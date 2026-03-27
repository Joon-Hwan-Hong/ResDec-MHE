"""
Cognitive Resilience Model - Full end-to-end architecture.

Combines all branches (PseudobulkEncoder, HGTEncoderBatched, CellTransformer) with
RegionHandler, FusionLayer, PathologyEncoder, PathologyStratifiedAttention,
and prediction heads to predict cognitive resilience from multi-modal inputs.

Data flow:
    region_pseudobulk [B, n_regions, 31, G] -> PseudobulkEncoder (per region)
        -> RegionHandler -> pooled [B, 31, d] + region_context [B, d]
    ccc_graph (edge_index_dict_list, edge_attr_dict_list) -> HGTEncoderBatched
        -> hgt_emb [B, 31, d]
    cells [B, 31, max_cells, G] -> CellTransformer -> cell_emb [B, 31, d]

    [pooled, hgt_emb, cell_emb] -> FusionLayer -> fused [B, 31, d_fused]
    [pathology, region_context] -> PathologyEncoder -> path_emb [B, d_cond]
    [fused, path_emb] -> PathologyStratifiedAttention -> attended [B, d_fused] + weights
    attended -> PredictionHead -> mean [B, 1] (+ std [B, 1] if Bayesian)

Expected Input Format:
    This model expects data from collate_for_hgt_multiregion() which provides:
    - region_pseudobulk: [B, n_regions, n_cell_types, n_genes]
    - region_mask: [B, n_regions]
    - edge_index_dict_list: List[Dict[(src, rel, dst): Tensor[2, n_edges]]]
    - edge_attr_dict_list: List[Dict[(src, rel, dst): Tensor[n_edges, 1]]]
    - cells, cell_mask, pathology, etc.

    For single-region data, use pseudobulk [B, n_cell_types, n_genes] which will
    be automatically expanded to region format.
"""

import warnings
from typing import Optional

import torch
import logging
import torch.nn as nn
from pyro.nn import PyroModule

from src.data.constants import CELL_TYPE_ORDER, ALL_EDGE_TYPES, N_REGIONS, PFC_REGION_IDX, sanitize_key
from src.data.collate import build_x_dict_list_from_embeddings
from src.models.branches import PseudobulkEncoder, CellTransformer
from src.models.branches.hgt_encoder import HGTEncoderBatched
from src.models.components import RegionHandler
from src.models.fusion import FusionLayer, PathologyEncoder, PathologyStratifiedAttention
from src.models.heads import BayesianPredictionHead, DeterministicPredictionHead

logger = logging.getLogger(__name__)


class CognitiveResilienceModel(PyroModule):
    """
    Full end-to-end model for cognitive resilience prediction.

    Integrates three encoding branches:
    - PseudobulkEncoder: Gene expression -> cell-type embeddings
    - HGTEncoderBatched: Cell-cell communication graph -> CCC embeddings (per-sample)
    - CellTransformer: Cell-level data -> heterogeneity embeddings

    These are fused and processed through pathology-conditioned attention
    to produce a final cognition prediction.

    Args:
        n_genes: Number of input genes
        n_cell_types: Number of cell types (default: 31)
        d_embed: Embedding dimension for all branches
        d_fused: Fused representation dimension
        d_cond: Pathology conditioning dimension
        n_regions: Number of brain regions (default: 6)
        n_hgt_layers: Number of HGT layers (default: 3)
        n_hgt_heads: Number of HGT attention heads (default: 4)
        n_cell_transformer_heads: Number of Set Transformer attention heads (default: 4).
            Independent from n_hgt_heads — HGT heads attend over graph edges while
            Set Transformer heads attend over cells within a type.
        n_isab_layers: Number of ISAB layers in CellTransformer (default: 2)
        n_inducing_points: Number of inducing points in ISAB (default: 32)
        n_attention_heads: Number of attention heads in PathologyStratifiedAttention (default: 4)
        gene_gate_temperature: Initial temperature for gene attention gate (default: 2.0,
            per design doc τ_max). Higher = softer attention. Annealed during training.
        selection_temperature: Temperature for cell type selection softmax (default: 1.0,
            per design doc). Fixed during training unless explicitly annealed.
        use_bayesian_head: Whether to use Bayesian prediction head (default: True)
        d_head_hidden: Hidden dimension in prediction head (default: 64)
        dropout: Dropout probability (default: 0.1)
        node_types: Cell type names (default: from constants)
        edge_categories: Edge type names (default: from constants)

    Forward inputs (collate_for_hgt_multiregion format):
        region_pseudobulk: [B, n_regions, n_cell_types, n_genes] OR
        pseudobulk: [B, n_cell_types, n_genes] (single-region, auto-expanded)
        region_mask: [B, n_regions]
        edge_index_dict_list: List of {(src, rel, dst): Tensor[2, n_edges]} per sample
        edge_attr_dict_list: List of {(src, rel, dst): Tensor[n_edges, 1]} per sample
        cells: [B, n_cell_types, max_cells, n_genes]
        cell_mask: [B, n_cell_types, max_cells]
        cell_type_mask: [B, n_cell_types] (optional, for masking missing cell types)
        pathology: [B, 3]
        cognition: [B, 1] (optional, for training)

    Forward outputs:
        dict with 'mean', 'std' (if Bayesian), 'attention_weights',
        and optionally 'hgt_attention' if return_hgt_attention=True
    """

    def __init__(
        self,
        n_genes: int,
        n_cell_types: int = 31,
        d_embed: int = 128,
        d_fused: int = 128,
        d_cond: int = 64,
        n_regions: int = 6,
        n_hgt_layers: int = 3,
        n_hgt_heads: int = 4,
        n_cell_transformer_heads: int = 4,
        n_isab_layers: int = 2,
        n_inducing_points: int = 32,
        n_attention_heads: int = 4,
        gene_gate_temperature: float = 2.0,
        selection_temperature: float = 1.0,
        use_bayesian_head: bool = True,
        d_head_hidden: int = 64,
        dropout: float = 0.1,
        n_pathology_features: int = 3,
        n_pma_seeds: int = 1,
        mlp_hidden: list[int] | None = None,
        use_layer_norm: bool = True,
        node_types: Optional[list[str]] = None,
        edge_categories: Optional[list[str]] = None,
        use_gradient_checkpointing: bool = False,
        use_torch_compile: bool = False,
    ):
        super().__init__("cognitive_resilience_model")

        # Validate inputs
        if n_genes <= 0:
            raise ValueError(f"n_genes must be positive, got {n_genes}")
        if n_cell_types <= 0:
            raise ValueError(f"n_cell_types must be positive, got {n_cell_types}")
        if d_embed <= 0:
            raise ValueError(f"d_embed must be positive, got {d_embed}")
        if d_fused <= 0:
            raise ValueError(f"d_fused must be positive, got {d_fused}")
        if d_cond <= 0:
            raise ValueError(f"d_cond must be positive, got {d_cond}")
        if d_fused % n_attention_heads != 0:
            raise ValueError(
                f"d_fused ({d_fused}) must be divisible by n_attention_heads ({n_attention_heads})"
            )

        # Store configuration
        self.n_genes = n_genes
        self.n_cell_types = n_cell_types
        self.d_embed = d_embed
        self.d_fused = d_fused
        self.d_cond = d_cond
        self.n_regions = n_regions
        self.use_bayesian_head = use_bayesian_head

        # Node and edge type configuration
        self.node_types = node_types if node_types is not None else list(CELL_TYPE_ORDER)
        self.edge_categories = edge_categories if edge_categories is not None else list(ALL_EDGE_TYPES)

        # Sanitized node types for HGT dict lookups
        self._sanitized_node_types = [sanitize_key(nt) for nt in self.node_types]
        self._node_type_to_idx = {nt: idx for idx, nt in enumerate(self.node_types)}
        self._sanitized_to_idx = {sanitize_key(nt): idx for idx, nt in enumerate(self.node_types)}

        # Pre-compute HGT output index mapping (avoids creating index tensor every forward)
        _hgt_indices = [self._sanitized_to_idx[sanitize_key(nt)] for nt in self.node_types]
        self.register_buffer(
            "_hgt_idx_tensor",
            torch.tensor(_hgt_indices, dtype=torch.long),
            persistent=False,
        )

        # Branch 1: Pseudobulk Encoder (applied per region)
        self.pseudobulk_encoder = PseudobulkEncoder(
            n_cell_types=n_cell_types,
            n_genes=n_genes,
            d_embed=d_embed,
            mlp_hidden=mlp_hidden,
            dropout=dropout,
            temperature=gene_gate_temperature,
            use_layer_norm=use_layer_norm,
        )

        # Region Handler (pools across regions)
        self.region_handler = RegionHandler(
            d_model=d_embed,
            n_regions=n_regions,
        )

        # Branch 2: HGT Encoder (cell-cell communication) - BATCHED version
        # HGT takes encoded pseudobulk embeddings as node features
        # Uses HGTEncoderBatched to process per-sample graphs correctly
        self.hgt_encoder = HGTEncoderBatched(
            d_input=d_embed,  # Takes encoded embeddings, not raw genes
            d_hidden=d_embed,
            d_output=d_embed,
            n_heads=n_hgt_heads,
            n_layers=n_hgt_layers,
            dropout=dropout,
            edge_dim=1,  # LIANA magnitude scores
            node_types=self.node_types,
            edge_categories=self.edge_categories,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

        # Branch 3: Cell Transformer (cell-level heterogeneity)
        self.cell_transformer = CellTransformer(
            n_genes=n_genes,
            n_cell_types=n_cell_types,
            d_model=d_embed,
            n_heads=n_cell_transformer_heads,
            n_isab_layers=n_isab_layers,
            n_inducing=n_inducing_points,
            n_pma_seeds=n_pma_seeds,
            dropout=dropout,
            selection_temperature=selection_temperature,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

        # Fusion Layer
        self.fusion_layer = FusionLayer(
            d_embed=d_embed,
            d_fused=d_fused,
            n_cell_types=n_cell_types,
            dropout=dropout,
        )

        # Pathology Encoder (combines pathology with region context)
        self.pathology_encoder = PathologyEncoder(
            n_pathology_features=n_pathology_features,
            d_region=d_embed,
            d_cond=d_cond,
            dropout=dropout,
        )

        # Pathology-Stratified Attention
        self.pathology_attention = PathologyStratifiedAttention(
            d_fused=d_fused,
            d_cond=d_cond,
            n_heads=n_attention_heads,
            n_cell_types=n_cell_types,
        )

        # Prediction Head (Bayesian or Deterministic)
        if use_bayesian_head:
            self.prediction_head = BayesianPredictionHead(
                d_input=d_fused,
                d_hidden=d_head_hidden,
            )
        else:
            self.prediction_head = DeterministicPredictionHead(
                d_input=d_fused,
                d_hidden=d_head_hidden,
                dropout=dropout,
            )

        # torch.compile fuses Linear+LN+GELU+Dropout into fewer CUDA kernels.
        # Only applied to pure tensor modules (no dict ops, no dynamic shapes).
        # BayesianPredictionHead uses pyro.sample which is incompatible with
        # torch.compile, so only compile the deterministic head.
        # Gated by config flag since torch.compile adds startup latency.
        if use_torch_compile:
            self.pseudobulk_encoder = torch.compile(self.pseudobulk_encoder)
            self.fusion_layer = torch.compile(self.fusion_layer)
            self.pathology_encoder = torch.compile(self.pathology_encoder)
            if not use_bayesian_head:
                self.prediction_head = torch.compile(self.prediction_head)

    def _encode_pseudobulk_per_region(
        self,
        region_pseudobulk: torch.Tensor,  # [B, n_regions, n_cell_types, n_genes]
        region_mask: Optional[torch.Tensor] = None,  # [B, n_regions]
    ) -> torch.Tensor:
        """
        Apply PseudobulkEncoder to each region independently.

        When region_mask is provided, only encodes regions that are active in
        at least one sample (avoids wasting compute on zero-filled regions).
        For single-region subjects this means encoding B instead of B*6 samples.

        Returns:
            [B, n_regions, n_cell_types, d_embed]
        """
        B, R, C, G = region_pseudobulk.shape

        if region_mask is not None:
            # Only encode regions active in ANY sample in the batch
            active_regions = region_mask.any(dim=0)  # [R]
            if not active_regions.all():
                active_indices = active_regions.nonzero(as_tuple=True)[0]
                active_data = region_pseudobulk[:, active_indices]  # [B, n_active, C, G]
                n_active = active_indices.size(0)

                flat = active_data.reshape(B * n_active, C, G)
                encoded_active = self.pseudobulk_encoder(flat)  # [B*n_active, C, d_embed]
                encoded_active = encoded_active.view(B, n_active, C, self.d_embed)

                # Reconstruct full-sized tensor with zeros for inactive regions
                encoded = torch.zeros(
                    B, R, C, self.d_embed,
                    device=region_pseudobulk.device, dtype=encoded_active.dtype,
                )
                encoded[:, active_indices] = encoded_active
                return encoded

        # Default: encode all regions
        flat = region_pseudobulk.view(B * R, C, G)
        encoded = self.pseudobulk_encoder(flat)  # [B * R, C, d_embed]
        return encoded.view(B, R, C, self.d_embed)

    def _convert_hgt_batched_output_to_tensor(
        self,
        hgt_out_dict: dict[str, torch.Tensor],  # {cell_type: [B, 1, d_embed]}
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Convert HGTEncoderBatched output to tensor [B, n_cell_types, d_embed].

        HGTEncoderBatched returns {cell_type: [B, 1, d_embed]} after stacking.

        Args:
            hgt_out_dict: Dict mapping cell type to batched embeddings
            batch_size: Expected batch size
            device: Device for output tensor

        Returns:
            [B, n_cell_types, d_embed] tensor with embeddings for all cell types
        """
        # Initialize output tensor, inferring dtype from HGT output for AMP
        # compatibility. Under torch.autocast, HGT layers may produce float16
        # intermediates; using the same dtype here avoids an implicit cast that
        # would break the autocast graph.
        if not hgt_out_dict:
            return torch.zeros(
                batch_size, self.n_cell_types, self.d_embed, device=device,
            )
        sample_tensor = next(iter(hgt_out_dict.values()))

        # Fast path: when all sanitized node types are present (common case),
        # use pre-computed index buffer to avoid per-forward tensor creation.
        sanitized_keys = set(self._sanitized_to_idx.keys())
        if set(hgt_out_dict.keys()) == sanitized_keys:
            # Stack in canonical order matching _hgt_idx_tensor
            stacked = torch.stack(
                [hgt_out_dict[snt].squeeze(1) for snt in self._sanitized_to_idx],
                dim=1,
            )  # [B, n_cell_types, d_embed]
            output = torch.zeros(
                batch_size, self.n_cell_types, self.d_embed,
                device=device, dtype=sample_tensor.dtype,
            )
            idx_tensor = self._hgt_idx_tensor.view(1, -1, 1).expand(
                batch_size, -1, self.d_embed
            )
            output.scatter_(1, idx_tensor, stacked)
            return output

        # Slow path: dynamic index building for partial node sets
        indices = []
        tensors = []
        for node_type, emb in hgt_out_dict.items():
            if node_type in self._node_type_to_idx:
                ct_idx = self._node_type_to_idx[node_type]
            elif node_type in self._sanitized_to_idx:
                ct_idx = self._sanitized_to_idx[node_type]
            else:
                logger.warning("Skipping unknown HGT output key: %s", node_type)
                continue
            indices.append(ct_idx)
            tensors.append(emb.squeeze(1))  # [B, d_embed]

        if not tensors:
            return torch.zeros(
                batch_size, self.n_cell_types, self.d_embed,
                device=device, dtype=sample_tensor.dtype,
            )

        # Stack and scatter in batched operations
        stacked = torch.stack(tensors, dim=1)  # [B, n_found, d_embed]
        output = torch.zeros(
            batch_size, self.n_cell_types, self.d_embed,
            device=device, dtype=sample_tensor.dtype,
        )
        idx_tensor = torch.tensor(indices, device=device).view(1, -1, 1).expand(
            batch_size, -1, self.d_embed
        )
        output.scatter_(1, idx_tensor, stacked)
        return output

    def forward(
        self,
        # Multi-region format (from collate_for_hgt_multiregion)
        region_pseudobulk: Optional[torch.Tensor] = None,  # [B, n_regions, n_cell_types, n_genes]
        region_mask: Optional[torch.Tensor] = None,        # [B, n_regions]
        # Single-region fallback
        pseudobulk: Optional[torch.Tensor] = None,         # [B, n_cell_types, n_genes]
        # HGT graph inputs (per-sample dicts from collate_for_hgt)
        edge_index_dict_list: Optional[list[dict]] = None,
        edge_attr_dict_list: Optional[list[dict]] = None,
        # Cell-level inputs
        cells: Optional[torch.Tensor] = None,              # [B, n_cell_types, max_cells, n_genes]
        cell_mask: Optional[torch.Tensor] = None,          # [B, n_cell_types, max_cells]
        cell_type_mask: Optional[torch.Tensor] = None,     # [B, n_cell_types] (optional)
        # Pathology and target
        pathology: Optional[torch.Tensor] = None,          # [B, 3]
        cognition: Optional[torch.Tensor] = None,          # [B, 1] for training
        # Interpretability options
        return_hgt_attention: bool = False,
        return_pma_attention: bool = False,
        return_region_attention: bool = False,
        return_embeddings: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through the full model.

        Supports two input formats:
        1. Multi-region (preferred): region_pseudobulk [B, R, C, G] + region_mask
        2. Single-region (fallback): pseudobulk [B, C, G] (auto-expanded to region format)

        For HGT, uses per-sample dict format (edge_index_dict_list, edge_attr_dict_list)
        from collate_for_hgt.

        Args:
            region_pseudobulk: [B, n_regions, n_cell_types, n_genes] regional pseudobulk
            region_mask: [B, n_regions] bool mask for available regions
            pseudobulk: [B, n_cell_types, n_genes] single-region fallback
            edge_index_dict_list: List of {(src, rel, dst): [2, n_edges]} per sample
            edge_attr_dict_list: List of {(src, rel, dst): [n_edges, 1]} per sample
            cells: [B, n_cell_types, max_cells, n_genes] cell-level expression
            cell_mask: [B, n_cell_types, max_cells] bool mask for valid cells
            cell_type_mask: [B, n_cell_types] optional mask for missing cell types.
                Only affects PathologyStratifiedAttention (masked types get -inf scores
                before softmax). HGT and CellTransformer still process all 31 types:
                HGT needs all node types for correct message-passing topology, and
                CellTransformer already handles empty types via cell_mask (producing
                zero embeddings).
            pathology: [B, 3] pathology features (amyloid, tau, global)
            cognition: [B, 1] target cognition scores (optional, for training)
            return_hgt_attention: Whether to return HGT attention weights (for interpretability)
            return_pma_attention: Whether to return PMA cell-level attention (for interpretability)
            return_region_attention: Whether to return region-level attention weights
            return_embeddings: Whether to return intermediate embeddings dict

        Returns:
            dict with keys:
                - 'mean': [B, 1] predicted cognition
                - 'std': [B, 1] uncertainty (only if use_bayesian_head)
                - 'attention_weights': [B, n_heads, n_cell_types] pathology attention
                - 'hgt_attention': List of attention dicts per layer (if return_hgt_attention)
                - 'pma_attention': List of [B, n_heads, n_seeds, max_cells] per cell type (if return_pma_attention)
                - 'region_attention': [B, n_regions] normalized region weights (if return_region_attention)
                - 'embeddings': dict of branch/fused/attended embeddings (if return_embeddings)
        """
        # ─────────────────────────────────────────────────────────────────────
        # Handle single-region vs multi-region input
        # ─────────────────────────────────────────────────────────────────────
        if region_pseudobulk is not None:
            # Multi-region format
            B = region_pseudobulk.size(0)
            device = region_pseudobulk.device

            if region_mask is None:
                # Default: all regions available
                region_mask = torch.ones(B, self.n_regions, dtype=torch.bool, device=device)
        elif pseudobulk is not None:
            # Single-region fallback: expand to multi-region format
            B = pseudobulk.size(0)
            device = pseudobulk.device

            # Expand [B, C, G] -> [B, n_regions, C, G] with only PFC region filled
            region_pseudobulk = torch.zeros(
                B, self.n_regions, self.n_cell_types, self.n_genes,
                device=device, dtype=pseudobulk.dtype
            )
            region_pseudobulk[:, PFC_REGION_IDX, :, :] = pseudobulk

            # Mask: only PFC region is available
            region_mask = torch.zeros(B, self.n_regions, dtype=torch.bool, device=device)
            region_mask[:, PFC_REGION_IDX] = True
        else:
            raise ValueError("Must provide either region_pseudobulk or pseudobulk")

        # Validate optional cell_type_mask shape early
        if cell_type_mask is not None and cell_type_mask.shape != (B, self.n_cell_types):
            raise ValueError(
                f"cell_type_mask shape must be [{B}, {self.n_cell_types}], "
                f"got {list(cell_type_mask.shape)}"
            )

        # Validate required inputs — these should always come from the collate function.
        # Explicit checks here provide clear error messages if a data pipeline bug
        # passes None instead of a tensor.
        if cells is None:
            raise ValueError("cells tensor is required but got None — check collate function output")
        if cell_mask is None:
            raise ValueError("cell_mask tensor is required but got None — check collate function output")
        if pathology is None:
            raise ValueError("pathology tensor is required but got None — check collate function output")

        # ─────────────────────────────────────────────────────────────────────
        # Branch 1: Pseudobulk encoding + region handling
        # ─────────────────────────────────────────────────────────────────────
        # [B, n_regions, n_cell_types, n_genes] -> [B, n_regions, n_cell_types, d_embed]
        region_encoded = self._encode_pseudobulk_per_region(region_pseudobulk, region_mask)

        # Pool across regions: [B, n_cell_types, d_embed] + [B, d_embed] + [B, n_regions]
        pseudobulk_emb, region_context, region_attn = self.region_handler(region_encoded, region_mask)

        # ─────────────────────────────────────────────────────────────────────
        # Branch 2: HGT encoding (cell-cell communication) - Per-sample graphs
        # Design decision: HGT receives region-pooled (not region-specific) features
        # because CCC edges represent communication patterns across the subject's
        # cell types, while RegionHandler's learned attention weights naturally
        # prioritize the region with strongest signal (typically PFC, which is also
        # where LIANA CCC edges originate). See architecture doc Part 1, §3.2.
        # ─────────────────────────────────────────────────────────────────────
        # Build x_dict_list from pooled pseudobulk embeddings
        x_dict_list = build_x_dict_list_from_embeddings(
            pseudobulk_emb, self._sanitized_node_types
        )

        # Handle edge dicts — pass through as-is; HGTConv handles empty dicts
        # correctly (isolated nodes get zero communication via received_messages mask).
        # Do NOT inject synthetic edges — that would encode phantom communication.
        if edge_index_dict_list is not None and edge_attr_dict_list is not None:
            # Use provided edge dicts (may contain empty dicts for edgeless samples)
            pass
        else:
            # No edges provided at all — create empty dicts for each sample
            edge_index_dict_list = [{} for _ in range(B)]
            edge_attr_dict_list = [{} for _ in range(B)]

        # Run HGTEncoderBatched - processes each sample's graph separately
        hgt_out_dict, hgt_attention = self.hgt_encoder(
            x_dict_list,
            edge_index_dict_list,
            edge_attr_dict_list,
            return_attention=return_hgt_attention,
        )

        # Convert HGT output to tensor: [B, n_cell_types, d_embed]
        hgt_emb = self._convert_hgt_batched_output_to_tensor(hgt_out_dict, B, device)

        # ─────────────────────────────────────────────────────────────────────
        # Branch 3: Cell transformer (cell-level heterogeneity)
        # ─────────────────────────────────────────────────────────────────────
        # [B, n_cell_types, max_cells, n_genes] -> [B, n_cell_types, d_embed]
        cell_emb, selection_weights, pma_attention = self.cell_transformer(
            cells, cell_mask, return_attention=return_pma_attention, apply_selection_weights=True
        )

        # ─────────────────────────────────────────────────────────────────────
        # Fusion: combine all three branches
        # ─────────────────────────────────────────────────────────────────────
        # [B, n_cell_types, d_fused]
        fused = self.fusion_layer(pseudobulk_emb, hgt_emb, cell_emb)

        # ─────────────────────────────────────────────────────────────────────
        # Pathology encoding with region context
        # ─────────────────────────────────────────────────────────────────────
        # [B, d_cond]
        path_emb = self.pathology_encoder(pathology, region_context)

        # ─────────────────────────────────────────────────────────────────────
        # Pathology-stratified attention over cell types
        # ─────────────────────────────────────────────────────────────────────
        # [B, d_fused], [B, n_heads, n_cell_types] or None
        attended, attention_weights = self.pathology_attention(
            fused, path_emb, cell_type_mask=cell_type_mask,
            return_attention_weights=not self.training,
        )

        # ─────────────────────────────────────────────────────────────────────
        # Prediction
        # ─────────────────────────────────────────────────────────────────────
        output = {'attention_weights': attention_weights, 'attended': attended}

        if return_hgt_attention and hgt_attention is not None:
            output['hgt_attention'] = hgt_attention

        if return_pma_attention and pma_attention is not None:
            output['pma_attention'] = pma_attention

        if return_region_attention:
            output['region_attention'] = region_attn

        if return_embeddings:
            output['embeddings'] = {
                'pseudobulk': pseudobulk_emb,  # [B, n_cell_types, d_embed]
                'hgt': hgt_emb,                # [B, n_cell_types, d_embed]
                'cell': cell_emb,              # [B, n_cell_types, d_embed]
                'fused': fused,                # [B, n_cell_types, d_fused]
                'attended': attended,           # [B, d_fused]
            }

        if self.use_bayesian_head:
            mean, std = self.prediction_head(attended, cognition)
            output['mean'] = mean
            output['std'] = std
        else:
            mean = self.prediction_head(attended)
            output['mean'] = mean

        return output

    def forward_encoder_only(
        self,
        region_pseudobulk=None,
        region_mask=None,
        pseudobulk=None,
        edge_index_dict_list=None,
        edge_attr_dict_list=None,
        cells=None,
        cell_mask=None,
        cell_type_mask=None,
        pathology=None,
    ) -> dict[str, torch.Tensor]:
        """Run encoder branches + fusion + attention, skip prediction head.

        Used by validation to run the deterministic encoder once and reuse
        the attended vector for both predictions (via head with median weights)
        and metrics, avoiding a redundant full forward pass.

        Returns:
            dict with 'attended' [B, d_fused] and 'attention_weights'
        """
        # Handle single-region vs multi-region input
        if region_pseudobulk is not None:
            B = region_pseudobulk.size(0)
            device = region_pseudobulk.device
            if region_mask is None:
                region_mask = torch.ones(B, self.n_regions, dtype=torch.bool, device=device)
        elif pseudobulk is not None:
            B = pseudobulk.size(0)
            device = pseudobulk.device
            region_pseudobulk = torch.zeros(
                B, self.n_regions, self.n_cell_types, self.n_genes,
                device=device, dtype=pseudobulk.dtype,
            )
            region_pseudobulk[:, PFC_REGION_IDX, :, :] = pseudobulk
            region_mask = torch.zeros(B, self.n_regions, dtype=torch.bool, device=device)
            region_mask[:, PFC_REGION_IDX] = True
        else:
            raise ValueError("Must provide either region_pseudobulk or pseudobulk")

        if cell_type_mask is not None and cell_type_mask.shape != (B, self.n_cell_types):
            raise ValueError(
                f"cell_type_mask shape must be [{B}, {self.n_cell_types}], "
                f"got {list(cell_type_mask.shape)}"
            )
        if cells is None:
            raise ValueError("cells tensor is required")
        if cell_mask is None:
            raise ValueError("cell_mask tensor is required")
        if pathology is None:
            raise ValueError("pathology tensor is required")

        # Branch 1: Pseudobulk + region
        region_encoded = self._encode_pseudobulk_per_region(region_pseudobulk, region_mask)
        pseudobulk_emb, region_context, _ = self.region_handler(region_encoded, region_mask)

        # Branch 2: HGT
        x_dict_list = build_x_dict_list_from_embeddings(pseudobulk_emb, self._sanitized_node_types)
        if edge_index_dict_list is None:
            edge_index_dict_list = [{} for _ in range(B)]
        if edge_attr_dict_list is None:
            edge_attr_dict_list = [{} for _ in range(B)]
        hgt_out_dict, _ = self.hgt_encoder(x_dict_list, edge_index_dict_list, edge_attr_dict_list)
        hgt_emb = self._convert_hgt_batched_output_to_tensor(hgt_out_dict, B, device)

        # Branch 3: Cell transformer
        cell_emb, _, _ = self.cell_transformer(
            cells, cell_mask, return_attention=False, apply_selection_weights=True,
        )

        # Fusion + pathology + attention
        fused = self.fusion_layer(pseudobulk_emb, hgt_emb, cell_emb)
        path_emb = self.pathology_encoder(pathology, region_context)
        attended, attention_weights = self.pathology_attention(
            fused, path_emb, cell_type_mask=cell_type_mask,
            return_attention_weights=not self.training,
        )

        return {"attended": attended, "attention_weights": attention_weights}

    def get_cell_type_importance(self) -> dict[str, float]:
        """
        Get cell type selection weights from CellTransformer.

        Returns:
            Dict mapping cell type name to importance weight
        """
        weights = self.cell_transformer.get_selection_weights()
        return {
            name: weights[idx].item()
            for idx, name in enumerate(self.node_types)
        }

    def get_region_importance(self) -> dict[str, float]:
        """
        Get region importance weights from RegionHandler.

        Returns:
            Dict mapping region name to importance weight
        """
        return self.region_handler.get_region_importance_dict()

    def get_hgt_layer_scales(self) -> dict[str, torch.Tensor]:
        """
        Get HGT LayerScale values for interpretability.

        Returns:
            Dict with scales per cell type across layers
        """
        return self.hgt_encoder.get_layer_scales()

    def num_parameters(self, trainable_only: bool = True) -> dict[str, int]:
        """
        Count parameters, total and per component.

        Args:
            trainable_only: If True, count only parameters with requires_grad.

        Returns:
            Dict with 'total' and per-component counts:
                'pseudobulk_encoder', 'region_handler', 'hgt_encoder',
                'cell_transformer', 'fusion_layer', 'pathology_encoder',
                'pathology_attention', 'prediction_head'.
        """
        def _count(module: nn.Module) -> int:
            return sum(
                p.numel() for p in module.parameters()
                if not trainable_only or p.requires_grad
            )

        components = {
            'pseudobulk_encoder': self.pseudobulk_encoder,
            'region_handler': self.region_handler,
            'hgt_encoder': self.hgt_encoder,
            'cell_transformer': self.cell_transformer,
            'fusion_layer': self.fusion_layer,
            'pathology_encoder': self.pathology_encoder,
            'pathology_attention': self.pathology_attention,
            'prediction_head': self.prediction_head,
        }

        counts = {name: _count(mod) for name, mod in components.items()}
        counts['total'] = _count(self)
        return counts

    def extra_repr(self) -> str:
        return (
            f"n_genes={self.n_genes}, n_cell_types={self.n_cell_types}, "
            f"d_embed={self.d_embed}, d_fused={self.d_fused}, d_cond={self.d_cond}, "
            f"n_regions={self.n_regions}, use_bayesian_head={self.use_bayesian_head}"
        )


def _cfg_get(cfg, key, default, section_name="model"):
    """Get config value with warning when using default."""
    val = cfg.get(key)
    if val is None:
        warnings.warn(
            f"Config key '{section_name}.{key}' not found, using default={default}. "
            f"Check for typos if you expected this to be set.",
            stacklevel=3,
        )
        return default
    return val


def build_model_from_config(model_cfg) -> CognitiveResilienceModel:
    """Build a CognitiveResilienceModel from a config dict/DictConfig.

    Single source of truth for model construction. Used by both the
    Lightning training module and inference Predictor to ensure identical
    model architecture.

    Args:
        model_cfg: Model config section (config.model) with nested keys
            for hgt, set_transformer, pathology_attention, gene_gate,
            cell_type_selector, head, and pseudobulk sub-configs.

    Returns:
        Configured CognitiveResilienceModel instance.
    """
    head_type = model_cfg.get("head", {}).get("type", "deterministic")
    use_bayesian = head_type == "bayesian"
    use_torch_compile = model_cfg.get("use_torch_compile", False)
    # node_types and edge_categories are intentionally not configurable —
    # they are fixed to the 31 Allen ABC cell types (CELL_TYPE_ORDER) and
    # 5 CellChatDB edge categories (ALL_EDGE_TYPES) from constants.py.
    return CognitiveResilienceModel(
        n_genes=model_cfg.n_genes,
        n_cell_types=model_cfg.n_cell_types,
        d_embed=model_cfg.d_embed,
        d_fused=model_cfg.d_fused,
        d_cond=model_cfg.pathology_attention.d_cond,
        n_regions=N_REGIONS,
        n_hgt_layers=model_cfg.hgt.n_layers,
        n_hgt_heads=model_cfg.hgt.n_heads,
        n_cell_transformer_heads=_cfg_get(model_cfg.set_transformer, "n_heads", 4, "model.set_transformer"),
        n_isab_layers=model_cfg.set_transformer.n_isab_layers,
        n_inducing_points=model_cfg.set_transformer.n_inducing_points,
        n_attention_heads=model_cfg.pathology_attention.n_heads,
        gene_gate_temperature=_cfg_get(model_cfg.gene_gate, "initial_temperature", 2.0, "model.gene_gate"),
        selection_temperature=_cfg_get(model_cfg.cell_type_selector, "selection_temperature", 1.0, "model.cell_type_selector"),
        use_bayesian_head=use_bayesian,
        d_head_hidden=model_cfg.head.d_hidden,
        dropout=_cfg_get(model_cfg, "dropout", 0.1, "model"),
        n_pathology_features=_cfg_get(model_cfg.pathology_attention, "n_pathology_features", 3, "model.pathology_attention"),
        n_pma_seeds=_cfg_get(model_cfg.set_transformer, "n_pma_seeds", 1, "model.set_transformer"),
        mlp_hidden=list(model_cfg.pseudobulk.mlp_hidden) if model_cfg.get("pseudobulk", {}).get("mlp_hidden") is not None else None,
        use_layer_norm=model_cfg.get("pseudobulk", {}).get("use_layer_norm", True),
        use_torch_compile=use_torch_compile,
    )
