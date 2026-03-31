"""
Cognitive Resilience Model - Full end-to-end architecture.

Two-branch architecture combining HGT and CellTransformer encoders with
RegionHandler, FusionLayer, PathologyEncoder, PathologyStratifiedAttention,
and prediction heads to predict cognitive resilience from multi-modal inputs.

Data flow:
    region_pseudobulk [B, n_regions, 31, G] -> GeneAttentionGate_HGT -> Linear(G, d)
        -> RegionHandler -> pooled [B, 31, d] + region_context [B, d]
        -> HGTEncoderTensor -> hgt_emb [B, 31, d]
    cell_data [total_cells, G] + cell_offsets [B, 32] -> CellTransformer -> cell_emb [B, 31, d]

    [hgt_emb, cell_emb] -> FusionLayer -> fused [B, 31, d_fused]
    [pathology, region_context] -> PathologyEncoder -> path_emb [B, d_cond]
    [fused, path_emb] -> PathologyStratifiedAttention -> attended [B, d_fused] + weights
    attended -> PredictionHead -> mean [B, 1] (+ std [B, 1] if Bayesian)

Expected Input Format:
    This model expects data from collate_for_hgt_multiregion() which provides:
    - region_pseudobulk: [B, n_regions, n_cell_types, n_genes]
    - region_mask: [B, n_regions]
    - ccc_edge_index: [2, E_total] flat edge indices
    - ccc_edge_type: [E_total] edge type indices
    - ccc_edge_attr: [E_total, 1] edge attributes
    - cell_data: [total_cells, n_genes] concatenated cell expressions
    - cell_offsets: [B, n_cell_types + 1] cumulative offsets
    - pathology, etc.

    For single-region data, use pseudobulk [B, n_cell_types, n_genes] which will
    be automatically expanded to region format.
"""

import warnings
from typing import Optional

import torch
import logging
import torch.nn as nn
from pyro.nn import PyroModule

from src.data.constants import CELL_TYPE_ORDER, ALL_EDGE_TYPES, N_REGIONS, PFC_REGION_IDX
from src.models.branches import CellTransformer
from src.models.branches.hgt_encoder_tensor import HGTEncoderTensor
from src.models.components import GeneAttentionGate, RegionHandler
from src.models.fusion import FusionLayer, PathologyEncoder, PathologyStratifiedAttention
from src.models.fusion.cross_attention_fusion import CrossAttentionFusionLayer
from src.models.fusion.normalized_concat_fusion import NormalizedConcatFusionLayer
from src.models.heads import BayesianPredictionHead, DeterministicPredictionHead

logger = logging.getLogger(__name__)


class CognitiveResilienceModel(PyroModule):
    """
    Full end-to-end model for cognitive resilience prediction.

    Two-branch architecture:
    - HGT branch: GeneAttentionGate → Linear → RegionHandler → HGTEncoderTensor
    - CellTransformer: GeneAttentionGate → SetTransformer → cell-level embeddings

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
            Applied to both HGT and CellTransformer gene gates.
        use_bayesian_head: Whether to use Bayesian prediction head (default: True)
        d_head_hidden: Hidden dimension in prediction head (default: 64)
        dropout: Dropout probability (default: 0.1)
        node_types: Cell type names (default: from constants)
        edge_categories: Edge type names (default: from constants)

    Forward inputs (collate_for_hgt_multiregion format):
        region_pseudobulk: [B, n_regions, n_cell_types, n_genes] OR
        pseudobulk: [B, n_cell_types, n_genes] (single-region, auto-expanded)
        region_mask: [B, n_regions]
        ccc_edge_index: [2, E_total] flat edge indices
        ccc_edge_type: [E_total] edge type indices
        ccc_edge_attr: [E_total, 1] edge attributes
        cell_data: [total_cells, n_genes] concatenated cell expressions
        cell_offsets: [B, n_cell_types + 1] cumulative offsets
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
        use_bayesian_head: bool = True,
        d_head_hidden: int = 64,
        dropout: float = 0.1,
        n_pathology_features: int = 3,
        n_pma_seeds: int = 1,
        node_types: Optional[list[str]] = None,
        edge_categories: Optional[list[str]] = None,
        use_hgt_encoder: bool = True,
        use_cell_transformer: bool = True,
        use_pathology_attention: bool = True,
        condition_on_cell_type: bool = True,
        use_gradient_checkpointing: bool = False,
        use_torch_compile: bool = False,
        target_mean: float = 0.0,
        fusion_type: str = "concat",
        fusion_n_heads: int = 4,
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
        self.n_pma_seeds = n_pma_seeds
        self.use_hgt_encoder = use_hgt_encoder
        self.use_cell_transformer = use_cell_transformer
        self.use_pathology_attention = use_pathology_attention
        self.use_gradient_checkpointing = use_gradient_checkpointing

        disabled = [name for name, on in [
            ("hgt_encoder", use_hgt_encoder),
            ("cell_transformer", use_cell_transformer),
        ] if not on]
        if disabled:
            logger.warning("Branch ablation: disabled branches = %s", disabled)

        # Node and edge type configuration
        self.node_types = node_types if node_types is not None else list(CELL_TYPE_ORDER)
        self.edge_categories = edge_categories if edge_categories is not None else list(ALL_EDGE_TYPES)

        # HGT input pipeline: GeneAttentionGate → Linear → RegionHandler → HGT
        # Gene attention gate + linear projection from gene space to embedding space
        self.hgt_gene_gate = GeneAttentionGate(
            n_cell_types=n_cell_types,
            n_genes=n_genes,
            temperature=gene_gate_temperature,
        )
        self.hgt_input_proj = nn.Linear(n_genes, d_embed)

        # Region Handler (pools across regions)
        self.region_handler = RegionHandler(
            d_model=d_embed,
            n_regions=n_regions,
        )

        # Branch 2: HGT Encoder (cell-cell communication) - tensor-native version
        # HGT takes encoded pseudobulk embeddings as node features
        # Uses HGTEncoderTensor for batched tensor operations (no dict round-trip)
        self.hgt_encoder = HGTEncoderTensor(
            d_input=d_embed,  # Takes encoded embeddings, not raw genes
            d_hidden=d_embed,
            d_output=d_embed,
            n_heads=n_hgt_heads,
            n_layers=n_hgt_layers,
            n_node_types=n_cell_types,
            n_edge_types=len(self.edge_categories),
            edge_dim=1,  # LIANA magnitude scores
            dropout=dropout,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

        # Branch 2: Cell Transformer (cell-level heterogeneity)
        # CellTransformer has its own GeneAttentionGate internally (Task 2).
        # gene_gate_temperature is passed to control both gates identically.
        self.cell_transformer = CellTransformer(
            n_genes=n_genes,
            n_cell_types=n_cell_types,
            d_model=d_embed,
            n_heads=n_cell_transformer_heads,
            n_isab_layers=n_isab_layers,
            n_inducing=n_inducing_points,
            n_pma_seeds=n_pma_seeds,
            dropout=dropout,
            gene_gate_temperature=gene_gate_temperature,
            use_gradient_checkpointing=use_gradient_checkpointing,
            condition_on_cell_type=condition_on_cell_type,
        )

        # Fusion Layer
        _xattn_types = {"cross_attention": "standard", "crossfuse": "reverse", "crossfuse_blend": "blend"}
        if fusion_type in _xattn_types:
            self.fusion_layer = CrossAttentionFusionLayer(
                d_embed=d_embed,
                d_fused=d_fused,
                n_cell_types=n_cell_types,
                n_heads=fusion_n_heads,
                dropout=dropout,
                n_pma_seeds=n_pma_seeds,
                attention_mode=_xattn_types[fusion_type],
            )
        elif fusion_type == "concat_normalized":
            self.fusion_layer = NormalizedConcatFusionLayer(
                d_embed=d_embed,
                d_fused=d_fused,
                n_cell_types=n_cell_types,
                dropout=dropout,
                n_pma_seeds=n_pma_seeds,
            )
        else:
            self.fusion_layer = FusionLayer(
                d_embed=d_embed,
                d_fused=d_fused,
                n_cell_types=n_cell_types,
                dropout=dropout,
                n_pma_seeds=n_pma_seeds,
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
                target_mean=target_mean,
            )
        else:
            self.prediction_head = DeterministicPredictionHead(
                d_input=d_fused,
                d_hidden=d_head_hidden,
                dropout=dropout,
            )

        # torch.compile fuses Linear+LN+GELU+Dropout into fewer CUDA kernels.
        # BayesianPredictionHead uses pyro.sample which is incompatible.
        # hgt_encoder excluded: scatter_add produces symbolic strides that
        # trip an Inductor codegen assertion (even with dynamic=True).
        # cell_transformer uses dynamic=True: total_cells varies per batch.
        # cache_size_limit raised from 8 to 64: DDP with variable shapes can
        # exhaust the default 8-entry cache, triggering excessive recompilation.
        self.use_torch_compile = use_torch_compile
        if use_torch_compile:
            import torch._dynamo.config
            torch._dynamo.config.cache_size_limit = 64
            self.hgt_input_proj = torch.compile(self.hgt_input_proj)
            self.fusion_layer = torch.compile(self.fusion_layer)
            self.pathology_encoder = torch.compile(self.pathology_encoder)
            self.cell_transformer = torch.compile(self.cell_transformer, dynamic=True)
            if not use_bayesian_head:
                self.prediction_head = torch.compile(self.prediction_head)

    def _encode_hgt_input_per_region(
        self,
        region_pseudobulk: torch.Tensor,  # [B, n_regions, n_cell_types, n_genes]
        region_mask: Optional[torch.Tensor] = None,  # [B, n_regions]
    ) -> torch.Tensor:
        """
        Apply GeneAttentionGate + Linear projection to each region independently.

        Replaces PseudobulkEncoder. The gate learns cell-type-specific gene
        importance, and the linear layer projects from gene space to d_embed.

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
                gated = self.hgt_gene_gate(flat)          # [B*n_active, C, G]
                projected = self.hgt_input_proj(gated)    # [B*n_active, C, d_embed]
                projected = projected.view(B, n_active, C, self.d_embed)

                # Reconstruct full-sized tensor with zeros for inactive regions
                encoded = torch.zeros(
                    B, R, C, self.d_embed,
                    device=region_pseudobulk.device, dtype=projected.dtype,
                )
                encoded[:, active_indices] = projected
                return encoded

        # Default: encode all regions
        flat = region_pseudobulk.view(B * R, C, G)
        gated = self.hgt_gene_gate(flat)          # [B*R, C, G]
        projected = self.hgt_input_proj(gated)    # [B*R, C, d_embed]
        return projected.view(B, R, C, self.d_embed)

    def forward(
        self,
        # Multi-region format (from collate_for_hgt_multiregion)
        region_pseudobulk: Optional[torch.Tensor] = None,  # [B, n_regions, n_cell_types, n_genes]
        region_mask: Optional[torch.Tensor] = None,        # [B, n_regions]
        # Single-region fallback
        pseudobulk: Optional[torch.Tensor] = None,         # [B, n_cell_types, n_genes]
        # CCC edge tensors (from collate)
        ccc_edge_index: Optional[torch.Tensor] = None,     # [2, E_total] flat
        ccc_edge_type: Optional[torch.Tensor] = None,      # [E_total]
        ccc_edge_attr: Optional[torch.Tensor] = None,      # [E_total, 1]
        # Cell-level inputs (flat format)
        cell_data: Optional[torch.Tensor] = None,          # [total_cells, n_genes]
        cell_offsets: Optional[torch.Tensor] = None,        # [B, n_types + 1]
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

        HGT receives batched edge tensors from the collate function.

        Args:
            region_pseudobulk: [B, n_regions, n_cell_types, n_genes] regional pseudobulk
            region_mask: [B, n_regions] bool mask for available regions
            pseudobulk: [B, n_cell_types, n_genes] single-region fallback
            ccc_edge_index: [2, E_total] flat edge indices
            ccc_edge_type: [E_total] edge type indices
            ccc_edge_attr: [E_total, 1] edge attributes
            cell_data: [total_cells, n_genes] concatenated cell expressions
            cell_offsets: [B, n_types + 1] cumulative offsets into cell_data
            cell_type_mask: [B, n_cell_types] optional mask for missing cell types.
                Only affects PathologyStratifiedAttention (masked types get -inf scores
                before softmax). HGT and CellTransformer still process all 31 types:
                HGT needs all node types for correct message-passing topology, and
                CellTransformer already handles empty types via zero cell counts
                (producing zero embeddings).
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
                - 'pma_attention': [B, n_cell_types, n_heads, n_seeds, max_cells] tensor (if return_pma_attention)
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

        # Validate cell inputs — need flat format (cell_data + cell_offsets)
        if cell_data is None or cell_offsets is None:
            raise ValueError(
                "Must provide (cell_data, cell_offsets) — "
                "check collate function output"
            )
        if pathology is None:
            raise ValueError("pathology tensor is required but got None — check collate function output")

        # ─────────────────────────────────────────────────────────────────────
        # Branch 1: HGT input pipeline + region handling + HGT encoding
        # GeneAttentionGate → Linear → RegionHandler → HGT
        # ─────────────────────────────────────────────────────────────────────
        # [B, n_regions, n_cell_types, n_genes] -> [B, n_regions, n_cell_types, d_embed]
        region_encoded = self._encode_hgt_input_per_region(region_pseudobulk, region_mask)
        # Pool across regions: [B, n_cell_types, d_embed] + [B, d_embed] + [B, n_regions]
        hgt_node_features, region_context, region_attn = self.region_handler(region_encoded, region_mask)

        hgt_attention = None
        if self.use_hgt_encoder:
            # Handle no-edges case
            if ccc_edge_index is None:
                ccc_edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
                ccc_edge_type = torch.zeros(0, dtype=torch.long, device=device)
                ccc_edge_attr = torch.zeros(0, 1, device=device)

            hgt_result = self.hgt_encoder(
                hgt_node_features, ccc_edge_index, ccc_edge_type, ccc_edge_attr,
                return_attention=return_hgt_attention,
            )
            if return_hgt_attention:
                hgt_emb, hgt_attention = hgt_result
            else:
                hgt_emb = hgt_result
        else:
            hgt_emb = torch.zeros(B, self.n_cell_types, self.d_embed, device=device)

        # ─────────────────────────────────────────────────────────────────────
        # Branch 2: Cell transformer (cell-level heterogeneity)
        # ─────────────────────────────────────────────────────────────────────
        pma_attention = None
        if self.use_cell_transformer:
            if self.use_gradient_checkpointing and self.training:
                cell_emb, pma_attention = torch.utils.checkpoint.checkpoint(
                    self.cell_transformer,
                    cell_data, cell_offsets,
                    use_reentrant=False,
                    return_attention=return_pma_attention,
                )
            else:
                cell_emb, pma_attention = self.cell_transformer(
                    cell_data, cell_offsets,
                    return_attention=return_pma_attention,
                )
        else:
            cell_emb = torch.zeros(B, self.n_cell_types, self.n_pma_seeds * self.d_embed, device=device)

        # ─────────────────────────────────────────────────────────────────────
        # Fusion: combine HGT + Cell branches
        # ─────────────────────────────────────────────────────────────────────
        # [B, n_cell_types, d_fused]
        if self.use_gradient_checkpointing and self.training:
            fused = torch.utils.checkpoint.checkpoint(
                self.fusion_layer,
                hgt_emb, cell_emb,
                use_reentrant=False,
            )
        else:
            fused = self.fusion_layer(hgt_emb, cell_emb)

        # ─────────────────────────────────────────────────────────────────────
        # Pathology encoding with region context
        # ─────────────────────────────────────────────────────────────────────
        # [B, d_cond]
        path_emb = self.pathology_encoder(pathology, region_context)

        # ─────────────────────────────────────────────────────────────────────
        # Pathology-stratified attention over cell types (or mean pooling)
        # ─────────────────────────────────────────────────────────────────────
        if self.use_pathology_attention:
            # [B, d_fused], [B, n_heads, n_cell_types] or None
            attended, attention_weights = self.pathology_attention(
                fused, path_emb, cell_type_mask=cell_type_mask,
                return_attention_weights=not self.training,
            )
        else:
            # Mean pooling over cell types (ablation: no pathology conditioning)
            if cell_type_mask is not None:
                mask = cell_type_mask.unsqueeze(-1).float()  # [B, n_cell_types, 1]
                attended = (fused * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                attended = fused.mean(dim=1)  # [B, d_fused]
            attention_weights = None

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
        ccc_edge_index=None,
        ccc_edge_type=None,
        ccc_edge_attr=None,
        cell_data=None,
        cell_offsets=None,
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

        if cell_data is None or cell_offsets is None:
            raise ValueError(
                "Must provide (cell_data, cell_offsets)"
            )
        if pathology is None:
            raise ValueError("pathology tensor is required")

        # Branch 1: HGT input pipeline + region handling + HGT
        region_encoded = self._encode_hgt_input_per_region(region_pseudobulk, region_mask)
        hgt_node_features, region_context, _ = self.region_handler(region_encoded, region_mask)

        if self.use_hgt_encoder:
            if ccc_edge_index is None:
                ccc_edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
                ccc_edge_type = torch.zeros(0, dtype=torch.long, device=device)
                ccc_edge_attr = torch.zeros(0, 1, device=device)

            hgt_emb = self.hgt_encoder(
                hgt_node_features, ccc_edge_index, ccc_edge_type, ccc_edge_attr,
            )
        else:
            hgt_emb = torch.zeros(B, self.n_cell_types, self.d_embed, device=device)

        # Branch 2: Cell transformer
        if self.use_cell_transformer:
            cell_emb, _ = self.cell_transformer(
                cell_data, cell_offsets, return_attention=False,
            )
        else:
            cell_emb = torch.zeros(B, self.n_cell_types, self.n_pma_seeds * self.d_embed, device=device)

        # Fusion + pathology + attention
        fused = self.fusion_layer(hgt_emb, cell_emb)
        path_emb = self.pathology_encoder(pathology, region_context)
        if self.use_pathology_attention:
            attended, attention_weights = self.pathology_attention(
                fused, path_emb, cell_type_mask=cell_type_mask,
                return_attention_weights=not self.training,
            )
        else:
            if cell_type_mask is not None:
                mask = cell_type_mask.unsqueeze(-1).float()
                attended = (fused * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                attended = fused.mean(dim=1)
            attention_weights = None

        return {"attended": attended, "attention_weights": attention_weights}

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
            Dict with 'scales' [n_layers, n_node_types], 'cell_types' list,
            and 'per_cell_type' {name: [n_layers]} mapping.
        """
        result = self.hgt_encoder.get_layer_scales()
        # Add cell type names for interpretability
        result['cell_types'] = list(self.node_types)
        scales = result['scales']
        result['per_cell_type'] = {
            ct: scales[:, idx].clone()
            for idx, ct in enumerate(self.node_types)
        }
        return result

    def num_parameters(self, trainable_only: bool = True) -> dict[str, int]:
        """
        Count parameters, total and per component.

        Args:
            trainable_only: If True, count only parameters with requires_grad.

        Returns:
            Dict with 'total' and per-component counts:
                'hgt_gene_gate', 'hgt_input_proj', 'region_handler',
                'hgt_encoder', 'cell_transformer', 'fusion_layer',
                'pathology_encoder', 'pathology_attention', 'prediction_head'.
        """
        def _count(module: nn.Module) -> int:
            return sum(
                p.numel() for p in module.parameters()
                if not trainable_only or p.requires_grad
            )

        components = {
            'hgt_gene_gate': self.hgt_gene_gate,
            'hgt_input_proj': self.hgt_input_proj,
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
            head, and fusion sub-configs.

    Returns:
        Configured CognitiveResilienceModel instance.
    """
    head_type = model_cfg.get("head", {}).get("type", "deterministic")
    use_bayesian = head_type == "bayesian"
    use_torch_compile = model_cfg.get("use_torch_compile", False)
    fusion_cfg = model_cfg.get("fusion", {})
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
        use_bayesian_head=use_bayesian,
        d_head_hidden=model_cfg.head.d_hidden,
        dropout=_cfg_get(model_cfg, "dropout", 0.1, "model"),
        n_pathology_features=_cfg_get(model_cfg.pathology_attention, "n_pathology_features", 3, "model.pathology_attention"),
        n_pma_seeds=_cfg_get(model_cfg.set_transformer, "n_pma_seeds", 1, "model.set_transformer"),
        use_hgt_encoder=model_cfg.get("use_hgt_encoder", True),
        use_cell_transformer=model_cfg.get("use_cell_transformer", True),
        use_pathology_attention=model_cfg.get("use_pathology_attention", True),
        condition_on_cell_type=_cfg_get(model_cfg.set_transformer, "condition_on_cell_type", True, "model.set_transformer"),
        use_gradient_checkpointing=model_cfg.get("use_gradient_checkpointing", False),
        use_torch_compile=use_torch_compile,
        target_mean=model_cfg.head.get("target_mean", 0.0) or 0.0,
        fusion_type=fusion_cfg.get("type", "concat"),
        fusion_n_heads=fusion_cfg.get("n_heads", 4),
    )
