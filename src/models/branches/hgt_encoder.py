"""
Heterogeneous Graph Transformer (HGT) Encoder for cell-cell communication.

Encodes cell-cell communication context using LIANA+ interaction scores
via custom HGTConvWithEdgeAttr, which implements type-aware attention
with edge feature support.

Architecture:
    Input: x_dict {cell_type: [1, d_input]} + edge_index_dict + edge_attr_dict
    → Input projection per node type
    → HGTConvWithEdgeAttr layers × n_layers (Pre-LN + LayerScale + residual)
    → Output: x_dict {cell_type: [1, d_output]}

Key features:
    - True heterogeneous: 31 cell types as distinct node types
    - Edge attributes: LIANA magnitude scores as attention bias
    - Pre-LN architecture: LayerNorm before HGT layer, direct residual add after
      (Xiong et al., 2020 - "On Layer Normalization in the Transformer Architecture")
      This ensures proper gradient flow through the message passing pathway.
    - LayerScale: Per-cell-type learnable scaling for message contribution
      After training, values show how much each cell type relies on communication.
    - Interpretable: Attention weights and LayerScale values extractable
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

from src.data.constants import ALL_EDGE_TYPES, CELL_TYPE_ORDER, sanitize_key
from src.models.components.hgt_conv import HGTConvWithEdgeAttr

logger = logging.getLogger(__name__)


class HGTEncoder(nn.Module):
    """
    Heterogeneous Graph Transformer encoder for cell-cell communication.

    Uses custom HGTConvWithEdgeAttr which implements type-specific projections,
    relation-specific attention, and edge feature support for LIANA magnitudes.

    Input Preparation:
        In the full model, HGT consumes ENCODED pseudobulk features from
        PseudobulkEncoder, not raw gene expression. The typical flow:

            pseudobulk_emb = pseudobulk_encoder(batch["pseudobulk"])  # [B, 31, d_embed]
            x_dict_list = build_x_dict_list_from_embeddings(pseudobulk_emb, node_types)
            hgt_out = hgt_encoder.forward_batched(x_dict_list, edge_dicts, ...)

        This ensures gene attention gating (from PseudobulkEncoder) influences
        the HGT's view of cell state, and keeps d_input = d_embed (typically 128)
        rather than n_genes (~4000).

    Args:
        d_input: Input feature dimension. Should match PseudobulkEncoder's d_embed
            (typically 128). This is the dimension of encoded pseudobulk embeddings.
        d_hidden: Hidden dimension for HGT layers
        d_output: Output dimension
        n_heads: Number of attention heads
        n_layers: Number of HGT layers
        dropout: Dropout probability
        edge_dim: Dimension of edge features (default: 1 for LIANA magnitude)
        node_types: List of cell type names (default: Allen ABC 31 types)
        edge_categories: List of edge category names (default: CellChatDB categories)
        layer_scale_init: Initial value for LayerScale parameters (default: 1.0)
            Controls how much message (communication) contributes vs residual.
            With Pre-LN architecture, can start at 1.0 (balanced contribution).
            Values are per-cell-type and learnable during training.
            After training, values show how much each cell type relies on communication.

    Shape:
        - x_dict: {cell_type: (1, d_input)} node features per cell type
        - edge_index_dict: {(src, rel, dst): (2, n_edges)} edges per type
        - edge_attr_dict: {(src, rel, dst): (n_edges, edge_dim)} edge features
        - Output: {cell_type: (1, d_output)} updated node features

    Attributes:
        layer_scales: nn.ParameterList of [n_node_types] tensors per layer.
            Each value represents how much that cell type relies on communication.
            Higher values = more reliance on cell-cell communication.
            Extractable via get_layer_scales() for interpretability analysis.
    """

    def __init__(
        self,
        d_input: int,
        d_hidden: int = 128,
        d_output: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        edge_dim: int = 1,
        node_types: Optional[list[str]] = None,
        edge_categories: Optional[list[str]] = None,
        layer_scale_init: float = 1.0,
    ):
        super().__init__()

        if d_input <= 0:
            raise ValueError(f"d_input must be positive, got {d_input}")
        if d_hidden <= 0:
            raise ValueError(f"d_hidden must be positive, got {d_hidden}")
        if d_output <= 0:
            raise ValueError(f"d_output must be positive, got {d_output}")
        if n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {n_heads}")
        if n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {n_layers}")
        if d_hidden % n_heads != 0:
            raise ValueError(
                f"d_hidden ({d_hidden}) must be divisible by n_heads ({n_heads})"
            )

        self.d_input = d_input
        self.d_hidden = d_hidden
        self.d_output = d_output
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.edge_dim = edge_dim
        self.layer_scale_init = layer_scale_init

        # Node types (31 cell types - each is a distinct node type)
        self.node_types = node_types if node_types is not None else list(CELL_TYPE_ORDER)
        self.n_node_types = len(self.node_types)

        # Edge categories from CellChatDB
        self.edge_categories = (
            edge_categories if edge_categories is not None else list(ALL_EDGE_TYPES)
        )
        self.n_edge_types = len(self.edge_categories)

        # Input projection per node type (to d_hidden)
        self.input_projs = nn.ModuleDict({
            sanitize_key(node_type): nn.Linear(d_input, d_hidden)
            for node_type in self.node_types
        })
        self.input_norms = nn.ModuleDict({
            sanitize_key(node_type): nn.LayerNorm(d_hidden)
            for node_type in self.node_types
        })

        # Mapping from original names to sanitized keys
        self._node_type_to_key = {
            nt: sanitize_key(nt) for nt in self.node_types
        }

        # Reverse mapping: sanitized key -> node type index (for LayerScale lookup)
        # This allows handling both sanitized and unsanitized input keys
        self._key_to_node_idx = {
            sanitize_key(nt): idx for idx, nt in enumerate(self.node_types)
        }
        # Also map original names to indices for unsanitized input
        self._key_to_node_idx.update({
            nt: idx for idx, nt in enumerate(self.node_types)
        })

        # HGT layers with edge attribute support
        self.hgt_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for _ in range(n_layers):
            layer = HGTConvWithEdgeAttr(
                in_channels=d_hidden,
                out_channels=d_hidden,
                node_types=self.node_types,
                edge_categories=self.edge_categories,
                heads=n_heads,
                edge_dim=edge_dim,
                dropout=dropout,
            )
            self.hgt_layers.append(layer)
            # LayerNorm per node type
            self.layer_norms.append(nn.ModuleDict({
                sanitize_key(nt): nn.LayerNorm(d_hidden)
                for nt in self.node_types
            }))

        # LayerScale: per-cell-type learnable scaling for message contribution
        # With Pre-LN, initialized to 1.0 (balanced contribution) since gradient
        # flow is already stable. After training, values show how much each
        # cell type relies on communication vs intrinsic state.
        # Shape: [n_layers, n_node_types] - one scale per cell type per layer
        self.layer_scales = nn.ParameterList([
            nn.Parameter(torch.ones(self.n_node_types) * layer_scale_init)
            for _ in range(n_layers)
        ])

        # Output projection per node type (if d_output != d_hidden)
        if d_output != d_hidden:
            self.output_projs = nn.ModuleDict({
                sanitize_key(nt): nn.Linear(d_hidden, d_output)
                for nt in self.node_types
            })
        else:
            self.output_projs = None

        self.dropout = nn.Dropout(dropout)

        # Create mapping from edge category to index
        self.edge_category_to_idx = {
            cat: idx for idx, cat in enumerate(self.edge_categories)
        }

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
        edge_attr_dict: Optional[dict[tuple[str, str, str], torch.Tensor]] = None,
        return_attention: bool = False,
    ) -> tuple[dict[str, torch.Tensor], Optional[list[dict]]]:
        """
        Forward pass through HGT encoder.

        Args:
            x_dict: Node features per cell type {cell_type: (n_nodes, d_input)}
            edge_index_dict: Edge indices {(src, rel, dst): (2, n_edges)}
            edge_attr_dict: Edge features {(src, rel, dst): (n_edges, edge_dim)}
            return_attention: Whether to return attention weights per layer

        Returns:
            output_dict: Updated node features {cell_type: (n_nodes, d_output)}
            attention_weights: List of attention dicts per layer (if return_attention)
        """
        attention_weights = [] if return_attention else None

        # Input projection per node type
        h_dict = {}
        for node_type, x in x_dict.items():
            key = self._node_type_to_key.get(node_type, sanitize_key(node_type))
            if key in self.input_projs:
                h = self.input_projs[key](x)
                h = self.input_norms[key](h)
                h = self.dropout(h)
                h_dict[node_type] = h
            else:
                # Node type not in our learned types, skip
                pass

        # Apply HGT layers with Pre-LN + LayerScale
        # Pre-LN: normalize BEFORE the HGT layer, add after (no norm on residual path)
        # This ensures proper gradient flow (Xiong et al., 2020)
        # LayerScale: optional per-cell-type scaling for interpretability
        for layer_idx, (hgt_layer, norms, scales) in enumerate(
            zip(self.hgt_layers, self.layer_norms, self.layer_scales)
        ):
            # Pre-LN: normalize h_dict BEFORE passing to HGT layer
            h_normed_dict = {}
            for node_type in h_dict:
                key = self._node_type_to_key.get(node_type, sanitize_key(node_type))
                h_normed_dict[node_type] = norms[key](h_dict[node_type])

            # HGTConv forward on normalized features
            h_new_dict, attn = hgt_layer(
                h_normed_dict,
                edge_index_dict,
                edge_attr_dict,
                return_attention=return_attention,
            )

            if return_attention and attn is not None:
                attention_weights.append(attn)

            # Residual + LayerScale(message) per node type
            # Pre-LN: NO LayerNorm after the add - gradient flows directly!
            # NOTE: Iterate over h_dict.keys() (not self.node_types) to handle both
            # sanitized keys (from collate_for_hgt) and unsanitized keys.
            for node_key in list(h_dict.keys()):
                # Look up node index for LayerScale (handles both sanitized/unsanitized)
                node_type_idx = self._key_to_node_idx.get(node_key)
                if node_type_idx is None:
                    # Unknown node type, skip
                    continue

                h_new = h_new_dict.get(node_key, torch.zeros_like(h_dict[node_key]))

                # Apply LayerScale: scale the message contribution per cell type
                # Initialized to 1.0 for Pre-LN (we're not trying to start at identity)
                scale = scales[node_type_idx]
                scaled_message = scale * self.dropout(h_new)

                # Residual + scaled message (no norm after - this is Pre-LN!)
                h_dict[node_key] = h_dict[node_key] + scaled_message

        # Output projection
        output_dict = {}
        for node_type, h in h_dict.items():
            if self.output_projs is not None:
                key = self._node_type_to_key.get(node_type, sanitize_key(node_type))
                output_dict[node_type] = self.output_projs[key](h)
            else:
                output_dict[node_type] = h

        return output_dict, attention_weights

    def get_edge_type_index(self, category: str) -> int:
        """
        Get the index for an edge category.

        Args:
            category: Edge category name (e.g., "Secreted_Signaling")

        Returns:
            Integer index for the category
        """
        if category not in self.edge_category_to_idx:
            raise ValueError(
                f"Unknown edge category: {category}. "
                f"Valid categories: {list(self.edge_category_to_idx.keys())}"
            )
        return self.edge_category_to_idx[category]

    def get_layer_scales(self) -> dict[str, torch.Tensor]:
        """
        Get LayerScale values for interpretability analysis.

        Returns a dict mapping cell type names to their scale values across layers.
        Higher values indicate the cell type relies more on communication.

        Returns:
            Dict with keys:
                - 'scales': Tensor of shape [n_layers, n_node_types]
                - 'cell_types': List of cell type names (order matches dim 1)
                - 'per_cell_type': Dict mapping cell type -> [n_layers] scale values
        """
        # Stack all layer scales: [n_layers, n_node_types]
        scales = torch.stack([s.detach() for s in self.layer_scales], dim=0)

        # Create per-cell-type mapping
        per_cell_type = {
            cell_type: scales[:, idx].clone()
            for idx, cell_type in enumerate(self.node_types)
        }

        return {
            'scales': scales,
            'cell_types': list(self.node_types),
            'per_cell_type': per_cell_type,
        }

    def get_mean_layer_scales(self) -> dict[str, float]:
        """
        Get mean LayerScale value per cell type (averaged across layers).

        Useful for quick interpretability: which cell types rely most on communication?

        Returns:
            Dict mapping cell type name -> mean scale value (float)
        """
        scales_info = self.get_layer_scales()
        mean_scales = scales_info['scales'].mean(dim=0)  # [n_node_types]

        return {
            cell_type: mean_scales[idx].item()
            for idx, cell_type in enumerate(self.node_types)
        }

    def extra_repr(self) -> str:
        return (
            f"d_input={self.d_input}, d_hidden={self.d_hidden}, "
            f"d_output={self.d_output}, n_heads={self.n_heads}, "
            f"n_layers={self.n_layers}, n_node_types={self.n_node_types}, "
            f"n_edge_types={self.n_edge_types}, edge_dim={self.edge_dim}, "
            f"layer_scale_init={self.layer_scale_init}"
        )


class HGTEncoderBatched(nn.Module):
    """
    Batched wrapper for HGTEncoder that handles batch dimension.

    For batch processing, each subject has its own graph. This wrapper
    processes them sequentially and stacks results.

    Args:
        Same as HGTEncoder

    Shape:
        - x_dict_list: List of {cell_type: (1, d_input)} per batch sample
        - edge_index_dict_list: List of edge_index_dict per sample
        - edge_attr_dict_list: List of edge_attr_dict per sample
        - Output: {cell_type: (batch, 1, d_output)} stacked outputs
    """

    def __init__(
        self,
        d_input: int,
        d_hidden: int = 128,
        d_output: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        edge_dim: int = 1,
        node_types: Optional[list[str]] = None,
        edge_categories: Optional[list[str]] = None,
        layer_scale_init: float = 1.0,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.encoder = HGTEncoder(
            d_input=d_input,
            d_hidden=d_hidden,
            d_output=d_output,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            edge_dim=edge_dim,
            node_types=node_types,
            edge_categories=edge_categories,
            layer_scale_init=layer_scale_init,
        )

    def forward(
        self,
        x_dict_list: list[dict[str, torch.Tensor]],
        edge_index_dict_list: list[dict[tuple[str, str, str], torch.Tensor]],
        edge_attr_dict_list: Optional[list[dict[tuple[str, str, str], torch.Tensor]]] = None,
        return_attention: bool = False,
    ) -> tuple[dict[str, torch.Tensor], Optional[list]]:
        """
        Forward pass for batched inputs.

        Args:
            x_dict_list: List of x_dict, one per batch sample
            edge_index_dict_list: List of edge_index_dict per sample
            edge_attr_dict_list: List of edge_attr_dict per sample (optional)
            return_attention: Whether to return attention weights

        Returns:
            output_dict: {cell_type: (batch, n_nodes, d_output)} stacked
            attention: List of attention weights per sample (if return_attention)
        """
        batch_size = len(x_dict_list)
        if len(edge_index_dict_list) != batch_size:
            raise ValueError(
                f"edge_index_dict_list length ({len(edge_index_dict_list)}) "
                f"must match batch size ({batch_size})"
            )
        if edge_attr_dict_list is not None and len(edge_attr_dict_list) != batch_size:
            raise ValueError(
                f"edge_attr_dict_list length ({len(edge_attr_dict_list)}) "
                f"must match batch size ({batch_size})"
            )

        # Process each sample independently through HGTEncoder.
        # Per-sample loop is required because each subject has a different graph
        # topology (different edge types present, different number of edges).
        # Heterogeneous graphs with variable topology cannot be batched without
        # padding to a supergraph, which would waste more compute for sparse graphs
        # than the sequential approach.
        # Memory: at B=16, d_embed=128, 31 nodes, 2 layers, per-sample activations
        # are ~32KB → total ~512KB, negligible on any modern GPU.
        # For production optimization if scale increases:
        # (1) torch.compile on self.encoder can fuse small ops and reduce Python overhead
        # (2) gradient checkpointing per sample can trade compute for memory
        # (3) if graphs become more uniform, a padded-supergraph approach could batch
        outputs_list = []
        all_attention = [] if return_attention else None

        for i in range(batch_size):
            edge_attr = edge_attr_dict_list[i] if edge_attr_dict_list else None
            if self.use_gradient_checkpointing and self.training:
                # Trade compute for memory: recompute activations during backward.
                # use_reentrant=False is required for dict inputs/outputs.
                out_dict, attn = torch.utils.checkpoint.checkpoint(
                    self.encoder,
                    x_dict_list[i],
                    edge_index_dict_list[i],
                    edge_attr,
                    return_attention,
                    use_reentrant=False,
                )
            else:
                out_dict, attn = self.encoder(
                    x_dict_list[i],
                    edge_index_dict_list[i],
                    edge_attr,
                    return_attention=return_attention,
                )
            outputs_list.append(out_dict)
            if return_attention:
                all_attention.append(attn)

        # Stack outputs per node type
        # Collect all node types across samples and validate consistency
        all_node_types = set()
        for out_dict in outputs_list:
            all_node_types.update(out_dict.keys())

        # Fill missing node types with zero embeddings so batch stacking succeeds
        for i, out_dict in enumerate(outputs_list):
            missing = all_node_types - set(out_dict.keys())
            if missing:
                logger.debug(
                    "Sample %d missing node types %s, filling with zeros", i, missing
                )
                for node_type in missing:
                    # Get embedding dim from any existing output
                    any_existing = next(iter(out_dict.values()))
                    out_dict[node_type] = torch.zeros_like(any_existing)

        output_dict = {}
        for node_type in all_node_types:
            tensors = [out_dict[node_type] for out_dict in outputs_list]
            output_dict[node_type] = torch.stack(tensors, dim=0)

        return output_dict, all_attention

    @property
    def n_edge_types(self) -> int:
        """Number of edge types."""
        return self.encoder.n_edge_types

    @property
    def n_node_types(self) -> int:
        """Number of node types."""
        return self.encoder.n_node_types

    @property
    def node_types(self) -> list[str]:
        """List of node type names."""
        return self.encoder.node_types

    @property
    def edge_categories(self) -> list[str]:
        """List of edge category names."""
        return self.encoder.edge_categories

    def get_edge_type_index(self, category: str) -> int:
        """Get edge type index for a category."""
        return self.encoder.get_edge_type_index(category)

    def get_layer_scales(self) -> dict[str, torch.Tensor]:
        """Get LayerScale values for interpretability. See HGTEncoder.get_layer_scales."""
        return self.encoder.get_layer_scales()

    def get_mean_layer_scales(self) -> dict[str, float]:
        """Get mean LayerScale per cell type. See HGTEncoder.get_mean_layer_scales."""
        return self.encoder.get_mean_layer_scales()

    def extra_repr(self) -> str:
        return self.encoder.extra_repr()
