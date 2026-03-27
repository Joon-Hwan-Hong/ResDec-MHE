"""HGTEncoderTensor — batched tensor-native HGT encoder.

Replaces HGTEncoder + HGTEncoderBatched with a single module that operates
on batched tensors [B, N, d] instead of per-sample dict-based processing.

Architecture:
    Input projection: [N, d_input, d_hidden] weight tensor + einsum
    Per-type LayerNorm: F.layer_norm + per-type ln_weight/ln_bias
    Layer loop: Pre-LN -> HGTConvTensor -> dropout -> LayerScale -> residual
    Output projection: optional [N, d_hidden, d_output] when dims differ
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


class HGTEncoderTensor(nn.Module):
    """Heterogeneous Graph Transformer encoder using batched tensor operations.

    Args:
        d_input: Input feature dimensionality.
        d_hidden: Hidden feature dimensionality.
        d_output: Output feature dimensionality.
        n_heads: Number of attention heads.
        n_layers: Number of HGT layers.
        n_node_types: Number of node types.
        n_edge_types: Number of edge/relation types.
        edge_dim: Dimensionality of edge attributes (default 1).
        dropout: Dropout rate.
        layer_scale_init: Initial value for LayerScale parameters.
        use_gradient_checkpointing: Whether to use gradient checkpointing on HGT layers.
    """

    def __init__(
        self,
        d_input: int,
        d_hidden: int,
        d_output: int,
        n_heads: int,
        n_layers: int,
        n_node_types: int,
        n_edge_types: int,
        edge_dim: int = 1,
        dropout: float = 0.0,
        layer_scale_init: float = 1.0,
        use_gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()

        # Validation
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
        self.n_node_types = n_node_types
        self.n_edge_types = n_edge_types
        self.edge_dim = edge_dim
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Input projection: batched einsum [N, d_input, d_hidden]
        self.input_proj_weight = nn.Parameter(torch.empty(n_node_types, d_input, d_hidden))
        self.input_proj_bias = nn.Parameter(torch.zeros(n_node_types, d_hidden))
        self.input_norm = nn.LayerNorm(d_hidden)
        self.dropout = nn.Dropout(dropout)

        # Per-type LayerNorm affine parameters: [n_layers, N, d_hidden]
        self.ln_weight = nn.Parameter(torch.ones(n_layers, n_node_types, d_hidden))
        self.ln_bias = nn.Parameter(torch.zeros(n_layers, n_node_types, d_hidden))

        # HGT convolution layers
        from src.models.components.hgt_conv_tensor import HGTConvTensor

        self.hgt_layers = nn.ModuleList([
            HGTConvTensor(
                in_channels=d_hidden,
                out_channels=d_hidden,
                n_node_types=n_node_types,
                n_edge_types=n_edge_types,
                heads=n_heads,
                edge_dim=edge_dim,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # LayerScale: per-cell-type learnable scaling [N] per layer
        self.layer_scales = nn.ParameterList([
            nn.Parameter(torch.ones(n_node_types) * layer_scale_init)
            for _ in range(n_layers)
        ])

        # Output projection (only when d_output != d_hidden)
        if d_output != d_hidden:
            self.output_proj_weight = nn.Parameter(
                torch.empty(n_node_types, d_hidden, d_output)
            )
            self.output_proj_bias = nn.Parameter(torch.zeros(n_node_types, d_output))
        else:
            self.output_proj_weight = None
            self.output_proj_bias = None

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.input_proj_weight)
        if self.output_proj_weight is not None:
            nn.init.xavier_uniform_(self.output_proj_weight)

    def _per_type_layer_norm(
        self, h: torch.Tensor, layer_idx: int
    ) -> torch.Tensor:
        """Apply per-type LayerNorm: shared normalization, per-type affine.

        Args:
            h: [B, N, d_hidden]
            layer_idx: Which layer's affine parameters to use.

        Returns:
            [B, N, d_hidden] normalized tensor.
        """
        h_normed = F.layer_norm(h, [self.d_hidden])
        return h_normed * self.ln_weight[layer_idx] + self.ln_bias[layer_idx]

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,  # [2, E_total]
        edge_type: torch.Tensor,   # [E_total]
        edge_attr: torch.Tensor | None,  # [E_total, 1] or None
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass with flat/concatenated edge format.

        Args:
            x: Node features [B, N, d_input]
            edge_index: [2, E_total] with batch-offset node indices
            edge_type: [E_total] edge type indices
            edge_attr: [E_total, edge_dim] or None
            return_attention: Whether to return per-layer attention weights

        Returns:
            Node outputs [B, N, d_output], or (outputs, list of attention)
        """
        attention_list = [] if return_attention else None

        # Input projection via batched einsum
        h = torch.einsum("bnd,ndo->bno", x, self.input_proj_weight) + self.input_proj_bias
        h = self.input_norm(h)
        h = self.dropout(h)

        # Layer loop: Pre-LN + HGTConvTensor + dropout + LayerScale + residual
        for layer_idx in range(self.n_layers):
            h_normed = self._per_type_layer_norm(h, layer_idx)

            if self.use_gradient_checkpointing and self.training:
                # Wrap HGT layer in checkpoint for memory savings.
                def _run_layer(
                    _h_normed, _edge_index, _edge_type, _edge_attr,
                    _layer_idx=layer_idx, _return_attention=return_attention,
                ):
                    layer = self.hgt_layers[_layer_idx]
                    return layer(
                        _h_normed, _edge_index, _edge_type, _edge_attr,
                        return_attention=_return_attention,
                    )

                result = torch.utils.checkpoint.checkpoint(
                    _run_layer,
                    h_normed, edge_index, edge_type, edge_attr,
                    use_reentrant=False,
                )
            else:
                result = self.hgt_layers[layer_idx](
                    h_normed, edge_index, edge_type, edge_attr,
                    return_attention=return_attention,
                )

            if return_attention:
                h_new, attn = result
                attention_list.append(attn)
            else:
                h_new = result

            # LayerScale + residual
            scale = self.layer_scales[layer_idx].view(1, -1, 1)  # [1, N, 1]
            h = h + scale * self.dropout(h_new)

        # Output projection (if needed)
        if self.output_proj_weight is not None:
            h = torch.einsum(
                "bnd,ndo->bno", h, self.output_proj_weight
            ) + self.output_proj_bias

        if return_attention:
            return h, attention_list
        return h

    def get_layer_scales(self) -> dict[str, torch.Tensor]:
        """Get LayerScale values for interpretability.

        Returns:
            Dict with 'scales': [n_layers, n_node_types] tensor.
        """
        scales = torch.stack([s.detach() for s in self.layer_scales], dim=0)
        return {"scales": scales}

    def get_mean_layer_scales(self) -> torch.Tensor:
        """Get mean LayerScale value per node type (averaged across layers).

        Returns:
            [n_node_types] tensor of mean scale values.
        """
        scales = torch.stack([s.detach() for s in self.layer_scales], dim=0)
        return scales.mean(dim=0)

    def extra_repr(self) -> str:
        return (
            f"d_input={self.d_input}, d_hidden={self.d_hidden}, "
            f"d_output={self.d_output}, n_heads={self.n_heads}, "
            f"n_layers={self.n_layers}, n_node_types={self.n_node_types}, "
            f"n_edge_types={self.n_edge_types}, edge_dim={self.edge_dim}"
        )
