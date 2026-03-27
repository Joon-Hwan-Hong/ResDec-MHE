"""HGTConvTensor — batched tensor-native HGT convolution.

Replaces dict-based HGTConvWithEdgeAttr with fully tensorized operations.
All node-type and edge-type specific projections use batched tensor ops
instead of nn.ModuleDict with per-type nn.Linear modules.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.data.constants import EPSILON_SOFTMAX


class HGTConvTensor(nn.Module):
    """Heterogeneous Graph Transformer convolution using batched tensor operations.

    Args:
        in_channels: Input feature dimensionality.
        out_channels: Output feature dimensionality.
        n_node_types: Number of node types.
        n_edge_types: Number of edge/relation types.
        heads: Number of attention heads.
        edge_dim: Dimensionality of edge attributes. None to disable edge features.
        dropout: Dropout rate on attention weights.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_node_types: int,
        n_edge_types: int,
        heads: int = 1,
        edge_dim: int | None = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # Validation
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if out_channels <= 0:
            raise ValueError("out_channels must be positive")
        if n_node_types <= 0:
            raise ValueError("n_node_types must be positive")
        if n_edge_types <= 0:
            raise ValueError("n_edge_types must be positive")
        if heads <= 0:
            raise ValueError("heads must be positive")
        if out_channels % heads != 0:
            raise ValueError(
                f"out_channels ({out_channels}) must be divisible by heads ({heads})"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_node_types = n_node_types
        self.n_edge_types = n_edge_types
        self.heads = heads
        self.d_k = out_channels // heads

        # Node-type-specific Q/K/V projections: [n_node_types, in_channels, out_channels]
        self.q_weight = nn.Parameter(torch.empty(n_node_types, in_channels, out_channels))
        self.q_bias = nn.Parameter(torch.zeros(n_node_types, out_channels))
        self.k_weight = nn.Parameter(torch.empty(n_node_types, in_channels, out_channels))
        self.k_bias = nn.Parameter(torch.zeros(n_node_types, out_channels))
        self.v_weight = nn.Parameter(torch.empty(n_node_types, in_channels, out_channels))
        self.v_bias = nn.Parameter(torch.zeros(n_node_types, out_channels))

        # Relation-specific attention and message weights
        self.w_att = nn.Parameter(torch.empty(n_edge_types, heads, self.d_k, self.d_k))
        self.w_msg = nn.Parameter(torch.empty(n_edge_types, heads, self.d_k, self.d_k))

        # Edge attribute projections
        if edge_dim is not None:
            self.edge_lin = nn.Linear(edge_dim, heads)
            self.edge_scale_lin = nn.Linear(edge_dim, 1)
        else:
            self.edge_lin = None
            self.edge_scale_lin = None

        # Output projection
        self.out_lin = nn.Linear(out_channels, out_channels)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize parameters following the original HGTConvWithEdgeAttr."""
        nn.init.xavier_uniform_(self.q_weight)
        nn.init.xavier_uniform_(self.k_weight)
        nn.init.xavier_uniform_(self.v_weight)
        nn.init.xavier_uniform_(self.w_att)
        nn.init.xavier_uniform_(self.w_msg)

        if self.edge_lin is not None:
            # Positive init for edge_lin: uniform 0.1-0.5 weight, zero bias
            nn.init.uniform_(self.edge_lin.weight, 0.1, 0.5)
            nn.init.zeros_(self.edge_lin.bias)

        if self.edge_scale_lin is not None:
            # edge_scale_lin: uniform 0.5-1.0 weight, zero bias
            nn.init.uniform_(self.edge_scale_lin.weight, 0.5, 1.0)
            nn.init.zeros_(self.edge_scale_lin.bias)

    def _batched_softmax_by_target(
        self,
        scores: torch.Tensor,
        dst_idx: torch.Tensor,
        edge_mask: torch.Tensor,
        B: int,
        N: int,
    ) -> torch.Tensor:
        """Batched scatter softmax grouped by (sample, destination node).

        Args:
            scores: [B, E, H] attention scores (padded edges have -inf)
            dst_idx: [B, E] destination node indices
            edge_mask: [B, E] True for valid edges
            B: batch size
            N: number of node types

        Returns:
            [B, E, H] softmax weights (0 for padding edges)
        """
        H = scores.size(-1)
        E = scores.size(1)
        device = scores.device

        # Flatten batch: offset destinations by batch index
        batch_offset = torch.arange(B, device=device).unsqueeze(1) * N  # [B, 1]
        dst_flat = (dst_idx + batch_offset).reshape(-1)  # [B*E]
        scores_flat = scores.reshape(-1, H)  # [B*E, H]
        total_nodes = B * N

        # Float32 for numerical stability
        scores_f32 = scores_flat.float()

        # Max per target (numerically stable softmax)
        max_scores = torch.full(
            (total_nodes, H), float("-inf"), device=device, dtype=torch.float32
        )
        max_scores.scatter_reduce_(
            0,
            dst_flat[:, None].expand(-1, H),
            scores_f32,
            reduce="amax",
            include_self=True,
        )

        # Subtract max and exp (guard against -inf - (-inf) = NaN)
        gathered_max = max_scores[dst_flat]
        scores_norm = scores_f32 - gathered_max
        # Where max is -inf (no valid edges to that target), set to -inf to get exp=0
        scores_norm = torch.where(
            gathered_max == float("-inf"), scores_f32, scores_norm
        )
        exp_scores = torch.exp(scores_norm)

        # Sum per target
        sum_exp = torch.zeros(total_nodes, H, device=device, dtype=torch.float32)
        sum_exp.scatter_add_(0, dst_flat[:, None].expand(-1, H), exp_scores)

        # Normalize
        result = exp_scores / (sum_exp[dst_flat] + EPSILON_SOFTMAX)

        return result.reshape(B, E, H)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_attr: torch.Tensor | None,
        edge_counts: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Node features [B, N, d_in]
            edge_index: Edge indices [B, 2, E] (src, dst)
            edge_type: Edge type indices [B, E]
            edge_attr: Edge attributes [B, E, edge_dim] or None
            edge_counts: Number of valid edges per sample [B]
            return_attention: Whether to return attention weights

        Returns:
            Node outputs [B, N, d_out], or (outputs, attention [B, E, H])
        """
        B, N, D = x.shape
        E = edge_index.shape[2]

        # Handle zero edges case
        if E == 0:
            out = torch.zeros(B, N, self.out_channels, device=x.device, dtype=x.dtype)
            if return_attention:
                return out, torch.zeros(B, 0, self.heads, device=x.device)
            return out

        # 1. Batched Q/K/V projections via einsum
        q = torch.einsum("bnd,ndo->bno", x, self.q_weight) + self.q_bias  # [B, N, d_out]
        k = torch.einsum("bnd,ndo->bno", x, self.k_weight) + self.k_bias
        v = torch.einsum("bnd,ndo->bno", x, self.v_weight) + self.v_bias

        # Reshape for multi-head: [B, N, H, dk]
        q = q.view(B, N, self.heads, self.d_k)
        k = k.view(B, N, self.heads, self.d_k)
        v = v.view(B, N, self.heads, self.d_k)

        # 2. Gather edge endpoints: [B, E, H, dk]
        src_idx = edge_index[:, 0, :]  # [B, E]
        dst_idx = edge_index[:, 1, :]  # [B, E]

        def idx_expand(idx: torch.Tensor) -> torch.Tensor:
            return idx[:, :, None, None].expand(-1, -1, self.heads, self.d_k)

        q_i = torch.gather(q, 1, idx_expand(dst_idx))
        k_j = torch.gather(k, 1, idx_expand(src_idx))
        v_j = torch.gather(v, 1, idx_expand(src_idx))

        # 3. Relation-specific attention (loop over edge types to avoid
        #    materializing [B, E, H, dk, dk] which OOMs with large E)
        k_transformed = torch.zeros_like(k_j)  # [B, E, H, dk]
        for et in range(self.n_edge_types):
            mask = (edge_type == et).unsqueeze(-1).unsqueeze(-1)  # [B, E, 1, 1]
            # w_att[et]: [H, dk, dk], k_j: [B, E, H, dk] -> [B, E, H, dk]
            contrib = torch.einsum("behd,hdk->behk", k_j, self.w_att[et])
            k_transformed = torch.where(mask, contrib, k_transformed)
        attn_scores = (q_i * k_transformed).sum(dim=-1) / (self.d_k**0.5)  # [B, E, H]

        # Edge attribute bias
        if self.edge_lin is not None and edge_attr is not None:
            edge_bias = self.edge_lin(edge_attr)  # [B, E, H]
            attn_scores = attn_scores + edge_bias

        # 4. Edge validity masking
        edge_mask = (
            torch.arange(E, device=x.device).unsqueeze(0) < edge_counts.unsqueeze(1)
        )  # [B, E]
        attn_scores = attn_scores.masked_fill(~edge_mask.unsqueeze(-1), float("-inf"))

        # 5. Batched scatter softmax (float32 precision)
        attn_weights = self._batched_softmax_by_target(
            attn_scores, dst_idx, edge_mask, B, N
        )

        # Extract attention BEFORE dropout
        if return_attention:
            attn_out = attn_weights.detach()

        attn_weights = self.dropout(attn_weights)

        # 6. Relation-specific message (loop over edge types, same reason as above)
        v_transformed = torch.zeros_like(v_j)  # [B, E, H, dk]
        for et in range(self.n_edge_types):
            mask = (edge_type == et).unsqueeze(-1).unsqueeze(-1)  # [B, E, 1, 1]
            contrib = torch.einsum("behd,hdk->behk", v_j, self.w_msg[et])
            v_transformed = torch.where(mask, contrib, v_transformed)
        messages = attn_weights.unsqueeze(-1) * v_transformed  # [B, E, H, dk]

        # Edge scaling
        if self.edge_scale_lin is not None and edge_attr is not None:
            edge_scale = torch.sigmoid(self.edge_scale_lin(edge_attr))  # [B, E, 1]
            messages = messages * edge_scale.unsqueeze(-1)

        # Mask padding edges
        messages = messages * edge_mask[:, :, None, None]

        # 7. Scatter add to destinations
        messages_flat = messages.reshape(B, E, self.out_channels)
        out = torch.zeros(B, N, self.out_channels, device=x.device, dtype=torch.float32)
        out.scatter_add_(
            1,
            dst_idx[:, :, None].expand(-1, -1, self.out_channels),
            messages_flat,
        )

        # Received mask: only nodes that received at least one valid edge message
        # Use scatter_add on a float tensor to count valid edges per dst node
        received_count = torch.zeros(B, N, dtype=torch.float32, device=x.device)
        edge_mask_float = edge_mask.float()  # [B, E]
        received_count.scatter_add_(
            1, dst_idx, edge_mask_float
        )
        received = received_count > 0

        # Output projection + masking
        input_dtype = x.dtype
        out = self.out_lin(out.to(input_dtype))
        out = out * received.unsqueeze(-1).to(out.dtype)

        if return_attention:
            return out, attn_out
        return out
