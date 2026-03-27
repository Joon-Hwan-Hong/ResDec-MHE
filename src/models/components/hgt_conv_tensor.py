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

    def _scatter_softmax(
        self,
        scores: torch.Tensor,
        dst_idx: torch.Tensor,
        total_nodes: int,
    ) -> torch.Tensor:
        """Scatter softmax for flat (concatenated) edges.

        Args:
            scores: [E_total, H] attention scores
            dst_idx: [E_total] destination node indices (already batch-offset)
            total_nodes: Total number of nodes across all samples (B * N)

        Returns:
            [E_total, H] softmax weights
        """
        H = scores.size(-1)
        device = scores.device

        # Float32 for numerical stability
        scores_f32 = scores.float()

        # Max per target (numerically stable softmax)
        max_scores = torch.full(
            (total_nodes, H), float("-inf"), device=device, dtype=torch.float32
        )
        max_scores.scatter_reduce_(
            0,
            dst_idx[:, None].expand(-1, H),
            scores_f32,
            reduce="amax",
            include_self=True,
        )

        # Subtract max and exp (guard against -inf - (-inf) = NaN)
        gathered_max = max_scores[dst_idx]
        scores_norm = scores_f32 - gathered_max
        # Where max is -inf (no valid edges to that target), set to -inf to get exp=0
        scores_norm = torch.where(
            gathered_max == float("-inf"), scores_f32, scores_norm
        )
        exp_scores = torch.exp(scores_norm)

        # Sum per target
        sum_exp = torch.zeros(total_nodes, H, device=device, dtype=torch.float32)
        sum_exp.scatter_add_(0, dst_idx[:, None].expand(-1, H), exp_scores)

        # Normalize
        result = exp_scores / (sum_exp[dst_idx] + EPSILON_SOFTMAX)

        return result

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,  # [2, E_total]
        edge_type: torch.Tensor,   # [E_total]
        edge_attr: torch.Tensor | None,  # [E_total, 1] or None
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with flat/concatenated edge format.

        Args:
            x: Node features [B, N, d_in]
            edge_index: [2, E_total] with batch-offset node indices
            edge_type: [E_total] edge type indices
            edge_attr: [E_total, 1] edge attributes, or None
            return_attention: Whether to return attention weights

        Returns:
            Node outputs [B, N, d_out], or (outputs, attention [E_total, H])
        """
        B, N, D = x.shape
        E_total = edge_index.shape[1]

        # Handle zero edges case
        if E_total == 0:
            out = torch.zeros(B, N, self.out_channels, device=x.device, dtype=x.dtype)
            if return_attention:
                return out, torch.zeros(0, self.heads, device=x.device)
            return out

        total_nodes = B * N

        # 1. Batched Q/K/V projections via einsum
        q = torch.einsum("bnd,ndo->bno", x, self.q_weight) + self.q_bias  # [B, N, d_out]
        k = torch.einsum("bnd,ndo->bno", x, self.k_weight) + self.k_bias
        v = torch.einsum("bnd,ndo->bno", x, self.v_weight) + self.v_bias

        # Reshape for multi-head and flatten: [B*N, H, dk]
        q = q.view(B, N, self.heads, self.d_k).reshape(total_nodes, self.heads, self.d_k)
        k = k.view(B, N, self.heads, self.d_k).reshape(total_nodes, self.heads, self.d_k)
        v = v.view(B, N, self.heads, self.d_k).reshape(total_nodes, self.heads, self.d_k)

        # 2. Gather edge endpoints: [E_total, H, dk]
        src_idx = edge_index[0]  # [E_total]
        dst_idx = edge_index[1]  # [E_total]

        q_i = q[dst_idx]  # [E_total, H, dk]
        k_j = k[src_idx]  # [E_total, H, dk]
        v_j = v[src_idx]  # [E_total, H, dk]

        # 3. Relation-specific attention — all edge types at once.
        # einsum: edges have no B dim → "ehd,nhdk->nehk"
        all_k = torch.einsum("ehd,nhdk->nehk", k_j, self.w_att)  # [n_et, E_total, H, dk]
        e_idx = torch.arange(E_total, device=x.device)
        k_transformed = all_k[edge_type, e_idx]  # [E_total, H, dk]
        del all_k
        attn_scores = (q_i * k_transformed).sum(dim=-1) / (self.d_k**0.5)  # [E_total, H]

        # Edge attribute bias
        if self.edge_lin is not None and edge_attr is not None:
            edge_bias = self.edge_lin(edge_attr)  # [E_total, H]
            attn_scores = attn_scores + edge_bias

        # 4. Scatter softmax (no masking needed — all edges are valid)
        attn_weights = self._scatter_softmax(attn_scores, dst_idx, total_nodes)

        # Extract attention BEFORE dropout
        if return_attention:
            attn_out = attn_weights.detach()

        attn_weights = self.dropout(attn_weights)

        # 5. Relation-specific message — all types at once
        all_v = torch.einsum("ehd,nhdk->nehk", v_j, self.w_msg)  # [n_et, E_total, H, dk]
        v_transformed = all_v[edge_type, e_idx]  # [E_total, H, dk]
        del all_v
        messages = attn_weights.unsqueeze(-1) * v_transformed  # [E_total, H, dk]

        # Edge scaling
        if self.edge_scale_lin is not None and edge_attr is not None:
            edge_scale = torch.sigmoid(self.edge_scale_lin(edge_attr))  # [E_total, 1]
            messages = messages * edge_scale.unsqueeze(-1)

        # 6. Scatter add to destinations
        messages_flat = messages.reshape(E_total, self.out_channels)  # [E_total, d_out]
        out = torch.zeros(total_nodes, self.out_channels, device=x.device, dtype=torch.float32)
        out.scatter_add_(
            0,
            dst_idx[:, None].expand(-1, self.out_channels),
            messages_flat,
        )

        # Received mask: nodes that received at least one edge message
        received_count = torch.zeros(total_nodes, dtype=torch.float32, device=x.device)
        received_count.scatter_add_(0, dst_idx, torch.ones(E_total, device=x.device))
        received = received_count > 0  # [B*N]

        # Reshape to [B, N, d_out]
        out = out.reshape(B, N, self.out_channels)
        received = received.reshape(B, N)

        # Output projection + masking
        input_dtype = x.dtype
        out = self.out_lin(out.to(input_dtype))
        out = out * received.unsqueeze(-1).to(out.dtype)

        if return_attention:
            return out, attn_out
        return out
