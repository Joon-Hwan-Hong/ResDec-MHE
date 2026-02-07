"""
Custom Heterogeneous Graph Transformer convolution with edge attribute support.

Implements HGT (Hu et al., 2020) with extensions for:
- Edge features projected to attention bias (for LIANA magnitude scores)
- Per-node-type Q/K/V projections (for cell-type-specific representations)
- Per-relation attention and message weights (for CellChatDB categories)
- Attention weight extraction (for interpretability)
"""

from typing import Optional

import torch
import torch.nn as nn

from src.data.constants import sanitize_key, EPSILON_SOFTMAX

# Verify PyTorch version for scatter_reduce_ support
_torch_version = tuple(int(x) for x in torch.__version__.split('.')[:2])
if _torch_version < (2, 1):
    raise ImportError(
        f"HGTConvWithEdgeAttr requires PyTorch >= 2.1 for scatter_reduce_ support, "
        f"got {torch.__version__}"
    )


class HGTConvWithEdgeAttr(nn.Module):
    """
    Heterogeneous Graph Transformer convolution with edge attribute support.

    Architecture:
        1. Type-specific projections: Q, K, V per node type
        2. Relation-specific attention: W_ATT per edge category
        3. Relation-specific message: W_MSG per edge category
        4. Edge attribute bias: projects edge features to attention bias

    The attention mechanism computes:
        α = softmax((K · W_ATT · Q^T) / √d_k + edge_bias)

    Where edge_bias = Linear(edge_attr) allows LIANA magnitude scores
    to modulate attention strength.

    Args:
        in_channels: Input feature dimension
        out_channels: Output feature dimension
        node_types: List of node type names (e.g., 31 cell types)
        edge_categories: List of edge category names (e.g., 5 CellChatDB types)
        heads: Number of attention heads (default: 4)
        edge_dim: Dimension of edge features, None for no edge features
        dropout: Dropout probability (default: 0.1)

    Shape:
        - x_dict: {node_type: (n_nodes, in_channels)}
        - edge_index_dict: {(src_type, rel, dst_type): (2, n_edges)}
        - edge_attr_dict: {(src_type, rel, dst_type): (n_edges, edge_dim)}
        - Output: {node_type: (n_nodes, out_channels)}
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        node_types: list[str],
        edge_categories: list[str],
        heads: int = 4,
        edge_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}")
        if heads <= 0:
            raise ValueError(f"heads must be positive, got {heads}")
        if out_channels % heads != 0:
            raise ValueError(
                f"out_channels ({out_channels}) must be divisible by heads ({heads})"
            )
        if len(node_types) == 0:
            raise ValueError("node_types must not be empty")
        if len(edge_categories) == 0:
            raise ValueError("edge_categories must not be empty")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.edge_dim = edge_dim
        self.d_k = out_channels // heads  # Dimension per head
        self.node_types = list(node_types)
        self.edge_categories = list(edge_categories)

        # ─────────────────────────────────────────────────────────────────────
        # 1. Type-specific node projections (Q, K, V per node type)
        # ─────────────────────────────────────────────────────────────────────
        # Each cell type gets its own learned projection
        # This captures cell-type-specific representation learning

        self.q_lin = nn.ModuleDict({
            sanitize_key(node_type): nn.Linear(in_channels, out_channels)
            for node_type in node_types
        })

        self.k_lin = nn.ModuleDict({
            sanitize_key(node_type): nn.Linear(in_channels, out_channels)
            for node_type in node_types
        })

        self.v_lin = nn.ModuleDict({
            sanitize_key(node_type): nn.Linear(in_channels, out_channels)
            for node_type in node_types
        })

        # ─────────────────────────────────────────────────────────────────────
        # 2. Relation-specific attention weights (W_ATT per edge category)
        # ─────────────────────────────────────────────────────────────────────
        # Each CellChatDB category (Secreted_Signaling, ECM_Receptor, etc.)
        # gets its own attention transformation
        # Shape: [heads, d_k, d_k] - transforms K before dot product with Q

        self.w_att = nn.ParameterDict({
            sanitize_key(edge_cat): nn.Parameter(
                torch.empty(heads, self.d_k, self.d_k)
            )
            for edge_cat in edge_categories
        })

        # ─────────────────────────────────────────────────────────────────────
        # 3. Relation-specific message weights (W_MSG per edge category)
        # ─────────────────────────────────────────────────────────────────────
        # Transforms value vectors based on relation type
        # Shape: [heads, d_k, d_k]

        self.w_msg = nn.ParameterDict({
            sanitize_key(edge_cat): nn.Parameter(
                torch.empty(heads, self.d_k, self.d_k)
            )
            for edge_cat in edge_categories
        })

        # ─────────────────────────────────────────────────────────────────────
        # 4. Edge attribute projections (for LIANA magnitude)
        # ─────────────────────────────────────────────────────────────────────
        # Two projections for edge features:
        # 1. edge_lin: attention bias (affects which sources get attention)
        # 2. edge_scale_lin: message scaling (affects how much influence flows)
        #
        # This ensures LIANA magnitude affects both:
        # - Relative importance (attention distribution)
        # - Absolute influence (message magnitude)

        if edge_dim is not None:
            # Attention bias: higher LIANA → higher attention score
            self.edge_lin = nn.Linear(edge_dim, heads)
            # Message scaling: higher LIANA → stronger message (sigmoid output)
            self.edge_scale_lin = nn.Linear(edge_dim, 1)
        else:
            self.edge_lin = None
            self.edge_scale_lin = None

        # ─────────────────────────────────────────────────────────────────────
        # 5. Output projection and dropout
        # ─────────────────────────────────────────────────────────────────────

        self.out_lin = nn.Linear(out_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

        # Mapping from original names to sanitized keys
        self._node_type_to_key = {
            nt: sanitize_key(nt) for nt in node_types
        }
        self._edge_cat_to_key = {
            ec: sanitize_key(ec) for ec in edge_categories
        }

        # Initialize parameters
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize attention and message weights.

        - W_ATT, W_MSG: Xavier initialization
        - edge_lin: Positive weights so higher LIANA → higher attention
        """
        for edge_cat in self.w_att:
            nn.init.xavier_uniform_(self.w_att[edge_cat])
            nn.init.xavier_uniform_(self.w_msg[edge_cat])

        # Initialize edge_lin with positive weights
        # This ensures higher LIANA magnitude = higher attention bias
        # which aligns with scientific intent (stronger interaction = more attention)
        if self.edge_lin is not None:
            nn.init.uniform_(self.edge_lin.weight, 0.1, 0.5)
            nn.init.zeros_(self.edge_lin.bias)

        # Initialize edge_scale_lin for message scaling
        # We want sigmoid(edge_scale_lin(x)) to produce reasonable values:
        # - For typical LIANA values (0-1), output should be ~0.5-0.8
        # - Positive weight ensures higher LIANA = stronger messages
        if self.edge_scale_lin is not None:
            nn.init.uniform_(self.edge_scale_lin.weight, 0.5, 1.0)
            # Bias of 0 means sigmoid(0)=0.5 for zero input
            nn.init.zeros_(self.edge_scale_lin.bias)

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
        edge_attr_dict: Optional[dict[tuple[str, str, str], torch.Tensor]] = None,
        return_attention: bool = False,
    ) -> tuple[dict[str, torch.Tensor], Optional[dict[tuple[str, str, str], torch.Tensor]]]:
        """
        Forward pass through HGT convolution.

        Args:
            x_dict: Node features per type {node_type: (n_nodes, in_channels)}
            edge_index_dict: Edge indices {(src_type, rel, dst_type): (2, n_edges)}
            edge_attr_dict: Edge features {(src_type, rel, dst_type): (n_edges, edge_dim)}
            return_attention: Whether to return attention weights

        Returns:
            out_dict: Updated node features {node_type: (n_nodes, out_channels)}
            attn_dict: Attention weights per edge type (if return_attention=True)
        """
        # Initialize output accumulators for each node type
        out_dict = {
            node_type: torch.zeros(
                x.size(0), self.out_channels, device=x.device, dtype=x.dtype
            )
            for node_type, x in x_dict.items()
        }

        attn_dict = {} if return_attention else None

        # Process each edge type
        for edge_type, edge_index in edge_index_dict.items():
            src_type, rel, dst_type = edge_type

            if edge_index.size(1) == 0:
                continue  # Skip empty edge sets

            # Validate node types exist
            if src_type not in x_dict or dst_type not in x_dict:
                continue

            # Get source and target node features
            x_src = x_dict[src_type]  # [n_src, in_channels]
            x_dst = x_dict[dst_type]  # [n_dst, in_channels]

            src_idx = edge_index[0]  # [n_edges]
            dst_idx = edge_index[1]  # [n_edges]

            # Get sanitized keys for module lookups
            src_key = self._node_type_to_key.get(src_type, sanitize_key(src_type))
            dst_key = self._node_type_to_key.get(dst_type, sanitize_key(dst_type))
            rel_key = self._edge_cat_to_key.get(rel, sanitize_key(rel))

            # Skip if node type or relation not in our learned parameters
            if src_key not in self.k_lin or dst_key not in self.q_lin:
                continue
            if rel_key not in self.w_att:
                continue

            # ─────────────────────────────────────────────────────────────────
            # Step 1: Type-specific projections
            # ─────────────────────────────────────────────────────────────────

            # Query from target nodes (receiving messages)
            q = self.q_lin[dst_key](x_dst)  # [n_dst, out_channels]
            q = q.view(-1, self.heads, self.d_k)  # [n_dst, heads, d_k]
            q_i = q[dst_idx]  # [n_edges, heads, d_k]

            # Key from source nodes (sending messages)
            k = self.k_lin[src_key](x_src)  # [n_src, out_channels]
            k = k.view(-1, self.heads, self.d_k)  # [n_src, heads, d_k]
            k_j = k[src_idx]  # [n_edges, heads, d_k]

            # Value from source nodes
            v = self.v_lin[src_key](x_src)  # [n_src, out_channels]
            v = v.view(-1, self.heads, self.d_k)  # [n_src, heads, d_k]
            v_j = v[src_idx]  # [n_edges, heads, d_k]

            # ─────────────────────────────────────────────────────────────────
            # Step 2: Relation-specific attention
            # ─────────────────────────────────────────────────────────────────

            # Transform key with relation-specific weight
            # k_j: [n_edges, heads, d_k]
            # w_att[rel]: [heads, d_k, d_k]
            # Result: [n_edges, heads, d_k]
            k_j_transformed = torch.einsum('ehd,hdk->ehk', k_j, self.w_att[rel_key])

            # Compute attention scores
            # q_i: [n_edges, heads, d_k], k_j_transformed: [n_edges, heads, d_k]
            # Result: [n_edges, heads] (dot product per head)
            attn_scores = (q_i * k_j_transformed).sum(dim=-1) / (self.d_k ** 0.5)

            # ─────────────────────────────────────────────────────────────────
            # Step 3: Add edge attribute bias (LIANA magnitude)
            # ─────────────────────────────────────────────────────────────────

            if self.edge_lin is not None and edge_attr_dict is not None:
                edge_attr = edge_attr_dict.get(edge_type)
                if edge_attr is not None:
                    # edge_attr: [n_edges, edge_dim]
                    # edge_bias: [n_edges, heads]
                    edge_bias = self.edge_lin(edge_attr)
                    attn_scores = attn_scores + edge_bias

            # ─────────────────────────────────────────────────────────────────
            # Step 4: Softmax over source nodes (per target)
            # ─────────────────────────────────────────────────────────────────

            # Compute softmax grouped by target node
            attn_weights = self._softmax_by_target(
                attn_scores, dst_idx, num_nodes=x_dst.size(0)
            )  # [n_edges, heads]

            attn_weights = self.dropout(attn_weights)

            if return_attention:
                attn_dict[edge_type] = attn_weights.detach().clone()

            # ─────────────────────────────────────────────────────────────────
            # Step 5: Relation-specific message passing with edge scaling
            # ─────────────────────────────────────────────────────────────────

            # Transform value with relation-specific weight
            v_j_transformed = torch.einsum('ehd,hdk->ehk', v_j, self.w_msg[rel_key])

            # Weight by attention
            # attn_weights: [n_edges, heads] -> [n_edges, heads, 1]
            messages = attn_weights.unsqueeze(-1) * v_j_transformed  # [n_edges, heads, d_k]

            # Apply edge-based message scaling (LIANA magnitude affects message strength)
            # This ensures higher LIANA = stronger influence on receiving cell
            if self.edge_scale_lin is not None and edge_attr_dict is not None:
                edge_attr = edge_attr_dict.get(edge_type)
                if edge_attr is not None:
                    # edge_scale: [n_edges, 1] -> [n_edges, 1, 1] for broadcasting
                    edge_scale = torch.sigmoid(self.edge_scale_lin(edge_attr))
                    messages = messages * edge_scale.unsqueeze(-1)

            # ─────────────────────────────────────────────────────────────────
            # Step 6: Aggregate messages to target nodes
            # ─────────────────────────────────────────────────────────────────

            # Reshape messages for scatter
            messages_flat = messages.view(-1, self.out_channels)  # [n_edges, out_channels]

            # Scatter-add messages to target nodes (non-in-place for gradient flow)
            out_dict[dst_type] = out_dict[dst_type].scatter_add(
                0,
                dst_idx.unsqueeze(-1).expand(-1, self.out_channels),
                messages_flat,
            )

        # Track which nodes received any messages (non-zero before out_lin)
        # This is used to zero out isolated nodes after out_lin
        # Rationale: "no LIANA edges = no communication contribution"
        received_messages = {
            node_type: (out.abs().sum(dim=-1) > 0)  # [n_nodes] bool
            for node_type, out in out_dict.items()
        }

        # Apply output projection
        out_dict = {
            node_type: self.out_lin(out)
            for node_type, out in out_dict.items()
        }

        # Zero out isolated nodes (those that received no messages)
        # Without this, out_lin's bias would give isolated nodes a non-zero
        # "communication" signal, conflating "no edges" with "baseline signal"
        out_dict = {
            node_type: torch.where(
                received_messages[node_type].unsqueeze(-1),
                out,
                torch.zeros_like(out)
            )
            for node_type, out in out_dict.items()
        }

        return out_dict, attn_dict

    def _softmax_by_target(
        self,
        scores: torch.Tensor,
        target_idx: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """
        Compute softmax of scores grouped by target node.

        Uses numerically stable softmax with max subtraction.

        Args:
            scores: [n_edges, heads] attention scores
            target_idx: [n_edges] target node indices
            num_nodes: Number of target nodes

        Returns:
            [n_edges, heads] softmax weights
        """
        n_heads = scores.size(1)
        device = scores.device
        dtype = scores.dtype

        # Compute max per target node for numerical stability
        max_scores = torch.full(
            (num_nodes, n_heads), float('-inf'), device=device, dtype=dtype
        )
        # scatter_reduce_ requires PyTorch >= 2.1 (checked at module import)
        max_scores.scatter_reduce_(
            0,
            target_idx.unsqueeze(-1).expand(-1, n_heads),
            scores,
            reduce='amax',
            include_self=True,
        )

        # Handle nodes with no incoming edges (keep as -inf, will become 0 after exp)
        scores_normalized = scores - max_scores[target_idx]

        # Compute exp
        exp_scores = torch.exp(scores_normalized)

        # Sum per target node
        sum_exp = torch.zeros(num_nodes, n_heads, device=device, dtype=dtype)
        sum_exp.scatter_add_(
            0,
            target_idx.unsqueeze(-1).expand(-1, n_heads),
            exp_scores,
        )

        # Normalize (add small epsilon to avoid division by zero)
        return exp_scores / (sum_exp[target_idx] + EPSILON_SOFTMAX)

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"heads={self.heads}, edge_dim={self.edge_dim}, "
            f"n_node_types={len(self.node_types)}, "
            f"n_edge_categories={len(self.edge_categories)}"
        )
