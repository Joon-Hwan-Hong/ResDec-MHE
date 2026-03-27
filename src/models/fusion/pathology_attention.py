"""
Pathology-stratified attention over cell types.

Attention mechanism where the query is derived from pathology embedding,
allowing different pathology levels to attend to different cell types.
This directly models the resilience question: "Given high pathology,
which cell states associate with preserved cognition?"

Pathology modulates attention additively: raw Q-K attention scores receive
an unbounded bias term computed from the pathology embedding and cell-type
embeddings.  This yields a clean decomposition:

    final_attention ~ softmax(base_Q-K_relevance + pathology_effect)

Because the bias is unbounded and additive, its sign and magnitude are
directly interpretable:
  * Positive bias  -> pathology increases attention to that cell type
  * Negative bias  -> pathology decreases attention to that cell type
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PathologyStratifiedAttention(nn.Module):
    """
    Attention over cell types, conditioned on pathology embedding.

    Key insight: The query is derived from pathology, so "what we're looking for"
    changes based on disease severity. This directly models resilience:
    "Given high pathology, which cell states associate with preserved cognition?"

    Design choices:
    - Pathology-conditioned query: Different pathology -> different attention pattern
    - Additive pathology bias: An unbounded bias term is *added* to the raw Q-K
      attention scores, shifting (not scaling) the attention distribution.
      The bias values directly represent pathology's effect on attention to each
      cell type (positive = pathology increases attention, negative = decreases).
      This enables a clean decomposition:
        final_attention ≈ base_Q-K_relevance + pathology_effect
    - Multi-head: Each head can learn different resilience patterns
    """

    def __init__(
        self,
        d_fused: int,
        d_cond: int = 64,
        n_heads: int = 4,
        n_cell_types: int = 31,
    ):
        super().__init__()

        # Input validation
        if d_fused <= 0:
            raise ValueError(f"d_fused must be positive, got {d_fused}")
        if d_cond <= 0:
            raise ValueError(f"d_cond must be positive, got {d_cond}")
        if n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {n_heads}")
        if n_cell_types <= 0:
            raise ValueError(f"n_cell_types must be positive, got {n_cell_types}")
        if d_fused % n_heads != 0:
            raise ValueError(
                f"d_fused ({d_fused}) must be divisible by n_heads ({n_heads})"
            )

        self.d_fused = d_fused
        self.d_cond = d_cond
        self.n_heads = n_heads
        self.n_cell_types = n_cell_types
        self.d_head = d_fused // n_heads

        # Pathology-conditioned query generator
        self.query_generator = nn.Linear(d_cond, d_fused)

        # Key/Value projections for cell types
        self.key_proj = nn.Linear(d_fused, d_fused)
        self.value_proj = nn.Linear(d_fused, d_fused)

        # Pathology-dependent attention bias (additive, directly interpretable)
        self.pathology_bias = nn.Sequential(
            nn.Linear(d_cond + d_fused, n_heads),
            # No activation — unbounded output acts as attention bias
        )

        self.out_proj = nn.Linear(d_fused, d_fused)

    def forward(
        self,
        cell_type_embeddings: torch.Tensor,  # [B, n_cell_types, d_fused]
        path_emb: torch.Tensor,               # [B, d_cond]
        cell_type_mask: torch.Tensor = None,  # [B, n_cell_types] optional mask
        return_attention_weights: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Attend over cell types conditioned on pathology.

        Args:
            cell_type_embeddings: [B, n_cell_types, d_fused] - fused cell type embeddings
            path_emb: [B, d_cond] - pathology embedding from PathologyEncoder
            cell_type_mask: [B, n_cell_types] - bool mask, True=available (optional)
                If not provided, all cell types are assumed available.
            return_attention_weights: Whether to compute attention weights for
                interpretability. When False, skips the Q-K score recomputation
                (separate from SDPA path) and returns None for attention_weights.
                Set to False during training to avoid unnecessary compute.

        Returns:
            attended: [B, d_fused] - pathology-weighted cell type summary
            attention_weights: [B, n_heads, n_cell_types] or None - for interpretability
        """
        # Input validation
        if cell_type_embeddings.dim() != 3:
            raise ValueError(
                f"Expected 3D cell_type_embeddings, got shape {cell_type_embeddings.shape}"
            )
        if path_emb.dim() != 2:
            raise ValueError(f"Expected 2D path_emb, got shape {path_emb.shape}")

        B, C, D = cell_type_embeddings.shape

        if C != self.n_cell_types:
            raise ValueError(
                f"Expected {self.n_cell_types} cell types, got {C}"
            )
        if D != self.d_fused:
            raise ValueError(
                f"Expected d_fused={self.d_fused}, got {D}"
            )
        if path_emb.size(1) != self.d_cond:
            raise ValueError(
                f"Expected d_cond={self.d_cond}, got {path_emb.size(1)}"
            )
        if cell_type_embeddings.size(0) != path_emb.size(0):
            raise ValueError(
                f"Batch size mismatch: cell_type_embeddings has {cell_type_embeddings.size(0)}, "
                f"path_emb has {path_emb.size(0)}"
            )

        # Generate pathology-conditioned query
        query = self.query_generator(path_emb)  # [B, d_fused]
        query = query.view(B, self.n_heads, 1, self.d_head)  # [B, H, 1, d_head]

        # Project cell types to keys and values
        keys = self.key_proj(cell_type_embeddings).view(B, C, self.n_heads, self.d_head)
        keys = keys.permute(0, 2, 1, 3)  # [B, H, C, d_head]
        values = self.value_proj(cell_type_embeddings).view(B, C, self.n_heads, self.d_head)
        values = values.permute(0, 2, 1, 3)  # [B, H, C, d_head]

        # Pathology-dependent additive bias → SDPA attn_mask
        path_emb_expanded = path_emb.unsqueeze(1).expand(-1, self.n_cell_types, -1)  # [B, C, d_cond]
        bias_input = torch.cat([path_emb_expanded, cell_type_embeddings], dim=-1)
        bias = self.pathology_bias(bias_input)  # [B, C, n_heads]
        attn_bias = bias.permute(0, 2, 1).unsqueeze(2)  # [B, H, 1, C]

        # Initialize all_masked before conditional
        all_masked = torch.zeros(B, dtype=torch.bool, device=cell_type_embeddings.device)

        # Apply cell type mask as additive bias (-inf for absent types)
        if cell_type_mask is not None:
            mask = cell_type_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, C]
            attn_bias = attn_bias.masked_fill(~mask, float('-inf'))
            all_masked = ~cell_type_mask.any(dim=1)  # [B]
            if all_masked.any():
                all_masked_expanded = all_masked.view(-1, 1, 1, 1).expand_as(attn_bias)
                attn_bias = attn_bias.masked_fill(all_masked_expanded, -1e9)

        # Fused attention via SDPA (dispatches to FlashAttention/memory-efficient backend).
        # SDPA handles float32 softmax internally — no explicit .float() promotion needed.
        attended = F.scaled_dot_product_attention(
            query, keys, values,
            attn_mask=attn_bias,
            dropout_p=0.0,
        )  # [B, H, 1, d_head]

        attended = attended.squeeze(2).reshape(B, self.d_fused)
        attended = self.out_proj(attended)

        # Zero out attended output for fully-masked samples
        if cell_type_mask is not None and all_masked.any():
            attended = attended.masked_fill(
                all_masked.unsqueeze(-1).expand_as(attended), 0.0
            )

        # Compute attention weights for interpretability (detached, no grad).
        # Separate from SDPA path since SDPA doesn't return weights.
        # Skipped during training (return_attention_weights=False) to avoid
        # the redundant Q-K matmul + softmax computation.
        if return_attention_weights:
            with torch.no_grad():
                scores = torch.einsum('bhqd,bhkd->bhqk', query, keys) / (self.d_head ** 0.5)
                scores = scores + attn_bias
                attention_weights = F.softmax(scores.float(), dim=-1)[:, :, 0, :]  # [B, H, C]
                if cell_type_mask is not None and all_masked.any():
                    attention_weights = attention_weights.masked_fill(
                        all_masked.unsqueeze(-1).unsqueeze(-1).expand_as(attention_weights), 0.0
                    )
        else:
            attention_weights = None

        return attended, attention_weights

    def extra_repr(self) -> str:
        """Return extra representation string for informative repr output."""
        return (
            f"d_fused={self.d_fused}, d_cond={self.d_cond}, "
            f"n_heads={self.n_heads}, n_cell_types={self.n_cell_types}"
        )
