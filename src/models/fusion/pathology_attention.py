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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Attend over cell types conditioned on pathology.

        Args:
            cell_type_embeddings: [B, n_cell_types, d_fused] - fused cell type embeddings
            path_emb: [B, d_cond] - pathology embedding from PathologyEncoder
            cell_type_mask: [B, n_cell_types] - bool mask, True=available (optional)
                If not provided, all cell types are assumed available.

        Returns:
            attended: [B, d_fused] - pathology-weighted cell type summary
            attention_weights: [B, n_heads, n_cell_types] - for interpretability
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
        query = query.view(B, 1, self.n_heads, self.d_head)

        # Project cell types to keys and values
        keys = self.key_proj(cell_type_embeddings).view(B, self.n_cell_types, self.n_heads, self.d_head)
        values = self.value_proj(cell_type_embeddings).view(B, self.n_cell_types, self.n_heads, self.d_head)

        # Attention scores
        scores = torch.einsum('bqhd,bkhd->bhqk', query, keys) / (self.d_head ** 0.5)

        # Pathology-dependent additive bias
        path_emb_expanded = path_emb.unsqueeze(1).expand(-1, self.n_cell_types, -1)  # [B, n_cell_types, d_cond]
        bias_input = torch.cat([path_emb_expanded, cell_type_embeddings], dim=-1)
        bias = self.pathology_bias(bias_input)  # [B, n_cell_types, n_heads]
        bias = bias.permute(0, 2, 1).unsqueeze(2)  # [B, n_heads, 1, n_cell_types]

        scores = scores + bias  # additive bias: directly shifts attention per cell type

        # Initialize all_masked before conditional to prevent UnboundLocalError
        # if masking logic is ever refactored. Default: no samples are fully masked.
        all_masked = torch.zeros(B, dtype=torch.bool, device=cell_type_embeddings.device)

        # Apply cell type mask if provided (mask out missing cell types)
        if cell_type_mask is not None:
            # cell_type_mask: [B, n_cell_types] -> [B, 1, 1, n_cell_types]
            mask = cell_type_mask.unsqueeze(1).unsqueeze(2)
            # Set scores for masked cell types to -inf so softmax gives 0
            scores = scores.masked_fill(~mask, float('-inf'))

            # For batch elements where ALL cell types are masked, all scores are -inf.
            # Softmax on all -inf produces NaN which poisons gradients even though
            # nan_to_num fixes forward values. Replace -inf with a large finite
            # negative so softmax produces near-zero uniform weights instead of NaN.
            all_masked = ~cell_type_mask.any(dim=1)  # [B]
            if all_masked.any():
                # Expand to match scores shape [B, n_heads, 1, n_cell_types]
                all_masked_expanded = all_masked.view(-1, 1, 1, 1).expand_as(scores)
                scores = scores.masked_fill(all_masked_expanded, -1e9)

        # Softmax and attend
        if scores.size(2) != 1:
            raise RuntimeError(f"Expected query dim 1, got {scores.size(2)}")
        attention_weights = F.softmax(scores.float(), dim=-1)[:, :, 0, :]  # [B, n_heads, n_cell_types]

        # Zero out attention for fully-masked samples (softmax gave near-uniform,
        # but these samples should contribute nothing)
        if cell_type_mask is not None and all_masked.any():
            # all_masked: [B] -> [B, 1, 1] for broadcasting
            attention_weights = attention_weights.masked_fill(
                all_masked.unsqueeze(-1).unsqueeze(-1).expand_as(attention_weights), 0.0
            )

        values = values.permute(0, 2, 1, 3)  # [B, n_heads, n_cell_types, d_head]
        # Compute weighted sum in float32 for precision — attention_weights are float32
        # from softmax promotion, values may be float16 under AMP.
        # einsum does not auto-promote mixed dtypes, so cast values up explicitly.
        # Cast result back to input dtype before out_proj (Linear layer stays in AMP dtype).
        attended = torch.einsum('bhk,bhkd->bhd', attention_weights, values.to(attention_weights.dtype))
        attended = attended.to(cell_type_embeddings.dtype)
        attended = attended.reshape(B, self.d_fused)
        attended = self.out_proj(attended)

        return attended, attention_weights

    def extra_repr(self) -> str:
        """Return extra representation string for informative repr output."""
        return (
            f"d_fused={self.d_fused}, d_cond={self.d_cond}, "
            f"n_heads={self.n_heads}, n_cell_types={self.n_cell_types}"
        )
