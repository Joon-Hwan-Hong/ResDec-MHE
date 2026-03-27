"""
Fusion layer for combining two-branch embeddings.

Concatenates HGT and cell transformer embeddings,
then projects to a unified dimension.
"""

import torch
import torch.nn as nn


class FusionLayer(nn.Module):
    """
    Fuse representations from two branches (HGT + Cell Transformer).

    All cell types get 2 embeddings (HGT + Cell Transformer).
    Project to uniform dimension.
    """

    def __init__(
        self,
        d_embed: int,
        d_fused: int,
        n_cell_types: int = 31,
        dropout: float = 0.1,
        n_pma_seeds: int = 1,
    ):
        super().__init__()

        # Input validation
        if d_embed <= 0:
            raise ValueError(f"d_embed must be positive, got {d_embed}")
        if d_fused <= 0:
            raise ValueError(f"d_fused must be positive, got {d_fused}")
        if n_cell_types <= 0:
            raise ValueError(f"n_cell_types must be positive, got {n_cell_types}")

        # Store attributes for debugging
        self.d_embed = d_embed
        self.d_fused = d_fused
        self.n_cell_types = n_cell_types
        self.n_pma_seeds = n_pma_seeds
        self.d_cell_emb = n_pma_seeds * d_embed

        # Concat branches: HGT(d_embed) + cell_transformer(n_pma_seeds * d_embed)
        self.proj = nn.Linear(d_embed + self.d_cell_emb, d_fused)

        self.layer_norm = nn.LayerNorm(d_fused)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hgt_emb: torch.Tensor,          # [B, 31, d_embed]
        cell_emb: torch.Tensor,         # [B, 31, n_pma_seeds * d_embed]
    ) -> torch.Tensor:
        """
        Fuse two branches for all cell types.

        When n_pma_seeds > 1, cell_emb carries multiple subpopulation summaries
        concatenated along the feature dim.

        Returns:
            fused: [B, 31, d_fused]
        """
        # Input dimension validation
        if hgt_emb.dim() != 3:
            raise ValueError(f"Expected 3D input for hgt_emb, got shape {hgt_emb.shape}")
        if cell_emb.dim() != 3:
            raise ValueError(f"Expected 3D input for cell_emb, got shape {cell_emb.shape}")

        # Validate cell type and embedding dimensions
        B, C, D = hgt_emb.shape
        if C != self.n_cell_types:
            raise ValueError(f"Expected {self.n_cell_types} cell types, got {C}")
        if D != self.d_embed:
            raise ValueError(f"Expected d_embed={self.d_embed} for hgt_emb, got {D}")
        if cell_emb.shape[1] != self.n_cell_types:
            raise ValueError(f"Expected {self.n_cell_types} cell types for cell_emb, got {cell_emb.shape[1]}")
        if cell_emb.shape[2] != self.d_cell_emb:
            raise ValueError(f"Expected d_cell_emb={self.d_cell_emb} for cell_emb, got {cell_emb.shape[2]}")

        # Concat 2 branches: [B, 31, d_embed + d_cell_emb]
        concat = torch.cat([hgt_emb, cell_emb], dim=-1)

        # Project to fused dimension: [B, 31, d_fused]
        fused = self.proj(concat)

        return self.dropout(self.layer_norm(fused))

    def extra_repr(self) -> str:
        """Return extra representation string for informative repr output."""
        return f"d_embed={self.d_embed}, d_fused={self.d_fused}, n_cell_types={self.n_cell_types}"
