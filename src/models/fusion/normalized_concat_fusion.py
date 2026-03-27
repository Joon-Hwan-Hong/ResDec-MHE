"""
Normalized concat fusion layer.

Same as FusionLayer but with per-branch LayerNorm before concatenation,
equalizing activation magnitudes across branches.
"""

import torch
import torch.nn as nn


class NormalizedConcatFusionLayer(nn.Module):
    """
    Concat fusion with per-branch LayerNorm normalization (HGT + CT).

    Each branch is normalized to unit variance before concatenation,
    so the linear projection weights determine branch contributions
    purely from learned importance, not activation magnitude.

    Args:
        d_embed: Branch embedding dimension.
        d_fused: Output fused dimension.
        n_cell_types: Number of cell types (default: 31).
        dropout: Dropout probability.
        n_pma_seeds: Number of PMA seeds (affects cell_emb dimension).
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

        if d_embed <= 0:
            raise ValueError(f"d_embed must be positive, got {d_embed}")
        if d_fused <= 0:
            raise ValueError(f"d_fused must be positive, got {d_fused}")
        if n_cell_types <= 0:
            raise ValueError(f"n_cell_types must be positive, got {n_cell_types}")

        self.d_embed = d_embed
        self.d_fused = d_fused
        self.n_cell_types = n_cell_types
        self.n_pma_seeds = n_pma_seeds
        self.d_cell_emb = n_pma_seeds * d_embed

        # Per-branch normalization (equalizes magnitudes before concat)
        self.hgt_norm = nn.LayerNorm(d_embed)
        self.ct_norm = nn.LayerNorm(self.d_cell_emb)

        # Projection: HGT(d_embed) + CT(d_cell_emb)
        self.proj = nn.Linear(d_embed + self.d_cell_emb, d_fused)

        self.layer_norm = nn.LayerNorm(d_fused)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hgt_emb: torch.Tensor,          # [B, 31, d_embed]
        cell_emb: torch.Tensor,         # [B, 31, d_cell_emb]
    ) -> torch.Tensor:
        """
        Fuse two branches with per-branch normalization before concat.

        Returns:
            fused: [B, n_cell_types, d_fused]
        """
        if hgt_emb.dim() != 3:
            raise ValueError(f"Expected 3D hgt_emb, got shape {hgt_emb.shape}")
        if cell_emb.dim() != 3:
            raise ValueError(f"Expected 3D cell_emb, got shape {cell_emb.shape}")

        # Per-branch LayerNorm — equalizes magnitudes
        hgt_normed = self.hgt_norm(hgt_emb)
        ct_normed = self.ct_norm(cell_emb)

        concat = torch.cat([hgt_normed, ct_normed], dim=-1)
        fused = self.proj(concat)

        return self.dropout(self.layer_norm(fused))

    def extra_repr(self) -> str:
        return f"d_embed={self.d_embed}, d_fused={self.d_fused}, n_cell_types={self.n_cell_types}"
