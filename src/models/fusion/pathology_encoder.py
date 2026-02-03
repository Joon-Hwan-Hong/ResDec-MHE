"""
Unified pathology encoder incorporating region context.

Combines pathology burden (amyloid, tau, global) with region availability
information to produce a single pathology embedding used throughout the model.
"""

import torch
import torch.nn as nn


class PathologyEncoder(nn.Module):
    """
    Unified pathology encoder incorporating region context.

    Design decision (2026-01-27): Single encoder replaces duplicate encoders
    in the original design. Region context informs pathology interpretation
    since subjects with different region availability may differ.
    """

    def __init__(
        self,
        n_pathology_features: int = 3,
        d_region: int = 128,
        d_cond: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Input validation
        if n_pathology_features <= 0:
            raise ValueError(f"n_pathology_features must be positive, got {n_pathology_features}")
        if d_region <= 0:
            raise ValueError(f"d_region must be positive, got {d_region}")
        if d_cond <= 0:
            raise ValueError(f"d_cond must be positive, got {d_cond}")

        self.n_pathology_features = n_pathology_features
        self.d_region = d_region
        self.d_cond = d_cond

        # Pathology MLP (post-activation dropout: LN → GELU → Dropout)
        self.pathology_mlp = nn.Sequential(
            nn.Linear(n_pathology_features, d_cond),
            nn.LayerNorm(d_cond),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_cond, d_cond),
        )

        # Region context projection
        self.region_proj = nn.Linear(d_region, d_cond)

        # Combined projection (post-activation dropout: LN → GELU → Dropout)
        self.combine = nn.Sequential(
            nn.Linear(d_cond * 2, d_cond),
            nn.LayerNorm(d_cond),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        pathology: torch.Tensor,       # [B, n_pathology_features]
        region_context: torch.Tensor,  # [B, d_region]
    ) -> torch.Tensor:
        """
        Encode pathology with region context.

        Returns:
            [B, d_cond] pathology-region embedding
        """
        # Input validation
        if pathology.dim() != 2:
            raise ValueError(f"Expected 2D pathology input, got shape {pathology.shape}")
        if region_context.dim() != 2:
            raise ValueError(f"Expected 2D region_context input, got shape {region_context.shape}")
        if pathology.size(1) != self.n_pathology_features:
            raise ValueError(
                f"Expected {self.n_pathology_features} pathology features, got {pathology.size(1)}"
            )
        if region_context.size(1) != self.d_region:
            raise ValueError(
                f"Expected region_context dim {self.d_region}, got {region_context.size(1)}"
            )
        if pathology.size(0) != region_context.size(0):
            raise ValueError(
                f"Batch size mismatch: pathology has {pathology.size(0)}, "
                f"region_context has {region_context.size(0)}"
            )

        path_emb = self.pathology_mlp(pathology)
        region_emb = self.region_proj(region_context)
        combined = torch.cat([path_emb, region_emb], dim=-1)
        return self.combine(combined)

    def extra_repr(self) -> str:
        return f"n_pathology_features={self.n_pathology_features}, d_region={self.d_region}, d_cond={self.d_cond}"
