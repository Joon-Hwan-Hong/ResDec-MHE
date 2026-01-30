"""
Region Handler for multi-region data pooling.

Pools per-region cell-type embeddings using learned weighted mean,
producing interpretable region importance weights for Phase 6 analysis.
"""

from typing import ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.constants import REGION_ORDER


class RegionHandler(nn.Module):
    """
    Pool multi-region embeddings with learned region importance weights.

    Architecture:
        Input: [B, n_regions, 31, d_model] (encoded per-region embeddings)
        → Weighted mean pooling (learned region_weights)
        → Output: [B, 31, d_model] (pooled) + [B, d_model] (region_context)

    Design decisions:
        - Weighted mean only (no mean/attention variants)
        - Mask-based single-region handling (no tiling)
        - Uniform weight initialization (no PFC bias)
        - region_context = average of available region embeddings

    Args:
        d_model: Embedding dimension (must match PseudobulkEncoder output)
        n_regions: Number of brain regions (default: 6)

    Attributes:
        REGIONS: Class variable listing region names in order
        region_weights: Learnable importance weights [n_regions]
        region_embedding: Learnable region identity embeddings [n_regions, d_model]
    """

    # Import from constants for single source of truth
    REGIONS: ClassVar[list[str]] = REGION_ORDER

    def __init__(self, d_model: int, n_regions: int = 6):
        super().__init__()

        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if n_regions <= 0:
            raise ValueError(f"n_regions must be positive, got {n_regions}")

        self.d_model = d_model
        self.n_regions = n_regions

        # Learnable region importance (uniform init)
        self.region_weights = nn.Parameter(torch.zeros(n_regions))

        # Region identity embeddings for region_context
        self.region_embedding = nn.Embedding(n_regions, d_model)

    def forward(
        self,
        x: torch.Tensor,
        region_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Pool multi-region embeddings.

        Args:
            x: Per-region embeddings [B, n_regions, 31, d_model]
            region_mask: Available regions [B, n_regions] (bool or float)

        Returns:
            pooled: Weighted combination [B, 31, d_model]
            region_context: Region identity encoding [B, d_model]
        """
        # Input validation
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input, got shape {x.shape}")
        B, R, C, D = x.shape
        if R != self.n_regions:
            raise ValueError(f"Expected {self.n_regions} regions, got {R}")
        if D != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, got {D}")

        # Ensure mask is float for computation
        mask_float = region_mask.float()  # [B, R]

        # Weighted mean pooling
        weights = F.softmax(self.region_weights, dim=0)  # [R]
        masked_weights = weights.unsqueeze(0) * mask_float  # [B, R]
        weight_sum = masked_weights.sum(dim=1, keepdim=True).clamp(min=torch.finfo(x.dtype).tiny)
        normalized_weights = masked_weights / weight_sum  # [B, R]

        # Apply weights: [B, R, 1, 1] for broadcasting over [B, R, C, D]
        pooled = (x * normalized_weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
        # pooled: [B, C, D]

        # Region context (average of available region embeddings)
        all_emb = self.region_embedding.weight  # [R, d_model]
        masked_emb = all_emb.unsqueeze(0) * mask_float.unsqueeze(-1)  # [B, R, d_model]
        region_count = mask_float.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
        region_context = masked_emb.sum(dim=1) / region_count  # [B, d_model]

        return pooled, region_context

    def get_region_weights(self) -> torch.Tensor:
        """Get normalized region importance weights [n_regions]."""
        return F.softmax(self.region_weights, dim=0)

    def get_region_importance_dict(self) -> dict[str, float]:
        """Get region weights as {name: weight} for analysis."""
        weights = self.get_region_weights().detach().cpu().tolist()
        return dict(zip(self.REGIONS, weights))

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, n_regions={self.n_regions}"
