"""TabM-style BatchEnsemble wrapper (Gorishniy et al., ICLR 2025, arXiv 2410.24210).

Applies a shared submodule k times with per-member rank-1 (s_k, r_k) scaling:
    y_k = submodule(x * s_k) * r_k
Returns (mean, std) across the k members for both prediction and uncertainty.
Used by ResDecMHEHead to wrap each boosting stage.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class TabMWrapper(nn.Module):
    def __init__(self, submodule: nn.Module, d_in: int, d_out: int | None = None, k: int = 8):
        super().__init__()
        self.submodule = submodule
        self.k = k
        self.d_in = d_in
        self.d_out = d_out if d_out is not None else d_in
        # Per-member rank-1 scalings, init near 1 with small noise so members diverge
        self.s = nn.Parameter(torch.randn(k, self.d_in) * 0.01 + 1.0)
        self.r = nn.Parameter(torch.randn(k, self.d_out) * 0.01 + 1.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, d_in]
        outputs = []
        for ki in range(self.k):
            scaled_in = x * self.s[ki]
            sub_out = self.submodule(scaled_in)
            if isinstance(sub_out, tuple):
                sub_out = sub_out[0]
            if sub_out.shape[-1] != self.d_out:
                raise RuntimeError(
                    f"TabMWrapper: expected submodule output dim {self.d_out}, got {sub_out.shape[-1]}. "
                    f"Pass d_out={sub_out.shape[-1]} in constructor."
                )
            outputs.append(sub_out * self.r[ki])
        stacked = torch.stack(outputs, dim=1)  # [B, k, d_out]
        mean = stacked.mean(dim=1)
        # unbiased=False (population std) so k=1 returns 0 instead of NaN.
        # For k≥2 this differs from the Bessel-corrected estimate by a factor
        # of sqrt(k/(k-1)), which is ≤1.07 for k=8 (the project default) — a
        # negligible bias compared to the benefit of well-defined k=1 behavior.
        std = stacked.std(dim=1, unbiased=False)
        return mean, std
