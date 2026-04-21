"""TabM-style BatchEnsemble wrapper (Gorishniy et al., ICLR 2025, arXiv 2410.24210).

Applies a shared submodule k times with per-member rank-1 (s_k, r_k) scaling:
    y_k = (x * s_k) then submodule(...) then * r_k
Returns (mean, std) across the k members for both prediction and uncertainty.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class TabMWrapper(nn.Module):
    def __init__(self, submodule: nn.Module, d_io: int, k: int = 8):
        super().__init__()
        self.submodule = submodule
        self.k = k
        self.d_io = d_io
        # Per-member rank-1 scalings, init near 1 with small noise so members diverge
        self.s = nn.Parameter(torch.randn(k, d_io) * 0.01 + 1.0)
        self.r = nn.Parameter(torch.randn(k, d_io) * 0.01 + 1.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, d_io]
        outputs = []
        for ki in range(self.k):
            scaled_in = x * self.s[ki]
            sub_out = self.submodule(scaled_in)
            if isinstance(sub_out, tuple):
                sub_out = sub_out[0]
            outputs.append(sub_out * self.r[ki])
        stacked = torch.stack(outputs, dim=1)  # [B, k, d_out]
        mean = stacked.mean(dim=1)
        std = stacked.std(dim=1)
        return mean, std
