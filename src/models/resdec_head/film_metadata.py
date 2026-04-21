"""FiLM metadata conditioning: z_cond = γ(m) ⊙ z + β(m).

Initialized so γ ≈ 1, β ≈ 0 → forward pass is near-identity at start of training.
Used by ResDec-H3 head to condition the subject embedding on APOE/sex/age.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class FiLMMetadata(nn.Module):
    def __init__(self, d_subject: int, d_metadata: int):
        super().__init__()
        self.gamma_proj = nn.Linear(d_metadata, d_subject)
        self.beta_proj = nn.Linear(d_metadata, d_subject)
        # Init: zero weights, bias=1 for gamma and bias=0 for beta → near-identity
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.ones_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, z: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma_proj(metadata)
        beta = self.beta_proj(metadata)
        return gamma * z + beta
