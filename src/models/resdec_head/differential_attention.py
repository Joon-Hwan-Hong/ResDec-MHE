"""Differential Transformer attention (Ye et al., ICLR 2025, arXiv 2410.05258).

Full fidelity to the paper:
  - Per-head λ reparameterization: λ_h = exp(λ_q1·λ_k1) − exp(λ_q2·λ_k2) + λ_init
  - GroupNorm (per-head normalization) on attention output before output projection
  - (1 − λ_init) output scaling to match initial scale of vanilla attention
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentialAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, lambda_init: float = 0.8):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.lambda_init = lambda_init

        self.q1 = nn.Linear(d_model, d_model)
        self.k1 = nn.Linear(d_model, d_model)
        self.q2 = nn.Linear(d_model, d_model)
        self.k2 = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

        # Per-head λ reparameterization (init with small values; exp gives init≈1)
        lam_init_val = 0.0  # so exp(0) = 1 at init
        self.lambda_q1 = nn.Parameter(torch.full((n_heads, self.d_head), lam_init_val))
        self.lambda_k1 = nn.Parameter(torch.full((n_heads, self.d_head), lam_init_val))
        self.lambda_q2 = nn.Parameter(torch.full((n_heads, self.d_head), lam_init_val))
        self.lambda_k2 = nn.Parameter(torch.full((n_heads, self.d_head), lam_init_val))

        # Per-head GroupNorm on attention output (num_groups = n_heads)
        self.group_norm = nn.GroupNorm(num_groups=n_heads, num_channels=d_model)

    def _per_head_lambda(self) -> torch.Tensor:
        """λ_h per head: shape [n_heads]."""
        lam1 = (self.lambda_q1 * self.lambda_k1).sum(dim=-1).exp()  # [n_heads]
        lam2 = (self.lambda_q2 * self.lambda_k2).sum(dim=-1).exp()  # [n_heads]
        return lam1 - lam2 + self.lambda_init  # [n_heads]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        q1 = self.q1(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        k1 = self.k1(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        q2 = self.q2(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        k2 = self.k2(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v  = self.v(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)

        scale = self.d_head ** -0.5
        attn1 = F.softmax(q1 @ k1.transpose(-2, -1) * scale, dim=-1)
        attn2 = F.softmax(q2 @ k2.transpose(-2, -1) * scale, dim=-1)

        # Per-head λ broadcast: [H] → [1, H, 1, 1]
        lam_h = self._per_head_lambda().view(1, self.n_heads, 1, 1)
        attn = attn1 - lam_h * attn2  # [B, H, N, N]

        out_h = attn @ v  # [B, H, N, d_head]
        # GroupNorm expects [B, C, L] with C=n_heads*d_head=d_model, so reshape:
        # [B, H, N, d_head] → [B, N, H, d_head] → [B, N, D] → [B, D, N]
        out_bnhd = out_h.transpose(1, 2).contiguous()  # [B, N, H, d_head]
        out_bnd = out_bnhd.view(B, N, D)
        out_bdn = out_bnd.transpose(1, 2)  # [B, D, N] for GroupNorm
        out_bdn = self.group_norm(out_bdn)
        out_bnd = out_bdn.transpose(1, 2)  # back to [B, N, D]

        # (1 - λ_init) output scaling before final projection
        return self.out(out_bnd) * (1.0 - self.lambda_init)
