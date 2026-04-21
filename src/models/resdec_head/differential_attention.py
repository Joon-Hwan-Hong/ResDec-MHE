"""Differential Transformer attention (Ye et al., ICLR 2025, arXiv 2410.05258).

Two softmax attention maps subtracted via learned λ to cancel common-mode noise:
    attn = softmax(Q1·K1ᵀ/√d) - λ · softmax(Q2·K2ᵀ/√d)
    out  = attn · V
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
        self.q1 = nn.Linear(d_model, d_model)
        self.k1 = nn.Linear(d_model, d_model)
        self.q2 = nn.Linear(d_model, d_model)
        self.k2 = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.lambda_param = nn.Parameter(torch.tensor(float(lambda_init)))

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
        attn = attn1 - self.lambda_param * attn2

        out = attn @ v  # [B, H, N, d_head]
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out(out)
