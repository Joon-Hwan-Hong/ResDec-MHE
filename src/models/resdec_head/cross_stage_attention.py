"""Cross-stage attention for ResDec-H3's multi-stage head.

(Wired in Phase 3 for multi-stage heads. Unused in Phase 1 but committed for
Phase 3 use per plan Task 3.1.)

Query: current stage's conditioned subject embedding z_cond [B, d_subject]
Keys+Values: concatenation of prior stages' latents [B, d_subject] each
Returns: context vector [B, d_subject] to add to z_cond before the current stage's work
"""
from __future__ import annotations
import torch
import torch.nn as nn


class CrossStageAttention(nn.Module):
    def __init__(self, d_subject: int = 64, n_heads: int = 4):
        super().__init__()
        if d_subject % n_heads != 0:
            raise ValueError(f"d_subject={d_subject} must be divisible by n_heads={n_heads}")
        self.d_subject = d_subject
        self.n_heads = n_heads
        self.d_head = d_subject // n_heads
        self.q = nn.Linear(d_subject, d_subject)
        self.kv = nn.Linear(d_subject, d_subject * 2)
        self.out = nn.Linear(d_subject, d_subject)

    def forward(self, z_cond: torch.Tensor, prior_latents: list[torch.Tensor]) -> torch.Tensor:
        """
        z_cond: [B, d_subject]
        prior_latents: list of [B, d_subject] tensors (one per prior stage)
        Returns: context [B, d_subject]
        """
        B, D = z_cond.shape
        if len(prior_latents) == 0:
            return torch.zeros_like(z_cond)

        # Stack priors as seq: [B, n_prior, d]
        ctx_seq = torch.stack(prior_latents, dim=1)
        n_prior = ctx_seq.size(1)

        q = self.q(z_cond).view(B, 1, self.n_heads, self.d_head).transpose(1, 2)
        kv = self.kv(ctx_seq).view(B, n_prior, 2, self.n_heads, self.d_head)
        k = kv[:, :, 0].permute(0, 2, 1, 3)  # [B, H, n_prior, d_head]
        v = kv[:, :, 1].permute(0, 2, 1, 3)

        scale = self.d_head ** -0.5
        attn = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
        out = attn @ v  # [B, H, 1, d_head]
        out = out.transpose(1, 2).contiguous().view(B, D)
        return self.out(out)
