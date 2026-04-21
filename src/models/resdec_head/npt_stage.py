"""Single head stage: NPT row-attention (subjects as tokens) + DiffAttn + HyperConn.

Full-cohort NPT mode: the whole batch is treated as a sequence of subjects.
Attention operates across the batch axis, letting each subject attend to all
other subjects in the batch. This matches the NPT (Non-Parametric Transformers)
original formulation (Kossen et al., NeurIPS 2021).
"""
from __future__ import annotations
import torch
import torch.nn as nn

from .differential_attention import DifferentialAttention
from .hyper_connections import HyperConnection


class NPTStage(nn.Module):
    def __init__(self, d_subject: int = 64, n_heads: int = 4,
                 n_hc_streams: int = 4, lambda_init: float = 0.8):
        super().__init__()
        self.diff_attn = DifferentialAttention(d_subject, n_heads=n_heads,
                                               lambda_init=lambda_init)
        self.norm1 = nn.LayerNorm(d_subject)
        self.ffn = nn.Sequential(
            nn.Linear(d_subject, d_subject * 2),
            nn.GELU(),
            nn.Linear(d_subject * 2, d_subject),
        )
        self.norm2 = nn.LayerNorm(d_subject)
        self.hc = HyperConnection(d_subject, n_streams=n_hc_streams)
        self.readout = nn.Linear(d_subject, 1)

    def forward(self, z_cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        z_cond: [B, d_subject] — all subjects in one batch (full-cohort NPT)
        Returns: (latent [B, d_subject], scalar [B])
        """
        # Reshape to [1, B, d] so DiffAttn sees B subjects as seq length
        x_seq = z_cond.unsqueeze(0)
        attn_out = self.diff_attn(self.norm1(x_seq))
        x_seq = x_seq + attn_out
        x = x_seq.squeeze(0)  # back to [B, d]

        # FFN wrapped by HyperConnection
        x = self.hc(x, lambda xx: self.ffn(self.norm2(xx)))

        scalar = self.readout(x).squeeze(-1)  # [B]
        return x, scalar
