"""Single head stage: NPT row-attention (subjects as tokens) + DiffAttn + HyperConn.

Full-cohort NPT mode: the whole batch is treated as a sequence of subjects.
Attention operates across the batch axis, letting each subject attend to all
other subjects in the batch. This matches the NPT (Non-Parametric Transformers)
original formulation (Kossen et al., NeurIPS 2021).

The FFN that follows the DiffAttn step is wrapped in a multi-stream
HyperConnection (Zhu et al., ICLR 2025). We lift the subject embedding ``x``
from ``[B, d]`` to ``[B, N, d]`` (one stream per HC branch) via expansion,
run the HC block, and pool the N streams back to ``[B, d]`` by mean-reduction.
Stream initialisation is identical across streams but ``HyperConnection.A``
breaks the symmetry after the first layer.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .differential_attention import DifferentialAttention
from .hyper_connections import HyperConnection


class NPTStage(nn.Module):
    def __init__(self, d_subject: int = 64, n_heads: int = 4,
                 n_hc_streams: int = 4, lambda_init: float = 0.8,
                 emit_scalar: bool = True,
                 use_diff_attn: bool = True,
                 use_hyper_conn: bool = True):
        """
        Args:
            d_subject, n_heads, n_hc_streams, lambda_init: see DifferentialAttention
                / HyperConnection for semantics.
            emit_scalar: if True (default), build a ``readout`` Linear(d_subject, 1)
                and return ``(latent, scalar)``. If False, omit the readout and
                return ``(latent, None)``. Set False when wrapped in a TabMWrapper
                that already supplies its own per-stage readout (e.g. ResDecH3Head's
                N-stage composer, N ∈ {1, 2, 3}, default 1), so that the unused
                scalar Linear isn't added to the optimizer / weight-decay.
            use_diff_attn: if True (default), use DifferentialAttention; if False,
                fall back to vanilla nn.MultiheadAttention. Phase-5.3 ablation #6.
            use_hyper_conn: if True (default), wrap FFN in multi-stream
                HyperConnection; if False, use plain residual (single stream).
                Phase-5.3 ablation #7.
        """
        super().__init__()
        self.use_diff_attn = use_diff_attn
        self.use_hyper_conn = use_hyper_conn
        if use_diff_attn:
            self.diff_attn = DifferentialAttention(d_subject, n_heads=n_heads,
                                                   lambda_init=lambda_init)
        else:
            # Vanilla MHA: same Q/K/V projections as DiffAttn but no λ-pair
            # subtraction. batch_first so [B_seq, N, d] layout is preserved.
            self.vanilla_attn = nn.MultiheadAttention(
                d_subject, num_heads=n_heads, batch_first=True,
            )
        self.norm1 = nn.LayerNorm(d_subject)
        self.ffn = nn.Sequential(
            nn.Linear(d_subject, d_subject * 2),
            nn.GELU(),
            nn.Linear(d_subject * 2, d_subject),
        )
        self.norm2 = nn.LayerNorm(d_subject)
        self.n_hc_streams = n_hc_streams if use_hyper_conn else 1
        if use_hyper_conn:
            self.hc = HyperConnection(d_subject, n_streams=n_hc_streams)
        self.emit_scalar = emit_scalar
        if emit_scalar:
            self.readout = nn.Linear(d_subject, 1)

    def _ffn_block(self, xx: torch.Tensor) -> torch.Tensor:
        """Pre-norm + FFN sublayer, kept as a bound method so HyperConnection
        can call it uniformly across all streams."""
        return self.ffn(self.norm2(xx))

    def forward(self, z_cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        z_cond: [B, d_subject] — all subjects in one batch (full-cohort NPT)
        Returns: (latent [B, d_subject], scalar [B] | None).
                 scalar is None when ``emit_scalar=False`` (callers like TabMWrapper
                 pick latent via ``sub_out[0]`` and ignore the second return).
        """
        # Reshape to [1, B, d] so attention sees B subjects as seq length.
        x_seq = z_cond.unsqueeze(0)
        normed = self.norm1(x_seq)
        if self.use_diff_attn:
            attn_out = self.diff_attn(normed)
        else:
            # nn.MultiheadAttention with batch_first=True expects [bsz, seq, d],
            # which matches our [1, B, d] layout (bsz=1, seq=B).
            attn_out, _ = self.vanilla_attn(normed, normed, normed, need_weights=False)
        x_seq = x_seq + attn_out
        x = x_seq.squeeze(0)  # back to [B, d]

        if self.use_hyper_conn:
            # Lift [B, d] → [B, N, d] multi-stream state for the HyperConnection.
            # Streams start identical; HC's learnable A matrix breaks the symmetry
            # layer-by-layer as it trains.
            streams_init = x.unsqueeze(1).expand(-1, self.n_hc_streams, -1).contiguous()
            streams_out = self.hc(streams_init, self._ffn_block)  # [B, N, d]
            # Reduce streams → [B, d] via mean-pool (simplest, no extra params).
            x = streams_out.mean(dim=1)
        else:
            # Plain residual FFN (no multi-stream): standard transformer block.
            x = x + self._ffn_block(x)

        if self.emit_scalar:
            scalar = self.readout(x).squeeze(-1)  # [B]
            return x, scalar
        return x, None
