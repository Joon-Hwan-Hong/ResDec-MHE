"""Hyper-Connections (Zhu et al., ICLR 2025, arXiv 2409.19606).

Dynamic learnable residual replacement that propagates N **distinct** residual
streams end-to-end instead of a single "x + sublayer(x)" trunk. At each
hyper-connected layer:

  1. Streams mix via a learnable matrix ``A`` of shape ``[N, N]``:
         streams_mixed = A @ streams
     so each output stream is a learned linear combination of prior streams.
  2. A learnable gate ``α`` picks a softmax-weighted combination of the mixed
     streams to feed into the sublayer (single FFN / attention call).
  3. The sublayer output is broadcast back into each stream, weighted by a
     learnable per-stream coefficient ``B``.

This is the multi-stream formulation spelled out in §3 of Zhu et al. (DHC —
Dynamic Hyper-Connection). The previous (buggy) implementation kept a single
residual ``x`` and called ``sublayer`` N times on identical inputs, so under a
deterministic sublayer the N streams collapsed and ``n_streams`` was a
structural no-op.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class HyperConnection(nn.Module):
    """Multi-stream Hyper-Connection block (Zhu et al., 2024).

    Takes multi-stream residual state ``streams`` of shape ``[B, N, d]``,
    applies a sublayer to a learned weighted mix of streams, and updates each
    stream via a learnable combination of the mixed prior state and the
    sublayer output.

    Args:
        d_model: Stream feature dimension ``d``.
        n_streams: Number of parallel residual streams ``N``.
    """

    def __init__(self, d_model: int, n_streams: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_streams = n_streams

        # A: stream-to-stream mixing matrix. Init as identity so the layer
        # starts as a no-op mix (each stream passes through unchanged).
        self.A = nn.Parameter(torch.eye(n_streams))

        # α: logits over streams for building the sublayer input
        #    (softmaxed on use). Uniform init → equivalent to mean-pooling at
        #    start of training.
        self.alpha = nn.Parameter(torch.zeros(n_streams))

        # B: per-stream coefficient for re-injecting the sublayer output back
        #    into each stream. Init to 1/N so each stream gets an even mean
        #    update, matching the original residual "x + sublayer(x)" scale
        #    when streams are identical.
        self.B = nn.Parameter(torch.full((n_streams,), 1.0 / n_streams))

    def forward(self, streams: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        """Forward through one hyper-connected layer.

        Args:
            streams: ``[B, N, d]`` multi-stream residual state.
            sublayer: Module or callable mapping ``[B, d] -> [B, d]``.

        Returns:
            ``[B, N, d]`` updated multi-stream state.
        """
        if streams.dim() != 3:
            raise ValueError(
                f"HyperConnection expects [B, N, d] streams; got shape {tuple(streams.shape)}"
            )
        _, N, _ = streams.shape
        if N != self.n_streams:
            raise ValueError(
                f"Stream count mismatch: got N={N}, expected {self.n_streams}"
            )

        # 1. Stream mixing: streams_mixed[b, m, :] = sum_n A[m, n] * streams[b, n, :]
        streams_mixed = torch.einsum("bnd,mn->bmd", streams, self.A)

        # 2. Build sublayer input as softmax-weighted mix of the mixed streams.
        alpha_weights = torch.softmax(self.alpha, dim=0)  # [N]
        sublayer_input = (streams_mixed * alpha_weights.view(1, -1, 1)).sum(dim=1)  # [B, d]
        sublayer_output = sublayer(sublayer_input)  # [B, d]

        # 3. Broadcast sublayer output back into each stream, scaled per-stream by B.
        B_weights = self.B.view(1, -1, 1)  # [1, N, 1]
        updated = streams_mixed + B_weights * sublayer_output.unsqueeze(1)
        return updated
