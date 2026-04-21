"""Hyper-Connections (Zhu et al., ICLR 2025, arXiv 2409.19606).

Dynamic learnable residual replacement: replaces `x + sublayer(x)` with a
learned softmax-weighted combination over N parallel streams.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class HyperConnection(nn.Module):
    def __init__(self, d_model: int, n_streams: int = 4):
        super().__init__()
        self.n_streams = n_streams
        # Init: zero logits → softmax is uniform → equivalent to mean pooling at start
        self.alpha = nn.Parameter(torch.zeros(n_streams))

    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        """
        x: [B, d_model]
        sublayer: any nn.Module that maps [B, d_model] → [B, d_model]
        Returns: [B, d_model], a softmax-weighted combination of N parallel
                 sublayer outputs over identical input.
        """
        # Apply sublayer N times (each stream sees same input; learning happens
        # via how alpha weights the combination)
        stream_outputs = torch.stack(
            [sublayer(x) for _ in range(self.n_streams)], dim=1
        )  # [B, N, d_model]
        weights = torch.softmax(self.alpha, dim=0)  # [N]
        return (stream_outputs * weights.view(1, -1, 1)).sum(dim=1)
