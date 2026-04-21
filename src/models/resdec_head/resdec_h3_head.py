"""ResDec-H3 head composer (Phase 1 single-stage).

Assembles FiLM metadata conditioning + NPTStage + scalar readout. Consumes a
subject embedding produced by the existing CognitiveResilienceModel encoder.

Phase 1 scope: single stage only.
Phase 2 adds TabPFN residual base (prediction = ŷ_tabpfn + f̂_1) and aug-U uncertainty-weighted loss.
Phase 3 adds 3-stage boosting with cross-stage attention and TabMWrapper for ensemble uncertainty.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from .film_metadata import FiLMMetadata
from .npt_stage import NPTStage


class ResDecH3Head(nn.Module):
    def __init__(
        self,
        d_subject: int = 64,
        d_metadata: int = 8,
        n_heads: int = 4,
        n_hc_streams: int = 4,
        lambda_init: float = 0.8,
    ):
        super().__init__()
        self.d_subject = d_subject
        self.film = FiLMMetadata(d_subject=d_subject, d_metadata=d_metadata)
        self.stage_1 = NPTStage(
            d_subject=d_subject,
            n_heads=n_heads,
            n_hc_streams=n_hc_streams,
            lambda_init=lambda_init,
        )

    def forward(self, z_encoder: torch.Tensor, metadata: torch.Tensor) -> dict:
        """
        z_encoder: [B, d_subject] subject embedding from CognitiveResilienceModel
        metadata: [B, d_metadata] FiLM-conditioning vector (APOE/sex/age)
        Returns dict with:
          prediction: [B] scalar cogn residual prediction
          latent_1:   [B, d_subject] stage-1 latent (for cross-stage attention in Phase 3)
        """
        z_cond = self.film(z_encoder, metadata)
        latent_1, scalar_1 = self.stage_1(z_cond)
        return {"prediction": scalar_1, "latent_1": latent_1}
