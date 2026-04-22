"""ResDec-H3 head composer.

Assembles FiLM metadata conditioning + a 3-stage H3 boosting stack with
cross-stage attention + TabM BatchEnsemble wrapping. Consumes a subject
embedding produced by the existing CognitiveResilienceModel encoder.

Contract (Phase 3):
    forward(z_encoder, metadata) -> dict with
        prediction: [B]    = stage_1 + stage_2 + stage_3  (residual sum f̂_1+f̂_2+f̂_3)
        stage_1:    [B]    = f̂_1 scalar
        stage_2:    [B]    = f̂_2 scalar
        stage_3:    [B]    = f̂_3 scalar
        latent_1:   [B, d_subject] = h_1 pre-readout latent
        latent_2:   [B, d_subject] = h_2 pre-readout latent
        latent_3:   [B, d_subject] = h_3 pre-readout latent

Architecture (matches docs/plans/2026-04-21-resdec-h3-architecture.md §Phase 3):
    z_cond = FiLM(z_encoder, metadata)
    Stage 1:  h_1 = TabM[NPTStage](z_cond);                  f̂_1 = readout_1(h_1)
    Stage 2:  ctx_2 = cross_stage_attention(z_cond, [h_1])
              h_2 = TabM[NPTStage](z_cond + ctx_2);           f̂_2 = readout_2(h_2)
    Stage 3:  ctx_3 = cross_stage_attention(z_cond, [h_1, h_2])
              h_3 = TabM[NPTStage](z_cond + ctx_3);           f̂_3 = readout_3(h_3)

The **scalar aux losses** (detached residual decomposition, aug-U uncertainty
weighting) live in ``ResDecLightningModule.training_step``; the composer is
responsible only for producing the per-stage scalars so the trainer can build
those losses from them.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from .cross_stage_attention import CrossStageAttention
from .film_metadata import FiLMMetadata
from .npt_stage import NPTStage
from .tabm_wrapper import TabMWrapper


class ResDecH3Head(nn.Module):
    def __init__(
        self,
        d_subject: int = 64,
        d_metadata: int = 8,
        n_heads: int = 4,
        n_hc_streams: int = 4,
        lambda_init: float = 0.8,
        k_tabm: int = 8,
    ):
        super().__init__()
        self.d_subject = d_subject
        self.k_tabm = k_tabm

        self.film = FiLMMetadata(d_subject=d_subject, d_metadata=d_metadata)

        # ---- Stage 1: TabM-wrapped NPTStage + dedicated scalar readout ----
        # NPTStage.forward returns (latent[B, d], scalar[B]); TabMWrapper picks
        # [0] from that tuple and ensembles the latent across k members.
        self.stage_1_npt = NPTStage(
            d_subject=d_subject, n_heads=n_heads,
            n_hc_streams=n_hc_streams, lambda_init=lambda_init,
        )
        self.stage_1_tabm = TabMWrapper(
            submodule=self.stage_1_npt, d_in=d_subject, d_out=d_subject, k=k_tabm,
        )
        self.stage_1_readout = nn.Linear(d_subject, 1)

        # ---- Stage 2: cross-attn over [h_1] → add to z_cond → TabM[NPT] → readout ----
        self.cross_attn_2 = CrossStageAttention(d_subject=d_subject, n_heads=n_heads)
        self.stage_2_npt = NPTStage(
            d_subject=d_subject, n_heads=n_heads,
            n_hc_streams=n_hc_streams, lambda_init=lambda_init,
        )
        self.stage_2_tabm = TabMWrapper(
            submodule=self.stage_2_npt, d_in=d_subject, d_out=d_subject, k=k_tabm,
        )
        self.stage_2_readout = nn.Linear(d_subject, 1)

        # ---- Stage 3: cross-attn over [h_1, h_2] → TabM[NPT] → readout ----
        self.cross_attn_3 = CrossStageAttention(d_subject=d_subject, n_heads=n_heads)
        self.stage_3_npt = NPTStage(
            d_subject=d_subject, n_heads=n_heads,
            n_hc_streams=n_hc_streams, lambda_init=lambda_init,
        )
        self.stage_3_tabm = TabMWrapper(
            submodule=self.stage_3_npt, d_in=d_subject, d_out=d_subject, k=k_tabm,
        )
        self.stage_3_readout = nn.Linear(d_subject, 1)

    def forward(self, z_encoder: torch.Tensor, metadata: torch.Tensor) -> dict:
        """
        z_encoder: [B, d_subject] subject embedding from CognitiveResilienceModel
        metadata:  [B, d_metadata] FiLM-conditioning vector (APOE/sex/age)

        Returns the Phase-3 dict contract (see module docstring).
        """
        z_cond = self.film(z_encoder, metadata)

        # ------- Stage 1 -------
        h_1, _ = self.stage_1_tabm(z_cond)                # [B, d_subject] (mean, std)
        scalar_1 = self.stage_1_readout(h_1).squeeze(-1)  # [B]

        # ------- Stage 2 -------
        # Prior latents are .detach()'d before being fed to cross-stage attention
        # so that aux_k losses (k > 1) can't back-propagate into earlier stages'
        # TabM/readout params — matching the H3 detached-residual-boosting
        # contract (plan §Phase 3, "stage-2 gradient does NOT flow into stage-1").
        # L_main still trains every stage jointly via its own non-detached path.
        ctx_2 = self.cross_attn_2(z_cond, [h_1.detach()])  # [B, d_subject]
        h_2, _ = self.stage_2_tabm(z_cond + ctx_2)
        scalar_2 = self.stage_2_readout(h_2).squeeze(-1)

        # ------- Stage 3 -------
        ctx_3 = self.cross_attn_3(z_cond, [h_1.detach(), h_2.detach()])
        h_3, _ = self.stage_3_tabm(z_cond + ctx_3)
        scalar_3 = self.stage_3_readout(h_3).squeeze(-1)

        # The composer's "prediction" is the residual sum. Main loss in the
        # Lightning module trains this sum against ``y - ŷ_tabpfn``; val-time
        # composite prediction adds ``ŷ_tabpfn`` back.
        prediction = scalar_1 + scalar_2 + scalar_3
        return {
            "prediction": prediction,
            "stage_1": scalar_1,
            "stage_2": scalar_2,
            "stage_3": scalar_3,
            "latent_1": h_1,
            "latent_2": h_2,
            "latent_3": h_3,
        }
