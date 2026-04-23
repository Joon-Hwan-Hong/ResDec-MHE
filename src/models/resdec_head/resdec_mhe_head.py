"""ResDec-MHE head composer.

Assembles FiLM metadata conditioning + an N-stage boosting stack
(N ∈ {1, 2, 3}) with cross-stage attention + TabM BatchEnsemble wrapping.
Consumes a subject embedding produced by the existing CognitiveResilienceModel
encoder.

Why configurable n_stages: the multi-stage (N > 1) path supports detached
residual-decomposition ablations. In the canonical configuration (N=1),
multi-stage boosting did not earn its parameters in this small-N residual-
target regime; TabM ensembling alone drove the gain. The N>1 paths are
retained so ablations can toggle the multi-stage behaviour without
additional scaffolding.

Contract:
    forward(z_encoder, metadata) -> dict with
        prediction: [B]    = sum of present stage scalars
        stage_k:    [B]    = f̂_k scalar  (only for k <= n_stages)
        latent_k:   [B, d] = h_k pre-readout latent  (only for k <= n_stages)

Architecture:
    z_cond = FiLM(z_encoder, metadata)
    Stage 1:  h_1 = TabM[NPTStage](z_cond);                  f̂_1 = readout_1(h_1)
    Stage 2:  ctx_2 = cross_stage_attention(z_cond, [h_1.detach()])
              h_2 = TabM[NPTStage](z_cond + ctx_2);           f̂_2 = readout_2(h_2)
    Stage 3:  ctx_3 = cross_stage_attention(z_cond, [h_1.detach(), h_2.detach()])
              h_3 = TabM[NPTStage](z_cond + ctx_3);           f̂_3 = readout_3(h_3)

Stages 2 and 3 are only built when n_stages >= 2 / 3 — saves parameters /
optimizer state / weight decay regularization on dropped stages.

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

# Number of TabM BatchEnsemble members per stage. 8 is the TabM paper default and
# matches the value used across HPO configs / ablation sweeps in this project.
DEFAULT_K_TABM = 8

# Default boosting depth. Multi-stage boosting did not earn its parameters in
# this small-N residual-target regime; TabM ensembling alone drove the gain,
# so the canonical configuration uses a single stage. n_stages > 1 remains
# available for ablation comparisons.
DEFAULT_N_STAGES = 1

# Allowed values for n_stages. Keep narrow — anything beyond 3 has no design
# justification in the plan doc and would silently expand the loss formula.
_VALID_N_STAGES = (1, 2, 3)


def _make_npt_tabm(d_subject: int, n_heads: int, n_hc_streams: int,
                   lambda_init: float, k_tabm: int,
                   use_diff_attn: bool = True,
                   use_hyper_conn: bool = True) -> tuple[NPTStage, TabMWrapper]:
    """Build a (NPTStage, TabMWrapper) pair for one boosting stage.

    NPTStage is constructed with emit_scalar=False because TabMWrapper discards
    sub_out[1] — keeping a stage-internal scalar readout would be dead weight.

    use_diff_attn / use_hyper_conn forwarded to NPTStage so ablation configs can
    toggle those components (vanilla MHA / plain residual) without rewiring.
    """
    npt = NPTStage(
        d_subject=d_subject, n_heads=n_heads,
        n_hc_streams=n_hc_streams, lambda_init=lambda_init,
        emit_scalar=False,
        use_diff_attn=use_diff_attn,
        use_hyper_conn=use_hyper_conn,
    )
    tabm = TabMWrapper(submodule=npt, d_in=d_subject, d_out=d_subject, k=k_tabm)
    return npt, tabm


class ResDecMHEHead(nn.Module):
    def __init__(
        self,
        d_subject: int = 64,
        d_metadata: int = 8,
        n_heads: int = 4,
        n_hc_streams: int = 4,
        lambda_init: float = 0.8,
        k_tabm: int = DEFAULT_K_TABM,
        n_stages: int = DEFAULT_N_STAGES,
        use_film: bool = True,
        use_diff_attn: bool = False,   # canonical: vanilla MHA
        use_hyper_conn: bool = True,
    ):
        super().__init__()
        if n_stages not in _VALID_N_STAGES:
            raise ValueError(
                f"n_stages must be one of {_VALID_N_STAGES}; got {n_stages}"
            )
        self.d_subject = d_subject
        self.k_tabm = k_tabm
        self.n_stages = n_stages
        self.use_film = use_film

        if use_film:
            self.film = FiLMMetadata(d_subject=d_subject, d_metadata=d_metadata)
        # When use_film=False, no module is constructed; forward bypasses it
        # entirely and feeds z_encoder directly to stage_1 (no-FiLM ablation).

        # NPTStage internals are toggleable via use_diff_attn / use_hyper_conn
        # (ablation flags) — forwarded into _make_npt_tabm.
        npt_kwargs = {
            "d_subject": d_subject, "n_heads": n_heads,
            "n_hc_streams": n_hc_streams, "lambda_init": lambda_init,
            "k_tabm": k_tabm,
            "use_diff_attn": use_diff_attn,
            "use_hyper_conn": use_hyper_conn,
        }

        # ---- Stage 1: always present ----
        self.stage_1_npt, self.stage_1_tabm = _make_npt_tabm(**npt_kwargs)
        self.stage_1_readout = nn.Linear(d_subject, 1)

        # ---- Stage 2: only if n_stages >= 2 ----
        if n_stages >= 2:
            self.stage_2_cross_attn = CrossStageAttention(
                d_subject=d_subject, n_heads=n_heads,
            )
            self.stage_2_npt, self.stage_2_tabm = _make_npt_tabm(**npt_kwargs)
            self.stage_2_readout = nn.Linear(d_subject, 1)

        # ---- Stage 3: only if n_stages >= 3 ----
        if n_stages >= 3:
            self.stage_3_cross_attn = CrossStageAttention(
                d_subject=d_subject, n_heads=n_heads,
            )
            self.stage_3_npt, self.stage_3_tabm = _make_npt_tabm(**npt_kwargs)
            self.stage_3_readout = nn.Linear(d_subject, 1)

    def forward(self, z_encoder: torch.Tensor, metadata: torch.Tensor) -> dict:
        """
        z_encoder: [B, d_subject] subject embedding from CognitiveResilienceModel
        metadata:  [B, d_metadata] FiLM-conditioning vector (APOE/sex/age)

        Returns dict with `prediction` (sum of present stage scalars) plus
        `stage_k` and `latent_k` for each k in [1, n_stages]. Absent stages
        are NOT in the dict — downstream consumers must use `.get()` or
        guard on n_stages.
        """
        # Bypass FiLM entirely when use_film=False (no-FiLM ablation). When
        # enabled, FiLM modulates z_encoder by metadata-derived γ/β.
        if self.use_film:
            z_cond = self.film(z_encoder, metadata)
        else:
            z_cond = z_encoder

        # ------- Stage 1 (always) -------
        h_1, _ = self.stage_1_tabm(z_cond)                # [B, d_subject]
        scalar_1 = self.stage_1_readout(h_1).squeeze(-1)  # [B]
        out: dict[str, torch.Tensor] = {
            "stage_1": scalar_1,
            "latent_1": h_1,
        }
        prediction = scalar_1

        # ------- Stage 2 -------
        # Prior latents are .detach()'d before cross-stage attention so that
        # aux_k losses (k > 1) cannot back-propagate into earlier stages'
        # TabM/readout params — this is the detached-residual-boosting contract
        # (stage-2 gradient does NOT flow into stage-1).
        # L_main still trains every stage jointly via its own non-detached path.
        if self.n_stages >= 2:
            ctx_2 = self.stage_2_cross_attn(z_cond, [h_1.detach()])  # [B, d]
            h_2, _ = self.stage_2_tabm(z_cond + ctx_2)
            scalar_2 = self.stage_2_readout(h_2).squeeze(-1)
            out["stage_2"] = scalar_2
            out["latent_2"] = h_2
            prediction = prediction + scalar_2

        # ------- Stage 3 -------
        if self.n_stages >= 3:
            ctx_3 = self.stage_3_cross_attn(
                z_cond, [h_1.detach(), h_2.detach()],
            )
            h_3, _ = self.stage_3_tabm(z_cond + ctx_3)
            scalar_3 = self.stage_3_readout(h_3).squeeze(-1)
            out["stage_3"] = scalar_3
            out["latent_3"] = h_3
            prediction = prediction + scalar_3

        # The composer's "prediction" is the residual sum of present stages.
        # Main loss in the Lightning module trains this sum against
        # ``y - ŷ_tabpfn``; val-time composite prediction adds ``ŷ_tabpfn`` back.
        out["prediction"] = prediction
        return out
