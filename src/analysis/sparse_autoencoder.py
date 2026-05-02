"""Sparse autoencoder for ResDec-MHE encoder hidden states.

Implementation of Orlov 2026 (bioRxiv 2026.03.04.709491v1) faithful to the
TopK and Batch-TopK formulations, with the Gao 2024 / Bussmann 2024 reference
implementation details adopted (auxiliary-K loss for dead-feature revival,
unit-norm decoder columns, decoder-bias init = mean of activations).

Reference: Orlov, A. V. et al. (2026). *What Do Biological Foundation Models
Compute? Sparse Autoencoders from Feature Recovery to Mechanistic
Interpretability.* bioRxiv 2026.03.04.709491v1.
PDF on disk at ``docs/2026.03.04.709491v1.full.pdf``.

Design summary (see ``docs/plans/2026-04-28-sparse-autoencoder-design.md``):

* SAE form (Orlov §3.1, p.6): encoder ``h = activation(W_enc @ x + b_enc)``,
  decoder ``x_hat = W_dec @ h + b_dec``; loss balances reconstruction MSE
  against sparsity of ``h``.
* Primary architecture: Batch-TopK (Bussmann et al. 2024, Orlov ref [85];
  §3.1.2, p.7-8). Secondary: TopK (Gao et al. 2024, Orlov ref [17]).
* Extraction sites in ResDec-MHE (``src/models/full_model.py``):

  - ``attended`` ``[B, d_fused=64]`` — line 547, post-PathologyStratifiedAttention,
    sole input to the prediction head.
  - ``fused`` ``[B, 31, d_fused=64]`` — line 534, post-FusionLayer per cell type.

  Both are returned in the ``embeddings`` dict at ``full_model.py:574-580``
  when ``forward(..., return_embeddings=True)``.

* d_fused = 64 from ``configs/default.yaml:78`` (canonical inherits, see
  ``configs/resdec_mhe/canonical.yaml`` which does not override).
* Expansion sweep: ``{8, 16, 32}x`` per Orlov §3.3.3 — small models like ours
  benefit from larger expansion.
* Sparsity sweep (TopK / Batch-TopK): ``K in {4, 8, 16, 32, 64}``.
* Training data: 5 canonical-checkpoint encoders × 516 subjects × ``return_embeddings``
  forward, persisted as ``.npz`` per fold under ``outputs/canonical/sae/``.

DEVIATIONS FROM ORLOV LITERAL EQUATIONS (user-approved):

1. Unit-norm decoder column constraint (Bussmann 2024, Gao 2024 standard) —
   ``W_dec[:, j] /= ||W_dec[:, j]||₂`` projected after every optimizer step.
   Required for stable magnitude/direction decomposition that makes feature
   interpretation meaningful.
2. Decoder bias initialised to the mean of the training activations
   (per Bussmann 2024 reference impl).

DEVIATION FROM PAPER (Gao 2024) — DEAD-WINDOW DEFINITION:

Gao 2024 specifies the dead-feature window as a *fixed* 1e7 tokens absolute
budget (independent of training duration). We instead use a *fraction* of
total optimisation steps (``DEAD_WINDOW_FRAC = 0.125``) so the window scales
naturally with sweep budgets that vary in length (e.g. 50 K vs 100 K steps).
For our N=516 cohort × 50 K steps × batch 64, this corresponds to ~6.25 K
steps ≈ 12.5 epochs — comparable to Gao 2024 in absolute terms at our scale.

DEVIATION FROM DESIGN DOC (§8.2 criterion 3) — INPUT-GRADIENT XREF:

The design doc §8.2 defines the third per-feature interpretability criterion
as "the top-3 input-gradient (CT, gene) pairs include at least one short-list
gene." This criterion is NOT computed inside :func:`interpret_features` (which
operates on raw activation matrices, not gradients through the full model).
Cross-referencing against Captum / GradShap / Wasserstein / CMI top-pair
short-lists is deferred to the post-hoc orchestrator
``run_feature_xref_consensus.py``. This deviation is documented here AND in
the design doc; the orchestrator delivers the equivalent functionality at the
JSON-comparison level rather than inside the per-feature loop.

DEVIATION FROM USER BRIEF (flagged):

* The brief says "load checkpoint via Predictor.from_checkpoint(...)". The
  canonical ResDec-MHE checkpoints under
  ``outputs/canonical/p5_canonical_seed42/fold{0..4}/checkpoints/`` are saved
  by ``ResDecLightningModule`` and contain only encoder + head state_dicts —
  no ``full_config`` or ``model_config`` keys, so ``Predictor.from_checkpoint``
  raises ValueError. We use ``ResDecLightningModule.load_from_checkpoint(...)``
  instead and call ``model.encoder(...)`` (the full ``CognitiveResilienceModel``)
  with ``return_embeddings=True``. This matches the canonical interpretability
  pattern in ``scripts/resdec_mhe/interpretability/captum_composite_attribution.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import mannwhitneyu

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Project root resolved from this file's location.
# /host/.../refinement-two/src/analysis/sparse_autoencoder.py → parents[2] = repo root.
# (parents[0] = src/analysis, [1] = src, [2] = repo root.)
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


# Dead-feature window per Gao 2024: a feature is "dead" if its
# fraction_active < ``DEAD_FRACTION_THRESHOLD`` over the most recent
# ``DEAD_WINDOW_FRAC`` × n_steps optimization steps. The user-approved
# 12.5-epoch window for our N≈80k samples corresponds to a fixed fraction of
# total optimisation steps, capped at the value below.
DEAD_FRACTION_THRESHOLD: float = 1e-4
DEAD_WINDOW_FRAC: float = 0.125  # 12.5% of total steps (≈12.5 epochs at 1 epoch ≈ 1% steps)


@dataclass
class SAEConfig:
    """Hyperparameter container for one SAE training run.

    Parameters
    ----------
    architecture
        ``"topk"`` (Gao et al. 2024 / Orlov §3.1.1) or ``"batch_topk"``
        (Bussmann et al. 2024 / Orlov §3.1.2). The L1 / shrinkage architecture
        is intentionally NOT supported (subject to shrinkage bias per Orlov
        §3.1.1; not used in the canonical sweep).
    expansion
        Dictionary expansion factor m / n. Orlov §3.3.3: 8x for VT, 16-32x
        for small PLM (our scale).
    k
        For ``topk`` / ``batch_topk``: number of active features per sample
        (TopK) or per ``n``-sample batch averaged (Batch-TopK). Orlov Table 2.
    aux_lambda
        Auxiliary-K loss weight for dead-feature revival (Gao et al. 2024).
        Default ``1/32`` per Orlov ref [17].
    aux_k
        Number of dead features used in the auxiliary reconstruction loss.
    decoder_unit_norm
        If True, normalize each decoder column to unit L2 norm after every
        optimizer step. *Not stated in Orlov's literal equations* — adopted
        from Gao 2024 / Bussmann 2024 implementations. Marked as a deviation
        from the paper in the design doc; user-approved.
    learning_rate
    batch_size
    n_steps
        Optimizer settings. Defaults match Bussmann et al. 2024 / Gao 2024.
    seed
    """

    architecture: Literal["topk", "batch_topk"]
    expansion: int
    k: int | None = None
    aux_lambda: float = 1.0 / 32.0
    aux_k: int = 256
    decoder_unit_norm: bool = True
    learning_rate: float = 1e-4
    batch_size: int = 64
    n_steps: int = 100_000
    seed: int = 0


@dataclass
class SAEModel:
    """Trained SAE state — pure-numpy container.

    Parameters
    ----------
    W_enc
        ``[m, n]`` encoder weight matrix.
    b_enc
        ``[m]`` encoder bias.
    W_dec
        ``[n, m]`` decoder weight matrix. If ``config.decoder_unit_norm``,
        each column ``W_dec[:, j]`` has unit L2 norm.
    b_dec
        ``[n]`` decoder bias (a.k.a. pre-encoder centering term in some
        implementations).
    config
        The ``SAEConfig`` used for training.
    activation_stats
        Per-feature activation statistics computed at the end of training:
        ``{"mean": [m], "std": [m], "fraction_active": [m], "is_dead": [m]}``.
        For ``architecture == "batch_topk"``, also includes ``"threshold"``
        (the inference-time scalar threshold derived from a running average
        of the batch K-th largest pre-activation; per Bussmann 2024).
    """

    W_enc: np.ndarray
    b_enc: np.ndarray
    W_dec: np.ndarray
    b_dec: np.ndarray
    config: SAEConfig
    activation_stats: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class ActivationBundle:
    """Container for ResDec-MHE activations extracted at a single layer.

    Parameters
    ----------
    activations
        ``[N, n]`` for ``layer == "attended"`` (one vector per subject)
        or ``[N, 31, n]`` for ``layer == "fused"`` (one vector per
        (subject, cell-type) pair).
    subject_ids
        ``[N]`` ROSMAP projid (or equivalent) per row.
    fold_indices
        ``[N]`` integer fold ∈ {0,1,2,3,4} that produced this activation.
    is_val
        ``[N]`` boolean — whether the subject was in val of its fold.
    cell_types
        ``[31]`` cell-type names; populated only when ``layer == "fused"``.
    layer
        Either ``"attended"`` or ``"fused"`` — matches the ResDec-MHE
        embedding-dict keys (see ``CognitiveResilienceModel.forward``
        ``return_embeddings=True`` output at ``full_model.py:574-580``).
    """

    activations: np.ndarray
    subject_ids: np.ndarray
    fold_indices: np.ndarray
    is_val: np.ndarray
    cell_types: np.ndarray | None
    layer: Literal["attended", "fused"]


# ─────────────────────────────────────────────────────────────────────────────
# Internal SAE torch module — not exported. The public API converts to/from
# numpy arrays so that downstream callers don't take a torch dependency.
# ─────────────────────────────────────────────────────────────────────────────


class _SAETorch(nn.Module):
    """Torch implementation of the (Batch-)TopK SAE used internally for training.

    Encoder: ``h_pre = ReLU(W_enc @ (x - b_dec) + b_enc)``.

    Per Bussmann 2024 / Gao 2024, we subtract ``b_dec`` from ``x`` before the
    encoder so that ``b_dec`` plays the role of the "pre-encoder centering
    term" — equivalent to absorbing the data mean into the decoder bias.

    Decoder: ``x_hat = W_dec @ h + b_dec``.

    For TopK: ``h = TopK(h_pre, k)`` per-sample.
    For Batch-TopK: keep the largest ``n_batch * k`` values across the entire
    batch's flattened ``h_pre``.
    """

    def __init__(self, n: int, m: int, config: SAEConfig):
        super().__init__()
        self.n = n
        self.m = m
        self.config = config

        # Encoder bias init = zero (per user spec).
        self.b_enc = nn.Parameter(torch.zeros(m))
        # Decoder bias init = mean of activations (set later via init_decoder_bias).
        self.b_dec = nn.Parameter(torch.zeros(n))

        # Decoder columns: random unit-norm (per user spec).
        # Sample standard-normal then normalize columns.
        W_dec = torch.randn(n, m)
        W_dec = W_dec / (W_dec.norm(dim=0, keepdim=True) + 1e-8)
        self.W_dec = nn.Parameter(W_dec)

        # Encoder weights = decoder transpose at init (Gao 2024 / Bussmann 2024
        # standard). This is a sensible starting point and is overwritten by
        # training.
        self.W_enc = nn.Parameter(W_dec.t().clone())

        # Inference threshold for Batch-TopK (set during training; running mean
        # of the per-batch K-th-largest pre-activation). Stored as a buffer so
        # it survives state_dict round-trips.
        self.register_buffer("inference_threshold", torch.tensor(0.0))

    @torch.no_grad()
    def init_decoder_bias(self, activations: torch.Tensor) -> None:
        """Set decoder bias to the mean of training activations (Bussmann 2024)."""
        self.b_dec.copy_(activations.mean(dim=0))

    def encode_pre(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-activation: ReLU(W_enc @ (x - b_dec) + b_enc)."""
        return F.relu(F.linear(x - self.b_dec, self.W_enc, self.b_enc))

    def encode_topk(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """Per-sample TopK: keep the k largest pre-activations per row."""
        h_pre = self.encode_pre(x)  # [B, m]
        if k >= h_pre.shape[1]:
            return h_pre
        topk_vals, topk_idx = h_pre.topk(k, dim=1)
        h = torch.zeros_like(h_pre)
        h.scatter_(1, topk_idx, topk_vals)
        return h

    def encode_batch_topk(
        self, x: torch.Tensor, k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch-TopK: keep the n_batch * k largest pre-activations across batch.

        Returns the sparse code AND the K-th-largest pre-activation value for
        the batch (used to maintain a running threshold for inference).

        F4: the threshold is returned as a 0-d torch tensor to avoid the
        per-step GPU→CPU sync that ``.item()`` would force. The training
        loop only realises ``.item()`` once at the end of the run for
        ``activation_stats["threshold"]``.
        """
        h_pre = self.encode_pre(x)  # [B, m]
        n_batch = h_pre.shape[0]
        budget = n_batch * k
        flat = h_pre.reshape(-1)
        if budget >= flat.numel():
            return h_pre, torch.zeros((), device=h_pre.device, dtype=h_pre.dtype)
        topk_vals, topk_idx = flat.topk(budget)
        h = torch.zeros_like(flat)
        h.scatter_(0, topk_idx, topk_vals)
        h = h.reshape(h_pre.shape)
        # Threshold: smallest of the kept values (the K-th largest in the
        # batch). 0-d tensor; .detach() so it doesn't track the autograd
        # graph (we only use it for the inference-time EMA, not for the loss).
        threshold_val = topk_vals[-1].detach()
        return h, threshold_val

    def encode_threshold(
        self, x: torch.Tensor, threshold: float | torch.Tensor,
    ) -> torch.Tensor:
        """Inference-time encode for Batch-TopK: zero out values <= threshold.

        ``threshold`` may be a Python float OR a 0-d tensor (the F4 path keeps
        the EMA on-GPU through training; the final ``activation_stats``
        materialises a Python float).
        """
        h_pre = self.encode_pre(x)
        return torch.where(h_pre > threshold, h_pre, torch.zeros_like(h_pre))

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        return F.linear(h, self.W_dec, self.b_dec)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (h_pre, h, x_hat) for a per-sample TopK forward."""
        # ``self.config.k`` is validated to be a positive int by SAEConfig
        # (see _train_sae_torch's positive-int check), but the class
        # annotation is ``int | None``. Hoist into a local int so the
        # encode_* call sites do not need a per-call type: ignore.
        k_int: int = int(self.config.k) if self.config.k is not None else 0
        if self.config.architecture == "topk":
            h_pre = self.encode_pre(x)
            h = self.encode_topk(x, k_int)
        else:  # "batch_topk" (Literal in SAEConfig restricts to {topk, batch_topk}).
            h_pre = self.encode_pre(x)
            h, _ = self.encode_batch_topk(x, k_int)
        x_hat = self.decode(h)
        return h_pre, h, x_hat

    @torch.no_grad()
    def project_decoder_unit_norm(self) -> None:
        """Renormalize each decoder column to unit L2 norm.

        F5: in-place norm + clamp + div to avoid the temporary tensors that
        the ``.norm(...).clamp(...)`` chain would otherwise produce. Numerics
        are bit-equivalent to the prior expression because ``vector_norm`` is
        the same reduction as ``.norm`` and the clamp + div are element-wise
        operations on identical inputs.
        """
        norms = torch.linalg.vector_norm(self.W_dec, dim=0, keepdim=True)
        norms.clamp_(min=1e-8)
        self.W_dec.div_(norms)


# ─────────────────────────────────────────────────────────────────────────────
# Internal training loop shared by TopK and Batch-TopK
# ─────────────────────────────────────────────────────────────────────────────


def _train_sae_torch(
    activations: np.ndarray,
    config: SAEConfig,
) -> SAEModel:
    """Generic torch training loop for TopK / Batch-TopK SAE.

    Implements Adam + cosine decay (Bussmann 2024 / Gao 2024 defaults) and
    the Gao 2024 auxiliary-K dead-feature loss.
    """
    if activations.ndim != 2:
        raise ValueError(
            f"activations must be [N, n] 2D; got shape {activations.shape}"
        )
    n = activations.shape[1]
    m = int(config.expansion * n)

    if config.k is None or config.k <= 0:
        raise ValueError("k must be a positive integer")
    if config.k > m:
        raise ValueError(f"k ({config.k}) must be <= m={m}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    sae = _SAETorch(n=n, m=m, config=config).to(device)
    x_full = torch.from_numpy(activations.astype(np.float32)).to(device)

    # Initialize decoder bias at the data mean (Bussmann 2024 ref impl).
    sae.init_decoder_bias(x_full)

    optimizer = torch.optim.Adam(sae.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, config.n_steps), eta_min=0.0
    )

    # Activity tracking: number of consecutive steps for which each feature
    # has been "inactive" (zero activation). Re-set to 0 whenever the feature
    # fires.
    last_active_step = torch.full((m,), -1, device=device, dtype=torch.long)

    # Window length (in optimisation steps) for "dead" status determination.
    dead_window = max(1, int(DEAD_WINDOW_FRAC * config.n_steps))

    # Running mean of the per-batch K-th largest pre-activation (used as
    # inference threshold for Batch-TopK). Bussmann 2024 uses an EMA with
    # heavy weight on the recent values; the threshold only matters for
    # inference, so we want it to track the converged distribution rather
    # than the early-training one.
    #
    # F4: keep the EMA on-device as a 0-d float32 tensor to avoid a per-step
    # GPU→CPU sync. ``activation_stats["threshold"]`` is materialised once at
    # the end of training. EMA arithmetic is mathematically identical (same
    # alpha, same operands); only the precision changes from Python fp64 to
    # GPU fp32. Verified within fp32 epsilon (1e-6) by the unit test
    # ``test_train_sae_batch_topk_threshold_within_epsilon``.
    running_threshold = torch.zeros((), device=device, dtype=torch.float32)
    running_threshold_alpha: float = 0.95
    one_minus_alpha: float = 1.0 - running_threshold_alpha
    # Warm-up: start the EMA only after this many steps so it represents the
    # converged regime rather than untrained-encoder noise.
    threshold_warmup_steps: int = max(1, config.n_steps // 5)

    # Random batch sampling — N may be small, so we just sample with
    # replacement.
    N = x_full.shape[0]

    for step in range(config.n_steps):
        idx = torch.randint(low=0, high=N, size=(config.batch_size,), device=device)
        x = x_full[idx]

        # Forward
        h_pre = sae.encode_pre(x)
        if config.architecture == "topk":
            h = sae.encode_topk(x, config.k)
            batch_threshold: torch.Tensor | None = None
        else:  # "batch_topk" (Literal in SAEConfig restricts to {topk, batch_topk}).
            h, batch_threshold = sae.encode_batch_topk(x, config.k)

        x_hat = sae.decode(h)
        recon_loss = F.mse_loss(x_hat, x)

        # Auxiliary-K loss for dead-feature revival (Gao 2024).
        # A feature is "dead" if it has not activated in the last
        # ``dead_window`` steps. Use the top-k_aux *dead* pre-activations.
        is_dead = (last_active_step < (step - dead_window))
        aux_loss = torch.zeros((), device=device)
        if is_dead.any() and config.aux_lambda > 0 and config.aux_k > 0:
            n_dead = int(is_dead.sum().item())
            kk = min(config.aux_k, n_dead)
            if kk > 0:
                # Mask non-dead pre-activations to -inf, then take top-k_aux
                # *across the dead-feature columns only*. By construction
                # ``kk <= n_dead``, so every column index returned by topk
                # corresponds to a dead feature whose pre-activation is finite
                # (ReLU output ≥ 0). The -inf values cannot survive the
                # top-kk selection: they are smaller than every finite
                # ReLU output, so torch.topk never chooses them while at least
                # ``kk`` finite columns are available — and that condition is
                # guaranteed by ``kk = min(aux_k, n_dead)``.
                masked = h_pre.clone()
                masked[:, ~is_dead] = float("-inf")
                topk_vals, topk_idx = masked.topk(kk, dim=1)
                assert torch.isfinite(topk_vals).all(), (
                    "aux-K topk returned -inf — invariant violation: "
                    f"kk={kk}, n_dead={n_dead}, h_pre.shape={tuple(h_pre.shape)}"
                )
                h_aux = torch.zeros_like(h_pre)
                h_aux.scatter_(1, topk_idx, topk_vals)
                # Reconstruct the residual (x - x_hat) from the dead features.
                # This is the Gao 2024 formulation: dead features try to
                # explain what main features missed.
                with torch.no_grad():
                    residual = (x - x_hat).detach()
                x_aux_hat = sae.decode(h_aux) - sae.b_dec  # decoder without bias
                aux_loss = F.mse_loss(x_aux_hat, residual)

        loss = recon_loss + config.aux_lambda * aux_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Re-project decoder columns to unit norm (Bussmann 2024 / Gao 2024).
        if config.decoder_unit_norm:
            sae.project_decoder_unit_norm()

        # Update activity tracker: features with any positive activation in
        # this batch are "live".
        with torch.no_grad():
            active_mask = (h.abs().sum(dim=0) > 0)
            # F8: in-place ``masked_fill_`` instead of
            # ``where(active_mask, full_like(...), last_active_step)`` —
            # avoids the per-step ``full_like`` allocation. Numerically
            # identical: both paths set positions where ``active_mask``
            # is true to the integer ``step`` and leave the rest unchanged.
            last_active_step.masked_fill_(active_mask, step)
            # Update running threshold for Batch-TopK inference. Skip the
            # warm-up phase so the threshold reflects the converged regime,
            # then EMA-track during the rest of training.
            #
            # F4: arithmetic stays on-GPU as a 0-d fp32 tensor; we never call
            # ``.item()`` here. The result is materialised to a Python float
            # exactly once after training finishes (see below).
            if config.architecture == "batch_topk" and step >= threshold_warmup_steps:
                # batch_threshold is set in the batch_topk branch above; the
                # outer architecture check above guarantees we don't enter
                # this block in the topk path. Hoist into a local non-None
                # alias so downstream calls don't need a per-call type:
                # ignore.
                assert batch_threshold is not None
                bt: torch.Tensor = batch_threshold
                if step == threshold_warmup_steps:
                    running_threshold.copy_(bt)
                else:
                    running_threshold.mul_(running_threshold_alpha).add_(
                        bt, alpha=one_minus_alpha,
                    )

    # Final stats over the full dataset.
    # F4: materialise the running EMA exactly once here (single GPU→CPU sync
    # at end-of-training) instead of per-step.
    running_threshold_value = float(running_threshold.detach().item())
    sae.eval()
    with torch.no_grad():
        if config.architecture == "topk":
            h_full = sae.encode_topk(x_full, config.k)
        else:  # "batch_topk" — at inference, use the running threshold (Bussmann 2024).
            h_full = sae.encode_threshold(x_full, running_threshold_value)

        feature_active = (h_full.abs() > 0).float()
        fraction_active = feature_active.mean(dim=0).cpu().numpy()
        feature_mean = h_full.mean(dim=0).cpu().numpy()
        feature_std = h_full.std(dim=0, unbiased=False).cpu().numpy()
        is_dead_final = (fraction_active < DEAD_FRACTION_THRESHOLD)

    activation_stats: dict[str, np.ndarray] = {
        "mean": feature_mean.astype(np.float32),
        "std": feature_std.astype(np.float32),
        "fraction_active": fraction_active.astype(np.float32),
        "is_dead": is_dead_final.astype(np.bool_),
    }
    if config.architecture == "batch_topk":
        activation_stats["threshold"] = np.array(
            [running_threshold_value], dtype=np.float32,
        )

    return SAEModel(
        W_enc=sae.W_enc.detach().cpu().numpy().astype(np.float32),
        b_enc=sae.b_enc.detach().cpu().numpy().astype(np.float32),
        W_dec=sae.W_dec.detach().cpu().numpy().astype(np.float32),
        b_dec=sae.b_dec.detach().cpu().numpy().astype(np.float32),
        config=config,
        activation_stats=activation_stats,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Encode helpers (numpy-only inference) — used by evaluate / interpret
# ─────────────────────────────────────────────────────────────────────────────


def _encode_numpy(sae: SAEModel, x: np.ndarray) -> np.ndarray:
    """Compute the sparse code h for one or many activation vectors.

    Implements the same forward as ``_SAETorch`` but in pure numpy so
    downstream callers don't need torch.
    """
    if x.ndim == 1:
        x = x[None, :]
    # Pre-encoder centering: subtract decoder bias.
    z = x - sae.b_dec[None, :]
    h_pre = z @ sae.W_enc.T + sae.b_enc[None, :]
    h_pre = np.maximum(h_pre, 0.0)

    cfg = sae.config
    if cfg.architecture == "topk":
        k = cfg.k
        if k is None or k >= h_pre.shape[1]:
            return h_pre.astype(np.float32)
        # Per-sample top-k: zero out everything except the k largest per row.
        h = np.zeros_like(h_pre)
        topk_idx = np.argpartition(-h_pre, k - 1, axis=1)[:, :k]
        rows = np.arange(h_pre.shape[0])[:, None]
        h[rows, topk_idx] = h_pre[rows, topk_idx]
        return h.astype(np.float32)
    else:  # "batch_topk" (Literal in SAEConfig restricts to {topk, batch_topk}).
        threshold = float(sae.activation_stats.get("threshold", np.array([0.0]))[0])
        h = np.where(h_pre > threshold, h_pre, 0.0)
        return h.astype(np.float32)


def _decode_numpy(sae: SAEModel, h: np.ndarray) -> np.ndarray:
    if h.ndim == 1:
        h = h[None, :]
    return (h @ sae.W_dec.T + sae.b_dec[None, :]).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def extract_activations(
    checkpoint_paths: list[Path],
    layer: Literal["attended", "fused"],
    output_dir: Path,
    *,
    device: str = "cuda",
    batch_size: int = 32,
) -> ActivationBundle:
    """Forward all subjects through each fold's canonical encoder and persist activations.

    For each checkpoint, build the canonical dataloader (PFC slice, 31 CTs,
    4785 genes), call the model with ``return_embeddings=True``, collect the
    requested ``embeddings[layer]`` tensor for every subject, and concatenate
    across folds into a single ``ActivationBundle``.

    Persists ``output_dir / f"activations_{layer}_fold{f}.npz"`` per fold and a
    combined ``output_dir / f"activations_{layer}_all_folds.npz"`` for
    reproducibility.

    Parameters
    ----------
    checkpoint_paths
        List of (typically 5) paths to ``best-*.ckpt`` files under
        ``outputs/canonical/p5_canonical_seed42/fold{0..4}/checkpoints/``.
    layer
        ``"attended"`` (``[B, 64]`` post-PathologyStratifiedAttention,
        ``full_model.py:547``) or ``"fused"`` (``[B, 31, 64]``
        post-FusionLayer, ``full_model.py:534``).
    output_dir
        Destination directory; usually ``PROJECT_ROOT / "outputs" / "redesign" / "sae"``.
    device
        ``"cuda"`` or ``"cpu"``.
    batch_size
        Forward-pass batch size; inference is non-backprop, so batch can
        be larger than training.

    Returns
    -------
    ActivationBundle
        With ``activations`` (``[N_total, 64]`` for ``attended`` or
        ``[N_total, 31, 64]`` for ``fused``), ``subject_ids``, ``fold_indices``,
        ``is_val``, ``cell_types`` (only when ``layer == "fused"``), ``layer``.

    Notes
    -----
    Loads the ResDec-MHE canonical checkpoint via
    ``ResDecLightningModule.load_from_checkpoint`` (NOT
    ``Predictor.from_checkpoint`` — see DEVIATION FROM USER BRIEF in module
    docstring). Forward is wrapped in ``torch.no_grad()`` and ``model.eval()``.

    Implementation note (F6 deferred — design constraint):
        The per-fold ``CognitiveResilienceDataModule`` rebuild on lines below
        cannot be hoisted out of the loop. ``CognitiveResilienceDataModule``
        is constructed with ``fold_idx`` as a required parameter (see
        ``src/data/datamodule.py:56-67``) and ``setup`` partitions the cohort
        into train / val / test using ``get_fold_subjects(splits, fold_idx,
        ...)``; there is no cohort-wide enumeration mode in the data layer.
        Adding one would require a parallel "sequencer" path that bypasses
        the train / val split entirely — outside the scope of this
        optimisation pass. ``extract_activations`` is one-shot per project
        (5-fold extract is run once at the start of the SAE pipeline), so
        the cost is bounded.
    """
    if layer not in ("attended", "fused"):
        raise ValueError(f"layer must be 'attended' or 'fused'; got {layer!r}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Local imports — keep the module light when called only for SAE training.
    import pandas as pd
    from omegaconf import OmegaConf

    from src.data.constants import CELL_TYPE_ORDER
    from src.data.datamodule import CognitiveResilienceDataModule
    from src.data.splits import get_fold_subjects, load_splits
    from src.training.resdec_lightning_module import ResDecLightningModule
    from src.utils.provenance import git_sha

    # Resolve config — canonical phase config merged on top of default. The
    # checkpoint itself does not carry a config (verified: no
    # ``full_config`` / ``model_config`` key), so we rely on the canonical
    # config files.
    cfg = OmegaConf.merge(
        OmegaConf.load(PROJECT_ROOT / "configs" / "default.yaml"),
        OmegaConf.load(PROJECT_ROOT / "configs" / "resdec_mhe" / "canonical.yaml"),
    )
    OmegaConf.set_struct(cfg, False)
    cfg.model.head.type = "deterministic"

    splits_path = PROJECT_ROOT / "outputs" / "splits.json"
    splits = load_splits(str(splits_path))
    metadata_path = Path(cfg.data.metadata_path) / "metadata.csv"
    if not metadata_path.is_absolute():
        metadata_path = PROJECT_ROOT / metadata_path
    metadata_csv = pd.read_csv(metadata_path)

    torch_device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")

    per_fold_activations: list[np.ndarray] = []
    per_fold_sids: list[np.ndarray] = []
    per_fold_idx: list[np.ndarray] = []
    per_fold_is_val: list[np.ndarray] = []

    cell_types_array: np.ndarray | None = None
    if layer == "fused":
        cell_types_array = np.array(list(CELL_TYPE_ORDER), dtype=object)

    for fold, ckpt_path in enumerate(checkpoint_paths):
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        fold_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        OmegaConf.set_struct(fold_cfg, False)
        fold_cfg.data.fold = int(fold)

        dm = CognitiveResilienceDataModule(
            config=fold_cfg, metadata=metadata_csv, splits=splits,
            fold_idx=int(fold),
            precomputed_dir=fold_cfg.data.precomputed_dir,
            adata=None,
        )
        dm.setup(stage="fit")

        # Build the set of val subjects for this fold to mark the is_val flag.
        val_subjects: set[str] = set(
            map(str, get_fold_subjects(splits, fold_idx=int(fold), split_type="val"))
        )

        logger.info("fold %d: loading %s", fold, ckpt_path.name)
        model = ResDecLightningModule.load_from_checkpoint(
            str(ckpt_path), config=fold_cfg, map_location="cpu",
        ).to(torch_device).eval()

        # Iterate over both train and val loaders so we cover all 516 subjects
        # for this fold.
        loaders = [dm.train_dataloader(), dm.val_dataloader()]

        fold_acts: list[np.ndarray] = []
        fold_sids: list[str] = []
        fold_is_val: list[bool] = []

        with torch.no_grad():
            for loader in loaders:
                if loader is None:
                    continue
                for batch in loader:
                    # Move tensors to device.
                    batch_d = {
                        k: (v.to(torch_device) if torch.is_tensor(v) else v)
                        for k, v in batch.items()
                    }
                    sids = list(batch_d.get("subject_ids", []))

                    enc_out = model.encoder(
                        region_pseudobulk=batch_d.get("region_pseudobulk"),
                        region_mask=batch_d.get("region_mask"),
                        pseudobulk=batch_d.get("pseudobulk"),
                        ccc_edge_index=batch_d.get("ccc_edge_index"),
                        ccc_edge_type=batch_d.get("ccc_edge_type"),
                        ccc_edge_attr=batch_d.get("ccc_edge_attr"),
                        cell_data=batch_d.get("cell_data"),
                        cell_offsets=batch_d.get("cell_offsets"),
                        cell_type_mask=batch_d.get("cell_type_mask"),
                        pathology=batch_d.get("pathology"),
                        return_embeddings=True,
                    )
                    embeddings = enc_out["embeddings"]
                    if layer not in embeddings:
                        raise KeyError(
                            f"layer {layer!r} not present in embeddings dict; "
                            f"available: {list(embeddings.keys())}"
                        )
                    arr = embeddings[layer].detach().cpu().numpy()
                    fold_acts.append(arr.astype(np.float32))
                    fold_sids.extend(sids)
                    fold_is_val.extend(str(s) in val_subjects for s in sids)

        del model
        if torch_device.type == "cuda":
            torch.cuda.empty_cache()

        fold_acts_arr = np.concatenate(fold_acts, axis=0)
        fold_sids_arr = np.array(fold_sids, dtype=object)
        fold_idx_arr = np.full(len(fold_sids), int(fold), dtype=np.int64)
        fold_is_val_arr = np.array(fold_is_val, dtype=bool)

        per_fold_activations.append(fold_acts_arr)
        per_fold_sids.append(fold_sids_arr)
        per_fold_idx.append(fold_idx_arr)
        per_fold_is_val.append(fold_is_val_arr)

        # Persist per-fold .npz.
        fold_npz = output_dir / f"activations_{layer}_fold{fold}.npz"
        np.savez(
            fold_npz,
            activations=fold_acts_arr,
            subject_ids=fold_sids_arr,
            fold_indices=fold_idx_arr,
            is_val=fold_is_val_arr,
            **({"cell_types": cell_types_array} if cell_types_array is not None else {}),
            layer=np.array(layer, dtype=object),
        )
        logger.info(
            "fold %d: wrote %s (shape=%s)",
            fold, fold_npz.name, fold_acts_arr.shape,
        )

    activations = np.concatenate(per_fold_activations, axis=0)
    subject_ids = np.concatenate(per_fold_sids, axis=0)
    fold_indices = np.concatenate(per_fold_idx, axis=0)
    is_val = np.concatenate(per_fold_is_val, axis=0)

    git_commit = git_sha(PROJECT_ROOT)
    combined_npz = output_dir / f"activations_{layer}_all_folds.npz"
    np.savez(
        combined_npz,
        activations=activations,
        subject_ids=subject_ids,
        fold_indices=fold_indices,
        is_val=is_val,
        **({"cell_types": cell_types_array} if cell_types_array is not None else {}),
        layer=np.array(layer, dtype=object),
        git_commit=np.array(git_commit, dtype=object),
    )
    logger.info("wrote combined %s (shape=%s)", combined_npz.name, activations.shape)

    return ActivationBundle(
        activations=activations,
        subject_ids=subject_ids,
        fold_indices=fold_indices,
        is_val=is_val,
        cell_types=cell_types_array,
        layer=layer,
    )


def train_sae_topk(
    activations: np.ndarray,
    config: SAEConfig,
) -> SAEModel:
    """Train a TopK SAE (Gao et al. 2024, Orlov §3.1.1).

    Encoder: ``h = TopK(ReLU(W_enc @ (x - b_dec) + b_enc), k=config.k)``.
    Decoder: ``x_hat = W_dec @ h + b_dec``.
    Loss: ``||x - x_hat||² + config.aux_lambda * ||residual - x_aux_hat||²``,
    where ``x_aux_hat`` is reconstructed using the top ``config.aux_k``
    *dead* features (features with ``fraction_active < 1e-4`` over the last
    ``DEAD_WINDOW_FRAC * n_steps`` window). Per Orlov Table 2: "L_aux
    encourages feature utilization."

    If ``config.decoder_unit_norm``, project each ``W_dec[:, j]`` to unit
    L2 norm after every optimizer step (deviation from Orlov's literal
    equations; required for stable magnitude/direction decomposition).

    Parameters
    ----------
    activations
        ``[N_total, n]`` flattened activation matrix. For ``layer == "fused"``,
        flatten ``[N, 31, n]`` to ``[N*31, n]`` first.
    config
        ``SAEConfig`` with ``architecture == "topk"``.

    Returns
    -------
    SAEModel
        Fitted SAE.
    """
    if config.architecture != "topk":
        raise ValueError(
            f"train_sae_topk requires architecture='topk', got {config.architecture!r}"
        )
    return _train_sae_torch(activations, config)


def train_sae_batch_topk(
    activations: np.ndarray,
    config: SAEConfig,
) -> SAEModel:
    """Train a Batch-TopK SAE (Bussmann et al. 2024, Orlov §3.1.2).

    Encoder: across a batch ``X`` of ``n_batch`` samples, the per-batch budget
    is ``n_batch * config.k`` activations total; only the largest
    ``n_batch * config.k`` pre-activations across the entire batch are kept,
    zeros elsewhere. Per Orlov §3.1.2: "Batch-TopK SAEs modify the TopK
    operation to select the top n×K activations across an entire batch of n
    samples rather than independently per sample. This allows variable
    per-sample sparsity."

    At inference time (single sample), apply the threshold from the running
    average of the per-batch K-th-largest pre-activation during training
    (per Bussmann et al. 2024; stored in ``activation_stats["threshold"]``).

    Decoder, loss, and unit-norm constraint identical to ``train_sae_topk``.

    Parameters
    ----------
    activations
        ``[N_total, n]`` flattened activation matrix.
    config
        ``SAEConfig`` with ``architecture == "batch_topk"``.

    Returns
    -------
    SAEModel
        Fitted SAE.
    """
    if config.architecture != "batch_topk":
        raise ValueError(
            f"train_sae_batch_topk requires architecture='batch_topk', got "
            f"{config.architecture!r}"
        )
    return _train_sae_torch(activations, config)


def evaluate_reconstruction(
    sae: SAEModel,
    activations: np.ndarray,
    *,
    dead_fraction_threshold: float = DEAD_FRACTION_THRESHOLD,
) -> dict[str, float]:
    """Compute reconstruction quality and sparsity metrics on a batch.

    Per Orlov §4.1: SAEs on biological foundation models typically explain
    90-95 % of activation variance at moderate sparsity. We use the same FVE
    metric.

    Parameters
    ----------
    sae
        Fitted SAE model.
    activations
        ``[N_eval, n]`` activation matrix to reconstruct.
    dead_fraction_threshold
        Per-feature ``fraction_active`` strictly-below threshold for the
        ``"dead"`` flag. Defaults to ``DEAD_FRACTION_THRESHOLD`` (1e-4) and
        matches the threshold used by :func:`interpret_features` so the two
        functions report the same dead-feature counts on identical inputs.

    Returns
    -------
    dict with keys
        - ``"mse"``: mean squared error between ``x`` and ``x_hat``.
        - ``"fve"``: fraction-of-variance-explained, ``1 - mse / Var(x)``.
        - ``"l0_mean"``: mean number of active features per sample.
        - ``"l0_std"``: per-sample std of active count.
        - ``"dead_fraction"``: fraction of dictionary features whose
          fraction_active < ``dead_fraction_threshold`` (matches
          :func:`interpret_features`).
    """
    if activations.ndim != 2:
        raise ValueError(
            f"activations must be [N, n] 2D; got shape {activations.shape}"
        )
    h = _encode_numpy(sae, activations)
    x_hat = _decode_numpy(sae, h)
    return _reconstruction_metrics_from_codes(
        activations, h, x_hat, dead_fraction_threshold=dead_fraction_threshold,
    )


def _reconstruction_metrics_from_codes(
    activations: np.ndarray,
    h: np.ndarray,
    x_hat: np.ndarray,
    *,
    dead_fraction_threshold: float = DEAD_FRACTION_THRESHOLD,
) -> dict[str, float]:
    """F10 helper: compute the same scalar metrics as
    :func:`evaluate_reconstruction` from precomputed sparse codes ``h`` and
    reconstruction ``x_hat``.

    Use this from per-fold evaluators that already encoded the full union to
    avoid the redundant per-fold encode/decode passes. Numerically identical
    to :func:`evaluate_reconstruction` when given the same slice indexing.
    """
    mse = float(np.mean((activations - x_hat) ** 2))
    var = float(np.var(activations))
    fve = 1.0 - mse / var if var > 0 else float("nan")

    l0_per_sample = (h > 0).sum(axis=1)
    l0_mean = float(np.mean(l0_per_sample))
    l0_std = float(np.std(l0_per_sample))

    # Dead-feature definition: fraction_active < dead_fraction_threshold.
    # Matches :func:`interpret_features` so the two outputs are consistent.
    fraction_active = (h > 0).mean(axis=0)  # [m]
    dead_fraction = float((fraction_active < dead_fraction_threshold).mean())

    return {
        "mse": mse,
        "fve": fve,
        "l0_mean": l0_mean,
        "l0_std": l0_std,
        "dead_fraction": dead_fraction,
    }


def evaluate_reconstruction_with_cached_codes(
    sae: SAEModel,
    activations: np.ndarray,
    *,
    dead_fraction_threshold: float = DEAD_FRACTION_THRESHOLD,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Encode + decode once; return metrics and the cached ``h``, ``x_hat``.

    F10: drop-in replacement for :func:`evaluate_reconstruction` that
    additionally returns the sparse codes and the reconstruction so callers
    can slice them per-fold without re-encoding. Used by
    ``run_sae_train.py`` and ``run_sae_random_null.py`` to compute full +
    per-fold metrics with a single encode pass over the union.

    Numerically identical to calling :func:`evaluate_reconstruction` on
    ``activations``, plus the same per-fold slice yields bit-equivalent
    metrics to calling :func:`evaluate_reconstruction` on
    ``activations[mask]`` (verified at fp32 epsilon by the unit test
    ``test_evaluate_reconstruction_with_cached_codes_matches_per_fold``).
    """
    if activations.ndim != 2:
        raise ValueError(
            f"activations must be [N, n] 2D; got shape {activations.shape}"
        )
    h = _encode_numpy(sae, activations)
    x_hat = _decode_numpy(sae, h)
    metrics = _reconstruction_metrics_from_codes(
        activations, h, x_hat, dead_fraction_threshold=dead_fraction_threshold,
    )
    return metrics, h, x_hat


def reconstruction_metrics_from_slice(
    activations: np.ndarray,
    h: np.ndarray,
    x_hat: np.ndarray,
    mask: np.ndarray,
    *,
    dead_fraction_threshold: float = DEAD_FRACTION_THRESHOLD,
) -> dict[str, float]:
    """F10 helper: compute reconstruction metrics on a per-fold slice of
    cached union-level codes / reconstruction.

    ``activations``, ``h``, ``x_hat`` should be the union arrays from
    :func:`evaluate_reconstruction_with_cached_codes`. ``mask`` selects the
    rows belonging to one fold. Numerically identical to
    ``evaluate_reconstruction(sae, activations[mask])`` because the encoder
    is row-wise (so ``_encode_numpy(sae, activations[mask]) == h[mask]``)
    and the metrics are simple reductions over the slice.
    """
    return _reconstruction_metrics_from_codes(
        activations[mask], h[mask], x_hat[mask],
        dead_fraction_threshold=dead_fraction_threshold,
    )


def interpret_features(
    sae: SAEModel,
    bundle: ActivationBundle,
    metadata: dict[str, np.ndarray],
    *,
    top_k_subjects: int = 20,
) -> list[dict]:
    """Build a per-feature interpretability report (Orlov §3.3.1).

    For each feature ``j`` in ``[0, m)``:

    1. **Top-activating subjects.** Rank by ``h_j(x_i)``. Take the top
       ``top_k_subjects``. Mann-Whitney U on cognition / pathology of top vs
       bottom ``top_k_subjects`` as a crude monosemanticity proxy.

    2. **Decoder-direction CT decomposition** (only when ``bundle.layer == "fused"``).
       For each cell-type ``c``, project that CT's mean fused embedding
       ``μ_c`` onto ``W_dec[:, j]`` and report top-3 CTs by absolute squared
       projection.

    3. **Quality flags.** ``"dead"`` if ``fraction_active < 1e-4``,
       ``"ubiquitous"`` if ``> 0.5``, ``"interpretable_candidate"`` if it
       passes (a) Mann-Whitney p < 0.05, (b) one-CT-dominant for ``fused``,
       (c) fraction_active in [1e-4, 0.5].

    Parameters
    ----------
    sae
        Fitted SAE.
    bundle
        ``ActivationBundle`` used to compute the activations.
    metadata
        Dict with at least ``"cognition"``, ``"amyloid"``, ``"tau"``,
        ``"global_pathology"``, all ``[N_subjects]``. Indexed by
        ``bundle.subject_ids``.
    top_k_subjects
        How many subjects to use for the top/bottom comparison.

    Returns
    -------
    list of dict, one per feature, with keys
        ``feature_idx``, ``top_subjects`` (list of subject_id),
        ``top_cell_types`` (list of CT name + projection magnitude),
        ``mw_p_cognition``, ``mw_p_pathology``, ``fraction_active``,
        ``flags`` (set of ``{"dead", "ubiquitous", "interpretable_candidate"}``).
    """
    layer = bundle.layer
    activations = bundle.activations  # [N, n] or [N, 31, n]
    sids = np.asarray(bundle.subject_ids)

    # Build per-row aggregations:
    #   - for "attended": one row per subject -> directly use sids.
    #   - for "fused": rows are (subject, CT) pairs in C-major order [N*31, n].
    if layer == "attended":
        if activations.ndim != 2:
            raise ValueError(f"attended activations must be 2D; got {activations.shape}")
        flat = activations
        flat_sids = sids
        flat_ct_idx = None
    elif layer == "fused":
        if activations.ndim != 3:
            raise ValueError(f"fused activations must be 3D; got {activations.shape}")
        N, C, n = activations.shape
        flat = activations.reshape(N * C, n)
        flat_sids = np.repeat(sids, C)
        flat_ct_idx = np.tile(np.arange(C), N)
    else:
        raise ValueError(f"Unsupported layer: {layer!r}")

    # Encode all rows.
    h = _encode_numpy(sae, flat)  # [N_rows, m]
    m = h.shape[1]

    # Per-feature fraction_active.
    fraction_active = (h > 0).mean(axis=0)

    # Sort sids -> metadata index for fast lookup.
    cog = metadata.get("cognition")
    path = metadata.get("global_pathology", metadata.get("pathology"))
    sid_to_idx: dict[str, int] = {}
    sids_meta = metadata.get("subject_ids")
    if sids_meta is not None:
        sid_to_idx = {str(s): i for i, s in enumerate(sids_meta)}

    def _values_at(arr: np.ndarray | None, sids: np.ndarray) -> np.ndarray:
        """Look up `arr` values by subject_id; return NaN where missing."""
        if arr is None or sid_to_idx == {}:
            return np.full(len(sids), np.nan, dtype=np.float64)
        out = np.full(len(sids), np.nan, dtype=np.float64)
        for i, s in enumerate(sids):
            j = sid_to_idx.get(str(s))
            if j is not None and j < len(arr):
                v = arr[j]
                if v is not None and not (isinstance(v, float) and math.isnan(v)):
                    out[i] = float(v)
        return out

    cell_types = (
        list(map(str, bundle.cell_types)) if bundle.cell_types is not None else None
    )

    # For "fused" layer: precompute per-CT mean of the *raw activations*
    # used for decoder-direction projection.
    per_ct_means: np.ndarray | None = None
    if layer == "fused":
        # Mean across subjects within each CT -> [C, n].
        per_ct_means = activations.mean(axis=0)

    # Mann-Whitney is per-subject (collapsing CTs for "fused"). For a
    # given feature j, build a "subject score" by max-pool across CTs of
    # h_j (subjects with at least one strongly-activating CT rank high).
    # For "attended", the subject score is just h_j directly.
    if layer == "attended":
        h_subject = h  # [N, m]
        subject_sids = sids
    else:
        # h is [N*C, m]; reshape to [N, C, m] then max over C -> [N, m].
        h_3d = h.reshape(activations.shape[0], activations.shape[1], m)
        h_subject = h_3d.max(axis=1)
        subject_sids = sids

    cog_per_subject = _values_at(cog, subject_sids)
    path_per_subject = _values_at(path, subject_sids)

    # Guard against top/bottom overlap when 2 * top_k_subjects > N: fall back
    # to floor(N / 2) so the two groups are disjoint. Without this guard, the
    # Mann-Whitney U test compares overlapping samples and inflates type-1
    # error.
    n_subjects = int(h_subject.shape[0])
    top_k_eff = int(min(top_k_subjects, n_subjects // 2))
    if top_k_eff < top_k_subjects:
        logger.warning(
            "interpret_features: requested top_k_subjects=%d but n_subjects=%d "
            "(2k > N); reducing to top_k_eff=%d to keep top/bottom disjoint.",
            top_k_subjects, n_subjects, top_k_eff,
        )

    # F14: parallelise the per-feature MW + CT-decomposition loop with joblib
    # threading. The inner work is mostly numpy + scipy.stats.mannwhitneyu —
    # both release the GIL — so threads avoid the loky pickle of ``sae`` and
    # the activation arrays. Numerics are deterministic per-feature and do
    # not depend on iteration order, so the threaded results are bit-equal
    # to the serial path.
    from joblib import Parallel, delayed

    def _mw(values_top: np.ndarray, values_bot: np.ndarray) -> float:
        v_top = values_top[~np.isnan(values_top)]
        v_bot = values_bot[~np.isnan(values_bot)]
        if len(v_top) < 2 or len(v_bot) < 2:
            return float("nan")
        try:
            _stat, p = mannwhitneyu(v_top, v_bot, alternative="two-sided")
            return float(p)
        except ValueError:
            # mannwhitneyu raises if all values are identical (no rank var).
            return float("nan")

    def _per_feature_report(j: int) -> dict:
        scores = h_subject[:, j]
        order = np.argsort(-scores)
        top_idx = order[:top_k_eff]
        bottom_idx = (
            order[-top_k_eff:] if top_k_eff > 0
            else np.empty(0, dtype=order.dtype)
        )
        top_subjects = [str(s) for s in subject_sids[top_idx]]

        mw_p_cog = _mw(cog_per_subject[top_idx], cog_per_subject[bottom_idx])
        mw_p_path = _mw(path_per_subject[top_idx], path_per_subject[bottom_idx])

        top_cell_types: list[dict] = []
        ct_dominance = 0.0
        if layer == "fused" and per_ct_means is not None and cell_types is not None:
            decoder_col = sae.W_dec[:, j]
            proj = per_ct_means @ decoder_col
            sq = proj ** 2
            total = float(sq.sum())
            top3 = np.argsort(-sq)[:3]
            top_cell_types = [
                {
                    "cell_type": cell_types[c] if c < len(cell_types) else f"ct_{c}",
                    "projection": float(proj[c]),
                    "squared_projection": float(sq[c]),
                }
                for c in top3
            ]
            ct_dominance = float(sq[top3].sum() / total) if total > 0 else 0.0

        flags: set[str] = set()
        if fraction_active[j] < DEAD_FRACTION_THRESHOLD:
            flags.add("dead")
        if fraction_active[j] > 0.5:
            flags.add("ubiquitous")
        is_interpretable = (
            (not math.isnan(mw_p_cog) and mw_p_cog < 0.05)
            and (DEAD_FRACTION_THRESHOLD <= fraction_active[j] <= 0.5)
        )
        if layer == "fused":
            is_interpretable = is_interpretable and (ct_dominance > 0.7)
        if is_interpretable:
            flags.add("interpretable_candidate")

        return {
            "feature_idx": int(j),
            "top_subjects": top_subjects,
            "top_cell_types": top_cell_types,
            "mw_p_cognition": float(mw_p_cog) if not math.isnan(mw_p_cog) else None,
            "mw_p_pathology": float(mw_p_path) if not math.isnan(mw_p_path) else None,
            "fraction_active": float(fraction_active[j]),
            "ct_dominance": float(ct_dominance),
            "flags": flags,
        }

    # Heuristic: only spawn threads when m > 64 (the loop is fast enough that
    # thread setup overhead dominates for tiny dictionaries used in tests).
    if m > 64:
        reports = Parallel(n_jobs=8, prefer="threads")(
            delayed(_per_feature_report)(j) for j in range(m)
        )
    else:
        reports = [_per_feature_report(j) for j in range(m)]

    return list(reports)


def cross_seed_stability(
    sae_models: list[SAEModel],
    *,
    cosine_threshold: float = 0.7,
) -> dict[str, np.ndarray | float]:
    """Quantify SAE feature stability across random seeds (Paulo & Belrose 2025).

    Per Orlov §4.1: Paulo & Belrose found ~30 % of features shared at
    cosine-similarity ≥ 0.7 across SAE training runs differing only in
    random seed. We adopt their threshold.

    For S input SAE models trained with different seeds:

    1. For every pair (s, s'), compute the cosine-similarity matrix
       ``C_{ss'}[j, k] = cos(W_dec_s[:, j], W_dec_{s'}[:, k])``.
    2. For each feature in seed 0, find its match in every other seed via
       **bipartite (Hungarian) matching** that maximises total similarity
       over the entire feature dictionary (one-to-one). This is the
       Paulo & Belrose 2025 canonical procedure: the simpler
       argmax-per-feature can let many seed-0 features collapse onto the
       same seed-s' target and inflates ``stable_fraction``.
    3. Count features whose Hungarian-assigned pair cosine is
       ``>= cosine_threshold`` in every other seed — this is the
       canonical ``stable_fraction``.
    4. Also report the argmax-per-feature stable_fraction (informational
       only; not the canonical metric) so the difference between the two
       definitions is visible in the JSON output.
    5. Return raw cosine matrices (for plotting), the canonical Hungarian
       fraction, and the argmax fraction.

    Parameters
    ----------
    sae_models
        List of ``S >= 2`` ``SAEModel`` instances differing only by training seed.
    cosine_threshold
        Per Orlov / Paulo & Belrose: 0.7.

    Returns
    -------
    dict with keys
        ``"cosine_matrices"`` ``[S, S, m, m]``,
        ``"stable_fraction"`` scalar in ``[0, 1]`` — canonical Hungarian-aligned,
        ``"stable_fraction_argmax"`` scalar in ``[0, 1]`` — informational
            best-match-per-feature fraction (legacy definition),
        ``"per_feature_stability"`` ``[m]`` boolean (stable under Hungarian
            assignment across every other seed),
        ``"per_feature_stability_argmax"`` ``[m]`` boolean (legacy).
    """
    from scipy.optimize import linear_sum_assignment

    S = len(sae_models)
    if S < 2:
        raise ValueError(f"Need at least 2 SAE models; got {S}")
    m = sae_models[0].W_dec.shape[1]
    n = sae_models[0].W_dec.shape[0]
    for i, sae in enumerate(sae_models):
        if sae.W_dec.shape != (n, m):
            raise ValueError(
                f"sae_models[{i}].W_dec shape={sae.W_dec.shape} != "
                f"sae_models[0].W_dec shape={(n, m)}"
            )

    # F3: compute upper-triangular off-diagonal pairs only and fill the rest
    # via cheap transposes / self-pair shortcuts. The original cube allocates
    # ``S*S*m*m`` entries (≈150 MB at S=3, m=2048) and runs ``S*S = 9``
    # matmuls; the new path runs ``S*(S-1)/2 = 3`` upper-triangular matmuls
    # plus ``S`` self-pair matmuls, with bit-equivalent results for the
    # off-diagonal panels (``cm[sp, s] = (A_s.T @ A_{sp}).T`` exactly under
    # fp32 since transpose is a memory layout op).
    cosine_matrices = np.zeros((S, S, m, m), dtype=np.float32)
    A_unit: list[np.ndarray] = []
    for sae in sae_models:
        col_norms = np.linalg.norm(sae.W_dec, axis=0, keepdims=True) + 1e-12  # [1, m]
        A_unit.append(sae.W_dec / col_norms)

    # Self-pairs (s == sp): A_s.T @ A_s. Diagonal is exactly 1.0 by unit-norm.
    for s in range(S):
        cosine_matrices[s, s] = (A_unit[s].T @ A_unit[s]).astype(np.float32)

    # Upper-triangular off-diagonal pairs; mirror to the lower triangle via
    # transpose (cosine is bilinear: (A.T @ B).T == B.T @ A bit-exactly in
    # fp32 since both are the same product after transpose layout).
    for s in range(S):
        for sp in range(s + 1, S):
            block = (A_unit[s].T @ A_unit[sp]).astype(np.float32)  # [m, m]
            cosine_matrices[s, sp] = block
            cosine_matrices[sp, s] = block.T

    # ── Canonical: bipartite (Hungarian) matching, seed 0 → seed sp ──────
    # Hungarian solves min over -cosine, equivalent to max over cosine.
    # Returns row_ind (= np.arange(m), since we pass a square cost matrix
    # without permutation) and col_ind (the assignment).
    per_feature_stability = np.ones(m, dtype=bool)
    for sp in range(1, S):
        cos = cosine_matrices[0, sp]  # [m, m]
        row_ind, col_ind = linear_sum_assignment(-cos)
        # row_ind == arange(m); col_ind[j] is the seed-sp match for seed-0 j.
        assigned_cosine = cos[row_ind, col_ind]  # [m]
        per_feature_stability &= (assigned_cosine >= cosine_threshold)
    stable_fraction = float(per_feature_stability.mean())

    # ── Informational: argmax-per-feature (legacy) ──────────────────────
    # Reported alongside the canonical fraction so the difference between
    # the two definitions is visible in the JSON output. Argmax allows
    # multiple seed-0 features to collapse onto the same seed-sp target
    # (a known failure mode of the simpler procedure).
    per_feature_stability_argmax = np.ones(m, dtype=bool)
    for sp in range(1, S):
        best_per_feat = cosine_matrices[0, sp].max(axis=1)  # [m]
        per_feature_stability_argmax &= (best_per_feat >= cosine_threshold)
    stable_fraction_argmax = float(per_feature_stability_argmax.mean())

    return {
        "cosine_matrices": cosine_matrices,
        "stable_fraction": stable_fraction,
        "stable_fraction_argmax": stable_fraction_argmax,
        "per_feature_stability": per_feature_stability,
        "per_feature_stability_argmax": per_feature_stability_argmax,
    }
