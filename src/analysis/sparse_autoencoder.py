"""Sparse autoencoder for ResDec-MHE encoder hidden states (DESIGN-ONLY skeleton).

This module is a SKELETON. All public functions raise ``NotImplementedError``.
Implementation is gated on the design doc at::

    docs/plans/2026-04-28-sparse-autoencoder-design.md

being approved by the user. Do NOT add behavior to these functions until
that approval is on record.

Reference: Orlov, A. V. et al. (2026). *What Do Biological Foundation Models
Compute? Sparse Autoencoders from Feature Recovery to Mechanistic
Interpretability.* bioRxiv 2026.03.04.709491v1.
PDF on disk at ``docs/2026.03.04.709491v1.full.pdf``.

Design summary (see plan for full detail and equations):

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
  forward, persisted as ``.npz`` per fold under ``outputs/redesign/sae/``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Project root resolved from this file's location.
# /host/.../refinement-two/src/analysis/sparse_autoencoder.py → parents[2] = repo root.
# (parents[0] = src/analysis, [1] = src, [2] = repo root.)
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


@dataclass
class SAEConfig:
    """Hyperparameter container for one SAE training run.

    Parameters
    ----------
    architecture
        ``"topk"`` (Gao et al. 2024 / Orlov §3.1.1) or ``"batch_topk"``
        (Bussmann et al. 2024 / Orlov §3.1.2). ``"l1"`` is allowed for
        comparison only (subject to shrinkage bias per Orlov §3.1.1).
    expansion
        Dictionary expansion factor m / n. Orlov §3.3.3: 8x for VT, 16-32x
        for small PLM (our scale).
    k
        For ``topk`` / ``batch_topk``: number of active features per sample
        (TopK) or per ``n``-sample batch averaged (Batch-TopK). Orlov Table 2.
    l1_lambda
        L1 sparsity coefficient. Used only when ``architecture == "l1"``.
    aux_lambda
        Auxiliary-K loss weight for dead-feature revival (Gao et al. 2024).
        Default ``1/32`` per Orlov ref [17].
    aux_k
        Number of dead features used in the auxiliary reconstruction loss.
    decoder_unit_norm
        If True, normalize each decoder column to unit L2 norm after every
        optimizer step. *Not stated in Orlov's literal equations* — adopted
        from Gao 2024 / Bussmann 2024 implementations. Marked as a deviation
        from the paper in the design doc; requires user approval.
    learning_rate
    batch_size
    n_steps
        Optimizer settings. Defaults are placeholders; finalize before
        training.
    seed
    """

    architecture: Literal["topk", "batch_topk", "l1"]
    expansion: int
    k: int | None = None
    l1_lambda: float | None = None
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
# Public API — SKELETON ONLY
# ─────────────────────────────────────────────────────────────────────────────


def extract_activations(
    checkpoint_paths: list[Path],
    layer: Literal["attended", "fused"],
    output_dir: Path,
    *,
    device: str = "cuda",
    batch_size: int = 32,
) -> ActivationBundle:
    """Forward N=516 subjects through each fold's canonical encoder and persist activations.

    For each checkpoint, build the canonical dataloader (PFC slice, 31 CTs,
    4785 genes), call the model with ``return_embeddings=True``, collect the
    requested ``embeddings[layer]`` tensor for every subject, and concatenate
    across folds into a single ``ActivationBundle``.

    Persists ``output_dir / f"activations_{layer}_fold{f}.npz"`` and a combined
    ``output_dir / f"activations_{layer}_all_folds.npz"`` for reproducibility.

    Parameters
    ----------
    checkpoint_paths
        List of 5 paths to ``best-*.ckpt`` files under
        ``outputs/redesign/p5_canonical_seed42/fold{0..4}/checkpoints/``.
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
        be larger than training (e.g., 64-128).

    Returns
    -------
    ActivationBundle
        With ``activations``, ``subject_ids``, ``fold_indices``, ``is_val``,
        ``cell_types`` (if ``layer == "fused"``), ``layer``.

    Notes
    -----
    Uses ``Predictor.from_checkpoint`` (see ``src/inference/predict.py``).
    Forward is wrapped in ``torch.no_grad()`` and ``model.eval()``.
    """
    raise NotImplementedError(
        "Skeleton — implementation gated on design doc approval at "
        "docs/plans/2026-04-28-sparse-autoencoder-design.md"
    )


def train_sae_topk(
    activations: np.ndarray,
    config: SAEConfig,
) -> SAEModel:
    """Train a TopK SAE (Gao et al. 2024, Orlov §3.1.1).

    Encoder: ``h = TopK(ReLU(W_enc @ x + b_enc), k=config.k)``.
    Decoder: ``x_hat = W_dec @ h + b_dec``.
    Loss: ``||x - x_hat||² + config.aux_lambda * ||x - x_aux_hat||²``,
    where ``x_aux_hat`` is reconstructed using the top ``config.aux_k``
    *dead* features (features with ``fraction_active < 1e-4`` over a 1e7-token
    window per Gao 2024). Per Orlov Table 2: "L_aux encourages feature
    utilization."

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
        Fitted SAE with ``W_enc``, ``b_enc``, ``W_dec``, ``b_dec`` and
        end-of-training ``activation_stats``.
    """
    raise NotImplementedError(
        "Skeleton — implementation gated on design doc approval"
    )


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
    average of batch-K-th value during training (per Bussmann et al. 2024).

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
    raise NotImplementedError(
        "Skeleton — implementation gated on design doc approval"
    )


def evaluate_reconstruction(
    sae: SAEModel,
    activations: np.ndarray,
) -> dict[str, float]:
    """Compute reconstruction quality and sparsity metrics on a held-out batch.

    Per Orlov §4.1: SAEs on biological foundation models typically explain
    90-95 % of activation variance at moderate sparsity. We use the same FVE
    metric.

    Parameters
    ----------
    sae
        Fitted SAE model.
    activations
        ``[N_eval, n]`` activation matrix to reconstruct.

    Returns
    -------
    dict with keys
        - ``"mse"``: mean squared error between ``x`` and ``x_hat``.
        - ``"fve"``: fraction-of-variance-explained, ``1 - mse / Var(x)``.
        - ``"l0_mean"``: mean number of active features per sample.
        - ``"l0_std"``: per-sample std of active count.
        - ``"dead_fraction"``: fraction of dictionary features that never
          activate on the eval set.
    """
    raise NotImplementedError(
        "Skeleton — implementation gated on design doc approval"
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

    1. **Top-activating subjects.** Rank by ``h_j(x_i)`` over all rows in
       ``bundle.activations``. Take the top ``top_k_subjects``. Compute their
       cognition-score and pathology distributions; report Mann-Whitney U vs
       the bottom ``top_k_subjects`` as a crude monosemanticity proxy.

    2. **Decoder-direction CT decomposition** (only when ``bundle.layer == "fused"``).
       For each cell-type ``c``, compute the mean activation
       ``mu_c = mean(activations[bundle.cell_types == c])``, project onto
       ``W_dec[:, j]``, and report top-3 CTs by absolute squared projection.

    3. **Quality flags.** Mark feature as "dead" if ``fraction_active < 1e-4``,
       "ubiquitous" if ``> 0.5``, "interpretable_candidate" if it passes
       (a) Mann-Whitney p < 0.05, (b) one-CT-dominant for ``fused``, (c)
       fraction_active in [1e-4, 0.5].

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
    raise NotImplementedError(
        "Skeleton — implementation gated on design doc approval"
    )


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
    2. For each feature in seed 0, find its best match in every other seed.
       Count features with all best-match cosines ``>= cosine_threshold`` —
       this is the "stable feature count" reported as a fraction of m.
    3. Return raw cosine matrices (for plotting) and the fraction.

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
        ``"stable_fraction"`` scalar in ``[0, 1]``,
        ``"per_feature_stability"`` ``[m]`` boolean (stable across all seed pairs).
    """
    raise NotImplementedError(
        "Skeleton — implementation gated on design doc approval"
    )
