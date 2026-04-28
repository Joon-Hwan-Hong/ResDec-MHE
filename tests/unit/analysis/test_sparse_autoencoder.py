"""Tests for src/analysis/sparse_autoencoder.py — SKELETON ONLY.

All tests are marked ``@pytest.mark.skip("Implementation deferred — design only")``
until the design at ``docs/plans/2026-04-28-sparse-autoencoder-design.md`` is
approved by the user. Once approved, remove the skip markers and implement
each test alongside the corresponding ``sparse_autoencoder.py`` function.

Test plan (per design §10):

* Construction tests for ``SAEConfig`` and ``SAEModel`` dataclasses.
* ``train_sae_topk`` reconstruction-quality regression on a synthetic
  superposition example with known sparse ground-truth dictionary.
* ``train_sae_batch_topk`` variable-per-sample sparsity behavior.
* ``evaluate_reconstruction`` returns FVE in ``[0, 1]`` and ``l0_mean``
  matches ``config.k`` for TopK at convergence.
* ``interpret_features`` flags dead/ubiquitous/interpretable correctly on
  a hand-built mini SAE.
* ``cross_seed_stability`` returns ``stable_fraction == 1.0`` when all
  input SAE models are identical.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.analysis.sparse_autoencoder import (
    ActivationBundle,
    SAEConfig,
    SAEModel,
    cross_seed_stability,
    evaluate_reconstruction,
    extract_activations,
    interpret_features,
    train_sae_batch_topk,
    train_sae_topk,
)


# ─────────────────────────────────────────────────────────────────────────────
# Dataclass construction (does NOT need implementation — passes immediately)
# These two tests are NOT skipped; they verify the public API surface.
# ─────────────────────────────────────────────────────────────────────────────


def test_sae_config_topk_minimal_construction():
    """Smoke test: SAEConfig with TopK fields constructs without error."""
    cfg = SAEConfig(architecture="topk", expansion=16, k=8)
    assert cfg.architecture == "topk"
    assert cfg.expansion == 16
    assert cfg.k == 8
    assert cfg.decoder_unit_norm is True


def test_sae_config_l1_construction():
    """Smoke test: SAEConfig with L1 fields constructs without error."""
    cfg = SAEConfig(architecture="l1", expansion=8, l1_lambda=1e-3)
    assert cfg.architecture == "l1"
    assert cfg.l1_lambda == pytest.approx(1e-3)


def test_sae_model_dataclass_holds_arrays():
    """Smoke test: SAEModel can hold the four weight arrays + config."""
    cfg = SAEConfig(architecture="topk", expansion=2, k=2, n_steps=10)
    n, m = 4, 8
    sae = SAEModel(
        W_enc=np.zeros((m, n)),
        b_enc=np.zeros(m),
        W_dec=np.zeros((n, m)),
        b_dec=np.zeros(n),
        config=cfg,
    )
    assert sae.W_enc.shape == (m, n)
    assert sae.W_dec.shape == (n, m)
    assert sae.config is cfg


def test_activation_bundle_dataclass():
    """Smoke test: ActivationBundle holds the expected fields."""
    bundle = ActivationBundle(
        activations=np.zeros((3, 64)),
        subject_ids=np.array([1, 2, 3]),
        fold_indices=np.array([0, 0, 0]),
        is_val=np.array([False, False, True]),
        cell_types=None,
        layer="attended",
    )
    assert bundle.activations.shape == (3, 64)
    assert bundle.layer == "attended"


# ─────────────────────────────────────────────────────────────────────────────
# Implementation-gated tests — all skipped until design is approved.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.skip(reason="Implementation deferred — design only at docs/plans/2026-04-28-sparse-autoencoder-design.md")
def test_extract_activations_attended_shape():
    """``extract_activations(layer='attended')`` returns ``[N=516, 64]``."""
    raise NotImplementedError


@pytest.mark.skip(reason="Implementation deferred")
def test_extract_activations_fused_shape():
    """``extract_activations(layer='fused')`` returns ``[N=516, 31, 64]``."""
    raise NotImplementedError


@pytest.mark.skip(reason="Implementation deferred")
def test_train_sae_topk_recovers_synthetic_dictionary():
    """On synthetic ``x = D @ z`` with sparse z and unit-norm D columns,
    TopK SAE should recover D up to permutation/sign."""
    raise NotImplementedError


@pytest.mark.skip(reason="Implementation deferred")
def test_train_sae_topk_l0_at_convergence_matches_k():
    """At convergence, mean L0 of TopK SAE equals ``config.k`` exactly."""
    raise NotImplementedError


@pytest.mark.skip(reason="Implementation deferred")
def test_train_sae_topk_decoder_columns_unit_norm():
    """When ``config.decoder_unit_norm=True``, every ``W_dec[:, j]`` has
    L2 norm 1.0 ± 1e-6 after training."""
    raise NotImplementedError


@pytest.mark.skip(reason="Implementation deferred")
def test_train_sae_batch_topk_variable_per_sample_sparsity():
    """Batch-TopK allows per-sample L0 to vary; some samples should have
    > k active and some < k active when sample variance is high."""
    raise NotImplementedError


@pytest.mark.skip(reason="Implementation deferred")
def test_evaluate_reconstruction_fve_bounds():
    """``evaluate_reconstruction`` returns FVE in ``[-inf, 1]``; for a
    converged SAE on its training data, FVE should exceed 0.85
    (Orlov §4.1 expects 0.90-0.95 in published biological SAEs)."""
    raise NotImplementedError


@pytest.mark.skip(reason="Implementation deferred")
def test_evaluate_reconstruction_dead_fraction_in_unit_interval():
    """``dead_fraction`` is in ``[0, 1]``."""
    raise NotImplementedError


@pytest.mark.skip(reason="Implementation deferred")
def test_interpret_features_flags_dead_features():
    """Features with zero activation across all subjects should receive
    the ``"dead"`` flag and not the ``"interpretable_candidate"`` flag."""
    raise NotImplementedError


@pytest.mark.skip(reason="Implementation deferred")
def test_interpret_features_top_subjects_count():
    """``len(report[j]['top_subjects']) <= top_k_subjects`` for every j."""
    raise NotImplementedError


@pytest.mark.skip(reason="Implementation deferred")
def test_interpret_features_fused_layer_returns_top_cell_types():
    """When the ``ActivationBundle`` has ``layer == 'fused'``, the
    per-feature report includes a non-empty ``top_cell_types`` list."""
    raise NotImplementedError


@pytest.mark.skip(reason="Implementation deferred")
def test_cross_seed_stability_identical_models_yields_full_stability():
    """Three identical ``SAEModel``s should yield ``stable_fraction == 1.0``
    and a cosine-matrix diagonal of all 1.0."""
    raise NotImplementedError


@pytest.mark.skip(reason="Implementation deferred")
def test_cross_seed_stability_random_models_yield_low_stability():
    """Three SAE models with i.i.d. Gaussian decoders should yield
    ``stable_fraction`` near 0 (random unit vectors in 64D rarely have
    cosine ≥ 0.7)."""
    raise NotImplementedError
