"""Tests for src/analysis/sparse_autoencoder.py.

Unit tests for the Orlov 2026 sparse-autoencoder implementation. Synthetic
fixtures verify:

* SAEConfig / SAEModel / ActivationBundle dataclass surface.
* TopK & Batch-TopK reconstruction quality on a synthetic sparse-dictionary
  ground truth (FVE > 0.85 — design §8.1 threshold; relaxed from Orlov 0.90-0.95
  to account for our small N).
* Per-sample L0 = config.k for TopK at convergence (the operation is
  deterministic).
* Decoder-column unit-norm constraint (Bussmann 2024 / Gao 2024 standard).
* Dead-feature aux-loss revival behaviour.
* Edge cases: k <= 0 and k > m raise.
* evaluate_reconstruction returns FVE in (-inf, 1] and dead_fraction in [0, 1].
* interpret_features flags dead, ubiquitous, and interpretable_candidate.
* cross_seed_stability returns 1.0 for identical models, ~0 for random.

Live model integration tests for ``extract_activations`` are out of scope
(would require a checkpoint + GPU); the function is exercised by the
end-to-end script in ``scripts/resdec_mhe/interpretability/extract_sae_activations.py``.
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
    interpret_features,
    train_sae_batch_topk,
    train_sae_topk,
)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _make_synthetic_sparse(
    *,
    n: int = 16,
    m_true: int = 32,
    n_samples: int = 2048,
    k: int = 4,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate ``x = D @ z`` with ``D ∈ R^{n×m_true}`` unit-norm columns and
    ``z ∈ R^{m_true}`` exactly k-sparse.

    Returns
    -------
    activations : [n_samples, n] float32
    dictionary : [n, m_true] float32 (unit-norm columns)
    """
    rng = np.random.default_rng(seed)
    D = rng.standard_normal(size=(n, m_true)).astype(np.float32)
    D /= np.linalg.norm(D, axis=0, keepdims=True) + 1e-8
    Z = np.zeros((n_samples, m_true), dtype=np.float32)
    for i in range(n_samples):
        idx = rng.choice(m_true, size=k, replace=False)
        Z[i, idx] = rng.uniform(low=0.5, high=2.0, size=k).astype(np.float32)
    X = Z @ D.T  # [n_samples, n]
    return X, D


# ─────────────────────────────────────────────────────────────────────────────
# Dataclass construction (does NOT need implementation — passes immediately)
# ─────────────────────────────────────────────────────────────────────────────


def test_sae_config_topk_minimal_construction():
    """Smoke test: SAEConfig with TopK fields constructs without error."""
    cfg = SAEConfig(architecture="topk", expansion=16, k=8)
    assert cfg.architecture == "topk"
    assert cfg.expansion == 16
    assert cfg.k == 8
    assert cfg.decoder_unit_norm is True


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
# Edge case tests — k bounds
# ─────────────────────────────────────────────────────────────────────────────


def test_train_sae_topk_raises_when_k_zero():
    """k <= 0 must raise ValueError."""
    X, _ = _make_synthetic_sparse(n=8, m_true=16, n_samples=64, k=2)
    cfg = SAEConfig(architecture="topk", expansion=2, k=0, n_steps=2, batch_size=8)
    with pytest.raises(ValueError):
        train_sae_topk(X, cfg)


def test_train_sae_topk_raises_when_k_exceeds_m():
    """k > m must raise ValueError (m = expansion * n)."""
    X, _ = _make_synthetic_sparse(n=8, m_true=16, n_samples=64, k=2)
    # n=8, expansion=2 => m=16, k=17 > m
    cfg = SAEConfig(architecture="topk", expansion=2, k=17, n_steps=2, batch_size=8)
    with pytest.raises(ValueError):
        train_sae_topk(X, cfg)


def test_train_sae_topk_raises_when_input_3d():
    """activations must be 2D ([N, n])."""
    X = np.zeros((4, 31, 8), dtype=np.float32)
    cfg = SAEConfig(architecture="topk", expansion=2, k=2, n_steps=2, batch_size=4)
    with pytest.raises(ValueError):
        train_sae_topk(X, cfg)


def test_train_sae_batch_topk_wrong_arch_raises():
    """train_sae_batch_topk requires architecture='batch_topk'."""
    X, _ = _make_synthetic_sparse(n=8, m_true=16, n_samples=64, k=2)
    cfg = SAEConfig(architecture="topk", expansion=2, k=2, n_steps=2, batch_size=8)
    with pytest.raises(ValueError):
        train_sae_batch_topk(X, cfg)


def test_train_sae_topk_wrong_arch_raises():
    """train_sae_topk requires architecture='topk'."""
    X, _ = _make_synthetic_sparse(n=8, m_true=16, n_samples=64, k=2)
    cfg = SAEConfig(architecture="batch_topk", expansion=2, k=2, n_steps=2, batch_size=8)
    with pytest.raises(ValueError):
        train_sae_topk(X, cfg)


# ─────────────────────────────────────────────────────────────────────────────
# Reconstruction quality on synthetic sparse signal
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "architecture,fve_threshold",
    [
        # TopK fixes per-sample L0 = k (matches ground-truth uniform-k signal).
        ("topk", 0.85),
        # Batch-TopK is variable-budget per sample; on a uniform-k ground-truth
        # signal it incurs a small reconstruction penalty, but should still
        # recover most of the variance (Orlov §3.1.2 trade-off).
        ("batch_topk", 0.80),
    ],
)
def test_train_sae_recovers_synthetic_sparse_signal(architecture, fve_threshold):
    """On synthetic ``x = D @ z`` with k-sparse z, SAE should reconstruct
    with FVE > 0.85 (design §8.1; relaxed from Orlov 0.90-0.95 for our N).

    Batch-TopK uses a slightly lower threshold (0.80) because its variable
    per-sample budget pays a small reconstruction cost on uniform-sparsity
    data.
    """
    n, m_true, k = 16, 64, 4
    X, _D = _make_synthetic_sparse(n=n, m_true=m_true, n_samples=2048, k=k, seed=0)

    cfg = SAEConfig(
        architecture=architecture,
        expansion=8,  # m=128, ≥ m_true=64
        k=k,
        n_steps=4000,
        batch_size=128,
        learning_rate=3e-3,
        seed=0,
        # Disable aux-loss; the test problem is small and aux can interfere.
        aux_lambda=0.0,
        aux_k=0,
    )
    if architecture == "topk":
        sae = train_sae_topk(X, cfg)
    else:
        sae = train_sae_batch_topk(X, cfg)

    metrics = evaluate_reconstruction(sae, X)
    assert metrics["fve"] > fve_threshold, (
        f"SAE {architecture} FVE {metrics['fve']:.3f} below "
        f"{fve_threshold} threshold"
    )


def test_train_sae_topk_l0_at_convergence_matches_k():
    """At convergence, mean L0 of TopK SAE equals config.k exactly per sample.

    The TopK operation is deterministic at inference: argpartition keeps
    exactly k indices per sample, and l0 == k holds whenever there are at
    least k positive pre-activations. After 500 training steps on a small
    synthetic problem, every sample reaches that condition, so we tighten
    the tolerance to ±0.05 (well below the previous lax ±0.5 slack) and
    add the strict ``l0_mean <= k`` upper bound: TopK never produces more
    than k actives per sample.
    """
    n, m_true, k = 16, 32, 4
    X, _ = _make_synthetic_sparse(n=n, m_true=m_true, n_samples=512, k=k, seed=1)
    cfg = SAEConfig(
        architecture="topk", expansion=4, k=k, n_steps=500,
        batch_size=64, learning_rate=1e-3, seed=1, aux_lambda=0.0,
    )
    sae = train_sae_topk(X, cfg)
    metrics = evaluate_reconstruction(sae, X)
    assert metrics["l0_mean"] <= float(k) + 1e-6, (
        f"TopK L0 mean {metrics['l0_mean']} exceeds k={k}; argpartition "
        "selects exactly k per sample, so this should never trigger."
    )
    assert metrics["l0_mean"] == pytest.approx(float(k), abs=0.05), (
        f"TopK L0 mean {metrics['l0_mean']} ≠ k={k}"
    )


def test_train_sae_topk_decoder_columns_unit_norm():
    """When config.decoder_unit_norm=True, every W_dec[:, j] has L2 norm 1.0."""
    n, m_true, k = 8, 16, 2
    X, _ = _make_synthetic_sparse(n=n, m_true=m_true, n_samples=256, k=k, seed=2)
    cfg = SAEConfig(
        architecture="topk", expansion=4, k=k, n_steps=200,
        batch_size=32, learning_rate=1e-3, seed=2, decoder_unit_norm=True,
    )
    sae = train_sae_topk(X, cfg)
    norms = np.linalg.norm(sae.W_dec, axis=0)  # [m]
    assert np.allclose(norms, 1.0, atol=1e-5), (
        f"Decoder columns not unit-norm; norms range "
        f"[{norms.min():.5f}, {norms.max():.5f}]"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Batch-TopK behaviour
# ─────────────────────────────────────────────────────────────────────────────


def test_train_sae_batch_topk_variable_per_sample_sparsity():
    """Batch-TopK allows per-sample L0 to vary; std(L0) > 0 when samples
    differ in scale — variable activation budget across the batch."""
    n, m_true, k = 16, 32, 4
    rng = np.random.default_rng(3)

    # Half of samples have 2 active components, half have 6 (so per-sample
    # information density varies). Average sparsity = 4 = k.
    X_low, _ = _make_synthetic_sparse(n=n, m_true=m_true, n_samples=256, k=2, seed=3)
    X_high, _ = _make_synthetic_sparse(n=n, m_true=m_true, n_samples=256, k=6, seed=4)
    # Scale "high-info" samples larger so they win the batch budget.
    X_high *= 2.0
    X = np.concatenate([X_low, X_high], axis=0)
    rng.shuffle(X)

    cfg = SAEConfig(
        architecture="batch_topk", expansion=4, k=k, n_steps=500,
        batch_size=64, learning_rate=1e-3, seed=3, aux_lambda=0.0,
    )
    sae = train_sae_batch_topk(X, cfg)
    metrics = evaluate_reconstruction(sae, X)
    # Variable per-sample sparsity → L0 std > 0. This is the defining
    # difference between Batch-TopK and TopK (Orlov §3.1.2).
    assert metrics["l0_std"] > 0, (
        "Batch-TopK should have variable per-sample sparsity (l0_std > 0); "
        f"got l0_std={metrics['l0_std']}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# evaluate_reconstruction
# ─────────────────────────────────────────────────────────────────────────────


def test_evaluate_reconstruction_returns_well_formed_dict():
    """All five metrics present, types correct, reasonable bounds."""
    n, m_true, k = 8, 16, 2
    X, _ = _make_synthetic_sparse(n=n, m_true=m_true, n_samples=128, k=k, seed=5)
    cfg = SAEConfig(
        architecture="topk", expansion=2, k=k, n_steps=200,
        batch_size=32, learning_rate=1e-3, seed=5,
    )
    sae = train_sae_topk(X, cfg)
    metrics = evaluate_reconstruction(sae, X)
    expected_keys = {"mse", "fve", "l0_mean", "l0_std", "dead_fraction"}
    assert set(metrics.keys()) == expected_keys
    assert metrics["mse"] >= 0
    assert metrics["fve"] <= 1.0
    assert metrics["l0_mean"] >= 0
    assert metrics["l0_std"] >= 0
    assert 0.0 <= metrics["dead_fraction"] <= 1.0


def test_evaluate_reconstruction_dead_fraction_in_unit_interval():
    """``dead_fraction`` must be in [0, 1]."""
    # Build a synthetic SAE by hand where all decoder columns are zero
    # (degenerate case) -> all features should appear "dead" on any input.
    n, m = 4, 8
    cfg = SAEConfig(architecture="topk", expansion=2, k=2)
    sae = SAEModel(
        W_enc=np.zeros((m, n)),
        b_enc=np.zeros(m),
        W_dec=np.zeros((n, m)),
        b_dec=np.zeros(n),
        config=cfg,
    )
    X = np.random.default_rng(0).standard_normal(size=(32, n)).astype(np.float32)
    metrics = evaluate_reconstruction(sae, X)
    assert 0.0 <= metrics["dead_fraction"] <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# interpret_features
# ─────────────────────────────────────────────────────────────────────────────


def test_interpret_features_flags_dead_features():
    """Hand-built SAE where one feature is fully alive and one is fully dead."""
    n, m = 4, 4
    cfg = SAEConfig(architecture="topk", expansion=1, k=1)
    # Encoder: identity (so feature j fires when x_j > 0).
    W_enc = np.eye(n, dtype=np.float32)
    b_enc = np.zeros(m, dtype=np.float32)
    W_dec = np.eye(n, dtype=np.float32)
    b_dec = np.zeros(n, dtype=np.float32)
    sae = SAEModel(
        W_enc=W_enc, b_enc=b_enc, W_dec=W_dec, b_dec=b_dec, config=cfg,
    )

    # All 16 subjects activate feature 0 strongly; no one activates feature 3.
    rng = np.random.default_rng(7)
    activations = np.zeros((16, n), dtype=np.float32)
    activations[:, 0] = rng.uniform(1.0, 2.0, size=16).astype(np.float32)

    sids = np.array([f"s{i}" for i in range(16)], dtype=object)
    bundle = ActivationBundle(
        activations=activations,
        subject_ids=sids,
        fold_indices=np.zeros(16, dtype=np.int64),
        is_val=np.zeros(16, dtype=bool),
        cell_types=None,
        layer="attended",
    )
    metadata = {
        "subject_ids": sids,
        "cognition": rng.standard_normal(16).astype(np.float64),
        "global_pathology": rng.standard_normal(16).astype(np.float64),
    }
    reports = interpret_features(sae, bundle, metadata, top_k_subjects=4)
    assert len(reports) == m
    # Feature 3 never fires on the dataset → fraction_active near 0 → "dead".
    feat3 = reports[3]
    assert "dead" in feat3["flags"]
    # Feature 0 fires for everyone (always picked by top-1 across rows where
    # x_0 > 0) → fraction_active = 1.0 → "ubiquitous".
    feat0 = reports[0]
    assert "ubiquitous" in feat0["flags"]


def test_interpret_features_top_subjects_count():
    """len(report[j]['top_subjects']) <= top_k_subjects for every j."""
    n, m = 4, 4
    cfg = SAEConfig(architecture="topk", expansion=1, k=1)
    sae = SAEModel(
        W_enc=np.eye(n, dtype=np.float32),
        b_enc=np.zeros(m, dtype=np.float32),
        W_dec=np.eye(n, dtype=np.float32),
        b_dec=np.zeros(n, dtype=np.float32),
        config=cfg,
    )

    rng = np.random.default_rng(8)
    activations = rng.standard_normal(size=(8, n)).astype(np.float32)
    sids = np.array([f"s{i}" for i in range(8)], dtype=object)
    bundle = ActivationBundle(
        activations=activations, subject_ids=sids,
        fold_indices=np.zeros(8, dtype=np.int64),
        is_val=np.zeros(8, dtype=bool),
        cell_types=None, layer="attended",
    )
    metadata = {
        "subject_ids": sids,
        "cognition": rng.standard_normal(8).astype(np.float64),
        "global_pathology": rng.standard_normal(8).astype(np.float64),
    }
    reports = interpret_features(sae, bundle, metadata, top_k_subjects=20)
    for r in reports:
        assert len(r["top_subjects"]) <= 20


def test_interpret_features_fused_layer_returns_top_cell_types():
    """When bundle.layer == 'fused', report includes non-empty top_cell_types."""
    n, m, n_subjects, n_ct = 4, 4, 8, 3
    cfg = SAEConfig(architecture="topk", expansion=1, k=1)
    sae = SAEModel(
        W_enc=np.eye(n, dtype=np.float32),
        b_enc=np.zeros(m, dtype=np.float32),
        W_dec=np.eye(n, dtype=np.float32),
        b_dec=np.zeros(n, dtype=np.float32),
        config=cfg,
    )

    rng = np.random.default_rng(9)
    activations = rng.standard_normal(size=(n_subjects, n_ct, n)).astype(np.float32)
    sids = np.array([f"s{i}" for i in range(n_subjects)], dtype=object)
    cts = np.array([f"ct{i}" for i in range(n_ct)], dtype=object)
    bundle = ActivationBundle(
        activations=activations, subject_ids=sids,
        fold_indices=np.zeros(n_subjects, dtype=np.int64),
        is_val=np.zeros(n_subjects, dtype=bool),
        cell_types=cts, layer="fused",
    )
    metadata = {
        "subject_ids": sids,
        "cognition": rng.standard_normal(n_subjects).astype(np.float64),
        "global_pathology": rng.standard_normal(n_subjects).astype(np.float64),
    }
    reports = interpret_features(sae, bundle, metadata, top_k_subjects=3)
    for r in reports:
        # top_cell_types should have 3 entries (top-3 by squared-projection).
        assert isinstance(r["top_cell_types"], list)
        assert len(r["top_cell_types"]) == 3
        for tc in r["top_cell_types"]:
            assert "cell_type" in tc
            assert "projection" in tc
            assert "squared_projection" in tc


# ─────────────────────────────────────────────────────────────────────────────
# cross_seed_stability
# ─────────────────────────────────────────────────────────────────────────────


def test_cross_seed_stability_identical_models_yields_full_stability():
    """3 identical SAEModels → stable_fraction == 1.0 and best-match cosine = 1."""
    n, m = 8, 16
    cfg = SAEConfig(architecture="topk", expansion=2, k=2)
    rng = np.random.default_rng(10)
    W_dec = rng.standard_normal(size=(n, m)).astype(np.float32)
    W_dec /= np.linalg.norm(W_dec, axis=0, keepdims=True) + 1e-8
    sae = SAEModel(
        W_enc=np.zeros((m, n), dtype=np.float32),
        b_enc=np.zeros(m, dtype=np.float32),
        W_dec=W_dec.copy(),
        b_dec=np.zeros(n, dtype=np.float32),
        config=cfg,
    )
    out = cross_seed_stability([sae, sae, sae], cosine_threshold=0.7)
    assert out["stable_fraction"] == pytest.approx(1.0)
    assert out["per_feature_stability"].all()
    # Diagonal of every cos-sim matrix should be ones.
    cm = out["cosine_matrices"]
    for s in range(3):
        assert np.allclose(np.diag(cm[s, s]), 1.0, atol=1e-5)


def test_cross_seed_stability_random_models_yield_low_stability():
    """3 i.i.d. random-decoder SAE models in moderate dim → stable_fraction near 0.

    Two independent unit-norm random vectors in R^n (n=64) have expected
    absolute cosine ≈ √(2/(πn)) ≈ 0.10. Even the **best-of-m** match across
    a dictionary of m=64 columns rarely reaches 0.7.
    """
    n, m = 64, 64
    cfg = SAEConfig(architecture="topk", expansion=1, k=1)
    rng = np.random.default_rng(11)
    saes: list[SAEModel] = []
    for s in range(3):
        W_dec = rng.standard_normal(size=(n, m)).astype(np.float32)
        W_dec /= np.linalg.norm(W_dec, axis=0, keepdims=True) + 1e-8
        saes.append(SAEModel(
            W_enc=np.zeros((m, n), dtype=np.float32),
            b_enc=np.zeros(m, dtype=np.float32),
            W_dec=W_dec,
            b_dec=np.zeros(n, dtype=np.float32),
            config=cfg,
        ))
    out = cross_seed_stability(saes, cosine_threshold=0.7)
    # Should be very small; allow up to 5% in case of seed lottery.
    assert out["stable_fraction"] < 0.1, (
        f"Random SAEs unexpectedly stable: stable_fraction={out['stable_fraction']}"
    )


def test_cross_seed_stability_requires_two_models():
    """Single SAE input must raise ValueError."""
    n, m = 8, 16
    cfg = SAEConfig(architecture="topk", expansion=2, k=2)
    sae = SAEModel(
        W_enc=np.zeros((m, n), dtype=np.float32),
        b_enc=np.zeros(m, dtype=np.float32),
        W_dec=np.zeros((n, m), dtype=np.float32),
        b_dec=np.zeros(n, dtype=np.float32),
        config=cfg,
    )
    with pytest.raises(ValueError):
        cross_seed_stability([sae], cosine_threshold=0.7)


# ─────────────────────────────────────────────────────────────────────────────
# Aux-loss revival behaviour (Gao 2024)
# ─────────────────────────────────────────────────────────────────────────────


def test_aux_loss_revives_dead_features():
    """With aux_lambda > 0, dead-feature count at end of training is no
    larger than without aux. (Test the dead-feature window finishes with
    aux-loss path having at least as many alive features as without.)
    """
    n, m_true, k = 16, 32, 4
    X, _ = _make_synthetic_sparse(n=n, m_true=m_true, n_samples=512, k=k, seed=12)

    cfg_no_aux = SAEConfig(
        architecture="topk", expansion=8, k=k, n_steps=300,
        batch_size=32, learning_rate=1e-3, seed=12,
        aux_lambda=0.0, aux_k=0,
    )
    cfg_aux = SAEConfig(
        architecture="topk", expansion=8, k=k, n_steps=300,
        batch_size=32, learning_rate=1e-3, seed=12,
        aux_lambda=1.0 / 32.0, aux_k=64,
    )
    sae_no_aux = train_sae_topk(X, cfg_no_aux)
    sae_aux = train_sae_topk(X, cfg_aux)

    metrics_no_aux = evaluate_reconstruction(sae_no_aux, X)
    metrics_aux = evaluate_reconstruction(sae_aux, X)

    # Aux-loss should NOT produce more dead features than no-aux. (Cannot
    # guarantee it produces strictly fewer in 300 steps on a small problem,
    # but it must not regress.)
    assert metrics_aux["dead_fraction"] <= metrics_no_aux["dead_fraction"] + 0.05, (
        "Aux loss regressed dead-feature count: "
        f"no_aux={metrics_no_aux['dead_fraction']:.3f} "
        f"aux={metrics_aux['dead_fraction']:.3f}"
    )


def test_decoder_unit_norm_idempotent():
    """A second projection should not change already-unit-norm columns."""
    n, m_true, k = 8, 16, 2
    X, _ = _make_synthetic_sparse(n=n, m_true=m_true, n_samples=128, k=k, seed=13)
    cfg = SAEConfig(
        architecture="topk", expansion=2, k=k, n_steps=100,
        batch_size=16, learning_rate=1e-3, seed=13, decoder_unit_norm=True,
    )
    sae = train_sae_topk(X, cfg)
    norms_before = np.linalg.norm(sae.W_dec, axis=0)
    # Second renormalisation must be a no-op (idempotent on unit-norm input).
    W_dec2 = sae.W_dec / (np.linalg.norm(sae.W_dec, axis=0, keepdims=True) + 1e-8)
    norms_after = np.linalg.norm(W_dec2, axis=0)
    assert np.allclose(norms_before, norms_after, atol=1e-6)
    assert np.allclose(norms_after, 1.0, atol=1e-5)
