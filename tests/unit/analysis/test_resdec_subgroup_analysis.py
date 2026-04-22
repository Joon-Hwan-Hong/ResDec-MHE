"""Unit tests for :mod:`src.analysis.resdec_subgroup_analysis`.

Contract (from docs/plans/2026-04-22-resdec-h3-phase5-finish.md Task C.2):

    stratified_metrics(y_true, y_pred, subgroup_masks, *, n_bootstrap, seed)

returns a flat dict keyed by ``group_name`` (e.g. ``"APOE_e4_0"``). Each value
is a dict with:

    - n: int
    - r2, rmse, pearson_r, spearman_rho: point estimates (float)
    - r2_ci, rmse_ci, pearson_r_ci, spearman_rho_ci: (lower, upper) 95% CIs
    - n_valid_bootstraps: int

Bootstrap uses the percentile method (2.5% / 97.5%) with
``np.random.default_rng(seed)``.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.analysis.resdec_subgroup_analysis import stratified_metrics


def test_trivial_single_subgroup_matches_overall():
    """With a single 'all-subjects' mask, subgroup R² equals overall R²."""
    rng = np.random.default_rng(0)
    n = 200
    y = rng.standard_normal(n)
    y_pred = y + rng.standard_normal(n) * 0.5
    all_mask = np.ones(n, dtype=bool)
    out = stratified_metrics(y, y_pred, {"all": all_mask}, n_bootstrap=100)
    from sklearn.metrics import r2_score
    assert out["all"]["r2"] == pytest.approx(r2_score(y, y_pred))
    assert out["all"]["n"] == n


def test_two_subgroup_case_returns_different_r2():
    """Two subgroups with deliberately different noise levels produce different R²."""
    rng = np.random.default_rng(42)
    n_per = 100
    y = rng.standard_normal(2 * n_per)
    # Group A: low noise (high R²). Group B: high noise (low R²).
    y_pred = np.empty_like(y)
    y_pred[:n_per] = y[:n_per] + rng.standard_normal(n_per) * 0.1
    y_pred[n_per:] = y[n_per:] + rng.standard_normal(n_per) * 2.0
    mask_a = np.concatenate([np.ones(n_per, dtype=bool), np.zeros(n_per, dtype=bool)])
    mask_b = ~mask_a
    out = stratified_metrics(y, y_pred, {"A": mask_a, "B": mask_b}, n_bootstrap=100)
    assert out["A"]["r2"] > out["B"]["r2"]  # low noise beats high noise
    assert out["A"]["n"] == n_per
    assert out["B"]["n"] == n_per


def test_bootstrap_ci_wider_for_smaller_n():
    """CI width should increase as subgroup n decreases."""
    rng = np.random.default_rng(7)
    y = rng.standard_normal(500)
    y_pred = y + rng.standard_normal(500) * 0.5
    large = np.ones(500, dtype=bool); large[-100:] = False  # n=400
    small = np.zeros(500, dtype=bool); small[:50] = True   # n=50
    out = stratified_metrics(y, y_pred, {"large": large, "small": small}, n_bootstrap=500)
    ci_large = out["large"]["r2_ci"][1] - out["large"]["r2_ci"][0]
    ci_small = out["small"]["r2_ci"][1] - out["small"]["r2_ci"][0]
    assert ci_small > ci_large


def test_tiny_subgroup_returns_nan():
    """Subgroup with n<3 gives NaN metrics / NaN CIs."""
    y = np.array([1.0, 2.0])
    y_pred = np.array([1.1, 1.9])
    mask = np.array([True, True])
    out = stratified_metrics(y, y_pred, {"tiny": mask}, n_bootstrap=50)
    assert out["tiny"]["n"] == 2
    # n<3 → metrics undefined by spec
    assert np.isnan(out["tiny"]["r2"])
    assert np.isnan(out["tiny"]["rmse"])
    assert np.isnan(out["tiny"]["pearson_r"])
    assert np.isnan(out["tiny"]["spearman_rho"])
    assert "n_valid_bootstraps" in out["tiny"]


def test_empty_subgroup_returns_nan_and_warns():
    """Subgroup with n=0 gives NaN metrics, n_valid_bootstraps=0, AND emits UserWarning."""
    y = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    mask = np.array([False, False, False])
    with pytest.warns(UserWarning, match="empty"):
        out = stratified_metrics(y, y_pred, {"empty": mask}, n_bootstrap=50)
    assert out["empty"]["n"] == 0
    assert np.isnan(out["empty"]["r2"])
    assert out["empty"]["n_valid_bootstraps"] == 0


def test_percentile_ci_not_normal_approx():
    """CI must be percentile method (q2.5, q97.5), NOT mean ± 1.96*std.

    On a skewed target distribution, the bootstrap resample distribution of
    R² is asymmetric. The percentile method preserves this asymmetry (the
    left and right arms of the CI differ). A normal-approximation CI would
    force symmetry (lo = r2 - k*std, hi = r2 + k*std), which would be wrong
    for skewed distributions.
    """
    rng = np.random.default_rng(0)
    n = 500
    y = rng.exponential(scale=2.0, size=n)
    y_pred = y + rng.standard_normal(n) * 0.3
    mask = np.ones(n, dtype=bool)
    out = stratified_metrics(y, y_pred, {"skewed": mask}, n_bootstrap=2000, seed=0)
    r2 = out["skewed"]["r2"]
    lo, hi = out["skewed"]["r2_ci"]
    assert r2 > 0.8
    left_arm = r2 - lo
    right_arm = hi - r2
    # Skewed bootstrap distribution → asymmetric arms. Normal-approx would force
    # symmetry exactly; percentile preserves the empirical asymmetry.
    assert abs(left_arm - right_arm) > 1e-4


def test_orchestration_uses_public_names():
    """Verify subgroup_r2.py imports public (non-underscore) helpers.

    If the quantile helpers are ever re-privatised (renamed back to
    ``_age_quartile_labels`` or similar), the module-load of
    ``scripts.redesign.interpretability.subgroup_r2`` would fail on the
    import statement and this test would error out rather than pass.
    """
    import scripts.redesign.interpretability.subgroup_r2 as mod
    assert hasattr(mod, "apoe_e4_count_label")
    # M1 consolidated the quartile helpers into the shared `quantile_labels`;
    # either name is acceptable for rename-safety.
    assert hasattr(mod, "age_quartile_labels") or hasattr(mod, "quantile_labels")


def test_reproducibility_across_seeds():
    """Same seed produces the same CIs; different seeds produce different CIs."""
    rng = np.random.default_rng(123)
    n = 300
    y = rng.standard_normal(n)
    y_pred = y + rng.standard_normal(n) * 0.3
    mask = np.ones(n, dtype=bool)

    out_a = stratified_metrics(y, y_pred, {"g": mask}, n_bootstrap=200, seed=42)
    out_b = stratified_metrics(y, y_pred, {"g": mask}, n_bootstrap=200, seed=42)
    out_c = stratified_metrics(y, y_pred, {"g": mask}, n_bootstrap=200, seed=99)
    assert out_a["g"]["r2_ci"] == out_b["g"]["r2_ci"]
    assert out_a["g"]["r2_ci"] != out_c["g"]["r2_ci"]
