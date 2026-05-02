"""Unit tests for :mod:`src.analysis.resdec_statistical_rigor`.

Contract:

1. ``paired_wilcoxon(fold_r2s_ours, fold_r2s_baseline, alternative="greater")``
   wraps :func:`scipy.stats.wilcoxon`. Identical arrays → all differences
   zero → return ``p_value=1.0`` (scipy raises ``ValueError`` in that case,
   which the wrapper must catch).
2. ``bootstrap_r2_ci(y_true, y_pred, n_boot=1000, conf=0.95, seed=42)``
   resamples with replacement ``n_boot`` times via ``rng.integers`` and
   reports percentile CI. Larger N → tighter CI.
3. ``calibration_coverage(y_true, y_pred, sigma, nominal=[...])`` returns
   the empirical coverage at each nominal level using the Gaussian
   z-score threshold ``|y_true - y_pred| <= z * sigma``. Well-calibrated
   Gaussian residuals should hit nominal coverage; too-small sigma →
   under-coverage (overconfident).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.analysis.resdec_statistical_rigor import (
    bootstrap_r2_ci,
    calibration_coverage,
    paired_wilcoxon,
)

def test_paired_wilcoxon_identical_arrays():
    """Identical arrays → all differences zero → p=1.0 (no evidence of improvement)."""
    r2 = np.array([0.4, 0.5, 0.3, 0.6, 0.45])
    out = paired_wilcoxon(r2, r2, alternative="greater")
    assert out["p_value"] == 1.0
    assert out["n_folds"] == 5
    assert out["median_diff"] == 0.0

def test_paired_wilcoxon_clear_improvement():
    """Ours uniformly > baseline → small p-value for 'greater'."""
    baseline = np.array([0.2, 0.3, 0.25, 0.35, 0.28])
    ours = baseline + 0.1  # uniform +0.1 improvement
    out = paired_wilcoxon(ours, baseline, alternative="greater")
    assert out["p_value"] < 0.1  # n=5 Wilcoxon limited power but detectable
    assert out["median_diff"] > 0

def test_bootstrap_r2_ci_contains_truth():
    """Bootstrap CI on synthetic well-behaved data should contain the sample R²."""
    rng = np.random.default_rng(0)
    n = 500
    y = rng.standard_normal(n)
    y_pred = y + rng.standard_normal(n) * 0.5
    out = bootstrap_r2_ci(y, y_pred, n_boot=1000, seed=0)
    # Sample R² is a point estimate; CI should contain it
    from sklearn.metrics import r2_score
    r2_point = r2_score(y, y_pred)
    assert out["ci_lower"] < r2_point < out["ci_upper"]
    assert out["n_boot"] == 1000
    assert out["conf"] == 0.95

def test_bootstrap_r2_ci_tighter_for_larger_n():
    """More subjects → tighter CI."""
    rng = np.random.default_rng(42)
    y_small = rng.standard_normal(100)
    y_pred_small = y_small + rng.standard_normal(100) * 0.5
    y_large = rng.standard_normal(2000)
    y_pred_large = y_large + rng.standard_normal(2000) * 0.5
    out_small = bootstrap_r2_ci(y_small, y_pred_small, n_boot=500, seed=0)
    out_large = bootstrap_r2_ci(y_large, y_pred_large, n_boot=500, seed=0)
    width_small = out_small["ci_upper"] - out_small["ci_lower"]
    width_large = out_large["ci_upper"] - out_large["ci_lower"]
    assert width_small > width_large

def test_calibration_coverage_well_calibrated():
    """Gaussian residuals with true sigma → coverage ≈ nominal."""
    rng = np.random.default_rng(0)
    n = 5000  # large n for tight empirical coverage
    sigma_true = 0.5
    y_true = rng.standard_normal(n)
    y_pred = y_true + rng.standard_normal(n) * sigma_true
    sigma_reported = np.full(n, sigma_true)
    out = calibration_coverage(y_true, y_pred, sigma_reported, nominal=[0.5, 0.68, 0.95])
    # At large n, empirical coverage should be within ±0.03 of nominal
    assert abs(out["coverage_at_0.5"] - 0.5) < 0.03
    assert abs(out["coverage_at_0.68"] - 0.68) < 0.03
    assert abs(out["coverage_at_0.95"] - 0.95) < 0.02

def test_calibration_coverage_overconfident_sigma():
    """Reported sigma too small → empirical coverage < nominal (overconfident)."""
    rng = np.random.default_rng(1)
    n = 5000
    sigma_true = 0.5
    y_true = rng.standard_normal(n)
    y_pred = y_true + rng.standard_normal(n) * sigma_true
    sigma_reported = np.full(n, sigma_true * 0.5)  # half the true noise
    out = calibration_coverage(y_true, y_pred, sigma_reported, nominal=[0.95])
    assert out["coverage_at_0.95"] < 0.95 - 0.05  # meaningfully under-covered

# ---------------------------------------------------------------------------
# I-1: fail loud on non-finite inputs to paired_wilcoxon.
# ---------------------------------------------------------------------------

def test_paired_wilcoxon_raises_on_nan_input():
    ours = np.array([0.4, 0.5, np.nan, 0.6, 0.45])
    baseline = np.array([0.3, 0.4, 0.3, 0.5, 0.35])
    with pytest.raises(ValueError, match="Non-finite"):
        paired_wilcoxon(ours, baseline, alternative="greater")

def test_paired_wilcoxon_raises_on_inf_input():
    ours = np.array([0.4, 0.5, 0.3, 0.6, 0.45])
    baseline = np.array([0.3, 0.4, np.inf, 0.5, 0.35])
    with pytest.raises(ValueError, match="Non-finite"):
        paired_wilcoxon(ours, baseline, alternative="greater")

# ---------------------------------------------------------------------------
# M-12: vectorized bootstrap must match the old Python-loop version.
# ---------------------------------------------------------------------------

def test_bootstrap_r2_ci_matches_loop_implementation():
    """Vectorized resampling should match a naive loop at the same seed."""
    from sklearn.metrics import r2_score

    rng_gold = np.random.default_rng(0)
    n = 200
    y = rng_gold.standard_normal(n)
    y_pred = y + rng_gold.standard_normal(n) * 0.5

    # Gold-standard: reproduce the old loop implementation exactly.
    n_boot = 500
    seed = 0
    rng = np.random.default_rng(seed)
    loop_r2s = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        loop_r2s[b] = r2_score(y[idx], y_pred[idx])
    lo_loop = float(np.percentile(loop_r2s, 2.5))
    hi_loop = float(np.percentile(loop_r2s, 97.5))

    out = bootstrap_r2_ci(y, y_pred, n_boot=n_boot, conf=0.95, seed=seed)
    # Vectorized path must match the loop endpoints within float round-off.
    assert abs(out["ci_lower"] - lo_loop) < 1e-10
    assert abs(out["ci_upper"] - hi_loop) < 1e-10

# ---------------------------------------------------------------------------
# M-13: filter non-finite / non-positive values robustly.
# ---------------------------------------------------------------------------

def test_bootstrap_r2_ci_filters_nan():
    y = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    yp = np.array([1.1, 1.9, 3.0, 4.1, 4.9])
    out = bootstrap_r2_ci(y, yp, n_boot=50, seed=0)
    assert out["n"] == 4

def test_calibration_coverage_filters_nonpositive_sigma():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    yp = np.array([1.0, 2.0, 3.0, 4.0])
    s = np.array([0.5, 0.0, -0.1, 0.5])
    out = calibration_coverage(y, yp, s, nominal=[0.95])
    assert out["n"] == 2
