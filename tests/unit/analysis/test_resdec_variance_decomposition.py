"""Unit tests for :mod:`src.analysis.resdec_variance_decomposition`.

Contract (from docs/plans/2026-04-22-resdec-h3-phase5-finish.md Task C.1):

    Var(y) = Var(y_tabpfn) + Var(f_1) + 2 * Cov(y_tabpfn, f_1) + Var(resid)

with ``resid = y_true - (y_tabpfn + f_1)``.

**Exact vs. approximate additivity.** The full expansion of ``Var(y)`` when
``y = y_tabpfn + f_1 + resid`` contains *six* covariance terms:
``Var(y) = Var(y_tabpfn) + Var(f_1) + Var(resid) + 2 Cov(y_tabpfn, f_1)
          + 2 Cov(y_tabpfn, resid) + 2 Cov(f_1, resid)``.
The reporting formula in the Task C.1 spec keeps only the
``2 Cov(y_tabpfn, f_1)`` cross term. The two dropped terms vanish in the
population limit whenever the composite prediction is OLS-orthogonal to
``resid``. They do **not** vanish in a finite sample unless the test is
set up so the residual is constructed orthogonal to the predictors.

These tests use orthogonal-by-construction residuals (Gram-Schmidt-style
residualisation) so the four-component additivity holds to floating-point
precision, matching the spec's ``abs=1e-6`` tolerance.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.analysis.resdec_variance_decomposition import decompose_variance


def _orthogonal_residual(rng: np.random.Generator, *predictors: np.ndarray) -> np.ndarray:
    """Return a residual with sample covariance zero against every ``predictor``.

    Builds the orthogonal complement via the OLS residual of a fresh random
    draw regressed on ``[1, *predictors]``. The intercept column absorbs the
    mean, and the OLS normal equations guarantee Cov(r, p_k) = 0 for all k
    at floating-point precision.
    """
    n = predictors[0].shape[0]
    r = rng.standard_normal(n)
    X = np.column_stack([np.ones(n), *predictors])
    # OLS projection; residual of r is orthogonal to every column of X
    # (including the intercept, so r is also mean-zero by construction).
    beta, *_ = np.linalg.lstsq(X, r, rcond=None)
    return r - X @ beta


def test_identity_case_f1_zero():
    """When f_1 == 0, var_resid == Var(y - y_tabpfn) exactly."""
    rng = np.random.default_rng(0)
    n = 200
    y_tabpfn = rng.standard_normal(n)
    f1 = np.zeros(n)
    # Construct noise orthogonal to y_tabpfn so the four-component identity is exact.
    noise = _orthogonal_residual(rng, y_tabpfn) * 0.5
    y_true = y_tabpfn + noise

    out = decompose_variance(y_true, y_tabpfn, f1)
    g = out["global"]
    assert g["var_f1"] == pytest.approx(0.0)
    assert g["cov_tabpfn_f1"] == pytest.approx(0.0)
    assert g["var_resid"] == pytest.approx(np.var(y_true - y_tabpfn, ddof=1))
    # Identity: var_y = var_tabpfn + var_resid (since var_f1 = cov_tabpfn_f1 = 0
    # and noise is orthogonal to y_tabpfn by construction).
    assert g["var_y"] == pytest.approx(g["var_tabpfn"] + g["var_resid"], abs=1e-6)
    assert g["n"] == n


def test_known_covariance():
    """Crafted orthogonal-residual case where additivity reconstructs Var(y) exactly."""
    n = 500
    rng = np.random.default_rng(42)
    y_tabpfn = rng.standard_normal(n) * 1.0
    f1 = rng.standard_normal(n) * 0.3
    # Noise orthogonal to both predictors → cov(y_tabpfn, noise) = cov(f1, noise) = 0.
    noise = _orthogonal_residual(rng, y_tabpfn, f1) * 0.2
    y_true = y_tabpfn + f1 + noise

    out = decompose_variance(y_true, y_tabpfn, f1)
    g = out["global"]
    assert g["var_tabpfn"] == pytest.approx(np.var(y_tabpfn, ddof=1))
    assert g["var_f1"] == pytest.approx(np.var(f1, ddof=1))
    # Additivity — spec contract: var_y = var_tabpfn + var_f1 + 2 cov + var_resid.
    # Exact at float precision because noise was constructed orthogonal to the predictors.
    reconstructed = g["var_tabpfn"] + g["var_f1"] + 2 * g["cov_tabpfn_f1"] + g["var_resid"]
    assert g["var_y"] == pytest.approx(reconstructed, abs=1e-6)
    assert g["n"] == n
    # Total explained fraction is 1 - var_resid / var_y.
    assert g["total_explained_fraction"] == pytest.approx(
        1.0 - g["var_resid"] / g["var_y"], abs=1e-9
    )


def test_subgroup_decomposition():
    """Subgroup split produces per-group dicts with same keys as global."""
    rng = np.random.default_rng(7)
    n = 300
    y_tabpfn = rng.standard_normal(n)
    f1 = rng.standard_normal(n) * 0.3
    noise = _orthogonal_residual(rng, y_tabpfn, f1) * 0.2
    y_true = y_tabpfn + f1 + noise

    labels = np.array(["A"] * 150 + ["B"] * 150)
    out = decompose_variance(
        y_true, y_tabpfn, f1, subgroups={"by_label": labels},
    )
    assert "by_label" in out
    assert set(out["by_label"].keys()) == {"A", "B"}
    assert out["by_label"]["A"]["n"] == 150
    assert out["by_label"]["B"]["n"] == 150

    expected_keys = {
        "var_y", "var_tabpfn", "var_f1", "cov_tabpfn_f1", "var_resid",
        "total_explained_fraction", "n",
    }
    assert expected_keys.issubset(out["global"].keys())
    assert expected_keys.issubset(out["by_label"]["A"].keys())
    assert expected_keys.issubset(out["by_label"]["B"].keys())

    # Per-group additivity: each group's variance budget reconstructs that group's var_y
    # using the same four-component formula. Global orthogonality does NOT imply
    # per-subgroup orthogonality, so we use a practical tolerance that still pins
    # the decomposition to within ~1% of var_y for n=150.
    for grp in (out["by_label"]["A"], out["by_label"]["B"]):
        reconstructed = (
            grp["var_tabpfn"] + grp["var_f1"] + 2 * grp["cov_tabpfn_f1"] + grp["var_resid"]
        )
        # Full expansion has two cross-cov terms (y_tabpfn↔resid, f1↔resid) that
        # vanish only in the full sample; per-subgroup they are O(1/sqrt(n_group)).
        assert grp["var_y"] == pytest.approx(reconstructed, abs=0.05)


def test_global_full_expansion_matches_var_y():
    """Independent of orthogonality, the full 6-term expansion must equal Var(y)."""
    rng = np.random.default_rng(11)
    n = 400
    y_tabpfn = rng.standard_normal(n)
    f1 = rng.standard_normal(n) * 0.3
    y_true = y_tabpfn + f1 + rng.standard_normal(n) * 0.2

    out = decompose_variance(y_true, y_tabpfn, f1)
    g = out["global"]
    resid = y_true - y_tabpfn - f1
    cov_tabpfn_resid = float(np.cov(y_tabpfn, resid, ddof=1)[0, 1])
    cov_f1_resid = float(np.cov(f1, resid, ddof=1)[0, 1])
    full = (
        g["var_tabpfn"] + g["var_f1"] + g["var_resid"]
        + 2 * g["cov_tabpfn_f1"]
        + 2 * cov_tabpfn_resid + 2 * cov_f1_resid
    )
    # Full expansion is exact at float precision.
    assert g["var_y"] == pytest.approx(full, abs=1e-8)


def test_subgroup_skips_missing_labels():
    """None / NaN labels are excluded from a subgroup (no 'nan' bucket)."""
    rng = np.random.default_rng(3)
    n = 100
    y_tabpfn = rng.standard_normal(n)
    f1 = rng.standard_normal(n) * 0.3
    y_true = y_tabpfn + f1 + rng.standard_normal(n) * 0.2

    labels = np.array(["A"] * 50 + ["B"] * 40 + ["MISS"] * 10, dtype=object)
    labels_with_nulls = np.where(labels == "MISS", None, labels)

    out = decompose_variance(
        y_true, y_tabpfn, f1, subgroups={"by_label": labels_with_nulls},
    )
    assert set(out["by_label"].keys()) == {"A", "B"}
    assert out["by_label"]["A"]["n"] == 50
    assert out["by_label"]["B"]["n"] == 40


def test_length_mismatch_raises():
    """Mismatched input lengths raise ValueError."""
    y_true = np.zeros(10)
    y_tabpfn = np.zeros(10)
    f1 = np.zeros(9)
    with pytest.raises(ValueError):
        decompose_variance(y_true, y_tabpfn, f1)


def test_subgroup_length_mismatch_raises():
    """Subgroup labels with wrong length raise ValueError."""
    y_true = np.zeros(10)
    y_tabpfn = np.zeros(10)
    f1 = np.zeros(10)
    with pytest.raises(ValueError):
        decompose_variance(y_true, y_tabpfn, f1, subgroups={"by_x": np.array(["a"] * 9)})


def test_tiny_group_returns_nan():
    """A subgroup with <2 samples has undefined sample variance; n=1 returns NaN."""
    rng = np.random.default_rng(1)
    n = 50
    y_tabpfn = rng.standard_normal(n)
    f1 = rng.standard_normal(n) * 0.3
    y_true = y_tabpfn + f1 + rng.standard_normal(n) * 0.2

    labels = np.array(["A"] + ["B"] * 49)
    out = decompose_variance(
        y_true, y_tabpfn, f1, subgroups={"by_label": labels},
    )
    assert out["by_label"]["A"]["n"] == 1
    assert np.isnan(out["by_label"]["A"]["var_y"])
    assert out["by_label"]["B"]["n"] == 49
    assert not np.isnan(out["by_label"]["B"]["var_y"])


def test_zero_variance_y_returns_nan_fraction():
    """When Var(y) == 0, total_explained_fraction is NaN (avoids 0/0)."""
    y = np.zeros(10)
    t = np.zeros(10)
    f = np.zeros(10)
    out = decompose_variance(y, t, f)
    assert np.isnan(out["global"]["total_explained_fraction"])
    assert out["global"]["var_y"] == 0.0
