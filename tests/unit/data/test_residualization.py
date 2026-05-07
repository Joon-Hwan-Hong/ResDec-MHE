"""Unit tests for src.data.residualization (per-fold OLS residualization helpers)."""
import numpy as np
import pandas as pd
import pytest
from src.data.residualization import fit_pathology_residual, apply_residual


def test_fit_returns_alpha_beta_per_axis():
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame({
        "cogn_global": rng.normal(0, 1, n),
        "gpath":       rng.uniform(0, 3, n),
    })
    fit = fit_pathology_residual(df, target="cogn_global", axes=["gpath"])
    assert "alpha" in fit
    assert "beta" in fit
    assert isinstance(fit["beta"], dict)
    assert "gpath" in fit["beta"]
    assert isinstance(fit["alpha"], float)
    assert isinstance(fit["beta"]["gpath"], float)


def test_apply_residual_yields_zero_mean_residuals_on_train_set():
    """Residuals on the training set should have zero mean by OLS construction."""
    rng = np.random.default_rng(42)
    n = 200
    gpath = rng.uniform(0, 3, n)
    cogn = -0.7 * gpath + 0.1 + rng.normal(0, 0.3, n)
    df_train = pd.DataFrame({"cogn_global": cogn, "gpath": gpath})

    fit = fit_pathology_residual(df_train, target="cogn_global", axes=["gpath"])
    residuals = apply_residual(df_train, target="cogn_global", fit=fit)

    assert abs(residuals.mean()) < 1e-9, "OLS train residuals should have zero mean"


def test_multi_axis_fit_recovers_all_coefficients():
    rng = np.random.default_rng(42)
    n = 200
    gpath = rng.uniform(0, 3, n)
    tang = rng.uniform(0, 5, n)
    amyl = rng.uniform(0, 4, n)
    cogn = (-0.5 * gpath - 0.3 * tang - 0.2 * amyl + 0.1
            + rng.normal(0, 0.2, n))
    df = pd.DataFrame({
        "cogn_global": cogn, "gpath": gpath,
        "tangsqrt": tang, "amylsqrt": amyl,
    })
    fit = fit_pathology_residual(
        df, target="cogn_global",
        axes=["gpath", "tangsqrt", "amylsqrt"],
    )
    assert abs(fit["beta"]["gpath"] - (-0.5)) < 0.1
    assert abs(fit["beta"]["tangsqrt"] - (-0.3)) < 0.1
    assert abs(fit["beta"]["amylsqrt"] - (-0.2)) < 0.1


def test_apply_to_held_out_set_works():
    """Fit on one subset, apply to another."""
    rng = np.random.default_rng(42)
    n = 200
    gpath_train = rng.uniform(0, 3, n)
    cogn_train = -0.5 * gpath_train + rng.normal(0, 0.3, n)
    df_train = pd.DataFrame({"cogn_global": cogn_train, "gpath": gpath_train})

    fit = fit_pathology_residual(df_train, target="cogn_global", axes=["gpath"])

    n_val = 50
    gpath_val = rng.uniform(0, 3, n_val)
    cogn_val = -0.5 * gpath_val + rng.normal(0, 0.3, n_val)
    df_val = pd.DataFrame({"cogn_global": cogn_val, "gpath": gpath_val})

    residuals_val = apply_residual(df_val, target="cogn_global", fit=fit)
    assert residuals_val.shape == (n_val,)


def test_missing_axis_raises():
    df = pd.DataFrame({"cogn_global": [0.1, 0.2], "gpath": [1.0, 2.0]})
    with pytest.raises(KeyError):
        fit_pathology_residual(df, target="cogn_global", axes=["nonexistent"])


def test_nan_in_target_propagates_to_nan_residual():
    df = pd.DataFrame({
        "cogn_global": [0.1, np.nan, 0.3],
        "gpath":       [1.0, 2.0, 3.0],
    })
    fit = fit_pathology_residual(
        df.dropna(), target="cogn_global", axes=["gpath"],
    )
    res = apply_residual(df, target="cogn_global", fit=fit)
    assert np.isnan(res[1])
    assert not np.isnan(res[0])
    assert not np.isnan(res[2])


def test_nan_in_axis_propagates_to_nan_residual():
    df = pd.DataFrame({
        "cogn_global": [0.1, 0.2, 0.3],
        "gpath":       [1.0, np.nan, 3.0],
    })
    fit = fit_pathology_residual(
        df.dropna(), target="cogn_global", axes=["gpath"],
    )
    res = apply_residual(df, target="cogn_global", fit=fit)
    assert np.isnan(res[1])
    assert not np.isnan(res[0])
    assert not np.isnan(res[2])
