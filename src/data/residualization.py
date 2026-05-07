"""Per-fold OLS residualization of cognition against pathology.

The cognitive residual at subject i is:
    target_i  =  cogn_global_i  -  E[cogn_global | pathology_i]
              =  cogn_global_i  -  (alpha + sum_k beta_k * pathology_k_i)

where (alpha, beta_k) are fit by OLS on the training subset only (per fold)
and applied to ALL subjects to compute the residual target.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def fit_pathology_residual(
    df: pd.DataFrame,
    *,
    target: str,
    axes: Sequence[str],
) -> dict:
    """OLS-fit cogn ~ alpha + sum_k beta_k * pathology_k on `df`.

    Returns
    -------
    dict with:
        - "alpha": float - OLS intercept
        - "beta":  dict[axis_name -> float] - OLS slopes per axis
        - "axes":  list[str] - copy of input axes (for reproducibility)
        - "n_train": int - training-set size used for the fit
    """
    missing = [a for a in axes if a not in df.columns]
    if missing:
        raise KeyError(f"axes not in df: {missing}")
    if target not in df.columns:
        raise KeyError(f"target '{target}' not in df")

    df_clean = df.dropna(subset=[target] + list(axes))
    X = df_clean[list(axes)].to_numpy(dtype=float)
    y = df_clean[target].to_numpy(dtype=float)

    reg = LinearRegression()
    reg.fit(X, y)

    return {
        "alpha": float(reg.intercept_),
        "beta": {a: float(reg.coef_[i]) for i, a in enumerate(axes)},
        "axes": list(axes),
        "n_train": int(len(df_clean)),
    }


def apply_residual(
    df: pd.DataFrame,
    *,
    target: str,
    fit: dict,
) -> np.ndarray:
    """Apply the OLS fit to compute residuals for every row in `df`.

    Returns
    -------
    np.ndarray of shape (len(df),) - the residualized target.
    Subjects with NaN in `target` or in any axis produce NaN.
    """
    axes = fit["axes"]
    alpha = fit["alpha"]
    beta = fit["beta"]

    expected = np.full(len(df), alpha, dtype=float)
    for a in axes:
        col = df[a].to_numpy(dtype=float)
        expected = expected + beta[a] * col

    y = df[target].to_numpy(dtype=float)
    return y - expected
