"""Subgroup stratified metrics for ResDec-H3 composite predictions.

For each named subgroup (e.g. ``"APOE_e4_0"``, ``"msex_1"``,
``"age_quartile_Q4"``), compute R², RMSE, Pearson r, and Spearman ρ of
``y_pred`` vs ``y_true`` along with 95% bootstrap CIs (percentile method).

This module is pure: inputs are NumPy arrays + a flat boolean-mask dict,
output is a nested dict, no I/O. Orchestration (loading per-fold npz,
metadata join, building masks, JSON/CSV write) lives in
``scripts/redesign/interpretability/subgroup_r2.py``.

Conventions
-----------
- Bootstrap: draw ``n_bootstrap`` resamples of the subgroup's indices
  WITH replacement, recompute all 4 metrics per resample, and report
  ``(q2.5, q97.5)`` as the 95% CI (percentile method — no normal
  approximation).
- Degenerate cases: subgroup with ``n < 3`` → all metrics NaN and all CIs
  ``(nan, nan)`` (Pearson/Spearman undefined for <3 points; we apply the
  same threshold to R²/RMSE for consistency).
- Bootstrap NaN handling: a resample may have zero variance in ``y_true``
  (all-same target), which makes R²/Pearson/Spearman NaN for that
  resample. NaN resamples are filtered before taking the percentile, and
  ``n_valid_bootstraps`` reports the count per subgroup.
- Reproducibility: ``np.random.default_rng(seed)`` seeds the full run; the
  same ``(y_true, y_pred, masks, n_bootstrap, seed)`` always yields the
  same CIs.
"""
from __future__ import annotations

import logging
import warnings
from typing import Mapping

import numpy as np
from scipy.stats import ConstantInputWarning, pearsonr, spearmanr
from sklearn.metrics import r2_score

logger = logging.getLogger(__name__)

# Minimum group size for metric computation. Pearson/Spearman are undefined
# for fewer than 3 paired points; we apply the same floor to R²/RMSE so the
# output schema is internally consistent (a subgroup either has all metrics
# defined or none).
_MIN_N_FOR_METRICS = 3

_NAN_CI: tuple[float, float] = (float("nan"), float("nan"))

# 95% percentile-method CI quantiles. Named so the bootstrap code reads as
# "take the 95% percentile CI" rather than a bare ``[0.025, 0.975]`` literal.
_CI_QUANTILES: tuple[float, float] = (0.025, 0.975)


def _point_metrics(y: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Point estimates: R², RMSE, Pearson r, Spearman ρ on paired arrays.

    Returns all-NaN if ``len(y) < _MIN_N_FOR_METRICS`` or if either array
    is constant (Pearson/Spearman ill-defined).
    """
    n = int(y.shape[0])
    if n < _MIN_N_FOR_METRICS:
        return {
            "r2": float("nan"),
            "rmse": float("nan"),
            "pearson_r": float("nan"),
            "spearman_rho": float("nan"),
        }

    rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
    # r2_score handles constant-y by returning 0.0 when both series are
    # identical; when y is constant but y != y_pred, it returns negative
    # values, which is fine (R² = 1 - SS_res/SS_tot is simply degenerate).
    # We still NaN-guard the edge case of identical constant series to
    # keep the convention uniform with Pearson/Spearman.
    var_y = float(np.var(y))
    if var_y == 0.0:
        # Both constant-and-equal and constant-but-different make R² ill-defined
        # for a per-subgroup interpretation. Emit NaN to be explicit.
        r2 = float("nan")
    else:
        r2 = float(r2_score(y, y_pred))

    # scipy raises a ConstantInputWarning when variance is 0. Suppress it
    # narrowly here and emit NaN — the warning adds noise without new
    # information since the caller already sees NaN + n_valid_bootstraps.
    # Only ConstantInputWarning is filtered so any other scipy warning
    # (e.g. NearConstantInputWarning) still surfaces.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        pearson_r = float(pearsonr(y, y_pred).statistic)
        spearman_rho = float(spearmanr(y, y_pred).statistic)

    return {
        "r2": r2,
        "rmse": rmse,
        "pearson_r": pearson_r,
        "spearman_rho": spearman_rho,
    }


def _bootstrap_cis(
    y: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[dict[str, tuple[float, float]], int]:
    """Percentile-method 95% CIs for (R², RMSE, Pearson r, Spearman ρ).

    Returns a dict keyed ``{metric}_ci → (lower, upper)`` plus
    ``n_valid_bootstraps`` (count of resamples producing all-finite metrics).
    """
    n = int(y.shape[0])
    if n < _MIN_N_FOR_METRICS or n_bootstrap <= 0:
        return (
            {
                "r2_ci": _NAN_CI,
                "rmse_ci": _NAN_CI,
                "pearson_r_ci": _NAN_CI,
                "spearman_rho_ci": _NAN_CI,
            },
            0,
        )

    r2_samples = np.empty(n_bootstrap, dtype=np.float64)
    rmse_samples = np.empty(n_bootstrap, dtype=np.float64)
    pearson_samples = np.empty(n_bootstrap, dtype=np.float64)
    spearman_samples = np.empty(n_bootstrap, dtype=np.float64)

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        yp = y_pred[idx]
        m = _point_metrics(yb, yp)
        r2_samples[b] = m["r2"]
        rmse_samples[b] = m["rmse"]
        pearson_samples[b] = m["pearson_r"]
        spearman_samples[b] = m["spearman_rho"]

    def _percentile_ci(samples: np.ndarray) -> tuple[float, float]:
        finite = samples[np.isfinite(samples)]
        if finite.size == 0:
            return _NAN_CI
        lo, hi = np.quantile(finite, _CI_QUANTILES)
        return (float(lo), float(hi))

    # ``n_valid_bootstraps`` = count of resamples where ALL four metrics
    # were finite. A single-metric view would hide the fact that e.g. a
    # Pearson-NaN resample is still a degenerate resample.
    all_finite = (
        np.isfinite(r2_samples)
        & np.isfinite(rmse_samples)
        & np.isfinite(pearson_samples)
        & np.isfinite(spearman_samples)
    )
    n_valid = int(all_finite.sum())

    return (
        {
            "r2_ci": _percentile_ci(r2_samples),
            "rmse_ci": _percentile_ci(rmse_samples),
            "pearson_r_ci": _percentile_ci(pearson_samples),
            "spearman_rho_ci": _percentile_ci(spearman_samples),
        },
        n_valid,
    )


def stratified_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subgroup_masks: Mapping[str, np.ndarray],
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, dict]:
    """Per-subgroup R²/RMSE/Pearson/Spearman with percentile bootstrap CIs.

    Parameters
    ----------
    y_true, y_pred : np.ndarray, shape ``[N]``
        Per-subject true targets and composite predictions.
    subgroup_masks : dict[str, np.ndarray]
        FLAT mapping ``{group_name: boolean_mask[N]}``. Each mask selects
        the subjects belonging to that group. Masks do NOT need to
        partition the set — multiple masks may overlap.
    n_bootstrap : int, default 1000
        Number of bootstrap resamples per subgroup.
    seed : int, default 42
        Seed for ``np.random.default_rng``; the same seed always yields
        the same CIs.

    Returns
    -------
    dict
        Flat dict keyed by ``group_name``. Each value is a dict with:

        - ``n``: int, subjects in the group
        - ``r2``, ``rmse``, ``pearson_r``, ``spearman_rho``: float point estimates
        - ``r2_ci``, ``rmse_ci``, ``pearson_r_ci``, ``spearman_rho_ci``:
          (lower, upper) 95% CI tuples
        - ``n_valid_bootstraps``: int, resamples with all-finite metrics

        Subgroups with ``n < 3`` get NaN metrics and ``(nan, nan)`` CIs.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must share shape; got {y_true.shape} vs {y_pred.shape}."
        )
    if y_true.ndim != 1:
        raise ValueError(f"y_true/y_pred must be 1-D; got ndim={y_true.ndim}.")

    n_total = y_true.shape[0]
    rng = np.random.default_rng(seed)

    out: dict[str, dict] = {}
    for group_name, mask in subgroup_masks.items():
        mask_arr = np.asarray(mask)
        if mask_arr.shape != (n_total,):
            raise ValueError(
                f"Mask for group {group_name!r} has shape {mask_arr.shape}; "
                f"expected ({n_total},)."
            )
        if mask_arr.dtype != bool:
            raise ValueError(
                f"Mask for group {group_name!r} must be boolean; got dtype {mask_arr.dtype}."
            )

        y_grp = y_true[mask_arr]
        yp_grp = y_pred[mask_arr]
        n = int(y_grp.shape[0])

        # Explicit warning on empty subgroups: n=0 is almost always a sign of
        # a bad mask (typo in family label, missing metadata, etc.), not a
        # legitimate "no members" case. Emit a UserWarning so pytest can
        # assert it and so the caller sees it in normal runtime.
        if n == 0:
            warnings.warn(
                f"Subgroup {group_name!r} is empty (n=0); metrics are NaN.",
                UserWarning,
                stacklevel=2,
            )

        point = _point_metrics(y_grp, yp_grp)
        cis, n_valid = _bootstrap_cis(
            y_grp, yp_grp, n_bootstrap=n_bootstrap, rng=rng,
        )
        out[group_name] = {
            "n": n,
            **point,
            **cis,
            "n_valid_bootstraps": n_valid,
        }
        logger.debug(
            "stratified_metrics: group=%s n=%d r2=%.4f r2_ci=%s n_valid_boot=%d",
            group_name, n, point["r2"], cis["r2_ci"], n_valid,
        )

    return out


__all__ = ["stratified_metrics"]
