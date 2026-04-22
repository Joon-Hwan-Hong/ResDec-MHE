"""Statistical rigor primitives for ResDec-H3 evaluation.

Three pure, I/O-free functions used by the ResDec-H3 paper-prep pipeline:

1. :func:`paired_wilcoxon` — paired Wilcoxon signed-rank over per-fold R²
   values (``n=5`` in the canonical 5-fold CV). Wraps
   :func:`scipy.stats.wilcoxon` with ``alternative="greater"`` by default
   and catches the degenerate ``ValueError`` scipy raises when all
   differences are exactly zero (identical arrays).
2. :func:`bootstrap_r2_ci` — percentile bootstrap 95% CI on the pooled R²
   of a prediction vector by resampling the subject index with
   replacement ``n_boot`` times. NaN-safe: non-finite predictions are
   filtered before resampling.
3. :func:`calibration_coverage` — empirical coverage of nominal levels
   under a Gaussian assumption, using the z-score threshold
   ``|y_true - y_pred| <= z * sigma`` with
   ``z = scipy.stats.norm.ppf(0.5 + nominal / 2)``.

Orchestration (loading per-fold npz + baseline CSVs, writing JSON/MD)
lives in ``scripts/redesign/interpretability/paired_tests_and_bootstrap.py``.
"""
from __future__ import annotations

import logging
from typing import Iterable, Sequence

import numpy as np
from scipy import stats
from sklearn.metrics import r2_score

logger = logging.getLogger(__name__)


def paired_wilcoxon(
    fold_r2s_ours: Sequence[float] | np.ndarray,
    fold_r2s_baseline: Sequence[float] | np.ndarray,
    alternative: str = "greater",
) -> dict:
    """Paired Wilcoxon signed-rank test on per-fold R² values.

    Parameters
    ----------
    fold_r2s_ours, fold_r2s_baseline : array-like, shape ``[n_folds]``
        Per-fold R² (or any scalar metric); paired by index.
    alternative : {"two-sided", "greater", "less"}, default "greater"
        Passed to :func:`scipy.stats.wilcoxon`. ``"greater"`` tests the
        hypothesis that ``ours > baseline`` in median.

    Returns
    -------
    dict
        ``{statistic, p_value, n_folds, median_diff}``. When all
        differences are exactly zero, scipy raises ``ValueError``; we
        catch it and return ``statistic=0.0, p_value=1.0`` (no evidence
        of improvement).

    Notes
    -----
    With ``n_folds=5``, a one-sided Wilcoxon has minimum achievable
    p-value ``1 / 2**5 = 0.03125`` (all positive differences). The test
    is under-powered at this sample size; it is included per the spec
    for honest reporting rather than strong inferential power.
    """
    ours = np.asarray(fold_r2s_ours, dtype=np.float64)
    base = np.asarray(fold_r2s_baseline, dtype=np.float64)
    if ours.shape != base.shape:
        raise ValueError(
            f"Shape mismatch: ours={ours.shape}, baseline={base.shape}."
        )
    if ours.ndim != 1:
        raise ValueError(f"Inputs must be 1-D; got ndim={ours.ndim}.")

    n = int(ours.shape[0])
    diffs = ours - base
    median_diff = float(np.median(diffs))

    try:
        result = stats.wilcoxon(ours, base, alternative=alternative)
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
    except ValueError as exc:
        # scipy.stats.wilcoxon raises when all differences are zero (or, in
        # older scipy, any diff is zero without zero_method set). In that
        # case there is no signed-rank signal → p_value = 1.0.
        logger.debug(
            "paired_wilcoxon: scipy raised %r; returning degenerate p=1.0.",
            exc,
        )
        statistic = 0.0
        p_value = 1.0

    return {
        "statistic": statistic,
        "p_value": p_value,
        "n_folds": n,
        "median_diff": median_diff,
    }


def bootstrap_r2_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 1000,
    conf: float = 0.95,
    seed: int = 42,
) -> dict:
    """Percentile bootstrap CI on the pooled R² of ``y_pred`` vs ``y_true``.

    Resamples subject indices with replacement ``n_boot`` times via
    :meth:`numpy.random.Generator.integers`, computes :func:`sklearn.metrics.r2_score`
    on each resample, and reports the ``[(1-conf)/2, (1+conf)/2]``
    percentile interval.

    Parameters
    ----------
    y_true, y_pred : np.ndarray, shape ``[N]``
        Per-subject truth and prediction vectors. Non-finite entries
        in either vector are filtered out before resampling.
    n_boot : int, default 1000
        Number of bootstrap resamples.
    conf : float, default 0.95
        Nominal CI level.
    seed : int, default 42
        Seed for the ``numpy.random.default_rng`` resampler.

    Returns
    -------
    dict
        ``{point_r2, ci_lower, ci_upper, n_boot, conf, n}`` where ``n``
        is the post-filter subject count.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}."
        )
    if y_true.ndim != 1:
        raise ValueError(f"Inputs must be 1-D; got ndim={y_true.ndim}.")

    finite_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[finite_mask]
    y_pred = y_pred[finite_mask]
    n = int(y_true.shape[0])
    if n < 2:
        raise ValueError(
            f"Bootstrap requires at least 2 finite subject pairs; got {n}."
        )

    point_r2 = float(r2_score(y_true, y_pred))
    rng = np.random.default_rng(seed)
    boot_r2s = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_r2s[b] = r2_score(y_true[idx], y_pred[idx])

    lower_q = (1.0 - conf) / 2.0
    upper_q = 1.0 - lower_q
    ci_lower = float(np.percentile(boot_r2s, lower_q * 100.0))
    ci_upper = float(np.percentile(boot_r2s, upper_q * 100.0))

    return {
        "point_r2": point_r2,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_boot": int(n_boot),
        "conf": float(conf),
        "n": n,
    }


def calibration_coverage(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sigma: np.ndarray,
    nominal: Iterable[float] = (0.5, 0.68, 0.8, 0.95),
) -> dict:
    """Empirical coverage at each nominal level under a Gaussian assumption.

    For each ``p in nominal``, compute ``z = Φ⁻¹(0.5 + p/2)`` and report
    the empirical fraction of subjects satisfying
    ``|y_true - y_pred| <= z * sigma``. When the reported ``sigma``
    is well-calibrated and residuals are Gaussian, empirical coverage
    matches ``p`` at large sample size.

    Parameters
    ----------
    y_true, y_pred, sigma : np.ndarray, shape ``[N]``
        Per-subject truth, point prediction, and reported standard
        deviation. All three must share shape. Non-finite or non-positive
        ``sigma`` entries are filtered out (division-by-zero guard).
    nominal : iterable of floats in (0, 1)
        Nominal coverage levels to evaluate.

    Returns
    -------
    dict
        ``{"coverage_at_<p>": empirical_fraction, ...,
           "mean_sigma": ..., "mean_abs_residual": ..., "n": ...}``.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    if not (y_true.shape == y_pred.shape == sigma.shape):
        raise ValueError(
            "Shape mismatch: "
            f"y_true={y_true.shape}, y_pred={y_pred.shape}, sigma={sigma.shape}."
        )
    if y_true.ndim != 1:
        raise ValueError(f"Inputs must be 1-D; got ndim={y_true.ndim}.")

    finite_mask = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
        & np.isfinite(sigma)
        & (sigma > 0.0)
    )
    y_true_f = y_true[finite_mask]
    y_pred_f = y_pred[finite_mask]
    sigma_f = sigma[finite_mask]
    n = int(y_true_f.shape[0])
    if n == 0:
        raise ValueError("No finite / positive-sigma subject pairs for calibration.")

    abs_resid = np.abs(y_true_f - y_pred_f)
    out: dict = {}
    for p in nominal:
        if not (0.0 < p < 1.0):
            raise ValueError(f"Nominal level must be in (0, 1); got {p}.")
        z = float(stats.norm.ppf(0.5 + p / 2.0))
        covered = abs_resid <= z * sigma_f
        out[f"coverage_at_{p}"] = float(np.mean(covered))

    out["mean_sigma"] = float(np.mean(sigma_f))
    out["mean_abs_residual"] = float(np.mean(abs_resid))
    out["n"] = n
    return out


__all__ = [
    "bootstrap_r2_ci",
    "calibration_coverage",
    "paired_wilcoxon",
]
