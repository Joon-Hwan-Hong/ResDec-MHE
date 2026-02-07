"""
Shared statistical utility functions.

Provides common statistical computations used across training and analysis modules:
- Calibration error metrics for uncertainty quantification
- Gini coefficient for heterogeneity analysis
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from src.data.constants import EPSILON_DIVISION


# Standard Gaussian calibration levels (expected coverage at 1σ, 2σ, 3σ)
CALIBRATION_LEVELS: dict[str, float] = {
    "1_sigma": 0.6827,
    "2_sigma": 0.9545,
    "3_sigma": 0.9973,
}


@dataclass
class CalibrationResult:
    """Results from calibration analysis.

    Attributes:
        detailed: DataFrame with per-level calibration metrics
        mean_error: Mean calibration error across all levels (0 = perfect)
        is_overconfident: True if model is systematically overconfident
        is_underconfident: True if model is systematically underconfident
    """
    detailed: pd.DataFrame
    mean_error: float
    is_overconfident: bool
    is_underconfident: bool


def compute_calibration_metrics(
    predicted_mean: np.ndarray,
    predicted_std: np.ndarray,
    actual: np.ndarray,
    epsilon: float = EPSILON_DIVISION,
) -> CalibrationResult:
    """
    Compute calibration error at different sigma levels.

    For a well-calibrated model:
    - ~68.27% of actual values should fall within 1σ of predictions
    - ~95.45% should fall within 2σ
    - ~99.73% should fall within 3σ

    Args:
        predicted_mean: [N] predicted values
        predicted_std: [N] predicted standard deviations
        actual: [N] ground truth values
        epsilon: Small value for numerical stability

    Returns:
        CalibrationResult with detailed metrics and summary statistics
    """
    z_scores = np.abs(actual - predicted_mean) / (predicted_std + epsilon)

    rows = []
    gaps = []

    for level_name, expected in CALIBRATION_LEVELS.items():
        n_sigma = int(level_name[0])

        # Observed coverage: fraction of z-scores <= n_sigma
        observed = float(np.mean(z_scores <= n_sigma))

        # Calibration error (positive = underconfident, negative = overconfident)
        error = observed - expected
        gaps.append(error)

        # Interpretation
        if error > 0.05:
            interp = "underconfident"
        elif error < -0.05:
            interp = "overconfident"
        else:
            interp = "well_calibrated"

        rows.append({
            "level": level_name,
            "n_sigma": n_sigma,
            "expected_coverage": expected,
            "observed_coverage": observed,
            "calibration_error": error,
            "interpretation": interp,
        })

    detailed = pd.DataFrame(rows)
    mean_error = float(np.mean(gaps))

    return CalibrationResult(
        detailed=detailed,
        mean_error=mean_error,
        is_overconfident=mean_error < -0.05,
        is_underconfident=mean_error > 0.05,
    )


def calibration_error(
    predicted_mean: np.ndarray,
    predicted_std: np.ndarray,
    actual: np.ndarray,
    epsilon: float = EPSILON_DIVISION,
) -> float:
    """
    Compute mean calibration error (convenience wrapper).

    For a well-calibrated model: 68.3% of z-scores ≤ 1, 95.4% ≤ 2, 99.7% ≤ 3.
    Returns mean gap across these levels.

    Args:
        predicted_mean: [N] predicted values
        predicted_std: [N] predicted standard deviations
        actual: [N] ground truth values
        epsilon: Small value for numerical stability

    Returns:
        Mean calibration error (0 = perfect, negative = overconfident)
    """
    return compute_calibration_metrics(
        predicted_mean, predicted_std, actual, epsilon
    ).mean_error


def gini_coefficient(values: np.ndarray) -> float:
    """
    Compute Gini coefficient (measure of inequality/concentration).

    The Gini coefficient ranges from 0 to 1:
    - 0 = perfect equality (all values identical)
    - 1 = perfect inequality (one value has everything)

    Used in heterogeneity analysis to measure concentration of attention weights.

    Args:
        values: Array of non-negative values

    Returns:
        Gini coefficient in [0, 1]
    """
    values = np.asarray(values, dtype=np.float64)
    values = np.sort(values)
    n = len(values)

    if n == 0:
        return 0.0

    total = values.sum()
    if total == 0:
        return 0.0

    index = np.arange(1, n + 1)
    return float(((2 * index - n - 1) * values).sum() / (n * total + EPSILON_DIVISION))


def cohens_d(
    group1: np.ndarray,
    group2: np.ndarray,
) -> float:
    """
    Compute Cohen's d effect size.

    Args:
        group1: First group of values
        group2: Second group of values

    Returns:
        Cohen's d (positive = group1 > group2)
    """
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    return float((mean1 - mean2) / (pooled_std + EPSILON_DIVISION))


def cohens_d_with_ci(
    group1: np.ndarray,
    group2: np.ndarray,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """
    Compute Cohen's d effect size with confidence interval.

    Uses the non-central t-distribution approximation.

    Args:
        group1: First group of values
        group2: Second group of values
        confidence: Confidence level (default: 0.95)

    Returns:
        Tuple of (d, ci_lower, ci_upper)
    """
    from scipy import stats

    d = cohens_d(group1, group2)
    n1, n2 = len(group1), len(group2)

    # Standard error of d
    se = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))

    # t critical value
    alpha = 1 - confidence
    df = n1 + n2 - 2
    t_crit = stats.t.ppf(1 - alpha / 2, df)

    ci_lower = d - t_crit * se
    ci_upper = d + t_crit * se

    return float(d), float(ci_lower), float(ci_upper)


def cohens_d_vectorized(
    group1_mean: np.ndarray,
    group1_std: np.ndarray,
    n1: int,
    group2_mean: np.ndarray,
    group2_std: np.ndarray,
    n2: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized Cohen's d across multiple features.

    Same formula as cohens_d() but operates on pre-computed summary statistics
    to avoid repeated loop overhead.

    Args:
        group1_mean: Mean of group 1 per feature [n_features]
        group1_std: Std (ddof=1) of group 1 per feature [n_features]
        n1: Number of samples in group 1
        group2_mean: Mean of group 2 per feature [n_features]
        group2_std: Std (ddof=1) of group 2 per feature [n_features]
        n2: Number of samples in group 2

    Returns:
        Tuple of (d, pooled_std) arrays [n_features]
    """
    pooled_var = ((n1 - 1) * group1_std**2 + (n2 - 1) * group2_std**2) / (n1 + n2 - 2)
    pooled_std = np.sqrt(pooled_var)
    d = np.zeros_like(group1_mean)
    nonzero = pooled_std > EPSILON_DIVISION
    d[nonzero] = (group1_mean[nonzero] - group2_mean[nonzero]) / pooled_std[nonzero]
    return d, pooled_std


def cohens_d_ci_vectorized(
    d: np.ndarray,
    n1: int,
    n2: int,
    confidence: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized CI for Cohen's d using Hedges & Olkin (1985) approximation.

    Args:
        d: Cohen's d values per feature [n_features]
        n1: Number of samples in group 1
        n2: Number of samples in group 2
        confidence: Confidence level (default: 0.95)

    Returns:
        Tuple of (ci_lower, ci_upper) arrays [n_features]
    """
    from scipy import stats as sp_stats

    se = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))
    df = n1 + n2 - 2
    t_crit = sp_stats.t.ppf(1 - (1 - confidence) / 2, df)
    return d - t_crit * se, d + t_crit * se


def benjamini_hochberg(
    pvalues: np.ndarray,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Benjamini-Hochberg FDR correction.

    Args:
        pvalues: Array of p-values
        alpha: Significance threshold (default: 0.05)

    Returns:
        Tuple of (adjusted_pvalues, significant_mask)
    """
    pvalues = np.asarray(pvalues)
    n = len(pvalues)

    if n == 0:
        return np.array([]), np.array([], dtype=bool)

    # Sort p-values
    sorted_idx = np.argsort(pvalues)
    sorted_pvals = pvalues[sorted_idx]

    # Compute adjusted p-values
    adjusted = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if i == n - 1:
            adjusted[sorted_idx[i]] = sorted_pvals[i]
        else:
            adjusted[sorted_idx[i]] = min(
                sorted_pvals[i] * n / (i + 1),
                adjusted[sorted_idx[i + 1]]
            )

    # Clip to [0, 1]
    adjusted = np.clip(adjusted, 0, 1)

    # Significant if adjusted p-value < alpha
    significant = adjusted < alpha

    return adjusted, significant


def derive_resilience_groups(
    cognition_scores: np.ndarray,
    pathology_scores: np.ndarray,
    pathology_percentile: float = 66.7,
    cognition_low_percentile: float = 33.3,
    cognition_high_percentile: float = 66.7,
) -> np.ndarray:
    """
    Derive resilient/vulnerable group labels from cognition and pathology scores.

    Two-stage stratification:
    1. Select high-pathology subjects (>= pathology_percentile)
    2. Within those, label top cognition tertile as "resilient",
       bottom tertile as "vulnerable", rest excluded ("")

    Mirrors the logic in ResilienceSignatureAnalyzer._identify_groups()
    but as a stateless utility for reuse across pipelines.

    Args:
        cognition_scores: [n_subjects] cognitive performance scores
        pathology_scores: [n_subjects] pathology burden scores
        pathology_percentile: Percentile threshold for "high pathology" (default: 66.7)
        cognition_low_percentile: Below this = "vulnerable" within high-path (default: 33.3)
        cognition_high_percentile: Above this = "resilient" within high-path (default: 66.7)

    Returns:
        [n_subjects] string array with values "resilient", "vulnerable", or ""
    """
    n = len(cognition_scores)
    labels = np.full(n, "", dtype=object)

    # Handle NaN in either score — exclude from grouping
    valid = ~(np.isnan(cognition_scores) | np.isnan(pathology_scores))
    if valid.sum() < 6:
        return labels.astype(str)

    # Stage 1: high pathology threshold
    path_threshold = np.nanpercentile(pathology_scores[valid], pathology_percentile)
    high_path_mask = valid & (pathology_scores >= path_threshold)

    if high_path_mask.sum() < 4:
        return labels.astype(str)

    # Stage 2: cognition split within high-pathology subjects
    high_path_cognition = cognition_scores[high_path_mask]
    cog_low = np.nanpercentile(high_path_cognition, cognition_low_percentile)
    cog_high = np.nanpercentile(high_path_cognition, cognition_high_percentile)

    high_path_indices = np.where(high_path_mask)[0]
    for idx in high_path_indices:
        cog = cognition_scores[idx]
        if cog >= cog_high:
            labels[idx] = "resilient"
        elif cog <= cog_low:
            labels[idx] = "vulnerable"

    return labels.astype(str)


def attention_entropy(
    attention: np.ndarray,
    axis: int | None = None,
    epsilon: float = EPSILON_DIVISION,
) -> float | np.ndarray:
    """
    Compute Shannon entropy of attention distribution.

    Can operate on flat arrays (returning scalar) or along an axis
    (returning array of entropies).

    Args:
        attention: Attention weights (non-negative, will be normalized)
        axis: Axis along which to compute entropy. If None, flattens array.
        epsilon: Small value for numerical stability

    Returns:
        Entropy value(s). Higher entropy = more uniform attention.
    """
    attention = np.asarray(attention, dtype=np.float64)

    if axis is None:
        # Flatten and compute single entropy
        values = attention.flatten()
        values = values[values > 0]
        if len(values) == 0:
            return 0.0

        # Normalize to probability distribution
        p = values / values.sum()
        return float(-np.sum(p * np.log(p + epsilon)))
    else:
        # Compute entropy along axis
        # Normalize along axis
        sums = attention.sum(axis=axis, keepdims=True)
        sums = np.where(sums > 0, sums, 1)  # Avoid division by zero
        p = attention / sums

        # Clip for numerical stability
        p = np.clip(p, epsilon, 1.0)

        # Shannon entropy: -sum(p * log(p))
        entropy = -np.sum(p * np.log(p), axis=axis)
        return entropy
