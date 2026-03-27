"""
Uncertainty analysis for Bayesian predictions.

Produces analysis outputs:
1. prediction_uncertainty.csv - Subject ID, predicted mean, predicted std, actual
2. uncertainty_correlates.csv - Correlation of uncertainty with covariates
3. calibration_summary.csv - Calibration error at 1σ, 2σ, 3σ levels

Output format: Tidy DataFrames saved as Parquet (primary) and CSV (human-readable).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.io import save_dataframe
from src.utils.statistics import compute_calibration_metrics, CALIBRATION_LEVELS

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyAnalysisResult:
    """
    Container for uncertainty analysis results.

    Attributes:
        prediction_summary: DataFrame with per-subject predictions and uncertainty
        calibration: DataFrame with calibration error at each sigma level
        correlates: DataFrame with uncertainty correlations with covariates
        metadata: Additional analysis metadata
    """

    prediction_summary: pd.DataFrame
    calibration: pd.DataFrame | None = None
    correlates: pd.DataFrame | None = None
    metadata: dict = field(default_factory=dict)


class UncertaintyAnalyzer:
    """
    Analyze prediction uncertainty quality and correlates.

    Evaluates whether predicted uncertainty (std) is well-calibrated
    and investigates what factors correlate with higher uncertainty.

    Example:
        >>> analyzer = UncertaintyAnalyzer(
        ...     predicted_mean=means,     # [n_subjects]
        ...     predicted_std=stds,       # [n_subjects]
        ...     actual=targets,           # [n_subjects] (optional)
        ...     covariates=covariates_df, # DataFrame with covariates
        ... )
        >>> result = analyzer.analyze()
        >>> analyzer.save(result, output_dir)
    """

    def __init__(
        self,
        predicted_mean: np.ndarray,
        predicted_std: np.ndarray,
        actual: np.ndarray | None = None,
        subject_ids: list[str] | None = None,
        covariates: pd.DataFrame | None = None,
    ):
        """
        Initialize analyzer with predictions and optional covariates.

        Args:
            predicted_mean: Predicted mean values [n_subjects]
            predicted_std: Predicted uncertainty (std) [n_subjects]
            actual: Actual target values [n_subjects] (required for calibration)
            subject_ids: Subject identifiers
            covariates: DataFrame with covariates for correlation analysis
                       (e.g., cell_count, n_regions, pathology)
        """
        self.predicted_mean = np.asarray(predicted_mean).flatten()
        self.predicted_std = np.asarray(predicted_std).flatten()
        self.actual = np.asarray(actual).flatten() if actual is not None else None
        self.subject_ids = subject_ids or [f"subject_{i}" for i in range(len(self.predicted_mean))]
        self.covariates = covariates

        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """Validate input array shapes."""
        n = len(self.predicted_mean)

        if len(self.predicted_std) != n:
            raise ValueError(
                f"predicted_std has {len(self.predicted_std)} entries "
                f"but predicted_mean has {n}"
            )

        if self.actual is not None and len(self.actual) != n:
            raise ValueError(
                f"actual has {len(self.actual)} entries "
                f"but predicted_mean has {n}"
            )

        if len(self.subject_ids) != n:
            raise ValueError(
                f"subject_ids has {len(self.subject_ids)} entries "
                f"but predicted_mean has {n}"
            )

        if self.covariates is not None and len(self.covariates) != n:
            raise ValueError(
                f"covariates has {len(self.covariates)} rows "
                f"but predicted_mean has {n}"
            )

        # Check for valid std values
        if (self.predicted_std <= 0).any():
            raise ValueError("predicted_std must be positive")

    def analyze(self) -> UncertaintyAnalysisResult:
        """
        Run all uncertainty analyses.

        Returns:
            UncertaintyAnalysisResult with all analyses
        """
        # Prediction summary
        prediction_summary = self._compute_prediction_summary()

        # Calibration analysis (requires actual values)
        calibration = None
        if self.actual is not None:
            calibration = self._compute_calibration()

        # Correlation analysis (requires covariates)
        correlates = None
        if self.covariates is not None:
            correlates = self._compute_correlates()

        metadata = {
            "n_subjects": len(self.predicted_mean),
            "has_actual": self.actual is not None,
            "has_covariates": self.covariates is not None,
            "mean_std": float(self.predicted_std.mean()),
            "std_std": float(self.predicted_std.std()),
        }

        return UncertaintyAnalysisResult(
            prediction_summary=prediction_summary,
            calibration=calibration,
            correlates=correlates,
            metadata=metadata,
        )

    def _compute_prediction_summary(self) -> pd.DataFrame:
        """
        Compute per-subject prediction summary.

        Returns:
            DataFrame with columns: subject_id, predicted_mean, predicted_std,
                                   actual, residual, z_score
        """
        df = pd.DataFrame({
            "subject_id": self.subject_ids,
            "predicted_mean": self.predicted_mean,
            "predicted_std": self.predicted_std,
        })

        if self.actual is not None:
            df["actual"] = self.actual
            df["residual"] = self.actual - self.predicted_mean
            df["z_score"] = np.abs(df["residual"]) / self.predicted_std
        else:
            df["actual"] = np.nan
            df["residual"] = np.nan
            df["z_score"] = np.nan

        return df

    def _compute_calibration(self) -> pd.DataFrame:
        """
        Compute calibration error at different sigma levels.

        Delegates to shared implementation in src.utils.statistics.

        Returns:
            DataFrame with columns: level, expected_coverage, observed_coverage,
                                   calibration_error, interpretation
        """
        if self.actual is None:
            raise ValueError("actual values required for calibration")

        result = compute_calibration_metrics(
            self.predicted_mean,
            self.predicted_std,
            self.actual,
        )
        return result.detailed

    def _compute_correlates(self) -> pd.DataFrame:
        """
        Compute correlation of uncertainty with covariates.

        Returns:
            DataFrame with columns: covariate, correlation, p_value, interpretation
        """
        if self.covariates is None:
            raise ValueError("covariates required for correlation analysis")

        rows = []
        for col in self.covariates.columns:
            values = self.covariates[col].values

            # Skip non-numeric columns
            if not np.issubdtype(values.dtype, np.number):
                continue

            # Filter NaN values for safe correlation
            valid = ~(np.isnan(self.predicted_std) | np.isnan(values))
            if valid.sum() < 3:
                continue

            pred_valid = self.predicted_std[valid]
            values_valid = values[valid]

            # Skip columns with no variance
            if np.std(values_valid) == 0:
                continue

            # Compute Spearman correlation (more robust to outliers)
            corr, pval = stats.spearmanr(pred_valid, values_valid)

            # Interpretation
            if pval > 0.05:
                interp = "not_significant"
            elif abs(corr) > 0.5:
                interp = "strong"
            elif abs(corr) > 0.3:
                interp = "moderate"
            else:
                interp = "weak"

            rows.append({
                "covariate": col,
                "correlation": float(corr),
                "p_value": float(pval),
                "significant": pval < 0.05,
                "interpretation": interp,
            })

        df = pd.DataFrame(rows)
        if len(df) > 0:
            # FDR correction (Benjamini-Hochberg) for multiple comparisons
            from src.utils.statistics import benjamini_hochberg

            fdr_values, sig_mask = benjamini_hochberg(df["p_value"].values)
            df["p_value_fdr"] = fdr_values
            df["significant_fdr"] = sig_mask
            df = df.sort_values("p_value").reset_index(drop=True)

        return df

    def save(
        self,
        result: UncertaintyAnalysisResult,
        output_dir: str | Path,
        formats: list[Literal["parquet", "csv"]] | None = None,
    ) -> dict[str, Path]:
        """
        Save analysis results to files.

        Args:
            result: UncertaintyAnalysisResult to save
            output_dir: Directory for output files
            formats: Output formats (default: ["parquet", "csv"])

        Returns:
            Dict mapping output name to file path
        """
        if formats is None:
            formats = ["parquet", "csv"]

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        # Save prediction summary
        for fmt in formats:
            path = output_dir / f"prediction_uncertainty.{fmt}"
            save_dataframe(result.prediction_summary, path, fmt)
            saved_files[f"prediction_summary_{fmt}"] = path

        # Save calibration (if available)
        if result.calibration is not None:
            for fmt in formats:
                path = output_dir / f"calibration_summary.{fmt}"
                save_dataframe(result.calibration, path, fmt)
                saved_files[f"calibration_{fmt}"] = path

        # Save correlates (if available)
        if result.correlates is not None:
            for fmt in formats:
                path = output_dir / f"uncertainty_correlates.{fmt}"
                save_dataframe(result.correlates, path, fmt)
                saved_files[f"correlates_{fmt}"] = path

        logger.info(f"Saved uncertainty analysis to {output_dir}")
        return saved_files


def compute_uncertainty_analysis(
    predicted_mean: np.ndarray,
    predicted_std: np.ndarray,
    actual: np.ndarray | None = None,
    subject_ids: list[str] | None = None,
    covariates: pd.DataFrame | None = None,
    output_dir: str | Path | None = None,
) -> UncertaintyAnalysisResult:
    """
    Convenience function to compute and optionally save uncertainty analysis.

    Args:
        predicted_mean: Predicted mean values [n_subjects]
        predicted_std: Predicted uncertainty (std) [n_subjects]
        actual: Actual target values (required for calibration)
        subject_ids: Subject identifiers
        covariates: DataFrame with covariates for correlation analysis
        output_dir: If provided, save results to this directory

    Returns:
        UncertaintyAnalysisResult with analysis results
    """
    analyzer = UncertaintyAnalyzer(
        predicted_mean=predicted_mean,
        predicted_std=predicted_std,
        actual=actual,
        subject_ids=subject_ids,
        covariates=covariates,
    )

    result = analyzer.analyze()

    if output_dir is not None:
        analyzer.save(result, output_dir)

    return result


def compute_ece_regression(
    predicted_mean: np.ndarray,
    predicted_std: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE) for regression via binned analysis.

    This is distinct from z-score calibration in src.utils.statistics.calibration_error,
    which checks coverage at 1σ/2σ/3σ levels. ECE bins by predicted uncertainty and
    measures whether higher predicted std corresponds to larger actual errors.

    Args:
        predicted_mean: Predicted mean values
        predicted_std: Predicted uncertainty (std)
        actual: Actual target values
        n_bins: Number of bins for binning by uncertainty

    Returns:
        ECE score (lower is better, 0 = perfectly calibrated)
    """
    z_scores = np.abs(actual - predicted_mean) / predicted_std

    # Bin by predicted std
    std_percentiles = np.percentile(predicted_std, np.linspace(0, 100, n_bins + 1))
    bin_indices = np.digitize(predicted_std, std_percentiles[1:-1])

    ece = 0.0
    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        if mask.sum() == 0:
            continue

        # Expected coverage (at 1 sigma)
        expected = CALIBRATION_LEVELS["1_sigma"]

        # Observed coverage
        observed = (z_scores[mask] <= 1.0).mean()

        # Weight by bin size
        weight = mask.sum() / len(z_scores)

        ece += weight * abs(observed - expected)

    return float(ece)
