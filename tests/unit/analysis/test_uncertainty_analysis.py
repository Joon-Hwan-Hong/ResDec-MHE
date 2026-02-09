"""Tests for uncertainty analysis module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from src.analysis.uncertainty_analysis import (
    UncertaintyAnalyzer,
    UncertaintyAnalysisResult,
    compute_uncertainty_analysis,
    compute_ece_regression,
    CALIBRATION_LEVELS,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_predictions():
    """Sample prediction data (well-calibrated)."""
    np.random.seed(42)
    n_subjects = 50
    actual = np.random.randn(n_subjects) * 2 + 5  # Mean 5, std 2
    true_std = np.abs(np.random.randn(n_subjects)) * 0.5 + 0.3
    predicted_mean = actual + np.random.randn(n_subjects) * true_std
    predicted_std = true_std + np.abs(np.random.randn(n_subjects)) * 0.1

    return {
        "predicted_mean": predicted_mean,
        "predicted_std": predicted_std,
        "actual": actual,
    }


@pytest.fixture
def sample_subject_ids():
    """Sample subject IDs."""
    return [f"ROSMAP_{i:03d}" for i in range(50)]


@pytest.fixture
def sample_covariates():
    """Sample covariates DataFrame."""
    np.random.seed(42)
    n = 50
    return pd.DataFrame({
        "cell_count": np.random.randint(1000, 10000, n),
        "n_regions": np.random.randint(1, 5, n),
        "pathology_level": np.random.rand(n) * 10,
        "age": np.random.randint(60, 95, n),
        "education": np.random.randint(8, 20, n),
    })


# =============================================================================
# UncertaintyAnalyzer Tests
# =============================================================================


class TestUncertaintyAnalyzerInit:
    """Test UncertaintyAnalyzer initialization."""

    def test_init_basic(self, sample_predictions):
        """Test basic initialization."""
        analyzer = UncertaintyAnalyzer(
            predicted_mean=sample_predictions["predicted_mean"],
            predicted_std=sample_predictions["predicted_std"],
        )

        assert len(analyzer.predicted_mean) == 50
        assert len(analyzer.predicted_std) == 50
        assert analyzer.actual is None

    def test_init_with_actual(self, sample_predictions):
        """Test initialization with actual values."""
        analyzer = UncertaintyAnalyzer(
            predicted_mean=sample_predictions["predicted_mean"],
            predicted_std=sample_predictions["predicted_std"],
            actual=sample_predictions["actual"],
        )

        assert analyzer.actual is not None
        assert len(analyzer.actual) == 50

    def test_init_with_covariates(self, sample_predictions, sample_covariates):
        """Test initialization with covariates."""
        analyzer = UncertaintyAnalyzer(
            predicted_mean=sample_predictions["predicted_mean"],
            predicted_std=sample_predictions["predicted_std"],
            covariates=sample_covariates,
        )

        assert analyzer.covariates is not None
        assert len(analyzer.covariates) == 50

    def test_init_with_subject_ids(self, sample_predictions, sample_subject_ids):
        """Test initialization with subject IDs."""
        analyzer = UncertaintyAnalyzer(
            predicted_mean=sample_predictions["predicted_mean"],
            predicted_std=sample_predictions["predicted_std"],
            subject_ids=sample_subject_ids,
        )

        assert analyzer.subject_ids == sample_subject_ids

    def test_init_auto_subject_ids(self, sample_predictions):
        """Test automatic subject ID generation."""
        analyzer = UncertaintyAnalyzer(
            predicted_mean=sample_predictions["predicted_mean"],
            predicted_std=sample_predictions["predicted_std"],
        )

        assert len(analyzer.subject_ids) == 50
        assert all("subject_" in sid for sid in analyzer.subject_ids)


class TestUncertaintyAnalyzerValidation:
    """Test input validation."""

    def test_mismatched_lengths(self, sample_predictions):
        """Test error on mismatched array lengths."""
        with pytest.raises(ValueError, match="predicted_std"):
            UncertaintyAnalyzer(
                predicted_mean=sample_predictions["predicted_mean"],
                predicted_std=sample_predictions["predicted_std"][:40],
            )

    def test_mismatched_actual_length(self, sample_predictions):
        """Test error on mismatched actual length."""
        with pytest.raises(ValueError, match="actual"):
            UncertaintyAnalyzer(
                predicted_mean=sample_predictions["predicted_mean"],
                predicted_std=sample_predictions["predicted_std"],
                actual=sample_predictions["actual"][:40],
            )

    def test_mismatched_subject_ids(self, sample_predictions):
        """Test error on mismatched subject ID count."""
        with pytest.raises(ValueError, match="subject_ids"):
            UncertaintyAnalyzer(
                predicted_mean=sample_predictions["predicted_mean"],
                predicted_std=sample_predictions["predicted_std"],
                subject_ids=["a", "b", "c"],
            )

    def test_mismatched_covariates(self, sample_predictions):
        """Test error on mismatched covariate rows."""
        with pytest.raises(ValueError, match="covariates"):
            UncertaintyAnalyzer(
                predicted_mean=sample_predictions["predicted_mean"],
                predicted_std=sample_predictions["predicted_std"],
                covariates=pd.DataFrame({"x": [1, 2, 3]}),
            )

    def test_negative_std(self, sample_predictions):
        """Test error on negative std values."""
        bad_std = sample_predictions["predicted_std"].copy()
        bad_std[0] = -1.0

        with pytest.raises(ValueError, match="positive"):
            UncertaintyAnalyzer(
                predicted_mean=sample_predictions["predicted_mean"],
                predicted_std=bad_std,
            )

    def test_zero_std(self, sample_predictions):
        """Test error on zero std values."""
        bad_std = sample_predictions["predicted_std"].copy()
        bad_std[0] = 0.0

        with pytest.raises(ValueError, match="positive"):
            UncertaintyAnalyzer(
                predicted_mean=sample_predictions["predicted_mean"],
                predicted_std=bad_std,
            )


class TestUncertaintyAnalyzerAnalyze:
    """Test UncertaintyAnalyzer.analyze()."""

    def test_analyze_basic(self, sample_predictions):
        """Test basic analysis without actual values."""
        analyzer = UncertaintyAnalyzer(
            predicted_mean=sample_predictions["predicted_mean"],
            predicted_std=sample_predictions["predicted_std"],
        )

        result = analyzer.analyze()

        assert isinstance(result, UncertaintyAnalysisResult)
        assert isinstance(result.prediction_summary, pd.DataFrame)
        assert result.calibration is None  # No actual values
        assert result.correlates is None  # No covariates

    def test_analyze_with_actual(self, sample_predictions):
        """Test analysis with actual values (calibration)."""
        analyzer = UncertaintyAnalyzer(
            predicted_mean=sample_predictions["predicted_mean"],
            predicted_std=sample_predictions["predicted_std"],
            actual=sample_predictions["actual"],
        )

        result = analyzer.analyze()

        assert result.calibration is not None
        assert len(result.calibration) == 3  # 1σ, 2σ, 3σ

    def test_analyze_with_covariates(self, sample_predictions, sample_covariates):
        """Test analysis with covariates."""
        analyzer = UncertaintyAnalyzer(
            predicted_mean=sample_predictions["predicted_mean"],
            predicted_std=sample_predictions["predicted_std"],
            covariates=sample_covariates,
        )

        result = analyzer.analyze()

        assert result.correlates is not None
        assert len(result.correlates) > 0

    def test_analyze_full(self, sample_predictions, sample_covariates, sample_subject_ids):
        """Test full analysis with all data."""
        analyzer = UncertaintyAnalyzer(
            predicted_mean=sample_predictions["predicted_mean"],
            predicted_std=sample_predictions["predicted_std"],
            actual=sample_predictions["actual"],
            subject_ids=sample_subject_ids,
            covariates=sample_covariates,
        )

        result = analyzer.analyze()

        assert result.prediction_summary is not None
        assert result.calibration is not None
        assert result.correlates is not None


class TestPredictionSummary:
    """Test prediction summary DataFrame."""

    def test_prediction_summary_schema(self, sample_predictions, sample_subject_ids):
        """Test prediction summary schema."""
        analyzer = UncertaintyAnalyzer(
            predicted_mean=sample_predictions["predicted_mean"],
            predicted_std=sample_predictions["predicted_std"],
            actual=sample_predictions["actual"],
            subject_ids=sample_subject_ids,
        )

        result = analyzer.analyze()
        df = result.prediction_summary

        expected_cols = {"subject_id", "predicted_mean", "predicted_std", "actual", "residual", "z_score"}
        assert expected_cols == set(df.columns)
        assert len(df) == 50

    def test_prediction_summary_without_actual(self, sample_predictions):
        """Test prediction summary without actual values."""
        analyzer = UncertaintyAnalyzer(
            predicted_mean=sample_predictions["predicted_mean"],
            predicted_std=sample_predictions["predicted_std"],
        )

        result = analyzer.analyze()
        df = result.prediction_summary

        assert df["actual"].isna().all()
        assert df["residual"].isna().all()
        assert df["z_score"].isna().all()

    def test_prediction_summary_z_scores(self, sample_predictions):
        """Test z-scores are computed correctly."""
        analyzer = UncertaintyAnalyzer(
            predicted_mean=sample_predictions["predicted_mean"],
            predicted_std=sample_predictions["predicted_std"],
            actual=sample_predictions["actual"],
        )

        result = analyzer.analyze()
        df = result.prediction_summary

        # Verify z-score calculation
        expected_z = np.abs(sample_predictions["actual"] - sample_predictions["predicted_mean"]) / sample_predictions["predicted_std"]
        np.testing.assert_array_almost_equal(df["z_score"].values, expected_z)


class TestCalibration:
    """Test calibration analysis."""

    def test_calibration_schema(self, sample_predictions):
        """Test calibration DataFrame schema."""
        analyzer = UncertaintyAnalyzer(
            predicted_mean=sample_predictions["predicted_mean"],
            predicted_std=sample_predictions["predicted_std"],
            actual=sample_predictions["actual"],
        )

        result = analyzer.analyze()
        df = result.calibration

        expected_cols = {"level", "n_sigma", "expected_coverage", "observed_coverage", "calibration_error", "interpretation"}
        assert expected_cols == set(df.columns)
        assert len(df) == 3

    def test_calibration_levels(self, sample_predictions):
        """Test calibration levels match expected."""
        analyzer = UncertaintyAnalyzer(
            predicted_mean=sample_predictions["predicted_mean"],
            predicted_std=sample_predictions["predicted_std"],
            actual=sample_predictions["actual"],
        )

        result = analyzer.analyze()
        df = result.calibration

        assert set(df["level"]) == {"1_sigma", "2_sigma", "3_sigma"}
        assert df[df["level"] == "1_sigma"]["expected_coverage"].values[0] == pytest.approx(0.6827, abs=0.001)
        assert df[df["level"] == "2_sigma"]["expected_coverage"].values[0] == pytest.approx(0.9545, abs=0.001)
        assert df[df["level"] == "3_sigma"]["expected_coverage"].values[0] == pytest.approx(0.9973, abs=0.001)

    def test_calibration_interpretation(self):
        """Test calibration interpretation labels."""
        # Create underconfident model (too wide predictions)
        np.random.seed(42)
        actual = np.random.randn(100)
        predicted_mean = actual + np.random.randn(100) * 0.1  # Small errors
        predicted_std = np.ones(100) * 2.0  # Large std

        analyzer = UncertaintyAnalyzer(
            predicted_mean=predicted_mean,
            predicted_std=predicted_std,
            actual=actual,
        )

        result = analyzer.analyze()

        # Should be mostly underconfident (coverage > expected)
        assert any(result.calibration["interpretation"] == "underconfident")


class TestCorrelates:
    """Test uncertainty correlates analysis."""

    def test_correlates_schema(self, sample_predictions, sample_covariates):
        """Test correlates DataFrame schema."""
        analyzer = UncertaintyAnalyzer(
            predicted_mean=sample_predictions["predicted_mean"],
            predicted_std=sample_predictions["predicted_std"],
            covariates=sample_covariates,
        )

        result = analyzer.analyze()
        df = result.correlates

        expected_cols = {"covariate", "correlation", "p_value", "significant", "interpretation"}
        assert expected_cols == set(df.columns)

    def test_correlates_numeric_only(self, sample_predictions):
        """Test that only numeric columns are analyzed."""
        covariates = pd.DataFrame({
            "numeric_col": np.random.rand(50),
            "string_col": ["a"] * 50,
            "another_numeric": np.random.rand(50),
        })

        analyzer = UncertaintyAnalyzer(
            predicted_mean=sample_predictions["predicted_mean"],
            predicted_std=sample_predictions["predicted_std"],
            covariates=covariates,
        )

        result = analyzer.analyze()

        # Only numeric columns should be in correlates
        assert "numeric_col" in result.correlates["covariate"].values
        assert "another_numeric" in result.correlates["covariate"].values
        assert "string_col" not in result.correlates["covariate"].values

    def test_correlates_sorted_by_pvalue(self, sample_predictions, sample_covariates):
        """Test correlates are sorted by p-value."""
        analyzer = UncertaintyAnalyzer(
            predicted_mean=sample_predictions["predicted_mean"],
            predicted_std=sample_predictions["predicted_std"],
            covariates=sample_covariates,
        )

        result = analyzer.analyze()

        # Check sorted ascending by p-value
        pvalues = result.correlates["p_value"].values
        assert (pvalues[:-1] <= pvalues[1:]).all()


class TestUncertaintyAnalyzerSave:
    """Test UncertaintyAnalyzer.save()."""

    def test_save_all_outputs(self, tmp_path, sample_predictions, sample_covariates):
        """Test saving all outputs."""
        analyzer = UncertaintyAnalyzer(
            predicted_mean=sample_predictions["predicted_mean"],
            predicted_std=sample_predictions["predicted_std"],
            actual=sample_predictions["actual"],
            covariates=sample_covariates,
        )

        result = analyzer.analyze()
        saved_files = analyzer.save(result, tmp_path)

        assert "prediction_summary_parquet" in saved_files
        assert "prediction_summary_csv" in saved_files
        assert "calibration_parquet" in saved_files
        assert "calibration_csv" in saved_files
        assert "correlates_parquet" in saved_files
        assert "correlates_csv" in saved_files

    def test_save_without_optional(self, tmp_path, sample_predictions):
        """Test saving without calibration/correlates."""
        analyzer = UncertaintyAnalyzer(
            predicted_mean=sample_predictions["predicted_mean"],
            predicted_std=sample_predictions["predicted_std"],
        )

        result = analyzer.analyze()
        saved_files = analyzer.save(result, tmp_path)

        assert "prediction_summary_parquet" in saved_files
        assert "calibration_parquet" not in saved_files
        assert "correlates_parquet" not in saved_files


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestComputeUncertaintyAnalysis:
    """Test compute_uncertainty_analysis convenience function."""

    def test_compute_basic(self, sample_predictions):
        """Test basic computation."""
        result = compute_uncertainty_analysis(
            predicted_mean=sample_predictions["predicted_mean"],
            predicted_std=sample_predictions["predicted_std"],
        )

        assert isinstance(result, UncertaintyAnalysisResult)

    def test_compute_with_save(self, tmp_path, sample_predictions):
        """Test computation with saving."""
        result = compute_uncertainty_analysis(
            predicted_mean=sample_predictions["predicted_mean"],
            predicted_std=sample_predictions["predicted_std"],
            actual=sample_predictions["actual"],
            output_dir=tmp_path,
        )

        assert isinstance(result, UncertaintyAnalysisResult)
        assert (tmp_path / "prediction_uncertainty.csv").exists()
        assert (tmp_path / "calibration_summary.csv").exists()


class TestComputeEceRegression:
    """Test ECE computation."""

    def test_ece_well_calibrated(self):
        """Test ECE for well-calibrated model."""
        np.random.seed(42)
        actual = np.random.randn(1000)
        predicted_mean = actual + np.random.randn(1000) * 0.5
        predicted_std = np.ones(1000) * 0.5

        ece = compute_ece_regression(
            predicted_mean=predicted_mean,
            predicted_std=predicted_std,
            actual=actual,
        )

        # ECE should be relatively low for well-calibrated model
        assert 0 <= ece <= 1
        assert ece < 0.3  # Rough threshold

    def test_ece_poorly_calibrated(self):
        """Test ECE for poorly calibrated model."""
        np.random.seed(42)
        actual = np.random.randn(1000)
        predicted_mean = actual + np.random.randn(1000) * 2.0  # Large errors
        predicted_std = np.ones(1000) * 0.1  # Overconfident

        ece = compute_ece_regression(
            predicted_mean=predicted_mean,
            predicted_std=predicted_std,
            actual=actual,
        )

        # ECE should be higher for poorly calibrated model
        assert 0 <= ece <= 1
        assert ece > 0.1  # Should have notable calibration error

    def test_ece_n_bins(self):
        """Test ECE with different bin counts."""
        np.random.seed(42)
        actual = np.random.randn(100)
        predicted_mean = actual + np.random.randn(100) * 0.5
        predicted_std = np.abs(np.random.randn(100)) * 0.5 + 0.1

        ece_5 = compute_ece_regression(
            predicted_mean=predicted_mean,
            predicted_std=predicted_std,
            actual=actual,
            n_bins=5,
        )

        ece_20 = compute_ece_regression(
            predicted_mean=predicted_mean,
            predicted_std=predicted_std,
            actual=actual,
            n_bins=20,
        )

        # Both should be valid
        assert 0 <= ece_5 <= 1
        assert 0 <= ece_20 <= 1


# =============================================================================
# Property-Based Tests
# =============================================================================


class TestUncertaintyAnalysisProperties:
    """Property-based tests using Hypothesis."""

    @given(n_subjects=st.integers(min_value=10, max_value=100))
    @settings(max_examples=15)
    def test_prediction_summary_length(self, n_subjects):
        """Test prediction summary has correct length."""
        predicted_mean = np.random.randn(n_subjects)
        predicted_std = np.abs(np.random.randn(n_subjects)) + 0.1

        analyzer = UncertaintyAnalyzer(
            predicted_mean=predicted_mean,
            predicted_std=predicted_std,
        )

        result = analyzer.analyze()
        assert len(result.prediction_summary) == n_subjects

    @given(n_subjects=st.integers(min_value=20, max_value=100))
    @settings(max_examples=15)
    def test_z_scores_nonnegative(self, n_subjects):
        """Test z-scores are always non-negative."""
        np.random.seed(42)
        predicted_mean = np.random.randn(n_subjects)
        predicted_std = np.abs(np.random.randn(n_subjects)) + 0.1
        actual = np.random.randn(n_subjects)

        analyzer = UncertaintyAnalyzer(
            predicted_mean=predicted_mean,
            predicted_std=predicted_std,
            actual=actual,
        )

        result = analyzer.analyze()
        assert (result.prediction_summary["z_score"] >= 0).all()

    @given(n_subjects=st.integers(min_value=20, max_value=100))
    @settings(max_examples=15)
    def test_calibration_coverage_bounds(self, n_subjects):
        """Test observed coverage is between 0 and 1."""
        np.random.seed(42)
        predicted_mean = np.random.randn(n_subjects)
        predicted_std = np.abs(np.random.randn(n_subjects)) + 0.1
        actual = np.random.randn(n_subjects)

        analyzer = UncertaintyAnalyzer(
            predicted_mean=predicted_mean,
            predicted_std=predicted_std,
            actual=actual,
        )

        result = analyzer.analyze()
        assert (result.calibration["observed_coverage"] >= 0).all()
        assert (result.calibration["observed_coverage"] <= 1).all()


# =============================================================================
# Schema Validation Tests
# =============================================================================


class TestUncertaintyAnalysisSchema:
    """Test output DataFrame schemas."""

    def test_prediction_summary_dtypes(self, sample_predictions, sample_subject_ids):
        """Test prediction summary column types."""
        analyzer = UncertaintyAnalyzer(
            predicted_mean=sample_predictions["predicted_mean"],
            predicted_std=sample_predictions["predicted_std"],
            actual=sample_predictions["actual"],
            subject_ids=sample_subject_ids,
        )

        result = analyzer.analyze()
        df = result.prediction_summary

        assert df["subject_id"].dtype == object
        assert np.issubdtype(df["predicted_mean"].dtype, np.floating)
        assert np.issubdtype(df["predicted_std"].dtype, np.floating)
        assert np.issubdtype(df["actual"].dtype, np.floating)
        assert np.issubdtype(df["residual"].dtype, np.floating)
        assert np.issubdtype(df["z_score"].dtype, np.floating)

    def test_calibration_dtypes(self, sample_predictions):
        """Test calibration column types."""
        analyzer = UncertaintyAnalyzer(
            predicted_mean=sample_predictions["predicted_mean"],
            predicted_std=sample_predictions["predicted_std"],
            actual=sample_predictions["actual"],
        )

        result = analyzer.analyze()
        df = result.calibration

        assert df["level"].dtype == object
        assert np.issubdtype(df["n_sigma"].dtype, np.integer)
        assert np.issubdtype(df["expected_coverage"].dtype, np.floating)
        assert np.issubdtype(df["observed_coverage"].dtype, np.floating)
        assert np.issubdtype(df["calibration_error"].dtype, np.floating)
        assert df["interpretation"].dtype == object

    def test_correlates_dtypes(self, sample_predictions, sample_covariates):
        """Test correlates column types."""
        analyzer = UncertaintyAnalyzer(
            predicted_mean=sample_predictions["predicted_mean"],
            predicted_std=sample_predictions["predicted_std"],
            covariates=sample_covariates,
        )

        result = analyzer.analyze()
        df = result.correlates

        assert df["covariate"].dtype == object
        assert np.issubdtype(df["correlation"].dtype, np.floating)
        assert np.issubdtype(df["p_value"].dtype, np.floating)
        assert df["significant"].dtype == bool
        assert df["interpretation"].dtype == object


# =============================================================================
# Round-Trip Tests
# =============================================================================


class TestUncertaintyAnalysisRoundTrip:
    """Test save and load round-trips."""

    def test_prediction_summary_roundtrip(self, tmp_path, sample_predictions, sample_subject_ids):
        """Test prediction summary round-trip."""
        analyzer = UncertaintyAnalyzer(
            predicted_mean=sample_predictions["predicted_mean"],
            predicted_std=sample_predictions["predicted_std"],
            actual=sample_predictions["actual"],
            subject_ids=sample_subject_ids,
        )

        result = analyzer.analyze()
        saved_files = analyzer.save(result, tmp_path, formats=["parquet"])

        loaded = pd.read_parquet(saved_files["prediction_summary_parquet"])
        pd.testing.assert_frame_equal(
            result.prediction_summary.reset_index(drop=True),
            loaded.reset_index(drop=True),
        )

    def test_calibration_roundtrip(self, tmp_path, sample_predictions):
        """Test calibration round-trip."""
        analyzer = UncertaintyAnalyzer(
            predicted_mean=sample_predictions["predicted_mean"],
            predicted_std=sample_predictions["predicted_std"],
            actual=sample_predictions["actual"],
        )

        result = analyzer.analyze()
        saved_files = analyzer.save(result, tmp_path, formats=["parquet"])

        loaded = pd.read_parquet(saved_files["calibration_parquet"])
        pd.testing.assert_frame_equal(
            result.calibration.reset_index(drop=True),
            loaded.reset_index(drop=True),
        )


# =============================================================================
# Edge Cases
# =============================================================================


class TestUncertaintyAnalysisEdgeCases:
    """Test edge cases."""

    def test_constant_covariates_skipped(self, sample_predictions):
        """Test that constant covariates are skipped."""
        covariates = pd.DataFrame({
            "varying": np.random.rand(50),
            "constant": np.ones(50),
        })

        analyzer = UncertaintyAnalyzer(
            predicted_mean=sample_predictions["predicted_mean"],
            predicted_std=sample_predictions["predicted_std"],
            covariates=covariates,
        )

        result = analyzer.analyze()

        # Constant column should be excluded
        assert "varying" in result.correlates["covariate"].values
        assert "constant" not in result.correlates["covariate"].values

    def test_metadata_fields(self, sample_predictions, sample_covariates):
        """Test metadata contains expected fields."""
        analyzer = UncertaintyAnalyzer(
            predicted_mean=sample_predictions["predicted_mean"],
            predicted_std=sample_predictions["predicted_std"],
            actual=sample_predictions["actual"],
            covariates=sample_covariates,
        )

        result = analyzer.analyze()

        assert "n_subjects" in result.metadata
        assert "has_actual" in result.metadata
        assert "has_covariates" in result.metadata
        assert "mean_std" in result.metadata
        assert "std_std" in result.metadata

        assert result.metadata["n_subjects"] == 50
        assert result.metadata["has_actual"] is True
        assert result.metadata["has_covariates"] is True


# ============================================================================
# Phase 6 Review Round 8 — M2: NaN-safe spearmanr
# ============================================================================


class TestSpearmanrNaNHandling:
    """Tests for NaN-safe Spearman correlation in uncertainty correlates."""

    def test_nan_in_covariates_does_not_crash(self):
        """spearmanr with NaN covariate values should not crash or produce NaN."""
        np.random.seed(42)
        n = 50
        predicted_std = np.random.rand(n)
        covariates = pd.DataFrame({
            "cell_count": np.random.rand(n),
            "with_nans": np.where(np.arange(n) < 40, np.random.rand(n), np.nan),
        })

        analyzer = UncertaintyAnalyzer(
            predicted_mean=np.random.rand(n),
            predicted_std=predicted_std,
            covariates=covariates,
        )
        result = analyzer.analyze()
        if result.correlates is not None and len(result.correlates) > 0:
            assert not result.correlates["correlation"].isna().any()

    def test_few_valid_values_skips_correlation(self):
        """Covariate with fewer than 3 valid values should be skipped."""
        np.random.seed(42)
        n = 10
        predicted_std = np.random.rand(n)
        covariates = pd.DataFrame({
            "mostly_nan": np.full(n, np.nan),
        })
        covariates.loc[0, "mostly_nan"] = 1.0
        covariates.loc[1, "mostly_nan"] = 2.0

        analyzer = UncertaintyAnalyzer(
            predicted_mean=np.random.rand(n),
            predicted_std=predicted_std,
            covariates=covariates,
        )
        result = analyzer.analyze()
        # "mostly_nan" should be absent from correlates (too few valid values)
        if result.correlates is not None and len(result.correlates) > 0:
            assert "mostly_nan" not in result.correlates["covariate"].values
