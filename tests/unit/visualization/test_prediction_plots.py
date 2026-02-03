"""Tests for prediction visualization plots."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for testing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from src.visualization.prediction_plots import (
    plot_predicted_vs_actual,
    plot_calibration_curve,
    plot_residuals,
    plot_uncertainty_vs_error,
    plot_uncertainty_correlates,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_predictions():
    """Sample prediction data."""
    np.random.seed(42)
    n = 50
    actual = np.random.randn(n) * 2 + 5
    predicted_mean = actual + np.random.randn(n) * 0.5
    predicted_std = np.abs(np.random.randn(n)) * 0.3 + 0.2
    return {
        "predicted_mean": predicted_mean,
        "predicted_std": predicted_std,
        "actual": actual,
    }


@pytest.fixture
def sample_calibration_df():
    """Sample calibration DataFrame."""
    return pd.DataFrame({
        "level": ["1sigma", "2sigma", "3sigma"],
        "expected_coverage": [0.6827, 0.9545, 0.9973],
        "observed_coverage": [0.70, 0.92, 0.99],
        "calibration_error": [0.02, -0.03, -0.01],
    })


@pytest.fixture
def sample_correlates_df():
    """Sample uncertainty correlates DataFrame."""
    return pd.DataFrame({
        "covariate": ["cell_count", "pathology", "age", "n_regions", "education"],
        "correlation": [0.35, 0.22, -0.15, 0.42, -0.08],
        "p_value": [0.01, 0.03, 0.12, 0.002, 0.45],
        "significant": [True, True, False, True, False],
    })


# =============================================================================
# plot_predicted_vs_actual Tests
# =============================================================================


class TestPlotPredictedVsActual:
    """Test plot_predicted_vs_actual function."""

    def test_basic_plot(self, sample_predictions):
        """Test basic scatter plot creation."""
        fig = plot_predicted_vs_actual(
            predicted_mean=sample_predictions["predicted_mean"],
            actual=sample_predictions["actual"],
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_uncertainty(self, sample_predictions):
        """Test with uncertainty coloring."""
        fig = plot_predicted_vs_actual(
            predicted_mean=sample_predictions["predicted_mean"],
            actual=sample_predictions["actual"],
            predicted_std=sample_predictions["predicted_std"],
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_figsize(self, sample_predictions):
        """Test custom figure size."""
        fig = plot_predicted_vs_actual(
            predicted_mean=sample_predictions["predicted_mean"],
            actual=sample_predictions["actual"],
            figsize=(10, 10),
        )

        assert fig.get_figwidth() == 10
        assert fig.get_figheight() == 10
        plt.close(fig)

    def test_custom_title(self, sample_predictions):
        """Test custom title."""
        title = "Custom Prediction Plot"
        fig = plot_predicted_vs_actual(
            predicted_mean=sample_predictions["predicted_mean"],
            actual=sample_predictions["actual"],
            title=title,
        )

        ax = fig.get_axes()[0]
        assert ax.get_title() == title
        plt.close(fig)

    def test_save_path(self, tmp_path, sample_predictions):
        """Test saving figure."""
        save_path = tmp_path / "pred_vs_actual.png"
        fig = plot_predicted_vs_actual(
            predicted_mean=sample_predictions["predicted_mean"],
            actual=sample_predictions["actual"],
            save_path=save_path,
        )

        assert save_path.exists()
        plt.close(fig)

    def test_includes_metrics(self, sample_predictions):
        """Test that RMSE, MAE, R² are annotated."""
        fig = plot_predicted_vs_actual(
            predicted_mean=sample_predictions["predicted_mean"],
            actual=sample_predictions["actual"],
        )

        # Get text from figure
        ax = fig.get_axes()[0]
        texts = [t.get_text() for t in ax.texts]
        text_content = " ".join(texts)

        assert "RMSE" in text_content
        assert "MAE" in text_content
        assert "R²" in text_content
        plt.close(fig)

    def test_identity_and_regression_lines(self, sample_predictions):
        """Test that identity and regression lines are present."""
        fig = plot_predicted_vs_actual(
            predicted_mean=sample_predictions["predicted_mean"],
            actual=sample_predictions["actual"],
        )

        ax = fig.get_axes()[0]
        # Should have multiple lines (identity and regression)
        assert len(ax.lines) >= 2
        plt.close(fig)


# =============================================================================
# plot_calibration_curve Tests
# =============================================================================


class TestPlotCalibrationCurve:
    """Test plot_calibration_curve function."""

    def test_basic_plot(self, sample_calibration_df):
        """Test basic bar chart creation."""
        fig = plot_calibration_curve(sample_calibration_df)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_calibration_error(self, sample_calibration_df):
        """Test with calibration_error annotations."""
        fig = plot_calibration_curve(sample_calibration_df)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_without_calibration_error(self, sample_calibration_df):
        """Test without calibration_error column."""
        df = sample_calibration_df.drop(columns=["calibration_error"])
        fig = plot_calibration_curve(df)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_figsize(self, sample_calibration_df):
        """Test custom figure size."""
        fig = plot_calibration_curve(
            sample_calibration_df,
            figsize=(10, 8),
        )

        assert fig.get_figwidth() == 10
        plt.close(fig)

    def test_custom_title(self, sample_calibration_df):
        """Test custom title."""
        title = "Custom Calibration Title"
        fig = plot_calibration_curve(
            sample_calibration_df,
            title=title,
        )

        ax = fig.get_axes()[0]
        assert ax.get_title() == title
        plt.close(fig)

    def test_save_path(self, tmp_path, sample_calibration_df):
        """Test saving figure."""
        save_path = tmp_path / "calibration.png"
        fig = plot_calibration_curve(
            sample_calibration_df,
            save_path=save_path,
        )

        assert save_path.exists()
        plt.close(fig)


# =============================================================================
# plot_residuals Tests
# =============================================================================


class TestPlotResiduals:
    """Test plot_residuals function."""

    def test_basic_plot(self, sample_predictions):
        """Test basic residual plot creation."""
        fig = plot_residuals(
            predicted_mean=sample_predictions["predicted_mean"],
            actual=sample_predictions["actual"],
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_uncertainty(self, sample_predictions):
        """Test with uncertainty coloring."""
        fig = plot_residuals(
            predicted_mean=sample_predictions["predicted_mean"],
            actual=sample_predictions["actual"],
            predicted_std=sample_predictions["predicted_std"],
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_two_subplots(self, sample_predictions):
        """Test that figure has two subplots."""
        fig = plot_residuals(
            predicted_mean=sample_predictions["predicted_mean"],
            actual=sample_predictions["actual"],
        )

        axes = fig.get_axes()
        # Should have 2 main axes (scatter and histogram)
        # Plus potentially a colorbar
        assert len(axes) >= 2
        plt.close(fig)

    def test_custom_figsize(self, sample_predictions):
        """Test custom figure size."""
        fig = plot_residuals(
            predicted_mean=sample_predictions["predicted_mean"],
            actual=sample_predictions["actual"],
            figsize=(14, 6),
        )

        assert fig.get_figwidth() == 14
        plt.close(fig)

    def test_custom_title(self, sample_predictions):
        """Test custom title."""
        title = "Custom Residual Title"
        fig = plot_residuals(
            predicted_mean=sample_predictions["predicted_mean"],
            actual=sample_predictions["actual"],
            title=title,
        )

        # Title should be the suptitle
        assert title in fig._suptitle.get_text()
        plt.close(fig)

    def test_save_path(self, tmp_path, sample_predictions):
        """Test saving figure."""
        save_path = tmp_path / "residuals.png"
        fig = plot_residuals(
            predicted_mean=sample_predictions["predicted_mean"],
            actual=sample_predictions["actual"],
            save_path=save_path,
        )

        assert save_path.exists()
        plt.close(fig)


# =============================================================================
# plot_uncertainty_vs_error Tests
# =============================================================================


class TestPlotUncertaintyVsError:
    """Test plot_uncertainty_vs_error function."""

    def test_basic_plot(self, sample_predictions):
        """Test basic scatter plot creation."""
        fig = plot_uncertainty_vs_error(
            predicted_mean=sample_predictions["predicted_mean"],
            actual=sample_predictions["actual"],
            predicted_std=sample_predictions["predicted_std"],
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_figsize(self, sample_predictions):
        """Test custom figure size."""
        fig = plot_uncertainty_vs_error(
            predicted_mean=sample_predictions["predicted_mean"],
            actual=sample_predictions["actual"],
            predicted_std=sample_predictions["predicted_std"],
            figsize=(10, 8),
        )

        assert fig.get_figwidth() == 10
        plt.close(fig)

    def test_custom_title(self, sample_predictions):
        """Test custom title."""
        title = "Custom Uncertainty vs Error Title"
        fig = plot_uncertainty_vs_error(
            predicted_mean=sample_predictions["predicted_mean"],
            actual=sample_predictions["actual"],
            predicted_std=sample_predictions["predicted_std"],
            title=title,
        )

        ax = fig.get_axes()[0]
        assert ax.get_title() == title
        plt.close(fig)

    def test_save_path(self, tmp_path, sample_predictions):
        """Test saving figure."""
        save_path = tmp_path / "uncertainty_vs_error.png"
        fig = plot_uncertainty_vs_error(
            predicted_mean=sample_predictions["predicted_mean"],
            actual=sample_predictions["actual"],
            predicted_std=sample_predictions["predicted_std"],
            save_path=save_path,
        )

        assert save_path.exists()
        plt.close(fig)

    def test_includes_regression_info(self, sample_predictions):
        """Test that regression line info is in legend."""
        fig = plot_uncertainty_vs_error(
            predicted_mean=sample_predictions["predicted_mean"],
            actual=sample_predictions["actual"],
            predicted_std=sample_predictions["predicted_std"],
        )

        ax = fig.get_axes()[0]
        legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
        legend_str = " ".join(legend_texts)

        assert "Fit" in legend_str or "r=" in legend_str
        plt.close(fig)


# =============================================================================
# plot_uncertainty_correlates Tests
# =============================================================================


class TestPlotUncertaintyCorrelates:
    """Test plot_uncertainty_correlates function."""

    def test_basic_plot(self, sample_correlates_df):
        """Test basic bar chart creation."""
        fig = plot_uncertainty_correlates(sample_correlates_df)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_figsize(self, sample_correlates_df):
        """Test custom figure size."""
        fig = plot_uncertainty_correlates(
            sample_correlates_df,
            figsize=(12, 8),
        )

        assert fig.get_figwidth() == 12
        plt.close(fig)

    def test_custom_title(self, sample_correlates_df):
        """Test custom title."""
        title = "Custom Correlates Title"
        fig = plot_uncertainty_correlates(
            sample_correlates_df,
            title=title,
        )

        ax = fig.get_axes()[0]
        assert ax.get_title() == title
        plt.close(fig)

    def test_save_path(self, tmp_path, sample_correlates_df):
        """Test saving figure."""
        save_path = tmp_path / "correlates.png"
        fig = plot_uncertainty_correlates(
            sample_correlates_df,
            save_path=save_path,
        )

        assert save_path.exists()
        plt.close(fig)

    def test_handles_all_significant(self):
        """Test with all correlates significant."""
        df = pd.DataFrame({
            "covariate": ["a", "b", "c"],
            "correlation": [0.5, -0.3, 0.4],
            "p_value": [0.001, 0.01, 0.02],
            "significant": [True, True, True],
        })

        fig = plot_uncertainty_correlates(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_handles_none_significant(self):
        """Test with no correlates significant."""
        df = pd.DataFrame({
            "covariate": ["a", "b", "c"],
            "correlation": [0.1, -0.05, 0.08],
            "p_value": [0.5, 0.6, 0.7],
            "significant": [False, False, False],
        })

        fig = plot_uncertainty_correlates(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# =============================================================================
# Property-Based Tests
# =============================================================================


class TestPredictionPlotsProperties:
    """Property-based tests for prediction plots."""

    @given(n_subjects=st.integers(min_value=10, max_value=100))
    @settings(max_examples=10)
    def test_pred_vs_actual_various_sizes(self, n_subjects):
        """Test predicted vs actual plot with various sample sizes."""
        np.random.seed(42)
        actual = np.random.randn(n_subjects)
        predicted_mean = actual + np.random.randn(n_subjects) * 0.5

        fig = plot_predicted_vs_actual(
            predicted_mean=predicted_mean,
            actual=actual,
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    @given(n_subjects=st.integers(min_value=10, max_value=100))
    @settings(max_examples=10)
    def test_residuals_various_sizes(self, n_subjects):
        """Test residuals plot with various sample sizes."""
        np.random.seed(42)
        actual = np.random.randn(n_subjects)
        predicted_mean = actual + np.random.randn(n_subjects) * 0.5

        fig = plot_residuals(
            predicted_mean=predicted_mean,
            actual=actual,
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    @given(n_covariates=st.integers(min_value=1, max_value=10))
    @settings(max_examples=10)
    def test_correlates_various_counts(self, n_covariates):
        """Test correlates plot with various covariate counts."""
        df = pd.DataFrame({
            "covariate": [f"cov_{i}" for i in range(n_covariates)],
            "correlation": np.random.randn(n_covariates) * 0.5,
            "p_value": np.random.rand(n_covariates),
            "significant": np.random.rand(n_covariates) < 0.5,
        })

        fig = plot_uncertainty_correlates(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# =============================================================================
# Edge Cases
# =============================================================================


class TestPredictionPlotsEdgeCases:
    """Test edge cases for prediction plots."""

    def test_perfect_predictions(self):
        """Test with perfect predictions (no error)."""
        np.random.seed(42)
        actual = np.random.randn(50)
        predicted_mean = actual.copy()

        fig = plot_predicted_vs_actual(
            predicted_mean=predicted_mean,
            actual=actual,
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_high_uncertainty(self):
        """Test with very high predicted uncertainty."""
        np.random.seed(42)
        actual = np.random.randn(50)
        predicted_mean = actual + np.random.randn(50) * 0.5
        # Very high uncertainty but with some variance to allow regression
        predicted_std = np.abs(np.random.randn(50)) * 2 + 8  # High but varying

        fig = plot_uncertainty_vs_error(
            predicted_mean=predicted_mean,
            actual=actual,
            predicted_std=predicted_std,
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_single_calibration_level(self):
        """Test calibration curve with single level."""
        df = pd.DataFrame({
            "level": ["1sigma"],
            "expected_coverage": [0.6827],
            "observed_coverage": [0.70],
        })

        fig = plot_calibration_curve(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_single_covariate(self):
        """Test correlates with single covariate."""
        df = pd.DataFrame({
            "covariate": ["cell_count"],
            "correlation": [0.35],
            "p_value": [0.01],
            "significant": [True],
        })

        fig = plot_uncertainty_correlates(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_very_small_sample(self):
        """Test with very small sample size."""
        np.random.seed(42)
        actual = np.random.randn(5)
        predicted_mean = actual + np.random.randn(5) * 0.5

        fig = plot_predicted_vs_actual(
            predicted_mean=predicted_mean,
            actual=actual,
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# =============================================================================
# Cleanup
# =============================================================================


@pytest.fixture(autouse=True)
def cleanup():
    """Cleanup matplotlib figures after each test."""
    yield
    plt.close("all")
