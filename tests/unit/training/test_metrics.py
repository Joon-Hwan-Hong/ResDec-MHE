"""
Tests for training metrics.

Tests ResilienceMetrics which computes:
- R², RMSE, MAE, Pearson r, Spearman ρ
- Mean uncertainty (mean_std)
- Calibration error at 1σ, 2σ, 3σ levels
"""

import pytest
import torch
import numpy as np


class TestResilienceMetrics:
    """Tests for ResilienceMetrics class."""

    # ─────────────────────────────────────────────────────────────────
    # Instantiation
    # ─────────────────────────────────────────────────────────────────

    def test_instantiation(self):
        """ResilienceMetrics instantiates without error."""
        from src.training.metrics import ResilienceMetrics
        metrics = ResilienceMetrics()
        assert metrics is not None

    # ─────────────────────────────────────────────────────────────────
    # Compute returns correct keys
    # ─────────────────────────────────────────────────────────────────

    def test_compute_returns_expected_keys(self):
        """compute() returns dict with all expected metric keys."""
        from src.training.metrics import ResilienceMetrics
        metrics = ResilienceMetrics()
        mean = torch.randn(16, 1)
        std = torch.rand(16, 1) + 0.1
        target = torch.randn(16, 1)
        result = metrics.compute(mean, std, target)
        expected_keys = {"r2", "rmse", "mae", "pearson_r", "spearman_rho", "mean_std", "calibration_error", "crps"}
        assert set(result.keys()) == expected_keys

    def test_compute_without_std(self):
        """compute() works without std (for deterministic head), skipping uncertainty metrics."""
        from src.training.metrics import ResilienceMetrics
        metrics = ResilienceMetrics()
        mean = torch.randn(16, 1)
        target = torch.randn(16, 1)
        result = metrics.compute(mean, target=target)
        # Should still have r2, rmse, mae, pearson_r, spearman_rho
        assert "r2" in result
        assert "rmse" in result
        assert "mae" in result

    # ─────────────────────────────────────────────────────────────────
    # Individual metric correctness
    # ─────────────────────────────────────────────────────────────────

    def test_r2_perfect_prediction(self):
        """R² = 1.0 for perfect prediction."""
        from src.training.metrics import ResilienceMetrics
        metrics = ResilienceMetrics()
        target = torch.randn(32, 1)
        result = metrics.compute(target.clone(), torch.ones(32, 1), target)
        assert abs(result["r2"] - 1.0) < 1e-5

    def test_r2_mean_prediction(self):
        """R² ≈ 0.0 when predicting the mean."""
        from src.training.metrics import ResilienceMetrics
        metrics = ResilienceMetrics()
        target = torch.randn(100, 1)
        mean_pred = torch.full_like(target, target.mean().item())
        result = metrics.compute(mean_pred, torch.ones(100, 1), target)
        assert abs(result["r2"]) < 0.05  # Should be near 0

    def test_rmse_perfect_prediction(self):
        """RMSE = 0 for perfect prediction."""
        from src.training.metrics import ResilienceMetrics
        metrics = ResilienceMetrics()
        target = torch.randn(32, 1)
        result = metrics.compute(target.clone(), torch.ones(32, 1), target)
        assert abs(result["rmse"]) < 1e-5

    def test_mae_perfect_prediction(self):
        """MAE = 0 for perfect prediction."""
        from src.training.metrics import ResilienceMetrics
        metrics = ResilienceMetrics()
        target = torch.randn(32, 1)
        result = metrics.compute(target.clone(), torch.ones(32, 1), target)
        assert abs(result["mae"]) < 1e-5

    def test_pearson_r_perfect_correlation(self):
        """Pearson r = 1.0 for perfect positive correlation."""
        from src.training.metrics import ResilienceMetrics
        metrics = ResilienceMetrics()
        target = torch.arange(50, dtype=torch.float32).unsqueeze(1)
        # Perfect linear relationship
        mean = target * 2.0 + 1.0
        result = metrics.compute(mean, torch.ones(50, 1), target)
        assert abs(result["pearson_r"] - 1.0) < 1e-4

    def test_spearman_rho_perfect_rank_correlation(self):
        """Spearman ρ = 1.0 for perfect rank correlation."""
        from src.training.metrics import ResilienceMetrics
        metrics = ResilienceMetrics()
        target = torch.arange(50, dtype=torch.float32).unsqueeze(1)
        # Monotonic (not necessarily linear) relationship
        mean = target ** 2
        result = metrics.compute(mean, torch.ones(50, 1), target)
        assert abs(result["spearman_rho"] - 1.0) < 1e-4

    def test_mean_std_computed_correctly(self):
        """mean_std is the mean of the std values."""
        from src.training.metrics import ResilienceMetrics
        metrics = ResilienceMetrics()
        std = torch.tensor([[0.1], [0.2], [0.3], [0.4]])
        target = torch.randn(4, 1)
        result = metrics.compute(torch.randn(4, 1), std, target)
        assert abs(result["mean_std"] - 0.25) < 1e-5

    # ─────────────────────────────────────────────────────────────────
    # Calibration Error
    # ─────────────────────────────────────────────────────────────────

    def test_calibration_error_well_calibrated(self):
        """Calibration error near 0 for well-calibrated Gaussian predictions."""
        from src.training.metrics import ResilienceMetrics
        metrics = ResilienceMetrics()
        # Generate well-calibrated data: target = mean + std * N(0,1)
        torch.manual_seed(123)
        n = 1000
        mean = torch.zeros(n, 1)
        std = torch.ones(n, 1)
        noise = torch.randn(n, 1)
        target = mean + std * noise
        result = metrics.compute(mean, std, target)
        # Calibration error should be small for well-calibrated predictions
        assert abs(result["calibration_error"]) < 0.1

    def test_calibration_error_overconfident(self):
        """Calibration error is negative when model is overconfident (std too small)."""
        from src.training.metrics import ResilienceMetrics
        metrics = ResilienceMetrics()
        torch.manual_seed(123)
        n = 1000
        mean = torch.zeros(n, 1)
        # True noise is much larger than predicted std
        true_std = 5.0
        predicted_std = torch.ones(n, 1) * 0.1  # Way too confident
        target = mean + true_std * torch.randn(n, 1)
        result = metrics.compute(mean, predicted_std, target)
        # Overconfident: observed fraction outside 1σ >> 31.7%
        assert result["calibration_error"] < -0.1

    # ─────────────────────────────────────────────────────────────────
    # Edge Cases
    # ─────────────────────────────────────────────────────────────────

    def test_metrics_small_sample(self):
        """Metrics work with small sample (n=3)."""
        from src.training.metrics import ResilienceMetrics
        metrics = ResilienceMetrics()
        result = metrics.compute(
            torch.randn(3, 1),
            torch.rand(3, 1) + 0.1,
            torch.randn(3, 1),
        )
        assert all(np.isfinite(v) for v in result.values())

    def test_all_values_are_python_floats(self):
        """All returned values are Python floats (not tensors)."""
        from src.training.metrics import ResilienceMetrics
        metrics = ResilienceMetrics()
        result = metrics.compute(
            torch.randn(16, 1),
            torch.rand(16, 1) + 0.1,
            torch.randn(16, 1),
        )
        for key, val in result.items():
            assert isinstance(val, float), f"{key} is {type(val)}, expected float"


class TestCRPSMetric:
    """Tests for CRPS metric in ResilienceMetrics."""

    def test_crps_present_when_std_provided(self):
        """CRPS is present and finite when std is provided."""
        from src.training.metrics import ResilienceMetrics
        metrics = ResilienceMetrics()
        mean = torch.randn(16, 1)
        std = torch.rand(16, 1) + 0.1
        target = torch.randn(16, 1)
        result = metrics.compute(mean, std, target)
        assert "crps" in result
        assert np.isfinite(result["crps"])

    def test_crps_nan_when_std_not_provided(self):
        """CRPS is NaN when std is not provided."""
        from src.training.metrics import ResilienceMetrics
        metrics = ResilienceMetrics()
        mean = torch.randn(16, 1)
        target = torch.randn(16, 1)
        result = metrics.compute(mean, target=target)
        assert "crps" in result
        assert np.isnan(result["crps"])
