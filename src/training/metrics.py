"""
Metrics for cognitive resilience model evaluation.

ResilienceMetrics computes prediction quality and uncertainty calibration:
- R², RMSE, MAE: Prediction accuracy
- Pearson r, Spearman ρ: Correlation measures
- mean_std: Average predicted uncertainty
- Calibration error: Whether predicted σ matches actual errors
"""

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from src.data.constants import EPSILON_DIVISION, EPSILON_POSITIVE_FLOOR
from src.utils.statistics import (
    calibration_error as _shared_calibration_error,
    crps_gaussian as _shared_crps_gaussian,
)


class ResilienceMetrics:
    """
    Comprehensive metrics for cognition prediction.

    Computes prediction quality metrics and uncertainty calibration.
    All returned values are Python floats.
    """

    def compute(
        self,
        mean: torch.Tensor,
        std: torch.Tensor | None = None,
        target: torch.Tensor = None,
    ) -> dict[str, float]:
        """
        Compute all metrics.

        Args:
            mean: [N, 1] predicted values
            std: [N, 1] predicted uncertainty (optional, for calibration)
            target: [N, 1] ground truth values (required)

        Returns:
            Dict of metric name -> float value
        """
        if mean.numel() == 0 or target is None or target.numel() == 0:
            return {
                "r2": float("nan"), "r2_calibrated": float("nan"),
                "rmse": float("nan"), "mae": float("nan"),
                "pearson_r": float("nan"), "spearman_rho": float("nan"),
                "mean_std": float("nan"), "calibration_error": float("nan"),
                "crps": float("nan"),
            }

        mean_np = mean.detach().cpu().float().numpy().flatten()
        target_np = target.detach().cpu().float().numpy().flatten()

        # Prediction quality metrics
        residuals = target_np - mean_np
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((target_np - target_np.mean()) ** 2)

        # Constant target → R² is undefined (not 0). Return NaN to match the
        # convention in resdec_lightning_module.py:on_validation_epoch_end.
        r2 = (
            1.0 - ss_res / (ss_tot + EPSILON_DIVISION)
            if ss_tot > EPSILON_DIVISION
            else float("nan")
        )
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        mae = float(np.mean(np.abs(residuals)))

        # Calibrated R²: R² achievable with optimal linear recalibration.
        # Diagnostic only — measures discrimination independent of calibration.
        if len(mean_np) >= 3 and np.std(mean_np) > EPSILON_POSITIVE_FLOOR:
            coeffs = np.polyfit(mean_np, target_np, 1)
            calibrated_pred = coeffs[0] * mean_np + coeffs[1]
            ss_res_cal = np.sum((target_np - calibrated_pred) ** 2)
            # Same convention: NaN for constant target.
            r2_calibrated = (
                1.0 - ss_res_cal / (ss_tot + EPSILON_DIVISION)
                if ss_tot > EPSILON_DIVISION
                else float("nan")
            )
        else:
            r2_calibrated = float('nan')

        # Correlation
        if len(mean_np) >= 3 and np.std(mean_np) > EPSILON_POSITIVE_FLOOR and np.std(target_np) > EPSILON_POSITIVE_FLOOR:
            pearson_r = float(pearsonr(mean_np, target_np)[0])
            spearman_rho = float(spearmanr(mean_np, target_np)[0])
        else:
            pearson_r = float('nan')
            spearman_rho = float('nan')

        result = {
            "r2": float(r2),
            "r2_calibrated": float(r2_calibrated),
            "rmse": rmse,
            "mae": mae,
            "pearson_r": pearson_r,
            "spearman_rho": spearman_rho,
        }

        # Uncertainty metrics (only if std is a non-empty tensor)
        if std is not None and std.numel() > 0:
            std_np = std.detach().cpu().float().numpy().flatten()
            result["mean_std"] = float(np.mean(std_np))
            result["calibration_error"] = float(
                self._calibration_error(mean_np, std_np, target_np)
            )
            result["crps"] = float(self._crps(mean_np, std_np, target_np))
        else:
            result["mean_std"] = float('nan')
            result["calibration_error"] = float('nan')
            result["crps"] = float('nan')

        return result

    def bootstrap_ci(
        self,
        mean: torch.Tensor,
        std: torch.Tensor | None = None,
        target: torch.Tensor = None,
        metrics: list[str] | None = None,
        n_bootstrap: int = 1000,
        ci: float = 0.95,
        seed: int = 42,
    ) -> dict[str, tuple[float, float]]:
        """Compute bootstrap confidence intervals for selected metrics.

        Args:
            mean: [N, 1] predicted values
            std: [N, 1] predicted uncertainty (unused, kept for API consistency)
            target: [N, 1] ground truth values
            metrics: List of metric names to bootstrap (default: ["r2"])
            n_bootstrap: Number of bootstrap resamples
            ci: Confidence level (default: 0.95)
            seed: Random seed for reproducibility

        Returns:
            Dict of metric name -> (lower, upper) confidence interval
        """
        if metrics is None:
            metrics = ["r2"]

        mean_np = mean.detach().cpu().float().numpy().flatten()
        target_np = target.detach().cpu().float().numpy().flatten()
        n = len(mean_np)

        rng = np.random.default_rng(seed)
        boot_results: dict[str, list[float]] = {m: [] for m in metrics}

        for _ in range(n_bootstrap):
            idx = rng.choice(n, n, replace=True)
            m_boot = mean_np[idx]
            t_boot = target_np[idx]

            if "r2" in metrics:
                residuals = t_boot - m_boot
                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((t_boot - t_boot.mean()) ** 2)
                # Constant resample → R² undefined. NaN matches the
                # convention used in compute() and resdec_lightning_module.
                r2 = (
                    1.0 - ss_res / (ss_tot + EPSILON_DIVISION)
                    if ss_tot > EPSILON_DIVISION
                    else float("nan")
                )
                boot_results["r2"].append(r2)

        alpha = (1 - ci) / 2
        result = {}
        for m in metrics:
            vals = np.array(boot_results[m])
            # Drop NaNs (e.g., constant-target resamples for R²) before
            # percentile so the CI reflects only well-defined draws.
            finite = vals[np.isfinite(vals)]
            if finite.size == 0:
                result[m] = (float("nan"), float("nan"))
            else:
                result[m] = (
                    float(np.percentile(finite, 100 * alpha)),
                    float(np.percentile(finite, 100 * (1 - alpha))),
                )
        return result

    # NOTE: CRPS is a standard proper scoring rule for evaluating
    # probabilistic predictions; the canonical implementation lives in
    # src.utils.statistics.crps_gaussian. This thin wrapper preserves the
    # ResilienceMetrics class API (kept for backward compatibility with
    # any direct callers); new code should call the shared helper.
    @staticmethod
    def _crps(
        mean: np.ndarray,
        std: np.ndarray,
        target: np.ndarray,
    ) -> float:
        """Mean CRPS for Gaussian predictions (delegates to shared helper)."""
        return _shared_crps_gaussian(mean, std, target)

    @staticmethod
    def _calibration_error(
        mean: np.ndarray,
        std: np.ndarray,
        target: np.ndarray,
    ) -> float:
        """
        Compute regression calibration error.

        Delegates to shared implementation in src.utils.statistics.

        Args:
            mean: [N] predicted values
            std: [N] predicted standard deviations
            target: [N] ground truth values

        Returns:
            Mean calibration error (0 = perfect, negative = overconfident)
        """
        return _shared_calibration_error(mean, std, target)
