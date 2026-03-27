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
from scipy.special import erf as scipy_erf
from scipy.stats import pearsonr, spearmanr

from src.data.constants import EPSILON_DIVISION, EPSILON_POSITIVE_FLOOR
from src.utils.statistics import calibration_error as _shared_calibration_error


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
                "r2": float("nan"), "rmse": float("nan"), "mae": float("nan"),
                "pearson_r": float("nan"), "spearman_rho": float("nan"),
                "mean_std": float("nan"), "calibration_error": float("nan"),
                "crps": float("nan"),
            }

        mean_np = mean.detach().cpu().numpy().flatten()
        target_np = target.detach().cpu().numpy().flatten()

        # Prediction quality metrics
        residuals = target_np - mean_np
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((target_np - target_np.mean()) ** 2)

        r2 = 1.0 - ss_res / (ss_tot + EPSILON_DIVISION) if ss_tot > EPSILON_DIVISION else 0.0
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        mae = float(np.mean(np.abs(residuals)))

        # Correlation
        if len(mean_np) >= 3 and np.std(mean_np) > EPSILON_POSITIVE_FLOOR and np.std(target_np) > EPSILON_POSITIVE_FLOOR:
            pearson_r_val = float(pearsonr(mean_np, target_np)[0])
            spearman_rho_val = float(spearmanr(mean_np, target_np)[0])
        else:
            pearson_r_val = float('nan')
            spearman_rho_val = float('nan')

        result = {
            "r2": float(r2),
            "rmse": rmse,
            "mae": mae,
            "pearson_r": pearson_r_val,
            "spearman_rho": spearman_rho_val,
        }

        # Uncertainty metrics (only if std provided)
        if std is not None:
            std_np = std.detach().cpu().numpy().flatten()
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

    # NOTE: This is an evaluation metric (numpy, non-differentiable), NOT a
    # training loss. The differentiable CRPSLoss class was intentionally removed
    # (designed but deferred; see design doc changelog Round 14). This metric is
    # retained because CRPS is a standard proper scoring rule for evaluating
    # probabilistic predictions alongside calibration error.
    @staticmethod
    def _crps(
        mean: np.ndarray,
        std: np.ndarray,
        target: np.ndarray,
    ) -> float:
        """
        Compute mean CRPS for Gaussian predictions (numpy version).

        CRPS(N(μ,σ), y) = σ * [z*(2Φ(z) - 1) + 2φ(z) - 1/√π]
        where z = (y - μ) / σ

        Args:
            mean: [N] predicted values
            std: [N] predicted standard deviations
            target: [N] ground truth values

        Returns:
            Mean CRPS across all samples
        """
        z = (target - mean) / (std + EPSILON_DIVISION)
        cdf_z = 0.5 * (1.0 + scipy_erf(z / np.sqrt(2.0)))
        pdf_z = np.exp(-0.5 * z ** 2) / np.sqrt(2.0 * np.pi)
        crps = std * (z * (2.0 * cdf_z - 1.0) + 2.0 * pdf_z - 1.0 / np.sqrt(np.pi))
        return float(np.mean(crps))

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
