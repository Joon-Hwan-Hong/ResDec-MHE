"""
Metrics for cognitive resilience model evaluation.

ResilienceMetrics computes prediction quality and uncertainty calibration:
- R², RMSE, MAE: Prediction accuracy
- Pearson r, Spearman ρ: Correlation measures
- mean_std: Average predicted uncertainty
- Calibration error: Whether predicted σ matches actual errors
"""

from typing import Optional

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr


class ResilienceMetrics:
    """
    Comprehensive metrics for cognition prediction.

    Computes prediction quality metrics and uncertainty calibration.
    All returned values are Python floats.
    """

    def compute(
        self,
        mean: torch.Tensor,
        std: Optional[torch.Tensor] = None,
        target: torch.Tensor = None,
    ) -> dict[str, float]:
        """
        Compute all metrics.

        Args:
            mean: [N, 1] predicted values
            std: [N, 1] predicted uncertainty (optional, for calibration)
            target: [N, 1] ground truth values

        Returns:
            Dict of metric name -> float value
        """
        mean_np = mean.detach().cpu().numpy().flatten()
        target_np = target.detach().cpu().numpy().flatten()

        # Prediction quality metrics
        residuals = target_np - mean_np
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((target_np - target_np.mean()) ** 2)

        r2 = 1.0 - ss_res / (ss_tot + 1e-10) if ss_tot > 1e-10 else 0.0
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        mae = float(np.mean(np.abs(residuals)))

        # Correlation
        if len(mean_np) >= 3:
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
        else:
            result["mean_std"] = float('nan')
            result["calibration_error"] = float('nan')

        return result

    @staticmethod
    def _calibration_error(
        mean: np.ndarray,
        std: np.ndarray,
        target: np.ndarray,
    ) -> float:
        """
        Compute regression calibration error.

        For a well-calibrated model: 68.3% of z-scores ≤ 1, 95.4% ≤ 2, 99.7% ≤ 3.
        Returns mean gap across these levels (negative = overconfident).

        Args:
            mean: [N] predicted values
            std: [N] predicted standard deviations
            target: [N] ground truth values

        Returns:
            Mean calibration error (0 = perfect, negative = overconfident)
        """
        z_scores = np.abs(target - mean) / (std + 1e-10)

        # Expected coverage at 1σ, 2σ, 3σ
        expected = [0.6827, 0.9545, 0.9973]
        thresholds = [1.0, 2.0, 3.0]

        gaps = []
        for threshold, exp_coverage in zip(thresholds, expected):
            observed_coverage = float(np.mean(z_scores <= threshold))
            gaps.append(observed_coverage - exp_coverage)

        return float(np.mean(gaps))
