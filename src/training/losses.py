"""
Loss functions for cognitive resilience model training.

BetaNLLLoss: β-NLL loss for heteroscedastic regression that prevents
uncertainty exploitation (model inflating σ to reduce loss without
improving predictions).

mse_loss: Simple MSE fallback for deterministic head.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BetaNLLLoss(nn.Module):
    """
    β-NLL Loss for heteroscedastic regression.

    Standard Gaussian NLL can be exploited — the model increases σ on hard
    examples to reduce loss without improving predictions. β-NLL prevents this
    by detaching variance from part of the gradient computation.

    β=0: Gradient of MSE term ignores variance entirely
    β=1: Standard NLL (can exploit uncertainty)
    β=0.5: Balanced (recommended)

    Args:
        beta: Balance parameter in [0, 1]. Default: 0.5.

    Reference:
        Seitzer et al., "On the Pitfalls of Heteroscedastic Uncertainty Estimation
        with Probabilistic Neural Networks" (ICLR 2022)
    """

    def __init__(self, beta: float = 0.5):
        super().__init__()
        if not (0.0 <= beta <= 1.0):
            raise ValueError(f"beta must be in [0, 1], got {beta}")
        self.beta = beta

    def forward(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute β-NLL loss.

        Args:
            mean: [B, 1] predicted mean
            std: [B, 1] predicted standard deviation (must be > 0)
            target: [B, 1] ground truth

        Returns:
            Scalar loss value
        """
        if (std <= 0).any():
            raise ValueError("std must be strictly positive everywhere")

        var = std ** 2

        # Log-variance term (unchanged by β)
        log_var_term = 0.5 * torch.log(var)

        # MSE term with β-weighted variance detachment
        # var.detach() ** β stops gradients through that portion of variance
        mse_term = 0.5 * (target - mean) ** 2 / (var.detach() ** self.beta * var ** (1 - self.beta))

        return (log_var_term + mse_term).mean()

    def extra_repr(self) -> str:
        return f"beta={self.beta}"


def mse_loss(mean: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Simple MSE loss for deterministic prediction head.

    Args:
        mean: [B, 1] predicted values
        target: [B, 1] ground truth

    Returns:
        Scalar MSE loss
    """
    return F.mse_loss(mean, target)
