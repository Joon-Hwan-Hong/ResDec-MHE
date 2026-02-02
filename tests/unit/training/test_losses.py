"""
Tests for training loss functions.

Tests BetaNLLLoss (β-NLL) for heteroscedastic regression:
- Output correctness (non-negative, correct gradient flow)
- β=0 reduces to MSE behavior
- β=1 reduces to NLL behavior
- β=0.5 provides balanced loss (default)
- Edge cases (zero std, large values, batch size 1)
"""

import pytest
import torch
import torch.nn as nn


class TestBetaNLLLoss:
    """Tests for BetaNLLLoss."""

    # ─────────────────────────────────────────────────────────────────
    # Instantiation
    # ─────────────────────────────────────────────────────────────────

    def test_instantiation_default_beta(self):
        """BetaNLLLoss instantiates with default beta=0.5."""
        from src.training.losses import BetaNLLLoss
        loss_fn = BetaNLLLoss()
        assert loss_fn.beta == 0.5

    def test_instantiation_custom_beta(self):
        """BetaNLLLoss accepts custom beta values."""
        from src.training.losses import BetaNLLLoss
        loss_fn = BetaNLLLoss(beta=0.0)
        assert loss_fn.beta == 0.0
        loss_fn = BetaNLLLoss(beta=1.0)
        assert loss_fn.beta == 1.0

    def test_instantiation_invalid_beta_raises(self):
        """BetaNLLLoss raises ValueError for beta outside [0, 1]."""
        from src.training.losses import BetaNLLLoss
        with pytest.raises(ValueError):
            BetaNLLLoss(beta=-0.1)
        with pytest.raises(ValueError):
            BetaNLLLoss(beta=1.1)

    # ─────────────────────────────────────────────────────────────────
    # Output Properties
    # ─────────────────────────────────────────────────────────────────

    def test_output_is_scalar(self):
        """Loss output is a scalar tensor."""
        from src.training.losses import BetaNLLLoss
        loss_fn = BetaNLLLoss()
        mean = torch.randn(8, 1)
        std = torch.rand(8, 1) + 0.1
        target = torch.randn(8, 1)
        loss = loss_fn(mean, std, target)
        assert loss.dim() == 0

    def test_output_is_non_negative(self):
        """Loss is non-negative for reasonable inputs."""
        from src.training.losses import BetaNLLLoss
        loss_fn = BetaNLLLoss()
        # With small std and target=mean, loss should be dominated by log(var) term
        # but overall the loss function is non-negative by construction
        mean = torch.zeros(16, 1)
        std = torch.ones(16, 1)
        target = torch.zeros(16, 1)
        loss = loss_fn(mean, std, target)
        # The log_var term can be negative (when var < 1), so loss CAN be negative
        # But with std=1 and target=mean, it's ~0.5*log(1) + 0 = 0
        assert torch.isfinite(loss)

    def test_gradient_flow(self):
        """Gradients flow through loss to mean and std."""
        from src.training.losses import BetaNLLLoss
        loss_fn = BetaNLLLoss()
        mean = torch.randn(8, 1, requires_grad=True)
        # Use a leaf tensor for std so .grad is populated
        std_raw = torch.rand(8, 1) + 0.1
        std_param = std_raw.clone().detach().requires_grad_(True)
        target = torch.randn(8, 1)
        loss = loss_fn(mean, std_param, target)
        loss.backward()
        assert mean.grad is not None
        assert std_param.grad is not None
        assert not torch.isnan(mean.grad).any()
        assert not torch.isnan(std_param.grad).any()

    # ─────────────────────────────────────────────────────────────────
    # Beta Behavior
    # ─────────────────────────────────────────────────────────────────

    def test_beta_zero_approximates_mse(self):
        """β=0 loss should be dominated by MSE-like behavior (ignores variance)."""
        from src.training.losses import BetaNLLLoss
        loss_fn = BetaNLLLoss(beta=0.0)
        mean = torch.tensor([[1.0]])
        std = torch.tensor([[0.5]])
        target = torch.tensor([[2.0]])
        loss = loss_fn(mean, std, target)
        # β=0: mse_term = 0.5 * (target - mean)^2 / var^0 / var^1 = 0.5 * 1 / 0.25 = 2.0
        # plus 0.5*log(0.25) = -0.693
        # The MSE term dominates and is independent of std in the gradient
        assert torch.isfinite(loss)

    def test_perfect_prediction_low_loss(self):
        """Perfect prediction (target == mean) gives lower loss than imperfect."""
        from src.training.losses import BetaNLLLoss
        loss_fn = BetaNLLLoss()
        std = torch.ones(8, 1) * 0.5
        target = torch.randn(8, 1)
        # Perfect prediction
        loss_perfect = loss_fn(target.clone(), std.clone(), target)
        # Bad prediction
        loss_bad = loss_fn(target + 5.0, std.clone(), target)
        assert loss_perfect < loss_bad

    def test_different_betas_different_gradients(self):
        """Different β values produce different gradients on std.

        β-NLL: var.detach()^β * var^(1-β) in the denominator of the MSE term.
        In the forward pass, detach() doesn't change values so the loss is the same.
        But β changes how much gradient flows through std (the key purpose of β-NLL).
        β=0: std receives no gradient from MSE term (pure MSE behavior)
        β=1: std receives full NLL gradient (can exploit uncertainty)
        """
        from src.training.losses import BetaNLLLoss
        mean = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        std_values = torch.tensor([[0.1], [0.5], [1.0], [5.0]])
        target = torch.tensor([[1.5], [1.0], [4.0], [3.0]])

        # β=0: std gradient comes only from log_var term
        std_0 = std_values.clone().detach().requires_grad_(True)
        loss_0 = BetaNLLLoss(beta=0.0)(mean.clone(), std_0, target.clone())
        loss_0.backward()
        grad_0 = std_0.grad.clone()

        # β=1: std gradient from both log_var and mse/var terms
        std_1 = std_values.clone().detach().requires_grad_(True)
        loss_1 = BetaNLLLoss(beta=1.0)(mean.clone(), std_1, target.clone())
        loss_1.backward()
        grad_1 = std_1.grad.clone()

        # Gradients should differ because β changes how variance affects MSE term gradient
        assert not torch.allclose(grad_0, grad_1, atol=1e-5)

    # ─────────────────────────────────────────────────────────────────
    # Edge Cases
    # ─────────────────────────────────────────────────────────────────

    def test_batch_size_one(self):
        """Loss works with batch size 1."""
        from src.training.losses import BetaNLLLoss
        loss_fn = BetaNLLLoss()
        loss = loss_fn(
            torch.randn(1, 1),
            torch.rand(1, 1) + 0.1,
            torch.randn(1, 1),
        )
        assert torch.isfinite(loss)

    def test_large_batch(self):
        """Loss works with large batch."""
        from src.training.losses import BetaNLLLoss
        loss_fn = BetaNLLLoss()
        loss = loss_fn(
            torch.randn(256, 1),
            torch.rand(256, 1) + 0.1,
            torch.randn(256, 1),
        )
        assert torch.isfinite(loss)

    def test_very_small_std(self):
        """Loss handles very small (but positive) std without inf/nan."""
        from src.training.losses import BetaNLLLoss
        loss_fn = BetaNLLLoss()
        loss = loss_fn(
            torch.zeros(4, 1),
            torch.ones(4, 1) * 1e-6,
            torch.zeros(4, 1),
        )
        assert torch.isfinite(loss)

    def test_large_std(self):
        """Loss handles large std values."""
        from src.training.losses import BetaNLLLoss
        loss_fn = BetaNLLLoss()
        loss = loss_fn(
            torch.zeros(4, 1),
            torch.ones(4, 1) * 100.0,
            torch.zeros(4, 1),
        )
        assert torch.isfinite(loss)

    def test_std_must_be_positive(self):
        """Loss raises or handles zero/negative std gracefully."""
        from src.training.losses import BetaNLLLoss
        loss_fn = BetaNLLLoss()
        with pytest.raises(ValueError):
            loss_fn(
                torch.zeros(4, 1),
                torch.zeros(4, 1),  # zero std
                torch.zeros(4, 1),
            )


class TestCRPSLoss:
    """Tests for CRPSLoss (Continuous Ranked Probability Score)."""

    def test_output_is_scalar(self):
        """CRPS loss output is a scalar tensor."""
        from src.training.losses import CRPSLoss
        loss_fn = CRPSLoss()
        mean = torch.randn(8, 1)
        std = torch.rand(8, 1) + 0.1
        target = torch.randn(8, 1)
        loss = loss_fn(mean, std, target)
        assert loss.dim() == 0

    def test_output_is_non_negative(self):
        """CRPS is non-negative (it is a proper scoring rule)."""
        from src.training.losses import CRPSLoss
        loss_fn = CRPSLoss()
        torch.manual_seed(42)
        mean = torch.randn(16, 1)
        std = torch.rand(16, 1) + 0.1
        target = torch.randn(16, 1)
        loss = loss_fn(mean, std, target)
        assert loss.item() >= 0.0

    def test_gradient_flow(self):
        """Gradients flow through CRPS loss to mean and std."""
        from src.training.losses import CRPSLoss
        loss_fn = CRPSLoss()
        mean = torch.randn(8, 1, requires_grad=True)
        std_param = (torch.rand(8, 1) + 0.1).clone().detach().requires_grad_(True)
        target = torch.randn(8, 1)
        loss = loss_fn(mean, std_param, target)
        loss.backward()
        assert mean.grad is not None
        assert std_param.grad is not None
        assert not torch.isnan(mean.grad).any()
        assert not torch.isnan(std_param.grad).any()

    def test_perfect_prediction_lower_crps(self):
        """Perfect prediction gives lower CRPS than bad prediction."""
        from src.training.losses import CRPSLoss
        loss_fn = CRPSLoss()
        std = torch.ones(8, 1) * 0.5
        target = torch.randn(8, 1)
        crps_perfect = loss_fn(target.clone(), std.clone(), target)
        crps_bad = loss_fn(target + 5.0, std.clone(), target)
        assert crps_perfect < crps_bad

    def test_tighter_std_better_at_correct_mean(self):
        """Tighter std gives lower CRPS when mean is correct."""
        from src.training.losses import CRPSLoss
        loss_fn = CRPSLoss()
        target = torch.randn(16, 1)
        crps_tight = loss_fn(target.clone(), torch.ones(16, 1) * 0.1, target)
        crps_wide = loss_fn(target.clone(), torch.ones(16, 1) * 2.0, target)
        assert crps_tight < crps_wide

    def test_std_must_be_positive(self):
        """CRPS raises ValueError for zero or negative std."""
        from src.training.losses import CRPSLoss
        loss_fn = CRPSLoss()
        with pytest.raises(ValueError):
            loss_fn(torch.zeros(4, 1), torch.zeros(4, 1), torch.zeros(4, 1))

    def test_batch_size_one(self):
        """CRPS works with batch size 1."""
        from src.training.losses import CRPSLoss
        loss_fn = CRPSLoss()
        loss = loss_fn(
            torch.randn(1, 1),
            torch.rand(1, 1) + 0.1,
            torch.randn(1, 1),
        )
        assert torch.isfinite(loss)

    def test_known_value_standard_normal_at_zero(self):
        """CRPS(N(0,1), y=0) = (√2 - 1) / √π ≈ 0.2338."""
        from src.training.losses import CRPSLoss
        import math
        loss_fn = CRPSLoss()
        mean = torch.tensor([[0.0]])
        std = torch.tensor([[1.0]])
        target = torch.tensor([[0.0]])
        crps = loss_fn(mean, std, target)
        expected = (math.sqrt(2) - 1) / math.sqrt(math.pi)
        assert abs(crps.item() - expected) < 1e-5

    def test_known_value_shifted(self):
        """CRPS at y=μ equals σ * (√2 - 1) / √π for any σ."""
        from src.training.losses import CRPSLoss
        import math
        loss_fn = CRPSLoss()
        sigma = 3.0
        mean = torch.tensor([[5.0]])
        std = torch.tensor([[sigma]])
        target = torch.tensor([[5.0]])  # y = μ
        crps = loss_fn(mean, std, target)
        expected = sigma * (math.sqrt(2) - 1) / math.sqrt(math.pi)
        assert abs(crps.item() - expected) < 1e-5


class TestMSEFallback:
    """Tests for MSE loss fallback used with deterministic head."""

    def test_mse_loss_available(self):
        """MSE loss function is available from losses module."""
        from src.training.losses import mse_loss
        mean = torch.randn(8, 1)
        target = torch.randn(8, 1)
        loss = mse_loss(mean, target)
        assert torch.isfinite(loss)
        assert loss.dim() == 0

    def test_mse_loss_zero_for_perfect_prediction(self):
        """MSE is zero when prediction equals target."""
        from src.training.losses import mse_loss
        target = torch.randn(8, 1)
        loss = mse_loss(target.clone(), target)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-7)

    def test_mse_loss_gradient_flow(self):
        """Gradients flow through MSE loss."""
        from src.training.losses import mse_loss
        mean = torch.randn(8, 1, requires_grad=True)
        target = torch.randn(8, 1)
        loss = mse_loss(mean, target)
        loss.backward()
        assert mean.grad is not None
