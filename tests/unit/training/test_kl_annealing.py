"""
Tests for KL annealing: KLAnnealedELBO and KLAnnealingCallback.

Tests the custom ELBO class that decomposes loss into NLL and KL terms
with a tunable KL weight, and the callback that ramps the weight from
alpha_min to 1.0 over warmup_epochs.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer import TraceMeanField_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal


# ---------------------------------------------------------------------------
# Shared toy model for ELBO tests
# ---------------------------------------------------------------------------

class ToyModel(PyroModule):
    """Minimal PyroModule for testing ELBO decomposition."""

    def __init__(self, d_in: int = 5, d_out: int = 1):
        super().__init__()
        self.linear = PyroModule[nn.Linear](d_in, d_out)
        # Place N(0,1) priors on weight and bias
        self.linear.weight = PyroSample(
            dist.Normal(0.0, 1.0).expand([d_out, d_in]).to_event(2)
        )
        self.linear.bias = PyroSample(
            dist.Normal(0.0, 1.0).expand([d_out]).to_event(1)
        )

    def forward(self, x, y=None):
        mean = self.linear(x)
        sigma = pyro.sample("sigma", dist.LogNormal(0.0, 1.0))
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mean, sigma).to_event(1), obs=y)
        return obs


# ---------------------------------------------------------------------------
# TestKLAnnealedELBO
# ---------------------------------------------------------------------------

class TestKLAnnealedELBO:
    """Tests for the KLAnnealedELBO custom ELBO class."""

    def setup_method(self):
        pyro.clear_param_store()

    def _make_model_guide_data(self, seed: int = 0):
        """Create a ToyModel, guide, and dummy data."""
        torch.manual_seed(seed)
        model = ToyModel(d_in=5, d_out=1)
        guide = AutoDiagonalNormal(model)
        x = torch.randn(8, 5)
        y = torch.randn(8, 1)
        # Prototype the guide
        guide(x, y)
        return model, guide, x, y

    def test_kl_weight_one_matches_standard_elbo(self):
        """With kl_weight=1.0, loss should match TraceMeanField_ELBO within tolerance."""
        from src.training.kl_annealing import KLAnnealedELBO

        model, guide, x, y = self._make_model_guide_data(seed=42)

        standard = TraceMeanField_ELBO()
        custom = KLAnnealedELBO(kl_weight=1.0)

        # Use same seed for both to get identical stochastic traces
        torch.manual_seed(99)
        loss_standard = standard.differentiable_loss(model, guide, x, y)

        torch.manual_seed(99)
        loss_custom = custom.differentiable_loss(model, guide, x, y)

        assert abs(loss_standard.item() - loss_custom.item()) < 1e-4, (
            f"Standard ELBO {loss_standard.item():.6f} vs "
            f"Custom ELBO {loss_custom.item():.6f}"
        )

    def test_kl_weight_reduced_gives_lower_loss(self):
        """With kl_weight=0.1, loss should be less than or equal to standard ELBO."""
        from src.training.kl_annealing import KLAnnealedELBO

        model, guide, x, y = self._make_model_guide_data(seed=42)

        standard = KLAnnealedELBO(kl_weight=1.0)
        reduced = KLAnnealedELBO(kl_weight=0.1)

        torch.manual_seed(99)
        loss_full = standard.differentiable_loss(model, guide, x, y)

        torch.manual_seed(99)
        loss_reduced = reduced.differentiable_loss(model, guide, x, y)

        # Reduced KL weight means less KL penalty, so total loss should be <= full
        assert loss_reduced.item() <= loss_full.item() + 1e-6, (
            f"Reduced KL loss {loss_reduced.item():.6f} should be <= "
            f"full KL loss {loss_full.item():.6f}"
        )

    def test_nll_and_kl_returned(self):
        """differentiable_loss_with_parts returns (nll, kl, total) where total = nll + kl."""
        from src.training.kl_annealing import KLAnnealedELBO

        model, guide, x, y = self._make_model_guide_data(seed=42)

        kl_weight = 0.5
        elbo = KLAnnealedELBO(kl_weight=kl_weight)

        torch.manual_seed(99)
        nll, kl, total = elbo.differentiable_loss_with_parts(model, guide, x, y)

        # All should be tensors
        assert isinstance(nll, torch.Tensor)
        assert isinstance(kl, torch.Tensor)
        assert isinstance(total, torch.Tensor)

        # total = nll + kl (where kl already has kl_weight applied)
        assert abs(total.item() - (nll.item() + kl.item())) < 1e-4, (
            f"total {total.item():.6f} != nll {nll.item():.6f} + kl {kl.item():.6f}"
        )

        # NLL should be positive (negative log-likelihood)
        # KL should be non-negative (weighted KL divergence)
        assert nll.item() > -1e6, "NLL should be finite"
        assert kl.item() >= -1e-6, "Weighted KL should be non-negative"

    def test_n_train_scales_kl(self):
        """With n_train=N, KL is divided by N compared to n_train=1."""
        from src.training.kl_annealing import KLAnnealedELBO

        model, guide, x, y = self._make_model_guide_data(seed=42)

        elbo_no_scale = KLAnnealedELBO(kl_weight=1.0, n_train=1)
        elbo_scaled = KLAnnealedELBO(kl_weight=1.0, n_train=100)

        torch.manual_seed(99)
        nll_1, kl_1, total_1 = elbo_no_scale.differentiable_loss_with_parts(model, guide, x, y)

        torch.manual_seed(99)
        nll_100, kl_100, total_100 = elbo_scaled.differentiable_loss_with_parts(model, guide, x, y)

        # NLL should be identical (not scaled)
        assert abs(nll_1.item() - nll_100.item()) < 1e-4, (
            f"NLL should be identical: {nll_1.item():.6f} vs {nll_100.item():.6f}"
        )

        # KL should be 100x smaller with n_train=100
        assert abs(kl_1.item() / 100 - kl_100.item()) < 1e-3, (
            f"KL should be 100x smaller: {kl_1.item()/100:.6f} vs {kl_100.item():.6f}"
        )

    def test_n_train_default_is_1(self):
        """Default n_train=1 preserves backward compatibility."""
        from src.training.kl_annealing import KLAnnealedELBO

        model, guide, x, y = self._make_model_guide_data(seed=42)

        elbo_default = KLAnnealedELBO(kl_weight=0.5)
        elbo_explicit = KLAnnealedELBO(kl_weight=0.5, n_train=1)

        torch.manual_seed(99)
        loss_default = elbo_default.differentiable_loss(model, guide, x, y)

        torch.manual_seed(99)
        loss_explicit = elbo_explicit.differentiable_loss(model, guide, x, y)

        assert abs(loss_default.item() - loss_explicit.item()) < 1e-4


# ---------------------------------------------------------------------------
# TestKLAnnealingCallback
# ---------------------------------------------------------------------------

class TestKLAnnealingCallback:
    """Tests for KLAnnealingCallback schedule computation."""

    def test_weight_at_epoch_0(self):
        """KL weight at epoch 0 should be alpha_min."""
        from src.training.callbacks import KLAnnealingCallback

        callback = KLAnnealingCallback(alpha_min=0.01, warmup_epochs=5)
        weight = callback.get_kl_weight(epoch=0)
        assert abs(weight - 0.01) < 1e-6

    def test_weight_at_warmup_end(self):
        """KL weight at epoch=warmup_epochs should be 1.0."""
        from src.training.callbacks import KLAnnealingCallback

        callback = KLAnnealingCallback(alpha_min=0.01, warmup_epochs=5)
        weight = callback.get_kl_weight(epoch=5)
        assert abs(weight - 1.0) < 1e-6

    def test_weight_after_warmup(self):
        """KL weight should stay 1.0 after warmup is complete."""
        from src.training.callbacks import KLAnnealingCallback

        callback = KLAnnealingCallback(alpha_min=0.01, warmup_epochs=5)
        for epoch in [5, 6, 10, 50, 100]:
            weight = callback.get_kl_weight(epoch=epoch)
            assert abs(weight - 1.0) < 1e-6, (
                f"Weight at epoch {epoch} should be 1.0, got {weight}"
            )

    def test_weight_midway(self):
        """KL weight at midpoint should be between alpha_min and 1.0."""
        from src.training.callbacks import KLAnnealingCallback

        callback = KLAnnealingCallback(alpha_min=0.01, warmup_epochs=10)
        weight = callback.get_kl_weight(epoch=5)
        assert 0.01 < weight < 1.0, (
            f"Midpoint weight should be between 0.01 and 1.0, got {weight}"
        )

    def test_linear_schedule(self):
        """Linear schedule produces evenly spaced weights."""
        from src.training.callbacks import KLAnnealingCallback

        alpha_min = 0.0
        warmup_epochs = 4
        callback = KLAnnealingCallback(
            alpha_min=alpha_min, warmup_epochs=warmup_epochs, schedule="linear",
        )

        weights = [callback.get_kl_weight(epoch=e) for e in range(warmup_epochs + 1)]
        # Expected: [0.0, 0.25, 0.5, 0.75, 1.0]
        expected = [e / warmup_epochs for e in range(warmup_epochs + 1)]
        for w, exp in zip(weights, expected):
            assert abs(w - exp) < 1e-6, (
                f"Weight {w} != expected {exp}"
            )

    def test_on_train_epoch_start_sets_kl_weight(self):
        """Callback sets elbo.kl_weight and logs it on epoch start."""
        from src.training.callbacks import KLAnnealingCallback

        callback = KLAnnealingCallback(alpha_min=0.01, warmup_epochs=5)

        trainer = MagicMock()
        trainer.current_epoch = 0

        # Mock pl_module with an elbo object that has kl_weight
        elbo = MagicMock()
        elbo.kl_weight = 1.0
        pl_module = MagicMock()
        pl_module.elbo = elbo

        callback.on_train_epoch_start(trainer, pl_module)

        # kl_weight should be set to alpha_min at epoch 0
        assert abs(elbo.kl_weight - 0.01) < 1e-6
        pl_module.log.assert_called_once_with(
            "kl_weight", pytest.approx(0.01, abs=1e-6), rank_zero_only=True, sync_dist=True,
        )

    def test_on_train_epoch_start_no_elbo_is_noop(self):
        """Callback does nothing if pl_module has no elbo attribute."""
        from src.training.callbacks import KLAnnealingCallback

        callback = KLAnnealingCallback(alpha_min=0.01, warmup_epochs=5)

        trainer = MagicMock()
        trainer.current_epoch = 0

        pl_module = MagicMock(spec=[])  # No attributes at all
        # Should not raise
        callback.on_train_epoch_start(trainer, pl_module)

    def test_warmup_zero_returns_one(self):
        """With warmup_epochs=0, weight is always 1.0."""
        from src.training.callbacks import KLAnnealingCallback

        callback = KLAnnealingCallback(alpha_min=0.01, warmup_epochs=0)
        for epoch in [0, 1, 5]:
            weight = callback.get_kl_weight(epoch=epoch)
            assert abs(weight - 1.0) < 1e-6
