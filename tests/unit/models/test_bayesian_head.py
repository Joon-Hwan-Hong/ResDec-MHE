"""
Tests for src/models/heads/bayesian_head.py

Test organization:
1. Initialization - PyroModule subclass, layer creation, validation
2. Forward pass - output shapes, std positivity, batch sizes
3. Pyro integration - trace compatibility, weight sampling
4. Numerical stability - large inputs, minimum std
5. Validation - invalid inputs at constructor and forward time
"""

import math

import pytest
import torch


class TestInitialization:
    """Tests for BayesianPredictionHead initialization."""

    def test_is_pyro_module(self):
        """Should be a PyroModule subclass."""
        from pyro.nn import PyroModule
        from src.models.heads.bayesian_head import BayesianPredictionHead

        head = BayesianPredictionHead(d_input=64, d_hidden=32)

        assert isinstance(head, PyroModule)

    def test_has_fc_layers(self):
        """Should have fc1, fc2, fc_mean, and fc_log_std layers."""
        from src.models.heads.bayesian_head import BayesianPredictionHead

        head = BayesianPredictionHead(d_input=64, d_hidden=32)

        assert hasattr(head, 'fc1')
        assert hasattr(head, 'fc2')
        assert hasattr(head, 'fc_mean')
        assert hasattr(head, 'fc_log_std')

    def test_fc_log_std_is_regular_linear(self):
        """fc_log_std should be a regular nn.Linear, not PyroModule."""
        import torch.nn as nn
        from pyro.nn import PyroModule
        from src.models.heads.bayesian_head import BayesianPredictionHead

        head = BayesianPredictionHead(d_input=64, d_hidden=32)

        # fc_log_std should be regular nn.Linear
        assert isinstance(head.fc_log_std, nn.Linear)
        # It should NOT be a PyroModule (no priors)
        assert not isinstance(head.fc_log_std, PyroModule)

    def test_default_hidden_dim(self):
        """Should use d_hidden=64 by default."""
        from src.models.heads.bayesian_head import BayesianPredictionHead

        head = BayesianPredictionHead(d_input=128)

        assert head.d_hidden == 64

    def test_custom_hidden_dim(self):
        """Should accept custom d_hidden."""
        from src.models.heads.bayesian_head import BayesianPredictionHead

        head = BayesianPredictionHead(d_input=128, d_hidden=256)

        assert head.d_hidden == 256

    def test_stores_d_input(self):
        """Should store d_input attribute."""
        from src.models.heads.bayesian_head import BayesianPredictionHead

        head = BayesianPredictionHead(d_input=128, d_hidden=64)

        assert head.d_input == 128

    def test_fc1_layer_dimensions(self):
        """fc1 should have correct input/output dimensions."""
        from src.models.heads.bayesian_head import BayesianPredictionHead

        head = BayesianPredictionHead(d_input=128, d_hidden=64)

        # Check dimensions via module attributes
        assert head.fc1.in_features == 128
        assert head.fc1.out_features == 64

    def test_fc2_layer_dimensions(self):
        """fc2 should have correct input/output dimensions."""
        from src.models.heads.bayesian_head import BayesianPredictionHead

        head = BayesianPredictionHead(d_input=128, d_hidden=64)

        assert head.fc2.in_features == 64
        assert head.fc2.out_features == 64

    def test_fc_mean_layer_dimensions(self):
        """fc_mean should output single value."""
        from src.models.heads.bayesian_head import BayesianPredictionHead

        head = BayesianPredictionHead(d_input=128, d_hidden=64)

        assert head.fc_mean.in_features == 64
        assert head.fc_mean.out_features == 1

    def test_fc_log_std_layer_dimensions(self):
        """fc_log_std should output single value."""
        from src.models.heads.bayesian_head import BayesianPredictionHead

        head = BayesianPredictionHead(d_input=128, d_hidden=64)

        assert head.fc_log_std.in_features == 64
        assert head.fc_log_std.out_features == 1


class TestForwardPass:
    """Tests for BayesianPredictionHead forward pass."""

    def test_output_shapes(self):
        """Forward should return mean [B, 1] and std [B, 1]."""
        from src.models.heads.bayesian_head import BayesianPredictionHead

        head = BayesianPredictionHead(d_input=64, d_hidden=32)

        x = torch.randn(8, 64)
        mean, std = head(x)

        assert mean.shape == (8, 1)
        assert std.shape == (8, 1)

    def test_std_is_positive(self):
        """std output should always be positive."""
        from src.models.heads.bayesian_head import BayesianPredictionHead

        head = BayesianPredictionHead(d_input=64, d_hidden=32)

        # Test multiple random inputs
        for _ in range(5):
            x = torch.randn(16, 64)
            _, std = head(x)

            assert torch.all(std > 0), "std must be positive"

    def test_different_batch_sizes(self):
        """Should work with various batch sizes."""
        from src.models.heads.bayesian_head import BayesianPredictionHead

        head = BayesianPredictionHead(d_input=64, d_hidden=32)

        for B in [1, 2, 8, 16, 32]:
            x = torch.randn(B, 64)
            mean, std = head(x)

            assert mean.shape == (B, 1)
            assert std.shape == (B, 1)

    def test_accepts_observation(self):
        """Forward should accept y observation for training."""
        from src.models.heads.bayesian_head import BayesianPredictionHead

        head = BayesianPredictionHead(d_input=64, d_hidden=32)

        x = torch.randn(8, 64)
        y = torch.randn(8, 1)

        # Should not raise
        mean, std = head(x, y=y)

        assert mean.shape == (8, 1)
        assert std.shape == (8, 1)

    def test_works_without_observation(self):
        """Forward should work without y for prediction."""
        from src.models.heads.bayesian_head import BayesianPredictionHead

        head = BayesianPredictionHead(d_input=64, d_hidden=32)

        x = torch.randn(8, 64)

        # Should not raise - y defaults to None
        mean, std = head(x)

        assert mean.shape == (8, 1)
        assert std.shape == (8, 1)

    def test_output_varies_across_forward_passes(self):
        """Multiple forward passes should produce different outputs due to weight sampling."""
        from src.models.heads.bayesian_head import BayesianPredictionHead

        head = BayesianPredictionHead(d_input=64, d_hidden=32)
        x = torch.randn(4, 64)
        outputs = [head(x)[0] for _ in range(5)]
        all_same = all(torch.allclose(outputs[0], o, atol=1e-7) for o in outputs[1:])
        assert not all_same

    def test_train_vs_eval_mode_behavior(self):
        """Should produce correct shapes in both train and eval mode."""
        from src.models.heads.bayesian_head import BayesianPredictionHead

        head = BayesianPredictionHead(d_input=64, d_hidden=32)
        x = torch.randn(4, 64)
        y = torch.randn(4, 1)
        head.train()
        mean_train, std_train = head(x, y)
        assert mean_train.shape == (4, 1)
        head.eval()
        mean_eval, std_eval = head(x)
        assert mean_eval.shape == (4, 1)
        assert std_eval.shape == (4, 1)


class TestPyroIntegration:
    """Tests for Pyro integration."""

    def test_works_with_pyro_trace(self):
        """Should be traceable with pyro.poutine.trace."""
        import pyro
        import pyro.poutine as poutine
        from src.models.heads.bayesian_head import BayesianPredictionHead

        head = BayesianPredictionHead(d_input=64, d_hidden=32)

        x = torch.randn(4, 64)

        # Trace the model
        trace = poutine.trace(head).get_trace(x)

        # Should have sample sites for weight priors
        sample_sites = [name for name, node in trace.nodes.items()
                       if node["type"] == "sample"]

        # Should have samples for fc1, fc2, fc_mean weights and biases
        assert len(sample_sites) > 0, "Should have sample sites"

    def test_different_calls_can_sample_different_weights(self):
        """Different forward calls should be able to sample different weights."""
        import pyro
        import pyro.poutine as poutine
        from src.models.heads.bayesian_head import BayesianPredictionHead

        head = BayesianPredictionHead(d_input=64, d_hidden=32)

        x = torch.randn(4, 64)

        # Get two traces
        trace1 = poutine.trace(head).get_trace(x)
        trace2 = poutine.trace(head).get_trace(x)

        # Extract fc1 weight samples
        fc1_weight_1 = trace1.nodes["fc1.weight"]["value"]
        fc1_weight_2 = trace2.nodes["fc1.weight"]["value"]

        # Different traces should sample different weights
        assert not torch.allclose(fc1_weight_1, fc1_weight_2), \
            "Different traces should sample different weights"

    def test_has_weight_priors_on_fc1(self):
        """fc1 should have weight and bias priors."""
        import pyro.poutine as poutine
        from src.models.heads.bayesian_head import BayesianPredictionHead

        head = BayesianPredictionHead(d_input=64, d_hidden=32)

        x = torch.randn(4, 64)
        trace = poutine.trace(head).get_trace(x)

        assert "fc1.weight" in trace.nodes
        assert "fc1.bias" in trace.nodes

    def test_has_weight_priors_on_fc2(self):
        """fc2 should have weight and bias priors."""
        import pyro.poutine as poutine
        from src.models.heads.bayesian_head import BayesianPredictionHead

        head = BayesianPredictionHead(d_input=64, d_hidden=32)

        x = torch.randn(4, 64)
        trace = poutine.trace(head).get_trace(x)

        assert "fc2.weight" in trace.nodes
        assert "fc2.bias" in trace.nodes

    def test_has_weight_priors_on_fc_mean(self):
        """fc_mean should have weight and bias priors."""
        import pyro.poutine as poutine
        from src.models.heads.bayesian_head import BayesianPredictionHead

        head = BayesianPredictionHead(d_input=64, d_hidden=32)

        x = torch.randn(4, 64)
        trace = poutine.trace(head).get_trace(x)

        assert "fc_mean.weight" in trace.nodes
        assert "fc_mean.bias" in trace.nodes

    def test_no_prior_on_fc_log_std(self):
        """fc_log_std should NOT have priors (aleatoric only)."""
        import pyro.poutine as poutine
        from src.models.heads.bayesian_head import BayesianPredictionHead

        head = BayesianPredictionHead(d_input=64, d_hidden=32)

        x = torch.randn(4, 64)
        trace = poutine.trace(head).get_trace(x)

        sample_sites = [name for name in trace.nodes.keys()]

        # fc_log_std should not be in the sample sites
        assert "fc_log_std.weight" not in sample_sites
        assert "fc_log_std.bias" not in sample_sites

    def test_has_obs_sample_site(self):
        """Should have 'obs' sample site for likelihood."""
        import pyro.poutine as poutine
        from src.models.heads.bayesian_head import BayesianPredictionHead

        head = BayesianPredictionHead(d_input=64, d_hidden=32)

        x = torch.randn(4, 64)
        y = torch.randn(4, 1)

        trace = poutine.trace(head).get_trace(x, y=y)

        assert "obs" in trace.nodes


class TestNumericalStability:
    """Tests for numerical stability."""

    def test_large_input_values(self):
        """Should handle large input values without NaN."""
        from src.models.heads.bayesian_head import BayesianPredictionHead

        head = BayesianPredictionHead(d_input=64, d_hidden=32)

        x = torch.randn(8, 64) * 100  # Large values
        mean, std = head(x)

        assert not torch.isnan(mean).any(), "mean should not have NaN"
        assert not torch.isnan(std).any(), "std should not have NaN"
        assert not torch.isinf(mean).any(), "mean should not have Inf"
        assert not torch.isinf(std).any(), "std should not have Inf"

    def test_std_has_minimum_value(self):
        """std should have a minimum value to prevent division by zero."""
        from src.models.heads.bayesian_head import BayesianPredictionHead

        head = BayesianPredictionHead(d_input=64, d_hidden=32)

        # Use inputs that might produce very small log_std
        x = torch.zeros(8, 64)
        _, std = head(x)

        # std should be at least 1e-6 (the minimum added in implementation)
        assert torch.all(std >= 1e-6), "std should have minimum value"

    def test_small_input_values(self):
        """Should handle small input values."""
        from src.models.heads.bayesian_head import BayesianPredictionHead

        head = BayesianPredictionHead(d_input=64, d_hidden=32)

        x = torch.randn(8, 64) * 1e-6
        mean, std = head(x)

        assert not torch.isnan(mean).any()
        assert not torch.isnan(std).any()
        assert torch.all(std > 0)

    def test_negative_log_std_produces_positive_std(self):
        """Even negative log_std values should produce positive std."""
        from src.models.heads.bayesian_head import BayesianPredictionHead

        head = BayesianPredictionHead(d_input=64, d_hidden=32)

        # Initialize fc_log_std to produce very negative values
        with torch.no_grad():
            head.fc_log_std.weight.fill_(-10.0)
            head.fc_log_std.bias.fill_(-10.0)

        x = torch.randn(8, 64)
        _, std = head(x)

        assert torch.all(std > 0), "std must be positive even with negative log_std"


class TestValidation:
    """Tests for input validation."""

    # Constructor validation tests
    def test_rejects_non_positive_d_input(self):
        """Should reject d_input <= 0."""
        from src.models.heads.bayesian_head import BayesianPredictionHead

        with pytest.raises(ValueError, match="d_input must be positive"):
            BayesianPredictionHead(d_input=0)

        with pytest.raises(ValueError, match="d_input must be positive"):
            BayesianPredictionHead(d_input=-1)

    def test_rejects_non_positive_d_hidden(self):
        """Should reject d_hidden <= 0."""
        from src.models.heads.bayesian_head import BayesianPredictionHead

        with pytest.raises(ValueError, match="d_hidden must be positive"):
            BayesianPredictionHead(d_input=64, d_hidden=0)

        with pytest.raises(ValueError, match="d_hidden must be positive"):
            BayesianPredictionHead(d_input=64, d_hidden=-1)

    # Forward pass validation tests
    def test_rejects_wrong_input_dim(self):
        """Should reject input with wrong number of dimensions."""
        from src.models.heads.bayesian_head import BayesianPredictionHead

        head = BayesianPredictionHead(d_input=64, d_hidden=32)

        # 1D tensor
        x_1d = torch.randn(64)
        with pytest.raises(ValueError, match="Expected 2D input"):
            head(x_1d)

        # 3D tensor
        x_3d = torch.randn(8, 64, 1)
        with pytest.raises(ValueError, match="Expected 2D input"):
            head(x_3d)

    def test_rejects_wrong_feature_dim(self):
        """Should reject input with wrong feature dimension."""
        from src.models.heads.bayesian_head import BayesianPredictionHead

        head = BayesianPredictionHead(d_input=64, d_hidden=32)

        x = torch.randn(8, 128)  # Wrong: 128 instead of 64
        with pytest.raises(ValueError, match="Expected d_input=64"):
            head(x)

    def test_rejects_wrong_y_dim(self):
        """Should reject y with wrong number of dimensions."""
        from src.models.heads.bayesian_head import BayesianPredictionHead

        head = BayesianPredictionHead(d_input=64, d_hidden=32)

        x = torch.randn(8, 64)
        y_1d = torch.randn(8)  # Should be [8, 1]

        with pytest.raises(ValueError, match="Expected 2D y"):
            head(x, y=y_1d)

    def test_rejects_y_batch_size_mismatch(self):
        """Should reject y with different batch size than x."""
        from src.models.heads.bayesian_head import BayesianPredictionHead

        head = BayesianPredictionHead(d_input=64, d_hidden=32)

        x = torch.randn(8, 64)
        y = torch.randn(4, 1)  # Wrong batch size

        with pytest.raises(ValueError, match="Batch size mismatch"):
            head(x, y=y)

    def test_rejects_wrong_y_feature_dim(self):
        """Should reject y with wrong feature dimension."""
        from src.models.heads.bayesian_head import BayesianPredictionHead

        head = BayesianPredictionHead(d_input=64, d_hidden=32)

        x = torch.randn(8, 64)
        y = torch.randn(8, 2)  # Wrong: 2 instead of 1

        with pytest.raises(ValueError, match="Expected y feature dim=1"):
            head(x, y=y)


class TestExtraRepr:
    """Tests for extra_repr method."""

    def test_extra_repr_contains_parameters(self):
        """extra_repr should show key parameters."""
        from src.models.heads.bayesian_head import BayesianPredictionHead

        head = BayesianPredictionHead(d_input=128, d_hidden=256)

        repr_str = head.extra_repr()

        assert "d_input=128" in repr_str
        assert "d_hidden=256" in repr_str

    def test_str_contains_extra_repr(self):
        """String representation should include extra_repr info."""
        from src.models.heads.bayesian_head import BayesianPredictionHead

        head = BayesianPredictionHead(d_input=64, d_hidden=32)

        str_repr = str(head)

        assert "d_input=64" in str_repr
        assert "d_hidden=32" in str_repr


class TestGradientFlow:
    """Tests for gradient flow through the Bayesian head."""

    def test_gradients_flow_to_input(self):
        """Gradients should reach input tensor."""
        from src.models.heads.bayesian_head import BayesianPredictionHead

        head = BayesianPredictionHead(d_input=64, d_hidden=32)

        x = torch.randn(8, 64, requires_grad=True)
        mean, std = head(x)
        loss = mean.sum() + std.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_gradients_flow_to_fc_log_std(self):
        """Gradients should reach fc_log_std parameters."""
        from src.models.heads.bayesian_head import BayesianPredictionHead

        head = BayesianPredictionHead(d_input=64, d_hidden=32)

        x = torch.randn(8, 64)
        mean, std = head(x)
        loss = mean.sum() + std.sum()
        loss.backward()

        assert head.fc_log_std.weight.grad is not None
        assert head.fc_log_std.bias.grad is not None


class TestSVITraining:
    """Tests for Pyro SVI training compatibility."""

    def test_autodiagonalnormal_guide_compatible(self):
        """AutoDiagonalNormal guide should work with the model."""
        from src.models.heads.bayesian_head import BayesianPredictionHead
        from pyro.infer.autoguide import AutoDiagonalNormal
        import pyro

        pyro.clear_param_store()
        head = BayesianPredictionHead(d_input=32, d_hidden=16)

        # Guide should be creatable
        guide = AutoDiagonalNormal(head)

        # Guide should have parameters
        x = torch.randn(4, 32)
        y = torch.randn(4, 1)

        # Run once to initialize
        guide(x, y)

        # Should have created variational parameters
        assert len(list(pyro.get_param_store().keys())) > 0

    def test_trace_elbo_computes(self):
        """Trace_ELBO loss should compute without error."""
        from src.models.heads.bayesian_head import BayesianPredictionHead
        from pyro.infer import SVI, Trace_ELBO
        from pyro.infer.autoguide import AutoDiagonalNormal
        from pyro.optim import Adam
        import pyro

        pyro.clear_param_store()
        head = BayesianPredictionHead(d_input=32, d_hidden=16)
        guide = AutoDiagonalNormal(head)

        svi = SVI(head, guide, Adam({"lr": 0.01}), loss=Trace_ELBO())

        x = torch.randn(8, 32)
        y = torch.randn(8, 1)

        # Should compute loss without error
        loss = svi.step(x, y)

        assert isinstance(loss, float)
        assert not math.isnan(loss)
        assert not math.isinf(loss)

    def test_svi_loss_decreases(self):
        """SVI loss should decrease over training steps."""
        from src.models.heads.bayesian_head import BayesianPredictionHead
        from pyro.infer import SVI, Trace_ELBO
        from pyro.infer.autoguide import AutoDiagonalNormal
        from pyro.optim import Adam
        import pyro

        pyro.clear_param_store()
        head = BayesianPredictionHead(d_input=32, d_hidden=16)
        guide = AutoDiagonalNormal(head)

        svi = SVI(head, guide, Adam({"lr": 0.01}), loss=Trace_ELBO())

        # Generate simple learnable pattern
        torch.manual_seed(42)
        x = torch.randn(32, 32)
        y = x[:, 0:1] * 0.5 + 0.1  # Simple linear relationship

        # Collect losses
        losses = []
        for _ in range(50):
            loss = svi.step(x, y)
            losses.append(loss)

        # Loss should generally decrease (compare first 10 avg vs last 10 avg)
        early_avg = sum(losses[:10]) / 10
        late_avg = sum(losses[-10:]) / 10

        assert late_avg < early_avg, f"Loss did not decrease: early={early_avg:.2f}, late={late_avg:.2f}"

    def test_predictive_sampling(self):
        """Predictive sampling should work for inference."""
        from src.models.heads.bayesian_head import BayesianPredictionHead
        from pyro.infer import SVI, Trace_ELBO, Predictive
        from pyro.infer.autoguide import AutoDiagonalNormal
        from pyro.optim import Adam
        import pyro

        pyro.clear_param_store()
        head = BayesianPredictionHead(d_input=32, d_hidden=16)
        guide = AutoDiagonalNormal(head)

        svi = SVI(head, guide, Adam({"lr": 0.01}), loss=Trace_ELBO())

        # Train briefly
        x_train = torch.randn(16, 32)
        y_train = torch.randn(16, 1)
        for _ in range(10):
            svi.step(x_train, y_train)

        # Create predictive
        predictive = Predictive(head, guide=guide, num_samples=20)

        # Sample predictions
        x_test = torch.randn(4, 32)
        samples = predictive(x_test)

        # Should have 'obs' samples
        assert 'obs' in samples
        assert samples['obs'].shape == (20, 4, 1)  # [num_samples, batch, 1]

    def test_multiple_training_runs_independent(self):
        """Multiple training runs with clear_param_store should be independent."""
        from src.models.heads.bayesian_head import BayesianPredictionHead
        from pyro.infer import SVI, Trace_ELBO
        from pyro.infer.autoguide import AutoDiagonalNormal
        from pyro.optim import Adam
        import pyro

        losses_run1 = []
        losses_run2 = []

        for losses_list in [losses_run1, losses_run2]:
            pyro.clear_param_store()
            torch.manual_seed(123)  # Same seed for reproducibility

            head = BayesianPredictionHead(d_input=16, d_hidden=8)
            guide = AutoDiagonalNormal(head)
            svi = SVI(head, guide, Adam({"lr": 0.01}), loss=Trace_ELBO())

            x = torch.randn(8, 16)
            y = torch.randn(8, 1)

            for _ in range(5):
                loss = svi.step(x, y)
                losses_list.append(loss)

        # Both runs should produce similar loss trajectories
        for l1, l2 in zip(losses_run1, losses_run2):
            assert abs(l1 - l2) < 1e-3, "Training runs not reproducible"


class TestDataDrivenPrior:
    """Tests for data-driven prior on fc_mean.bias."""

    def test_default_target_mean_is_zero(self):
        """Default target_mean should be 0.0 (backward compatibility)."""
        from src.models.heads.bayesian_head import BayesianPredictionHead

        head = BayesianPredictionHead(d_input=64, d_hidden=32)

        assert head.target_mean == 0.0

    def test_custom_target_mean(self):
        """target_mean=-0.89 should shift the fc_mean.bias prior loc."""
        import pyro.poutine as poutine
        from src.models.heads.bayesian_head import BayesianPredictionHead

        head = BayesianPredictionHead(d_input=64, d_hidden=32, target_mean=-0.89)

        assert head.target_mean == -0.89

        # Trace the model to inspect the fc_mean.bias prior distribution
        x = torch.randn(4, 64)
        trace = poutine.trace(head).get_trace(x)

        bias_node = trace.nodes["fc_mean.bias"]
        prior_dist = bias_node["fn"]
        # The prior loc should be shifted to -0.89
        assert torch.allclose(prior_dist.base_dist.loc, torch.tensor([-0.89])), \
            f"Expected prior loc=-0.89, got {prior_dist.base_dist.loc}"

    def test_other_priors_unchanged(self):
        """fc1, fc2, fc_mean.weight priors should remain N(0,1) even with custom target_mean."""
        import pyro.poutine as poutine
        from src.models.heads.bayesian_head import BayesianPredictionHead

        head = BayesianPredictionHead(d_input=64, d_hidden=32, target_mean=-0.89)

        x = torch.randn(4, 64)
        trace = poutine.trace(head).get_trace(x)

        # Check that fc1, fc2, fc_mean.weight priors are all N(0,1)
        for site_name in ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias", "fc_mean.weight"]:
            node = trace.nodes[site_name]
            prior_dist = node["fn"]
            loc = prior_dist.base_dist.loc
            scale = prior_dist.base_dist.scale
            assert torch.all(loc == 0.0), \
                f"{site_name} prior loc should be 0.0, got {loc}"
            assert torch.all(scale == 1.0), \
                f"{site_name} prior scale should be 1.0, got {scale}"
