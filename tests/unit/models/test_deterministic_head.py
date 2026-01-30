"""
Tests for src/models/heads/deterministic_head.py

Test organization:
1. Initialization - MLP structure, layer dimensions, validation
2. Forward pass - output shapes, batch sizes, determinism
3. Gradient flow - gradients reach input and parameters
4. Validation - invalid inputs at constructor and forward time
5. ExtraRepr - string representation

DeterministicPredictionHead is a simple fallback for BayesianPredictionHead
when Bayesian inference is too slow or uncertainty quantification is not needed.
"""

import pytest
import torch
import torch.nn as nn


class TestInitialization:
    """Tests for DeterministicPredictionHead initialization."""

    def test_is_nn_module(self):
        """Should be an nn.Module subclass."""
        from src.models.heads.deterministic_head import DeterministicPredictionHead

        head = DeterministicPredictionHead(d_input=64, d_hidden=32)

        assert isinstance(head, nn.Module)

    def test_has_mlp(self):
        """Should have an mlp attribute."""
        from src.models.heads.deterministic_head import DeterministicPredictionHead

        head = DeterministicPredictionHead(d_input=64, d_hidden=32)

        assert hasattr(head, 'mlp')

    def test_mlp_structure(self):
        """MLP should have correct structure: Linear-GELU-Dropout-Linear-GELU-Dropout-Linear."""
        from src.models.heads.deterministic_head import DeterministicPredictionHead

        head = DeterministicPredictionHead(d_input=128, d_hidden=64)

        # Check MLP is nn.Sequential
        assert isinstance(head.mlp, nn.Sequential)

        # Check structure: Linear, GELU, Dropout, Linear, GELU, Dropout, Linear
        assert len(head.mlp) == 7
        assert isinstance(head.mlp[0], nn.Linear)
        assert isinstance(head.mlp[1], nn.GELU)
        assert isinstance(head.mlp[2], nn.Dropout)
        assert isinstance(head.mlp[3], nn.Linear)
        assert isinstance(head.mlp[4], nn.GELU)
        assert isinstance(head.mlp[5], nn.Dropout)
        assert isinstance(head.mlp[6], nn.Linear)

    def test_mlp_layer_dimensions(self):
        """MLP layers should have correct dimensions."""
        from src.models.heads.deterministic_head import DeterministicPredictionHead

        head = DeterministicPredictionHead(d_input=128, d_hidden=64)

        # First linear: d_input -> d_hidden
        assert head.mlp[0].in_features == 128
        assert head.mlp[0].out_features == 64

        # Second linear: d_hidden -> d_hidden
        assert head.mlp[3].in_features == 64
        assert head.mlp[3].out_features == 64

        # Third linear: d_hidden -> 1
        assert head.mlp[6].in_features == 64
        assert head.mlp[6].out_features == 1

    def test_output_dimension(self):
        """Final layer should output dimension 1."""
        from src.models.heads.deterministic_head import DeterministicPredictionHead

        head = DeterministicPredictionHead(d_input=64, d_hidden=32)

        # Check final linear layer outputs 1
        final_linear = head.mlp[-1]
        assert final_linear.out_features == 1

    def test_stores_d_input(self):
        """Should store d_input attribute."""
        from src.models.heads.deterministic_head import DeterministicPredictionHead

        head = DeterministicPredictionHead(d_input=128, d_hidden=64)

        assert head.d_input == 128

    def test_stores_d_hidden(self):
        """Should store d_hidden attribute."""
        from src.models.heads.deterministic_head import DeterministicPredictionHead

        head = DeterministicPredictionHead(d_input=128, d_hidden=64)

        assert head.d_hidden == 64

    def test_default_hidden_dim(self):
        """Should use d_hidden=64 by default."""
        from src.models.heads.deterministic_head import DeterministicPredictionHead

        head = DeterministicPredictionHead(d_input=128)

        assert head.d_hidden == 64

    def test_custom_hidden_dim(self):
        """Should accept custom d_hidden."""
        from src.models.heads.deterministic_head import DeterministicPredictionHead

        head = DeterministicPredictionHead(d_input=128, d_hidden=256)

        assert head.d_hidden == 256

    def test_default_dropout(self):
        """Should use dropout=0.1 by default."""
        from src.models.heads.deterministic_head import DeterministicPredictionHead

        head = DeterministicPredictionHead(d_input=128)

        assert head.dropout_rate == 0.1

    def test_custom_dropout(self):
        """Should accept custom dropout rate."""
        from src.models.heads.deterministic_head import DeterministicPredictionHead

        head = DeterministicPredictionHead(d_input=128, dropout=0.5)

        assert head.dropout_rate == 0.5

    def test_zero_dropout(self):
        """Should accept dropout=0.0 (no dropout)."""
        from src.models.heads.deterministic_head import DeterministicPredictionHead

        head = DeterministicPredictionHead(d_input=128, dropout=0.0)

        assert head.dropout_rate == 0.0

    def test_dropout_layers_have_correct_rate(self):
        """Dropout layers in MLP should use the specified dropout rate."""
        from src.models.heads.deterministic_head import DeterministicPredictionHead

        head = DeterministicPredictionHead(d_input=128, d_hidden=64, dropout=0.3)

        dropout_layers = [m for m in head.mlp if isinstance(m, nn.Dropout)]
        assert len(dropout_layers) == 2
        for layer in dropout_layers:
            assert layer.p == 0.3


class TestForwardPass:
    """Tests for DeterministicPredictionHead forward pass."""

    def test_output_shape(self):
        """Forward should return tensor of shape [B, 1]."""
        from src.models.heads.deterministic_head import DeterministicPredictionHead

        head = DeterministicPredictionHead(d_input=64, d_hidden=32)

        x = torch.randn(8, 64)
        output = head(x)

        assert output.shape == (8, 1)

    def test_different_batch_sizes(self):
        """Should work with various batch sizes."""
        from src.models.heads.deterministic_head import DeterministicPredictionHead

        head = DeterministicPredictionHead(d_input=64, d_hidden=32)

        for B in [1, 2, 8, 16, 32]:
            x = torch.randn(B, 64)
            output = head(x)

            assert output.shape == (B, 1)

    def test_deterministic(self):
        """Same input should produce same output (deterministic)."""
        from src.models.heads.deterministic_head import DeterministicPredictionHead

        head = DeterministicPredictionHead(d_input=64, d_hidden=32)
        head.eval()  # Set to eval mode to be sure

        x = torch.randn(8, 64)

        output1 = head(x)
        output2 = head(x)

        assert torch.allclose(output1, output2), "Outputs should be identical for same input"

    def test_output_dtype(self):
        """Output should have same dtype as input."""
        from src.models.heads.deterministic_head import DeterministicPredictionHead

        head = DeterministicPredictionHead(d_input=64, d_hidden=32)

        x = torch.randn(8, 64, dtype=torch.float32)
        output = head(x)

        assert output.dtype == torch.float32

    def test_output_device(self):
        """Output should be on same device as input."""
        from src.models.heads.deterministic_head import DeterministicPredictionHead

        head = DeterministicPredictionHead(d_input=64, d_hidden=32)

        x = torch.randn(8, 64)
        output = head(x)

        assert output.device == x.device

    def test_dropout_active_in_train_mode(self):
        """Dropout should cause stochastic outputs in train mode (with non-zero dropout)."""
        from src.models.heads.deterministic_head import DeterministicPredictionHead

        head = DeterministicPredictionHead(d_input=64, d_hidden=32, dropout=0.5)
        head.train()

        x = torch.randn(8, 64)

        # Run many times; with dropout=0.5 outputs should differ across runs
        outputs = [head(x) for _ in range(10)]
        all_same = all(torch.allclose(outputs[0], o) for o in outputs[1:])

        assert not all_same, "Dropout should cause stochastic outputs in train mode"

    def test_deterministic_in_eval_mode(self):
        """Outputs should be deterministic in eval mode (dropout disabled)."""
        from src.models.heads.deterministic_head import DeterministicPredictionHead

        head = DeterministicPredictionHead(d_input=64, d_hidden=32, dropout=0.5)
        head.eval()

        x = torch.randn(8, 64)

        output1 = head(x)
        output2 = head(x)

        assert torch.allclose(output1, output2), "Outputs should be identical in eval mode"

    def test_zero_dropout_deterministic_in_train_mode(self):
        """With dropout=0, outputs should be deterministic even in train mode."""
        from src.models.heads.deterministic_head import DeterministicPredictionHead

        head = DeterministicPredictionHead(d_input=64, d_hidden=32, dropout=0.0)
        head.train()

        x = torch.randn(8, 64)

        output1 = head(x)
        output2 = head(x)

        assert torch.allclose(output1, output2), "Zero dropout should be deterministic"


class TestGradientFlow:
    """Tests for gradient flow through the deterministic head."""

    def test_gradients_flow_to_input(self):
        """Gradients should reach input tensor."""
        from src.models.heads.deterministic_head import DeterministicPredictionHead

        head = DeterministicPredictionHead(d_input=64, d_hidden=32)

        x = torch.randn(8, 64, requires_grad=True)
        output = head(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_gradients_flow_to_parameters(self):
        """Gradients should reach all MLP parameters."""
        from src.models.heads.deterministic_head import DeterministicPredictionHead

        head = DeterministicPredictionHead(d_input=64, d_hidden=32)

        x = torch.randn(8, 64)
        output = head(x)
        loss = output.sum()
        loss.backward()

        # Check gradients for all linear layers
        for i, layer in enumerate(head.mlp):
            if isinstance(layer, nn.Linear):
                assert layer.weight.grad is not None, f"Layer {i} weight should have grad"
                assert layer.bias.grad is not None, f"Layer {i} bias should have grad"
                assert not torch.all(layer.weight.grad == 0), f"Layer {i} weight grad should not be all zeros"

    def test_gradients_flow_through_all_layers(self):
        """Check gradients reach first layer (full backprop through network)."""
        from src.models.heads.deterministic_head import DeterministicPredictionHead

        head = DeterministicPredictionHead(d_input=64, d_hidden=32)

        x = torch.randn(8, 64)
        output = head(x)
        loss = output.sum()
        loss.backward()

        # First linear layer (index 0)
        first_linear = head.mlp[0]
        assert first_linear.weight.grad is not None
        assert not torch.all(first_linear.weight.grad == 0)


class TestValidation:
    """Tests for input validation."""

    # Constructor validation tests
    def test_rejects_non_positive_d_input(self):
        """Should reject d_input <= 0."""
        from src.models.heads.deterministic_head import DeterministicPredictionHead

        with pytest.raises(ValueError, match="d_input must be positive"):
            DeterministicPredictionHead(d_input=0)

        with pytest.raises(ValueError, match="d_input must be positive"):
            DeterministicPredictionHead(d_input=-1)

    def test_rejects_non_positive_d_hidden(self):
        """Should reject d_hidden <= 0."""
        from src.models.heads.deterministic_head import DeterministicPredictionHead

        with pytest.raises(ValueError, match="d_hidden must be positive"):
            DeterministicPredictionHead(d_input=64, d_hidden=0)

        with pytest.raises(ValueError, match="d_hidden must be positive"):
            DeterministicPredictionHead(d_input=64, d_hidden=-1)

    def test_rejects_negative_dropout(self):
        """Should reject dropout < 0."""
        from src.models.heads.deterministic_head import DeterministicPredictionHead

        with pytest.raises(ValueError, match="dropout must be in"):
            DeterministicPredictionHead(d_input=64, dropout=-0.1)

    def test_rejects_dropout_one(self):
        """Should reject dropout >= 1."""
        from src.models.heads.deterministic_head import DeterministicPredictionHead

        with pytest.raises(ValueError, match="dropout must be in"):
            DeterministicPredictionHead(d_input=64, dropout=1.0)

    def test_rejects_dropout_greater_than_one(self):
        """Should reject dropout > 1."""
        from src.models.heads.deterministic_head import DeterministicPredictionHead

        with pytest.raises(ValueError, match="dropout must be in"):
            DeterministicPredictionHead(d_input=64, dropout=1.5)

    # Forward pass validation tests
    def test_rejects_wrong_input_dim(self):
        """Should reject input with wrong number of dimensions."""
        from src.models.heads.deterministic_head import DeterministicPredictionHead

        head = DeterministicPredictionHead(d_input=64, d_hidden=32)

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
        from src.models.heads.deterministic_head import DeterministicPredictionHead

        head = DeterministicPredictionHead(d_input=64, d_hidden=32)

        x = torch.randn(8, 128)  # Wrong: 128 instead of 64
        with pytest.raises(ValueError, match="Expected d_input=64"):
            head(x)


class TestExtraRepr:
    """Tests for extra_repr method."""

    def test_extra_repr_contains_parameters(self):
        """extra_repr should show key parameters."""
        from src.models.heads.deterministic_head import DeterministicPredictionHead

        head = DeterministicPredictionHead(d_input=128, d_hidden=256, dropout=0.2)

        repr_str = head.extra_repr()

        assert "d_input=128" in repr_str
        assert "d_hidden=256" in repr_str
        assert "dropout=0.2" in repr_str

    def test_str_contains_extra_repr(self):
        """String representation should include extra_repr info."""
        from src.models.heads.deterministic_head import DeterministicPredictionHead

        head = DeterministicPredictionHead(d_input=64, d_hidden=32)

        str_repr = str(head)

        assert "d_input=64" in str_repr
        assert "d_hidden=32" in str_repr
        assert "dropout=0.1" in str_repr


class TestNumericalStability:
    """Tests for numerical stability."""

    def test_large_input_values(self):
        """Should handle large input values without NaN."""
        from src.models.heads.deterministic_head import DeterministicPredictionHead

        head = DeterministicPredictionHead(d_input=64, d_hidden=32)

        x = torch.randn(8, 64) * 100  # Large values
        output = head(x)

        assert not torch.isnan(output).any(), "output should not have NaN"
        assert not torch.isinf(output).any(), "output should not have Inf"

    def test_small_input_values(self):
        """Should handle small input values."""
        from src.models.heads.deterministic_head import DeterministicPredictionHead

        head = DeterministicPredictionHead(d_input=64, d_hidden=32)

        x = torch.randn(8, 64) * 1e-6
        output = head(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_zero_input(self):
        """Should handle zero input."""
        from src.models.heads.deterministic_head import DeterministicPredictionHead

        head = DeterministicPredictionHead(d_input=64, d_hidden=32)

        x = torch.zeros(8, 64)
        output = head(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
