"""
Unit tests for PseudobulkEncoder.

Tests cover:
- Basic functionality and shape validation
- Gene attention gating integration
- Temperature control
- Gradient flow
- Edge cases and error handling
"""

import pytest
import torch
import torch.nn as nn

from src.models.branches.pseudobulk_encoder import PseudobulkEncoder


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def encoder_config():
    """Standard encoder configuration for testing."""
    return {
        "n_cell_types": 31,
        "n_genes": 3000,
        "d_embed": 128,
        "mlp_hidden": [512, 256],
        "dropout": 0.1,
        "temperature": 1.0,
    }


@pytest.fixture
def small_encoder_config():
    """Small encoder for faster tests."""
    return {
        "n_cell_types": 8,
        "n_genes": 100,
        "d_embed": 32,
        "mlp_hidden": [64, 32],
        "dropout": 0.0,
        "temperature": 1.0,
    }


@pytest.fixture
def encoder(encoder_config):
    """Standard PseudobulkEncoder instance."""
    return PseudobulkEncoder(**encoder_config)


@pytest.fixture
def small_encoder(small_encoder_config):
    """Small PseudobulkEncoder for faster tests."""
    return PseudobulkEncoder(**small_encoder_config)


# ============================================================================
# Basic Functionality Tests
# ============================================================================


class TestBasicFunctionality:
    """Test basic encoder operations."""

    def test_initialization(self, encoder_config):
        """Test encoder initializes correctly."""
        encoder = PseudobulkEncoder(**encoder_config)
        assert encoder.n_cell_types == encoder_config["n_cell_types"]
        assert encoder.n_genes == encoder_config["n_genes"]
        assert encoder.d_embed == encoder_config["d_embed"]
        assert encoder.mlp_hidden == encoder_config["mlp_hidden"]

    def test_forward_shape(self, small_encoder, small_encoder_config):
        """Test forward pass produces correct output shape."""
        batch_size = 4
        x = torch.randn(
            batch_size,
            small_encoder_config["n_cell_types"],
            small_encoder_config["n_genes"],
        )
        output = small_encoder(x)

        expected_shape = (
            batch_size,
            small_encoder_config["n_cell_types"],
            small_encoder_config["d_embed"],
        )
        assert output.shape == expected_shape

    def test_forward_batch_sizes(self, small_encoder, small_encoder_config):
        """Test forward pass works with various batch sizes."""
        for batch_size in [1, 2, 8, 16]:
            x = torch.randn(
                batch_size,
                small_encoder_config["n_cell_types"],
                small_encoder_config["n_genes"],
            )
            output = small_encoder(x)
            assert output.shape[0] == batch_size

    def test_default_mlp_hidden(self):
        """Test default MLP hidden dimensions are used when not specified."""
        encoder = PseudobulkEncoder(
            n_cell_types=31,
            n_genes=3000,
            d_embed=128,
        )
        assert encoder.mlp_hidden == [512, 256]

    def test_custom_mlp_hidden(self):
        """Test custom MLP hidden dimensions."""
        custom_hidden = [256, 128, 64]
        encoder = PseudobulkEncoder(
            n_cell_types=31,
            n_genes=1000,
            d_embed=64,
            mlp_hidden=custom_hidden,
        )
        assert encoder.mlp_hidden == custom_hidden


# ============================================================================
# Gene Attention Gate Integration Tests
# ============================================================================


class TestGeneAttentionIntegration:
    """Test gene attention gate integration."""

    def test_gene_gate_exists(self, small_encoder):
        """Test gene attention gate is properly initialized."""
        assert hasattr(small_encoder, "gene_gate")
        assert small_encoder.gene_gate is not None

    def test_gene_weights_shape(self, small_encoder, small_encoder_config):
        """Test gene weights have correct shape."""
        weights = small_encoder.get_gene_weights()
        expected_shape = (
            small_encoder_config["n_cell_types"],
            small_encoder_config["n_genes"],
        )
        assert weights.shape == expected_shape

    def test_gene_weights_sum_to_one(self, small_encoder):
        """Test gene weights sum to 1 per cell type."""
        weights = small_encoder.get_gene_weights()
        row_sums = weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_get_top_genes(self, small_encoder, small_encoder_config):
        """Test top genes extraction."""
        k = 10
        top_genes = small_encoder.get_top_genes_per_cell_type(k=k)

        assert len(top_genes) == small_encoder_config["n_cell_types"]
        for ct_idx, genes in top_genes.items():
            assert len(genes) == k
            # Check descending order of weights
            weights = [w for _, w in genes]
            assert weights == sorted(weights, reverse=True)

    def test_get_top_genes_with_names(self, small_encoder, small_encoder_config):
        """Test top genes extraction with gene names."""
        gene_names = [f"gene_{i}" for i in range(small_encoder_config["n_genes"])]
        top_genes = small_encoder.get_top_genes_per_cell_type(k=5, gene_names=gene_names)

        for ct_idx, genes in top_genes.items():
            for gene_id, weight in genes:
                assert isinstance(gene_id, str)
                assert gene_id.startswith("gene_")


# ============================================================================
# Temperature Control Tests
# ============================================================================


class TestTemperatureControl:
    """Test temperature control functionality."""

    def test_initial_temperature(self, small_encoder, small_encoder_config):
        """Test initial temperature is set correctly."""
        assert small_encoder.temperature == small_encoder_config["temperature"]

    def test_temperature_setter(self, small_encoder):
        """Test temperature can be set."""
        new_temp = 2.0
        small_encoder.temperature = new_temp
        assert small_encoder.temperature == new_temp

    def test_temperature_affects_gate_weights(self, small_encoder_config):
        """Test temperature affects gate weight distribution."""
        # Create encoder with non-uniform logits
        encoder = PseudobulkEncoder(**{**small_encoder_config, "temperature": 1.0})

        # Manually set non-uniform logits to test temperature effect
        with torch.no_grad():
            encoder.gene_gate.gate_logits.copy_(
                torch.randn_like(encoder.gene_gate.gate_logits)
            )

        # Get weights at different temperatures
        encoder.temperature = 5.0
        weights_high_temp = encoder.get_gene_weights().detach().clone()

        encoder.temperature = 0.1
        weights_low_temp = encoder.get_gene_weights().detach().clone()

        # High temp should give more uniform weights (higher entropy)
        entropy_high = -(weights_high_temp * torch.log(weights_high_temp + 1e-10)).sum(dim=-1).mean()
        entropy_low = -(weights_low_temp * torch.log(weights_low_temp + 1e-10)).sum(dim=-1).mean()

        assert entropy_high > entropy_low

    def test_temperature_validation(self, small_encoder):
        """Test temperature validation on setter."""
        with pytest.raises(ValueError, match="temperature must be positive"):
            small_encoder.temperature = 0.0

        with pytest.raises(ValueError, match="temperature must be positive"):
            small_encoder.temperature = -1.0


# ============================================================================
# Gradient Flow Tests
# ============================================================================


class TestGradientFlow:
    """Test gradient flow through the encoder."""

    def test_gradients_flow(self, small_encoder, small_encoder_config):
        """Test gradients flow through the encoder."""
        x = torch.randn(
            2,
            small_encoder_config["n_cell_types"],
            small_encoder_config["n_genes"],
            requires_grad=True,
        )
        output = small_encoder(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_gradients_to_gate_logits(self, small_encoder, small_encoder_config):
        """Test gradients reach gene gate logits."""
        x = torch.randn(
            2,
            small_encoder_config["n_cell_types"],
            small_encoder_config["n_genes"],
        )
        output = small_encoder(x)
        loss = output.sum()
        loss.backward()

        assert small_encoder.gene_gate.gate_logits.grad is not None
        assert not torch.all(small_encoder.gene_gate.gate_logits.grad == 0)

    def test_gradients_to_mlp_weights(self, small_encoder, small_encoder_config):
        """Test gradients reach MLP weights."""
        x = torch.randn(
            2,
            small_encoder_config["n_cell_types"],
            small_encoder_config["n_genes"],
        )
        output = small_encoder(x)
        loss = output.sum()
        loss.backward()

        # Check first linear layer has gradients
        for module in small_encoder.shared_mlp:
            if isinstance(module, nn.Linear):
                assert module.weight.grad is not None
                assert not torch.all(module.weight.grad == 0)
                break


# ============================================================================
# Edge Cases and Error Handling Tests
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_n_cell_types(self):
        """Test error on invalid n_cell_types."""
        with pytest.raises(ValueError, match="n_cell_types must be positive"):
            PseudobulkEncoder(n_cell_types=0, n_genes=100, d_embed=32)

        with pytest.raises(ValueError, match="n_cell_types must be positive"):
            PseudobulkEncoder(n_cell_types=-1, n_genes=100, d_embed=32)

    def test_invalid_n_genes(self):
        """Test error on invalid n_genes."""
        with pytest.raises(ValueError, match="n_genes must be positive"):
            PseudobulkEncoder(n_cell_types=31, n_genes=0, d_embed=32)

    def test_invalid_d_embed(self):
        """Test error on invalid d_embed."""
        with pytest.raises(ValueError, match="d_embed must be positive"):
            PseudobulkEncoder(n_cell_types=31, n_genes=100, d_embed=0)

    def test_wrong_input_dim(self, small_encoder):
        """Test error on wrong input dimensions."""
        # 2D input
        x_2d = torch.randn(4, 100)
        with pytest.raises(ValueError, match="Expected 3D input"):
            small_encoder(x_2d)

        # 4D input
        x_4d = torch.randn(4, 8, 100, 10)
        with pytest.raises(ValueError, match="Expected 3D input"):
            small_encoder(x_4d)

    def test_wrong_n_cell_types(self, small_encoder, small_encoder_config):
        """Test error on wrong number of cell types."""
        wrong_ct = small_encoder_config["n_cell_types"] + 5
        x = torch.randn(4, wrong_ct, small_encoder_config["n_genes"])
        with pytest.raises(ValueError, match="Expected .* cell types"):
            small_encoder(x)

    def test_wrong_n_genes(self, small_encoder, small_encoder_config):
        """Test error on wrong number of genes."""
        wrong_genes = small_encoder_config["n_genes"] + 50
        x = torch.randn(4, small_encoder_config["n_cell_types"], wrong_genes)
        with pytest.raises(ValueError, match="Expected .* genes"):
            small_encoder(x)

    def test_single_cell_type(self):
        """Test encoder works with single cell type."""
        encoder = PseudobulkEncoder(
            n_cell_types=1,
            n_genes=50,
            d_embed=16,
            mlp_hidden=[32],
        )
        x = torch.randn(2, 1, 50)
        output = encoder(x)
        assert output.shape == (2, 1, 16)

    def test_minimal_mlp(self):
        """Test encoder with minimal MLP (no hidden layers)."""
        encoder = PseudobulkEncoder(
            n_cell_types=4,
            n_genes=50,
            d_embed=16,
            mlp_hidden=[],
        )
        x = torch.randn(2, 4, 50)
        output = encoder(x)
        assert output.shape == (2, 4, 16)


# ============================================================================
# Numerical Stability Tests
# ============================================================================


class TestNumericalStability:
    """Test numerical stability."""

    def test_no_nan_output(self, small_encoder, small_encoder_config):
        """Test no NaN in output."""
        x = torch.randn(
            4,
            small_encoder_config["n_cell_types"],
            small_encoder_config["n_genes"],
        )
        output = small_encoder(x)
        assert not torch.isnan(output).any()

    def test_no_inf_output(self, small_encoder, small_encoder_config):
        """Test no Inf in output."""
        x = torch.randn(
            4,
            small_encoder_config["n_cell_types"],
            small_encoder_config["n_genes"],
        )
        output = small_encoder(x)
        assert not torch.isinf(output).any()

    def test_large_input_values(self, small_encoder, small_encoder_config):
        """Test stability with large input values."""
        x = torch.randn(
            2,
            small_encoder_config["n_cell_types"],
            small_encoder_config["n_genes"],
        ) * 100
        output = small_encoder(x)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_small_input_values(self, small_encoder, small_encoder_config):
        """Test stability with small input values."""
        x = torch.randn(
            2,
            small_encoder_config["n_cell_types"],
            small_encoder_config["n_genes"],
        ) * 1e-6
        output = small_encoder(x)
        assert not torch.isnan(output).any()

    def test_zero_input(self, small_encoder, small_encoder_config):
        """Test behavior with zero input."""
        x = torch.zeros(
            2,
            small_encoder_config["n_cell_types"],
            small_encoder_config["n_genes"],
        )
        output = small_encoder(x)
        assert not torch.isnan(output).any()


# ============================================================================
# Determinism Tests
# ============================================================================


class TestDeterminism:
    """Test deterministic behavior."""

    def test_eval_mode_determinism(self, small_encoder, small_encoder_config):
        """Test deterministic output in eval mode."""
        small_encoder.eval()
        x = torch.randn(
            2,
            small_encoder_config["n_cell_types"],
            small_encoder_config["n_genes"],
        )
        output1 = small_encoder(x)
        output2 = small_encoder(x)
        assert torch.allclose(output1, output2)

    def test_seed_determinism(self, small_encoder_config):
        """Test reproducibility with fixed seed."""
        torch.manual_seed(42)
        encoder1 = PseudobulkEncoder(**small_encoder_config)

        torch.manual_seed(42)
        encoder2 = PseudobulkEncoder(**small_encoder_config)

        # Weights should be identical
        for p1, p2 in zip(encoder1.parameters(), encoder2.parameters()):
            assert torch.allclose(p1, p2)


# ============================================================================
# Extra Repr Test
# ============================================================================


class TestExtraRepr:
    """Test string representation."""

    def test_extra_repr(self, small_encoder, small_encoder_config):
        """Test extra_repr contains key info."""
        repr_str = small_encoder.extra_repr()
        assert str(small_encoder_config["n_cell_types"]) in repr_str
        assert str(small_encoder_config["n_genes"]) in repr_str
        assert str(small_encoder_config["d_embed"]) in repr_str
