"""
Tests for src/models/components/gene_attention_gate.py

Tests cover:
- Output shape correctness
- Gate weights sum to 1 per cell type
- Temperature affects attention sharpness
- Gradient flow through gating
- Input validation
"""

import numpy as np
import pytest
import torch


class TestGeneAttentionGateInit:
    """Tests for GeneAttentionGate initialization."""

    def test_creates_correct_shape_parameters(self):
        """Gate logits should have shape (n_cell_types, n_genes)."""
        from src.models.components.gene_attention_gate import GeneAttentionGate

        gate = GeneAttentionGate(n_cell_types=31, n_genes=3000)

        assert gate.gate_logits.shape == (31, 3000)

    def test_uniform_init_starts_at_zero(self):
        """Uniform initialization should start with zero logits."""
        from src.models.components.gene_attention_gate import GeneAttentionGate

        gate = GeneAttentionGate(n_cell_types=5, n_genes=10, init_uniform=True)

        assert torch.allclose(gate.gate_logits, torch.zeros(5, 10))

    def test_rejects_invalid_n_cell_types(self):
        """Should reject non-positive n_cell_types."""
        from src.models.components.gene_attention_gate import GeneAttentionGate

        with pytest.raises(ValueError, match="n_cell_types must be positive"):
            GeneAttentionGate(n_cell_types=0, n_genes=100)

        with pytest.raises(ValueError, match="n_cell_types must be positive"):
            GeneAttentionGate(n_cell_types=-5, n_genes=100)

    def test_rejects_invalid_n_genes(self):
        """Should reject non-positive n_genes."""
        from src.models.components.gene_attention_gate import GeneAttentionGate

        with pytest.raises(ValueError, match="n_genes must be positive"):
            GeneAttentionGate(n_cell_types=31, n_genes=0)

    def test_rejects_invalid_temperature(self):
        """Should reject non-positive temperature."""
        from src.models.components.gene_attention_gate import GeneAttentionGate

        with pytest.raises(ValueError, match="temperature must be positive"):
            GeneAttentionGate(n_cell_types=31, n_genes=100, temperature=0)

        with pytest.raises(ValueError, match="temperature must be positive"):
            GeneAttentionGate(n_cell_types=31, n_genes=100, temperature=-1.0)


class TestGeneAttentionGateForward:
    """Tests for GeneAttentionGate forward pass."""

    @pytest.fixture
    def gate(self):
        """Create a small gate for testing."""
        from src.models.components.gene_attention_gate import GeneAttentionGate

        return GeneAttentionGate(n_cell_types=5, n_genes=10, temperature=1.0)

    def test_output_shape_matches_input(self, gate):
        """Output should have same shape as input."""
        x = torch.randn(4, 5, 10)  # batch=4, cell_types=5, genes=10
        output = gate(x)

        assert output.shape == x.shape

    def test_output_is_gated_input(self, gate):
        """Output should be input * gate_weights."""
        x = torch.randn(2, 5, 10)
        gate_weights = gate.get_gate_weights()
        output = gate(x)

        expected = x * gate_weights.unsqueeze(0)
        assert torch.allclose(output, expected)

    def test_batch_size_one(self, gate):
        """Should handle batch size of 1."""
        x = torch.randn(1, 5, 10)
        output = gate(x)

        assert output.shape == (1, 5, 10)

    def test_rejects_wrong_dimensions(self, gate):
        """Should reject input with wrong number of dimensions."""
        with pytest.raises(ValueError, match="Expected 3D input"):
            gate(torch.randn(5, 10))  # 2D

        with pytest.raises(ValueError, match="Expected 3D input"):
            gate(torch.randn(2, 3, 5, 10))  # 4D

    def test_rejects_wrong_cell_type_count(self, gate):
        """Should reject input with wrong number of cell types."""
        with pytest.raises(ValueError, match="Input shape mismatch"):
            gate(torch.randn(4, 3, 10))  # 3 cell types instead of 5

    def test_rejects_wrong_gene_count(self, gate):
        """Should reject input with wrong number of genes."""
        with pytest.raises(ValueError, match="Input shape mismatch"):
            gate(torch.randn(4, 5, 20))  # 20 genes instead of 10


class TestGeneAttentionGateWeights:
    """Tests for gate weight properties."""

    def test_weights_sum_to_one(self):
        """Gate weights should sum to 1 for each cell type."""
        from src.models.components.gene_attention_gate import GeneAttentionGate

        gate = GeneAttentionGate(n_cell_types=31, n_genes=100)
        weights = gate.get_gate_weights()

        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(31), atol=1e-5)

    def test_weights_are_positive(self):
        """All gate weights should be non-negative."""
        from src.models.components.gene_attention_gate import GeneAttentionGate

        gate = GeneAttentionGate(n_cell_types=10, n_genes=50)
        # Set some random logits
        gate.gate_logits.data = torch.randn(10, 50) * 2

        weights = gate.get_gate_weights()
        assert (weights >= 0).all()

    def test_weights_bounded_by_one(self):
        """All gate weights should be at most 1."""
        from src.models.components.gene_attention_gate import GeneAttentionGate

        gate = GeneAttentionGate(n_cell_types=10, n_genes=50)
        gate.gate_logits.data = torch.randn(10, 50) * 5

        weights = gate.get_gate_weights()
        assert (weights <= 1).all()


class TestTemperatureEffects:
    """Tests for temperature behavior."""

    def test_high_temperature_gives_uniform_weights(self):
        """High temperature should give nearly uniform weights."""
        from src.models.components.gene_attention_gate import GeneAttentionGate

        gate = GeneAttentionGate(n_cell_types=5, n_genes=10, temperature=100.0)
        # Set varied logits
        gate.gate_logits.data = torch.randn(5, 10)

        weights = gate.get_gate_weights()
        uniform = torch.ones(5, 10) / 10

        assert torch.allclose(weights, uniform, atol=0.01)

    def test_low_temperature_gives_sharp_weights(self):
        """Low temperature should give sharp (nearly one-hot) weights."""
        from src.models.components.gene_attention_gate import GeneAttentionGate

        gate = GeneAttentionGate(n_cell_types=5, n_genes=10, temperature=0.01)
        # Set clear preference for gene 0
        gate.gate_logits.data = torch.zeros(5, 10)
        gate.gate_logits.data[:, 0] = 1.0

        weights = gate.get_gate_weights()

        # Gene 0 should have weight close to 1
        assert weights[:, 0].min() > 0.99

    def test_temperature_setter_works(self):
        """Temperature property setter should update correctly."""
        from src.models.components.gene_attention_gate import GeneAttentionGate

        gate = GeneAttentionGate(n_cell_types=5, n_genes=10, temperature=1.0)
        assert gate.temperature == 1.0

        gate.temperature = 2.0
        assert gate.temperature == 2.0

    def test_temperature_setter_rejects_invalid(self):
        """Temperature setter should reject non-positive values."""
        from src.models.components.gene_attention_gate import GeneAttentionGate

        gate = GeneAttentionGate(n_cell_types=5, n_genes=10)

        with pytest.raises(ValueError, match="temperature must be positive"):
            gate.temperature = 0

        with pytest.raises(ValueError, match="temperature must be positive"):
            gate.temperature = -1.0


class TestGradientFlow:
    """Tests for gradient computation."""

    def test_gradients_flow_through_gate(self):
        """Gradients should flow back to gate logits."""
        from src.models.components.gene_attention_gate import GeneAttentionGate

        gate = GeneAttentionGate(n_cell_types=5, n_genes=10)
        x = torch.randn(2, 5, 10, requires_grad=True)

        output = gate(x)
        loss = output.sum()
        loss.backward()

        assert gate.gate_logits.grad is not None
        assert gate.gate_logits.grad.shape == (5, 10)

    def test_gradients_flow_to_input(self):
        """Gradients should flow back to input."""
        from src.models.components.gene_attention_gate import GeneAttentionGate

        gate = GeneAttentionGate(n_cell_types=5, n_genes=10)
        x = torch.randn(2, 5, 10, requires_grad=True)

        output = gate(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == (2, 5, 10)


class TestTopGenesExtraction:
    """Tests for interpretability methods."""

    def test_get_top_genes_returns_correct_k(self):
        """Should return exactly k genes per cell type."""
        from src.models.components.gene_attention_gate import GeneAttentionGate

        gate = GeneAttentionGate(n_cell_types=3, n_genes=20)
        # Set some varied logits
        gate.gate_logits.data = torch.randn(3, 20)

        top_genes = gate.get_top_genes_per_cell_type(k=5)

        assert len(top_genes) == 3
        for ct_idx in range(3):
            assert len(top_genes[ct_idx]) == 5

    def test_get_top_genes_with_names(self):
        """Should use gene names when provided."""
        from src.models.components.gene_attention_gate import GeneAttentionGate

        gate = GeneAttentionGate(n_cell_types=2, n_genes=5)
        gene_names = ["APOE", "GFAP", "SYP", "OLIG2", "SLC17A7"]

        top_genes = gate.get_top_genes_per_cell_type(k=3, gene_names=gene_names)

        # Check that names are strings
        for ct_idx in range(2):
            for gene_name, weight in top_genes[ct_idx]:
                assert isinstance(gene_name, str)
                assert gene_name in gene_names

    def test_get_top_genes_weights_are_sorted(self):
        """Top genes should be sorted by weight (descending)."""
        from src.models.components.gene_attention_gate import GeneAttentionGate

        gate = GeneAttentionGate(n_cell_types=2, n_genes=10)
        gate.gate_logits.data = torch.randn(2, 10) * 2

        top_genes = gate.get_top_genes_per_cell_type(k=5)

        for ct_idx in range(2):
            weights = [w for _, w in top_genes[ct_idx]]
            assert weights == sorted(weights, reverse=True)