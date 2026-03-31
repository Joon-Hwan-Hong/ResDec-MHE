"""
Tests for src/models/components/gene_attention_gate.py

Test organization:
1. Initialization - parameter shapes, validation, defaults
2. Forward pass - shapes, correctness, input validation
3. Weight properties - sigmoid properties (bounded [0,1], independent per gene)
4. Temperature - stored for compat but is a no-op for sigmoid
5. Gradients - flow through gate and to input
6. Interpretability - top genes extraction
7. Edge cases - single gene/cell type, extreme sizes, boundary conditions
8. Numerical stability - NaN, Inf, extreme logits
9. Determinism - reproducibility
10. Device - CPU/CUDA consistency
"""

import pytest
import torch

from src.data.constants import N_CELL_TYPES


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def small_gate():
    """Small gate for fast tests."""
    from src.models.components.gene_attention_gate import GeneAttentionGate
    return GeneAttentionGate(n_cell_types=5, n_genes=10, temperature=1.0)


@pytest.fixture
def production_gate():
    """Production-sized gate (31 cell types, 3000 genes)."""
    from src.models.components.gene_attention_gate import GeneAttentionGate
    return GeneAttentionGate(n_cell_types=N_CELL_TYPES, n_genes=3000)


# =============================================================================
# 1. INITIALIZATION TESTS
# =============================================================================

class TestInitialization:
    """Tests for GeneAttentionGate initialization."""

    def test_creates_correct_shape_parameters(self):
        """Gate logits should have shape (n_cell_types, n_genes)."""
        from src.models.components.gene_attention_gate import GeneAttentionGate

        gate = GeneAttentionGate(n_cell_types=N_CELL_TYPES, n_genes=3000)
        assert gate.gate_logits.shape == (N_CELL_TYPES, 3000)

    def test_uniform_init_starts_at_zero(self):
        """Uniform initialization should start with zero logits."""
        from src.models.components.gene_attention_gate import GeneAttentionGate

        gate = GeneAttentionGate(n_cell_types=5, n_genes=10, init_uniform=True)
        assert torch.allclose(gate.gate_logits, torch.zeros(5, 10))

    def test_non_uniform_init_has_variation(self):
        """Non-uniform initialization should have small random values."""
        from src.models.components.gene_attention_gate import GeneAttentionGate

        torch.manual_seed(42)
        gate = GeneAttentionGate(n_cell_types=5, n_genes=10, init_uniform=False)
        assert not torch.allclose(gate.gate_logits, torch.zeros(5, 10))
        assert gate.gate_logits.abs().max() < 0.1  # Small initialization

    def test_default_temperature_is_one(self):
        """Default temperature should be 1.0."""
        from src.models.components.gene_attention_gate import GeneAttentionGate

        gate = GeneAttentionGate(n_cell_types=5, n_genes=10)
        assert gate.temperature == 1.0

    def test_rejects_zero_n_cell_types(self):
        """Should reject n_cell_types=0."""
        from src.models.components.gene_attention_gate import GeneAttentionGate

        with pytest.raises(ValueError, match="n_cell_types must be positive"):
            GeneAttentionGate(n_cell_types=0, n_genes=100)

    def test_rejects_negative_n_cell_types(self):
        """Should reject negative n_cell_types."""
        from src.models.components.gene_attention_gate import GeneAttentionGate

        with pytest.raises(ValueError, match="n_cell_types must be positive"):
            GeneAttentionGate(n_cell_types=-5, n_genes=100)

    def test_rejects_zero_n_genes(self):
        """Should reject n_genes=0."""
        from src.models.components.gene_attention_gate import GeneAttentionGate

        with pytest.raises(ValueError, match="n_genes must be positive"):
            GeneAttentionGate(n_cell_types=N_CELL_TYPES, n_genes=0)

    def test_rejects_zero_temperature(self):
        """Should reject temperature=0."""
        from src.models.components.gene_attention_gate import GeneAttentionGate

        with pytest.raises(ValueError, match="temperature must be positive"):
            GeneAttentionGate(n_cell_types=N_CELL_TYPES, n_genes=100, temperature=0)

    def test_rejects_negative_temperature(self):
        """Should reject negative temperature."""
        from src.models.components.gene_attention_gate import GeneAttentionGate

        with pytest.raises(ValueError, match="temperature must be positive"):
            GeneAttentionGate(n_cell_types=N_CELL_TYPES, n_genes=100, temperature=-1.0)


# =============================================================================
# 2. FORWARD PASS TESTS
# =============================================================================

class TestForwardPass:
    """Tests for GeneAttentionGate forward pass."""

    def test_output_shape_matches_input(self, small_gate):
        """Output should have same shape as input."""
        x = torch.randn(4, 5, 10)
        output = small_gate(x)
        assert output.shape == x.shape

    def test_output_is_gated_input(self, small_gate):
        """Output should be input * sigmoid(logits)."""
        x = torch.randn(2, 5, 10)
        gate_weights = small_gate.get_gate_weights()
        output = small_gate(x)

        # Sigmoid gate: output = input * sigmoid(logits), no scaling
        expected = x * gate_weights.unsqueeze(0)
        assert torch.allclose(output, expected)

    def test_batch_size_one(self, small_gate):
        """Should handle batch size of 1."""
        x = torch.randn(1, 5, 10)
        output = small_gate(x)
        assert output.shape == (1, 5, 10)

    def test_large_batch_size(self, small_gate):
        """Should handle large batch sizes."""
        x = torch.randn(256, 5, 10)
        output = small_gate(x)
        assert output.shape == (256, 5, 10)

    def test_rejects_2d_input(self, small_gate):
        """Should reject 2D input."""
        with pytest.raises(ValueError, match="Expected 3D input"):
            small_gate(torch.randn(5, 10))

    def test_rejects_4d_input(self, small_gate):
        """Should reject 4D input."""
        with pytest.raises(ValueError, match="Expected 3D input"):
            small_gate(torch.randn(2, 3, 5, 10))

    def test_rejects_wrong_cell_type_count(self, small_gate):
        """Should reject input with wrong number of cell types."""
        with pytest.raises(ValueError, match="Input shape mismatch"):
            small_gate(torch.randn(4, 3, 10))  # 3 instead of 5

    def test_rejects_wrong_gene_count(self, small_gate):
        """Should reject input with wrong number of genes."""
        with pytest.raises(ValueError, match="Input shape mismatch"):
            small_gate(torch.randn(4, 5, 20))  # 20 instead of 10

    def test_empty_batch_produces_empty_output(self, small_gate):
        """Empty batch should produce empty output."""
        x = torch.randn(0, 5, 10)
        output = small_gate(x)
        assert output.shape == (0, 5, 10)


# =============================================================================
# 3. WEIGHT PROPERTIES TESTS
# =============================================================================

class TestWeightProperties:
    """Tests for mathematical properties of sigmoid gate weights."""

    def test_weights_in_zero_one_range(self):
        """All gate weights should be in [0, 1]."""
        from src.models.components.gene_attention_gate import GeneAttentionGate

        gate = GeneAttentionGate(n_cell_types=10, n_genes=50)
        gate.gate_logits.data = torch.randn(10, 50) * 5  # Varied logits
        weights = gate.get_gate_weights()
        assert (weights >= 0).all()
        assert (weights <= 1).all()

    def test_init_weights_are_half(self):
        """At initialization (logits=0), all weights should be 0.5."""
        from src.models.components.gene_attention_gate import GeneAttentionGate

        gate = GeneAttentionGate(n_cell_types=10, n_genes=50, init_uniform=True)
        weights = gate.get_gate_weights()
        assert torch.allclose(weights, torch.full((10, 50), 0.5), atol=1e-6)

    def test_positive_logits_give_weights_above_half(self):
        """Positive logits should produce weights > 0.5."""
        from src.models.components.gene_attention_gate import GeneAttentionGate

        gate = GeneAttentionGate(n_cell_types=5, n_genes=10)
        gate.gate_logits.data = torch.abs(torch.randn(5, 10)) + 0.1  # All positive
        weights = gate.get_gate_weights()
        assert (weights > 0.5).all()

    def test_negative_logits_give_weights_below_half(self):
        """Negative logits should produce weights < 0.5."""
        from src.models.components.gene_attention_gate import GeneAttentionGate

        gate = GeneAttentionGate(n_cell_types=5, n_genes=10)
        gate.gate_logits.data = -torch.abs(torch.randn(5, 10)) - 0.1  # All negative
        weights = gate.get_gate_weights()
        assert (weights < 0.5).all()

    def test_weights_independent_per_gene(self):
        """Changing one gene's logit should not affect other genes' weights."""
        from src.models.components.gene_attention_gate import GeneAttentionGate

        gate = GeneAttentionGate(n_cell_types=3, n_genes=10)
        gate.gate_logits.data = torch.randn(3, 10)

        weights_before = gate.get_gate_weights().clone()

        # Change only gene 0's logit for cell type 0
        gate.gate_logits.data[0, 0] += 5.0
        weights_after = gate.get_gate_weights()

        # Gene 0 in cell type 0 should change
        assert not torch.allclose(weights_before[0, 0], weights_after[0, 0])
        # All other genes in cell type 0 should be unaffected
        assert torch.equal(weights_before[0, 1:], weights_after[0, 1:])
        # All genes in other cell types should be unaffected
        assert torch.equal(weights_before[1:], weights_after[1:])

    def test_weights_strictly_positive_with_moderate_logits(self):
        """With moderate logits, all sigmoid weights are strictly > 0.

        Note: With very extreme logits (< -80), float32 sigmoid CAN underflow
        to exactly 0. This test uses moderate logits where all weights stay
        positive, which covers the realistic training range.
        """
        from src.models.components.gene_attention_gate import GeneAttentionGate

        gate = GeneAttentionGate(n_cell_types=5, n_genes=10)
        gate.gate_logits.data = torch.randn(5, 10) * 5  # Moderate range
        weights = gate.get_gate_weights()
        assert (weights > 0).all()


# =============================================================================
# 4. TEMPERATURE TESTS
# =============================================================================

class TestTemperature:
    """Tests for temperature compatibility (no-op for sigmoid gate).

    Temperature is stored for backward compatibility with checkpoints and
    the TemperatureAnnealing callback, but has no effect on sigmoid output.
    """

    def test_temperature_does_not_affect_weights(self):
        """Changing temperature should NOT change sigmoid gate weights."""
        from src.models.components.gene_attention_gate import GeneAttentionGate

        gate = GeneAttentionGate(n_cell_types=5, n_genes=10, temperature=1.0)
        gate.gate_logits.data = torch.randn(5, 10)

        weights_t1 = gate.get_gate_weights().clone()
        gate.temperature = 0.1
        weights_t01 = gate.get_gate_weights()

        assert torch.equal(weights_t1, weights_t01), \
            "Temperature should be a no-op for sigmoid gate"

    def test_temperature_does_not_affect_forward(self):
        """Forward pass output should be identical regardless of temperature."""
        from src.models.components.gene_attention_gate import GeneAttentionGate

        gate = GeneAttentionGate(n_cell_types=5, n_genes=10, temperature=1.0)
        gate.gate_logits.data = torch.randn(5, 10)
        x = torch.randn(2, 5, 10)

        out_t1 = gate(x).clone()
        gate.temperature = 100.0
        out_t100 = gate(x)

        assert torch.equal(out_t1, out_t100)

    def test_temperature_setter_works(self):
        """Temperature property setter should update the stored value."""
        from src.models.components.gene_attention_gate import GeneAttentionGate

        gate = GeneAttentionGate(n_cell_types=5, n_genes=10, temperature=1.0)
        assert gate.temperature == 1.0

        gate.temperature = 2.0
        assert gate.temperature == 2.0

    def test_temperature_setter_rejects_zero(self):
        """Temperature setter should reject zero."""
        from src.models.components.gene_attention_gate import GeneAttentionGate

        gate = GeneAttentionGate(n_cell_types=5, n_genes=10)
        with pytest.raises(ValueError, match="temperature must be positive"):
            gate.temperature = 0

    def test_temperature_setter_rejects_negative(self):
        """Temperature setter should reject negative values."""
        from src.models.components.gene_attention_gate import GeneAttentionGate

        gate = GeneAttentionGate(n_cell_types=5, n_genes=10)
        with pytest.raises(ValueError, match="temperature must be positive"):
            gate.temperature = -1.0

    def test_temperature_floor_applied(self):
        """Very small temperature should be floored to 0.05."""
        from src.models.components.gene_attention_gate import GeneAttentionGate

        gate = GeneAttentionGate(n_cell_types=5, n_genes=10, temperature=1e-6)
        assert gate.temperature == pytest.approx(0.05)


# =============================================================================
# 5. GRADIENT TESTS
# =============================================================================

class TestGradients:
    """Tests for gradient computation."""

    def test_gradients_flow_to_gate_logits(self, small_gate):
        """Gradients should flow back to gate logits."""
        x = torch.randn(2, 5, 10, requires_grad=True)
        output = small_gate(x)
        loss = output.sum()
        loss.backward()

        assert small_gate.gate_logits.grad is not None
        assert small_gate.gate_logits.grad.shape == (5, 10)

    def test_gradients_flow_to_input(self, small_gate):
        """Gradients should flow back to input."""
        x = torch.randn(2, 5, 10, requires_grad=True)
        output = small_gate(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == (2, 5, 10)

    def test_gradients_are_not_nan(self, small_gate):
        """Gradients should not be NaN."""
        small_gate.gate_logits.data = torch.randn(5, 10) * 10
        x = torch.randn(2, 5, 10, requires_grad=True)

        output = small_gate(x)
        loss = output.sum()
        loss.backward()

        assert not torch.isnan(small_gate.gate_logits.grad).any()
        assert not torch.isnan(x.grad).any()

    def test_gradients_are_not_inf(self, small_gate):
        """Gradients should not be Inf."""
        x = torch.randn(2, 5, 10, requires_grad=True)
        output = small_gate(x)
        loss = output.sum()
        loss.backward()

        assert not torch.isinf(small_gate.gate_logits.grad).any()
        assert not torch.isinf(x.grad).any()

    def test_gradient_checkpointing_compatible(self, small_gate):
        """Should work with gradient checkpointing."""
        x = torch.randn(2, 5, 10, requires_grad=True)
        output = torch.utils.checkpoint.checkpoint(
            small_gate, x, use_reentrant=False
        )
        loss = output.sum()
        loss.backward()

        assert x.grad is not None


# =============================================================================
# 6. INTERPRETABILITY TESTS
# =============================================================================

class TestInterpretability:
    """Tests for interpretability methods (top genes extraction)."""

    def test_get_top_genes_returns_correct_k(self):
        """Should return exactly k genes per cell type."""
        from src.models.components.gene_attention_gate import GeneAttentionGate

        gate = GeneAttentionGate(n_cell_types=3, n_genes=20)
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

        for ct_idx in range(2):
            for gene_name, weight in top_genes[ct_idx]:
                assert isinstance(gene_name, str)
                assert gene_name in gene_names

    def test_get_top_genes_weights_sorted_descending(self):
        """Top genes should be sorted by weight (descending)."""
        from src.models.components.gene_attention_gate import GeneAttentionGate

        gate = GeneAttentionGate(n_cell_types=2, n_genes=10)
        gate.gate_logits.data = torch.randn(2, 10) * 2

        top_genes = gate.get_top_genes_per_cell_type(k=5)

        for ct_idx in range(2):
            weights = [w for _, w in top_genes[ct_idx]]
            assert weights == sorted(weights, reverse=True)

    def test_get_top_genes_k_equals_n(self):
        """k=n_genes should return all genes."""
        from src.models.components.gene_attention_gate import GeneAttentionGate

        gate = GeneAttentionGate(n_cell_types=3, n_genes=5)
        top_genes = gate.get_top_genes_per_cell_type(k=5)

        for ct_idx in range(3):
            indices = [idx for idx, _ in top_genes[ct_idx]]
            assert set(indices) == {0, 1, 2, 3, 4}

    def test_get_top_genes_k_exceeds_n(self):
        """k > n_genes should return n_genes."""
        from src.models.components.gene_attention_gate import GeneAttentionGate

        gate = GeneAttentionGate(n_cell_types=2, n_genes=5)
        top_genes = gate.get_top_genes_per_cell_type(k=100)

        for ct_idx in range(2):
            assert len(top_genes[ct_idx]) == 5


class TestGetTopGenesEdgeCases:
    """Tests for get_top_genes_per_cell_type with k<=0."""

    def test_get_top_genes_k_zero_returns_empty(self):
        """k=0 silently returns empty gene lists (torch.topk accepts k=0)."""
        from src.models.components.gene_attention_gate import GeneAttentionGate
        gate = GeneAttentionGate(n_cell_types=3, n_genes=50)
        result = gate.get_top_genes_per_cell_type(k=0)
        assert len(result) == 3
        for ct_idx in range(3):
            assert result[ct_idx] == []

    def test_get_top_genes_k_negative_raises_runtime_error(self):
        """Negative k raises RuntimeError from torch.topk."""
        from src.models.components.gene_attention_gate import GeneAttentionGate
        gate = GeneAttentionGate(n_cell_types=3, n_genes=50)
        with pytest.raises(RuntimeError):
            gate.get_top_genes_per_cell_type(k=-1)


# =============================================================================
# 7. EDGE CASES TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for boundary conditions and edge cases."""

    def test_single_gene(self):
        """Single gene at init: weight = sigmoid(0) = 0.5."""
        from src.models.components.gene_attention_gate import GeneAttentionGate

        gate = GeneAttentionGate(n_cell_types=5, n_genes=1)
        x = torch.randn(2, 5, 1)

        output = gate(x)
        weights = gate.get_gate_weights()

        assert output.shape == (2, 5, 1)
        assert torch.allclose(weights, torch.full((5, 1), 0.5))

    def test_single_cell_type(self):
        """Single cell type should work."""
        from src.models.components.gene_attention_gate import GeneAttentionGate

        gate = GeneAttentionGate(n_cell_types=1, n_genes=100)
        x = torch.randn(4, 1, 100)

        output = gate(x)
        assert output.shape == (4, 1, 100)

    def test_single_gene_single_cell_type(self):
        """Degenerate case: 1x1 gate. At init, output = x * 0.5."""
        from src.models.components.gene_attention_gate import GeneAttentionGate

        gate = GeneAttentionGate(n_cell_types=1, n_genes=1)
        x = torch.randn(2, 1, 1)

        output = gate(x)
        assert output.shape == (2, 1, 1)
        assert torch.allclose(output, x * 0.5)  # Weight is sigmoid(0) = 0.5

    def test_all_zero_input(self):
        """All-zero input should produce all-zero output."""
        from src.models.components.gene_attention_gate import GeneAttentionGate

        gate = GeneAttentionGate(n_cell_types=5, n_genes=10)
        x = torch.zeros(2, 5, 10)

        output = gate(x)
        assert torch.allclose(output, torch.zeros_like(output))

    def test_all_negative_input(self):
        """Negative input: output sign should match input sign."""
        from src.models.components.gene_attention_gate import GeneAttentionGate

        gate = GeneAttentionGate(n_cell_types=5, n_genes=10)
        x = -torch.abs(torch.randn(2, 5, 10))  # All negative

        output = gate(x)
        assert (output <= 0).all()

    def test_very_large_input_values(self):
        """Large input values should be scaled by weights."""
        from src.models.components.gene_attention_gate import GeneAttentionGate

        gate = GeneAttentionGate(n_cell_types=5, n_genes=10)
        x = torch.randn(2, 5, 10) * 1e6

        output = gate(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


# =============================================================================
# 8. NUMERICAL STABILITY TESTS
# =============================================================================

class TestNumericalStability:
    """Tests for numerical stability with extreme values."""

    def test_nan_input_propagates(self):
        """NaN in input should propagate (not crash)."""
        from src.models.components.gene_attention_gate import GeneAttentionGate

        gate = GeneAttentionGate(n_cell_types=5, n_genes=10)
        x = torch.randn(2, 5, 10)
        x[0, 2, 5] = float('nan')

        output = gate(x)

        assert torch.isnan(output[0, 2, 5])
        assert not torch.isnan(output[1]).any()

    def test_inf_input_handled(self):
        """Inf in input should not crash."""
        from src.models.components.gene_attention_gate import GeneAttentionGate

        gate = GeneAttentionGate(n_cell_types=5, n_genes=10)
        x = torch.randn(2, 5, 10)
        x[0, 0, 0] = float('inf')

        output = gate(x)
        assert output.shape == x.shape

    def test_large_positive_logits_no_overflow(self):
        """Large positive logits should saturate to 1.0, not overflow."""
        from src.models.components.gene_attention_gate import GeneAttentionGate

        gate = GeneAttentionGate(n_cell_types=5, n_genes=10)
        gate.gate_logits.data = torch.randn(5, 10) * 1000

        weights = gate.get_gate_weights()

        assert not torch.isnan(weights).any()
        assert not torch.isinf(weights).any()
        # Large positive logits -> sigmoid saturates near 1.0
        assert (weights >= 0).all() and (weights <= 1).all()

    def test_large_negative_logits_saturate_near_zero(self):
        """Large negative logits should saturate near 0, not produce NaN.

        Sigmoid is numerically stable: large negative logits saturate to 0
        (unlike softmax which can have competitive underflow issues).
        """
        from src.models.components.gene_attention_gate import GeneAttentionGate

        gate = GeneAttentionGate(n_cell_types=5, n_genes=10)
        gate.gate_logits.data = torch.ones(5, 10) * -1000

        weights = gate.get_gate_weights()

        # Should not produce NaN
        assert not torch.isnan(weights).any()
        # All weights should be near zero
        assert (weights < 1e-6).all()

    def test_mixed_extreme_logits(self):
        """Mix of very large and very small logits."""
        from src.models.components.gene_attention_gate import GeneAttentionGate

        gate = GeneAttentionGate(n_cell_types=5, n_genes=10)
        gate.gate_logits.data = torch.zeros(5, 10)
        gate.gate_logits.data[:, 0] = 1000
        gate.gate_logits.data[:, 1] = -1000

        weights = gate.get_gate_weights()

        assert not torch.isnan(weights).any()
        assert not torch.isinf(weights).any()

    def test_gene_gate_sigmoid_float16_stability(self):
        """Gene gate sigmoid should not produce NaN with float16 logits."""
        from src.models.components.gene_attention_gate import GeneAttentionGate

        gate = GeneAttentionGate(n_cell_types=2, n_genes=2000)
        gate.gate_logits.data = torch.randn(2, 2000) * 0.5  # Typical after training

        # Simulate AMP: cast logits to float16
        gate.gate_logits.data = gate.gate_logits.data.half()
        weights = gate.get_gate_weights()
        assert torch.isfinite(weights).all(), "NaN/Inf in gate weights"
        # gate_logits is nn.Parameter — get_gate_weights() casts to float32 via .float()
        # so weights are always float32 regardless of logit dtype.
        assert weights.dtype == torch.float32, "Gate weights should stay float32 for precision"


# =============================================================================
# 9. DETERMINISM TESTS
# =============================================================================

class TestDeterminism:
    """Tests for reproducibility."""

    def test_same_input_same_output(self, small_gate):
        """Same input should produce identical output."""
        small_gate.eval()
        x = torch.randn(2, 5, 10)

        with torch.no_grad():
            out1 = small_gate(x.clone())
            out2 = small_gate(x.clone())

        assert torch.equal(out1, out2)

    def test_weights_deterministic(self, small_gate):
        """Gate weights should be deterministic."""
        small_gate.gate_logits.data = torch.randn(5, 10)

        weights1 = small_gate.get_gate_weights()
        weights2 = small_gate.get_gate_weights()

        assert torch.equal(weights1, weights2)

    def test_seeded_initialization_reproducible(self):
        """Seeded initialization should be reproducible."""
        from src.models.components.gene_attention_gate import GeneAttentionGate

        torch.manual_seed(42)
        gate1 = GeneAttentionGate(n_cell_types=5, n_genes=10, init_uniform=False)

        torch.manual_seed(42)
        gate2 = GeneAttentionGate(n_cell_types=5, n_genes=10, init_uniform=False)

        assert torch.equal(gate1.gate_logits, gate2.gate_logits)


# =============================================================================
# 10. DEVICE TESTS
# =============================================================================

class TestDevice:
    """Tests for device placement and consistency."""

    def test_output_on_same_device_as_input(self, small_gate):
        """Output should be on same device as input."""
        x = torch.randn(2, 5, 10)
        output = small_gate(x)
        assert output.device == x.device

    def test_weights_on_same_device_as_parameters(self, small_gate):
        """Weights should be on same device as gate parameters."""
        weights = small_gate.get_gate_weights()
        assert weights.device == small_gate.gate_logits.device

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_forward(self):
        """Should work on CUDA."""
        from src.models.components.gene_attention_gate import GeneAttentionGate

        gate = GeneAttentionGate(n_cell_types=5, n_genes=10).cuda()
        x = torch.randn(2, 5, 10).cuda()

        output = gate(x)

        assert output.device.type == "cuda"
        assert output.shape == (2, 5, 10)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cpu_cuda_outputs_close(self):
        """CPU and CUDA outputs should be numerically close."""
        from src.models.components.gene_attention_gate import GeneAttentionGate

        gate_cpu = GeneAttentionGate(n_cell_types=5, n_genes=10)
        gate_cuda = GeneAttentionGate(n_cell_types=5, n_genes=10).cuda()
        gate_cuda.gate_logits.data = gate_cpu.gate_logits.data.cuda()

        x_cpu = torch.randn(2, 5, 10)
        x_cuda = x_cpu.cuda()

        gate_cpu.eval()
        gate_cuda.eval()

        with torch.no_grad():
            out_cpu = gate_cpu(x_cpu)
            out_cuda = gate_cuda(x_cuda)

        assert torch.allclose(out_cpu, out_cuda.cpu(), atol=1e-5)