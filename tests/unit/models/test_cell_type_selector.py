"""
Tests for src/models/components/cell_type_selector.py

Tests cover:
- Selection weights are independent per-type values in (0, 1)
- Top-k selection correctness
- Temperature effects on selection sharpness
- Input validation
"""

import pytest
import torch

from src.data.constants import N_CELL_TYPES, N_REGIONS


class TestCellTypeSelectorInit:
    """Tests for CellTypeSelector initialization."""

    def test_creates_correct_shape_parameters(self):
        """Selection logits should have shape (n_cell_types,)."""
        from src.models.components.cell_type_selector import CellTypeSelector

        selector = CellTypeSelector(n_cell_types=N_CELL_TYPES)

        assert selector.selection_logits.shape == (N_CELL_TYPES,)

    def test_uniform_init_starts_at_zero(self):
        """Uniform initialization should start with zero logits."""
        from src.models.components.cell_type_selector import CellTypeSelector

        selector = CellTypeSelector(n_cell_types=10, init_uniform=True)

        assert torch.allclose(selector.selection_logits, torch.zeros(10))

    def test_default_temperature(self):
        """Default temperature should be 1.0."""
        from src.models.components.cell_type_selector import CellTypeSelector

        selector = CellTypeSelector(n_cell_types=N_CELL_TYPES)

        assert selector.temperature == 1.0

    def test_rejects_invalid_n_cell_types(self):
        """Should reject non-positive n_cell_types."""
        from src.models.components.cell_type_selector import CellTypeSelector

        with pytest.raises(ValueError, match="n_cell_types must be positive"):
            CellTypeSelector(n_cell_types=0)

        with pytest.raises(ValueError, match="n_cell_types must be positive"):
            CellTypeSelector(n_cell_types=-5)

    def test_rejects_invalid_temperature(self):
        """Should reject non-positive temperature."""
        from src.models.components.cell_type_selector import CellTypeSelector

        with pytest.raises(ValueError, match="temperature must be positive"):
            CellTypeSelector(n_cell_types=N_CELL_TYPES, temperature=0)

        with pytest.raises(ValueError, match="temperature must be positive"):
            CellTypeSelector(n_cell_types=N_CELL_TYPES, temperature=-1.0)


class TestSelectionWeights:
    """Tests for selection weight properties."""

    def test_weights_in_unit_interval(self):
        """Selection weights should be in (0, 1) per type (sigmoid)."""
        from src.models.components.cell_type_selector import CellTypeSelector

        selector = CellTypeSelector(n_cell_types=N_CELL_TYPES)
        weights = selector.get_selection_weights()

        assert (weights > 0).all()
        assert (weights < 1).all()

    def test_uniform_init_gives_half(self):
        """Uniform init (logits=0) should give weights = 0.5 (sigmoid(0))."""
        from src.models.components.cell_type_selector import CellTypeSelector

        selector = CellTypeSelector(n_cell_types=N_CELL_TYPES)
        weights = selector.get_selection_weights()

        assert torch.allclose(weights, torch.full((N_CELL_TYPES,), 0.5), atol=1e-5)

    def test_weights_are_positive(self):
        """All selection weights should be positive."""
        from src.models.components.cell_type_selector import CellTypeSelector

        selector = CellTypeSelector(n_cell_types=20)
        # Set random logits
        selector.selection_logits.data = torch.randn(20) * 2

        weights = selector.get_selection_weights()
        assert (weights > 0).all()

    def test_weights_bounded_by_one(self):
        """All selection weights should be strictly less than 1."""
        from src.models.components.cell_type_selector import CellTypeSelector

        selector = CellTypeSelector(n_cell_types=20)
        selector.selection_logits.data = torch.randn(20) * 5

        weights = selector.get_selection_weights()
        assert (weights < 1).all()

    def test_weights_are_independent(self):
        """Each weight depends only on its own logit (sigmoid, not softmax)."""
        from src.models.components.cell_type_selector import CellTypeSelector

        selector = CellTypeSelector(n_cell_types=5)
        selector.selection_logits.data = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])

        weights = selector.get_selection_weights()

        # Higher logit → higher weight, monotonically
        for i in range(4):
            assert weights[i] < weights[i + 1]

    def test_forward_returns_weights(self):
        """forward() should return selection weights."""
        from src.models.components.cell_type_selector import CellTypeSelector

        selector = CellTypeSelector(n_cell_types=10)
        weights_forward = selector()
        weights_method = selector.get_selection_weights()

        assert torch.allclose(weights_forward, weights_method)


class TestTopKSelection:
    """Tests for top-k cell type selection."""

    def test_returns_correct_k(self):
        """Should return exactly k indices."""
        from src.models.components.cell_type_selector import CellTypeSelector

        selector = CellTypeSelector(n_cell_types=N_CELL_TYPES)
        selector.selection_logits.data = torch.randn(N_CELL_TYPES)

        selected = selector.get_selected_types(k=8)

        assert selected.shape == (8,)

    def test_returns_highest_logit_indices(self):
        """Should return indices with highest logits."""
        from src.models.components.cell_type_selector import CellTypeSelector

        selector = CellTypeSelector(n_cell_types=10)
        # Set clear ranking: type 5 > type 3 > type 7 > ...
        selector.selection_logits.data = torch.arange(10, dtype=torch.float)

        selected = selector.get_selected_types(k=3)

        # Should be indices 9, 8, 7 (highest logits)
        assert 9 in selected
        assert 8 in selected
        assert 7 in selected

    def test_rejects_k_greater_than_n(self):
        """Should reject k > n_cell_types."""
        from src.models.components.cell_type_selector import CellTypeSelector

        selector = CellTypeSelector(n_cell_types=10)

        with pytest.raises(ValueError, match="cannot exceed"):
            selector.get_selected_types(k=15)

    def test_rejects_non_positive_k(self):
        """Should reject k <= 0."""
        from src.models.components.cell_type_selector import CellTypeSelector

        selector = CellTypeSelector(n_cell_types=10)

        with pytest.raises(ValueError, match="k must be positive"):
            selector.get_selected_types(k=0)

        with pytest.raises(ValueError, match="k must be positive"):
            selector.get_selected_types(k=-1)

    def test_k_equals_n(self):
        """Should work when k == n_cell_types."""
        from src.models.components.cell_type_selector import CellTypeSelector

        selector = CellTypeSelector(n_cell_types=5)
        selected = selector.get_selected_types(k=5)

        assert selected.shape == (5,)
        # Should contain all indices
        assert set(selected.tolist()) == {0, 1, 2, 3, 4}


class TestRanking:
    """Tests for cell type ranking."""

    def test_ranking_returns_all_indices(self):
        """Ranking should return all cell type indices."""
        from src.models.components.cell_type_selector import CellTypeSelector

        selector = CellTypeSelector(n_cell_types=15)
        ranking = selector.get_ranking()

        assert ranking.shape == (15,)
        assert set(ranking.tolist()) == set(range(15))

    def test_ranking_is_sorted_by_importance(self):
        """Ranking should be sorted by logit value (descending)."""
        from src.models.components.cell_type_selector import CellTypeSelector

        selector = CellTypeSelector(n_cell_types=5)
        # Set clear ordering
        selector.selection_logits.data = torch.tensor([0.5, 0.1, 0.9, 0.3, 0.7])

        ranking = selector.get_ranking()

        # Expected order: 2 (0.9), 4 (0.7), 0 (0.5), 3 (0.3), 1 (0.1)
        assert ranking.tolist() == [2, 4, 0, 3, 1]


class TestTemperatureEffects:
    """Tests for temperature behavior."""

    def test_high_temperature_gives_uniform_weights(self):
        """High temperature should give nearly uniform weights (all ≈ 0.5)."""
        from src.models.components.cell_type_selector import CellTypeSelector

        selector = CellTypeSelector(n_cell_types=10, temperature=100.0)
        selector.selection_logits.data = torch.randn(10)

        weights = selector.get_selection_weights()
        # sigmoid(x/100) ≈ 0.5 for any reasonable x
        assert torch.allclose(weights, torch.full((10,), 0.5), atol=0.01)

    def test_low_temperature_gives_sharp_weights(self):
        """Low temperature should push weights toward 0 or 1."""
        from src.models.components.cell_type_selector import CellTypeSelector

        selector = CellTypeSelector(n_cell_types=10, temperature=0.01)
        # Positive logit → weight near 1, negative → near 0
        selector.selection_logits.data = torch.zeros(10)
        selector.selection_logits.data[3] = 1.0
        selector.selection_logits.data[7] = -1.0

        weights = selector.get_selection_weights()

        assert weights[3] > 0.99  # sigmoid(1.0/0.01) ≈ 1.0
        assert weights[7] < 0.01  # sigmoid(-1.0/0.01) ≈ 0.0

    def test_temperature_setter_works(self):
        """Temperature property setter should update correctly."""
        from src.models.components.cell_type_selector import CellTypeSelector

        selector = CellTypeSelector(n_cell_types=10, temperature=1.0)
        assert selector.temperature == 1.0

        selector.temperature = 2.0
        assert selector.temperature == 2.0

    def test_temperature_setter_rejects_invalid(self):
        """Temperature setter should reject non-positive values."""
        from src.models.components.cell_type_selector import CellTypeSelector

        selector = CellTypeSelector(n_cell_types=10)

        with pytest.raises(ValueError, match="temperature must be positive"):
            selector.temperature = 0

        with pytest.raises(ValueError, match="temperature must be positive"):
            selector.temperature = -1.0

    def test_extra_repr_contains_parameters(self):
        from src.models.components.cell_type_selector import CellTypeSelector
        selector = CellTypeSelector(n_cell_types=N_CELL_TYPES)
        repr_str = selector.extra_repr()
        assert f"n_cell_types={N_CELL_TYPES}" in repr_str
        assert "temperature=" in repr_str


class TestGradientFlow:
    """Tests for gradient computation."""

    def test_gradients_flow_through_weights(self):
        """Gradients should flow back to selection logits."""
        from src.models.components.cell_type_selector import CellTypeSelector

        selector = CellTypeSelector(n_cell_types=10)
        weights = selector.get_selection_weights()

        # Simulate a loss that depends on weights
        loss = weights.sum()
        loss.backward()

        assert selector.selection_logits.grad is not None
        assert selector.selection_logits.grad.shape == (10,)

    def test_gradients_are_nonzero(self):
        """Gradients should be non-zero for non-trivial loss."""
        from src.models.components.cell_type_selector import CellTypeSelector

        selector = CellTypeSelector(n_cell_types=5)
        selector.selection_logits.data = torch.randn(5)

        weights = selector.get_selection_weights()
        # Weight index 0 more heavily
        loss = (weights * torch.tensor([10.0, 1.0, 1.0, 1.0, 1.0])).sum()
        loss.backward()

        # Gradient should push logit[0] higher
        assert selector.selection_logits.grad[0] > 0


class TestDevicePlacement:
    """Tests for device handling."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_works_on_cuda(self):
        """Should work correctly on CUDA device."""
        from src.models.components.cell_type_selector import CellTypeSelector

        selector = CellTypeSelector(n_cell_types=10).cuda()
        weights = selector.get_selection_weights()
        selected = selector.get_selected_types(k=3)

        assert weights.device.type == "cuda"
        assert selected.device.type == "cuda"


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_single_cell_type(self):
        """Single cell type: weight = sigmoid(0) = 0.5."""
        from src.models.components.cell_type_selector import CellTypeSelector

        selector = CellTypeSelector(n_cell_types=1)
        weights = selector.get_selection_weights()
        selected = selector.get_selected_types(k=1)

        assert torch.allclose(weights, torch.tensor([0.5]))
        assert selected.item() == 0

    def test_k_equals_n(self):
        """Select all cell types."""
        from src.models.components.cell_type_selector import CellTypeSelector

        selector = CellTypeSelector(n_cell_types=5)
        selected = selector.get_selected_types(k=5)

        assert set(selected.tolist()) == {0, 1, 2, 3, 4}

    def test_k_equals_one(self):
        """Select single most important cell type."""
        from src.models.components.cell_type_selector import CellTypeSelector

        selector = CellTypeSelector(n_cell_types=10)
        selector.selection_logits.data = torch.arange(10, dtype=torch.float)

        selected = selector.get_selected_types(k=1)

        assert selected.item() == 9  # Highest logit

    def test_all_equal_logits(self):
        """All logits equal (zero): all weights = 0.5."""
        from src.models.components.cell_type_selector import CellTypeSelector

        selector = CellTypeSelector(n_cell_types=5, init_uniform=True)
        weights = selector.get_selection_weights()

        expected = torch.full((5,), 0.5)
        assert torch.allclose(weights, expected, atol=1e-5)


# =============================================================================
# NUMERICAL STABILITY
# =============================================================================

class TestNumericalStability:
    """Numerical stability tests."""

    def test_large_positive_logits(self):
        """Large positive logits should saturate at 1, not overflow."""
        from src.models.components.cell_type_selector import CellTypeSelector

        selector = CellTypeSelector(n_cell_types=10)
        selector.selection_logits.data = torch.ones(10) * 1000

        weights = selector.get_selection_weights()

        assert not torch.isnan(weights).any()
        assert not torch.isinf(weights).any()
        # sigmoid(1000) ≈ 1.0
        assert torch.allclose(weights, torch.ones(10), atol=1e-4)

    def test_large_negative_logits(self):
        """Large negative logits should saturate at 0, not produce NaN."""
        from src.models.components.cell_type_selector import CellTypeSelector

        selector = CellTypeSelector(n_cell_types=10)
        selector.selection_logits.data = torch.ones(10) * -1000

        weights = selector.get_selection_weights()

        assert not torch.isnan(weights).any()
        # sigmoid(-1000) ≈ 0.0
        assert torch.allclose(weights, torch.zeros(10), atol=1e-4)

    def test_gradient_stability(self):
        """Gradients should be stable, not NaN or Inf."""
        from src.models.components.cell_type_selector import CellTypeSelector

        selector = CellTypeSelector(n_cell_types=10)
        selector.selection_logits.data = torch.randn(10) * 5

        weights = selector.get_selection_weights()
        loss = weights.sum()
        loss.backward()

        assert not torch.isnan(selector.selection_logits.grad).any()
        assert not torch.isinf(selector.selection_logits.grad).any()

    def test_near_zero_temperature_numerical_stability(self):
        from src.models.components.cell_type_selector import CellTypeSelector
        selector = CellTypeSelector(n_cell_types=N_CELL_TYPES, temperature=1e-6)
        # Set non-uniform logits to test near-zero temperature behavior
        selector.selection_logits.data = torch.randn(N_CELL_TYPES)
        weights = selector.get_selection_weights()
        assert torch.isfinite(weights).all()
        # Near-zero T pushes sigmoid toward 0 or 1 (hard gating)
        assert (weights > 0.9).any() or (weights < 0.1).any()


# =============================================================================
# DETERMINISM
# =============================================================================

class TestDeterminism:
    """Reproducibility tests."""

    def test_same_logits_same_weights(self):
        """Same logits should produce identical weights."""
        from src.models.components.cell_type_selector import CellTypeSelector

        selector = CellTypeSelector(n_cell_types=10)
        selector.selection_logits.data = torch.randn(10)

        weights1 = selector.get_selection_weights()
        weights2 = selector.get_selection_weights()

        assert torch.equal(weights1, weights2)

    def test_same_logits_same_selection(self):
        """Same logits should produce identical selection."""
        from src.models.components.cell_type_selector import CellTypeSelector

        selector = CellTypeSelector(n_cell_types=10)
        selector.selection_logits.data = torch.randn(10)

        selected1 = selector.get_selected_types(k=5)
        selected2 = selector.get_selected_types(k=5)

        assert torch.equal(selected1, selected2)

    def test_seeded_init_reproducible(self):
        """Seeded initialization should be reproducible."""
        from src.models.components.cell_type_selector import CellTypeSelector

        torch.manual_seed(42)
        sel1 = CellTypeSelector(n_cell_types=10, init_uniform=False)

        torch.manual_seed(42)
        sel2 = CellTypeSelector(n_cell_types=10, init_uniform=False)

        assert torch.equal(sel1.selection_logits, sel2.selection_logits)