"""
Unit tests for CellTransformer.

Tests cover:
- Basic functionality and shape validation
- Cell type selection with soft attention (differentiable)
- Attention extraction
- Gradient flow (including to selector)
- Edge cases and error handling
"""

import pytest
import torch

from src.models.branches.cell_transformer import CellTransformer


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def transformer_config():
    """Standard transformer configuration."""
    return {
        "n_genes": 100,
        "n_cell_types": 31,
        "d_model": 64,
        "n_heads": 4,
        "n_isab_layers": 2,
        "n_inducing": 16,
        "n_pma_seeds": 1,
        "dropout": 0.1,
        "selection_temperature": 1.0,
    }


@pytest.fixture
def small_config():
    """Small configuration for faster tests."""
    return {
        "n_genes": 50,
        "n_cell_types": 8,
        "d_model": 32,
        "n_heads": 2,
        "n_isab_layers": 1,
        "n_inducing": 8,
        "n_pma_seeds": 1,
        "dropout": 0.0,
        "selection_temperature": 1.0,
    }


@pytest.fixture
def transformer(transformer_config):
    """Standard CellTransformer instance."""
    return CellTransformer(**transformer_config)


@pytest.fixture
def small_transformer(small_config):
    """Small CellTransformer for faster tests."""
    return CellTransformer(**small_config)


@pytest.fixture
def sample_data(small_config):
    """Sample cell data for testing."""
    batch_size = 4
    max_cells = 100

    cells = torch.randn(
        batch_size,
        small_config["n_cell_types"],
        max_cells,
        small_config["n_genes"],
    )
    # Create mask with some valid cells per type
    cell_mask = torch.zeros(batch_size, small_config["n_cell_types"], max_cells, dtype=torch.bool)
    for b in range(batch_size):
        for ct in range(small_config["n_cell_types"]):
            n_valid = torch.randint(20, 80, (1,)).item()
            cell_mask[b, ct, :n_valid] = True

    return cells, cell_mask


# ============================================================================
# Basic Functionality Tests
# ============================================================================


class TestBasicFunctionality:
    """Test basic transformer operations."""

    def test_initialization(self, transformer_config):
        """Test transformer initializes correctly."""
        transformer = CellTransformer(**transformer_config)
        assert transformer.n_genes == transformer_config["n_genes"]
        assert transformer.n_cell_types == transformer_config["n_cell_types"]
        assert transformer.d_model == transformer_config["d_model"]

    def test_forward_shape(self, small_transformer, sample_data, small_config):
        """Test forward pass produces correct output shape."""
        cells, cell_mask = sample_data
        batch_size = cells.size(0)

        embeddings, selection_weights, _ = small_transformer(cells, cell_mask)

        # Output is for ALL cell types now (not just selected)
        expected_emb_shape = (
            batch_size,
            small_config["n_cell_types"],
            small_config["d_model"],
        )
        assert embeddings.shape == expected_emb_shape
        assert selection_weights.shape == (small_config["n_cell_types"],)

    def test_forward_without_mask(self, small_transformer, sample_data, small_config):
        """Test forward pass without cell mask."""
        cells, _ = sample_data
        batch_size = cells.size(0)

        embeddings, selection_weights, _ = small_transformer(cells, cell_mask=None)

        expected_shape = (
            batch_size,
            small_config["n_cell_types"],
            small_config["d_model"],
        )
        assert embeddings.shape == expected_shape

    def test_forward_batch_sizes(self, small_transformer, small_config):
        """Test forward with various batch sizes."""
        max_cells = 50

        for batch_size in [1, 2, 8]:
            cells = torch.randn(
                batch_size,
                small_config["n_cell_types"],
                max_cells,
                small_config["n_genes"],
            )
            embeddings, _, _ = small_transformer(cells)
            assert embeddings.shape[0] == batch_size

    def test_forward_without_selection_weights(self, small_transformer, sample_data, small_config):
        """Test forward without applying selection weights."""
        cells, cell_mask = sample_data

        emb_weighted, weights, _ = small_transformer(
            cells, cell_mask, apply_selection_weights=True
        )
        emb_unweighted, _, _ = small_transformer(
            cells, cell_mask, apply_selection_weights=False
        )

        # Weighted embeddings should be scaled version of unweighted
        expected = emb_unweighted * weights.view(1, -1, 1)
        assert torch.allclose(emb_weighted, expected, atol=1e-5)


# ============================================================================
# Cell Type Selection Tests
# ============================================================================


class TestCellTypeSelection:
    """Test cell type selection integration."""

    def test_selector_exists(self, small_transformer):
        """Test selector is properly initialized."""
        assert hasattr(small_transformer, "selector")
        assert small_transformer.selector is not None

    def test_selection_weights_shape(self, small_transformer, small_config):
        """Test selection weights have correct shape."""
        weights = small_transformer.get_selection_weights()
        assert weights.shape == (small_config["n_cell_types"],)

    def test_selection_weights_sum_to_one(self, small_transformer):
        """Test selection weights sum to 1."""
        weights = small_transformer.get_selection_weights()
        assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-5)

    def test_selection_ranking(self, small_transformer, small_config):
        """Test selection ranking returns all indices."""
        ranking = small_transformer.get_selection_ranking()
        assert ranking.shape == (small_config["n_cell_types"],)
        # All indices should be present
        assert set(ranking.tolist()) == set(range(small_config["n_cell_types"]))

    def test_selection_temperature_property(self, small_transformer, small_config):
        """Test selection temperature property."""
        assert small_transformer.selection_temperature == small_config["selection_temperature"]

        new_temp = 2.0
        small_transformer.selection_temperature = new_temp
        assert small_transformer.selection_temperature == new_temp

    def test_get_top_k_types(self, small_transformer, small_config):
        """Test getting top-k most important cell types."""
        k = 3
        top_k = small_transformer.get_top_k_types(k)
        assert top_k.shape == (k,)
        # All indices should be valid
        assert (top_k >= 0).all()
        assert (top_k < small_config["n_cell_types"]).all()


# ============================================================================
# Attention Tests
# ============================================================================


class TestAttention:
    """Test attention weight extraction."""

    def test_return_attention(self, small_transformer, sample_data, small_config):
        """Test attention weights are returned when requested."""
        cells, cell_mask = sample_data

        embeddings, selection_weights, attention = small_transformer(
            cells, cell_mask, return_attention=True
        )

        assert attention is not None
        # Attention for ALL cell types now
        assert len(attention) == small_config["n_cell_types"]

    def test_no_attention_by_default(self, small_transformer, sample_data):
        """Test attention is None when not requested."""
        cells, cell_mask = sample_data

        embeddings, selection_weights, attention = small_transformer(
            cells, cell_mask, return_attention=False
        )

        assert attention is None


# ============================================================================
# Gradient Flow Tests
# ============================================================================


class TestGradientFlow:
    """Test gradient flow through the transformer."""

    def test_gradients_flow_to_input(self, small_transformer, sample_data):
        """Test gradients flow back to input."""
        cells, cell_mask = sample_data
        cells.requires_grad = True

        embeddings, _, _ = small_transformer(cells, cell_mask)
        loss = embeddings.sum()
        loss.backward()

        assert cells.grad is not None
        assert not torch.all(cells.grad == 0)

    def test_gradients_flow_to_selector(self, small_transformer, sample_data):
        """Test gradients flow to selector (now differentiable with soft attention)."""
        cells, cell_mask = sample_data

        embeddings, _, _ = small_transformer(cells, cell_mask)
        loss = embeddings.sum()
        loss.backward()

        # With soft attention, selector logits should receive gradients
        assert small_transformer.selector.selection_logits.grad is not None
        # Gradients should be non-zero (selection weights affect output)
        assert not torch.all(small_transformer.selector.selection_logits.grad == 0)

    def test_gradients_to_set_encoder(self, small_transformer, sample_data):
        """Test gradients reach set encoder."""
        cells, cell_mask = sample_data

        embeddings, _, _ = small_transformer(cells, cell_mask)
        loss = embeddings.sum()
        loss.backward()

        # Check input projection has gradients
        has_grad = False
        for name, param in small_transformer.set_encoder.named_parameters():
            if param.requires_grad and param.grad is not None:
                if not torch.all(param.grad == 0):
                    has_grad = True
                    break
        assert has_grad


# ============================================================================
# Edge Cases and Error Handling Tests
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_n_genes(self):
        """Test error on invalid n_genes."""
        with pytest.raises(ValueError, match="n_genes must be positive"):
            CellTransformer(n_genes=0, n_cell_types=31)

    def test_invalid_n_cell_types(self):
        """Test error on invalid n_cell_types."""
        with pytest.raises(ValueError, match="n_cell_types must be positive"):
            CellTransformer(n_genes=100, n_cell_types=0)

    def test_wrong_input_dim(self, small_transformer):
        """Test error on wrong input dimensions."""
        # 3D input (missing cell type dimension)
        cells_3d = torch.randn(4, 100, 50)
        with pytest.raises(ValueError, match="Expected 4D"):
            small_transformer(cells_3d)

    def test_wrong_n_cell_types(self, small_transformer, small_config):
        """Test error on wrong number of cell types."""
        wrong_ct = small_config["n_cell_types"] + 5
        cells = torch.randn(4, wrong_ct, 100, small_config["n_genes"])
        with pytest.raises(ValueError, match="Expected .* cell types"):
            small_transformer(cells)

    def test_wrong_n_genes(self, small_transformer, small_config):
        """Test error on wrong number of genes."""
        wrong_genes = small_config["n_genes"] + 50
        cells = torch.randn(4, small_config["n_cell_types"], 100, wrong_genes)
        with pytest.raises(ValueError, match="Expected .* genes"):
            small_transformer(cells)

    def test_single_cell_type(self):
        """Test with single cell type."""
        transformer = CellTransformer(
            n_genes=50,
            n_cell_types=1,
            d_model=32,
            n_heads=2,
            n_isab_layers=1,
            n_inducing=8,
        )
        cells = torch.randn(2, 1, 50, 50)
        embeddings, weights, _ = transformer(cells)
        assert embeddings.shape == (2, 1, 32)
        assert weights.shape == (1,)


# ============================================================================
# Numerical Stability Tests
# ============================================================================


class TestNumericalStability:
    """Test numerical stability."""

    def test_no_nan_output(self, small_transformer, sample_data):
        """Test no NaN in output."""
        cells, cell_mask = sample_data
        embeddings, _, _ = small_transformer(cells, cell_mask)
        assert not torch.isnan(embeddings).any()

    def test_no_inf_output(self, small_transformer, sample_data):
        """Test no Inf in output."""
        cells, cell_mask = sample_data
        embeddings, _, _ = small_transformer(cells, cell_mask)
        assert not torch.isinf(embeddings).any()

    def test_large_input_values(self, small_transformer, small_config):
        """Test stability with large input values."""
        cells = torch.randn(2, small_config["n_cell_types"], 50, small_config["n_genes"]) * 100
        embeddings, _, _ = small_transformer(cells)
        assert not torch.isnan(embeddings).any()
        assert not torch.isinf(embeddings).any()

    def test_small_input_values(self, small_transformer, small_config):
        """Test stability with small input values."""
        cells = torch.randn(2, small_config["n_cell_types"], 50, small_config["n_genes"]) * 1e-6
        embeddings, _, _ = small_transformer(cells)
        assert not torch.isnan(embeddings).any()

    def test_sparse_mask(self, small_transformer, small_config):
        """Test with very sparse valid cells."""
        cells = torch.randn(2, small_config["n_cell_types"], 100, small_config["n_genes"])
        # Only 5 valid cells per type
        cell_mask = torch.zeros(2, small_config["n_cell_types"], 100, dtype=torch.bool)
        cell_mask[:, :, :5] = True

        embeddings, _, _ = small_transformer(cells, cell_mask)
        assert not torch.isnan(embeddings).any()

    def test_all_masked_cell_types(self, small_transformer, small_config):
        """Test with some cell types having all cells masked."""
        cells = torch.randn(2, small_config["n_cell_types"], 100, small_config["n_genes"])
        cell_mask = torch.zeros(2, small_config["n_cell_types"], 100, dtype=torch.bool)
        # Only first 3 cell types have valid cells
        cell_mask[:, :3, :50] = True

        embeddings, _, _ = small_transformer(cells, cell_mask)
        # All outputs should be finite (empty types get empty_embedding)
        assert torch.isfinite(embeddings).all()


# ============================================================================
# Determinism Tests
# ============================================================================


class TestDeterminism:
    """Test deterministic behavior."""

    def test_eval_mode_determinism(self, small_transformer, sample_data):
        """Test deterministic output in eval mode."""
        small_transformer.eval()
        cells, cell_mask = sample_data

        embeddings1, weights1, _ = small_transformer(cells, cell_mask)
        embeddings2, weights2, _ = small_transformer(cells, cell_mask)

        assert torch.equal(weights1, weights2)
        assert torch.allclose(embeddings1, embeddings2)


# ============================================================================
# Extra Repr Test
# ============================================================================


class TestExtraRepr:
    """Test string representation."""

    def test_extra_repr(self, small_transformer, small_config):
        """Test extra_repr contains key info."""
        repr_str = small_transformer.extra_repr()
        assert str(small_config["n_genes"]) in repr_str
        assert str(small_config["n_cell_types"]) in repr_str
        assert "temperature" in repr_str
