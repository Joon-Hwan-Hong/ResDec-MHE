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

from src.data.constants import N_CELL_TYPES
from src.models.branches.cell_transformer import CellTransformer


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def transformer_config():
    """Standard transformer configuration."""
    return {
        "n_genes": 100,
        "n_cell_types": N_CELL_TYPES,
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

    def test_selection_weights_in_unit_interval(self, small_transformer):
        """Test selection weights are independent values in (0, 1)."""
        weights = small_transformer.get_selection_weights()
        assert (weights > 0).all()
        assert (weights < 1).all()

    def test_selection_temperature_property(self, small_transformer, small_config):
        """Test selection temperature property."""
        assert small_transformer.selection_temperature == small_config["selection_temperature"]

        new_temp = 2.0
        small_transformer.selection_temperature = new_temp
        assert small_transformer.selection_temperature == new_temp


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
        # Attention tensor: [B, n_cell_types, n_heads, n_seeds, max_cells]
        assert attention.shape[1] == small_config["n_cell_types"]

    def test_no_attention_by_default(self, small_transformer, sample_data):
        """Test attention is None when not requested."""
        cells, cell_mask = sample_data

        embeddings, selection_weights, attention = small_transformer(
            cells, cell_mask, return_attention=False
        )

        assert attention is None

    def test_return_attention_shape_detail(self, small_transformer, sample_data, small_config):
        """Attention tensors should have expected dimensionality."""
        cells, cell_mask = sample_data
        batch_size = cells.size(0)

        embeddings, selection_weights, attention = small_transformer(
            cells, cell_mask, return_attention=True
        )

        assert attention is not None
        # attention is now a single tensor: [B, n_cell_types, n_heads, n_pma_seeds, max_cells]
        assert attention.dim() == 5
        assert attention.shape[0] == batch_size
        assert attention.shape[1] == small_config["n_cell_types"]
        assert attention.shape[2] == small_config["n_heads"]
        assert attention.shape[3] == small_config["n_pma_seeds"]
        assert attention.shape[4] == cells.size(2)


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
            CellTransformer(n_genes=0, n_cell_types=N_CELL_TYPES)

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

    def test_dropout_train_vs_eval_differs(self):
        """Outputs should differ between train/eval with dropout > 0."""
        ct = CellTransformer(
            n_genes=50,
            n_cell_types=N_CELL_TYPES,
            d_model=32,
            n_heads=2,
            n_isab_layers=1,
            n_inducing=8,
            dropout=0.5,
        )
        x = torch.randn(2, N_CELL_TYPES, 10, 50)
        mask = torch.ones(2, N_CELL_TYPES, 10, dtype=torch.bool)

        ct.train()
        out_train = ct(x, mask)[0]

        ct.eval()
        out_eval = ct(x, mask)[0]

        assert not torch.allclose(out_train, out_eval, atol=1e-6)


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


# ============================================================================
# CT-A2: apply_selection_weights gradient blocking contract
# ============================================================================


class TestSelectionWeightGradientBlocking:
    """
    CT-A2: Verify that apply_selection_weights controls gradient flow to
    CellTypeSelector.selection_logits.

    Design contract:
        - apply_selection_weights=True  -> selection_logits receive gradients
        - apply_selection_weights=False -> selection_logits do NOT receive gradients

    This ensures ablation studies can disable cell-type selection learning
    without affecting the rest of the model.
    """

    def test_selection_weights_true_grads_flow_to_selector(self, small_config):
        """With apply_selection_weights=True, selector.selection_logits must receive gradients."""
        transformer = CellTransformer(**small_config)
        transformer.train()

        cells = torch.randn(2, small_config["n_cell_types"], 20, small_config["n_genes"])
        mask = torch.ones(2, small_config["n_cell_types"], 20, dtype=torch.bool)

        embeddings, _, _ = transformer(cells, mask, apply_selection_weights=True)
        loss = embeddings.sum()
        loss.backward()

        logits_grad = transformer.selector.selection_logits.grad
        assert logits_grad is not None, (
            "selection_logits.grad is None when apply_selection_weights=True — "
            "gradient must flow to CellTypeSelector"
        )
        assert not torch.all(logits_grad == 0), (
            "selection_logits.grad is all zeros when apply_selection_weights=True — "
            "gradient must be non-trivial"
        )

    def test_selection_weights_false_blocks_grads_to_selector(self, small_config):
        """With apply_selection_weights=False, selector.selection_logits must NOT receive gradients."""
        transformer = CellTransformer(**small_config)
        transformer.train()

        cells = torch.randn(2, small_config["n_cell_types"], 20, small_config["n_genes"])
        mask = torch.ones(2, small_config["n_cell_types"], 20, dtype=torch.bool)

        embeddings, _, _ = transformer(cells, mask, apply_selection_weights=False)
        loss = embeddings.sum()
        loss.backward()

        logits_grad = transformer.selector.selection_logits.grad
        assert logits_grad is None or torch.all(logits_grad == 0), (
            f"selection_logits received non-zero gradient when apply_selection_weights=False — "
            f"this violates the CT-A2 design contract (grad={logits_grad})"
        )

    def test_selection_weight_gradient_blocking_same_model(self, small_config):
        """
        Both modes on the same model instance: first with=False (no grad),
        then with=True (grad flows). Confirms the flag is the sole control.
        """
        transformer = CellTransformer(**small_config)
        transformer.train()

        cells = torch.randn(2, small_config["n_cell_types"], 20, small_config["n_genes"])
        mask = torch.ones(2, small_config["n_cell_types"], 20, dtype=torch.bool)

        # --- Pass 1: apply_selection_weights=False ---
        transformer.zero_grad()
        embeddings_off, _, _ = transformer(cells, mask, apply_selection_weights=False)
        loss_off = embeddings_off.sum()
        loss_off.backward()

        grad_off = transformer.selector.selection_logits.grad
        assert grad_off is None or torch.all(grad_off == 0), (
            "selection_logits should have no gradient with apply_selection_weights=False"
        )

        # --- Pass 2: apply_selection_weights=True ---
        transformer.zero_grad()
        embeddings_on, _, _ = transformer(cells, mask, apply_selection_weights=True)
        loss_on = embeddings_on.sum()
        loss_on.backward()

        grad_on = transformer.selector.selection_logits.grad
        assert grad_on is not None, (
            "selection_logits.grad is None with apply_selection_weights=True"
        )
        assert not torch.all(grad_on == 0), (
            "selection_logits.grad is all zeros with apply_selection_weights=True"
        )


# ============================================================================
# H3: Returned selection_weights should be detached
# ============================================================================


class TestSelectionWeightsDetached:
    """H3: Returned selection_weights should be detached for safety."""

    def test_returned_selection_weights_are_detached(self, small_transformer, sample_data):
        """selection_weights returned by forward() should not have grad_fn."""
        cells, cell_mask = sample_data
        embeddings, selection_weights, _ = small_transformer(cells, cell_mask)

        assert selection_weights.grad_fn is None, \
            "Returned selection_weights should be detached (no grad_fn)"


# ============================================================================
# T2: Divergent Cell Masks In Batch
# ============================================================================


class TestDivergentCellMasksInBatch:
    """T2: Test behavior when samples have divergent cell availability per type."""

    @pytest.fixture
    def divergent_config(self):
        return {
            "n_cell_types": 5,
            "n_genes": 20,
            "d_model": 16,
            "n_heads": 2,
            "n_isab_layers": 1,
            "n_inducing": 8,
        }

    @pytest.fixture
    def divergent_transformer(self, divergent_config):
        return CellTransformer(**divergent_config)

    def test_one_sample_all_masked_other_has_cells(self, divergent_transformer, divergent_config):
        """Sample 0 has all cells masked for type 0; sample 1 has valid cells.
        This triggers Tier 2 (mixed batch) in SetTransformerEncoder."""
        B, C, max_cells, G = 2, divergent_config["n_cell_types"], 30, divergent_config["n_genes"]
        cells = torch.randn(B, C, max_cells, G)
        cell_mask = torch.ones(B, C, max_cells, dtype=torch.bool)
        cell_mask[0, 0, :] = False  # Sample 0, type 0: no valid cells

        embeddings, _, _ = divergent_transformer(cells, cell_mask)
        assert torch.isfinite(embeddings).all(), "NaN/Inf in embeddings with divergent masks"
        assert embeddings.shape == (B, C, divergent_config["d_model"])

    def test_entire_sample_all_types_masked(self, divergent_transformer, divergent_config):
        """One sample has ALL cell types fully masked (total empty sample)."""
        B, C, max_cells, G = 2, divergent_config["n_cell_types"], 30, divergent_config["n_genes"]
        cells = torch.randn(B, C, max_cells, G)
        cell_mask = torch.ones(B, C, max_cells, dtype=torch.bool)
        cell_mask[0, :, :] = False  # Sample 0: everything masked

        embeddings, _, _ = divergent_transformer(cells, cell_mask)
        assert torch.isfinite(embeddings).all(), "NaN/Inf with one fully-empty sample"
        assert embeddings.shape == (B, C, divergent_config["d_model"])

    def test_gradients_flow_through_mixed_mask_batch(self, divergent_transformer, divergent_config):
        """Gradients flow to both samples even when one has empty cell types.

        Note: gradients to raw cells are very small (~1e-8) because the Set
        Transformer has 5 chained softmax attention operations that contract
        gradients.  We verify the computation graph is connected by checking
        that embeddings (the CellTransformer output) have grad_fn and that
        the gradient to cells is not None.  We use (embeddings**2).sum() as
        the loss to amplify the gradient signal above float32 precision.
        """
        B, C, max_cells, G = 2, divergent_config["n_cell_types"], 30, divergent_config["n_genes"]
        torch.manual_seed(42)
        cells = torch.randn(B, C, max_cells, G, requires_grad=True)
        cell_mask = torch.ones(B, C, max_cells, dtype=torch.bool)
        cell_mask[0, 0, :] = False

        embeddings, _, _ = divergent_transformer(cells, cell_mask)
        assert embeddings.grad_fn is not None  # Graph is connected
        loss = (embeddings ** 2).sum()  # Squared loss amplifies gradient
        loss.backward()

        assert cells.grad is not None
        assert cells.grad[1].abs().sum() > 0  # Sample 1 has gradients


# ============================================================================
# T3: NaN Input Handling
# ============================================================================


class TestNaNInputHandling:
    """T3: Test NaN in cells tensor input through CellTransformer."""

    @pytest.fixture
    def nan_config(self):
        return {
            "n_cell_types": 5,
            "n_genes": 20,
            "d_model": 16,
            "n_heads": 2,
            "n_isab_layers": 1,
            "n_inducing": 8,
        }

    @pytest.fixture
    def nan_transformer(self, nan_config):
        return CellTransformer(**nan_config)

    def test_nan_in_masked_positions_does_not_propagate(self, nan_transformer, nan_config):
        """NaN in masked (invalid) cell positions should not affect output."""
        B, C, max_cells, G = 2, nan_config["n_cell_types"], 30, nan_config["n_genes"]
        cells = torch.randn(B, C, max_cells, G)
        cell_mask = torch.ones(B, C, max_cells, dtype=torch.bool)
        cell_mask[:, :, 20:] = False
        cells[:, :, 20:, :] = float("nan")

        embeddings, _, _ = nan_transformer(cells, cell_mask)
        assert torch.isfinite(embeddings).all(), "NaN propagated from masked positions"

    def test_nan_in_valid_positions_propagates(self, nan_transformer, nan_config):
        """NaN in valid (unmasked) cells SHOULD propagate.

        This documents CURRENT expected behavior, not a permanent invariant.
        If NaN-safe attention is added, update this test to match the new contract.
        """
        B, C, max_cells, G = 2, nan_config["n_cell_types"], 30, nan_config["n_genes"]
        cells = torch.randn(B, C, max_cells, G)
        cell_mask = torch.ones(B, C, max_cells, dtype=torch.bool)
        cells[0, 0, 0, :] = float("nan")

        embeddings, _, _ = nan_transformer(cells, cell_mask)
        assert torch.isnan(embeddings).any(), "Expected NaN propagation from valid-position NaN"


# ============================================================================
# Task 4: forward_flat tests
# ============================================================================


class TestCellTransformerFlatInput:
    """Test forward_flat with flat cell representation."""

    def test_forward_flat_output_shape(self):
        """forward_flat produces correct output shape."""
        n_genes, n_types, d_model = 20, 31, 16
        ct = CellTransformer(
            n_genes=n_genes, n_cell_types=n_types, d_model=d_model,
            n_heads=2, n_isab_layers=1, n_inducing=4,
        )
        ct.eval()

        B = 2
        # Sample 0: type 0 has 5 cells, type 1 has 3 -> 8 total
        # Sample 1: type 0 has 8 cells -> 8 total
        cell_data = torch.randn(16, n_genes)  # 8+8=16 cells total
        cell_offsets = torch.zeros(B, n_types + 1, dtype=torch.long)
        # Sample 0
        cell_offsets[0, 1] = 5
        cell_offsets[0, 2:] = 8
        # Sample 1 (offset from 8)
        cell_offsets[1, :] = 8  # start at 8
        cell_offsets[1, 1:] = 16  # type 0 has 8 cells

        with torch.no_grad():
            emb, sel, attn = ct.forward_flat(cell_data, cell_offsets)

        assert emb.shape == (B, n_types, d_model)
        assert sel.shape == (n_types,)

    def test_forward_flat_matches_padded(self):
        """forward_flat produces identical output to forward with equivalent input."""
        n_genes, n_types, d_model = 20, 31, 16
        ct = CellTransformer(
            n_genes=n_genes, n_cell_types=n_types, d_model=d_model,
            n_heads=2, n_isab_layers=1, n_inducing=4,
        )
        ct.eval()

        B = 2
        # Create padded input
        cells_padded = torch.zeros(B, n_types, 10, n_genes)
        cell_mask = torch.zeros(B, n_types, 10, dtype=torch.bool)

        # Sample 0: type 0 has 5 cells, type 1 has 3
        data_0_0 = torch.randn(5, n_genes)
        data_0_1 = torch.randn(3, n_genes)
        cells_padded[0, 0, :5] = data_0_0
        cells_padded[0, 1, :3] = data_0_1
        cell_mask[0, 0, :5] = True
        cell_mask[0, 1, :3] = True

        # Sample 1: type 0 has 8 cells
        data_1_0 = torch.randn(8, n_genes)
        cells_padded[1, 0, :8] = data_1_0
        cell_mask[1, 0, :8] = True

        with torch.no_grad():
            emb_padded, sel_padded, _ = ct(cells_padded, cell_mask)

        # Create equivalent flat input
        cell_data = torch.cat([data_0_0, data_0_1, data_1_0], dim=0)  # [16, n_genes]
        cell_offsets = torch.zeros(B, n_types + 1, dtype=torch.long)
        cell_offsets[0, 1] = 5
        cell_offsets[0, 2:] = 8
        cell_offsets[1, :] = 8
        cell_offsets[1, 1:] = 16

        with torch.no_grad():
            emb_flat, sel_flat, _ = ct.forward_flat(cell_data, cell_offsets)

        torch.testing.assert_close(emb_flat, emb_padded, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(sel_flat, sel_padded)

    def test_forward_flat_gradient_flow(self):
        """Gradients flow through forward_flat to cell_data."""
        n_genes, n_types, d_model = 20, 8, 16
        ct = CellTransformer(
            n_genes=n_genes, n_cell_types=n_types, d_model=d_model,
            n_heads=2, n_isab_layers=1, n_inducing=4,
        )
        ct.train()

        cell_data = torch.randn(10, n_genes, requires_grad=True)
        cell_offsets = torch.zeros(1, n_types + 1, dtype=torch.long)
        cell_offsets[0, 1] = 5
        cell_offsets[0, 2:] = 10

        emb, _, _ = ct.forward_flat(cell_data, cell_offsets)
        loss = emb.sum()
        loss.backward()

        assert cell_data.grad is not None
        assert not torch.all(cell_data.grad == 0)
