"""
Unit tests for CellTransformer.

Tests cover:
- Basic functionality and shape validation
- Attention extraction
- Gradient flow
- Edge cases and error handling
"""

import pytest
import torch

from src.data.constants import N_CELL_TYPES
from src.models.branches.cell_transformer import CellTransformer


# ============================================================================
# Helpers
# ============================================================================


def _padded_to_flat(cells_padded, cell_mask):
    """Convert padded 4D cells tensor to flat format for testing.

    Args:
        cells_padded: [B, n_types, max_cells, n_genes]
        cell_mask: [B, n_types, max_cells] bool

    Returns:
        cell_data: [total_cells, n_genes]
        cell_offsets: [B, n_types + 1] long  (global offsets into cell_data)
    """
    B, n_types, max_cells, n_genes = cells_padded.shape
    flat_parts = []
    cell_offsets = torch.zeros(B, n_types + 1, dtype=torch.long)

    cumulative = 0
    for b in range(B):
        cell_offsets[b, 0] = cumulative
        for ct in range(n_types):
            n = int(cell_mask[b, ct].sum().item())
            if n > 0:
                flat_parts.append(cells_padded[b, ct, :n])
            cumulative += n
            cell_offsets[b, ct + 1] = cumulative

    if flat_parts:
        cell_data = torch.cat(flat_parts, dim=0)
    else:
        cell_data = torch.empty(0, n_genes)
    return cell_data, cell_offsets


def _make_flat_data(batch_size, n_cell_types, cells_per_type, n_genes):
    """Create flat cell_data + cell_offsets from scratch.

    Args:
        batch_size: Number of samples
        n_cell_types: Number of cell types
        cells_per_type: int or list[int] — cells per type (uniform or per-type)
        n_genes: Number of genes

    Returns:
        cell_data: [total_cells, n_genes]
        cell_offsets: [B, n_types + 1] long
    """
    if isinstance(cells_per_type, int):
        counts = [cells_per_type] * n_cell_types
    else:
        counts = list(cells_per_type)

    total_per_sample = sum(counts)
    total = batch_size * total_per_sample
    cell_data = torch.randn(total, n_genes)

    cell_offsets = torch.zeros(batch_size, n_cell_types + 1, dtype=torch.long)
    for b in range(batch_size):
        base = b * total_per_sample
        for ct in range(n_cell_types):
            cell_offsets[b, ct + 1] = cell_offsets[b, ct] + counts[ct]
        cell_offsets[b] += base

    return cell_data, cell_offsets


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
    """Sample cell data in flat format for testing."""
    batch_size = 4
    n_types = small_config["n_cell_types"]
    n_genes = small_config["n_genes"]
    max_cells = 100

    # Build padded first, then convert to flat (to get variable counts)
    cells_padded = torch.randn(batch_size, n_types, max_cells, n_genes)
    cell_mask = torch.zeros(batch_size, n_types, max_cells, dtype=torch.bool)
    for b in range(batch_size):
        for ct in range(n_types):
            n_valid = torch.randint(20, 80, (1,)).item()
            cell_mask[b, ct, :n_valid] = True

    cell_data, cell_offsets = _padded_to_flat(cells_padded, cell_mask)
    return cell_data, cell_offsets


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
        cell_data, cell_offsets = sample_data
        batch_size = cell_offsets.size(0)

        embeddings, _ = small_transformer(cell_data, cell_offsets)

        expected_emb_shape = (
            batch_size,
            small_config["n_cell_types"],
            small_config["d_model"],
        )
        assert embeddings.shape == expected_emb_shape

    def test_forward_with_empty_cells(self, small_transformer, small_config):
        """Test forward pass with zero total cells (all types empty)."""
        B = 2
        n_types = small_config["n_cell_types"]
        n_genes = small_config["n_genes"]

        cell_data = torch.empty(0, n_genes)
        cell_offsets = torch.zeros(B, n_types + 1, dtype=torch.long)

        embeddings, _ = small_transformer(cell_data, cell_offsets)

        expected_shape = (
            B,
            small_config["n_cell_types"],
            small_config["d_model"],
        )
        assert embeddings.shape == expected_shape

    def test_forward_batch_sizes(self, small_transformer, small_config):
        """Test forward with various batch sizes."""
        for batch_size in [1, 2, 8]:
            cell_data, cell_offsets = _make_flat_data(
                batch_size, small_config["n_cell_types"], 20, small_config["n_genes"]
            )
            embeddings, _ = small_transformer(cell_data, cell_offsets)
            assert embeddings.shape[0] == batch_size



# ============================================================================
# Attention Tests
# ============================================================================


class TestAttention:
    """Test attention weight extraction."""

    def test_return_attention(self, small_transformer, sample_data, small_config):
        """Test attention weights are returned when requested."""
        cell_data, cell_offsets = sample_data

        embeddings, attention = small_transformer(
            cell_data, cell_offsets, return_attention=True
        )

        assert attention is not None
        # Attention tensor: [B, n_cell_types, n_heads, n_seeds, max_cells]
        assert attention.shape[1] == small_config["n_cell_types"]

    def test_no_attention_by_default(self, small_transformer, sample_data):
        """Test attention is None when not requested."""
        cell_data, cell_offsets = sample_data

        embeddings, attention = small_transformer(
            cell_data, cell_offsets, return_attention=False
        )

        assert attention is None

    def test_return_attention_shape_detail(self, small_transformer, sample_data, small_config):
        """Attention tensors should have expected dimensionality."""
        cell_data, cell_offsets = sample_data
        batch_size = cell_offsets.size(0)

        embeddings, attention = small_transformer(
            cell_data, cell_offsets, return_attention=True
        )

        assert attention is not None
        # attention: [B, n_cell_types, n_heads, n_pma_seeds, max_cells_in_batch]
        assert attention.dim() == 5
        assert attention.shape[0] == batch_size
        assert attention.shape[1] == small_config["n_cell_types"]
        assert attention.shape[2] == small_config["n_heads"]
        assert attention.shape[3] == small_config["n_pma_seeds"]


# ============================================================================
# Gradient Flow Tests
# ============================================================================


class TestGradientFlow:
    """Test gradient flow through the transformer."""

    def test_gradients_flow_to_input(self, small_transformer, small_config):
        """Test gradients flow back to input."""
        cell_data, cell_offsets = _make_flat_data(
            2, small_config["n_cell_types"], 20, small_config["n_genes"]
        )
        cell_data.requires_grad = True

        embeddings, _ = small_transformer(cell_data, cell_offsets)
        loss = embeddings.sum()
        loss.backward()

        assert cell_data.grad is not None
        assert not torch.all(cell_data.grad == 0)

    def test_gradients_to_set_encoder(self, small_transformer, sample_data):
        """Test gradients reach set encoder."""
        cell_data, cell_offsets = sample_data

        embeddings, _ = small_transformer(cell_data, cell_offsets)
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
        cell_data, cell_offsets = _make_flat_data(2, 1, 50, 50)
        embeddings, _ = transformer(cell_data, cell_offsets)
        assert embeddings.shape == (2, 1, 32)


# ============================================================================
# Numerical Stability Tests
# ============================================================================


class TestNumericalStability:
    """Test numerical stability."""

    def test_no_nan_output(self, small_transformer, sample_data):
        """Test no NaN in output."""
        cell_data, cell_offsets = sample_data
        embeddings, _ = small_transformer(cell_data, cell_offsets)
        assert not torch.isnan(embeddings).any()

    def test_no_inf_output(self, small_transformer, sample_data):
        """Test no Inf in output."""
        cell_data, cell_offsets = sample_data
        embeddings, _ = small_transformer(cell_data, cell_offsets)
        assert not torch.isinf(embeddings).any()

    def test_large_input_values(self, small_transformer, small_config):
        """Test stability with large input values."""
        cell_data, cell_offsets = _make_flat_data(
            2, small_config["n_cell_types"], 50, small_config["n_genes"]
        )
        cell_data = cell_data * 100
        embeddings, _ = small_transformer(cell_data, cell_offsets)
        assert not torch.isnan(embeddings).any()
        assert not torch.isinf(embeddings).any()

    def test_small_input_values(self, small_transformer, small_config):
        """Test stability with small input values."""
        cell_data, cell_offsets = _make_flat_data(
            2, small_config["n_cell_types"], 50, small_config["n_genes"]
        )
        cell_data = cell_data * 1e-6
        embeddings, _ = small_transformer(cell_data, cell_offsets)
        assert not torch.isnan(embeddings).any()

    def test_sparse_cells(self, small_transformer, small_config):
        """Test with very few valid cells per type (5 each)."""
        cell_data, cell_offsets = _make_flat_data(
            2, small_config["n_cell_types"], 5, small_config["n_genes"]
        )
        embeddings, _ = small_transformer(cell_data, cell_offsets)
        assert not torch.isnan(embeddings).any()

    def test_some_empty_cell_types(self, small_transformer, small_config):
        """Test with some cell types having zero cells."""
        B = 2
        n_types = small_config["n_cell_types"]
        n_genes = small_config["n_genes"]

        # Only first 3 cell types have cells (50 each), rest have 0
        counts = [50] * 3 + [0] * (n_types - 3)
        cell_data, cell_offsets = _make_flat_data(B, n_types, counts, n_genes)

        embeddings, _ = small_transformer(cell_data, cell_offsets)
        assert torch.isfinite(embeddings).all()


# ============================================================================
# Determinism Tests
# ============================================================================


class TestDeterminism:
    """Test deterministic behavior."""

    def test_eval_mode_determinism(self, small_transformer, sample_data):
        """Test deterministic output in eval mode."""
        small_transformer.eval()
        cell_data, cell_offsets = sample_data

        embeddings1, _ = small_transformer(cell_data, cell_offsets)
        embeddings2, _ = small_transformer(cell_data, cell_offsets)

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
        cell_data, cell_offsets = _make_flat_data(2, N_CELL_TYPES, 10, 50)

        ct.train()
        out_train = ct(cell_data, cell_offsets)[0]

        ct.eval()
        out_eval = ct(cell_data, cell_offsets)[0]

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


# ============================================================================
# T2: Divergent Cell Counts In Batch
# ============================================================================


class TestDivergentCellCountsInBatch:
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

    def test_one_sample_empty_type_other_has_cells(self, divergent_transformer, divergent_config):
        """Sample 0 has zero cells for type 0; sample 1 has valid cells.
        This triggers mixed-count behavior in the flat forward path."""
        B, C, G = 2, divergent_config["n_cell_types"], divergent_config["n_genes"]
        # Sample 0: type 0 has 0 cells, types 1-4 have 10 each
        # Sample 1: all types have 10 cells
        counts_s0 = [0, 10, 10, 10, 10]
        counts_s1 = [10, 10, 10, 10, 10]

        parts = []
        offsets = torch.zeros(B, C + 1, dtype=torch.long)
        cum = 0
        for b, counts in enumerate([counts_s0, counts_s1]):
            for ct in range(C):
                if counts[ct] > 0:
                    parts.append(torch.randn(counts[ct], G))
                offsets[b, ct + 1] = offsets[b, ct] + counts[ct]
            offsets[b] += cum
            cum += sum(counts)

        cell_data = torch.cat(parts) if parts else torch.empty(0, G)

        embeddings, _ = divergent_transformer(cell_data, offsets)
        assert torch.isfinite(embeddings).all(), "NaN/Inf in embeddings with divergent counts"
        assert embeddings.shape == (B, C, divergent_config["d_model"])

    def test_entire_sample_all_types_empty(self, divergent_transformer, divergent_config):
        """One sample has ALL cell types with zero cells (total empty sample)."""
        B, C, G = 2, divergent_config["n_cell_types"], divergent_config["n_genes"]

        # Sample 0: all empty, Sample 1: 10 cells per type
        counts_s0 = [0] * C
        counts_s1 = [10] * C

        parts = []
        offsets = torch.zeros(B, C + 1, dtype=torch.long)
        cum = 0
        for b, counts in enumerate([counts_s0, counts_s1]):
            for ct in range(C):
                if counts[ct] > 0:
                    parts.append(torch.randn(counts[ct], G))
                offsets[b, ct + 1] = offsets[b, ct] + counts[ct]
            offsets[b] += cum
            cum += sum(counts)

        cell_data = torch.cat(parts) if parts else torch.empty(0, G)

        embeddings, _ = divergent_transformer(cell_data, offsets)
        assert torch.isfinite(embeddings).all(), "NaN/Inf with one fully-empty sample"
        assert embeddings.shape == (B, C, divergent_config["d_model"])

    def test_gradients_flow_through_mixed_count_batch(self, divergent_transformer, divergent_config):
        """Gradients flow to cell_data even when one sample has empty cell types."""
        B, C, G = 2, divergent_config["n_cell_types"], divergent_config["n_genes"]
        torch.manual_seed(42)

        cell_data, cell_offsets = _make_flat_data(B, C, 10, G)
        cell_data.requires_grad = True

        embeddings, _ = divergent_transformer(cell_data, cell_offsets)
        assert embeddings.grad_fn is not None
        loss = (embeddings ** 2).sum()
        loss.backward()

        assert cell_data.grad is not None
        assert cell_data.grad.abs().sum() > 0


# ============================================================================
# T3: NaN Input Handling
# ============================================================================


class TestNaNInputHandling:
    """T3: Test NaN in cell_data input through CellTransformer."""

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

    def test_nan_in_valid_positions_propagates(self, nan_transformer, nan_config):
        """NaN in cell_data SHOULD propagate to output.

        This documents CURRENT expected behavior, not a permanent invariant.
        If NaN-safe attention is added, update this test to match the new contract.
        """
        C, G = nan_config["n_cell_types"], nan_config["n_genes"]
        cell_data, cell_offsets = _make_flat_data(2, C, 10, G)
        cell_data[0, :] = float("nan")  # First cell is NaN

        embeddings, _ = nan_transformer(cell_data, cell_offsets)
        assert torch.isnan(embeddings).any(), "Expected NaN propagation from valid-position NaN"


# ============================================================================
# Forward with flat input tests (formerly TestCellTransformerFlatInput)
# ============================================================================


class TestCellTransformerFlatInput:
    """Test forward() with flat cell representation."""

    def test_forward_output_shape(self):
        """forward() produces correct output shape."""
        n_genes, n_types, d_model = 20, 31, 16
        ct = CellTransformer(
            n_genes=n_genes, n_cell_types=n_types, d_model=d_model,
            n_heads=2, n_isab_layers=1, n_inducing=4,
        )
        ct.eval()

        B = 2
        cell_data = torch.randn(16, n_genes)
        cell_offsets = torch.zeros(B, n_types + 1, dtype=torch.long)
        # Sample 0: type 0 has 5 cells, type 1 has 3
        cell_offsets[0, 1] = 5
        cell_offsets[0, 2:] = 8
        # Sample 1 (offset from 8): type 0 has 8 cells
        cell_offsets[1, :] = 8
        cell_offsets[1, 1:] = 16

        with torch.no_grad():
            emb, attn = ct(cell_data, cell_offsets)

        assert emb.shape == (B, n_types, d_model)

    def test_forward_gradient_flow(self):
        """Gradients flow through forward() to cell_data."""
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

        emb, _ = ct(cell_data, cell_offsets)
        loss = emb.sum()
        loss.backward()

        assert cell_data.grad is not None
        assert not torch.all(cell_data.grad == 0)


# ============================================================================
# Task 5: Cell-type conditioning in CellTransformer
# ============================================================================


class TestCellTransformerCellTypeConditioning:
    """Tests for cell-type-conditioned inducing points in CellTransformer."""

    def test_condition_on_cell_type_default_true(self):
        """CellTransformer should enable cell type conditioning by default."""
        ct = CellTransformer(n_genes=100, n_cell_types=31, d_model=64, n_heads=4)
        assert ct.condition_on_cell_type is True
        for isab in ct.set_encoder.isab_layers:
            assert isab.cell_type_embed is not None

    def test_condition_on_cell_type_disabled(self):
        """Setting condition_on_cell_type=False should disable conditioning."""
        ct = CellTransformer(
            n_genes=100, n_cell_types=31, d_model=64, n_heads=4,
            condition_on_cell_type=False,
        )
        assert ct.condition_on_cell_type is False
        for isab in ct.set_encoder.isab_layers:
            assert isab.cell_type_embed is None

    def test_forward_with_conditioning(self):
        """forward() should pass ct_idx to SetTransformerEncoder."""
        torch.manual_seed(42)
        ct = CellTransformer(
            n_genes=100, n_cell_types=4, d_model=64, n_heads=4,
            n_isab_layers=1, n_inducing=8,
            condition_on_cell_type=True,
        )
        for isab in ct.set_encoder.isab_layers:
            torch.nn.init.normal_(isab.cell_type_embed, std=0.1)

        cell_data, cell_offsets = _make_flat_data(2, 4, 10, 100)
        emb, _ = ct(cell_data, cell_offsets)
        assert emb.shape == (2, 4, 64)

    def test_forward_with_conditioning_flat(self):
        """forward() with flat representation should also pass ct_idx."""
        torch.manual_seed(42)
        ct = CellTransformer(
            n_genes=100, n_cell_types=4, d_model=64, n_heads=4,
            n_isab_layers=1, n_inducing=8,
            condition_on_cell_type=True,
        )

        # Build flat representation: 2 samples, 4 types, 5 cells each
        n_cells = 5
        B, n_types = 2, 4
        cell_data = torch.randn(B * n_types * n_cells, 100)
        offsets = torch.stack([
            torch.arange(0, n_types + 1) * n_cells + b * n_types * n_cells
            for b in range(B)
        ])

        emb, _ = ct(cell_data, offsets)
        assert emb.shape == (2, 4, 64)


# ============================================================================
# Gene Attention Gate Integration Tests
# ============================================================================


class TestGeneAttentionGateIntegration:
    """Test GeneAttentionGate integration in CellTransformer."""

    @pytest.fixture
    def gate_config(self):
        return {
            "n_genes": 50,
            "n_cell_types": 8,
            "d_model": 32,
            "n_heads": 2,
            "n_isab_layers": 1,
            "n_inducing": 8,
            "n_pma_seeds": 1,
            "dropout": 0.0,
            "gene_gate_temperature": 2.0,
        }

    def test_gene_gate_changes_output(self, gate_config):
        """Gene gate should alter output compared to uniform gating."""
        torch.manual_seed(42)
        ct = CellTransformer(**gate_config)
        ct.eval()

        cell_data, cell_offsets = _make_flat_data(
            2, gate_config["n_cell_types"], 10, gate_config["n_genes"]
        )

        # Output with uniform gate (init)
        with torch.no_grad():
            out_uniform, _ = ct(cell_data, cell_offsets)

        # Perturb gate logits so they are no longer uniform
        with torch.no_grad():
            ct.gene_gate.gate_logits.normal_(0, 1.0)

        with torch.no_grad():
            out_perturbed, _ = ct(cell_data, cell_offsets)

        assert not torch.allclose(out_uniform, out_perturbed, atol=1e-5), (
            "Gene gate perturbation should change CellTransformer output"
        )

    def test_gate_weights_shape(self, gate_config):
        """Gate weights should be [n_cell_types, n_genes]."""
        ct = CellTransformer(**gate_config)
        weights = ct.gene_gate.get_gate_weights()
        assert weights.shape == (gate_config["n_cell_types"], gate_config["n_genes"])

    def test_gradient_flows_through_gate(self, gate_config):
        """Gradients should flow back through the gene gate logits."""
        ct = CellTransformer(**gate_config)
        ct.train()

        cell_data, cell_offsets = _make_flat_data(
            2, gate_config["n_cell_types"], 10, gate_config["n_genes"]
        )

        emb, _ = ct(cell_data, cell_offsets)
        loss = emb.sum()
        loss.backward()

        assert ct.gene_gate.gate_logits.grad is not None
        assert ct.gene_gate.gate_logits.grad.abs().sum() > 0, (
            "Gene gate logits should receive non-zero gradients"
        )

    def test_gene_gate_temperature_property(self, gate_config):
        """gene_gate_temperature property should read/write (no-op for sigmoid gate)."""
        ct = CellTransformer(**gate_config)

        # Sigmoid gate: temperature is a no-op buffer, default=1.0
        # (kept for backward compatibility with checkpoints and callbacks)
        assert ct.gene_gate_temperature == pytest.approx(gate_config.get("gene_gate_temperature", 2.0))

        # Set new value (no-op but shouldn't crash)
        ct.gene_gate_temperature = 0.5
        assert ct.gene_gate_temperature == pytest.approx(0.5)
        assert ct.gene_gate.temperature == pytest.approx(0.5)

        # Validate it rejects invalid values
        with pytest.raises(ValueError):
            ct.gene_gate_temperature = -1.0
