"""
Tests for src/models/branches/pseudobulk_encoder.py

Test organization:
1. Initialization - parameter creation, validation, defaults
2. Forward pass - shapes, input validation
3. Gene gate integration - temperature, gating behavior
4. Gradient flow - through MLP and gate
5. Interpretability - gene weights, top genes
6. Edge cases - single cell type, single gene
"""

import pytest
import torch

from src.data.constants import N_CELL_TYPES


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def small_encoder():
    """Small encoder for fast tests."""
    from src.models.branches.pseudobulk_encoder import PseudobulkEncoder
    return PseudobulkEncoder(
        n_cell_types=5, n_genes=20, d_embed=16,
        mlp_hidden=[32, 16], dropout=0.0, temperature=1.0,
    )


@pytest.fixture
def production_encoder():
    """Production-sized encoder (31 types, 3000 genes)."""
    from src.models.branches.pseudobulk_encoder import PseudobulkEncoder
    return PseudobulkEncoder(
        n_cell_types=N_CELL_TYPES, n_genes=3000, d_embed=128,
        mlp_hidden=[512, 256], dropout=0.1, temperature=2.0,
    )


# =============================================================================
# 1. INITIALIZATION TESTS
# =============================================================================

class TestInitialization:
    """Tests for PseudobulkEncoder initialization."""

    def test_creates_gene_gate(self, small_encoder):
        """Encoder should have a gene attention gate."""
        assert hasattr(small_encoder, 'gene_gate')
        assert small_encoder.gene_gate.n_cell_types == 5
        assert small_encoder.gene_gate.n_genes == 20

    def test_creates_shared_mlp(self, small_encoder):
        """Encoder should have a shared MLP."""
        assert hasattr(small_encoder, 'shared_mlp')

    def test_stores_dimensions(self, small_encoder):
        """Encoder should store key dimensions."""
        assert small_encoder.n_cell_types == 5
        assert small_encoder.n_genes == 20
        assert small_encoder.d_embed == 16

    def test_default_mlp_hidden(self):
        """Default MLP hidden dims should be [512, 256]."""
        from src.models.branches.pseudobulk_encoder import PseudobulkEncoder
        enc = PseudobulkEncoder(n_cell_types=5, n_genes=20, d_embed=16)
        assert enc.mlp_hidden == [512, 256]

    def test_custom_mlp_hidden(self):
        """Custom MLP hidden dims should be stored."""
        from src.models.branches.pseudobulk_encoder import PseudobulkEncoder
        enc = PseudobulkEncoder(
            n_cell_types=5, n_genes=20, d_embed=16, mlp_hidden=[64, 32]
        )
        assert enc.mlp_hidden == [64, 32]

    def test_temperature_forwarded_to_gate(self):
        """Temperature parameter should be forwarded to gene gate."""
        from src.models.branches.pseudobulk_encoder import PseudobulkEncoder
        enc = PseudobulkEncoder(
            n_cell_types=5, n_genes=20, d_embed=16, temperature=2.0
        )
        assert enc.gene_gate.temperature == 2.0

    def test_invalid_n_cell_types_raises(self):
        """Non-positive n_cell_types should raise ValueError."""
        from src.models.branches.pseudobulk_encoder import PseudobulkEncoder
        with pytest.raises(ValueError, match="n_cell_types must be positive"):
            PseudobulkEncoder(n_cell_types=0, n_genes=20, d_embed=16)

    def test_invalid_n_genes_raises(self):
        """Non-positive n_genes should raise ValueError."""
        from src.models.branches.pseudobulk_encoder import PseudobulkEncoder
        with pytest.raises(ValueError, match="n_genes must be positive"):
            PseudobulkEncoder(n_cell_types=5, n_genes=0, d_embed=16)

    def test_invalid_d_embed_raises(self):
        """Non-positive d_embed should raise ValueError."""
        from src.models.branches.pseudobulk_encoder import PseudobulkEncoder
        with pytest.raises(ValueError, match="d_embed must be positive"):
            PseudobulkEncoder(n_cell_types=5, n_genes=20, d_embed=0)


# =============================================================================
# 2. FORWARD PASS TESTS
# =============================================================================

class TestForwardPass:
    """Tests for PseudobulkEncoder forward pass."""

    def test_output_shape(self, small_encoder):
        """Output should be [batch, n_cell_types, d_embed]."""
        x = torch.randn(4, 5, 20)
        out = small_encoder(x)
        assert out.shape == (4, 5, 16)

    def test_batch_size_one(self, small_encoder):
        """Should work with batch size 1."""
        x = torch.randn(1, 5, 20)
        out = small_encoder(x)
        assert out.shape == (1, 5, 16)

    def test_output_dtype(self, small_encoder):
        """Output should match input dtype."""
        x = torch.randn(4, 5, 20)
        out = small_encoder(x)
        assert out.dtype == x.dtype

    def test_no_nan_in_output(self, small_encoder):
        """Output should not contain NaN."""
        x = torch.randn(4, 5, 20)
        out = small_encoder(x)
        assert not torch.isnan(out).any()

    def test_wrong_dims_raises(self, small_encoder):
        """2D input should raise ValueError."""
        x = torch.randn(5, 20)
        with pytest.raises(ValueError, match="Expected 3D input"):
            small_encoder(x)

    def test_wrong_n_cell_types_raises(self, small_encoder):
        """Mismatched cell type dim should raise ValueError."""
        x = torch.randn(4, 10, 20)  # 10 != 5
        with pytest.raises(ValueError, match="Expected 5 cell types"):
            small_encoder(x)

    def test_wrong_n_genes_raises(self, small_encoder):
        """Mismatched gene dim should raise ValueError."""
        x = torch.randn(4, 5, 50)  # 50 != 20
        with pytest.raises(ValueError, match="Expected 20 genes"):
            small_encoder(x)

    def test_deterministic_in_eval(self, small_encoder):
        """Same input should produce same output in eval mode."""
        small_encoder.eval()
        x = torch.randn(4, 5, 20)
        out1 = small_encoder(x)
        out2 = small_encoder(x)
        assert torch.allclose(out1, out2)

    def test_production_shape(self, production_encoder):
        """Production-sized encoder should produce correct shape."""
        production_encoder.eval()
        x = torch.randn(2, N_CELL_TYPES, 3000)
        out = production_encoder(x)
        assert out.shape == (2, N_CELL_TYPES, 128)


# =============================================================================
# 3. GENE GATE INTEGRATION TESTS
# =============================================================================

class TestGeneGateIntegration:
    """Tests for gene attention gate behavior within encoder."""

    def test_temperature_property_get(self, small_encoder):
        """Temperature property should return gate temperature."""
        assert small_encoder.temperature == 1.0

    def test_temperature_property_set(self, small_encoder):
        """Setting temperature should update gate temperature."""
        small_encoder.temperature = 2.0
        assert small_encoder.temperature == 2.0
        assert small_encoder.gene_gate.temperature == 2.0

    def test_temperature_affects_output(self, small_encoder):
        """Different temperatures should produce different outputs."""
        small_encoder.eval()
        x = torch.randn(2, 5, 20)

        # Manually set non-uniform logits so temperature has an effect
        small_encoder.gene_gate.gate_logits.data = torch.randn(5, 20)

        small_encoder.temperature = 0.1
        out_sharp = small_encoder(x).detach().clone()

        small_encoder.temperature = 10.0
        out_soft = small_encoder(x).detach().clone()

        assert not torch.allclose(out_sharp, out_soft)

    def test_gate_weights_accessible(self, small_encoder):
        """get_gene_weights should return valid gate weights."""
        weights = small_encoder.get_gene_weights()
        assert weights.shape == (5, 20)
        # Each row should sum to ~1 (softmax)
        row_sums = weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(5), atol=1e-5)


# =============================================================================
# 4. GRADIENT FLOW TESTS
# =============================================================================

class TestGradientFlow:
    """Tests for gradient flow through the encoder."""

    def test_gradients_flow_to_gate_logits(self, small_encoder):
        """Gradients should reach gene gate logits."""
        x = torch.randn(4, 5, 20)
        out = small_encoder(x)
        loss = out.sum()
        loss.backward()
        assert small_encoder.gene_gate.gate_logits.grad is not None
        assert (small_encoder.gene_gate.gate_logits.grad != 0).any()

    def test_gradients_flow_to_mlp(self, small_encoder):
        """Gradients should reach shared MLP parameters."""
        x = torch.randn(4, 5, 20)
        out = small_encoder(x)
        loss = out.sum()
        loss.backward()

        for name, param in small_encoder.shared_mlp.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_gradients_flow_to_input(self, small_encoder):
        """Gradients should flow back to input."""
        x = torch.randn(4, 5, 20, requires_grad=True)
        out = small_encoder(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert (x.grad != 0).any()


# =============================================================================
# 5. INTERPRETABILITY TESTS
# =============================================================================

class TestInterpretability:
    """Tests for interpretability methods."""

    def test_get_top_genes_per_cell_type(self, small_encoder):
        """Should return top genes for each cell type."""
        results = small_encoder.get_top_genes_per_cell_type(k=5)
        assert len(results) == 5  # 5 cell types
        for ct_idx in range(5):
            assert len(results[ct_idx]) == 5  # k=5

    def test_get_top_genes_with_names(self, small_encoder):
        """Should use gene names when provided."""
        gene_names = [f"gene_{i}" for i in range(20)]
        results = small_encoder.get_top_genes_per_cell_type(k=3, gene_names=gene_names)
        # Check that results contain gene name strings, not indices
        for ct_idx in range(5):
            name, weight = results[ct_idx][0]
            assert isinstance(name, str)
            assert name.startswith("gene_")

    def test_get_top_genes_k_exceeds_n_genes(self, small_encoder):
        """k > n_genes should return all genes."""
        results = small_encoder.get_top_genes_per_cell_type(k=100)
        for ct_idx in range(5):
            assert len(results[ct_idx]) == 20  # capped at n_genes

    def test_extra_repr(self, small_encoder):
        """extra_repr should contain key info."""
        repr_str = small_encoder.extra_repr()
        assert "n_cell_types=5" in repr_str
        assert "n_genes=20" in repr_str
        assert "d_embed=16" in repr_str


# =============================================================================
# 6. EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_cell_type(self):
        """Should work with a single cell type."""
        from src.models.branches.pseudobulk_encoder import PseudobulkEncoder
        enc = PseudobulkEncoder(
            n_cell_types=1, n_genes=10, d_embed=8,
            mlp_hidden=[16], dropout=0.0,
        )
        x = torch.randn(2, 1, 10)
        out = enc(x)
        assert out.shape == (2, 1, 8)

    def test_single_gene(self):
        """Should work with a single gene."""
        from src.models.branches.pseudobulk_encoder import PseudobulkEncoder
        enc = PseudobulkEncoder(
            n_cell_types=3, n_genes=1, d_embed=8,
            mlp_hidden=[16], dropout=0.0,
        )
        x = torch.randn(2, 3, 1)
        out = enc(x)
        assert out.shape == (2, 3, 8)

    def test_large_batch(self, small_encoder):
        """Should handle large batch sizes."""
        small_encoder.eval()
        x = torch.randn(64, 5, 20)
        out = small_encoder(x)
        assert out.shape == (64, 5, 16)

    def test_zero_input(self, small_encoder):
        """Zero input should not produce NaN."""
        x = torch.zeros(2, 5, 20)
        out = small_encoder(x)
        assert not torch.isnan(out).any()

    def test_large_input_values(self, small_encoder):
        """Large input values should not produce NaN."""
        x = torch.randn(2, 5, 20) * 1000
        out = small_encoder(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_layer_norm_disabled(self):
        """Should work without LayerNorm in MLP."""
        from src.models.branches.pseudobulk_encoder import PseudobulkEncoder
        enc = PseudobulkEncoder(n_cell_types=5, n_genes=50, d_embed=64, use_layer_norm=False)
        x = torch.randn(2, 5, 50)
        out = enc(x)
        assert out.shape == (2, 5, 64)
        assert torch.isfinite(out).all()

    def test_dropout_train_vs_eval(self):
        """Dropout should cause different outputs in train vs eval mode."""
        from src.models.branches.pseudobulk_encoder import PseudobulkEncoder
        enc = PseudobulkEncoder(n_cell_types=5, n_genes=50, d_embed=64, dropout=0.5)
        x = torch.randn(2, 5, 50)
        enc.train()
        out_train = enc(x)
        enc.eval()
        out_eval = enc(x)
        assert not torch.allclose(out_train, out_eval, atol=1e-6)

    def test_cell_type_independence(self):
        """Modifying one cell type should not affect others (shared MLP but independent gating)."""
        from src.models.branches.pseudobulk_encoder import PseudobulkEncoder
        enc = PseudobulkEncoder(n_cell_types=5, n_genes=50, d_embed=64)
        enc.eval()
        x = torch.randn(1, 5, 50)
        out1 = enc(x).clone()
        x_modified = x.clone()
        x_modified[0, 0, :] = torch.randn(50)
        out2 = enc(x_modified)
        # First cell type output should change
        assert not torch.allclose(out1[0, 0], out2[0, 0])
        # Other cell types should remain the same
        assert torch.allclose(out1[0, 1:], out2[0, 1:])
