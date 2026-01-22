"""
Tests for src/models/components/set_transformer.py

Tests cover:
- MultiheadAttentionBlock: shape, residual connection
- ISAB: permutation equivariance, shape, masking
- PMA: permutation invariance, attention extraction, shape
- SetTransformerEncoder: end-to-end, masking
"""

import pytest
import torch
import numpy as np


class TestMultiheadAttentionBlock:
    """Tests for MultiheadAttentionBlock."""

    def test_output_shape_matches_query(self):
        """Output should have same shape as query."""
        from src.models.components.set_transformer import MultiheadAttentionBlock

        mab = MultiheadAttentionBlock(d_model=64, n_heads=4)
        query = torch.randn(2, 10, 64)
        key_value = torch.randn(2, 20, 64)

        output, _ = mab(query, key_value)

        assert output.shape == query.shape

    def test_self_attention(self):
        """Should work with query == key_value (self-attention)."""
        from src.models.components.set_transformer import MultiheadAttentionBlock

        mab = MultiheadAttentionBlock(d_model=64, n_heads=4)
        x = torch.randn(2, 15, 64)

        output, _ = mab(x, x)

        assert output.shape == x.shape

    def test_returns_attention_weights_when_requested(self):
        """Should return attention weights when return_attention=True."""
        from src.models.components.set_transformer import MultiheadAttentionBlock

        mab = MultiheadAttentionBlock(d_model=64, n_heads=4)
        query = torch.randn(2, 10, 64)
        key_value = torch.randn(2, 20, 64)

        output, attention = mab(query, key_value, return_attention=True)

        assert attention is not None
        assert attention.shape == (2, 4, 10, 20)  # (batch, heads, query, kv)

    def test_returns_none_attention_by_default(self):
        """Should return None for attention by default."""
        from src.models.components.set_transformer import MultiheadAttentionBlock

        mab = MultiheadAttentionBlock(d_model=64, n_heads=4)
        x = torch.randn(2, 10, 64)

        output, attention = mab(x, x)

        assert attention is None

    def test_rejects_misaligned_dimensions(self):
        """d_model must be divisible by n_heads."""
        from src.models.components.set_transformer import MultiheadAttentionBlock

        with pytest.raises(ValueError, match="must be divisible"):
            MultiheadAttentionBlock(d_model=65, n_heads=4)

    def test_padding_mask_works(self):
        """Key padding mask should affect output."""
        from src.models.components.set_transformer import MultiheadAttentionBlock

        mab = MultiheadAttentionBlock(d_model=64, n_heads=4)
        query = torch.randn(2, 5, 64)
        key_value = torch.randn(2, 10, 64)

        # Mask out last 5 positions
        mask = torch.zeros(2, 10, dtype=torch.bool)
        mask[:, 5:] = True  # True = ignore

        out_masked, _ = mab(query, key_value, key_padding_mask=mask)
        out_unmasked, _ = mab(query, key_value)

        # Outputs should differ when mask is applied
        assert not torch.allclose(out_masked, out_unmasked)


class TestISAB:
    """Tests for Induced Set Attention Block."""

    @pytest.fixture
    def isab(self):
        """Create ISAB for testing."""
        from src.models.components.set_transformer import ISAB

        return ISAB(d_model=64, n_heads=4, n_inducing=16)

    def test_output_shape_preserved(self, isab):
        """Output should have same shape as input."""
        x = torch.randn(2, 100, 64)
        output = isab(x)

        assert output.shape == x.shape

    def test_permutation_equivariance(self, isab):
        """ISAB output should permute with input permutation."""
        torch.manual_seed(42)
        x = torch.randn(1, 50, 64)

        # Create random permutation
        perm = torch.randperm(50)
        x_perm = x[:, perm, :]

        # Run ISAB on both
        isab.eval()
        with torch.no_grad():
            out_original = isab(x)
            out_perm = isab(x_perm)

        # Permuting output of original should match output of permuted
        out_original_perm = out_original[:, perm, :]
        assert torch.allclose(out_original_perm, out_perm, atol=1e-5)

    def test_handles_small_set(self, isab):
        """Should handle sets smaller than inducing points."""
        x = torch.randn(2, 5, 64)  # Only 5 elements, but 16 inducing points
        output = isab(x)

        assert output.shape == (2, 5, 64)

    def test_handles_single_element(self, isab):
        """Should handle single-element sets."""
        x = torch.randn(2, 1, 64)
        output = isab(x)

        assert output.shape == (2, 1, 64)

    def test_respects_mask(self, isab):
        """Masked elements should not affect valid elements."""
        x = torch.randn(2, 20, 64)

        # Full mask (all valid)
        mask_full = torch.ones(2, 20, dtype=torch.bool)

        # Partial mask (first 10 valid)
        mask_partial = torch.zeros(2, 20, dtype=torch.bool)
        mask_partial[:, :10] = True

        isab.eval()
        with torch.no_grad():
            out_full = isab(x, mask=mask_full)
            out_partial = isab(x, mask=mask_partial)

        # Valid positions should differ because context differs
        # (This is expected behavior - masked elements don't contribute)
        assert not torch.allclose(out_full[:, :10], out_partial[:, :10])


class TestPMA:
    """Tests for Pooling by Multihead Attention."""

    @pytest.fixture
    def pma(self):
        """Create PMA for testing."""
        from src.models.components.set_transformer import PMA

        return PMA(d_model=64, n_heads=4, n_seeds=1)

    def test_output_shape(self, pma):
        """Output should be (batch, n_seeds, d_model)."""
        x = torch.randn(2, 100, 64)
        pooled, _ = pma(x)

        assert pooled.shape == (2, 1, 64)

    def test_permutation_invariance(self, pma):
        """PMA output should be invariant to input permutation."""
        torch.manual_seed(42)
        x = torch.randn(1, 50, 64)

        # Create random permutation
        perm = torch.randperm(50)
        x_perm = x[:, perm, :]

        # Run PMA on both
        pma.eval()
        with torch.no_grad():
            out_original, _ = pma(x)
            out_perm, _ = pma(x_perm)

        # Outputs should be identical
        assert torch.allclose(out_original, out_perm, atol=1e-5)

    def test_returns_attention_weights(self, pma):
        """Should return attention weights when requested."""
        x = torch.randn(2, 50, 64)
        pooled, attention = pma(x, return_attention=True)

        assert attention is not None
        assert attention.shape == (2, 4, 1, 50)  # (batch, heads, seeds, cells)

    def test_attention_weights_sum_to_one(self, pma):
        """Attention weights should sum to 1 over cells (in eval mode)."""
        pma.eval()  # Disable dropout for exact summation
        x = torch.randn(2, 50, 64)
        with torch.no_grad():
            _, attention = pma(x, return_attention=True)

        # Sum over cell dimension (last dim)
        sums = attention.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_multiple_seeds(self):
        """Should work with multiple seed vectors."""
        from src.models.components.set_transformer import PMA

        pma = PMA(d_model=64, n_heads=4, n_seeds=4)
        x = torch.randn(2, 50, 64)

        pooled, attention = pma(x, return_attention=True)

        assert pooled.shape == (2, 4, 64)
        assert attention.shape == (2, 4, 4, 50)  # (batch, heads, seeds, cells)

    def test_respects_mask(self, pma):
        """Masked cells should have zero attention."""
        x = torch.randn(2, 20, 64)

        # Mask out last 10 cells
        mask = torch.zeros(2, 20, dtype=torch.bool)
        mask[:, :10] = True  # First 10 valid

        _, attention = pma(x, mask=mask, return_attention=True)

        # Attention on masked cells should be ~0
        # (softmax with -inf gives 0)
        masked_attention = attention[:, :, :, 10:]
        assert masked_attention.abs().max() < 1e-4


class TestSetTransformerEncoder:
    """Tests for complete SetTransformerEncoder."""

    @pytest.fixture
    def encoder(self):
        """Create encoder for testing."""
        from src.models.components.set_transformer import SetTransformerEncoder

        return SetTransformerEncoder(
            d_input=100,
            d_model=64,
            n_heads=4,
            n_isab_layers=2,
            n_inducing=16,
            n_pma_seeds=1,
        )

    def test_output_shape_single_seed(self, encoder):
        """With 1 seed, output should be (batch, d_model)."""
        x = torch.randn(2, 50, 100)
        output, _ = encoder(x)

        assert output.shape == (2, 64)

    def test_output_shape_multiple_seeds(self):
        """With multiple seeds, output should be (batch, n_seeds, d_model)."""
        from src.models.components.set_transformer import SetTransformerEncoder

        encoder = SetTransformerEncoder(
            d_input=100,
            d_model=64,
            n_heads=4,
            n_pma_seeds=4,
        )
        x = torch.randn(2, 50, 100)
        output, _ = encoder(x)

        assert output.shape == (2, 4, 64)

    def test_permutation_invariance(self, encoder):
        """Encoder output should be permutation invariant."""
        torch.manual_seed(42)
        x = torch.randn(1, 50, 100)

        perm = torch.randperm(50)
        x_perm = x[:, perm, :]

        encoder.eval()
        with torch.no_grad():
            out_original, _ = encoder(x)
            out_perm, _ = encoder(x_perm)

        assert torch.allclose(out_original, out_perm, atol=1e-4)

    def test_returns_pma_attention(self, encoder):
        """Should return PMA attention weights when requested."""
        x = torch.randn(2, 50, 100)
        output, attention = encoder(x, return_attention=True)

        assert attention is not None
        # Shape: (batch, n_heads, n_seeds, n_cells)
        assert attention.shape == (2, 4, 1, 50)

    def test_handles_variable_set_sizes_with_mask(self, encoder):
        """Should handle batches with different set sizes via masking."""
        # Batch of 2, max 50 cells, but sample 1 has only 30
        x = torch.randn(2, 50, 100)
        mask = torch.ones(2, 50, dtype=torch.bool)
        mask[0, 30:] = False  # Sample 0 has only 30 valid cells

        output, attention = encoder(x, mask=mask, return_attention=True)

        assert output.shape == (2, 64)
        # Check masked attention is zero for sample 0
        assert attention[0, :, :, 30:].abs().max() < 1e-4

    def test_handles_single_cell(self, encoder):
        """Should handle single-cell inputs."""
        x = torch.randn(2, 1, 100)
        output, _ = encoder(x)

        assert output.shape == (2, 64)

    def test_gradient_flow(self, encoder):
        """Gradients should flow through entire encoder."""
        x = torch.randn(2, 50, 100, requires_grad=True)
        output, _ = encoder(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_no_nan_output(self, encoder):
        """Output should not contain NaN values."""
        x = torch.randn(2, 50, 100)
        output, _ = encoder(x)

        assert not torch.isnan(output).any()

    def test_handles_large_set(self):
        """Should handle large sets efficiently."""
        from src.models.components.set_transformer import SetTransformerEncoder

        encoder = SetTransformerEncoder(
            d_input=100,
            d_model=64,
            n_heads=4,
            n_inducing=32,
        )
        x = torch.randn(2, 1000, 100)  # 1000 cells
        output, _ = encoder(x)

        assert output.shape == (2, 64)


class TestSetTransformerEquivarianceProperties:
    """Mathematical property tests for Set Transformer."""

    def test_isab_multiple_permutations(self):
        """ISAB equivariance should hold for multiple permutations."""
        from src.models.components.set_transformer import ISAB

        isab = ISAB(d_model=32, n_heads=4, n_inducing=8)
        isab.eval()

        x = torch.randn(1, 20, 32)

        with torch.no_grad():
            out_original = isab(x)

            for _ in range(5):
                perm = torch.randperm(20)
                x_perm = x[:, perm, :]
                out_perm = isab(x_perm)

                # Verify equivariance
                out_original_perm = out_original[:, perm, :]
                assert torch.allclose(out_original_perm, out_perm, atol=1e-5)

    def test_pma_multiple_permutations(self):
        """PMA invariance should hold for multiple permutations."""
        from src.models.components.set_transformer import PMA

        pma = PMA(d_model=32, n_heads=4, n_seeds=1)
        pma.eval()

        x = torch.randn(1, 30, 32)

        with torch.no_grad():
            out_original, _ = pma(x)

            for _ in range(5):
                perm = torch.randperm(30)
                x_perm = x[:, perm, :]
                out_perm, _ = pma(x_perm)

                # Verify invariance
                assert torch.allclose(out_original, out_perm, atol=1e-5)

    def test_attention_weights_are_positive(self):
        """All attention weights should be non-negative."""
        from src.models.components.set_transformer import PMA

        pma = PMA(d_model=64, n_heads=4, n_seeds=2)
        x = torch.randn(4, 50, 64)

        _, attention = pma(x, return_attention=True)

        assert (attention >= 0).all()

    def test_attention_weights_bounded(self):
        """All attention weights should be at most 1."""
        from src.models.components.set_transformer import PMA

        pma = PMA(d_model=64, n_heads=4, n_seeds=2)
        x = torch.randn(4, 50, 64)

        _, attention = pma(x, return_attention=True)

        assert (attention <= 1).all()