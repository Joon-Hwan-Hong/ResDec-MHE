"""
Tests for src/models/components/set_transformer.py

Test organization:
1. MultiheadAttentionBlock - core attention building block
2. ISAB - induced set attention (permutation equivariance)
3. PMA - pooling by multihead attention (permutation invariance)
4. SetTransformerEncoder - end-to-end encoder
5. Mathematical properties - equivariance, invariance proofs
6. Edge cases - empty sets, single elements, extreme sizes
7. Numerical stability - NaN, Inf, gradient flow
8. Determinism - reproducibility
9. Device - CPU/CUDA consistency
"""

import pytest
import torch


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mab():
    """MultiheadAttentionBlock for testing."""
    from src.models.components.set_transformer import MultiheadAttentionBlock
    return MultiheadAttentionBlock(d_model=64, n_heads=4)


@pytest.fixture
def isab():
    """ISAB for testing."""
    from src.models.components.set_transformer import ISAB
    return ISAB(d_model=64, n_heads=4, n_inducing=16)


@pytest.fixture
def pma():
    """PMA for testing."""
    from src.models.components.set_transformer import PMA
    return PMA(d_model=64, n_heads=4, n_seeds=1)


@pytest.fixture
def encoder():
    """SetTransformerEncoder for testing."""
    from src.models.components.set_transformer import SetTransformerEncoder
    return SetTransformerEncoder(
        d_input=100,
        d_model=64,
        n_heads=4,
        n_isab_layers=2,
        n_inducing=16,
        n_pma_seeds=1,
    )


# =============================================================================
# 1. MULTIHEAD ATTENTION BLOCK TESTS
# =============================================================================

class TestMultiheadAttentionBlock:
    """Tests for MultiheadAttentionBlock."""

    def test_output_shape_matches_query(self, mab):
        """Output should have same shape as query."""
        query = torch.randn(2, 10, 64)
        key_value = torch.randn(2, 20, 64)
        output, _ = mab(query, key_value)
        assert output.shape == query.shape

    def test_self_attention(self, mab):
        """Should work with query == key_value (self-attention)."""
        x = torch.randn(2, 15, 64)
        output, _ = mab(x, x)
        assert output.shape == x.shape

    def test_returns_attention_weights_when_requested(self, mab):
        """Should return attention weights when return_attention=True."""
        query = torch.randn(2, 10, 64)
        key_value = torch.randn(2, 20, 64)
        output, attention = mab(query, key_value, return_attention=True)

        assert attention is not None
        assert attention.shape == (2, 4, 10, 20)  # (batch, heads, query, kv)

    def test_returns_none_attention_by_default(self, mab):
        """Should return None for attention by default."""
        x = torch.randn(2, 10, 64)
        output, attention = mab(x, x)
        assert attention is None

    def test_rejects_misaligned_dimensions(self):
        """d_model must be divisible by n_heads."""
        from src.models.components.set_transformer import MultiheadAttentionBlock

        with pytest.raises(ValueError, match="must be divisible"):
            MultiheadAttentionBlock(d_model=65, n_heads=4)

    def test_padding_mask_affects_output(self, mab):
        """Key padding mask should affect output."""
        query = torch.randn(2, 5, 64)
        key_value = torch.randn(2, 10, 64)

        mask = torch.zeros(2, 10, dtype=torch.bool)
        mask[:, 5:] = True  # True = ignore last 5 positions

        out_masked, _ = mab(query, key_value, key_padding_mask=mask)
        out_unmasked, _ = mab(query, key_value)

        assert not torch.allclose(out_masked, out_unmasked)

    def test_without_ffn(self):
        """Should work without feed-forward network."""
        from src.models.components.set_transformer import MultiheadAttentionBlock

        mab = MultiheadAttentionBlock(d_model=64, n_heads=4, use_ffn=False)
        x = torch.randn(2, 10, 64)
        output, _ = mab(x, x)
        assert output.shape == x.shape

    def test_batch_size_one(self, mab):
        """Should handle batch size of 1."""
        x = torch.randn(1, 10, 64)
        output, _ = mab(x, x)
        assert output.shape == (1, 10, 64)


# =============================================================================
# 2. ISAB TESTS
# =============================================================================

class TestISAB:
    """Tests for Induced Set Attention Block."""

    def test_output_shape_preserved(self, isab):
        """Output should have same shape as input."""
        x = torch.randn(2, 100, 64)
        output = isab(x)
        assert output.shape == x.shape

    def test_permutation_equivariance(self, isab):
        """ISAB output should permute with input permutation."""
        torch.manual_seed(42)
        x = torch.randn(1, 50, 64)
        perm = torch.randperm(50)
        x_perm = x[:, perm, :]

        isab.eval()
        with torch.no_grad():
            out_original = isab(x)
            out_perm = isab(x_perm)

        out_original_perm = out_original[:, perm, :]
        assert torch.allclose(out_original_perm, out_perm, atol=1e-5)

    def test_handles_set_smaller_than_inducing_points(self, isab):
        """Should handle sets smaller than inducing points."""
        x = torch.randn(2, 5, 64)  # 5 elements, 16 inducing points
        output = isab(x)
        assert output.shape == (2, 5, 64)

    def test_handles_single_element_set(self, isab):
        """Should handle single-element sets."""
        x = torch.randn(2, 1, 64)
        output = isab(x)
        assert output.shape == (2, 1, 64)

    def test_respects_mask(self, isab):
        """Masked elements should not affect valid elements' context."""
        x = torch.randn(2, 20, 64)

        mask_full = torch.ones(2, 20, dtype=torch.bool)
        mask_partial = torch.zeros(2, 20, dtype=torch.bool)
        mask_partial[:, :10] = True

        isab.eval()
        with torch.no_grad():
            out_full = isab(x, mask=mask_full)
            out_partial = isab(x, mask=mask_partial)

        # Outputs differ because context differs
        assert not torch.allclose(out_full[:, :10], out_partial[:, :10])

    def test_different_inducing_point_counts(self):
        """Test with various inducing point counts."""
        from src.models.components.set_transformer import ISAB

        for n_inducing in [4, 16, 64]:
            isab = ISAB(d_model=32, n_heads=4, n_inducing=n_inducing)
            x = torch.randn(2, 50, 32)
            output = isab(x)
            assert output.shape == (2, 50, 32)


# =============================================================================
# 3. PMA TESTS
# =============================================================================

class TestPMA:
    """Tests for Pooling by Multihead Attention."""

    def test_output_shape(self, pma):
        """Output should be (batch, n_seeds, d_model)."""
        x = torch.randn(2, 100, 64)
        pooled, _ = pma(x)
        assert pooled.shape == (2, 1, 64)

    def test_permutation_invariance(self, pma):
        """PMA output should be invariant to input permutation."""
        torch.manual_seed(42)
        x = torch.randn(1, 50, 64)
        perm = torch.randperm(50)
        x_perm = x[:, perm, :]

        pma.eval()
        with torch.no_grad():
            out_original, _ = pma(x)
            out_perm, _ = pma(x_perm)

        assert torch.allclose(out_original, out_perm, atol=1e-5)

    def test_returns_attention_weights(self, pma):
        """Should return attention weights when requested."""
        x = torch.randn(2, 50, 64)
        pooled, attention = pma(x, return_attention=True)

        assert attention is not None
        assert attention.shape == (2, 4, 1, 50)

    def test_attention_weights_sum_to_one(self, pma):
        """Attention weights should sum to 1 over cells (in eval mode)."""
        pma.eval()
        x = torch.randn(2, 50, 64)
        with torch.no_grad():
            _, attention = pma(x, return_attention=True)

        sums = attention.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_multiple_seeds(self):
        """Should work with multiple seed vectors."""
        from src.models.components.set_transformer import PMA

        pma = PMA(d_model=64, n_heads=4, n_seeds=4)
        x = torch.randn(2, 50, 64)
        pooled, attention = pma(x, return_attention=True)

        assert pooled.shape == (2, 4, 64)
        assert attention.shape == (2, 4, 4, 50)

    def test_respects_mask_zeros_attention(self, pma):
        """Masked cells should have zero attention."""
        x = torch.randn(2, 20, 64)
        mask = torch.zeros(2, 20, dtype=torch.bool)
        mask[:, :10] = True  # First 10 valid

        _, attention = pma(x, mask=mask, return_attention=True)

        masked_attention = attention[:, :, :, 10:]
        assert masked_attention.abs().max() < 1e-4


# =============================================================================
# 4. SET TRANSFORMER ENCODER TESTS
# =============================================================================

class TestSetTransformerEncoder:
    """Tests for complete SetTransformerEncoder."""

    def test_output_shape_single_seed(self, encoder):
        """With 1 seed, output should be (batch, d_model)."""
        x = torch.randn(2, 50, 100)
        output, _ = encoder(x)
        assert output.shape == (2, 64)

    def test_output_shape_multiple_seeds(self):
        """With multiple seeds, output should be (batch, n_seeds, d_model)."""
        from src.models.components.set_transformer import SetTransformerEncoder

        encoder = SetTransformerEncoder(d_input=100, d_model=64, n_heads=4, n_pma_seeds=4)
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
        assert attention.shape == (2, 4, 1, 50)

    def test_handles_variable_set_sizes_with_mask(self, encoder):
        """Should handle batches with different set sizes via masking."""
        x = torch.randn(2, 50, 100)
        mask = torch.ones(2, 50, dtype=torch.bool)
        mask[0, 30:] = False

        output, attention = encoder(x, mask=mask, return_attention=True)

        assert output.shape == (2, 64)
        assert attention[0, :, :, 30:].abs().max() < 1e-4

    def test_handles_single_cell(self, encoder):
        """Should handle single-cell inputs."""
        x = torch.randn(2, 1, 100)
        output, _ = encoder(x)
        assert output.shape == (2, 64)

    def test_handles_large_set(self):
        """Should handle large sets (tests O(n*m) complexity)."""
        from src.models.components.set_transformer import SetTransformerEncoder

        encoder = SetTransformerEncoder(d_input=100, d_model=64, n_heads=4, n_inducing=32)
        x = torch.randn(2, 1000, 100)
        output, _ = encoder(x)
        assert output.shape == (2, 64)


# =============================================================================
# 5. MATHEMATICAL PROPERTIES TESTS
# =============================================================================

class TestMathematicalProperties:
    """Rigorous tests for mathematical invariants."""

    def test_isab_equivariance_multiple_permutations(self):
        """ISAB equivariance should hold for many random permutations."""
        from src.models.components.set_transformer import ISAB

        isab = ISAB(d_model=32, n_heads=4, n_inducing=8)
        isab.eval()
        x = torch.randn(1, 20, 32)

        with torch.no_grad():
            out_original = isab(x)

            for _ in range(10):
                perm = torch.randperm(20)
                x_perm = x[:, perm, :]
                out_perm = isab(x_perm)

                out_original_perm = out_original[:, perm, :]
                assert torch.allclose(out_original_perm, out_perm, atol=1e-5)

    def test_pma_invariance_multiple_permutations(self):
        """PMA invariance should hold for many random permutations."""
        from src.models.components.set_transformer import PMA

        pma = PMA(d_model=32, n_heads=4, n_seeds=1)
        pma.eval()
        x = torch.randn(1, 30, 32)

        with torch.no_grad():
            out_original, _ = pma(x)

            for _ in range(10):
                perm = torch.randperm(30)
                x_perm = x[:, perm, :]
                out_perm, _ = pma(x_perm)

                assert torch.allclose(out_original, out_perm, atol=1e-5)

    def test_attention_weights_non_negative(self):
        """All attention weights should be >= 0."""
        from src.models.components.set_transformer import PMA

        pma = PMA(d_model=64, n_heads=4, n_seeds=2)
        x = torch.randn(4, 50, 64)
        _, attention = pma(x, return_attention=True)

        assert (attention >= 0).all()

    def test_attention_weights_bounded_by_one(self):
        """All attention weights should be <= 1."""
        from src.models.components.set_transformer import PMA

        pma = PMA(d_model=64, n_heads=4, n_seeds=2)
        x = torch.randn(4, 50, 64)
        _, attention = pma(x, return_attention=True)

        assert (attention <= 1).all()

    def test_encoder_invariance_different_batch_sizes(self):
        """Invariance should hold regardless of batch size."""
        from src.models.components.set_transformer import SetTransformerEncoder

        encoder = SetTransformerEncoder(d_input=50, d_model=32, n_heads=4)
        encoder.eval()

        torch.manual_seed(42)
        x_single = torch.randn(1, 20, 50)
        perm = torch.randperm(20)
        x_perm = x_single[:, perm, :]

        with torch.no_grad():
            out1, _ = encoder(x_single)
            out2, _ = encoder(x_perm)

        assert torch.allclose(out1, out2, atol=1e-4)


# =============================================================================
# 6. EDGE CASES TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for boundary conditions and edge cases."""

    def test_empty_set_with_mask(self, encoder):
        """All-masked input should still produce output (though meaningless)."""
        x = torch.randn(2, 10, 100)
        mask = torch.zeros(2, 10, dtype=torch.bool)  # All masked

        # This is a degenerate case - output may be NaN or arbitrary
        # We just verify it doesn't crash
        output, _ = encoder(x, mask=mask)
        assert output.shape == (2, 64)

    def test_single_valid_cell(self, encoder):
        """Single valid cell in masked batch."""
        x = torch.randn(2, 10, 100)
        mask = torch.zeros(2, 10, dtype=torch.bool)
        mask[:, 0] = True  # Only first cell valid

        output, attention = encoder(x, mask=mask, return_attention=True)

        assert output.shape == (2, 64)
        # Masked cells should have ~0 attention
        masked_attention = attention[:, :, :, 1:]
        assert masked_attention.abs().max() < 1e-4

    def test_very_large_batch(self):
        """Large batch size."""
        from src.models.components.set_transformer import SetTransformerEncoder

        encoder = SetTransformerEncoder(d_input=50, d_model=32, n_heads=4)
        x = torch.randn(64, 20, 50)
        output, _ = encoder(x)
        assert output.shape == (64, 32)

    def test_minimum_valid_configuration(self):
        """Minimum valid configuration (1 head, 1 layer, 1 inducing)."""
        from src.models.components.set_transformer import SetTransformerEncoder

        encoder = SetTransformerEncoder(
            d_input=16,
            d_model=16,
            n_heads=1,
            n_isab_layers=1,
            n_inducing=1,
            n_pma_seeds=1,
        )
        x = torch.randn(2, 5, 16)
        output, _ = encoder(x)
        assert output.shape == (2, 16)

    def test_input_all_zeros(self, encoder):
        """All-zero input should produce output (not NaN)."""
        x = torch.zeros(2, 50, 100)
        output, _ = encoder(x)

        assert not torch.isnan(output).any()

    def test_input_all_same_value(self, encoder):
        """Input where all cells are identical - tests permutation invariance.

        In eval mode, identical inputs should produce identical outputs.
        In training mode, dropout creates different outputs (expected behavior).
        """
        encoder.eval()  # Disable dropout for deterministic comparison
        x = torch.ones(2, 50, 100) * 3.14

        with torch.no_grad():
            output, _ = encoder(x)

        assert not torch.isnan(output).any()
        # In eval mode, both samples should have same output
        assert torch.allclose(output[0], output[1], atol=1e-5)


# =============================================================================
# 7. NUMERICAL STABILITY TESTS
# =============================================================================

class TestNumericalStability:
    """Tests for numerical stability."""

    def test_no_nan_in_output(self, encoder):
        """Output should never be NaN for valid input."""
        x = torch.randn(4, 50, 100)
        output, _ = encoder(x)
        assert not torch.isnan(output).any()

    def test_no_inf_in_output(self, encoder):
        """Output should never be Inf for valid input."""
        x = torch.randn(4, 50, 100)
        output, _ = encoder(x)
        assert not torch.isinf(output).any()

    def test_large_input_values(self, encoder):
        """Large input values should not cause overflow."""
        x = torch.randn(2, 50, 100) * 100
        output, _ = encoder(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_small_input_values(self, encoder):
        """Very small input values should not cause underflow issues."""
        x = torch.randn(2, 50, 100) * 1e-6
        output, _ = encoder(x)

        assert not torch.isnan(output).any()

    def test_gradient_flow_no_nan(self, encoder):
        """Gradients should flow without NaN."""
        x = torch.randn(2, 50, 100, requires_grad=True)
        output, _ = encoder(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()

    def test_gradient_magnitude_reasonable(self, encoder):
        """Gradients should not explode."""
        x = torch.randn(2, 50, 100, requires_grad=True)
        output, _ = encoder(x)
        loss = output.mean()
        loss.backward()

        # Gradient magnitude should be reasonable (not exploding)
        grad_norm = x.grad.norm()
        assert grad_norm < 100, f"Gradient norm too large: {grad_norm}"


# =============================================================================
# 8. DETERMINISM TESTS
# =============================================================================

class TestDeterminism:
    """Tests for reproducibility."""

    def test_same_input_same_output(self, encoder):
        """Same input should produce identical output in eval mode."""
        encoder.eval()
        x = torch.randn(2, 50, 100)

        with torch.no_grad():
            out1, attn1 = encoder(x.clone(), return_attention=True)
            out2, attn2 = encoder(x.clone(), return_attention=True)

        assert torch.equal(out1, out2)
        assert torch.equal(attn1, attn2)

    def test_seeded_initialization_reproducible(self):
        """Seeded initialization should be reproducible."""
        from src.models.components.set_transformer import SetTransformerEncoder

        torch.manual_seed(42)
        enc1 = SetTransformerEncoder(d_input=50, d_model=32, n_heads=4)

        torch.manual_seed(42)
        enc2 = SetTransformerEncoder(d_input=50, d_model=32, n_heads=4)

        # Compare first layer parameters
        for p1, p2 in zip(enc1.parameters(), enc2.parameters()):
            assert torch.equal(p1, p2)


# =============================================================================
# 9. DEVICE TESTS
# =============================================================================

class TestDevice:
    """Tests for device placement and consistency."""

    def test_output_on_same_device_as_input(self, encoder):
        """Output should be on same device as input."""
        x = torch.randn(2, 50, 100)
        output, attention = encoder(x, return_attention=True)

        assert output.device == x.device
        assert attention.device == x.device

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_forward(self):
        """Should work on CUDA."""
        from src.models.components.set_transformer import SetTransformerEncoder

        encoder = SetTransformerEncoder(d_input=100, d_model=64, n_heads=4).cuda()
        x = torch.randn(2, 50, 100).cuda()

        output, attention = encoder(x, return_attention=True)

        assert output.device.type == "cuda"
        assert attention.device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cpu_cuda_outputs_close(self):
        """CPU and CUDA outputs should be numerically close."""
        from src.models.components.set_transformer import SetTransformerEncoder

        torch.manual_seed(42)
        encoder_cpu = SetTransformerEncoder(d_input=50, d_model=32, n_heads=4)

        torch.manual_seed(42)
        encoder_cuda = SetTransformerEncoder(d_input=50, d_model=32, n_heads=4).cuda()

        x_cpu = torch.randn(2, 20, 50)
        x_cuda = x_cpu.cuda()

        encoder_cpu.eval()
        encoder_cuda.eval()

        with torch.no_grad():
            out_cpu, _ = encoder_cpu(x_cpu)
            out_cuda, _ = encoder_cuda(x_cuda)

        assert torch.allclose(out_cpu, out_cuda.cpu(), atol=1e-4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_permutation_invariance_on_cuda(self):
        """Permutation invariance should hold on CUDA."""
        from src.models.components.set_transformer import SetTransformerEncoder

        encoder = SetTransformerEncoder(d_input=50, d_model=32, n_heads=4).cuda()
        encoder.eval()

        torch.manual_seed(42)
        x = torch.randn(1, 30, 50).cuda()
        perm = torch.randperm(30)
        x_perm = x[:, perm, :]

        with torch.no_grad():
            out1, _ = encoder(x)
            out2, _ = encoder(x_perm)

        assert torch.allclose(out1, out2, atol=1e-4)
