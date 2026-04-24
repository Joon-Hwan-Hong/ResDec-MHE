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
        """All-masked input should produce finite output (learned empty embedding)."""
        x = torch.randn(2, 10, 100)
        mask = torch.zeros(2, 10, dtype=torch.bool)  # All masked

        output, _ = encoder(x, mask=mask)
        assert output.shape == (2, 64)
        # Output should be finite (no NaN, no Inf)
        assert torch.isfinite(output).all(), "All-masked input produced NaN/Inf"

    def test_all_masked_produces_finite_gradients(self):
        """All-masked input should allow gradient flow through empty_embedding."""
        from src.models.components.set_transformer import SetTransformerEncoder

        encoder = SetTransformerEncoder(d_input=50, d_model=32, n_heads=4)
        x = torch.randn(2, 10, 50, requires_grad=True)
        mask = torch.zeros(2, 10, dtype=torch.bool)  # All masked

        output, _ = encoder(x, mask=mask)

        # Compute loss and backprop
        loss = output.sum()
        loss.backward()

        # empty_embedding should have gradient (it's being used)
        assert encoder.empty_embedding.grad is not None
        assert torch.isfinite(encoder.empty_embedding.grad).all()

    def test_mixed_empty_and_valid_batches(self):
        """Batch with some all-masked and some valid samples."""
        from src.models.components.set_transformer import SetTransformerEncoder

        encoder = SetTransformerEncoder(d_input=50, d_model=32, n_heads=4)
        x = torch.randn(4, 10, 50)
        mask = torch.zeros(4, 10, dtype=torch.bool)
        # Samples 0,2 have valid cells; samples 1,3 are all-masked
        mask[0, :5] = True
        mask[2, :3] = True

        output, _ = encoder(x, mask=mask)

        assert output.shape == (4, 32)
        # All outputs should be finite
        assert torch.isfinite(output).all()
        # Empty samples should get empty_embedding
        assert torch.allclose(output[1], output[3], atol=1e-6)  # Both use empty_embedding

    def test_mixed_empty_and_valid_with_attention(self):
        """Batch with mixed empty/valid samples should produce valid attention weights.

        This tests the NaN risk when some samples are all-masked but others have valid cells.
        The attention computation for empty samples should not produce NaN values.
        """
        from src.models.components.set_transformer import SetTransformerEncoder

        encoder = SetTransformerEncoder(d_input=50, d_model=32, n_heads=4)
        x = torch.randn(4, 10, 50)
        mask = torch.zeros(4, 10, dtype=torch.bool)
        # Samples 0,2 have valid cells; samples 1,3 are all-masked
        mask[0, :5] = True
        mask[2, :3] = True

        output, attention = encoder(x, mask=mask, return_attention=True)

        # Output should be finite for all samples
        assert torch.isfinite(output).all()

        # Attention should be finite (no NaN from empty samples)
        assert attention is not None
        assert torch.isfinite(attention).all(), (
            "Attention weights contain NaN or Inf - empty samples may not be handled correctly"
        )

        # Empty samples (1,3) should have zero attention (replaced by fix)
        assert (attention[1] == 0).all(), "Empty sample 1 should have zero attention"
        assert (attention[3] == 0).all(), "Empty sample 3 should have zero attention"

        # Valid samples (0,2) should have non-zero attention for valid cells
        assert attention[0, :, :, :5].abs().max() > 0, "Valid cells should have attention"
        assert attention[2, :, :, :3].abs().max() > 0, "Valid cells should have attention"

        # Masked cells in valid samples should have ~0 attention
        assert attention[0, :, :, 5:].abs().max() < 1e-4, "Masked cells should have ~0 attention"
        assert attention[2, :, :, 3:].abs().max() < 1e-4, "Masked cells should have ~0 attention"

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

    def test_mixed_batch_gradient_flow(self):
        """Gradients should flow for mixed empty/valid batches."""
        from src.models.components.set_transformer import SetTransformerEncoder
        enc = SetTransformerEncoder(d_input=50, d_model=32, n_heads=2, n_isab_layers=1, n_inducing=8)
        x = torch.randn(4, 10, 50, requires_grad=True)
        mask = torch.ones(4, 10, dtype=torch.bool)
        mask[0, :] = False  # First sample fully masked
        mask[2, :] = False  # Third sample fully masked
        out, _ = enc(x, mask)
        loss = out.sum()
        loss.backward()
        # Encoder parameters should receive gradients from valid samples
        has_param_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in enc.parameters()
        )
        assert has_param_grad, "No encoder parameters received gradients"


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


# =============================================================================
# 10. MULTI-SEED MASKING TESTS
# =============================================================================

class TestMultiSeedMasking:
    """A11-2/A11-3: Tests for n_pma_seeds > 1 with masking."""

    def test_multi_seed_all_masked_output_shape(self):
        """A11-2: All-masked with n_pma_seeds>1 should return [B, n_seeds, d_model]."""
        from src.models.components.set_transformer import SetTransformerEncoder

        enc = SetTransformerEncoder(
            d_input=50, d_model=32, n_heads=4, n_isab_layers=1,
            n_inducing=8, n_pma_seeds=3, dropout=0.0
        )
        enc.eval()

        B, n_cells = 2, 10
        x = torch.randn(B, n_cells, 50)
        mask = torch.zeros(B, n_cells, dtype=torch.bool)  # All masked

        with torch.no_grad():
            pooled, attention = enc(x, mask)

        assert pooled.shape == (B, 3, 32), f"Expected (2, 3, 32), got {pooled.shape}"
        assert torch.isfinite(pooled).all()

        # All outputs should be the empty_embedding repeated
        for b in range(B):
            for s in range(3):
                assert torch.allclose(pooled[b, s], enc.empty_embedding, atol=1e-6)

    def test_multi_seed_all_masked_with_return_attention(self):
        """A11-2: All-masked with return_attention should return zero attention."""
        from src.models.components.set_transformer import SetTransformerEncoder

        enc = SetTransformerEncoder(
            d_input=50, d_model=32, n_heads=4, n_isab_layers=1,
            n_inducing=8, n_pma_seeds=3, dropout=0.0
        )
        enc.eval()

        B, n_cells = 2, 10
        x = torch.randn(B, n_cells, 50)
        mask = torch.zeros(B, n_cells, dtype=torch.bool)

        with torch.no_grad():
            pooled, attention = enc(x, mask, return_attention=True)

        assert attention is not None
        assert attention.shape == (B, 4, 3, n_cells)  # [B, n_heads, n_seeds, n_cells]
        assert torch.allclose(attention, torch.zeros_like(attention))

    def test_multi_seed_mixed_batch(self):
        """A11-3: Mixed batch with n_pma_seeds>1: empty samples get empty_embedding."""
        from src.models.components.set_transformer import SetTransformerEncoder

        enc = SetTransformerEncoder(
            d_input=50, d_model=32, n_heads=4, n_isab_layers=1,
            n_inducing=8, n_pma_seeds=3, dropout=0.0
        )
        enc.eval()

        B, n_cells = 3, 10
        x = torch.randn(B, n_cells, 50)
        mask = torch.zeros(B, n_cells, dtype=torch.bool)
        mask[0, :5] = True   # Sample 0: 5 valid cells
        # Sample 1: all masked
        mask[2, :8] = True   # Sample 2: 8 valid cells

        with torch.no_grad():
            pooled, attention = enc(x, mask)

        assert pooled.shape == (B, 3, 32)
        assert torch.isfinite(pooled).all()

        # Empty sample (1) should have empty_embedding for all seeds
        for s in range(3):
            assert torch.allclose(pooled[1, s], enc.empty_embedding, atol=1e-6)

        # Valid samples should differ from empty_embedding (with high probability)
        # Check at least one seed differs
        assert not torch.allclose(pooled[0], pooled[1], atol=1e-4)
        assert not torch.allclose(pooled[2], pooled[1], atol=1e-4)

    def test_multi_seed_mixed_batch_attention_replacement(self):
        """A11-3: Empty samples should get zero attention in mixed batch."""
        from src.models.components.set_transformer import SetTransformerEncoder

        enc = SetTransformerEncoder(
            d_input=50, d_model=32, n_heads=4, n_isab_layers=1,
            n_inducing=8, n_pma_seeds=3, dropout=0.0
        )
        enc.eval()

        B, n_cells = 3, 10
        x = torch.randn(B, n_cells, 50)
        mask = torch.zeros(B, n_cells, dtype=torch.bool)
        mask[0, :5] = True
        mask[2, :8] = True

        with torch.no_grad():
            pooled, attention = enc(x, mask, return_attention=True)

        assert attention is not None
        assert attention.shape == (B, 4, 3, n_cells)

        # Empty sample attention should be zero
        assert torch.allclose(attention[1], torch.zeros_like(attention[1]))

        # Valid samples should have non-zero attention
        assert attention[0].abs().sum() > 0
        assert attention[2].abs().sum() > 0

    def test_multi_seed_gradient_flow_mixed_batch(self):
        """A11-3: Gradients should flow correctly through mixed batch with multi-seed."""
        from src.models.components.set_transformer import SetTransformerEncoder

        enc = SetTransformerEncoder(
            d_input=50, d_model=32, n_heads=4, n_isab_layers=1,
            n_inducing=8, n_pma_seeds=3, dropout=0.0
        )

        B, n_cells = 3, 10
        x = torch.randn(B, n_cells, 50, requires_grad=True)
        mask = torch.zeros(B, n_cells, dtype=torch.bool)
        mask[0, :5] = True
        mask[2, :8] = True

        pooled, _ = enc(x, mask)
        loss = pooled.sum()
        loss.backward()

        # Input gradients should be finite
        assert torch.isfinite(x.grad).all()

        # Encoder parameters should get gradients
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in enc.parameters())
        assert has_grad, "No parameter received gradients"


# =============================================================================
# 11. ISAB XAVIER INIT TESTS
# =============================================================================

class TestISABXavierInit:
    """Tests for ISAB inducing point initialization scale."""

    def test_inducing_points_xavier_scale(self):
        """ISAB inducing points should use Xavier init, not small randn * 0.02."""
        from src.models.components.set_transformer import ISAB
        isab = ISAB(d_model=128, n_heads=4, n_inducing=32)
        # Xavier uniform for [32, 128]: limit = sqrt(6 / (32 + 128)) = 0.1936
        # std of uniform(-limit, limit) = limit / sqrt(3) ≈ 0.112
        # With randn * 0.02, std would be ~0.02
        std = isab.inducing_points.data.std().item()
        assert std > 0.05, (
            f"Inducing points std={std:.4f}, expected Xavier scale (~0.11), "
            f"not randn*0.02 scale (~0.02)"
        )

    def test_inducing_points_match_pma_init_style(self):
        """ISAB inducing points should use same init style as PMA seed vectors."""
        from src.models.components.set_transformer import ISAB, PMA
        isab = ISAB(d_model=128, n_heads=4, n_inducing=32)
        pma = PMA(d_model=128, n_heads=4, n_seeds=1)
        # Both should use Xavier — similar scale (not identical due to different shapes)
        isab_std = isab.inducing_points.data.std().item()
        pma_std = pma.seed_vectors.data.std().item()
        ratio = max(isab_std, pma_std) / min(isab_std, pma_std)
        assert ratio < 3.0, (
            f"ISAB std={isab_std:.4f}, PMA std={pma_std:.4f}, ratio={ratio:.1f} — "
            f"expected similar init scales"
        )


# =============================================================================
# 12. ISAB SKIP CONNECTION TESTS
# =============================================================================

class TestISABSkipConnection:
    """Tests for ISAB input-to-output skip connection."""

    def test_output_includes_input_skip(self):
        """ISAB output should be mab2(x, h) + x (skip connection)."""
        import torch
        from src.models.components.set_transformer import ISAB
        torch.manual_seed(42)
        isab = ISAB(d_model=64, n_heads=4, n_inducing=16, dropout=0.0)

        x = torch.randn(1, 5, 64)

        # Compute what the forward does internally
        inducing = isab.inducing_points.unsqueeze(0)
        h, _ = isab.mab1(inducing, x)
        mab2_output, _ = isab.mab2(x, h)

        # With skip connection: output = mab2_output + x
        expected = mab2_output + x
        output = isab(x)

        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)

    def test_skip_connection_with_mask(self):
        """Skip connection should work correctly with padding mask."""
        import torch
        from src.models.components.set_transformer import ISAB
        torch.manual_seed(42)
        isab = ISAB(d_model=64, n_heads=4, n_inducing=16, dropout=0.0)

        x = torch.randn(2, 8, 64)
        mask = torch.ones(2, 8, dtype=torch.bool)
        mask[0, 5:] = False  # Mask last 3 cells in sample 0

        output = isab(x, mask=mask)

        # Masked positions should be zeroed out (mask applied after skip)
        assert (output[0, 5:] == 0).all(), "Masked positions should be zero"
        # Unmasked positions should be non-zero
        assert (output[0, :5] != 0).any(), "Unmasked positions should be non-zero"

    def test_skip_connection_gradient_magnitude(self):
        """Skip connection should improve gradient flow vs no-skip baseline."""
        import torch
        from src.models.components.set_transformer import ISAB
        torch.manual_seed(42)
        isab = ISAB(d_model=64, n_heads=4, n_inducing=16, dropout=0.0)

        x = torch.randn(2, 10, 64, requires_grad=True)
        output = isab(x)
        output.sum().backward()

        # With skip connection, average gradient magnitude should be substantial
        # (identity Jacobian contributes ~1.0 per element)
        avg_grad = x.grad.abs().mean().item()
        assert avg_grad > 0.5, (
            f"Average |gradient|={avg_grad:.4f}, expected >0.5 with skip connection"
        )


# =============================================================================
# 13. ISAB CELL-TYPE CONDITIONING TESTS
# =============================================================================

class TestISABCellTypeConditioning:
    """Tests for cell-type-conditioned inducing points in ISAB."""

    def test_different_cell_types_produce_different_outputs(self):
        """Different ct_idx values should produce different ISAB outputs."""
        import torch
        import torch.nn as nn
        from src.models.components.set_transformer import ISAB
        torch.manual_seed(42)
        isab = ISAB(d_model=64, n_heads=4, n_inducing=16, n_cell_types=31)

        # Initialize cell_type_embed with non-zero values
        nn.init.normal_(isab.cell_type_embed, std=0.1)

        x = torch.randn(2, 10, 64)
        ct_idx_a = torch.tensor([0, 1])
        ct_idx_b = torch.tensor([15, 25])

        out_a = isab(x, ct_idx=ct_idx_a)
        out_b = isab(x, ct_idx=ct_idx_b)

        assert not torch.allclose(out_a, out_b, atol=1e-6), (
            "Different cell type indices should produce different outputs"
        )

    def test_same_cell_type_same_output(self):
        """Same ct_idx should produce identical outputs."""
        import torch
        from src.models.components.set_transformer import ISAB
        torch.manual_seed(42)
        isab = ISAB(d_model=64, n_heads=4, n_inducing=16, n_cell_types=31)
        isab.eval()

        x = torch.randn(2, 10, 64)
        ct_idx = torch.tensor([5, 5])

        with torch.no_grad():
            out1 = isab(x, ct_idx=ct_idx)
            out2 = isab(x, ct_idx=ct_idx)

        torch.testing.assert_close(out1, out2)

    def test_no_conditioning_without_n_cell_types(self):
        """Without n_cell_types, ct_idx should have no effect."""
        import torch
        from src.models.components.set_transformer import ISAB
        torch.manual_seed(42)
        isab = ISAB(d_model=64, n_heads=4, n_inducing=16)  # no n_cell_types
        isab.eval()

        x = torch.randn(2, 10, 64)
        with torch.no_grad():
            out_none = isab(x, ct_idx=None)
            out_with = isab(x, ct_idx=torch.tensor([0, 1]))

        torch.testing.assert_close(out_none, out_with)

    def test_cell_type_embed_shape(self):
        """cell_type_embed should have shape [n_cell_types, d_model]."""
        from src.models.components.set_transformer import ISAB
        isab = ISAB(d_model=128, n_heads=4, n_inducing=32, n_cell_types=31)
        assert isab.cell_type_embed is not None
        assert isab.cell_type_embed.shape == (31, 128)

    def test_cell_type_embed_none_when_disabled(self):
        """cell_type_embed should be None when n_cell_types not provided."""
        from src.models.components.set_transformer import ISAB
        isab = ISAB(d_model=128, n_heads=4, n_inducing=32)
        assert isab.cell_type_embed is None

    def test_cell_type_embed_zero_init(self):
        """cell_type_embed should be zero-initialized (no change at init)."""
        import torch
        from src.models.components.set_transformer import ISAB
        isab = ISAB(d_model=64, n_heads=4, n_inducing=16, n_cell_types=31)
        assert (isab.cell_type_embed.data == 0).all(), (
            "cell_type_embed should be zero-initialized"
        )

    def test_conditioning_gradient_flows_to_embed(self):
        """Gradients should flow back to cell_type_embed."""
        import torch
        from src.models.components.set_transformer import ISAB
        torch.manual_seed(42)
        isab = ISAB(d_model=64, n_heads=4, n_inducing=16, dropout=0.0, n_cell_types=31)

        x = torch.randn(4, 10, 64)
        ct_idx = torch.tensor([0, 5, 10, 20])
        output = isab(x, ct_idx=ct_idx)
        output.sum().backward()

        assert isab.cell_type_embed.grad is not None
        # Only the used cell types should have non-zero gradient
        used_types = set(ct_idx.tolist())
        for i in range(31):
            if i in used_types:
                assert isab.cell_type_embed.grad[i].abs().sum() > 0, (
                    f"cell_type_embed[{i}] should have gradient (was used)"
                )
            else:
                assert isab.cell_type_embed.grad[i].abs().sum() == 0, (
                    f"cell_type_embed[{i}] should have zero gradient (not used)"
                )


# =============================================================================
# 14. SET TRANSFORMER ENCODER CELL-TYPE CONDITIONING TESTS
# =============================================================================

class TestSetTransformerEncoderCellTypeConditioning:
    """Tests for ct_idx plumbing through SetTransformerEncoder."""

    def test_encoder_accepts_n_cell_types(self):
        """SetTransformerEncoder should accept n_cell_types parameter."""
        from src.models.components.set_transformer import SetTransformerEncoder
        encoder = SetTransformerEncoder(
            d_input=100, d_model=64, n_heads=4, n_cell_types=31,
        )
        # Verify it propagated to ISAB layers
        for isab in encoder.isab_layers:
            assert isab.cell_type_embed is not None
            assert isab.cell_type_embed.shape == (31, 64)

    def test_encoder_accepts_ct_idx_in_forward(self):
        """SetTransformerEncoder.forward should accept and use ct_idx."""
        import torch
        import torch.nn as nn
        from src.models.components.set_transformer import SetTransformerEncoder
        torch.manual_seed(42)
        encoder = SetTransformerEncoder(
            d_input=100, d_model=64, n_heads=4, n_cell_types=31,
        )
        # Initialize embeddings for visible effect
        for isab in encoder.isab_layers:
            nn.init.normal_(isab.cell_type_embed, std=0.1)

        x = torch.randn(4, 10, 100)
        mask = torch.ones(4, 10, dtype=torch.bool)
        ct_a = torch.tensor([0, 1, 2, 3])
        ct_b = torch.tensor([10, 11, 12, 13])

        encoder.eval()
        with torch.no_grad():
            out_a, _ = encoder(x, mask=mask, ct_idx=ct_a)
            out_b, _ = encoder(x, mask=mask, ct_idx=ct_b)

        assert not torch.allclose(out_a, out_b, atol=1e-5), (
            "Different ct_idx should produce different encoder outputs"
        )

    def test_encoder_without_conditioning(self):
        """Encoder without n_cell_types should work the same as before."""
        import torch
        from src.models.components.set_transformer import SetTransformerEncoder
        torch.manual_seed(42)
        encoder = SetTransformerEncoder(d_input=100, d_model=64, n_heads=4)

        x = torch.randn(2, 10, 100)
        encoder.eval()
        with torch.no_grad():
            out1, _ = encoder(x)
            out2, _ = encoder(x, ct_idx=torch.tensor([0, 1]))

        # Without n_cell_types, ct_idx is ignored
        torch.testing.assert_close(out1, out2)

    def test_encoder_ct_idx_with_mixed_batch(self):
        """ct_idx should work correctly with mixed valid/empty samples."""
        import torch
        from src.models.components.set_transformer import SetTransformerEncoder
        torch.manual_seed(42)
        encoder = SetTransformerEncoder(
            d_input=100, d_model=64, n_heads=4, n_cell_types=31,
        )

        x = torch.randn(3, 10, 100)
        mask = torch.ones(3, 10, dtype=torch.bool)
        mask[1, :] = False  # Sample 1 is all-empty
        ct_idx = torch.tensor([0, 5, 10])

        # Should not crash on mixed batch with ct_idx
        output, _ = encoder(x, mask=mask, ct_idx=ct_idx)
        assert output.shape == (3, 64)
        # Empty sample should get empty_embedding (not NaN)
        assert not output[1].isnan().any()
