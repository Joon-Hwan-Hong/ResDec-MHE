"""
Tests for src/models/fusion/pathology_attention.py

Test organization:
1. Initialization - parameter shapes, layer creation, validation
2. Forward pass - output shapes, attention weights properties
3. Pathology bias - different pathology gives different attention
4. Gradient flow - all inputs and parameters receive gradients
5. Validation - invalid inputs at constructor and forward time
"""

import pytest
import torch

from src.data.constants import N_CELL_TYPES


class TestInitialization:
    """Tests for PathologyStratifiedAttention initialization."""

    def test_creates_query_generator(self):
        """Should create linear layer for query generation from pathology."""
        from src.models.fusion.pathology_attention import PathologyStratifiedAttention

        attn = PathologyStratifiedAttention(d_fused=64, d_cond=32, n_heads=4)

        assert hasattr(attn, 'query_generator')
        assert attn.query_generator.in_features == 32
        assert attn.query_generator.out_features == 64

    def test_creates_key_value_projections(self):
        """Should create key and value projection layers."""
        from src.models.fusion.pathology_attention import PathologyStratifiedAttention

        attn = PathologyStratifiedAttention(d_fused=64, d_cond=32, n_heads=4)

        assert hasattr(attn, 'key_proj')
        assert hasattr(attn, 'value_proj')
        assert attn.key_proj.in_features == 64
        assert attn.key_proj.out_features == 64
        assert attn.value_proj.in_features == 64
        assert attn.value_proj.out_features == 64

    def test_creates_pathology_bias(self):
        """Should create pathology bias network (additive, no sigmoid)."""
        from src.models.fusion.pathology_attention import PathologyStratifiedAttention

        attn = PathologyStratifiedAttention(d_fused=64, d_cond=32, n_heads=4)

        assert hasattr(attn, 'pathology_bias')
        # First layer takes d_cond + d_fused, outputs n_heads
        first_layer = attn.pathology_bias[0]
        assert first_layer.in_features == 32 + 64  # d_cond + d_fused
        assert first_layer.out_features == 4  # n_heads
        # Should only have one layer (Linear) — no Sigmoid
        assert len(list(attn.pathology_bias.children())) == 1

    def test_stores_n_heads(self):
        """Should store n_heads and compute d_head correctly."""
        from src.models.fusion.pathology_attention import PathologyStratifiedAttention

        attn = PathologyStratifiedAttention(d_fused=64, d_cond=32, n_heads=4)

        assert attn.n_heads == 4
        assert attn.d_head == 16  # 64 // 4

    def test_stores_d_fused(self):
        """Should store d_fused attribute."""
        from src.models.fusion.pathology_attention import PathologyStratifiedAttention

        attn = PathologyStratifiedAttention(d_fused=128, d_cond=64, n_heads=8)

        assert attn.d_fused == 128

    def test_stores_d_cond(self):
        """Should store d_cond attribute for validation."""
        from src.models.fusion.pathology_attention import PathologyStratifiedAttention

        attn = PathologyStratifiedAttention(d_fused=64, d_cond=32, n_heads=4)

        assert attn.d_cond == 32

    def test_stores_n_cell_types(self):
        """Should store n_cell_types attribute."""
        from src.models.fusion.pathology_attention import PathologyStratifiedAttention

        attn = PathologyStratifiedAttention(d_fused=64, d_cond=32, n_heads=4, n_cell_types=N_CELL_TYPES)

        assert attn.n_cell_types == N_CELL_TYPES

    def test_creates_output_projection(self):
        """Should create output projection layer."""
        from src.models.fusion.pathology_attention import PathologyStratifiedAttention

        attn = PathologyStratifiedAttention(d_fused=64, d_cond=32, n_heads=4)

        assert hasattr(attn, 'out_proj')
        assert attn.out_proj.in_features == 64
        assert attn.out_proj.out_features == 64


class TestForwardPass:
    """Tests for PathologyStratifiedAttention forward pass."""

    def test_output_shapes(self):
        """Forward should return attended [B, d_fused] and weights [B, n_heads, n_cell_types]."""
        from src.models.fusion.pathology_attention import PathologyStratifiedAttention

        attn = PathologyStratifiedAttention(d_fused=64, d_cond=32, n_heads=4, n_cell_types=N_CELL_TYPES)

        cell_type_embs = torch.randn(4, N_CELL_TYPES, 64)
        path_emb = torch.randn(4, 32)

        attended, weights = attn(cell_type_embs, path_emb)

        assert attended.shape == (4, 64)
        assert weights.shape == (4, 4, N_CELL_TYPES)  # [B, n_heads, n_cell_types]

    def test_attention_weights_sum_to_one(self):
        """Attention weights should sum to 1 across cell types for each head."""
        from src.models.fusion.pathology_attention import PathologyStratifiedAttention

        attn = PathologyStratifiedAttention(d_fused=64, d_cond=32, n_heads=4, n_cell_types=N_CELL_TYPES)
        attn.eval()

        cell_type_embs = torch.randn(4, N_CELL_TYPES, 64)
        path_emb = torch.randn(4, 32)

        with torch.no_grad():
            _, weights = attn(cell_type_embs, path_emb)

        # Sum across cell types dimension (dim=-1)
        sums = weights.sum(dim=-1)  # [B, n_heads]

        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_different_batch_sizes(self):
        """Should work with various batch sizes."""
        from src.models.fusion.pathology_attention import PathologyStratifiedAttention

        attn = PathologyStratifiedAttention(d_fused=64, d_cond=32, n_heads=4, n_cell_types=N_CELL_TYPES)

        for B in [1, 2, 8, 16]:
            cell_type_embs = torch.randn(B, N_CELL_TYPES, 64)
            path_emb = torch.randn(B, 32)

            attended, weights = attn(cell_type_embs, path_emb)

            assert attended.shape == (B, 64)
            assert weights.shape == (B, 4, N_CELL_TYPES)

    def test_different_n_cell_types(self):
        """Should work with different n_cell_types configurations."""
        from src.models.fusion.pathology_attention import PathologyStratifiedAttention

        for n_cell_types in [10, 20, N_CELL_TYPES, 50]:
            attn = PathologyStratifiedAttention(
                d_fused=64, d_cond=32, n_heads=4, n_cell_types=n_cell_types
            )

            cell_type_embs = torch.randn(4, n_cell_types, 64)
            path_emb = torch.randn(4, 32)

            attended, weights = attn(cell_type_embs, path_emb)

            assert attended.shape == (4, 64)
            assert weights.shape == (4, 4, n_cell_types)

    def test_attention_weights_non_negative(self):
        """Attention weights should be non-negative (softmax output)."""
        from src.models.fusion.pathology_attention import PathologyStratifiedAttention

        attn = PathologyStratifiedAttention(d_fused=64, d_cond=32, n_heads=4, n_cell_types=N_CELL_TYPES)
        attn.eval()

        cell_type_embs = torch.randn(4, N_CELL_TYPES, 64)
        path_emb = torch.randn(4, 32)

        with torch.no_grad():
            _, weights = attn(cell_type_embs, path_emb)

        assert torch.all(weights >= 0)


class TestPathologyBias:
    """Tests for pathology-dependent additive attention bias."""

    def test_different_pathology_gives_different_attention(self):
        """Different pathology embeddings should produce different attention patterns."""
        from src.models.fusion.pathology_attention import PathologyStratifiedAttention

        attn = PathologyStratifiedAttention(d_fused=64, d_cond=32, n_heads=4, n_cell_types=N_CELL_TYPES)
        attn.eval()

        cell_type_embs = torch.randn(1, N_CELL_TYPES, 64)
        path_low = torch.zeros(1, 32)
        path_high = torch.ones(1, 32)

        with torch.no_grad():
            _, weights_low = attn(cell_type_embs, path_low)
            _, weights_high = attn(cell_type_embs, path_high)

        assert not torch.allclose(weights_low, weights_high)

    def test_same_input_gives_same_output(self):
        """Same inputs should produce same outputs (deterministic)."""
        from src.models.fusion.pathology_attention import PathologyStratifiedAttention

        attn = PathologyStratifiedAttention(d_fused=64, d_cond=32, n_heads=4, n_cell_types=N_CELL_TYPES)
        attn.eval()

        cell_type_embs = torch.randn(2, N_CELL_TYPES, 64)
        path_emb = torch.randn(2, 32)

        with torch.no_grad():
            attended1, weights1 = attn(cell_type_embs, path_emb)
            attended2, weights2 = attn(cell_type_embs, path_emb)

        assert torch.allclose(attended1, attended2)
        assert torch.allclose(weights1, weights2)

    def test_pathology_bias_affects_scores(self):
        """Pathology bias should shift attention scores additively."""
        from src.models.fusion.pathology_attention import PathologyStratifiedAttention

        # Create with known seed for reproducibility
        torch.manual_seed(42)
        attn = PathologyStratifiedAttention(d_fused=64, d_cond=32, n_heads=4, n_cell_types=N_CELL_TYPES)
        attn.eval()

        cell_type_embs = torch.randn(1, N_CELL_TYPES, 64)
        # Create very different pathology embeddings
        path_emb1 = torch.randn(1, 32) * 5
        path_emb2 = -path_emb1

        with torch.no_grad():
            attended1, _ = attn(cell_type_embs, path_emb1)
            attended2, _ = attn(cell_type_embs, path_emb2)

        # Attended outputs should differ
        assert not torch.allclose(attended1, attended2)


class TestGradientFlow:
    """Tests for gradient flow through PathologyStratifiedAttention."""

    def test_gradients_flow_to_cell_embeddings(self):
        """Gradients should reach cell type embeddings input."""
        from src.models.fusion.pathology_attention import PathologyStratifiedAttention

        attn = PathologyStratifiedAttention(d_fused=64, d_cond=32, n_heads=4, n_cell_types=N_CELL_TYPES)

        cell_type_embs = torch.randn(2, N_CELL_TYPES, 64, requires_grad=True)
        path_emb = torch.randn(2, 32)

        attended, weights = attn(cell_type_embs, path_emb)
        loss = attended.sum()
        loss.backward()

        assert cell_type_embs.grad is not None
        assert not torch.all(cell_type_embs.grad == 0)

    def test_gradients_flow_to_path_emb(self):
        """Gradients should reach pathology embedding input."""
        from src.models.fusion.pathology_attention import PathologyStratifiedAttention

        attn = PathologyStratifiedAttention(d_fused=64, d_cond=32, n_heads=4, n_cell_types=N_CELL_TYPES)

        cell_type_embs = torch.randn(2, N_CELL_TYPES, 64)
        path_emb = torch.randn(2, 32, requires_grad=True)

        attended, weights = attn(cell_type_embs, path_emb)
        loss = attended.sum()
        loss.backward()

        assert path_emb.grad is not None
        assert not torch.all(path_emb.grad == 0)

    def test_gradients_flow_to_bias(self):
        """Gradients should reach pathology bias parameters."""
        from src.models.fusion.pathology_attention import PathologyStratifiedAttention

        attn = PathologyStratifiedAttention(d_fused=64, d_cond=32, n_heads=4, n_cell_types=N_CELL_TYPES)

        cell_type_embs = torch.randn(2, N_CELL_TYPES, 64)
        path_emb = torch.randn(2, 32)

        attended, weights = attn(cell_type_embs, path_emb)
        loss = attended.sum()
        loss.backward()

        # Check pathology_bias has gradients
        for name, param in attn.pathology_bias.named_parameters():
            assert param.grad is not None, f"No gradient for pathology_bias.{name}"
            assert not torch.all(param.grad == 0), f"Zero gradient for pathology_bias.{name}"

    def test_gradients_flow_to_all_parameters(self):
        """Gradients should reach all module parameters."""
        from src.models.fusion.pathology_attention import PathologyStratifiedAttention

        attn = PathologyStratifiedAttention(d_fused=64, d_cond=32, n_heads=4, n_cell_types=N_CELL_TYPES)

        cell_type_embs = torch.randn(2, N_CELL_TYPES, 64, requires_grad=True)
        path_emb = torch.randn(2, 32, requires_grad=True)

        attended, weights = attn(cell_type_embs, path_emb)
        loss = attended.sum()
        loss.backward()

        for name, param in attn.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.all(param.grad == 0), f"Zero gradient for {name}"


class TestValidation:
    """Tests for input validation."""

    # Constructor validation tests
    def test_rejects_invalid_d_fused(self):
        """Should reject non-positive d_fused."""
        from src.models.fusion.pathology_attention import PathologyStratifiedAttention

        with pytest.raises(ValueError, match="d_fused must be positive"):
            PathologyStratifiedAttention(d_fused=0, d_cond=32, n_heads=4)

        with pytest.raises(ValueError, match="d_fused must be positive"):
            PathologyStratifiedAttention(d_fused=-1, d_cond=32, n_heads=4)

    def test_rejects_invalid_d_cond(self):
        """Should reject non-positive d_cond."""
        from src.models.fusion.pathology_attention import PathologyStratifiedAttention

        with pytest.raises(ValueError, match="d_cond must be positive"):
            PathologyStratifiedAttention(d_fused=64, d_cond=0, n_heads=4)

        with pytest.raises(ValueError, match="d_cond must be positive"):
            PathologyStratifiedAttention(d_fused=64, d_cond=-1, n_heads=4)

    def test_rejects_invalid_n_heads(self):
        """Should reject non-positive n_heads."""
        from src.models.fusion.pathology_attention import PathologyStratifiedAttention

        with pytest.raises(ValueError, match="n_heads must be positive"):
            PathologyStratifiedAttention(d_fused=64, d_cond=32, n_heads=0)

        with pytest.raises(ValueError, match="n_heads must be positive"):
            PathologyStratifiedAttention(d_fused=64, d_cond=32, n_heads=-1)

    def test_rejects_invalid_n_cell_types(self):
        """Should reject non-positive n_cell_types."""
        from src.models.fusion.pathology_attention import PathologyStratifiedAttention

        with pytest.raises(ValueError, match="n_cell_types must be positive"):
            PathologyStratifiedAttention(d_fused=64, d_cond=32, n_heads=4, n_cell_types=0)

        with pytest.raises(ValueError, match="n_cell_types must be positive"):
            PathologyStratifiedAttention(d_fused=64, d_cond=32, n_heads=4, n_cell_types=-1)

    def test_rejects_d_fused_not_divisible_by_n_heads(self):
        """Should reject d_fused that is not divisible by n_heads."""
        from src.models.fusion.pathology_attention import PathologyStratifiedAttention

        with pytest.raises(ValueError, match="d_fused.*must be divisible by n_heads"):
            PathologyStratifiedAttention(d_fused=65, d_cond=32, n_heads=4)

        with pytest.raises(ValueError, match="d_fused.*must be divisible by n_heads"):
            PathologyStratifiedAttention(d_fused=64, d_cond=32, n_heads=5)

    # Forward pass validation tests
    def test_rejects_wrong_cell_embeddings_dim(self):
        """Should reject cell_type_embeddings with wrong number of dimensions."""
        from src.models.fusion.pathology_attention import PathologyStratifiedAttention

        attn = PathologyStratifiedAttention(d_fused=64, d_cond=32, n_heads=4, n_cell_types=N_CELL_TYPES)

        # 2D tensor (missing batch dimension)
        cell_type_embs_2d = torch.randn(N_CELL_TYPES, 64)
        path_emb = torch.randn(2, 32)

        with pytest.raises(ValueError, match="Expected 3D cell_type_embeddings"):
            attn(cell_type_embs_2d, path_emb)

        # 4D tensor
        cell_type_embs_4d = torch.randn(2, N_CELL_TYPES, 64, 1)
        with pytest.raises(ValueError, match="Expected 3D cell_type_embeddings"):
            attn(cell_type_embs_4d, path_emb)

    def test_rejects_wrong_path_emb_dim(self):
        """Should reject path_emb with wrong number of dimensions."""
        from src.models.fusion.pathology_attention import PathologyStratifiedAttention

        attn = PathologyStratifiedAttention(d_fused=64, d_cond=32, n_heads=4, n_cell_types=N_CELL_TYPES)

        cell_type_embs = torch.randn(2, N_CELL_TYPES, 64)

        # 1D tensor (missing batch dimension)
        path_emb_1d = torch.randn(32)
        with pytest.raises(ValueError, match="Expected 2D path_emb"):
            attn(cell_type_embs, path_emb_1d)

        # 3D tensor
        path_emb_3d = torch.randn(2, 32, 1)
        with pytest.raises(ValueError, match="Expected 2D path_emb"):
            attn(cell_type_embs, path_emb_3d)

    def test_rejects_wrong_n_cell_types(self):
        """Should reject cell_type_embeddings with wrong n_cell_types."""
        from src.models.fusion.pathology_attention import PathologyStratifiedAttention

        attn = PathologyStratifiedAttention(d_fused=64, d_cond=32, n_heads=4, n_cell_types=N_CELL_TYPES)

        cell_type_embs = torch.randn(2, 20, 64)  # Wrong: 20 instead of 31
        path_emb = torch.randn(2, 32)

        with pytest.raises(ValueError, match=f"Expected {N_CELL_TYPES} cell types"):
            attn(cell_type_embs, path_emb)

    def test_rejects_wrong_d_fused(self):
        """Should reject cell_type_embeddings with wrong d_fused."""
        from src.models.fusion.pathology_attention import PathologyStratifiedAttention

        attn = PathologyStratifiedAttention(d_fused=64, d_cond=32, n_heads=4, n_cell_types=N_CELL_TYPES)

        cell_type_embs = torch.randn(2, N_CELL_TYPES, 128)  # Wrong: 128 instead of 64
        path_emb = torch.randn(2, 32)

        with pytest.raises(ValueError, match="Expected d_fused=64"):
            attn(cell_type_embs, path_emb)

    def test_rejects_wrong_d_cond(self):
        """Should reject path_emb with wrong d_cond."""
        from src.models.fusion.pathology_attention import PathologyStratifiedAttention

        attn = PathologyStratifiedAttention(d_fused=64, d_cond=32, n_heads=4, n_cell_types=N_CELL_TYPES)

        cell_type_embs = torch.randn(2, N_CELL_TYPES, 64)
        path_emb = torch.randn(2, 64)  # Wrong: 64 instead of 32

        with pytest.raises(ValueError, match="Expected d_cond=32"):
            attn(cell_type_embs, path_emb)

    def test_rejects_batch_size_mismatch(self):
        """Should reject mismatched batch sizes."""
        from src.models.fusion.pathology_attention import PathologyStratifiedAttention

        attn = PathologyStratifiedAttention(d_fused=64, d_cond=32, n_heads=4, n_cell_types=N_CELL_TYPES)

        cell_type_embs = torch.randn(4, N_CELL_TYPES, 64)
        path_emb = torch.randn(8, 32)  # Different batch size

        with pytest.raises(ValueError, match="Batch size mismatch"):
            attn(cell_type_embs, path_emb)


class TestExtraRepr:
    """Tests for extra_repr method."""

    def test_repr_contains_parameters(self):
        """extra_repr and str should show key parameters."""
        from src.models.fusion.pathology_attention import PathologyStratifiedAttention

        attn = PathologyStratifiedAttention(d_fused=128, d_cond=64, n_heads=8, n_cell_types=N_CELL_TYPES)

        repr_str = attn.extra_repr()
        assert "d_fused=128" in repr_str
        assert "d_cond=64" in repr_str
        assert "n_heads=8" in repr_str
        assert f"n_cell_types={N_CELL_TYPES}" in repr_str

        str_repr = str(attn)
        assert "d_fused=128" in str_repr
        assert "d_cond=64" in str_repr
        assert "n_heads=8" in str_repr
        assert f"n_cell_types={N_CELL_TYPES}" in str_repr


class TestAMPSoftmaxPromotion:
    """Manual softmax must promote to float32 under AMP for numerical stability."""

    def test_softmax_stable_with_large_float16_scores(self):
        """Attention weights should remain valid even with large-magnitude float16 inputs."""
        from src.models.fusion.pathology_attention import PathologyStratifiedAttention

        module = PathologyStratifiedAttention(
            d_fused=64, d_cond=32, n_heads=4, n_cell_types=31
        )
        module = module.half()

        cell_emb = torch.randn(2, 31, 64, dtype=torch.float16)
        path_emb = torch.randn(2, 32, dtype=torch.float16)

        # Large-magnitude inputs that would cause softmax saturation in float16
        cell_emb_large = cell_emb * 100.0
        _, attn_weights = module(cell_emb_large, path_emb)

        # Attention should sum to ~1 per head, not degenerate
        attn_sums = attn_weights.sum(dim=-1)
        assert torch.allclose(attn_sums, torch.ones_like(attn_sums), atol=0.05), \
            f"Attention sums should be ~1.0, got {attn_sums}"
        assert not torch.isnan(attn_weights).any(), "Attention weights contain NaN"


class TestCellTypeMasking:
    """PA-G1/G2/G3: Tests for cell_type_mask masking behavior."""

    def test_masked_cell_types_get_zero_attention(self):
        """PA-G1: Masked cell types should get zero attention weight."""
        from src.models.fusion.pathology_attention import PathologyStratifiedAttention

        attn = PathologyStratifiedAttention(d_fused=64, d_cond=32, n_heads=4, n_cell_types=8)
        attn.eval()

        cell_embs = torch.randn(2, 8, 64)
        path_emb = torch.randn(2, 32)

        # Mask out last 4 cell types
        mask = torch.ones(2, 8, dtype=torch.bool)
        mask[:, 4:] = False

        with torch.no_grad():
            attended, weights = attn(cell_embs, path_emb, cell_type_mask=mask)

        # Masked cell types should have zero attention
        assert torch.allclose(weights[:, :, 4:], torch.zeros_like(weights[:, :, 4:]), atol=1e-6)
        # Unmasked cell types should have non-zero attention that sums to 1
        unmasked_sum = weights[:, :, :4].sum(dim=-1)
        assert torch.allclose(unmasked_sum, torch.ones_like(unmasked_sum), atol=1e-5)

    def test_all_masked_produces_zero_attention_and_output(self):
        """PA-G2: All cell types masked should produce zero attention weights and near-zero output."""
        from src.models.fusion.pathology_attention import PathologyStratifiedAttention

        attn = PathologyStratifiedAttention(d_fused=64, d_cond=32, n_heads=4, n_cell_types=8)
        attn.eval()

        cell_embs = torch.randn(2, 8, 64)
        path_emb = torch.randn(2, 32)

        # All cell types masked
        mask = torch.zeros(2, 8, dtype=torch.bool)

        with torch.no_grad():
            attended, weights = attn(cell_embs, path_emb, cell_type_mask=mask)

        # Attention weights should all be zero
        assert torch.allclose(weights, torch.zeros_like(weights), atol=1e-6)
        # Output should be finite (not NaN)
        assert torch.isfinite(attended).all()

    def test_all_masked_no_nan_in_gradients(self):
        """PA-G2: All-masked should not produce NaN gradients."""
        from src.models.fusion.pathology_attention import PathologyStratifiedAttention

        attn = PathologyStratifiedAttention(d_fused=64, d_cond=32, n_heads=4, n_cell_types=8)

        cell_embs = torch.randn(2, 8, 64, requires_grad=True)
        path_emb = torch.randn(2, 32, requires_grad=True)
        mask = torch.zeros(2, 8, dtype=torch.bool)

        attended, weights = attn(cell_embs, path_emb, cell_type_mask=mask)
        loss = attended.sum()
        loss.backward()

        # No NaN in gradients
        assert torch.isfinite(cell_embs.grad).all()
        assert torch.isfinite(path_emb.grad).all()
        for p in attn.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"NaN gradient in {p.shape}"

    def test_mixed_batch_partial_masking(self):
        """PA-G3: Mixed batch where some samples are fully masked and others have valid cells."""
        from src.models.fusion.pathology_attention import PathologyStratifiedAttention

        attn = PathologyStratifiedAttention(d_fused=64, d_cond=32, n_heads=4, n_cell_types=8)
        attn.eval()

        cell_embs = torch.randn(3, 8, 64)
        path_emb = torch.randn(3, 32)

        mask = torch.zeros(3, 8, dtype=torch.bool)
        mask[0, :4] = True   # Sample 0: first 4 valid
        mask[1, :] = False    # Sample 1: all masked
        mask[2, :] = True     # Sample 2: all valid

        with torch.no_grad():
            attended, weights = attn(cell_embs, path_emb, cell_type_mask=mask)

        # Sample 0: masked cells get 0, unmasked sum to 1
        assert torch.allclose(weights[0, :, 4:], torch.zeros(4, 4), atol=1e-6)
        assert torch.allclose(weights[0, :, :4].sum(dim=-1), torch.ones(4), atol=1e-5)

        # Sample 1: all zero attention
        assert torch.allclose(weights[1], torch.zeros_like(weights[1]), atol=1e-6)

        # Sample 2: all cells valid, sum to 1
        assert torch.allclose(weights[2].sum(dim=-1), torch.ones(4), atol=1e-5)

        # All outputs should be finite
        assert torch.isfinite(attended).all()

    def test_mask_changes_output(self):
        """PA-G1: Applying a mask should change the output compared to no mask."""
        from src.models.fusion.pathology_attention import PathologyStratifiedAttention

        attn = PathologyStratifiedAttention(d_fused=64, d_cond=32, n_heads=4, n_cell_types=8)
        attn.eval()

        cell_embs = torch.randn(2, 8, 64)
        path_emb = torch.randn(2, 32)

        with torch.no_grad():
            attended_no_mask, _ = attn(cell_embs, path_emb)

            # Mask out half the cell types
            mask = torch.ones(2, 8, dtype=torch.bool)
            mask[:, 4:] = False
            attended_masked, _ = attn(cell_embs, path_emb, cell_type_mask=mask)

        assert not torch.allclose(attended_no_mask, attended_masked, atol=1e-6)


class TestComputeAttentionWithGrad:
    """Flag-gated path that keeps attention in the autograd graph."""

    def test_default_flag_is_false(self):
        from src.models.fusion.pathology_attention import PathologyStratifiedAttention

        attn = PathologyStratifiedAttention(
            d_fused=64, d_cond=32, n_heads=4, n_cell_types=8,
        )
        assert attn.compute_attention_with_grad is False

    def test_grad_path_returns_same_shape(self):
        from src.models.fusion.pathology_attention import PathologyStratifiedAttention

        attn = PathologyStratifiedAttention(
            d_fused=64, d_cond=32, n_heads=4, n_cell_types=8,
        )
        attn.compute_attention_with_grad = True
        cell_embs = torch.randn(2, 8, 64)
        path_emb = torch.randn(2, 32)
        attended, weights = attn(cell_embs, path_emb)
        assert attended.shape == (2, 64)
        assert weights.shape == (2, 4, 8)
        # Each (batch, head) row sums to 1 (proper softmax distribution)
        torch.testing.assert_close(
            weights.sum(dim=-1), torch.ones(2, 4), atol=1e-5, rtol=1e-5,
        )

    def test_grad_path_attention_is_differentiable(self):
        """Attention weights MUST flow gradients in the grad path. In the
        canonical SDPA path attention is detached; here it should backprop
        into the encoder."""
        from src.models.fusion.pathology_attention import PathologyStratifiedAttention

        attn = PathologyStratifiedAttention(
            d_fused=64, d_cond=32, n_heads=4, n_cell_types=8,
        )
        attn.compute_attention_with_grad = True
        cell_embs = torch.randn(2, 8, 64, requires_grad=True)
        path_emb = torch.randn(2, 32, requires_grad=True)
        _, weights = attn(cell_embs, path_emb)
        # Loss on attention weights (e.g. entropy bonus): if backprop flows,
        # cell_embs and path_emb must receive non-zero gradients.
        loss = -(weights * (weights + 1e-12).log()).sum(dim=-1).mean()
        loss.backward()
        assert cell_embs.grad is not None
        assert path_emb.grad is not None
        assert cell_embs.grad.abs().sum() > 0
        assert path_emb.grad.abs().sum() > 0

    def test_grad_path_canonical_path_numerical_close(self):
        """The grad path's attended output should be close to the canonical
        SDPA path output (drift expected from FlashAttention reorderings)."""
        from src.models.fusion.pathology_attention import PathologyStratifiedAttention

        torch.manual_seed(0)
        attn = PathologyStratifiedAttention(
            d_fused=64, d_cond=32, n_heads=4, n_cell_types=8,
        )
        attn.eval()
        cell_embs = torch.randn(2, 8, 64)
        path_emb = torch.randn(2, 32)
        # Canonical SDPA path
        attn.compute_attention_with_grad = False
        with torch.no_grad():
            canonical_att, _ = attn(cell_embs, path_emb)
        # Grad-enabled einsum path
        attn.compute_attention_with_grad = True
        with torch.no_grad():
            grad_att, _ = attn(cell_embs, path_emb)
        torch.testing.assert_close(canonical_att, grad_att, atol=1e-4, rtol=1e-4)


class TestNoGradBlockSkippedDuringTraining:
    """§31.7 fix: when `return_attention_weights=False` and
    `compute_attention_with_grad=False`, the SDPA fast path runs WITHOUT the
    no_grad einsum+softmax re-compute block. This prevents the cudaMallocAsync
    pool perturbation that shifts SDPA's non-deterministic backward atomic-add
    reduction order (~0.07 R² regression). The full_model.py refactor at lines
    549/676 ensures `return_attention_weights=False` is passed during training,
    so this contract is what the fix relies on."""

    def test_no_grad_block_skipped_when_weights_not_requested(self):
        """SDPA path with return_attention_weights=False returns None for
        attention_weights, i.e. the no_grad einsum block at lines 224-234 of
        pathology_attention.py is bypassed."""
        from src.models.fusion.pathology_attention import PathologyStratifiedAttention

        attn = PathologyStratifiedAttention(
            d_fused=64, d_cond=32, n_heads=4, n_cell_types=8,
        )
        attn.train()  # simulate training mode (the regression scenario)
        attn.compute_attention_with_grad = False  # SDPA fast path
        cell_embs = torch.randn(2, 8, 64)
        path_emb = torch.randn(2, 32)
        attended, weights = attn(
            cell_embs, path_emb, return_attention_weights=False,
        )
        assert weights is None, (
            "Expected attention_weights=None when return_attention_weights=False; "
            "the no_grad einsum re-compute block must NOT fire during training."
        )
        # attended must still be a valid SDPA forward output
        assert attended.shape == (2, 64)
        assert torch.isfinite(attended).all()

    def test_sdpa_attended_output_independent_of_flag(self):
        """The SDPA path's attended output is identical whether or not the
        no_grad re-compute block runs. This guarantees the forward pass is
        bit-equivalent — only the backward atomic-add order is affected by
        the eliminated allocations."""
        from src.models.fusion.pathology_attention import PathologyStratifiedAttention

        torch.manual_seed(0)
        attn = PathologyStratifiedAttention(
            d_fused=64, d_cond=32, n_heads=4, n_cell_types=8,
        )
        attn.eval()
        attn.compute_attention_with_grad = False  # SDPA path
        cell_embs = torch.randn(2, 8, 64)
        path_emb = torch.randn(2, 32)
        with torch.no_grad():
            attended_with_block, weights_with = attn(
                cell_embs, path_emb, return_attention_weights=True,
            )
            attended_no_block, weights_no = attn(
                cell_embs, path_emb, return_attention_weights=False,
            )
        torch.testing.assert_close(
            attended_with_block, attended_no_block, atol=1e-6, rtol=1e-6,
        )
        assert weights_with is not None
        assert weights_no is None
