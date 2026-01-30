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

        attn = PathologyStratifiedAttention(d_fused=64, d_cond=32, n_heads=4, n_cell_types=31)

        assert attn.n_cell_types == 31

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

        attn = PathologyStratifiedAttention(d_fused=64, d_cond=32, n_heads=4, n_cell_types=31)

        cell_type_embs = torch.randn(4, 31, 64)
        path_emb = torch.randn(4, 32)

        attended, weights = attn(cell_type_embs, path_emb)

        assert attended.shape == (4, 64)
        assert weights.shape == (4, 4, 31)  # [B, n_heads, n_cell_types]

    def test_attention_weights_sum_to_one(self):
        """Attention weights should sum to 1 across cell types for each head."""
        from src.models.fusion.pathology_attention import PathologyStratifiedAttention

        attn = PathologyStratifiedAttention(d_fused=64, d_cond=32, n_heads=4, n_cell_types=31)
        attn.eval()

        cell_type_embs = torch.randn(4, 31, 64)
        path_emb = torch.randn(4, 32)

        with torch.no_grad():
            _, weights = attn(cell_type_embs, path_emb)

        # Sum across cell types dimension (dim=-1)
        sums = weights.sum(dim=-1)  # [B, n_heads]

        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_different_batch_sizes(self):
        """Should work with various batch sizes."""
        from src.models.fusion.pathology_attention import PathologyStratifiedAttention

        attn = PathologyStratifiedAttention(d_fused=64, d_cond=32, n_heads=4, n_cell_types=31)

        for B in [1, 2, 8, 16]:
            cell_type_embs = torch.randn(B, 31, 64)
            path_emb = torch.randn(B, 32)

            attended, weights = attn(cell_type_embs, path_emb)

            assert attended.shape == (B, 64)
            assert weights.shape == (B, 4, 31)

    def test_different_n_cell_types(self):
        """Should work with different n_cell_types configurations."""
        from src.models.fusion.pathology_attention import PathologyStratifiedAttention

        for n_cell_types in [10, 20, 31, 50]:
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

        attn = PathologyStratifiedAttention(d_fused=64, d_cond=32, n_heads=4, n_cell_types=31)
        attn.eval()

        cell_type_embs = torch.randn(4, 31, 64)
        path_emb = torch.randn(4, 32)

        with torch.no_grad():
            _, weights = attn(cell_type_embs, path_emb)

        assert torch.all(weights >= 0)


class TestPathologyBias:
    """Tests for pathology-dependent additive attention bias."""

    def test_different_pathology_gives_different_attention(self):
        """Different pathology embeddings should produce different attention patterns."""
        from src.models.fusion.pathology_attention import PathologyStratifiedAttention

        attn = PathologyStratifiedAttention(d_fused=64, d_cond=32, n_heads=4, n_cell_types=31)
        attn.eval()

        cell_type_embs = torch.randn(1, 31, 64)
        path_low = torch.zeros(1, 32)
        path_high = torch.ones(1, 32)

        with torch.no_grad():
            _, weights_low = attn(cell_type_embs, path_low)
            _, weights_high = attn(cell_type_embs, path_high)

        assert not torch.allclose(weights_low, weights_high)

    def test_same_input_gives_same_output(self):
        """Same inputs should produce same outputs (deterministic)."""
        from src.models.fusion.pathology_attention import PathologyStratifiedAttention

        attn = PathologyStratifiedAttention(d_fused=64, d_cond=32, n_heads=4, n_cell_types=31)
        attn.eval()

        cell_type_embs = torch.randn(2, 31, 64)
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
        attn = PathologyStratifiedAttention(d_fused=64, d_cond=32, n_heads=4, n_cell_types=31)
        attn.eval()

        cell_type_embs = torch.randn(1, 31, 64)
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

        attn = PathologyStratifiedAttention(d_fused=64, d_cond=32, n_heads=4, n_cell_types=31)

        cell_type_embs = torch.randn(2, 31, 64, requires_grad=True)
        path_emb = torch.randn(2, 32)

        attended, weights = attn(cell_type_embs, path_emb)
        loss = attended.sum()
        loss.backward()

        assert cell_type_embs.grad is not None
        assert not torch.all(cell_type_embs.grad == 0)

    def test_gradients_flow_to_path_emb(self):
        """Gradients should reach pathology embedding input."""
        from src.models.fusion.pathology_attention import PathologyStratifiedAttention

        attn = PathologyStratifiedAttention(d_fused=64, d_cond=32, n_heads=4, n_cell_types=31)

        cell_type_embs = torch.randn(2, 31, 64)
        path_emb = torch.randn(2, 32, requires_grad=True)

        attended, weights = attn(cell_type_embs, path_emb)
        loss = attended.sum()
        loss.backward()

        assert path_emb.grad is not None
        assert not torch.all(path_emb.grad == 0)

    def test_gradients_flow_to_bias(self):
        """Gradients should reach pathology bias parameters."""
        from src.models.fusion.pathology_attention import PathologyStratifiedAttention

        attn = PathologyStratifiedAttention(d_fused=64, d_cond=32, n_heads=4, n_cell_types=31)

        cell_type_embs = torch.randn(2, 31, 64)
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

        attn = PathologyStratifiedAttention(d_fused=64, d_cond=32, n_heads=4, n_cell_types=31)

        cell_type_embs = torch.randn(2, 31, 64, requires_grad=True)
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

        attn = PathologyStratifiedAttention(d_fused=64, d_cond=32, n_heads=4, n_cell_types=31)

        # 2D tensor (missing batch dimension)
        cell_type_embs_2d = torch.randn(31, 64)
        path_emb = torch.randn(2, 32)

        with pytest.raises(ValueError, match="Expected 3D cell_type_embeddings"):
            attn(cell_type_embs_2d, path_emb)

        # 4D tensor
        cell_type_embs_4d = torch.randn(2, 31, 64, 1)
        with pytest.raises(ValueError, match="Expected 3D cell_type_embeddings"):
            attn(cell_type_embs_4d, path_emb)

    def test_rejects_wrong_path_emb_dim(self):
        """Should reject path_emb with wrong number of dimensions."""
        from src.models.fusion.pathology_attention import PathologyStratifiedAttention

        attn = PathologyStratifiedAttention(d_fused=64, d_cond=32, n_heads=4, n_cell_types=31)

        cell_type_embs = torch.randn(2, 31, 64)

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

        attn = PathologyStratifiedAttention(d_fused=64, d_cond=32, n_heads=4, n_cell_types=31)

        cell_type_embs = torch.randn(2, 20, 64)  # Wrong: 20 instead of 31
        path_emb = torch.randn(2, 32)

        with pytest.raises(ValueError, match="Expected 31 cell types"):
            attn(cell_type_embs, path_emb)

    def test_rejects_wrong_d_fused(self):
        """Should reject cell_type_embeddings with wrong d_fused."""
        from src.models.fusion.pathology_attention import PathologyStratifiedAttention

        attn = PathologyStratifiedAttention(d_fused=64, d_cond=32, n_heads=4, n_cell_types=31)

        cell_type_embs = torch.randn(2, 31, 128)  # Wrong: 128 instead of 64
        path_emb = torch.randn(2, 32)

        with pytest.raises(ValueError, match="Expected d_fused=64"):
            attn(cell_type_embs, path_emb)

    def test_rejects_wrong_d_cond(self):
        """Should reject path_emb with wrong d_cond."""
        from src.models.fusion.pathology_attention import PathologyStratifiedAttention

        attn = PathologyStratifiedAttention(d_fused=64, d_cond=32, n_heads=4, n_cell_types=31)

        cell_type_embs = torch.randn(2, 31, 64)
        path_emb = torch.randn(2, 64)  # Wrong: 64 instead of 32

        with pytest.raises(ValueError, match="Expected d_cond=32"):
            attn(cell_type_embs, path_emb)

    def test_rejects_batch_size_mismatch(self):
        """Should reject mismatched batch sizes."""
        from src.models.fusion.pathology_attention import PathologyStratifiedAttention

        attn = PathologyStratifiedAttention(d_fused=64, d_cond=32, n_heads=4, n_cell_types=31)

        cell_type_embs = torch.randn(4, 31, 64)
        path_emb = torch.randn(8, 32)  # Different batch size

        with pytest.raises(ValueError, match="Batch size mismatch"):
            attn(cell_type_embs, path_emb)


class TestExtraRepr:
    """Tests for extra_repr method."""

    def test_extra_repr_contains_parameters(self):
        """extra_repr should show key parameters."""
        from src.models.fusion.pathology_attention import PathologyStratifiedAttention

        attn = PathologyStratifiedAttention(d_fused=128, d_cond=64, n_heads=8, n_cell_types=31)

        repr_str = attn.extra_repr()

        assert "d_fused=128" in repr_str
        assert "d_cond=64" in repr_str
        assert "n_heads=8" in repr_str
        assert "n_cell_types=31" in repr_str

    def test_str_contains_extra_repr(self):
        """String representation should include extra_repr info."""
        from src.models.fusion.pathology_attention import PathologyStratifiedAttention

        attn = PathologyStratifiedAttention(d_fused=64, d_cond=32, n_heads=4, n_cell_types=31)

        str_repr = str(attn)

        assert "d_fused=64" in str_repr
        assert "d_cond=32" in str_repr
        assert "n_heads=4" in str_repr
        assert "n_cell_types=31" in str_repr
