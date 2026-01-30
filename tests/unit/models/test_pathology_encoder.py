"""
Tests for src/models/fusion/pathology_encoder.py

Test organization:
1. Initialization - parameter shapes, layer creation
2. Forward pass - output shapes, correctness
3. Gradient flow - all inputs and parameters
4. Validation - invalid inputs
"""

import pytest
import torch


class TestInitialization:
    """Tests for PathologyEncoder initialization."""

    def test_creates_pathology_mlp(self):
        """Should create MLP for pathology encoding."""
        from src.models.fusion.pathology_encoder import PathologyEncoder

        encoder = PathologyEncoder(n_pathology_features=3, d_region=128, d_cond=64)

        assert hasattr(encoder, 'pathology_mlp')

    def test_creates_region_projection(self):
        """Should create linear projection for region context."""
        from src.models.fusion.pathology_encoder import PathologyEncoder

        encoder = PathologyEncoder(n_pathology_features=3, d_region=128, d_cond=64)

        assert encoder.region_proj.in_features == 128
        assert encoder.region_proj.out_features == 64

    def test_creates_combine_layer(self):
        """Should create combination MLP."""
        from src.models.fusion.pathology_encoder import PathologyEncoder

        encoder = PathologyEncoder(d_cond=64)

        assert hasattr(encoder, 'combine')

    def test_default_parameters(self):
        """Should use sensible defaults."""
        from src.models.fusion.pathology_encoder import PathologyEncoder

        encoder = PathologyEncoder()

        assert encoder.region_proj.in_features == 128
        assert encoder.region_proj.out_features == 64

    def test_custom_parameters(self):
        """Should accept custom parameters."""
        from src.models.fusion.pathology_encoder import PathologyEncoder

        encoder = PathologyEncoder(n_pathology_features=5, d_region=256, d_cond=128)

        assert encoder.n_pathology_features == 5
        assert encoder.d_region == 256
        assert encoder.d_cond == 128
        assert encoder.region_proj.in_features == 256
        assert encoder.region_proj.out_features == 128


class TestForwardPass:
    """Tests for PathologyEncoder forward pass."""

    def test_output_shape(self):
        """Forward should return [B, d_cond]."""
        from src.models.fusion.pathology_encoder import PathologyEncoder

        encoder = PathologyEncoder(n_pathology_features=3, d_region=128, d_cond=64)

        pathology = torch.randn(4, 3)
        region_context = torch.randn(4, 128)

        output = encoder(pathology, region_context)

        assert output.shape == (4, 64)

    def test_different_batch_sizes(self):
        """Should work with various batch sizes."""
        from src.models.fusion.pathology_encoder import PathologyEncoder

        encoder = PathologyEncoder()

        for B in [1, 2, 8, 16]:
            pathology = torch.randn(B, 3)
            region_context = torch.randn(B, 128)

            output = encoder(pathology, region_context)
            assert output.shape == (B, 64)

    def test_different_pathology_values_give_different_outputs(self):
        """Different pathology inputs should produce different embeddings."""
        from src.models.fusion.pathology_encoder import PathologyEncoder

        encoder = PathologyEncoder()
        encoder.eval()

        region_context = torch.randn(1, 128)

        pathology_low = torch.tensor([[0.1, 0.1, 0.1]])
        pathology_high = torch.tensor([[0.9, 0.9, 0.9]])

        with torch.no_grad():
            out_low = encoder(pathology_low, region_context)
            out_high = encoder(pathology_high, region_context)

        assert not torch.allclose(out_low, out_high)

    def test_different_region_context_gives_different_outputs(self):
        """Different region contexts should produce different embeddings."""
        from src.models.fusion.pathology_encoder import PathologyEncoder

        encoder = PathologyEncoder()
        encoder.eval()

        pathology = torch.randn(1, 3)

        region_single = torch.randn(1, 128)
        region_multi = torch.randn(1, 128)

        with torch.no_grad():
            out_single = encoder(pathology, region_single)
            out_multi = encoder(pathology, region_multi)

        assert not torch.allclose(out_single, out_multi)

    def test_deterministic_output(self):
        """Same inputs should produce same outputs."""
        from src.models.fusion.pathology_encoder import PathologyEncoder

        encoder = PathologyEncoder()
        encoder.eval()

        pathology = torch.randn(2, 3)
        region_context = torch.randn(2, 128)

        with torch.no_grad():
            out1 = encoder(pathology, region_context)
            out2 = encoder(pathology, region_context)

        assert torch.allclose(out1, out2)

    def test_custom_dimensions(self):
        """Should work with custom feature dimensions."""
        from src.models.fusion.pathology_encoder import PathologyEncoder

        encoder = PathologyEncoder(n_pathology_features=5, d_region=256, d_cond=128)

        pathology = torch.randn(4, 5)
        region_context = torch.randn(4, 256)

        output = encoder(pathology, region_context)

        assert output.shape == (4, 128)


class TestGradientFlow:
    """Tests for gradient flow through PathologyEncoder."""

    def test_gradients_flow_to_pathology(self):
        """Gradients should reach pathology input."""
        from src.models.fusion.pathology_encoder import PathologyEncoder

        encoder = PathologyEncoder()

        pathology = torch.randn(2, 3, requires_grad=True)
        region_context = torch.randn(2, 128)

        output = encoder(pathology, region_context)
        loss = output.sum()
        loss.backward()

        assert pathology.grad is not None
        assert not torch.all(pathology.grad == 0)

    def test_gradients_flow_to_region_context(self):
        """Gradients should reach region_context input."""
        from src.models.fusion.pathology_encoder import PathologyEncoder

        encoder = PathologyEncoder()

        pathology = torch.randn(2, 3)
        region_context = torch.randn(2, 128, requires_grad=True)

        output = encoder(pathology, region_context)
        loss = output.sum()
        loss.backward()

        assert region_context.grad is not None
        assert not torch.all(region_context.grad == 0)

    def test_gradients_flow_to_all_parameters(self):
        """Gradients should reach all encoder parameters."""
        from src.models.fusion.pathology_encoder import PathologyEncoder

        encoder = PathologyEncoder()

        pathology = torch.randn(2, 3, requires_grad=True)
        region_context = torch.randn(2, 128, requires_grad=True)

        output = encoder(pathology, region_context)
        loss = output.sum()
        loss.backward()

        for name, param in encoder.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.all(param.grad == 0), f"Zero gradient for {name}"


class TestValidation:
    """Tests for input validation."""

    def test_rejects_invalid_n_pathology_features(self):
        """Should reject non-positive n_pathology_features."""
        from src.models.fusion.pathology_encoder import PathologyEncoder

        with pytest.raises(ValueError, match="n_pathology_features must be positive"):
            PathologyEncoder(n_pathology_features=0)

        with pytest.raises(ValueError, match="n_pathology_features must be positive"):
            PathologyEncoder(n_pathology_features=-1)

    def test_rejects_invalid_d_region(self):
        """Should reject non-positive d_region."""
        from src.models.fusion.pathology_encoder import PathologyEncoder

        with pytest.raises(ValueError, match="d_region must be positive"):
            PathologyEncoder(d_region=0)

        with pytest.raises(ValueError, match="d_region must be positive"):
            PathologyEncoder(d_region=-1)

    def test_rejects_invalid_d_cond(self):
        """Should reject non-positive d_cond."""
        from src.models.fusion.pathology_encoder import PathologyEncoder

        with pytest.raises(ValueError, match="d_cond must be positive"):
            PathologyEncoder(d_cond=0)

        with pytest.raises(ValueError, match="d_cond must be positive"):
            PathologyEncoder(d_cond=-1)

    def test_rejects_wrong_pathology_dim(self):
        """Should reject pathology with wrong number of dimensions."""
        from src.models.fusion.pathology_encoder import PathologyEncoder

        encoder = PathologyEncoder()

        # 1D tensor (missing batch dimension)
        pathology_1d = torch.randn(3)
        region_context = torch.randn(2, 128)

        with pytest.raises(ValueError, match="Expected 2D pathology input"):
            encoder(pathology_1d, region_context)

        # 3D tensor
        pathology_3d = torch.randn(2, 3, 1)
        with pytest.raises(ValueError, match="Expected 2D pathology input"):
            encoder(pathology_3d, region_context)

    def test_rejects_wrong_region_context_dim(self):
        """Should reject region_context with wrong number of dimensions."""
        from src.models.fusion.pathology_encoder import PathologyEncoder

        encoder = PathologyEncoder()

        pathology = torch.randn(2, 3)

        # 1D tensor (missing batch dimension)
        region_1d = torch.randn(128)
        with pytest.raises(ValueError, match="Expected 2D region_context input"):
            encoder(pathology, region_1d)

        # 3D tensor
        region_3d = torch.randn(2, 128, 1)
        with pytest.raises(ValueError, match="Expected 2D region_context input"):
            encoder(pathology, region_3d)

    def test_rejects_wrong_pathology_features(self):
        """Should reject pathology with wrong feature count."""
        from src.models.fusion.pathology_encoder import PathologyEncoder

        encoder = PathologyEncoder(n_pathology_features=3)

        pathology = torch.randn(2, 5)  # Wrong: 5 instead of 3
        region_context = torch.randn(2, 128)

        with pytest.raises(ValueError, match="Expected 3 pathology features"):
            encoder(pathology, region_context)

    def test_rejects_wrong_region_context_size(self):
        """Should reject region_context with wrong dimension size."""
        from src.models.fusion.pathology_encoder import PathologyEncoder

        encoder = PathologyEncoder(d_region=128)

        pathology = torch.randn(2, 3)
        region_context = torch.randn(2, 256)  # Wrong: 256 instead of 128

        with pytest.raises(ValueError, match="Expected region_context dim 128"):
            encoder(pathology, region_context)

    def test_batch_size_mismatch(self):
        """Should raise ValueError for mismatched batch sizes."""
        from src.models.fusion.pathology_encoder import PathologyEncoder

        encoder = PathologyEncoder()

        pathology = torch.randn(4, 3)
        region_context = torch.randn(8, 128)  # Different batch size

        with pytest.raises(ValueError, match="Batch size mismatch"):
            encoder(pathology, region_context)


class TestExtraRepr:
    """Tests for extra_repr method."""

    def test_extra_repr_contains_parameters(self):
        """extra_repr should show key parameters."""
        from src.models.fusion.pathology_encoder import PathologyEncoder

        encoder = PathologyEncoder(n_pathology_features=5, d_region=256, d_cond=128)

        repr_str = encoder.extra_repr()

        assert "n_pathology_features=5" in repr_str
        assert "d_region=256" in repr_str
        assert "d_cond=128" in repr_str

    def test_str_contains_extra_repr(self):
        """String representation should include extra_repr info."""
        from src.models.fusion.pathology_encoder import PathologyEncoder

        encoder = PathologyEncoder(n_pathology_features=3, d_region=128, d_cond=64)

        str_repr = str(encoder)

        assert "n_pathology_features=3" in str_repr
        assert "d_region=128" in str_repr
        assert "d_cond=64" in str_repr
