"""
Tests for src/models/fusion/fusion_layer.py

Test organization:
1. Initialization - parameter shapes, validation
2. Forward pass - output shapes, correctness
3. Gradient flow - all inputs receive gradients
"""

import pytest
import torch

from src.data.constants import N_CELL_TYPES


class TestInitialization:
    """Tests for FusionLayer initialization."""

    def test_creates_projection_layer(self):
        """Should create projection from 3*d_embed to d_fused."""
        from src.models.fusion.fusion_layer import FusionLayer

        layer = FusionLayer(d_embed=64, d_fused=128)

        assert layer.proj.in_features == 3 * 64
        assert layer.proj.out_features == 128

    def test_creates_layer_norm(self):
        """Should create LayerNorm with d_fused dimensions."""
        from src.models.fusion.fusion_layer import FusionLayer

        layer = FusionLayer(d_embed=64, d_fused=128)

        assert layer.layer_norm.normalized_shape == (128,)

    def test_stores_n_cell_types(self):
        """Should store n_cell_types attribute."""
        from src.models.fusion.fusion_layer import FusionLayer

        layer = FusionLayer(d_embed=64, d_fused=128, n_cell_types=N_CELL_TYPES)

        assert layer.n_cell_types == N_CELL_TYPES


class TestForwardPass:
    """Tests for FusionLayer forward pass."""

    def test_output_shape(self):
        """Forward should return [B, N_CELL_TYPES, d_fused]."""
        from src.models.fusion.fusion_layer import FusionLayer

        layer = FusionLayer(d_embed=64, d_fused=128, n_cell_types=N_CELL_TYPES)

        pseudobulk = torch.randn(4, N_CELL_TYPES, 64)
        hgt = torch.randn(4, N_CELL_TYPES, 64)
        cell = torch.randn(4, N_CELL_TYPES, 64)

        output = layer(pseudobulk, hgt, cell)

        assert output.shape == (4, N_CELL_TYPES, 128)

    def test_different_batch_sizes(self):
        """Should work with various batch sizes."""
        from src.models.fusion.fusion_layer import FusionLayer

        layer = FusionLayer(d_embed=32, d_fused=64)

        for B in [1, 2, 8, 16]:
            pb = torch.randn(B, N_CELL_TYPES, 32)
            hgt = torch.randn(B, N_CELL_TYPES, 32)
            cell = torch.randn(B, N_CELL_TYPES, 32)

            output = layer(pb, hgt, cell)
            assert output.shape == (B, N_CELL_TYPES, 64)

    def test_output_is_normalized(self):
        """Output should have approximately zero mean and unit variance per feature."""
        from src.models.fusion.fusion_layer import FusionLayer

        layer = FusionLayer(d_embed=64, d_fused=128)
        layer.eval()  # Disable dropout so LayerNorm statistics are preserved

        pb = torch.randn(32, N_CELL_TYPES, 64)
        hgt = torch.randn(32, N_CELL_TYPES, 64)
        cell = torch.randn(32, N_CELL_TYPES, 64)

        output = layer(pb, hgt, cell)

        # LayerNorm normalizes over last dimension
        mean = output.mean(dim=-1)
        var = output.var(dim=-1, unbiased=False)

        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
        assert torch.allclose(var, torch.ones_like(var), atol=1e-1)


class TestGradientFlow:
    """Tests for gradient flow through FusionLayer."""

    def test_gradients_flow_to_all_inputs(self):
        """Gradients should reach all three input tensors."""
        from src.models.fusion.fusion_layer import FusionLayer

        layer = FusionLayer(d_embed=32, d_fused=64)

        pb = torch.randn(2, N_CELL_TYPES, 32, requires_grad=True)
        hgt = torch.randn(2, N_CELL_TYPES, 32, requires_grad=True)
        cell = torch.randn(2, N_CELL_TYPES, 32, requires_grad=True)

        output = layer(pb, hgt, cell)
        loss = output.sum()
        loss.backward()

        assert pb.grad is not None
        assert hgt.grad is not None
        assert cell.grad is not None
        assert not torch.all(pb.grad == 0)
        assert not torch.all(hgt.grad == 0)
        assert not torch.all(cell.grad == 0)

    def test_gradients_flow_to_parameters(self):
        """Gradients should reach layer parameters."""
        from src.models.fusion.fusion_layer import FusionLayer

        layer = FusionLayer(d_embed=32, d_fused=64)

        pb = torch.randn(2, N_CELL_TYPES, 32)
        hgt = torch.randn(2, N_CELL_TYPES, 32)
        cell = torch.randn(2, N_CELL_TYPES, 32)

        output = layer(pb, hgt, cell)
        loss = output.sum()
        loss.backward()

        assert layer.proj.weight.grad is not None
        assert layer.proj.bias.grad is not None

    def test_mismatched_d_embed_raises_error(self):
        """Inputs with wrong feature dimension should raise RuntimeError."""
        from src.models.fusion.fusion_layer import FusionLayer

        layer = FusionLayer(d_embed=64, d_fused=128, n_cell_types=N_CELL_TYPES)

        wrong_pb = torch.randn(2, N_CELL_TYPES, 32)  # d_embed=32 != 64
        correct = torch.randn(2, N_CELL_TYPES, 64)

        with pytest.raises(ValueError):
            layer(wrong_pb, correct, correct)

    def test_all_three_branches_contribute_to_output(self):
        """Changing any single branch input should change the output."""
        from src.models.fusion.fusion_layer import FusionLayer

        layer = FusionLayer(d_embed=64, d_fused=128, n_cell_types=N_CELL_TYPES)

        pb = torch.randn(2, N_CELL_TYPES, 64)
        hgt = torch.randn(2, N_CELL_TYPES, 64)
        cell = torch.randn(2, N_CELL_TYPES, 64)

        layer.eval()
        base = layer(pb, hgt, cell)

        for i, modified in enumerate([
            (torch.randn_like(pb), hgt, cell),
            (pb, torch.randn_like(hgt), cell),
            (pb, hgt, torch.randn_like(cell)),
        ]):
            alt = layer(*modified)
            assert not torch.allclose(base, alt, atol=1e-6), f"Branch {i} doesn't affect output"


class TestValidation:
    """Tests for input validation."""

    def test_invalid_d_embed(self):
        """Should raise ValueError for d_embed <= 0."""
        from src.models.fusion.fusion_layer import FusionLayer

        with pytest.raises(ValueError, match="d_embed must be positive"):
            FusionLayer(d_embed=0, d_fused=64)

        with pytest.raises(ValueError, match="d_embed must be positive"):
            FusionLayer(d_embed=-1, d_fused=64)

    def test_invalid_d_fused(self):
        """Should raise ValueError for d_fused <= 0."""
        from src.models.fusion.fusion_layer import FusionLayer

        with pytest.raises(ValueError, match="d_fused must be positive"):
            FusionLayer(d_embed=64, d_fused=0)

        with pytest.raises(ValueError, match="d_fused must be positive"):
            FusionLayer(d_embed=64, d_fused=-1)

    def test_invalid_n_cell_types(self):
        """Should raise ValueError for n_cell_types <= 0."""
        from src.models.fusion.fusion_layer import FusionLayer

        with pytest.raises(ValueError, match="n_cell_types must be positive"):
            FusionLayer(d_embed=64, d_fused=128, n_cell_types=0)

        with pytest.raises(ValueError, match="n_cell_types must be positive"):
            FusionLayer(d_embed=64, d_fused=128, n_cell_types=-1)

    def test_invalid_input_dimension_pseudobulk(self):
        """Should raise ValueError for non-3D pseudobulk_emb input."""
        from src.models.fusion.fusion_layer import FusionLayer

        layer = FusionLayer(d_embed=32, d_fused=64)

        # 2D input for pseudobulk
        with pytest.raises(ValueError, match="Expected 3D input for pseudobulk_emb"):
            layer(torch.randn(N_CELL_TYPES, 32), torch.randn(2, N_CELL_TYPES, 32), torch.randn(2, N_CELL_TYPES, 32))

        # 4D input for pseudobulk
        with pytest.raises(ValueError, match="Expected 3D input for pseudobulk_emb"):
            layer(torch.randn(2, 1, N_CELL_TYPES, 32), torch.randn(2, N_CELL_TYPES, 32), torch.randn(2, N_CELL_TYPES, 32))

    def test_invalid_input_dimension_hgt(self):
        """Should raise ValueError for non-3D hgt_emb input."""
        from src.models.fusion.fusion_layer import FusionLayer

        layer = FusionLayer(d_embed=32, d_fused=64)

        # 2D input for hgt
        with pytest.raises(ValueError, match="Expected 3D input for hgt_emb"):
            layer(torch.randn(2, N_CELL_TYPES, 32), torch.randn(N_CELL_TYPES, 32), torch.randn(2, N_CELL_TYPES, 32))

    def test_invalid_input_dimension_cell(self):
        """Should raise ValueError for non-3D cell_emb input."""
        from src.models.fusion.fusion_layer import FusionLayer

        layer = FusionLayer(d_embed=32, d_fused=64)

        # 2D input for cell
        with pytest.raises(ValueError, match="Expected 3D input for cell_emb"):
            layer(torch.randn(2, N_CELL_TYPES, 32), torch.randn(2, N_CELL_TYPES, 32), torch.randn(N_CELL_TYPES, 32))

    def test_invalid_cell_type_count(self):
        """Should raise ValueError when n_cell_types doesn't match input."""
        from src.models.fusion.fusion_layer import FusionLayer

        layer = FusionLayer(d_embed=32, d_fused=64, n_cell_types=N_CELL_TYPES)

        # Wrong number of cell types
        with pytest.raises(ValueError, match=f"Expected {N_CELL_TYPES} cell types"):
            layer(torch.randn(2, 20, 32), torch.randn(2, 20, 32), torch.randn(2, 20, 32))


class TestExtraRepr:
    """Tests for extra_repr method."""

    def test_extra_repr_output(self):
        """Should return informative string representation."""
        from src.models.fusion.fusion_layer import FusionLayer

        layer = FusionLayer(d_embed=64, d_fused=128, n_cell_types=N_CELL_TYPES)

        repr_str = layer.extra_repr()

        assert "d_embed=64" in repr_str
        assert "d_fused=128" in repr_str
        assert f"n_cell_types={N_CELL_TYPES}" in repr_str

    def test_stored_attributes(self):
        """Should store d_embed and d_fused as attributes for debugging."""
        from src.models.fusion.fusion_layer import FusionLayer

        layer = FusionLayer(d_embed=64, d_fused=128, n_cell_types=N_CELL_TYPES)

        assert layer.d_embed == 64
        assert layer.d_fused == 128
        assert layer.n_cell_types == N_CELL_TYPES
