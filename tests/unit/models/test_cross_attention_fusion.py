"""
Tests for src/models/fusion/cross_attention_fusion.py

Test organization:
1. Initialization - parameter shapes, validation
2. Forward pass - output shapes, drop-in compatibility with FusionLayer
3. Scale invariance - scaling one branch input shouldn't proportionally change output
4. B2 branch weights - shape, normalization, interpretability methods
5. Attention maps - shape and extraction
6. Gradient flow - all inputs receive gradients
7. n_pma_seeds > 1 handling
"""

import pytest
import torch

from src.data.constants import N_CELL_TYPES
from src.models.fusion.cross_attention_fusion import (
    CrossAttentionFusionLayer,
    PairwiseCrossAttention,
)
from src.models.fusion.normalized_concat_fusion import NormalizedConcatFusionLayer


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def layer():
    return CrossAttentionFusionLayer(
        d_embed=64, d_fused=64, n_cell_types=N_CELL_TYPES, n_heads=4, dropout=0.0,
    )


@pytest.fixture
def branch_inputs():
    B = 4
    return (
        torch.randn(B, N_CELL_TYPES, 64),  # hgt_emb
        torch.randn(B, N_CELL_TYPES, 64),  # cell_emb
    )


# ── 1. Initialization ───────────────────────────────────────────────────────

class TestInitialization:

    def test_creates_two_cross_attention_ops(self, layer):
        assert hasattr(layer, "hgt_from_ct")
        assert hasattr(layer, "ct_from_hgt")

    def test_no_pseudobulk_ops(self, layer):
        assert not hasattr(layer, "pb_from_hgt")
        assert not hasattr(layer, "pb_from_ct")
        assert not hasattr(layer, "hgt_from_pb")
        assert not hasattr(layer, "ct_from_pb")

    def test_creates_branch_weight_logits(self, layer):
        assert layer.branch_weight_logits.shape == (2, N_CELL_TYPES)

    def test_creates_enrichment_norms(self, layer):
        assert layer.hgt_enrich_norm.normalized_shape == (64,)
        assert layer.ct_enrich_norm.normalized_shape == (64,)

    def test_no_pb_enrich_norm(self, layer):
        assert not hasattr(layer, "pb_enrich_norm")

    def test_no_output_proj_when_dims_match(self, layer):
        assert layer.output_proj is None

    def test_output_proj_when_dims_differ(self):
        layer = CrossAttentionFusionLayer(d_embed=64, d_fused=32)
        assert layer.output_proj is not None
        assert layer.output_proj.in_features == 64
        assert layer.output_proj.out_features == 32

    def test_invalid_d_embed_raises(self):
        with pytest.raises(ValueError, match="d_embed must be positive"):
            CrossAttentionFusionLayer(d_embed=0, d_fused=64)

    def test_invalid_n_heads_raises(self):
        with pytest.raises(ValueError, match="divisible by n_heads"):
            PairwiseCrossAttention(d_model=64, n_heads=5)


# ── 2. Forward Pass ─────────────────────────────────────────────────────────

class TestForwardPass:

    def test_output_shape(self, layer, branch_inputs):
        hgt, ct = branch_inputs
        out = layer(hgt, ct)
        assert out.shape == (4, N_CELL_TYPES, 64)

    def test_output_shape_different_d_fused(self, branch_inputs):
        layer = CrossAttentionFusionLayer(d_embed=64, d_fused=32, dropout=0.0)
        hgt, ct = branch_inputs
        out = layer(hgt, ct)
        assert out.shape == (4, N_CELL_TYPES, 32)

    def test_drop_in_compatible_with_fusion_layer(self, branch_inputs):
        """Output shape matches FusionLayer for same config."""
        from src.models.fusion.fusion_layer import FusionLayer

        concat_layer = FusionLayer(d_embed=64, d_fused=64, dropout=0.0)
        xattn_layer = CrossAttentionFusionLayer(d_embed=64, d_fused=64, dropout=0.0)

        hgt, ct = branch_inputs
        out_concat = concat_layer(hgt, ct)
        out_xattn = xattn_layer(hgt, ct)

        assert out_concat.shape == out_xattn.shape

    def test_batch_size_one(self, layer):
        hgt = torch.randn(1, N_CELL_TYPES, 64)
        ct = torch.randn(1, N_CELL_TYPES, 64)
        out = layer(hgt, ct)
        assert out.shape == (1, N_CELL_TYPES, 64)

    def test_invalid_input_dim_raises(self, layer):
        with pytest.raises(ValueError, match="3D"):
            layer(torch.randn(4, 64), torch.randn(4, 31, 64))


# ── 3. Scale Invariance ─────────────────────────────────────────────────────

class TestScaleInvariance:

    def test_scaling_branch_does_not_proportionally_scale_output(self, layer, branch_inputs):
        """Cross-attention is scale-invariant: scaling input by 10x should NOT
        scale output by 10x (unlike concat+linear which would)."""
        layer.eval()
        hgt, ct = branch_inputs

        with torch.no_grad():
            out_normal = layer(hgt, ct)
            out_scaled = layer(hgt * 10.0, ct)

        # If output scaled proportionally, ratio would be ~10.
        # With attention (scale-invariant), ratio should be much less.
        ratio = out_scaled.norm() / out_normal.norm()
        assert ratio < 5.0, f"Output scaled by {ratio:.1f}x — not scale-invariant"


# ── 4. B2 Branch Weights ────────────────────────────────────────────────────

class TestBranchWeights:

    def test_get_branch_weights_shape(self, layer):
        w = layer.get_branch_weights()
        assert w.shape == (2, N_CELL_TYPES)

    def test_branch_weights_sum_to_one(self, layer):
        w = layer.get_branch_weights()
        sums = w.sum(dim=0)
        assert torch.allclose(sums, torch.ones(N_CELL_TYPES), atol=1e-5)

    def test_branch_weights_initialized_uniform(self, layer):
        w = layer.get_branch_weights()
        expected = torch.ones(2, N_CELL_TYPES) / 2.0
        assert torch.allclose(w, expected, atol=1e-5)

    def test_branch_weight_dict_keys(self, layer):
        d = layer.get_branch_weight_dict()
        assert set(d.keys()) == {"hgt", "cell_transformer"}

    def test_branch_weight_dict_values_shape(self, layer):
        d = layer.get_branch_weight_dict()
        for v in d.values():
            assert v.shape == (N_CELL_TYPES,)


# ── 5. Attention Maps ───────────────────────────────────────────────────────

class TestAttentionMaps:

    def test_return_attention_provides_maps(self, layer, branch_inputs):
        hgt, ct = branch_inputs
        out, attn_info = layer(hgt, ct, return_attention=True)
        assert isinstance(attn_info, dict)

    def test_two_attention_maps_returned(self, layer, branch_inputs):
        hgt, ct = branch_inputs
        _, attn_info = layer(hgt, ct, return_attention=True)
        expected_keys = {
            "hgt_from_ct", "ct_from_hgt",
            "branch_weights",
        }
        assert set(attn_info.keys()) == expected_keys

    def test_attention_map_shape(self, layer, branch_inputs):
        hgt, ct = branch_inputs
        _, attn_info = layer(hgt, ct, return_attention=True)
        for key in ["hgt_from_ct", "ct_from_hgt"]:
            assert attn_info[key].shape == (4, 4, N_CELL_TYPES, N_CELL_TYPES)

    def test_no_attention_by_default(self, layer, branch_inputs):
        hgt, ct = branch_inputs
        result = layer(hgt, ct)
        assert isinstance(result, torch.Tensor)  # Not a tuple


# ── 6. Gradient Flow ────────────────────────────────────────────────────────

class TestGradientFlow:

    def test_gradients_flow_to_all_branches(self, layer):
        hgt = torch.randn(2, N_CELL_TYPES, 64, requires_grad=True)
        ct = torch.randn(2, N_CELL_TYPES, 64, requires_grad=True)

        out = layer(hgt, ct)
        out.sum().backward()

        assert hgt.grad is not None and hgt.grad.norm() > 0
        assert ct.grad is not None and ct.grad.norm() > 0

    def test_branch_weight_logits_receive_gradient(self, layer):
        hgt = torch.randn(2, N_CELL_TYPES, 64)
        ct = torch.randn(2, N_CELL_TYPES, 64)

        out = layer(hgt, ct)
        out.sum().backward()

        assert layer.branch_weight_logits.grad is not None
        assert layer.branch_weight_logits.grad.norm() > 0


# ── 7. n_pma_seeds > 1 ──────────────────────────────────────────────────────

class TestPmaSeeds:

    def test_pma_seeds_2_creates_projection(self):
        layer = CrossAttentionFusionLayer(
            d_embed=64, d_fused=64, n_pma_seeds=2, dropout=0.0,
        )
        assert layer.cell_input_proj is not None
        assert layer.cell_input_proj.in_features == 128
        assert layer.cell_input_proj.out_features == 64

    def test_pma_seeds_1_no_projection(self, layer):
        assert layer.cell_input_proj is None

    def test_pma_seeds_2_forward_works(self):
        layer = CrossAttentionFusionLayer(
            d_embed=64, d_fused=64, n_pma_seeds=2, dropout=0.0,
        )
        hgt = torch.randn(2, N_CELL_TYPES, 64)
        ct = torch.randn(2, N_CELL_TYPES, 128)  # n_pma_seeds * d_embed

        out = layer(hgt, ct)
        assert out.shape == (2, N_CELL_TYPES, 64)


# ── 8. Attention Modes (CrossFuse, Blend) ────────────────────────────────────

class TestAttentionModes:

    def test_reverse_mode_forward(self, branch_inputs):
        layer = CrossAttentionFusionLayer(
            d_embed=64, d_fused=64, n_heads=4, dropout=0.0, attention_mode="reverse",
        )
        hgt, ct = branch_inputs
        out = layer(hgt, ct)
        assert out.shape == (4, N_CELL_TYPES, 64)

    def test_blend_mode_forward(self, branch_inputs):
        layer = CrossAttentionFusionLayer(
            d_embed=64, d_fused=64, n_heads=4, dropout=0.0, attention_mode="blend",
        )
        hgt, ct = branch_inputs
        out = layer(hgt, ct)
        assert out.shape == (4, N_CELL_TYPES, 64)

    def test_blend_mode_has_blend_logits(self):
        layer = CrossAttentionFusionLayer(
            d_embed=64, d_fused=64, n_heads=4, dropout=0.0, attention_mode="blend",
        )
        # 2 cross-attention ops × 4 heads = 8 blend_logits total
        blend_params = [p for n, p in layer.named_parameters() if "blend_logits" in n]
        assert len(blend_params) == 2
        assert all(p.shape == (4,) for p in blend_params)

    def test_reverse_mode_attention_maps(self, branch_inputs):
        layer = CrossAttentionFusionLayer(
            d_embed=64, d_fused=64, n_heads=4, dropout=0.0, attention_mode="reverse",
        )
        hgt, ct = branch_inputs
        _, attn_info = layer(hgt, ct, return_attention=True)
        assert "hgt_from_ct" in attn_info
        assert attn_info["hgt_from_ct"].shape == (4, 4, N_CELL_TYPES, N_CELL_TYPES)

    def test_blend_gradient_to_blend_logits(self):
        layer = CrossAttentionFusionLayer(
            d_embed=64, d_fused=64, n_heads=4, dropout=0.0, attention_mode="blend",
        )
        hgt = torch.randn(2, N_CELL_TYPES, 64)
        ct = torch.randn(2, N_CELL_TYPES, 64)
        out = layer(hgt, ct)
        out.sum().backward()
        for name, p in layer.named_parameters():
            if "blend_logits" in name:
                assert p.grad is not None and p.grad.norm() > 0, f"{name} has no gradient"

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be"):
            PairwiseCrossAttention(d_model=64, n_heads=4, mode="invalid")


# ── 9. NormalizedConcatFusionLayer ───────────────────────────────────────────

class TestNormalizedConcat:

    def test_output_shape(self, branch_inputs):
        layer = NormalizedConcatFusionLayer(d_embed=64, d_fused=64, dropout=0.0)
        hgt, ct = branch_inputs
        out = layer(hgt, ct)
        assert out.shape == (4, N_CELL_TYPES, 64)

    def test_drop_in_compatible(self, branch_inputs):
        from src.models.fusion.fusion_layer import FusionLayer
        concat = FusionLayer(d_embed=64, d_fused=64, dropout=0.0)
        normed = NormalizedConcatFusionLayer(d_embed=64, d_fused=64, dropout=0.0)
        hgt, ct = branch_inputs
        assert concat(hgt, ct).shape == normed(hgt, ct).shape

    def test_reduces_magnitude_imbalance(self):
        """Normalized concat should equalize branch contributions."""
        layer = NormalizedConcatFusionLayer(d_embed=64, d_fused=64, dropout=0.0)
        layer.eval()
        # Simulate magnitude imbalance: HGT 10x larger than CT
        hgt = torch.randn(2, N_CELL_TYPES, 64) * 10.0
        ct = torch.randn(2, N_CELL_TYPES, 64) * 0.1
        with torch.no_grad():
            out = layer(hgt, ct)
        # Output should be finite and reasonably scaled
        assert torch.isfinite(out).all()
        assert out.abs().mean() < 10.0

    def test_pma_seeds_2(self):
        layer = NormalizedConcatFusionLayer(
            d_embed=64, d_fused=64, n_pma_seeds=2, dropout=0.0,
        )
        hgt = torch.randn(2, N_CELL_TYPES, 64)
        ct = torch.randn(2, N_CELL_TYPES, 128)
        out = layer(hgt, ct)
        assert out.shape == (2, N_CELL_TYPES, 64)
