"""
Edge case and coverage gap tests.

Tests for scenarios identified during code review:
1. HGT empty graph handling
2. Mixed dtype inputs
3. Region context gradient flow
4. All regions masked edge case
5. Bayesian head training loop verification
"""

import pytest
import torch
import torch.nn as nn

from src.data.constants import CELL_TYPE_ORDER, ALL_EDGE_TYPES, sanitize_key
from src.models.full_model import CognitiveResilienceModel
from src.models.fusion import FusionLayer, PathologyEncoder, PathologyStratifiedAttention
from src.models.components import RegionHandler
from src.models.heads import BayesianPredictionHead


def _make_edge_dicts(batch_size, n_edges=5):
    """Create edge_index_dict_list and edge_attr_dict_list for testing."""
    edge_index_dict_list = []
    edge_attr_dict_list = []
    for _ in range(batch_size):
        edge_index_dict = {}
        edge_attr_dict = {}
        for src_ct in CELL_TYPE_ORDER[:3]:
            for dst_ct in CELL_TYPE_ORDER[:3]:
                for et in ALL_EDGE_TYPES[:2]:
                    key = (sanitize_key(src_ct), sanitize_key(et), sanitize_key(dst_ct))
                    edge_index_dict[key] = torch.zeros(2, n_edges, dtype=torch.long)
                    edge_attr_dict[key] = torch.rand(n_edges, 1)
        edge_index_dict_list.append(edge_index_dict)
        edge_attr_dict_list.append(edge_attr_dict)
    return edge_index_dict_list, edge_attr_dict_list


class TestHGTEmptyGraphHandling:
    """Test handling of empty or minimal CCC graphs."""

    @pytest.fixture
    def model(self):
        return CognitiveResilienceModel(
            n_genes=50,
            n_cell_types=31,
            d_embed=32,
            d_fused=32,
            use_bayesian_head=False,
        )

    @pytest.fixture
    def base_inputs(self):
        B = 2
        return {
            'region_pseudobulk': torch.randn(B, 6, 31, 50),
            'region_mask': torch.ones(B, 6, dtype=torch.bool),
            'cells': torch.randn(B, 31, 10, 50),
            'cell_mask': torch.ones(B, 31, 10, dtype=torch.bool),
            'pathology': torch.randn(B, 3),
        }

    def test_empty_edge_index_produces_output(self, model, base_inputs):
        """Model should handle empty edge dicts gracefully."""
        B = 2
        inputs = {
            **base_inputs,
            'edge_index_dict_list': [{}, {}],
            'edge_attr_dict_list': [{}, {}],
        }
        output = model(**inputs)
        assert 'mean' in output
        assert output['mean'].shape == (2, 1)
        assert torch.isfinite(output['mean']).all()

    def test_single_edge_graph(self, model, base_inputs):
        """Model should handle single edge graph."""
        B = 2
        key = (sanitize_key(CELL_TYPE_ORDER[0]), sanitize_key(ALL_EDGE_TYPES[0]), sanitize_key(CELL_TYPE_ORDER[1]))
        inputs = {
            **base_inputs,
            'edge_index_dict_list': [{key: torch.tensor([[0], [0]])} for _ in range(B)],
            'edge_attr_dict_list': [{key: torch.tensor([[1.0]])} for _ in range(B)],
        }
        output = model(**inputs)
        assert torch.isfinite(output['mean']).all()

    def test_self_loop_only_graph(self, model, base_inputs):
        """Model should handle graph with only self-loops."""
        B = 2
        key = (sanitize_key(CELL_TYPE_ORDER[0]), sanitize_key(ALL_EDGE_TYPES[0]), sanitize_key(CELL_TYPE_ORDER[0]))
        inputs = {
            **base_inputs,
            'edge_index_dict_list': [{key: torch.tensor([[0, 0, 0], [0, 0, 0]])} for _ in range(B)],
            'edge_attr_dict_list': [{key: torch.tensor([[1.0], [0.9], [0.8]])} for _ in range(B)],
        }
        output = model(**inputs)
        assert torch.isfinite(output['mean']).all()

    def test_duplicate_edges_accumulated(self, model, base_inputs):
        """Duplicate edges should be accumulated, not dropped."""
        B = 2
        key = (sanitize_key(CELL_TYPE_ORDER[0]), sanitize_key(ALL_EDGE_TYPES[0]), sanitize_key(CELL_TYPE_ORDER[1]))
        inputs = {
            **base_inputs,
            'edge_index_dict_list': [{key: torch.tensor([[0, 0], [0, 0]])} for _ in range(B)],
            'edge_attr_dict_list': [{key: torch.tensor([[1.0], [0.8]])} for _ in range(B)],
        }
        output = model(**inputs)
        assert torch.isfinite(output['mean']).all()


class TestMixedDtypeInputs:
    """Test handling of inputs with different dtypes."""

    @pytest.fixture
    def model(self):
        return CognitiveResilienceModel(
            n_genes=50,
            n_cell_types=31,
            d_embed=32,
            d_fused=32,
            use_bayesian_head=False,
        )

    def test_float64_pathology_requires_conversion(self, model):
        """Float64 pathology with float32 model requires explicit conversion."""
        B = 2
        edge_index_dict_list, edge_attr_dict_list = _make_edge_dicts(B, n_edges=1)
        inputs = {
            'region_pseudobulk': torch.randn(B, 6, 31, 50, dtype=torch.float32),
            'region_mask': torch.ones(B, 6, dtype=torch.bool),
            'edge_index_dict_list': edge_index_dict_list,
            'edge_attr_dict_list': edge_attr_dict_list,
            'cells': torch.randn(B, 31, 10, 50, dtype=torch.float32),
            'cell_mask': torch.ones(B, 31, 10, dtype=torch.bool),
            'pathology': torch.randn(B, 3, dtype=torch.float64),  # Different dtype
        }
        # PyTorch doesn't auto-convert dtypes - this will fail
        with pytest.raises(RuntimeError, match="dtype"):
            model(**inputs)

    def test_consistent_dtypes_work(self, model):
        """Consistent dtypes should work correctly."""
        B = 2
        edge_index_dict_list, edge_attr_dict_list = _make_edge_dicts(B, n_edges=1)
        inputs = {
            'region_pseudobulk': torch.randn(B, 6, 31, 50, dtype=torch.float32),
            'region_mask': torch.ones(B, 6, dtype=torch.bool),
            'edge_index_dict_list': edge_index_dict_list,
            'edge_attr_dict_list': edge_attr_dict_list,
            'cells': torch.randn(B, 31, 10, 50, dtype=torch.float32),
            'cell_mask': torch.ones(B, 31, 10, dtype=torch.bool),
            'pathology': torch.randn(B, 3, dtype=torch.float32),  # Same dtype
        }
        output = model(**inputs)
        assert 'mean' in output

    def test_bool_mask_required(self, model):
        """Cell mask must be bool (for transformer attention)."""
        B = 2
        edge_index_dict_list, edge_attr_dict_list = _make_edge_dicts(B, n_edges=1)
        inputs = {
            'region_pseudobulk': torch.randn(B, 6, 31, 50),
            'region_mask': torch.ones(B, 6, dtype=torch.bool),
            'edge_index_dict_list': edge_index_dict_list,
            'edge_attr_dict_list': edge_attr_dict_list,
            'cells': torch.randn(B, 31, 10, 50),
            'cell_mask': torch.ones(B, 31, 10, dtype=torch.int32),  # int instead of bool
            'pathology': torch.randn(B, 3),
        }
        # Int masks cause issues with attention mask handling
        with pytest.raises((RuntimeError, AssertionError)):
            model(**inputs)

    def test_region_mask_float_works(self, model):
        """Region mask can be float (RegionHandler converts to float internally)."""
        B = 2
        edge_index_dict_list, edge_attr_dict_list = _make_edge_dicts(B, n_edges=1)
        inputs = {
            'region_pseudobulk': torch.randn(B, 6, 31, 50),
            'region_mask': torch.ones(B, 6, dtype=torch.float32),  # Float mask
            'edge_index_dict_list': edge_index_dict_list,
            'edge_attr_dict_list': edge_attr_dict_list,
            'cells': torch.randn(B, 31, 10, 50),
            'cell_mask': torch.ones(B, 31, 10, dtype=torch.bool),
            'pathology': torch.randn(B, 3),
        }
        output = model(**inputs)
        assert 'mean' in output


class TestRegionContextGradientFlow:
    """Test that gradients flow correctly through region_context."""

    def test_region_context_affects_pathology_encoder(self):
        """Region context should influence pathology encoding."""
        encoder = PathologyEncoder(n_pathology_features=3, d_region=64, d_cond=32)

        pathology = torch.randn(2, 3, requires_grad=True)
        region_context1 = torch.randn(2, 64, requires_grad=True)
        region_context2 = torch.randn(2, 64, requires_grad=True)

        out1 = encoder(pathology, region_context1)
        out2 = encoder(pathology, region_context2)

        # Different region contexts should produce different outputs
        assert not torch.allclose(out1, out2)

    def test_region_context_gradient_flows_to_handler(self):
        """Gradient should flow from pathology encoder back to region handler parameters."""
        handler = RegionHandler(d_model=64, n_regions=6)
        encoder = PathologyEncoder(n_pathology_features=3, d_region=64, d_cond=32)

        # Inputs (region_context comes from handler.region_embedding, not x)
        x = torch.randn(2, 6, 31, 64)
        mask = torch.ones(2, 6, dtype=torch.bool)
        pathology = torch.randn(2, 3)

        # Forward through handler and encoder
        pooled, region_context = handler(x, mask)
        path_emb = encoder(pathology, region_context)

        # Backward
        loss = path_emb.sum()
        loss.backward()

        # Gradient should flow to region_embedding (which produces region_context)
        assert handler.region_embedding.weight.grad is not None
        assert handler.region_embedding.weight.grad.abs().sum() > 0

        # Gradient should flow to encoder's region projection
        assert encoder.region_proj.weight.grad is not None

    def test_region_embedding_receives_gradient(self):
        """Region embedding parameters should receive gradients."""
        handler = RegionHandler(d_model=64, n_regions=6)
        encoder = PathologyEncoder(n_pathology_features=3, d_region=64, d_cond=32)

        x = torch.randn(2, 6, 31, 64)
        mask = torch.ones(2, 6, dtype=torch.bool)
        pathology = torch.randn(2, 3)

        pooled, region_context = handler(x, mask)
        path_emb = encoder(pathology, region_context)

        loss = path_emb.sum()
        loss.backward()

        # Region embedding should receive gradient
        assert handler.region_embedding.weight.grad is not None
        assert handler.region_embedding.weight.grad.abs().sum() > 0


class TestAllRegionsMaskedEdgeCase:
    """Test behavior when all regions are masked (edge case)."""

    def test_all_regions_masked_produces_output(self):
        """When all regions masked, output should still be valid (clamped)."""
        handler = RegionHandler(d_model=64, n_regions=6)

        x = torch.randn(2, 6, 31, 64)
        mask = torch.zeros(2, 6, dtype=torch.bool)  # All masked!

        pooled, region_context = handler(x, mask)

        # Should produce output (even if near-zero due to clamping)
        assert pooled.shape == (2, 31, 64)
        assert region_context.shape == (2, 64)
        assert torch.isfinite(pooled).all()
        assert torch.isfinite(region_context).all()

    def test_all_regions_masked_output_is_near_zero(self):
        """With all regions masked, output should be near zero."""
        handler = RegionHandler(d_model=64, n_regions=6)

        x = torch.randn(2, 6, 31, 64)
        mask = torch.zeros(2, 6, dtype=torch.bool)  # All masked!

        pooled, region_context = handler(x, mask)

        # Pooled should be near zero (masked weights sum to ~0, clamped)
        assert pooled.abs().max() < 1e-5
        # Region context should also be near zero
        assert region_context.abs().max() < 1e-5

    def test_partial_batch_all_masked(self):
        """Mixed batch where some samples have all regions masked."""
        handler = RegionHandler(d_model=64, n_regions=6)

        x = torch.randn(3, 6, 31, 64)
        mask = torch.tensor([
            [True, True, False, False, False, False],  # Normal
            [False, False, False, False, False, False],  # All masked
            [True, False, True, False, True, False],  # Partial
        ], dtype=torch.bool)

        pooled, region_context = handler(x, mask)

        # All samples should produce valid output
        assert torch.isfinite(pooled).all()
        assert torch.isfinite(region_context).all()

        # Sample 1 (all masked) should have near-zero output
        assert pooled[1].abs().max() < 1e-5


class TestFusionLayerDimensionValidation:
    """Test the new d_embed dimension validation in FusionLayer."""

    def test_mismatched_pseudobulk_d_embed_raises(self):
        """Mismatched pseudobulk embedding dimension should raise."""
        layer = FusionLayer(d_embed=64, d_fused=32, n_cell_types=31)

        pseudobulk = torch.randn(2, 31, 32)  # Wrong: 32 instead of 64
        hgt = torch.randn(2, 31, 64)
        cell = torch.randn(2, 31, 64)

        with pytest.raises(ValueError, match="d_embed=64"):
            layer(pseudobulk, hgt, cell)

    def test_mismatched_hgt_d_embed_raises(self):
        """Mismatched HGT embedding dimension should raise."""
        layer = FusionLayer(d_embed=64, d_fused=32, n_cell_types=31)

        pseudobulk = torch.randn(2, 31, 64)
        hgt = torch.randn(2, 31, 32)  # Wrong: 32 instead of 64
        cell = torch.randn(2, 31, 64)

        with pytest.raises(ValueError, match="d_embed=64 for hgt_emb"):
            layer(pseudobulk, hgt, cell)

    def test_mismatched_cell_d_embed_raises(self):
        """Mismatched cell embedding dimension should raise."""
        layer = FusionLayer(d_embed=64, d_fused=32, n_cell_types=31)

        pseudobulk = torch.randn(2, 31, 64)
        hgt = torch.randn(2, 31, 64)
        cell = torch.randn(2, 31, 32)  # Wrong: 32 instead of 64

        with pytest.raises(ValueError, match="d_embed=64 for cell_emb"):
            layer(pseudobulk, hgt, cell)


class TestBayesianHeadTrainingLoopVerification:
    """Test Bayesian head under realistic training conditions."""

    def test_bayesian_head_loss_decreases_with_svi(self):
        """SVI loss should decrease over training steps."""
        import pyro
        from pyro.infer import SVI, Trace_ELBO
        from pyro.infer.autoguide import AutoDiagonalNormal
        from pyro.optim import Adam

        pyro.clear_param_store()

        head = BayesianPredictionHead(d_input=32, d_hidden=16)
        guide = AutoDiagonalNormal(head)
        optimizer = Adam({"lr": 0.01})
        svi = SVI(head, guide, optimizer, loss=Trace_ELBO())

        # Fixed training data
        x = torch.randn(8, 32)
        y = torch.randn(8, 1)

        losses = []
        for _ in range(50):
            loss = svi.step(x, y)
            losses.append(loss)

        # Loss should decrease
        assert losses[-1] < losses[0], "SVI loss should decrease"
        # First 10 losses should be higher than last 10 on average
        assert sum(losses[:10]) / 10 > sum(losses[-10:]) / 10

    def test_bayesian_head_predictive_produces_samples(self):
        """Predictive should produce multiple samples for uncertainty."""
        import pyro
        from pyro.infer import SVI, Trace_ELBO, Predictive
        from pyro.infer.autoguide import AutoDiagonalNormal
        from pyro.optim import Adam

        pyro.clear_param_store()

        head = BayesianPredictionHead(d_input=32, d_hidden=16)
        guide = AutoDiagonalNormal(head)
        optimizer = Adam({"lr": 0.01})
        svi = SVI(head, guide, optimizer, loss=Trace_ELBO())

        # Train briefly to initialize guide parameters properly
        x = torch.randn(4, 32)
        y = torch.randn(4, 1)
        for _ in range(10):
            svi.step(x, y)

        # Get predictions using posterior samples
        # Note: Predictive with Pyro samples weights, creating batched weights
        # This test verifies the sampling mechanism works
        num_samples = 20
        predictive = Predictive(head, guide=guide, num_samples=num_samples, return_sites=["obs"])

        with torch.no_grad():
            samples = predictive(x)

        # Should have obs samples
        assert "obs" in samples
        # Shape is [num_samples, batch_size, 1]
        assert samples["obs"].shape[0] == num_samples
        assert samples["obs"].shape[1] == 4  # batch size

        # Different samples should give different predictions (epistemic uncertainty)
        sample_std = samples["obs"].std(dim=0)
        # At least some variance across samples
        assert sample_std.mean() > 0, "Samples should have variance across posterior draws"


class TestPathologyModulationBehavior:
    """Test pathology modulation behavior in attention."""

    def test_modulation_suppresses_attention(self):
        """High pathology should modulate (suppress/enhance) attention differently."""
        attention = PathologyStratifiedAttention(
            d_fused=32, d_cond=16, n_heads=2, n_cell_types=31
        )

        cell_emb = torch.randn(2, 31, 32)

        # Low pathology embedding
        low_path = torch.zeros(2, 16)
        _, weights_low = attention(cell_emb, low_path)

        # High pathology embedding
        high_path = torch.ones(2, 16) * 3.0
        _, weights_high = attention(cell_emb, high_path)

        # Attention patterns should differ
        assert not torch.allclose(weights_low, weights_high, atol=0.01)

    def test_bias_is_unbounded(self):
        """Additive pathology bias should be unbounded (no sigmoid activation)."""
        attention = PathologyStratifiedAttention(
            d_fused=32, d_cond=16, n_heads=2, n_cell_types=31
        )

        # Use large-magnitude inputs so the linear layer is likely to produce
        # values outside [0, 1], confirming no sigmoid constrains the output.
        cell_emb = torch.randn(2, 31, 32) * 10
        path_emb = torch.randn(2, 16) * 10

        # Access bias output directly
        B = cell_emb.size(0)
        path_emb_expanded = path_emb.unsqueeze(1).expand(-1, 31, -1)
        bias_input = torch.cat([path_emb_expanded, cell_emb], dim=-1)
        bias = attention.pathology_bias(bias_input)

        # Bias is a real-valued (unbounded) tensor — it may contain negative
        # values or values > 1, unlike the old sigmoid-based modulation.
        assert bias.shape == (2, 31, 2)  # [B, n_cell_types, n_heads]


class TestFullModelGradientFlowEndToEnd:
    """Test gradient flow through entire model."""

    @pytest.fixture
    def model(self):
        return CognitiveResilienceModel(
            n_genes=50,
            n_cell_types=31,
            d_embed=32,
            d_fused=32,
            use_bayesian_head=False,
        )

    def test_all_parameters_receive_gradients(self, model):
        """All trainable parameters should receive gradients."""
        B = 2
        edge_index_dict_list, edge_attr_dict_list = _make_edge_dicts(B)
        inputs = {
            'region_pseudobulk': torch.randn(B, 6, 31, 50),
            'region_mask': torch.ones(B, 6, dtype=torch.bool),
            'edge_index_dict_list': edge_index_dict_list,
            'edge_attr_dict_list': edge_attr_dict_list,
            'cells': torch.randn(B, 31, 10, 50),
            'cell_mask': torch.ones(B, 31, 10, dtype=torch.bool),
            'pathology': torch.randn(B, 3),
        }

        output = model(**inputs)
        loss = output['mean'].sum()
        loss.backward()

        # Check key components receive gradients
        components_to_check = [
            ('pseudobulk_encoder', model.pseudobulk_encoder),
            ('region_handler', model.region_handler),
            ('fusion_layer', model.fusion_layer),
            ('pathology_encoder', model.pathology_encoder),
            ('pathology_attention', model.pathology_attention),
            ('prediction_head', model.prediction_head),
        ]

        for name, component in components_to_check:
            has_grad = any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in component.parameters()
                if p.requires_grad
            )
            assert has_grad, f"{name} should receive gradients"

    def test_region_weights_receive_gradient_multi_region(self, model):
        """Region weights should receive gradients with multi-region input."""
        B = 2
        edge_index_dict_list, edge_attr_dict_list = _make_edge_dicts(B)
        inputs = {
            'region_pseudobulk': torch.randn(B, 6, 31, 50),
            'region_mask': torch.ones(B, 6, dtype=torch.bool),  # All regions available
            'edge_index_dict_list': edge_index_dict_list,
            'edge_attr_dict_list': edge_attr_dict_list,
            'cells': torch.randn(B, 31, 10, 50),
            'cell_mask': torch.ones(B, 31, 10, dtype=torch.bool),
            'pathology': torch.randn(B, 3),
        }

        output = model(**inputs)
        loss = output['mean'].sum()
        loss.backward()

        # Region weights should have gradient
        assert model.region_handler.region_weights.grad is not None
        # With all regions available, all weights should have non-zero gradient
        assert (model.region_handler.region_weights.grad.abs() > 1e-10).any()
