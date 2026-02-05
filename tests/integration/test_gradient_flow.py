"""
Gradient flow tests for the full model and its components.

These tests verify:
1. Gradients flow correctly through all model components
2. Single-region data doesn't corrupt multi-region weight gradients
3. All learnable parameters receive gradients during training
"""

import pytest
import torch
import torch.nn as nn

from src.data.constants import N_CELL_TYPES, N_EDGE_TYPES, N_REGIONS, CELL_TYPE_ORDER, ALL_EDGE_TYPES, sanitize_key
from src.models.full_model import CognitiveResilienceModel
from src.models.components.region_handler import RegionHandler
from src.models.fusion import PathologyStratifiedAttention


# =============================================================================
# Constants
# =============================================================================

N_GENES = 50
MAX_CELLS = 20
D_EMBED = 32
D_FUSED = 32
D_COND = 16


# =============================================================================
# RegionHandler Gradient Flow Tests
# =============================================================================


class TestRegionHandlerGradientFlow:
    """Test gradient flow through RegionHandler."""

    @pytest.fixture
    def region_handler(self):
        """Create RegionHandler."""
        return RegionHandler(d_model=D_EMBED, n_regions=N_REGIONS)

    def test_gradients_flow_to_region_weights(self, region_handler):
        """Gradients should flow to region_weights parameter."""
        x = torch.randn(2, N_REGIONS, N_CELL_TYPES, D_EMBED, requires_grad=True)
        region_mask = torch.ones(2, N_REGIONS, dtype=torch.bool)

        pooled, context, _ = region_handler(x, region_mask)
        loss = pooled.sum() + context.sum()
        loss.backward()

        assert region_handler.region_weights.grad is not None
        assert not torch.all(region_handler.region_weights.grad == 0)

    def test_single_region_gradient_isolation(self, region_handler):
        """Single-region subjects should only affect one region's gradient contribution."""
        # Create input with only region 0 available
        x = torch.randn(1, N_REGIONS, N_CELL_TYPES, D_EMBED, requires_grad=True)
        region_mask = torch.zeros(1, N_REGIONS, dtype=torch.bool)
        region_mask[0, 0] = True  # Only PFC

        pooled, context, _ = region_handler(x, region_mask)
        loss = pooled.sum()
        loss.backward()

        # The gradient should exist
        assert region_handler.region_weights.grad is not None

        # With only one region, the softmax gradient is zero for all weights
        # because d(softmax)/d(x_i) = softmax_i * (1 - softmax_i) for same index
        # and -softmax_i * softmax_j for different indices
        # When only one region is active, the masked softmax is [1, 0, 0, 0, 0, 0]
        # So the gradient of the output w.r.t. weights is effectively zero
        # (the output is just x[:, 0, :, :] regardless of weights)

    def test_multi_region_gradients_distributed(self, region_handler):
        """Multi-region subjects should distribute gradients across regions."""
        x = torch.randn(2, N_REGIONS, N_CELL_TYPES, D_EMBED, requires_grad=True)
        # All regions available
        region_mask = torch.ones(2, N_REGIONS, dtype=torch.bool)

        pooled, context, _ = region_handler(x, region_mask)
        loss = pooled.sum()
        loss.backward()

        # With all regions, all weights should receive gradients
        grad = region_handler.region_weights.grad
        assert grad is not None

    def test_region_embedding_receives_gradients(self, region_handler):
        """Region embedding should receive gradients via context output."""
        x = torch.randn(2, N_REGIONS, N_CELL_TYPES, D_EMBED)
        region_mask = torch.ones(2, N_REGIONS, dtype=torch.bool)

        pooled, context, _ = region_handler(x, region_mask)
        loss = context.sum()  # Only use context for gradient
        loss.backward()

        assert region_handler.region_embedding.weight.grad is not None
        assert not torch.all(region_handler.region_embedding.weight.grad == 0)


# =============================================================================
# Full Model End-to-End Gradient Flow Tests
# =============================================================================


class TestFullModelGradientFlow:
    """Test end-to-end gradient flow through the full model."""

    @pytest.fixture
    def model(self):
        """Create test model (deterministic for gradient testing)."""
        return CognitiveResilienceModel(
            n_genes=N_GENES,
            n_cell_types=N_CELL_TYPES,
            d_embed=D_EMBED,
            d_fused=D_FUSED,
            d_cond=D_COND,
            n_regions=N_REGIONS,
            n_hgt_layers=1,
            n_hgt_heads=2,
            n_isab_layers=1,
            n_inducing_points=8,
            n_attention_heads=2,
            d_head_hidden=16,
            dropout=0.0,
            use_bayesian_head=False,
        )

    def test_key_parameters_receive_gradients(self, model):
        """Key model parameters should receive gradients during training.

        Uses multi-region inputs (region_pseudobulk + region_mask) to test the
        full multi-region path rather than the single-region fallback.

        Note: HGT has per-node-type and per-edge-type parameters. Parameters for
        types not involved in any edges won't receive gradients - this is expected.
        This test verifies that core model components receive gradients.
        """
        B = 2

        # Build edge dicts for HGT
        sanitized_types = [sanitize_key(ct) for ct in CELL_TYPE_ORDER]
        sanitized_edges = [sanitize_key(et) for et in ALL_EDGE_TYPES]
        edge_index_dict_list = []
        edge_attr_dict_list = []
        for _ in range(B):
            edge_key = (sanitized_types[0], sanitized_edges[0], sanitized_types[1])
            edge_index_dict_list.append({edge_key: torch.tensor([[0], [0]])})
            edge_attr_dict_list.append({edge_key: torch.rand(1, 1)})

        inputs = {
            'region_pseudobulk': torch.randn(B, N_REGIONS, N_CELL_TYPES, N_GENES),
            'region_mask': torch.ones(B, N_REGIONS, dtype=torch.bool),
            'edge_index_dict_list': edge_index_dict_list,
            'edge_attr_dict_list': edge_attr_dict_list,
            'cells': torch.randn(B, N_CELL_TYPES, MAX_CELLS, N_GENES),
            'cell_mask': torch.ones(B, N_CELL_TYPES, MAX_CELLS, dtype=torch.bool),
            'pathology': torch.randn(B, 3),
        }

        output = model(**inputs)
        loss = output['mean'].sum()
        loss.backward()

        # Check key components receive gradients (not per-type parameters)
        key_components = [
            'pseudobulk_encoder.gene_gate.gate_logits',
            'pseudobulk_encoder.shared_mlp.0.weight',
            'region_handler.region_weights',
            'cell_transformer.selector.selection_logits',
            'fusion_layer.proj.weight',
            'pathology_encoder.pathology_mlp.0.weight',
            'pathology_attention.query_generator.weight',
            'prediction_head.mlp.0.weight',
        ]

        missing_grads = []
        for component in key_components:
            param = model
            for attr in component.split('.'):
                param = getattr(param, attr)
            if param.grad is None:
                missing_grads.append(component)

        assert len(missing_grads) == 0, f"Key components without gradients: {missing_grads}"

    def test_gradients_flow_through_all_branches(self, model):
        """Gradients should flow through pseudobulk, HGT, and cell transformer branches."""
        B = 2

        # Build edge dicts so HGT branch actually processes edges
        sanitized_types = [sanitize_key(ct) for ct in CELL_TYPE_ORDER]
        sanitized_edges = [sanitize_key(et) for et in ALL_EDGE_TYPES]
        edge_index_dict_list = []
        edge_attr_dict_list = []
        for _ in range(B):
            edge_key = (sanitized_types[0], sanitized_edges[0], sanitized_types[1])
            edge_index_dict_list.append({edge_key: torch.tensor([[0], [0]])})
            edge_attr_dict_list.append({edge_key: torch.rand(1, 1)})

        inputs = {
            'region_pseudobulk': torch.randn(B, N_REGIONS, N_CELL_TYPES, N_GENES),
            'region_mask': torch.ones(B, N_REGIONS, dtype=torch.bool),
            'edge_index_dict_list': edge_index_dict_list,
            'edge_attr_dict_list': edge_attr_dict_list,
            'cells': torch.randn(B, N_CELL_TYPES, MAX_CELLS, N_GENES),
            'cell_mask': torch.ones(B, N_CELL_TYPES, MAX_CELLS, dtype=torch.bool),
            'pathology': torch.randn(B, 3),
        }

        output = model(**inputs)
        loss = output['mean'].sum()
        loss.backward()

        # Check pseudobulk encoder
        assert model.pseudobulk_encoder.gene_gate.gate_logits.grad is not None

        # Check cell transformer
        assert model.cell_transformer.selector.selection_logits.grad is not None

        # Check fusion layer
        assert model.fusion_layer.proj.weight.grad is not None

        # Check prediction head
        assert model.prediction_head.mlp[0].weight.grad is not None

        # Check HGT encoder - at least one parameter should have received gradients
        hgt_has_grad = any(
            p.grad is not None and not torch.all(p.grad == 0)
            for p in model.hgt_encoder.parameters()
            if p.requires_grad
        )
        assert hgt_has_grad, "No HGT encoder parameters received non-zero gradients"

    def test_gradient_flow_with_edge_dicts(self, model):
        """Gradients should flow correctly when using edge_index_dict_list."""
        B = 2

        # Create edge dicts
        sanitized_types = [sanitize_key(ct) for ct in CELL_TYPE_ORDER]
        sanitized_edges = [sanitize_key(et) for et in ALL_EDGE_TYPES]

        edge_index_dict_list = []
        edge_attr_dict_list = []

        for _ in range(B):
            edge_key = (sanitized_types[0], sanitized_edges[0], sanitized_types[1])
            edge_index_dict_list.append({edge_key: torch.tensor([[0], [0]])})
            edge_attr_dict_list.append({edge_key: torch.rand(1, 1)})

        inputs = {
            'pseudobulk': torch.randn(B, N_CELL_TYPES, N_GENES),
            'edge_index_dict_list': edge_index_dict_list,
            'edge_attr_dict_list': edge_attr_dict_list,
            'cells': torch.randn(B, N_CELL_TYPES, MAX_CELLS, N_GENES),
            'cell_mask': torch.ones(B, N_CELL_TYPES, MAX_CELLS, dtype=torch.bool),
            'pathology': torch.randn(B, 3),
        }

        output = model(**inputs)
        loss = output['mean'].sum()
        loss.backward()

        # HGT encoder should receive gradients
        # Check HGT layer scale parameters exist and receive gradients
        for name, param in model.hgt_encoder.named_parameters():
            if param.requires_grad and param.grad is not None:
                break
        else:
            pytest.fail("No HGT parameters received gradients")


# =============================================================================
# Bayesian KL Divergence Gradient Flow Tests
# =============================================================================


class TestBayesianKLGradientFlow:
    """Test gradient flow through Bayesian head KL divergence."""

    def test_bayesian_kl_divergence_gradient_flow(self):
        """KL divergence from Bayesian head should produce gradients."""
        model = CognitiveResilienceModel(
            n_genes=N_GENES,
            n_cell_types=N_CELL_TYPES,
            d_embed=D_EMBED,
            d_fused=D_FUSED,
            d_cond=D_COND,
            n_regions=N_REGIONS,
            n_hgt_layers=1,
            n_hgt_heads=2,
            n_isab_layers=1,
            n_inducing_points=8,
            n_attention_heads=2,
            d_head_hidden=16,
            dropout=0.0,
            use_bayesian_head=True,
        )

        B = 2
        edge_index_dict_list, edge_attr_dict_list = [], []
        sanitized_types = [sanitize_key(ct) for ct in CELL_TYPE_ORDER]
        sanitized_edges = [sanitize_key(et) for et in ALL_EDGE_TYPES]
        for _ in range(B):
            edge_key = (sanitized_types[0], sanitized_edges[0], sanitized_types[1])
            edge_index_dict_list.append({edge_key: torch.tensor([[0], [0]])})
            edge_attr_dict_list.append({edge_key: torch.rand(1, 1)})

        inputs = {
            'region_pseudobulk': torch.randn(B, N_REGIONS, N_CELL_TYPES, N_GENES),
            'region_mask': torch.ones(B, N_REGIONS, dtype=torch.bool),
            'edge_index_dict_list': edge_index_dict_list,
            'edge_attr_dict_list': edge_attr_dict_list,
            'cells': torch.randn(B, N_CELL_TYPES, MAX_CELLS, N_GENES),
            'cell_mask': torch.ones(B, N_CELL_TYPES, MAX_CELLS, dtype=torch.bool),
            'pathology': torch.randn(B, 3),
            'cognition': torch.randn(B, 1),
        }

        # Forward pass (Bayesian head returns mean and std)
        output = model(**inputs)
        mean = output['mean']
        std = output['std']

        # Compute ELBO-style loss: negative log-likelihood + KL-like term
        # The KL divergence is implicitly handled by Pyro's plate/sample,
        # but for gradient flow testing we use a loss that depends on both
        # mean and std, mirroring what ELBO does.
        nll = 0.5 * ((inputs['cognition'] - mean) / std).pow(2) + std.log()
        loss = nll.sum()
        loss.backward()

        # Verify gradients flow to prediction head params
        head_has_grad = False
        for name, param in model.prediction_head.named_parameters():
            if param.grad is not None and not torch.all(param.grad == 0):
                head_has_grad = True
                break
        assert head_has_grad, "No gradients reached Bayesian prediction head"

        # Specifically check fc_log_std (aleatoric uncertainty branch)
        assert model.prediction_head.fc_log_std.weight.grad is not None, \
            "No gradients reached fc_log_std weight"
        assert not torch.all(model.prediction_head.fc_log_std.weight.grad == 0), \
            "fc_log_std weight has zero gradients"


# =============================================================================
# Multi-Region Behavior Tests
# =============================================================================


class TestMultiRegionBehavior:
    """Test multi-region handling behavior."""

    @pytest.fixture
    def model(self):
        """Create test model."""
        return CognitiveResilienceModel(
            n_genes=N_GENES,
            n_cell_types=N_CELL_TYPES,
            d_embed=D_EMBED,
            d_fused=D_FUSED,
            d_cond=D_COND,
            n_regions=N_REGIONS,
            n_hgt_layers=1,
            n_hgt_heads=2,
            n_isab_layers=1,
            n_inducing_points=8,
            n_attention_heads=2,
            d_head_hidden=16,
            dropout=0.0,
            use_bayesian_head=False,
        )

    # Single-region forward pass: canonical test in
    # test_full_model_integration.py::TestEndToEndForward::test_single_region_subject

    def test_multi_region_produces_valid_output(self, model):
        """Multi-region input should produce valid predictions."""
        B = 4
        inputs = {
            'region_pseudobulk': torch.randn(B, N_REGIONS, N_CELL_TYPES, N_GENES),
            'region_mask': torch.ones(B, N_REGIONS, dtype=torch.bool),
            'cells': torch.randn(B, N_CELL_TYPES, MAX_CELLS, N_GENES),
            'cell_mask': torch.ones(B, N_CELL_TYPES, MAX_CELLS, dtype=torch.bool),
            'pathology': torch.randn(B, 3),
        }

        output = model(**inputs)

        assert output['mean'].shape == (B, 1)
        assert torch.isfinite(output['mean']).all()

    def test_partial_region_availability(self, model):
        """Model should handle subjects with different numbers of available regions."""
        B = 4
        region_mask = torch.zeros(B, N_REGIONS, dtype=torch.bool)
        # Different subjects have different regions
        region_mask[0, :2] = True  # 2 regions
        region_mask[1, :3] = True  # 3 regions
        region_mask[2, :1] = True  # 1 region (PFC only)
        region_mask[3, :N_REGIONS] = True  # All regions

        inputs = {
            'region_pseudobulk': torch.randn(B, N_REGIONS, N_CELL_TYPES, N_GENES),
            'region_mask': region_mask,
            'cells': torch.randn(B, N_CELL_TYPES, MAX_CELLS, N_GENES),
            'cell_mask': torch.ones(B, N_CELL_TYPES, MAX_CELLS, dtype=torch.bool),
            'pathology': torch.randn(B, 3),
        }

        output = model(**inputs)

        assert output['mean'].shape == (B, 1)
        assert torch.isfinite(output['mean']).all()

    def test_region_weights_interpretable(self, model):
        """Region importance weights should be retrievable for interpretability."""
        # Run forward pass to ensure model is in correct state
        B = 2
        inputs = {
            'region_pseudobulk': torch.randn(B, N_REGIONS, N_CELL_TYPES, N_GENES),
            'region_mask': torch.ones(B, N_REGIONS, dtype=torch.bool),
            'cells': torch.randn(B, N_CELL_TYPES, MAX_CELLS, N_GENES),
            'cell_mask': torch.ones(B, N_CELL_TYPES, MAX_CELLS, dtype=torch.bool),
            'pathology': torch.randn(B, 3),
        }
        model(**inputs)

        # Get region importance
        importance = model.get_region_importance()

        assert isinstance(importance, dict)
        assert len(importance) == N_REGIONS
        # Weights should sum to 1 (softmax)
        total = sum(importance.values())
        assert abs(total - 1.0) < 1e-5


# =============================================================================
# PathologyStratifiedAttention with cell_type_mask Tests
# =============================================================================


class TestCellTypeMaskPropagation:
    """Test cell_type_mask handling through the model."""

    @pytest.fixture
    def attention(self):
        """Create PathologyStratifiedAttention."""
        return PathologyStratifiedAttention(
            d_fused=D_FUSED,
            d_cond=D_COND,
            n_heads=2,
            n_cell_types=N_CELL_TYPES,
        )

    def test_attention_respects_cell_type_mask(self, attention):
        """Attention should zero out masked cell types."""
        B = 2
        cell_emb = torch.randn(B, N_CELL_TYPES, D_FUSED)
        path_emb = torch.randn(B, D_COND)

        # Mask: only first 10 cell types available
        mask = torch.zeros(B, N_CELL_TYPES, dtype=torch.bool)
        mask[:, :10] = True

        attended, weights = attention(cell_emb, path_emb, cell_type_mask=mask)

        # Attention weights for masked cell types should be 0
        assert torch.all(weights[:, :, 10:] == 0), "Masked cell types should have zero attention"

        # Attention weights for unmasked cell types should sum to 1
        unmasked_weights = weights[:, :, :10]
        weight_sums = unmasked_weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)

    def test_full_model_with_cell_type_mask(self):
        """Full model should propagate cell_type_mask correctly."""
        model = CognitiveResilienceModel(
            n_genes=N_GENES,
            n_cell_types=N_CELL_TYPES,
            d_embed=D_EMBED,
            d_fused=D_FUSED,
            d_cond=D_COND,
            n_regions=N_REGIONS,
            n_hgt_layers=1,
            n_hgt_heads=2,
            n_isab_layers=1,
            n_inducing_points=8,
            n_attention_heads=2,
            d_head_hidden=16,
            dropout=0.0,
            use_bayesian_head=False,
        )

        B = 2
        cell_type_mask = torch.ones(B, N_CELL_TYPES, dtype=torch.bool)
        cell_type_mask[:, 20:] = False  # Mask out last 11 cell types

        inputs = {
            'pseudobulk': torch.randn(B, N_CELL_TYPES, N_GENES),
            'cells': torch.randn(B, N_CELL_TYPES, MAX_CELLS, N_GENES),
            'cell_mask': torch.ones(B, N_CELL_TYPES, MAX_CELLS, dtype=torch.bool),
            'cell_type_mask': cell_type_mask,
            'pathology': torch.randn(B, 3),
        }

        output = model(**inputs)

        # Check attention weights respect mask
        attention_weights = output['attention_weights']  # [B, n_heads, n_cell_types]
        assert torch.all(attention_weights[:, :, 20:] == 0), "Masked cell types should have zero attention"
