"""
Unit tests for CognitiveResilienceModel (full end-to-end model).

Focus areas:
- TestInitialization: Component creation verification
- TestForwardPass: Output structure and shapes

Note: Full gradient flow and integration tests are in Task 8.
"""

import pytest
import torch

from src.data.constants import N_CELL_TYPES, N_REGIONS
from src.models.full_model import CognitiveResilienceModel


class TestInitialization:
    """Test model initialization and component creation."""

    @pytest.fixture
    def model_config(self):
        """Small model configuration for testing."""
        return {
            'n_genes': 100,
            'n_cell_types': N_CELL_TYPES,
            'd_embed': 32,
            'd_fused': 32,
            'd_cond': 16,
            'n_regions': N_REGIONS,
            'n_hgt_layers': 2,
            'n_hgt_heads': 4,
            'n_isab_layers': 1,
            'n_inducing_points': 8,
            'n_attention_heads': 4,
            'd_head_hidden': 16,
            'dropout': 0.1,
        }

    def test_creates_all_branches(self, model_config):
        """Test that all three encoder branches are created."""
        model = CognitiveResilienceModel(**model_config, use_bayesian_head=True)

        # Verify all branches exist
        assert hasattr(model, 'pseudobulk_encoder')
        assert hasattr(model, 'hgt_encoder')
        assert hasattr(model, 'cell_transformer')

        # Verify branch types
        from src.models.branches import PseudobulkEncoder, HGTEncoderTensor, CellTransformer
        assert isinstance(model.pseudobulk_encoder, PseudobulkEncoder)
        assert isinstance(model.hgt_encoder, HGTEncoderTensor)
        assert isinstance(model.cell_transformer, CellTransformer)

    def test_creates_region_handler(self, model_config):
        """Test that RegionHandler is created with correct parameters."""
        model = CognitiveResilienceModel(**model_config)

        assert hasattr(model, 'region_handler')

        from src.models.components import RegionHandler
        assert isinstance(model.region_handler, RegionHandler)
        assert model.region_handler.d_model == model_config['d_embed']
        assert model.region_handler.n_regions == model_config['n_regions']

    def test_creates_fusion_components(self, model_config):
        """Test that fusion components are created correctly."""
        model = CognitiveResilienceModel(**model_config)

        # FusionLayer
        assert hasattr(model, 'fusion_layer')
        from src.models.fusion import FusionLayer
        assert isinstance(model.fusion_layer, FusionLayer)
        assert model.fusion_layer.d_embed == model_config['d_embed']
        assert model.fusion_layer.d_fused == model_config['d_fused']

        # PathologyEncoder
        assert hasattr(model, 'pathology_encoder')
        from src.models.fusion import PathologyEncoder
        assert isinstance(model.pathology_encoder, PathologyEncoder)
        assert model.pathology_encoder.d_cond == model_config['d_cond']

        # PathologyStratifiedAttention
        assert hasattr(model, 'pathology_attention')
        from src.models.fusion import PathologyStratifiedAttention
        assert isinstance(model.pathology_attention, PathologyStratifiedAttention)
        assert model.pathology_attention.d_fused == model_config['d_fused']
        assert model.pathology_attention.d_cond == model_config['d_cond']

    def test_creates_bayesian_head_by_default(self, model_config):
        """Test that Bayesian head is created by default."""
        model = CognitiveResilienceModel(**model_config, use_bayesian_head=True)

        assert hasattr(model, 'prediction_head')
        from src.models.heads import BayesianPredictionHead
        assert isinstance(model.prediction_head, BayesianPredictionHead)
        assert model.use_bayesian_head is True

    def test_creates_deterministic_head_when_specified(self, model_config):
        """Test that deterministic head is created when specified."""
        model = CognitiveResilienceModel(**model_config, use_bayesian_head=False)

        assert hasattr(model, 'prediction_head')
        from src.models.heads import DeterministicPredictionHead
        assert isinstance(model.prediction_head, DeterministicPredictionHead)
        assert model.use_bayesian_head is False

    def test_invalid_n_genes_raises_error(self, model_config):
        """Test that invalid n_genes raises ValueError."""
        model_config['n_genes'] = 0
        with pytest.raises(ValueError, match="n_genes must be positive"):
            CognitiveResilienceModel(**model_config)

    def test_invalid_d_fused_attention_heads_raises_error(self, model_config):
        """Test that d_fused not divisible by n_attention_heads raises error."""
        model_config['d_fused'] = 33  # Not divisible by 4
        model_config['n_attention_heads'] = 4
        with pytest.raises(ValueError, match="d_fused.*must be divisible by n_attention_heads"):
            CognitiveResilienceModel(**model_config)

    def test_invalid_n_cell_types_raises_error(self):
        """n_cell_types=0 should raise ValueError."""
        with pytest.raises(ValueError, match="n_cell_types must be positive"):
            CognitiveResilienceModel(n_genes=50, n_cell_types=0, d_embed=32, d_fused=32, d_cond=16)

    def test_invalid_d_embed_raises_error(self):
        """d_embed=0 should raise ValueError."""
        with pytest.raises(ValueError, match="d_embed must be positive"):
            CognitiveResilienceModel(n_genes=50, d_embed=0, d_fused=32, d_cond=16)

    def test_invalid_d_cond_raises_error(self):
        """d_cond=0 should raise ValueError."""
        with pytest.raises(ValueError, match="d_cond must be positive"):
            CognitiveResilienceModel(n_genes=50, d_embed=32, d_fused=32, d_cond=0)


class TestForwardPass:
    """Test forward pass output structure and shapes."""

    @pytest.fixture
    def small_model_bayesian(self):
        """Create small Bayesian model for testing."""
        return CognitiveResilienceModel(
            n_genes=50,
            n_cell_types=N_CELL_TYPES,
            d_embed=32,
            d_fused=32,
            d_cond=16,
            n_regions=N_REGIONS,
            n_hgt_layers=1,
            n_hgt_heads=4,
            n_isab_layers=1,
            n_inducing_points=4,
            n_attention_heads=4,
            use_bayesian_head=True,
            d_head_hidden=16,
            dropout=0.0,  # Disable dropout for deterministic testing
        )

    @pytest.fixture
    def small_model_deterministic(self):
        """Create small deterministic model for testing."""
        return CognitiveResilienceModel(
            n_genes=50,
            n_cell_types=N_CELL_TYPES,
            d_embed=32,
            d_fused=32,
            d_cond=16,
            n_regions=N_REGIONS,
            n_hgt_layers=1,
            n_hgt_heads=4,
            n_isab_layers=1,
            n_inducing_points=4,
            n_attention_heads=4,
            use_bayesian_head=False,
            d_head_hidden=16,
            dropout=0.0,
        )

    @pytest.fixture
    def sample_inputs(self):
        """Create sample inputs for forward pass."""
        B = 2
        n_regions = N_REGIONS
        n_cell_types = N_CELL_TYPES
        n_genes = 50
        max_cells = 10
        n_edges = 5

        src = torch.cat([torch.randint(0, n_cell_types, (n_edges,)) + b * n_cell_types for b in range(B)])
        dst = torch.cat([torch.randint(0, n_cell_types, (n_edges,)) + b * n_cell_types for b in range(B)])
        return {
            'region_pseudobulk': torch.randn(B, n_regions, n_cell_types, n_genes),
            'region_mask': torch.ones(B, n_regions, dtype=torch.bool),
            'ccc_edge_index': torch.stack([src, dst]),
            'ccc_edge_type': torch.randint(0, 5, (B * n_edges,)),
            'ccc_edge_attr': torch.rand(B * n_edges, 1),
            'cells': torch.randn(B, n_cell_types, max_cells, n_genes),
            'cell_mask': torch.ones(B, n_cell_types, max_cells, dtype=torch.bool),
            'pathology': torch.randn(B, 3),
            'cognition': torch.randn(B, 1),
        }

    def test_output_structure_bayesian(self, small_model_bayesian, sample_inputs):
        """Test that Bayesian model returns correct output structure."""
        output = small_model_bayesian(**sample_inputs)

        assert isinstance(output, dict)
        assert 'mean' in output
        assert 'std' in output
        assert 'attention_weights' in output

    def test_output_structure_deterministic(self, small_model_deterministic, sample_inputs):
        """Test that deterministic model returns correct output structure."""
        output = small_model_deterministic(**sample_inputs)

        assert isinstance(output, dict)
        assert 'mean' in output
        assert 'std' not in output  # No std for deterministic
        assert 'attention_weights' in output

    def test_output_shapes_bayesian(self, small_model_bayesian, sample_inputs):
        """Test output shapes for Bayesian model."""
        B = sample_inputs['region_pseudobulk'].size(0)
        n_cell_types = N_CELL_TYPES
        n_attention_heads = 4

        # Training mode: attention_weights skipped for performance
        output_train = small_model_bayesian(**sample_inputs)
        assert output_train['mean'].shape == (B, 1)
        assert output_train['std'].shape == (B, 1)
        assert output_train['attention_weights'] is None

        # Eval mode: attention_weights computed for interpretability
        small_model_bayesian.eval()
        output = small_model_bayesian(**sample_inputs)
        small_model_bayesian.train()

        assert output['mean'].shape == (B, 1)
        assert output['std'].shape == (B, 1)
        assert output['attention_weights'].shape == (B, n_attention_heads, n_cell_types)

    def test_output_shapes_deterministic(self, small_model_deterministic, sample_inputs):
        """Test output shapes for deterministic model."""
        B = sample_inputs['region_pseudobulk'].size(0)
        n_cell_types = N_CELL_TYPES
        n_attention_heads = 4

        # Training mode: attention_weights skipped for performance
        output_train = small_model_deterministic(**sample_inputs)
        assert output_train['mean'].shape == (B, 1)
        assert output_train['attention_weights'] is None

        # Eval mode: attention_weights computed for interpretability
        small_model_deterministic.eval()
        output = small_model_deterministic(**sample_inputs)
        small_model_deterministic.train()

        assert output['mean'].shape == (B, 1)
        assert output['attention_weights'].shape == (B, n_attention_heads, n_cell_types)

    def test_forward_without_cognition(self, small_model_deterministic, sample_inputs):
        """Test forward pass works without cognition (inference mode)."""
        # Remove cognition from inputs
        inputs = {k: v for k, v in sample_inputs.items() if k != 'cognition'}

        output = small_model_deterministic(**inputs)

        assert 'mean' in output
        assert output['mean'].shape == (2, 1)

    def test_attention_weights_sum_to_one(self, small_model_deterministic, sample_inputs):
        """Test that attention weights sum to approximately 1 across cell types."""
        # Eval mode: attention_weights only computed outside training
        small_model_deterministic.eval()
        output = small_model_deterministic(**sample_inputs)
        small_model_deterministic.train()

        attention_weights = output['attention_weights']  # [B, n_heads, n_cell_types]

        # Sum across cell types dimension
        weight_sums = attention_weights.sum(dim=-1)  # [B, n_heads]

        # Should sum to approximately 1
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)

    def test_std_is_positive(self, small_model_bayesian, sample_inputs):
        """Test that std output is always positive (Bayesian model)."""
        output = small_model_bayesian(**sample_inputs)

        assert (output['std'] > 0).all()


class TestInterpretability:
    """Test interpretability methods."""

    @pytest.fixture
    def model(self):
        """Create model for interpretability testing."""
        return CognitiveResilienceModel(
            n_genes=50,
            n_cell_types=N_CELL_TYPES,
            d_embed=32,
            d_fused=32,
            d_cond=16,
            n_regions=N_REGIONS,
            n_hgt_layers=1,
            n_hgt_heads=4,
            n_isab_layers=1,
            n_inducing_points=4,
            n_attention_heads=4,
            use_bayesian_head=False,
            d_head_hidden=16,
        )

    def test_get_cell_type_importance(self, model):
        """Test cell type importance extraction."""
        importance = model.get_cell_type_importance()

        assert isinstance(importance, dict)
        assert len(importance) == N_CELL_TYPES  # All cell types
        assert all(isinstance(v, float) for v in importance.values())

        # Weights are independent sigmoid values in (0, 1) per type
        assert all(0 < v < 1 for v in importance.values())

    def test_get_hgt_layer_scales(self, model):
        """Test HGT layer scale extraction."""
        scales = model.get_hgt_layer_scales()

        assert isinstance(scales, dict)
        assert 'scales' in scales
        assert 'cell_types' in scales
        assert 'per_cell_type' in scales

        # scales tensor: [n_layers, n_node_types]
        assert scales['scales'].dim() == 2
        n_layers = scales['scales'].size(0)
        assert n_layers > 0
        assert scales['scales'].size(1) == N_CELL_TYPES

        # cell_types list matches node count
        assert len(scales['cell_types']) == N_CELL_TYPES

        # per_cell_type has one entry per cell type, each [n_layers]
        assert len(scales['per_cell_type']) == N_CELL_TYPES
        for ct, vals in scales['per_cell_type'].items():
            assert vals.shape == (n_layers,)

    def test_get_region_importance(self, model):
        """Test region importance extraction."""
        importance = model.get_region_importance()

        assert isinstance(importance, dict)
        assert len(importance) == N_REGIONS  # All regions

        # Weights are independent sigmoid values in (0, 1) per type
        assert all(0 < v < 1 for v in importance.values())

    def test_num_parameters_structure(self, model):
        """Test num_parameters returns all expected components."""
        counts = model.num_parameters()

        expected_keys = {
            'total', 'pseudobulk_encoder', 'region_handler', 'hgt_encoder',
            'cell_transformer', 'fusion_layer', 'pathology_encoder',
            'pathology_attention', 'prediction_head',
        }
        assert set(counts.keys()) == expected_keys
        assert all(isinstance(v, int) and v > 0 for v in counts.values())

        # Component counts should sum to total (no shared parameters)
        component_sum = sum(v for k, v in counts.items() if k != 'total')
        assert component_sum == counts['total']

    def test_num_parameters_count_stability(self, model):
        """Pin parameter count for standard test config to catch accidental architecture changes.

        Config: n_genes=50, d_embed=32, d_fused=32, d_cond=16, n_hgt_layers=1,
        n_hgt_heads=4, n_isab_layers=1, n_inducing=4, n_attention_heads=4,
        use_bayesian_head=False, d_head_hidden=16, dropout default.
        """
        counts = model.num_parameters()

        assert counts['pseudobulk_encoder'] == 168_750
        assert counts['region_handler'] == 198
        assert counts['hgt_encoder'] == 136_649
        assert counts['cell_transformer'] == 40_031
        assert counts['fusion_layer'] == 3_168
        assert counts['pathology_encoder'] == 1_456
        assert counts['pathology_attention'] == 3_908
        assert counts['prediction_head'] == 817
        assert counts['total'] == 354_977

    def test_num_parameters_trainable_only_false(self, model):
        """num_parameters(trainable_only=False) should include all params."""
        counts_all = model.num_parameters(trainable_only=False)
        counts_trainable = model.num_parameters(trainable_only=True)
        assert counts_all['total'] >= counts_trainable['total']
        assert counts_all['total'] == counts_trainable['total']  # all trainable by default


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def small_model(self):
        """Create small model for edge case testing."""
        return CognitiveResilienceModel(
            n_genes=50,
            n_cell_types=N_CELL_TYPES,
            d_embed=32,
            d_fused=32,
            d_cond=16,
            n_regions=N_REGIONS,
            n_hgt_layers=1,
            n_hgt_heads=4,
            n_isab_layers=1,
            n_inducing_points=4,
            n_attention_heads=4,
            use_bayesian_head=False,
            d_head_hidden=16,
            dropout=0.0,
        )

    def test_empty_edges(self, small_model):
        """Test forward pass with no CCC edges (empty edge tensors)."""
        B = 2
        n_cell_types = N_CELL_TYPES
        n_genes = 50
        max_cells = 10

        inputs = {
            'region_pseudobulk': torch.randn(B, N_REGIONS, n_cell_types, n_genes),
            'region_mask': torch.ones(B, N_REGIONS, dtype=torch.bool),
            'ccc_edge_index': torch.zeros(2, 0, dtype=torch.long),
            'ccc_edge_type': torch.zeros(0, dtype=torch.long),
            'ccc_edge_attr': torch.zeros(0, 1),
            'cells': torch.randn(B, n_cell_types, max_cells, n_genes),
            'cell_mask': torch.ones(B, n_cell_types, max_cells, dtype=torch.bool),
            'pathology': torch.randn(B, 3),
        }

        output = small_model(**inputs)
        assert output['mean'].shape == (B, 1)

    def test_partial_cell_mask(self, small_model):
        """Test forward pass with some cells masked out."""
        B = 2
        n_cell_types = N_CELL_TYPES
        n_genes = 50
        max_cells = 10

        # Mask out half of cells
        cell_mask = torch.ones(B, n_cell_types, max_cells, dtype=torch.bool)
        cell_mask[:, :, max_cells//2:] = False

        n_edges = 5
        src = torch.cat([torch.randint(0, n_cell_types, (n_edges,)) + b * n_cell_types for b in range(B)])
        dst = torch.cat([torch.randint(0, n_cell_types, (n_edges,)) + b * n_cell_types for b in range(B)])
        inputs = {
            'region_pseudobulk': torch.randn(B, N_REGIONS, n_cell_types, n_genes),
            'region_mask': torch.ones(B, N_REGIONS, dtype=torch.bool),
            'ccc_edge_index': torch.stack([src, dst]),
            'ccc_edge_type': torch.randint(0, 5, (B * n_edges,)),
            'ccc_edge_attr': torch.rand(B * n_edges, 1),
            'cells': torch.randn(B, n_cell_types, max_cells, n_genes),
            'cell_mask': cell_mask,
            'pathology': torch.randn(B, 3),
        }

        output = small_model(**inputs)
        assert output['mean'].shape == (B, 1)

    def test_single_region_pseudobulk_only_fallback(self, small_model):
        """Test forward pass using pseudobulk (not region_pseudobulk) auto-expansion.

        The model should auto-expand [B, C, G] to [B, n_regions, C, G] with only PFC
        filled, and produce the same output structure as the multi-region path.
        """
        B = 2
        n_cell_types = N_CELL_TYPES
        n_genes = 50
        max_cells = 10

        n_edges = 5
        src = torch.cat([torch.randint(0, n_cell_types, (n_edges,)) + b * n_cell_types for b in range(B)])
        dst = torch.cat([torch.randint(0, n_cell_types, (n_edges,)) + b * n_cell_types for b in range(B)])
        inputs = {
            'pseudobulk': torch.randn(B, n_cell_types, n_genes),
            # NO region_pseudobulk or region_mask
            'ccc_edge_index': torch.stack([src, dst]),
            'ccc_edge_type': torch.randint(0, 5, (B * n_edges,)),
            'ccc_edge_attr': torch.rand(B * n_edges, 1),
            'cells': torch.randn(B, n_cell_types, max_cells, n_genes),
            'cell_mask': torch.ones(B, n_cell_types, max_cells, dtype=torch.bool),
            'pathology': torch.randn(B, 3),
        }

        # Eval mode for attention_weights check (skipped during training)
        small_model.eval()
        output = small_model(**inputs)
        small_model.train()
        assert output['mean'].shape == (B, 1)
        assert 'attention_weights' in output
        assert output['attention_weights'].shape == (B, 4, n_cell_types)

    def test_single_region_fallback_matches_explicit_multi_region(self, small_model):
        """forward(pseudobulk=X) produces same output as forward(region_pseudobulk=X_expanded)."""
        from src.data.constants import PFC_REGION_IDX

        B = 2
        n_cell_types = N_CELL_TYPES
        n_genes = 50
        max_cells = 10

        # Build a multi-region input with only PFC filled
        region_pb = torch.randn(B, N_REGIONS, n_cell_types, n_genes)
        # Zero out all regions except PFC
        region_pb[:, :PFC_REGION_IDX, :, :] = 0.0
        if PFC_REGION_IDX + 1 < N_REGIONS:
            region_pb[:, PFC_REGION_IDX + 1:, :, :] = 0.0

        # Single-region: extract PFC
        pseudobulk = region_pb[:, PFC_REGION_IDX, :, :].clone()  # [B, C, G]

        # Build explicit multi-region with only PFC active
        region_mask = torch.zeros(B, N_REGIONS, dtype=torch.bool)
        region_mask[:, PFC_REGION_IDX] = True

        n_edges = 5
        src = torch.cat([torch.randint(0, n_cell_types, (n_edges,)) + b * n_cell_types for b in range(B)])
        dst = torch.cat([torch.randint(0, n_cell_types, (n_edges,)) + b * n_cell_types for b in range(B)])
        common_kwargs = {
            "ccc_edge_index": torch.stack([src, dst]),
            "ccc_edge_type": torch.randint(0, 5, (B * n_edges,)),
            "ccc_edge_attr": torch.rand(B * n_edges, 1),
            "cells": torch.randn(B, n_cell_types, max_cells, n_genes),
            "cell_mask": torch.ones(B, n_cell_types, max_cells, dtype=torch.bool),
            "pathology": torch.randn(B, 3),
        }

        small_model.eval()
        with torch.no_grad():
            out_fallback = small_model(pseudobulk=pseudobulk, **common_kwargs)
            out_explicit = small_model(
                region_pseudobulk=region_pb, region_mask=region_mask, **common_kwargs
            )

        torch.testing.assert_close(
            out_fallback["mean"], out_explicit["mean"], atol=1e-5, rtol=1e-5
        )

    def test_batch_size_one(self, small_model):
        """Test forward pass with batch size of 1."""
        B = 1
        n_cell_types = N_CELL_TYPES
        n_genes = 50
        max_cells = 10

        n_edges = 5
        src = torch.randint(0, n_cell_types, (n_edges,))
        dst = torch.randint(0, n_cell_types, (n_edges,))
        inputs = {
            'region_pseudobulk': torch.randn(B, N_REGIONS, n_cell_types, n_genes),
            'region_mask': torch.ones(B, N_REGIONS, dtype=torch.bool),
            'ccc_edge_index': torch.stack([src, dst]),
            'ccc_edge_type': torch.randint(0, 5, (n_edges,)),
            'ccc_edge_attr': torch.rand(n_edges, 1),
            'cells': torch.randn(B, n_cell_types, max_cells, n_genes),
            'cell_mask': torch.ones(B, n_cell_types, max_cells, dtype=torch.bool),
            'pathology': torch.randn(B, 3),
        }

        output = small_model(**inputs)
        assert output['mean'].shape == (1, 1)

    def test_invalid_cell_type_mask_shape_raises_error(self, small_model):
        """Test that wrong cell_type_mask shape raises ValueError early."""
        B = 2
        n_cell_types = N_CELL_TYPES
        n_genes = 50
        max_cells = 10

        n_edges = 5
        src = torch.cat([torch.randint(0, n_cell_types, (n_edges,)) + b * n_cell_types for b in range(B)])
        dst = torch.cat([torch.randint(0, n_cell_types, (n_edges,)) + b * n_cell_types for b in range(B)])
        inputs = {
            'region_pseudobulk': torch.randn(B, N_REGIONS, n_cell_types, n_genes),
            'region_mask': torch.ones(B, N_REGIONS, dtype=torch.bool),
            'ccc_edge_index': torch.stack([src, dst]),
            'ccc_edge_type': torch.randint(0, 5, (B * n_edges,)),
            'ccc_edge_attr': torch.rand(B * n_edges, 1),
            'cells': torch.randn(B, n_cell_types, max_cells, n_genes),
            'cell_mask': torch.ones(B, n_cell_types, max_cells, dtype=torch.bool),
            'pathology': torch.randn(B, 3),
            'cell_type_mask': torch.ones(B, n_cell_types, n_cell_types, dtype=torch.bool),  # Wrong: 3D
        }

        with pytest.raises(ValueError, match="cell_type_mask shape must be"):
            small_model(**inputs)

    def test_forward_without_any_pseudobulk_raises_error(self, small_model):
        """A-14: forward() with neither region_pseudobulk nor pseudobulk raises ValueError."""
        B = 2
        n_cell_types = N_CELL_TYPES
        n_genes = 50
        max_cells = 10

        n_edges = 5
        src = torch.cat([torch.randint(0, n_cell_types, (n_edges,)) + b * n_cell_types for b in range(B)])
        dst = torch.cat([torch.randint(0, n_cell_types, (n_edges,)) + b * n_cell_types for b in range(B)])
        inputs = {
            # Neither region_pseudobulk nor pseudobulk provided
            'ccc_edge_index': torch.stack([src, dst]),
            'ccc_edge_type': torch.randint(0, 5, (B * n_edges,)),
            'ccc_edge_attr': torch.rand(B * n_edges, 1),
            'cells': torch.randn(B, n_cell_types, max_cells, n_genes),
            'cell_mask': torch.ones(B, n_cell_types, max_cells, dtype=torch.bool),
            'pathology': torch.randn(B, 3),
        }

        with pytest.raises(ValueError, match="Must provide either region_pseudobulk or pseudobulk"):
            small_model(**inputs)

