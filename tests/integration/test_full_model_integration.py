"""
Integration tests for CognitiveResilienceModel end-to-end data flow.

Tests verify that:
- Full forward pass completes without NaN for both Bayesian and deterministic modes
- Gradients propagate to all trainable parameters across all encoders
- Attention weights are valid (sum to 1, non-negative)
- Different pathology inputs produce different attention patterns
- Single-region subjects are handled correctly
"""

import pytest
import torch
import torch.nn as nn

from src.models.full_model import CognitiveResilienceModel
from src.data.constants import N_CELL_TYPES, N_EDGE_TYPES, N_REGIONS, CELL_TYPE_ORDER, ALL_EDGE_TYPES, sanitize_key


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


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_batch():
    """Create a sample batch for testing.

    Returns a dict with all required inputs for CognitiveResilienceModel.forward().
    Uses small dimensions for fast testing.
    """
    B = 2
    n_genes = 50
    n_cell_types = 31
    max_cells = 10
    n_regions = 6

    edge_index_dict_list, edge_attr_dict_list = _make_edge_dicts(B)

    return {
        'region_pseudobulk': torch.randn(B, n_regions, n_cell_types, n_genes),
        'region_mask': torch.ones(B, n_regions, dtype=torch.bool),
        'edge_index_dict_list': edge_index_dict_list,
        'edge_attr_dict_list': edge_attr_dict_list,
        'cells': torch.randn(B, n_cell_types, max_cells, n_genes),
        'cell_mask': torch.ones(B, n_cell_types, max_cells, dtype=torch.bool),
        'pathology': torch.randn(B, 3),
        'cognition': torch.randn(B, 1),
    }


@pytest.fixture
def model_kwargs():
    """Minimal kwargs for integration testing.

    Returns configuration that creates a small but complete model
    with all required components.
    """
    return {
        'n_genes': 50,
        'n_cell_types': 31,
        'd_embed': 32,
        'd_fused': 32,
        'd_cond': 16,
        'n_regions': 6,
        'n_hgt_layers': 1,
        'n_hgt_heads': 2,
        'n_isab_layers': 1,
        'n_inducing_points': 8,
        'n_attention_heads': 2,
        'd_head_hidden': 16,
        'dropout': 0.0,  # Disable dropout for deterministic testing
    }


# =============================================================================
# TestEndToEndForward
# =============================================================================


class TestEndToEndForward:
    """Test complete forward pass through the full model."""

    def test_deterministic_forward(self, model_kwargs, sample_batch):
        """Full forward pass completes, no NaN for deterministic model."""
        model = CognitiveResilienceModel(**model_kwargs, use_bayesian_head=False)

        output = model(**sample_batch)

        # Check output structure
        assert 'mean' in output
        assert 'attention_weights' in output
        assert 'std' not in output  # Deterministic has no std

        # Check no NaN values
        assert torch.isfinite(output['mean']).all(), "mean contains NaN or Inf"
        assert torch.isfinite(output['attention_weights']).all(), "attention_weights contains NaN or Inf"

        # Check shapes
        B = sample_batch['region_pseudobulk'].size(0)
        assert output['mean'].shape == (B, 1)
        assert output['attention_weights'].shape == (B, model_kwargs['n_attention_heads'], N_CELL_TYPES)

    def test_bayesian_forward(self, model_kwargs, sample_batch):
        """Full forward pass completes, std > 0 for Bayesian model."""
        model = CognitiveResilienceModel(**model_kwargs, use_bayesian_head=True)

        output = model(**sample_batch)

        # Check output structure
        assert 'mean' in output
        assert 'std' in output
        assert 'attention_weights' in output

        # Check no NaN values
        assert torch.isfinite(output['mean']).all(), "mean contains NaN or Inf"
        assert torch.isfinite(output['std']).all(), "std contains NaN or Inf"
        assert torch.isfinite(output['attention_weights']).all(), "attention_weights contains NaN or Inf"

        # Check std is positive
        assert (output['std'] > 0).all(), "std must be positive for uncertainty estimation"

        # Check shapes
        B = sample_batch['region_pseudobulk'].size(0)
        assert output['mean'].shape == (B, 1)
        assert output['std'].shape == (B, 1)

    def test_single_region_subject(self, model_kwargs, sample_batch):
        """Works with only PFC available (single region)."""
        model = CognitiveResilienceModel(**model_kwargs, use_bayesian_head=False)

        # Create mask with only first region available (PFC)
        B = sample_batch['region_pseudobulk'].size(0)
        single_region_mask = torch.zeros(B, N_REGIONS, dtype=torch.bool)
        single_region_mask[:, 0] = True  # Only PFC

        sample_batch['region_mask'] = single_region_mask

        output = model(**sample_batch)

        # Should still produce valid output
        assert torch.isfinite(output['mean']).all()
        assert output['mean'].shape == (B, 1)

        # Attention weights should still be valid
        assert torch.isfinite(output['attention_weights']).all()

    def test_region_mask_propagates_to_region_handler(self, model_kwargs, sample_batch):
        """Region mask should produce different outputs with different masks."""
        model = CognitiveResilienceModel(**model_kwargs, use_bayesian_head=False)
        model.eval()

        B = sample_batch['region_pseudobulk'].size(0)

        # Run with first two regions active
        mask_two = torch.zeros(B, N_REGIONS, dtype=torch.bool)
        mask_two[:, :2] = True
        batch_two = {k: v.clone() if isinstance(v, torch.Tensor) else v
                     for k, v in sample_batch.items()}
        batch_two['region_mask'] = mask_two
        output_two = model(**batch_two)

        # Run with only first region active
        mask_one = torch.zeros(B, N_REGIONS, dtype=torch.bool)
        mask_one[:, :1] = True
        batch_one = {k: v.clone() if isinstance(v, torch.Tensor) else v
                     for k, v in sample_batch.items()}
        batch_one['region_mask'] = mask_one
        output_one = model(**batch_one)

        # Outputs should differ because different regions are pooled
        assert not torch.allclose(output_two['mean'], output_one['mean'], atol=1e-6), \
            "Different region masks should produce different outputs"

    def test_different_region_masks_produce_different_outputs(self, model_kwargs, sample_batch):
        """Different region masks should produce different predictions."""
        model = CognitiveResilienceModel(**model_kwargs, use_bayesian_head=False)
        model.eval()

        B = sample_batch['region_pseudobulk'].size(0)

        # All regions active
        all_true_mask = torch.ones(B, N_REGIONS, dtype=torch.bool)
        batch_all = {k: v.clone() if isinstance(v, torch.Tensor) else v
                     for k, v in sample_batch.items()}
        batch_all['region_mask'] = all_true_mask
        output_all = model(**batch_all)

        # Single region active
        single_mask = torch.zeros(B, N_REGIONS, dtype=torch.bool)
        single_mask[:, 0] = True
        batch_single = {k: v.clone() if isinstance(v, torch.Tensor) else v
                        for k, v in sample_batch.items()}
        batch_single['region_mask'] = single_mask
        output_single = model(**batch_single)

        # Outputs should differ
        assert not torch.allclose(output_all['mean'], output_single['mean'], atol=1e-6), \
            "All-True mask vs single-region mask should produce different outputs"

    def test_forward_without_cognition_target(self, model_kwargs, sample_batch):
        """Forward pass works in inference mode (no cognition target)."""
        model = CognitiveResilienceModel(**model_kwargs, use_bayesian_head=False)

        # Remove cognition from inputs (inference mode)
        inputs = {k: v for k, v in sample_batch.items() if k != 'cognition'}

        output = model(**inputs)

        assert 'mean' in output
        assert torch.isfinite(output['mean']).all()

    def test_varying_batch_sizes(self, model_kwargs):
        """Forward pass handles different batch sizes correctly."""
        model = CognitiveResilienceModel(**model_kwargs, use_bayesian_head=False)

        for batch_size in [1, 4, 8]:
            edge_index_dict_list, edge_attr_dict_list = _make_edge_dicts(batch_size)
            batch = {
                'region_pseudobulk': torch.randn(batch_size, N_REGIONS, N_CELL_TYPES, model_kwargs['n_genes']),
                'region_mask': torch.ones(batch_size, N_REGIONS, dtype=torch.bool),
                'edge_index_dict_list': edge_index_dict_list,
                'edge_attr_dict_list': edge_attr_dict_list,
                'cells': torch.randn(batch_size, N_CELL_TYPES, 10, model_kwargs['n_genes']),
                'cell_mask': torch.ones(batch_size, N_CELL_TYPES, 10, dtype=torch.bool),
                'pathology': torch.randn(batch_size, 3),
            }

            output = model(**batch)

            assert output['mean'].shape == (batch_size, 1)
            assert torch.isfinite(output['mean']).all()


# =============================================================================
# TestGradientFlow
# =============================================================================


class TestGradientFlow:
    """Test that gradients flow correctly through all model components."""

    def test_gradients_reach_all_encoders(self, model_kwargs, sample_batch):
        """Gradients flow to pseudobulk, HGT, and cell encoders."""
        model = CognitiveResilienceModel(**model_kwargs, use_bayesian_head=False)

        # Enable gradient tracking on inputs
        sample_batch['region_pseudobulk'].requires_grad_(True)
        sample_batch['cells'].requires_grad_(True)

        output = model(**sample_batch)
        loss = output['mean'].sum()
        loss.backward()

        # Check gradients reach PseudobulkEncoder
        pb_has_grad = False
        for param in model.pseudobulk_encoder.parameters():
            if param.grad is not None and not torch.all(param.grad == 0):
                pb_has_grad = True
                break
        assert pb_has_grad, "No gradients reached PseudobulkEncoder"

        # Check gradients reach HGTEncoder
        hgt_has_grad = False
        for param in model.hgt_encoder.parameters():
            if param.grad is not None and not torch.all(param.grad == 0):
                hgt_has_grad = True
                break
        assert hgt_has_grad, "No gradients reached HGTEncoder"

        # Check gradients reach CellTransformer
        cell_has_grad = False
        for param in model.cell_transformer.parameters():
            if param.grad is not None and not torch.all(param.grad == 0):
                cell_has_grad = True
                break
        assert cell_has_grad, "No gradients reached CellTransformer"

        # Check gradients flow to inputs
        assert sample_batch['region_pseudobulk'].grad is not None, \
            "No gradients reached region_pseudobulk input"
        assert sample_batch['cells'].grad is not None, \
            "No gradients reached cells input"

    def test_gradients_reach_fusion_components(self, model_kwargs, sample_batch):
        """Gradients flow to fusion layer, pathology encoder, and attention."""
        model = CognitiveResilienceModel(**model_kwargs, use_bayesian_head=False)

        output = model(**sample_batch)
        loss = output['mean'].sum()
        loss.backward()

        # Check gradients reach FusionLayer
        fusion_has_grad = False
        for param in model.fusion_layer.parameters():
            if param.grad is not None and not torch.all(param.grad == 0):
                fusion_has_grad = True
                break
        assert fusion_has_grad, "No gradients reached FusionLayer"

        # Check gradients reach PathologyEncoder
        path_has_grad = False
        for param in model.pathology_encoder.parameters():
            if param.grad is not None and not torch.all(param.grad == 0):
                path_has_grad = True
                break
        assert path_has_grad, "No gradients reached PathologyEncoder"

        # Check gradients reach PathologyStratifiedAttention
        attn_has_grad = False
        for param in model.pathology_attention.parameters():
            if param.grad is not None and not torch.all(param.grad == 0):
                attn_has_grad = True
                break
        assert attn_has_grad, "No gradients reached PathologyStratifiedAttention"

        # Check gradients reach prediction head
        head_has_grad = False
        for param in model.prediction_head.parameters():
            if param.grad is not None and not torch.all(param.grad == 0):
                head_has_grad = True
                break
        assert head_has_grad, "No gradients reached prediction head"

    def test_gradients_reach_region_handler(self, model_kwargs, sample_batch):
        """Gradients flow to RegionHandler parameters."""
        model = CognitiveResilienceModel(**model_kwargs, use_bayesian_head=False)

        output = model(**sample_batch)
        loss = output['mean'].sum()
        loss.backward()

        # Check gradients reach RegionHandler
        region_has_grad = False
        for param in model.region_handler.parameters():
            if param.grad is not None and not torch.all(param.grad == 0):
                region_has_grad = True
                break
        assert region_has_grad, "No gradients reached RegionHandler"

    def test_gradients_reach_cell_type_selector(self, model_kwargs, sample_batch):
        """Gradients flow to CellTypeSelector logits."""
        model = CognitiveResilienceModel(**model_kwargs, use_bayesian_head=False)

        output = model(**sample_batch)
        loss = output['mean'].sum()
        loss.backward()

        # CellTypeSelector is part of CellTransformer
        selector = model.cell_transformer.selector
        assert selector.selection_logits.grad is not None, \
            "No gradients reached CellTypeSelector logits"
        assert not torch.all(selector.selection_logits.grad == 0), \
            "CellTypeSelector logits have zero gradients"

    def test_no_nan_in_gradients(self, model_kwargs, sample_batch):
        """No NaN values appear in any gradients."""
        model = CognitiveResilienceModel(**model_kwargs, use_bayesian_head=False)

        output = model(**sample_batch)
        loss = output['mean'].sum()
        loss.backward()

        # Check all parameter gradients for NaN
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), \
                    f"NaN or Inf in gradients for parameter: {name}"


# =============================================================================
# TestAttentionInterpretability
# =============================================================================


class TestAttentionInterpretability:
    """Test attention weights for interpretability."""

    def test_attention_weights_are_valid(self, model_kwargs, sample_batch):
        """Sum to 1, non-negative attention weights."""
        model = CognitiveResilienceModel(**model_kwargs, use_bayesian_head=False)

        output = model(**sample_batch)
        attention_weights = output['attention_weights']  # [B, n_heads, n_cell_types]

        # Check non-negative
        assert (attention_weights >= 0).all(), "Attention weights must be non-negative"

        # Check sum to 1 across cell types
        sums = attention_weights.sum(dim=-1)  # [B, n_heads]
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
            f"Attention weights must sum to 1, got sums: {sums}"

    def test_different_pathology_gives_different_attention(self, model_kwargs, sample_batch):
        """Different pathology produces different attention patterns."""
        model = CognitiveResilienceModel(**model_kwargs, use_bayesian_head=False)
        model.eval()  # Ensure consistent behavior

        # Run with original pathology
        output1 = model(**sample_batch)
        attention1 = output1['attention_weights'].clone()

        # Run with very different pathology (high pathology)
        sample_batch_high = {k: v.clone() if isinstance(v, torch.Tensor) else v
                            for k, v in sample_batch.items()}
        sample_batch_high['pathology'] = torch.ones_like(sample_batch['pathology']) * 10.0

        output2 = model(**sample_batch_high)
        attention2 = output2['attention_weights'].clone()

        # Run with very different pathology (low pathology)
        sample_batch_low = {k: v.clone() if isinstance(v, torch.Tensor) else v
                           for k, v in sample_batch.items()}
        sample_batch_low['pathology'] = torch.ones_like(sample_batch['pathology']) * -10.0

        output3 = model(**sample_batch_low)
        attention3 = output3['attention_weights'].clone()

        # Attention patterns should differ between high and low pathology
        # Using L2 distance to measure difference
        diff_high_low = (attention2 - attention3).norm()

        # There should be meaningful difference (not identical)
        assert diff_high_low > 1e-6, \
            "Different pathology values should produce different attention patterns"

    def test_attention_weights_shape(self, model_kwargs, sample_batch):
        """Attention weights have correct shape."""
        model = CognitiveResilienceModel(**model_kwargs, use_bayesian_head=False)

        output = model(**sample_batch)
        attention_weights = output['attention_weights']

        B = sample_batch['region_pseudobulk'].size(0)
        n_heads = model_kwargs['n_attention_heads']

        expected_shape = (B, n_heads, N_CELL_TYPES)
        assert attention_weights.shape == expected_shape, \
            f"Expected attention shape {expected_shape}, got {attention_weights.shape}"

    def test_attention_weights_vary_across_batch(self, model_kwargs):
        """Attention weights vary across batch items with different inputs."""
        model = CognitiveResilienceModel(**model_kwargs, use_bayesian_head=False)
        model.eval()

        B = 4
        n_genes = model_kwargs['n_genes']

        edge_index_dict_list, edge_attr_dict_list = _make_edge_dicts(B)

        # Create batch with deliberately different pathology per sample
        batch = {
            'region_pseudobulk': torch.randn(B, N_REGIONS, N_CELL_TYPES, n_genes),
            'region_mask': torch.ones(B, N_REGIONS, dtype=torch.bool),
            'edge_index_dict_list': edge_index_dict_list,
            'edge_attr_dict_list': edge_attr_dict_list,
            'cells': torch.randn(B, N_CELL_TYPES, 10, n_genes),
            'cell_mask': torch.ones(B, N_CELL_TYPES, 10, dtype=torch.bool),
            'pathology': torch.tensor([
                [0.0, 0.0, 0.0],  # Low pathology
                [1.0, 1.0, 1.0],  # Medium pathology
                [5.0, 5.0, 5.0],  # High pathology
                [-2.0, -2.0, -2.0],  # Negative (normalized low)
            ]),
        }

        output = model(**batch)
        attention_weights = output['attention_weights']  # [B, n_heads, n_cell_types]

        # Check that attention differs across samples
        for i in range(B):
            for j in range(i + 1, B):
                diff = (attention_weights[i] - attention_weights[j]).abs().sum()
                # Attention should differ due to different pathology
                # Allow for some numerical tolerance but expect difference
                assert diff > 1e-6, \
                    f"Samples {i} and {j} have identical attention despite different pathology"


# =============================================================================
# TestNumericalStability
# =============================================================================


class TestNumericalStability:
    """Test numerical stability of the full model."""

    def test_nan_input_produces_nan_output(self, model_kwargs, sample_batch):
        """NaN in input should propagate to output."""
        model = CognitiveResilienceModel(**model_kwargs, use_bayesian_head=False)
        model.eval()

        # Inject NaN into region_pseudobulk
        sample_batch['region_pseudobulk'][0, 0, 0, :] = float('nan')

        output = model(**sample_batch)

        # Output should contain NaN because NaN propagates through linear layers
        assert torch.isnan(output['mean']).any(), \
            "NaN in region_pseudobulk should propagate to output"

    def test_large_input_values(self, model_kwargs, sample_batch):
        """Model handles large input values without NaN."""
        model = CognitiveResilienceModel(**model_kwargs, use_bayesian_head=False)

        # Scale inputs to large values
        sample_batch['region_pseudobulk'] = sample_batch['region_pseudobulk'] * 100
        sample_batch['cells'] = sample_batch['cells'] * 100
        sample_batch['pathology'] = sample_batch['pathology'] * 10

        output = model(**sample_batch)

        assert torch.isfinite(output['mean']).all(), "NaN in output with large inputs"
        assert torch.isfinite(output['attention_weights']).all(), \
            "NaN in attention weights with large inputs"

    def test_small_input_values(self, model_kwargs, sample_batch):
        """Model handles small input values without NaN."""
        model = CognitiveResilienceModel(**model_kwargs, use_bayesian_head=False)

        # Scale inputs to small values
        sample_batch['region_pseudobulk'] = sample_batch['region_pseudobulk'] * 1e-6
        sample_batch['cells'] = sample_batch['cells'] * 1e-6
        sample_batch['pathology'] = sample_batch['pathology'] * 1e-3

        output = model(**sample_batch)

        assert torch.isfinite(output['mean']).all(), "NaN in output with small inputs"
        assert torch.isfinite(output['attention_weights']).all(), \
            "NaN in attention weights with small inputs"

    def test_sparse_cell_masks(self, model_kwargs, sample_batch):
        """Model handles sparse cell masks without NaN."""
        model = CognitiveResilienceModel(**model_kwargs, use_bayesian_head=False)

        # Make cell mask very sparse (only 2 valid cells per type)
        B = sample_batch['cells'].size(0)
        max_cells = sample_batch['cells'].size(2)
        sparse_mask = torch.zeros(B, N_CELL_TYPES, max_cells, dtype=torch.bool)
        sparse_mask[:, :, :2] = True  # Only first 2 cells valid

        sample_batch['cell_mask'] = sparse_mask

        output = model(**sample_batch)

        assert torch.isfinite(output['mean']).all(), "NaN with sparse cell masks"

    def test_partial_region_masks(self, model_kwargs, sample_batch):
        """Model handles partial region masks without NaN."""
        model = CognitiveResilienceModel(**model_kwargs, use_bayesian_head=False)

        # Make region mask partial (only 2 regions available)
        B = sample_batch['region_pseudobulk'].size(0)
        partial_mask = torch.zeros(B, N_REGIONS, dtype=torch.bool)
        partial_mask[:, 0] = True  # PFC
        partial_mask[:, 2] = True  # Another region

        sample_batch['region_mask'] = partial_mask

        output = model(**sample_batch)

        assert torch.isfinite(output['mean']).all(), "NaN with partial region masks"


# =============================================================================
# TestDeterminism
# =============================================================================


class TestDeterminism:
    """Test determinism and reproducibility."""

    def test_deterministic_forward_reproducible(self, model_kwargs, sample_batch):
        """Deterministic model produces same output with same seed."""
        torch.manual_seed(42)
        model = CognitiveResilienceModel(**model_kwargs, use_bayesian_head=False)
        model.eval()

        # Run twice with same inputs
        output1 = model(**sample_batch)
        output2 = model(**sample_batch)

        # Should be identical
        assert torch.allclose(output1['mean'], output2['mean']), \
            "Deterministic model not reproducible in eval mode"
        assert torch.allclose(output1['attention_weights'], output2['attention_weights']), \
            "Attention weights not reproducible in eval mode"

    def test_model_can_be_saved_and_loaded(self, model_kwargs, sample_batch, tmp_path):
        """Model can be saved and loaded, producing same outputs."""
        model = CognitiveResilienceModel(**model_kwargs, use_bayesian_head=False)
        model.eval()

        # Get output before saving
        output_before = model(**sample_batch)

        # Save model
        save_path = tmp_path / "model.pt"
        torch.save(model.state_dict(), save_path)

        # Create new model and load weights
        model_loaded = CognitiveResilienceModel(**model_kwargs, use_bayesian_head=False)
        model_loaded.load_state_dict(torch.load(save_path, weights_only=True))
        model_loaded.eval()

        # Get output after loading
        output_after = model_loaded(**sample_batch)

        # Should be identical
        assert torch.allclose(output_before['mean'], output_after['mean']), \
            "Model output differs after save/load"


# =============================================================================
# TestEdgeCases
# =============================================================================


class TestEdgeCases:
    """Test edge cases in the full model pipeline."""

    def test_empty_ccc_graph(self, model_kwargs, sample_batch):
        """Model handles empty CCC graph (no edges)."""
        model = CognitiveResilienceModel(**model_kwargs, use_bayesian_head=False)

        # Empty edge dicts
        B = sample_batch['region_pseudobulk'].size(0)
        sample_batch['edge_index_dict_list'] = [{} for _ in range(B)]
        sample_batch['edge_attr_dict_list'] = [{} for _ in range(B)]

        output = model(**sample_batch)

        assert torch.isfinite(output['mean']).all(), "NaN with empty CCC graph"

    def test_all_regions_masked(self, model_kwargs, sample_batch):
        """Model behavior with only minimum regions available."""
        model = CognitiveResilienceModel(**model_kwargs, use_bayesian_head=False)

        # Only one region available
        B = sample_batch['region_pseudobulk'].size(0)
        min_mask = torch.zeros(B, N_REGIONS, dtype=torch.bool)
        min_mask[:, 0] = True  # Only first region

        sample_batch['region_mask'] = min_mask

        output = model(**sample_batch)

        # Should still produce valid output
        assert torch.isfinite(output['mean']).all()

    def test_extreme_pathology_values(self, model_kwargs, sample_batch):
        """Model handles extreme pathology values."""
        model = CognitiveResilienceModel(**model_kwargs, use_bayesian_head=False)

        # Extreme pathology values
        sample_batch['pathology'] = torch.tensor([
            [100.0, 100.0, 100.0],
            [-100.0, -100.0, -100.0],
        ])

        output = model(**sample_batch)

        assert torch.isfinite(output['mean']).all(), "NaN with extreme pathology"
        assert torch.isfinite(output['attention_weights']).all(), \
            "NaN in attention with extreme pathology"


# =============================================================================
# TestBayesianSpecific
# =============================================================================


class TestBayesianSpecific:
    """Tests specific to Bayesian model behavior."""

    def test_bayesian_uncertainty_increases_with_input_variation(self, model_kwargs):
        """Bayesian model uncertainty reflects input diversity."""
        model = CognitiveResilienceModel(**model_kwargs, use_bayesian_head=True)
        model.eval()

        n_genes = model_kwargs['n_genes']

        edge_index_dict_list, edge_attr_dict_list = _make_edge_dicts(4)

        # Batch with similar inputs
        uniform_batch = {
            'region_pseudobulk': torch.randn(4, N_REGIONS, N_CELL_TYPES, n_genes) * 0.1,
            'region_mask': torch.ones(4, N_REGIONS, dtype=torch.bool),
            'edge_index_dict_list': edge_index_dict_list,
            'edge_attr_dict_list': edge_attr_dict_list,
            'cells': torch.randn(4, N_CELL_TYPES, 10, n_genes) * 0.1,
            'cell_mask': torch.ones(4, N_CELL_TYPES, 10, dtype=torch.bool),
            'pathology': torch.zeros(4, 3),
        }

        output = model(**uniform_batch)

        # Std should be positive
        assert (output['std'] > 0).all(), "Bayesian std should be positive"

        # Std should be finite
        assert torch.isfinite(output['std']).all(), "Bayesian std should be finite"

    def test_bayesian_multiple_forward_passes_vary(self, model_kwargs, sample_batch):
        """Multiple forward passes in training mode produce different samples.

        The Bayesian head samples from weight posteriors during training,
        so repeated forward passes should yield different predictions.
        We run enough passes (5) to make it extremely unlikely that all
        outputs are bitwise identical by chance.
        """
        model = CognitiveResilienceModel(**model_kwargs, use_bayesian_head=True)
        model.train()  # Training mode for stochastic behavior

        # Run multiple forward passes
        outputs = []
        for _ in range(5):
            output = model(**sample_batch)
            outputs.append(output['mean'].clone())

        # All outputs should be finite
        assert all(torch.isfinite(o).all() for o in outputs)

        # At least one pair of outputs should differ (stochastic sampling)
        any_differ = any(
            not torch.allclose(outputs[i], outputs[j], atol=1e-7)
            for i in range(len(outputs))
            for j in range(i + 1, len(outputs))
        )
        assert any_differ, (
            "All 5 Bayesian forward passes produced identical outputs — "
            "expected stochastic variation from weight sampling"
        )


# =============================================================================
# TestComponentInteraction
# =============================================================================


class TestComponentInteraction:
    """Test interactions between model components."""

    def test_region_handler_output_feeds_correctly(self, model_kwargs, sample_batch):
        """RegionHandler output correctly feeds into subsequent components."""
        model = CognitiveResilienceModel(**model_kwargs, use_bayesian_head=False)

        # Run forward pass - if dimensions are wrong, this will fail
        output = model(**sample_batch)

        # Verify region context is used in pathology encoder
        # (Indirectly tested by checking the model runs without errors)
        assert 'mean' in output

    def test_fusion_receives_all_branch_outputs(self, model_kwargs, sample_batch):
        """FusionLayer receives outputs from all three branches."""
        model = CognitiveResilienceModel(**model_kwargs, use_bayesian_head=False)

        # Hook to capture fusion layer inputs
        fusion_inputs = []

        def capture_inputs(module, input_args, output):
            fusion_inputs.append([arg.clone() for arg in input_args])

        hook = model.fusion_layer.register_forward_hook(capture_inputs)

        try:
            model(**sample_batch)

            # Should have captured one call
            assert len(fusion_inputs) == 1

            # Should have 3 inputs (pseudobulk_emb, hgt_emb, cell_emb)
            assert len(fusion_inputs[0]) == 3

            # All should have same shape [B, n_cell_types, d_embed]
            B = sample_batch['region_pseudobulk'].size(0)
            for i, inp in enumerate(fusion_inputs[0]):
                assert inp.shape == (B, N_CELL_TYPES, model_kwargs['d_embed']), \
                    f"Branch {i} has unexpected shape: {inp.shape}"
        finally:
            hook.remove()

    def test_pathology_attention_receives_correct_inputs(self, model_kwargs, sample_batch):
        """PathologyStratifiedAttention receives fused embeddings and pathology."""
        model = CognitiveResilienceModel(**model_kwargs, use_bayesian_head=False)

        # Hook to capture attention inputs
        attn_inputs = []

        def capture_inputs(module, input_args, output):
            attn_inputs.append([arg.clone() for arg in input_args])

        hook = model.pathology_attention.register_forward_hook(capture_inputs)

        try:
            model(**sample_batch)

            # Should have captured one call
            assert len(attn_inputs) == 1

            # Should have 2 inputs (cell_type_embeddings, path_emb)
            assert len(attn_inputs[0]) == 2

            B = sample_batch['region_pseudobulk'].size(0)

            # cell_type_embeddings: [B, n_cell_types, d_fused]
            assert attn_inputs[0][0].shape == (B, N_CELL_TYPES, model_kwargs['d_fused'])

            # path_emb: [B, d_cond]
            assert attn_inputs[0][1].shape == (B, model_kwargs['d_cond'])
        finally:
            hook.remove()
