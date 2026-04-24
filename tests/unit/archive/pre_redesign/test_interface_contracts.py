"""
Interface contract tests to ensure consistent data formats between modules.

These tests verify that:
1. Data pipeline output formats match model input expectations
2. Edge type/attribute tensors have correct shapes across modules
3. Changes to one module don't silently break another module's assumptions

The key interfaces being tested:
- Data pipeline (collate_for_hgt_multiregion) produces raw padded edge tensors
- Model (full_model.py) accepts raw ccc_edge_* tensors
"""

import pytest
import torch
from torch.utils.data import Dataset, DataLoader

from src.data.constants import N_CELL_TYPES, N_EDGE_TYPES, N_REGIONS, ALL_EDGE_TYPES
from src.data.collate import collate_fn, collate_for_hgt, collate_for_hgt_multiregion
from src.models.full_model import CognitiveResilienceModel


# =============================================================================
# Constants
# =============================================================================

N_GENES = 50
MAX_CELLS = 20


# =============================================================================
# Helper Functions
# =============================================================================


def _make_flat_cell_tensors_single(n_cell_types: int, n_cells_per_type: int, n_genes: int):
    """Create flat cell_data and cell_offsets for a single sample (pre-collation)."""
    total_cells = n_cell_types * n_cells_per_type
    cell_data = torch.randn(total_cells, n_genes)
    cell_offsets = torch.arange(0, (n_cell_types + 1) * n_cells_per_type, n_cells_per_type)
    return cell_data, cell_offsets


def _make_flat_cell_tensors_batch(B: int, n_cell_types: int, n_cells_per_type: int, n_genes: int):
    """Create flat cell_data and cell_offsets for a batch (post-collation)."""
    cells_per_sample = n_cell_types * n_cells_per_type
    total_cells = B * cells_per_sample
    cell_data = torch.randn(total_cells, n_genes)
    offsets_one = torch.arange(0, (n_cell_types + 1) * n_cells_per_type, n_cells_per_type)
    cell_offsets = torch.stack([offsets_one + i * cells_per_sample for i in range(B)])
    return cell_data, cell_offsets


def create_mock_sample(n_edges: int = 15) -> dict:
    """Create a mock dataset sample matching CognitiveResilienceDataset output.

    Uses production constants (N_CELL_TYPES, N_REGIONS, N_EDGE_TYPES).
    Divergences: random dense tensors (real data is sparse non-negative expression),
    all masks True (real data has masked types/cells). Update if dataset schema changes.
    """
    cell_data, cell_offsets = _make_flat_cell_tensors_single(N_CELL_TYPES, MAX_CELLS, N_GENES)
    return {
        "subject_id": "TEST_SUBJECT",
        "pseudobulk": torch.randn(N_CELL_TYPES, N_GENES),
        "cell_type_mask": torch.ones(N_CELL_TYPES, dtype=torch.bool),
        "cell_counts": torch.randint(10, 100, (N_CELL_TYPES,)),
        "cell_data": cell_data,
        "cell_offsets": cell_offsets,
        "ccc_edge_index": torch.randint(0, N_CELL_TYPES, (2, n_edges)),
        "ccc_edge_type": torch.randint(0, N_EDGE_TYPES, (n_edges,)),  # Integer indices
        "ccc_edge_attr": torch.rand(n_edges, 1),  # LIANA magnitude [n_edges, 1]
        "pathology": torch.rand(3),
        "cognition": torch.randn(1),
        "region_mask": torch.ones(N_REGIONS, dtype=torch.bool),
    }


def create_mock_multiregion_sample(n_edges: int = 15, n_available_regions: int = 3) -> dict:
    """Create a mock multi-region sample."""
    sample = create_mock_sample(n_edges)

    # Add region-specific pseudobulk
    available_regions = list(range(n_available_regions))
    sample["available_regions"] = available_regions

    for region_idx in available_regions:
        sample[f"region_{region_idx}_pseudobulk"] = torch.randn(N_CELL_TYPES, N_GENES)

    # Update region mask
    sample["region_mask"] = torch.zeros(N_REGIONS, dtype=torch.bool)
    sample["region_mask"][:n_available_regions] = True

    return sample


# =============================================================================
# Interface Contract Tests for collate_for_hgt
# =============================================================================


class TestCollateForHgtFormat:
    """Test that collate_for_hgt produces correct format for HGTEncoderTensor."""

    def test_collate_for_hgt_returns_raw_edge_tensors(self):
        """collate_for_hgt should return flat ccc_edge_* tensors."""
        batch = [create_mock_sample(n_edges=10) for _ in range(4)]
        collated = collate_for_hgt(batch)

        assert "ccc_edge_index" in collated
        assert "ccc_edge_type" in collated
        assert "ccc_edge_attr" in collated
        assert "ccc_edge_counts" not in collated

        B = 4
        E_total = B * 10
        assert collated["ccc_edge_index"].shape == (2, E_total)
        assert collated["ccc_edge_type"].shape == (E_total,)
        assert collated["ccc_edge_attr"].shape == (E_total, 1)

    def test_edge_type_indices_valid(self):
        """Edge type indices should be within [0, N_EDGE_TYPES)."""
        batch = [create_mock_sample(n_edges=20) for _ in range(2)]
        collated = collate_for_hgt(batch)

        edge_type = collated["ccc_edge_type"]
        assert (edge_type >= 0).all()
        assert (edge_type < N_EDGE_TYPES).all()

    def test_edge_attr_has_correct_shape(self):
        """Edge attributes should have shape [E_total, 1] for LIANA magnitude."""
        batch = [create_mock_sample(n_edges=15) for _ in range(2)]
        collated = collate_for_hgt(batch)

        edge_attr = collated["ccc_edge_attr"]
        assert edge_attr.dim() == 2
        assert edge_attr.shape[0] == 2 * 15  # E_total
        assert edge_attr.shape[1] == 1  # LIANA magnitude dimension


class TestCollateForHgtMultiregion:
    """Test collate_for_hgt_multiregion format."""

    def test_multiregion_has_region_pseudobulk(self):
        """collate_for_hgt_multiregion should produce region_pseudobulk tensor."""
        batch = [create_mock_multiregion_sample(n_edges=10, n_available_regions=3) for _ in range(4)]
        collated = collate_for_hgt_multiregion(batch)

        assert "region_pseudobulk" in collated
        assert collated["region_pseudobulk"].shape == (4, N_REGIONS, N_CELL_TYPES, N_GENES)

    def test_multiregion_has_region_mask(self):
        """collate_for_hgt_multiregion should produce region_mask."""
        batch = [create_mock_multiregion_sample(n_edges=10, n_available_regions=2) for _ in range(3)]
        collated = collate_for_hgt_multiregion(batch)

        assert "region_mask" in collated
        assert collated["region_mask"].shape == (3, N_REGIONS)
        assert collated["region_mask"].dtype == torch.bool


# =============================================================================
# Model Input Contract Tests
# =============================================================================


class TestModelInputContract:
    """Test that model forward signature matches expected input format."""

    @pytest.fixture
    def model(self):
        """Create test model."""
        return CognitiveResilienceModel(
            n_genes=N_GENES,
            n_cell_types=N_CELL_TYPES,
            d_embed=32,
            d_fused=32,
            d_cond=16,
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

    def test_model_accepts_single_region_input(self, model):
        """Model should accept single-region input via pseudobulk parameter."""
        B = 2
        cell_data, cell_offsets = _make_flat_cell_tensors_batch(B, N_CELL_TYPES, MAX_CELLS, N_GENES)
        inputs = {
            'pseudobulk': torch.randn(B, N_CELL_TYPES, N_GENES),
            'cell_data': cell_data,
            'cell_offsets': cell_offsets,
            'pathology': torch.randn(B, 3),
        }

        output = model(**inputs)
        assert 'mean' in output
        assert output['mean'].shape == (B, 1)

    def test_model_accepts_multiregion_input(self, model):
        """Model should accept multi-region input via region_pseudobulk parameter."""
        B = 2
        cell_data, cell_offsets = _make_flat_cell_tensors_batch(B, N_CELL_TYPES, MAX_CELLS, N_GENES)
        inputs = {
            'region_pseudobulk': torch.randn(B, N_REGIONS, N_CELL_TYPES, N_GENES),
            'region_mask': torch.ones(B, N_REGIONS, dtype=torch.bool),
            'cell_data': cell_data,
            'cell_offsets': cell_offsets,
            'pathology': torch.randn(B, 3),
        }

        output = model(**inputs)
        assert 'mean' in output
        assert output['mean'].shape == (B, 1)

    def test_model_accepts_edge_tensors(self, model):
        """Model should accept ccc_edge_* flat tensor format."""
        B = 2
        n_edges = 5
        E = B * n_edges

        src = torch.cat([torch.randint(0, N_CELL_TYPES, (n_edges,)) + b * N_CELL_TYPES for b in range(B)])
        dst = torch.cat([torch.randint(0, N_CELL_TYPES, (n_edges,)) + b * N_CELL_TYPES for b in range(B)])
        cell_data, cell_offsets = _make_flat_cell_tensors_batch(B, N_CELL_TYPES, MAX_CELLS, N_GENES)
        inputs = {
            'pseudobulk': torch.randn(B, N_CELL_TYPES, N_GENES),
            'ccc_edge_index': torch.stack([src, dst]),
            'ccc_edge_type': torch.randint(0, N_EDGE_TYPES, (E,)),
            'ccc_edge_attr': torch.rand(E, 1),
            'cell_data': cell_data,
            'cell_offsets': cell_offsets,
            'pathology': torch.randn(B, 3),
        }

        output = model(**inputs)
        assert 'mean' in output

    def test_model_returns_hgt_attention_when_requested(self, model):
        """Model should return HGT attention weights when return_hgt_attention=True."""
        B = 2
        cell_data, cell_offsets = _make_flat_cell_tensors_batch(B, N_CELL_TYPES, MAX_CELLS, N_GENES)
        inputs = {
            'pseudobulk': torch.randn(B, N_CELL_TYPES, N_GENES),
            'cell_data': cell_data,
            'cell_offsets': cell_offsets,
            'pathology': torch.randn(B, 3),
            'return_hgt_attention': True,
        }

        output = model(**inputs)
        assert 'hgt_attention' in output


# =============================================================================
# Collate to Model Integration Tests
# =============================================================================


class TestCollateToModelIntegration:
    """Test that collate output can be directly fed to model."""

    @pytest.fixture
    def model(self):
        """Create test model."""
        return CognitiveResilienceModel(
            n_genes=N_GENES,
            n_cell_types=N_CELL_TYPES,
            d_embed=32,
            d_fused=32,
            d_cond=16,
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

    def test_collate_for_hgt_output_usable_by_model(self, model):
        """collate_for_hgt output should work directly with model."""
        batch = [create_mock_sample(n_edges=15) for _ in range(4)]
        collated = collate_for_hgt(batch)

        # Build model inputs from collated data
        model_input = {
            "pseudobulk": collated["pseudobulk"],
            "ccc_edge_index": collated["ccc_edge_index"],
            "ccc_edge_type": collated["ccc_edge_type"],
            "ccc_edge_attr": collated["ccc_edge_attr"],
            "cell_data": collated["cell_data"],
            "cell_offsets": collated["cell_offsets"],
            "pathology": collated["pathology"],
        }

        output = model(**model_input)
        assert torch.isfinite(output["mean"]).all()

    def test_collate_for_hgt_multiregion_output_usable_by_model(self, model):
        """collate_for_hgt_multiregion output should work directly with model."""
        batch = [create_mock_multiregion_sample(n_edges=15, n_available_regions=3) for _ in range(4)]
        collated = collate_for_hgt_multiregion(batch)

        model_input = {
            "region_pseudobulk": collated["region_pseudobulk"],
            "region_mask": collated["region_mask"],
            "ccc_edge_index": collated["ccc_edge_index"],
            "ccc_edge_type": collated["ccc_edge_type"],
            "ccc_edge_attr": collated["ccc_edge_attr"],
            "cell_data": collated["cell_data"],
            "cell_offsets": collated["cell_offsets"],
            "pathology": collated["pathology"],
        }

        output = model(**model_input)
        assert torch.isfinite(output["mean"]).all()


# =============================================================================
# Empty Edge Contract Tests
# =============================================================================


class TestEmptyEdgeContract:
    """Test that empty edge handling is consistent across modules."""

    @pytest.fixture
    def model(self):
        """Create test model."""
        return CognitiveResilienceModel(
            n_genes=N_GENES,
            n_cell_types=N_CELL_TYPES,
            d_embed=32,
            d_fused=32,
            d_cond=16,
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

    def test_model_handles_no_edges(self, model):
        """Model should handle samples with no edges."""
        B = 2
        cell_data, cell_offsets = _make_flat_cell_tensors_batch(B, N_CELL_TYPES, MAX_CELLS, N_GENES)
        inputs = {
            'pseudobulk': torch.randn(B, N_CELL_TYPES, N_GENES),
            'ccc_edge_index': torch.zeros(2, 0, dtype=torch.long),
            'ccc_edge_type': torch.zeros(0, dtype=torch.long),
            'ccc_edge_attr': torch.zeros(0, 1),
            'cell_data': cell_data,
            'cell_offsets': cell_offsets,
            'pathology': torch.randn(B, 3),
        }

        output = model(**inputs)
        assert torch.isfinite(output["mean"]).all()


# =============================================================================
# Edge Type Category Tests
# =============================================================================


class TestEdgeTypeCategories:
    """Test that edge type categories are used correctly."""

    @pytest.fixture
    def model(self):
        """Create test model."""
        return CognitiveResilienceModel(
            n_genes=N_GENES,
            n_cell_types=N_CELL_TYPES,
            d_embed=32,
            d_fused=32,
            d_cond=16,
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

    def test_all_edge_types_can_be_processed(self, model):
        """Model should handle all 5 edge type categories."""
        B = 2
        n_edge_types = N_EDGE_TYPES
        E = B * n_edge_types

        # Create flat edges with one of each edge type per sample
        src = torch.cat([torch.randint(0, N_CELL_TYPES, (n_edge_types,)) + b * N_CELL_TYPES for b in range(B)])
        dst = torch.cat([torch.randint(0, N_CELL_TYPES, (n_edge_types,)) + b * N_CELL_TYPES for b in range(B)])
        edge_type = torch.arange(n_edge_types).repeat(B)

        cell_data, cell_offsets = _make_flat_cell_tensors_batch(B, N_CELL_TYPES, MAX_CELLS, N_GENES)
        inputs = {
            'pseudobulk': torch.randn(B, N_CELL_TYPES, N_GENES),
            'ccc_edge_index': torch.stack([src, dst]),
            'ccc_edge_type': edge_type,
            'ccc_edge_attr': torch.rand(E, 1),
            'cell_data': cell_data,
            'cell_offsets': cell_offsets,
            'pathology': torch.randn(B, 3),
        }

        output = model(**inputs)
        assert torch.isfinite(output["mean"]).all()

    def test_edge_categories_match_constants(self):
        """Model's edge categories should match constants."""
        model = CognitiveResilienceModel(
            n_genes=N_GENES,
            n_cell_types=N_CELL_TYPES,
            d_embed=32,
            d_fused=32,
            use_bayesian_head=False,
        )

        assert len(model.edge_categories) == len(ALL_EDGE_TYPES)
        assert set(model.edge_categories) == set(ALL_EDGE_TYPES)


# =============================================================================
# Raw Edge Tensor Collate Tests
# =============================================================================


class TestCollateRawEdgeTensors:
    """Verify collate_for_hgt returns raw edge tensors instead of dicts."""

    def test_collate_returns_raw_edge_tensors(self):
        """collate_for_hgt should return flat ccc_edge_* tensors, not dict lists."""
        batch = [create_mock_sample(n_edges=15) for _ in range(4)]
        collated = collate_for_hgt(batch)

        # Flat tensor format
        assert "ccc_edge_index" in collated, "Missing ccc_edge_index"
        assert "ccc_edge_type" in collated, "Missing ccc_edge_type"
        assert "ccc_edge_attr" in collated, "Missing ccc_edge_attr"
        assert "ccc_edge_counts" not in collated

        B = 4
        E_total = B * 15
        assert collated["ccc_edge_index"].shape == (2, E_total)
        assert collated["ccc_edge_type"].shape == (E_total,)
        assert collated["ccc_edge_attr"].shape == (E_total, 1)

        # Old dict format should NOT be present
        assert "edge_index_dict_list" not in collated
        assert "edge_attr_dict_list" not in collated

    def test_collate_flat_edge_count_matches_total(self):
        """Flat edge tensors should have total edges = sum of per-sample edges."""
        batch = [create_mock_sample(n_edges=10), create_mock_sample(n_edges=5)]
        collated = collate_for_hgt(batch)

        E_total = 10 + 5
        assert collated["ccc_edge_index"].shape == (2, E_total)
        assert collated["ccc_edge_type"].shape == (E_total,)
        assert collated["ccc_edge_attr"].shape == (E_total, 1)

    def test_collate_excludes_non_tensor_metadata(self):
        """collate_for_hgt should not include non-tensor metadata strings."""
        batch = [create_mock_sample(n_edges=10) for _ in range(2)]
        collated = collate_for_hgt(batch)
        assert "node_types" not in collated
        assert "edge_types" not in collated
        assert "cell_type_order" not in collated
        assert "cell_barcodes" not in collated


# =============================================================================
# Full Model Raw Edge Tensor Tests
# =============================================================================


class TestFullModelRawEdgeTensors:
    """Test that full model accepts raw edge tensors."""

    @pytest.fixture
    def model(self):
        return CognitiveResilienceModel(
            n_genes=N_GENES,
            n_cell_types=N_CELL_TYPES,
            d_embed=32,
            d_fused=32,
            d_cond=16,
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

    def test_model_accepts_raw_edge_tensors(self, model):
        """Full model forward should work with flat ccc_edge_* tensors."""
        B = 2
        n_edges = 30
        E = B * n_edges

        src = torch.cat([torch.randint(0, N_CELL_TYPES, (n_edges,)) + b * N_CELL_TYPES for b in range(B)])
        dst = torch.cat([torch.randint(0, N_CELL_TYPES, (n_edges,)) + b * N_CELL_TYPES for b in range(B)])
        cell_data, cell_offsets = _make_flat_cell_tensors_batch(B, N_CELL_TYPES, MAX_CELLS, N_GENES)

        with torch.no_grad():
            out = model(
                pseudobulk=torch.randn(B, N_CELL_TYPES, N_GENES),
                ccc_edge_index=torch.stack([src, dst]),
                ccc_edge_type=torch.randint(0, 5, (E,)),
                ccc_edge_attr=torch.rand(E, 1),
                cell_data=cell_data,
                cell_offsets=cell_offsets,
                pathology=torch.randn(B, 3),
            )

        assert "mean" in out
        assert out["mean"].shape == (B, 1)
        assert torch.isfinite(out["mean"]).all()
