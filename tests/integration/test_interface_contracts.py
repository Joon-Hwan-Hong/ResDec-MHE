"""
Interface contract tests to ensure consistent data formats between modules.

These tests verify that:
1. Data pipeline output formats match model input expectations
2. Edge type/attribute tensors have correct shapes across modules
3. Changes to one module don't silently break another module's assumptions

The key interfaces being tested:
- Data pipeline (collate_for_hgt_multiregion) produces per-sample dicts
- Model (full_model.py) expects edge_index_dict_list, edge_attr_dict_list
"""

import pytest
import torch
from torch.utils.data import Dataset, DataLoader

from src.data.constants import N_CELL_TYPES, N_EDGE_TYPES, N_REGIONS, CELL_TYPE_ORDER, ALL_EDGE_TYPES, sanitize_key
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


def create_mock_sample(n_edges: int = 15) -> dict:
    """Create a mock dataset sample matching CognitiveResilienceDataset output."""
    return {
        "subject_id": "TEST_SUBJECT",
        "pseudobulk": torch.randn(N_CELL_TYPES, N_GENES),
        "cell_type_mask": torch.ones(N_CELL_TYPES, dtype=torch.bool),
        "cell_counts": torch.randint(10, 100, (N_CELL_TYPES,)),
        "cells": torch.randn(N_CELL_TYPES, MAX_CELLS, N_GENES),
        "cell_mask": torch.ones(N_CELL_TYPES, MAX_CELLS, dtype=torch.bool),
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
    """Test that collate_for_hgt produces correct format for HGTEncoderBatched."""

    def test_collate_for_hgt_returns_dict_lists(self):
        """collate_for_hgt should return edge_index_dict_list, edge_attr_dict_list."""
        batch = [create_mock_sample(n_edges=10) for _ in range(4)]
        collated = collate_for_hgt(batch)

        assert "edge_index_dict_list" in collated
        assert "edge_attr_dict_list" in collated

        assert isinstance(collated["edge_index_dict_list"], list)
        assert isinstance(collated["edge_attr_dict_list"], list)

        assert len(collated["edge_index_dict_list"]) == 4
        assert len(collated["edge_attr_dict_list"]) == 4

    def test_x_dict_list_has_all_cell_types(self):
        """build_x_dict_list_from_embeddings should produce x_dicts with all cell types."""
        from src.data.collate import build_x_dict_list_from_embeddings

        batch = [create_mock_sample(n_edges=10) for _ in range(2)]
        collated = collate_for_hgt(batch)

        # Build x_dict_list from pseudobulk for standalone HGT testing
        x_dict_list = build_x_dict_list_from_embeddings(
            collated["pseudobulk"], collated["node_types"]
        )

        for x_dict in x_dict_list:
            assert len(x_dict) == N_CELL_TYPES
            for key, tensor in x_dict.items():
                assert tensor.shape == (1, N_GENES)

    def test_edge_index_dict_has_triplet_keys(self):
        """Edge dicts should use (src_type, rel, dst_type) triplet keys."""
        batch = [create_mock_sample(n_edges=20) for _ in range(2)]
        collated = collate_for_hgt(batch)

        for edge_index_dict in collated["edge_index_dict_list"]:
            for key in edge_index_dict:
                assert isinstance(key, tuple)
                assert len(key) == 3
                src, rel, dst = key
                assert isinstance(src, str)
                assert isinstance(rel, str)
                assert isinstance(dst, str)

    def test_edge_attr_has_correct_shape(self):
        """Edge attributes should be [n_edges, 1] for LIANA magnitude."""
        batch = [create_mock_sample(n_edges=15) for _ in range(2)]
        collated = collate_for_hgt(batch)

        for edge_attr_dict in collated["edge_attr_dict_list"]:
            for key, attr in edge_attr_dict.items():
                assert attr.dim() == 2
                assert attr.shape[1] == 1  # LIANA magnitude dimension


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
        inputs = {
            'pseudobulk': torch.randn(B, N_CELL_TYPES, N_GENES),
            'cells': torch.randn(B, N_CELL_TYPES, MAX_CELLS, N_GENES),
            'cell_mask': torch.ones(B, N_CELL_TYPES, MAX_CELLS, dtype=torch.bool),
            'pathology': torch.randn(B, 3),
        }

        output = model(**inputs)
        assert 'mean' in output
        assert output['mean'].shape == (B, 1)

    def test_model_accepts_multiregion_input(self, model):
        """Model should accept multi-region input via region_pseudobulk parameter."""
        B = 2
        inputs = {
            'region_pseudobulk': torch.randn(B, N_REGIONS, N_CELL_TYPES, N_GENES),
            'region_mask': torch.ones(B, N_REGIONS, dtype=torch.bool),
            'cells': torch.randn(B, N_CELL_TYPES, MAX_CELLS, N_GENES),
            'cell_mask': torch.ones(B, N_CELL_TYPES, MAX_CELLS, dtype=torch.bool),
            'pathology': torch.randn(B, 3),
        }

        output = model(**inputs)
        assert 'mean' in output
        assert output['mean'].shape == (B, 1)

    def test_model_accepts_edge_dict_lists(self, model):
        """Model should accept edge_index_dict_list and edge_attr_dict_list."""
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
        assert 'mean' in output

    def test_model_returns_hgt_attention_when_requested(self, model):
        """Model should return HGT attention weights when return_hgt_attention=True."""
        B = 2
        inputs = {
            'pseudobulk': torch.randn(B, N_CELL_TYPES, N_GENES),
            'cells': torch.randn(B, N_CELL_TYPES, MAX_CELLS, N_GENES),
            'cell_mask': torch.ones(B, N_CELL_TYPES, MAX_CELLS, dtype=torch.bool),
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
            "edge_index_dict_list": collated["edge_index_dict_list"],
            "edge_attr_dict_list": collated["edge_attr_dict_list"],
            "cells": collated["cells"],
            "cell_mask": collated["cell_mask"],
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
            "edge_index_dict_list": collated["edge_index_dict_list"],
            "edge_attr_dict_list": collated["edge_attr_dict_list"],
            "cells": collated["cells"],
            "cell_mask": collated["cell_mask"],
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
        """Model should handle samples with no edges (empty edge dicts)."""
        B = 2
        inputs = {
            'pseudobulk': torch.randn(B, N_CELL_TYPES, N_GENES),
            'edge_index_dict_list': [{}, {}],  # Empty dicts
            'edge_attr_dict_list': [{}, {}],
            'cells': torch.randn(B, N_CELL_TYPES, MAX_CELLS, N_GENES),
            'cell_mask': torch.ones(B, N_CELL_TYPES, MAX_CELLS, dtype=torch.bool),
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

        # Create edge dicts with all 5 edge types
        sanitized_types = [sanitize_key(ct) for ct in CELL_TYPE_ORDER]
        sanitized_edges = [sanitize_key(et) for et in ALL_EDGE_TYPES]

        edge_index_dict_list = []
        edge_attr_dict_list = []

        for _ in range(B):
            edge_index = {}
            edge_attr = {}

            for i, edge_type in enumerate(sanitized_edges):
                src_idx = i % len(sanitized_types)
                dst_idx = (i + 1) % len(sanitized_types)
                key = (sanitized_types[src_idx], edge_type, sanitized_types[dst_idx])
                edge_index[key] = torch.tensor([[0], [0]])
                edge_attr[key] = torch.rand(1, 1)

            edge_index_dict_list.append(edge_index)
            edge_attr_dict_list.append(edge_attr)

        inputs = {
            'pseudobulk': torch.randn(B, N_CELL_TYPES, N_GENES),
            'edge_index_dict_list': edge_index_dict_list,
            'edge_attr_dict_list': edge_attr_dict_list,
            'cells': torch.randn(B, N_CELL_TYPES, MAX_CELLS, N_GENES),
            'cell_mask': torch.ones(B, N_CELL_TYPES, MAX_CELLS, dtype=torch.bool),
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
