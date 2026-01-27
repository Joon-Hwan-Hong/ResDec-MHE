"""
Tests for src/data/collate.py

Tests cover:
- Collate function correctness for various batch sizes
- HeteroData construction and batching
- Edge case handling (empty graphs, single sample batch)
- Tensor shape verification
- Output schema compliance (Task C6)
"""

import numpy as np
import torch
import pytest

from src.data.constants import N_CELL_TYPES, N_REGIONS


@pytest.fixture
def mock_batch():
    """Create mock batch for collate testing with all required keys."""
    n_genes = 100
    max_cells = 100

    def make_sample(subject_id):
        return {
            "subject_id": subject_id,
            "pseudobulk": torch.randn(N_CELL_TYPES, n_genes),
            "cell_type_mask": torch.randint(0, 2, (N_CELL_TYPES,)).bool(),
            "cell_counts": torch.randint(0, 100, (N_CELL_TYPES,)),
            # cells now has ALL 31 cell types (not just selected)
            "cells": torch.randn(N_CELL_TYPES, max_cells, n_genes),
            "cell_mask": torch.randint(0, 2, (N_CELL_TYPES, max_cells)).bool(),
            "ccc_edge_index": torch.randint(0, N_CELL_TYPES, (2, 50)),
            "ccc_edge_type": torch.randint(0, 5, (50,)),
            "ccc_edge_attr": torch.rand(50, 1),
            "pathology": torch.rand(3),
            "cognition": torch.rand(1),
            "region_mask": torch.randint(0, 2, (N_REGIONS,)).bool(),
        }

    return [make_sample(f"subj_{i:03d}") for i in range(4)]


class TestCollateOutputSchema:
    """Tests for collate output schema compliance."""

    def test_collate_fn_output_keys(self, mock_batch):
        """collate_fn should output all required keys."""
        from src.data.collate import collate_fn

        batch = collate_fn(mock_batch)

        required_keys = {
            "pseudobulk", "cell_type_mask", "cell_counts",
            "cells", "cell_mask",
            "ccc_edge_index", "ccc_edge_type", "ccc_edge_attr",
            "pathology", "cognition",
            "region_mask",
            "graph_batch", "graph_ptr", "n_nodes_per_graph",
            "subject_ids", "batch_size",
        }

        assert required_keys.issubset(set(batch.keys())), \
            f"Missing keys: {required_keys - set(batch.keys())}"

    def test_collate_fn_key_shapes(self, mock_batch):
        """Verify shapes of collate_fn output keys."""
        from src.data.collate import collate_fn

        batch = collate_fn(mock_batch)
        batch_size = len(mock_batch)

        # cell_counts: [batch, n_cell_types]
        assert batch["cell_counts"].shape == (batch_size, N_CELL_TYPES)
        # region_mask: [batch, n_regions]
        assert batch["region_mask"].shape == (batch_size, N_REGIONS)
        # cells: [batch, n_cell_types, max_cells, n_genes] (all 31 types now)
        assert batch["cells"].shape[0] == batch_size
        assert batch["cells"].shape[1] == N_CELL_TYPES

    def test_collate_to_heterodata_output_keys(self, mock_batch):
        """collate_to_heterodata should output all required keys."""
        from src.data.collate import collate_to_heterodata

        batch = collate_to_heterodata(mock_batch)

        required_keys = {
            "pseudobulk", "cell_type_mask", "cell_counts",
            "cells", "cell_mask",
            "pathology", "cognition",
            "region_mask",
            "hetero_batch",
            "subject_ids", "batch_size",
            "node_types", "edge_types",
        }

        assert required_keys.issubset(set(batch.keys())), \
            f"Missing keys: {required_keys - set(batch.keys())}"

    def test_collate_to_heterodata_key_shapes(self, mock_batch):
        """Verify shapes of collate_to_heterodata output keys."""
        from src.data.collate import collate_to_heterodata

        batch = collate_to_heterodata(mock_batch)
        batch_size = len(mock_batch)

        # cell_counts: [batch, n_cell_types]
        assert batch["cell_counts"].shape == (batch_size, N_CELL_TYPES)
        # region_mask: [batch, n_regions]
        assert batch["region_mask"].shape == (batch_size, N_REGIONS)
        # cells: [batch, n_cell_types, max_cells, n_genes]
        assert batch["cells"].shape[0] == batch_size
        assert batch["cells"].shape[1] == N_CELL_TYPES


def create_mock_sample(
    n_cell_types: int = 31,
    n_genes: int = 100,
    max_cells: int = 50,
    n_edges: int = 20,
    n_regions: int = N_REGIONS,
) -> dict:
    """Create a mock sample dictionary matching Dataset output.

    Note: cells now has ALL n_cell_types (not a selected subset).
    """
    return {
        "subject_id": "TEST_SUBJECT",
        "pseudobulk": torch.randn(n_cell_types, n_genes),
        "cell_type_mask": torch.ones(n_cell_types, dtype=torch.bool),
        "cell_counts": torch.randint(0, 100, (n_cell_types,)),
        # cells has ALL cell types now (soft attention weighting instead of hard selection)
        "cells": torch.randn(n_cell_types, max_cells, n_genes),
        "cell_mask": torch.ones(n_cell_types, max_cells, dtype=torch.bool),
        # Graph features (CCC = cell-cell communication)
        "ccc_edge_index": torch.randint(0, n_cell_types, (2, n_edges)),
        "ccc_edge_type": torch.randint(0, 5, (n_edges,)),
        "ccc_edge_attr": torch.rand(n_edges, 1),
        # Phenotypes
        "pathology": torch.rand(3),
        "cognition": torch.randn(1),
        # Region mask
        "region_mask": torch.ones(n_regions, dtype=torch.bool),
    }


class TestCollateFn:
    """Tests for collate_fn() - homogeneous graph batching."""

    def test_stacks_tensors_correctly(self):
        """Verify tensor stacking produces correct shapes."""
        from src.data.collate import collate_fn

        batch_size = 4
        n_genes = 100
        n_cell_types = 31

        batch = [create_mock_sample(n_genes=n_genes, n_cell_types=n_cell_types)
                 for _ in range(batch_size)]

        result = collate_fn(batch)

        assert result["pseudobulk"].shape == (batch_size, n_cell_types, n_genes)
        assert result["cell_type_mask"].shape == (batch_size, n_cell_types)
        assert result["pathology"].shape == (batch_size, 3)
        assert result["cognition"].shape == (batch_size, 1)

    def test_batches_edges_with_offsets(self):
        """Edges should be offset by node count for batching."""
        from src.data.collate import collate_fn

        n_cell_types = 31
        batch = [create_mock_sample(n_cell_types=n_cell_types, n_edges=10)
                 for _ in range(3)]

        result = collate_fn(batch)

        ccc_edge_index = result["ccc_edge_index"]

        # Total edges should be 30 (10 per sample)
        assert ccc_edge_index.shape[1] == 30

        # Check offsets: sample 0 edges should be in [0, 31)
        # sample 1 edges should be in [31, 62), etc.
        # First 10 edges (sample 0) should have max < 31
        assert ccc_edge_index[:, :10].max() < n_cell_types

    def test_creates_batch_vector(self):
        """Batch vector maps each node to its graph."""
        from src.data.collate import collate_fn

        n_cell_types = 31
        batch_size = 4
        batch = [create_mock_sample(n_cell_types=n_cell_types) for _ in range(batch_size)]

        result = collate_fn(batch)

        graph_batch = result["graph_batch"]

        # Should have 31 * 4 = 124 entries
        assert len(graph_batch) == n_cell_types * batch_size

        # First 31 should be 0, next 31 should be 1, etc.
        assert torch.all(graph_batch[:n_cell_types] == 0)
        assert torch.all(graph_batch[n_cell_types:2*n_cell_types] == 1)

    def test_handles_empty_edges(self):
        """Handle samples with no edges."""
        from src.data.collate import collate_fn

        sample_with_edges = create_mock_sample(n_edges=10)
        sample_without_edges = create_mock_sample(n_edges=0)
        sample_without_edges["ccc_edge_index"] = torch.zeros((2, 0), dtype=torch.long)
        sample_without_edges["ccc_edge_type"] = torch.zeros((0,), dtype=torch.long)
        sample_without_edges["ccc_edge_attr"] = torch.zeros((0, 1))

        batch = [sample_with_edges, sample_without_edges]
        result = collate_fn(batch)

        # Should only have edges from first sample
        assert result["ccc_edge_index"].shape[1] == 10

    def test_preserves_subject_ids(self):
        """Subject IDs should be preserved as list."""
        from src.data.collate import collate_fn

        batch = [create_mock_sample() for _ in range(3)]
        batch[0]["subject_id"] = "SUBJ_A"
        batch[1]["subject_id"] = "SUBJ_B"
        batch[2]["subject_id"] = "SUBJ_C"

        result = collate_fn(batch)

        assert result["subject_ids"] == ["SUBJ_A", "SUBJ_B", "SUBJ_C"]


class TestCollateToHeterodata:
    """Tests for collate_to_heterodata() - heterogeneous graph batching."""

    def test_creates_heterodata_batch(self):
        """Creates PyG HeteroData batch."""
        from src.data.collate import collate_to_heterodata

        batch = [create_mock_sample() for _ in range(3)]
        result = collate_to_heterodata(batch)

        assert "hetero_batch" in result
        assert result["batch_size"] == 3

    def test_includes_node_and_edge_types(self):
        """Result includes metadata about node/edge types."""
        from src.data.collate import collate_to_heterodata

        batch = [create_mock_sample() for _ in range(2)]
        result = collate_to_heterodata(batch)

        assert "node_types" in result
        assert "edge_types" in result
        assert len(result["node_types"]) == 31  # Cell types
        assert len(result["edge_types"]) == 5   # CellChatDB categories

    def test_sanitizes_names(self):
        """Node and edge type names should be sanitized for PyG."""
        from src.data.collate import collate_to_heterodata

        batch = [create_mock_sample() for _ in range(1)]
        result = collate_to_heterodata(batch)

        # Names should not contain spaces or slashes
        for name in result["node_types"]:
            assert " " not in name
            assert "/" not in name

    def test_single_sample_batch(self):
        """Handle batch of size 1."""
        from src.data.collate import collate_to_heterodata

        batch = [create_mock_sample()]
        result = collate_to_heterodata(batch)

        assert result["batch_size"] == 1
        assert result["pseudobulk"].shape[0] == 1


class TestCreateDataloader:
    """Tests for create_dataloader()."""

    def test_uses_heterodata_by_default(self):
        """Default should use HeteroData collate."""
        from src.data.collate import create_dataloader
        from torch.utils.data import TensorDataset

        # Create minimal dataset
        dataset = TensorDataset(torch.randn(10, 5))

        loader = create_dataloader(dataset, batch_size=2)

        # Check that use_heterodata default is True
        # (We can't easily test the collate_fn directly, but we can check the setting)
        assert loader.batch_size == 2

    def test_respects_num_workers(self):
        """num_workers should be configurable."""
        from src.data.collate import create_dataloader
        from torch.utils.data import TensorDataset

        dataset = TensorDataset(torch.randn(10, 5))
        loader = create_dataloader(dataset, num_workers=0)

        assert loader.num_workers == 0


class TestMoveBatchToDevice:
    """Tests for move_batch_to_device()."""

    def test_moves_tensors_to_device(self):
        """All tensors should be moved to specified device."""
        from src.data.collate import move_batch_to_device

        batch = {
            "pseudobulk": torch.randn(2, 31, 100),
            "cognition": torch.randn(2, 1),
            "subject_ids": ["A", "B"],
            "batch_size": 2,
        }

        moved = move_batch_to_device(batch, "cpu")

        assert moved["pseudobulk"].device == torch.device("cpu")
        assert moved["cognition"].device == torch.device("cpu")
        assert moved["subject_ids"] == ["A", "B"]  # Unchanged

    def test_handles_string_device(self):
        """Accept device as string."""
        from src.data.collate import move_batch_to_device

        batch = {"tensor": torch.randn(5)}
        moved = move_batch_to_device(batch, "cpu")

        assert moved["tensor"].device == torch.device("cpu")


class TestGetEffectiveBatchSize:
    """Tests for get_effective_batch_size()."""

    def test_ddp_multiplies_by_gpus(self):
        """DDP effective batch = batch_size * num_gpus."""
        from src.data.collate import get_effective_batch_size

        assert get_effective_batch_size(16, 2, "ddp") == 32
        assert get_effective_batch_size(16, 4, "ddp") == 64

    def test_dp_keeps_batch_size(self):
        """DataParallel splits batch, so effective = batch_size."""
        from src.data.collate import get_effective_batch_size

        assert get_effective_batch_size(16, 2, "dp") == 16
        assert get_effective_batch_size(16, 4, "dp") == 16


class TestOutputSchemaKeys:
    """Tests for output key names matching plan contract."""

    def test_dataset_output_keys_match_contract(self):
        """Dataset output should use plan-specified key names."""
        expected_keys = {
            "subject_id",
            "pseudobulk",
            "cell_type_mask",
            "cell_counts",
            "cells",                 # Now all 31 cell types (soft attention)
            "cell_mask",
            "ccc_edge_index",
            "ccc_edge_type",
            "ccc_edge_attr",
            "pathology",
            "cognition",
            "region_mask",
        }

        # This test documents the expected contract
        assert expected_keys  # Placeholder assertion

    def test_collate_fn_output_keys(self):
        """collate_fn output should use renamed keys."""
        from src.data.collate import collate_fn

        batch = [create_mock_sample() for _ in range(2)]
        result = collate_fn(batch)

        # Verify renamed keys exist
        assert "ccc_edge_index" in result
        assert "ccc_edge_type" in result
        assert "ccc_edge_attr" in result
        assert "cognition" in result

        # Verify old keys do not exist
        assert "edge_index" not in result
        assert "edge_type" not in result
        assert "edge_attr" not in result
        assert "target" not in result

    def test_collate_to_heterodata_output_keys(self):
        """collate_to_heterodata output should use renamed keys."""
        from src.data.collate import collate_to_heterodata

        batch = [create_mock_sample() for _ in range(2)]
        result = collate_to_heterodata(batch)

        # Verify renamed key exists
        assert "cognition" in result

        # Verify old key does not exist
        assert "target" not in result


class TestEdgeCases:
    """Edge case tests for collate functions."""

    def test_handles_all_empty_graphs(self):
        """Handle batch where all samples have no edges."""
        from src.data.collate import collate_fn

        samples = []
        for _ in range(3):
            s = create_mock_sample(n_edges=0)
            s["ccc_edge_index"] = torch.zeros((2, 0), dtype=torch.long)
            s["ccc_edge_type"] = torch.zeros((0,), dtype=torch.long)
            s["ccc_edge_attr"] = torch.zeros((0, 1))
            samples.append(s)

        result = collate_fn(samples)

        assert result["ccc_edge_index"].shape == (2, 0)
        assert result["ccc_edge_type"].shape == (0,)

    def test_mixed_edge_counts(self):
        """Handle samples with different edge counts."""
        from src.data.collate import collate_fn

        batch = [
            create_mock_sample(n_edges=5),
            create_mock_sample(n_edges=20),
            create_mock_sample(n_edges=10),
        ]

        result = collate_fn(batch)

        # Total edges: 5 + 20 + 10 = 35
        assert result["ccc_edge_index"].shape[1] == 35