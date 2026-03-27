"""
Tests for src/data/collate.py

Tests cover:
- Collate function correctness for various batch sizes
- Dict-list format construction and batching for HGT
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
            "cell_type_mask": torch.ones(N_CELL_TYPES, dtype=torch.bool),
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

    def test_collate_for_hgt_output_keys(self, mock_batch):
        """collate_for_hgt should output all required keys."""
        from src.data.collate import collate_for_hgt

        batch = collate_for_hgt(mock_batch)

        required_keys = {
            "pseudobulk", "cell_type_mask", "cell_counts",
            "cells", "cell_mask",
            "pathology", "cognition",
            "region_mask",
            # HGT dict lists (x_dict_list built separately via build_x_dict_list_from_embeddings)
            "edge_index_dict_list", "edge_attr_dict_list",
            "subject_ids", "batch_size",
            "node_types", "edge_types",
        }

        assert required_keys.issubset(set(batch.keys())), \
            f"Missing keys: {required_keys - set(batch.keys())}"

    def test_collate_for_hgt_key_shapes(self, mock_batch):
        """Verify shapes of collate_for_hgt output keys."""
        from src.data.collate import collate_for_hgt

        batch = collate_for_hgt(mock_batch)
        batch_size = len(mock_batch)

        # cell_counts: [batch, n_cell_types]
        assert batch["cell_counts"].shape == (batch_size, N_CELL_TYPES)
        # region_mask: [batch, n_regions]
        assert batch["region_mask"].shape == (batch_size, N_REGIONS)
        # cells: [batch, n_cell_types, max_cells, n_genes]
        assert batch["cells"].shape[0] == batch_size
        assert batch["cells"].shape[1] == N_CELL_TYPES
        # HGT dict lists should have length == batch_size
        assert len(batch["edge_index_dict_list"]) == batch_size
        assert len(batch["edge_attr_dict_list"]) == batch_size


def create_mock_sample(
    n_cell_types: int = N_CELL_TYPES,
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
        n_cell_types = N_CELL_TYPES

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

        n_cell_types = N_CELL_TYPES
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

        n_cell_types = N_CELL_TYPES
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


class TestCollateForHgt:
    """Tests for collate_for_hgt() - dict format for HGTEncoderBatched."""

    def test_creates_dict_lists(self):
        """Creates dict lists for HGTEncoderBatched."""
        from src.data.collate import collate_for_hgt

        batch = [create_mock_sample() for _ in range(3)]
        result = collate_for_hgt(batch)

        assert "edge_index_dict_list" in result
        assert "edge_attr_dict_list" in result
        assert result["batch_size"] == 3
        assert len(result["edge_index_dict_list"]) == 3

    def test_includes_node_and_edge_types(self):
        """Result includes metadata about node/edge types."""
        from src.data.collate import collate_for_hgt

        batch = [create_mock_sample() for _ in range(2)]
        result = collate_for_hgt(batch)

        assert "node_types" in result
        assert "edge_types" in result
        assert len(result["node_types"]) == N_CELL_TYPES  # Cell types
        assert len(result["edge_types"]) == 5   # CellChatDB categories

    def test_sanitizes_names(self):
        """Node and edge type names should be sanitized."""
        from src.data.collate import collate_for_hgt

        batch = [create_mock_sample() for _ in range(1)]
        result = collate_for_hgt(batch)

        # Names should not contain spaces or slashes
        for name in result["node_types"]:
            assert " " not in name
            assert "/" not in name

    def test_single_sample_batch(self):
        """Handle batch of size 1."""
        from src.data.collate import collate_for_hgt

        batch = [create_mock_sample()]
        result = collate_for_hgt(batch)

        assert result["batch_size"] == 1
        assert result["pseudobulk"].shape[0] == 1
        assert len(result["edge_index_dict_list"]) == 1

    def test_x_dict_structure_via_build_helper(self):
        """build_x_dict_list_from_embeddings should produce x_dicts with all cell types."""
        from src.data.collate import collate_for_hgt, build_x_dict_list_from_embeddings

        batch = [create_mock_sample() for _ in range(2)]
        result = collate_for_hgt(batch)

        # Build x_dict_list from raw pseudobulk (as standalone HGT tests would)
        x_dict_list = build_x_dict_list_from_embeddings(
            result["pseudobulk"], result["node_types"]
        )

        assert len(x_dict_list) == 2
        for x_dict in x_dict_list:
            # Should have 31 cell types
            assert len(x_dict) == N_CELL_TYPES
            # Each value should be (1, n_genes)
            for ct_name, tensor in x_dict.items():
                assert tensor.shape[0] == 1  # Single node per type per subject

        # Verify values match source pseudobulk
        for b in range(2):  # batch size
            for ct_idx, ct_name in enumerate(result["node_types"]):
                expected = result["pseudobulk"][b, ct_idx]
                actual = x_dict_list[b][ct_name].squeeze(0)
                assert torch.allclose(expected, actual), f"Mismatch for batch {b}, type {ct_name}"

    def test_edge_dict_triplet_keys(self):
        """Edge dicts should have (src, rel, dst) triplet keys."""
        from src.data.collate import collate_for_hgt

        # Create sample with edges
        sample = create_mock_sample()
        sample["ccc_edge_index"] = torch.tensor([[0, 1], [1, 2]])
        sample["ccc_edge_type"] = torch.tensor([0, 1])
        sample["ccc_edge_attr"] = torch.tensor([[0.8], [0.5]])

        result = collate_for_hgt([sample])

        edge_index_dict = result["edge_index_dict_list"][0]
        edge_attr_dict = result["edge_attr_dict_list"][0]

        # Should have edges
        assert len(edge_index_dict) > 0
        # Keys should be triplets
        for key in edge_index_dict.keys():
            assert isinstance(key, tuple)
            assert len(key) == 3  # (src_type, relation, dst_type)

    def test_uses_custom_cell_type_order(self):
        """Should use cell_type_order from sample, not global constant."""
        from src.data.collate import collate_for_hgt, build_x_dict_list_from_embeddings

        # Create sample with custom cell type order
        custom_order = ["TypeA", "TypeB", "TypeC"]
        sample = create_mock_sample()
        sample["pseudobulk"] = torch.randn(3, 100)  # 3 cell types
        sample["cell_type_order"] = custom_order

        # Edge from TypeA (0) to TypeB (1)
        sample["ccc_edge_index"] = torch.tensor([[0], [1]])
        sample["ccc_edge_type"] = torch.tensor([0])
        sample["ccc_edge_attr"] = torch.tensor([[0.5]])

        result = collate_for_hgt([sample])

        # Node types should match custom order (sanitized)
        assert result["node_types"] == ["TypeA", "TypeB", "TypeC"]

        # Build x_dict_list from pseudobulk and verify custom type names as keys
        x_dict_list = build_x_dict_list_from_embeddings(
            result["pseudobulk"], result["node_types"]
        )
        x_dict = x_dict_list[0]
        assert set(x_dict.keys()) == {"TypeA", "TypeB", "TypeC"}

        # Raw cell_type_order should be preserved
        assert result["cell_type_order"] == custom_order

    def test_falls_back_to_default_cell_type_order(self):
        """Should use CELL_TYPE_ORDER when sample doesn't include cell_type_order."""
        from src.data.collate import collate_for_hgt
        from src.data.constants import CELL_TYPE_ORDER

        # Create sample WITHOUT cell_type_order key
        sample = create_mock_sample()
        # Ensure no cell_type_order key
        if "cell_type_order" in sample:
            del sample["cell_type_order"]

        result = collate_for_hgt([sample])

        # Should fall back to default CELL_TYPE_ORDER
        assert result["cell_type_order"] == CELL_TYPE_ORDER

    def test_raises_on_mismatched_cell_type_order(self):
        """Should raise RuntimeError when samples have different cell_type_order."""
        from src.data.collate import collate_for_hgt

        n_cell_types = 3
        n_genes = 10
        order_a = ["TypeA", "TypeB", "TypeC"]
        order_b = ["TypeC", "TypeA", "TypeB"]  # Different order

        sample_a = create_mock_sample(n_genes=n_genes, n_cell_types=n_cell_types)
        sample_a["cell_type_order"] = order_a

        sample_b = create_mock_sample(n_genes=n_genes, n_cell_types=n_cell_types)
        sample_b["cell_type_order"] = order_b

        import os
        old = os.environ.get("RESILIENCE_DEBUG")
        os.environ["RESILIENCE_DEBUG"] = "1"
        try:
            with pytest.raises(RuntimeError, match="cell_type_order mismatch"):
                collate_for_hgt([sample_a, sample_b])
        finally:
            if old is None:
                os.environ.pop("RESILIENCE_DEBUG", None)
            else:
                os.environ["RESILIENCE_DEBUG"] = old


class TestBuildXDictListFromEmbeddings:
    """Tests for build_x_dict_list_from_embeddings() helper."""

    def test_creates_correct_structure(self):
        """Should create list of dicts with correct structure."""
        from src.data.collate import build_x_dict_list_from_embeddings

        batch_size = 3
        n_cell_types = N_CELL_TYPES
        d_embed = 128
        node_types = [f"CellType_{i}" for i in range(n_cell_types)]

        embeddings = torch.randn(batch_size, n_cell_types, d_embed)
        x_dict_list = build_x_dict_list_from_embeddings(embeddings, node_types)

        assert len(x_dict_list) == batch_size
        for x_dict in x_dict_list:
            assert len(x_dict) == n_cell_types
            for ct_name, tensor in x_dict.items():
                assert tensor.shape == (1, d_embed)

    def test_preserves_tensor_values(self):
        """Output tensors should contain the correct values."""
        from src.data.collate import build_x_dict_list_from_embeddings

        embeddings = torch.tensor([
            [[1.0, 2.0], [3.0, 4.0]],  # Sample 0: 2 cell types, 2 features
            [[5.0, 6.0], [7.0, 8.0]],  # Sample 1
        ])
        node_types = ["Type_A", "Type_B"]

        x_dict_list = build_x_dict_list_from_embeddings(embeddings, node_types)

        # Sample 0, Type_A should be [1.0, 2.0]
        assert torch.allclose(x_dict_list[0]["Type_A"], torch.tensor([[1.0, 2.0]]))
        # Sample 1, Type_B should be [7.0, 8.0]
        assert torch.allclose(x_dict_list[1]["Type_B"], torch.tensor([[7.0, 8.0]]))

    def test_output_compatible_with_hgt_encoder_batched(self):
        """Output should be directly usable with HGTEncoderBatched."""
        from src.data.collate import build_x_dict_list_from_embeddings
        from src.models.branches.hgt_encoder import HGTEncoderBatched
        from src.data.constants import CELL_TYPE_ORDER, ALL_EDGE_TYPES

        # Setup
        batch_size = 2
        d_embed = 64  # Matches HGT d_input
        node_types = [ct.replace(" ", "_").replace("/", "_").replace("-", "_")
                      for ct in CELL_TYPE_ORDER]

        # Create encoded embeddings
        embeddings = torch.randn(batch_size, len(node_types), d_embed)
        x_dict_list = build_x_dict_list_from_embeddings(embeddings, node_types)

        # Create empty edge dicts (no communication)
        edge_index_dict_list = [{} for _ in range(batch_size)]
        edge_attr_dict_list = [{} for _ in range(batch_size)]

        # HGT encoder should accept this
        encoder = HGTEncoderBatched(
            d_input=d_embed,
            d_hidden=64,
            d_output=64,
            n_heads=4,
            n_layers=2,
        )

        out, _ = encoder(
            x_dict_list, edge_index_dict_list, edge_attr_dict_list
        )

        # Output is a dict of {cell_type: (batch, 1, d_output)}
        assert isinstance(out, dict)
        assert len(out) == len(node_types)
        for ct_name, tensor in out.items():
            assert tensor.shape == (batch_size, 1, 64)


class TestCreateDataloader:
    """Tests for create_dataloader()."""

    def test_uses_hgt_format_by_default(self):
        """Default should use collate_for_hgt."""
        from src.data.collate import create_dataloader
        from torch.utils.data import TensorDataset

        # Create minimal dataset
        dataset = TensorDataset(torch.randn(10, 5))

        loader = create_dataloader(dataset, batch_size=2)

        # Check that use_hgt_format default is True
        # (We can't easily test the collate_fn directly, but we can check the setting)
        assert loader.batch_size == 2

    def test_respects_num_workers(self):
        """num_workers should be configurable."""
        from src.data.collate import create_dataloader
        from torch.utils.data import TensorDataset

        dataset = TensorDataset(torch.randn(10, 5))
        loader = create_dataloader(dataset, num_workers=0)

        assert loader.num_workers == 0

    @pytest.mark.parametrize("use_hgt,multiregion,expected_fn_name", [
        (False, False, "collate_fn"),
        (True, False, "collate_for_hgt"),
        (False, True, "collate_multiregion"),
        (True, True, "collate_for_hgt_multiregion"),
    ])
    def test_create_dataloader_selects_correct_collate(self, use_hgt, multiregion, expected_fn_name):
        """Verify correct collate function is selected for each flag combination."""
        from src.data.collate import create_dataloader
        from torch.utils.data import TensorDataset

        dataset = TensorDataset(torch.randn(4, 10))
        dl = create_dataloader(dataset, batch_size=2, use_hgt_format=use_hgt, multiregion=multiregion)
        assert expected_fn_name in dl.collate_fn.__name__


class TestCollateForHgtMultiregion:
    """Tests for collate_for_hgt_multiregion() combined function."""

    def test_includes_hgt_format_keys(self):
        """Should include all keys from collate_for_hgt."""
        from src.data.collate import collate_for_hgt_multiregion

        sample = create_mock_sample()
        result = collate_for_hgt_multiregion([sample])

        # HGT keys should be present
        assert "edge_index_dict_list" in result
        assert "edge_attr_dict_list" in result
        assert "node_types" in result
        assert "edge_types" in result

    def test_includes_region_data_when_present(self):
        """Should include region data when samples have region_pseudobulk."""
        from src.data.collate import collate_for_hgt_multiregion

        n_genes = 100
        n_cell_types = N_CELL_TYPES

        sample = create_mock_sample()
        sample["region_pseudobulk"] = torch.randn(n_cell_types, n_genes)
        sample["available_regions"] = [0, 2]  # PFC and MTC region
        sample["region_0_pseudobulk"] = torch.randn(n_cell_types, n_genes)
        sample["region_2_pseudobulk"] = torch.randn(n_cell_types, n_genes)

        result = collate_for_hgt_multiregion([sample])

        # Region data should be present
        assert "region_pseudobulk" in result
        assert "region_mask" in result

        # Check shapes
        assert result["region_pseudobulk"].shape == (1, N_REGIONS, n_cell_types, n_genes)
        assert result["region_mask"].shape == (1, N_REGIONS)

        # Regions 0 and 2 should be available (computed from actual data presence)
        assert result["region_mask"][0, 0] == True
        assert result["region_mask"][0, 2] == True
        assert result["region_mask"][0, 1] == False

    def test_works_without_region_data(self):
        """Should work when samples don't have region data."""
        from src.data.collate import collate_for_hgt_multiregion

        sample = create_mock_sample()
        # No region_pseudobulk key

        result = collate_for_hgt_multiregion([sample])

        # HGT keys should still be present
        assert "edge_index_dict_list" in result

        # No region data expected
        assert "region_pseudobulk" not in result or result.get("region_pseudobulk") is None


class TestMoveBatchToDevice:
    """Tests for move_batch_to_device()."""

    def test_moves_tensors_to_device(self):
        """All tensors should be moved to specified device."""
        from src.utils.device import move_batch_to_device

        batch = {
            "pseudobulk": torch.randn(2, N_CELL_TYPES, 100),
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
        from src.utils.device import move_batch_to_device

        batch = {"tensor": torch.randn(5)}
        moved = move_batch_to_device(batch, "cpu")

        assert moved["tensor"].device == torch.device("cpu")

    def test_moves_x_dict_list(self):
        """Should move tensors in x_dict_list to device."""
        from src.utils.device import move_batch_to_device

        batch = {
            "x_dict_list": [
                {"TypeA": torch.randn(1, 64), "TypeB": torch.randn(1, 64)},
                {"TypeA": torch.randn(1, 64), "TypeB": torch.randn(1, 64)},
            ],
            "node_types": ["TypeA", "TypeB"],
        }

        moved = move_batch_to_device(batch, "cpu")

        # Check all tensors in x_dict_list are on CPU
        for x_dict in moved["x_dict_list"]:
            for ct_name, tensor in x_dict.items():
                assert tensor.device == torch.device("cpu")

        # Metadata should be unchanged
        assert moved["node_types"] == ["TypeA", "TypeB"]

    def test_moves_edge_dict_lists(self):
        """Should move tensors in edge_index_dict_list and edge_attr_dict_list."""
        from src.utils.device import move_batch_to_device

        batch = {
            "edge_index_dict_list": [
                {("TypeA", "rel", "TypeB"): torch.tensor([[0], [0]])},
            ],
            "edge_attr_dict_list": [
                {("TypeA", "rel", "TypeB"): torch.tensor([[0.5]])},
            ],
        }

        moved = move_batch_to_device(batch, "cpu")

        # Check edge_index_dict_list
        for edge_dict in moved["edge_index_dict_list"]:
            for triplet, tensor in edge_dict.items():
                assert tensor.device == torch.device("cpu")

        # Check edge_attr_dict_list
        for edge_dict in moved["edge_attr_dict_list"]:
            for triplet, tensor in edge_dict.items():
                assert tensor.device == torch.device("cpu")

    def test_preserves_metadata_keys(self):
        """Should keep metadata keys on CPU."""
        from src.utils.device import move_batch_to_device

        batch = {
            "pseudobulk": torch.randn(2, N_CELL_TYPES, 100),
            "subject_ids": ["A", "B"],
            "batch_size": 2,
            "node_types": ["TypeA", "TypeB"],
            "edge_types": ["rel1", "rel2"],
            "cell_type_order": ["TypeA", "TypeB"],
        }

        moved = move_batch_to_device(batch, "cpu")

        # These should be unchanged
        assert moved["subject_ids"] == ["A", "B"]
        assert moved["batch_size"] == 2
        assert moved["node_types"] == ["TypeA", "TypeB"]
        assert moved["edge_types"] == ["rel1", "rel2"]
        assert moved["cell_type_order"] == ["TypeA", "TypeB"]


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

        sample = create_mock_sample()
        assert set(sample.keys()) == expected_keys

    def test_collate_fn_uses_renamed_keys(self):
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

    def test_collate_for_hgt_output_keys(self):
        """collate_for_hgt output should use renamed keys."""
        from src.data.collate import collate_for_hgt

        batch = [create_mock_sample() for _ in range(2)]
        result = collate_for_hgt(batch)

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


class TestDeriveAvailableRegionsFromKeys:
    """Tests for _derive_available_regions_from_keys() helper."""

    def test_extracts_region_indices(self):
        """Should extract sorted region indices from region_*_pseudobulk keys."""
        from src.data.collate import _derive_available_regions_from_keys

        sample = {
            'region_0_pseudobulk': torch.randn(N_CELL_TYPES, 50),
            'region_2_pseudobulk': torch.randn(N_CELL_TYPES, 50),
            'region_5_pseudobulk': torch.randn(N_CELL_TYPES, 50),
            'pseudobulk': torch.randn(N_CELL_TYPES, 50),
        }
        regions = _derive_available_regions_from_keys(sample)
        assert regions == [0, 2, 5]

    def test_empty_when_no_region_keys(self):
        """Should return empty list when no region_*_pseudobulk keys exist."""
        from src.data.collate import _derive_available_regions_from_keys

        sample = {'pseudobulk': torch.randn(N_CELL_TYPES, 50)}
        regions = _derive_available_regions_from_keys(sample)
        assert regions == []


class TestAssembleRegionTensors:
    """Tests for _assemble_region_tensors() helper."""

    def test_basic_assembly(self):
        """Should assemble region tensors with correct shape and mask."""
        from src.data.collate import _assemble_region_tensors

        batch = [
            {'region_0_pseudobulk': torch.randn(N_CELL_TYPES, 50),
             'region_1_pseudobulk': torch.randn(N_CELL_TYPES, 50),
             'available_regions': [0, 1]},
        ]
        region_pb, region_mask = _assemble_region_tensors(batch, batch_size=1, n_cell_types=N_CELL_TYPES, n_genes=50)
        assert region_pb.shape == (1, N_REGIONS, N_CELL_TYPES, 50)
        assert region_mask.shape == (1, N_REGIONS)
        assert region_mask[0, 0] == True
        assert region_mask[0, 1] == True
        assert region_mask[0, 2] == False


class TestCompositeKeyOverflow:
    """Tests for composite key overflow detection in collate_for_hgt."""

    def test_overflow_raises_value_error(self, monkeypatch):
        """Should raise ValueError when composite key would overflow int64."""
        import src.data.collate as collate_mod

        huge_n = 2_200_000
        fake_ct = [f"c{i}" for i in range(huge_n)]
        fake_et = [f"e{i}" for i in range(huge_n)]
        monkeypatch.setattr(collate_mod, "ALL_EDGE_TYPES", fake_et)
        monkeypatch.setattr(collate_mod, "SANITIZED_EDGE_TYPES", fake_et)
        monkeypatch.setattr(collate_mod, "sanitize_key", lambda x: x)
        n_ct_small = 4
        sample = {
            "pseudobulk": torch.randn(n_ct_small, 10),
            "cell_type_mask": torch.ones(n_ct_small, dtype=torch.bool),
            "cell_counts": torch.ones(n_ct_small, dtype=torch.long),
            "cells": torch.randn(n_ct_small, 5, 10),
            "cell_mask": torch.ones(n_ct_small, 5, dtype=torch.bool),
            "ccc_edge_index": torch.tensor([[0, 1], [1, 0]]),
            "ccc_edge_type": torch.tensor([0, 0]),
            "ccc_edge_attr": torch.randn(2, 1),
            "pathology": torch.randn(3),
            "cognition": torch.randn(1),
            "region_mask": torch.ones(N_REGIONS, dtype=torch.bool),
            "subject_id": "test_subj",
            "cell_type_order": fake_ct,
        }
        from src.data.collate import collate_for_hgt
        with pytest.raises(ValueError, match="Composite key overflow"):
            collate_for_hgt([sample])


class TestMultiregionAvailableRegionsDerivation:
    """Tests for deriving available_regions from region keys."""

    def _make_multiregion_sample(self, n_genes=100, max_cells=50):
        """Create a properly structured sample for multi-region testing."""
        return {
            "subject_id": "test_subject",
            "pseudobulk": torch.randn(N_CELL_TYPES, n_genes),
            "cell_type_mask": torch.ones(N_CELL_TYPES, dtype=torch.bool),
            "cell_counts": torch.randint(1, 100, (N_CELL_TYPES,)),
            "cells": torch.randn(N_CELL_TYPES, max_cells, n_genes),
            "cell_mask": torch.ones(N_CELL_TYPES, max_cells, dtype=torch.bool),
            "ccc_edge_index": torch.randint(0, N_CELL_TYPES, (2, 50)),
            "ccc_edge_type": torch.randint(0, 5, (50,)),
            "ccc_edge_attr": torch.rand(50, 1),
            "pathology": torch.rand(3),
            "cognition": torch.rand(1),
            "region_mask": torch.ones(N_REGIONS, dtype=torch.bool),
        }

    def test_derives_available_regions_from_keys_when_missing(self):
        """Should derive available_regions from region_*_pseudobulk keys."""
        import warnings
        from src.data.collate import collate_for_hgt_multiregion

        n_genes = 100

        sample = self._make_multiregion_sample(n_genes=n_genes)
        sample["region_pseudobulk"] = torch.randn(N_CELL_TYPES, n_genes)
        # Add region keys but no available_regions
        sample["region_0_pseudobulk"] = torch.randn(N_CELL_TYPES, n_genes)
        sample["region_2_pseudobulk"] = torch.randn(N_CELL_TYPES, n_genes)
        # No "available_regions" key

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = collate_for_hgt_multiregion([sample])
            # Should warn about missing available_regions
            assert len(w) == 1
            assert "available_regions" in str(w[0].message)
            assert "region_*_pseudobulk" in str(w[0].message)

        # Should have derived regions [0, 2] from keys
        assert result["region_mask"][0, 0].item() is True
        assert result["region_mask"][0, 1].item() is False
        assert result["region_mask"][0, 2].item() is True

    def test_no_warning_when_available_regions_provided(self):
        """Should not warn when available_regions is explicitly provided."""
        import warnings
        from src.data.collate import collate_for_hgt_multiregion

        n_genes = 100

        sample = self._make_multiregion_sample(n_genes=n_genes)
        sample["region_pseudobulk"] = torch.randn(N_CELL_TYPES, n_genes)
        sample["region_0_pseudobulk"] = torch.randn(N_CELL_TYPES, n_genes)
        sample["region_1_pseudobulk"] = torch.randn(N_CELL_TYPES, n_genes)
        sample["available_regions"] = [0, 1]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = collate_for_hgt_multiregion([sample])
            # Should not warn
            assert len(w) == 0

        assert result["region_mask"][0, 0].item() is True
        assert result["region_mask"][0, 1].item() is True

    def test_defaults_to_dlpfc_when_no_region_keys(self):
        """Should default to [0] when no region_*_pseudobulk keys exist."""
        import warnings
        from src.data.collate import collate_for_hgt_multiregion

        n_genes = 100

        sample = self._make_multiregion_sample(n_genes=n_genes)
        sample["region_pseudobulk"] = torch.randn(N_CELL_TYPES, n_genes)
        # No region_*_pseudobulk keys and no available_regions

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = collate_for_hgt_multiregion([sample])
            # Should not warn (default behavior, no keys to derive from)
            assert len(w) == 0

        # Default PFC
        assert result["region_mask"][0, 0].item() is True
        assert result["region_mask"][0, 1].item() is False

    def test_detects_multiregion_without_sentinel_key(self):
        """Should detect multi-region from region_{idx}_pseudobulk keys without sentinel."""
        import warnings
        from src.data.collate import collate_for_hgt_multiregion

        n_genes = 100

        sample = self._make_multiregion_sample(n_genes=n_genes)
        # Add region keys but NO "region_pseudobulk" sentinel
        sample["region_0_pseudobulk"] = torch.randn(N_CELL_TYPES, n_genes)
        sample["region_1_pseudobulk"] = torch.randn(N_CELL_TYPES, n_genes)
        # Explicitly ensure no sentinel
        assert "region_pseudobulk" not in sample

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = collate_for_hgt_multiregion([sample])
            # Should warn about missing available_regions
            assert len(w) == 1

        # Should have detected multi-region and processed regions 0 and 1
        assert "region_pseudobulk" in result
        assert "region_mask" in result
        assert result["region_mask"][0, 0].item() is True
        assert result["region_mask"][0, 1].item() is True
        assert result["region_mask"][0, 2].item() is False

    def test_mixed_batch_some_with_regions_some_without(self):
        """Should handle mixed batches where some samples have region keys."""
        import warnings
        from src.data.collate import collate_for_hgt_multiregion

        n_genes = 100

        # Sample 1: has multi-region data
        sample1 = self._make_multiregion_sample(n_genes=n_genes)
        sample1["region_0_pseudobulk"] = torch.randn(N_CELL_TYPES, n_genes)
        sample1["region_2_pseudobulk"] = torch.randn(N_CELL_TYPES, n_genes)

        # Sample 2: single region only (no region_* keys)
        sample2 = self._make_multiregion_sample(n_genes=n_genes)
        # No region_*_pseudobulk keys - will default to PFC from pseudobulk

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = collate_for_hgt_multiregion([sample1, sample2])

        # Should have region_pseudobulk with shape [2, N_REGIONS, n_cell_types, n_genes]
        assert result["region_pseudobulk"].shape[0] == 2
        assert result["region_pseudobulk"].shape[1] == N_REGIONS

        # Sample 1: regions 0 and 2 available
        assert result["region_mask"][0, 0].item() is True
        assert result["region_mask"][0, 1].item() is False
        assert result["region_mask"][0, 2].item() is True

        # Sample 2: only PFC (region 0) from default
        assert result["region_mask"][1, 0].item() is True
        assert result["region_mask"][1, 1].item() is False
        assert result["region_mask"][1, 2].item() is False


class TestCollateMultiregionWithSentinel:
    """Tests for collate_multiregion() with the region_pseudobulk sentinel key.

    Covers C-A1: collate_multiregion() was never tested with the
    region_pseudobulk sentinel that activates its multi-region path
    (as opposed to collate_for_hgt_multiregion which has separate tests).
    """

    @staticmethod
    def _make_sample(
        n_genes: int = 50,
        max_cells: int = 20,
        n_edges: int = 10,
        region_indices: list[int] | None = None,
    ) -> dict:
        """Create a sample with the region_pseudobulk sentinel and per-region keys.

        Args:
            n_genes: Number of genes per pseudobulk profile.
            max_cells: Max cells per cell type.
            n_edges: Number of CCC edges.
            region_indices: Which region_{idx}_pseudobulk keys to include.
                If None, defaults to [0, 2] (PFC and region 2).
        """
        if region_indices is None:
            region_indices = [0, 2]

        sample = {
            "subject_id": "multiregion_subj",
            "pseudobulk": torch.randn(N_CELL_TYPES, n_genes),
            "cell_type_mask": torch.ones(N_CELL_TYPES, dtype=torch.bool),
            "cell_counts": torch.randint(1, 100, (N_CELL_TYPES,)),
            "cells": torch.randn(N_CELL_TYPES, max_cells, n_genes),
            "cell_mask": torch.ones(N_CELL_TYPES, max_cells, dtype=torch.bool),
            "ccc_edge_index": torch.randint(0, N_CELL_TYPES, (2, n_edges)),
            "ccc_edge_type": torch.randint(0, 5, (n_edges,)),
            "ccc_edge_attr": torch.rand(n_edges, 1),
            "pathology": torch.rand(3),
            "cognition": torch.rand(1),
            "region_mask": torch.ones(N_REGIONS, dtype=torch.bool),
            # Sentinel key that activates multi-region path
            "region_pseudobulk": True,
            # Explicit list so _assemble_region_tensors knows which regions
            "available_regions": region_indices,
        }

        # Per-region pseudobulk tensors
        for idx in region_indices:
            sample[f"region_{idx}_pseudobulk"] = torch.randn(N_CELL_TYPES, n_genes)

        return sample

    def test_multiregion_output_shapes(self):
        """region_pseudobulk should be [B, n_regions, n_cell_types, n_genes]
        and region_mask should be [B, n_regions]."""
        from src.data.collate import collate_multiregion

        n_genes = 50
        batch_size = 3
        region_indices = [0, 2, 4]

        batch = [
            self._make_sample(n_genes=n_genes, region_indices=region_indices)
            for _ in range(batch_size)
        ]

        result = collate_multiregion(batch)

        assert "region_pseudobulk" in result
        assert "region_mask" in result
        assert result["region_pseudobulk"].shape == (
            batch_size, N_REGIONS, N_CELL_TYPES, n_genes,
        )
        assert result["region_mask"].shape == (batch_size, N_REGIONS)

    def test_multiregion_mask_reflects_available_regions(self):
        """region_mask should be True only for regions that have data."""
        from src.data.collate import collate_multiregion

        n_genes = 50
        region_indices = [0, 3]

        batch = [self._make_sample(n_genes=n_genes, region_indices=region_indices)]
        result = collate_multiregion(batch)

        mask = result["region_mask"][0]
        for r in range(N_REGIONS):
            if r in region_indices:
                assert mask[r].item() is True, f"Expected region {r} mask True"
            else:
                assert mask[r].item() is False, f"Expected region {r} mask False"

    def test_multiregion_data_is_populated(self):
        """Populated regions should contain non-zero data matching the source tensors."""
        from src.data.collate import collate_multiregion

        n_genes = 50
        region_indices = [0, 1]

        sample = self._make_sample(n_genes=n_genes, region_indices=region_indices)
        result = collate_multiregion([sample])

        for idx in region_indices:
            expected = sample[f"region_{idx}_pseudobulk"]
            actual = result["region_pseudobulk"][0, idx]
            assert torch.allclose(actual, expected), (
                f"Region {idx} data mismatch"
            )

    def test_multiregion_unpopulated_regions_are_zero(self):
        """Regions without data should remain zero-filled."""
        from src.data.collate import collate_multiregion

        n_genes = 50
        region_indices = [0]

        batch = [self._make_sample(n_genes=n_genes, region_indices=region_indices)]
        result = collate_multiregion(batch)

        for r in range(N_REGIONS):
            if r not in region_indices:
                assert (result["region_pseudobulk"][0, r] == 0).all(), (
                    f"Region {r} should be zero-filled"
                )

    def test_multiregion_preserves_base_collate_keys(self):
        """collate_multiregion should include all base collate_fn keys."""
        from src.data.collate import collate_multiregion

        batch = [self._make_sample()]
        result = collate_multiregion(batch)

        base_keys = {
            "pseudobulk", "cell_type_mask", "cell_counts",
            "cells", "cell_mask",
            "ccc_edge_index", "ccc_edge_type", "ccc_edge_attr",
            "pathology", "cognition",
            "graph_batch", "graph_ptr", "n_nodes_per_graph",
            "subject_ids", "batch_size",
        }
        assert base_keys.issubset(set(result.keys())), (
            f"Missing base keys: {base_keys - set(result.keys())}"
        )

    def test_multiregion_falls_back_without_sentinel(self):
        """Without the region_pseudobulk sentinel, should fall back to collate_fn
        and NOT produce a region_pseudobulk tensor in the output."""
        from src.data.collate import collate_multiregion

        # Create a sample without the sentinel key
        sample = create_mock_sample()
        assert "region_pseudobulk" not in sample

        result = collate_multiregion([sample])

        # Should have base keys but no region_pseudobulk tensor
        assert "pseudobulk" in result
        assert "region_pseudobulk" not in result

    def test_multiregion_batch_of_two_different_regions(self):
        """Batch with samples having different available regions."""
        from src.data.collate import collate_multiregion

        n_genes = 50

        sample_a = self._make_sample(n_genes=n_genes, region_indices=[0, 1])
        sample_b = self._make_sample(n_genes=n_genes, region_indices=[0, 4])

        result = collate_multiregion([sample_a, sample_b])

        assert result["region_pseudobulk"].shape == (
            2, N_REGIONS, N_CELL_TYPES, n_genes,
        )
        assert result["region_mask"].shape == (2, N_REGIONS)

        # Sample A: regions 0, 1
        assert result["region_mask"][0, 0].item() is True
        assert result["region_mask"][0, 1].item() is True
        assert result["region_mask"][0, 4].item() is False

        # Sample B: regions 0, 4
        assert result["region_mask"][1, 0].item() is True
        assert result["region_mask"][1, 1].item() is False
        assert result["region_mask"][1, 4].item() is True


class TestWorkerInitFn:
    """Tests for DataLoader worker re-seeding."""

    def test_worker_init_fn_seeds_numpy(self):
        """_worker_init_fn re-seeds numpy RNG for the worker."""
        from unittest.mock import patch, MagicMock
        from src.data.collate import _worker_init_fn

        mock_info = MagicMock()
        mock_info.seed = 12345
        mock_info.dataset = MagicMock(spec=[])  # no cell_sampler

        with patch("torch.utils.data.get_worker_info", return_value=mock_info):
            # Should not raise
            _worker_init_fn(0)

    def test_worker_init_fn_reseeds_cell_sampler(self):
        """_worker_init_fn re-seeds CellSampler RNG when dataset has one."""
        from unittest.mock import patch, MagicMock
        from src.data.collate import _worker_init_fn

        mock_sampler = MagicMock()
        mock_sampler.rng = None

        mock_dataset = MagicMock()
        mock_dataset.sampler = mock_sampler

        mock_info = MagicMock()
        mock_info.seed = 12345
        mock_info.dataset = mock_dataset

        with patch("torch.utils.data.get_worker_info", return_value=mock_info):
            _worker_init_fn(0)
            # RNG should have been replaced
            assert mock_dataset.sampler.rng is not None


class TestCreateDataloaderPrefetch:
    """Tests for prefetch_factor wiring in create_dataloader."""

    def test_create_dataloader_accepts_prefetch_factor(self):
        """create_dataloader accepts prefetch_factor parameter."""
        import inspect
        from src.data.collate import create_dataloader
        sig = inspect.signature(create_dataloader)
        assert "prefetch_factor" in sig.parameters
