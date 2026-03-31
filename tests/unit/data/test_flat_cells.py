"""Tests for flat cell storage and loading format.

The flat format replaces the padded 3D cells array [n_types, max_cells, n_genes]
with cell_data [total_real_cells, n_genes] + cell_offsets [n_types+1] (cumulative).
This eliminates ~87% zero padding.
"""
import pytest
import torch
import numpy as np
import pandas as pd

from src.data.constants import CELL_TYPE_ORDER, REGION_ORDER


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_metadata():
    """Metadata for 5 test subjects."""
    return pd.DataFrame({
        "ROSMAP_IndividualID": [f"flat_subj_{i}" for i in range(5)],
        "cogn_global": np.random.randn(5),
    })


@pytest.fixture
def flat_precomputed_dir(tmp_path, mock_metadata):
    """Create a tmp dir with flat-format .pt files for 5 subjects."""
    n_cell_types = len(CELL_TYPE_ORDER)
    n_genes = 10
    n_regions = len(REGION_ORDER)

    rng = np.random.RandomState(42)

    for i in range(5):
        sid = f"flat_subj_{i}"

        # Generate random cell counts per type (0-8 cells each)
        cell_counts_np = rng.randint(0, 9, size=n_cell_types)
        # Ensure at least 2 cell types have cells (for edge validation)
        cell_counts_np[0] = max(cell_counts_np[0], 3)
        cell_counts_np[1] = max(cell_counts_np[1], 3)

        total_cells = int(cell_counts_np.sum())

        # Build flat cell_data and cell_offsets
        cell_offsets = torch.zeros(n_cell_types + 1, dtype=torch.long)
        for ct in range(n_cell_types):
            cell_offsets[ct + 1] = cell_offsets[ct] + int(cell_counts_np[ct])

        cell_data = torch.from_numpy(rng.randn(total_cells, n_genes).astype(np.float32))

        torch.save({
            "pseudobulk": torch.from_numpy(rng.randn(n_cell_types, n_genes).astype(np.float32)),
            "cell_type_mask": torch.tensor(cell_counts_np > 0, dtype=torch.bool),
            "cell_counts": torch.from_numpy(cell_counts_np.astype(np.int64)),
            "region_mask": torch.tensor([True] + [False] * (n_regions - 1), dtype=torch.bool),
            "cell_data": cell_data,
            "cell_offsets": cell_offsets,
            "ccc_edge_index": torch.zeros(2, 0, dtype=torch.long),
            "ccc_edge_type": torch.zeros(0, dtype=torch.long),
            "ccc_edge_attr": torch.zeros(0, 1),
            "cell_type_order": list(CELL_TYPE_ORDER),
            "available_regions": [0],
        }, tmp_path / f"{sid}.pt")
    return tmp_path


# ---------------------------------------------------------------------------
# Task 1: Flat cell storage in save_precomputed_features
# ---------------------------------------------------------------------------

class TestFlatCellStorage:
    """Verify save_precomputed_features writes flat format."""

    def test_flat_cells_keys_present(self, tmp_path):
        """Saved .pt has cell_data and cell_offsets (not cells/cell_mask)."""
        from src.data.datasets import save_precomputed_features

        n_cell_types = len(CELL_TYPE_ORDER)
        n_genes = 8
        max_cells = 4

        # Build flat cell data for mock dataset
        total_cells = n_cell_types * max_cells
        cell_data = torch.randn(total_cells, n_genes)
        cell_offsets = torch.zeros(n_cell_types + 1, dtype=torch.long)
        for ct in range(n_cell_types):
            cell_offsets[ct + 1] = cell_offsets[ct] + max_cells

        mock_sample = {
            "subject_id": "test_subj_0",
            "pseudobulk": torch.randn(n_cell_types, n_genes),
            "cell_type_mask": torch.ones(n_cell_types, dtype=torch.bool),
            "cell_counts": torch.full((n_cell_types,), max_cells, dtype=torch.long),
            "region_mask": torch.tensor(
                [True] + [False] * (len(REGION_ORDER) - 1), dtype=torch.bool
            ),
            "cell_data": cell_data,
            "cell_offsets": cell_offsets,
            "ccc_edge_index": torch.zeros(2, 0, dtype=torch.long),
            "ccc_edge_type": torch.zeros(0, dtype=torch.long),
            "ccc_edge_attr": torch.zeros(0, 1),
        }

        class FakeDataset:
            def __len__(self):
                return 1

            def __getitem__(self, idx):
                return mock_sample

            def get_gene_names(self):
                return None

            cell_type_order = CELL_TYPE_ORDER

        save_precomputed_features(FakeDataset(), tmp_path, verbose=False)
        pt_file = tmp_path / "test_subj_0.pt"
        assert pt_file.exists()

        data = torch.load(pt_file, weights_only=False)
        keys = set(data.keys())
        assert "cell_data" in keys, f"Missing cell_data. Keys: {keys}"
        assert "cell_offsets" in keys, f"Missing cell_offsets. Keys: {keys}"
        assert "cells" not in keys, f"Old 'cells' key should not be present"
        assert "cell_mask" not in keys, f"Old 'cell_mask' key should not be present"

    def test_flat_round_trip_preserves_data(self, tmp_path):
        """Flat data is preserved through save_precomputed_features."""
        from src.data.datasets import save_precomputed_features

        n_cell_types = len(CELL_TYPE_ORDER)
        n_genes = 6

        rng = np.random.RandomState(123)

        # Create flat cell data with varying counts per type
        real_counts = []
        flat_parts = []
        for ct in range(n_cell_types):
            n = rng.randint(0, 6)  # 0 to 5
            real_counts.append(n)
            if n > 0:
                flat_parts.append(rng.randn(n, n_genes).astype(np.float32))

        if flat_parts:
            cell_data_np = np.concatenate(flat_parts, axis=0)
        else:
            cell_data_np = np.empty((0, n_genes), dtype=np.float32)

        cell_offsets_np = np.zeros(n_cell_types + 1, dtype=np.int64)
        for ct in range(n_cell_types):
            cell_offsets_np[ct + 1] = cell_offsets_np[ct] + real_counts[ct]

        mock_sample = {
            "subject_id": "rt_subj",
            "pseudobulk": torch.randn(n_cell_types, n_genes),
            "cell_type_mask": torch.tensor(
                [c > 0 for c in real_counts], dtype=torch.bool
            ),
            "cell_counts": torch.tensor(real_counts, dtype=torch.long),
            "region_mask": torch.tensor(
                [True] + [False] * (len(REGION_ORDER) - 1), dtype=torch.bool
            ),
            "cell_data": torch.from_numpy(cell_data_np),
            "cell_offsets": torch.from_numpy(cell_offsets_np),
            "ccc_edge_index": torch.zeros(2, 0, dtype=torch.long),
            "ccc_edge_type": torch.zeros(0, dtype=torch.long),
            "ccc_edge_attr": torch.zeros(0, 1),
        }

        class FakeDataset:
            def __len__(self):
                return 1

            def __getitem__(self, idx):
                return mock_sample

            def get_gene_names(self):
                return None

            cell_type_order = CELL_TYPE_ORDER

        save_precomputed_features(FakeDataset(), tmp_path, verbose=False)

        data = torch.load(tmp_path / "rt_subj.pt", weights_only=False)
        cell_data = data["cell_data"]
        cell_offsets = data["cell_offsets"]

        # Verify offsets shape and monotonicity
        assert cell_offsets.shape == (n_cell_types + 1,)
        assert cell_offsets[0] == 0
        for ct in range(n_cell_types):
            assert cell_offsets[ct + 1] - cell_offsets[ct] == real_counts[ct]

        # Verify total cell count
        assert cell_data.shape[0] == sum(real_counts)
        assert cell_data.shape[1] == n_genes

        # Verify data matches original
        for ct in range(n_cell_types):
            start = int(cell_offsets[ct])
            end = int(cell_offsets[ct + 1])
            n = real_counts[ct]
            if n > 0:
                torch.testing.assert_close(
                    cell_data[start:end],
                    torch.from_numpy(cell_data_np[cell_offsets_np[ct]:cell_offsets_np[ct + 1]]),
                    msg=f"Mismatch at cell type {ct}",
                )


# ---------------------------------------------------------------------------
# Task 2: Flat cell loading in PrecomputedDataset
# ---------------------------------------------------------------------------

class TestFlatCellLoading:
    """Verify PrecomputedDataset loads flat-format .pt correctly."""

    def test_getitem_returns_flat_tensors(self, flat_precomputed_dir, mock_metadata):
        """Loading flat .pt returns cell_data (2D) and cell_offsets (1D len 32)."""
        from src.data.datasets import PrecomputedDataset

        ds = PrecomputedDataset(
            feature_dir=flat_precomputed_dir,
            subject_ids=[f"flat_subj_{i}" for i in range(5)],
            metadata=mock_metadata,
            subject_column="ROSMAP_IndividualID",
            target_column="cogn_global",
            pathology_columns=[],
            max_missing_subject_fraction=1.0,
        )
        sample = ds[0]

        n_cell_types = len(CELL_TYPE_ORDER)

        # Must have flat keys
        assert "cell_data" in sample, f"Missing cell_data. Keys: {list(sample.keys())}"
        assert "cell_offsets" in sample, f"Missing cell_offsets. Keys: {list(sample.keys())}"

        # Must NOT have old padded keys
        assert "cells" not in sample, "Old 'cells' key should not be present"
        assert "cell_mask" not in sample, "Old 'cell_mask' key should not be present"

        # Shape checks
        assert sample["cell_data"].ndim == 2, f"cell_data should be 2D, got {sample['cell_data'].ndim}D"
        assert sample["cell_offsets"].ndim == 1, f"cell_offsets should be 1D, got {sample['cell_offsets'].ndim}D"
        assert sample["cell_offsets"].shape[0] == n_cell_types + 1
        assert sample["cell_offsets"][-1] == sample["cell_data"].shape[0]

        # dtype checks
        assert sample["cell_data"].dtype == torch.float32
        assert sample["cell_offsets"].dtype == torch.int64

    def test_flat_pt_loads_correctly(self, tmp_path):
        """PrecomputedDataset loads flat .pt format with correct cell data."""
        from src.data.datasets import PrecomputedDataset

        n_cell_types = len(CELL_TYPE_ORDER)
        n_genes = 10
        n_regions = len(REGION_ORDER)

        rng = np.random.RandomState(99)
        metadata = pd.DataFrame({
            "ROSMAP_IndividualID": ["compat_subj_0"],
            "cogn_global": [0.5],
        })

        # Build flat cell data with known counts per type
        # type 0 has 3 cells, type 1 has 2, rest have 5
        counts = [3, 2] + [5] * (n_cell_types - 2)
        total_cells = sum(counts)

        cell_offsets = torch.zeros(n_cell_types + 1, dtype=torch.long)
        for ct in range(n_cell_types):
            cell_offsets[ct + 1] = cell_offsets[ct] + counts[ct]

        cell_data = torch.from_numpy(rng.randn(total_cells, n_genes).astype(np.float32))

        torch.save({
            "pseudobulk": torch.from_numpy(rng.randn(n_cell_types, n_genes).astype(np.float32)),
            "cell_type_mask": torch.ones(n_cell_types, dtype=torch.bool),
            "cell_counts": torch.tensor(counts, dtype=torch.long),
            "region_mask": torch.tensor([True] + [False] * (n_regions - 1), dtype=torch.bool),
            "cell_data": cell_data,
            "cell_offsets": cell_offsets,
            "ccc_edge_index": torch.zeros(2, 0, dtype=torch.long),
            "ccc_edge_type": torch.zeros(0, dtype=torch.long),
            "ccc_edge_attr": torch.zeros(0, 1),
            "cell_type_order": list(CELL_TYPE_ORDER),
            "available_regions": [0],
        }, tmp_path / "compat_subj_0.pt")

        ds = PrecomputedDataset(
            feature_dir=tmp_path,
            subject_ids=["compat_subj_0"],
            metadata=metadata,
            subject_column="ROSMAP_IndividualID",
            target_column="cogn_global",
            pathology_columns=[],
            max_missing_subject_fraction=1.0,
        )
        sample = ds[0]

        # Should return flat format
        assert "cell_data" in sample
        assert "cell_offsets" in sample
        assert sample["cell_data"].ndim == 2
        assert sample["cell_offsets"].shape[0] == n_cell_types + 1

        # Verify data integrity: type 0 should have 3 cells, type 1 should have 2
        offsets = sample["cell_offsets"]
        assert offsets[1] - offsets[0] == 3  # type 0
        assert offsets[2] - offsets[1] == 2  # type 1

        # Check actual values match
        torch.testing.assert_close(
            sample["cell_data"][offsets[0]:offsets[1]],
            cell_data[:3],
        )
        torch.testing.assert_close(
            sample["cell_data"][offsets[1]:offsets[2]],
            cell_data[3:5],
        )

    def test_mmap_loading_flat_format(self, flat_precomputed_dir, mock_metadata):
        """PrecomputedDataset with mmap loading works with flat format."""
        from src.data.datasets import PrecomputedDataset

        ds = PrecomputedDataset(
            feature_dir=flat_precomputed_dir,
            subject_ids=[f"flat_subj_{i}" for i in range(5)],
            metadata=mock_metadata,
            subject_column="ROSMAP_IndividualID",
            target_column="cogn_global",
            pathology_columns=[],
            max_missing_subject_fraction=1.0,
        )
        sample = ds[0]

        n_cell_types = len(CELL_TYPE_ORDER)
        assert "cell_data" in sample
        assert "cell_offsets" in sample
        assert sample["cell_data"].ndim == 2
        assert sample["cell_offsets"].shape[0] == n_cell_types + 1


# ---------------------------------------------------------------------------
# Task 3: Flat collation — _pad_and_stack_cells + flat batch output
# ---------------------------------------------------------------------------

class TestFlatCollation:
    """Verify collation handles flat-format samples."""

    @staticmethod
    def _make_flat_sample(cell_counts_per_type, n_genes=10):
        """Helper to create a flat-format sample dict."""
        n_types = len(cell_counts_per_type)
        total = sum(cell_counts_per_type)
        cell_data = torch.randn(total, n_genes)
        offsets = torch.zeros(n_types + 1, dtype=torch.long)
        for i, c in enumerate(cell_counts_per_type):
            offsets[i + 1] = offsets[i] + c
        ct_mask = torch.tensor([c > 0 for c in cell_counts_per_type])
        return {
            "cell_data": cell_data,
            "cell_offsets": offsets,
            "cell_type_mask": ct_mask,
            "cell_counts": torch.tensor(cell_counts_per_type, dtype=torch.long),
            "pseudobulk": torch.zeros(n_types, n_genes),
            "ccc_edge_index": torch.zeros(2, 0, dtype=torch.long),
            "ccc_edge_type": torch.zeros(0, dtype=torch.long),
            "ccc_edge_attr": torch.zeros(0, 1),
            "pathology": torch.zeros(3),
            "cognition": torch.tensor([1.0]),
            "region_mask": torch.ones(6, dtype=torch.bool),
            "subject_id": "S001",
        }

    def test_collate_flat_outputs_only_flat_keys(self):
        """Flat-format collation outputs cell_data/cell_offsets, NOT cells/cell_mask."""
        from src.data.collate import collate_for_hgt_multiregion

        n_types, n_genes = 31, 10

        counts1 = [5, 3] + [0] * 29
        counts2 = [8, 2] + [0] * 29

        s1 = self._make_flat_sample(counts1, n_genes)
        s2 = self._make_flat_sample(counts2, n_genes)

        batch = collate_for_hgt_multiregion([s1, s2])

        # Flat keys present
        assert "cell_data" in batch
        assert "cell_offsets" in batch
        # Padded keys NOT present (no redundant 9.5 GB tensor)
        assert "cells" not in batch
        assert "cell_mask" not in batch

        # Verify flat data shape and content
        assert batch["cell_data"].shape == (18, n_genes)  # 8 + 10 cells
        assert batch["cell_offsets"].shape == (2, n_types + 1)
        # Sample 0 type 0 data preserved
        torch.testing.assert_close(
            batch["cell_data"][int(batch["cell_offsets"][0, 0]):int(batch["cell_offsets"][0, 1])],
            s1["cell_data"][:5],
        )

    def test_collate_flat_preserves_values(self):
        """Cell values are preserved exactly through flat collation."""
        from src.data.collate import collate_for_hgt_multiregion

        n_types, n_genes = 31, 10
        # Single sample with known values
        cell_data = torch.arange(30, dtype=torch.float32).reshape(3, 10)
        offsets = torch.zeros(n_types + 1, dtype=torch.long)
        offsets[1:] = 3  # type 0 has 3 cells, rest have 0

        sample = {
            "cell_data": cell_data,
            "cell_offsets": offsets,
            "cell_type_mask": torch.tensor([True] + [False] * 30),
            "cell_counts": torch.tensor([3] + [0] * 30, dtype=torch.long),
            "pseudobulk": torch.zeros(n_types, n_genes),
            "ccc_edge_index": torch.zeros(2, 0, dtype=torch.long),
            "ccc_edge_type": torch.zeros(0, dtype=torch.long),
            "ccc_edge_attr": torch.zeros(0, 1),
            "pathology": torch.zeros(3),
            "cognition": torch.tensor([1.0]),
            "region_mask": torch.ones(6, dtype=torch.bool),
            "subject_id": "S001",
        }

        batch = collate_for_hgt_multiregion([sample])
        # Flat output: cell_data[0:3] should match original
        torch.testing.assert_close(batch["cell_data"][:3], cell_data)

    def test_collate_flat_outputs_flat_batch_tensors(self):
        """Collation also outputs cell_data and cell_offsets for forward()."""
        from src.data.collate import collate_for_hgt_multiregion

        n_types, n_genes = 31, 10
        counts1 = [5, 3] + [0] * 29
        counts2 = [8, 2] + [0] * 29

        s1 = self._make_flat_sample(counts1, n_genes)
        s2 = self._make_flat_sample(counts2, n_genes)

        batch = collate_for_hgt_multiregion([s1, s2])

        # Must have flat batch keys
        assert "cell_data" in batch, f"Missing cell_data. Keys: {list(batch.keys())}"
        assert "cell_offsets" in batch, f"Missing cell_offsets. Keys: {list(batch.keys())}"

        # cell_data: concatenated across samples
        assert batch["cell_data"].shape == (8 + 10, n_genes)  # s1 has 8, s2 has 10
        # cell_offsets: [B, n_types + 1]
        assert batch["cell_offsets"].shape == (2, n_types + 1)

        # Sample 0 offsets start at 0
        assert batch["cell_offsets"][0, 0] == 0
        assert batch["cell_offsets"][0, 1] == 5   # type 0: 5 cells
        assert batch["cell_offsets"][0, 2] == 8   # type 1: 3 cells

        # Sample 1 offsets start where sample 0 ended
        total_s1 = 8  # 5+3
        assert batch["cell_offsets"][1, 0] == total_s1
        assert batch["cell_offsets"][1, 1] == total_s1 + 8  # type 0: 8 cells
        assert batch["cell_offsets"][1, 2] == total_s1 + 10  # type 1: 2 cells

        # Verify data consistency: indexing with batch offsets recovers originals
        o = batch["cell_offsets"]
        torch.testing.assert_close(
            batch["cell_data"][o[0, 0]:o[0, 1]],
            s1["cell_data"][:5],
        )
        torch.testing.assert_close(
            batch["cell_data"][o[1, 0]:o[1, 1]],
            s2["cell_data"][:8],
        )


# ---------------------------------------------------------------------------
# Task 6: Flat npz conversion script
# ---------------------------------------------------------------------------

class TestConvertScript:
    """Tests for scripts/convert_to_flat_npz.py conversion logic."""

    def test_convert_padded_to_flat(self, tmp_path):
        """Conversion script converts padded npz to flat format."""
        n_types, n_genes, max_cells = 31, 10, 5
        # Create a padded-format npz
        cells = np.random.randn(n_types, max_cells, n_genes).astype(np.float32)
        cell_mask = np.zeros((n_types, max_cells), dtype=bool)
        cell_mask[0, :3] = True
        cell_mask[5, :2] = True
        cells[~cell_mask] = 0.0

        np.savez_compressed(
            tmp_path / "test_subject.npz",
            pseudobulk=np.zeros((n_types, n_genes), dtype=np.float32),
            cell_type_mask=np.array([True, False] * 15 + [True], dtype=bool),
            cell_counts=cell_mask.sum(axis=1).astype(np.int64),
            region_mask=np.array([True] + [False] * 5, dtype=bool),
            cells=cells,
            cell_mask=cell_mask,
            edge_index=np.zeros((2, 0), dtype=np.int64),
            edge_type=np.zeros(0, dtype=np.int64),
            edge_attr=np.zeros((0, 1), dtype=np.float32),
        )

        # Run conversion
        from scripts.data.convert_to_flat_npz import convert_npz
        from pathlib import Path

        status = convert_npz(
            tmp_path / "test_subject.npz", tmp_path / "test_subject.npz"
        )
        assert status == "converted"

        # Verify converted file
        with np.load(tmp_path / "test_subject.npz", allow_pickle=True) as data:
            assert "cell_data" in data
            assert "cell_offsets" in data
            assert "cells" not in data
            assert "cell_mask" not in data
            assert data["cell_data"].shape == (5, n_genes)  # 3 + 2 cells
            assert data["cell_offsets"][-1] == 5
            # Verify data matches original
            np.testing.assert_array_equal(data["cell_data"][:3], cells[0, :3])
            np.testing.assert_array_equal(data["cell_data"][3:5], cells[5, :2])

    def test_convert_skips_already_flat(self, tmp_path):
        """Conversion script skips files already in flat format."""
        np.savez_compressed(
            tmp_path / "flat.npz",
            cell_data=np.zeros((5, 10), dtype=np.float32),
            cell_offsets=np.arange(32, dtype=np.int64),
        )
        from scripts.data.convert_to_flat_npz import convert_npz

        status = convert_npz(tmp_path / "flat.npz", tmp_path / "flat.npz")
        assert status == "skipped"

    def test_convert_error_on_missing_keys(self, tmp_path):
        """Conversion returns 'error' for npz without cells or cell_data."""
        np.savez_compressed(
            tmp_path / "bad.npz",
            pseudobulk=np.zeros((31, 10), dtype=np.float32),
        )
        from scripts.data.convert_to_flat_npz import convert_npz

        status = convert_npz(tmp_path / "bad.npz", tmp_path / "bad.npz")
        assert status == "error"

    def test_convert_to_output_dir(self, tmp_path):
        """Conversion can write to a separate output directory."""
        src_dir = tmp_path / "src"
        dst_dir = tmp_path / "dst"
        src_dir.mkdir()

        n_types, n_genes, max_cells = 31, 10, 3
        cells = np.random.randn(n_types, max_cells, n_genes).astype(np.float32)
        cell_mask = np.ones((n_types, max_cells), dtype=bool)

        np.savez_compressed(
            src_dir / "subj.npz",
            cells=cells,
            cell_mask=cell_mask,
            pseudobulk=np.zeros((n_types, n_genes), dtype=np.float32),
        )

        from scripts.data.convert_to_flat_npz import convert_npz

        status = convert_npz(src_dir / "subj.npz", dst_dir / "subj.npz")
        assert status == "converted"
        assert (dst_dir / "subj.npz").exists()
        # Source should be unchanged
        with np.load(src_dir / "subj.npz") as data:
            assert "cells" in data  # original still has padded format

    def test_convert_empty_cells(self, tmp_path):
        """Conversion handles a file where all cell types have 0 cells."""
        n_types, n_genes, max_cells = 31, 10, 5
        cells = np.zeros((n_types, max_cells, n_genes), dtype=np.float32)
        cell_mask = np.zeros((n_types, max_cells), dtype=bool)

        np.savez_compressed(
            tmp_path / "empty.npz",
            cells=cells,
            cell_mask=cell_mask,
        )

        from scripts.data.convert_to_flat_npz import convert_npz

        status = convert_npz(tmp_path / "empty.npz", tmp_path / "empty.npz")
        assert status == "converted"

        with np.load(tmp_path / "empty.npz") as data:
            assert data["cell_data"].shape == (0, n_genes)
            assert data["cell_offsets"].shape == (n_types + 1,)
            assert (data["cell_offsets"] == 0).all()
