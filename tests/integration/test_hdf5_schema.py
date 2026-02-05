"""
Integration tests for HDF5 schema version 2.0.

Tests that the Phase 6 HDF5 changes work correctly:
1. Schema version consistency
2. Variable-length string encoding
3. Per-sample HGT attention storage
4. Nested attention groups (hgt_attention, pma_attention)
"""

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from src.data.constants import CELL_TYPE_ORDER, ALL_EDGE_TYPES
from src.inference.extract_attention import aggregate_hgt_attention


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def n_samples() -> int:
    return 10


@pytest.fixture
def n_layers() -> int:
    return 3


@pytest.fixture
def n_heads() -> int:
    return 4


@pytest.fixture
def n_edge_types() -> int:
    return len(ALL_EDGE_TYPES)


@pytest.fixture
def cell_type_names() -> list[str]:
    """Cell type names including long ones."""
    return list(CELL_TYPE_ORDER)


@pytest.fixture
def long_cell_type_names() -> list[str]:
    """Cell type names with intentionally long names for truncation testing."""
    return [
        "Committed oligodendrocyte precursor",  # 35 chars
        "This is a very long cell type name for testing truncation issues in HDF5 files",  # 78 chars
        "Short",
        "Another_medium_length_cell_type_name",
    ]


@pytest.fixture
def synthetic_hgt_attention(n_samples, n_layers, n_heads):
    """Generate synthetic HGT attention in the model's output format.

    Returns: list[list[dict]] where each sample has a list of per-layer dicts
             mapping edge_type -> [n_edges, n_heads]
    """
    np.random.seed(42)
    edge_types = [
        ("Microglia", "Secreted_Signaling", "Astrocyte"),
        ("Neuron", "ECM_Receptor", "Oligodendrocyte"),
        ("Astrocyte", "Cell_Cell_Contact", "Microglia"),
    ]

    hgt_attention = []
    for sample_idx in range(n_samples):
        sample_layers = []
        for layer_idx in range(n_layers):
            layer_dict = {}
            for et in edge_types:
                # Variable number of edges per sample (5-15)
                n_edges = np.random.randint(5, 15)
                # Random attention weights
                layer_dict[et] = np.random.rand(n_edges, n_heads).astype(np.float32)
            sample_layers.append(layer_dict)
        hgt_attention.append(sample_layers)

    return hgt_attention


# ─────────────────────────────────────────────────────────────────────────────
# Schema Version Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSchemaVersion:
    """Test HDF5 schema version consistency."""

    def test_aggregate_hgt_attention_returns_per_sample(self, synthetic_hgt_attention):
        """Test that aggregate_hgt_attention returns per-sample summaries."""
        result = aggregate_hgt_attention(synthetic_hgt_attention, include_per_sample=True)

        assert "per_sample" in result
        assert result["per_sample"] is not None

        # Shape should be [n_samples, n_edge_types, n_layers, n_heads]
        per_sample = result["per_sample"]
        assert per_sample.shape[0] == len(synthetic_hgt_attention)  # n_samples
        assert per_sample.shape[2] == len(synthetic_hgt_attention[0])  # n_layers

    def test_aggregate_hgt_attention_without_per_sample(self, synthetic_hgt_attention):
        """Test that include_per_sample=False returns None for per_sample."""
        result = aggregate_hgt_attention(synthetic_hgt_attention, include_per_sample=False)

        assert result["per_sample"] is None

    def test_aggregated_stats_match_per_sample(self, synthetic_hgt_attention):
        """Test that aggregated mean/std match per-sample computation."""
        result = aggregate_hgt_attention(synthetic_hgt_attention, include_per_sample=True)

        per_sample = result["per_sample"]  # [n_samples, n_edge_types, n_layers, n_heads]

        # Mean across layers, then across samples
        attention_per_sample = per_sample.mean(axis=2)  # [n_samples, n_edge_types, n_heads]
        expected_mean = attention_per_sample.mean(axis=0)  # [n_edge_types, n_heads]
        expected_std = attention_per_sample.std(axis=0)

        np.testing.assert_allclose(result["mean_by_edge_type"], expected_mean, rtol=1e-5)
        np.testing.assert_allclose(result["std_by_edge_type"], expected_std, rtol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# Variable-Length String Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestVariableLengthStrings:
    """Test variable-length string encoding preserves long names."""

    def test_long_strings_preserved_in_hdf5(self, long_cell_type_names):
        """Test that long cell type names are preserved without truncation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_strings.h5"

            # Write with variable-length strings
            vlen_str = h5py.special_dtype(vlen=str)
            with h5py.File(path, "w") as f:
                f.create_dataset(
                    "cell_type_names",
                    data=np.array(long_cell_type_names, dtype=object),
                    dtype=vlen_str
                )

            # Read back
            with h5py.File(path, "r") as f:
                loaded = [
                    n.decode() if isinstance(n, bytes) else n
                    for n in f["cell_type_names"][:]
                ]

            # All names should match exactly
            assert loaded == long_cell_type_names

            # Verify the long name is fully preserved
            assert len(loaded[1]) > 64  # Was over 64 chars
            assert loaded[1] == long_cell_type_names[1]

    def test_fixed_length_would_truncate(self, long_cell_type_names):
        """Verify that S64 encoding would truncate long names (documenting why we changed)."""
        # This test documents why we switched from S64 to vlen strings
        encoded = np.array(long_cell_type_names, dtype="S64")

        # The long string should be truncated
        decoded = [s.decode() for s in encoded]

        # Short names preserved
        assert decoded[2] == "Short"

        # Long name truncated to 64 bytes
        assert len(decoded[1]) <= 64
        assert decoded[1] != long_cell_type_names[1]


# ─────────────────────────────────────────────────────────────────────────────
# HGT Attention Storage Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestHGTAttentionStorage:
    """Test HGT attention storage with per-sample summaries."""

    def test_hdf5_roundtrip_with_per_sample(self, synthetic_hgt_attention, n_heads):
        """Test that HGT attention with per-sample data survives HDF5 roundtrip."""
        result = aggregate_hgt_attention(synthetic_hgt_attention, include_per_sample=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_hgt.h5"

            # Write
            vlen_str = h5py.special_dtype(vlen=str)
            with h5py.File(path, "w") as f:
                f.attrs["schema_version"] = "2.0"

                hgt_group = f.create_group("hgt_attention")
                hgt_group.attrs["n_samples"] = result["n_samples"]
                hgt_group.attrs["n_layers"] = result["n_layers"]

                # Edge type names
                hgt_group.create_dataset(
                    "edge_type_names",
                    data=np.array(result["edge_type_names"], dtype=object),
                    dtype=vlen_str
                )

                # Aggregated
                agg_group = hgt_group.create_group("aggregated")
                agg_group.create_dataset("mean_by_edge_type", data=result["mean_by_edge_type"])
                agg_group.create_dataset("std_by_edge_type", data=result["std_by_edge_type"])

                # Per-sample
                if result["per_sample"] is not None:
                    ps_group = hgt_group.create_group("per_sample")
                    ps_group.create_dataset("attention", data=result["per_sample"])

            # Read
            with h5py.File(path, "r") as f:
                assert f.attrs["schema_version"] == "2.0"

                hgt = f["hgt_attention"]
                assert hgt.attrs["n_samples"] == result["n_samples"]
                assert hgt.attrs["n_layers"] == result["n_layers"]

                # Check aggregated
                np.testing.assert_allclose(
                    hgt["aggregated"]["mean_by_edge_type"][:],
                    result["mean_by_edge_type"]
                )

                # Check per-sample
                assert "per_sample" in hgt
                loaded_per_sample = hgt["per_sample"]["attention"][:]
                np.testing.assert_allclose(loaded_per_sample, result["per_sample"])

    def test_per_sample_shape_is_correct(self, synthetic_hgt_attention, n_heads):
        """Test that per_sample has correct shape [n_samples, n_edge_types, n_layers, n_heads]."""
        result = aggregate_hgt_attention(synthetic_hgt_attention, include_per_sample=True)

        per_sample = result["per_sample"]
        n_samples = len(synthetic_hgt_attention)
        n_layers = len(synthetic_hgt_attention[0])
        n_edge_types = len(result["edge_type_names"])

        assert per_sample.shape == (n_samples, n_edge_types, n_layers, n_heads)


# ─────────────────────────────────────────────────────────────────────────────
# Empty/Edge Case Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Test edge cases in HGT attention aggregation."""

    def test_empty_hgt_attention(self):
        """Test handling of empty HGT attention list."""
        result = aggregate_hgt_attention([])

        assert result["edge_type_names"] == []
        assert result["n_samples"] == 0
        assert result["n_layers"] == 0

    def test_single_sample_hgt_attention(self, n_layers, n_heads):
        """Test HGT attention with single sample."""
        edge_type = ("A", "Secreted_Signaling", "B")
        single_sample = [[{edge_type: np.random.rand(5, n_heads)} for _ in range(n_layers)]]

        result = aggregate_hgt_attention(single_sample, include_per_sample=True)

        assert result["n_samples"] == 1
        assert result["per_sample"].shape[0] == 1

    def test_missing_edge_type_in_some_samples(self, n_samples, n_layers, n_heads):
        """Test handling when edge types are missing in some samples."""
        et1 = ("A", "Secreted_Signaling", "B")
        et2 = ("C", "ECM_Receptor", "D")

        hgt_attention = []
        for i in range(n_samples):
            sample_layers = []
            for _ in range(n_layers):
                layer_dict = {et1: np.random.rand(5, n_heads)}
                # Only add et2 in even samples
                if i % 2 == 0:
                    layer_dict[et2] = np.random.rand(3, n_heads)
                sample_layers.append(layer_dict)
            hgt_attention.append(sample_layers)

        result = aggregate_hgt_attention(hgt_attention, include_per_sample=True)

        # Both edge types should be in the result
        assert len(result["edge_type_names"]) == 2

        # Per-sample should have zeros where edge type was missing
        per_sample = result["per_sample"]
        # Odd samples should have zeros for et2
        et2_idx = result["edge_type_names"].index("C|ECM_Receptor|D")
        for i in range(1, n_samples, 2):  # Odd samples
            assert np.allclose(per_sample[i, et2_idx, :, :], 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Save/Load Round-Trip Tests with Unpacking
# ─────────────────────────────────────────────────────────────────────────────


class TestSaveLoadRoundTrip:
    """Test save → load → unpack round-trip for HGT and PMA data."""

    def test_hgt_roundtrip_with_unpack(self, synthetic_hgt_attention, n_heads):
        """Save HGT via io.save_attention_weights, load, then unpack_hgt_for_ccc."""
        from src.utils.io import save_attention_weights, load_attention_weights, unpack_hgt_for_ccc

        hgt_agg = aggregate_hgt_attention(synthetic_hgt_attention, include_per_sample=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "attention_weights.h5"

            save_attention_weights(
                path=path,
                gene_gate=np.random.rand(8, 100).astype(np.float32),
                hgt_attention=hgt_agg,
                subject_ids=[f"S{i}" for i in range(len(synthetic_hgt_attention))],
                cell_type_names=["Ast", "Mic", "Oli", "OPC", "Exc", "Inh", "End", "Per"],
            )

            loaded = load_attention_weights(path)
            assert "hgt_attention" in loaded
            assert isinstance(loaded["hgt_attention"], dict)

            # Unpack for CCC
            scores, metadata_df, names = unpack_hgt_for_ccc(loaded["hgt_attention"])
            assert scores is not None
            assert metadata_df is not None
            assert names is not None
            assert len(names) == len(hgt_agg["edge_type_names"])
            assert "source" in metadata_df.columns
            assert "target" in metadata_df.columns
            assert "edge_type" in metadata_df.columns

            # Scores should be 2D [n_samples, n_edge_types] since per_sample was stored
            assert scores.ndim == 2
            assert scores.shape[0] == len(synthetic_hgt_attention)
            assert scores.shape[1] == len(names)

            # Verify edge name parsing follows PyG convention (src|edge_type|dst)
            # Fixture uses ("Microglia", "Secreted_Signaling", "Astrocyte") etc.
            first_name = names[0]  # e.g. "Microglia|Secreted_Signaling|Astrocyte"
            parts = first_name.split("|")
            first_row = metadata_df.iloc[0]
            assert first_row["source"] == parts[0]      # source = first part
            assert first_row["edge_type"] == parts[1]    # edge_type = middle part
            assert first_row["target"] == parts[2]       # target = last part

    def test_pma_roundtrip_with_unpack(self):
        """Save PMA via io.save_attention_weights, load, then unpack_pma_attention."""
        from src.utils.io import save_attention_weights, load_attention_weights, unpack_pma_attention

        np.random.seed(42)
        n_subjects = 10
        n_heads = 4
        n_seeds = 1
        max_cells = 50
        ct_names = ["Ast", "Mic", "Oli", "OPC"]

        # Create per-cell-type arrays [n_subjects, n_heads, n_seeds, max_cells]
        pma_list = [
            np.random.rand(n_subjects, n_heads, n_seeds, max_cells).astype(np.float32)
            for _ in ct_names
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "attention_weights.h5"

            save_attention_weights(
                path=path,
                pma_attention=pma_list,
                cell_type_names=ct_names,
            )

            loaded = load_attention_weights(path)
            assert "pma_attention" in loaded
            assert isinstance(loaded["pma_attention"], dict)

            # Unpack to 3D
            pma_3d = unpack_pma_attention(loaded["pma_attention"], ct_names)
            assert pma_3d is not None
            assert pma_3d.shape == (n_subjects, len(ct_names), max_cells)

    def test_backward_compat_flat_file(self):
        """Flat HDF5 without nested groups should still load correctly."""
        from src.utils.io import load_attention_weights

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "flat_attention.h5"

            with h5py.File(path, "w") as f:
                f.attrs["schema_version"] = "2.0"
                f.create_dataset("gene_gate", data=np.random.rand(8, 100))
                f.create_dataset("pathology_attention", data=np.random.rand(20, 4, 8))
                f.attrs["cell_type_names"] = ["A", "B", "C"]

            loaded = load_attention_weights(path)
            assert "gene_gate" in loaded
            assert loaded["gene_gate"].shape == (8, 100)
            assert "pathology_attention" in loaded
            assert loaded["pathology_attention"].shape == (20, 4, 8)
            assert "metadata" in loaded

    def test_gene_gate_weights_alias(self):
        """gene_gate_weights key should be aliased to gene_gate."""
        from src.utils.io import load_attention_weights

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "alias_test.h5"

            with h5py.File(path, "w") as f:
                f.create_dataset("gene_gate_weights", data=np.random.rand(8, 100))

            loaded = load_attention_weights(path)
            assert "gene_gate" in loaded
            assert loaded["gene_gate"].shape == (8, 100)

    def test_region_pseudobulk_roundtrip(self):
        """region_pseudobulk survives save/load round-trip."""
        from src.utils.io import save_attention_weights, load_attention_weights

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "region_pb.h5"
            gene_gate = np.random.rand(8, 100).astype(np.float32)
            region_pb = np.random.rand(6, 8, 100).astype(np.float32)

            save_attention_weights(
                path=path,
                gene_gate=gene_gate,
                region_pseudobulk=region_pb,
            )

            loaded = load_attention_weights(path)
            assert "region_pseudobulk" in loaded
            np.testing.assert_array_almost_equal(
                loaded["region_pseudobulk"], region_pb
            )
            assert loaded["region_pseudobulk"].shape == (6, 8, 100)
