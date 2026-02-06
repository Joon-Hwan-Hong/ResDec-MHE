"""Tests for CellHeterogeneityAnalyzer class."""

import numpy as np
import pandas as pd
import pytest
import h5py

from src.analysis.cell_heterogeneity import (
    CellHeterogeneityAnalyzer,
    CellHeterogeneityResult,
    compute_cell_heterogeneity,
    analyze_cell_heterogeneity,  # backward compat
)


@pytest.fixture
def sample_data():
    """Create sample PMA attention data."""
    rng = np.random.default_rng(42)
    n_subjects, n_cell_types, n_cells = 5, 3, 20
    pma = rng.random((n_subjects, n_cell_types, n_cells))
    # Zero out some cells (padding)
    pma[:, :, 15:] = 0.0
    return {
        "pma_attention": pma,
        "cell_type_names": ["Excitatory", "Inhibitory", "Astrocyte"],
        "subject_ids": [f"subj_{i}" for i in range(n_subjects)],
    }


class TestCellHeterogeneityResult:
    """Test the result dataclass."""

    def test_result_has_expected_fields(self):
        result = CellHeterogeneityResult(
            summary=pd.DataFrame({"cell_type": ["A"]}),
            high_attention_cells=pd.DataFrame(),
            all_scores=pd.DataFrame(),
        )
        assert isinstance(result.summary, pd.DataFrame)
        assert isinstance(result.high_attention_cells, pd.DataFrame)
        assert isinstance(result.all_scores, pd.DataFrame)
        assert isinstance(result.metadata, dict)


class TestCellHeterogeneityAnalyzer:
    """Test the analyzer class."""

    def test_analyze_returns_result(self, sample_data):
        analyzer = CellHeterogeneityAnalyzer(
            pma_attention=sample_data["pma_attention"],
            cell_type_names=sample_data["cell_type_names"],
            subject_ids=sample_data["subject_ids"],
        )
        result = analyzer.analyze()
        assert isinstance(result, CellHeterogeneityResult)
        assert len(result.summary) > 0
        assert "gini_coefficient" in result.summary.columns
        assert "attention_entropy" in result.summary.columns

    def test_save_creates_h5_with_vlen_strings(self, sample_data, tmp_path):
        analyzer = CellHeterogeneityAnalyzer(
            pma_attention=sample_data["pma_attention"],
            cell_type_names=sample_data["cell_type_names"],
            subject_ids=sample_data["subject_ids"],
        )
        result = analyzer.analyze()
        analyzer.save(result, tmp_path)

        # Verify HDF5 file
        h5_path = tmp_path / "cell_attention.h5"
        assert h5_path.exists()
        with h5py.File(h5_path, "r") as f:
            assert f.attrs["schema_version"] == "2.0"
            assert "pma_attention" in f
            assert "cell_type_names" in f
            assert "subject_ids" in f
            # Verify shape attributes
            assert f.attrs["n_subjects"] == 5
            assert f.attrs["n_cell_types"] == 3
            assert f.attrs["max_cells"] == 20
            # Verify vlen strings (not fixed-length S64)
            ds = f["cell_type_names"]
            assert ds.dtype.kind != "S", f"Expected vlen string, got fixed-length {ds.dtype}"
            # Decode values (h5py may return bytes or str depending on version)
            ct_names = [
                x.decode("utf-8") if isinstance(x, bytes) else str(x)
                for x in ds[:]
            ]
            assert ct_names == ["Excitatory", "Inhibitory", "Astrocyte"]

    def test_save_creates_dataframe_files(self, sample_data, tmp_path):
        analyzer = CellHeterogeneityAnalyzer(
            pma_attention=sample_data["pma_attention"],
            cell_type_names=sample_data["cell_type_names"],
        )
        result = analyzer.analyze()
        saved = analyzer.save(result, tmp_path, formats=["parquet", "csv"])
        assert (tmp_path / "cell_attention_summary.parquet").exists()
        assert (tmp_path / "cell_attention_summary.csv").exists()
        assert (tmp_path / "high_attention_cells.parquet").exists()
        assert (tmp_path / "cell_attention_scores.parquet").exists()
        assert isinstance(saved, dict)

    def test_save_with_no_subject_ids(self, sample_data, tmp_path):
        """Save should work when subject_ids not provided."""
        analyzer = CellHeterogeneityAnalyzer(
            pma_attention=sample_data["pma_attention"],
            cell_type_names=sample_data["cell_type_names"],
        )
        result = analyzer.analyze()
        analyzer.save(result, tmp_path)

        h5_path = tmp_path / "cell_attention.h5"
        with h5py.File(h5_path, "r") as f:
            assert "subject_ids" in f  # auto-generated subject IDs

    def test_min_cells_filter(self):
        """Cell types with too few valid cells should be skipped."""
        pma = np.zeros((2, 2, 10))
        pma[0, 0, 0:3] = 0.5  # Only 3 valid cells for type 0
        pma[:, 1, :] = 0.1    # Full cells for type 1

        analyzer = CellHeterogeneityAnalyzer(
            pma_attention=pma,
            cell_type_names=["sparse_type", "full_type"],
            min_cells_per_type=10,
        )
        result = analyzer.analyze()
        # sparse_type should be excluded (only 3 cells < 10 threshold)
        assert "sparse_type" not in result.summary["cell_type"].values
        assert "full_type" in result.summary["cell_type"].values


class TestBackwardCompat:
    """Test backward compatibility with function API."""

    def test_analyze_cell_heterogeneity_returns_tuple(self, sample_data):
        summary, high, scores = analyze_cell_heterogeneity(
            pma_attention=sample_data["pma_attention"],
            cell_type_names=sample_data["cell_type_names"],
        )
        assert isinstance(summary, pd.DataFrame)
        assert isinstance(high, pd.DataFrame)
        assert isinstance(scores, pd.DataFrame)

    def test_backward_compat_with_cell_metadata(self, sample_data):
        """Function API should still accept cell_metadata."""
        metadata = pd.DataFrame(
            {"some_col": [1]},
            index=["barcode_0"],
        )
        summary, high, scores = analyze_cell_heterogeneity(
            pma_attention=sample_data["pma_attention"],
            cell_type_names=sample_data["cell_type_names"],
            cell_metadata=metadata,
        )
        assert isinstance(summary, pd.DataFrame)


class TestComputeConvenience:
    """Test the convenience function."""

    def test_compute_with_output_dir(self, tmp_path):
        rng = np.random.default_rng(42)
        pma = rng.random((5, 3, 20))
        result = compute_cell_heterogeneity(
            pma_attention=pma,
            cell_type_names=["A", "B", "C"],
            output_dir=tmp_path,
        )
        assert isinstance(result, CellHeterogeneityResult)
        assert (tmp_path / "cell_attention.h5").exists()
        assert (tmp_path / "cell_attention_summary.parquet").exists()

    def test_compute_without_output_dir(self):
        rng = np.random.default_rng(42)
        pma = rng.random((5, 3, 20))
        result = compute_cell_heterogeneity(
            pma_attention=pma,
            cell_type_names=["A", "B", "C"],
        )
        assert isinstance(result, CellHeterogeneityResult)
        assert len(result.summary) == 3
