"""
Tests for src/inference/extract_attention.py.

Test coverage includes:
- AttentionWeights dataclass behavior
- AttentionExtractor static weight extraction
- DataFrame conversion methods
- HDF5 serialization round-trip
- Schema validation for output formats
- Edge cases (empty arrays, single subject)
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st
from unittest.mock import MagicMock, patch

from src.data.constants import CELL_TYPE_ORDER, N_CELL_TYPES
from src.inference.extract_attention import (
    AttentionWeights,
    AttentionExtractor,
    save_attention_weights_hdf5,
    load_attention_weights_hdf5,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_gene_gate():
    """Sample gene gate weights [n_cell_types, n_genes]."""
    np.random.seed(42)
    return np.random.rand(N_CELL_TYPES, 100).astype(np.float32)


@pytest.fixture
def sample_pathology_attention():
    """Sample pathology attention [n_subjects, n_heads, n_cell_types]."""
    np.random.seed(42)
    return np.random.rand(10, 4, N_CELL_TYPES).astype(np.float32)


@pytest.fixture
def sample_attention_weights(sample_gene_gate, sample_pathology_attention):
    """Complete AttentionWeights instance."""
    return AttentionWeights(
        gene_gate=sample_gene_gate,
        pathology_attention=sample_pathology_attention,
        cell_type_selection=np.random.rand(N_CELL_TYPES).astype(np.float32),
        region_weights=np.random.rand(6).astype(np.float32),
        hgt_layer_scales={"layer_0": np.array([1.0, 0.9, 0.8])},
        subject_ids=[f"subj_{i}" for i in range(10)],
        cell_type_names=list(CELL_TYPE_ORDER),
    )


@pytest.fixture
def mock_model():
    """Mock CognitiveResilienceModel for testing AttentionExtractor."""
    model = MagicMock()
    model.n_cell_types = N_CELL_TYPES
    model.n_genes = 100

    # Mock gene gate weights
    import torch
    model.pseudobulk_encoder.gene_gate.get_gate_weights.return_value = torch.rand(N_CELL_TYPES, 100)

    # Mock cell type selection weights
    model.cell_transformer.get_selection_weights.return_value = torch.rand(N_CELL_TYPES)

    # Mock region importance
    model.get_region_importance.return_value = {
        "PFC": 0.3, "AG": 0.2, "MTC": 0.15, "EC": 0.15, "HC": 0.1, "TH": 0.1
    }

    # Mock HGT layer scales
    model.get_hgt_layer_scales.return_value = {
        "layer_0": torch.tensor([1.0, 0.9, 0.8]),
        "layer_1": torch.tensor([0.8, 0.7, 0.6]),
    }

    return model


# ============================================================================
# AttentionWeights Dataclass Tests
# ============================================================================


class TestAttentionWeightsDataclass:
    """Tests for AttentionWeights dataclass."""

    def test_init_minimal(self, sample_gene_gate):
        """AttentionWeights can be initialized with only gene_gate."""
        weights = AttentionWeights(gene_gate=sample_gene_gate)
        assert weights.gene_gate.shape == (N_CELL_TYPES, 100)
        assert weights.pathology_attention is None
        assert weights.cell_type_selection is None

    def test_post_init_sets_default_cell_types(self, sample_gene_gate):
        """__post_init__ sets cell_type_names from CELL_TYPE_ORDER if None."""
        weights = AttentionWeights(gene_gate=sample_gene_gate)
        assert weights.cell_type_names == list(CELL_TYPE_ORDER)

    def test_post_init_preserves_custom_cell_types(self, sample_gene_gate):
        """__post_init__ preserves custom cell_type_names if provided."""
        custom_names = ["Type_A", "Type_B"]
        weights = AttentionWeights(
            gene_gate=sample_gene_gate,
            cell_type_names=custom_names,
        )
        assert weights.cell_type_names == custom_names

    def test_all_fields_accessible(self, sample_attention_weights):
        """All fields are accessible on fully populated instance."""
        w = sample_attention_weights
        assert w.gene_gate is not None
        assert w.pathology_attention is not None
        assert w.cell_type_selection is not None
        assert w.region_weights is not None
        assert w.hgt_layer_scales is not None
        assert w.subject_ids is not None


# ============================================================================
# AttentionExtractor Tests
# ============================================================================


class TestAttentionExtractor:
    """Tests for AttentionExtractor class."""

    def test_init_extracts_model_params(self, mock_model):
        """Extractor initializes and extracts model configuration."""
        extractor = AttentionExtractor(mock_model)
        assert extractor.n_cell_types == N_CELL_TYPES
        assert extractor.n_genes == 100
        mock_model.eval.assert_called_once()

    def test_extract_static_weights_returns_attention_weights(self, mock_model):
        """extract_static_weights returns AttentionWeights instance."""
        extractor = AttentionExtractor(mock_model)
        weights = extractor.extract_static_weights()
        assert isinstance(weights, AttentionWeights)

    def test_extract_static_weights_gene_gate_shape(self, mock_model):
        """Extracted gene gate has correct shape."""
        extractor = AttentionExtractor(mock_model)
        weights = extractor.extract_static_weights()
        assert weights.gene_gate.shape == (N_CELL_TYPES, 100)

    def test_extract_static_weights_cell_type_selection_shape(self, mock_model):
        """Extracted cell type selection has correct shape."""
        extractor = AttentionExtractor(mock_model)
        weights = extractor.extract_static_weights()
        assert weights.cell_type_selection.shape == (N_CELL_TYPES,)

    def test_extract_static_weights_region_weights_present(self, mock_model):
        """Region weights are extracted."""
        extractor = AttentionExtractor(mock_model)
        weights = extractor.extract_static_weights()
        assert weights.region_weights is not None
        assert len(weights.region_weights) == 6

    def test_extract_static_weights_hgt_layer_scales_present(self, mock_model):
        """HGT layer scales are extracted."""
        extractor = AttentionExtractor(mock_model)
        weights = extractor.extract_static_weights()
        assert weights.hgt_layer_scales is not None
        assert "layer_0" in weights.hgt_layer_scales


# ============================================================================
# DataFrame Conversion Tests
# ============================================================================


class TestDataFrameConversions:
    """Tests for DataFrame conversion methods."""

    def test_gene_gate_to_dataframe_shape(self, mock_model):
        """gene_gate_to_dataframe produces correct number of rows."""
        extractor = AttentionExtractor(mock_model)
        gene_gate = np.random.rand(5, 10).astype(np.float32)
        gene_names = [f"gene_{i}" for i in range(10)]

        # Temporarily set cell_type_names to match test data
        extractor.cell_type_names = [f"type_{i}" for i in range(5)]

        df = extractor.gene_gate_to_dataframe(gene_gate, gene_names)
        assert len(df) == 5 * 10  # n_cell_types * n_genes
        assert set(df.columns) == {"cell_type", "gene", "gene_idx", "weight"}

    def test_gene_gate_to_dataframe_generates_gene_names(self, mock_model):
        """gene_gate_to_dataframe generates gene names if not provided."""
        extractor = AttentionExtractor(mock_model)
        gene_gate = np.random.rand(5, 10).astype(np.float32)
        extractor.cell_type_names = [f"type_{i}" for i in range(5)]

        df = extractor.gene_gate_to_dataframe(gene_gate, gene_names=None)
        assert "gene_0" in df["gene"].values

    def test_get_top_genes_per_cell_type(self, mock_model):
        """get_top_genes_per_cell_type returns top-k genes ranked."""
        extractor = AttentionExtractor(mock_model)
        gene_gate = np.random.rand(3, 20).astype(np.float32)
        gene_names = [f"gene_{i}" for i in range(20)]
        extractor.cell_type_names = ["A", "B", "C"]

        df = extractor.get_top_genes_per_cell_type(gene_gate, gene_names, top_k=5)
        assert len(df) == 3 * 5  # 3 cell types * 5 top genes
        assert set(df.columns) == {"cell_type", "rank", "gene", "gene_idx", "weight"}
        assert df["rank"].min() == 1
        assert df["rank"].max() == 5

    def test_cell_type_selection_to_dataframe(self, mock_model):
        """cell_type_selection_to_dataframe produces ranked DataFrame."""
        extractor = AttentionExtractor(mock_model)
        weights = np.array([0.1, 0.3, 0.2])
        extractor.cell_type_names = ["A", "B", "C"]

        df = extractor.cell_type_selection_to_dataframe(weights)
        assert list(df.columns) == ["cell_type", "weight", "rank"]
        assert df.iloc[0]["rank"] == 1
        assert df.iloc[0]["weight"] == 0.3  # Highest weight

    def test_pathology_attention_to_dataframe(self, mock_model):
        """pathology_attention_to_dataframe produces tidy format."""
        extractor = AttentionExtractor(mock_model)
        attention = np.random.rand(2, 3, 4).astype(np.float32)  # 2 subjects, 3 heads, 4 types
        extractor.cell_type_names = ["A", "B", "C", "D"]
        subject_ids = ["S1", "S2"]

        df = extractor.pathology_attention_to_dataframe(attention, subject_ids)
        assert len(df) == 2 * 3 * 4  # n_subjects * n_heads * n_cell_types
        assert set(df.columns) == {"subject_id", "head", "cell_type", "weight"}

    def test_aggregate_pathology_attention_mean(self, mock_model):
        """aggregate_pathology_attention with mean aggregation."""
        extractor = AttentionExtractor(mock_model)
        attention = np.random.rand(2, 3, 4).astype(np.float32)
        extractor.cell_type_names = ["A", "B", "C", "D"]

        df = extractor.aggregate_pathology_attention(attention, aggregation="mean")
        assert "mean_attention" in df.columns
        assert len(df) == 2 * 4  # n_subjects * n_cell_types


# ============================================================================
# HDF5 Serialization Tests
# ============================================================================


class TestHDF5Serialization:
    """Tests for HDF5 save/load round-trip."""

    def test_save_load_roundtrip_minimal(self, sample_gene_gate):
        """Minimal AttentionWeights survives save/load round-trip."""
        weights = AttentionWeights(gene_gate=sample_gene_gate)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "weights.h5"
            save_attention_weights_hdf5(weights, path)
            loaded = load_attention_weights_hdf5(path)

        np.testing.assert_array_almost_equal(loaded.gene_gate, weights.gene_gate)

    def test_save_load_roundtrip_full(self, sample_attention_weights):
        """Full AttentionWeights survives save/load round-trip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "weights.h5"
            save_attention_weights_hdf5(sample_attention_weights, path)
            loaded = load_attention_weights_hdf5(path)

        np.testing.assert_array_almost_equal(
            loaded.gene_gate, sample_attention_weights.gene_gate
        )
        np.testing.assert_array_almost_equal(
            loaded.pathology_attention, sample_attention_weights.pathology_attention
        )
        np.testing.assert_array_almost_equal(
            loaded.cell_type_selection, sample_attention_weights.cell_type_selection
        )
        np.testing.assert_array_almost_equal(
            loaded.region_weights, sample_attention_weights.region_weights
        )
        assert loaded.subject_ids == sample_attention_weights.subject_ids
        assert loaded.cell_type_names == sample_attention_weights.cell_type_names

    def test_save_creates_parent_directories(self, sample_gene_gate):
        """save_attention_weights_hdf5 creates parent directories."""
        weights = AttentionWeights(gene_gate=sample_gene_gate)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dir" / "weights.h5"
            save_attention_weights_hdf5(weights, path)
            assert path.exists()

    def test_save_includes_schema_version(self, sample_gene_gate):
        """Saved HDF5 includes schema version attribute."""
        import h5py

        weights = AttentionWeights(gene_gate=sample_gene_gate)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "weights.h5"
            save_attention_weights_hdf5(weights, path)

            with h5py.File(path, "r") as f:
                assert "schema_version" in f.attrs
                assert f.attrs["schema_version"] == "2.0"

    def test_save_with_gene_names(self, sample_gene_gate):
        """Gene names are saved when provided."""
        import h5py

        weights = AttentionWeights(gene_gate=sample_gene_gate)
        gene_names = [f"gene_{i}" for i in range(100)]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "weights.h5"
            save_attention_weights_hdf5(weights, path, gene_names=gene_names)

            with h5py.File(path, "r") as f:
                assert "gene_names" in f
                loaded_names = [x.decode("utf-8") for x in f["gene_names"][:]]
                assert loaded_names == gene_names


# ============================================================================
# Schema Validation Tests
# ============================================================================


class TestOutputSchemaValidation:
    """Tests validating output DataFrame schemas."""

    def test_gene_gate_dataframe_schema(self, mock_model):
        """gene_gate_to_dataframe output has expected schema."""
        extractor = AttentionExtractor(mock_model)
        gene_gate = np.random.rand(3, 5).astype(np.float32)
        extractor.cell_type_names = ["A", "B", "C"]

        df = extractor.gene_gate_to_dataframe(gene_gate, [f"g{i}" for i in range(5)])

        # Schema validation
        assert df["cell_type"].dtype == object
        assert df["gene"].dtype == object
        assert df["gene_idx"].dtype in [np.int64, np.int32, int]
        assert df["weight"].dtype in [np.float64, np.float32, float]

    def test_top_genes_dataframe_schema(self, mock_model):
        """get_top_genes_per_cell_type output has expected schema."""
        extractor = AttentionExtractor(mock_model)
        gene_gate = np.random.rand(3, 10).astype(np.float32)
        extractor.cell_type_names = ["A", "B", "C"]

        df = extractor.get_top_genes_per_cell_type(gene_gate, top_k=5)

        # Schema validation
        assert "rank" in df.columns
        assert df["rank"].dtype in [np.int64, np.int32, int]
        assert (df["rank"] >= 1).all()

    def test_pathology_attention_dataframe_schema(self, mock_model):
        """pathology_attention_to_dataframe output has expected schema."""
        extractor = AttentionExtractor(mock_model)
        attention = np.random.rand(2, 2, 3).astype(np.float32)
        extractor.cell_type_names = ["A", "B", "C"]

        df = extractor.pathology_attention_to_dataframe(attention, ["S1", "S2"])

        # Schema validation
        assert df["subject_id"].dtype == object
        assert df["head"].dtype in [np.int64, np.int32, int]
        assert df["cell_type"].dtype == object
        assert df["weight"].dtype in [np.float64, np.float32, float]


# ============================================================================
# Property-Based Tests (Hypothesis)
# ============================================================================


class TestPropertyBased:
    """Property-based tests using Hypothesis."""

    @given(
        n_cell_types=st.integers(min_value=1, max_value=10),
        n_genes=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=20)
    def test_gene_gate_dataframe_row_count(self, n_cell_types, n_genes):
        """gene_gate_to_dataframe always produces n_cell_types * n_genes rows."""
        gene_gate = np.random.rand(n_cell_types, n_genes).astype(np.float32)
        cell_type_names = [f"type_{i}" for i in range(n_cell_types)]

        mock_model = MagicMock()
        mock_model.n_cell_types = n_cell_types
        mock_model.n_genes = n_genes

        extractor = AttentionExtractor.__new__(AttentionExtractor)
        extractor.cell_type_names = cell_type_names

        df = extractor.gene_gate_to_dataframe(gene_gate)
        assert len(df) == n_cell_types * n_genes

    @given(
        n_subjects=st.integers(min_value=1, max_value=10),
        n_heads=st.integers(min_value=1, max_value=4),
        n_cell_types=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=20)
    def test_pathology_attention_dataframe_row_count(self, n_subjects, n_heads, n_cell_types):
        """pathology_attention_to_dataframe produces correct row count."""
        attention = np.random.rand(n_subjects, n_heads, n_cell_types).astype(np.float32)
        cell_type_names = [f"type_{i}" for i in range(n_cell_types)]
        subject_ids = [f"subj_{i}" for i in range(n_subjects)]

        mock_model = MagicMock()
        extractor = AttentionExtractor.__new__(AttentionExtractor)
        extractor.cell_type_names = cell_type_names

        df = extractor.pathology_attention_to_dataframe(attention, subject_ids)
        assert len(df) == n_subjects * n_heads * n_cell_types

    @given(
        n_cell_types=st.integers(min_value=1, max_value=10),
        n_genes=st.integers(min_value=5, max_value=50),
        top_k=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=20)
    def test_top_genes_respects_top_k(self, n_cell_types, n_genes, top_k):
        """get_top_genes_per_cell_type returns exactly top_k per cell type."""
        gene_gate = np.random.rand(n_cell_types, n_genes).astype(np.float32)
        cell_type_names = [f"type_{i}" for i in range(n_cell_types)]

        extractor = AttentionExtractor.__new__(AttentionExtractor)
        extractor.cell_type_names = cell_type_names

        actual_k = min(top_k, n_genes)
        df = extractor.get_top_genes_per_cell_type(gene_gate, top_k=top_k)
        assert len(df) == n_cell_types * actual_k


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Edge case tests for boundary conditions."""

    def test_single_cell_type(self):
        """Handles single cell type correctly."""
        gene_gate = np.random.rand(1, 10).astype(np.float32)
        weights = AttentionWeights(
            gene_gate=gene_gate,
            cell_type_names=["SingleType"],
        )
        assert weights.gene_gate.shape == (1, 10)
        assert len(weights.cell_type_names) == 1

    def test_single_gene(self, mock_model):
        """Handles single gene correctly."""
        mock_model.n_genes = 1
        import torch
        mock_model.pseudobulk_encoder.gene_gate.get_gate_weights.return_value = torch.rand(N_CELL_TYPES, 1)

        extractor = AttentionExtractor(mock_model)
        weights = extractor.extract_static_weights()
        assert weights.gene_gate.shape[1] == 1

    def test_single_subject_pathology_attention(self, mock_model):
        """Handles single subject pathology attention."""
        extractor = AttentionExtractor(mock_model)
        attention = np.random.rand(1, 4, N_CELL_TYPES).astype(np.float32)
        extractor.cell_type_names = list(CELL_TYPE_ORDER)

        df = extractor.pathology_attention_to_dataframe(attention, ["single_subject"])
        assert len(df) == 1 * 4 * N_CELL_TYPES
        assert df["subject_id"].unique().tolist() == ["single_subject"]

    def test_empty_hgt_layer_scales(self, mock_model):
        """Handles empty HGT layer scales."""
        mock_model.get_hgt_layer_scales.return_value = {}

        extractor = AttentionExtractor(mock_model)
        weights = extractor.extract_static_weights()
        assert weights.hgt_layer_scales == {}

    def test_hdf5_roundtrip_with_none_optional_fields(self):
        """HDF5 round-trip works with None optional fields."""
        gene_gate = np.random.rand(5, 10).astype(np.float32)
        weights = AttentionWeights(gene_gate=gene_gate)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "weights.h5"
            save_attention_weights_hdf5(weights, path)
            loaded = load_attention_weights_hdf5(path)

        assert loaded.pathology_attention is None
        assert loaded.cell_type_selection is None
        assert loaded.region_weights is None
        assert loaded.subject_ids is None
