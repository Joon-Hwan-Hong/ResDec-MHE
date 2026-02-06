"""
Tests for src/analysis/ccc_importance.py.

Test coverage includes:
- CCCImportanceResult dataclass behavior
- CCCImportanceAnalyzer initialization
- Edge importance computation
- Top interactions extraction
- Region-stratified importance
- Network summary by edge type
- Schema validation
- Edge cases
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from src.data.constants import CELL_TYPE_ORDER, ALL_EDGE_TYPES, N_CELL_TYPES
from src.analysis.ccc_importance import (
    CCCImportanceResult,
    CCCImportanceAnalyzer,
    compute_ccc_importance,
    create_edge_metadata_from_graph,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_edge_metadata():
    """Sample edge metadata DataFrame."""
    rows = []
    for edge_type in ALL_EDGE_TYPES[:3]:  # Use 3 edge types
        for i in range(10):  # 10 edges per type
            rows.append({
                "source": f"cell_{i % 5}",
                "target": f"cell_{(i + 1) % 5}",
                "edge_type": edge_type,
                "source_idx": i % 5,
                "target_idx": (i + 1) % 5,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_edge_attention():
    """Sample edge attention scores [n_subjects, n_edges]."""
    np.random.seed(42)
    n_subjects = 20
    n_edges = 30  # Matches sample_edge_metadata (3 edge types × 10 edges)
    return np.random.rand(n_subjects, n_edges).astype(np.float32)


@pytest.fixture
def sample_region_labels():
    """Sample region labels."""
    np.random.seed(42)
    regions = ["PFC", "AG", "MTC"]
    return np.array([regions[i % len(regions)] for i in range(20)])


@pytest.fixture
def analyzer(sample_edge_attention, sample_edge_metadata, sample_region_labels):
    """CCCImportanceAnalyzer instance."""
    return CCCImportanceAnalyzer(
        edge_attention_scores=sample_edge_attention,
        edge_metadata=sample_edge_metadata,
        region_labels=sample_region_labels,
    )


# ============================================================================
# CCCImportanceResult Dataclass Tests
# ============================================================================


class TestCCCImportanceResult:
    """Tests for CCCImportanceResult dataclass."""

    def test_init_with_required_fields(self):
        """Result can be initialized with required fields."""
        edge_importance = pd.DataFrame({"source": ["A"], "target": ["B"], "edge_type": ["X"], "mean_attention": [0.5]})
        top_interactions = pd.DataFrame({"rank": [1], "source": ["A"], "target": ["B"], "edge_type": ["X"], "mean_attention": [0.5]})
        result = CCCImportanceResult(edge_importance=edge_importance, top_interactions=top_interactions)
        assert result.edge_importance is not None
        assert result.by_region is None

    def test_metadata_defaults_to_empty_dict(self):
        """metadata defaults to empty dict."""
        edge_importance = pd.DataFrame({"source": ["A"], "target": ["B"], "mean_attention": [0.5]})
        top_interactions = pd.DataFrame({"rank": [1], "source": ["A"], "target": ["B"], "mean_attention": [0.5]})
        result = CCCImportanceResult(edge_importance=edge_importance, top_interactions=top_interactions)
        assert result.metadata == {}


# ============================================================================
# CCCImportanceAnalyzer Initialization Tests
# ============================================================================


class TestAnalyzerInit:
    """Tests for CCCImportanceAnalyzer initialization."""

    def test_init_with_metadata_only(self, sample_edge_metadata):
        """Analyzer initializes with only edge metadata."""
        analyzer = CCCImportanceAnalyzer(edge_metadata=sample_edge_metadata)
        result = analyzer.analyze()
        assert result.edge_importance is not None

    def test_init_without_any_data(self):
        """Analyzer can be created without data (generates placeholder)."""
        analyzer = CCCImportanceAnalyzer()
        result = analyzer.analyze()
        assert result.edge_importance is not None

    def test_init_uses_default_cell_types(self):
        """Analyzer uses CELL_TYPE_ORDER as default."""
        analyzer = CCCImportanceAnalyzer()
        assert analyzer.cell_type_names == list(CELL_TYPE_ORDER)

    def test_init_uses_default_edge_types(self):
        """Analyzer uses ALL_EDGE_TYPES as default."""
        analyzer = CCCImportanceAnalyzer()
        assert analyzer.edge_types == list(ALL_EDGE_TYPES)


# ============================================================================
# Edge Importance Tests
# ============================================================================


class TestEdgeImportance:
    """Tests for edge importance computation."""

    def test_analyze_returns_result(self, analyzer):
        """analyze() returns CCCImportanceResult."""
        result = analyzer.analyze()
        assert isinstance(result, CCCImportanceResult)

    def test_edge_importance_merges_with_metadata(self, analyzer, sample_edge_metadata):
        """edge_importance includes metadata columns."""
        result = analyzer.analyze()
        assert "source" in result.edge_importance.columns
        assert "target" in result.edge_importance.columns
        assert "edge_type" in result.edge_importance.columns
        assert len(result.edge_importance) == len(sample_edge_metadata)

    def test_edge_importance_has_attention_columns(self, analyzer):
        """edge_importance includes attention statistics."""
        result = analyzer.analyze()
        assert "mean_attention" in result.edge_importance.columns
        assert "std_attention" in result.edge_importance.columns

    def test_edge_importance_aggregates_across_subjects(self, sample_edge_attention, sample_edge_metadata):
        """Mean attention is correctly computed across subjects."""
        analyzer = CCCImportanceAnalyzer(
            edge_attention_scores=sample_edge_attention,
            edge_metadata=sample_edge_metadata,
        )
        result = analyzer.analyze()

        expected_mean = sample_edge_attention.mean(axis=0)[0]
        actual_mean = result.edge_importance["mean_attention"].iloc[0]
        assert np.isclose(actual_mean, expected_mean, atol=1e-6)


# ============================================================================
# Top Interactions Tests
# ============================================================================


class TestTopInteractions:
    """Tests for top interactions extraction."""

    def test_top_interactions_has_expected_columns(self, analyzer):
        """top_interactions has expected columns."""
        result = analyzer.analyze(top_k=10)
        assert "rank" in result.top_interactions.columns
        assert "source" in result.top_interactions.columns
        assert "target" in result.top_interactions.columns
        assert "mean_attention" in result.top_interactions.columns

    def test_top_interactions_respects_top_k(self, analyzer):
        """top_interactions extracts at most top_k interactions."""
        result = analyzer.analyze(top_k=5)
        assert len(result.top_interactions) <= 5

    def test_top_interactions_sorted_by_attention(self, analyzer):
        """top_interactions sorted by mean_attention descending."""
        result = analyzer.analyze(top_k=10)
        if len(result.top_interactions) > 1:
            attentions = result.top_interactions["mean_attention"].tolist()
            assert attentions == sorted(attentions, reverse=True)

    def test_top_interactions_has_sequential_ranks(self, analyzer):
        """top_interactions has sequential ranks starting from 1."""
        result = analyzer.analyze(top_k=10)
        n_rows = len(result.top_interactions)
        expected_ranks = list(range(1, n_rows + 1))
        assert result.top_interactions["rank"].tolist() == expected_ranks


# ============================================================================
# Region-Stratified Importance Tests
# ============================================================================


class TestRegionStratified:
    """Tests for region-stratified importance computation."""

    def test_by_region_present_when_labels_provided(self, analyzer):
        """by_region is present when region_labels provided."""
        result = analyzer.analyze()
        assert result.by_region is not None

    def test_by_region_absent_when_labels_missing(self, sample_edge_attention, sample_edge_metadata):
        """by_region is None when region_labels not provided."""
        analyzer = CCCImportanceAnalyzer(
            edge_attention_scores=sample_edge_attention,
            edge_metadata=sample_edge_metadata,
        )
        result = analyzer.analyze()
        assert result.by_region is None

    def test_by_region_has_expected_columns(self, analyzer):
        """by_region has expected columns."""
        result = analyzer.analyze()
        expected_cols = {"region", "source", "target", "edge_type", "mean_attention", "n_subjects"}
        assert set(result.by_region.columns) == expected_cols

    def test_by_region_includes_all_regions(self, analyzer, sample_region_labels):
        """by_region includes all unique regions from labels."""
        result = analyzer.analyze()
        regions = set(result.by_region["region"].unique())
        expected_regions = set(np.unique(sample_region_labels))
        assert regions == expected_regions


# ============================================================================
# Network Summary Tests
# ============================================================================


class TestNetworkSummary:
    """Tests for network summary computation."""

    def test_network_summary_present(self, analyzer):
        """network_summary is computed."""
        result = analyzer.analyze()
        assert result.network_summary is not None

    def test_network_summary_has_expected_columns(self, analyzer):
        """network_summary has expected columns."""
        result = analyzer.analyze()
        expected_cols = {"edge_type", "display_name", "mean_attention", "std_attention", "n_edges"}
        assert set(result.network_summary.columns) == expected_cols

    def test_network_summary_aggregates_by_edge_type(self, analyzer):
        """network_summary aggregates across edges of same type."""
        result = analyzer.analyze()
        # Each edge type should have one row
        edge_types_in_summary = result.network_summary["edge_type"].unique()
        edge_types_in_data = result.edge_importance["edge_type"].unique()
        assert set(edge_types_in_summary) == set(edge_types_in_data)


# ============================================================================
# Save/Load Tests
# ============================================================================


class TestSaveLoad:
    """Tests for save functionality."""

    def test_save_creates_files(self, analyzer):
        """save() creates expected files."""
        result = analyzer.analyze()
        with tempfile.TemporaryDirectory() as tmpdir:
            saved = analyzer.save(result, tmpdir)
            assert (Path(tmpdir) / "ccc_importance.parquet").exists()
            assert (Path(tmpdir) / "top_interactions.csv").exists()
            assert (Path(tmpdir) / "ccc_network_summary.csv").exists()

    def test_save_creates_region_files_when_present(self, analyzer):
        """save() creates region files when by_region available."""
        result = analyzer.analyze()
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer.save(result, tmpdir)
            assert (Path(tmpdir) / "ccc_importance_by_region.parquet").exists()


# ============================================================================
# Schema Validation Tests
# ============================================================================


class TestOutputSchemaValidation:
    """Tests validating output DataFrame schemas."""

    def test_edge_importance_schema(self, analyzer):
        """edge_importance has expected schema."""
        result = analyzer.analyze()
        df = result.edge_importance
        assert df["source"].dtype == object
        assert df["target"].dtype == object
        assert df["edge_type"].dtype == object
        assert np.issubdtype(df["mean_attention"].dtype, np.floating)

    def test_attention_values_bounded(self, analyzer):
        """Attention values are bounded [0, 1]."""
        result = analyzer.analyze()
        assert (result.edge_importance["mean_attention"] >= 0).all()
        assert (result.edge_importance["mean_attention"] <= 1).all()


# ============================================================================
# Convenience Function Tests
# ============================================================================


class TestConvenienceFunction:
    """Tests for compute_ccc_importance function."""

    def test_compute_returns_result(self, sample_edge_attention, sample_edge_metadata):
        """compute_ccc_importance returns CCCImportanceResult."""
        result = compute_ccc_importance(
            edge_attention_scores=sample_edge_attention,
            edge_metadata=sample_edge_metadata,
        )
        assert isinstance(result, CCCImportanceResult)

    def test_compute_with_output_dir_saves_files(self, sample_edge_attention, sample_edge_metadata):
        """compute_ccc_importance saves when output_dir provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = compute_ccc_importance(
                edge_attention_scores=sample_edge_attention,
                edge_metadata=sample_edge_metadata,
                output_dir=tmpdir,
            )
            assert (Path(tmpdir) / "ccc_importance.parquet").exists()


# ============================================================================
# Edge Metadata Creation Tests
# ============================================================================


class TestEdgeMetadataCreation:
    """Tests for create_edge_metadata_from_graph."""

    def test_creates_dataframe(self):
        """create_edge_metadata_from_graph returns DataFrame."""
        edge_index_dict = {
            ("A", "interacts", "B"): np.array([[0, 1], [1, 2]]),
        }
        df = create_edge_metadata_from_graph(edge_index_dict, ["A", "B", "C"])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2  # 2 edges

    def test_has_expected_columns(self):
        """Output has expected columns."""
        edge_index_dict = {
            ("A", "interacts", "B"): np.array([[0], [1]]),
        }
        df = create_edge_metadata_from_graph(edge_index_dict)
        expected_cols = {"source", "target", "edge_type", "source_idx", "target_idx"}
        assert set(df.columns) == expected_cols


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_edge_metadata(self):
        """Handles empty edge metadata."""
        analyzer = CCCImportanceAnalyzer(edge_metadata=pd.DataFrame())
        result = analyzer.analyze()
        # Should generate placeholder data
        assert result.edge_importance is not None

    def test_single_edge(self):
        """Handles single edge correctly."""
        edge_metadata = pd.DataFrame({
            "source": ["A"],
            "target": ["B"],
            "edge_type": ["interaction"],
        })
        edge_attention = np.array([[0.5]])  # 1 subject, 1 edge

        analyzer = CCCImportanceAnalyzer(
            edge_attention_scores=edge_attention,
            edge_metadata=edge_metadata,
        )
        result = analyzer.analyze()
        assert len(result.edge_importance) == 1
        assert len(result.top_interactions) == 1

    def test_pre_aggregated_attention(self):
        """Handles pre-aggregated 1D attention scores."""
        edge_metadata = pd.DataFrame({
            "source": ["A", "B"],
            "target": ["B", "C"],
            "edge_type": ["X", "X"],
        })
        edge_attention = np.array([0.5, 0.3])  # Already aggregated

        analyzer = CCCImportanceAnalyzer(
            edge_attention_scores=edge_attention,
            edge_metadata=edge_metadata,
        )
        result = analyzer.analyze()
        assert len(result.edge_importance) == 2
        # Std should be 0 for pre-aggregated
        assert (result.edge_importance["std_attention"] == 0).all()

    def test_top_k_larger_than_n_edges(self, sample_edge_attention, sample_edge_metadata):
        """Handles top_k > n_edges correctly."""
        analyzer = CCCImportanceAnalyzer(
            edge_attention_scores=sample_edge_attention,
            edge_metadata=sample_edge_metadata,
        )
        result = analyzer.analyze(top_k=1000)  # Much larger than actual edges
        assert len(result.top_interactions) == len(sample_edge_metadata)


# ============================================================================
# CCC Length Validation Tests (Phase 6 Review Round 5)
# ============================================================================


class TestCCCLengthValidation:
    """Test CCC analyzer rejects mismatched attention/metadata lengths."""

    def test_mismatched_lengths_raises_error(self):
        """Mismatched attention and edge_metadata lengths should raise ValueError."""
        attention = np.random.rand(10, 5)  # 10 subjects, 5 edges
        metadata = pd.DataFrame({
            "source": ["A"] * 3,  # only 3 edges
            "target": ["B"] * 3,
            "edge_type": ["Secreted_Signaling"] * 3,
        })
        analyzer = CCCImportanceAnalyzer(
            edge_attention_scores=attention,
            edge_metadata=metadata,
            cell_type_names=["A", "B"],
        )
        with pytest.raises(ValueError, match="does not match edge metadata"):
            analyzer.analyze()

    def test_matching_lengths_succeeds(self):
        """Matching attention and edge_metadata lengths should work."""
        attention = np.random.rand(10, 5)  # 10 subjects, 5 edges
        metadata = pd.DataFrame({
            "source": ["A"] * 5,
            "target": ["B"] * 5,
            "edge_type": ["Secreted_Signaling"] * 5,
        })
        analyzer = CCCImportanceAnalyzer(
            edge_attention_scores=attention,
            edge_metadata=metadata,
            cell_type_names=["A", "B"],
        )
        result = analyzer.analyze()
        assert result.edge_importance is not None


# ============================================================================
# Phase 6 Review Round 8 — H1: Empty region label filtering
# ============================================================================


class TestEmptyRegionLabelFiltering:
    """Tests for empty-string region label filtering in CCC importance."""

    def test_empty_region_labels_excluded_from_stratification(self):
        """Subjects with empty region labels are excluded from by-region analysis."""
        np.random.seed(42)
        n_subjects, n_edges = 20, 10
        attention = np.random.rand(n_subjects, n_edges).astype(np.float32)
        metadata = pd.DataFrame({
            "source": ["A"] * n_edges,
            "target": ["B"] * n_edges,
            "edge_type": ["Secreted_Signaling"] * n_edges,
        })
        # Mix of real regions and empty strings
        region_labels = np.array(
            ["PFC"] * 8 + ["AG"] * 8 + [""] * 4
        )
        analyzer = CCCImportanceAnalyzer(
            edge_attention_scores=attention,
            edge_metadata=metadata,
            cell_type_names=["A", "B"],
            region_labels=region_labels,
        )
        result = analyzer.analyze()
        assert result.by_region is not None
        unique_regions = result.by_region["region"].unique()
        assert "" not in unique_regions
        assert "PFC" in unique_regions
        assert "AG" in unique_regions
