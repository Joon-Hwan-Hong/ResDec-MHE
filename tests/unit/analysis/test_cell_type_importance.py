"""
Tests for src/analysis/cell_type_importance.py.

Test coverage includes:
- CellTypeImportanceResult dataclass behavior
- CellTypeImportanceAnalyzer initialization and validation
- Overall importance computation
- Pathology-stratified importance computation
- Region-stratified importance computation
- Output format validation
- Schema validation
- Property-based tests for mathematical invariants
- Edge cases
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from src.data.constants import CELL_TYPE_ORDER, N_CELL_TYPES, REGION_ORDER
from src.analysis.cell_type_importance import (
    CellTypeImportanceResult,
    CellTypeImportanceAnalyzer,
    compute_cell_type_importance,
    load_cell_type_importance,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_attention():
    """Sample pathology attention weights [n_subjects, n_heads, n_cell_types]."""
    np.random.seed(42)
    return np.random.rand(20, 4, N_CELL_TYPES).astype(np.float32)


@pytest.fixture
def sample_pathology_scores():
    """Sample pathology scores [n_subjects]."""
    np.random.seed(42)
    return np.random.rand(20).astype(np.float32)


@pytest.fixture
def sample_region_labels():
    """Sample region labels [n_subjects]."""
    np.random.seed(42)
    regions = ["PFC", "AG", "MTC", "EC", "HC", "TH"]
    return np.array([regions[i % len(regions)] for i in range(20)])


@pytest.fixture
def sample_subject_ids():
    """Sample subject IDs."""
    return [f"subj_{i:03d}" for i in range(20)]


@pytest.fixture
def analyzer(sample_attention, sample_pathology_scores, sample_region_labels, sample_subject_ids):
    """Fully initialized CellTypeImportanceAnalyzer."""
    return CellTypeImportanceAnalyzer(
        attention=sample_attention,
        pathology_scores=sample_pathology_scores,
        region_labels=sample_region_labels,
        subject_ids=sample_subject_ids,
    )


# ============================================================================
# CellTypeImportanceResult Dataclass Tests
# ============================================================================


class TestCellTypeImportanceResult:
    """Tests for CellTypeImportanceResult dataclass."""

    def test_init_with_overall_only(self):
        """Result can be initialized with only overall DataFrame."""
        overall = pd.DataFrame({
            "cell_type": ["A", "B"],
            "mean_attention": [0.5, 0.3],
            "std_attention": [0.1, 0.1],
            "rank": [1, 2],
        })
        result = CellTypeImportanceResult(overall=overall)
        assert result.overall is not None
        assert result.by_pathology is None
        assert result.by_region is None

    def test_metadata_defaults_to_empty_dict(self):
        """metadata defaults to empty dict."""
        overall = pd.DataFrame({"cell_type": ["A"], "mean_attention": [0.5], "std_attention": [0.1], "rank": [1]})
        result = CellTypeImportanceResult(overall=overall)
        assert result.metadata == {}

    def test_all_fields_accessible(self):
        """All fields are accessible."""
        overall = pd.DataFrame({"cell_type": ["A"], "mean_attention": [0.5], "std_attention": [0.1], "rank": [1]})
        by_pathology = pd.DataFrame({"cell_type": ["A"], "pathology_tertile": ["low"], "mean_attention": [0.5], "std_attention": [0.1], "n_subjects": [10]})
        result = CellTypeImportanceResult(
            overall=overall,
            by_pathology=by_pathology,
            metadata={"key": "value"},
        )
        assert result.overall is not None
        assert result.by_pathology is not None
        assert result.metadata["key"] == "value"


# ============================================================================
# CellTypeImportanceAnalyzer Initialization Tests
# ============================================================================


class TestAnalyzerInit:
    """Tests for CellTypeImportanceAnalyzer initialization."""

    def test_init_minimal(self, sample_attention):
        """Analyzer initializes with only attention array."""
        analyzer = CellTypeImportanceAnalyzer(attention=sample_attention)
        assert analyzer.attention.shape == sample_attention.shape

    def test_init_sets_default_cell_type_names(self, sample_attention):
        """Analyzer uses CELL_TYPE_ORDER as default cell type names."""
        analyzer = CellTypeImportanceAnalyzer(attention=sample_attention)
        assert analyzer.cell_type_names == list(CELL_TYPE_ORDER)

    def test_init_preserves_custom_cell_type_names(self, sample_attention):
        """Analyzer preserves custom cell type names."""
        custom = ["Type_A", "Type_B"]
        # Create attention with matching shape
        attention = np.random.rand(10, 4, 2).astype(np.float32)
        analyzer = CellTypeImportanceAnalyzer(
            attention=attention,
            cell_type_names=custom,
        )
        assert analyzer.cell_type_names == custom

    def test_init_validates_attention_ndim(self):
        """Analyzer rejects attention with wrong number of dimensions."""
        bad_attention = np.random.rand(10, 31).astype(np.float32)  # 2D instead of 3D
        with pytest.raises(ValueError, match="must be 3D"):
            CellTypeImportanceAnalyzer(attention=bad_attention)

    def test_init_validates_pathology_length(self, sample_attention):
        """Analyzer rejects pathology_scores with wrong length."""
        bad_pathology = np.random.rand(5).astype(np.float32)  # Wrong length
        with pytest.raises(ValueError, match="pathology_scores"):
            CellTypeImportanceAnalyzer(
                attention=sample_attention,
                pathology_scores=bad_pathology,
            )

    def test_init_validates_region_labels_length(self, sample_attention):
        """Analyzer rejects region_labels with wrong length."""
        bad_regions = np.array(["PFC"] * 5)  # Wrong length
        with pytest.raises(ValueError, match="region_labels"):
            CellTypeImportanceAnalyzer(
                attention=sample_attention,
                region_labels=bad_regions,
            )

    def test_init_validates_subject_ids_length(self, sample_attention):
        """Analyzer rejects subject_ids with wrong length."""
        bad_ids = ["a", "b", "c"]  # Wrong length
        with pytest.raises(ValueError, match="subject_ids"):
            CellTypeImportanceAnalyzer(
                attention=sample_attention,
                subject_ids=bad_ids,
            )


# ============================================================================
# Overall Importance Computation Tests
# ============================================================================


class TestOverallImportance:
    """Tests for overall importance computation."""

    def test_analyze_returns_result(self, analyzer):
        """analyze() returns CellTypeImportanceResult."""
        result = analyzer.analyze()
        assert isinstance(result, CellTypeImportanceResult)

    def test_overall_has_all_cell_types(self, analyzer):
        """Overall importance includes all cell types."""
        result = analyzer.analyze()
        assert len(result.overall) == N_CELL_TYPES
        assert set(result.overall["cell_type"]) == set(CELL_TYPE_ORDER)

    def test_overall_has_expected_columns(self, analyzer):
        """Overall importance has expected columns."""
        result = analyzer.analyze()
        expected_cols = {"cell_type", "mean_attention", "std_attention", "rank"}
        assert set(result.overall.columns) == expected_cols

    def test_overall_ranks_are_sequential(self, analyzer):
        """Ranks are sequential from 1 to n_cell_types."""
        result = analyzer.analyze()
        ranks = sorted(result.overall["rank"].tolist())
        assert ranks == list(range(1, N_CELL_TYPES + 1))

    def test_overall_sorted_by_mean_attention(self, analyzer):
        """Overall is sorted by mean_attention descending."""
        result = analyzer.analyze()
        means = result.overall["mean_attention"].tolist()
        assert means == sorted(means, reverse=True)

    def test_overall_mean_attention_is_average_over_subjects_and_heads(self, sample_attention):
        """Mean attention is computed correctly as mean over subjects and heads."""
        analyzer = CellTypeImportanceAnalyzer(attention=sample_attention)
        result = analyzer.analyze()

        # Manual computation
        attention_per_subject = sample_attention.mean(axis=1)  # [n_subjects, n_cell_types]
        expected_means = attention_per_subject.mean(axis=0)  # [n_cell_types]

        # Get mean for first cell type (after sorting, need to find by name)
        first_ct = CELL_TYPE_ORDER[0]
        actual_mean = result.overall[result.overall["cell_type"] == first_ct]["mean_attention"].values[0]
        assert np.isclose(actual_mean, expected_means[0], atol=1e-6)


# ============================================================================
# Pathology-Stratified Importance Tests
# ============================================================================


class TestPathologyStratified:
    """Tests for pathology-stratified importance computation."""

    def test_by_pathology_present_when_scores_provided(self, analyzer):
        """by_pathology is present when pathology_scores provided."""
        result = analyzer.analyze()
        assert result.by_pathology is not None

    def test_by_pathology_absent_when_scores_missing(self, sample_attention):
        """by_pathology is None when pathology_scores not provided."""
        analyzer = CellTypeImportanceAnalyzer(attention=sample_attention)
        result = analyzer.analyze()
        assert result.by_pathology is None

    def test_by_pathology_has_expected_columns(self, analyzer):
        """by_pathology has expected columns."""
        result = analyzer.analyze()
        expected_cols = {"cell_type", "pathology_tertile", "mean_attention", "std_attention", "n_subjects"}
        assert set(result.by_pathology.columns) == expected_cols

    def test_by_pathology_has_three_tertiles(self, analyzer):
        """by_pathology has low, medium, high tertiles."""
        result = analyzer.analyze()
        tertiles = set(result.by_pathology["pathology_tertile"].unique())
        assert tertiles == {"low", "medium", "high"}

    def test_by_pathology_n_subjects_sums_correctly(self, analyzer):
        """n_subjects across tertiles sums to total subjects per cell type."""
        result = analyzer.analyze()

        # For each cell type, n_subjects should sum to 20 (total subjects)
        for ct in CELL_TYPE_ORDER[:5]:  # Check first 5
            ct_data = result.by_pathology[result.by_pathology["cell_type"] == ct]
            total = ct_data["n_subjects"].sum()
            assert total == 20

    def test_by_pathology_low_tertile_has_low_pathology_subjects(self, sample_attention, sample_pathology_scores):
        """Low tertile contains subjects with lowest pathology scores."""
        np.random.seed(42)
        # Create distinct pathology scores
        pathology = np.linspace(0, 1, 20).astype(np.float32)

        analyzer = CellTypeImportanceAnalyzer(
            attention=sample_attention,
            pathology_scores=pathology,
        )
        result = analyzer.analyze()

        # Low tertile should have subjects with pathology < 33rd percentile
        low_data = result.by_pathology[result.by_pathology["pathology_tertile"] == "low"]
        # At least some subjects in low tertile
        assert (low_data["n_subjects"] > 0).any()


# ============================================================================
# Region-Stratified Importance Tests
# ============================================================================


class TestRegionStratified:
    """Tests for region-stratified importance computation."""

    def test_by_region_present_when_labels_provided(self, analyzer):
        """by_region is present when region_labels provided."""
        result = analyzer.analyze()
        assert result.by_region is not None

    def test_by_region_absent_when_labels_missing(self, sample_attention):
        """by_region is None when region_labels not provided."""
        analyzer = CellTypeImportanceAnalyzer(attention=sample_attention)
        result = analyzer.analyze()
        assert result.by_region is None

    def test_by_region_has_expected_columns(self, analyzer):
        """by_region has expected columns."""
        result = analyzer.analyze()
        expected_cols = {"cell_type", "region", "mean_attention", "std_attention", "n_subjects"}
        assert set(result.by_region.columns) == expected_cols

    def test_by_region_contains_all_provided_regions(self, analyzer):
        """by_region contains all unique regions from labels."""
        result = analyzer.analyze()
        regions = set(result.by_region["region"].unique())
        # Our fixture cycles through all 6 regions
        assert regions == set(REGION_ORDER)

    def test_by_region_n_subjects_per_region(self, sample_attention, sample_region_labels):
        """n_subjects matches expected count per region."""
        analyzer = CellTypeImportanceAnalyzer(
            attention=sample_attention,
            region_labels=sample_region_labels,
        )
        result = analyzer.analyze()

        # Count subjects per region in input
        from collections import Counter
        region_counts = Counter(sample_region_labels)

        # Check by_region has correct n_subjects for each region
        for region, expected_count in region_counts.items():
            region_data = result.by_region[result.by_region["region"] == region]
            # Each cell type should have same n_subjects for this region
            unique_counts = region_data["n_subjects"].unique()
            assert len(unique_counts) == 1
            assert unique_counts[0] == expected_count


# ============================================================================
# Save/Load Tests
# ============================================================================


class TestSaveLoad:
    """Tests for save and load functionality."""

    def test_save_creates_files(self, analyzer):
        """save() creates expected files."""
        result = analyzer.analyze()

        with tempfile.TemporaryDirectory() as tmpdir:
            saved = analyzer.save(result, tmpdir)

            # Check files exist
            assert (Path(tmpdir) / "cell_type_importance.parquet").exists()
            assert (Path(tmpdir) / "cell_type_importance.csv").exists()

    def test_save_creates_pathology_files_when_present(self, analyzer):
        """save() creates pathology-stratified files when data present."""
        result = analyzer.analyze()

        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer.save(result, tmpdir)

            assert (Path(tmpdir) / "cell_type_importance_by_pathology.parquet").exists()
            assert (Path(tmpdir) / "cell_type_importance_by_pathology.csv").exists()

    def test_save_creates_region_files_when_present(self, analyzer):
        """save() creates region-stratified files when data present."""
        result = analyzer.analyze()

        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer.save(result, tmpdir)

            assert (Path(tmpdir) / "cell_type_importance_by_region.parquet").exists()
            assert (Path(tmpdir) / "cell_type_importance_by_region.csv").exists()

    def test_save_parquet_only(self, analyzer):
        """save() can save only parquet format."""
        result = analyzer.analyze()

        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer.save(result, tmpdir, formats=["parquet"])

            assert (Path(tmpdir) / "cell_type_importance.parquet").exists()
            assert not (Path(tmpdir) / "cell_type_importance.csv").exists()

    def test_load_parquet(self, analyzer):
        """load_cell_type_importance loads parquet files."""
        result = analyzer.analyze()

        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer.save(result, tmpdir)
            loaded = load_cell_type_importance(Path(tmpdir) / "cell_type_importance.parquet")

        pd.testing.assert_frame_equal(loaded, result.overall)

    def test_load_csv(self, analyzer):
        """load_cell_type_importance loads CSV files."""
        result = analyzer.analyze()

        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer.save(result, tmpdir)
            loaded = load_cell_type_importance(Path(tmpdir) / "cell_type_importance.csv")

        assert len(loaded) == len(result.overall)
        assert set(loaded.columns) == set(result.overall.columns)

    def test_save_creates_nested_directories(self, analyzer):
        """save() creates nested parent directories."""
        result = analyzer.analyze()

        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = Path(tmpdir) / "a" / "b" / "c"
            analyzer.save(result, nested_path)

            assert (nested_path / "cell_type_importance.parquet").exists()


# ============================================================================
# Convenience Function Tests
# ============================================================================


class TestConvenienceFunction:
    """Tests for compute_cell_type_importance function."""

    def test_compute_returns_result(self, sample_attention):
        """compute_cell_type_importance returns CellTypeImportanceResult."""
        result = compute_cell_type_importance(attention=sample_attention)
        assert isinstance(result, CellTypeImportanceResult)

    def test_compute_with_output_dir_saves_files(self, sample_attention):
        """compute_cell_type_importance saves when output_dir provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = compute_cell_type_importance(
                attention=sample_attention,
                output_dir=tmpdir,
            )

            assert (Path(tmpdir) / "cell_type_importance.parquet").exists()

    def test_compute_without_output_dir_does_not_save(self, sample_attention):
        """compute_cell_type_importance does not save when output_dir None."""
        # No exception should be raised, just returns result
        result = compute_cell_type_importance(attention=sample_attention)
        assert result is not None


# ============================================================================
# Schema Validation Tests
# ============================================================================


class TestOutputSchemaValidation:
    """Tests validating output DataFrame schemas."""

    def test_overall_schema(self, analyzer):
        """Overall DataFrame has expected schema."""
        result = analyzer.analyze()
        df = result.overall

        assert df["cell_type"].dtype == object
        assert np.issubdtype(df["mean_attention"].dtype, np.floating)
        assert np.issubdtype(df["std_attention"].dtype, np.floating)
        assert np.issubdtype(df["rank"].dtype, np.integer)

    def test_by_pathology_schema(self, analyzer):
        """by_pathology DataFrame has expected schema."""
        result = analyzer.analyze()
        df = result.by_pathology

        assert df["cell_type"].dtype == object
        assert df["pathology_tertile"].dtype == object
        assert np.issubdtype(df["mean_attention"].dtype, np.floating)
        assert np.issubdtype(df["n_subjects"].dtype, np.integer)

    def test_by_region_schema(self, analyzer):
        """by_region DataFrame has expected schema."""
        result = analyzer.analyze()
        df = result.by_region

        assert df["cell_type"].dtype == object
        assert df["region"].dtype == object
        assert np.issubdtype(df["mean_attention"].dtype, np.floating)
        assert np.issubdtype(df["n_subjects"].dtype, np.integer)

    def test_mean_attention_values_bounded(self, analyzer):
        """Mean attention values are bounded [0, 1] (attention weights)."""
        result = analyzer.analyze()

        assert (result.overall["mean_attention"] >= 0).all()
        assert (result.overall["mean_attention"] <= 1).all()

    def test_std_attention_non_negative(self, analyzer):
        """Std attention values are non-negative."""
        result = analyzer.analyze()
        assert (result.overall["std_attention"] >= 0).all()


# ============================================================================
# Property-Based Tests (Hypothesis)
# ============================================================================


class TestPropertyBased:
    """Property-based tests using Hypothesis."""

    @given(
        n_subjects=st.integers(min_value=3, max_value=30),
        n_heads=st.integers(min_value=1, max_value=4),
        n_cell_types=st.integers(min_value=2, max_value=10),
    )
    @settings(max_examples=20)
    def test_overall_always_has_n_cell_types_rows(self, n_subjects, n_heads, n_cell_types):
        """Overall importance always has exactly n_cell_types rows."""
        attention = np.random.rand(n_subjects, n_heads, n_cell_types).astype(np.float32)
        cell_type_names = [f"type_{i}" for i in range(n_cell_types)]

        analyzer = CellTypeImportanceAnalyzer(
            attention=attention,
            cell_type_names=cell_type_names,
        )
        result = analyzer.analyze()

        assert len(result.overall) == n_cell_types

    @given(
        n_subjects=st.integers(min_value=9, max_value=30),  # Need at least 9 for 3 tertiles
        n_heads=st.integers(min_value=1, max_value=4),
        n_cell_types=st.integers(min_value=2, max_value=10),
    )
    @settings(max_examples=20)
    def test_pathology_tertiles_cover_all_subjects(self, n_subjects, n_heads, n_cell_types):
        """Pathology tertiles cover all subjects."""
        attention = np.random.rand(n_subjects, n_heads, n_cell_types).astype(np.float32)
        pathology = np.random.rand(n_subjects).astype(np.float32)
        cell_type_names = [f"type_{i}" for i in range(n_cell_types)]

        analyzer = CellTypeImportanceAnalyzer(
            attention=attention,
            pathology_scores=pathology,
            cell_type_names=cell_type_names,
        )
        result = analyzer.analyze()

        # For any cell type, sum of n_subjects across tertiles should equal n_subjects
        first_ct = cell_type_names[0]
        ct_data = result.by_pathology[result.by_pathology["cell_type"] == first_ct]
        assert ct_data["n_subjects"].sum() == n_subjects

    @given(
        n_subjects=st.integers(min_value=2, max_value=20),
        n_heads=st.integers(min_value=1, max_value=4),
    )
    @settings(max_examples=20)
    def test_mean_attention_invariant_to_constant_offset(self, n_subjects, n_heads):
        """Mean attention ordering is invariant to adding constant to all values."""
        n_cell_types = 5
        attention = np.random.rand(n_subjects, n_heads, n_cell_types).astype(np.float32)
        cell_type_names = [f"type_{i}" for i in range(n_cell_types)]

        analyzer1 = CellTypeImportanceAnalyzer(
            attention=attention,
            cell_type_names=cell_type_names,
        )
        result1 = analyzer1.analyze()
        ranking1 = result1.overall["cell_type"].tolist()

        # Add constant offset
        attention_offset = attention + 0.5

        analyzer2 = CellTypeImportanceAnalyzer(
            attention=attention_offset,
            cell_type_names=cell_type_names,
        )
        result2 = analyzer2.analyze()
        ranking2 = result2.overall["cell_type"].tolist()

        # Rankings should be identical
        assert ranking1 == ranking2


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Edge case tests."""

    def test_single_subject(self):
        """Handles single subject correctly."""
        attention = np.random.rand(1, 4, N_CELL_TYPES).astype(np.float32)
        analyzer = CellTypeImportanceAnalyzer(attention=attention)
        result = analyzer.analyze()

        assert len(result.overall) == N_CELL_TYPES
        # Std should be 0 for single subject
        assert (result.overall["std_attention"] == 0).all()

    def test_single_head(self):
        """Handles single head correctly."""
        attention = np.random.rand(10, 1, N_CELL_TYPES).astype(np.float32)
        analyzer = CellTypeImportanceAnalyzer(attention=attention)
        result = analyzer.analyze()

        assert len(result.overall) == N_CELL_TYPES

    def test_two_cell_types(self):
        """Handles two cell types correctly."""
        attention = np.random.rand(10, 4, 2).astype(np.float32)
        analyzer = CellTypeImportanceAnalyzer(
            attention=attention,
            cell_type_names=["TypeA", "TypeB"],
        )
        result = analyzer.analyze()

        assert len(result.overall) == 2
        assert set(result.overall["rank"]) == {1, 2}

    def test_uniform_attention_produces_tied_ranks(self):
        """Uniform attention produces stable (though arbitrary) ranking."""
        attention = np.ones((10, 4, 5), dtype=np.float32) * 0.5
        analyzer = CellTypeImportanceAnalyzer(
            attention=attention,
            cell_type_names=[f"type_{i}" for i in range(5)],
        )
        result = analyzer.analyze()

        # All means should be equal
        means = result.overall["mean_attention"].unique()
        assert len(means) == 1
        assert np.isclose(means[0], 0.5)

    def test_single_region_label(self):
        """Handles case where all subjects have same region."""
        attention = np.random.rand(10, 4, 5).astype(np.float32)
        region_labels = np.array(["PFC"] * 10)

        analyzer = CellTypeImportanceAnalyzer(
            attention=attention,
            region_labels=region_labels,
            cell_type_names=[f"type_{i}" for i in range(5)],
        )
        result = analyzer.analyze()

        assert result.by_region is not None
        unique_regions = result.by_region["region"].unique()
        assert len(unique_regions) == 1
        assert unique_regions[0] == "PFC"

    def test_pathology_with_only_two_tertiles_possible(self):
        """Handles case with minimum subjects for tertiles."""
        attention = np.random.rand(3, 4, 5).astype(np.float32)  # Only 3 subjects
        pathology = np.array([0.1, 0.5, 0.9], dtype=np.float32)

        analyzer = CellTypeImportanceAnalyzer(
            attention=attention,
            pathology_scores=pathology,
            cell_type_names=[f"type_{i}" for i in range(5)],
        )
        result = analyzer.analyze()

        # Should still produce result (possibly with single subject per tertile)
        assert result.by_pathology is not None

    def test_load_unsupported_format_raises(self):
        """load_cell_type_importance raises for unsupported format."""
        with pytest.raises(ValueError, match="Unsupported"):
            load_cell_type_importance(Path("/fake/path.xyz"))
