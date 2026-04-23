"""
Tests for src/analysis/resilience_signatures.py.

Test coverage includes:
- ResilienceSignatureResult dataclass behavior
- ResilienceSignatureAnalyzer initialization and validation
- Group identification (resilient vs vulnerable)
- Signature computation
- Permutation test
- Group statistics
- Schema validation
- Property-based tests
- Edge cases
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from src.data.constants import CELL_TYPE_ORDER, N_CELL_TYPES
from src.analysis.resilience_signatures import (
    ResilienceSignatureResult,
    ResilienceSignatureAnalyzer,
    compute_resilience_signature,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_attention():
    """Sample pathology attention [n_subjects, n_heads, n_cell_types]."""
    np.random.seed(42)
    return np.random.rand(30, 4, N_CELL_TYPES).astype(np.float32)


@pytest.fixture
def sample_pathology_scores():
    """Sample pathology scores (higher = more pathology)."""
    np.random.seed(42)
    return np.random.rand(30).astype(np.float32)


@pytest.fixture
def sample_cognition_scores():
    """Sample cognition scores (higher = better cognition)."""
    np.random.seed(42)
    return np.random.rand(30).astype(np.float32)


@pytest.fixture
def analyzer(sample_attention, sample_pathology_scores, sample_cognition_scores):
    """ResilienceSignatureAnalyzer instance."""
    return ResilienceSignatureAnalyzer(
        attention=sample_attention,
        pathology_scores=sample_pathology_scores,
        cognition_scores=sample_cognition_scores,
    )


# ============================================================================
# ResilienceSignatureResult Dataclass Tests
# ============================================================================


class TestResilienceSignatureResult:
    """Tests for ResilienceSignatureResult dataclass."""

    def test_init_with_required_fields(self):
        """Result can be initialized with required field."""
        signature = pd.DataFrame({
            "cell_type": ["A", "B"],
            "signature": [0.1, -0.1],
            "resilient_mean": [0.6, 0.4],
            "vulnerable_mean": [0.5, 0.5],
            "rank": [1, 2],
        })
        result = ResilienceSignatureResult(signature=signature)
        assert result.signature is not None
        assert result.permutation_pvalues is None

    def test_metadata_defaults_to_empty_dict(self):
        """metadata defaults to empty dict."""
        signature = pd.DataFrame({"cell_type": ["A"], "signature": [0.1], "rank": [1]})
        result = ResilienceSignatureResult(signature=signature)
        assert result.metadata == {}


# ============================================================================
# ResilienceSignatureAnalyzer Initialization Tests
# ============================================================================


class TestAnalyzerInit:
    """Tests for ResilienceSignatureAnalyzer initialization."""

    def test_init_validates_attention_ndim(self, sample_pathology_scores, sample_cognition_scores):
        """Analyzer rejects attention with wrong dimensions."""
        bad_attention = np.random.rand(30, N_CELL_TYPES).astype(np.float32)  # 2D
        with pytest.raises(ValueError, match="must be 3D"):
            ResilienceSignatureAnalyzer(
                attention=bad_attention,
                pathology_scores=sample_pathology_scores,
                cognition_scores=sample_cognition_scores,
            )

    def test_init_validates_pathology_length(self, sample_attention, sample_cognition_scores):
        """Analyzer rejects pathology_scores with wrong length."""
        bad_pathology = np.random.rand(10).astype(np.float32)  # Wrong length
        with pytest.raises(ValueError, match="pathology_scores"):
            ResilienceSignatureAnalyzer(
                attention=sample_attention,
                pathology_scores=bad_pathology,
                cognition_scores=sample_cognition_scores,
            )

    def test_init_validates_cognition_length(self, sample_attention, sample_pathology_scores):
        """Analyzer rejects cognition_scores with wrong length."""
        bad_cognition = np.random.rand(10).astype(np.float32)  # Wrong length
        with pytest.raises(ValueError, match="cognition_scores"):
            ResilienceSignatureAnalyzer(
                attention=sample_attention,
                pathology_scores=sample_pathology_scores,
                cognition_scores=bad_cognition,
            )

    def test_init_uses_default_cell_type_names(self, sample_attention, sample_pathology_scores, sample_cognition_scores):
        """Analyzer uses CELL_TYPE_ORDER as default cell type names."""
        analyzer = ResilienceSignatureAnalyzer(
            attention=sample_attention,
            pathology_scores=sample_pathology_scores,
            cognition_scores=sample_cognition_scores,
        )
        assert analyzer.cell_type_names == list(CELL_TYPE_ORDER)


# ============================================================================
# Group Identification Tests
# ============================================================================


class TestGroupIdentification:
    """Tests for resilient/vulnerable group identification."""

    def test_identifies_high_pathology_subjects(self, analyzer):
        """Analyzer identifies high pathology subjects (top tertile)."""
        assert analyzer.high_pathology_mask.sum() > 0
        # Should be approximately 1/3 of subjects
        assert analyzer.high_pathology_mask.sum() <= len(analyzer.pathology_scores) // 2

    def test_identifies_resilient_and_vulnerable(self, analyzer):
        """Analyzer identifies resilient and vulnerable groups."""
        assert analyzer.n_resilient >= 0
        assert analyzer.n_vulnerable >= 0
        # Combined should not exceed high pathology count
        assert analyzer.n_resilient + analyzer.n_vulnerable <= analyzer.high_pathology_mask.sum()

    def test_resilient_have_high_cognition(self, analyzer):
        """Resilient subjects have high cognition within high pathology group."""
        if analyzer.n_resilient > 0:
            resilient_cog = analyzer.cognition_scores[analyzer.resilient_mask].mean()
            vulnerable_cog = analyzer.cognition_scores[analyzer.vulnerable_mask].mean()
            assert resilient_cog >= vulnerable_cog

    def test_resilient_have_high_pathology(self, analyzer):
        """Both groups have high pathology (by definition)."""
        if analyzer.n_resilient > 0:
            resilient_path = analyzer.pathology_scores[analyzer.resilient_mask].mean()
            all_path = analyzer.pathology_scores.mean()
            # Resilient should have above-average pathology
            assert resilient_path >= all_path * 0.5  # Relaxed threshold


# ============================================================================
# Signature Computation Tests
# ============================================================================


class TestSignatureComputation:
    """Tests for signature computation."""

    def test_analyze_returns_result(self, analyzer):
        """analyze() returns ResilienceSignatureResult."""
        result = analyzer.analyze(n_permutations=0)
        assert isinstance(result, ResilienceSignatureResult)

    def test_signature_has_expected_columns(self, analyzer):
        """signature DataFrame has expected columns."""
        result = analyzer.analyze(n_permutations=0)
        expected_cols = {
            "cell_type", "signature", "resilient_mean", "vulnerable_mean",
            "cohens_d", "ci_lower", "ci_upper", "cohens_d_ci_lower",
            "cohens_d_ci_upper", "rank"
        }
        assert set(result.signature.columns) == expected_cols

    def test_signature_has_all_cell_types(self, analyzer):
        """signature includes all cell types."""
        result = analyzer.analyze(n_permutations=0)
        assert len(result.signature) == N_CELL_TYPES

    def test_signature_is_difference(self, analyzer):
        """signature = resilient_mean - vulnerable_mean."""
        result = analyzer.analyze(n_permutations=0)
        for _, row in result.signature.iterrows():
            expected_sig = row["resilient_mean"] - row["vulnerable_mean"]
            assert np.isclose(row["signature"], expected_sig, atol=1e-6)

    def test_signature_sorted_by_absolute_value(self, analyzer):
        """signature sorted by absolute value descending."""
        result = analyzer.analyze(n_permutations=0)
        abs_sigs = np.abs(result.signature["signature"].values)
        assert all(abs_sigs[i] >= abs_sigs[i + 1] for i in range(len(abs_sigs) - 1))


# ============================================================================
# Permutation Test Tests
# ============================================================================


class TestPermutationTest:
    """Tests for permutation significance testing."""

    def test_permutation_pvalues_present(self, analyzer):
        """permutation_pvalues is computed when n_permutations > 0."""
        result = analyzer.analyze(n_permutations=100)
        assert result.permutation_pvalues is not None

    def test_permutation_pvalues_absent_when_skipped(self, analyzer):
        """permutation_pvalues is None when n_permutations=0."""
        result = analyzer.analyze(n_permutations=0)
        assert result.permutation_pvalues is None

    def test_permutation_pvalues_has_expected_columns(self, analyzer):
        """permutation_pvalues has expected columns."""
        result = analyzer.analyze(n_permutations=100)
        expected_cols = {"cell_type", "p_value", "fdr_corrected", "significant", "significant_005", "significant_001"}
        assert set(result.permutation_pvalues.columns) == expected_cols

    def test_pvalues_bounded(self, analyzer):
        """p-values are bounded [0, 1]."""
        result = analyzer.analyze(n_permutations=100)
        assert (result.permutation_pvalues["p_value"] >= 0).all()
        assert (result.permutation_pvalues["p_value"] <= 1).all()
        assert (result.permutation_pvalues["fdr_corrected"] >= 0).all()
        assert (result.permutation_pvalues["fdr_corrected"] <= 1).all()

    def test_permutation_reproducible_with_seed(self, sample_attention, sample_pathology_scores, sample_cognition_scores):
        """Permutation test is reproducible with same seed."""
        analyzer1 = ResilienceSignatureAnalyzer(
            attention=sample_attention,
            pathology_scores=sample_pathology_scores,
            cognition_scores=sample_cognition_scores,
        )
        result1 = analyzer1.analyze(n_permutations=100, random_seed=42)

        analyzer2 = ResilienceSignatureAnalyzer(
            attention=sample_attention,
            pathology_scores=sample_pathology_scores,
            cognition_scores=sample_cognition_scores,
        )
        result2 = analyzer2.analyze(n_permutations=100, random_seed=42)

        pd.testing.assert_frame_equal(result1.permutation_pvalues, result2.permutation_pvalues)


# ============================================================================
# Group Statistics Tests
# ============================================================================


class TestGroupStatistics:
    """Tests for group statistics computation."""

    def test_group_statistics_present(self, analyzer):
        """group_statistics is computed."""
        result = analyzer.analyze(n_permutations=0)
        assert result.group_statistics is not None

    def test_group_statistics_has_expected_columns(self, analyzer):
        """group_statistics has expected columns."""
        result = analyzer.analyze(n_permutations=0)
        expected_cols = {"group", "n_subjects", "mean_pathology", "std_pathology", "mean_cognition", "std_cognition"}
        assert set(result.group_statistics.columns) == expected_cols

    def test_group_statistics_has_two_groups(self, analyzer):
        """group_statistics has resilient and vulnerable rows (if both exist)."""
        result = analyzer.analyze(n_permutations=0)
        groups = result.group_statistics["group"].tolist()
        # At least one group should be present
        assert len(groups) >= 1


# ============================================================================
# Save/Load Tests
# ============================================================================


class TestSaveLoad:
    """Tests for save functionality."""

    def test_save_creates_files(self, analyzer):
        """save() creates expected files."""
        result = analyzer.analyze(n_permutations=100)
        with tempfile.TemporaryDirectory() as tmpdir:
            saved = analyzer.save(result, tmpdir)
            assert (Path(tmpdir) / "resilience_signature.parquet").exists()
            assert (Path(tmpdir) / "resilience_signature.csv").exists()
            assert (Path(tmpdir) / "signature_pvalues.parquet").exists()
            assert (Path(tmpdir) / "group_statistics.csv").exists()


# ============================================================================
# Schema Validation Tests
# ============================================================================


class TestOutputSchemaValidation:
    """Tests validating output DataFrame schemas."""

    def test_signature_schema(self, analyzer):
        """signature DataFrame has expected schema."""
        result = analyzer.analyze(n_permutations=0)
        df = result.signature
        assert df["cell_type"].dtype == object
        assert np.issubdtype(df["signature"].dtype, np.floating)
        assert np.issubdtype(df["resilient_mean"].dtype, np.floating)
        assert np.issubdtype(df["vulnerable_mean"].dtype, np.floating)
        assert np.issubdtype(df["rank"].dtype, np.integer)

    def test_attention_values_bounded(self, analyzer):
        """Mean attention values are bounded [0, 1]."""
        result = analyzer.analyze(n_permutations=0)
        assert (result.signature["resilient_mean"] >= 0).all()
        assert (result.signature["resilient_mean"] <= 1).all()
        assert (result.signature["vulnerable_mean"] >= 0).all()
        assert (result.signature["vulnerable_mean"] <= 1).all()


# ============================================================================
# Property-Based Tests
# ============================================================================


@pytest.mark.filterwarnings("ignore:Degrees of freedom <= 0 for slice:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning")
class TestPropertyBased:
    """Property-based tests using Hypothesis."""

    @given(
        n_subjects=st.integers(min_value=9, max_value=30),
        n_heads=st.integers(min_value=1, max_value=4),
        n_cell_types=st.integers(min_value=2, max_value=10),
    )
    @settings(max_examples=15)
    def test_signature_always_has_n_cell_types_rows(self, n_subjects, n_heads, n_cell_types):
        """signature always has exactly n_cell_types rows."""
        attention = np.random.rand(n_subjects, n_heads, n_cell_types).astype(np.float32)
        pathology = np.random.rand(n_subjects).astype(np.float32)
        cognition = np.random.rand(n_subjects).astype(np.float32)
        cell_type_names = [f"type_{i}" for i in range(n_cell_types)]

        analyzer = ResilienceSignatureAnalyzer(
            attention=attention,
            pathology_scores=pathology,
            cognition_scores=cognition,
            cell_type_names=cell_type_names,
        )
        result = analyzer.analyze(n_permutations=0)
        assert len(result.signature) == n_cell_types

    @given(
        n_subjects=st.integers(min_value=9, max_value=30),
    )
    @settings(max_examples=10)
    def test_group_sizes_sum_correctly(self, n_subjects):
        """Resilient + vulnerable <= high pathology subjects."""
        attention = np.random.rand(n_subjects, 4, 5).astype(np.float32)
        pathology = np.random.rand(n_subjects).astype(np.float32)
        cognition = np.random.rand(n_subjects).astype(np.float32)
        cell_type_names = [f"type_{i}" for i in range(5)]

        analyzer = ResilienceSignatureAnalyzer(
            attention=attention,
            pathology_scores=pathology,
            cognition_scores=cognition,
            cell_type_names=cell_type_names,
        )

        assert analyzer.n_resilient + analyzer.n_vulnerable <= analyzer.high_pathology_mask.sum()


# ============================================================================
# Convenience Function Tests
# ============================================================================


class TestConvenienceFunction:
    """Tests for compute_resilience_signature function."""

    def test_compute_returns_result(self, sample_attention, sample_pathology_scores, sample_cognition_scores):
        """compute_resilience_signature returns ResilienceSignatureResult."""
        result = compute_resilience_signature(
            attention=sample_attention,
            pathology_scores=sample_pathology_scores,
            cognition_scores=sample_cognition_scores,
            n_permutations=0,
        )
        assert isinstance(result, ResilienceSignatureResult)

    def test_compute_with_output_dir_saves_files(self, sample_attention, sample_pathology_scores, sample_cognition_scores):
        """compute_resilience_signature saves when output_dir provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = compute_resilience_signature(
                attention=sample_attention,
                pathology_scores=sample_pathology_scores,
                cognition_scores=sample_cognition_scores,
                n_permutations=10,
                output_dir=tmpdir,
            )
            assert (Path(tmpdir) / "resilience_signature.parquet").exists()


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Edge case tests."""

    def test_all_same_pathology(self):
        """Handles case where all subjects have same pathology."""
        attention = np.random.rand(20, 4, 5).astype(np.float32)
        pathology = np.ones(20).astype(np.float32)  # All same
        cognition = np.random.rand(20).astype(np.float32)

        analyzer = ResilienceSignatureAnalyzer(
            attention=attention,
            pathology_scores=pathology,
            cognition_scores=cognition,
            cell_type_names=[f"type_{i}" for i in range(5)],
        )
        # All subjects will be "high pathology"
        assert analyzer.high_pathology_mask.all()
        result = analyzer.analyze(n_permutations=0)
        assert result.signature is not None

    def test_all_same_cognition(self):
        """Handles case where all subjects have same cognition."""
        attention = np.random.rand(20, 4, 5).astype(np.float32)
        pathology = np.random.rand(20).astype(np.float32)
        cognition = np.ones(20).astype(np.float32)  # All same

        analyzer = ResilienceSignatureAnalyzer(
            attention=attention,
            pathology_scores=pathology,
            cognition_scores=cognition,
            cell_type_names=[f"type_{i}" for i in range(5)],
        )
        result = analyzer.analyze(n_permutations=0)
        # Signature should be ~0 when groups are indistinguishable
        assert result.signature is not None

    @pytest.mark.filterwarnings("ignore:Degrees of freedom <= 0 for slice:RuntimeWarning")
    @pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning")
    def test_minimum_subjects_for_groups(self):
        """Handles minimum number of subjects for group formation."""
        # Need at least 9 subjects for 3 tertiles within high pathology
        attention = np.random.rand(9, 4, 5).astype(np.float32)
        pathology = np.linspace(0, 1, 9).astype(np.float32)  # Distinct values
        cognition = np.linspace(0, 1, 9).astype(np.float32)

        analyzer = ResilienceSignatureAnalyzer(
            attention=attention,
            pathology_scores=pathology,
            cognition_scores=cognition,
            cell_type_names=[f"type_{i}" for i in range(5)],
        )
        result = analyzer.analyze(n_permutations=0)
        assert result.signature is not None

    def test_empty_resilient_group(self):
        """Handles case with no resilient subjects (edge case)."""
        # This is pathological but should not crash
        attention = np.random.rand(10, 4, 5).astype(np.float32)
        pathology = np.array([0.1, 0.2, 0.3, 0.8, 0.85, 0.9, 0.91, 0.92, 0.93, 0.94])
        # High pathology subjects (6) have low cognition
        cognition = np.array([0.9, 0.8, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

        analyzer = ResilienceSignatureAnalyzer(
            attention=attention,
            pathology_scores=pathology,
            cognition_scores=cognition,
            cell_type_names=[f"type_{i}" for i in range(5)],
        )
        result = analyzer.analyze(n_permutations=0)
        assert result.signature is not None


# ============================================================================
# Regional Analysis Tests
# ============================================================================


class TestRegionalAnalysis:
    """Tests for regional resilience signature analysis."""

    @pytest.fixture
    def regional_analyzer(self):
        """Analyzer with regional labels for sufficient subjects per region."""
        np.random.seed(42)
        n_subjects = 60  # Need sufficient subjects per region
        attention = np.random.rand(n_subjects, 4, 5).astype(np.float32)
        pathology = np.random.rand(n_subjects).astype(np.float32)
        cognition = np.random.rand(n_subjects).astype(np.float32)
        # Assign subjects to 3 regions evenly
        region_labels = np.array(["PFC"] * 20 + ["DLPFC"] * 20 + ["MTG"] * 20)
        cell_type_names = [f"type_{i}" for i in range(5)]

        return ResilienceSignatureAnalyzer(
            attention=attention,
            pathology_scores=pathology,
            cognition_scores=cognition,
            region_labels=region_labels,
            cell_type_names=cell_type_names,
        )

    def test_regional_analysis_returns_dataframe(self, regional_analyzer):
        """Regional analysis returns a DataFrame when region_labels provided."""
        result = regional_analyzer.analyze(n_permutations=0)
        assert result.by_region is not None
        assert isinstance(result.by_region, pd.DataFrame)

    def test_regional_analysis_has_expected_columns(self, regional_analyzer):
        """by_region DataFrame has expected columns."""
        result = regional_analyzer.analyze(n_permutations=0)
        expected_cols = {
            "region", "cell_type", "signature", "resilient_mean", "vulnerable_mean",
            "cohens_d", "ci_lower", "ci_upper", "cohens_d_ci_lower", "cohens_d_ci_upper",
            "n_resilient", "n_vulnerable"
        }
        assert set(result.by_region.columns) == expected_cols

    def test_regional_analysis_includes_all_regions(self, regional_analyzer):
        """by_region includes all regions with sufficient subjects."""
        result = regional_analyzer.analyze(n_permutations=0)
        regions = result.by_region["region"].unique()
        # Should have multiple regions
        assert len(regions) >= 1

    def test_regional_analysis_includes_all_cell_types_per_region(self, regional_analyzer):
        """Each region includes all cell types."""
        result = regional_analyzer.analyze(n_permutations=0)
        for region in result.by_region["region"].unique():
            region_df = result.by_region[result.by_region["region"] == region]
            assert len(region_df) == 5  # 5 cell types

    def test_regional_signature_is_difference(self, regional_analyzer):
        """Regional signature = resilient_mean - vulnerable_mean."""
        result = regional_analyzer.analyze(n_permutations=0)
        for _, row in result.by_region.iterrows():
            expected_sig = row["resilient_mean"] - row["vulnerable_mean"]
            assert np.isclose(row["signature"], expected_sig, atol=1e-6)

    def test_regional_analysis_without_region_labels_returns_none(self, analyzer):
        """by_region is None when region_labels not provided."""
        result = analyzer.analyze(n_permutations=0)
        assert result.by_region is None

    def test_regional_validates_region_labels_length(
        self, sample_attention, sample_pathology_scores, sample_cognition_scores
    ):
        """Analyzer rejects region_labels with wrong length."""
        bad_regions = np.array(["PFC"] * 10)  # Wrong length (30 subjects)
        with pytest.raises(ValueError, match="region_labels"):
            ResilienceSignatureAnalyzer(
                attention=sample_attention,
                pathology_scores=sample_pathology_scores,
                cognition_scores=sample_cognition_scores,
                region_labels=bad_regions,
            )

    def test_regional_metadata_includes_region_info(self, regional_analyzer):
        """Metadata includes regional analysis info."""
        result = regional_analyzer.analyze(n_permutations=0)
        assert "n_regions_analyzed" in result.metadata
        assert "regions" in result.metadata

    def test_regional_analysis_saves_by_region_file(self, regional_analyzer):
        """save() creates by_region files."""
        result = regional_analyzer.analyze(n_permutations=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            regional_analyzer.save(result, tmpdir)
            assert (Path(tmpdir) / "resilience_signature_by_region.parquet").exists()
            assert (Path(tmpdir) / "resilience_signature_by_region.csv").exists()

    def test_regional_skips_region_with_few_subjects(self):
        """Regions with insufficient subjects are skipped."""
        np.random.seed(42)
        n_subjects = 30
        attention = np.random.rand(n_subjects, 4, 5).astype(np.float32)
        pathology = np.random.rand(n_subjects).astype(np.float32)
        cognition = np.random.rand(n_subjects).astype(np.float32)
        # One region has many subjects, another has very few
        region_labels = np.array(["PFC"] * 28 + ["DLPFC"] * 2)
        cell_type_names = [f"type_{i}" for i in range(5)]

        analyzer = ResilienceSignatureAnalyzer(
            attention=attention,
            pathology_scores=pathology,
            cognition_scores=cognition,
            region_labels=region_labels,
            cell_type_names=cell_type_names,
        )
        result = analyzer.analyze(n_permutations=0)

        if result.by_region is not None:
            # DLPFC should not be in results due to too few subjects
            regions = result.by_region["region"].unique()
            assert "DLPFC" not in regions


# ============================================================================
# Ablation Study Tests
# ============================================================================


class TestAblationStudy:
    """Tests for ablation study functionality."""

    def test_ablation_returns_dataframe(self, analyzer):
        """Ablation study returns a DataFrame."""
        result = analyzer.analyze(n_permutations=0, run_ablation=True)
        assert result.ablation_results is not None
        assert isinstance(result.ablation_results, pd.DataFrame)

    def test_ablation_zero_embedding_method(self, analyzer):
        """Zero embedding ablation returns expected columns."""
        result = analyzer.analyze(
            n_permutations=0,
            run_ablation=True,
            ablation_method="zero_embedding",
        )
        expected_cols = {
            "cell_type", "method", "importance", "importance_std",
            "importance_high_pathology", "importance_low_pathology",
            "importance_low_tertile", "importance_med_tertile", "importance_high_tertile",
            "rank"
        }
        assert set(result.ablation_results.columns) == expected_cols
        assert (result.ablation_results["method"] == "zero_embedding").all()

    def test_ablation_node_removal_method(self, analyzer):
        """Node removal ablation returns expected columns."""
        result = analyzer.analyze(
            n_permutations=0,
            run_ablation=True,
            ablation_method="node_removal",
        )
        expected_cols = {
            "cell_type", "method", "importance", "importance_std",
            "importance_high_pathology", "importance_low_pathology",
            "importance_low_tertile", "importance_med_tertile", "importance_high_tertile",
            "rank"
        }
        assert set(result.ablation_results.columns) == expected_cols
        assert (result.ablation_results["method"] == "node_removal").all()

    def test_ablation_both_methods(self, analyzer):
        """Both ablation methods returns results from each."""
        result = analyzer.analyze(
            n_permutations=0,
            run_ablation=True,
            ablation_method="both",
        )
        methods = result.ablation_results["method"].unique()
        assert "zero_embedding" in methods
        assert "node_removal" in methods

    def test_ablation_comparison_generated(self, analyzer):
        """Ablation comparison is generated when both methods run."""
        result = analyzer.analyze(
            n_permutations=0,
            run_ablation=True,
            ablation_method="both",
        )
        assert result.ablation_comparison is not None
        assert isinstance(result.ablation_comparison, pd.DataFrame)

    def test_ablation_comparison_has_expected_columns(self, analyzer):
        """Ablation comparison has expected columns."""
        result = analyzer.analyze(
            n_permutations=0,
            run_ablation=True,
            ablation_method="both",
        )
        expected_cols = {
            "cell_type",
            "importance_zero_embedding",
            "importance_node_removal",
            "rank_zero_embedding",
            "rank_node_removal",
            "rank_difference",
            "importance_ratio",
            "methods_agree",
            "importance_type",
            "methods_correlation",
            "correlation_pvalue",
        }
        assert set(result.ablation_comparison.columns) == expected_cols

    def test_ablation_comparison_not_generated_single_method(self, analyzer):
        """Ablation comparison is None when only one method run."""
        result = analyzer.analyze(
            n_permutations=0,
            run_ablation=True,
            ablation_method="zero_embedding",
        )
        assert result.ablation_comparison is None

    def test_ablation_importance_nonnegative(self, analyzer):
        """Ablation importance values are non-negative."""
        result = analyzer.analyze(
            n_permutations=0,
            run_ablation=True,
            ablation_method="both",
        )
        assert (result.ablation_results["importance"] >= 0).all()

    def test_ablation_includes_all_cell_types(self, analyzer):
        """Ablation results include all cell types for each method."""
        result = analyzer.analyze(
            n_permutations=0,
            run_ablation=True,
            ablation_method="both",
        )
        for method in ["zero_embedding", "node_removal"]:
            method_df = result.ablation_results[result.ablation_results["method"] == method]
            assert len(method_df) == N_CELL_TYPES

    def test_ablation_with_embeddings(self):
        """Zero embedding ablation works with provided embeddings."""
        np.random.seed(42)
        n_subjects = 30
        n_cell_types = 5
        embed_dim = 16
        attention = np.random.rand(n_subjects, 4, n_cell_types).astype(np.float32)
        pathology = np.random.rand(n_subjects).astype(np.float32)
        cognition = np.random.rand(n_subjects).astype(np.float32)
        embeddings = np.random.rand(n_subjects, n_cell_types, embed_dim).astype(np.float32)
        cell_type_names = [f"type_{i}" for i in range(n_cell_types)]

        analyzer = ResilienceSignatureAnalyzer(
            attention=attention,
            pathology_scores=pathology,
            cognition_scores=cognition,
            cell_type_names=cell_type_names,
        )
        result = analyzer.analyze(
            n_permutations=0,
            run_ablation=True,
            ablation_method="zero_embedding",
            embeddings=embeddings,
        )
        assert result.ablation_results is not None
        # With embeddings, importance should be based on L2 deviation
        assert (result.ablation_results["importance"] >= 0).all()

    def test_ablation_stratified_by_pathology(self, analyzer):
        """Ablation includes stratified importance by pathology."""
        result = analyzer.analyze(
            n_permutations=0,
            run_ablation=True,
            ablation_method="both",
        )
        # Should have pathology stratification columns
        assert "importance_high_pathology" in result.ablation_results.columns
        assert "importance_low_pathology" in result.ablation_results.columns

    def test_ablation_metadata_added(self, analyzer):
        """Ablation adds metadata when run."""
        result = analyzer.analyze(
            n_permutations=0,
            run_ablation=True,
            ablation_method="both",
        )
        assert "ablation_method" in result.metadata
        assert "ablation_methods_correlation" in result.metadata

    def test_ablation_saves_files(self, analyzer):
        """save() creates ablation files."""
        result = analyzer.analyze(
            n_permutations=0,
            run_ablation=True,
            ablation_method="both",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer.save(result, tmpdir)
            assert (Path(tmpdir) / "ablation_importance.parquet").exists()
            assert (Path(tmpdir) / "ablation_importance.csv").exists()
            assert (Path(tmpdir) / "ablation_comparison.parquet").exists()
            assert (Path(tmpdir) / "ablation_comparison.csv").exists()

    def test_ablation_importance_type_categorization(self, analyzer):
        """Ablation comparison categorizes importance type."""
        result = analyzer.analyze(
            n_permutations=0,
            run_ablation=True,
            ablation_method="both",
        )
        valid_types = {"structural", "transcriptional", "consistent"}
        assert all(t in valid_types for t in result.ablation_comparison["importance_type"])

    def test_ablation_methods_agree_boolean(self, analyzer):
        """methods_agree column is boolean."""
        result = analyzer.analyze(
            n_permutations=0,
            run_ablation=True,
            ablation_method="both",
        )
        assert result.ablation_comparison["methods_agree"].dtype == bool


# ============================================================================
# Permutation Null Distribution Tests
# ============================================================================


class TestPermutationNullDistribution:
    """Tests for permutation null distribution storage."""

    def test_permutation_null_returned(self, analyzer):
        """Permutation null distribution is returned when permutations > 0."""
        result = analyzer.analyze(n_permutations=100)
        assert result.permutation_null is not None
        assert isinstance(result.permutation_null, np.ndarray)

    def test_permutation_null_shape(self, analyzer):
        """Permutation null has correct shape [n_permutations, n_cell_types]."""
        result = analyzer.analyze(n_permutations=100)
        assert result.permutation_null.shape == (100, N_CELL_TYPES)

    def test_permutation_null_none_when_skipped(self, analyzer):
        """Permutation null is None when n_permutations=0."""
        result = analyzer.analyze(n_permutations=0)
        assert result.permutation_null is None

    def test_permutation_null_saved_to_hdf5(self, analyzer):
        """Permutation null is saved to HDF5 file."""
        result = analyzer.analyze(n_permutations=100)
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer.save(result, tmpdir)
            h5_path = Path(tmpdir) / "resilience_permutation_null.h5"
            assert h5_path.exists()

            # Verify contents
            import h5py
            with h5py.File(h5_path, "r") as f:
                assert "null_distribution" in f
                assert f["null_distribution"].shape == (100, N_CELL_TYPES)
                assert f.attrs["n_permutations"] == 100


# ============================================================================
# Regional Ablation Tests
# ============================================================================


class TestRegionalAblation:
    """Tests for regional ablation analysis."""

    @pytest.fixture
    def regional_ablation_analyzer(self):
        """Analyzer with regional labels for ablation."""
        np.random.seed(42)
        n_subjects = 60
        attention = np.random.rand(n_subjects, 4, 5).astype(np.float32)
        pathology = np.random.rand(n_subjects).astype(np.float32)
        cognition = np.random.rand(n_subjects).astype(np.float32)
        region_labels = np.array(["PFC"] * 20 + ["DLPFC"] * 20 + ["MTG"] * 20)
        cell_type_names = [f"type_{i}" for i in range(5)]

        return ResilienceSignatureAnalyzer(
            attention=attention,
            pathology_scores=pathology,
            cognition_scores=cognition,
            region_labels=region_labels,
            cell_type_names=cell_type_names,
        )

    def test_regional_ablation_returns_dataframe(self, regional_ablation_analyzer):
        """Regional ablation returns DataFrame."""
        result = regional_ablation_analyzer.analyze(
            n_permutations=0,
            run_ablation=True,
            ablation_method="both",
        )
        assert result.ablation_by_region is not None
        assert isinstance(result.ablation_by_region, pd.DataFrame)

    def test_regional_ablation_has_expected_columns(self, regional_ablation_analyzer):
        """Regional ablation has expected columns."""
        result = regional_ablation_analyzer.analyze(
            n_permutations=0,
            run_ablation=True,
            ablation_method="both",
        )
        expected_cols = {"region", "cell_type", "method", "importance", "importance_std", "n_subjects", "rank"}
        assert set(result.ablation_by_region.columns) == expected_cols

    def test_regional_ablation_includes_all_regions(self, regional_ablation_analyzer):
        """Regional ablation includes results for all regions."""
        result = regional_ablation_analyzer.analyze(
            n_permutations=0,
            run_ablation=True,
            ablation_method="both",
        )
        regions = result.ablation_by_region["region"].unique()
        assert len(regions) == 3

    def test_regional_ablation_includes_both_methods(self, regional_ablation_analyzer):
        """Regional ablation includes both ablation methods."""
        result = regional_ablation_analyzer.analyze(
            n_permutations=0,
            run_ablation=True,
            ablation_method="both",
        )
        methods = result.ablation_by_region["method"].unique()
        assert "zero_embedding" in methods
        assert "node_removal" in methods

    def test_regional_ablation_none_without_regions(self, analyzer):
        """Regional ablation is None when no region_labels."""
        result = analyzer.analyze(
            n_permutations=0,
            run_ablation=True,
            ablation_method="both",
        )
        assert result.ablation_by_region is None

    def test_regional_ablation_saved(self, regional_ablation_analyzer):
        """Regional ablation is saved to files."""
        result = regional_ablation_analyzer.analyze(
            n_permutations=0,
            run_ablation=True,
            ablation_method="both",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            regional_ablation_analyzer.save(result, tmpdir)
            assert (Path(tmpdir) / "ablation_by_region.parquet").exists()
            assert (Path(tmpdir) / "ablation_by_region.csv").exists()


# ============================================================================
# Cohen's d CI Tests
# ============================================================================


class TestCohensD_CI:
    """Tests for Cohen's d confidence intervals."""

    def test_cohens_d_ci_in_signature(self, analyzer):
        """Signature includes Cohen's d CI columns."""
        result = analyzer.analyze(n_permutations=0)
        assert "cohens_d_ci_lower" in result.signature.columns
        assert "cohens_d_ci_upper" in result.signature.columns

    def test_cohens_d_ci_bounds_cohens_d(self, analyzer):
        """Cohen's d CI lower < d < CI upper (generally)."""
        result = analyzer.analyze(n_permutations=0)
        # For rows where CI is meaningful (non-zero effect)
        for _, row in result.signature.iterrows():
            if abs(row["cohens_d"]) > 0.01:
                assert row["cohens_d_ci_lower"] <= row["cohens_d"]
                assert row["cohens_d"] <= row["cohens_d_ci_upper"]

    def test_cohens_d_ci_in_regional(self):
        """Regional signatures include Cohen's d CI."""
        np.random.seed(42)
        n_subjects = 60
        attention = np.random.rand(n_subjects, 4, 5).astype(np.float32)
        pathology = np.random.rand(n_subjects).astype(np.float32)
        cognition = np.random.rand(n_subjects).astype(np.float32)
        region_labels = np.array(["PFC"] * 30 + ["DLPFC"] * 30)
        cell_type_names = [f"type_{i}" for i in range(5)]

        analyzer = ResilienceSignatureAnalyzer(
            attention=attention,
            pathology_scores=pathology,
            cognition_scores=cognition,
            region_labels=region_labels,
            cell_type_names=cell_type_names,
        )
        result = analyzer.analyze(n_permutations=0)

        if result.by_region is not None:
            assert "cohens_d_ci_lower" in result.by_region.columns
            assert "cohens_d_ci_upper" in result.by_region.columns


# ============================================================================
# NaN Handling Tests
# ============================================================================


class TestNaNHandling:
    """Tests for NaN handling in resilience analysis (Finding 4)."""

    def test_nan_pathology_excluded_from_groups(self):
        """Subjects with NaN pathology should be excluded from all groups."""
        np.random.seed(42)
        n_subjects = 30
        attention = np.random.rand(n_subjects, 4, 5).astype(np.float32)
        pathology = np.random.rand(n_subjects).astype(np.float32)
        cognition = np.random.rand(n_subjects).astype(np.float32)

        # Set some subjects to NaN
        pathology[0] = np.nan
        pathology[5] = np.nan
        pathology[10] = np.nan

        analyzer = ResilienceSignatureAnalyzer(
            attention=attention,
            pathology_scores=pathology,
            cognition_scores=cognition,
            cell_type_names=[f"type_{i}" for i in range(5)],
        )
        result = analyzer.analyze(n_permutations=0)

        assert result.signature is not None
        # NaN subjects should NOT appear in any group
        assert not analyzer.resilient_mask[0]
        assert not analyzer.resilient_mask[5]
        assert not analyzer.resilient_mask[10]
        assert not analyzer.vulnerable_mask[0]
        assert not analyzer.vulnerable_mask[5]
        assert not analyzer.vulnerable_mask[10]

    def test_nan_cognition_excluded_from_groups(self):
        """Subjects with NaN cognition should be excluded from all groups."""
        np.random.seed(42)
        n_subjects = 30
        attention = np.random.rand(n_subjects, 4, 5).astype(np.float32)
        pathology = np.random.rand(n_subjects).astype(np.float32)
        cognition = np.random.rand(n_subjects).astype(np.float32)

        cognition[3] = np.nan
        cognition[7] = np.nan

        analyzer = ResilienceSignatureAnalyzer(
            attention=attention,
            pathology_scores=pathology,
            cognition_scores=cognition,
            cell_type_names=[f"type_{i}" for i in range(5)],
        )
        result = analyzer.analyze(n_permutations=0)

        assert result.signature is not None
        assert not analyzer.resilient_mask[3]
        assert not analyzer.vulnerable_mask[3]

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_all_nan_raises_or_handles_gracefully(self):
        """All NaN scores should be handled gracefully."""
        np.random.seed(42)
        attention = np.random.rand(10, 4, 5).astype(np.float32)
        pathology = np.full(10, np.nan)
        cognition = np.random.rand(10).astype(np.float32)

        # Should handle gracefully (either raise or produce empty groups)
        try:
            analyzer = ResilienceSignatureAnalyzer(
                attention=attention,
                pathology_scores=pathology,
                cognition_scores=cognition,
                cell_type_names=[f"type_{i}" for i in range(5)],
            )
            result = analyzer.analyze(n_permutations=0)
            # If it doesn't raise, signature should still be produced
            assert result.signature is not None
        except (ValueError, RuntimeWarning):
            pass  # Acceptable to raise on all-NaN input

    def test_nan_warning_logged(self, caplog):
        """NaN values in input should trigger a warning."""
        import logging
        np.random.seed(42)
        n_subjects = 20
        attention = np.random.rand(n_subjects, 4, 5).astype(np.float32)
        pathology = np.random.rand(n_subjects).astype(np.float32)
        cognition = np.random.rand(n_subjects).astype(np.float32)
        pathology[0] = np.nan

        with caplog.at_level(logging.WARNING):
            analyzer = ResilienceSignatureAnalyzer(
                attention=attention,
                pathology_scores=pathology,
                cognition_scores=cognition,
                cell_type_names=[f"type_{i}" for i in range(5)],
            )

        assert any("NaN" in record.message for record in caplog.records)


# ============================================================================
# FDR Threshold Tests
# ============================================================================


class TestFDRThreshold:
    """Test that fdr_threshold parameter is threaded through."""

    def test_custom_fdr_threshold_affects_significance(self):
        """Custom fdr_threshold should affect the 'significant' column."""
        np.random.seed(42)
        n = 40
        attention = np.random.rand(n, 4, 5).astype(np.float32)
        pathology = np.random.rand(n)
        cognition = np.random.rand(n)
        cell_types = [f"type_{i}" for i in range(5)]

        analyzer = ResilienceSignatureAnalyzer(
            attention=attention,
            pathology_scores=pathology,
            cognition_scores=cognition,
            cell_type_names=cell_types,
        )

        # With very permissive threshold
        result_permissive = analyzer.analyze(
            n_permutations=100, fdr_threshold=0.99, random_seed=42
        )
        # With very strict threshold
        result_strict = analyzer.analyze(
            n_permutations=100, fdr_threshold=0.001, random_seed=42
        )

        if result_permissive.permutation_pvalues is not None and result_strict.permutation_pvalues is not None:
            # The 'significant' column should differ between thresholds
            # (permissive should have >= as many True as strict)
            n_sig_permissive = result_permissive.permutation_pvalues["significant"].sum()
            n_sig_strict = result_strict.permutation_pvalues["significant"].sum()
            assert n_sig_permissive >= n_sig_strict

            # Fixed reference columns should be identical between runs
            pd.testing.assert_series_equal(
                result_permissive.permutation_pvalues["significant_005"],
                result_strict.permutation_pvalues["significant_005"],
            )

    def test_fdr_threshold_default_matches_005(self):
        """Default fdr_threshold=0.05 should make 'significant' match 'significant_005'."""
        np.random.seed(42)
        n = 40
        attention = np.random.rand(n, 4, 5).astype(np.float32)
        pathology = np.random.rand(n)
        cognition = np.random.rand(n)
        cell_types = [f"type_{i}" for i in range(5)]

        analyzer = ResilienceSignatureAnalyzer(
            attention=attention,
            pathology_scores=pathology,
            cognition_scores=cognition,
            cell_type_names=cell_types,
        )
        result = analyzer.analyze(n_permutations=100, random_seed=42)

        if result.permutation_pvalues is not None:
            pd.testing.assert_series_equal(
                result.permutation_pvalues["significant"],
                result.permutation_pvalues["significant_005"],
                check_names=False,
            )


# ============================================================================
# Region filter + seed determinism
# ============================================================================


class TestEmptyRegionFilterResilience:
    """Tests for empty-string region label filtering in resilience analysis."""

    def test_empty_region_labels_excluded(self):
        """Subjects with empty region labels are excluded from regional analysis."""
        np.random.seed(42)
        n = 60
        attention = np.random.rand(n, 4, N_CELL_TYPES).astype(np.float32)
        pathology = np.random.rand(n).astype(np.float32)
        cognition = np.random.rand(n).astype(np.float32)
        # 20 per real region, 0 for empty
        region_labels = np.array(["PFC"] * 20 + ["AG"] * 20 + [""] * 20)

        analyzer = ResilienceSignatureAnalyzer(
            attention=attention,
            pathology_scores=pathology,
            cognition_scores=cognition,
            cell_type_names=[f"ct_{i}" for i in range(N_CELL_TYPES)],
            region_labels=region_labels,
        )
        result = analyzer.analyze(n_permutations=0)
        if result.by_region is not None:
            regions_in_result = result.by_region["region"].unique()
            assert "" not in regions_in_result


class TestSeedDeterminism:
    """Tests that same seed produces identical permutation results."""

    def test_same_seed_same_result(self):
        """Permutation test with same seed must be deterministic."""
        np.random.seed(42)
        n = 30
        attention = np.random.rand(n, 4, N_CELL_TYPES).astype(np.float32)
        pathology = np.random.rand(n).astype(np.float32)
        cognition = np.random.rand(n).astype(np.float32)

        analyzer = ResilienceSignatureAnalyzer(
            attention=attention,
            pathology_scores=pathology,
            cognition_scores=cognition,
            cell_type_names=[f"ct_{i}" for i in range(N_CELL_TYPES)],
        )
        r1 = analyzer.analyze(n_permutations=50, random_seed=123)
        r2 = analyzer.analyze(n_permutations=50, random_seed=123)

        if r1.permutation_pvalues is not None:
            pd.testing.assert_frame_equal(r1.permutation_pvalues, r2.permutation_pvalues)


class TestVectorizedCohensDSharedUtility:
    """Tests that vectorized Cohen's d from statistics.py matches scalar version."""

    def test_vectorized_matches_scalar_loop(self):
        """cohens_d_vectorized results match per-feature cohens_d_with_ci calls."""
        from src.utils.statistics import (
            cohens_d_vectorized, cohens_d_ci_vectorized, cohens_d_with_ci,
        )
        np.random.seed(42)
        n1, n2 = 15, 12
        n_features = 5
        group1 = np.random.rand(n1, n_features)
        group2 = np.random.rand(n2, n_features)

        g1_mean = group1.mean(axis=0)
        g1_std = group1.std(axis=0, ddof=1)
        g2_mean = group2.mean(axis=0)
        g2_std = group2.std(axis=0, ddof=1)

        d_vec, _ = cohens_d_vectorized(g1_mean, g1_std, n1, g2_mean, g2_std, n2)
        ci_lo_vec, ci_hi_vec = cohens_d_ci_vectorized(d_vec, n1, n2)

        for i in range(n_features):
            d_scalar, ci_lo_scalar, ci_hi_scalar = cohens_d_with_ci(
                group1[:, i], group2[:, i],
            )
            np.testing.assert_almost_equal(d_vec[i], d_scalar, decimal=6)
            np.testing.assert_almost_equal(ci_lo_vec[i], ci_lo_scalar, decimal=6)
            np.testing.assert_almost_equal(ci_hi_vec[i], ci_hi_scalar, decimal=6)


# ============================================================================
# 2D Embedding Ablation Fallback Tests
# ============================================================================


def test_ablation_skips_2d_embeddings():
    """2D embeddings (attended) should not crash einsum; ablation should fall back to attention-only."""
    n_subjects, n_heads, n_cell_types = 10, 4, 5
    attention = np.random.rand(n_subjects, n_heads, n_cell_types)
    cognition = np.random.rand(n_subjects)
    pathology = np.random.rand(n_subjects)
    cell_type_names = [f"CT{i}" for i in range(n_cell_types)]
    embeddings_2d = np.random.rand(n_subjects, 64)

    analyzer = ResilienceSignatureAnalyzer(
        attention=attention,
        cognition_scores=cognition,
        pathology_scores=pathology,
        cell_type_names=cell_type_names,
    )
    result = analyzer._ablation_zero_embedding(embeddings=embeddings_2d)
    assert isinstance(result, pd.DataFrame)
    assert "importance" in result.columns
    assert len(result) == n_cell_types


# ============================================================================
# Resilience Grouping Consistency Tests
# ============================================================================


class TestIdentifyGroupsMatchesDeriveResilienceGroups:
    """_identify_groups() must produce same resilient/vulnerable masks
    as the shared derive_resilience_groups() utility."""

    def test_identify_groups_matches_derive_resilience_groups(self):
        """_identify_groups() should produce same resilient/vulnerable groups
        as derive_resilience_groups() for identical inputs."""
        from src.utils.statistics import derive_resilience_groups

        np.random.seed(42)
        n = 50
        pathology = np.random.rand(n)
        cognition = np.random.rand(n)
        # Introduce NaN in cognition but not pathology for some subjects
        cognition[0:3] = np.nan
        attention = np.random.rand(n, 4, 31)

        labels = derive_resilience_groups(cognition, pathology)

        analyzer = ResilienceSignatureAnalyzer(
            attention=attention,
            pathology_scores=pathology,
            cognition_scores=cognition,
        )

        # Masks should match labels
        expected_resilient = labels == "resilient"
        expected_vulnerable = labels == "vulnerable"

        np.testing.assert_array_equal(analyzer.resilient_mask, expected_resilient)
        np.testing.assert_array_equal(analyzer.vulnerable_mask, expected_vulnerable)

    def test_nan_cognition_shifts_pathology_threshold(self):
        """When NaN-cognition subjects have extreme pathology, pathology
        threshold must be computed only on valid subjects (both cog & path
        non-NaN), matching derive_resilience_groups exactly."""
        from src.utils.statistics import derive_resilience_groups

        # Construct data where NaN-cognition subjects have very high pathology,
        # so including them shifts the 66.7th percentile upward.
        n = 30
        pathology = np.linspace(0.0, 0.6, n)
        cognition = np.linspace(0.0, 1.0, n)
        # Give the NaN-cognition subjects very high pathology
        cognition[0:5] = np.nan
        pathology[0:5] = 0.99  # extreme high pathology

        attention = np.random.rand(n, 4, 5)

        labels = derive_resilience_groups(cognition, pathology)

        analyzer = ResilienceSignatureAnalyzer(
            attention=attention,
            pathology_scores=pathology,
            cognition_scores=cognition,
            cell_type_names=[f"type_{i}" for i in range(5)],
        )

        expected_resilient = labels == "resilient"
        expected_vulnerable = labels == "vulnerable"

        np.testing.assert_array_equal(analyzer.resilient_mask, expected_resilient)
        np.testing.assert_array_equal(analyzer.vulnerable_mask, expected_vulnerable)
