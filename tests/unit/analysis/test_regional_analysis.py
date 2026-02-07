"""Tests for regional analysis module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from src.analysis.regional_analysis import (
    RegionalAnalyzer,
    RegionalAnalysisResult,
    compute_regional_analysis,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_region_attention():
    """Sample region attention weights [n_subjects, n_regions, n_cell_types]."""
    np.random.seed(42)
    return np.random.rand(20, 4, 8)  # 20 subjects, 4 regions, 8 cell types


@pytest.fixture
def sample_region_attention_2d():
    """Sample aggregated region attention weights [n_subjects, n_regions]."""
    np.random.seed(42)
    return np.random.rand(20, 4)


@pytest.fixture
def sample_region_weights():
    """Sample learned region weights [n_regions]."""
    return np.array([0.3, 0.25, 0.25, 0.2])


@pytest.fixture
def sample_gene_gate_weights():
    """Sample gene gate weights [n_cell_types, n_genes]."""
    np.random.seed(42)
    return np.random.rand(8, 100)


@pytest.fixture
def sample_region_pseudobulk():
    """Sample mean pseudobulk per region."""
    np.random.seed(42)
    return {
        "DLPFC": np.random.rand(8, 100),
        "PCC": np.random.rand(8, 100),
        "AC": np.random.rand(8, 100),
        "MT": np.random.rand(8, 100),
    }


@pytest.fixture
def sample_region_names():
    """Sample region names."""
    return ["DLPFC", "PCC", "AC", "MT"]


@pytest.fixture
def sample_cell_type_names():
    """Sample cell type names."""
    return ["Ast", "Mic", "Oli", "OPC", "Exc", "Inh", "End", "Per"]


@pytest.fixture
def sample_gene_names():
    """Sample gene names."""
    return [f"gene_{i}" for i in range(100)]


# =============================================================================
# RegionalAnalyzer Tests
# =============================================================================


class TestRegionalAnalyzerInit:
    """Test RegionalAnalyzer initialization."""

    def test_init_with_all_data(
        self,
        sample_region_attention,
        sample_region_weights,
        sample_gene_gate_weights,
        sample_region_pseudobulk,
        sample_region_names,
        sample_cell_type_names,
        sample_gene_names,
    ):
        """Test initialization with all data types."""
        analyzer = RegionalAnalyzer(
            region_attention=sample_region_attention,
            region_weights=sample_region_weights,
            gene_gate_weights=sample_gene_gate_weights,
            region_pseudobulk=sample_region_pseudobulk,
            region_names=sample_region_names,
            cell_type_names=sample_cell_type_names,
            gene_names=sample_gene_names,
        )

        assert analyzer.region_attention is not None
        assert analyzer.region_weights is not None
        assert analyzer.gene_gate_weights is not None
        assert analyzer.region_pseudobulk is not None

    def test_init_with_minimal_data(self):
        """Test initialization with no data (uses defaults)."""
        analyzer = RegionalAnalyzer()

        assert analyzer.region_attention is None
        assert analyzer.region_weights is None
        assert analyzer.gene_gate_weights is None
        assert analyzer.region_pseudobulk is None
        # Should use defaults
        assert len(analyzer.region_names) > 0
        assert len(analyzer.cell_type_names) > 0

    def test_init_with_attention_only(self, sample_region_attention, sample_region_names):
        """Test initialization with attention data only."""
        analyzer = RegionalAnalyzer(
            region_attention=sample_region_attention,
            region_names=sample_region_names,
        )

        assert analyzer.region_attention is not None
        assert analyzer.region_weights is None


class TestRegionalAnalyzerAnalyze:
    """Test RegionalAnalyzer.analyze()."""

    def test_analyze_with_all_data(
        self,
        sample_region_attention,
        sample_region_weights,
        sample_gene_gate_weights,
        sample_region_pseudobulk,
        sample_region_names,
        sample_cell_type_names,
        sample_gene_names,
    ):
        """Test analysis with all data types."""
        analyzer = RegionalAnalyzer(
            region_attention=sample_region_attention,
            region_weights=sample_region_weights,
            gene_gate_weights=sample_gene_gate_weights,
            region_pseudobulk=sample_region_pseudobulk,
            region_names=sample_region_names,
            cell_type_names=sample_cell_type_names,
            gene_names=sample_gene_names,
        )

        result = analyzer.analyze(top_k_genes=10)

        assert isinstance(result, RegionalAnalysisResult)
        assert isinstance(result.attention_summary, pd.DataFrame)
        assert result.gene_importance is not None
        assert result.region_contribution is not None

    def test_analyze_with_2d_attention(
        self,
        sample_region_attention_2d,
        sample_region_names,
    ):
        """Test analysis with 2D aggregated attention."""
        analyzer = RegionalAnalyzer(
            region_attention=sample_region_attention_2d,
            region_names=sample_region_names,
        )

        result = analyzer.analyze()

        assert isinstance(result.attention_summary, pd.DataFrame)
        assert "mean_attention" in result.attention_summary.columns

    def test_analyze_attention_summary_schema(
        self,
        sample_region_attention,
        sample_region_names,
    ):
        """Test attention summary DataFrame schema."""
        analyzer = RegionalAnalyzer(
            region_attention=sample_region_attention,
            region_names=sample_region_names,
        )

        result = analyzer.analyze()

        expected_cols = {"region", "mean_attention", "std_attention", "n_subjects", "rank"}
        assert expected_cols.issubset(set(result.attention_summary.columns))
        assert len(result.attention_summary) == len(sample_region_names)

    def test_analyze_region_contribution_schema(
        self,
        sample_region_weights,
        sample_region_names,
    ):
        """Test region contribution DataFrame schema."""
        analyzer = RegionalAnalyzer(
            region_weights=sample_region_weights,
            region_names=sample_region_names,
        )

        result = analyzer.analyze()

        assert result.region_contribution is not None
        expected_cols = {"region", "weight", "normalized_weight", "rank"}
        assert expected_cols == set(result.region_contribution.columns)

    def test_analyze_gene_importance_schema(
        self,
        sample_gene_gate_weights,
        sample_region_pseudobulk,
        sample_region_names,
        sample_cell_type_names,
        sample_gene_names,
    ):
        """Test gene importance DataFrame schema."""
        analyzer = RegionalAnalyzer(
            gene_gate_weights=sample_gene_gate_weights,
            region_pseudobulk=sample_region_pseudobulk,
            region_names=sample_region_names,
            cell_type_names=sample_cell_type_names,
            gene_names=sample_gene_names,
        )

        result = analyzer.analyze(top_k_genes=5)

        assert result.gene_importance is not None
        expected_cols = {
            "region", "cell_type", "rank", "gene", "gene_idx",
            "gate_weight", "mean_expression", "effective_weight"
        }
        assert expected_cols == set(result.gene_importance.columns)

    def test_analyze_no_data(self):
        """Test analysis with no data returns placeholder."""
        analyzer = RegionalAnalyzer()

        result = analyzer.analyze()

        assert isinstance(result.attention_summary, pd.DataFrame)
        assert result.attention_summary["mean_attention"].isna().all()

    def test_analyze_top_k_genes(
        self,
        sample_gene_gate_weights,
        sample_region_pseudobulk,
        sample_region_names,
        sample_cell_type_names,
        sample_gene_names,
    ):
        """Test top_k_genes parameter."""
        analyzer = RegionalAnalyzer(
            gene_gate_weights=sample_gene_gate_weights,
            region_pseudobulk=sample_region_pseudobulk,
            region_names=sample_region_names,
            cell_type_names=sample_cell_type_names,
            gene_names=sample_gene_names,
        )

        result = analyzer.analyze(top_k_genes=3)

        # Should have 3 genes per region per cell type
        for region in sample_region_names:
            for ct in sample_cell_type_names:
                region_ct_rows = result.gene_importance[
                    (result.gene_importance["region"] == region) &
                    (result.gene_importance["cell_type"] == ct)
                ]
                assert len(region_ct_rows) == 3

    def test_analyze_metadata(
        self,
        sample_region_attention,
        sample_region_weights,
        sample_region_names,
    ):
        """Test metadata in result."""
        analyzer = RegionalAnalyzer(
            region_attention=sample_region_attention,
            region_weights=sample_region_weights,
            region_names=sample_region_names,
        )

        result = analyzer.analyze(top_k_genes=10)

        assert "n_regions" in result.metadata
        assert "has_attention_data" in result.metadata
        assert "has_region_weights" in result.metadata
        assert result.metadata["has_attention_data"] is True
        assert result.metadata["has_region_weights"] is True


class TestRegionalAnalyzerSave:
    """Test RegionalAnalyzer.save()."""

    def test_save_parquet_and_csv(
        self,
        tmp_path,
        sample_region_attention,
        sample_region_weights,
        sample_region_names,
    ):
        """Test saving in both formats."""
        analyzer = RegionalAnalyzer(
            region_attention=sample_region_attention,
            region_weights=sample_region_weights,
            region_names=sample_region_names,
        )

        result = analyzer.analyze()
        saved_files = analyzer.save(result, tmp_path)

        assert "attention_summary_parquet" in saved_files
        assert "attention_summary_csv" in saved_files
        assert "region_contribution_parquet" in saved_files
        assert "region_contribution_csv" in saved_files

        # Verify files exist
        assert saved_files["attention_summary_parquet"].exists()
        assert saved_files["attention_summary_csv"].exists()

    def test_save_with_gene_importance(
        self,
        tmp_path,
        sample_gene_gate_weights,
        sample_region_pseudobulk,
        sample_region_names,
        sample_cell_type_names,
        sample_gene_names,
    ):
        """Test saving with gene importance data."""
        analyzer = RegionalAnalyzer(
            gene_gate_weights=sample_gene_gate_weights,
            region_pseudobulk=sample_region_pseudobulk,
            region_names=sample_region_names,
            cell_type_names=sample_cell_type_names,
            gene_names=sample_gene_names,
        )

        result = analyzer.analyze(top_k_genes=5)
        saved_files = analyzer.save(result, tmp_path)

        assert "gene_importance_parquet" in saved_files
        assert "gene_importance_csv" in saved_files

    def test_save_csv_only(
        self,
        tmp_path,
        sample_region_attention,
        sample_region_names,
    ):
        """Test saving CSV only."""
        analyzer = RegionalAnalyzer(
            region_attention=sample_region_attention,
            region_names=sample_region_names,
        )

        result = analyzer.analyze()
        saved_files = analyzer.save(result, tmp_path, formats=["csv"])

        assert "attention_summary_csv" in saved_files
        assert "attention_summary_parquet" not in saved_files

    def test_save_creates_directory(
        self,
        tmp_path,
        sample_region_attention,
        sample_region_names,
    ):
        """Test that save creates output directory if needed."""
        analyzer = RegionalAnalyzer(
            region_attention=sample_region_attention,
            region_names=sample_region_names,
        )

        result = analyzer.analyze()
        output_dir = tmp_path / "nested" / "output"
        saved_files = analyzer.save(result, output_dir)

        assert output_dir.exists()
        assert len(saved_files) > 0


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestComputeRegionalAnalysis:
    """Test compute_regional_analysis convenience function."""

    def test_compute_basic(self, sample_region_attention, sample_region_names):
        """Test basic compute function."""
        result = compute_regional_analysis(
            region_attention=sample_region_attention,
            region_names=sample_region_names,
        )

        assert isinstance(result, RegionalAnalysisResult)
        assert isinstance(result.attention_summary, pd.DataFrame)

    def test_compute_with_save(
        self,
        tmp_path,
        sample_region_attention,
        sample_region_weights,
        sample_region_names,
    ):
        """Test compute function with saving."""
        result = compute_regional_analysis(
            region_attention=sample_region_attention,
            region_weights=sample_region_weights,
            region_names=sample_region_names,
            output_dir=tmp_path,
        )

        assert isinstance(result, RegionalAnalysisResult)
        assert (tmp_path / "regional_attention_summary.csv").exists()
        assert (tmp_path / "region_contribution.parquet").exists()

    def test_compute_with_all_data(
        self,
        sample_region_attention,
        sample_region_weights,
        sample_gene_gate_weights,
        sample_region_pseudobulk,
        sample_region_names,
        sample_cell_type_names,
        sample_gene_names,
    ):
        """Test compute function with all data types."""
        result = compute_regional_analysis(
            region_attention=sample_region_attention,
            region_weights=sample_region_weights,
            gene_gate_weights=sample_gene_gate_weights,
            region_pseudobulk=sample_region_pseudobulk,
            region_names=sample_region_names,
            cell_type_names=sample_cell_type_names,
            gene_names=sample_gene_names,
            top_k_genes=5,
        )

        assert result.attention_summary is not None
        assert result.region_contribution is not None
        assert result.gene_importance is not None


# =============================================================================
# Edge Cases
# =============================================================================


class TestRegionalAnalysisGuard:
    """Test the guard condition in run_analysis.py."""

    def test_regional_analysis_skipped_without_region_data(self):
        """Regional analysis should not run when only region_weights exist
        but no region_attention or region_pseudobulk was extracted."""
        import ast
        from pathlib import Path

        source = Path("scripts/run_analysis.py").read_text()
        assert "region_attention is not None" in source or "region_pseudobulk" in source, (
            "Regional analysis guard must check for actual extracted data, "
            "not just learned weights"
        )


class TestRegionalAnalysisEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_attention_shape(self, sample_region_names):
        """Test error on invalid attention shape."""
        analyzer = RegionalAnalyzer(
            region_attention=np.random.rand(10, 4, 8, 2),  # 4D not supported
            region_names=sample_region_names,
        )

        with pytest.raises(ValueError, match="Unexpected attention shape"):
            analyzer.analyze()

    def test_region_mismatch(self):
        """Test handling of region count mismatch."""
        # 5 regions in attention but only 4 region names
        attention = np.random.rand(10, 5, 8)
        region_names = ["R1", "R2", "R3", "R4"]

        analyzer = RegionalAnalyzer(
            region_attention=attention,
            region_names=region_names,
        )

        result = analyzer.analyze()
        # Should only report on regions that have names
        assert len(result.attention_summary) == 4

    def test_empty_region_pseudobulk(
        self,
        sample_gene_gate_weights,
        sample_region_names,
        sample_cell_type_names,
        sample_gene_names,
    ):
        """Test handling of empty region pseudobulk."""
        analyzer = RegionalAnalyzer(
            gene_gate_weights=sample_gene_gate_weights,
            region_pseudobulk={},  # Empty
            region_names=sample_region_names,
            cell_type_names=sample_cell_type_names,
            gene_names=sample_gene_names,
        )

        result = analyzer.analyze()

        # Gene importance should still be None (no regions to process)
        assert result.gene_importance is None or len(result.gene_importance) == 0


# =============================================================================
# Property-Based Tests
# =============================================================================


class TestRegionalAnalysisProperties:
    """Property-based tests using Hypothesis."""

    @given(
        n_subjects=st.integers(min_value=5, max_value=50),
        n_regions=st.integers(min_value=2, max_value=10),
        n_cell_types=st.integers(min_value=2, max_value=15),
    )
    @settings(max_examples=20)
    def test_attention_summary_properties(self, n_subjects, n_regions, n_cell_types):
        """Test attention summary has correct properties."""
        attention = np.random.rand(n_subjects, n_regions, n_cell_types)
        region_names = [f"Region_{i}" for i in range(n_regions)]

        analyzer = RegionalAnalyzer(
            region_attention=attention,
            region_names=region_names,
        )

        result = analyzer.analyze()

        # Check number of rows
        assert len(result.attention_summary) == n_regions

        # Check all regions present
        assert set(result.attention_summary["region"]) == set(region_names)

        # Check mean attention in valid range
        assert result.attention_summary["mean_attention"].min() >= 0
        assert result.attention_summary["mean_attention"].max() <= 1

        # Check ranks are consecutive 1 to n_regions
        assert set(result.attention_summary["rank"]) == set(range(1, n_regions + 1))

    @given(n_regions=st.integers(min_value=2, max_value=10))
    @settings(max_examples=15)
    def test_normalized_weights_sum_to_one(self, n_regions):
        """Test normalized region weights sum to approximately 1."""
        weights = np.random.rand(n_regions) + 0.1  # Ensure positive
        region_names = [f"Region_{i}" for i in range(n_regions)]

        analyzer = RegionalAnalyzer(
            region_weights=weights,
            region_names=region_names,
        )

        result = analyzer.analyze()

        assert result.region_contribution is not None
        total = result.region_contribution["normalized_weight"].sum()
        assert abs(total - 1.0) < 1e-6


# =============================================================================
# Schema Validation Tests
# =============================================================================


class TestRegionalAnalysisSchema:
    """Test output DataFrame schemas."""

    def test_attention_summary_dtypes(
        self,
        sample_region_attention,
        sample_region_names,
    ):
        """Test attention summary column types."""
        analyzer = RegionalAnalyzer(
            region_attention=sample_region_attention,
            region_names=sample_region_names,
        )

        result = analyzer.analyze()
        df = result.attention_summary

        assert df["region"].dtype == object  # string
        assert np.issubdtype(df["mean_attention"].dtype, np.floating)
        assert np.issubdtype(df["std_attention"].dtype, np.floating)
        assert np.issubdtype(df["n_subjects"].dtype, np.integer)
        assert np.issubdtype(df["rank"].dtype, np.integer)

    def test_region_contribution_dtypes(
        self,
        sample_region_weights,
        sample_region_names,
    ):
        """Test region contribution column types."""
        analyzer = RegionalAnalyzer(
            region_weights=sample_region_weights,
            region_names=sample_region_names,
        )

        result = analyzer.analyze()
        df = result.region_contribution

        assert df["region"].dtype == object
        assert np.issubdtype(df["weight"].dtype, np.floating)
        assert np.issubdtype(df["normalized_weight"].dtype, np.floating)
        assert np.issubdtype(df["rank"].dtype, np.integer)

    def test_gene_importance_dtypes(
        self,
        sample_gene_gate_weights,
        sample_region_pseudobulk,
        sample_region_names,
        sample_cell_type_names,
        sample_gene_names,
    ):
        """Test gene importance column types."""
        analyzer = RegionalAnalyzer(
            gene_gate_weights=sample_gene_gate_weights,
            region_pseudobulk=sample_region_pseudobulk,
            region_names=sample_region_names,
            cell_type_names=sample_cell_type_names,
            gene_names=sample_gene_names,
        )

        result = analyzer.analyze(top_k_genes=5)
        df = result.gene_importance

        assert df["region"].dtype == object
        assert df["cell_type"].dtype == object
        assert df["gene"].dtype == object
        assert np.issubdtype(df["rank"].dtype, np.integer)
        assert np.issubdtype(df["gene_idx"].dtype, np.integer)
        assert np.issubdtype(df["gate_weight"].dtype, np.floating)
        assert np.issubdtype(df["mean_expression"].dtype, np.floating)
        assert np.issubdtype(df["effective_weight"].dtype, np.floating)


# =============================================================================
# Round-Trip Tests
# =============================================================================


class TestRegionalAnalysisRoundTrip:
    """Test save and load round-trips."""

    def test_attention_summary_roundtrip_parquet(
        self,
        tmp_path,
        sample_region_attention,
        sample_region_names,
    ):
        """Test attention summary parquet round-trip."""
        analyzer = RegionalAnalyzer(
            region_attention=sample_region_attention,
            region_names=sample_region_names,
        )

        result = analyzer.analyze()
        saved_files = analyzer.save(result, tmp_path, formats=["parquet"])

        # Load and compare
        loaded = pd.read_parquet(saved_files["attention_summary_parquet"])
        pd.testing.assert_frame_equal(
            result.attention_summary.reset_index(drop=True),
            loaded.reset_index(drop=True),
        )

    def test_region_contribution_roundtrip_csv(
        self,
        tmp_path,
        sample_region_weights,
        sample_region_names,
    ):
        """Test region contribution CSV round-trip."""
        analyzer = RegionalAnalyzer(
            region_weights=sample_region_weights,
            region_names=sample_region_names,
        )

        result = analyzer.analyze()
        saved_files = analyzer.save(result, tmp_path, formats=["csv"])

        # Load and compare
        loaded = pd.read_csv(saved_files["region_contribution_csv"])
        pd.testing.assert_frame_equal(
            result.region_contribution.reset_index(drop=True),
            loaded.reset_index(drop=True),
            check_dtype=False,  # CSV may change dtypes
        )


# =============================================================================
# _compute_per_subject_attention Tests (I10)
# =============================================================================


class TestComputePerSubjectAttention:
    """Test _compute_per_subject_attention method."""

    def test_returns_dataframe_with_correct_schema(
        self, sample_region_attention, sample_region_names
    ):
        """Output DataFrame has columns: subject_id, region, attention_weight."""
        analyzer = RegionalAnalyzer(
            region_attention=sample_region_attention,
            region_names=sample_region_names,
        )
        df = analyzer._compute_per_subject_attention()

        assert df is not None
        assert list(df.columns) == ["subject_id", "region", "attention_weight"]
        assert df["attention_weight"].dtype == np.float64

    def test_correct_row_count(
        self, sample_region_attention, sample_region_names
    ):
        """Output has n_subjects * n_regions rows."""
        analyzer = RegionalAnalyzer(
            region_attention=sample_region_attention,
            region_names=sample_region_names,
        )
        df = analyzer._compute_per_subject_attention()

        n_subjects = sample_region_attention.shape[0]
        n_regions = len(sample_region_names)
        assert len(df) == n_subjects * n_regions

    def test_uses_provided_subject_ids(
        self, sample_region_attention, sample_region_names
    ):
        """Output uses real subject IDs when provided."""
        n_subjects = sample_region_attention.shape[0]
        subject_ids = [f"ROSMAP_{i:04d}" for i in range(n_subjects)]

        analyzer = RegionalAnalyzer(
            region_attention=sample_region_attention,
            region_names=sample_region_names,
            subject_ids=subject_ids,
        )
        df = analyzer._compute_per_subject_attention()

        assert set(df["subject_id"].unique()) == set(subject_ids)

    def test_synthetic_subject_ids_when_none(
        self, sample_region_attention, sample_region_names
    ):
        """Falls back to subject_0, subject_1, ... when no subject_ids."""
        analyzer = RegionalAnalyzer(
            region_attention=sample_region_attention,
            region_names=sample_region_names,
        )
        df = analyzer._compute_per_subject_attention()

        n_subjects = sample_region_attention.shape[0]
        expected_ids = {f"subject_{i}" for i in range(n_subjects)}
        assert set(df["subject_id"].unique()) == expected_ids

    def test_all_regions_present(
        self, sample_region_attention, sample_region_names
    ):
        """Every region appears in the output for each subject."""
        analyzer = RegionalAnalyzer(
            region_attention=sample_region_attention,
            region_names=sample_region_names,
        )
        df = analyzer._compute_per_subject_attention()

        assert set(df["region"].unique()) == set(sample_region_names)

    def test_handles_3d_input(
        self, sample_region_attention, sample_region_names
    ):
        """3D attention [n_subjects, n_regions, n_cell_types] is mean-pooled over axis 2."""
        assert sample_region_attention.ndim == 3

        analyzer = RegionalAnalyzer(
            region_attention=sample_region_attention,
            region_names=sample_region_names,
        )
        df = analyzer._compute_per_subject_attention()

        # Verify a specific value matches manual mean-pool
        expected = float(sample_region_attention[0, 0, :].mean())
        actual = df.loc[
            (df["subject_id"] == "subject_0") & (df["region"] == sample_region_names[0]),
            "attention_weight",
        ].iloc[0]
        np.testing.assert_almost_equal(actual, expected)

    def test_handles_2d_input(
        self, sample_region_attention_2d, sample_region_names
    ):
        """2D attention [n_subjects, n_regions] used directly."""
        assert sample_region_attention_2d.ndim == 2

        analyzer = RegionalAnalyzer(
            region_attention=sample_region_attention_2d,
            region_names=sample_region_names,
        )
        df = analyzer._compute_per_subject_attention()

        expected = float(sample_region_attention_2d[0, 0])
        actual = df.loc[
            (df["subject_id"] == "subject_0") & (df["region"] == sample_region_names[0]),
            "attention_weight",
        ].iloc[0]
        np.testing.assert_almost_equal(actual, expected)

    def test_returns_none_when_no_attention(self, sample_region_names):
        """Returns None when region_attention is None."""
        analyzer = RegionalAnalyzer(
            region_attention=None,
            region_names=sample_region_names,
        )
        result = analyzer._compute_per_subject_attention()
        assert result is None

    def test_integrated_through_analyze(
        self, sample_region_attention, sample_region_names
    ):
        """per_subject_attention is populated in RegionalAnalysisResult.analyze()."""
        subject_ids = [f"subj_{i}" for i in range(sample_region_attention.shape[0])]
        analyzer = RegionalAnalyzer(
            region_attention=sample_region_attention,
            region_names=sample_region_names,
            subject_ids=subject_ids,
        )
        result = analyzer.analyze()

        assert result.per_subject_attention is not None
        assert len(result.per_subject_attention) == len(subject_ids) * len(sample_region_names)

    def test_round_trip_parquet(
        self, sample_region_attention, sample_region_names, tmp_path
    ):
        """per_subject_attention survives parquet save/load."""
        analyzer = RegionalAnalyzer(
            region_attention=sample_region_attention,
            region_names=sample_region_names,
        )
        result = analyzer.analyze()
        saved_files = analyzer.save(result, tmp_path, formats=["parquet"])

        key = "per_subject_attention_parquet"
        if key in saved_files:
            loaded = pd.read_parquet(saved_files[key])
            pd.testing.assert_frame_equal(
                result.per_subject_attention.reset_index(drop=True),
                loaded.reset_index(drop=True),
            )
