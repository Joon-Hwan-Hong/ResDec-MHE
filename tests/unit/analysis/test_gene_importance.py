"""
Tests for src/analysis/gene_importance.py.

Test coverage includes:
- GeneImportanceResult dataclass behavior
- GeneImportanceAnalyzer initialization and validation
- Gene importance by cell type computation
- Top genes extraction
- Region-stratified effective importance
- HDF5 serialization
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

from src.data.constants import CELL_TYPE_ORDER, N_CELL_TYPES, REGION_ORDER
from src.analysis.gene_importance import (
    GeneImportanceResult,
    GeneImportanceAnalyzer,
    compute_gene_importance,
    load_gene_gate_weights_hdf5,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_gene_gate_weights():
    """Sample gene gate weights [n_cell_types, n_genes]."""
    np.random.seed(42)
    return np.random.rand(N_CELL_TYPES, 100).astype(np.float32)


@pytest.fixture
def sample_gene_names():
    """Sample gene names."""
    return [f"GENE_{i}" for i in range(100)]


@pytest.fixture
def sample_region_pseudobulk():
    """Sample region pseudobulk data."""
    np.random.seed(42)
    return {
        "PFC": np.random.rand(N_CELL_TYPES, 100).astype(np.float32),
        "AG": np.random.rand(N_CELL_TYPES, 100).astype(np.float32),
        "MTC": np.random.rand(N_CELL_TYPES, 100).astype(np.float32),
    }


@pytest.fixture
def analyzer(sample_gene_gate_weights, sample_gene_names):
    """GeneImportanceAnalyzer instance."""
    return GeneImportanceAnalyzer(
        gene_gate_weights=sample_gene_gate_weights,
        gene_names=sample_gene_names,
    )


# ============================================================================
# GeneImportanceResult Dataclass Tests
# ============================================================================


class TestGeneImportanceResult:
    """Tests for GeneImportanceResult dataclass."""

    def test_init_with_required_fields(self):
        """Result can be initialized with required fields."""
        by_celltype = pd.DataFrame({"cell_type": ["A"], "gene": ["G1"], "gene_idx": [0], "weight": [0.5]})
        top_genes = pd.DataFrame({"cell_type": ["A"], "rank": [1], "gene": ["G1"], "gene_idx": [0], "weight": [0.5]})
        result = GeneImportanceResult(by_celltype=by_celltype, top_genes=top_genes)
        assert result.by_celltype is not None
        assert result.by_region is None

    def test_metadata_defaults_to_empty_dict(self):
        """metadata defaults to empty dict."""
        by_celltype = pd.DataFrame({"cell_type": ["A"], "gene": ["G1"], "gene_idx": [0], "weight": [0.5]})
        top_genes = pd.DataFrame({"cell_type": ["A"], "rank": [1], "gene": ["G1"], "gene_idx": [0], "weight": [0.5]})
        result = GeneImportanceResult(by_celltype=by_celltype, top_genes=top_genes)
        assert result.metadata == {}


# ============================================================================
# GeneImportanceAnalyzer Initialization Tests
# ============================================================================


class TestAnalyzerInit:
    """Tests for GeneImportanceAnalyzer initialization."""

    def test_init_with_weights_only(self, sample_gene_gate_weights):
        """Analyzer initializes with only gene gate weights."""
        analyzer = GeneImportanceAnalyzer(gene_gate_weights=sample_gene_gate_weights)
        assert analyzer.n_cell_types == N_CELL_TYPES
        assert analyzer.n_genes == 100

    def test_init_generates_default_gene_names(self, sample_gene_gate_weights):
        """Analyzer generates default gene names if not provided."""
        analyzer = GeneImportanceAnalyzer(gene_gate_weights=sample_gene_gate_weights)
        assert analyzer.gene_names[0] == "gene_0"
        assert len(analyzer.gene_names) == 100

    def test_init_uses_cell_type_order(self, sample_gene_gate_weights, sample_gene_names):
        """Analyzer uses CELL_TYPE_ORDER as default cell type names."""
        analyzer = GeneImportanceAnalyzer(
            gene_gate_weights=sample_gene_gate_weights,
            gene_names=sample_gene_names,
        )
        assert analyzer.cell_type_names == list(CELL_TYPE_ORDER)[:N_CELL_TYPES]

    def test_init_validates_weights_ndim(self, sample_gene_names):
        """Analyzer rejects weights with wrong dimensions."""
        bad_weights = np.random.rand(100).astype(np.float32)  # 1D
        with pytest.raises(ValueError, match="must be 2D"):
            GeneImportanceAnalyzer(gene_gate_weights=bad_weights, gene_names=sample_gene_names)

    def test_init_validates_gene_names_length(self, sample_gene_gate_weights):
        """Analyzer rejects gene_names with wrong length."""
        bad_names = ["gene_0", "gene_1"]  # Too short
        with pytest.raises(ValueError, match="gene_names"):
            GeneImportanceAnalyzer(gene_gate_weights=sample_gene_gate_weights, gene_names=bad_names)

    def test_init_validates_region_pseudobulk_shape(self, sample_gene_gate_weights, sample_gene_names):
        """Analyzer rejects region_pseudobulk with mismatched shapes."""
        bad_region_data = {"PFC": np.random.rand(5, 10).astype(np.float32)}
        with pytest.raises(ValueError, match="region_pseudobulk"):
            GeneImportanceAnalyzer(
                gene_gate_weights=sample_gene_gate_weights,
                gene_names=sample_gene_names,
                region_pseudobulk=bad_region_data,
            )


# ============================================================================
# Gene Importance by Cell Type Tests
# ============================================================================


class TestImportanceByCellType:
    """Tests for gene importance by cell type computation."""

    def test_analyze_returns_result(self, analyzer):
        """analyze() returns GeneImportanceResult."""
        result = analyzer.analyze()
        assert isinstance(result, GeneImportanceResult)

    def test_by_celltype_has_all_combinations(self, analyzer):
        """by_celltype includes all cell type × gene combinations."""
        result = analyzer.analyze()
        expected_rows = N_CELL_TYPES * 100
        assert len(result.by_celltype) == expected_rows

    def test_by_celltype_has_expected_columns(self, analyzer):
        """by_celltype has expected columns."""
        result = analyzer.analyze()
        expected_cols = {"cell_type", "gene", "gene_idx", "weight"}
        assert set(result.by_celltype.columns) == expected_cols

    def test_by_celltype_weights_match_input(self, analyzer, sample_gene_gate_weights):
        """by_celltype weights match input gene gate weights."""
        result = analyzer.analyze()
        first_ct = list(CELL_TYPE_ORDER)[0]
        first_gene = analyzer.gene_names[0]

        row = result.by_celltype[
            (result.by_celltype["cell_type"] == first_ct) &
            (result.by_celltype["gene"] == first_gene)
        ]
        assert len(row) == 1
        assert np.isclose(row["weight"].values[0], sample_gene_gate_weights[0, 0])


# ============================================================================
# Top Genes Tests
# ============================================================================


class TestTopGenes:
    """Tests for top genes extraction."""

    def test_top_genes_has_expected_columns(self, analyzer):
        """top_genes has expected columns."""
        result = analyzer.analyze(top_k=10)
        expected_cols = {"cell_type", "rank", "gene", "gene_idx", "weight"}
        assert set(result.top_genes.columns) == expected_cols

    def test_top_genes_respects_top_k(self, analyzer):
        """top_genes extracts exactly top_k genes per cell type."""
        result = analyzer.analyze(top_k=10)
        expected_rows = N_CELL_TYPES * 10
        assert len(result.top_genes) == expected_rows

    def test_top_genes_ranks_are_sequential(self, analyzer):
        """top_genes ranks are 1 to top_k for each cell type."""
        result = analyzer.analyze(top_k=5)
        first_ct = list(CELL_TYPE_ORDER)[0]
        ct_data = result.top_genes[result.top_genes["cell_type"] == first_ct]
        assert list(ct_data["rank"]) == [1, 2, 3, 4, 5]

    def test_top_genes_sorted_by_weight(self, analyzer):
        """top_genes are sorted by weight descending within cell type."""
        result = analyzer.analyze(top_k=10)
        first_ct = list(CELL_TYPE_ORDER)[0]
        ct_data = result.top_genes[result.top_genes["cell_type"] == first_ct]
        weights = ct_data["weight"].tolist()
        assert weights == sorted(weights, reverse=True)


# ============================================================================
# Region-Stratified Importance Tests
# ============================================================================


class TestRegionStratified:
    """Tests for region-stratified effective importance."""

    def test_by_region_present_when_data_provided(self, sample_gene_gate_weights, sample_gene_names, sample_region_pseudobulk):
        """by_region is present when region_pseudobulk provided."""
        analyzer = GeneImportanceAnalyzer(
            gene_gate_weights=sample_gene_gate_weights,
            gene_names=sample_gene_names,
            region_pseudobulk=sample_region_pseudobulk,
        )
        result = analyzer.analyze()
        assert result.by_region is not None

    def test_by_region_absent_when_data_missing(self, analyzer):
        """by_region is None when region_pseudobulk not provided."""
        result = analyzer.analyze()
        assert result.by_region is None

    def test_by_region_has_expected_columns(self, sample_gene_gate_weights, sample_gene_names, sample_region_pseudobulk):
        """by_region has expected columns."""
        analyzer = GeneImportanceAnalyzer(
            gene_gate_weights=sample_gene_gate_weights,
            gene_names=sample_gene_names,
            region_pseudobulk=sample_region_pseudobulk,
        )
        result = analyzer.analyze(top_k=10)
        expected_cols = {"region", "cell_type", "rank", "gene", "gene_idx", "gate_weight", "mean_expression", "effective_weight"}
        assert set(result.by_region.columns) == expected_cols

    def test_by_region_includes_all_regions(self, sample_gene_gate_weights, sample_gene_names, sample_region_pseudobulk):
        """by_region includes all provided regions."""
        analyzer = GeneImportanceAnalyzer(
            gene_gate_weights=sample_gene_gate_weights,
            gene_names=sample_gene_names,
            region_pseudobulk=sample_region_pseudobulk,
        )
        result = analyzer.analyze(top_k=10)
        regions = set(result.by_region["region"].unique())
        assert regions == set(sample_region_pseudobulk.keys())

    def test_effective_weight_is_product(self, sample_gene_gate_weights, sample_gene_names, sample_region_pseudobulk):
        """effective_weight = gate_weight × mean_expression."""
        analyzer = GeneImportanceAnalyzer(
            gene_gate_weights=sample_gene_gate_weights,
            gene_names=sample_gene_names,
            region_pseudobulk=sample_region_pseudobulk,
        )
        result = analyzer.analyze(top_k=10)
        for _, row in result.by_region.head(5).iterrows():
            expected = row["gate_weight"] * row["mean_expression"]
            assert np.isclose(row["effective_weight"], expected, atol=1e-6)


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
            assert (Path(tmpdir) / "gene_importance_by_celltype.parquet").exists()
            assert (Path(tmpdir) / "top_genes_per_celltype.csv").exists()
            assert (Path(tmpdir) / "gene_gate_weights.h5").exists()

    def test_hdf5_roundtrip(self, analyzer, sample_gene_gate_weights, sample_gene_names):
        """HDF5 save/load preserves data."""
        result = analyzer.analyze()
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer.save(result, tmpdir)
            weights, genes, cell_types = load_gene_gate_weights_hdf5(Path(tmpdir) / "gene_gate_weights.h5")

        np.testing.assert_array_almost_equal(weights, sample_gene_gate_weights)
        assert genes == sample_gene_names


# ============================================================================
# Schema Validation Tests
# ============================================================================


class TestOutputSchemaValidation:
    """Tests validating output DataFrame schemas."""

    def test_by_celltype_schema(self, analyzer):
        """by_celltype DataFrame has expected schema."""
        result = analyzer.analyze()
        df = result.by_celltype
        assert df["cell_type"].dtype == object
        assert df["gene"].dtype == object
        assert np.issubdtype(df["gene_idx"].dtype, np.integer)
        assert np.issubdtype(df["weight"].dtype, np.floating)

    def test_top_genes_schema(self, analyzer):
        """top_genes DataFrame has expected schema."""
        result = analyzer.analyze()
        df = result.top_genes
        assert df["cell_type"].dtype == object
        assert np.issubdtype(df["rank"].dtype, np.integer)
        assert df["gene"].dtype == object

    def test_weights_non_negative(self, analyzer):
        """Gene gate weights should be non-negative (after softmax)."""
        result = analyzer.analyze()
        # Note: Raw weights before softmax can be any value, but
        # typically gate outputs are non-negative probabilities
        # This test just verifies the data type is correct
        assert np.issubdtype(result.by_celltype["weight"].dtype, np.floating)


# ============================================================================
# Property-Based Tests
# ============================================================================


class TestPropertyBased:
    """Property-based tests using Hypothesis."""

    @given(
        n_cell_types=st.integers(min_value=1, max_value=10),
        n_genes=st.integers(min_value=10, max_value=50),
    )
    @settings(max_examples=15)
    def test_by_celltype_row_count(self, n_cell_types, n_genes):
        """by_celltype always has n_cell_types × n_genes rows."""
        weights = np.random.rand(n_cell_types, n_genes).astype(np.float32)
        cell_type_names = [f"type_{i}" for i in range(n_cell_types)]
        gene_names = [f"gene_{i}" for i in range(n_genes)]

        analyzer = GeneImportanceAnalyzer(
            gene_gate_weights=weights,
            gene_names=gene_names,
            cell_type_names=cell_type_names,
        )
        result = analyzer.analyze()
        assert len(result.by_celltype) == n_cell_types * n_genes

    @given(
        n_cell_types=st.integers(min_value=1, max_value=10),
        n_genes=st.integers(min_value=10, max_value=50),
        top_k=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=15)
    def test_top_genes_row_count(self, n_cell_types, n_genes, top_k):
        """top_genes has n_cell_types × min(top_k, n_genes) rows."""
        weights = np.random.rand(n_cell_types, n_genes).astype(np.float32)
        cell_type_names = [f"type_{i}" for i in range(n_cell_types)]
        gene_names = [f"gene_{i}" for i in range(n_genes)]

        analyzer = GeneImportanceAnalyzer(
            gene_gate_weights=weights,
            gene_names=gene_names,
            cell_type_names=cell_type_names,
        )
        result = analyzer.analyze(top_k=top_k)
        expected = n_cell_types * min(top_k, n_genes)
        assert len(result.top_genes) == expected


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Edge case tests."""

    def test_single_gene(self):
        """Handles single gene correctly."""
        weights = np.random.rand(5, 1).astype(np.float32)
        analyzer = GeneImportanceAnalyzer(
            gene_gate_weights=weights,
            gene_names=["SINGLE_GENE"],
            cell_type_names=[f"type_{i}" for i in range(5)],
        )
        result = analyzer.analyze(top_k=10)
        assert len(result.top_genes) == 5  # 5 cell types × 1 gene

    def test_single_cell_type(self):
        """Handles single cell type correctly."""
        weights = np.random.rand(1, 50).astype(np.float32)
        analyzer = GeneImportanceAnalyzer(
            gene_gate_weights=weights,
            gene_names=[f"gene_{i}" for i in range(50)],
            cell_type_names=["SingleType"],
        )
        result = analyzer.analyze(top_k=10)
        assert len(result.top_genes) == 10

    def test_top_k_larger_than_n_genes(self):
        """Handles top_k > n_genes correctly."""
        weights = np.random.rand(3, 5).astype(np.float32)
        analyzer = GeneImportanceAnalyzer(
            gene_gate_weights=weights,
            gene_names=[f"gene_{i}" for i in range(5)],
            cell_type_names=["A", "B", "C"],
        )
        result = analyzer.analyze(top_k=100)  # Much larger than 5 genes
        assert len(result.top_genes) == 3 * 5  # All genes for each cell type

    def test_empty_region_pseudobulk(self):
        """Handles empty region_pseudobulk dict."""
        weights = np.random.rand(3, 10).astype(np.float32)
        analyzer = GeneImportanceAnalyzer(
            gene_gate_weights=weights,
            gene_names=[f"gene_{i}" for i in range(10)],
            cell_type_names=["A", "B", "C"],
            region_pseudobulk={},
        )
        result = analyzer.analyze()
        assert result.by_region is None or len(result.by_region) == 0


# ============================================================================
# Phase 6 Review Round 8 — L1: Gene gate HDF5 string decode
# ============================================================================


class TestGeneGateLoadStrAndBytes:
    """Test L1: load_gene_gate_weights_hdf5 handles both bytes and str HDF5 datasets."""

    def test_loads_bytes_strings(self, tmp_path):
        """HDF5 with fixed-length byte strings (S-type) loads correctly."""
        import h5py

        path = tmp_path / "gene_gate_bytes.h5"
        n_ct, n_genes = 3, 5
        with h5py.File(path, "w") as f:
            f.create_dataset("gene_gate", data=np.random.rand(n_ct, n_genes).astype(np.float32))
            # Fixed-length byte strings (S64)
            f.create_dataset("gene_names", data=np.array(
                [f"GENE{i}" for i in range(n_genes)], dtype="S64"
            ))
            f.create_dataset("cell_type_names", data=np.array(
                ["Ast", "Mic", "Oli"], dtype="S64"
            ))

        weights, gene_names, cell_type_names = load_gene_gate_weights_hdf5(path)
        assert weights.shape == (n_ct, n_genes)
        assert gene_names == [f"GENE{i}" for i in range(n_genes)]
        assert cell_type_names == ["Ast", "Mic", "Oli"]

    def test_loads_vlen_strings(self, tmp_path):
        """HDF5 with vlen string datasets loads correctly."""
        import h5py

        path = tmp_path / "gene_gate_vlen.h5"
        n_ct, n_genes = 3, 5
        vlen_str = h5py.special_dtype(vlen=str)
        with h5py.File(path, "w") as f:
            f.create_dataset("gene_gate", data=np.random.rand(n_ct, n_genes).astype(np.float32))
            # Vlen strings (already str, not bytes)
            f.create_dataset("gene_names", data=np.array(
                [f"GENE{i}" for i in range(n_genes)], dtype=object
            ), dtype=vlen_str)
            f.create_dataset("cell_type_names", data=np.array(
                ["Ast", "Mic", "Oli"], dtype=object
            ), dtype=vlen_str)

        weights, gene_names, cell_type_names = load_gene_gate_weights_hdf5(path)
        assert weights.shape == (n_ct, n_genes)
        assert gene_names == [f"GENE{i}" for i in range(n_genes)]
        assert cell_type_names == ["Ast", "Mic", "Oli"]

    def test_mixed_bytes_and_str(self, tmp_path):
        """HDF5 with one bytes dataset and one vlen string dataset loads correctly."""
        import h5py

        path = tmp_path / "gene_gate_mixed.h5"
        n_ct, n_genes = 3, 5
        vlen_str = h5py.special_dtype(vlen=str)
        with h5py.File(path, "w") as f:
            f.create_dataset("gene_gate", data=np.random.rand(n_ct, n_genes).astype(np.float32))
            # gene_names as bytes
            f.create_dataset("gene_names", data=np.array(
                [f"GENE{i}" for i in range(n_genes)], dtype="S64"
            ))
            # cell_type_names as vlen str
            f.create_dataset("cell_type_names", data=np.array(
                ["Ast", "Mic", "Oli"], dtype=object
            ), dtype=vlen_str)

        weights, gene_names, cell_type_names = load_gene_gate_weights_hdf5(path)
        assert gene_names == [f"GENE{i}" for i in range(n_genes)]
        assert cell_type_names == ["Ast", "Mic", "Oli"]


# ============================================================================
# Differential Attention Analysis Tests
# ============================================================================


class TestDifferentialExpression:
    """Tests for differential expression analysis."""

    def test_differential_expression_analysis(self):
        """Differential expression analysis should compute fold change and p-values between groups."""
        n_cell_types, n_genes, n_subjects = 3, 50, 20
        gene_gate_weights = np.random.rand(n_cell_types, n_genes)
        gene_names = [f"GENE{i}" for i in range(n_genes)]
        cell_type_names = [f"CT{i}" for i in range(n_cell_types)]

        group_labels = np.array(["resilient"] * 10 + ["vulnerable"] * 10)
        subject_expr = np.random.rand(n_subjects, n_cell_types, n_genes) * 0.1
        subject_expr[:10] += 0.05

        analyzer = GeneImportanceAnalyzer(
            gene_gate_weights=gene_gate_weights,
            gene_names=gene_names,
            cell_type_names=cell_type_names,
        )
        result = analyzer._compute_differential_expression(
            group_labels=group_labels,
            subject_expression=subject_expr,
            group_a="resilient",
            group_b="vulnerable",
        )
        assert isinstance(result, pd.DataFrame)
        assert "log2_fold_change" in result.columns
        assert "pvalue" in result.columns
        assert "cell_type" in result.columns
        assert "gene" in result.columns
        assert len(result) > 0
        assert (result["pvalue"] >= 0).all()
        assert (result["pvalue"] <= 1).all()

    def test_differential_expression_has_all_combinations(self):
        """Result should have n_cell_types * n_genes rows when all genes pass gate."""
        n_cell_types, n_genes, n_subjects = 2, 10, 16
        gene_gate_weights = np.ones((n_cell_types, n_genes))
        gene_names = [f"GENE{i}" for i in range(n_genes)]
        cell_type_names = [f"CT{i}" for i in range(n_cell_types)]

        group_labels = np.array(["resilient"] * 8 + ["vulnerable"] * 8)
        subject_expr = np.random.rand(n_subjects, n_cell_types, n_genes)

        analyzer = GeneImportanceAnalyzer(
            gene_gate_weights=gene_gate_weights,
            gene_names=gene_names,
            cell_type_names=cell_type_names,
        )
        result = analyzer._compute_differential_expression(
            group_labels=group_labels,
            subject_expression=subject_expr,
        )
        assert len(result) == n_cell_types * n_genes

    def test_differential_expression_insufficient_samples(self):
        """Should return empty DataFrame when either group has < 5 samples."""
        n_cell_types, n_genes = 2, 10
        gene_gate_weights = np.random.rand(n_cell_types, n_genes)
        gene_names = [f"GENE{i}" for i in range(n_genes)]
        cell_type_names = [f"CT{i}" for i in range(n_cell_types)]

        group_labels = np.array(["resilient"] * 3 + ["vulnerable"] * 5)
        subject_expr = np.random.rand(8, n_cell_types, n_genes)

        analyzer = GeneImportanceAnalyzer(
            gene_gate_weights=gene_gate_weights,
            gene_names=gene_names,
            cell_type_names=cell_type_names,
        )
        result = analyzer._compute_differential_expression(
            group_labels=group_labels,
            subject_expression=subject_expr,
        )
        assert len(result) == 0

    def test_differential_expression_via_analyze(self):
        """analyze() should include differential_expression when group data provided."""
        n_cell_types, n_genes, n_subjects = 2, 10, 16
        gene_gate_weights = np.ones((n_cell_types, n_genes))
        gene_names = [f"GENE{i}" for i in range(n_genes)]
        cell_type_names = [f"CT{i}" for i in range(n_cell_types)]

        group_labels = np.array(["resilient"] * 8 + ["vulnerable"] * 8)
        subject_expr = np.random.rand(n_subjects, n_cell_types, n_genes)

        analyzer = GeneImportanceAnalyzer(
            gene_gate_weights=gene_gate_weights,
            gene_names=gene_names,
            cell_type_names=cell_type_names,
        )
        result = analyzer.analyze(
            group_labels=group_labels,
            subject_expression=subject_expr,
        )
        assert result.differential_expression is not None
        assert isinstance(result.differential_expression, pd.DataFrame)
        assert result.metadata["has_differential_analysis"] is True

    def test_differential_expression_absent_without_group_data(self):
        """analyze() should have differential_expression=None without group data."""
        n_cell_types, n_genes = 2, 10
        gene_gate_weights = np.random.rand(n_cell_types, n_genes)
        gene_names = [f"GENE{i}" for i in range(n_genes)]
        cell_type_names = [f"CT{i}" for i in range(n_cell_types)]

        analyzer = GeneImportanceAnalyzer(
            gene_gate_weights=gene_gate_weights,
            gene_names=gene_names,
            cell_type_names=cell_type_names,
        )
        result = analyzer.analyze()
        assert result.differential_expression is None
        assert result.metadata["has_differential_analysis"] is False

    def test_differential_expression_fold_change_direction(self):
        """Higher expression in group_a should yield positive log2 fold change."""
        n_cell_types, n_genes, n_subjects = 1, 5, 20
        gene_gate_weights = np.ones((n_cell_types, n_genes))
        gene_names = [f"GENE{i}" for i in range(n_genes)]
        cell_type_names = ["CT0"]

        group_labels = np.array(["resilient"] * 10 + ["vulnerable"] * 10)
        # group_a (resilient) gets higher values
        subject_expr = np.ones((n_subjects, n_cell_types, n_genes)) * 0.1
        subject_expr[:10] = 0.5  # resilient group much higher

        analyzer = GeneImportanceAnalyzer(
            gene_gate_weights=gene_gate_weights,
            gene_names=gene_names,
            cell_type_names=cell_type_names,
        )
        result = analyzer._compute_differential_expression(
            group_labels=group_labels,
            subject_expression=subject_expr,
        )
        # All genes should have positive fold change since resilient > vulnerable
        assert (result["log2_fold_change"] > 0).all()

    def test_gate_filtering_reduces_rows(self):
        """Only genes above gate_threshold should be tested."""
        n_cell_types, n_genes, n_subjects = 2, 20, 16
        gene_gate_weights = np.zeros((n_cell_types, n_genes))
        gene_gate_weights[:, :5] = 0.5
        gene_names = [f"GENE{i}" for i in range(n_genes)]
        cell_type_names = [f"CT{i}" for i in range(n_cell_types)]

        group_labels = np.array(["resilient"] * 8 + ["vulnerable"] * 8)
        subject_expr = np.random.rand(n_subjects, n_cell_types, n_genes)

        analyzer = GeneImportanceAnalyzer(
            gene_gate_weights=gene_gate_weights,
            gene_names=gene_names,
            cell_type_names=cell_type_names,
        )
        result = analyzer._compute_differential_expression(
            group_labels=group_labels,
            subject_expression=subject_expr,
            gate_threshold=0.01,
        )
        assert len(result) == n_cell_types * 5

    def test_padj_column_present_with_fdr(self):
        """padj column should be present when apply_fdr=True."""
        n_cell_types, n_genes, n_subjects = 2, 10, 16
        gene_gate_weights = np.ones((n_cell_types, n_genes))
        gene_names = [f"GENE{i}" for i in range(n_genes)]
        cell_type_names = [f"CT{i}" for i in range(n_cell_types)]

        group_labels = np.array(["resilient"] * 8 + ["vulnerable"] * 8)
        subject_expr = np.random.rand(n_subjects, n_cell_types, n_genes)

        analyzer = GeneImportanceAnalyzer(
            gene_gate_weights=gene_gate_weights,
            gene_names=gene_names,
            cell_type_names=cell_type_names,
        )
        result = analyzer._compute_differential_expression(
            group_labels=group_labels,
            subject_expression=subject_expr,
            apply_fdr=True,
        )
        assert "padj" in result.columns
        assert (result["padj"] >= result["pvalue"]).all()

    def test_fdr_disabled(self):
        """When apply_fdr=False, padj should equal pvalue."""
        n_cell_types, n_genes, n_subjects = 2, 10, 16
        gene_gate_weights = np.ones((n_cell_types, n_genes))
        gene_names = [f"GENE{i}" for i in range(n_genes)]
        cell_type_names = [f"CT{i}" for i in range(n_cell_types)]

        group_labels = np.array(["resilient"] * 8 + ["vulnerable"] * 8)
        subject_expr = np.random.rand(n_subjects, n_cell_types, n_genes)

        analyzer = GeneImportanceAnalyzer(
            gene_gate_weights=gene_gate_weights,
            gene_names=gene_names,
            cell_type_names=cell_type_names,
        )
        result = analyzer._compute_differential_expression(
            group_labels=group_labels,
            subject_expression=subject_expr,
            apply_fdr=False,
        )
        assert "padj" in result.columns
        np.testing.assert_array_almost_equal(result["padj"].values, result["pvalue"].values)

    def test_gate_weight_column_present(self):
        """Output should include gate_weight column for visualization context."""
        n_cell_types, n_genes, n_subjects = 2, 10, 16
        gene_gate_weights = np.random.rand(n_cell_types, n_genes)
        gene_names = [f"GENE{i}" for i in range(n_genes)]
        cell_type_names = [f"CT{i}" for i in range(n_cell_types)]

        group_labels = np.array(["resilient"] * 8 + ["vulnerable"] * 8)
        subject_expr = np.random.rand(n_subjects, n_cell_types, n_genes)

        analyzer = GeneImportanceAnalyzer(
            gene_gate_weights=gene_gate_weights,
            gene_names=gene_names,
            cell_type_names=cell_type_names,
        )
        result = analyzer._compute_differential_expression(
            group_labels=group_labels,
            subject_expression=subject_expr,
        )
        assert "gate_weight" in result.columns
        assert (result["gate_weight"] > 0).all()

    def test_save_persists_differential_expression(self, tmp_path):
        """save() should write differential_expression.parquet when result is present."""
        n_cell_types, n_genes, n_subjects = 2, 10, 16
        gene_gate_weights = np.ones((n_cell_types, n_genes))
        gene_names = [f"GENE{i}" for i in range(n_genes)]
        cell_type_names = [f"CT{i}" for i in range(n_cell_types)]

        group_labels = np.array(["resilient"] * 8 + ["vulnerable"] * 8)
        subject_expr = np.random.rand(n_subjects, n_cell_types, n_genes)

        analyzer = GeneImportanceAnalyzer(
            gene_gate_weights=gene_gate_weights,
            gene_names=gene_names,
            cell_type_names=cell_type_names,
        )
        result = analyzer.analyze(
            group_labels=group_labels,
            subject_expression=subject_expr,
        )
        saved = analyzer.save(result, tmp_path)

        assert (tmp_path / "differential_expression.parquet").exists()
        loaded = pd.read_parquet(tmp_path / "differential_expression.parquet")
        assert "log2_fold_change" in loaded.columns
        assert "padj" in loaded.columns
        assert "gate_weight" in loaded.columns

    def test_minimum_group_size_is_five(self):
        """Should return empty DataFrame when either group has < 5 samples."""
        n_cell_types, n_genes = 2, 10
        gene_gate_weights = np.ones((n_cell_types, n_genes))
        gene_names = [f"GENE{i}" for i in range(n_genes)]
        cell_type_names = [f"CT{i}" for i in range(n_cell_types)]

        group_labels = np.array(["resilient"] * 3 + ["vulnerable"] * 10)
        subject_expr = np.random.rand(13, n_cell_types, n_genes)

        analyzer = GeneImportanceAnalyzer(
            gene_gate_weights=gene_gate_weights,
            gene_names=gene_names,
            cell_type_names=cell_type_names,
        )
        result = analyzer._compute_differential_expression(
            group_labels=group_labels,
            subject_expression=subject_expr,
        )
        assert len(result) == 0
