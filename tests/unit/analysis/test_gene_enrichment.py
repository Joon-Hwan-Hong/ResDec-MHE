"""
Tests for src/analysis/gene_enrichment.py.

Test coverage includes:
- GeneEnrichmentResult dataclass behavior
- GeneEnrichmentAnalyzer initialization and validation
- decoupler ULM/consensus workflow (mocked)
- gseapy prerank workflow (mocked)
- Output DataFrame column validation
- Per-cell-type result generation
- Save method creates expected files
- Edge cases: empty networks, failed methods
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.analysis.gene_enrichment import (
    GeneEnrichmentAnalyzer,
    GeneEnrichmentResult,
    compute_gene_enrichment,
    _convert_net_to_gseapy_dict,
    MSIGDB_COLLECTION_MAP,
    DEFAULT_GENE_SET_COLLECTIONS,
)

# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_gene_names():
    """20 sample gene names."""
    return [f"GENE_{i}" for i in range(20)]

@pytest.fixture
def sample_cell_type_names():
    """3 sample cell type names."""
    return ["Astrocyte", "Microglia", "Oligodendrocyte"]

@pytest.fixture
def sample_gene_gate_weights():
    """Sample gene gate weights [3 cell_types, 20 genes]."""
    return np.random.rand(3, 20).astype(np.float32)

@pytest.fixture
def sample_network():
    """Sample decoupler network DataFrame."""
    genes = [f"GENE_{i}" for i in range(20)]
    return pd.DataFrame({
        "source": ["PathwayA"] * 10 + ["PathwayB"] * 10,
        "target": genes[:10] + genes[5:15],
        "weight": np.random.randn(20),
    })

@pytest.fixture
def sample_decouple_result():
    """Sample return from dc.mt.decouple()."""
    cell_types = ["Astrocyte", "Microglia", "Oligodendrocyte"]
    pathways = ["PathwayA", "PathwayB"]

    return {
        "score_ulm": pd.DataFrame(
            np.random.randn(3, 2), index=cell_types, columns=pathways
        ),
        "padj_ulm": pd.DataFrame(
            np.random.rand(3, 2), index=cell_types, columns=pathways
        ),
        "score_mlm": pd.DataFrame(
            np.random.randn(3, 2), index=cell_types, columns=pathways
        ),
        "padj_mlm": pd.DataFrame(
            np.random.rand(3, 2), index=cell_types, columns=pathways
        ),
    }

@pytest.fixture
def sample_consensus_result():
    """Sample return from dc.mt.consensus()."""
    cell_types = ["Astrocyte", "Microglia", "Oligodendrocyte"]
    pathways = ["PathwayA", "PathwayB"]

    cons_scores = pd.DataFrame(
        np.random.randn(3, 2), index=cell_types, columns=pathways
    )
    cons_pvals = pd.DataFrame(
        np.random.rand(3, 2), index=cell_types, columns=pathways
    )
    return cons_scores, cons_pvals

@pytest.fixture
def mock_gseapy_prerank_result():
    """Sample gseapy prerank result object."""
    result = MagicMock()
    result.res2d = pd.DataFrame({
        "Name": ["prerank", "prerank"],
        "Term": ["PathwayA", "PathwayB"],
        "ES": [0.65, -0.32],
        "NES": [1.85, -1.12],
        "NOM p-val": [0.001, 0.05],
        "FDR q-val": [0.005, 0.1],
        "FWER p-val": [0.002, 0.08],
        "Tag %": ["30%", "20%"],
        "Gene %": ["25%", "15%"],
        "Lead_genes": ["GENE_0;GENE_1;GENE_2", "GENE_5;GENE_6"],
    })
    return result

@pytest.fixture
def analyzer(sample_gene_gate_weights, sample_gene_names, sample_cell_type_names):
    """GeneEnrichmentAnalyzer with mocked collections."""
    return GeneEnrichmentAnalyzer(
        gene_gate_weights=sample_gene_gate_weights,
        gene_names=sample_gene_names,
        cell_type_names=sample_cell_type_names,
        gene_set_collections=["hallmark"],
    )

# ============================================================================
# GeneEnrichmentResult Dataclass Tests
# ============================================================================

class TestGeneEnrichmentResult:
    """Tests for GeneEnrichmentResult dataclass."""

    def test_init_with_required_fields(self):
        """Result can be initialized with required fields."""
        result = GeneEnrichmentResult(
            decoupler_scores=pd.DataFrame(),
            gsea_results=pd.DataFrame(),
            consensus=pd.DataFrame(),
        )
        assert isinstance(result.decoupler_scores, pd.DataFrame)
        assert isinstance(result.gsea_results, pd.DataFrame)
        assert isinstance(result.consensus, pd.DataFrame)

    def test_metadata_defaults_to_empty_dict(self):
        """metadata defaults to empty dict."""
        result = GeneEnrichmentResult(
            decoupler_scores=pd.DataFrame(),
            gsea_results=pd.DataFrame(),
            consensus=pd.DataFrame(),
        )
        assert result.metadata == {}

    def test_metadata_set_explicitly(self):
        """metadata can be set explicitly."""
        meta = {"n_cell_types": 5}
        result = GeneEnrichmentResult(
            decoupler_scores=pd.DataFrame(),
            gsea_results=pd.DataFrame(),
            consensus=pd.DataFrame(),
            metadata=meta,
        )
        assert result.metadata == {"n_cell_types": 5}

# ============================================================================
# GeneEnrichmentAnalyzer Initialization Tests
# ============================================================================

class TestAnalyzerInit:
    """Tests for GeneEnrichmentAnalyzer initialization."""

    def test_init_with_required_args(self, sample_gene_gate_weights, sample_gene_names):
        """Analyzer initializes with weights and gene names."""
        analyzer = GeneEnrichmentAnalyzer(
            gene_gate_weights=sample_gene_gate_weights,
            gene_names=sample_gene_names,
        )
        assert analyzer.n_cell_types == 3
        assert analyzer.n_genes == 20

    def test_init_generates_default_cell_type_names(
        self, sample_gene_gate_weights, sample_gene_names
    ):
        """Analyzer generates default cell type names if not provided."""
        analyzer = GeneEnrichmentAnalyzer(
            gene_gate_weights=sample_gene_gate_weights,
            gene_names=sample_gene_names,
        )
        assert analyzer.cell_type_names == [
            "cell_type_0",
            "cell_type_1",
            "cell_type_2",
        ]

    def test_init_uses_default_collections(
        self, sample_gene_gate_weights, sample_gene_names
    ):
        """Analyzer uses DEFAULT_GENE_SET_COLLECTIONS when not specified."""
        analyzer = GeneEnrichmentAnalyzer(
            gene_gate_weights=sample_gene_gate_weights,
            gene_names=sample_gene_names,
        )
        assert analyzer.gene_set_collections == DEFAULT_GENE_SET_COLLECTIONS

    def test_init_accepts_custom_collections(
        self, sample_gene_gate_weights, sample_gene_names
    ):
        """Analyzer accepts custom gene set collections."""
        analyzer = GeneEnrichmentAnalyzer(
            gene_gate_weights=sample_gene_gate_weights,
            gene_names=sample_gene_names,
            gene_set_collections=["hallmark", "progeny"],
        )
        assert analyzer.gene_set_collections == ["hallmark", "progeny"]

    def test_init_rejects_1d_weights(self, sample_gene_names):
        """Analyzer rejects 1D weight array."""
        with pytest.raises(ValueError, match="must be 2D"):
            GeneEnrichmentAnalyzer(
                gene_gate_weights=np.random.rand(20),
                gene_names=sample_gene_names,
            )

    def test_init_rejects_3d_weights(self, sample_gene_names):
        """Analyzer rejects 3D weight array."""
        with pytest.raises(ValueError, match="must be 2D"):
            GeneEnrichmentAnalyzer(
                gene_gate_weights=np.random.rand(3, 20, 5),
                gene_names=sample_gene_names,
            )

    def test_init_rejects_mismatched_gene_names(self, sample_gene_gate_weights):
        """Analyzer rejects gene_names with wrong length."""
        with pytest.raises(ValueError, match="gene_names"):
            GeneEnrichmentAnalyzer(
                gene_gate_weights=sample_gene_gate_weights,
                gene_names=["A", "B"],  # Too short
            )

    def test_init_rejects_mismatched_cell_type_names(
        self, sample_gene_gate_weights, sample_gene_names
    ):
        """Analyzer rejects cell_type_names with wrong length."""
        with pytest.raises(ValueError, match="cell_type_names"):
            GeneEnrichmentAnalyzer(
                gene_gate_weights=sample_gene_gate_weights,
                gene_names=sample_gene_names,
                cell_type_names=["A"],  # Too short
            )

# ============================================================================
# Helper Function Tests
# ============================================================================

class TestConvertNetToGseapyDict:
    """Tests for _convert_net_to_gseapy_dict helper."""

    def test_basic_conversion(self, sample_network):
        """Converts network DataFrame to dict of gene lists."""
        gene_sets = _convert_net_to_gseapy_dict(sample_network)
        assert isinstance(gene_sets, dict)
        assert "PathwayA" in gene_sets
        assert "PathwayB" in gene_sets
        assert all(isinstance(v, list) for v in gene_sets.values())

    def test_gene_lists_are_unique(self):
        """Duplicate targets within a source are deduplicated."""
        net = pd.DataFrame({
            "source": ["S1", "S1", "S1"],
            "target": ["A", "A", "B"],
            "weight": [1.0, 1.0, 1.0],
        })
        gene_sets = _convert_net_to_gseapy_dict(net)
        assert len(gene_sets["S1"]) == 2  # A, B

    def test_empty_network(self):
        """Empty network produces empty dict."""
        net = pd.DataFrame(columns=["source", "target", "weight"])
        gene_sets = _convert_net_to_gseapy_dict(net)
        assert gene_sets == {}

# ============================================================================
# Decoupler Workflow Tests (Mocked)
# ============================================================================

class TestDecouplerWorkflow:
    """Tests for the decoupler ULM + consensus workflow."""

    def test_run_decoupler_returns_rows(
        self,
        analyzer,
        sample_decouple_result,
        sample_consensus_result,
    ):
        """_run_decoupler returns non-empty row lists on success."""
        mat = pd.DataFrame(
            analyzer.gene_gate_weights,
            index=analyzer.cell_type_names,
            columns=analyzer.gene_names,
        )

        with patch("src.analysis.gene_enrichment.dc") as mock_dc:
            mock_dc.mt.decouple.return_value = sample_decouple_result
            mock_dc.mt.consensus.return_value = sample_consensus_result

            dc_rows, cons_rows = analyzer._run_decoupler(
                mat, pd.DataFrame({"source": [], "target": [], "weight": []}),
                "hallmark",
            )

        assert len(dc_rows) > 0
        assert len(cons_rows) > 0

    def test_decoupler_rows_have_expected_keys(
        self,
        analyzer,
        sample_decouple_result,
        sample_consensus_result,
    ):
        """Decoupler result rows contain expected keys."""
        mat = pd.DataFrame(
            analyzer.gene_gate_weights,
            index=analyzer.cell_type_names,
            columns=analyzer.gene_names,
        )

        with patch("src.analysis.gene_enrichment.dc") as mock_dc:
            mock_dc.mt.decouple.return_value = sample_decouple_result
            mock_dc.mt.consensus.return_value = sample_consensus_result

            dc_rows, cons_rows = analyzer._run_decoupler(
                mat, pd.DataFrame(), "hallmark"
            )

        expected_dc_keys = {"cell_type", "source", "score", "pvalue", "method", "collection"}
        expected_cons_keys = {
            "cell_type", "source", "consensus_score", "consensus_pvalue", "collection"
        }

        assert set(dc_rows[0].keys()) == expected_dc_keys
        assert set(cons_rows[0].keys()) == expected_cons_keys

    def test_decoupler_per_cell_type_results(
        self,
        analyzer,
        sample_decouple_result,
        sample_consensus_result,
    ):
        """Decoupler produces results for all cell types."""
        mat = pd.DataFrame(
            analyzer.gene_gate_weights,
            index=analyzer.cell_type_names,
            columns=analyzer.gene_names,
        )

        with patch("src.analysis.gene_enrichment.dc") as mock_dc:
            mock_dc.mt.decouple.return_value = sample_decouple_result
            mock_dc.mt.consensus.return_value = sample_consensus_result

            dc_rows, _ = analyzer._run_decoupler(mat, pd.DataFrame(), "hallmark")

        cell_types_in_results = {r["cell_type"] for r in dc_rows}
        assert cell_types_in_results == set(analyzer.cell_type_names)

    def test_decoupler_handles_assertion_error(self, analyzer):
        """_run_decoupler handles AssertionError gracefully."""
        mat = pd.DataFrame(
            analyzer.gene_gate_weights,
            index=analyzer.cell_type_names,
            columns=analyzer.gene_names,
        )

        with patch("src.analysis.gene_enrichment.dc") as mock_dc:
            mock_dc.mt.decouple.side_effect = AssertionError("No sources")

            dc_rows, cons_rows = analyzer._run_decoupler(
                mat, pd.DataFrame(), "hallmark"
            )

        assert dc_rows == []
        assert cons_rows == []

    def test_decoupler_handles_value_error(self, analyzer):
        """_run_decoupler handles ValueError gracefully."""
        mat = pd.DataFrame(
            analyzer.gene_gate_weights,
            index=analyzer.cell_type_names,
            columns=analyzer.gene_names,
        )

        with patch("src.analysis.gene_enrichment.dc") as mock_dc:
            mock_dc.mt.decouple.side_effect = ValueError("Bad data")

            dc_rows, cons_rows = analyzer._run_decoupler(
                mat, pd.DataFrame(), "hallmark"
            )

        assert dc_rows == []
        assert cons_rows == []

    def test_decoupler_both_methods_present(
        self,
        analyzer,
        sample_decouple_result,
        sample_consensus_result,
    ):
        """Results contain rows from both ULM and MLM methods."""
        mat = pd.DataFrame(
            analyzer.gene_gate_weights,
            index=analyzer.cell_type_names,
            columns=analyzer.gene_names,
        )

        with patch("src.analysis.gene_enrichment.dc") as mock_dc:
            mock_dc.mt.decouple.return_value = sample_decouple_result
            mock_dc.mt.consensus.return_value = sample_consensus_result

            dc_rows, _ = analyzer._run_decoupler(mat, pd.DataFrame(), "hallmark")

        methods = {r["method"] for r in dc_rows}
        assert methods == {"ulm", "mlm"}

# ============================================================================
# GSEA Prerank Workflow Tests (Mocked)
# ============================================================================

class TestGseapyPrerank:
    """Tests for the gseapy prerank workflow."""

    def test_prerank_returns_rows(
        self, analyzer, sample_network, mock_gseapy_prerank_result
    ):
        """_run_gseapy_prerank returns non-empty rows on success."""
        with patch("src.analysis.gene_enrichment.gp") as mock_gp:
            mock_gp.prerank.return_value = mock_gseapy_prerank_result

            rows = analyzer._run_gseapy_prerank(sample_network, "hallmark")

        assert len(rows) > 0

    def test_prerank_rows_have_expected_keys(
        self, analyzer, sample_network, mock_gseapy_prerank_result
    ):
        """GSEA result rows contain expected keys."""
        with patch("src.analysis.gene_enrichment.gp") as mock_gp:
            mock_gp.prerank.return_value = mock_gseapy_prerank_result

            rows = analyzer._run_gseapy_prerank(sample_network, "hallmark")

        expected_keys = {
            "cell_type", "term", "es", "nes", "pvalue", "fdr",
            "leading_edge", "collection",
        }
        assert set(rows[0].keys()) == expected_keys

    def test_prerank_per_cell_type(
        self, analyzer, sample_network, mock_gseapy_prerank_result
    ):
        """GSEA runs for all cell types."""
        with patch("src.analysis.gene_enrichment.gp") as mock_gp:
            mock_gp.prerank.return_value = mock_gseapy_prerank_result

            rows = analyzer._run_gseapy_prerank(sample_network, "hallmark")

        cell_types_in_results = {r["cell_type"] for r in rows}
        assert cell_types_in_results == set(analyzer.cell_type_names)

    def test_prerank_handles_exception(self, analyzer, sample_network):
        """_run_gseapy_prerank handles exceptions gracefully."""
        with patch("src.analysis.gene_enrichment.gp") as mock_gp:
            mock_gp.prerank.side_effect = RuntimeError("GSEA failed")

            rows = analyzer._run_gseapy_prerank(sample_network, "hallmark")

        assert rows == []

    def test_prerank_handles_empty_results(self, analyzer, sample_network):
        """_run_gseapy_prerank handles empty result table."""
        empty_result = MagicMock()
        empty_result.res2d = pd.DataFrame()

        with patch("src.analysis.gene_enrichment.gp") as mock_gp:
            mock_gp.prerank.return_value = empty_result

            rows = analyzer._run_gseapy_prerank(sample_network, "hallmark")

        assert rows == []

    def test_prerank_handles_none_res2d(self, analyzer, sample_network):
        """_run_gseapy_prerank handles None res2d."""
        none_result = MagicMock()
        none_result.res2d = None

        with patch("src.analysis.gene_enrichment.gp") as mock_gp:
            mock_gp.prerank.return_value = none_result

            rows = analyzer._run_gseapy_prerank(sample_network, "hallmark")

        assert rows == []

    def test_prerank_called_with_correct_params(
        self, analyzer, sample_network, mock_gseapy_prerank_result
    ):
        """gp.prerank called with expected parameters."""
        with patch("src.analysis.gene_enrichment.gp") as mock_gp:
            mock_gp.prerank.return_value = mock_gseapy_prerank_result

            analyzer._run_gseapy_prerank(sample_network, "hallmark")

        # Called once per cell type
        assert mock_gp.prerank.call_count == len(analyzer.cell_type_names)

        # Check kwargs of first call
        call_kwargs = mock_gp.prerank.call_args_list[0][1]
        assert call_kwargs["min_size"] == 5
        assert call_kwargs["max_size"] == 500
        assert call_kwargs["permutation_num"] == 1000
        assert call_kwargs["seed"] == 42
        assert call_kwargs["no_plot"] is True
        assert call_kwargs["threads"] == 1

# ============================================================================
# Full Analyze Workflow Tests (Mocked)
# ============================================================================

class TestAnalyzeWorkflow:
    """Tests for the full analyze() method."""

    def test_analyze_returns_result(
        self,
        analyzer,
        sample_network,
        sample_decouple_result,
        sample_consensus_result,
        mock_gseapy_prerank_result,
    ):
        """analyze() returns GeneEnrichmentResult."""
        with (
            patch.object(analyzer, "_fetch_network", return_value=sample_network),
            patch("src.analysis.gene_enrichment.dc") as mock_dc,
            patch("src.analysis.gene_enrichment.gp") as mock_gp,
        ):
            mock_dc.mt.decouple.return_value = sample_decouple_result
            mock_dc.mt.consensus.return_value = sample_consensus_result
            mock_gp.prerank.return_value = mock_gseapy_prerank_result

            result = analyzer.analyze()

        assert isinstance(result, GeneEnrichmentResult)

    def test_analyze_populates_all_dataframes(
        self,
        analyzer,
        sample_network,
        sample_decouple_result,
        sample_consensus_result,
        mock_gseapy_prerank_result,
    ):
        """analyze() populates decoupler_scores, gsea_results, and consensus."""
        with (
            patch.object(analyzer, "_fetch_network", return_value=sample_network),
            patch("src.analysis.gene_enrichment.dc") as mock_dc,
            patch("src.analysis.gene_enrichment.gp") as mock_gp,
        ):
            mock_dc.mt.decouple.return_value = sample_decouple_result
            mock_dc.mt.consensus.return_value = sample_consensus_result
            mock_gp.prerank.return_value = mock_gseapy_prerank_result

            result = analyzer.analyze()

        assert not result.decoupler_scores.empty
        assert not result.gsea_results.empty
        assert not result.consensus.empty

    def test_analyze_decoupler_columns(
        self,
        analyzer,
        sample_network,
        sample_decouple_result,
        sample_consensus_result,
        mock_gseapy_prerank_result,
    ):
        """decoupler_scores has expected columns."""
        with (
            patch.object(analyzer, "_fetch_network", return_value=sample_network),
            patch("src.analysis.gene_enrichment.dc") as mock_dc,
            patch("src.analysis.gene_enrichment.gp") as mock_gp,
        ):
            mock_dc.mt.decouple.return_value = sample_decouple_result
            mock_dc.mt.consensus.return_value = sample_consensus_result
            mock_gp.prerank.return_value = mock_gseapy_prerank_result

            result = analyzer.analyze()

        expected = {"cell_type", "source", "score", "pvalue", "method", "collection"}
        assert set(result.decoupler_scores.columns) == expected

    def test_analyze_gsea_columns(
        self,
        analyzer,
        sample_network,
        sample_decouple_result,
        sample_consensus_result,
        mock_gseapy_prerank_result,
    ):
        """gsea_results has expected columns."""
        with (
            patch.object(analyzer, "_fetch_network", return_value=sample_network),
            patch("src.analysis.gene_enrichment.dc") as mock_dc,
            patch("src.analysis.gene_enrichment.gp") as mock_gp,
        ):
            mock_dc.mt.decouple.return_value = sample_decouple_result
            mock_dc.mt.consensus.return_value = sample_consensus_result
            mock_gp.prerank.return_value = mock_gseapy_prerank_result

            result = analyzer.analyze()

        expected = {
            "cell_type", "term", "es", "nes", "pvalue", "fdr",
            "leading_edge", "collection",
        }
        assert set(result.gsea_results.columns) == expected

    def test_analyze_consensus_columns(
        self,
        analyzer,
        sample_network,
        sample_decouple_result,
        sample_consensus_result,
        mock_gseapy_prerank_result,
    ):
        """consensus has expected columns."""
        with (
            patch.object(analyzer, "_fetch_network", return_value=sample_network),
            patch("src.analysis.gene_enrichment.dc") as mock_dc,
            patch("src.analysis.gene_enrichment.gp") as mock_gp,
        ):
            mock_dc.mt.decouple.return_value = sample_decouple_result
            mock_dc.mt.consensus.return_value = sample_consensus_result
            mock_gp.prerank.return_value = mock_gseapy_prerank_result

            result = analyzer.analyze()

        expected = {
            "cell_type", "source", "consensus_score", "consensus_pvalue", "collection",
        }
        assert set(result.consensus.columns) == expected

    def test_analyze_metadata(
        self,
        analyzer,
        sample_network,
        sample_decouple_result,
        sample_consensus_result,
        mock_gseapy_prerank_result,
    ):
        """analyze() metadata contains expected keys."""
        with (
            patch.object(analyzer, "_fetch_network", return_value=sample_network),
            patch("src.analysis.gene_enrichment.dc") as mock_dc,
            patch("src.analysis.gene_enrichment.gp") as mock_gp,
        ):
            mock_dc.mt.decouple.return_value = sample_decouple_result
            mock_dc.mt.consensus.return_value = sample_consensus_result
            mock_gp.prerank.return_value = mock_gseapy_prerank_result

            result = analyzer.analyze(top_k_ora=50)

        assert result.metadata["n_cell_types"] == 3
        assert result.metadata["n_genes"] == 20
        assert result.metadata["top_k_ora"] == 50
        assert "collections" in result.metadata

    def test_analyze_empty_network_skipped(self, analyzer):
        """analyze() skips collections with empty networks."""
        empty_net = pd.DataFrame(columns=["source", "target", "weight"])

        with patch.object(analyzer, "_fetch_network", return_value=empty_net):
            result = analyzer.analyze()

        assert result.decoupler_scores.empty
        assert result.gsea_results.empty
        assert result.consensus.empty

    def test_analyze_multiple_collections(
        self,
        sample_gene_gate_weights,
        sample_gene_names,
        sample_cell_type_names,
        sample_network,
        sample_decouple_result,
        sample_consensus_result,
        mock_gseapy_prerank_result,
    ):
        """analyze() processes multiple collections."""
        multi_analyzer = GeneEnrichmentAnalyzer(
            gene_gate_weights=sample_gene_gate_weights,
            gene_names=sample_gene_names,
            cell_type_names=sample_cell_type_names,
            gene_set_collections=["hallmark", "kegg"],
        )

        with (
            patch.object(multi_analyzer, "_fetch_network", return_value=sample_network),
            patch("src.analysis.gene_enrichment.dc") as mock_dc,
            patch("src.analysis.gene_enrichment.gp") as mock_gp,
        ):
            mock_dc.mt.decouple.return_value = sample_decouple_result
            mock_dc.mt.consensus.return_value = sample_consensus_result
            mock_gp.prerank.return_value = mock_gseapy_prerank_result

            result = multi_analyzer.analyze()

        collections_in_dc = set(result.decoupler_scores["collection"].unique())
        assert collections_in_dc == {"hallmark", "kegg"}

# ============================================================================
# Save Tests
# ============================================================================

class TestSave:
    """Tests for save method."""

    def _make_result(self) -> GeneEnrichmentResult:
        """Create a non-empty GeneEnrichmentResult for save tests."""
        return GeneEnrichmentResult(
            decoupler_scores=pd.DataFrame({
                "cell_type": ["A", "B"],
                "source": ["P1", "P1"],
                "score": [1.0, 2.0],
                "pvalue": [0.01, 0.05],
                "method": ["ulm", "ulm"],
                "collection": ["hallmark", "hallmark"],
            }),
            gsea_results=pd.DataFrame({
                "cell_type": ["A", "B"],
                "term": ["P1", "P1"],
                "es": [0.5, -0.3],
                "nes": [1.5, -1.1],
                "pvalue": [0.01, 0.05],
                "fdr": [0.02, 0.1],
                "leading_edge": ["G1;G2", "G3"],
                "collection": ["hallmark", "hallmark"],
            }),
            consensus=pd.DataFrame({
                "cell_type": ["A", "B"],
                "source": ["P1", "P1"],
                "consensus_score": [1.2, 1.8],
                "consensus_pvalue": [0.03, 0.07],
                "collection": ["hallmark", "hallmark"],
            }),
            metadata={"n_cell_types": 2, "n_genes": 10},
        )

    def test_save_creates_expected_files(self, tmp_path):
        """save() creates parquet, csv, and json files."""
        result = self._make_result()
        analyzer = GeneEnrichmentAnalyzer(
            gene_gate_weights=np.random.rand(2, 10),
            gene_names=[f"G{i}" for i in range(10)],
            cell_type_names=["A", "B"],
        )

        saved = analyzer.save(result, tmp_path)

        assert (tmp_path / "decoupler_scores.parquet").exists()
        assert (tmp_path / "decoupler_scores.csv").exists()
        assert (tmp_path / "gsea_results.parquet").exists()
        assert (tmp_path / "gsea_results.csv").exists()
        assert (tmp_path / "consensus_scores.parquet").exists()
        assert (tmp_path / "consensus_scores.csv").exists()
        assert (tmp_path / "gene_enrichment_metadata.json").exists()

    def test_save_metadata_json_readable(self, tmp_path):
        """Saved metadata JSON is readable."""
        result = self._make_result()
        analyzer = GeneEnrichmentAnalyzer(
            gene_gate_weights=np.random.rand(2, 10),
            gene_names=[f"G{i}" for i in range(10)],
            cell_type_names=["A", "B"],
        )

        analyzer.save(result, tmp_path)

        with open(tmp_path / "gene_enrichment_metadata.json") as f:
            meta = json.load(f)
        assert meta["n_cell_types"] == 2

    def test_save_parquet_roundtrip(self, tmp_path):
        """Saved parquet files can be loaded back."""
        result = self._make_result()
        analyzer = GeneEnrichmentAnalyzer(
            gene_gate_weights=np.random.rand(2, 10),
            gene_names=[f"G{i}" for i in range(10)],
            cell_type_names=["A", "B"],
        )

        analyzer.save(result, tmp_path)

        loaded_dc = pd.read_parquet(tmp_path / "decoupler_scores.parquet")
        assert set(loaded_dc.columns) == set(result.decoupler_scores.columns)
        assert len(loaded_dc) == len(result.decoupler_scores)

    def test_save_csv_only_format(self, tmp_path):
        """save() with formats=['csv'] creates only csv files."""
        result = self._make_result()
        analyzer = GeneEnrichmentAnalyzer(
            gene_gate_weights=np.random.rand(2, 10),
            gene_names=[f"G{i}" for i in range(10)],
            cell_type_names=["A", "B"],
        )

        saved = analyzer.save(result, tmp_path, formats=["csv"])

        assert (tmp_path / "decoupler_scores.csv").exists()
        assert not (tmp_path / "decoupler_scores.parquet").exists()

    def test_save_creates_output_dir(self, tmp_path):
        """save() creates output directory if it doesn't exist."""
        result = self._make_result()
        analyzer = GeneEnrichmentAnalyzer(
            gene_gate_weights=np.random.rand(2, 10),
            gene_names=[f"G{i}" for i in range(10)],
            cell_type_names=["A", "B"],
        )

        new_dir = tmp_path / "nested" / "output"
        analyzer.save(result, new_dir)

        assert new_dir.exists()
        assert (new_dir / "decoupler_scores.parquet").exists()

    def test_save_empty_results(self, tmp_path):
        """save() handles empty DataFrames without error."""
        result = GeneEnrichmentResult(
            decoupler_scores=pd.DataFrame(),
            gsea_results=pd.DataFrame(),
            consensus=pd.DataFrame(),
            metadata={"empty": True},
        )
        analyzer = GeneEnrichmentAnalyzer(
            gene_gate_weights=np.random.rand(2, 10),
            gene_names=[f"G{i}" for i in range(10)],
            cell_type_names=["A", "B"],
        )

        saved = analyzer.save(result, tmp_path)

        # Empty DataFrames should not produce files
        assert not (tmp_path / "decoupler_scores.parquet").exists()
        assert not (tmp_path / "gsea_results.parquet").exists()
        # But metadata always saved
        assert (tmp_path / "gene_enrichment_metadata.json").exists()

    def test_save_returns_path_dict(self, tmp_path):
        """save() returns dict mapping names to file paths."""
        result = self._make_result()
        analyzer = GeneEnrichmentAnalyzer(
            gene_gate_weights=np.random.rand(2, 10),
            gene_names=[f"G{i}" for i in range(10)],
            cell_type_names=["A", "B"],
        )

        saved = analyzer.save(result, tmp_path)

        assert isinstance(saved, dict)
        assert "metadata" in saved
        assert "decoupler_scores_parquet" in saved
        assert all(isinstance(v, Path) for v in saved.values())

# ============================================================================
# Convenience Function Tests
# ============================================================================

class TestConvenienceFunction:
    """Tests for compute_gene_enrichment convenience function."""

    def test_compute_returns_result(
        self,
        sample_gene_gate_weights,
        sample_gene_names,
        sample_cell_type_names,
        sample_network,
        sample_decouple_result,
        sample_consensus_result,
        mock_gseapy_prerank_result,
    ):
        """compute_gene_enrichment returns GeneEnrichmentResult."""
        with (
            patch(
                "src.analysis.gene_enrichment.GeneEnrichmentAnalyzer._fetch_network",
                return_value=sample_network,
            ),
            patch("src.analysis.gene_enrichment.dc") as mock_dc,
            patch("src.analysis.gene_enrichment.gp") as mock_gp,
        ):
            mock_dc.mt.decouple.return_value = sample_decouple_result
            mock_dc.mt.consensus.return_value = sample_consensus_result
            mock_gp.prerank.return_value = mock_gseapy_prerank_result

            result = compute_gene_enrichment(
                gene_gate_weights=sample_gene_gate_weights,
                gene_names=sample_gene_names,
                cell_type_names=sample_cell_type_names,
                gene_set_collections=["hallmark"],
            )

        assert isinstance(result, GeneEnrichmentResult)

    def test_compute_with_output_dir(
        self,
        tmp_path,
        sample_gene_gate_weights,
        sample_gene_names,
        sample_cell_type_names,
        sample_network,
        sample_decouple_result,
        sample_consensus_result,
        mock_gseapy_prerank_result,
    ):
        """compute_gene_enrichment saves when output_dir provided."""
        with (
            patch(
                "src.analysis.gene_enrichment.GeneEnrichmentAnalyzer._fetch_network",
                return_value=sample_network,
            ),
            patch("src.analysis.gene_enrichment.dc") as mock_dc,
            patch("src.analysis.gene_enrichment.gp") as mock_gp,
        ):
            mock_dc.mt.decouple.return_value = sample_decouple_result
            mock_dc.mt.consensus.return_value = sample_consensus_result
            mock_gp.prerank.return_value = mock_gseapy_prerank_result

            compute_gene_enrichment(
                gene_gate_weights=sample_gene_gate_weights,
                gene_names=sample_gene_names,
                cell_type_names=sample_cell_type_names,
                gene_set_collections=["hallmark"],
                output_dir=tmp_path,
            )

        assert (tmp_path / "gene_enrichment_metadata.json").exists()

# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Edge case tests."""

    def test_single_cell_type(self, sample_gene_names):
        """Handles single cell type."""
        analyzer = GeneEnrichmentAnalyzer(
            gene_gate_weights=np.random.rand(1, 20).astype(np.float32),
            gene_names=sample_gene_names,
            cell_type_names=["SingleType"],
            gene_set_collections=["hallmark"],
        )
        assert analyzer.n_cell_types == 1

    def test_single_gene(self):
        """Handles single gene."""
        analyzer = GeneEnrichmentAnalyzer(
            gene_gate_weights=np.random.rand(3, 1).astype(np.float32),
            gene_names=["ONLY_GENE"],
            cell_type_names=["A", "B", "C"],
            gene_set_collections=["hallmark"],
        )
        assert analyzer.n_genes == 1

    def test_unknown_collection_raises(
        self, sample_gene_gate_weights, sample_gene_names
    ):
        """Unknown collection in _fetch_network raises ValueError."""
        analyzer = GeneEnrichmentAnalyzer(
            gene_gate_weights=sample_gene_gate_weights,
            gene_names=sample_gene_names,
            gene_set_collections=["nonexistent"],
        )
        with pytest.raises(ValueError, match="Unknown gene set collection"):
            analyzer._fetch_network("nonexistent")

    def test_collection_label_in_results(
        self,
        analyzer,
        sample_network,
        sample_decouple_result,
        sample_consensus_result,
        mock_gseapy_prerank_result,
    ):
        """Collection name appears in all result rows."""
        with (
            patch.object(analyzer, "_fetch_network", return_value=sample_network),
            patch("src.analysis.gene_enrichment.dc") as mock_dc,
            patch("src.analysis.gene_enrichment.gp") as mock_gp,
        ):
            mock_dc.mt.decouple.return_value = sample_decouple_result
            mock_dc.mt.consensus.return_value = sample_consensus_result
            mock_gp.prerank.return_value = mock_gseapy_prerank_result

            result = analyzer.analyze()

        assert (result.decoupler_scores["collection"] == "hallmark").all()
        assert (result.gsea_results["collection"] == "hallmark").all()
        assert (result.consensus["collection"] == "hallmark").all()
