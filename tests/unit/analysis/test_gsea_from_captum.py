"""Tests for scripts/resdec_mhe/interpretability/gsea_from_captum.py (Task C.6).

Covers the pure-function helpers used by the GSEA adapter:

- ``build_gene_list_from_summary``: pulls top-K gene symbols from the Captum
  ``composite_attribution_summary.json`` structure.
- ``hypergeometric_overlap_pvalue``: one-sided hypergeometric p-value for
  overlap of a ranked gene list with the AD-GWAS reference set, restricted to
  a specified universe (the 4785 HVG input to the model).
- ``compute_ad_gwas_overlap``: wraps the hypergeometric test with the
  manually-curated Bellenguez 2022 + Wightman 2021 AD-GWAS list.
- ``AD_GWAS_GENES``: the curated list itself (non-empty, valid HUGO symbols,
  uppercase).

The Enrichr-hitting path (``run_enrichr_for_gene_list``) is *not* exercised
here to keep unit tests offline; it is covered manually by running the
script and inspecting outputs.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.stats import hypergeom

# Make the script importable as a module.
_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS_DIR = _WORKTREE_ROOT / "scripts" / "resdec_mhe" / "interpretability"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from gsea_from_captum import (  # noqa: E402
    AD_GWAS_GENES,
    build_gene_list_from_npz,
    build_gene_list_from_summary,
    compute_ad_gwas_overlap,
    hypergeometric_overlap_pvalue,
    rank_cell_types_from_summary,
)


# ---------------------------------------------------------------------------
# AD_GWAS_GENES constant
# ---------------------------------------------------------------------------


class TestADGWASGenes:
    def test_is_nonempty_set_of_strings(self):
        assert isinstance(AD_GWAS_GENES, frozenset)
        assert len(AD_GWAS_GENES) >= 20
        for g in AD_GWAS_GENES:
            assert isinstance(g, str)

    def test_all_uppercase_hugo_symbols(self):
        # HUGO symbols are uppercase; no leading/trailing whitespace.
        # Dashes are allowed (e.g., HLA-DRB1). Typical length 2–20 chars.
        for g in AD_GWAS_GENES:
            assert g == g.upper(), f"{g} is not uppercase"
            assert g.strip() == g, f"{g} has whitespace"
            assert 2 <= len(g) <= 20, f"{g} has unusual length"

    def test_contains_core_ad_loci(self):
        # These are the canonical, widely-cited AD GWAS genes from both
        # Bellenguez 2022 and Wightman 2021. Any AD gene list missing these
        # is broken.
        core = {"APOE", "TREM2", "BIN1", "CLU", "CR1", "ABCA7", "PICALM"}
        assert core.issubset(AD_GWAS_GENES), (
            f"Missing core AD genes: {core - AD_GWAS_GENES}"
        )


# ---------------------------------------------------------------------------
# build_gene_list_from_summary
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_summary():
    return {
        "n_subjects": 10,
        "n_cell_types": 3,
        "n_genes": 5,
        "top_global_genes": [
            {"gene": "APOE", "mean_abs_attribution": 0.5},
            {"gene": "TREM2", "mean_abs_attribution": 0.4},
            {"gene": "BIN1", "mean_abs_attribution": 0.3},
            {"gene": "CLU", "mean_abs_attribution": 0.2},
            {"gene": "MT-CO3", "mean_abs_attribution": 0.1},
        ],
        "top_genes_per_cell_type": {
            "Splatter": [
                {"gene": "SCN3B", "mean_abs_attribution": 0.6},
                {"gene": "VAMP2", "mean_abs_attribution": 0.5},
                {"gene": "UNC5D", "mean_abs_attribution": 0.4},
            ],
            "Fibroblast": [
                {"gene": "USP53", "mean_abs_attribution": 0.55},
                {"gene": "SLC6A20", "mean_abs_attribution": 0.45},
            ],
        },
        "cell_types_ranked_by_total_attribution": [
            {"cell_type": "Splatter", "total_abs_attribution": 1.4},
            {"cell_type": "Fibroblast", "total_abs_attribution": 0.6},
            {"cell_type": "Vascular", "total_abs_attribution": 0.2},
        ],
    }


class TestBuildGeneListFromSummary:
    def test_global_top_k(self, mock_summary):
        genes = build_gene_list_from_summary(mock_summary, scope="global", top_k=3)
        assert genes == ["APOE", "TREM2", "BIN1"]

    def test_global_top_k_truncates_to_available(self, mock_summary):
        # Only 5 entries in mock; asking for 10 returns all 5.
        genes = build_gene_list_from_summary(mock_summary, scope="global", top_k=10)
        assert genes == ["APOE", "TREM2", "BIN1", "CLU", "MT-CO3"]

    def test_per_celltype_top_k(self, mock_summary):
        genes = build_gene_list_from_summary(
            mock_summary, scope="cell_type", cell_type="Splatter", top_k=2
        )
        assert genes == ["SCN3B", "VAMP2"]

    def test_per_celltype_missing_raises(self, mock_summary):
        with pytest.raises(KeyError):
            build_gene_list_from_summary(
                mock_summary, scope="cell_type", cell_type="Unknown", top_k=5
            )

    def test_bad_scope_raises(self, mock_summary):
        with pytest.raises(ValueError):
            build_gene_list_from_summary(mock_summary, scope="nonsense", top_k=3)

    def test_cell_type_scope_requires_cell_type_arg(self, mock_summary):
        with pytest.raises(ValueError):
            build_gene_list_from_summary(
                mock_summary, scope="cell_type", cell_type=None, top_k=5
            )


# ---------------------------------------------------------------------------
# rank_cell_types_from_summary
# ---------------------------------------------------------------------------


class TestBuildGeneListFromNpz:
    def _make_fixture(self):
        # 3 subjects, 2 cell types, 5 genes. Gene 3 has largest mean across
        # all cell types, Gene 0 is second, etc. For cell type 0, Gene 1
        # dominates; for cell type 1, Gene 4 dominates.
        attr = np.zeros((3, 2, 5), dtype=np.float32)
        attr[:, 0, :] = [[0.1, 0.9, 0.2, 0.8, 0.3]] * 3
        attr[:, 1, :] = [[0.5, 0.1, 0.3, 0.7, 0.9]] * 3
        genes = ["A", "B", "C", "D", "E"]
        cts = ["ct0", "ct1"]
        return attr, genes, cts

    def test_global_ranking(self):
        attr, genes, cts = self._make_fixture()
        # per-gene mean across subjects and CTs:
        #   A: (0.1+0.5)/2=0.3, B: (0.9+0.1)/2=0.5, C: (0.2+0.3)/2=0.25,
        #   D: (0.8+0.7)/2=0.75, E: (0.3+0.9)/2=0.6
        # Descending: D, E, B, A, C
        out = build_gene_list_from_npz(
            attr, genes, cts, scope="global", top_k=3
        )
        assert out == ["D", "E", "B"]

    def test_cell_type_ranking(self):
        attr, genes, cts = self._make_fixture()
        # ct0: B=0.9, D=0.8, E=0.3, C=0.2, A=0.1 → descending order
        out = build_gene_list_from_npz(
            attr, genes, cts, scope="cell_type", cell_type="ct0", top_k=3
        )
        assert out == ["B", "D", "E"]
        # ct1: E=0.9, D=0.7, A=0.5, C=0.3, B=0.1
        out2 = build_gene_list_from_npz(
            attr, genes, cts, scope="cell_type", cell_type="ct1", top_k=3
        )
        assert out2 == ["E", "D", "A"]

    def test_top_k_larger_than_genes(self):
        attr, genes, cts = self._make_fixture()
        out = build_gene_list_from_npz(
            attr, genes, cts, scope="global", top_k=100
        )
        assert len(out) == 5
        assert set(out) == set(genes)

    def test_unknown_cell_type_raises(self):
        attr, genes, cts = self._make_fixture()
        with pytest.raises(KeyError):
            build_gene_list_from_npz(
                attr, genes, cts, scope="cell_type",
                cell_type="nonexistent", top_k=3,
            )

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError):
            build_gene_list_from_npz(
                np.zeros((5, 3)), ["A", "B", "C"], ["ct0", "ct1", "ct2"],
                scope="global", top_k=2,
            )

    def test_uses_abs_value(self):
        # Negative attributions should rank by magnitude, not sign.
        attr = np.zeros((2, 1, 3), dtype=np.float32)
        attr[:, 0, :] = [[-0.9, 0.1, 0.5]] * 2
        out = build_gene_list_from_npz(
            attr, ["A", "B", "C"], ["ct0"],
            scope="cell_type", cell_type="ct0", top_k=3,
        )
        # |A|=0.9, |C|=0.5, |B|=0.1
        assert out == ["A", "C", "B"]


class TestRankCellTypesFromSummary:
    def test_returns_top_n(self, mock_summary):
        ranked = rank_cell_types_from_summary(mock_summary, top_n=2)
        assert ranked == ["Splatter", "Fibroblast"]

    def test_returns_all_if_top_n_large(self, mock_summary):
        ranked = rank_cell_types_from_summary(mock_summary, top_n=10)
        assert ranked == ["Splatter", "Fibroblast", "Vascular"]


# ---------------------------------------------------------------------------
# hypergeometric_overlap_pvalue — independent scipy sanity check
# ---------------------------------------------------------------------------


class TestHypergeometricOverlap:
    def test_strong_overlap_pvalue_significant(self):
        # Universe = 4785 HVG genes. AD-GWAS has 75 hits in universe.
        # Top-K = 200 attribution genes. 12 overlap (vs ~3.1 expected) is
        # ~4x the null expectation and should be well below 1e-3.
        p = hypergeometric_overlap_pvalue(
            overlap=12, sample_size=200, n_successes_pop=75, pop_size=4785
        )
        assert p < 1e-3, f"Expected p<1e-3, got {p}"

    def test_null_overlap_pvalue_not_significant(self):
        # Expected overlap ~3.1; observing 3 should NOT be significant.
        p = hypergeometric_overlap_pvalue(
            overlap=3, sample_size=200, n_successes_pop=75, pop_size=4785
        )
        assert p > 0.05

    def test_matches_scipy_reference(self):
        # Independent reference using scipy directly.
        overlap, K, n_GWAS, N = 8, 200, 75, 4785
        # One-sided: P(X >= overlap) = sf(overlap - 1)
        expected = hypergeom.sf(overlap - 1, N, n_GWAS, K)
        got = hypergeometric_overlap_pvalue(
            overlap=overlap, sample_size=K, n_successes_pop=n_GWAS, pop_size=N
        )
        assert got == pytest.approx(expected, rel=1e-10)

    def test_zero_overlap_gives_pvalue_one(self):
        p = hypergeometric_overlap_pvalue(
            overlap=0, sample_size=200, n_successes_pop=75, pop_size=4785
        )
        assert p == pytest.approx(1.0, rel=1e-10)

    def test_invalid_args_raise(self):
        with pytest.raises(ValueError):
            hypergeometric_overlap_pvalue(
                overlap=10, sample_size=5, n_successes_pop=75, pop_size=4785
            )  # overlap > sample_size
        with pytest.raises(ValueError):
            hypergeometric_overlap_pvalue(
                overlap=-1, sample_size=200, n_successes_pop=75, pop_size=4785
            )


# ---------------------------------------------------------------------------
# compute_ad_gwas_overlap
# ---------------------------------------------------------------------------


class TestComputeADGWASOverlap:
    def test_synthetic_known_overlap(self):
        # Simulate a realistic HVG-sized universe:
        #   - 10 real AD GWAS genes in list (big overlap)
        #   - 90 non-GWAS genes in list
        #   - universe of ~4800 genes, with all AD_GWAS genes present
        # Expected overlap under null ≈ 100 * 94 / 4700 ≈ 2.0;
        # observing 10 is strongly enriched.
        ad_in_list = [
            "APOE", "TREM2", "BIN1", "CLU", "ABCA7",
            "CR1", "PICALM", "SORL1", "CD33", "CD2AP",
        ]
        non_gwas = [f"FAKE_GENE_{i}" for i in range(90)]
        gene_list = ad_in_list + non_gwas

        # Universe: realistic-sized (~4785), containing list genes + AD_GWAS.
        pad = [f"PAD_GENE_{i}" for i in range(4500)]
        universe = (
            set(ad_in_list) | set(non_gwas) | set(AD_GWAS_GENES) | set(pad)
        )

        result = compute_ad_gwas_overlap(
            gene_list=gene_list, universe=universe, gwas_genes=AD_GWAS_GENES
        )

        assert result["n_overlap"] == 10
        assert set(result["overlap_genes"]) == set(ad_in_list)
        # 10 hits vs ~2 expected → highly enriched, p well below 1e-3.
        assert result["p_hypergeometric"] < 1e-3, (
            f"Expected p<1e-3, got {result['p_hypergeometric']}"
        )

    def test_random_list_null_case(self):
        # A 50-gene list of fake genes; 0 real AD overlap expected.
        fake_list = [f"FAKE_{i}" for i in range(50)]
        universe = set(fake_list) | set(AD_GWAS_GENES)
        result = compute_ad_gwas_overlap(
            gene_list=fake_list, universe=universe, gwas_genes=AD_GWAS_GENES
        )
        assert result["n_overlap"] == 0
        assert result["p_hypergeometric"] == pytest.approx(1.0, rel=1e-10)

    def test_result_schema(self):
        fake_list = ["APOE", "TREM2"]
        universe = set(fake_list) | set(AD_GWAS_GENES)
        result = compute_ad_gwas_overlap(
            gene_list=fake_list, universe=universe, gwas_genes=AD_GWAS_GENES
        )
        for key in [
            "n_overlap",
            "overlap_genes",
            "p_hypergeometric",
            "sample_size",
            "n_gwas_in_universe",
            "universe_size",
        ]:
            assert key in result, f"Missing key: {key}"

    def test_respects_universe_restriction(self):
        # If an AD GWAS gene is NOT in the universe, it must not count
        # toward n_gwas_in_universe or overlap.
        ad_genes_in = {"APOE"}  # only APOE is in universe
        ad_genes_out = {"TREM2", "BIN1"}  # not in universe
        gene_list = ["APOE", "FAKE_1", "FAKE_2"]
        universe = set(gene_list) | ad_genes_in
        # AD_GWAS_GENES parameter includes both in/out genes, but the code
        # should restrict to universe before testing.
        result = compute_ad_gwas_overlap(
            gene_list=gene_list,
            universe=universe,
            gwas_genes=ad_genes_in | ad_genes_out,
        )
        assert result["n_overlap"] == 1  # only APOE
        assert result["n_gwas_in_universe"] == 1  # only APOE in universe
        assert result["overlap_genes"] == ["APOE"]


# ---------------------------------------------------------------------------
# End-to-end smoke: load a real summary JSON, run through everything except
# Enrichr. Uses the actual Captum composite_attribution_summary.json.
# ---------------------------------------------------------------------------


SUMMARY_JSON_PATH = (
    _WORKTREE_ROOT
    / "outputs"
    / "redesign"
    / "interpretability"
    / "captum_ig"
    / "composite_attribution_summary.json"
)
GENE_NAMES_PATH = _WORKTREE_ROOT / "data" / "precomputed" / "gene_names.npy"


@pytest.mark.skipif(
    not SUMMARY_JSON_PATH.exists() or not GENE_NAMES_PATH.exists(),
    reason="requires composite_attribution_summary.json and gene_names.npy",
)
class TestRealDataSmoke:
    def test_build_global_gene_list_from_real_summary(self):
        summary = json.loads(SUMMARY_JSON_PATH.read_text())
        gene_names = np.load(GENE_NAMES_PATH, allow_pickle=True).tolist()

        top_genes = build_gene_list_from_summary(
            summary, scope="global", top_k=50
        )
        assert len(top_genes) == 50
        assert all(g in gene_names for g in top_genes), (
            "Summary contains genes not in HVG universe"
        )

    def test_ad_gwas_overlap_on_real_top200(self):
        summary = json.loads(SUMMARY_JSON_PATH.read_text())
        gene_names = np.load(GENE_NAMES_PATH, allow_pickle=True).tolist()
        universe = set(gene_names)

        # Real summary has only 50 entries; if < 200 provided, test still runs
        # on whatever is available.
        top_genes = build_gene_list_from_summary(
            summary, scope="global", top_k=50
        )
        result = compute_ad_gwas_overlap(
            gene_list=top_genes, universe=universe, gwas_genes=AD_GWAS_GENES
        )
        # Sanity: all fields present; values in valid ranges.
        assert 0 <= result["n_overlap"] <= len(top_genes)
        assert 0.0 <= result["p_hypergeometric"] <= 1.0
