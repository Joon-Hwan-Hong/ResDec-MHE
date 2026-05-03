"""Tests for make_readme_fig4_upset.py.

Covers:
1. ``load_top5_cts_per_method`` reads consensus_heatmap_data.json correctly:
   exactly 11 methods, top-5 sets per method, Splatter in 11/11, Fibroblast
   in 10/11.
2. ``load_captum_top_pairs`` slices to the requested top-K and returns
   ``(CT, gene)`` tuples.
3. ``load_wasserstein_top_pairs`` pools per-CT top-10 lists and returns
   the global top-K sorted by distance descending.
4. ``load_de_top_pairs`` pools per-CT (gene, p_value) records, drops
   non-finite p-values, and returns the global top-K by p-value ascending.
5. ``multiway_counts`` partitions items by membership-degree (exactly k).
6. ``pairwise_jaccard_median`` returns 0.5 on a hand-computed example.
7. End-to-end smoke test against the canonical artefacts (skipped if any
   primary input is unavailable). Confirms the script writes a non-empty
   PNG and prints the verification headers.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import WORKTREE_ROOT as _WORKTREE_ROOT

# Module under test
from scripts.resdec_mhe.interpretability.make_readme_fig4_upset import (
    EXPECTED_BOTTOM_PANEL_METHODS,
    EXPECTED_TOP_PANEL_METHODS,
    load_captum_top_pairs,
    load_de_top_pairs,
    load_top5_cts_per_method,
    load_wasserstein_top_pairs,
    multiway_counts,
    pairwise_jaccard_median,
)

CONSENSUS_JSON = (
    _WORKTREE_ROOT
    / "outputs/canonical/interpretability/figures/consensus_heatmap"
    / "consensus_heatmap_data.json"
)
CAPTUM_IG_JSON = (
    _WORKTREE_ROOT
    / "outputs/canonical/interpretability/captum_ig"
    / "composite_attribution_summary.json"
)
GRADIENTSHAP_JSON = (
    _WORKTREE_ROOT
    / "outputs/canonical/interpretability/captum_robustness"
    / "gradientshap_attribution_summary.json"
)
SMOOTHGRAD_JSON = (
    _WORKTREE_ROOT
    / "outputs/canonical/interpretability/captum_robustness"
    / "smoothgrad_attribution_summary.json"
)
WASSERSTEIN_JSON = (
    _WORKTREE_ROOT
    / "outputs/canonical/interpretability/distributional_resilience"
    / "wasserstein_per_celltype_pseudobulk.json"
)
DE_WILCOXON_DIR = (
    _WORKTREE_ROOT / "outputs/canonical/interpretability/de_resilient_vs_vulnerable"
)
DE_DESEQ2_DIR = (
    _WORKTREE_ROOT
    / "outputs/canonical/interpretability/de_resilient_vs_vulnerable_deseq2"
)
GENE_JACCARD_JSON = (
    _WORKTREE_ROOT
    / "outputs/canonical/interpretability/cross_method_gene_jaccard.json"
)


# ---------------------------------------------------------------------------
# Pure helpers (do not require canonical artefacts)
# ---------------------------------------------------------------------------


def test_multiway_counts_partition() -> None:
    """Each item is counted in exactly one bucket."""
    sets = {
        "A": {"x", "y", "z"},
        "B": {"x", "y"},
        "C": {"x"},
    }
    counts = multiway_counts(sets)
    # Items by degree: x in 3 sets, y in 2 sets, z in 1 set
    assert counts == {1: 1, 2: 1, 3: 1}
    assert sum(counts.values()) == 3  # 3 distinct items


def test_pairwise_jaccard_median_handcomputed() -> None:
    """Median of [J(A,B), J(A,C), J(B,C)] on a tiny hand-computable example."""
    sets = {
        "A": {"x", "y"},
        "B": {"y", "z"},
        "C": {"x", "y", "z"},
    }
    # J(A,B) = |{y}|/|{x,y,z}| = 1/3 ~= 0.3333
    # J(A,C) = |{x,y}|/|{x,y,z}| = 2/3 ~= 0.6667
    # J(B,C) = |{y,z}|/|{x,y,z}| = 2/3 ~= 0.6667
    # median = 0.6667
    assert math.isclose(
        pairwise_jaccard_median(sets), 2.0 / 3.0, rel_tol=1e-9,
    )


def test_pairwise_jaccard_median_disjoint() -> None:
    """Disjoint sets -> all pairwise Jaccards are 0 -> median is 0."""
    sets = {"A": {"x"}, "B": {"y"}, "C": {"z"}}
    assert pairwise_jaccard_median(sets) == 0.0


def test_expected_method_lists_match_design() -> None:
    """Hard-code the 11- and 6-method enumerations from the design."""
    assert len(EXPECTED_TOP_PANEL_METHODS) == 11
    assert "IG" in EXPECTED_TOP_PANEL_METHODS
    assert "AttnLRP" in EXPECTED_TOP_PANEL_METHODS
    assert "LOCO" in EXPECTED_TOP_PANEL_METHODS
    assert len(EXPECTED_BOTTOM_PANEL_METHODS) == 6
    assert "Captum IG" in EXPECTED_BOTTOM_PANEL_METHODS
    assert "DE DESeq2" in EXPECTED_BOTTOM_PANEL_METHODS
    assert "Wasserstein" in EXPECTED_BOTTOM_PANEL_METHODS
    # Methods explicitly OUT of the bottom panel (the design verified V3 list)
    for ct_only in ("AttnLRP", "GMAR", "GAF AF", "GAF AGF", "GAF GF",
                    "LOCO", "CMI"):
        assert ct_only not in EXPECTED_BOTTOM_PANEL_METHODS


def test_load_wasserstein_top_pairs_synthetic(tmp_path: Path) -> None:
    """``load_wasserstein_top_pairs`` returns global top-K by distance desc."""
    payload = {
        "n_resilient": 0,
        "n_vulnerable": 0,
        "per_cell_type": [
            {
                "cell_type": "CT_A",
                "wasserstein_per_gene_top10": [
                    ["GENE1", 0.9],
                    ["GENE2", 0.5],
                ],
            },
            {
                "cell_type": "CT_B",
                "wasserstein_per_gene_top10": [
                    ["GENE3", 0.8],
                    ["GENE4", 0.3],
                ],
            },
        ],
    }
    json_path = tmp_path / "ws.json"
    json_path.write_text(json.dumps(payload))
    pairs = load_wasserstein_top_pairs(json_path, top_k=2)
    # Top-2 by distance descending: (CT_A, GENE1, 0.9) and (CT_B, GENE3, 0.8)
    assert pairs == {("CT_A", "GENE1"), ("CT_B", "GENE3")}


def test_load_captum_top_pairs_synthetic(tmp_path: Path) -> None:
    """``load_captum_top_pairs`` reads the built-in pre-sorted top-K block."""
    payload = {
        "top_cell_type_gene_pairs": [
            {"cell_type": "CT_A", "gene": "G1", "mean_abs_attribution": 0.5},
            {"cell_type": "CT_A", "gene": "G2", "mean_abs_attribution": 0.4},
            {"cell_type": "CT_B", "gene": "G3", "mean_abs_attribution": 0.3},
        ],
    }
    json_path = tmp_path / "ig.json"
    json_path.write_text(json.dumps(payload))
    pairs = load_captum_top_pairs(json_path, top_k=2)
    # Top-2 (already sorted by source schema)
    assert pairs == {("CT_A", "G1"), ("CT_A", "G2")}


def test_load_de_top_pairs_synthetic(tmp_path: Path) -> None:
    """``load_de_top_pairs`` pools per-CT CSVs, sorts ascending by p_value."""
    de_dir = tmp_path / "de"
    de_dir.mkdir()
    summary_csv = de_dir / "per_ct_summary.csv"
    summary_csv.write_text(
        "cell_type_index,cell_type,n_genes_tested,n_sig_padj005,min_padj,min_pvalue\n"
        "0,CT_A,2,0,1.0,0.001\n"
        "1,CT_B,2,0,1.0,0.005\n"
    )
    (de_dir / "CT_00_de.csv").write_text(
        "gene,log2_fold_change,lfc_ci_lo,lfc_ci_hi,p_value,padj_fdr,"
        "rank_biserial,n_resilient,n_vulnerable,method\n"
        "G_A1,0.1,,,0.001,1.0,,1,1,wilcoxon\n"
        "G_A2,0.1,,,0.5,1.0,,1,1,wilcoxon\n"
    )
    (de_dir / "CT_01_de.csv").write_text(
        "gene,log2_fold_change,lfc_ci_lo,lfc_ci_hi,p_value,padj_fdr,"
        "rank_biserial,n_resilient,n_vulnerable,method\n"
        "G_B1,0.1,,,0.005,1.0,,1,1,wilcoxon\n"
        "G_B2,0.1,,,0.5,1.0,,1,1,wilcoxon\n"
        "G_B3,,,,,,,1,1,wilcoxon\n"  # non-finite p_value -> dropped
    )
    pairs = load_de_top_pairs(de_dir, top_k=2)
    # Top-2 by p-value ascending: (CT_A, G_A1, p=0.001), (CT_B, G_B1, p=0.005)
    assert pairs == {("CT_A", "G_A1"), ("CT_B", "G_B1")}


# ---------------------------------------------------------------------------
# Tests against the canonical artefacts (skipped if missing)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not CONSENSUS_JSON.exists(),
    reason=f"Consensus heatmap JSON missing: {CONSENSUS_JSON}",
)
def test_load_top5_cts_per_method_canonical() -> None:
    """Exactly 11 methods; Splatter in 11/11 and Fibroblast in 10/11."""
    sets, methods = load_top5_cts_per_method(CONSENSUS_JSON)
    assert len(methods) == 11
    assert set(methods) == set(EXPECTED_TOP_PANEL_METHODS)
    universe = set().union(*sets.values())
    splatter_count = sum(1 for s in sets.values() if "Splatter" in s)
    fibro_count = sum(1 for s in sets.values() if "Fibroblast" in s)
    assert splatter_count == 11, (
        f"Splatter must be in 11/11 method top-5; got {splatter_count}"
    )
    assert fibro_count == 10, (
        f"Fibroblast must be in 10/11 method top-5; got {fibro_count}"
    )
    # Every method must yield ~5 CTs (one method has 4 because of a tied rank).
    for m in EXPECTED_TOP_PANEL_METHODS:
        n = len(sets[m])
        assert 4 <= n <= 5, (
            f"Method {m!r} should have 4-5 top-K CTs (got {n})"
        )
    # Universe size = 10 (the row_cts list)
    assert len(universe) == 10


@pytest.mark.skipif(
    not (
        CAPTUM_IG_JSON.exists() and GRADIENTSHAP_JSON.exists()
        and SMOOTHGRAD_JSON.exists() and WASSERSTEIN_JSON.exists()
        and DE_WILCOXON_DIR.exists() and DE_DESEQ2_DIR.exists()
    ),
    reason="One or more bottom-panel primary inputs missing",
)
def test_load_top_pairs_per_method_canonical_sizes() -> None:
    """All 6 gene-rankable methods yield exactly 50 (CT, gene) pairs."""
    from scripts.resdec_mhe.interpretability.make_readme_fig4_upset import (
        load_top_pairs_per_method,
    )
    sets = load_top_pairs_per_method(
        captum_ig_path=CAPTUM_IG_JSON,
        gradientshap_path=GRADIENTSHAP_JSON,
        smoothgrad_path=SMOOTHGRAD_JSON,
        wasserstein_path=WASSERSTEIN_JSON,
        de_wilcoxon_dir=DE_WILCOXON_DIR,
        de_deseq2_dir=DE_DESEQ2_DIR,
        top_k=50,
    )
    for m in EXPECTED_BOTTOM_PANEL_METHODS:
        assert m in sets
        assert len(sets[m]) == 50, (
            f"Method {m!r} must yield exactly 50 (CT, gene) pairs; "
            f"got {len(sets[m])}"
        )
        # Every entry is a (str, str) tuple
        for entry in sets[m]:
            assert isinstance(entry, tuple)
            assert len(entry) == 2
            assert isinstance(entry[0], str)
            assert isinstance(entry[1], str)


@pytest.mark.skipif(
    not GENE_JACCARD_JSON.exists(),
    reason=f"Gene-set Jaccard JSON missing: {GENE_JACCARD_JSON}",
)
def test_gene_jaccard_reference_consistency() -> None:
    """The reference gene-set median Jaccard is the documented 0.16."""
    payload = json.loads(GENE_JACCARD_JSON.read_text())
    median = float(payload["pairwise_summary"]["median_jaccard"])
    # Design says ~0.16; allow tight tolerance.
    assert math.isclose(median, 0.16, abs_tol=1e-6)


@pytest.mark.skipif(
    not CONSENSUS_JSON.exists(),
    reason="Canonical consensus JSON missing",
)
def test_main_smoke(tmp_path: Path) -> None:
    """End-to-end: invoke the script's main() and confirm a non-empty PNG."""
    out_stem = tmp_path / "fig4_upset_test"
    cmd = [
        sys.executable,
        str(
            _WORKTREE_ROOT
            / "scripts/resdec_mhe/interpretability/make_readme_fig4_upset.py"
        ),
        "--out-stem", str(out_stem),
    ]
    env_pythonpath = str(_WORKTREE_ROOT)
    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = env_pythonpath
    proc = subprocess.run(
        cmd, capture_output=True, text=True, env=env,
    )
    assert proc.returncode == 0, (
        f"script failed: stdout={proc.stdout}\nstderr={proc.stderr}"
    )
    out_png = out_stem.with_suffix(".png")
    assert out_png.exists(), f"PNG not written: {out_png}"
    assert out_png.stat().st_size > 50_000, (
        f"PNG suspiciously small ({out_png.stat().st_size} bytes): {out_png}"
    )
    assert "README Figure 4" in proc.stdout
    assert "Top panel - top-5 CTs per method" in proc.stdout
    assert "Bottom panel - top-50 (CT, gene) pairs per method" in proc.stdout
