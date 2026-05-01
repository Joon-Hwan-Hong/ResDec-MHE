"""Unit + smoke tests for run_11method_gene_jaccard.py.

Coverage:
  - jaccard() returns 0 for empty union and exact rational for known sets
  - pairwise_jaccard_matrix produces 1.0 on diagonal and symmetric off-diagonal
  - multiway_support_counts sums and bucket counts agree
  - consensus_genes_at_threshold filters and sorts by count desc / gene asc
  - per_ct_to_union dedups across CTs
  - Captum loader filters by top-K and returns sets
  - Wasserstein loader caps at 10 (source schema)
  - DE loader sorts by p_value asc and returns the requested top-K size
  - End-to-end CLI smoke against canonical artefacts: writes JSON / MD /
    PNG / PDF; JSON has the documented schema (labels, matrix shape,
    multiway buckets) and the Jaccard min / max / median agree with the
    matrix.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]


def _import_mod():
    sys.path.insert(0, str(_WORKTREE_ROOT))
    from scripts.resdec_mhe.interpretability import (  # noqa: E402
        run_11method_gene_jaccard as mod,
    )
    return mod


def test_jaccard_empty_and_known_values() -> None:
    mod = _import_mod()
    assert mod.jaccard(set(), set()) == 0.0
    assert mod.jaccard({"x"}, set()) == 0.0
    assert mod.jaccard({"x"}, {"x"}) == 1.0
    # |A∩B|=1, |A∪B|=3 -> 1/3
    assert mod.jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1.0 / 3.0)
    # Disjoint sets
    assert mod.jaccard({"a"}, {"b"}) == 0.0


def test_pairwise_jaccard_matrix_symmetry_and_diagonal() -> None:
    mod = _import_mod()
    sets = {
        "M1": {"a", "b", "c"},
        "M2": {"b", "c", "d"},
        "M3": {"a", "x"},
    }
    matrix, labels = mod.pairwise_jaccard_matrix(sets)
    assert labels == sorted(sets.keys())
    n = len(labels)
    # Diagonal == 1.0
    assert all(matrix[i, i] == 1.0 for i in range(n))
    # Symmetric
    assert np.allclose(matrix, matrix.T)
    # Spot-check M1 vs M2: intersect={b,c}, union={a,b,c,d} -> 2/4 = 0.5
    i = labels.index("M1")
    j = labels.index("M2")
    assert matrix[i, j] == pytest.approx(0.5)
    # M1 vs M3: intersect={a}, union={a,b,c,x} -> 1/4 = 0.25
    k = labels.index("M3")
    assert matrix[i, k] == pytest.approx(0.25)


def test_multiway_support_counts_and_at_least_buckets() -> None:
    mod = _import_mod()
    sets = {
        "M1": {"x", "y"},
        "M2": {"y", "z"},
        "M3": {"z", "w"},
    }
    per_gene, at_least = mod.multiway_support_counts(sets)
    assert per_gene["x"] == 1
    assert per_gene["y"] == 2
    assert per_gene["z"] == 2
    assert per_gene["w"] == 1
    # >=1: all 4 distinct genes present in at least one set
    assert at_least[1] == 4
    # >=2: y, z
    assert at_least[2] == 2
    # >=3: none
    assert at_least[3] == 0


def test_consensus_genes_filter_and_sort() -> None:
    mod = _import_mod()
    sets = {
        "M1": {"a", "b", "c"},
        "M2": {"a", "b"},
        "M3": {"a"},
    }
    # Threshold=2: a in 3, b in 2, c in 1 -> a, b
    recs = mod.consensus_genes_at_threshold(sets, threshold=2)
    assert [r["gene"] for r in recs] == ["a", "b"]
    assert recs[0]["count"] == 3 and recs[0]["methods"] == ["M1", "M2", "M3"]
    assert recs[1]["count"] == 2 and recs[1]["methods"] == ["M1", "M2"]
    # Threshold=1: all genes; sorted by count desc then gene asc
    recs1 = mod.consensus_genes_at_threshold(sets, threshold=1)
    assert [r["gene"] for r in recs1] == ["a", "b", "c"]
    # Threshold above max yields empty
    assert mod.consensus_genes_at_threshold(sets, threshold=99) == []


def test_per_ct_to_union_dedups_across_cts() -> None:
    mod = _import_mod()
    per_ct = {"CT_A": {"x", "y"}, "CT_B": {"y", "z"}, "CT_C": {"z"}}
    assert mod.per_ct_to_union(per_ct) == {"x", "y", "z"}


def test_captum_loader_filters_by_top_k() -> None:
    mod = _import_mod()
    p = (
        _WORKTREE_ROOT
        / "outputs/canonical/interpretability/captum_ig/composite_attribution_summary.json"
    )
    if not p.is_file():
        pytest.skip("Captum IG summary missing")
    blocks = mod._captum_per_ct_top_k(p, top_k=5)
    # Every CT block should contribute <= 5 genes
    assert all(len(s) <= 5 for s in blocks.values())
    # 31 CTs in the canonical dataset
    assert len(blocks) == 31
    # The Splatter top-1 from EXP-034 / MEMORY is Splatter × SCN3B — pinned
    assert "SCN3B" in blocks["Splatter"]


def test_wasserstein_loader_capped_at_10() -> None:
    mod = _import_mod()
    p = (
        _WORKTREE_ROOT
        / "outputs/canonical/interpretability/distributional_resilience/"
          "wasserstein_per_celltype_pseudobulk.json"
    )
    if not p.is_file():
        pytest.skip("Wasserstein JSON missing")
    blocks = mod._wasserstein_per_ct_top_k(p, top_k=99)
    # 31 CTs
    assert len(blocks) == 31
    # Source caps at 10 → loader truncates to 10 even when caller asks for 99
    assert all(len(s) <= 10 for s in blocks.values())


def test_de_wilcoxon_loader_returns_top_k() -> None:
    mod = _import_mod()
    de_dir = (
        _WORKTREE_ROOT
        / "outputs/canonical/interpretability/de_resilient_vs_vulnerable"
    )
    if not (de_dir / "per_ct_summary.csv").is_file():
        pytest.skip("DE Wilcoxon dir missing")
    blocks = mod._de_per_ct_top_k(de_dir, top_k=5)
    # 31 CTs in summary
    assert len(blocks) == 31
    # Each non-empty CT has <= 5 genes; some CTs may legitimately yield <5
    # if their per-CT CSV has fewer rows or has non-finite p-values.
    for ct, s in blocks.items():
        assert len(s) <= 5, ct


def test_de_wilcoxon_loader_pins_splatter_top_1() -> None:
    """DE Wilcoxon top-1 for Splatter is ID2 (per EXP-034 test pin)."""
    mod = _import_mod()
    de_dir = (
        _WORKTREE_ROOT
        / "outputs/canonical/interpretability/de_resilient_vs_vulnerable"
    )
    if not (de_dir / "per_ct_summary.csv").is_file():
        pytest.skip("DE Wilcoxon dir missing")
    s = mod._de_per_ct_top_k(de_dir, top_k=1)
    assert s["Splatter"] == {"ID2"}


def test_end_to_end_runs(tmp_path: Path) -> None:
    """Smoke test: script runs against canonical inputs and writes outputs."""
    captum = (
        _WORKTREE_ROOT
        / "outputs/canonical/interpretability/captum_ig/composite_attribution_summary.json"
    )
    gradshap = (
        _WORKTREE_ROOT
        / "outputs/canonical/interpretability/captum_robustness/"
          "gradientshap_attribution_summary.json"
    )
    smoothg = (
        _WORKTREE_ROOT
        / "outputs/canonical/interpretability/captum_robustness/"
          "smoothgrad_attribution_summary.json"
    )
    wass = (
        _WORKTREE_ROOT
        / "outputs/canonical/interpretability/distributional_resilience/"
          "wasserstein_per_celltype_pseudobulk.json"
    )
    de_w = (
        _WORKTREE_ROOT
        / "outputs/canonical/interpretability/de_resilient_vs_vulnerable"
    )
    de_d = (
        _WORKTREE_ROOT
        / "outputs/canonical/interpretability/de_resilient_vs_vulnerable_deseq2"
    )
    needed = (captum, gradshap, smoothg, wass, de_w, de_d)
    if not all(p.exists() for p in needed):
        pytest.skip("required canonical inputs missing")

    out_json = tmp_path / "cross_method_gene_jaccard.json"
    out_md = tmp_path / "cross_method_gene_jaccard.md"
    out_fig_dir = tmp_path / "fig_cross_method_jaccard"
    script = (
        _WORKTREE_ROOT
        / "scripts/resdec_mhe/interpretability/run_11method_gene_jaccard.py"
    )
    cmd = [
        sys.executable, str(script),
        "--out-json", str(out_json),
        "--out-md", str(out_md),
        "--out-fig-dir", str(out_fig_dir),
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(_WORKTREE_ROOT),
    )
    assert result.returncode == 0, (
        f"Script failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert out_json.is_file() and out_json.stat().st_size > 200
    assert out_md.is_file() and out_md.stat().st_size > 200
    png = out_fig_dir / "fig_cross_method_gene_jaccard.png"
    pdf = out_fig_dir / "fig_cross_method_gene_jaccard.pdf"
    assert png.is_file() and png.stat().st_size > 5000
    assert pdf.is_file() and pdf.stat().st_size > 5000

    payload = json.loads(out_json.read_text())
    assert "labels_label_sorted" in payload
    assert "jaccard_matrix" in payload
    assert "pairwise_summary" in payload
    assert "multiway_at_least_k" in payload
    n = len(payload["labels_label_sorted"])
    assert n == 6, payload["labels_label_sorted"]
    matrix = payload["jaccard_matrix"]
    # Square shape
    assert all(len(row) == n for row in matrix)
    # Diagonal == 1
    assert all(matrix[i][i] == pytest.approx(1.0) for i in range(n))
    # Symmetric
    for i in range(n):
        for j in range(n):
            assert matrix[i][j] == pytest.approx(matrix[j][i])
    # Pairwise summary aligns with the off-diagonal upper triangle
    upper = [matrix[i][j] for i in range(n) for j in range(i + 1, n)]
    assert payload["pairwise_summary"]["n_pairs"] == len(upper) == n * (n - 1) // 2
    assert payload["pairwise_summary"]["min_jaccard"] == pytest.approx(min(upper))
    assert payload["pairwise_summary"]["max_jaccard"] == pytest.approx(max(upper))
    # Median agrees (numpy vs python median identical here)
    py_med = float(np.median(np.asarray(upper)))
    assert payload["pairwise_summary"]["median_jaccard"] == pytest.approx(py_med)
    # Buckets cover k=1..6 and are non-increasing
    buckets = payload["multiway_at_least_k"]
    assert sorted(int(k) for k in buckets) == list(range(1, 7))
    counts = [buckets[str(k)] for k in range(1, 7)]
    for a, b in zip(counts, counts[1:]):
        assert a >= b, counts
