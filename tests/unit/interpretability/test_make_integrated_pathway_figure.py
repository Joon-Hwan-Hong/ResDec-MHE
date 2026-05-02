"""Smoke + unit tests for make_integrated_pathway_figure.py.

Tests exercise:
  - end-to-end run against the actual canonical JSON / CSV inputs and
    verify PNG + PDF + caption.md + summary.json are produced
  - DE top-K loader filters by cell_type and sorts by p_value
  - Reactome CSV loader returns top-10 sorted by adjusted p
  - Splatter NT-pathway count parsing (descriptive, not from JSON)
  - Intersection table builder produces all 2^N - 1 non-empty cells when
    every set is distinct, and zero size for cells with no members
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import WORKTREE_ROOT as _WORKTREE_ROOT

def test_make_integrated_pathway_figure_runs(tmp_path: Path) -> None:
    """End-to-end smoke: script runs against canonical inputs and writes PNG/PDF."""
    consensus = (
        _WORKTREE_ROOT
        / "outputs/canonical/interpretability/figures/consensus_heatmap/"
          "consensus_heatmap_data.json"
    )
    reactome = (
        _WORKTREE_ROOT
        / "outputs/canonical/interpretability/gsea/"
          "gsea_Reactome_2022_top_50_Splatter.csv"
    )
    captum = (
        _WORKTREE_ROOT
        / "outputs/canonical/interpretability/captum_ig/"
          "composite_attribution_summary.json"
    )
    wasserstein = (
        _WORKTREE_ROOT
        / "outputs/canonical/interpretability/distributional_resilience/"
          "wasserstein_per_celltype_pseudobulk.json"
    )
    coverage = (
        _WORKTREE_ROOT
        / "outputs/canonical/interpretability/ct_coverage_full_cohort.json"
    )
    if not all(p.is_file() for p in (
        consensus, reactome, captum, wasserstein, coverage
    )):
        pytest.skip("required input artefacts missing")

    out_dir = tmp_path / "manuscript_composite"
    script = (
        _WORKTREE_ROOT
        / "scripts/resdec_mhe/interpretability/"
          "make_integrated_pathway_figure.py"
    )
    cmd = [
        sys.executable, str(script),
        "--out-dir", str(out_dir),
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(_WORKTREE_ROOT)
    )
    assert result.returncode == 0, (
        f"Script failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    png = out_dir / "fig_integrated_pathway_celltype.png"
    pdf = out_dir / "fig_integrated_pathway_celltype.pdf"
    cap = out_dir / "caption.md"
    summary = out_dir / "summary.json"
    assert png.is_file() and png.stat().st_size > 5000, png
    assert pdf.is_file() and pdf.stat().st_size > 5000, pdf
    assert cap.is_file() and cap.stat().st_size > 200, cap
    assert summary.is_file() and summary.stat().st_size > 100, summary

    payload = json.loads(summary.read_text())
    assert "panel_a" in payload and "panel_b" in payload
    assert "panel_c" in payload and "panel_d" in payload
    assert payload["panel_a"]["splatter_top5_count"] >= 5, (
        payload["panel_a"]
    )
    assert payload["panel_b"]["n_pathways"] == 10
    assert payload["panel_c"]["captum_top_k"] == 15
    assert "set_sizes" in payload["panel_d"]
    assert "all_method_intersection_size" in payload["panel_d"]

def test_load_de_top_genes_set_filters_and_sorts() -> None:
    """DE loader returns only genes for the requested CT, sorted by p_value."""
    from scripts.resdec_mhe.interpretability import (  # noqa: E402
        make_integrated_pathway_figure as mod,
    )

    de_dir = (
        _WORKTREE_ROOT
        / "outputs/canonical/interpretability/de_resilient_vs_vulnerable"
    )
    if not (de_dir / "per_ct_summary.csv").is_file():
        pytest.skip("DE-Wilcoxon per-CT directory missing")

    s5 = mod.load_de_top_genes_set(de_dir, "Splatter", 5)
    assert isinstance(s5, set)
    assert 1 <= len(s5) <= 5
    # First entry by p_value when filtered to Splatter is ID2 in the
    # canonical Wilcoxon table; guard against re-sort regression.
    assert "ID2" in s5

    # Filter by a cell-type that does not exist returns empty set.
    s_none = mod.load_de_top_genes_set(de_dir, "ZZZZZZZ", 50)
    assert s_none == set()

def test_load_de_top_genes_set_returns_top_50_when_full_table_available() -> None:
    """The per-CT DE table holds 4785 genes per CT; top-50 is satisfiable."""
    from scripts.resdec_mhe.interpretability import (  # noqa: E402
        make_integrated_pathway_figure as mod,
    )

    de_dir = (
        _WORKTREE_ROOT
        / "outputs/canonical/interpretability/de_resilient_vs_vulnerable"
    )
    if not (de_dir / "per_ct_summary.csv").is_file():
        pytest.skip("DE-Wilcoxon per-CT directory missing")
    s50 = mod.load_de_top_genes_set(de_dir, "Splatter", 50)
    assert len(s50) == 50, len(s50)

def test_load_reactome_top10_returns_sorted() -> None:
    """Reactome loader returns 10 rows sorted by adjusted p ascending."""
    from scripts.resdec_mhe.interpretability import (  # noqa: E402
        make_integrated_pathway_figure as mod,
    )

    csv = (
        _WORKTREE_ROOT
        / "outputs/canonical/interpretability/gsea/"
          "gsea_Reactome_2022_top_50_Splatter.csv"
    )
    if not csv.is_file():
        pytest.skip("Reactome CSV missing")

    rows = mod.load_reactome_top10_for_splatter(csv)
    assert len(rows) == 10
    padjs = [r["adjusted_p_value"] for r in rows]
    assert padjs == sorted(padjs), "rows must be ascending by adjusted p"
    # Splatter top-1 lead per docs/MEMORY: NT release / ACh / VAMP2 etc.
    leading = rows[0]["genes"]
    assert any(g in leading for g in ("VAMP2", "SNAP25", "CPLX1")), leading

def test_load_per_ct_top1_returns_splatter_with_six_nt() -> None:
    """Splatter row in the per-CT top-1 table has 6 NT-release pathways.

    This pins the F5 finding (6/10 NT-release Splatter-specific) so the
    Panel B annotation cannot silently regress.
    """
    from scripts.resdec_mhe.interpretability import (  # noqa: E402
        make_integrated_pathway_figure as mod,
    )

    csv = (
        _WORKTREE_ROOT
        / "outputs/canonical/interpretability/gsea/"
          "per_ct_reactome_top1_comparative.csv"
    )
    if not csv.is_file():
        pytest.skip("per-CT top1 comparative CSV missing")

    rows = mod.load_per_ct_top1(csv)
    splatter_row = next(
        (r for r in rows if r["cell_type"].lower().startswith("splatter")),
        None,
    )
    assert splatter_row is not None, rows
    assert splatter_row["n_nt_in_top10"] == 6, splatter_row

def test_intersection_table_enumerates_all_non_empty_cells() -> None:
    """Builder returns all 2^N - 1 cells when every set is distinct."""
    from scripts.resdec_mhe.interpretability import (  # noqa: E402
        make_integrated_pathway_figure as mod,
    )

    sets = {
        "A": {"x", "y"},
        "B": {"y", "z"},
        "C": {"z", "w"},
    }
    cells = mod._build_intersection_table(sets)
    # Build a label->size lookup
    by_members = {tuple(sorted(m)): s for m, s in cells}
    # Per-set sizes (singleton cells are exclusive)
    assert by_members[("A",)] == 1   # only "x"
    assert by_members[("B",)] == 0 if ("B",) in by_members else True
    # Pairwise: A∩B = {y} (excluding C), B∩C = {z} (excluding A)
    assert by_members[("A", "B")] == 1
    assert by_members[("B", "C")] == 1
    # Triple intersection = empty
    assert ("A", "B", "C") not in by_members
    # Sorted descending by size
    sizes = [s for _, s in cells]
    assert sizes == sorted(sizes, reverse=True)

def test_shorten_reactome_term_strips_id_and_falls_back() -> None:
    """Reactome short-name helper strips R-HSA-* and falls back to base."""
    from scripts.resdec_mhe.interpretability import (  # noqa: E402
        make_integrated_pathway_figure as mod,
    )

    short = mod._shorten_reactome_term(
        "Acetylcholine Neurotransmitter Release Cycle R-HSA-264642"
    )
    assert short == "Acetylcholine NT release"
    # Unknown term: strips ID, returns base
    fallback = mod._shorten_reactome_term(
        "An Unknown Pathway Name R-HSA-9999999"
    )
    assert fallback == "An Unknown Pathway Name"
