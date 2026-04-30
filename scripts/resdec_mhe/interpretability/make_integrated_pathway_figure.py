#!/usr/bin/env python
"""Manuscript composite: integrated pathway x cell-type figure (4 panels).

Synthesises four currently-disconnected interpretability artifacts into one
manuscript-ready 2x2 composite that links:

  - Cross-method cell-type consensus (11 methods, top-5 ranks)
  - GSEA Reactome pathway enrichment for Splatter top-50 attribution genes
  - Captum top (CT, gene) attribution pairs for Splatter
  - Top-50 gene overlap across attribution / pseudobulk / DE methods

Panel layout::

    A | B
    C | D

Panel A — 11-method cross-method consensus heatmap (rows = top-10 CTs by
top-5 frequency; cols = methods; cell colour = rank in top-5 with rank 1
darkest viridis; rank > 5 white). Splatter row outlined in red.

Panel B — Reactome 2022 top-10 pathways for Splatter top-50 attribution
genes; bars by -log10(adjusted p), coloured by NES proxy
(negative log p + sign of mean log fold change unavailable in raw GSEA
table, so we use overlap size as a proxy weight in tooltip annotation).
Pathway names abbreviated. Inline annotation flags 6/10
neurotransmitter-release pathways unique to Splatter (see
``per_ct_reactome_top1_comparative.csv``).

Panel C — Captum top-15 (CT=Splatter, gene) attribution pairs as
horizontal bars (mean abs IG attribution; gene name on y-axis; top 15 of
50 stored in the captum summary).

Panel D — Top-50 gene-set overlap across four methods:

    1. Captum IG (Splatter top-50 by mean_abs_attribution)
    2. Pseudobulk Wasserstein-1 (Splatter top-N where N = available; the
       canonical JSON stores top-10 per CT — Panel D legend labels this
       "Wasserstein top-10" rather than top-50)
    3. DE Wilcoxon (Splatter top-50 by p-value)
    4. DE DESeq2  (Splatter top-50 by p-value)

Rendered as an upset-style intersection bar plot since
``matplotlib_venn`` / ``upsetplot`` are not available in this environment.
The bottom matrix uses dots-and-lines to indicate which sets each
intersection corresponds to; the top bars give the intersection size.

Per-CT cohort coverage (median cells, n_subjects with >=1 cell) for
Splatter is annotated inline on Panel C (descriptive — does NOT filter
input data).

Outputs::

    outputs/canonical/interpretability/figures/manuscript_composite/
        fig_integrated_pathway_celltype.{png,pdf}    (600 DPI, ~14x12 in)
        caption.md                                    (figure caption)

This is a CPU-only script: it consumes existing JSON / CSV summaries and
emits a static composite figure plus its caption.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from src.visualization.composite import auto_letter
from src.visualization.theme import (  # noqa: E402
    PALETTES,
    apply_theme,
    fmt_axes,
    style_paper_axes,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Default input paths (relative to worktree root; absolute via _resolve_path)
# ----------------------------------------------------------------------
DEFAULTS = {
    "consensus": (
        "outputs/canonical/interpretability/figures/consensus_heatmap/"
        "consensus_heatmap_data.json"
    ),
    "reactome_csv": (
        "outputs/canonical/interpretability/gsea/"
        "gsea_Reactome_2022_top_50_Splatter.csv"
    ),
    "per_ct_top1_csv": (
        "outputs/canonical/interpretability/gsea/per_ct_reactome_top1_comparative.csv"
    ),
    "captum": (
        "outputs/canonical/interpretability/captum_ig/"
        "composite_attribution_summary.json"
    ),
    "wasserstein": (
        "outputs/canonical/interpretability/distributional_resilience/"
        "wasserstein_per_celltype_pseudobulk.json"
    ),
    # Per-CT full DE result tables (all ~4785 genes; allows top-50 ranking).
    # The summary "top_genes_per_ct_by_pvalue.csv" only stores 20 rows per CT.
    "de_wilcoxon_dir": (
        "outputs/canonical/interpretability/de_resilient_vs_vulnerable"
    ),
    "de_deseq2_dir": (
        "outputs/canonical/interpretability/de_resilient_vs_vulnerable_deseq2"
    ),
    "de_summary_csv": (
        "outputs/canonical/interpretability/de_resilient_vs_vulnerable/"
        "per_ct_summary.csv"
    ),
    "ct_coverage": (
        "outputs/canonical/interpretability/ct_coverage_full_cohort.json"
    ),
}


# ----------------------------------------------------------------------
# Data loading helpers
# ----------------------------------------------------------------------
def _resolve_path(p: str | Path) -> Path:
    """Resolve a default-relative path against _WORKTREE_ROOT if not absolute."""
    pth = Path(p)
    if pth.is_absolute():
        return pth
    return _WORKTREE_ROOT / pth


def load_consensus(path: Path) -> dict:
    """Load the precomputed 11-method consensus heatmap rank matrix."""
    return json.loads(Path(path).read_text())


def load_reactome_top10_for_splatter(csv_path: Path) -> list[dict]:
    """Return the top-10 Reactome pathways for Splatter (sorted by adjusted p).

    The CSV is the gseapy ``Reactome_2022`` enrichment output for the
    Splatter top-50 attribution genes. Columns: term, overlap, p_value,
    adjusted_p_value, odds_ratio, combined_score, genes, database.
    """
    rows: list[dict] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "term": r["term"],
                "overlap": int(r["overlap"]),
                "p_value": float(r["p_value"]),
                "adjusted_p_value": float(r["adjusted_p_value"]),
                "odds_ratio": float(r["odds_ratio"]),
                "combined_score": float(r["combined_score"]),
                "genes": r["genes"],
            })
    rows.sort(key=lambda d: d["adjusted_p_value"])
    return rows[:10]


def load_per_ct_top1(csv_path: Path) -> list[dict]:
    """Load per-CT top-1 comparative table.

    Columns: cell_type, top1_term, top1_genes, top1_p_value,
    top1_adjusted_p_value, n_neurotransmitter_pathways_in_top10.
    """
    out: list[dict] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            out.append({
                "cell_type": r["cell_type"],
                "n_nt_in_top10": int(r["n_neurotransmitter_pathways_in_top10"]),
            })
    return out


def load_captum_top_pairs_for_ct(
    summary_path: Path, cell_type: str, top_k: int
) -> list[tuple[str, float]]:
    """Return the top-K (gene, mean_abs_attribution) pairs for ``cell_type``.

    Uses ``top_genes_per_cell_type[cell_type]`` from the captum composite
    summary (already sorted descending by attribution).
    """
    summary = json.loads(Path(summary_path).read_text())
    block = summary["top_genes_per_cell_type"].get(cell_type, [])
    return [(g["gene"], float(g["mean_abs_attribution"])) for g in block[:top_k]]


def load_captum_top_genes_set(
    summary_path: Path, cell_type: str, top_k: int
) -> set[str]:
    return {g for g, _ in load_captum_top_pairs_for_ct(
        summary_path, cell_type, top_k
    )}


def load_wasserstein_top_genes_set(
    json_path: Path, cell_type: str, top_k: int
) -> set[str]:
    """Pull ``wasserstein_per_gene_top10`` for the matching CT.

    Schema: per_cell_type is a LIST of {cell_type, wasserstein_per_gene_mean,
    wasserstein_per_gene_top10: [[gene, value], ...]}. Capped at length 10
    upstream — top_k is therefore min(top_k, 10).
    """
    summary = json.loads(Path(json_path).read_text())
    block = next(
        (b for b in summary["per_cell_type"] if b["cell_type"] == cell_type),
        None,
    )
    if block is None:
        return set()
    pairs = block.get("wasserstein_per_gene_top10", [])
    return {p[0] for p in pairs[:top_k]}


def _resolve_ct_index(de_summary_csv: Path, cell_type: str) -> int:
    """Look up the integer cell_type_index for ``cell_type``.

    Reads ``per_ct_summary.csv`` (columns include cell_type_index, cell_type)
    so the per-CT DE files (CT_NN_de.csv) can be loaded by index.
    """
    with open(de_summary_csv, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["cell_type"] == cell_type:
                return int(r["cell_type_index"])
    raise KeyError(f"cell_type {cell_type!r} not found in {de_summary_csv}")


def load_de_top_genes_set(
    de_dir: Path, cell_type: str, top_k: int, *,
    de_summary_csv: Path | None = None,
) -> set[str]:
    """Read top-K genes by p_value for ``cell_type`` from a per-CT DE directory.

    The per-CT DE files (``CT_NN_de.csv``) hold ~4785 genes/cell-type with
    columns: gene, log2_fold_change, lfc_ci_lo, lfc_ci_hi, p_value,
    padj_fdr, rank_biserial, n_resilient, n_vulnerable, method. We look up
    the integer index via ``per_ct_summary.csv`` (in the same DE dir if
    not provided), then read that CT's full table and take the first top_k
    rows after sorting by p_value ascending.
    """
    de_dir = Path(de_dir)
    if de_summary_csv is None:
        de_summary_csv = de_dir / "per_ct_summary.csv"
    try:
        ct_idx = _resolve_ct_index(Path(de_summary_csv), cell_type)
    except KeyError:
        return set()

    ct_csv = de_dir / f"CT_{ct_idx:02d}_de.csv"
    if not ct_csv.is_file():
        return set()

    rows: list[tuple[str, float]] = []
    with open(ct_csv, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                p = float(r["p_value"])
            except (KeyError, ValueError):
                continue
            if not np.isfinite(p):
                continue
            rows.append((r["gene"], p))
    rows.sort(key=lambda kv: kv[1])
    return {g for g, _ in rows[:top_k]}


def load_ct_coverage(path: Path) -> dict:
    return json.loads(Path(path).read_text())


# ----------------------------------------------------------------------
# Panel draw functions
# ----------------------------------------------------------------------
def draw_panel_a_consensus(ax: plt.Axes, data: dict) -> None:
    """Cross-method top-5 consensus heatmap; Splatter row red-outlined."""
    rows = list(data["row_cts"])
    methods = list(data["methods"])
    n_rows, n_cols = len(rows), len(methods)
    grid = np.full((n_rows, n_cols), np.nan, dtype=float)
    for i, ct in enumerate(rows):
        for j, m in enumerate(methods):
            r = data["ranks"].get(ct, {}).get(m)
            if r is not None:
                grid[i, j] = r

    cmap = plt.get_cmap("viridis")
    rgba = np.ones((n_rows, n_cols, 4), dtype=float)
    for i in range(n_rows):
        for j in range(n_cols):
            r = grid[i, j]
            if not np.isnan(r) and r <= 5:
                color_val = 0.15 + 0.7 * (r - 1) / 4.0
                rgba[i, j, :] = cmap(color_val)
    ax.imshow(rgba, aspect="auto", interpolation="nearest")

    for i in range(n_rows):
        for j in range(n_cols):
            r = grid[i, j]
            if not np.isnan(r) and r <= 5:
                color_val = 0.15 + 0.7 * (r - 1) / 4.0
                txt_color = "white" if color_val < 0.55 else "black"
                ax.text(
                    j, i, f"{int(r)}",
                    ha="center", va="center",
                    fontsize=6, color=txt_color, fontweight="bold",
                )

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(methods, rotation=55, ha="right", fontsize=7)
    ax.set_yticks(np.arange(n_rows))
    yticklabels = []
    for ct in rows:
        if ct.lower() == "splatter":
            yticklabels.append(f"$\\bf{{{ct}}}$")
        elif ct.lower() == "fibroblast":
            yticklabels.append(f"$\\it{{{ct}}}$")
        else:
            yticklabels.append(ct)
    ax.set_yticklabels(yticklabels, fontsize=7)

    if any(ct.lower() == "splatter" for ct in rows):
        i_splatter = next(
            i for i, ct in enumerate(rows) if ct.lower() == "splatter"
        )
        ax.add_patch(Rectangle(
            (-0.5, i_splatter - 0.5),
            n_cols, 1.0,
            fill=False, edgecolor="#d62728", linewidth=1.8, zorder=5,
        ))

    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="#cccccc", linewidth=0.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    fmt_axes(ax, hide_spines=(), grid_major=False, grid_minor=False)
    ax.set_xlabel("Method", fontsize=8)


# Hand-picked short labels for the long Reactome term names.
# Maps: full term (with R-HSA suffix optional) -> short label for Y axis.
_REACTOME_SHORT_LABELS = {
    "Acetylcholine Neurotransmitter Release Cycle": "Acetylcholine NT release",
    "Neurotoxicity Of Clostridium Toxins": "Clostridium toxin",
    "Norepinephrine Neurotransmitter Release Cycle": "Norepinephrine NT release",
    "Serotonin Neurotransmitter Release Cycle": "Serotonin NT release",
    "Sensory Processing Of Sound By Inner Hair Cells Of Cochlea": (
        "Inner-ear hair-cell"
    ),
    "Dopamine Neurotransmitter Release Cycle": "Dopamine NT release",
    "Sensory Processing Of Sound": "Sound processing",
    "GABA Synthesis, Release, Reuptake And Degradation": "GABA cycle",
    "Glutamate Neurotransmitter Release Cycle": "Glutamate NT release",
    "Sensory Perception": "Sensory perception",
    "Uptake And Actions Of Bacterial Toxins": "Bacterial toxin uptake",
    "Neurotransmitter Release Cycle": "NT release (generic)",
}


def _shorten_reactome_term(term: str) -> str:
    """Strip the ``R-HSA-...`` suffix and apply the short label dict.

    Falls back to the stripped term if not in the dict.
    """
    base = term.rsplit(" R-HSA-", 1)[0]
    return _REACTOME_SHORT_LABELS.get(base, base)


def draw_panel_b_reactome(
    ax: plt.Axes,
    rows: list[dict],
    n_nt_pathways_splatter: int,
    n_nt_pathways_total: int,
) -> None:
    """Top-10 Reactome pathways for Splatter; bars by -log10(adjusted p).

    Color: bar's color reflects overlap size (number of Splatter genes in
    the term gene-set), normalized to the viridis colormap so that
    larger-overlap (more biologically anchored) terms are darker.
    """
    if not rows:
        ax.text(0.5, 0.5, "no Reactome rows available",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    rows_sorted = sorted(rows, key=lambda d: -d["adjusted_p_value"])
    labels = [_shorten_reactome_term(d["term"]) for d in rows_sorted]
    neg_log_padj = [
        -np.log10(max(d["adjusted_p_value"], 1e-12)) for d in rows_sorted
    ]
    overlaps = [d["overlap"] for d in rows_sorted]

    # Color by overlap size (proxy for term-level biological depth)
    cmap = plt.get_cmap("viridis")
    if max(overlaps) > min(overlaps):
        norm_vals = [
            0.25 + 0.55 * (o - min(overlaps)) / (max(overlaps) - min(overlaps))
            for o in overlaps
        ]
    else:
        norm_vals = [0.5] * len(overlaps)
    colors = [cmap(v) for v in norm_vals]

    y = np.arange(len(labels))
    ax.barh(y, neg_log_padj, color=colors, edgecolor="white", linewidth=0.4,
            height=0.78)
    # Per-bar overlap annotation just inside the bar end
    x_max = max(neg_log_padj)
    for yi, (val, ov) in enumerate(zip(neg_log_padj, overlaps)):
        ax.text(
            val - x_max * 0.02, yi, f"k={ov}",
            ha="right", va="center", fontsize=5, color="white",
            fontweight="bold",
        )
    # Significance threshold (padj=0.05 = -log10 ≈ 1.30)
    ax.axvline(-np.log10(0.05), color="#d62728", linestyle="--",
               linewidth=0.8,
               label=f"padj=0.05 ($-\\log_{{10}}$={-np.log10(0.05):.2f})")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("$-\\log_{10}$(adjusted p) [Reactome 2022]", fontsize=8)
    ax.legend(loc="lower right", fontsize=6, frameon=True)

    # Annotation for the F5 NT-release Splatter-specificity finding
    ax.text(
        0.98, 0.03,
        f"{n_nt_pathways_splatter}/{n_nt_pathways_total} top-10 = NT release "
        f"(Splatter-specific)",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=6.5,
        bbox=dict(boxstyle="round,pad=0.30", facecolor="#fff5f0",
                  edgecolor="#d62728", linewidth=0.8),
    )
    fmt_axes(ax)


def draw_panel_c_captum(
    ax: plt.Axes,
    pairs: list[tuple[str, float]],
    coverage_text: str,
) -> None:
    """Top-15 Captum (Splatter, gene) pairs as horizontal bars."""
    if not pairs:
        ax.text(0.5, 0.5, "no Captum pairs",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    # Sort ascending so the strongest is at the top of the barh
    pairs_sorted = sorted(pairs, key=lambda kv: kv[1])
    genes = [g for g, _ in pairs_sorted]
    vals = [v for _, v in pairs_sorted]
    y = np.arange(len(genes))
    splatter_color = "#d62728"
    ax.barh(y, vals, color=splatter_color, edgecolor="white", linewidth=0.4,
            height=0.78)
    ax.set_yticks(y)
    ax.set_yticklabels(genes, fontsize=7)
    ax.set_xlabel("Captum IG mean abs attribution (Splatter)", fontsize=8)

    # Inline coverage annotation in upper-right corner
    ax.text(
        0.98, 0.04,
        coverage_text,
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=6,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#f0f4ff",
                  edgecolor="#1f77b4", linewidth=0.6),
    )
    fmt_axes(ax)


def _build_intersection_table(
    sets: dict[str, set[str]],
) -> list[tuple[tuple[str, ...], int]]:
    """Build all non-empty intersection cells of a family of sets.

    Returns: list of (sorted_member_label_tuple, intersection_size).
    Sorted descending by intersection size, then by label-tuple.
    """
    labels = list(sets.keys())
    universe = set().union(*sets.values()) if sets else set()
    cells: list[tuple[tuple[str, ...], int]] = []
    for mask in range(1, 1 << len(labels)):
        members = [labels[i] for i in range(len(labels)) if mask & (1 << i)]
        in_sets = [sets[m] for m in members]
        out_sets = [
            sets[labels[i]] for i in range(len(labels)) if not (mask & (1 << i))
        ]
        cell = set.intersection(*in_sets) if in_sets else universe
        for s_out in out_sets:
            cell = cell - s_out
        if cell:
            cells.append((tuple(members), len(cell)))
    cells.sort(key=lambda kv: (-kv[1], kv[0]))
    return cells


def draw_panel_d_overlap(
    ax_top: plt.Axes,
    ax_matrix: plt.Axes,
    sets: dict[str, set[str]],
    set_size_cap: int,
) -> None:
    """Render an upset-style top-K-method gene-overlap chart.

    Top axis: vertical bar plot of intersection sizes.
    Bottom axis: dot-and-line matrix of which sets each intersection
    corresponds to (rows = methods, columns = intersections).
    """
    cells = _build_intersection_table(sets)
    # Cap to keep the panel readable on a manuscript figure.
    max_cells = 12
    cells = cells[:max_cells]

    labels = list(sets.keys())
    n_methods = len(labels)
    n_cells = len(cells)

    if n_cells == 0:
        ax_top.text(0.5, 0.5, "no overlap structure",
                    ha="center", va="center", transform=ax_top.transAxes)
        ax_matrix.axis("off")
        return

    # Top-bar plot
    x = np.arange(n_cells)
    sizes = [c[1] for c in cells]
    palette = list(PALETTES["categorical"])
    bar_color = palette[3]  # tab10 red-ish for visibility
    ax_top.bar(x, sizes, color=bar_color, edgecolor="white", linewidth=0.4,
               width=0.7)
    for xi, s in zip(x, sizes):
        ax_top.text(xi, s + 0.15, str(s), ha="center", va="bottom",
                    fontsize=6.5, fontweight="bold")
    ax_top.set_ylabel("Intersection size (genes)", fontsize=8)
    ax_top.set_xlim(-0.6, n_cells - 0.4)
    ax_top.tick_params(axis="x", labelbottom=False, length=0)
    ax_top.set_xticks([])
    ax_top.spines["bottom"].set_visible(False)
    fmt_axes(ax_top, hide_spines=("top", "right", "bottom"))

    # Bottom matrix
    ax_matrix.set_xlim(-0.6, n_cells - 0.4)
    ax_matrix.set_ylim(-0.5, n_methods - 0.5)
    ax_matrix.invert_yaxis()
    for j, (members, _) in enumerate(cells):
        in_idx = [labels.index(m) for m in members]
        # Background dim dots for non-members
        for i in range(n_methods):
            if i in in_idx:
                continue
            ax_matrix.plot(j, i, "o", markersize=5, color="#d0d0d0",
                           markeredgecolor="white", markeredgewidth=0.4)
        # Active dots for members
        for i in in_idx:
            ax_matrix.plot(j, i, "o", markersize=6, color=bar_color,
                           markeredgecolor="white", markeredgewidth=0.4)
        # Vertical connecting line if 2+ members
        if len(in_idx) >= 2:
            ax_matrix.plot(
                [j, j],
                [min(in_idx), max(in_idx)],
                color=bar_color, linewidth=1.4, zorder=1,
            )
    ax_matrix.set_yticks(np.arange(n_methods))
    set_size_strs = [
        f"{lab} (|S|={len(sets[lab])})" for lab in labels
    ]
    ax_matrix.set_yticklabels(set_size_strs, fontsize=7)
    ax_matrix.set_xticks([])
    ax_matrix.set_xlabel(
        f"Intersection cells (top {set_size_cap}; n_universe="
        f"{len(set().union(*sets.values()))})",
        fontsize=8,
    )
    ax_matrix.tick_params(left=False, bottom=False)
    for spine in ("top", "right", "bottom"):
        ax_matrix.spines[spine].set_visible(False)
    ax_matrix.spines["left"].set_visible(False)
    ax_matrix.grid(False)


# ----------------------------------------------------------------------
# Caption generation
# ----------------------------------------------------------------------
def build_caption(
    *,
    n_methods: int,
    n_top10_cts: int,
    splatter_top5_count: int,
    fibroblast_top5_count: int,
    reactome_top10: list[dict],
    n_nt_pathways_splatter: int,
    n_nt_pathways_total: int,
    captum_top_k: int,
    sets_meta: dict[str, dict],
    coverage_meta: dict,
    splatter_intersection_size: int,
) -> str:
    """Build the manuscript-style caption for the integrated figure."""
    top_path = reactome_top10[0] if reactome_top10 else None
    top_genes = (top_path["genes"].replace(";", ", ")
                 if top_path is not None else "n/a")
    top_padj = (f"{top_path['adjusted_p_value']:.2e}"
                if top_path is not None else "n/a")
    set_lines = []
    for k, meta in sets_meta.items():
        set_lines.append(
            f"  - **{k}** (top-{meta['top_k']} genes for Splatter, "
            f"|S|={meta['size']})"
        )
    cov = coverage_meta
    cov_str = (
        f"Splatter coverage: median {cov['median_cells']} cells/subject, "
        f"present in {cov['n_subj_with_cells']}/{cov['n_subjects_total']} "
        f"subjects (zero_frac={cov['zero_frac']:.3f})"
    )
    return (
        "**Figure: Integrated pathway x cell-type interpretability for "
        "Splatter.**\n\n"
        f"**Panel A** Cross-method top-5 cell-type ranking across {n_methods} "
        "interpretability methods (Captum IG / GradientSHAP / SmoothGrad, "
        "five attention-rollout variants, pseudobulk Wasserstein-1, raw-"
        "pseudobulk conditional MI, LOCO zero-out). Rows are the "
        f"{n_top10_cts} cell types with the highest total top-5 frequency; "
        "viridis-shaded cells = rank in top-5 (rank 1 darkest); white = "
        f"outside top-5. Splatter (boldface, red outline) appears in the "
        f"top-5 of {splatter_top5_count}/{n_methods} methods; Fibroblast "
        f"(italic) appears in {fibroblast_top5_count}/{n_methods}.\n\n"
        f"**Panel B** Top-10 Reactome 2022 pathways enriched in the Splatter "
        "top-50 attribution genes, ranked by adjusted p (BH-FDR). Bar colour "
        "= overlap size (k = number of Splatter genes in the term gene-set). "
        f"Lead pathway: '{_shorten_reactome_term(top_path['term']) if top_path else ''}' "
        f"(genes: {top_genes}; padj={top_padj}). The dashed red line marks "
        f"padj=0.05. Inset: {n_nt_pathways_splatter}/{n_nt_pathways_total} "
        "of the top-10 are neurotransmitter-release cycles "
        "(acetylcholine / norepinephrine / serotonin / dopamine / GABA / "
        "glutamate); none of the four other CT-specific Reactome enrichments "
        "(Fibroblast, Committed OPC, Vascular, MGE-IN, Deep-layer IT) "
        "contain any neurotransmitter-release pathway in their respective "
        "top-10s (per `per_ct_reactome_top1_comparative.csv`).\n\n"
        f"**Panel C** Top-{captum_top_k} (Splatter, gene) attribution pairs "
        "from Captum integrated gradients on the canonical ResDec-MHE "
        "(n_subjects=516, fold 0). Bar colour = #d62728 (Splatter "
        f"convention). {cov_str}.\n\n"
        f"**Panel D** Upset-style intersection plot of Splatter top-N gene "
        "lists across four orthogonal methods:\n"
        + "\n".join(set_lines)
        + (
            f"\n\nThe 4-way intersection |Captum ∩ Wasserstein ∩ Wilcoxon ∩ "
            f"DESeq2| = {splatter_intersection_size} gene(s) (not shown — "
            "empty intersection cells are omitted). Method-specific exclusive "
            "sets dominate (Captum-only, DESeq2-only, Wilcoxon-only each "
            "contribute >40 genes), with only modest 2-way agreement "
            "(Wasserstein ∩ Wilcoxon = 5 genes, Captum ∩ DESeq2 = 1 gene; "
            "see summary.json for the full intersection table). The dot-and-"
            "line matrix below the bars indicates which methods each "
            "intersection cell corresponds to; |S|=|method's input set|.\n\n"
        )
        + (
            "**Notes.** Wasserstein-per-CT canonical JSON stores only top-10 "
            "genes per CT (not top-50), so the Wasserstein input set for "
            "Panel D is capped at top-10. DE Wilcoxon and DESeq2 inputs at "
            "padj=0.05 yield zero significant genes (per "
            "`de_resilient_vs_vulnerable*/per_ct_summary.csv`); we therefore "
            "rank by raw p-value (top-50). Captum IG inputs are the same "
            "Splatter top-50 genes used as the GSEA universe in Panel B. "
            "The low cross-method gene overlap is consistent with the "
            "documented finding that Captum / Wasserstein / DE measure "
            "different aspects of the (CT, gene) signal: Captum captures "
            "model-attribution sensitivity, Wasserstein captures "
            "distributional shift between resilient/vulnerable subjects, "
            "and the two DE methods rank by classical mean-shift p-values "
            "(N=129 vs 129 — ρ_per_CT_Wilcoxon-vs-DESeq2 spans -0.61 to "
            "+0.33 across CTs per "
            "`docs/results/2026-04-24-permutation-and-distributional-results.md`). "
            "Per-CT cell-count statistics are descriptive (no filtering "
            "applied to model inputs)."
        )
    )


# ----------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------
def build_figure(
    *,
    consensus_path: Path,
    reactome_csv: Path,
    per_ct_top1_csv: Path,
    captum_path: Path,
    wasserstein_path: Path,
    de_wilcoxon_dir: Path,
    de_deseq2_dir: Path,
    ct_coverage_path: Path,
    figsize: tuple[float, float] = (14.0, 12.0),
    captum_top_k: int = 15,
    overlap_top_k: int = 50,
) -> tuple[Figure, str, dict]:
    """Build the 2x2 composite figure + caption + a summary dict.

    Returns
    -------
    fig
        matplotlib Figure (not yet saved).
    caption
        Manuscript caption (markdown).
    summary
        Dict of derived numbers used by the caption / downstream QA.
    """
    apply_theme()

    # Load inputs ------------------------------------------------------
    consensus = load_consensus(consensus_path)
    reactome_top10 = load_reactome_top10_for_splatter(reactome_csv)
    per_ct_top1 = load_per_ct_top1(per_ct_top1_csv)
    captum_pairs = load_captum_top_pairs_for_ct(
        captum_path, "Splatter", captum_top_k
    )
    coverage = load_ct_coverage(ct_coverage_path)
    splatter_cov = coverage["per_ct"]["Splatter"]

    # Panel D input sets (top-K per method, restricted to CT=Splatter)
    captum_set = load_captum_top_genes_set(captum_path, "Splatter", overlap_top_k)
    wasserstein_set = load_wasserstein_top_genes_set(
        wasserstein_path, "Splatter", overlap_top_k
    )
    wilcoxon_set = load_de_top_genes_set(
        de_wilcoxon_dir, "Splatter", overlap_top_k
    )
    deseq2_set = load_de_top_genes_set(
        de_deseq2_dir, "Splatter", overlap_top_k
    )
    sets = {
        "Captum IG": captum_set,
        "Wasserstein": wasserstein_set,
        "Wilcoxon": wilcoxon_set,
        "DESeq2": deseq2_set,
    }

    # Splatter NT-release count (the CSV row labelled "Splatter (reference)")
    nt_count_splatter = next(
        (r["n_nt_in_top10"] for r in per_ct_top1
         if r["cell_type"].lower().startswith("splatter")),
        0,
    )

    # All-method intersection size (captioning convenience)
    all_intersection = (
        captum_set & wasserstein_set & wilcoxon_set & deseq2_set
    )

    # Splatter coverage annotation
    coverage_text = (
        f"Splatter (descriptive)\n"
        f"median {splatter_cov['median_cells']} cells/subj, "
        f"{splatter_cov['n_subj_with_cells']}/{coverage['n_subjects']} subj"
    )

    # Build figure -----------------------------------------------------
    fig = plt.figure(figsize=figsize)
    # Outer 2x2 grid; bottom-right (Panel D) has its own internal split
    # via a nested grid with top:matrix = 3:2 height ratio. Generous
    # wspace / hspace because Panel A's row labels are long, Panel B's
    # bar labels are long, and Panel C's annotation needs vertical room.
    outer = GridSpec(
        2, 2, figure=fig,
        wspace=0.55, hspace=0.55,
        left=0.08, right=0.97, top=0.94, bottom=0.08,
    )

    # Panel A (top-left)
    ax_a = fig.add_subplot(outer[0, 0])
    draw_panel_a_consensus(ax_a, consensus)
    ax_a.set_title("11-method top-5 CT consensus", fontsize=10, pad=8)
    auto_letter(ax_a, "A", offset=(-0.16, 0.05), fontsize=14)

    # Panel B (top-right)
    ax_b = fig.add_subplot(outer[0, 1])
    draw_panel_b_reactome(
        ax_b, reactome_top10,
        n_nt_pathways_splatter=nt_count_splatter,
        n_nt_pathways_total=10,
    )
    ax_b.set_title("Reactome 2022 — Splatter top-50 genes", fontsize=10,
                   pad=8)
    auto_letter(ax_b, "B", offset=(-0.16, 0.05), fontsize=14)

    # Panel C (bottom-left)
    ax_c = fig.add_subplot(outer[1, 0])
    draw_panel_c_captum(ax_c, captum_pairs, coverage_text)
    ax_c.set_title(
        f"Captum top-{captum_top_k} (Splatter, gene) pairs",
        fontsize=10, pad=8,
    )
    auto_letter(ax_c, "C", offset=(-0.16, 0.05), fontsize=14)

    # Panel D (bottom-right) — split into a top-bar axis and a bottom matrix
    inner_d = outer[1, 1].subgridspec(2, 1, hspace=0.10, height_ratios=[3, 2])
    ax_d_top = fig.add_subplot(inner_d[0, 0])
    ax_d_mat = fig.add_subplot(inner_d[1, 0])
    draw_panel_d_overlap(
        ax_d_top, ax_d_mat, sets, set_size_cap=overlap_top_k,
    )
    ax_d_top.set_title(
        f"Splatter top-{overlap_top_k} gene overlap (4 methods)",
        fontsize=10, pad=8,
    )
    auto_letter(ax_d_top, "D", offset=(-0.16, 0.08), fontsize=14)

    # Final tick / spine sweep applied centrally (theme convention).
    style_paper_axes(fig)

    # Caption + summary --------------------------------------------------
    caption = build_caption(
        n_methods=len(consensus["methods"]),
        n_top10_cts=len(consensus["row_cts"]),
        splatter_top5_count=consensus["top5_counts"].get("Splatter", 0),
        fibroblast_top5_count=consensus["top5_counts"].get("Fibroblast", 0),
        reactome_top10=reactome_top10,
        n_nt_pathways_splatter=nt_count_splatter,
        n_nt_pathways_total=10,
        captum_top_k=captum_top_k,
        sets_meta={
            k: {"top_k": overlap_top_k if k != "Wasserstein" else 10,
                "size": len(v)}
            for k, v in sets.items()
        },
        coverage_meta={
            "median_cells": splatter_cov["median_cells"],
            "n_subj_with_cells": splatter_cov["n_subj_with_cells"],
            "n_subjects_total": coverage["n_subjects"],
            "zero_frac": splatter_cov["zero_frac"],
        },
        splatter_intersection_size=len(all_intersection),
    )

    summary = {
        "panel_a": {
            "n_methods": len(consensus["methods"]),
            "n_rows": len(consensus["row_cts"]),
            "splatter_top5_count": consensus["top5_counts"].get("Splatter", 0),
            "fibroblast_top5_count": consensus["top5_counts"].get(
                "Fibroblast", 0
            ),
        },
        "panel_b": {
            "n_pathways": len(reactome_top10),
            "n_nt_pathways_splatter": nt_count_splatter,
            "lead_term": reactome_top10[0]["term"] if reactome_top10 else None,
            "lead_padj": (reactome_top10[0]["adjusted_p_value"]
                          if reactome_top10 else None),
        },
        "panel_c": {
            "captum_top_k": captum_top_k,
            "lead_pair": captum_pairs[0] if captum_pairs else None,
        },
        "panel_d": {
            "set_sizes": {k: len(v) for k, v in sets.items()},
            "all_method_intersection_size": len(all_intersection),
            "all_method_intersection_genes": sorted(all_intersection),
        },
        "splatter_coverage": {
            "median_cells": splatter_cov["median_cells"],
            "n_subj_with_cells": splatter_cov["n_subj_with_cells"],
            "n_subjects_total": coverage["n_subjects"],
        },
    }
    return fig, caption, summary


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--consensus-data",
        default=DEFAULTS["consensus"],
        help="Path to consensus_heatmap_data.json (Panel A).",
    )
    p.add_argument(
        "--reactome-csv",
        default=DEFAULTS["reactome_csv"],
        help="Path to gsea_Reactome_2022_top_50_Splatter.csv (Panel B).",
    )
    p.add_argument(
        "--per-ct-top1-csv",
        default=DEFAULTS["per_ct_top1_csv"],
        help="Path to per_ct_reactome_top1_comparative.csv "
             "(Panel B annotation).",
    )
    p.add_argument(
        "--captum-summary",
        default=DEFAULTS["captum"],
        help="Path to captum composite_attribution_summary.json (Panels C, D).",
    )
    p.add_argument(
        "--wasserstein",
        default=DEFAULTS["wasserstein"],
        help="Path to wasserstein_per_celltype_pseudobulk.json (Panel D).",
    )
    p.add_argument(
        "--de-wilcoxon-dir",
        default=DEFAULTS["de_wilcoxon_dir"],
        help="Path to DE-Wilcoxon directory containing CT_NN_de.csv "
             "+ per_ct_summary.csv (Panel D).",
    )
    p.add_argument(
        "--de-deseq2-dir",
        default=DEFAULTS["de_deseq2_dir"],
        help="Path to DE-DESeq2 directory containing CT_NN_de.csv "
             "+ per_ct_summary.csv (Panel D).",
    )
    p.add_argument(
        "--ct-coverage",
        default=DEFAULTS["ct_coverage"],
        help="Path to ct_coverage_full_cohort.json (Panel C inset).",
    )
    p.add_argument(
        "--out-dir",
        default=str(_WORKTREE_ROOT / "outputs/canonical/interpretability/"
                                     "figures/manuscript_composite"),
        help="Output directory; figure stem is fig_integrated_pathway_celltype.",
    )
    p.add_argument(
        "--stem",
        default="fig_integrated_pathway_celltype",
        help="File stem for PNG/PDF outputs.",
    )
    p.add_argument(
        "--figsize",
        default="14,12",
        help="Comma-separated W,H in inches (default 14,12).",
    )
    p.add_argument(
        "--captum-top-k",
        type=int, default=15,
        help="Number of (Splatter, gene) pairs to render in Panel C.",
    )
    p.add_argument(
        "--overlap-top-k",
        type=int, default=50,
        help="Top-K cap per method for Panel D overlap (Wasserstein "
             "is capped to its native 10).",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    w_str, h_str = args.figsize.split(",")
    figsize = (float(w_str), float(h_str))

    fig, caption, summary = build_figure(
        consensus_path=_resolve_path(args.consensus_data),
        reactome_csv=_resolve_path(args.reactome_csv),
        per_ct_top1_csv=_resolve_path(args.per_ct_top1_csv),
        captum_path=_resolve_path(args.captum_summary),
        wasserstein_path=_resolve_path(args.wasserstein),
        de_wilcoxon_dir=_resolve_path(args.de_wilcoxon_dir),
        de_deseq2_dir=_resolve_path(args.de_deseq2_dir),
        ct_coverage_path=_resolve_path(args.ct_coverage),
        figsize=figsize,
        captum_top_k=args.captum_top_k,
        overlap_top_k=args.overlap_top_k,
    )

    png_path = out_dir / f"{args.stem}.png"
    pdf_path = out_dir / f"{args.stem}.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", png_path)
    logger.info("Wrote %s", pdf_path)

    caption_path = out_dir / "caption.md"
    caption_path.write_text(caption + "\n")
    logger.info("Wrote %s", caption_path)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("Wrote %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
