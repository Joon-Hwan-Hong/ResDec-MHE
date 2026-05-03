#!/usr/bin/env python
"""README Figure 4: cross-method set agreement (2 stacked UpSet plots).

Renders ``figures/fig4_upset.png`` with two stacked UpSet panels with method-
family color encoding:

  Top panel: UpSet on the top-5 cell-types ranked by each of 11 attribution /
             attention / distributional / information-theoretic / perturbation
             methods (IG, GradientSHAP, SmoothGrad, AttnLRP, GMAR, GAF AF,
             GAF AGF, GAF GF, Wasserstein, CMI, LOCO). Two CTs (Splatter,
             Fibroblast) appear in 11/11 and 10/11 sets respectively; agreement
             decays below those. Set bars (left totals) and intersection bars
             (top) are colored by method family (gradient-attribution = blue,
             attention-based = orange, distributional = green, information-
             theoretic = red, perturbation = purple); mixed-family intersections
             remain grey.

  Bottom panel: UpSet on the global top-50 (CT, gene) pairs across the 6
                gene-rankable methods (Captum IG, GradientSHAP, SmoothGrad,
                Wasserstein, DE Wilcoxon, DE DESeq2). The other 5 EXP-016
                methods (AttnLRP, GMAR, GAF AF/AGF/GF) are CT-only and the
                two EXP-016 task-excluded methods (LOCO, raw-pseudobulk CMI)
                are likewise CT-only, so neither family contributes a gene
                axis to this UpSet. Color encoding here uses the gradient-
                attribution family (blue) for Captum IG/GradientSHAP/
                SmoothGrad, the differential-expression family (pink) for DE
                Wilcoxon/DE DESeq2, and distributional (green) for Wasserstein.

  Legend: a single figure-level legend is rendered below both panels with
  color swatches for every family that appears in either panel. Legend is
  placed in figure-level coordinates via ``bbox_to_anchor`` so it never
  overlaps the upsetplot internal axes.

Data sources (read fresh on every run; no hardcoded numbers):
  - Top panel:
    ``outputs/canonical/interpretability/figures/consensus_heatmap/consensus_heatmap_data.json``
    Provides per-method per-CT rank under ``ranks[ct][method]``. Top-5 set
    membership = ``ranks[ct][method] in {1, 2, 3, 4, 5}``.
  - Bottom panel:
    Six per-method primary sources. Captum IG / GradientSHAP / SmoothGrad
    each ship a built-in ``top_cell_type_gene_pairs`` (length 50) in their
    summary JSON. Wasserstein, DE Wilcoxon, and DE DESeq2 ship per-CT lists
    only; the bottom-panel script pools ``(CT, gene, metric)`` triples across
    all CTs, sorts by the method's primary metric (Wasserstein distance for
    Wasserstein; ``p_value`` ascending for DE), and keeps the global top-50.
    Provenance for the input file paths is identical to the inputs declared
    in ``run_11method_gene_jaccard.py`` (the orchestrator that produced
    ``cross_method_gene_jaccard.json``).

Layout decision
---------------
``upsetplot==0.9.0``'s ``UpSet.plot()`` expects ``fig.get_figwidth()`` and
``fig.get_figheight()``, both of which raise ``AttributeError`` on
``matplotlib.figure.SubFigure``. Stacking two UpSet plots in one figure via
``fig.subfigures()`` is therefore not supported by this version. Workaround:
render each panel to a PNG byte buffer, load as a PIL ``Image``, and
composite both onto a 14x9 in canvas via ``Axes.imshow``. This preserves
the upsetplot internal layout while letting us stack and label both panels.

Verification (printed to stdout)
--------------------------------
  - Top panel: n_methods (=11), n_cts_in_universe (= union of top-5),
    count of CTs in 11/11, count of CTs in 10/11.
  - Bottom panel: n_methods (=6), n_pairs_in_universe (= union),
    count of pairs in 6/6, median pairwise Jaccard between methods.
    Reports the gene-set median Jaccard from
    ``cross_method_gene_jaccard.json`` (= 0.16 expected) for cross-check;
    note that the (CT, gene) pair median is computed independently and may
    differ from the gene-only number.

See ``docs/plans/2026-05-02-readme-redesign-design.md`` Figures section
"Figure 4 (PNG): set agreement" for the spec this implements.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import sys
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # must precede pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

# upsetplot 0.9.0 emits FutureWarnings from internal pandas inplace fillna usage
# that we cannot fix from caller code. Suppress to keep stdout focused on
# verification numbers; surface real warnings from our own code only.
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"upsetplot\.plotting",
)
from upsetplot import UpSet, from_indicators  # noqa: E402

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from src.visualization.theme import apply_theme, save_fig  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Top-K thresholds (mirrors the design + run_11method_gene_jaccard.py).
TOP_K_CT = 5  # top-5 CTs per method for top panel
TOP_K_PAIRS = 50  # top-50 (CT, gene) pairs per method for bottom panel

# Order matches the design's "11 methods" enumeration so the consensus heatmap
# JSON ordering can be re-used directly.
EXPECTED_TOP_PANEL_METHODS = (
    "IG", "GradientSHAP", "SmoothGrad",
    "AttnLRP", "GMAR", "GAF AF", "GAF AGF", "GAF GF",
    "Wasserstein", "CMI", "LOCO",
)

# Order matches ``cross_method_gene_jaccard.json::labels_label_sorted``
# (alphabetical, which is what UpSet uses for the dot-grid columns).
EXPECTED_BOTTOM_PANEL_METHODS = (
    "Captum IG", "DE DESeq2", "DE Wilcoxon",
    "GradientSHAP", "SmoothGrad", "Wasserstein",
)

# ---------------------------------------------------------------------------
# Method-family color encoding
# ---------------------------------------------------------------------------
#
# Five families distinguished by tab10 colors. Bottom panel introduces a new
# DE family (pink) that does not appear in the top panel because no DE method
# is in the 11-method top-panel enumeration. Mixed-family intersections in the
# UpSet (intersections that span 2+ families) keep the default ``MIXED_FAMILY``
# grey color so the family encoding only "fires" when the convergence story
# is intra-family.

GRAY_MIXED = "#404040"  # default fill for mixed-family intersections + bars

FAMILY_COLOR = {
    "Gradient-attribution": "#1f77b4",  # tab10 blue
    "Attention-based":      "#ff7f0e",  # tab10 orange
    "Distributional":       "#2ca02c",  # tab10 green
    "Information-theoretic": "#d62728",  # tab10 red
    "Perturbation":         "#9467bd",  # tab10 purple
    # Bottom-panel-only family (DE Wilcoxon + DE DESeq2). User spec allowed
    # either reusing attention-orange or introducing a new color; pink picked
    # for visual distinctness from the gradient-attribution blue.
    "Differential expression": "#e377c2",  # tab10 pink
}

# Top-panel: 11 methods -> family.
FAMILY_BY_METHOD_TOP = {
    "IG":            "Gradient-attribution",
    "GradientSHAP":  "Gradient-attribution",
    "SmoothGrad":    "Gradient-attribution",
    "AttnLRP":       "Attention-based",
    "GMAR":          "Attention-based",
    "GAF AF":        "Attention-based",
    "GAF AGF":       "Attention-based",
    "GAF GF":        "Attention-based",
    "Wasserstein":   "Distributional",
    "CMI":           "Information-theoretic",
    "LOCO":          "Perturbation",
}

# Bottom-panel: 6 methods -> family.
FAMILY_BY_METHOD_BOTTOM = {
    "Captum IG":    "Gradient-attribution",
    "GradientSHAP": "Gradient-attribution",
    "SmoothGrad":   "Gradient-attribution",
    "DE Wilcoxon":  "Differential expression",
    "DE DESeq2":    "Differential expression",
    "Wasserstein":  "Distributional",
}


def _family_color_by_method(family_by_method: dict[str, str]) -> dict[str, str]:
    """Return ``{method_label: family_hex_color}`` for a panel."""
    return {m: FAMILY_COLOR[fam] for m, fam in family_by_method.items()}


def _families_present(family_by_method: dict[str, str]) -> list[str]:
    """Return the unique families used by ``family_by_method``, in stable order."""
    seen: list[str] = []
    for fam in family_by_method.values():
        if fam not in seen:
            seen.append(fam)
    return seen


def _group_methods_by_family(
    family_by_method: dict[str, str],
    *,
    methods: list[str] | tuple[str, ...] | None = None,
) -> dict[str, list[str]]:
    """Return ``{family: [methods belonging to that family]}``.

    Restricts to ``methods`` (the panel-specific column order) when given so
    the caller can pass a subset to ``style_subsets(present=...)``.
    """
    if methods is None:
        methods = list(family_by_method.keys())
    out: dict[str, list[str]] = {}
    for m in methods:
        fam = family_by_method.get(m)
        if fam is None:
            continue
        out.setdefault(fam, []).append(m)
    return out


# ---------------------------------------------------------------------------
# Top panel: load top-5 CTs per method from the consensus heatmap JSON
# ---------------------------------------------------------------------------


def load_top5_cts_per_method(
    consensus_json: Path,
    *,
    top_k: int = TOP_K_CT,
) -> tuple[dict[str, set[str]], list[str]]:
    """Return ``{method: set of top-K cell types}`` from the consensus JSON.

    Schema (``consensus_heatmap_data.json``):
      - ``methods``: list[str], length 11
      - ``row_cts``: list[str], length 10 -- the union of all top-K CTs across
        methods (smaller than the 31 source CTs because most never enter any
        method's top-5)
      - ``ranks``: dict[ct -> dict[method -> int rank in 1..31]]
      - ``top5_counts``: dict[ct -> int count of methods that placed ct in
        their top-5]

    Returns
    -------
    sets : dict[str, set[str]]
        method label -> set of cell-type names ranked in top-K by that method.
    method_order : list[str]
        Source order from the JSON (for stable iteration).
    """
    payload = json.loads(consensus_json.read_text())
    methods = list(payload["methods"])
    ranks = payload["ranks"]
    sets: dict[str, set[str]] = {m: set() for m in methods}
    for ct, per_method in ranks.items():
        for method, rank in per_method.items():
            if isinstance(rank, (int, float)) and rank <= top_k:
                sets[method].add(ct)
    return sets, methods


# ---------------------------------------------------------------------------
# Bottom panel: load top-50 (CT, gene) pairs per method
# ---------------------------------------------------------------------------


def load_captum_top_pairs(
    summary_path: Path,
    *,
    top_k: int = TOP_K_PAIRS,
) -> set[tuple[str, str]]:
    """Read built-in top-K ``(CT, gene)`` pairs from a Captum-family summary.

    Schema: ``top_cell_type_gene_pairs`` is a list of dicts with keys
    ``cell_type``, ``gene``, ``mean_abs_attribution`` already sorted descending
    by attribution magnitude. Captum IG / GradientSHAP / SmoothGrad emit this
    block at length 50; we slice to ``top_k`` defensively.
    """
    summary = json.loads(summary_path.read_text())
    rows = summary["top_cell_type_gene_pairs"][:top_k]
    return {(r["cell_type"], r["gene"]) for r in rows}


def load_wasserstein_top_pairs(
    summary_path: Path,
    *,
    top_k: int = TOP_K_PAIRS,
) -> set[tuple[str, str]]:
    """Build top-K ``(CT, gene)`` pairs by Wasserstein-1 distance (descending).

    Schema: ``per_cell_type[i].wasserstein_per_gene_top10`` is a length-10 list
    of ``[gene, distance]`` pairs (the canonical JSON hard-caps at 10 per CT).
    With 31 CTs the candidate pool is 310 pairs; the global top-K is the
    sorted-by-distance head. If fewer than 10 per-CT entries are present the
    pool may be smaller than 310, but for this dataset every CT yields 10.
    """
    summary = json.loads(summary_path.read_text())
    pool: list[tuple[float, str, str]] = []
    for block in summary["per_cell_type"]:
        ct = block["cell_type"]
        for gene, value in block.get("wasserstein_per_gene_top10", []):
            pool.append((float(value), ct, gene))
    pool.sort(key=lambda kv: kv[0], reverse=True)
    return {(ct, gene) for _, ct, gene in pool[:top_k]}


def _resolve_ct_index(de_summary_csv: Path) -> dict[str, int]:
    """Map ``cell_type -> cell_type_index`` from per-CT summary CSV."""
    out: dict[str, int] = {}
    with de_summary_csv.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            out[row["cell_type"]] = int(row["cell_type_index"])
    return out


def load_de_top_pairs(
    de_dir: Path,
    *,
    top_k: int = TOP_K_PAIRS,
) -> set[tuple[str, str]]:
    """Build top-K ``(CT, gene)`` pairs by p-value (ascending) across all CTs.

    Reads ``per_ct_summary.csv`` for the CT -> CT-index mapping, then for
    each CT reads ``CT_NN_de.csv`` and pulls every (gene, p_value) row. Pools
    all (CT, gene, p_value) triples across all CTs, drops non-finite p_value
    rows, sorts ascending by p_value, and returns the top-K set.
    """
    de_dir = Path(de_dir)
    summary_csv = de_dir / "per_ct_summary.csv"
    if not summary_csv.is_file():
        raise FileNotFoundError(f"DE summary CSV missing: {summary_csv}")
    ct_index = _resolve_ct_index(summary_csv)
    pool: list[tuple[float, str, str]] = []
    for ct, idx in ct_index.items():
        ct_csv = de_dir / f"CT_{idx:02d}_de.csv"
        if not ct_csv.is_file():
            continue
        with ct_csv.open(newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                gene = row.get("gene")
                if not gene:
                    continue
                try:
                    p = float(row["p_value"])
                except (KeyError, ValueError, TypeError):
                    continue
                if not np.isfinite(p):
                    continue
                pool.append((p, ct, gene))
    pool.sort(key=lambda kv: kv[0])
    return {(ct, gene) for _, ct, gene in pool[:top_k]}


def load_top_pairs_per_method(
    *,
    captum_ig_path: Path,
    gradientshap_path: Path,
    smoothgrad_path: Path,
    wasserstein_path: Path,
    de_wilcoxon_dir: Path,
    de_deseq2_dir: Path,
    top_k: int = TOP_K_PAIRS,
) -> dict[str, set[tuple[str, str]]]:
    """Return ``{method: set of top-K (CT, gene) pairs}`` for the 6 methods.

    Method labels match the order/spelling in
    ``cross_method_gene_jaccard.json::labels_label_sorted``.
    """
    return {
        "Captum IG": load_captum_top_pairs(captum_ig_path, top_k=top_k),
        "DE DESeq2": load_de_top_pairs(de_deseq2_dir, top_k=top_k),
        "DE Wilcoxon": load_de_top_pairs(de_wilcoxon_dir, top_k=top_k),
        "GradientSHAP": load_captum_top_pairs(gradientshap_path, top_k=top_k),
        "SmoothGrad": load_captum_top_pairs(smoothgrad_path, top_k=top_k),
        "Wasserstein": load_wasserstein_top_pairs(wasserstein_path, top_k=top_k),
    }


# ---------------------------------------------------------------------------
# Set summaries (multiway counts + pairwise Jaccard)
# ---------------------------------------------------------------------------


def multiway_counts(sets: dict[str, set]) -> dict[int, int]:
    """Return ``{k: n_items in EXACTLY k sets}`` for k = 1..len(sets).

    Caller can derive ``>= k`` via cumulative-from-the-right.
    """
    member_count: dict[object, int] = {}
    for s in sets.values():
        for item in s:
            member_count[item] = member_count.get(item, 0) + 1
    out: dict[int, int] = {}
    for k in range(1, len(sets) + 1):
        out[k] = sum(1 for v in member_count.values() if v == k)
    return out


def pairwise_jaccard_median(sets: dict[str, set]) -> float:
    """Median Jaccard over all ``C(M, 2)`` unordered pairs of methods."""
    labels = sorted(sets.keys())
    vals: list[float] = []
    for a, b in combinations(labels, 2):
        sa = sets[a]
        sb = sets[b]
        if not sa and not sb:
            vals.append(0.0)
            continue
        union = len(sa | sb)
        inter = len(sa & sb)
        vals.append(inter / union if union else 0.0)
    if not vals:
        return float("nan")
    return float(np.median(np.asarray(vals, dtype=np.float64)))


# ---------------------------------------------------------------------------
# UpSet rendering helpers
# ---------------------------------------------------------------------------


def _build_indicator_df(
    sets: dict[str, set],
    *,
    column_order: list[str],
) -> pd.DataFrame:
    """Items x methods boolean DataFrame for ``from_indicators``.

    Items are stringified (``f"{ct}|{gene}"`` for pair tuples) so the index
    has unique hashable labels usable by upsetplot.
    """
    universe: list = []
    seen: set = set()
    for col in column_order:
        for item in sets[col]:
            if item not in seen:
                seen.add(item)
                universe.append(item)
    # Friendly index label (CT|gene) when items are tuples; else as-is.
    if universe and isinstance(universe[0], tuple):
        index = [f"{ct}|{gene}" for (ct, gene) in universe]
    else:
        index = list(universe)
    data = {col: [item in sets[col] for item in universe] for col in column_order}
    return pd.DataFrame(data, index=index)


def _render_upset_to_image(
    sets: dict[str, set],
    *,
    column_order: list[str],
    figsize: tuple[float, float],
    dpi: int,
    show_counts: str | bool = True,
    sort_by: str = "cardinality",
    min_subset_size: int | None = None,
    family_by_method: dict[str, str] | None = None,
) -> Image.Image:
    """Render a single UpSet plot to a PIL Image at the requested DPI.

    Strategy: build the upsetplot, ``savefig`` to a BytesIO buffer at the
    requested DPI, re-open as PIL, and return. Caller composites the result
    onto the final figure.

    ``min_subset_size`` is forwarded to ``UpSet`` to suppress empty / single-
    member intersections that would otherwise crowd the dot-grid; default
    ``None`` keeps every visible intersection.

    Family coloring (when ``family_by_method`` is provided)
    ------------------------------------------------------
    - **Set bars (totals, left side)**: colored by the family of each method
      via ``upset.style_categories(method, bar_facecolor=family_color)``.
    - **Intersection bars (top)** + **matrix dots**: for every family that
      appears in this panel, ``upset.style_subsets(present=[methods in
      family], absent=[methods NOT in family], facecolor=family_color)`` is
      called once. This colors *only* intersections whose membership is
      EXACTLY the methods of one family — so a "pure single-family"
      convergence is colored, while any intersection mixing two or more
      families remains the default ``GRAY_MIXED`` (#404040). Single-method
      intersections (degree=1) are colored by that method's family because
      they trivially "all members from same family".
    """
    df = _build_indicator_df(sets, column_order=column_order)
    if df.empty:
        # Defensive: empty universe -> single blank panel with a label so the
        # caller can still composite (rather than blowing up downstream).
        fig_blank = plt.figure(figsize=figsize, dpi=dpi)
        plt.text(
            0.5, 0.5, "(no items)", ha="center", va="center",
            fontsize=14, transform=plt.gca().transAxes,
        )
        plt.axis("off")
        buf = io.BytesIO()
        fig_blank.savefig(buf, dpi=dpi, bbox_inches="tight", format="png")
        plt.close(fig_blank)
        buf.seek(0)
        return Image.open(buf).convert("RGBA")

    membership = from_indicators(column_order, data=df)

    fig_temp = plt.figure(figsize=figsize, dpi=dpi)
    upset_kwargs = dict(
        sort_by=sort_by,
        sort_categories_by="cardinality",
        show_counts=show_counts,
        facecolor=GRAY_MIXED,
        other_dots_color=0.30,
        shading_color=0.05,
        with_lines=True,
    )
    if min_subset_size is not None:
        upset_kwargs["min_subset_size"] = min_subset_size
    upset = UpSet(membership, **upset_kwargs)

    # ------------------------------------------------------------------
    # Family color encoding (must happen BEFORE upset.plot)
    # ------------------------------------------------------------------
    if family_by_method is not None:
        # 1) Set-bar (totals) coloring per method.
        for method in column_order:
            fam = family_by_method.get(method)
            if fam is None:
                continue
            upset.style_categories(
                method,
                bar_facecolor=FAMILY_COLOR[fam],
            )

        # 2) Intersection bars + matrix dots. For every family present in
        #    this panel, color the EXACTLY-that-family intersection (i.e.
        #    every method in the family is "present" AND every method in
        #    OTHER families is "absent"). This implementation matches the
        #    spec: "if all members are from the same family, color by that
        #    family; if mixed, color grey".
        family_to_methods = _group_methods_by_family(
            family_by_method, methods=column_order,
        )
        all_methods_in_panel = list(column_order)
        for fam, fam_methods in family_to_methods.items():
            absent_methods = [
                m for m in all_methods_in_panel if m not in fam_methods
            ]
            # Note: we color *every* subset of the family, not just the
            # full-degree one — single-method intersections (degree=1) and
            # within-family pairs both qualify as "same family". Achieved
            # by setting absent=other-family methods and present=None
            # (matches subsets that may include any subset of fam_methods
            # while excluding all other-family methods).
            upset.style_subsets(
                absent=absent_methods if absent_methods else None,
                facecolor=FAMILY_COLOR[fam],
            )

    upset.plot(fig=fig_temp)

    buf = io.BytesIO()
    fig_temp.savefig(buf, dpi=dpi, bbox_inches="tight", format="png")
    plt.close(fig_temp)
    buf.seek(0)
    return Image.open(buf).convert("RGBA")


def _composite_two_panels(
    img_top: Image.Image,
    img_bot: Image.Image,
    *,
    figsize: tuple[float, float],
    dpi: int,
    families: list[str] | None = None,
    include_mixed_swatch: bool = True,
) -> plt.Figure:
    """Stack the two PNG-buffer images into a single matplotlib Figure.

    Uses ``Axes.imshow`` with ``set_axis_off()`` so each panel keeps its own
    upsetplot internal axes (totals + intersection + matrix) and we just
    arrange them vertically. ``figsize`` and ``dpi`` are the OUTPUT canvas
    dimensions; the ``img_top`` / ``img_bot`` already carry their own internal
    DPI from upsetplot's renderer.

    ``families`` (when provided) drives a figure-level legend with one color
    swatch per family. The legend is anchored at figure-coords below the two
    panels via ``bbox_to_anchor=(0.5, 0.0)``, so it cannot overlap the
    upsetplot internal axes (which were rasterised onto the panel images).
    ``include_mixed_swatch=True`` appends a "Mixed families" grey swatch so
    readers can map the default color too.
    """
    if families:
        # Reserve ~12 % of vertical space at the bottom for the legend strip.
        gridspec_kw = {"height_ratios": [1.0, 1.0], "hspace": 0.04}
        fig = plt.figure(figsize=figsize, dpi=dpi)
        # 2 panels stacked + 1 thin legend axis at the bottom (no axis -- just
        # acts as anchor for the legend). bbox_to_anchor uses figure coords so
        # the legend is guaranteed not to clip into the panel images.
        gs = fig.add_gridspec(
            nrows=2, ncols=1,
            top=0.98, bottom=0.10, left=0.01, right=0.99,
            **gridspec_kw,
        )
        axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])]
    else:
        fig, axes = plt.subplots(2, 1, figsize=figsize, dpi=dpi)
        fig.subplots_adjust(top=0.99, bottom=0.01, left=0.01, right=0.99,
                            hspace=0.04)
    for ax, img in zip(axes, (img_top, img_bot)):
        ax.imshow(np.asarray(img), interpolation="bilinear")
        ax.set_axis_off()

    if families:
        from matplotlib.patches import Patch
        handles = [
            Patch(facecolor=FAMILY_COLOR[fam], edgecolor="none", label=fam)
            for fam in families
        ]
        if include_mixed_swatch:
            handles.append(
                Patch(facecolor=GRAY_MIXED, edgecolor="none",
                      label="Mixed families"),
            )
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.005),
            ncol=min(len(handles), 7),
            frameon=False,
            fontsize=10,
            title="Method family",
            title_fontsize=11,
            handlelength=1.6,
            handleheight=1.2,
            columnspacing=1.6,
            borderaxespad=0.0,
        )
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--consensus-json",
        type=Path,
        default=(
            _WORKTREE_ROOT
            / "outputs/canonical/interpretability/figures/consensus_heatmap"
            / "consensus_heatmap_data.json"
        ),
        help="Top panel: consensus heatmap data with per-method CT ranks.",
    )
    parser.add_argument(
        "--captum-ig",
        type=Path,
        default=(
            _WORKTREE_ROOT
            / "outputs/canonical/interpretability/captum_ig"
            / "composite_attribution_summary.json"
        ),
    )
    parser.add_argument(
        "--gradientshap",
        type=Path,
        default=(
            _WORKTREE_ROOT
            / "outputs/canonical/interpretability/captum_robustness"
            / "gradientshap_attribution_summary.json"
        ),
    )
    parser.add_argument(
        "--smoothgrad",
        type=Path,
        default=(
            _WORKTREE_ROOT
            / "outputs/canonical/interpretability/captum_robustness"
            / "smoothgrad_attribution_summary.json"
        ),
    )
    parser.add_argument(
        "--wasserstein",
        type=Path,
        default=(
            _WORKTREE_ROOT
            / "outputs/canonical/interpretability/distributional_resilience"
            / "wasserstein_per_celltype_pseudobulk.json"
        ),
    )
    parser.add_argument(
        "--de-wilcoxon-dir",
        type=Path,
        default=(
            _WORKTREE_ROOT
            / "outputs/canonical/interpretability/de_resilient_vs_vulnerable"
        ),
    )
    parser.add_argument(
        "--de-deseq2-dir",
        type=Path,
        default=(
            _WORKTREE_ROOT
            / "outputs/canonical/interpretability"
            / "de_resilient_vs_vulnerable_deseq2"
        ),
    )
    parser.add_argument(
        "--gene-jaccard-json",
        type=Path,
        default=(
            _WORKTREE_ROOT
            / "outputs/canonical/interpretability/cross_method_gene_jaccard.json"
        ),
        help="Reference gene-set Jaccard JSON used for stdout cross-check.",
    )
    parser.add_argument(
        "--out-stem",
        type=Path,
        default=_WORKTREE_ROOT / "figures/fig4_upset",
        help="Output PNG stem (extension is appended).",
    )
    parser.add_argument(
        "--figsize",
        type=float, nargs=2, default=(14.0, 9.0),
        metavar=("W", "H"),
        help="Final canvas figsize in inches.",
    )
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--top-min-subset",
        type=int, default=1,
        help="Suppress intersections smaller than this in the top panel.",
    )
    parser.add_argument(
        "--bottom-min-subset",
        type=int, default=1,
        help="Suppress intersections smaller than this in the bottom panel.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    apply_theme("paper")

    # ------------------------------------------------------------------
    # Top panel: top-5 CTs per method
    # ------------------------------------------------------------------
    top_sets, top_methods = load_top5_cts_per_method(args.consensus_json)
    # Re-order to match design enumeration; defensively pad with any extras.
    top_methods_ordered = [m for m in EXPECTED_TOP_PANEL_METHODS if m in top_sets]
    extras_top = [m for m in top_methods if m not in EXPECTED_TOP_PANEL_METHODS]
    top_methods_ordered.extend(extras_top)

    # ------------------------------------------------------------------
    # Bottom panel: top-50 (CT, gene) pairs per method
    # ------------------------------------------------------------------
    inputs = {
        "Captum IG": args.captum_ig,
        "GradientSHAP": args.gradientshap,
        "SmoothGrad": args.smoothgrad,
        "Wasserstein": args.wasserstein,
        "DE Wilcoxon": args.de_wilcoxon_dir,
        "DE DESeq2": args.de_deseq2_dir,
    }
    for label, path in inputs.items():
        if not path.exists():
            logger.error("Missing input for %s: %s", label, path)
            return 1

    pair_sets = load_top_pairs_per_method(
        captum_ig_path=args.captum_ig,
        gradientshap_path=args.gradientshap,
        smoothgrad_path=args.smoothgrad,
        wasserstein_path=args.wasserstein,
        de_wilcoxon_dir=args.de_wilcoxon_dir,
        de_deseq2_dir=args.de_deseq2_dir,
        top_k=TOP_K_PAIRS,
    )
    bottom_methods_ordered = [
        m for m in EXPECTED_BOTTOM_PANEL_METHODS if m in pair_sets
    ]
    if len(bottom_methods_ordered) != len(EXPECTED_BOTTOM_PANEL_METHODS):
        missing = set(EXPECTED_BOTTOM_PANEL_METHODS) - set(bottom_methods_ordered)
        logger.error("Missing bottom-panel methods: %s", sorted(missing))
        return 1

    # ------------------------------------------------------------------
    # Render each UpSet panel to a PNG buffer + composite
    # ------------------------------------------------------------------
    # Each panel sized to ~half the final canvas (height-wise); the
    # downstream composite stacks them vertically.
    panel_dpi = args.dpi
    panel_figsize = (float(args.figsize[0]), float(args.figsize[1]) / 2.0)

    # Top panel: sort by ``-degree`` so the universal intersection (all 11
    # methods) comes first; this puts Splatter (in 11/11) leftmost and
    # Fibroblast (in 10/11) just to its right, telling the convergence
    # story directly.
    img_top = _render_upset_to_image(
        top_sets,
        column_order=top_methods_ordered,
        figsize=panel_figsize,
        dpi=panel_dpi,
        min_subset_size=args.top_min_subset,
        sort_by="-degree",
        family_by_method=FAMILY_BY_METHOD_TOP,
    )
    # Bottom panel: sort by cardinality (descending) so the largest
    # intersections come first; emphasizes that the within-family
    # intersections (Captum vs SmoothGrad: 49 of 50) dominate over
    # cross-family agreement (which is ~0).
    img_bot = _render_upset_to_image(
        pair_sets,
        column_order=bottom_methods_ordered,
        figsize=panel_figsize,
        dpi=panel_dpi,
        min_subset_size=args.bottom_min_subset,
        sort_by="cardinality",
        family_by_method=FAMILY_BY_METHOD_BOTTOM,
    )
    # Union of families across both panels for a single shared legend.
    families_top = _families_present(FAMILY_BY_METHOD_TOP)
    families_bot = _families_present(FAMILY_BY_METHOD_BOTTOM)
    families_union: list[str] = []
    for fam in families_top + families_bot:
        if fam not in families_union:
            families_union.append(fam)
    fig = _composite_two_panels(
        img_top, img_bot,
        figsize=tuple(args.figsize),
        dpi=panel_dpi,
        families=families_union,
        include_mixed_swatch=True,
    )
    written = save_fig(fig, args.out_stem, formats=("png",), dpi=panel_dpi)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Verification stdout
    # ------------------------------------------------------------------
    top_universe = set().union(*top_sets.values())
    top_member_count: dict[str, int] = {}
    for ct in top_universe:
        top_member_count[ct] = sum(1 for s in top_sets.values() if ct in s)
    n_in_11_of_11 = sum(1 for c in top_member_count.values() if c == 11)
    n_in_10_of_11 = sum(1 for c in top_member_count.values() if c == 10)
    top_pairwise_jacc = pairwise_jaccard_median(top_sets)

    bot_universe = set().union(*pair_sets.values())
    bot_member_count: dict[tuple[str, str], int] = {}
    for pair in bot_universe:
        bot_member_count[pair] = sum(1 for s in pair_sets.values() if pair in s)
    n_pairs_in_6_of_6 = sum(1 for c in bot_member_count.values() if c == 6)
    bot_pair_pairwise_jacc = pairwise_jaccard_median(pair_sets)

    # Gene-set median Jaccard (for cross-check vs the bottom panel pair-set)
    ref_gene_jacc = float("nan")
    if args.gene_jaccard_json.exists():
        ref = json.loads(args.gene_jaccard_json.read_text())
        ref_gene_jacc = float(ref["pairwise_summary"]["median_jaccard"])

    print("=" * 78)
    print("README Figure 4 - verification (all values from primary files)")
    print("=" * 78)
    print("Top panel - top-5 CTs per method:")
    print(f"  n_methods                         : {len(top_methods_ordered)}")
    print(f"  n_cts_in_universe (union of top-5): {len(top_universe)}")
    print(f"  n CTs appearing in 11/11 methods  : {n_in_11_of_11}")
    print(f"  n CTs appearing in 10/11 methods  : {n_in_10_of_11}")
    print(f"  median pairwise Jaccard (CT sets) : {top_pairwise_jacc:.4f}")
    print()
    print("Bottom panel - top-50 (CT, gene) pairs per method:")
    print(f"  n_methods                         : {len(bottom_methods_ordered)}")
    print(f"  n_pairs_in_universe (union)       : {len(bot_universe)}")
    print(f"  n pairs appearing in 6/6 methods  : {n_pairs_in_6_of_6}")
    print(
        f"  median pairwise Jaccard (pair sets): "
        f"{bot_pair_pairwise_jacc:.4f}  "
        f"(CROSS-CHECK: gene-set median = {ref_gene_jacc:.4f}; "
        f"pair-set is finer granularity so values may differ)"
    )
    print()
    print(f"DPI                                : {panel_dpi}")
    print()
    for path in written:
        size_mb = path.stat().st_size / (1024 * 1024)
        size_kb = path.stat().st_size / 1024
        print(
            f"Wrote: {path}  ({size_mb:.3f} MB / {size_kb:.1f} KB)"
        )
    print("=" * 78)

    # Per-method set-size breakdown for sanity / debugging.
    print("Per-method set sizes:")
    print("  TOP PANEL (top-5 CTs)")
    for m in top_methods_ordered:
        print(f"    {m:<14s} : {len(top_sets[m]):>3d}")
    print("  BOTTOM PANEL (top-50 (CT, gene) pairs)")
    for m in bottom_methods_ordered:
        print(f"    {m:<14s} : {len(pair_sets[m]):>3d}")

    # Family color encoding -- print which methods got which colors.
    print()
    print("Family color encoding (TOP PANEL, 11 methods):")
    fam_to_methods_top = _group_methods_by_family(
        FAMILY_BY_METHOD_TOP, methods=top_methods_ordered,
    )
    for fam in _families_present(FAMILY_BY_METHOD_TOP):
        members = fam_to_methods_top.get(fam, [])
        print(
            f"  {fam:<24s} {FAMILY_COLOR[fam]}  "
            f"-> [{', '.join(members)}]"
        )
    print()
    print("Family color encoding (BOTTOM PANEL, 6 methods):")
    fam_to_methods_bot = _group_methods_by_family(
        FAMILY_BY_METHOD_BOTTOM, methods=bottom_methods_ordered,
    )
    for fam in _families_present(FAMILY_BY_METHOD_BOTTOM):
        members = fam_to_methods_bot.get(fam, [])
        print(
            f"  {fam:<24s} {FAMILY_COLOR[fam]}  "
            f"-> [{', '.join(members)}]"
        )
    print(f"  {'Mixed-family (default)':<24s} {GRAY_MIXED}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
