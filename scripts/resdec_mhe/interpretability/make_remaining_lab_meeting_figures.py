"""Orchestrator: render the 3 remaining lab-meeting figures.

Produces three figures used in the 2026-04-29 lab meeting that the prior
``make_lab_meeting_figures.py`` orchestrator did NOT cover:

  Slot 4.3 — fig_slot4_3_statistical_rigor.{png,pdf}
      3-panel composite (1 row x 3 cols, figsize=(15, 5)) via
      ``src.visualization.composite.make_panel``:
        Panel A — permutation null histogram with canonical R² overlay.
        Panel B — bootstrap 95% CI forest plot with baseline reference points.
        Panel C — paired Wilcoxon heatmap, 22 baselines x 5 seeds.

  S1 — fig_S1_per_fold_r2_strip.{png,pdf}
      Single-panel strip plot of per-fold R² for ResDec-MHE (with optional
      TabPFN-2.6 overlay) plus mean line.  Figsize (8, 5).

  S8 — fig_S8_published_marker_concordance.{png,pdf}
      Horizontal bar chart summarizing 4 published marker panels (Madduri
      2026, Mathys 2023, Sun 2023, Mathys 2025) showing total -> in-HVG ->
      in top-K.  Annotates direct anchors PLP1, CPLX1; calls out PVALB rank
      #958 PVALB+ Inh10 replication failure.  Figsize (10, 6).

Default I/O paths are resolved from the project worktree root and can be
overridden via env-vars or argparse.

Usage:
    uv run python scripts/resdec_mhe/interpretability/make_remaining_lab_meeting_figures.py
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from src.visualization.composite import make_panel  # noqa: E402
from src.visualization.theme import (  # noqa: E402
    apply_theme,
    fmt_axes,
    save_fig,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# Defaults (env-var / argparse driven per project rules)
# ===========================================================================

_DEFAULT_PERM_SUMMARY = os.environ.get(
    "LAB_MEETING_PERM_SUMMARY",
    str(_WORKTREE_ROOT
        / "outputs/canonical/permutation_test/permutation_summary.json"),
)
_DEFAULT_STATISTICAL_RIGOR = os.environ.get(
    "LAB_MEETING_STAT_RIGOR_JSON",
    str(_WORKTREE_ROOT
        / "outputs/canonical/interpretability/statistical_rigor.json"),
)
_DEFAULT_SEED_WILCOXON = os.environ.get(
    "LAB_MEETING_SEED_WILCOXON_JSON",
    str(_WORKTREE_ROOT
        / "outputs/canonical/interpretability"
        / "seed_variation_wilcoxon_all_baselines.json"),
)
_DEFAULT_BASELINE_TABLE = os.environ.get(
    "LAB_MEETING_BASELINE_TABLE_CSV",
    str(_WORKTREE_ROOT
        / "outputs/canonical/interpretability/paper_baseline_table.csv"),
)
_DEFAULT_MARKER_MD = os.environ.get(
    "LAB_MEETING_MARKER_MD",
    str(_WORKTREE_ROOT
        / "outputs/canonical/interpretability/published_marker_concordance.md"),
)
_DEFAULT_OUT_DIR = os.environ.get(
    "LAB_MEETING_OUT_DIR",
    str(_WORKTREE_ROOT
        / "outputs/canonical/interpretability/figures/lab_meeting"),
)


# ===========================================================================
# Slot 4.3 — Statistical rigor 3-panel composite
# ===========================================================================


def _draw_panel_A_permutation(
    ax: plt.Axes,
    *,
    perm_summary: dict,
) -> None:
    """Panel A: histogram of null R² with canonical overlay."""
    null_r2 = np.asarray(
        perm_summary.get("null_mean_r2_per_perm", []), dtype=np.float64,
    )
    # Lab-meeting unified-R² choice: use pooled bootstrap point estimate (0.4493)
    # to match the R² shown in the slot 4.1/4.2 predicted-vs-actual scatter
    # legends. This is the same model as the per-fold-mean R²=0.4436 reported in
    # MASTER-INFO; the difference is statistic choice (pooled vs mean-of-folds)
    # and is < 0.01 R² — within fold variance. Single number throughout the deck
    # avoids audience confusion.
    canonical = 0.4493
    null_mean = float(perm_summary.get("null_mean", null_r2.mean() if null_r2.size else 0.0))
    null_std = float(
        perm_summary.get(
            "null_std", null_r2.std(ddof=1) if null_r2.size > 1 else 0.0,
        )
    )
    z = float(perm_summary.get("z_under_null", float("nan")))
    n_ge = int(perm_summary.get("n_perms_ge_canonical", 0))
    n_perms = int(perm_summary.get("n_permutations", null_r2.size))

    # Histogram of nulls.
    if null_r2.size > 0:
        bins = max(5, min(10, null_r2.size))
        ax.hist(
            null_r2, bins=bins,
            color="#888888", edgecolor="white", linewidth=0.6, alpha=0.8,
            label="null R² (N=%d)" % n_perms,
        )
    # Canonical line.
    ax.axvline(
        canonical, color="#1f77b4", linestyle="-", linewidth=1.6,
        label=f"canonical R² = {canonical:.4f}",
    )
    # Null mean line.
    ax.axvline(
        null_mean, color="#d62728", linestyle="--", linewidth=1.0,
        label=f"null mean = {null_mean:+.3f}",
    )

    # Pad x-limits so canonical R² is visible at far right.
    if null_r2.size > 0:
        x_lo = float(null_r2.min()) - 0.05
        x_hi = max(canonical + 0.05, float(null_r2.max()) + 0.05)
        ax.set_xlim(x_lo, x_hi)

    ax.set_xlabel("R²")
    ax.set_ylabel("# permutations")

    annotation = (
        f"z = {z:.2f}\n"
        f"null mean = {null_mean:+.3f} ± {null_std:.3f}\n"
        f"{n_ge}/{n_perms} ≥ canonical"
    )
    ax.text(
        0.03, 0.97, annotation,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=7,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#cccccc", alpha=0.9),
    )
    ax.legend(loc="upper right", fontsize=6)


def _draw_panel_B_bootstrap(
    ax: plt.Axes,
    *,
    statistical_rigor: dict,
    baseline_table: list[dict],
) -> None:
    """Panel B: bootstrap CI forest plot with baseline reference points."""
    boot = statistical_rigor.get("bootstrap_r2_ci", {})
    point = float(boot.get("point_r2", 0.4436))
    ci_lo = float(boot.get("ci_lower", 0.37))
    ci_hi = float(boot.get("ci_upper", 0.51))
    n_boot = int(boot.get("n_boot", 1000))

    # Reference baselines (read mean R² from CSV, top 4 strong points).
    refs: list[tuple[str, float]] = []
    name_map = {
        "tabpfn_2_6_standalone": "TabPFN-2.6",
        "ridge_A": "Ridge",
        "xgboost_A": "XGBoost",
        "randomforest_A": "RandomForest",
    }
    for row in baseline_table:
        m = row.get("model")
        if m in name_map:
            try:
                refs.append((name_map[m], float(row["r2_mean"])))
            except (KeyError, ValueError):
                continue
    # Order: ResDec-MHE on top row, baselines stacked below.
    rows = [("ResDec-MHE\n(95% CI)", point, ci_lo, ci_hi)]
    rows.extend([(name, val, None, None) for name, val in refs])

    ys = np.arange(len(rows))[::-1]  # top row at y=max so it appears first
    for y, item in zip(ys, rows):
        if len(item) == 4:
            name, val, lo, hi = item
        else:
            name, val = item[0], item[1]
            lo, hi = None, None
        if lo is not None and hi is not None:
            ax.plot([lo, hi], [y, y], color="#1f77b4", linewidth=2.4, zorder=2)
            ax.plot(
                [val], [y],
                marker="o", color="#1f77b4",
                markersize=6, markeredgecolor="white", markeredgewidth=0.8,
                zorder=3,
            )
        else:
            ax.plot(
                [val], [y],
                marker="s", color="#888888",
                markersize=5, markeredgecolor="white", markeredgewidth=0.6,
                zorder=3,
            )
    ax.axvline(0.0, color="black", linestyle=":", linewidth=0.6, alpha=0.6)
    ax.set_yticks(ys)
    ax.set_yticklabels([item[0] for item in rows], fontsize=7)
    ax.set_xlabel("R²")
    ax.set_xlim(min(-0.05, ci_lo - 0.05), max(0.6, ci_hi + 0.05))

    annotation = (
        f"n_boot = {n_boot}\n"
        f"95% CI = [{ci_lo:.2f}, {ci_hi:.2f}]"
    )
    ax.text(
        0.97, 0.05, annotation,
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=7,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#cccccc", alpha=0.9),
    )


def _draw_panel_C_wilcoxon_heatmap(
    ax: plt.Axes,
    *,
    seed_wilcoxon: dict,
) -> None:
    """Panel C: paired-Wilcoxon p heatmap, baselines x seeds."""
    seeds = [str(s) for s in seed_wilcoxon.get("seeds", [42, 67, 21, 2000, 426])]
    per_baseline: dict = seed_wilcoxon.get("per_baseline", {})
    # Sort baselines by Stouffer p (most significant on top).
    baseline_items = sorted(
        per_baseline.items(),
        key=lambda kv: kv[1].get("stouffer_p_one_sided", 1.0),
    )
    baselines = [name for name, _ in baseline_items]
    n_b, n_s = len(baselines), len(seeds)

    # Build p matrix; missing = NaN (gray).
    p_mat = np.full((n_b, n_s), np.nan, dtype=np.float64)
    for i, name in enumerate(baselines):
        per_seed = per_baseline[name].get("per_seed", {})
        for j, sd in enumerate(seeds):
            seed_block = per_seed.get(sd, {})
            p = seed_block.get("wilcoxon_p_one_sided_greater")
            if p is not None:
                p_mat[i, j] = float(p)

    # Color = log10(p) clipped at p <= 0.0625; emphasize p <= 0.0312.
    eps = 1e-6
    log_p = np.log10(np.clip(p_mat, eps, None))
    # Cap floor at log10(0.001) and top at log10(0.0625) so that 0.0312 shows.
    vmin = np.log10(1e-3)
    vmax = np.log10(0.0625)
    cmap = plt.get_cmap("viridis_r").copy()
    cmap.set_bad(color="#dddddd")
    masked = np.ma.array(log_p, mask=np.isnan(log_p))
    im = ax.imshow(
        masked, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
        interpolation="nearest",
    )
    ax.set_xticks(np.arange(n_s))
    ax.set_xticklabels([f"seed {s}" for s in seeds], rotation=45, ha="right",
                       fontsize=7)
    ax.set_yticks(np.arange(n_b))
    ax.set_yticklabels(baselines, fontsize=6)
    ax.set_xlabel("random seed")
    ax.set_ylabel("baseline")

    # Annotate cells with p-value (skip NaN).
    for i in range(n_b):
        for j in range(n_s):
            v = p_mat[i, j]
            if np.isnan(v):
                continue
            color = "white" if log_p[i, j] < (vmin + (vmax - vmin) * 0.5) else "black"
            ax.text(
                j, i, f"{v:.4f}",
                ha="center", va="center",
                fontsize=5.5, color=color,
            )

    # Tab-style colorbar.
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("log₁₀(p)", fontsize=6)
    cbar.ax.tick_params(labelsize=5)

    # Stouffer combined p annotation: pull TabPFN as the marquee one.
    # Place below the panel inside the figure to avoid colliding with title.
    tab = per_baseline.get("TabPFN-2.6", {})
    stouffer = tab.get("stouffer_p_one_sided")
    if stouffer is not None:
        ax.text(
            0.50, -0.32,
            f"Stouffer combined p (TabPFN, 5 seeds) = {stouffer:.2e}",
            transform=ax.transAxes,
            ha="center", va="top", fontsize=6,
        )

    ax.grid(False)


def _read_baseline_table(csv_path: Path) -> list[dict]:
    """Read the baseline-table CSV into a list of dicts."""
    rows: list[dict] = []
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
    return rows


def build_slot4_3_statistical_rigor(
    *,
    permutation_summary: Path,
    statistical_rigor: Path,
    seed_wilcoxon: Path,
    baseline_table: Path,
) -> plt.Figure:
    """3-panel statistical-rigor composite (1 row x 3 cols, figsize=(15, 5))."""
    permutation_summary = Path(permutation_summary)
    statistical_rigor = Path(statistical_rigor)
    seed_wilcoxon = Path(seed_wilcoxon)
    baseline_table = Path(baseline_table)

    perm = json.loads(permutation_summary.read_text())
    rigor = json.loads(statistical_rigor.read_text())
    wilcoxon = json.loads(seed_wilcoxon.read_text())
    blines = _read_baseline_table(baseline_table)

    panels = [
        {
            "title": "Permutation null (N=10)",
            "draw": lambda ax: _draw_panel_A_permutation(
                ax, perm_summary=perm,
            ),
        },
        {
            "title": "Bootstrap 95% CI",
            "draw": lambda ax: _draw_panel_B_bootstrap(
                ax, statistical_rigor=rigor, baseline_table=blines,
            ),
        },
        {
            "title": "Paired Wilcoxon vs baselines",
            "draw": lambda ax: _draw_panel_C_wilcoxon_heatmap(
                ax, seed_wilcoxon=wilcoxon,
            ),
        },
    ]

    fig = make_panel(
        panels,
        layout=(1, 3),
        figsize=(15.0, 4.5),
        labels=True,
        # Tuck A/B/C labels close to the panel content (user pref). NB
        # ``auto_letter`` *adds* offset to the anchor (0, 1) for top-left,
        # so an effective y-coord of 1.02 means offset y = 0.02. Previously
        # offset y = 1.03 placed the label at axes-frac y = 2.03 — way up
        # in the suptitle band.
        label_kwargs={"offset": (-0.02, 0.02), "fontsize": 12.0},
        wspace=0.35,
        hspace=0.30,
    )
    # Tight margins so the panels fill the figure and the A/B/C labels are
    # visually adjacent to the panel content (previous top=0.86 left a big
    # blank band above the panels which exaggerated the label-to-content gap).
    fig.subplots_adjust(top=0.92, bottom=0.18, left=0.06, right=0.97)
    return fig


# ===========================================================================
# S1 — Per-fold R² strip
# ===========================================================================


def build_S1_per_fold_r2_strip(
    *,
    statistical_rigor: Path,
) -> plt.Figure:
    """Single-panel strip plot of per-fold R² (ResDec-MHE + optional TabPFN)."""
    statistical_rigor = Path(statistical_rigor)
    rigor = json.loads(statistical_rigor.read_text())
    per_fold = (
        rigor.get("provenance", {}).get("per_fold_r2", {})
    )
    ours = np.asarray(per_fold.get("ours", []), dtype=np.float64)
    tabpfn = np.asarray(
        per_fold.get("tabpfn_2_6_standalone", []), dtype=np.float64,
    )
    if ours.size == 0:
        raise ValueError(
            f"No 'ours' per-fold R² found in {statistical_rigor}"
        )

    n_folds = ours.size
    folds = np.arange(n_folds)
    mean_r2 = float(ours.mean())
    std_r2 = float(ours.std(ddof=1)) if ours.size > 1 else 0.0

    apply_theme()
    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    # ResDec-MHE per-fold dots.
    ax.scatter(
        folds, ours,
        s=80, color="#1f77b4", edgecolor="white", linewidth=0.8, zorder=3,
        label="ResDec-MHE (canonical)",
    )
    # Optional TabPFN reference dots (different color/marker, slightly offset).
    if tabpfn.size == n_folds:
        ax.scatter(
            folds + 0.15, tabpfn,
            s=70, color="#d62728", marker="^",
            edgecolor="white", linewidth=0.8, zorder=2,
            label="TabPFN-2.6 standalone",
        )

    # Mean line for ResDec-MHE.
    ax.axhline(
        mean_r2, color="#1f77b4", linestyle="--", linewidth=1.0,
        label=f"ResDec-MHE mean = {mean_r2:.4f}",
    )
    # Reference y=0 line.
    ax.axhline(0.0, color="black", linestyle=":", linewidth=0.6, alpha=0.6)

    ax.set_xticks(folds)
    ax.set_xticklabels([f"fold {f}" for f in range(n_folds)])
    ax.set_xlabel("outer cross-validation fold")
    ax.set_ylabel("R²")
    ax.set_xlim(-0.5, n_folds - 0.5 + 0.3)

    annotation = (
        f"mean R² = {mean_r2:.4f} ± {std_r2:.4f}\n"
        f"(mean ± std across {n_folds} folds)"
    )
    ax.text(
        0.02, 0.05, annotation,
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#cccccc", alpha=0.9),
    )

    ax.legend(loc="lower right", fontsize=7)
    fmt_axes(ax)
    fig.tight_layout()
    return fig


# ===========================================================================
# S8 — Published-marker concordance horizontal bar
# ===========================================================================


# Hard-coded counts extracted from
# outputs/canonical/interpretability/published_marker_concordance.md (sections
# A.1, A.2, B.1, B.2, C.1, C.2, D.1, D.2 read 2026-04-28).  These are stable
# numbers; the .md file is the canonical source.
_MARKER_DATA: list[dict] = [
    {
        "source":  "Madduri 2026",
        "context": "ROSMAP+Mathys 2019 PFC, AD case/control",
        "total":   49,
        "in_hvg":  33,
        "in_topk": 16,
    },
    {
        "source":  "Mathys 2023",
        "context": "ROSMAP DLPFC, cognitive resilience clusters",
        "total":   20,
        "in_hvg":  15,
        "in_topk": 5,
    },
    {
        "source":  "Sun 2023",
        "context": "Microglial DAM-state markers",
        "total":   23,
        "in_hvg":  17,
        "in_topk": 4,
    },
    {
        "source":  "Mathys 2025",
        "context": "Multi-region snATAC + multi-omics, AD progression",
        "total":   13,
        "in_hvg":  11,
        "in_topk": 4,
    },
]


def build_S8_published_marker_concordance(
    *,
    marker_md: Path | None = None,
) -> plt.Figure:
    """Horizontal-bar summary of published marker concordance across 4 sources.

    Notes
    -----
    Mathys 2025 numbers come directly from §D of the .md (PLP1 anchor: 13
    panel genes, 11 in HVG, 4 in any top-K set).  The .md file
    (``published_marker_concordance.md``) is the binding canonical source —
    we encode the per-source counts inline so the figure remains
    deterministic without re-parsing markdown tables.
    """
    apply_theme()
    fig, ax = plt.subplots(figsize=(10.0, 6.0))

    sources = [d["source"] for d in _MARKER_DATA]
    totals = np.asarray([d["total"]   for d in _MARKER_DATA], dtype=int)
    hvgs   = np.asarray([d["in_hvg"]  for d in _MARKER_DATA], dtype=int)
    topks  = np.asarray([d["in_topk"] for d in _MARKER_DATA], dtype=int)

    n = len(sources)
    y = np.arange(n)[::-1]  # top source at top
    bar_h = 0.27

    # Three stacked horizontal bars per source: total (lightest), in-HVG, in-topK.
    color_total = "#dcdcdc"
    color_hvg   = "#a6cee3"
    color_topk  = "#1f78b4"

    # Place the three bars vertically centered: total spanning full width,
    # in-HVG on top of total at slight offset, in-topK on top of in-HVG.
    for i, yi in enumerate(y):
        ax.barh(
            yi + bar_h, totals[i], height=bar_h,
            color=color_total, edgecolor="#999999", linewidth=0.6,
            label="total panel" if i == 0 else None,
        )
        ax.barh(
            yi, hvgs[i], height=bar_h,
            color=color_hvg, edgecolor="#1f4f71", linewidth=0.6,
            label="in our HVG (4,785 genes)" if i == 0 else None,
        )
        ax.barh(
            yi - bar_h, topks[i], height=bar_h,
            color=color_topk, edgecolor="#102f49", linewidth=0.6,
            label="in our top-K (Captum/stab/DE)" if i == 0 else None,
        )

        # Annotate counts at end of each bar.
        ax.text(totals[i] + 0.5, yi + bar_h, f"{totals[i]}",
                va="center", ha="left", fontsize=7)
        ax.text(hvgs[i] + 0.5, yi, f"{hvgs[i]} ({hvgs[i] * 100 // totals[i]}%)",
                va="center", ha="left", fontsize=7)
        ax.text(topks[i] + 0.5, yi - bar_h,
                f"{topks[i]} ({topks[i] * 100 // totals[i]}%)",
                va="center", ha="left", fontsize=7)

    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{d['source']}\n({d['context']})" for d in _MARKER_DATA],
        fontsize=8,
    )
    ax.set_xlabel("# genes")
    xmax = float(totals.max()) * 1.20
    ax.set_xlim(0, xmax)

    # Direct anchor annotations.
    anchor_text = (
        "Direct anchors:\n"
        "  * PLP1 - Committed OPC, Captum top-2 pair (Mathys 2025)\n"
        "  * CPLX1 - Splatter, Captum top-30 pair (Mathys 2023)\n"
        "  * CLU - top-50 global, GWAS-anchored (Mathys 2025 + Madduri)"
    )
    ax.text(
        0.98, 0.97, anchor_text,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=7,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#e8f4d8",
                  edgecolor="#7da840", alpha=0.95),
    )

    # Replication-failure callout (PVALB).
    failure_text = (
        "Caveat: PVALB rank #958 in MGE int. — direct failure to replicate\n"
        "Mathys 2023 PVALB+ Inh10 cognitive-resilience cluster as a per-CT\n"
        "attribution-importance signal in our model."
    )
    ax.text(
        0.02, 0.02, failure_text,
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=7,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fde2e2",
                  edgecolor="#c34646", alpha=0.95),
    )

    ax.legend(loc="lower right", fontsize=7, framealpha=0.95)
    fmt_axes(ax)
    fig.tight_layout()
    return fig


# ===========================================================================
# Top-level orchestrator
# ===========================================================================


def build_all_figures(
    *,
    permutation_summary: Path,
    statistical_rigor: Path,
    seed_wilcoxon: Path,
    baseline_table: Path,
    marker_md: Path,
    out_dir: Path,
) -> dict[str, list[Path]]:
    """Build + save the 3 remaining lab-meeting figures."""
    permutation_summary = Path(permutation_summary)
    statistical_rigor = Path(statistical_rigor)
    seed_wilcoxon = Path(seed_wilcoxon)
    baseline_table = Path(baseline_table)
    marker_md = Path(marker_md)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    apply_theme()

    written: dict[str, list[Path]] = {}

    # Slot 4.3 — 3-panel composite.
    fig1 = build_slot4_3_statistical_rigor(
        permutation_summary=permutation_summary,
        statistical_rigor=statistical_rigor,
        seed_wilcoxon=seed_wilcoxon,
        baseline_table=baseline_table,
    )
    written["slot4_3"] = save_fig(
        fig1, out_dir / "fig_slot4_3_statistical_rigor", dpi=600,
    )
    plt.close(fig1)

    # S1 — per-fold strip.
    fig2 = build_S1_per_fold_r2_strip(statistical_rigor=statistical_rigor)
    written["S1"] = save_fig(
        fig2, out_dir / "fig_S1_per_fold_r2_strip", dpi=600,
    )
    plt.close(fig2)

    # S8 — published marker concordance.
    fig3 = build_S8_published_marker_concordance(marker_md=marker_md)
    written["S8"] = save_fig(
        fig3, out_dir / "fig_S8_published_marker_concordance", dpi=600,
    )
    plt.close(fig3)

    for slot, paths in written.items():
        for p in paths:
            logger.info("wrote %s (%.1f KB)", p, p.stat().st_size / 1024.0)

    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--permutation-summary", default=_DEFAULT_PERM_SUMMARY)
    parser.add_argument("--statistical-rigor", default=_DEFAULT_STATISTICAL_RIGOR)
    parser.add_argument("--seed-wilcoxon", default=_DEFAULT_SEED_WILCOXON)
    parser.add_argument("--baseline-table", default=_DEFAULT_BASELINE_TABLE)
    parser.add_argument("--marker-md", default=_DEFAULT_MARKER_MD)
    parser.add_argument("--out-dir", default=_DEFAULT_OUT_DIR)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    build_all_figures(
        permutation_summary=Path(args.permutation_summary),
        statistical_rigor=Path(args.statistical_rigor),
        seed_wilcoxon=Path(args.seed_wilcoxon),
        baseline_table=Path(args.baseline_table),
        marker_md=Path(args.marker_md),
        out_dir=Path(args.out_dir),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
