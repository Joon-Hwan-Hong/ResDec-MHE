"""Orchestrator: render the 3 lab-meeting slot-anchor figures.

Produces three single-panel slide figures used in the 2026-04-29 lab meeting:

  Slot 1 — fig_slot1_residual_definition.{png,pdf}
      Histogram of the cognitive resilience target across N=516 subjects
      (the cognition_residual = global_cognition − E[cognition | pathology])
      with vertical lines at the resilient / vulnerable cutoffs (top-10 /
      bottom-10 of target distribution) and an annotation describing the
      resilience definition.

  Slot 2 — fig_slot2_marker_validation.{png,pdf}
      Heatmap of canonical marker rank (1 = highest expression in atlas)
      for 9 cell types × 6-9 marker genes, sourced from the cached
      ``extended_marker_verification.json`` and
      ``splatter_marker_verification.json`` artifacts.

  Slot 6 — fig_slot6_methods_recap.{png,pdf}
      Block diagram (no data plotting) summarizing the 5-fold CV pipeline,
      the statistical-rigor stack, and the interpretability paradigms.

Default I/O paths are resolved from the project worktree root.  All paths can
be overridden via CLI / env-var defaults per project conventions.

Usage:
    uv run python scripts/resdec_mhe/interpretability/make_lab_meeting_figures.py

CLI flags:
    --canonical-dir         path to outputs/canonical/p5_canonical_seed42
    --extended-marker-json  path to extended_marker_verification.json
    --splatter-marker-json  path to splatter_marker_verification.json
    --out-dir               output directory for the 3 figures
    --n-folds               number of folds to concatenate (default 5)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from src.visualization.theme import (  # noqa: E402
    apply_theme,
    fmt_axes,
    save_fig,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# Defaults (env-var / argparse driven per project rules)
# ===========================================================================

_DEFAULT_CANONICAL_DIR = os.environ.get(
    "LAB_MEETING_CANONICAL_DIR",
    str(_WORKTREE_ROOT / "outputs/canonical/p5_canonical_seed42"),
)
_DEFAULT_EXTENDED_MARKER_JSON = os.environ.get(
    "LAB_MEETING_EXTENDED_MARKER_JSON",
    str(_WORKTREE_ROOT
        / "outputs/canonical/interpretability/extended_marker_verification.json"),
)
_DEFAULT_SPLATTER_MARKER_JSON = os.environ.get(
    "LAB_MEETING_SPLATTER_MARKER_JSON",
    str(_WORKTREE_ROOT
        / "outputs/canonical/interpretability/splatter_marker_verification.json"),
)
_DEFAULT_OUT_DIR = os.environ.get(
    "LAB_MEETING_OUT_DIR",
    str(_WORKTREE_ROOT
        / "outputs/canonical/interpretability/figures/lab_meeting"),
)


# ===========================================================================
# Helpers
# ===========================================================================


def _load_concatenated_targets(
    canonical_dir: Path, n_folds: int = 5,
) -> np.ndarray:
    """Concatenate the ``targets`` field across all per-fold val NPZ files.

    The ``targets`` field of ``val_predictions_best.npz`` IS the cognitive
    residual that ResDec-MHE was trained to predict — already the residual of
    global cognition after regressing on pathology.  Concatenating across the
    5 outer folds gives one residual per subject for the full N=516 cohort.
    """
    pieces: list[np.ndarray] = []
    for f in range(n_folds):
        npz_path = canonical_dir / f"fold{f}/val_predictions_best.npz"
        if not npz_path.exists():
            logger.warning("missing %s — skipping", npz_path)
            continue
        d = np.load(npz_path, allow_pickle=True)
        pieces.append(np.asarray(d["targets"], dtype=np.float64))
    if not pieces:
        raise FileNotFoundError(
            f"No per-fold val_predictions_best.npz under {canonical_dir}"
        )
    return np.concatenate(pieces, axis=0)


def _load_per_fold_residuals(
    canonical_dir: Path, n_folds: int = 5,
) -> np.ndarray:
    """Concatenate the model residual (target − prediction) across folds.

    F1 CF defines resilient / vulnerable as the top / bottom 10 of this model
    residual.  We need it for the threshold lines on the slot-1 histogram.
    """
    pieces: list[np.ndarray] = []
    for f in range(n_folds):
        npz_path = canonical_dir / f"fold{f}/val_predictions_best.npz"
        if not npz_path.exists():
            continue
        d = np.load(npz_path, allow_pickle=True)
        t = np.asarray(d["targets"], dtype=np.float64)
        p = np.asarray(d["predictions"], dtype=np.float64)
        pieces.append(t - p)
    if not pieces:
        raise FileNotFoundError(
            f"No per-fold val_predictions_best.npz under {canonical_dir}"
        )
    return np.concatenate(pieces, axis=0)


# ===========================================================================
# Slot 1: residual histogram (resilience definition)
# ===========================================================================


def build_slot1_residual_definition(
    *,
    canonical_dir: Path,
    n_folds: int = 5,
) -> plt.Figure:
    """Build the single-panel cognitive-residual histogram for slot 1.

    The histogram covers the cognition residual (= cognition − E[cognition |
    pathology]) for all N subjects across the 5 outer folds, with vertical
    lines at the resilient / vulnerable thresholds (top-10 / bottom-10 cuts of
    the target distribution).
    """
    canonical_dir = Path(canonical_dir)
    targets = _load_concatenated_targets(canonical_dir, n_folds=n_folds)

    # Top-10 / bottom-10 cuts on the target distribution → tail thresholds.
    n = targets.shape[0]
    sorted_t = np.sort(targets)
    if n >= 20:
        vuln_thresh = float(sorted_t[9])     # 10th smallest
        resi_thresh = float(sorted_t[-10])   # 10th largest
    else:
        # fallback for very small N (testing): use 5 / 95% quantiles.
        vuln_thresh = float(np.quantile(targets, 0.05))
        resi_thresh = float(np.quantile(targets, 0.95))

    fig, ax = plt.subplots(figsize=(8.0, 6.0))

    # Histogram (~50 bins) of the residual.
    bins = 50
    counts, edges, _ = ax.hist(
        targets,
        bins=bins,
        color="#4c72b0",
        edgecolor="white",
        linewidth=0.5,
        alpha=0.85,
    )
    ymax = counts.max() * 1.15

    # Mean + std summary annotation.
    mu = float(np.mean(targets))
    sigma = float(np.std(targets, ddof=1)) if n > 1 else 0.0

    # Vertical lines: resilient (right tail) + vulnerable (left tail).
    ax.axvline(resi_thresh, color="#2ca02c", linestyle="--", linewidth=1.4,
               label=f"resilient threshold (top-10) = {resi_thresh:+.2f}")
    ax.axvline(vuln_thresh, color="#d62728", linestyle="--", linewidth=1.4,
               label=f"vulnerable threshold (bot-10) = {vuln_thresh:+.2f}")
    ax.axvline(0.0, color="black", linestyle=":", linewidth=0.8, alpha=0.6)

    ax.set_xlabel(
        "cognitive resilience residual\n"
        r"$y = \mathrm{cognition} - \mathbb{E}[\mathrm{cognition}\;|\;\mathrm{pathology}]$"
    )
    ax.set_ylabel(f"# subjects (N = {n})")
    ax.set_ylim(0, ymax)

    # Inline annotation: mean ± std + tail interpretation.
    annotation = (
        f"mean = {mu:+.2f}, std = {sigma:.2f}\n"
        r"resilient $\rightarrow$ right tail"   "\n"
        r"vulnerable $\rightarrow$ left tail"
    )
    ax.text(
        0.02, 0.97, annotation,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#cccccc", alpha=0.9),
    )

    ax.legend(loc="upper right", fontsize=7)
    fmt_axes(ax)
    fig.tight_layout()
    return fig


# ===========================================================================
# Slot 2: 9-CT marker validation heatmap
# ===========================================================================


# Canonical 9 cell types in display order (matches atlas conventions).
_NINE_CT_ORDER: list[tuple[str, str]] = [
    # (display_label, JSON_key_in_extended)
    ("Splatter",                    "Splatter (SST+CHODL+ projection-IN)"),
    ("Vascular",                    "Vascular (endothelial)"),
    ("Fibroblast",                  "Fibroblast (perivascular)"),
    ("MGE interneuron",             "MGE interneuron (LHX6+ MGE)"),
    ("Deep-layer IT",               "Deep-layer IT (RORB/FEZF2)"),
    ("Committed OPC",               "Committed OPC (myelinating progenitor)"),
    ("OPC",                         "OPC (PDGFRA+)"),
    ("Astrocyte",                   "Astrocyte (GFAP+)"),
    ("Microglia",                   "Microglia (CSF1R+)"),
]


def _collect_marker_rank_table(
    extended: dict,
    splatter: dict | None = None,
) -> tuple[list[str], list[str], np.ndarray]:
    """Build (rows, cols, rank_matrix) for the 9-CT marker heatmap.

    Each row is one of the 9 canonical CTs; each column is a marker gene
    associated with at least one CT.  Cell value is the **target_rank** of
    that gene in that CT (1 = highest expression in atlas; lower = better).
    Cells without any defined target_rank for the (CT, gene) pair are NaN
    and rendered as gray.
    """
    # Splatter is special: extended JSON may be sparse for it; supplement
    # with splatter_marker_verification.json's splatter_rank_per_marker if
    # available.
    rows_display: list[str] = []
    rows_json_keys: list[str] = []
    for display, json_key in _NINE_CT_ORDER:
        if json_key in extended:
            rows_display.append(display)
            rows_json_keys.append(json_key)
    if not rows_display:
        # Defensive: fall back to whatever keys exist.
        rows_display = list(extended.keys())[:9]
        rows_json_keys = rows_display[:]

    # Collect all marker genes across all CTs (preserve declaration order
    # within each CT, dedupe across CTs).
    all_markers: list[str] = []
    seen: set[str] = set()
    for json_key in rows_json_keys:
        ct_block = extended.get(json_key, {})
        markers_dict = ct_block.get("markers", {})
        for m in markers_dict.keys():
            if m not in seen:
                seen.add(m)
                all_markers.append(m)
    # If splatter has extra markers not in extended (SST/CHODL/NPY/NOS1 etc.),
    # include them too — Splatter row.
    if splatter is not None:
        for m in splatter.get("splatter_rank_per_marker", {}):
            if m not in seen:
                seen.add(m)
                all_markers.append(m)

    n_rows = len(rows_display)
    n_cols = len(all_markers)
    rank = np.full((n_rows, n_cols), np.nan, dtype=np.float64)

    for i, json_key in enumerate(rows_json_keys):
        ct_block = extended.get(json_key, {})
        markers_dict = ct_block.get("markers", {})
        for m, info in markers_dict.items():
            if m in all_markers:
                j = all_markers.index(m)
                tr = info.get("target_rank")
                if tr is not None:
                    rank[i, j] = float(tr)

    # Splatter row supplementation from splatter_marker_verification.json.
    if splatter is not None:
        try:
            splat_row_idx = rows_display.index("Splatter")
        except ValueError:
            splat_row_idx = -1
        if splat_row_idx >= 0:
            spr = splatter.get("splatter_rank_per_marker", {})
            for m, r in spr.items():
                if m in all_markers:
                    j = all_markers.index(m)
                    if np.isnan(rank[splat_row_idx, j]):
                        rank[splat_row_idx, j] = float(r)

    return rows_display, all_markers, rank


def build_slot2_marker_validation(
    *,
    extended_marker_json: Path,
    splatter_marker_json: Path | None = None,
) -> plt.Figure:
    """Build the 9-CT × marker rank heatmap for slot 2."""
    extended_marker_json = Path(extended_marker_json)
    extended = json.loads(extended_marker_json.read_text())
    splatter = None
    if splatter_marker_json is not None:
        splatter_marker_json = Path(splatter_marker_json)
        if splatter_marker_json.exists():
            splatter = json.loads(splatter_marker_json.read_text())

    rows, cols, rank = _collect_marker_rank_table(extended, splatter)

    n_rows, n_cols = rank.shape
    # Square / 4:3 figure, scale modestly with col count.
    fig_w = max(10.0, 0.55 * n_cols + 4.0)
    fig_h = max(6.0, 0.55 * n_rows + 3.0)
    fig, ax = plt.subplots(figsize=(min(fig_w, 14.0), min(fig_h, 10.0)))

    # Plot — lower rank = better → use reversed viridis (yellow = good).
    cmap = plt.get_cmap("viridis_r").copy()
    cmap.set_bad(color="#dddddd")
    masked = np.ma.array(rank, mask=np.isnan(rank))
    # Cap colorbar at rank 10 so rank 1 stays distinct from rank ≥ 10.
    vmax = 10.0
    im = ax.imshow(masked, aspect="auto", cmap=cmap,
                   vmin=1.0, vmax=vmax, interpolation="nearest")

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(rows)
    ax.set_xlabel("canonical marker gene")
    ax.set_ylabel("cell type (atlas label)")

    # Annotate cells with their rank value (skip NaN).
    for i in range(n_rows):
        for j in range(n_cols):
            v = rank[i, j]
            if np.isnan(v):
                continue
            color = "white" if v <= 4 else "black"
            ax.text(j, i, f"{int(v)}", ha="center", va="center",
                    fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(f"rank in atlas (1 = highest, ≥{int(vmax)} clipped)")
    cbar.ax.invert_yaxis()  # rank 1 at top so "best = top of bar".

    # Disable internal grid (it conflicts visually with imshow cells).
    ax.grid(False)

    fig.tight_layout()
    return fig


# ===========================================================================
# Slot 6: methodology recap block diagram
# ===========================================================================


def _block(ax, x, y, w, h, text, *, fc="#dceaf5", ec="#1f4e79", fontsize=8):
    """Draw a rounded box with centered text."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=0.9, edgecolor=ec, facecolor=fc, zorder=2,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fontsize, color="#222222",
            wrap=True, zorder=3)


def _arrow(ax, x0, y0, x1, y1, *, color="#444444"):
    arr = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="-|>",
        mutation_scale=10,
        color=color,
        linewidth=1.0,
        zorder=2,
    )
    ax.add_patch(arr)


def build_slot6_methods_recap() -> plt.Figure:
    """Build the methodology-recap block diagram for slot 6.

    Three rows:
      Row 1 — pipeline: Data → 5-fold CV → ResDec-MHE encoder → composite pred
      Row 2 — statistical rigor stack
      Row 3 — interpretability paradigms (8+ methods grouped by family)
    """
    fig, ax = plt.subplots(figsize=(12.0, 9.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    ax.axis("off")
    fig.suptitle("", fontsize=1)  # placeholder, no in-figure title

    # =================================================================
    # Row 1 — pipeline (top)
    # =================================================================
    row1_y = 0.84
    row1_h = 0.10
    row1_blocks = [
        ("snRNA-seq\nN=516, ROSMAP", 0.02, 0.16, "#fde9d9"),
        ("5-fold CV\n(seed 42)",     0.21, 0.13, "#fde9d9"),
        ("ResDec-MHE\nencoder + FiLM\n+ TabPFN residual",  0.37, 0.20, "#fde9d9"),
        ("composite\nprediction\n$\\hat y$",            0.60, 0.13, "#fde9d9"),
        ("R² = 0.4436 ± 0.10\nbeats TabPFN, XGB",     0.76, 0.22, "#d4eee0"),
    ]
    centers_row1 = []
    for text, x, w, fc in row1_blocks:
        _block(ax, x, row1_y, w, row1_h, text, fc=fc)
        centers_row1.append((x + w, row1_y + row1_h / 2, x))
    # arrows between row-1 blocks
    for i in range(len(centers_row1) - 1):
        x_end_left, y_mid, _ = centers_row1[i]
        _, _, x_start_right = centers_row1[i + 1]
        _arrow(ax, x_end_left, y_mid, x_start_right, y_mid)

    ax.text(0.50, row1_y + row1_h + 0.025, "Pipeline",
            ha="center", va="bottom", fontsize=10, fontweight="bold")

    # =================================================================
    # Row 2 — statistical rigor stack (middle)
    # =================================================================
    row2_y = 0.50
    row2_h = 0.10
    row2_blocks = [
        ("Paired Wilcoxon\n5/5 wins, p=0.0312",         0.02, 0.18, "#e8eaf6"),
        ("Stouffer combined\np = 2.9e-5",               0.22, 0.16, "#e8eaf6"),
        ("Permutation null\nN=10, z = 8.73",            0.40, 0.16, "#e8eaf6"),
        ("Bootstrap CI\n(B = 1000)",                    0.58, 0.14, "#e8eaf6"),
        ("KSG-CMI CI\nPolitis–Romano subsample",        0.74, 0.24, "#e8eaf6"),
    ]
    centers_row2 = []
    for text, x, w, fc in row2_blocks:
        _block(ax, x, row2_y, w, row2_h, text, fc=fc)
        centers_row2.append((x + w, row2_y + row2_h / 2, x))
    for i in range(len(centers_row2) - 1):
        x_end, y_mid, _ = centers_row2[i]
        _, _, x_start = centers_row2[i + 1]
        _arrow(ax, x_end, y_mid, x_start, y_mid)

    ax.text(0.50, row2_y + row2_h + 0.025, "Statistical-rigor stack",
            ha="center", va="bottom", fontsize=10, fontweight="bold")

    # connector from row 1 → row 2 (column-aligned drop)
    _arrow(ax, 0.50, row1_y, 0.50, row2_y + row2_h)

    # =================================================================
    # Row 3 — interpretability paradigms (bottom)
    # =================================================================
    # Two sub-rows (3a + 3b) inside the bottom block.
    row3a_y = 0.26
    row3b_y = 0.10
    row3_h = 0.10
    paradigm_blocks_a = [
        ("Gradient attribution\n(Captum IG / GradSHAP\n/ SmoothGrad) ×3",  0.02, 0.24, "#fff2cc"),
        ("Attention attribution\n(AttnLRP / GMAR / GAF×3) ×5",            0.28, 0.24, "#fff2cc"),
        ("Distance\n(Wasserstein-1)",                                     0.54, 0.16, "#fff2cc"),
        ("Mutual info\n(KSG-CMI)",                                        0.72, 0.14, "#fff2cc"),
        ("Ablation\n(LOCO zero-out)",                                     0.88, 0.10, "#fff2cc"),
    ]
    paradigm_blocks_b = [
        ("Counterfactual\n(Wachter Mode-A literal)",                      0.02, 0.26, "#fff2cc"),
        ("Sparse autoencoder\n(Orlov 2026 features)",                     0.30, 0.24, "#fff2cc"),
        ("Conditional MI per CT\n(raw pseudobulk, n=516)",                0.56, 0.26, "#fff2cc"),
        ("DE / pseudobulk\n(Wilcoxon + DESeq2)",                          0.84, 0.14, "#fff2cc"),
    ]
    for text, x, w, fc in paradigm_blocks_a:
        _block(ax, x, row3a_y, w, row3_h, text, fc=fc, fontsize=7)
    for text, x, w, fc in paradigm_blocks_b:
        _block(ax, x, row3b_y, w, row3_h, text, fc=fc, fontsize=7)

    ax.text(0.50, row3a_y + row3_h + 0.025,
            "Interpretability paradigms (10+ methods, 6-way Splatter convergence)",
            ha="center", va="bottom", fontsize=10, fontweight="bold")

    # connector from row 2 → row 3 (mid drop)
    _arrow(ax, 0.50, row2_y, 0.50, row3a_y + row3_h)

    # No grid / no axis frame.
    fmt_axes(ax, hide_spines=("top", "right", "bottom", "left"),
             grid_major=False, grid_minor=False)
    ax.set_xticks([])
    ax.set_yticks([])

    return fig


# ===========================================================================
# Top-level orchestrator
# ===========================================================================


def build_all_figures(
    *,
    canonical_dir: Path,
    extended_marker_json: Path,
    splatter_marker_json: Path | None,
    out_dir: Path,
    n_folds: int = 5,
) -> dict[str, list[Path]]:
    """Build + save the 3 lab-meeting slot-anchor figures.

    Returns
    -------
    dict
        Mapping ``slot_name -> [paths_written]`` for each of the 3 figures.
    """
    canonical_dir = Path(canonical_dir)
    extended_marker_json = Path(extended_marker_json)
    if splatter_marker_json is not None:
        splatter_marker_json = Path(splatter_marker_json)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    apply_theme()

    written: dict[str, list[Path]] = {}

    # Slot 1
    fig1 = build_slot1_residual_definition(
        canonical_dir=canonical_dir, n_folds=n_folds,
    )
    written["slot1"] = save_fig(
        fig1, out_dir / "fig_slot1_residual_definition", dpi=300,
    )
    plt.close(fig1)

    # Slot 2
    fig2 = build_slot2_marker_validation(
        extended_marker_json=extended_marker_json,
        splatter_marker_json=splatter_marker_json,
    )
    written["slot2"] = save_fig(
        fig2, out_dir / "fig_slot2_marker_validation", dpi=300,
    )
    plt.close(fig2)

    # Slot 6
    fig3 = build_slot6_methods_recap()
    written["slot6"] = save_fig(
        fig3, out_dir / "fig_slot6_methods_recap", dpi=300,
    )
    plt.close(fig3)

    for slot_name, paths in written.items():
        for p in paths:
            logger.info("wrote %s (%.1f KB)", p, p.stat().st_size / 1024.0)

    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--canonical-dir", default=_DEFAULT_CANONICAL_DIR)
    parser.add_argument("--extended-marker-json",
                        default=_DEFAULT_EXTENDED_MARKER_JSON)
    parser.add_argument("--splatter-marker-json",
                        default=_DEFAULT_SPLATTER_MARKER_JSON)
    parser.add_argument("--out-dir", default=_DEFAULT_OUT_DIR)
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    build_all_figures(
        canonical_dir=Path(args.canonical_dir),
        extended_marker_json=Path(args.extended_marker_json),
        splatter_marker_json=Path(args.splatter_marker_json),
        out_dir=Path(args.out_dir),
        n_folds=args.n_folds,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
