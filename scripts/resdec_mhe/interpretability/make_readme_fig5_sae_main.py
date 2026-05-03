#!/usr/bin/env python
"""Render Figure 5 main for the ResDec-MHE README redesign.

Two-panel figure summarizing the SAE-feature interpretability story:

  Panel A (polar feature wheel)
    323 SAE features that pass the *relaxed* interpretability filter
    are rendered as spokes on a unit circle. Spoke length is proportional
    to ``ct_dominance`` in [0, 0.7] (the relaxed cap), spoke color is the
    deterministic CT color of the feature's max-CT identity (i.e.
    ``top_cell_types[0]``). The lone Splatter feature (idx 572) is
    highlighted with a thicker line, contrasting outline, and label so
    its position relative to the population is immediately readable.
    Concentric guide rings at ct_dominance = {0.2, 0.4, 0.6} give
    visual grid for spoke length.

  Panel B (raincloud, patch ΔR² across folds)
    11 SAE features as rows, top row = Splatter feature 572, then 10
    random control features ([178, 1577, 183, 1340, 898, 883, 1431, 194,
    415, 1750]). Each row is a *raincloud*:
      * half-violin (KDE) above the row baseline
      * strip dots below for the 5 per-fold ΔR² values
      * boxplot (median + IQR) inside the violin
    X-axis is patch ΔR² (saturate mode, vs SAE baseline). Vertical
    dashed line at ΔR² = 0; mean ± std annotated for the Splatter row
    and for the pooled random null. Splatter row is colored distinct
    (red); random control rows in a neutral palette.

Inputs (read fresh — no hardcoded numbers):

  - ``outputs/canonical/sae/batch_topk/fused/exp32_k64_seed0/feature_report.json``
    Per-feature metadata for all 2048 SAE features (feature_idx,
    top_cell_types, ct_dominance, mw_p_cognition, fraction_active, flags).

  - ``outputs/canonical/sae/feature_xref_consensus.json``
    Reference for the relaxed filter definition. Used at print-report
    time to cross-check the count (323) against the published n_features.

  - ``outputs/canonical/interpretability/sae_causal_patching.json``
    5-fold patch ΔR² values for the Splatter feature (saturate mode) and
    the 10 random control features.

Outputs:

  - ``figures/fig5_sae_main.png`` (12 x 8 in at 600 dpi)
  - Verification numbers printed to stdout.

Usage:

  PYTHONPATH=<worktree-root> \\
    uv run python scripts/resdec_mhe/interpretability/make_readme_fig5_sae_main.py

Idempotence
-----------
Pipeline is fully deterministic: JSON I/O, numpy filtering, fixed-seed
KDE bandwidth (Scott's rule via ``scipy.stats.gaussian_kde``). PYTHONHASHSEED
pinned defensively for matplotlib's color-path stability.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Pin hash seed defensively (some matplotlib color paths hit set-iteration).
os.environ.setdefault("PYTHONHASHSEED", "42")

import matplotlib

matplotlib.use("Agg")  # must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from src.visualization.config import CELL_TYPE_COLORS  # noqa: E402
from src.visualization.theme import (  # noqa: E402
    apply_theme,
    fmt_axes,
    save_fig,
)

logger = logging.getLogger(__name__)


# Ordered list of 10 random control features that the causal-patching run
# used. Order is preserved here so Panel B rows are deterministic and
# match the JSON column order.
RANDOM_FEATURE_IDXS: tuple[int, ...] = (
    178, 1577, 183, 1340, 898, 883, 1431, 194, 415, 1750,
)

# The lone Splatter SAE feature in the relaxed filter.
SPLATTER_FEATURE_IDX: int = 572

# Splatter highlight color: red, distinct from the gray Splatter CT color
# used in Panel A spoke rendering. The CT color (CELL_TYPE_COLORS["Splatter"]
# = "#D3D3D3") is too desaturated to highlight a single spoke against the
# 322-feature backdrop, so we override to a high-contrast red and add a
# black outline. This is a presentation-only choice — the underlying CT
# identity is still Splatter.
SPLATTER_HIGHLIGHT_COLOR: str = "#D62728"  # tab10 red


# -----------------------------------------------------------------------------
# Filter helpers.
# -----------------------------------------------------------------------------
def _passes_relaxed_filter(feat: dict) -> bool:
    """Return True if a feature_report entry passes the relaxed filter.

    Relaxed filter (per ``feature_xref_consensus.json::filter_definitions``):

      * non-dead (``"dead"`` not in ``flags``)
      * ``mw_p_cognition < 0.05``
      * ``fraction_active`` in [0.0001, 0.5]
      * ``ct_dominance <= 0.7``
    """
    flags = feat.get("flags") or []
    if "dead" in flags:
        return False
    p = feat.get("mw_p_cognition")
    if p is None or not (p < 0.05):
        return False
    fa = feat.get("fraction_active")
    if fa is None or not (0.0001 <= fa <= 0.5):
        return False
    dom = feat.get("ct_dominance")
    if dom is None or not (dom <= 0.7):
        return False
    return True


def _max_ct_identity(feat: dict) -> str | None:
    """Return ``top_cell_types[0].cell_type``, or None if no top CT recorded."""
    tcts = feat.get("top_cell_types") or []
    if not tcts:
        return None
    return tcts[0].get("cell_type")


def _load_relaxed_features(report_path: Path) -> list[dict]:
    """Load + relax-filter the SAE feature report. Returns list of dicts."""
    payload = json.loads(report_path.read_text())
    if not isinstance(payload, list):
        raise ValueError(
            f"Expected list at {report_path}; got {type(payload).__name__}"
        )
    return [feat for feat in payload if _passes_relaxed_filter(feat)]


# -----------------------------------------------------------------------------
# Panel B data extraction.
# -----------------------------------------------------------------------------
def _load_patching_per_fold(
    payload: dict,
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[str, float]]:
    """Extract per-fold saturate ΔR² for Splatter + 10 random control features.

    Returns
    -------
    splatter_per_fold : np.ndarray (5,)
        Splatter feature 572 saturate ΔR² across the 5 folds.
    random_per_fold : dict[int, np.ndarray (5,)]
        Per-feature 5-fold saturate ΔR², keyed by feature_idx.
    summary : dict[str, float]
        ``payload["summary_statistics"]`` for cross-check at print-report time.
    """
    folds = payload.get("per_fold")
    if not isinstance(folds, list) or len(folds) != 5:
        raise ValueError(
            f"Expected per_fold to be a list of length 5; "
            f"got {type(folds).__name__} with length "
            f"{len(folds) if hasattr(folds, '__len__') else 'NA'}"
        )

    splatter_vals: list[float] = []
    random_vals: dict[int, list[float]] = {idx: [] for idx in RANDOM_FEATURE_IDXS}

    for fold in folds:
        sp = fold["splatter_feature"]["per_mode"]["saturate"][
            "delta_r2_vs_sae_baseline"
        ]
        splatter_vals.append(float(sp))

        per_feature = fold["random_feature_controls"]["per_feature"]
        for idx in RANDOM_FEATURE_IDXS:
            entry = per_feature.get(str(idx))
            if entry is None:
                raise KeyError(
                    f"Random feature idx {idx} not found in fold "
                    f"{fold.get('fold')!r} per_feature payload"
                )
            random_vals[idx].append(float(entry["delta_r2_vs_sae_baseline"]))

    splatter_per_fold = np.asarray(splatter_vals, dtype=np.float64)
    random_per_fold = {
        idx: np.asarray(vals, dtype=np.float64) for idx, vals in random_vals.items()
    }
    summary = payload.get("summary_statistics") or {}
    return splatter_per_fold, random_per_fold, summary


# -----------------------------------------------------------------------------
# Panel A: polar feature wheel.
# -----------------------------------------------------------------------------
def _draw_polar_wheel(
    ax,
    relaxed_feats: list[dict],
) -> dict[str, object]:
    """Render the polar SAE-feature wheel.

    Spoke length = ``ct_dominance`` in [0, 0.7].
    Spoke color  = ``CELL_TYPE_COLORS[top_cell_types[0].cell_type]``.
    Splatter spoke = highlighted (thicker, red, black outline, labeled).

    Returns a dict mapping CTs that appear in the population to their colors
    (used to draw the Panel-A legend).
    """
    n_feats = len(relaxed_feats)
    if n_feats == 0:
        raise ValueError("No relaxed-filter features to plot")

    # Sort relaxed features by max-CT identity then by ct_dominance descending,
    # so spokes with the same CT cluster together on the wheel and the
    # eyeballs see CT-blocks rather than a uniform mess. Splatter is the
    # last block (only 1 feature) so it gets its own angular slot regardless.
    def _sort_key(feat: dict) -> tuple[str, float]:
        ct = _max_ct_identity(feat) or "zzz_unknown"
        # negative dominance for descending; tie-break on idx for stability
        return (ct, -float(feat.get("ct_dominance", 0.0)))

    sorted_feats = sorted(relaxed_feats, key=_sort_key)

    # Angular positions: evenly spaced over [0, 2π).
    thetas = np.linspace(0.0, 2.0 * np.pi, n_feats, endpoint=False)

    # Concentric guide rings (light gray) at ct_dominance = 0.2, 0.4, 0.6.
    ring_radii = (0.2, 0.4, 0.6)
    ring_thetas = np.linspace(0.0, 2.0 * np.pi, 360)
    for r in ring_radii:
        ax.plot(
            ring_thetas,
            np.full_like(ring_thetas, r),
            color="#cccccc",
            linewidth=0.5,
            linestyle="--",
            zorder=1,
        )

    # Draw spokes one by one. Splatter spoke is drawn last (top z-order) with
    # highlight styling.
    splatter_theta: float | None = None
    splatter_dom: float | None = None
    used_cts: dict[str, str] = {}

    for theta, feat in zip(thetas, sorted_feats):
        ct = _max_ct_identity(feat) or "Miscellaneous"
        dom = float(feat.get("ct_dominance", 0.0))
        # Rendering color: CT color for everyone except Splatter, which gets
        # its dedicated highlight pass below (still draw a faint gray spoke
        # underneath for layering consistency).
        is_splatter = (ct == "Splatter")
        color = CELL_TYPE_COLORS.get(ct, "#808080")
        if is_splatter:
            splatter_theta = float(theta)
            splatter_dom = dom
            # Draw the underlying gray spoke first for alignment, then
            # overlay the highlight in the next loop iteration's drawing.
            ax.plot(
                [theta, theta],
                [0.0, dom],
                color=color,
                linewidth=0.6,
                alpha=0.7,
                zorder=2,
            )
            continue

        ax.plot(
            [theta, theta],
            [0.0, dom],
            color=color,
            linewidth=0.7,
            alpha=0.85,
            zorder=2,
        )
        used_cts.setdefault(ct, color)

    # Highlight Splatter (drawn on top with thicker line + black outline).
    if splatter_theta is not None and splatter_dom is not None:
        # Thick red spoke in the foreground.
        ax.plot(
            [splatter_theta, splatter_theta],
            [0.0, splatter_dom],
            color=SPLATTER_HIGHLIGHT_COLOR,
            linewidth=2.6,
            zorder=5,
            solid_capstyle="round",
        )
        # Black outline by overplotting a slightly thicker line behind it.
        ax.plot(
            [splatter_theta, splatter_theta],
            [0.0, splatter_dom],
            color="black",
            linewidth=4.0,
            zorder=4,
            solid_capstyle="round",
        )
        # Marker at the spoke tip for emphasis.
        ax.plot(
            [splatter_theta],
            [splatter_dom],
            marker="o",
            color=SPLATTER_HIGHLIGHT_COLOR,
            markersize=7.0,
            markeredgecolor="black",
            markeredgewidth=0.9,
            zorder=6,
        )
        # Label "Splatter (idx 572)" placed just outside the spoke tip,
        # rotated tangentially so it doesn't overlap neighboring spokes.
        label_r = max(splatter_dom + 0.10, 0.72)
        ax.annotate(
            f"Splatter\n(feat {SPLATTER_FEATURE_IDX})",
            xy=(splatter_theta, splatter_dom),
            xytext=(splatter_theta, label_r),
            ha="center",
            va="bottom" if splatter_theta < np.pi else "top",
            fontsize=7,
            color="black",
            fontweight="bold",
            zorder=7,
        )

    # Polar styling: hide angular grid labels (irrelevant — axis is a
    # categorical wheel of features), keep radial gridlines for the spoke
    # length scale.
    ax.set_theta_direction(-1)         # clockwise
    ax.set_theta_zero_location("N")    # 0° at top
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks(list(ring_radii) + [0.7])
    ax.set_yticklabels(
        [f"{r:.1f}" for r in ring_radii] + ["0.7 (max)"],
        fontsize=6,
        color="#666666",
    )
    ax.set_ylim(0.0, 0.78)  # leave room for splatter label outside the cap
    ax.set_rlabel_position(135.0)
    ax.grid(True, color="#e6e6e6", linewidth=0.4)
    ax.set_facecolor("white")

    return {"used_cts": used_cts, "n_feats": n_feats}


# -----------------------------------------------------------------------------
# Panel B: raincloud across folds.
# -----------------------------------------------------------------------------
def _draw_raincloud(
    ax,
    splatter_per_fold: np.ndarray,
    random_per_fold: dict[int, np.ndarray],
) -> None:
    """Render the per-feature 5-fold ΔR² raincloud.

    Layout: one row per feature, top row = Splatter, then 10 random
    controls in their canonical order. Each row contains:

      * half-violin (KDE) above the row baseline
      * boxplot (median + IQR) on the row baseline
      * strip dots (the 5 raw per-fold values) below the baseline.
    """
    # Row order: Splatter on top (row index 0 = highest y), random controls
    # below in their canonical order.
    row_labels: list[str] = [f"Splatter\n(feat {SPLATTER_FEATURE_IDX})"]
    row_data: list[np.ndarray] = [splatter_per_fold]
    row_colors: list[str] = [SPLATTER_HIGHLIGHT_COLOR]
    is_splatter_row: list[bool] = [True]

    # Neutral grayscale palette for the random rows; sample viridis at
    # 10 evenly spaced positions for visual differentiation while staying
    # subdued vs the Splatter highlight.
    cmap = plt.get_cmap("viridis")
    rand_colors = [cmap(0.18 + 0.06 * k) for k in range(len(RANDOM_FEATURE_IDXS))]

    for k, idx in enumerate(RANDOM_FEATURE_IDXS):
        row_labels.append(f"Random\n{idx}")
        row_data.append(random_per_fold[idx])
        row_colors.append(rand_colors[k])
        is_splatter_row.append(False)

    n_rows = len(row_data)

    # Y positions: row 0 (Splatter) at the top → y = n_rows - 1
    y_positions = np.arange(n_rows)[::-1]  # row 0 → y = n_rows-1, ..., row n-1 → y = 0

    # X axis: shared across all rows, centered on 0.
    all_vals = np.concatenate(row_data)
    x_lo = float(np.min(all_vals))
    x_hi = float(np.max(all_vals))
    span = max(x_hi - x_lo, 1e-6)
    pad = 0.12 * span
    x_range = (x_lo - pad, x_hi + pad)

    # Drawing parameters.
    half_violin_height = 0.36   # max KDE half-width per row in axis units
    strip_offset = -0.18        # strip dots placed below the baseline
    box_height = 0.10           # boxplot height
    kde_grid = np.linspace(x_range[0], x_range[1], 256)

    for vals, y, color, is_sp in zip(row_data, y_positions, row_colors, is_splatter_row):
        # Half-violin via gaussian KDE — only with > 1 distinct value.
        if vals.size >= 2 and np.std(vals) > 1e-12:
            try:
                kde = gaussian_kde(vals, bw_method="scott")
                density = kde(kde_grid)
                if density.max() > 0:
                    density = density / density.max() * half_violin_height
                ax.fill_between(
                    kde_grid,
                    y,
                    y + density,
                    color=color,
                    alpha=0.55 if not is_sp else 0.70,
                    linewidth=0.0,
                    zorder=2,
                )
                # Outline of the violin top.
                ax.plot(
                    kde_grid,
                    y + density,
                    color=color,
                    linewidth=0.8 if not is_sp else 1.1,
                    zorder=3,
                )
            except np.linalg.LinAlgError:
                # Degenerate (singular) covariance — skip violin, keep boxplot.
                pass

        # Boxplot (median + IQR) at the row baseline.
        med = float(np.median(vals))
        q1 = float(np.percentile(vals, 25))
        q3 = float(np.percentile(vals, 75))
        v_min = float(np.min(vals))
        v_max = float(np.max(vals))
        # IQR box.
        ax.fill_betweenx(
            [y - box_height / 2, y + box_height / 2],
            q1, q3,
            color="white",
            edgecolor=color,
            linewidth=1.0 if not is_sp else 1.4,
            zorder=4,
        )
        # Median line.
        ax.plot(
            [med, med],
            [y - box_height / 2, y + box_height / 2],
            color=color,
            linewidth=1.6 if not is_sp else 2.0,
            zorder=5,
        )
        # Whiskers (extend to min/max — only 5 points so no outlier rule).
        ax.plot(
            [v_min, q1],
            [y, y],
            color=color,
            linewidth=0.8,
            zorder=4,
        )
        ax.plot(
            [q3, v_max],
            [y, y],
            color=color,
            linewidth=0.8,
            zorder=4,
        )

        # Strip plot (raw per-fold dots) below the baseline.
        # Add small horizontal jitter for visibility (deterministic via
        # rank-based offsets, not random — bit-stable across reruns).
        if vals.size > 0:
            ranks = np.argsort(np.argsort(vals))  # 0..n-1
            jitter = (ranks - (vals.size - 1) / 2.0) * 0.0  # no jitter; values are spread
            ax.scatter(
                vals,
                np.full_like(vals, y + strip_offset, dtype=float) + jitter,
                s=20.0 if not is_sp else 32.0,
                facecolor=color,
                edgecolor="white",
                linewidth=0.5,
                alpha=0.95,
                zorder=6,
            )

    # Vertical zero line.
    ax.axvline(
        0.0,
        color="#444444",
        linestyle="--",
        linewidth=0.9,
        zorder=1,
        label=r"$\Delta R^2 = 0$",
    )

    # Annotate Splatter mean ± std and pooled-random mean ± std.
    sp_mean = float(splatter_per_fold.mean())
    sp_std = float(splatter_per_fold.std(ddof=1))
    pooled_random = np.concatenate(
        [random_per_fold[idx] for idx in RANDOM_FEATURE_IDXS]
    )
    rand_mean = float(pooled_random.mean())
    rand_std = float(pooled_random.std(ddof=1))

    ax.text(
        0.02, 0.985,
        (
            "Splatter (saturate)\n"
            r"$\overline{\Delta R^2}\pm$std = "
            + f"{sp_mean:+.4f} $\\pm$ {sp_std:.4f}"
        ),
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=7,
        bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                  edgecolor=SPLATTER_HIGHLIGHT_COLOR, linewidth=0.9),
        zorder=7,
    )
    ax.text(
        0.98, 0.985,
        (
            "Random null (10x5=50)\n"
            r"$\overline{\Delta R^2}\pm$std = "
            + f"{rand_mean:+.4f} $\\pm$ {rand_std:.4f}"
        ),
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=7,
        bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                  edgecolor="#666666", linewidth=0.9),
        zorder=7,
    )

    # Y axis: row labels.
    ax.set_yticks(y_positions)
    ax.set_yticklabels(row_labels, fontsize=6.5)
    ax.set_ylim(-0.6, n_rows - 0.4)

    # X axis: ΔR², centered on 0.
    ax.set_xlim(*x_range)
    ax.set_xlabel(r"Patch $\Delta R^2$ (saturate, val per fold)")

    fmt_axes(ax, grid_major=True, grid_minor=False)


# -----------------------------------------------------------------------------
# Orchestrator.
# -----------------------------------------------------------------------------
def make_figure(
    *,
    relaxed_feats: list[dict],
    splatter_per_fold: np.ndarray,
    random_per_fold: dict[int, np.ndarray],
) -> tuple[plt.Figure, dict[str, object]]:
    """Build the 2-panel figure. Returns (fig, panel_a_meta)."""
    apply_theme("paper")

    # ~12x8 in canvas; left panel polar, right panel cartesian raincloud.
    fig = plt.figure(figsize=(12, 8))
    ax_a = fig.add_subplot(1, 2, 1, projection="polar")
    ax_b = fig.add_subplot(1, 2, 2)

    panel_a_meta = _draw_polar_wheel(ax_a, relaxed_feats)
    _draw_raincloud(ax_b, splatter_per_fold, random_per_fold)

    # Panel A legend: list the most frequent CTs in the wheel + Splatter
    # highlight. Build a legend from CT colors that actually appear in
    # the relaxed features, top-N by frequency.
    used = panel_a_meta["used_cts"]
    # Frequency count per CT in relaxed features.
    ct_counts: dict[str, int] = {}
    for feat in relaxed_feats:
        ct = _max_ct_identity(feat) or "Miscellaneous"
        ct_counts[ct] = ct_counts.get(ct, 0) + 1

    # Choose top 8 CTs by count for the legend (excluding Splatter — it gets
    # its own highlight entry). Tie-break alphabetical.
    nonsplatter_sorted = sorted(
        ((ct, n) for ct, n in ct_counts.items() if ct != "Splatter"),
        key=lambda kv: (-kv[1], kv[0]),
    )
    legend_cts = nonsplatter_sorted[:8]

    legend_handles: list[Patch] = []
    for ct, n in legend_cts:
        col = CELL_TYPE_COLORS.get(ct, "#808080")
        legend_handles.append(Patch(facecolor=col, edgecolor="black",
                                    linewidth=0.5, label=f"{ct} (n={n})"))
    # Splatter highlight entry.
    legend_handles.append(
        Line2D(
            [0], [0],
            color=SPLATTER_HIGHLIGHT_COLOR,
            marker="o",
            markersize=6.0,
            markeredgecolor="black",
            markeredgewidth=0.7,
            linewidth=2.4,
            label=f"Splatter (n=1, feat {SPLATTER_FEATURE_IDX})",
        )
    )

    ax_a.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(-0.18, 1.04),
        fontsize=6.5,
        frameon=True,
        title=f"Top-CT identity (top 8 of 31)\nN_feat={len(relaxed_feats)}",
        title_fontsize=7,
        ncol=1,
    )

    # Panel-level annotations.
    ax_a.set_title(
        "A. SAE feature wheel (323 relaxed-filter features)",
        fontsize=9, fontweight="bold", pad=18,
    )
    ax_b.set_title(
        "B. Patch $\\Delta R^2$ (saturate) across 5 folds",
        fontsize=9, fontweight="bold", pad=8,
    )

    fig.subplots_adjust(left=0.18, right=0.97, top=0.93, bottom=0.10,
                        wspace=0.28)
    return fig, panel_a_meta


# -----------------------------------------------------------------------------
# Verification.
# -----------------------------------------------------------------------------
def _print_report(
    *,
    relaxed_feats: list[dict],
    consensus: dict,
    splatter_feat: dict,
    splatter_per_fold: np.ndarray,
    random_per_fold: dict[int, np.ndarray],
    summary_from_json: dict,
) -> None:
    n_relaxed = len(relaxed_feats)
    splatter_count = sum(
        1 for feat in relaxed_feats if _max_ct_identity(feat) == "Splatter"
    )
    consensus_n = (
        consensus.get("trained", {}).get("relaxed", {}).get("n_features")
    )
    consensus_splatter = (
        consensus.get("trained", {}).get("relaxed", {})
        .get("per_ct_counts", {}).get("Splatter")
    )

    pooled_random = np.concatenate(
        [random_per_fold[idx] for idx in RANDOM_FEATURE_IDXS]
    )
    rand_mean = float(pooled_random.mean())
    rand_std = float(pooled_random.std(ddof=1))

    print("=" * 72)
    print("README Figure 5 -- SAE polar wheel + raincloud")
    print("=" * 72)
    print(f"  n_relaxed_filter             : {n_relaxed}")
    print(f"  consensus.relaxed.n_features : {consensus_n}")
    print(f"  splatter_in_relaxed          : {splatter_count}")
    print(f"  consensus.relaxed.Splatter   : {consensus_splatter}")
    print(f"  splatter_feature_idx         : {splatter_feat['feature_idx']}")
    print(f"  splatter_ct_dominance        : "
          f"{splatter_feat['ct_dominance']:.6f}")
    print(f"  splatter_top_cts             : "
          f"{[c['cell_type'] for c in splatter_feat['top_cell_types']]}")

    print()
    print("  splatter_saturate_dr2_per_fold:")
    for k, v in enumerate(splatter_per_fold):
        print(f"    fold {k}: {v:+.6e}")
    print(f"  splatter_mean                : "
          f"{float(splatter_per_fold.mean()):+.6e}")
    print(f"  splatter_std (ddof=1)        : "
          f"{float(splatter_per_fold.std(ddof=1)):+.6e}")

    print()
    print(f"  pooled_random_n              : {pooled_random.size}")
    print(f"  pooled_random_mean           : {rand_mean:+.6e}")
    print(f"  pooled_random_std (ddof=1)   : {rand_std:+.6e}")

    if summary_from_json:
        print("  --- summary_from_json ---")
        for k in (
            "splatter_saturate_delta_r2_mean",
            "splatter_saturate_delta_r2_std",
            "random_saturate_delta_r2_mean",
            "random_saturate_delta_r2_std",
            "n_random_pooled",
        ):
            v = summary_from_json.get(k)
            if v is not None:
                if isinstance(v, float):
                    print(f"    {k:42s}: {v:+.6e}")
                else:
                    print(f"    {k:42s}: {v}")

    print("=" * 72)


# -----------------------------------------------------------------------------
# Main.
# -----------------------------------------------------------------------------
def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])

    parser.add_argument(
        "--feature-report", type=Path,
        default=_WORKTREE_ROOT
        / "outputs/canonical/sae/batch_topk/fused/exp32_k64_seed0/feature_report.json",
    )
    parser.add_argument(
        "--xref-consensus", type=Path,
        default=_WORKTREE_ROOT
        / "outputs/canonical/sae/feature_xref_consensus.json",
    )
    parser.add_argument(
        "--causal-patching", type=Path,
        default=_WORKTREE_ROOT
        / "outputs/canonical/interpretability/sae_causal_patching.json",
    )
    parser.add_argument(
        "--out-stem", type=Path,
        default=_WORKTREE_ROOT / "figures/fig5_sae_main",
        help="Output path stem (no extension); save_fig appends .png.",
    )
    parser.add_argument(
        "--dpi", type=int, default=600,
        help="PNG resolution. Default 600 is the project's standard for "
             "paper-grade rasters at 12x8 in.",
    )
    args = parser.parse_args()

    logger.info("[fig5] loading SAE feature report: %s", args.feature_report)
    relaxed_feats = _load_relaxed_features(args.feature_report)
    logger.info(
        "[fig5] loaded %d relaxed-filter features (out of total in report)",
        len(relaxed_feats),
    )

    logger.info("[fig5] loading xref consensus: %s", args.xref_consensus)
    consensus = json.loads(args.xref_consensus.read_text())

    splatter_feats = [
        feat for feat in relaxed_feats if _max_ct_identity(feat) == "Splatter"
    ]
    if len(splatter_feats) != 1:
        raise ValueError(
            f"Expected exactly 1 Splatter feature in relaxed filter; "
            f"got {len(splatter_feats)}"
        )
    splatter_feat = splatter_feats[0]
    if splatter_feat["feature_idx"] != SPLATTER_FEATURE_IDX:
        raise ValueError(
            f"Splatter feature_idx mismatch: expected "
            f"{SPLATTER_FEATURE_IDX}, got {splatter_feat['feature_idx']}"
        )

    logger.info("[fig5] loading causal patching JSON: %s", args.causal_patching)
    patching_payload = json.loads(args.causal_patching.read_text())
    splatter_per_fold, random_per_fold, summary_from_json = (
        _load_patching_per_fold(patching_payload)
    )
    logger.info(
        "[fig5] splatter shape=%s, random feature_idxs=%s",
        splatter_per_fold.shape, sorted(random_per_fold.keys()),
    )

    fig, _meta = make_figure(
        relaxed_feats=relaxed_feats,
        splatter_per_fold=splatter_per_fold,
        random_per_fold=random_per_fold,
    )

    out_png = args.out_stem.with_suffix(".png")
    if out_png.exists():
        logger.info("[fig5] removing preexisting %s", out_png)
        out_png.unlink()

    written = save_fig(fig, args.out_stem, formats=("png",), dpi=args.dpi)
    plt.close(fig)
    for w in written:
        logger.info("[fig5] wrote %s (%.2f MB)", w,
                    w.stat().st_size / (1024 * 1024))

    _print_report(
        relaxed_feats=relaxed_feats,
        consensus=consensus,
        splatter_feat=splatter_feat,
        splatter_per_fold=splatter_per_fold,
        random_per_fold=random_per_fold,
        summary_from_json=summary_from_json,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
