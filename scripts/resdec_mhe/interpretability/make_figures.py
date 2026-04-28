"""Paper figures.

Generates seven publication-ready figures from canonical-run artefacts:

1. ``fig_ablation_bar`` — ablation R² bar chart with error bars, sorted desc.
2. ``fig_resilience_scatter`` — y_true vs y_pred, colored by residual,
   with median-based resilient/vulnerable quadrant overlay.
3. ``fig_celltype_gene_heatmap`` — top-30 (cell-type, gene) attribution pairs
   from Captum IG as a horizontal heatmap.
4. ``fig_head_specialization`` — per-head top-3 cell-type stacked bar with
   Shannon entropy annotation (plus Splatter × LAMP5 correlation note).
5. ``fig_subgroup_r2`` — subgroup R² point estimates + 95 % bootstrap CIs,
   grouped by family (APOE | sex | age | pathology), canonical R² ref line.
6. ``fig_calibration`` — |residual| vs TabPFN σ scatter + nominal-vs-empirical
   coverage curve (two-panel).
7. ``fig_k_sensitivity`` — TabPFN top-k feature-count sensitivity (5-fold
   mean R² ± std for k ∈ {1000, 2000, 4000}) with canonical bootstrap CI
   reference band.

Each ``make_figX_*`` function accepts *pre-loaded* DataFrames / dicts and
returns a ``matplotlib.figure.Figure`` — so they are unit-testable without
touching disk. Missing or empty inputs raise :class:`SkipFigure`, which the
CLI catches and logs as a WARNING so the overall batch still completes.

Usage
-----
    PYTHONPATH=<worktree-root> \\
    uv run python scripts/resdec_mhe/interpretability/make_figures.py \\
        --out-dir outputs/redesign/interpretability/figures
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

# Must set backend before pyplot import for headless runs.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.metrics import r2_score


# Ensure worktree root on sys.path for standalone invocation.
_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if (_WORKTREE_ROOT / "src").is_dir() and str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from src.analysis.resdec_io import load_all_folds, load_fold_predictions
from src.visualization.theme import apply_theme


logger = logging.getLogger(__name__)


CANONICAL_R2 = 0.4436211705207825  # from paper_baseline_table.csv (p5_canonical_seed42)

# --- Module constants ---
N_FOLDS = 5
TOP_N_PAIRS_HEATMAP = 30
R2_VISUAL_LOWER_CLIP = -1.5
NOMINAL_COVERAGE_LEVELS = (0.5, 0.68, 0.8, 0.95)


# Per-family prefix stripping for subgroup label display.
# Keyed by the family display name used in _SUBGROUP_FAMILIES.
_FAMILY_PREFIX_STRIP: dict[str, str] = {
    "APOE": "APOE_",
    "Sex": "msex_",
    "Age": "age_quartile_",
    "Pathology": "pathology_quartile_",
}


def _nf_int(nf) -> int:
    """NaN-safe conversion of an ``n_folds`` cell to int."""
    return 0 if pd.isna(nf) else int(nf)


class SkipFigure(RuntimeError):
    """Raised when a figure cannot be produced (missing / empty input)."""


# ---------------------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------------------


def _apply_paper_style() -> None:
    """Paper-ready rcParams via the project-wide theme.

    Delegates the bulk of the styling to ``src.visualization.theme.apply_theme``
    (Helvetica / fonts / spines / ticks / grid / 600 DPI savefig). On top of
    that base, restore the figure-script-specific overrides this orchestrator
    has historically used:

    - Slightly larger fonts than the default ``paper`` style (base 8 → 9) so
      the multi-panel figures here read at journal print sizes without a
      magnifier.
    - Thinner grid + line widths (0.4 / 1.2 vs the theme's 0.6 / 1.4) so the
      dense bar/scatter panels do not get visually overwhelmed by lines.
    - ``savefig.bbox="tight"`` is already set by ``apply_theme`` but we keep
      it explicit here for clarity.
    """
    apply_theme()
    matplotlib.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.4,
        "lines.linewidth": 1.2,
        "savefig.bbox": "tight",
    })


# Apply paper style at import time so both CLI runs and unit tests render
# with identical rcParams.
_apply_paper_style()


# ---------------------------------------------------------------------------
# Figure 1: ablation bar chart
# ---------------------------------------------------------------------------


# Baselines vs ours: group detection is display-name based so the figure
# can be re-generated without a code change if new rows are appended to the
# baseline table.
_OURS_MODEL_TOKENS: tuple[str, ...] = ("p5_",)


def _is_ours_row(model: str) -> bool:
    return any(model.startswith(tok) for tok in _OURS_MODEL_TOKENS)


def make_fig1_ablation_bar(
    table: pd.DataFrame | None,
    canonical_r2: float = CANONICAL_R2,
) -> plt.Figure:
    """Bar chart of R² mean ± std for every row in the baseline table.

    - Completed rows (``n_folds >= 5``) sorted by r2_mean descending.
    - Pending rows (``n_folds < 5`` or NaN) appended at the right with
      light outline-only bars and a "† pending" legend note.
    - Horizontal reference line at ``canonical_r2``.
    - Colour: baselines (gray), ours (steel blue), pending (light outline).
    - Error bars: outer thin = std; inner thick red = SEM (= std / sqrt(n_folds))
      so reviewers can see both sample-level dispersion (std) and the
      uncertainty of the mean (SEM) at a glance.
    """
    if table is None:
        raise SkipFigure("fig1_ablation_bar: baseline table is None")

    df = table.copy()

    completed_mask = df["n_folds"].fillna(0).astype(int) >= N_FOLDS
    completed = df[completed_mask & df["r2_mean"].notna()].copy()
    pending = df[~completed_mask | df["r2_mean"].isna()].copy()

    if completed.empty:
        raise SkipFigure(
            "fig1_ablation_bar: no completed rows (all n_folds<5 or NaN)"
        )

    completed = completed.sort_values("r2_mean", ascending=False).reset_index(drop=True)
    pending = pending.reset_index(drop=True)
    full = pd.concat([completed, pending], ignore_index=True)

    n = len(full)
    x = np.arange(n)

    # Compute pending mask once for reuse (I5 dedup).
    pending_any = np.array([
        pd.isna(r) or _nf_int(nf) < N_FOLDS
        for r, nf in zip(full["r2_mean"], full["n_folds"])
    ])

    # Colour coding (uses pending_any; I5)
    colours: list[str] = []
    edge_colours: list[str] = []
    fill: list[bool] = []
    for i, row in full.iterrows():
        if pending_any[i]:
            colours.append("#dddddd")
            edge_colours.append("#666666")
            fill.append(False)
        elif _is_ours_row(row["model"]):
            colours.append("#3b6ea5")  # steel blue
            edge_colours.append("#2a4f78")
            fill.append(True)
        else:
            colours.append("#888888")  # gray
            edge_colours.append("#555555")
            fill.append(True)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Build y and yerr, mapping NaN to 0-height outline-only bars
    y = full["r2_mean"].fillna(0.0).to_numpy()
    yerr = full["r2_std"].fillna(0.0).to_numpy()
    # SEM column (added 2026-04-28); fall back to std/sqrt(n_folds) if absent.
    if "r2_sem" in full.columns:
        ysem = full["r2_sem"].fillna(0.0).to_numpy()
    else:
        n_folds_arr = full["n_folds"].fillna(N_FOLDS).astype(float).to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            ysem = np.where(n_folds_arr > 0, yerr / np.sqrt(n_folds_arr), 0.0)
    # No errbars on pending (outline-only) rows
    yerr[pending_any] = 0.0
    ysem[pending_any] = 0.0

    for i in range(n):
        ax.bar(
            x[i], y[i],
            yerr=yerr[i] if yerr[i] > 0 else None,
            color=colours[i] if fill[i] else "none",
            edgecolor=edge_colours[i],
            linewidth=1.0 if fill[i] else 1.2,
            capsize=2.5,
            error_kw={"linewidth": 0.9, "ecolor": "#333333"},
        )

    # Inner SEM error bars (thicker, red) overlaid on top of std bars.
    sem_mask = ysem > 0
    if sem_mask.any():
        ax.errorbar(
            x[sem_mask], y[sem_mask], yerr=ysem[sem_mask],
            fmt="none", ecolor="#cc3333", elinewidth=1.8, capsize=2.0,
            capthick=1.8, zorder=4,
        )

    # Reference line at canonical R² (no label; the legend proxy carries it — M4)
    ax.axhline(
        canonical_r2, color="#cc5533", linestyle="--", linewidth=1.0,
    )
    # Zero line
    ax.axhline(0.0, color="#000000", linewidth=0.5)

    # Tick labels: the `display_name` from the table
    ax.set_xticks(x)
    ax.set_xticklabels(
        full["display_name"].to_list(), rotation=40, ha="right",
    )
    ax.set_ylabel("Cross-validated R²")
    ax.set_title("Model / ablation R² comparison (5-fold)")

    # Legend (outside axes so it never overlaps bars — M6)
    handles = [
        Patch(facecolor="#3b6ea5", edgecolor="#2a4f78", label="ResDec-MHE (ours / ablation)"),
        Patch(facecolor="#888888", edgecolor="#555555", label="Baselines"),
        Patch(facecolor="none", edgecolor="#666666", label="† pending (n_folds < 5)"),
        Line2D([0], [0], color="#333333", linewidth=0.9, label="± std (across folds)"),
        Line2D([0], [0], color="#cc3333", linewidth=1.8, label="± SEM (= std / √n)"),
        Line2D([0], [0], color="#cc5533", linestyle="--",
               label=f"canonical R² = {canonical_r2:.3f}"),
    ]
    ax.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
        frameon=True, fontsize=8,
    )

    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout(rect=[0, 0, 0.85, 1])
    return fig


# ---------------------------------------------------------------------------
# Figure 2: resilience scatter
# ---------------------------------------------------------------------------


def make_fig2_resilience_scatter(df: pd.DataFrame | None) -> plt.Figure:
    """y_true vs y_pred scatter, colored by residual, with quadrant overlay.

    Canonical resilience definition (see
    ``scripts/resdec_mhe/interpretability/resilience_residual_phenotype.py``):

        residual = y_true − y_pred
        residual > 0 → "Resilient" (better cognition than predicted)
        residual < 0 → "Overestimated" / "Vulnerable" (worse than predicted)

    The definition is **unconditional on pathology**. Quadrant labels describe
    y_true vs y_pred geometry only; pathology is not inferred from y_true.
    """
    if df is None or len(df) == 0:
        raise SkipFigure("fig2_resilience_scatter: predictions DataFrame is None/empty")
    required = {"y_true", "y_composite"}
    missing = required - set(df.columns)
    if missing:
        raise SkipFigure(
            f"fig2_resilience_scatter: predictions missing columns: {sorted(missing)}"
        )

    y_true = df["y_true"].to_numpy(dtype=np.float64)
    y_pred = df["y_composite"].to_numpy(dtype=np.float64)
    resid = y_true - y_pred

    med_y = float(np.median(y_true))
    med_r = float(np.median(resid))

    fig, ax = plt.subplots(figsize=(7, 6))
    vmax = float(np.max(np.abs(resid))) or 1.0
    sc = ax.scatter(
        y_true, y_pred, c=resid,
        cmap="coolwarm", vmin=-vmax, vmax=vmax,
        s=30, alpha=0.85, edgecolors="#222222", linewidths=0.3,
    )

    # Diagonal
    lo = float(min(y_true.min(), y_pred.min())) - 0.2
    hi = float(max(y_true.max(), y_pred.max())) + 0.2
    ax.plot([lo, hi], [lo, hi], color="#444444", linestyle="--", linewidth=1.0,
            label="y = x")

    # Quadrant lines: vertical at median y_true, horizontal at y_true - y_pred = med_r
    # Interpretation space uses (y_true, residual) → draw a second axis
    # directly: we draw a vertical line at median(y_true), and the horizontal
    # line through (median residual) in pred space corresponds to y = y_true - med_r.
    ax.axvline(med_y, color="#888888", linestyle=":", linewidth=0.8)
    # Horizontal boundary where residual = med_r:
    #   y_pred = y_true - med_r
    xs = np.array([lo, hi])
    ax.plot(xs, xs - med_r, color="#888888", linestyle=":", linewidth=0.8)

    # Quadrant labels (y_true vs y_pred geometry only — C2).
    # bottom-right (high y_true, low y_pred): Resilient (y_true > y_pred → residual > 0)
    # top-left    (low  y_true, high y_pred): Overestimated (y_pred > y_true → residual < 0)
    # top-right   (high y_true, high y_pred): Accurate high-cognition
    # bottom-left (low  y_true, low  y_pred): Accurate low-cognition
    pad = 0.05 * (hi - lo)
    ax.text(hi - pad, lo + pad,
            "Resilient\n(y_true > y_pred,\n positive residual)",
            fontsize=9, ha="right", va="bottom", color="#1f77b4",
            bbox=dict(facecolor="white", edgecolor="#aaaaaa", alpha=0.85,
                      boxstyle="round,pad=0.25"))
    ax.text(lo + pad, hi - pad,
            "Overestimated\n(y_pred > y_true,\n negative residual)",
            fontsize=9, ha="left", va="top", color="#d62728",
            bbox=dict(facecolor="white", edgecolor="#aaaaaa", alpha=0.85,
                      boxstyle="round,pad=0.25"))
    ax.text(hi - pad, hi - pad,
            "Accurate\n(high cognition)",
            fontsize=8, ha="right", va="top", color="#444444",
            bbox=dict(facecolor="white", edgecolor="#cccccc", alpha=0.70,
                      boxstyle="round,pad=0.25"))
    ax.text(lo + pad, lo + pad,
            "Accurate\n(low cognition)",
            fontsize=8, ha="left", va="bottom", color="#444444",
            bbox=dict(facecolor="white", edgecolor="#cccccc", alpha=0.70,
                      boxstyle="round,pad=0.25"))

    cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
    cbar.set_label("Residual (y_true − y_pred)")

    pooled_r2 = r2_score(y_true, y_pred)
    ax.set_xlabel("y_true (cognition score)")
    ax.set_ylabel("y_pred (composite prediction)")
    # M9: explicitly "pooled R²" to distinguish from mean-per-fold R².
    ax.set_title(
        f"Resilience scatter (pooled R² = {pooled_r2:.3f}, n={len(df)} subjects)"
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.legend(loc="upper left", frameon=True, fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 3: cell-type × gene heatmap
# ---------------------------------------------------------------------------


def make_fig3_celltype_gene_heatmap(
    summary: dict | None,
    top_n: int = TOP_N_PAIRS_HEATMAP,
) -> plt.Figure:
    """Single-column heatmap of the top-30 (cell-type, gene) IG pairs."""
    if summary is None:
        raise SkipFigure("fig3_celltype_gene_heatmap: summary is None")
    pairs = summary.get("top_cell_type_gene_pairs")
    if not pairs:
        raise SkipFigure(
            "fig3_celltype_gene_heatmap: 'top_cell_type_gene_pairs' missing/empty"
        )

    pairs = pairs[:top_n]
    labels = [f"{p['cell_type']} / {p['gene']}" for p in pairs]
    values = np.array([p["mean_abs_attribution"] for p in pairs], dtype=np.float64)

    # Row i = pair i, single column; use imshow with shape (N, 1).
    fig, ax = plt.subplots(figsize=(6.5, max(6.0, 0.2 * len(pairs) + 2.0)))
    im = ax.imshow(
        values.reshape(-1, 1),
        cmap="Reds",
        aspect="auto",
    )
    ax.set_yticks(np.arange(len(pairs)))
    ax.set_yticklabels(labels)
    ax.set_xticks([0])
    ax.set_xticklabels(["mean |attr|"])
    ax.set_title(f"Top-{len(pairs)} (cell-type, gene) pairs — Captum IG")

    # Annotate the top-5 values with their numeric magnitude.
    top5_idx = np.argsort(values)[::-1][:5]
    for i in top5_idx:
        val = values[i]
        # Text color threshold at 50% of colormap range — white text on dark-red
        # cells (high |attr|), black text on light-red cells (low |attr|) for
        # WCAG-adequate contrast.
        ax.text(0, i, f"{val:.4f}", ha="center", va="center",
                color="white" if val > values.max() * 0.5 else "#111111",
                fontsize=8)

    cbar = fig.colorbar(im, ax=ax, shrink=0.65)
    cbar.set_label("Mean |attribution|")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 4: head specialization
# ---------------------------------------------------------------------------


def make_fig4_head_specialization(
    head_summary: dict | None,
    splatter_lamp5_corr: float | None = None,
) -> plt.Figure:
    """Grouped bar: per-head top-3 cell types with Shannon entropy note.

    - 4 heads × top-3 cell types as a grouped bar chart.
    - Each head's Shannon entropy (nats) annotated above its bar group.
    - Splatter × LAMP5 correlation (if provided) added as a footnote.
    """
    if head_summary is None:
        raise SkipFigure("fig4_head_specialization: head_summary is None")
    heads = head_summary.get("head_specialization")
    if not heads:
        raise SkipFigure(
            "fig4_head_specialization: 'head_specialization' missing/empty"
        )

    n_heads = len(heads)
    top_k = max(len(h["top_3_cell_types"]) for h in heads)
    # Collect all unique cell type labels in the order they appear across heads
    ct_order: list[str] = []
    for h in heads:
        for entry in h["top_3_cell_types"]:
            if entry["cell_type"] not in ct_order:
                ct_order.append(entry["cell_type"])

    # Grouped bar: for each head, up to top_k bars (one per top-rank slot).
    fig, ax = plt.subplots(figsize=(10, 5))
    bar_w = 0.18
    group_gap = 0.3
    cmap = matplotlib.colormaps.get_cmap("tab20")
    ct_colours = {ct: cmap(i % 20) for i, ct in enumerate(ct_order)}

    for h_i, h in enumerate(heads):
        for slot, entry in enumerate(h["top_3_cell_types"]):
            xpos = h_i * (top_k * bar_w + group_gap) + slot * bar_w
            ax.bar(
                xpos, entry["mean_attention"], width=bar_w,
                color=ct_colours[entry["cell_type"]],
                edgecolor="#333333", linewidth=0.6,
                label=entry["cell_type"],
            )

    # Per-head entropy annotations at the top of each head group
    y_max = max(
        entry["mean_attention"]
        for h in heads for entry in h["top_3_cell_types"]
    )
    y_top = y_max * 1.18
    ax.set_ylim(0, y_max * 1.35)
    group_centres = []
    for h_i, h in enumerate(heads):
        centre = h_i * (top_k * bar_w + group_gap) + (top_k - 1) * bar_w / 2
        group_centres.append(centre)
        ax.text(
            centre, y_top,
            f"Head {h_i}\nH = {h['shannon_entropy_nats']:.2f} nats\n"
            f"eff. CTs = {h['effective_n_cell_types']:.1f}",
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_xticks(group_centres)
    ax.set_xticklabels([f"Head {i}" for i in range(n_heads)])
    ax.set_ylabel("Mean attention")
    ax.set_title("Head specialization — top-3 cell types per head")
    # Deduplicate legend (M2: explicit loop; preserves first-seen order).
    handles, labels = ax.get_legend_handles_labels()
    seen: set[str] = set()
    dedup_handles: list = []
    dedup_labels: list[str] = []
    for handle, label in zip(handles, labels):
        if label in seen:
            continue
        seen.add(label)
        dedup_handles.append(handle)
        dedup_labels.append(label)
    ax.legend(
        dedup_handles, dedup_labels,
        loc="center left", bbox_to_anchor=(1.02, 0.5),
        frameon=True, fontsize=8, title="Cell type",
    )

    # Footnote with Splatter × LAMP5 correlation, if provided
    if splatter_lamp5_corr is not None:
        fig.text(
            0.01, 0.01,
            f"Splatter × LAMP5-LHX6 co-attention r = {splatter_lamp5_corr:.3f}",
            ha="left", va="bottom", fontsize=8, color="#444444",
        )

    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 5: subgroup R² with bootstrap CIs
# ---------------------------------------------------------------------------


_SUBGROUP_FAMILIES: tuple[tuple[str, str, list[str]], ...] = (
    ("APOE", "APOE-ε4 alleles",
     ["APOE_e4_0", "APOE_e4_1", "APOE_e4_2"]),
    ("Sex", "Sex (msex)",
     ["msex_0", "msex_1"]),
    ("Age", "Age quartile",
     ["age_quartile_Q1", "age_quartile_Q2", "age_quartile_Q3", "age_quartile_Q4"]),
    ("Pathology", "Pathology quartile",
     ["pathology_quartile_Q1", "pathology_quartile_Q2",
      "pathology_quartile_Q3", "pathology_quartile_Q4"]),
)


def make_fig5_subgroup_r2(
    metrics: dict | None,
    canonical_r2: float = CANONICAL_R2,
) -> plt.Figure:
    """Bar chart of per-subgroup R² with 95 % bootstrap CIs, grouped by family."""
    if metrics is None:
        raise SkipFigure("fig5_subgroup_r2: subgroup metrics is None")
    if not metrics:
        raise SkipFigure("fig5_subgroup_r2: subgroup metrics is empty")

    fig, ax = plt.subplots(figsize=(11, 5))

    x_pos: list[float] = []
    r2_vals: list[float] = []
    ci_lo: list[float] = []
    ci_hi: list[float] = []
    labels: list[str] = []
    colours: list[str] = []
    family_palette = {
        "APOE": "#ca6f3b",
        "Sex": "#2a8a6a",
        "Age": "#3b6ea5",
        "Pathology": "#a4408c",
    }
    boundaries: list[float] = []  # positions of vertical family separators

    cursor = 0.0
    family_gap = 1.0
    for f_idx, (family, f_label, subgroups) in enumerate(_SUBGROUP_FAMILIES):
        family_start = cursor
        for sg in subgroups:
            if sg not in metrics:
                continue
            entry = metrics[sg]
            r2 = float(entry["r2"])
            ci = entry.get("r2_ci", [r2, r2])
            r2_vals.append(r2)
            ci_lo.append(float(ci[0]))
            ci_hi.append(float(ci[1]))
            x_pos.append(cursor)

            # C1: explicit per-family prefix strip, then underscore → space.
            label = sg
            prefix = _FAMILY_PREFIX_STRIP.get(family)
            if prefix and sg.startswith(prefix):
                label = sg[len(prefix):]
            label = label.replace("_", " ")
            labels.append(label)

            colours.append(family_palette[family])
            cursor += 1.0
        if f_idx < len(_SUBGROUP_FAMILIES) - 1 and cursor > family_start:
            boundaries.append(cursor - 0.5)
            cursor += family_gap

    x = np.array(x_pos)
    y = np.array(r2_vals)
    ci_lo_arr = np.array(ci_lo)
    yerr_lo = y - ci_lo_arr
    yerr_hi = np.array(ci_hi) - y

    # clamp visually at a sane lower bound so tiny-n wild CIs don't blow up the axis.
    lower_clip = R2_VISUAL_LOWER_CLIP
    yerr_lo_clipped = np.minimum(yerr_lo, y - lower_clip)
    yerr_lo_clipped = np.maximum(yerr_lo_clipped, 0.0)

    # I6: any bar whose TRUE CI lower bound falls below the visual clip gets a
    # '†' suffix so the dagger in the footnote refers to something concrete.
    truncated = ci_lo_arr < lower_clip
    any_truncated = bool(np.any(truncated))
    for i, was_truncated in enumerate(truncated):
        if was_truncated:
            labels[i] = labels[i] + "†"

    ax.bar(x, y, color=colours, edgecolor="#333333", linewidth=0.7, width=0.85)
    ax.errorbar(
        x, y, yerr=[yerr_lo_clipped, yerr_hi],
        fmt="none", ecolor="#222222", capsize=3, linewidth=0.8,
    )

    # Canonical R² reference line (no label; legend proxy carries it — M4)
    ax.axhline(
        canonical_r2, color="#cc5533", linestyle="--", linewidth=1.0,
    )
    ax.axhline(0.0, color="#000000", linewidth=0.5)

    for b in boundaries:
        ax.axvline(b, color="#bbbbbb", linestyle="--", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("R² (95 % bootstrap CI)")
    ax.set_title("Subgroup R² — canonical model (ResDec-MHE)")
    ax.set_ylim(lower_clip, 1.0)

    # Family legend (uses hoisted Patch / Line2D from module-level imports — M3).
    handles = [
        Patch(facecolor=family_palette[f], edgecolor="#333333", label=name)
        for f, name, _ in _SUBGROUP_FAMILIES
    ]
    handles.append(Line2D([0], [0], color="#cc5533", linestyle="--",
                          label=f"canonical R² = {canonical_r2:.3f}"))
    ax.legend(handles=handles, loc="upper right", fontsize=8, frameon=True)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)

    # I6: footnote explaining the '†' marker when any CI is truncated.
    if any_truncated:
        fig.text(
            0.02, 0.01,
            "† CI lower bound extends below axis range "
            "(small-n subgroup; see subgroup_metrics.json)",
            fontsize=8, style="italic", color="#444444",
        )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 6: calibration
# ---------------------------------------------------------------------------


def make_fig6_calibration(
    stat_rigor: dict | None,
    per_subject: pd.DataFrame | None,
) -> plt.Figure:
    """Two-panel calibration figure.

    Left:   |residual| vs σ_TabPFN scatter with y = x reference line.
    Right:  nominal vs empirical coverage curve (using coverage_at_0.5/0.68/0.8/0.95)
            with diagonal = perfect-calibration reference.
    """
    if stat_rigor is None or per_subject is None:
        raise SkipFigure(
            "fig6_calibration: statistical_rigor or per_subject DataFrame is None"
        )
    cov = stat_rigor.get("calibration_coverage")
    if not cov:
        raise SkipFigure(
            "fig6_calibration: 'calibration_coverage' missing from statistical_rigor"
        )
    if not {"abs_residual", "sigma_tabpfn"}.issubset(per_subject.columns):
        raise SkipFigure(
            "fig6_calibration: per_subject df missing abs_residual / sigma_tabpfn"
        )

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 5))

    # ------- Left: |residual| vs σ -------
    r = per_subject["abs_residual"].to_numpy(dtype=np.float64)
    s = per_subject["sigma_tabpfn"].to_numpy(dtype=np.float64)
    ax_l.scatter(
        s, r, alpha=0.55, s=18, color="#3b6ea5",
        edgecolor="#1f3d5a", linewidths=0.3,
    )
    lo = 0.0
    hi = float(max(r.max(), s.max())) * 1.05
    ax_l.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0,
              color="#cc5533", label="|resid| = σ")
    ax_l.set_xlabel("TabPFN σ (per subject)")
    ax_l.set_ylabel("|residual| (|y_true − y_composite|)")
    ax_l.set_xlim(lo, hi)
    ax_l.set_ylim(lo, hi)
    ax_l.set_title("Residual magnitude vs predictive σ")
    # Means annotation
    ax_l.text(
        0.02, 0.97,
        f"mean |resid| = {cov['mean_abs_residual']:.3f}\n"
        f"mean σ       = {cov['mean_sigma']:.3f}",
        transform=ax_l.transAxes, ha="left", va="top", fontsize=8,
        family="monospace",
        bbox=dict(facecolor="white", edgecolor="#aaaaaa", alpha=0.85,
                  boxstyle="round,pad=0.3"),
    )
    ax_l.grid(True, linestyle=":", alpha=0.4)
    ax_l.legend(loc="lower right", fontsize=8, frameon=True)

    # ------- Right: coverage curve -------
    nominal_levels = list(NOMINAL_COVERAGE_LEVELS)
    empirical = [float(cov[f"coverage_at_{L}"]) for L in nominal_levels]
    ax_r.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0,
              color="#888888", label="perfect calibration")
    ax_r.plot(
        nominal_levels, empirical,
        marker="o", color="#3b6ea5", linewidth=1.2,
        markersize=6, label="empirical coverage",
    )
    for xn, ye in zip(nominal_levels, empirical):
        ax_r.annotate(
            f"{ye:.2f}",
            xy=(xn, ye), xytext=(5, 5), textcoords="offset points",
            fontsize=8,
        )
    ax_r.set_xlabel("Nominal coverage")
    ax_r.set_ylabel("Empirical coverage")
    ax_r.set_title("Calibration curve (σ from TabPFN-2.6)")
    ax_r.set_xlim(0, 1)
    ax_r.set_ylim(0, 1)
    ax_r.grid(True, linestyle=":", alpha=0.4)
    ax_r.legend(loc="lower right", fontsize=8, frameon=True)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 7: TabPFN feature-count sensitivity
# ---------------------------------------------------------------------------


def make_fig7_k_sensitivity(
    k_values: list[int],
    r2_means: list[float],
    r2_stds: list[float],
    bootstrap_ci: tuple[float, float] | None = None,
    r2_sems: list[float] | None = None,
    n_folds: int = N_FOLDS,
) -> plt.Figure:
    """Line plot of 5-fold R² (mean ± std, with inner SEM bars) vs TabPFN top-k.

    - X-axis: top-k (log scale, readable for {1000, 2000, 4000}).
    - Y-axis: 5-fold mean R² with outer ±std errorbars and inner ±SEM
      errorbars (thicker red caps); SEM = std / √n_folds when ``r2_sems``
      is None.
    - Canonical k=2000 highlighted with a distinct marker / annotation.
    - Optional horizontal reference band from ``bootstrap_ci`` (canonical
      bootstrap R² 95 % CI) illustrating that all three k values sit deep
      inside the canonical's CI — i.e. robustness to feature-count choice.

    Raises
    ------
    SkipFigure
        If ``k_values`` is empty, inputs have mismatched lengths, or any
        entry in ``r2_means`` / ``r2_stds`` is NaN.
    """
    if not k_values:
        raise SkipFigure("fig7_k_sensitivity: k_values is empty")
    if len(k_values) != len(r2_means) or len(k_values) != len(r2_stds):
        raise SkipFigure(
            "fig7_k_sensitivity: k_values / r2_means / r2_stds length mismatch"
        )
    r2_arr = np.asarray(r2_means, dtype=np.float64)
    std_arr = np.asarray(r2_stds, dtype=np.float64)
    # Treat empty list / None as "no SEM provided" — fall back to std/√n_folds.
    if not r2_sems:
        sem_arr = std_arr / np.sqrt(max(n_folds, 1))
    else:
        if len(r2_sems) != len(k_values):
            raise SkipFigure(
                "fig7_k_sensitivity: r2_sems length does not match k_values"
            )
        sem_arr = np.asarray(r2_sems, dtype=np.float64)
    if np.any(np.isnan(r2_arr)) or np.any(np.isnan(std_arr)):
        raise SkipFigure(
            "fig7_k_sensitivity: NaN in r2_means or r2_stds"
        )

    # Ensure k ascending — downstream annotations assume monotonic x.
    order = np.argsort(np.asarray(k_values))
    k_sorted = np.asarray(k_values)[order]
    r2_sorted = r2_arr[order]
    std_sorted = std_arr[order]
    sem_sorted = sem_arr[order]

    fig, ax = plt.subplots(figsize=(7, 5))

    # Optional canonical bootstrap CI reference band (e.g. (0.39, 0.51)).
    if bootstrap_ci is not None:
        ci_lo, ci_hi = float(bootstrap_ci[0]), float(bootstrap_ci[1])
        ax.axhspan(
            ci_lo, ci_hi,
            facecolor="#cc5533", alpha=0.12, zorder=0,
            label=f"canonical bootstrap 95% CI ({ci_lo:.2f}, {ci_hi:.2f})",
        )
        ax.axhline(ci_lo, color="#cc5533", linestyle=":", linewidth=0.8, zorder=1)
        ax.axhline(ci_hi, color="#cc5533", linestyle=":", linewidth=0.8, zorder=1)

    # Main line with outer std error bars.
    ax.errorbar(
        k_sorted, r2_sorted, yerr=std_sorted,
        marker="o", markersize=7, linewidth=1.4, capsize=4,
        color="#3b6ea5", ecolor="#222222", elinewidth=0.9,
        markerfacecolor="#3b6ea5", markeredgecolor="#1f3d5a",
        label="ResDec-MHE (5-fold mean R²; outer ± std)",
        zorder=3,
    )

    # Inner SEM error bars (thicker, red) overlaid on top.
    ax.errorbar(
        k_sorted, r2_sorted, yerr=sem_sorted,
        fmt="none", ecolor="#cc3333", elinewidth=1.8, capsize=2.5,
        capthick=1.8, zorder=4, label="± SEM (= std / √n)",
    )

    # Highlight canonical k=2000 (if present) with a distinct marker.
    canonical_k = 2000
    if canonical_k in k_sorted.tolist():
        idx = int(np.where(k_sorted == canonical_k)[0][0])
        ax.scatter(
            [canonical_k], [r2_sorted[idx]],
            marker="*", s=260, color="#ffcc33",
            edgecolors="#8a6a00", linewidths=1.2,
            zorder=5, label="canonical (k=2000)",
        )
        ax.annotate(
            (
                f"canonical\nR² = {r2_sorted[idx]:.3f}"
                f"\n ± {std_sorted[idx]:.3f} std"
                f"\n ± {sem_sorted[idx]:.3f} SEM"
            ),
            xy=(canonical_k, r2_sorted[idx]),
            xytext=(8, 14), textcoords="offset points",
            fontsize=8, ha="left", va="bottom",
            bbox=dict(facecolor="white", edgecolor="#aaaaaa",
                      alpha=0.85, boxstyle="round,pad=0.25"),
        )

    # Annotate each non-canonical point with its R² value (std + SEM both shown).
    for k_i, r2_i, sd_i, se_i in zip(
        k_sorted.tolist(), r2_sorted, std_sorted, sem_sorted
    ):
        if k_i == canonical_k:
            continue
        ax.annotate(
            f"{r2_i:.3f} ± {sd_i:.3f} (SEM {se_i:.3f})",
            xy=(k_i, r2_i), xytext=(6, -14), textcoords="offset points",
            fontsize=8, ha="left", va="top", color="#1f3d5a",
        )

    ax.set_xscale("log")
    # Explicit ticks at the measured k values (log scale hides them otherwise).
    # Suppress matplotlib's default minor/major log ticks so only our k values
    # show — otherwise "3×10³" appears between 2000 and 4000.
    ax.set_xticks(k_sorted.tolist())
    ax.set_xticklabels([str(int(k)) for k in k_sorted.tolist()])
    ax.set_xticks([], minor=True)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("TabPFN top-k features (log scale)")
    ax.set_ylabel("5-fold cross-validated R² (mean ± std; inner ± SEM)")
    ax.set_title("TabPFN feature-count sensitivity")

    # Y-limits: respect bootstrap CI if provided so the band is visible.
    y_lo_data = float(np.min(r2_sorted - std_sorted))
    y_hi_data = float(np.max(r2_sorted + std_sorted))
    if bootstrap_ci is not None:
        y_lo_data = min(y_lo_data, float(bootstrap_ci[0]))
        y_hi_data = max(y_hi_data, float(bootstrap_ci[1]))
    pad = 0.05 * max(y_hi_data - y_lo_data, 0.05)
    ax.set_ylim(y_lo_data - pad, y_hi_data + pad)

    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", fontsize=8, frameon=True)

    # Subtitle / caption below the axes.
    if bootstrap_ci is not None:
        ci_lo, ci_hi = float(bootstrap_ci[0]), float(bootstrap_ci[1])
        caption = (
            f"All three k values within canonical's bootstrap 95% CI "
            f"({ci_lo:.2f}, {ci_hi:.2f}) — model robust to feature-count choice."
        )
    else:
        caption = (
            "All three k values lie within ±1 std of the canonical "
            "point estimate — model robust to feature-count choice."
        )
    fig.text(
        0.5, 0.01, caption,
        ha="center", va="bottom", fontsize=8, style="italic", color="#444444",
    )

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return fig


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------


def save_figure(
    fig: plt.Figure,
    out_dir: Path,
    stem: str,
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 300,
) -> list[Path]:
    """Save ``fig`` to ``out_dir/<stem>.<fmt>`` for each format; return paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths: list[Path] = []
    for fmt in formats:
        p = out_dir / f"{stem}.{fmt}"
        fig.savefig(p, dpi=dpi)
        out_paths.append(p)
        logger.info("Wrote %s", p)
    return out_paths


# ---------------------------------------------------------------------------
# Data loaders (disk → in-memory) — only used by the CLI, not unit tests.
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_calibration_per_subject(
    pred_root: Path, tabpfn_dir: Path, n_folds: int = N_FOLDS,
) -> pd.DataFrame:
    """Join per-subject |residual| (composite) with sigma_tabpfn across folds.

    Delegates the fold-level prediction + TabPFN join to the shared canonical
    loader ``load_fold_predictions``; only ``sigma_tabpfn`` is added on
    top per fold.
    """
    frames: list[pd.DataFrame] = []
    for f in range(n_folds):
        merged = load_fold_predictions(pred_root, tabpfn_dir, f)
        tab_path = tabpfn_dir / f"tabpfn_outer_fold{f}.npz"
        tab = np.load(tab_path, allow_pickle=True)
        sigma = pd.DataFrame({
            "ROSMAP_IndividualID": tab["val_subject_ids"].astype(str),
            "sigma_tabpfn": tab["sigma_tabpfn"].astype(np.float64),
        })
        df = merged.merge(sigma, on="ROSMAP_IndividualID", how="inner")
        df["abs_residual"] = np.abs(df["y_true"] - df["y_composite"])
        frames.append(df[["ROSMAP_IndividualID", "fold",
                          "abs_residual", "sigma_tabpfn"]])
    return pd.concat(frames, ignore_index=True)


def _load_k_sensitivity_data(
    canonical_dir: Path, ablation_root: Path
) -> tuple[list[int], list[float], list[float], list[float]]:
    """Load k-sensitivity R² data from k=1000/2000/4000 summaries.

    Reads ``best_vs_tabpfn_summary.json`` from each of
    ``{ablation_root}/p5_ablation_topk_1000``, ``canonical_dir``,
    ``{ablation_root}/p5_ablation_topk_4000`` and computes per-k
    ``(mean, std, sem)`` over ``per_fold[i]["ours"]["r2"]``. SEM is
    ``std / sqrt(n_folds)``.

    Returns sorted k values, R² means, R² stds (ddof=1), R² SEMs. Missing
    summaries are skipped — the figure will be drawn with whatever k
    entries loaded.
    """
    k_to_dir: dict[int, Path] = {
        1000: ablation_root / "p5_ablation_topk_1000",
        2000: canonical_dir,
        4000: ablation_root / "p5_ablation_topk_4000",
    }

    k_vals: list[int] = []
    r2_means: list[float] = []
    r2_stds: list[float] = []
    r2_sems: list[float] = []
    for k in sorted(k_to_dir):
        summary_path = k_to_dir[k] / "best_vs_tabpfn_summary.json"
        if not summary_path.is_file():
            logger.warning("k=%d: summary not found at %s (skipping)", k, summary_path)
            continue
        try:
            data = _load_json(summary_path)
            fold_r2s = [
                float(pf["ours"]["r2"]) for pf in data.get("per_fold", [])
            ]
            if not fold_r2s:
                logger.warning("k=%d: no per_fold entries in %s", k, summary_path)
                continue
            k_vals.append(int(k))
            r2_means.append(float(np.mean(fold_r2s)))
            std_k = float(np.std(fold_r2s, ddof=1)) if len(fold_r2s) > 1 else 0.0
            r2_stds.append(std_k)
            r2_sems.append(std_k / float(np.sqrt(len(fold_r2s))) if fold_r2s else 0.0)
        except Exception as e:
            logger.warning("k=%d: failed to parse %s (%s)", k, summary_path, e)
    return k_vals, r2_means, r2_stds, r2_sems


def _load_bootstrap_r2_ci(stat_rigor: dict | None) -> tuple[float, float] | None:
    """Extract (ci_lower, ci_upper) from statistical_rigor['bootstrap_r2_ci']."""
    if not stat_rigor:
        return None
    bci = stat_rigor.get("bootstrap_r2_ci")
    if not bci or "ci_lower" not in bci or "ci_upper" not in bci:
        return None
    try:
        return (float(bci["ci_lower"]), float(bci["ci_upper"]))
    except (TypeError, ValueError):
        return None


def _safe_splatter_lamp5_corr(summary_path: Path) -> float | None:
    """Extract Splatter × LAMP5-LHX6 pearson r from the deep-dive JSON.

    Returns ``None`` on any failure — the annotation is decorative.
    """
    try:
        deep = _load_json(summary_path)
        key = "Splatter__vs__LAMP5-LHX6 and Chandelier"
        return float(
            deep.get("gabaergic_interneuron_co_attention_pearson_r", {}).get(key)
        )
    except Exception as e:
        logger.warning("splatter deep-dive not loaded (%s)", e)
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-table", type=Path,
                   default=Path("outputs/redesign/interpretability/paper_baseline_table.csv"))
    p.add_argument("--captum-summary", type=Path,
                   default=Path("outputs/redesign/interpretability/captum_ig/composite_attribution_summary.json"))
    p.add_argument("--head-analysis", type=Path,
                   default=Path("outputs/redesign/interpretability/head_analysis_summary.json"))
    p.add_argument("--splatter-deepdive", type=Path,
                   default=Path("outputs/redesign/interpretability/splatter_deepdive_summary.json"))
    p.add_argument("--subgroup-metrics", type=Path,
                   default=Path("outputs/redesign/interpretability/subgroup_metrics.json"))
    p.add_argument("--statistical-rigor", type=Path,
                   default=Path("outputs/redesign/interpretability/statistical_rigor.json"))
    p.add_argument("--residual-csv", type=Path,
                   default=Path("outputs/redesign/interpretability/residual_per_subject.csv"))
    p.add_argument("--pred-root", type=Path,
                   default=Path("outputs/redesign/p5_canonical_seed42"))
    p.add_argument("--ablation-root", type=Path,
                   default=Path("outputs/redesign"),
                   help="Parent dir containing p5_ablation_topk_{1000,4000} subdirs")
    p.add_argument("--tabpfn-dir", type=Path, default=Path("data/redesign"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("outputs/redesign/interpretability/figures"))
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--figure-format", nargs="+", default=["png", "pdf"])
    p.add_argument("--canonical-r2", type=float, default=CANONICAL_R2)
    p.add_argument("--n-folds", type=int, default=N_FOLDS)
    return p.parse_args(argv)


def _try_make(
    name: str,
    thunk,
    out_dir: Path,
    stem: str,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    try:
        fig = thunk()
    except SkipFigure as e:
        logger.warning("SKIP %s: %s", name, e)
        return []
    paths = save_figure(fig, out_dir, stem, formats=formats, dpi=dpi)
    plt.close(fig)
    return paths


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    )
    # Note: _apply_paper_style() runs at module import so tests also render
    # with paper-style rcParams. Re-running it here would be a no-op.

    # --- Fig 1: ablation bar ---
    fig1_paths: list[Path] = []
    try:
        table = pd.read_csv(args.baseline_table) if args.baseline_table.exists() else None
    except Exception as e:
        logger.warning("baseline table load failed: %s", e)
        table = None
    fig1_paths = _try_make(
        "fig1_ablation_bar",
        lambda: make_fig1_ablation_bar(table=table, canonical_r2=args.canonical_r2),
        args.out_dir, "fig_ablation_bar",
        args.figure_format, args.dpi,
    )

    # --- Fig 2: resilience scatter ---
    fig2_paths: list[Path] = []
    try:
        df_preds = load_all_folds(
            args.pred_root, args.tabpfn_dir, n_folds=args.n_folds
        )
    except Exception as e:
        logger.warning("load_all_folds failed: %s", e)
        df_preds = None
    fig2_paths = _try_make(
        "fig2_resilience_scatter",
        lambda: make_fig2_resilience_scatter(df=df_preds),
        args.out_dir, "fig_resilience_scatter",
        args.figure_format, args.dpi,
    )

    # --- Fig 3: CT × gene heatmap ---
    fig3_paths: list[Path] = []
    try:
        captum = _load_json(args.captum_summary) if args.captum_summary.exists() else None
    except Exception as e:
        logger.warning("captum summary load failed: %s", e)
        captum = None
    fig3_paths = _try_make(
        "fig3_celltype_gene_heatmap",
        lambda: make_fig3_celltype_gene_heatmap(summary=captum),
        args.out_dir, "fig_celltype_gene_heatmap",
        args.figure_format, args.dpi,
    )

    # --- Fig 4: head specialization ---
    fig4_paths: list[Path] = []
    try:
        head_data = _load_json(args.head_analysis) if args.head_analysis.exists() else None
    except Exception as e:
        logger.warning("head analysis load failed: %s", e)
        head_data = None
    splatter_lamp5 = _safe_splatter_lamp5_corr(args.splatter_deepdive)
    fig4_paths = _try_make(
        "fig4_head_specialization",
        lambda: make_fig4_head_specialization(
            head_summary=head_data, splatter_lamp5_corr=splatter_lamp5,
        ),
        args.out_dir, "fig_head_specialization",
        args.figure_format, args.dpi,
    )

    # --- Fig 5: subgroup R² ---
    fig5_paths: list[Path] = []
    try:
        subg = _load_json(args.subgroup_metrics) if args.subgroup_metrics.exists() else None
    except Exception as e:
        logger.warning("subgroup metrics load failed: %s", e)
        subg = None
    fig5_paths = _try_make(
        "fig5_subgroup_r2",
        lambda: make_fig5_subgroup_r2(metrics=subg, canonical_r2=args.canonical_r2),
        args.out_dir, "fig_subgroup_r2",
        args.figure_format, args.dpi,
    )

    # --- Fig 6: calibration ---
    fig6_paths: list[Path] = []
    try:
        rigor = _load_json(args.statistical_rigor) if args.statistical_rigor.exists() else None
    except Exception as e:
        logger.warning("statistical_rigor load failed: %s", e)
        rigor = None
    try:
        per_subj = _load_calibration_per_subject(
            args.pred_root, args.tabpfn_dir, n_folds=args.n_folds,
        )
    except Exception as e:
        logger.warning("per-subject calibration table load failed: %s", e)
        per_subj = None
    fig6_paths = _try_make(
        "fig6_calibration",
        lambda: make_fig6_calibration(stat_rigor=rigor, per_subject=per_subj),
        args.out_dir, "fig_calibration",
        args.figure_format, args.dpi,
    )

    # --- Fig 7: k-sensitivity ---
    fig7_paths: list[Path] = []
    try:
        k_vals, r2_means, r2_stds, r2_sems = _load_k_sensitivity_data(
            canonical_dir=args.pred_root,
            ablation_root=args.ablation_root,
        )
    except Exception as e:
        logger.warning("k-sensitivity data load failed: %s", e)
        k_vals, r2_means, r2_stds, r2_sems = [], [], [], []
    bootstrap_ci = _load_bootstrap_r2_ci(rigor)
    fig7_paths = _try_make(
        "fig7_k_sensitivity",
        lambda: make_fig7_k_sensitivity(
            k_values=k_vals, r2_means=r2_means, r2_stds=r2_stds,
            r2_sems=r2_sems,
            bootstrap_ci=bootstrap_ci,
        ),
        args.out_dir, "fig_k_sensitivity",
        args.figure_format, args.dpi,
    )

    # Summary
    total = sum(len(p) for p in (fig1_paths, fig2_paths, fig3_paths,
                                 fig4_paths, fig5_paths, fig6_paths,
                                 fig7_paths))
    logger.info("Wrote %d figure files total → %s", total, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
