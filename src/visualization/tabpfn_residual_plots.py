"""TabPFN-residual decomposition figures for ResDec-MHE composite (§4d in MASTER-INFO).

Composite prediction: ŷ_composite = ŷ_TabPFN + f̂_residual

These figures expose the additive structure that ResDec-MHE relies on:
the TabPFN base accounts for ~42.5% of Var(y) and the residual head adds
~4.6% (per outputs/canonical/interpretability/variance_decomposition.json).

Four candidate figures (user picks visually after rendering):

1. ``plot_additive_3panel`` — three sub-panels in one figure: (A) y vs
   ŷ_TabPFN scatter; (B) y vs ŷ_composite scatter; (C) f̂_residual histogram
   color-coded by ŷ_TabPFN tertile. Shows what TabPFN captures vs what the
   residual closes.

2. ``plot_variance_partition_bar`` — single stacked bar of the variance
   decomposition: Var(ŷ_TabPFN), Var(f̂_residual), 2·Cov(...), Var(residual ε).

3. ``plot_per_subject_delta_scatter`` — f̂_residual vs ŷ_TabPFN, points
   colored by global pathology (gpath). Reveals whether residual gains
   are subject-modulated.

4. ``plot_residual_histogram_overlay`` — distribution of f̂_residual
   alongside the no-residual baseline (TabPFN-only error). Shows the
   subject-level "before / after" effect.

All functions use ``src/visualization/theme.py`` palettes + fmt_axes +
save_fig conventions. No new colormaps.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from sklearn.metrics import r2_score

from src.visualization.theme import (
    PALETTES,
    apply_theme,
    fmt_axes,
    save_fig,
)


def plot_additive_3panel(
    y_true: np.ndarray,
    y_tabpfn: np.ndarray,
    y_residual: np.ndarray,
    out_stem: str | Path,
    *,
    figsize: tuple[float, float] = (10.5, 3.5),
) -> Path:
    """Three-panel additive decomposition figure."""
    apply_theme()
    y_composite = y_tabpfn + y_residual
    fig = plt.figure(figsize=figsize, constrained_layout=False)
    gs = GridSpec(1, 3, figure=fig, wspace=0.35)

    palette = PALETTES["categorical"]
    color_a = palette[0]   # tab10 blue
    color_b = palette[3]   # tab10 red — primary "improvement" tone

    # (A) y vs ŷ_TabPFN
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.scatter(y_tabpfn, y_true, s=10, alpha=0.55, color=color_a, edgecolors="white",
                 linewidth=0.5)
    lo = float(min(y_true.min(), y_tabpfn.min(), y_composite.min()))
    hi = float(max(y_true.max(), y_tabpfn.max(), y_composite.max()))
    ax_a.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.6)
    ax_a.set_xlabel("ŷ_TabPFN")
    ax_a.set_ylabel("y_true")
    ax_a.set_title(f"(A) TabPFN-only  (R²={_r2(y_true, y_tabpfn):.3f})")
    fmt_axes(ax_a)

    # (B) y vs ŷ_composite
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.scatter(y_composite, y_true, s=10, alpha=0.55, color=color_b, edgecolors="white",
                 linewidth=0.5)
    ax_b.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.6)
    ax_b.set_xlabel("ŷ_composite = ŷ_TabPFN + f̂_residual")
    ax_b.set_ylabel("y_true")
    ax_b.set_title(f"(B) Composite  (R²={_r2(y_true, y_composite):.3f})")
    fmt_axes(ax_b)

    # (C) residual histogram color-coded by TabPFN tertile
    ax_c = fig.add_subplot(gs[0, 2])
    tert = np.quantile(y_tabpfn, [1/3, 2/3])
    bins_idx = np.digitize(y_tabpfn, tert)
    seq = PALETTES["sequential"](np.linspace(0.15, 0.85, 3))
    bin_labels = ["TabPFN low", "TabPFN mid", "TabPFN high"]
    bin_edges = np.linspace(y_residual.min(), y_residual.max(), 35)
    for k in range(3):
        ax_c.hist(y_residual[bins_idx == k], bins=bin_edges, color=seq[k],
                  alpha=0.55, label=bin_labels[k], edgecolor="white", linewidth=0.4)
    ax_c.axvline(0, ls="--", color="black", lw=0.6, alpha=0.6)
    ax_c.set_xlabel("f̂_residual")
    ax_c.set_ylabel("subjects")
    ax_c.set_title("(C) Residual head correction\nby TabPFN baseline")
    ax_c.legend(loc="upper left", frameon=True, fontsize=7)
    fmt_axes(ax_c)

    paths = save_fig(fig, out_stem)
    plt.close(fig)
    return paths[0]


def plot_variance_partition_bar(
    var_components: dict,
    out_stem: str | Path,
    *,
    figsize: tuple[float, float] = (4.2, 2.8),
) -> Path:
    """Stacked bar of the variance decomposition.

    Expected dict keys (raw values, not fractions): ``var_y``, ``var_tabpfn``,
    ``var_f1``, ``cov_tabpfn_f1``, ``var_resid``, optionally ``total_explained_fraction``.

    Sign-of-covariance handling
    ---------------------------
    The covariance term ``2·Cov(TabPFN, residual)`` can be negative, in which
    case the stacked bar visually "moves backwards" from the running base.
    We retain that single-bar layout (no separate positive/negative stacks) and
    instead emit an explicit caption beneath the bar so readers know to read
    the term as a correction. Choice rationale: the caption is one-line and
    keeps the bar geometry interpretable; splitting into two stacks would
    double the legend entries for what is, in practice, almost always a small
    correction (≤ a few percent of Var(y)).
    """
    apply_theme()
    var_y = var_components["var_y"]
    cov_frac = 2 * var_components["cov_tabpfn_f1"] / var_y
    components = [
        ("Var(ŷ_TabPFN)", var_components["var_tabpfn"] / var_y, PALETTES["categorical"][0]),
        ("Var(f̂_residual)", var_components["var_f1"] / var_y, PALETTES["categorical"][3]),
        ("2·Cov(TabPFN, residual)", cov_frac, PALETTES["categorical"][2]),
        ("Var(residual ε)", var_components["var_resid"] / var_y, PALETTES["categorical"][7]),
    ]
    fig, ax = plt.subplots(figsize=figsize)
    base = 0.0
    for name, frac, color in components:
        ax.barh([0], [frac], left=base, height=0.5, color=color, label=f"{name}: {100*frac:.1f}%",
                edgecolor="white", linewidth=0.6)
        if frac > 0.04:
            ax.text(base + frac / 2, 0, f"{100*frac:.1f}%",
                    ha="center", va="center", fontsize=7, color="white")
        base += frac
    explained = var_components.get("total_explained_fraction")
    title = "Variance decomposition (Var(y) split)"
    if explained is not None:
        title += f"\ntotal_explained_fraction = {100*explained:.1f}%"
    ax.set_title(title)
    # X limits must accommodate negative cov: use min/max over [0, base, cov_left_edge].
    x_left = min(0.0, base, base - max(0.0, -cov_frac))
    x_right = max(1.05, base * 1.02, base + max(0.0, cov_frac))
    ax.set_xlim(x_left - 0.02 if x_left < 0 else 0, x_right)
    ax.set_xlabel("fraction of Var(y)")
    ax.set_yticks([])
    if cov_frac < 0:
        ax.text(0.5, -0.55,
                "Negative covariance shown as right-edge correction term "
                "(bar segment overlaps preceding stacks).",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=6.5, style="italic", color="#444444")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=True, fontsize=7)
    fmt_axes(ax)
    paths = save_fig(fig, out_stem, bbox_inches="tight")
    plt.close(fig)
    return paths[0]


def plot_per_subject_delta_scatter(
    y_tabpfn: np.ndarray,
    y_residual: np.ndarray,
    pathology: np.ndarray,
    out_stem: str | Path,
    *,
    pathology_label: str = "Global pathology (gpath)",
    figsize: tuple[float, float] = (5.0, 4.0),
) -> Path:
    """Scatter of f̂_residual vs ŷ_TabPFN, color-coded by pathology covariate."""
    apply_theme()
    fig, ax = plt.subplots(figsize=figsize)
    cmap = PALETTES["sequential"]
    finite = np.isfinite(pathology) & np.isfinite(y_residual) & np.isfinite(y_tabpfn)
    sc = ax.scatter(y_tabpfn[finite], y_residual[finite], c=pathology[finite],
                    cmap=cmap, s=18, alpha=0.75, edgecolors="white", linewidth=0.4)
    ax.axhline(0, ls="--", color="black", lw=0.6, alpha=0.6)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label(pathology_label)
    ax.set_xlabel("ŷ_TabPFN")
    ax.set_ylabel("f̂_residual")
    ax.set_title("Residual correction vs TabPFN baseline\n(color = pathology)")
    fmt_axes(ax)
    paths = save_fig(fig, out_stem)
    plt.close(fig)
    return paths[0]


def plot_residual_histogram_overlay(
    y_true: np.ndarray,
    y_tabpfn: np.ndarray,
    y_residual: np.ndarray,
    out_stem: str | Path,
    *,
    figsize: tuple[float, float] = (5.0, 3.5),
) -> Path:
    """Overlay: |y - ŷ_TabPFN| (TabPFN-only error) vs |y - ŷ_composite| (with residual head)."""
    apply_theme()
    err_tabpfn = y_true - y_tabpfn
    err_composite = y_true - (y_tabpfn + y_residual)
    fig, ax = plt.subplots(figsize=figsize)
    bins = np.linspace(min(err_tabpfn.min(), err_composite.min()),
                       max(err_tabpfn.max(), err_composite.max()), 35)
    ax.hist(err_tabpfn, bins=bins, color=PALETTES["categorical"][0], alpha=0.55,
            label=f"TabPFN-only (MAE={np.mean(np.abs(err_tabpfn)):.3f})",
            edgecolor="white", linewidth=0.4)
    ax.hist(err_composite, bins=bins, color=PALETTES["categorical"][3], alpha=0.55,
            label=f"Composite (MAE={np.mean(np.abs(err_composite)):.3f})",
            edgecolor="white", linewidth=0.4)
    ax.axvline(0, ls="--", color="black", lw=0.6, alpha=0.6)
    ax.set_xlabel("residual error  y − ŷ")
    ax.set_ylabel("subjects")
    ax.set_title("Per-subject prediction-error distribution\nbefore (TabPFN-only) vs after (composite)")
    ax.legend(loc="upper left", frameon=True, fontsize=8)
    fmt_axes(ax)
    paths = save_fig(fig, out_stem)
    plt.close(fig)
    return paths[0]


# Helper
def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Finite-mask filter then ``sklearn.metrics.r2_score``.

    Project convention (cf. ``make_baseline_table.py``, ``make_figures.py``);
    matches sklearn semantics so panel R² values match the baseline tables.
    """
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    if finite.sum() < 2:
        return float("nan")
    return float(r2_score(y_true[finite], y_pred[finite]))
