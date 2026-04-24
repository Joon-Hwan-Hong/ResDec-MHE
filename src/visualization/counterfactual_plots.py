"""Counterfactual-explanation plots: per-subject success + feature-level aggregates.

Functions:

  - ``plot_counterfactual_movement`` — per-subject bar of |Δy| / |target − init|
    (fraction-of-target-reached), colored by regime (resilient vs vulnerable).
    Overlays a horizontal reference at the group mean.

  - ``plot_counterfactual_ct_aggregate`` — per-cell-type share of top-K feature
    counts aggregated across subjects. Horizontal bar chart sorted descending.
    Colors by viridis magnitude.

  - ``plot_counterfactual_top_pairs`` — top-N (cell type, gene) pairs by
    subject-count, horizontal bar chart with labels. Useful for the paper's
    per-method top-feature table.

All functions take pre-parsed numpy / Counter / DataFrame data; orchestrators
handle JSON → data conversion.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from src.visualization.theme import PALETTES, fmt_axes, save_fig


def plot_counterfactual_movement(
    subject_ids: Sequence[str],
    fraction_of_target: Sequence[float],
    regime: Sequence[str],
    *,
    figsize: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Per-subject bar chart of fraction-of-target reached.

    Bars are sorted descending by fraction. Each bar is colored by regime
    (resilient = tab10[0], vulnerable = tab10[1]). Mean line drawn across
    all subjects.
    """
    sids = list(subject_ids)
    frac = np.asarray(fraction_of_target, dtype=np.float64)
    if len(sids) != len(frac):
        raise ValueError("length mismatch: subjects vs fractions")
    if len(sids) == 0:
        raise ValueError("no subjects supplied")

    order = np.argsort(frac)[::-1]
    cmap = PALETTES["categorical"]
    color_res = cmap[0]
    color_vul = cmap[1]
    colors = [color_res if regime[i] == "resilient" else color_vul for i in order]
    x_labels = [sids[i] for i in order]
    frac_sorted = frac[order]

    if figsize is None:
        figsize = (max(6.5, 0.3 * len(sids)), 4.0)
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(sids))
    ax.bar(x, frac_sorted, color=colors, edgecolor="black", linewidth=0.3)
    ax.axhline(
        float(frac.mean()), color="black", linestyle="--", linewidth=0.8,
        label=f"mean = {frac.mean():.1%}",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=60, ha="right", fontsize=6)
    ax.set_ylabel("Fraction of target reached\n(|Δy| / |target − init|)")
    ax.set_ylim(0, 1.05)
    fmt_axes(ax)
    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(facecolor=color_res, edgecolor="black", label="resilient"),
            Patch(facecolor=color_vul, edgecolor="black", label="vulnerable"),
            plt.Line2D([], [], color="black", linestyle="--", label=f"mean {frac.mean():.1%}"),
        ],
        loc="upper right", fontsize=7,
    )
    fig.tight_layout()
    if save_path is not None:
        save_fig(fig, save_path)
    return fig


def plot_counterfactual_ct_aggregate(
    ct_counts: Mapping[str, int] | Counter,
    *,
    total: int | None = None,
    figsize: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Horizontal bar chart of per-cell-type aggregate counts from top-K features.

    Width proportional to count; bars sorted descending. If ``total`` is
    supplied, right-axis annotates percent share.
    """
    items = sorted(ct_counts.items(), key=lambda kv: kv[1], reverse=True)
    if not items:
        raise ValueError("empty counter")
    names = [k for k, _ in items]
    counts = np.asarray([v for _, v in items], dtype=np.int64)
    pct = counts / max(1, total) if total else counts / counts.sum()

    if figsize is None:
        figsize = (6.5, max(3.5, 0.22 * len(names)))
    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.get_cmap(PALETTES["sequential"])
    norm_max = float(counts.max())
    colors = [cmap(float(v) / norm_max) for v in counts]
    y = np.arange(len(names))
    ax.barh(y, counts, color=colors, edgecolor="black", linewidth=0.3)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Top-K feature-count aggregate")
    fmt_axes(ax)
    for i, (c, p) in enumerate(zip(counts, pct)):
        ax.text(c + max(counts) * 0.015, i, f"{c}  ({p:.0%})",
                va="center", fontsize=6)
    ax.set_xlim(0, counts.max() * 1.25)
    fig.tight_layout()
    if save_path is not None:
        save_fig(fig, save_path)
    return fig


def plot_counterfactual_top_pairs(
    pair_counts: Mapping[tuple[str, str], int] | Counter,
    top_n: int = 20,
    *,
    figsize: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Top-N (cell type, gene) pairs by subject count. Horizontal bar chart."""
    items = sorted(pair_counts.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    if not items:
        raise ValueError("empty pair counter")
    labels = [f"{ct} × {gene}" for (ct, gene), _ in items]
    counts = np.asarray([v for _, v in items], dtype=np.int64)

    if figsize is None:
        figsize = (6.5, max(3.5, 0.3 * len(labels)))
    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.get_cmap(PALETTES["sequential"])
    norm_max = float(counts.max())
    colors = [cmap(float(v) / norm_max) for v in counts]
    y = np.arange(len(labels))
    ax.barh(y, counts, color=colors, edgecolor="black", linewidth=0.3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("# subjects featuring this (CT, gene) in top-K")
    fmt_axes(ax)
    fig.tight_layout()
    if save_path is not None:
        save_fig(fig, save_path)
    return fig
