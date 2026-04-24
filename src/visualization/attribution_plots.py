"""Attribution plots: per-subject and aggregate views of model attribution signals.

Functions:

  - ``plot_subject_waterfall`` — single-subject Captum waterfall from TabPFN
    base through top-K (CT, gene) attribution steps to the composite prediction.

  - ``plot_tabpfn_vs_residual_stack`` — per-subject stacked bars of
    (TabPFN_pred, signed residual contribution), sorted by composite ŷ.
    Visualizes how much each model component contributes per subject.

  - ``plot_resilience_signature_radar`` — radar chart with top-N attributed
    genes as axes, polygon per residual quartile (resilient → vulnerable).

  - ``plot_per_quintile_attribution`` — heatmap of cell type × prediction-quintile
    mean |attribution|; reveals reasoning shifts across the prediction range.

All functions take pre-loaded numpy / pandas data + standard args and return
``matplotlib.figure.Figure``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.visualization.theme import PALETTES, baseline_color, fmt_axes, save_fig


def plot_subject_waterfall(
    subject_id: str,
    captum_attrs: np.ndarray,
    cell_type_names: Sequence[str],
    gene_names: Sequence[str],
    tabpfn_pred: float,
    composite_pred: float,
    true_y: float,
    *,
    top_n: int = 12,
    figsize: tuple[float, float] = (5.5, 4.0),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Waterfall: TabPFN base + top-N (CT, gene) attribution steps → composite.

    Each step shows how a (cell-type, gene) attribution moves the prediction
    toward the composite. The dashed horizontal line marks the true label.
    """
    flat_attrs = captum_attrs.ravel()
    if flat_attrs.size == 0 or not np.isfinite(flat_attrs).any():
        raise ValueError("captum_attrs empty or all-NaN")
    n_ct, n_gene = captum_attrs.shape
    abs_idx = np.argsort(np.abs(flat_attrs))[::-1][:top_n]
    ct_idx = abs_idx // n_gene
    gn_idx = abs_idx % n_gene
    contribs = flat_attrs[abs_idx]
    labels = [
        f"{cell_type_names[c]} × {gene_names[g]}"
        for c, g in zip(ct_idx, gn_idx)
    ]
    other_contrib = (composite_pred - tabpfn_pred) - contribs.sum()

    fig, ax = plt.subplots(figsize=figsize)
    bar_x = list(range(len(labels) + 3))
    bar_y = [tabpfn_pred] + list(contribs) + [other_contrib, composite_pred]
    bar_colors = (
        [baseline_color("TabPFN-2.6")]
        + ["#2ca02c" if c > 0 else "#d62728" for c in contribs]
        + ["#bbbbbb", baseline_color("ResDec-MHE")]
    )
    bar_labels = ["TabPFN base"] + labels + ["(other)", "Composite"]
    cum = 0.0
    for i, val in enumerate(bar_y):
        if i == 0 or i == len(bar_y) - 1:
            ax.bar(bar_x[i], val, bottom=0, width=0.7,
                   color=bar_colors[i], edgecolor="white",
                   linewidth=0.5, zorder=2)
            cum = val if i == 0 else cum
        else:
            ax.bar(bar_x[i], val, bottom=cum, width=0.7,
                   color=bar_colors[i], edgecolor="white",
                   linewidth=0.5, zorder=2)
            cum += val
    ax.axhline(y=true_y, color="black", linewidth=1.0, linestyle="--",
               label=f"True y = {true_y:.3f}", zorder=3)
    ax.set_xticks(bar_x)
    ax.set_xticklabels(bar_labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Resilience score")
    fmt_axes(ax)
    ax.legend(loc="upper left", fontsize=7)
    ax.text(
        0.02, 0.97,
        f"subject {subject_id}\nTabPFN={tabpfn_pred:.3f}, composite={composite_pred:.3f}",
        transform=ax.transAxes, fontsize=7, va="top",
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="none"),
    )
    if save_path is not None:
        save_fig(fig, save_path)
    return fig


def plot_tabpfn_vs_residual_stack(
    subject_ids: np.ndarray,
    tabpfn_preds: np.ndarray,
    composite_preds: np.ndarray,
    true_y: np.ndarray,
    *,
    figsize: tuple[float, float] = (7.0, 3.5),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Per-subject stacked bar: TabPFN_pred + signed residual, sorted by composite.

    Highlights subjects where the residual head matters for the prediction.
    True y overlaid as black dots.
    """
    n = len(subject_ids)
    if n == 0:
        raise ValueError("no subjects")
    residual_contrib = composite_preds - tabpfn_preds
    order = np.argsort(composite_preds)
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(
        x, tabpfn_preds[order], color=baseline_color("TabPFN-2.6"), width=1.0,
        label="TabPFN-2.6 base", linewidth=0,
    )
    pos = residual_contrib[order].copy()
    neg = residual_contrib[order].copy()
    pos[pos < 0] = 0
    neg[neg > 0] = 0
    ax.bar(x, pos, bottom=tabpfn_preds[order], color="#2ca02c", width=1.0,
           label="Residual head (+)", linewidth=0)
    ax.bar(x, neg, bottom=tabpfn_preds[order], color="#d62728", width=1.0,
           label="Residual head (−)", linewidth=0)
    ax.scatter(x, true_y[order], s=2, color="black", alpha=0.6,
               zorder=3, label="True y")
    ax.set_xlabel(f"Subjects (n={n}, sorted by composite ŷ)")
    ax.set_ylabel("Resilience score")
    ax.set_xlim(-0.5, n - 0.5)
    fmt_axes(ax)
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    if save_path is not None:
        save_fig(fig, save_path)
    return fig


def plot_resilience_signature_radar(
    captum_attrs_per_subject: np.ndarray,
    residuals_per_subject: np.ndarray,
    cell_type_names: Sequence[str],
    gene_names: Sequence[str],
    *,
    top_n_genes: int = 8,
    n_quartiles: int = 4,
    figsize: tuple[float, float] = (5.0, 5.0),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Radar: top-N attribution genes as axes, polygon per residual quartile."""
    n_subj, n_ct, n_gene = captum_attrs_per_subject.shape
    if n_subj == 0 or len(residuals_per_subject) != n_subj:
        raise ValueError("attribution / residual length mismatch or empty")
    global_abs = np.abs(captum_attrs_per_subject).sum(axis=(0, 1))
    top_g_idx = np.argsort(global_abs)[::-1][:top_n_genes]
    top_g_names = [str(gene_names[i]) for i in top_g_idx]

    finite = np.isfinite(residuals_per_subject)
    q_edges = np.quantile(residuals_per_subject[finite], np.linspace(0, 1, n_quartiles + 1))
    q_edges[0] -= 1e-9
    q_labels = pd.cut(residuals_per_subject, q_edges, labels=False, include_lowest=True)

    matrix = np.full((n_quartiles, top_n_genes), np.nan, dtype=np.float64)
    for q in range(n_quartiles):
        mask = (q_labels == q)
        if mask.sum() == 0:
            continue
        sub_attrs = captum_attrs_per_subject[mask]
        per_g = np.abs(sub_attrs).mean(axis=(0, 1))[top_g_idx]
        matrix[q] = per_g
    col_max = np.nanmax(matrix, axis=0, keepdims=True)
    col_max[col_max == 0] = 1.0
    matrix_norm = matrix / col_max

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"polar": True})
    angles = np.linspace(0, 2 * np.pi, top_n_genes, endpoint=False)
    angles_closed = np.concatenate([angles, angles[:1]])
    cmap = PALETTES["sequential"]
    for q in range(n_quartiles):
        vals = matrix_norm[q].copy()
        if not np.isfinite(vals).all():
            continue
        vals_closed = np.concatenate([vals, vals[:1]])
        color = cmap(q / max(1, n_quartiles - 1))
        label = f"Q{q+1}" + (" (resilient)" if q == 0 else
                              " (vulnerable)" if q == n_quartiles - 1 else "")
        ax.plot(angles_closed, vals_closed, color=color, linewidth=1.5, label=label)
        ax.fill(angles_closed, vals_closed, color=color, alpha=0.15)
    ax.set_xticks(angles)
    ax.set_xticklabels(top_g_names, fontsize=7)
    ax.set_yticks([])
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.0), fontsize=7)
    if save_path is not None:
        save_fig(fig, save_path)
    return fig


def plot_per_quintile_attribution(
    captum_attrs_per_subject: np.ndarray,
    composite_preds: np.ndarray,
    cell_type_names: Sequence[str],
    *,
    n_quintiles: int = 5,
    top_n_cts: int = 15,
    figsize: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Heatmap: top-CT × prediction-quintile mean |attribution|."""
    n_subj, n_ct, _ = captum_attrs_per_subject.shape
    if n_subj == 0:
        raise ValueError("no subjects")
    finite = np.isfinite(composite_preds)
    q_edges = np.quantile(composite_preds[finite], np.linspace(0, 1, n_quintiles + 1))
    q_edges[0] -= 1e-9
    q_labels = pd.cut(composite_preds, q_edges, labels=False, include_lowest=True)
    ct_total = np.abs(captum_attrs_per_subject).sum(axis=(0, 2))
    top_ct = np.argsort(ct_total)[::-1][:top_n_cts]

    matrix = np.full((top_n_cts, n_quintiles), np.nan, dtype=np.float64)
    for qi in range(n_quintiles):
        mask = (q_labels == qi)
        if mask.sum() == 0:
            continue
        attrs_q = captum_attrs_per_subject[mask]
        ct_attr_q = np.abs(attrs_q).mean(axis=(0, 2))[top_ct]
        matrix[:, qi] = ct_attr_q

    if figsize is None:
        figsize = (4.5, max(3.0, top_n_cts * 0.25))
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        matrix, ax=ax, cmap=PALETTES["sequential"],
        cbar_kws={"label": "mean |attribution|"},
        yticklabels=[str(cell_type_names[i]) for i in top_ct],
        xticklabels=[f"Q{q+1}" for q in range(n_quintiles)],
        linewidths=0.4, linecolor="white",
    )
    ax.set_xlabel("Composite ŷ quintile (Q1=lowest)")
    ax.set_ylabel("")
    if save_path is not None:
        save_fig(fig, save_path)
    return fig


def plot_attribution_stability_heatmap(
    captum_attrs: np.ndarray,
    fold_ids: np.ndarray,
    cell_type_names: Sequence[str],
    gene_names: Sequence[str],
    *,
    top_n_pairs: int = 30,
    figsize: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Cross-fold attribution rank stability heatmap.

    For the top-N (CT, gene) pairs by GLOBAL mean |attribution| (averaged
    across folds), show the rank within EACH fold as a heatmap. Stable
    pairs have low rank in all folds; pairs that pop up only in one fold
    have high rank-variance — highlighting which attributions reproduce.

    Parameters
    ----------
    captum_attrs
        Shape ``(n_subjects, n_ct, n_gene)`` per-subject attributions.
    fold_ids
        Shape ``(n_subjects,)`` integer fold ID per subject (0-indexed).
    cell_type_names, gene_names
        Names for axes.
    top_n_pairs
        How many top (CT, gene) pairs to display (sorted by global rank).
    """
    n_subj, n_ct, n_gene = captum_attrs.shape
    if n_subj == 0:
        raise ValueError("no subjects")
    folds = np.unique(fold_ids)
    n_folds = len(folds)
    # Per-fold mean |attribution| (CT, gene).
    per_fold = np.zeros((n_folds, n_ct, n_gene), dtype=np.float64)
    for fi, f in enumerate(folds):
        mask = fold_ids == f
        per_fold[fi] = np.abs(captum_attrs[mask]).mean(axis=0)
    # Global ranking by mean across folds.
    mean_attr = per_fold.mean(axis=0)
    flat = mean_attr.ravel()
    top_idx = np.argsort(flat)[::-1][:top_n_pairs]
    ct_idx = top_idx // n_gene
    gn_idx = top_idx % n_gene
    pair_labels = [
        f"{cell_type_names[c]} × {gene_names[g]}"
        for c, g in zip(ct_idx, gn_idx)
    ]
    # Per-fold rank (1-indexed) of each pair.
    ranks = np.zeros((top_n_pairs, n_folds), dtype=np.int64)
    for fi in range(n_folds):
        flat_fold = per_fold[fi].ravel()
        order = np.argsort(flat_fold)[::-1]
        rank_lookup = np.empty_like(order)
        rank_lookup[order] = np.arange(len(order))
        for pi, idx in enumerate(top_idx):
            ranks[pi, fi] = int(rank_lookup[idx]) + 1

    if figsize is None:
        figsize = (5.5, max(3.5, top_n_pairs * 0.18))
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        ranks, ax=ax, cmap=PALETTES["sequential"],
        cbar_kws={"label": "rank (1=top)"},
        yticklabels=pair_labels,
        xticklabels=[f"fold {int(f)}" for f in folds],
        annot=True, fmt="d", annot_kws={"size": 6},
        linewidths=0.4, linecolor="white",
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    if save_path is not None:
        save_fig(fig, save_path)
    return fig
