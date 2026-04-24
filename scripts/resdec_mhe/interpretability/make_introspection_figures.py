"""Additional paper figures (Tier 1 + Tier 2 + Tier 3-where-feasible).

Generates 10 figures using existing canonical artefacts + the project-wide
``src/visualization/theme`` style. Each ``make_figX_*`` function takes
pre-loaded data and returns a matplotlib Figure (or None on insufficient
data); the CLI loads + saves.

Figures produced:

  Tier 1 (paper-defining narrative):
    1. ``fig_subject_waterfall``         — single-subject Captum waterfall
       (TabPFN base → composite, per-(CT,gene) attribution steps)
    2. ``fig_tabpfn_vs_residual_stack``  — per-subject stacked bar of
       (TabPFN_pred, signed residual contribution), sorted by composite
    3. ``fig_head_attention_chord``      — head × cell-type attention as
       chord diagram (pyCirclize)
    4. ``fig_resilience_signature_radar``— top-8 attributed genes, polygon
       per residual quartile (resilient → vulnerable)

  Tier 2 (strong supporting):
    5. ``fig_attribution_stability``     — (CT, gene) × fold attribution
       rank stability heatmap
    6. ``fig_per_quintile_attribution``  — CT × prediction-quintile mean
       attribution shift heatmap
    7. ``fig_head_fingerprint_umap``     — UMAP of per-subject head×CT
       fingerprints, colored by residual quartile
    8. ``fig_calibration_overlay``       — TabPFN-only vs Composite
       reliability diagrams (two panels)
    9. ``fig_hgt_celltype_network``      — top-50 HGT edges as a CT graph

  Tier 3 (where feasible):
   10. ``fig_architecture_diagram``      — pure-matplotlib model schematic

Skipped here (require new compute / model hooks; landed later):
   - Captum-rank vs DE-rank scatter (needs DE output)
   - Per-stage activation cascade (needs model + hooks)
   - Head specialization bootstrap (light extra compute)
   - Loss landscape PCA (per-fold weights)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from src.visualization.theme import (  # noqa: E402
    PALETTES,
    apply_theme,
    baseline_color,
    fmt_axes,
    save_fig,
)


logger = logging.getLogger(__name__)


class SkipFigure(Exception):
    """Raised when a figure cannot be rendered due to missing data."""


# ----------------------------- TIER 1 ------------------------------------


def make_fig_subject_waterfall(
    subject_id: str,
    captum_attrs: np.ndarray,
    cell_type_names: Sequence[str],
    gene_names: Sequence[str],
    tabpfn_pred: float,
    composite_pred: float,
    true_y: float,
    *,
    top_n: int = 12,
):
    """Waterfall: TabPFN base + top-N (CT, gene) attribution steps → composite.

    Parameters
    ----------
    subject_id
        Subject identifier (for title).
    captum_attrs
        Shape ``(n_celltypes, n_genes)`` for THIS subject (signed).
    cell_type_names, gene_names
        Names for axes.
    tabpfn_pred, composite_pred, true_y
        Scalars.
    top_n
        Number of top |attribution| (CT, gene) steps to show.
    """
    flat_attrs = captum_attrs.ravel()
    if flat_attrs.size == 0 or not np.isfinite(flat_attrs).any():
        raise SkipFigure("captum_attrs empty or all-NaN")
    n_ct, n_gene = captum_attrs.shape
    abs_idx = np.argsort(np.abs(flat_attrs))[::-1][:top_n]
    ct_idx = abs_idx // n_gene
    gn_idx = abs_idx % n_gene
    contribs = flat_attrs[abs_idx]
    labels = [
        f"{cell_type_names[c]} × {gene_names[g]}"
        for c, g in zip(ct_idx, gn_idx)
    ]
    # The remaining attribution = composite - tabpfn - sum(top_n_contribs).
    other_contrib = (composite_pred - tabpfn_pred) - contribs.sum()

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    cumulative = tabpfn_pred
    bar_x = []
    bar_y = []
    bar_colors = []
    bar_labels = []
    bar_x.append(0)
    bar_y.append(tabpfn_pred)
    bar_colors.append(baseline_color("TabPFN-2.6"))
    bar_labels.append("TabPFN base")
    for i, (label, c) in enumerate(zip(labels, contribs)):
        x = i + 1
        bar_x.append(x)
        bar_y.append(c)
        bar_colors.append("#2ca02c" if c > 0 else "#d62728")
        bar_labels.append(label)
        # Connector line for waterfall visual.
        ax.plot(
            [x - 0.4, x - 0.4 + 0.8], [cumulative, cumulative],
            color="#888", linewidth=0.5, zorder=1,
        )
        cumulative += c
    bar_x.append(len(labels) + 1)
    bar_y.append(other_contrib)
    bar_colors.append("#bbbbbb")
    bar_labels.append("(other)")
    bar_x.append(len(labels) + 2)
    bar_y.append(composite_pred)
    bar_colors.append(baseline_color("ResDec-MHE"))
    bar_labels.append("Composite")

    # First bar: full bar from 0 to tabpfn_pred.
    # Step bars: from cumulative_before to cumulative_before+contrib.
    cum = 0.0
    rects = []
    bar_width = 0.7
    for i, val in enumerate(bar_y):
        if i == 0 or i == len(bar_y) - 1:
            # Anchor bars: from 0 to value.
            rect = ax.bar(bar_x[i], val, bottom=0, width=bar_width,
                          color=bar_colors[i], edgecolor="white",
                          linewidth=0.5, zorder=2)
            cum = val if i == 0 else cum
        else:
            rect = ax.bar(bar_x[i], val, bottom=cum, width=bar_width,
                          color=bar_colors[i], edgecolor="white",
                          linewidth=0.5, zorder=2)
            cum += val
        rects.append(rect)

    ax.axhline(y=true_y, color="black", linewidth=1.0, linestyle="--",
               label=f"True y = {true_y:.3f}", zorder=3)
    ax.set_xticks(bar_x)
    ax.set_xticklabels(bar_labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Resilience score")
    ax.set_xlabel("")
    fmt_axes(ax)
    ax.legend(loc="upper left", fontsize=7)
    ax.text(
        0.02, 0.97,
        f"subject {subject_id}\nTabPFN={tabpfn_pred:.3f}, composite={composite_pred:.3f}",
        transform=ax.transAxes, fontsize=7, va="top",
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="none"),
    )
    return fig


def make_fig_tabpfn_vs_residual_stack(
    subject_ids: np.ndarray,
    tabpfn_preds: np.ndarray,
    composite_preds: np.ndarray,
    true_y: np.ndarray,
):
    """Per-subject stacked bar: TabPFN_pred + (composite - TabPFN), sorted by composite.

    Highlights subjects where the residual head matters for the prediction.
    """
    n = len(subject_ids)
    if n == 0:
        raise SkipFigure("no subjects")
    residual_contrib = composite_preds - tabpfn_preds
    order = np.argsort(composite_preds)
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    # TabPFN pred as baseline bar.
    ax.bar(
        x, tabpfn_preds[order], color=baseline_color("TabPFN-2.6"), width=1.0,
        label="TabPFN-2.6 base", linewidth=0,
    )
    # Residual contribution stacked on top (positive: green, negative: red).
    pos = residual_contrib[order].copy()
    neg = residual_contrib[order].copy()
    pos[pos < 0] = 0
    neg[neg > 0] = 0
    ax.bar(
        x, pos, bottom=tabpfn_preds[order], color="#2ca02c", width=1.0,
        label="Residual head (+)", linewidth=0,
    )
    ax.bar(
        x, neg, bottom=tabpfn_preds[order], color="#d62728", width=1.0,
        label="Residual head (−)", linewidth=0,
    )
    # True y as overlaid dots.
    ax.scatter(
        x, true_y[order], s=2, color="black", alpha=0.6, zorder=3,
        label="True y",
    )
    ax.set_xlabel(f"Subjects (n={n}, sorted by composite ŷ)")
    ax.set_ylabel("Resilience score")
    ax.set_xlim(-0.5, n - 0.5)
    fmt_axes(ax)
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    return fig


def make_fig_head_attention_chord(
    head_attention: np.ndarray,
    cell_type_names: Sequence[str],
    *,
    top_k_cts: int = 12,
):
    """Chord diagram: 4 heads × top_k_cts cell types, chord width = mean attention.

    Uses ``pyCirclize.Circos.chord_diagram`` (matrix input form).
    """
    try:
        from pycirclize import Circos
    except ImportError as exc:
        raise SkipFigure("pyCirclize not installed") from exc
    n_subj, n_heads, n_ct = head_attention.shape
    if n_subj == 0:
        raise SkipFigure("no subjects in head_attention")
    mean_attn = head_attention.mean(axis=0)  # (n_heads, n_ct)

    head_labels = [f"H{i}" for i in range(n_heads)]
    ct_totals = mean_attn.sum(axis=0)
    top_ct_idx = np.argsort(ct_totals)[::-1][:top_k_cts]
    ct_labels = [str(cell_type_names[i]) for i in top_ct_idx]
    matrix = mean_attn[:, top_ct_idx]  # (n_heads, top_k_cts)
    df = pd.DataFrame(matrix, index=head_labels, columns=ct_labels)
    # Convert from rectangular to a square symmetric matrix expected by chord_diagram.
    nodes = head_labels + ct_labels
    n_nodes = len(nodes)
    sq = pd.DataFrame(
        np.zeros((n_nodes, n_nodes)), index=nodes, columns=nodes,
    )
    for h in head_labels:
        for c in ct_labels:
            v = df.loc[h, c]
            sq.loc[h, c] = v
            sq.loc[c, h] = v

    # Color: heads use Dark2, CTs use viridis sequential.
    cmap = {}
    for i, h in enumerate(head_labels):
        cmap[h] = PALETTES["categorical_paired"][i % len(PALETTES["categorical_paired"])]
    for j, c in enumerate(ct_labels):
        cmap[c] = PALETTES["sequential"](float(j) / max(1, len(ct_labels) - 1))

    circos = Circos.chord_diagram(
        sq, space=3, cmap=cmap, label_kws={"size": 7},
    )
    fig = circos.plotfig(figsize=(5.5, 5.5))
    return fig


def make_fig_resilience_signature_radar(
    captum_attrs_per_subject: np.ndarray,
    residuals_per_subject: np.ndarray,
    cell_type_names: Sequence[str],
    gene_names: Sequence[str],
    *,
    top_n_genes: int = 8,
    n_quartiles: int = 4,
):
    """Radar: top-N attribution genes as axes, polygon per residual quartile.

    Each quartile = mean attribution magnitude for the gene (averaged across
    subjects in that quartile and across all cell types).
    """
    n_subj, n_ct, n_gene = captum_attrs_per_subject.shape
    if n_subj == 0 or len(residuals_per_subject) != n_subj:
        raise SkipFigure("attribution / residual length mismatch or empty")
    # Find top-N genes by global |attribution| summed across subjects + CTs.
    global_abs = np.abs(captum_attrs_per_subject).sum(axis=(0, 1))
    top_g_idx = np.argsort(global_abs)[::-1][:top_n_genes]
    top_g_names = [str(gene_names[i]) for i in top_g_idx]

    # Quartile labels by ascending residual.
    finite = np.isfinite(residuals_per_subject)
    q_edges = np.quantile(residuals_per_subject[finite], np.linspace(0, 1, n_quartiles + 1))
    q_edges[0] -= 1e-9
    q_labels = pd.cut(residuals_per_subject, q_edges, labels=False, include_lowest=True)

    # Per-quartile mean attribution magnitude per top gene.
    matrix = np.full((n_quartiles, top_n_genes), np.nan, dtype=np.float64)
    for q in range(n_quartiles):
        mask = (q_labels == q)
        if mask.sum() == 0:
            continue
        sub_attrs = captum_attrs_per_subject[mask]  # (m, n_ct, n_gene)
        # Mean |attr| per top gene, averaged across CTs.
        per_g = np.abs(sub_attrs).mean(axis=(0, 1))[top_g_idx]
        matrix[q] = per_g

    # Normalize per gene for visual comparability across genes.
    col_max = np.nanmax(matrix, axis=0, keepdims=True)
    col_max[col_max == 0] = 1.0
    matrix_norm = matrix / col_max

    fig, ax = plt.subplots(figsize=(5.0, 5.0), subplot_kw={"polar": True})
    angles = np.linspace(0, 2 * np.pi, top_n_genes, endpoint=False)
    angles_closed = np.concatenate([angles, angles[:1]])
    cmap = PALETTES["sequential"]
    for q in range(n_quartiles):
        vals = matrix_norm[q].copy()
        if not np.isfinite(vals).all():
            continue
        vals_closed = np.concatenate([vals, vals[:1]])
        color = cmap(q / max(1, n_quartiles - 1))
        label = f"Q{q+1}" + (" (resilient)" if q == 0 else " (vulnerable)" if q == n_quartiles - 1 else "")
        ax.plot(angles_closed, vals_closed, color=color, linewidth=1.5, label=label)
        ax.fill(angles_closed, vals_closed, color=color, alpha=0.15)
    ax.set_xticks(angles)
    ax.set_xticklabels(top_g_names, fontsize=7)
    ax.set_yticks([])
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.0), fontsize=7)
    return fig


# ----------------------------- TIER 2 ------------------------------------


def make_fig_attribution_stability(
    per_fold_attrs: np.ndarray,
    cell_type_names: Sequence[str],
    gene_names: Sequence[str],
    *,
    top_n_pairs: int = 30,
):
    """Heatmap of (CT, gene) × fold attribution rank.

    Parameters
    ----------
    per_fold_attrs
        Shape ``(n_folds, n_celltypes, n_genes)`` global mean |attribution|
        per (CT, gene) per fold.
    """
    n_folds, n_ct, n_gene = per_fold_attrs.shape
    if n_folds == 0:
        raise SkipFigure("no folds")
    # Global ranking (mean across folds) → pick top_n_pairs.
    mean_attr = per_fold_attrs.mean(axis=0)  # (n_ct, n_gene)
    flat = mean_attr.ravel()
    top_idx = np.argsort(flat)[::-1][:top_n_pairs]
    ct_idx = top_idx // n_gene
    gn_idx = top_idx % n_gene
    pair_labels = [
        f"{cell_type_names[c]} × {gene_names[g]}"
        for c, g in zip(ct_idx, gn_idx)
    ]
    # For each pair, rank within each fold.
    ranks = np.full((top_n_pairs, n_folds), np.nan)
    for f in range(n_folds):
        flat_fold = per_fold_attrs[f].ravel()
        order = np.argsort(flat_fold)[::-1]
        rank_lookup = np.empty_like(order)
        rank_lookup[order] = np.arange(len(order))
        for pi, idx in enumerate(top_idx):
            ranks[pi, f] = int(rank_lookup[idx]) + 1  # 1-indexed

    fig, ax = plt.subplots(figsize=(5.5, max(3.5, top_n_pairs * 0.18)))
    sns.heatmap(
        ranks, ax=ax, cmap=PALETTES["sequential"], cbar_kws={"label": "rank"},
        yticklabels=pair_labels,
        xticklabels=[f"fold {i}" for i in range(n_folds)],
        annot=True, fmt=".0f", annot_kws={"size": 6},
        linewidths=0.4, linecolor="white",
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    return fig


def make_fig_per_quintile_attribution(
    captum_attrs_per_subject: np.ndarray,
    composite_preds: np.ndarray,
    cell_type_names: Sequence[str],
    *,
    n_quintiles: int = 5,
    top_n_cts: int = 15,
):
    """Heatmap: top-CT × prediction-quintile mean |attribution|."""
    n_subj, n_ct, _ = captum_attrs_per_subject.shape
    if n_subj == 0:
        raise SkipFigure("no subjects")
    finite = np.isfinite(composite_preds)
    q_edges = np.quantile(composite_preds[finite], np.linspace(0, 1, n_quintiles + 1))
    q_edges[0] -= 1e-9
    q_labels = pd.cut(composite_preds, q_edges, labels=False, include_lowest=True)
    # Per-CT total |attr| for top-CT selection.
    ct_total = np.abs(captum_attrs_per_subject).sum(axis=(0, 2))
    top_ct = np.argsort(ct_total)[::-1][:top_n_cts]

    matrix = np.full((top_n_cts, n_quintiles), np.nan, dtype=np.float64)
    for qi in range(n_quintiles):
        mask = (q_labels == qi)
        if mask.sum() == 0:
            continue
        attrs_q = captum_attrs_per_subject[mask]  # (m, n_ct, n_gene)
        ct_attr_q = np.abs(attrs_q).mean(axis=(0, 2))[top_ct]
        matrix[:, qi] = ct_attr_q

    fig, ax = plt.subplots(figsize=(4.5, max(3.0, top_n_cts * 0.25)))
    sns.heatmap(
        matrix, ax=ax, cmap=PALETTES["sequential"],
        cbar_kws={"label": "mean |attribution|"},
        yticklabels=[str(cell_type_names[i]) for i in top_ct],
        xticklabels=[f"Q{q+1}" for q in range(n_quintiles)],
        linewidths=0.4, linecolor="white",
    )
    ax.set_xlabel("Composite ŷ quintile (Q1=lowest)")
    ax.set_ylabel("")
    return fig


def make_fig_head_fingerprint_umap(
    head_fingerprints: np.ndarray,
    residuals: np.ndarray,
    *,
    n_quartiles: int = 4,
    seed: int = 42,
):
    """UMAP of per-subject (n_heads * n_celltypes) flattened head fingerprints."""
    try:
        import umap  # type: ignore
    except ImportError as exc:
        raise SkipFigure("umap-learn not installed") from exc
    n_subj = head_fingerprints.shape[0]
    if n_subj == 0:
        raise SkipFigure("no subjects")
    flat = head_fingerprints.reshape(n_subj, -1)
    finite = np.isfinite(flat).all(axis=1) & np.isfinite(residuals)
    if finite.sum() < 30:
        raise SkipFigure("too few finite subjects (<30)")
    flat = flat[finite]
    res = residuals[finite]

    reducer = umap.UMAP(
        n_neighbors=min(30, max(5, n_subj // 10)),
        min_dist=0.3, random_state=seed,
    )
    emb = reducer.fit_transform(flat)
    q_edges = np.quantile(res, np.linspace(0, 1, n_quartiles + 1))
    q_edges[0] -= 1e-9
    q_labels = pd.cut(res, q_edges, labels=False, include_lowest=True)
    cmap = PALETTES["sequential"]

    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    for q in range(n_quartiles):
        mask = q_labels == q
        ax.scatter(
            emb[mask, 0], emb[mask, 1],
            c=[cmap(q / max(1, n_quartiles - 1))], s=14, alpha=0.75,
            edgecolor="white", linewidth=0.4,
            label=f"Q{q+1}",
        )
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    fmt_axes(ax)
    ax.legend(loc="upper right", fontsize=7, title="Residual quartile")
    return fig


def make_fig_calibration_overlay(
    tabpfn_per_fold: list[tuple[np.ndarray, np.ndarray]],
    composite_per_fold: list[tuple[np.ndarray, np.ndarray]],
    *,
    n_bins: int = 10,
):
    """Two-panel reliability diagram: TabPFN-only vs Composite."""
    if not tabpfn_per_fold or not composite_per_fold:
        raise SkipFigure("no per-fold predictions")

    def reliability(y_true_all, y_pred_all, n_bins):
        bin_edges = np.quantile(y_pred_all, np.linspace(0, 1, n_bins + 1))
        bin_edges[0] -= 1e-9
        labels = pd.cut(y_pred_all, bin_edges, labels=False, include_lowest=True)
        means_pred = []
        means_true = []
        for b in range(n_bins):
            mask = labels == b
            if mask.sum() == 0:
                continue
            means_pred.append(float(y_pred_all[mask].mean()))
            means_true.append(float(y_true_all[mask].mean()))
        return np.array(means_pred), np.array(means_true)

    yt_t = np.concatenate([t for t, _ in tabpfn_per_fold])
    yp_t = np.concatenate([p for _, p in tabpfn_per_fold])
    yt_c = np.concatenate([t for t, _ in composite_per_fold])
    yp_c = np.concatenate([p for _, p in composite_per_fold])
    mp_t, mt_t = reliability(yt_t, yp_t, n_bins)
    mp_c, mt_c = reliability(yt_c, yp_c, n_bins)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5), sharey=True, sharex=True)
    for ax, (mp, mt, name, color) in zip(
        axes,
        [
            (mp_t, mt_t, "TabPFN-2.6", baseline_color("TabPFN-2.6")),
            (mp_c, mt_c, "Composite (ResDec-MHE)", baseline_color("ResDec-MHE")),
        ],
    ):
        lo = float(min(mp.min(), mt.min()))
        hi = float(max(mp.max(), mt.max()))
        ax.plot([lo, hi], [lo, hi], color="#888", linewidth=0.6, linestyle="--", zorder=1)
        ax.scatter(mp, mt, color=color, s=22, edgecolor="white", linewidth=0.6, zorder=3)
        ax.plot(mp, mt, color=color, linewidth=1.2, zorder=2)
        ax.set_xlabel(f"Mean predicted ({name})")
        fmt_axes(ax)
    axes[0].set_ylabel("Mean true")
    return fig


def make_fig_hgt_celltype_network(
    edge_attention_df: pd.DataFrame,
    *,
    top_k_edges: int = 50,
    edge_type_col: str = "edge_type_name",
    sender_col: str = "source_ct",
    receiver_col: str = "target_ct",
    weight_col: str = "mean_attention",
):
    """Network: top-K HGT edges as a directed graph between cell types."""
    try:
        import networkx as nx
    except ImportError as exc:
        raise SkipFigure("networkx not installed") from exc
    if edge_attention_df.empty:
        raise SkipFigure("empty edge_attention_df")
    df = edge_attention_df.sort_values(weight_col, ascending=False).head(top_k_edges)
    G = nx.DiGraph()
    edge_types = (
        sorted(df[edge_type_col].unique()) if edge_type_col in df.columns else ["edge"]
    )
    edge_color_map = {
        et: PALETTES["categorical_paired"][i % len(PALETTES["categorical_paired"])]
        for i, et in enumerate(edge_types)
    }
    for _, row in df.iterrows():
        s = row[sender_col]
        r = row[receiver_col]
        et = row.get(edge_type_col, "edge")
        w = float(row[weight_col])
        G.add_edge(s, r, weight=w, color=edge_color_map.get(et, "#888"))
    pos = nx.spring_layout(G, seed=42, k=1.5 / max(1, len(G) ** 0.5))
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    colors = [G[u][v]["color"] for u, v in G.edges()]
    if not weights:
        raise SkipFigure("no edges after filtering")
    w_arr = np.array(weights)
    w_norm = (w_arr - w_arr.min()) / (max(1e-9, w_arr.max() - w_arr.min()))
    nx.draw_networkx_nodes(
        G, pos, node_color="#cccccc", node_size=300, edgecolors="black",
        linewidths=0.6, ax=ax,
    )
    nx.draw_networkx_edges(
        G, pos, edge_color=colors, width=0.8 + 2.5 * w_norm,
        arrows=True, arrowsize=10, alpha=0.7, ax=ax,
    )
    nx.draw_networkx_labels(G, pos, font_size=6, ax=ax)
    # Legend: edge types.
    handles = [
        plt.Line2D([0], [0], color=color, lw=2, label=et)
        for et, color in edge_color_map.items()
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=7, title="Edge type")
    ax.set_axis_off()
    return fig


# ----------------------------- TIER 3 (feasible) -------------------------


def make_fig_architecture_diagram():
    """Pure-matplotlib model schematic of ResDec-MHE.

    Boxes for each component (HGT, CellTransformer, PMA, RegionHandler,
    PathologyAttention, gene_gate, ResDecMHE head, TabPFN base) with arrows
    showing data flow. No real data dependency.
    """
    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    def box(x, y, w, h, label, color):
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="black",
                             linewidth=0.8, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=7, fontweight="bold", zorder=3)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=0.7, color="#444"),
                    zorder=1)

    # Encoder column (left)
    box(0.05, 0.75, 0.20, 0.10, "Per-subject\npseudobulk", "#e6e6e6")
    box(0.30, 0.85, 0.20, 0.10, "HGT (CCC graph)", PALETTES["categorical"][0])
    box(0.30, 0.65, 0.20, 0.10, "CellTransformer", PALETTES["categorical"][1])
    box(0.55, 0.75, 0.18, 0.10, "PMA pool", PALETTES["categorical"][2])
    box(0.55, 0.55, 0.18, 0.10, "RegionHandler\n(6-scalar)", PALETTES["categorical"][3])
    box(0.55, 0.35, 0.18, 0.10, "Pathology\nAttention", PALETTES["categorical"][4])
    box(0.55, 0.15, 0.18, 0.10, "gene_gate", PALETTES["categorical"][5])
    box(0.78, 0.45, 0.18, 0.20, "z ∈ ℝ⁶⁴", "#fff2cc")

    # Head column
    box(0.78, 0.10, 0.18, 0.20, "ResDecMHE Head\n(NPT + TabM\nk=8 ensemble\n+ HyperConn\n+ FiLM)",
        baseline_color("ResDec-MHE"))

    # Bottom: TabPFN residual base + composite output
    box(0.05, 0.20, 0.35, 0.10, "TabPFN-2.6 (in-context, top-2K features)",
        baseline_color("TabPFN-2.6"))
    box(0.45, 0.20, 0.20, 0.10, "ŷ_tabpfn", "#ffd9d9")
    box(0.45, 0.02, 0.20, 0.10, "f̂_residual", "#d9ffd9")
    box(0.78, 0.02, 0.18, 0.10, "ŷ = ŷ_tabpfn + f̂", "#cce5ff")

    # Arrows
    arrow(0.25, 0.80, 0.30, 0.90)
    arrow(0.25, 0.80, 0.30, 0.70)
    arrow(0.50, 0.90, 0.55, 0.80)
    arrow(0.50, 0.70, 0.55, 0.60)
    arrow(0.50, 0.65, 0.55, 0.40)
    arrow(0.73, 0.80, 0.78, 0.60)
    arrow(0.73, 0.60, 0.78, 0.55)
    arrow(0.73, 0.40, 0.78, 0.50)
    arrow(0.73, 0.20, 0.78, 0.45)
    arrow(0.78, 0.50, 0.78, 0.30)  # z → head input
    arrow(0.40, 0.25, 0.45, 0.25)  # tabpfn → ŷ_tabpfn
    arrow(0.78, 0.10, 0.65, 0.07)  # head → f̂_residual
    arrow(0.65, 0.07, 0.78, 0.07)  # f̂_residual → composite
    arrow(0.55, 0.25, 0.78, 0.07)  # ŷ_tabpfn → composite

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return fig


# ----------------------------- CLI ---------------------------------------


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--canonical-dir",
        default="outputs/redesign/p5_canonical_seed42",
    )
    p.add_argument(
        "--captum-npz",
        default="outputs/redesign/interpretability/captum_ig/composite_attributions.npz",
    )
    p.add_argument(
        "--head-attention-npz",
        default="outputs/redesign/interpretability/pathology_attention_per_subject.npz",
    )
    p.add_argument(
        "--ccc-edge-csv",
        default="outputs/redesign/interpretability/ccc/ccc_edge_attention.csv",
    )
    p.add_argument(
        "--residual-csv",
        default="outputs/redesign/interpretability/residual_per_subject.csv",
    )
    p.add_argument(
        "--tabpfn-dir", default="data/redesign",
    )
    p.add_argument(
        "--out-dir",
        default="outputs/redesign/interpretability/figures_introspection",
    )
    p.add_argument("--example-subject", default=None,
                   help="Subject ID for the per-subject waterfall (default: top-resilient).")
    p.add_argument("--n-folds", type=int, default=5)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    apply_theme()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    canon = Path(args.canonical_dir)

    # Load canonical predictions per fold (BEST checkpoint).
    per_fold = []
    for f in range(args.n_folds):
        npz_path = canon / f"fold{f}/val_predictions_best.npz"
        if not npz_path.exists():
            logger.warning("missing %s; skipping fold %d", npz_path, f)
            continue
        d = np.load(npz_path, allow_pickle=True)
        per_fold.append({
            "subject_ids": np.asarray(d["subject_ids"]),
            "predictions": np.asarray(d["predictions"], dtype=np.float64),
            "targets": np.asarray(d["targets"], dtype=np.float64),
        })
    if not per_fold:
        logger.error("no per-fold predictions found; aborting")
        return 1

    composite_subj = np.concatenate([f["subject_ids"] for f in per_fold])
    composite_preds = np.concatenate([f["predictions"] for f in per_fold])
    composite_true = np.concatenate([f["targets"] for f in per_fold])

    # TabPFN per-fold predictions.
    tabpfn_per_fold = []
    tabpfn_subj_to_pred: dict[str, float] = {}
    for f in range(args.n_folds):
        npz_path = Path(args.tabpfn_dir) / f"tabpfn_outer_fold{f}.npz"
        if not npz_path.exists():
            logger.warning("missing %s", npz_path)
            continue
        d = np.load(npz_path, allow_pickle=True)
        tabpfn_per_fold.append((
            np.asarray(d["y_true"], dtype=np.float64),
            np.asarray(d["y_tabpfn"], dtype=np.float64),
        ))
        for s, p_ in zip(d["val_subject_ids"], d["y_tabpfn"]):
            tabpfn_subj_to_pred[str(s)] = float(p_)
    composite_tabpfn = np.array(
        [tabpfn_subj_to_pred.get(str(s), np.nan) for s in composite_subj],
    )

    # Captum.
    captum_path = Path(args.captum_npz)
    captum_data = None
    if captum_path.exists():
        c = np.load(captum_path, allow_pickle=True)
        captum_data = {k: c[k] for k in c.files}
    summary_path = Path(args.captum_npz).parent / "composite_attribution_summary.json"
    captum_summary = json.loads(summary_path.read_text()) if summary_path.exists() else None

    # Head attention.
    head_attention = None
    head_attention_path = Path(args.head_attention_npz)
    if head_attention_path.exists():
        d = np.load(head_attention_path, allow_pickle=True)
        # File schema: per-subject (n_heads, n_cell_types) — adapt as needed.
        for key in ("attention", "per_subject", "head_attention", "per_head_attention"):
            if key in d.files:
                head_attention = np.asarray(d[key], dtype=np.float64)
                break

    # Residuals.
    res_path = Path(args.residual_csv)
    residual_df = pd.read_csv(res_path) if res_path.exists() else None

    # Names: prefer pathology_attention_summary.json if available for CT names.
    ct_names = None
    gene_names = None
    if captum_summary is not None:
        ct_names = (
            captum_summary.get("cell_types")
            or captum_summary.get("cell_type_names")
            or captum_summary.get("cell_types_ranked_by_total_attribution")
        )
        gene_names = (
            captum_summary.get("genes")
            or captum_summary.get("gene_names")
        )
    if ct_names is None and head_attention is not None:
        ct_names = [f"CT_{i}" for i in range(head_attention.shape[-1])]
    if gene_names is None and captum_data is not None:
        for key in ("gene_names", "genes"):
            if key in captum_data:
                gene_names = list(captum_data[key])
                break
    if gene_names is None and captum_data is not None:
        # Fallback: infer count from attribution shape.
        for key in ("attributions", "attributions", "attrs"):
            if key in captum_data:
                arr = captum_data[key]
                if arr.ndim == 3:
                    gene_names = [f"gene_{j}" for j in range(arr.shape[2])]
                    break

    # ---------------------- render ----------------------
    rendered = []

    # 1. Subject waterfall — pick top-resilient subject by composite − TabPFN.
    if captum_data is not None and "attributions" in captum_data and gene_names is not None:
        attrs_all = captum_data["attributions"]  # (n_subjects, n_ct, n_gene)
        subj_attr_ids = (
            list(captum_data["subject_ids"]) if "subject_ids" in captum_data
            else list(composite_subj)
        )
        # Pick subject by --example-subject or by max residual contribution.
        sel_id = args.example_subject
        if sel_id is None:
            residual_contrib_arr = composite_preds - composite_tabpfn
            order = np.argsort(np.nan_to_num(residual_contrib_arr, nan=0.0))[::-1]
            sel_id = str(composite_subj[order[0]])
        if sel_id in subj_attr_ids and sel_id in [str(s) for s in composite_subj]:
            i_attr = subj_attr_ids.index(sel_id)
            i_comp = list(composite_subj).index(sel_id)
            try:
                fig = make_fig_subject_waterfall(
                    sel_id,
                    attrs_all[i_attr],
                    cell_type_names=ct_names or [f"CT_{i}" for i in range(attrs_all.shape[1])],
                    gene_names=gene_names,
                    tabpfn_pred=float(composite_tabpfn[i_comp]),
                    composite_pred=float(composite_preds[i_comp]),
                    true_y=float(composite_true[i_comp]),
                )
                save_fig(fig, out_dir / "fig_subject_waterfall")
                plt.close(fig)
                rendered.append("fig_subject_waterfall")
            except SkipFigure as exc:
                logger.warning("waterfall: %s", exc)

    # 2. TabPFN-vs-residual stack.
    try:
        if np.isfinite(composite_tabpfn).any():
            fig = make_fig_tabpfn_vs_residual_stack(
                composite_subj, composite_tabpfn, composite_preds, composite_true,
            )
            save_fig(fig, out_dir / "fig_tabpfn_vs_residual_stack")
            plt.close(fig)
            rendered.append("fig_tabpfn_vs_residual_stack")
    except SkipFigure as exc:
        logger.warning("tabpfn-vs-residual: %s", exc)

    # 3. Head attention chord.
    if head_attention is not None and ct_names is not None:
        try:
            fig = make_fig_head_attention_chord(head_attention, ct_names)
            save_fig(fig, out_dir / "fig_head_attention_chord")
            plt.close(fig)
            rendered.append("fig_head_attention_chord")
        except SkipFigure as exc:
            logger.warning("chord: %s", exc)

    # 4. Resilience signature radar.
    if (
        captum_data is not None
        and "attributions" in captum_data
        and residual_df is not None
        and gene_names is not None
        and ct_names is not None
    ):
        try:
            attrs_all = captum_data["attributions"]
            res_map = dict(zip(residual_df["ROSMAP_IndividualID"].astype(str), residual_df["residual"].astype(float)))
            subj_attr_ids = (
                list(captum_data["subject_ids"]) if "subject_ids" in captum_data
                else list(composite_subj)
            )
            res_per_attr_subj = np.array([
                res_map.get(str(s), np.nan) for s in subj_attr_ids
            ])
            fig = make_fig_resilience_signature_radar(
                attrs_all, res_per_attr_subj, ct_names, gene_names,
            )
            save_fig(fig, out_dir / "fig_resilience_signature_radar")
            plt.close(fig)
            rendered.append("fig_resilience_signature_radar")
        except SkipFigure as exc:
            logger.warning("radar: %s", exc)

    # 6. Per-quintile attribution shift.
    if captum_data is not None and "attributions" in captum_data and ct_names is not None:
        try:
            attrs_all = captum_data["attributions"]
            subj_attr_ids = (
                list(captum_data["subject_ids"]) if "subject_ids" in captum_data
                else list(composite_subj)
            )
            subj_to_pred = dict(zip([str(s) for s in composite_subj], composite_preds))
            preds_for_attrs = np.array([
                subj_to_pred.get(str(s), np.nan) for s in subj_attr_ids
            ])
            fig = make_fig_per_quintile_attribution(attrs_all, preds_for_attrs, ct_names)
            save_fig(fig, out_dir / "fig_per_quintile_attribution")
            plt.close(fig)
            rendered.append("fig_per_quintile_attribution")
        except SkipFigure as exc:
            logger.warning("per-quintile: %s", exc)

    # 7. Head fingerprint UMAP.
    if head_attention is not None and residual_df is not None:
        try:
            res_map = dict(zip(residual_df["ROSMAP_IndividualID"].astype(str), residual_df["residual"].astype(float)))
            res_arr = np.array([
                res_map.get(str(s), np.nan)
                for s in composite_subj
            ])
            # Need to align head_attention order to composite_subj order.
            # If head_attention has its own subject_ids in the npz, use that.
            if head_attention.shape[0] == len(composite_subj):
                fig = make_fig_head_fingerprint_umap(head_attention, res_arr)
                save_fig(fig, out_dir / "fig_head_fingerprint_umap")
                plt.close(fig)
                rendered.append("fig_head_fingerprint_umap")
        except SkipFigure as exc:
            logger.warning("UMAP: %s", exc)

    # 8. Calibration overlay.
    if tabpfn_per_fold:
        composite_tuples = [
            (f["targets"], f["predictions"]) for f in per_fold
        ]
        try:
            fig = make_fig_calibration_overlay(tabpfn_per_fold, composite_tuples)
            save_fig(fig, out_dir / "fig_calibration_overlay")
            plt.close(fig)
            rendered.append("fig_calibration_overlay")
        except SkipFigure as exc:
            logger.warning("calibration: %s", exc)

    # 9. HGT CT network.
    edge_csv_path = Path(args.ccc_edge_csv)
    if edge_csv_path.exists():
        try:
            df = pd.read_csv(edge_csv_path)
            fig = make_fig_hgt_celltype_network(df)
            save_fig(fig, out_dir / "fig_hgt_celltype_network")
            plt.close(fig)
            rendered.append("fig_hgt_celltype_network")
        except SkipFigure as exc:
            logger.warning("hgt network: %s", exc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("hgt network failed: %s", exc)

    # 10. Architecture diagram (no data dependency).
    try:
        fig = make_fig_architecture_diagram()
        save_fig(fig, out_dir / "fig_architecture_diagram")
        plt.close(fig)
        rendered.append("fig_architecture_diagram")
    except SkipFigure as exc:
        logger.warning("architecture: %s", exc)

    logger.info("rendered %d figures in %s: %s", len(rendered), out_dir, rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
