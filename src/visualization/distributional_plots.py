"""Distributional-analysis plots: Wasserstein, DE-method concordance, stability selection.

Functions:

  - ``plot_wasserstein_per_celltype_bar`` — horizontal bar chart of per-cell-type
    mean per-gene Wasserstein-1 distance (resilient vs vulnerable pseudobulk),
    sorted descending, with the top gene label annotated per bar.

  - ``plot_de_method_concordance_bar`` — per-cell-type Spearman ρ between two
    differential-expression pipelines (e.g. Wilcoxon vs DESeq2), highlighting
    the extent of disagreement across cell types. Bars diverging-colored on ρ.

  - ``plot_stability_selection_bar`` — per-cell-type count of genes passing a
    stability-selection threshold (pi, rank-biserial). Highlights which cell
    types harbor reproducible resilient-vs-vulnerable signatures.

All functions take pre-parsed numpy / pandas data and return a
``matplotlib.figure.Figure``. Data loading and JSON/CSV parsing live in the
orchestrator (``make_distributional_figures.py``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from src.visualization.theme import PALETTES, fmt_axes, save_fig


def plot_wasserstein_per_celltype_bar(
    cell_type_names: Sequence[str],
    mean_wasserstein: Sequence[float],
    top_gene_per_ct: Sequence[str] | None = None,
    *,
    figsize: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Per-cell-type mean per-gene Wasserstein-1 distance (resilient vs vulnerable).

    Bars sorted descending. If ``top_gene_per_ct`` is supplied, each bar is
    annotated on the right with its top-ranked gene (largest W-1 in that CT).
    Bar color encodes the distance magnitude via the viridis colormap.

    Parameters
    ----------
    cell_type_names
        Length ``n_ct`` cell type labels.
    mean_wasserstein
        Length ``n_ct`` mean per-gene W-1 distances.
    top_gene_per_ct
        Optional length ``n_ct`` per-CT top gene (by W-1). If ``None``, no
        gene annotations are drawn.
    """
    cts = list(cell_type_names)
    wass = np.asarray(mean_wasserstein, dtype=np.float64)
    n_ct = len(cts)
    if n_ct != len(wass):
        raise ValueError(
            f"length mismatch: names={n_ct} vs wasserstein={len(wass)}")
    if n_ct == 0:
        raise ValueError("no cell types supplied")
    if top_gene_per_ct is not None and len(top_gene_per_ct) != n_ct:
        raise ValueError(
            f"top_gene_per_ct length {len(top_gene_per_ct)} != n_ct={n_ct}")

    order = np.argsort(wass)[::-1]
    cts_sorted = [cts[i] for i in order]
    wass_sorted = wass[order]
    genes_sorted = (
        [top_gene_per_ct[i] for i in order]
        if top_gene_per_ct is not None else None
    )

    if figsize is None:
        figsize = (6.5, max(3.5, 0.22 * n_ct))
    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.get_cmap(PALETTES["sequential"])
    norm_max = float(max(wass_sorted.max(), 1e-6))
    colors = [cmap(w / norm_max) for w in wass_sorted]
    y_pos = np.arange(n_ct)
    ax.barh(y_pos, wass_sorted, color=colors, edgecolor="black", linewidth=0.3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cts_sorted, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Mean per-gene Wasserstein-1 (resilient vs vulnerable)")
    fmt_axes(ax)
    if genes_sorted is not None:
        for i, (w, g) in enumerate(zip(wass_sorted, genes_sorted)):
            ax.text(
                w * 1.02, i, f"  {g}", va="center", ha="left",
                fontsize=6, color="black",
            )
        ax.set_xlim(0, norm_max * 1.35)
    fig.tight_layout()
    if save_path is not None:
        save_fig(fig, save_path)
    return fig


def plot_de_method_concordance_bar(
    cell_type_names: Sequence[str],
    spearman_rho: Sequence[float],
    *,
    method_labels: tuple[str, str] = ("Wilcoxon", "DESeq2"),
    figsize: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Per-cell-type Spearman ρ between two DE pipelines' p-value rankings.

    Sorted descending. Diverging PiYG color on ρ: positive (green) = methods
    agree; negative (magenta) = methods disagree. NaN ρ (undefined for a CT
    with constant p-values) is rendered as a hollow bar.
    """
    cts = list(cell_type_names)
    rho = np.asarray(spearman_rho, dtype=np.float64)
    n_ct = len(cts)
    if n_ct != len(rho):
        raise ValueError(f"length mismatch: names={n_ct} vs rho={len(rho)}")
    if n_ct == 0:
        raise ValueError("no cell types supplied")

    finite = np.isfinite(rho)
    order_scores = np.where(finite, rho, -np.inf)
    order = np.argsort(order_scores)[::-1]
    cts_sorted = [cts[i] for i in order]
    rho_sorted = rho[order]
    finite_sorted = finite[order]

    if figsize is None:
        figsize = (6.5, max(3.5, 0.22 * n_ct))
    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.get_cmap(PALETTES["diverging"])
    norm_max = max(0.01, float(np.abs(rho_sorted[finite_sorted]).max()))
    y_pos = np.arange(n_ct)
    for i, (r, ok) in enumerate(zip(rho_sorted, finite_sorted)):
        if ok:
            color = cmap(0.5 + 0.5 * r / norm_max)
            ax.barh(
                y_pos[i], r, color=color, edgecolor="black", linewidth=0.3,
            )
        else:
            ax.barh(
                y_pos[i], 0.01, color="white", edgecolor="gray",
                linewidth=0.5, hatch="///",
            )
            ax.text(0.02, y_pos[i], "NaN", va="center", fontsize=6, color="gray")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cts_sorted, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(f"Spearman ρ ({method_labels[0]} vs {method_labels[1]} p-value rank)")
    fmt_axes(ax)
    fig.tight_layout()
    if save_path is not None:
        save_fig(fig, save_path)
    return fig


def plot_stability_selection_bar(
    cell_type_names: Sequence[str],
    n_stable_per_ct: Sequence[int],
    stable_genes_per_ct: Sequence[Sequence[str]] | None = None,
    *,
    figsize: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Per-cell-type stability-selection gene count (horizontal bar).

    Sorted descending by ``n_stable_per_ct``. If ``stable_genes_per_ct`` is
    supplied, cell types with ≥1 stable gene are annotated with the gene list
    (top-3) to the right of the bar.
    """
    cts = list(cell_type_names)
    n_stable = np.asarray(n_stable_per_ct, dtype=np.int64)
    n_ct = len(cts)
    if n_ct != len(n_stable):
        raise ValueError(
            f"length mismatch: names={n_ct} vs n_stable={len(n_stable)}")
    if n_ct == 0:
        raise ValueError("no cell types supplied")
    if (
        stable_genes_per_ct is not None
        and len(stable_genes_per_ct) != n_ct
    ):
        raise ValueError(
            f"stable_genes_per_ct length {len(stable_genes_per_ct)} "
            f"!= n_ct={n_ct}"
        )

    order = np.argsort(n_stable)[::-1]
    cts_sorted = [cts[i] for i in order]
    n_sorted = n_stable[order]
    if figsize is None:
        figsize = (6.5, max(3.5, 0.22 * n_ct))
    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.get_cmap(PALETTES["sequential"])
    norm_max = float(max(n_sorted.max(), 1))
    colors = [cmap(v / norm_max) for v in n_sorted]
    y_pos = np.arange(n_ct)
    ax.barh(y_pos, n_sorted, color=colors, edgecolor="black", linewidth=0.3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cts_sorted, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("# stable (CT, gene) pairs (pi≥0.8, |rank-biserial|≥0.2)")
    fmt_axes(ax)
    if stable_genes_per_ct is not None:
        for i, idx in enumerate(order):
            genes = list(stable_genes_per_ct[idx] or [])
            if not genes:
                continue
            annot = ", ".join(genes[:3])
            if len(genes) > 3:
                annot += ", …"
            ax.text(
                n_sorted[i] + max(0.02, norm_max * 0.03), i,
                annot, va="center", ha="left", fontsize=6,
            )
        ax.set_xlim(0, norm_max * 1.45)
    fig.tight_layout()
    if save_path is not None:
        save_fig(fig, save_path)
    return fig
