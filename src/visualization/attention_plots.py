"""
Attention visualization plots.

Provides publication-quality plots for:
- Cell type attention heatmaps
- Pathology-stratified attention
- Gene gate attention
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.visualization.config import (
    get_sequential_cmap,
    get_diverging_cmap,
    get_cell_type_color,
    setup_seaborn_style,
    save_figure,
    CELL_TYPE_COLORS,
)
from src.data.constants import CELL_TYPE_ORDER


def plot_cell_type_attention_heatmap(
    attention_df: pd.DataFrame,
    figsize: tuple[float, float] = (12, 8),
    cmap: str | None = None,
    title: str = "Cell Type Attention by Pathology Level",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot cell type attention heatmap stratified by pathology level.

    Args:
        attention_df: DataFrame with columns [cell_type, pathology_tertile, mean_attention]
        figsize: Figure size
        cmap: Colormap name (defaults to sequential colormap)
        title: Plot title
        save_path: If provided, save figure to this path

    Returns:
        Matplotlib Figure
    """
    setup_seaborn_style()

    # Pivot to matrix format
    pivot_df = attention_df.pivot(
        index="cell_type",
        columns="pathology_tertile",
        values="mean_attention",
    )

    # Reorder columns
    col_order = ["low", "medium", "high"]
    pivot_df = pivot_df[[c for c in col_order if c in pivot_df.columns]]

    # Reorder rows by cell type order
    row_order = [ct for ct in CELL_TYPE_ORDER if ct in pivot_df.index]
    pivot_df = pivot_df.reindex(row_order)

    fig, ax = plt.subplots(figsize=figsize)

    cmap_obj = get_sequential_cmap() if cmap is None else cmap

    sns.heatmap(
        pivot_df,
        cmap=cmap_obj,
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Mean Attention"},
    )

    ax.set_xlabel("Pathology Level")
    ax.set_ylabel("Cell Type")
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        save_figure(fig, str(save_path))

    return fig


def plot_cell_type_importance_bar(
    importance_df: pd.DataFrame,
    figsize: tuple[float, float] = (10, 8),
    title: str = "Cell Type Importance Ranking",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot cell type importance as horizontal bar chart.

    Args:
        importance_df: DataFrame with columns [cell_type, mean_attention, rank]
        figsize: Figure size
        title: Plot title
        save_path: If provided, save figure to this path

    Returns:
        Matplotlib Figure
    """
    setup_seaborn_style()

    # Sort by importance (descending)
    df = importance_df.sort_values("mean_attention", ascending=True)

    fig, ax = plt.subplots(figsize=figsize)

    # Get colors for each cell type
    colors = [get_cell_type_color(ct) for ct in df["cell_type"]]

    bars = ax.barh(df["cell_type"], df["mean_attention"], color=colors)

    # Add error bars if std available
    if "std_attention" in df.columns:
        ax.errorbar(
            df["mean_attention"],
            range(len(df)),
            xerr=df["std_attention"],
            fmt="none",
            color="black",
            capsize=2,
        )

    ax.set_xlabel("Mean Attention Weight")
    ax.set_ylabel("Cell Type")
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        save_figure(fig, str(save_path))

    return fig


def plot_attention_distribution(
    attention: np.ndarray,
    cell_type_names: list[str] | None = None,
    figsize: tuple[float, float] = (12, 6),
    title: str = "Attention Weight Distribution by Cell Type",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot distribution of attention weights across subjects.

    Args:
        attention: Attention weights [n_subjects, n_cell_types] (after head averaging)
        cell_type_names: Cell type names
        figsize: Figure size
        title: Plot title
        save_path: If provided, save figure to this path

    Returns:
        Matplotlib Figure
    """
    setup_seaborn_style()

    n_subjects, n_cell_types = attention.shape
    cell_type_names = cell_type_names or [f"Type_{i}" for i in range(n_cell_types)]

    # Create tidy DataFrame
    rows = []
    for ct_idx in range(n_cell_types):
        for subj_idx in range(n_subjects):
            rows.append({
                "cell_type": cell_type_names[ct_idx],
                "attention": attention[subj_idx, ct_idx],
            })
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=figsize)

    # Create palette from cell type colors
    palette = {ct: get_cell_type_color(ct) for ct in cell_type_names}

    sns.boxplot(
        data=df,
        x="cell_type",
        y="attention",
        hue="cell_type",
        palette=palette,
        legend=False,
        ax=ax,
    )

    ax.set_xlabel("Cell Type")
    ax.set_ylabel("Attention Weight")
    ax.set_title(title)

    # Rotate x labels
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        save_figure(fig, str(save_path))

    return fig


def plot_gene_gate_heatmap(
    gene_gate_weights: np.ndarray,
    gene_names: list[str] | None = None,
    cell_type_names: list[str] | None = None,
    top_k_genes: int = 100,
    figsize: tuple[float, float] = (14, 10),
    title: str = "Gene Attention Weights",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot gene gate attention weights as heatmap.

    Args:
        gene_gate_weights: Gene gate weights [n_cell_types, n_genes]
        gene_names: Gene names
        cell_type_names: Cell type names
        top_k_genes: Number of top genes to display (by max weight across cell types)
        figsize: Figure size
        title: Plot title
        save_path: If provided, save figure to this path

    Returns:
        Matplotlib Figure
    """
    setup_seaborn_style()

    n_cell_types, n_genes = gene_gate_weights.shape
    cell_type_names = cell_type_names or list(CELL_TYPE_ORDER)[:n_cell_types]
    gene_names = gene_names or [f"gene_{i}" for i in range(n_genes)]

    # Select top-k genes by max weight across cell types
    max_weights = gene_gate_weights.max(axis=0)
    top_indices = np.argsort(max_weights)[::-1][:top_k_genes]

    weights_subset = gene_gate_weights[:, top_indices]
    gene_names_subset = [gene_names[i] for i in top_indices]

    # Create DataFrame
    df = pd.DataFrame(
        weights_subset,
        index=cell_type_names,
        columns=gene_names_subset,
    )

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        df,
        cmap=get_sequential_cmap(),
        xticklabels=True,
        yticklabels=True,
        ax=ax,
        cbar_kws={"label": "Attention Weight"},
    )

    ax.set_xlabel("Gene")
    ax.set_ylabel("Cell Type")
    ax.set_title(f"{title} (Top {top_k_genes} Genes)")

    # Rotate x labels
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(fontsize=8)
    plt.tight_layout()

    if save_path:
        save_figure(fig, str(save_path))

    return fig


def plot_resilience_signature_heatmap(
    signature_df: pd.DataFrame,
    figsize: tuple[float, float] = (6, 10),
    title: str = "Resilience Signature",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot resilience signature as diverging heatmap.

    Args:
        signature_df: DataFrame with columns [cell_type, signature]
        figsize: Figure size
        title: Plot title
        save_path: If provided, save figure to this path

    Returns:
        Matplotlib Figure
    """
    setup_seaborn_style()

    # Sort by signature
    df = signature_df.sort_values("signature", ascending=False)

    fig, ax = plt.subplots(figsize=figsize)

    # Create single-column heatmap
    data = df[["signature"]].set_index(df["cell_type"])

    # Determine symmetric limits
    vmax = max(abs(data.values.min()), abs(data.values.max()))

    sns.heatmap(
        data,
        cmap=get_diverging_cmap(),
        center=0,
        vmin=-vmax,
        vmax=vmax,
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Signature (Resilient - Vulnerable)"},
    )

    ax.set_xlabel("")
    ax.set_ylabel("Cell Type")
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        save_figure(fig, str(save_path))

    return fig
