"""
Gene and cell type importance visualization plots.

Provides publication-quality plots for:
- Gene importance rankings
- Top genes per cell type
- Cell-cell communication importance
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.visualization.config import (
    get_sequential_cmap,
    get_cell_type_color,
    get_edge_type_color,
    get_edge_type_display_name,
    setup_seaborn_style,
    save_figure,
)
from src.data.constants import CELL_TYPE_ORDER, EPSILON_DIVISION


def plot_top_genes_per_cell_type(
    top_genes_df: pd.DataFrame,
    cell_types: list[str] | None = None,
    n_genes: int = 10,
    figsize: tuple[float, float] = (14, 10),
    title: str = "Top Genes per Cell Type",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot top genes per cell type as faceted bar chart.

    Args:
        top_genes_df: DataFrame with columns [cell_type, rank, gene, weight]
        cell_types: Cell types to include (defaults to first 6 in order)
        n_genes: Number of genes per cell type
        figsize: Figure size
        title: Plot title
        save_path: If provided, save figure to this path

    Returns:
        Matplotlib Figure
    """
    setup_seaborn_style()

    # Select cell types
    if cell_types is None:
        all_cts = top_genes_df["cell_type"].unique()
        # Try to use cell types from CELL_TYPE_ORDER, fallback to unique values
        cell_types = [ct for ct in CELL_TYPE_ORDER if ct in all_cts][:6]
        if len(cell_types) == 0:
            cell_types = list(all_cts)[:6]

    # Handle empty cell_types
    if len(cell_types) == 0:
        raise ValueError("No cell types to plot. Check your data.")

    # Filter data
    df = top_genes_df[
        (top_genes_df["cell_type"].isin(cell_types)) &
        (top_genes_df["rank"] <= n_genes)
    ].copy()

    # Create faceted plot
    n_cols = min(3, len(cell_types))
    n_rows = (len(cell_types) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, cell_type in enumerate(cell_types):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        ct_data = df[df["cell_type"] == cell_type].sort_values("weight", ascending=True)

        color = get_cell_type_color(cell_type)
        ax.barh(ct_data["gene"], ct_data["weight"], color=color)

        ax.set_xlabel("Attention Weight")
        ax.set_title(cell_type, fontsize=10)
        ax.tick_params(axis="y", labelsize=8)

    # Hide empty subplots
    for idx in range(len(cell_types), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        save_figure(fig, str(save_path))

    return fig


def plot_gene_importance_volcano(
    gene_df: pd.DataFrame,
    cell_type: str | None = None,
    significance_threshold: float = 0.05,
    figsize: tuple[float, float] = (10, 8),
    title: str | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot gene differential expression as volcano plot.

    Points are colored by significance (padj if available, else pvalue)
    and sized by gate_weight if available.

    Args:
        gene_df: DataFrame with columns [gene, log2_fold_change, pvalue/padj]
        cell_type: Cell type for title
        significance_threshold: P-value threshold for significance
        figsize: Figure size
        title: Plot title
        save_path: If provided, save figure to this path

    Returns:
        Matplotlib Figure
    """
    setup_seaborn_style()

    fig, ax = plt.subplots(figsize=figsize)

    # Determine p-value column (prefer FDR-corrected)
    p_col = None
    if "padj" in gene_df.columns:
        p_col = "padj"
    elif "pvalue" in gene_df.columns:
        p_col = "pvalue"
    elif "p_value" in gene_df.columns:
        p_col = "p_value"

    fc_col = "log2_fold_change" if "log2_fold_change" in gene_df.columns else "weight"

    if p_col is None:
        # No significance data, plot simple scatter
        ax.scatter(gene_df[fc_col], range(len(gene_df)), alpha=0.5)
        ax.set_xlabel(fc_col)
        ax.set_ylabel("Gene Index")
    else:
        # Volcano plot
        df = gene_df.copy()
        df["-log10(p)"] = -np.log10(df[p_col] + EPSILON_DIVISION)

        # Color by significance
        colors = np.where(df[p_col] < significance_threshold, "red", "gray")

        # Size by gate_weight if available
        if "gate_weight" in df.columns:
            gw = df["gate_weight"].values
            gw_min, gw_max = gw.min(), gw.max()
            if gw_max > gw_min:
                sizes = 10 + 90 * (gw - gw_min) / (gw_max - gw_min)
            else:
                sizes = 30
        else:
            sizes = 20

        ax.scatter(df[fc_col], df["-log10(p)"], c=colors, alpha=0.5, s=sizes)

        # Add threshold line
        ax.axhline(-np.log10(significance_threshold), color="red", linestyle="--", alpha=0.5)
        # Add vertical reference line at fold change = 0
        ax.axvline(0, color="gray", linestyle=":", alpha=0.3)

        p_label = "FDR-adjusted p-value" if p_col == "padj" else "p-value"
        ax.set_xlabel("log2(Fold Change)" if fc_col == "log2_fold_change" else "Importance Weight")
        ax.set_ylabel(f"-log10({p_label})")

    if title is None:
        title = f"Differential Expression{f' ({cell_type})' if cell_type else ''}"
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        save_figure(fig, str(save_path))

    return fig


def plot_ccc_network_summary(
    network_df: pd.DataFrame,
    figsize: tuple[float, float] = (10, 6),
    title: str = "Cell-Cell Communication by Category",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot CCC network summary as bar chart by edge type category.

    Args:
        network_df: DataFrame with columns [edge_type, display_name, mean_attention]
        figsize: Figure size
        title: Plot title
        save_path: If provided, save figure to this path

    Returns:
        Matplotlib Figure
    """
    setup_seaborn_style()

    # NaN/zero guard: if all attention values are NaN or zero, return placeholder figure
    if (
        len(network_df) == 0
        or "mean_attention" not in network_df.columns
        or network_df["mean_attention"].isna().all()
        or (network_df["mean_attention"].fillna(0) == 0).all()
    ):
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5, 0.5,
            "No CCC attention data available",
            ha="center", va="center", fontsize=12, color="gray",
            transform=ax.transAxes,
        )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        if save_path:
            save_figure(fig, str(save_path))
        return fig

    fig, ax = plt.subplots(figsize=figsize)

    # Sort by mean attention
    df = network_df.sort_values("mean_attention", ascending=True)

    # Get colors and display names
    colors = [get_edge_type_color(et) for et in df["edge_type"]]
    labels = df["display_name"] if "display_name" in df.columns else df["edge_type"]

    ax.barh(labels, df["mean_attention"], color=colors)

    # Add error bars if available
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
    ax.set_ylabel("Communication Category")
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        save_figure(fig, str(save_path))

    return fig


def plot_top_interactions_heatmap(
    interactions_df: pd.DataFrame,
    top_k: int = 20,
    figsize: tuple[float, float] = (10, 8),
    title: str = "Top Cell-Cell Interactions",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot top cell-cell interactions as heatmap.

    Args:
        interactions_df: DataFrame with columns [source, target, mean_attention]
        top_k: Number of top interactions to show
        figsize: Figure size
        title: Plot title
        save_path: If provided, save figure to this path

    Returns:
        Matplotlib Figure
    """
    setup_seaborn_style()

    # NaN/zero guard: if all attention values are NaN or zero, return placeholder figure
    if (
        len(interactions_df) == 0
        or "mean_attention" not in interactions_df.columns
        or interactions_df["mean_attention"].isna().all()
        or (interactions_df["mean_attention"].fillna(0) == 0).all()
    ):
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5, 0.5,
            "No CCC interaction data available",
            ha="center", va="center", fontsize=12, color="gray",
            transform=ax.transAxes,
        )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        if save_path:
            save_figure(fig, str(save_path))
        return fig

    # Get top interactions
    df = interactions_df.head(top_k)

    # Create interaction labels
    df = df.copy()
    df["interaction"] = df["source"] + " → " + df["target"]

    fig, ax = plt.subplots(figsize=figsize)

    # Sort by attention
    df = df.sort_values("mean_attention", ascending=True)

    colors = [get_cell_type_color(src) for src in df["source"]]

    ax.barh(df["interaction"], df["mean_attention"], color=colors)

    ax.set_xlabel("Mean Attention Weight")
    ax.set_ylabel("Interaction")
    ax.set_title(f"{title} (Top {top_k})")

    # Adjust label size for readability
    ax.tick_params(axis="y", labelsize=8)

    plt.tight_layout()

    if save_path:
        save_figure(fig, str(save_path))

    return fig


def plot_regional_gene_importance(
    regional_df: pd.DataFrame,
    regions: list[str] | None = None,
    n_genes: int = 10,
    figsize: tuple[float, float] = (14, 8),
    title: str = "Top Genes by Region",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot top genes per region as faceted bar chart.

    Args:
        regional_df: DataFrame with columns [region, cell_type, gene, effective_weight]
        regions: Regions to include (defaults to all)
        n_genes: Number of genes per region
        figsize: Figure size
        title: Plot title
        save_path: If provided, save figure to this path

    Returns:
        Matplotlib Figure
    """
    setup_seaborn_style()

    # Select regions
    if regions is None:
        regions = regional_df["region"].unique().tolist()

    # Aggregate across cell types to get top genes per region
    agg_df = regional_df.groupby(["region", "gene"]).agg({
        "effective_weight": "mean",
    }).reset_index()

    # Create faceted plot
    n_cols = min(3, len(regions))
    n_rows = (len(regions) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, region in enumerate(regions):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        region_data = agg_df[agg_df["region"] == region].nlargest(n_genes, "effective_weight")
        region_data = region_data.sort_values("effective_weight", ascending=True)

        ax.barh(region_data["gene"], region_data["effective_weight"])
        ax.set_xlabel("Effective Weight")
        ax.set_title(region, fontsize=10)
        ax.tick_params(axis="y", labelsize=8)

    # Hide empty subplots
    for idx in range(len(regions), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        save_figure(fig, str(save_path))

    return fig
