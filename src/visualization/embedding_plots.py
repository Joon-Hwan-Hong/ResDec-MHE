"""
Embedding visualization plots.

Provides publication-quality plots for:
- UMAP projections colored by covariates
- Cluster visualizations
- Linear probe results
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.visualization.config import (
    ACCENT_CORAL,
    ACCENT_TEAL,
    get_sequential_cmap,
    get_diverging_cmap,
    setup_seaborn_style,
    save_figure,
)

logger = logging.getLogger(__name__)


def plot_umap_scatter(
    umap_df: pd.DataFrame,
    color_by: str | None = None,
    figsize: tuple[float, float] = (10, 8),
    title: str = "Subject Embeddings (UMAP)",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot UMAP projection of subject embeddings.

    Args:
        umap_df: DataFrame with columns: subject_id, umap_1, umap_2, and optional color columns
        color_by: Column name to color points by (continuous or categorical)
        figsize: Figure size
        title: Plot title
        save_path: If provided, save figure to this path

    Returns:
        Matplotlib Figure
    """
    setup_seaborn_style()

    fig, ax = plt.subplots(figsize=figsize)

    if color_by is not None and color_by in umap_df.columns:
        values = umap_df[color_by]

        # Check if categorical or continuous
        if values.dtype == "object" or values.nunique() < 10:
            # Categorical
            scatter = ax.scatter(
                umap_df["umap_1"],
                umap_df["umap_2"],
                c=pd.Categorical(values).codes,
                cmap="Set2",
                alpha=0.7,
                s=50,
            )
            # Add legend for categories
            handles = [
                plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=plt.cm.Set2(i / values.nunique()),
                           markersize=8, label=cat)
                for i, cat in enumerate(values.unique())
            ]
            ax.legend(handles=handles, title=color_by, loc="best")
        else:
            # Continuous
            scatter = ax.scatter(
                umap_df["umap_1"],
                umap_df["umap_2"],
                c=values,
                cmap=get_sequential_cmap(),
                alpha=0.7,
                s=50,
            )
            plt.colorbar(scatter, ax=ax, label=color_by)
    else:
        ax.scatter(
            umap_df["umap_1"],
            umap_df["umap_2"],
            color=ACCENT_CORAL,
            alpha=0.7,
            s=50,
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        save_figure(fig, str(save_path))
        logger.info(f"Saved UMAP plot to {save_path}")

    return fig


def plot_cluster_composition(
    cluster_df: pd.DataFrame,
    covariate: str = "cluster",
    figsize: tuple[float, float] = (10, 6),
    title: str = "Cluster Composition",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot cluster composition as stacked bar chart.

    Args:
        cluster_df: DataFrame with cluster assignments and covariates
        covariate: Column to show composition by
        figsize: Figure size
        title: Plot title
        save_path: If provided, save figure to this path

    Returns:
        Matplotlib Figure
    """
    setup_seaborn_style()

    if "cluster" not in cluster_df.columns:
        logger.warning("No 'cluster' column in cluster_df")
        return None

    fig, ax = plt.subplots(figsize=figsize)

    # Count subjects per cluster
    cluster_counts = cluster_df["cluster"].value_counts().sort_index()

    ax.bar(
        cluster_counts.index.astype(str),
        cluster_counts.values,
        color=ACCENT_TEAL,
        alpha=0.8,
    )

    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Subjects")
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        save_figure(fig, str(save_path))
        logger.info(f"Saved cluster composition to {save_path}")

    return fig


def plot_linear_probe_results(
    probe_df: pd.DataFrame,
    figsize: tuple[float, float] = (10, 6),
    title: str = "Linear Probe Results (Embedding Quality)",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot linear probe results showing embedding quality per target.

    Args:
        probe_df: DataFrame with columns: target, r2_score, mae, etc.
        figsize: Figure size
        title: Plot title
        save_path: If provided, save figure to this path

    Returns:
        Matplotlib Figure
    """
    setup_seaborn_style()

    if "target" not in probe_df.columns or "r2_score" not in probe_df.columns:
        logger.warning("probe_df missing required columns (target, r2_score)")
        return None

    fig, ax = plt.subplots(figsize=figsize)

    # Sort by R2 score
    probe_df = probe_df.sort_values("r2_score", ascending=True)

    # Horizontal bar chart
    bars = ax.barh(
        probe_df["target"],
        probe_df["r2_score"],
        color=[ACCENT_TEAL if r2 >= 0 else ACCENT_CORAL for r2 in probe_df["r2_score"]],
        alpha=0.8,
    )

    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_xlabel("R² Score")
    ax.set_ylabel("Target Variable")
    ax.set_title(title)

    # Add value labels
    for bar, r2 in zip(bars, probe_df["r2_score"]):
        width = bar.get_width()
        ax.text(
            width + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{r2:.3f}",
            ha="left",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()

    if save_path:
        save_figure(fig, str(save_path))
        logger.info(f"Saved linear probe results to {save_path}")

    return fig


def plot_embedding_summary(
    umap_df: pd.DataFrame,
    cluster_df: pd.DataFrame | None = None,
    probe_df: pd.DataFrame | None = None,
    color_by: str = "cluster",
    figsize: tuple[float, float] = (16, 5),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Generate multi-panel embedding analysis summary.

    Args:
        umap_df: DataFrame with UMAP coordinates
        cluster_df: DataFrame with cluster assignments (optional)
        probe_df: DataFrame with linear probe results (optional)
        color_by: Column to color UMAP scatter by
        figsize: Figure size
        save_path: If provided, save figure to this path

    Returns:
        Matplotlib Figure
    """
    setup_seaborn_style()

    n_panels = 1
    if cluster_df is not None and "cluster" in cluster_df.columns:
        n_panels += 1
    if probe_df is not None and "r2_score" in probe_df.columns:
        n_panels += 1

    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    if n_panels == 1:
        axes = [axes]

    panel_idx = 0

    # Panel 1: UMAP scatter
    ax = axes[panel_idx]
    if color_by in umap_df.columns:
        values = umap_df[color_by]
        if values.dtype == "object" or values.nunique() < 10:
            scatter = ax.scatter(
                umap_df["umap_1"], umap_df["umap_2"],
                c=pd.Categorical(values).codes,
                cmap="Set2", alpha=0.7, s=30,
            )
        else:
            scatter = ax.scatter(
                umap_df["umap_1"], umap_df["umap_2"],
                c=values, cmap=get_sequential_cmap(), alpha=0.7, s=30,
            )
            plt.colorbar(scatter, ax=ax, label=color_by)
    else:
        ax.scatter(umap_df["umap_1"], umap_df["umap_2"], color=ACCENT_CORAL, alpha=0.7, s=30)

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Subject Embeddings")
    panel_idx += 1

    # Panel 2: Cluster composition
    if cluster_df is not None and "cluster" in cluster_df.columns and panel_idx < len(axes):
        ax = axes[panel_idx]
        cluster_counts = cluster_df["cluster"].value_counts().sort_index()
        ax.bar(cluster_counts.index.astype(str), cluster_counts.values, color=ACCENT_TEAL, alpha=0.8)
        ax.set_xlabel("Cluster")
        ax.set_ylabel("N Subjects")
        ax.set_title("Cluster Sizes")
        panel_idx += 1

    # Panel 3: Linear probe results
    if probe_df is not None and "r2_score" in probe_df.columns and panel_idx < len(axes):
        ax = axes[panel_idx]
        probe_sorted = probe_df.sort_values("r2_score", ascending=True)
        colors = [ACCENT_TEAL if r2 >= 0 else ACCENT_CORAL for r2 in probe_sorted["r2_score"]]
        ax.barh(probe_sorted["target"], probe_sorted["r2_score"], color=colors, alpha=0.8)
        ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
        ax.set_xlabel("R² Score")
        ax.set_title("Linear Probe Quality")

    plt.tight_layout()

    if save_path:
        save_figure(fig, str(save_path))
        logger.info(f"Saved embedding summary to {save_path}")

    return fig
