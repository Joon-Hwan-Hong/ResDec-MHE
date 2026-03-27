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
    covariate: str | None = None,
    figsize: tuple[float, float] = (10, 6),
    title: str = "Cluster Composition",
    save_path: str | Path | None = None,
) -> plt.Figure | None:
    """
    Plot cluster composition as bar chart, optionally stacked by covariate.

    When covariate is provided, shows a stacked bar chart where each cluster
    bar is broken down by covariate values (e.g., diagnosis, sex, pathology
    group). This reveals whether embedding clusters capture meaningful
    biological variation.

    Args:
        cluster_df: DataFrame with 'cluster' column and optional covariates
        covariate: Column to break down composition by. If None, shows
            simple cluster size bars.
        figsize: Figure size
        title: Plot title
        save_path: If provided, save figure to this path

    Returns:
        Matplotlib Figure, or None if required columns are missing
    """
    setup_seaborn_style()

    if "cluster" not in cluster_df.columns:
        logger.warning("No 'cluster' column in cluster_df")
        return None

    fig, ax = plt.subplots(figsize=figsize)

    if covariate is not None and covariate in cluster_df.columns:
        # Stacked bar chart: break down each cluster by covariate values
        cross_tab = pd.crosstab(cluster_df["cluster"], cluster_df[covariate])
        cross_tab = cross_tab.sort_index()

        cmap = plt.cm.get_cmap("tab10", len(cross_tab.columns))
        colors = [cmap(i) for i in range(len(cross_tab.columns))]

        cross_tab.plot.bar(
            stacked=True, ax=ax, color=colors, alpha=0.85,
            width=0.8,
        )
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Number of Subjects")
        ax.legend(title=covariate, bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.set_title(f"{title} by {covariate}")
    elif covariate is not None and covariate not in cluster_df.columns:
        logger.warning(f"Covariate '{covariate}' not found in cluster_df, showing cluster sizes")
        cluster_counts = cluster_df["cluster"].value_counts().sort_index()
        ax.bar(cluster_counts.index.astype(str), cluster_counts.values,
               color=ACCENT_TEAL, alpha=0.8)
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Number of Subjects")
        ax.set_title(title)
    else:
        # Simple cluster size bars
        cluster_counts = cluster_df["cluster"].value_counts().sort_index()
        ax.bar(cluster_counts.index.astype(str), cluster_counts.values,
               color=ACCENT_TEAL, alpha=0.8)
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
) -> plt.Figure | None:
    """
    Plot linear probe results showing embedding quality per target.

    Splits regression (R² score) and classification (accuracy) into
    separate subplots when both task types are present.

    Args:
        probe_df: DataFrame with columns: target, task_type, score_mean, etc.
        figsize: Figure size
        title: Plot title
        save_path: If provided, save figure to this path

    Returns:
        Matplotlib Figure, or None if required columns are missing
    """
    setup_seaborn_style()

    if "target" not in probe_df.columns or "score_mean" not in probe_df.columns:
        # Fall back to r2_score if score_mean not present (backward compatibility)
        if "target" not in probe_df.columns or "r2_score" not in probe_df.columns:
            logger.warning("probe_df missing required columns (target, score_mean or r2_score)")
            return None
        probe_df = probe_df.copy()
        probe_df["score_mean"] = probe_df["r2_score"]
        if "task_type" not in probe_df.columns:
            probe_df["task_type"] = "regression"

    # Split by task type
    has_task_type = "task_type" in probe_df.columns
    if has_task_type:
        reg_df = probe_df[probe_df["task_type"] == "regression"].copy()
        cls_df = probe_df[probe_df["task_type"] == "classification"].copy()
    else:
        reg_df = probe_df.copy()
        cls_df = pd.DataFrame()

    n_panels = (len(reg_df) > 0) + (len(cls_df) > 0)
    if n_panels == 0:
        logger.warning("No valid linear probe results to plot")
        return None

    fig, axes = plt.subplots(1, n_panels, figsize=figsize, squeeze=False)
    ax_idx = 0

    if len(reg_df) > 0:
        ax = axes[0, ax_idx]
        reg_df = reg_df.sort_values("score_mean", ascending=True)
        bars = ax.barh(
            reg_df["target"], reg_df["score_mean"],
            color=[ACCENT_TEAL if s >= 0 else ACCENT_CORAL for s in reg_df["score_mean"]],
            alpha=0.8,
        )
        ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
        ax.set_xlabel("R² Score")
        ax.set_ylabel("Target Variable")
        ax.set_title("Regression" if n_panels > 1 else title)
        for bar, score in zip(bars, reg_df["score_mean"]):
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{score:.3f}", ha="left", va="center", fontsize=9)
        ax_idx += 1

    if len(cls_df) > 0:
        ax = axes[0, ax_idx]
        cls_df = cls_df.sort_values("score_mean", ascending=True)
        bars = ax.barh(
            cls_df["target"], cls_df["score_mean"],
            color=ACCENT_TEAL, alpha=0.8,
        )
        ax.axvline(x=0.5, color="gray", linestyle="--", linewidth=0.8, label="Chance")
        ax.set_xlabel("Accuracy")
        ax.set_ylabel("Target Variable")
        ax.set_title("Classification" if n_panels > 1 else title)
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8)
        for bar, score in zip(bars, cls_df["score_mean"]):
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{score:.3f}", ha="left", va="center", fontsize=9)

    if n_panels > 1:
        fig.suptitle(title, fontsize=14, y=1.02)

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
    if probe_df is not None and ("score_mean" in probe_df.columns or "r2_score" in probe_df.columns):
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
    score_col = "score_mean" if probe_df is not None and "score_mean" in probe_df.columns else "r2_score"
    if probe_df is not None and score_col in probe_df.columns and panel_idx < len(axes):
        ax = axes[panel_idx]
        probe_sorted = probe_df.sort_values(score_col, ascending=True)
        colors = [ACCENT_TEAL if s >= 0 else ACCENT_CORAL for s in probe_sorted[score_col]]
        ax.barh(probe_sorted["target"], probe_sorted[score_col], color=colors, alpha=0.8)
        ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
        ax.set_xlabel("Score")
        ax.set_title("Linear Probe Quality")

    plt.tight_layout()

    if save_path:
        save_figure(fig, str(save_path))
        logger.info(f"Saved embedding summary to {save_path}")

    return fig
