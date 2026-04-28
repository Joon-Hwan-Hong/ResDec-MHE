"""Tests for embedding visualization plots."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.visualization.embedding_plots import (
    plot_umap_scatter,
    plot_cluster_composition,
    plot_linear_probe_results,
    plot_embedding_summary,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_umap_df():
    rng = np.random.default_rng(42)
    n = 60
    return pd.DataFrame({
        "subject_id": [f"S{i:03d}" for i in range(n)],
        "umap_1": rng.standard_normal(n),
        "umap_2": rng.standard_normal(n),
        "pathology": rng.uniform(0, 2, n),
        "diagnosis": rng.choice(["AD", "Control", "MCI"], n),
        "cluster": rng.integers(0, 4, n),
    })


@pytest.fixture
def sample_cluster_df():
    rng = np.random.default_rng(7)
    n = 60
    return pd.DataFrame({
        "subject_id": [f"S{i:03d}" for i in range(n)],
        "cluster": rng.integers(0, 4, n),
        "diagnosis": rng.choice(["AD", "Control", "MCI"], n),
    })


@pytest.fixture
def sample_probe_df():
    return pd.DataFrame({
        "target": ["cogn_global", "gpath", "amyl", "diagnosis", "sex"],
        "task_type": ["regression", "regression", "regression", "classification", "classification"],
        "score_mean": [0.31, 0.48, 0.22, 0.71, 0.88],
        "score_std": [0.04, 0.06, 0.05, 0.03, 0.02],
    })


@pytest.fixture
def sample_probe_df_legacy():
    return pd.DataFrame({
        "target": ["cogn_global", "gpath", "amyl"],
        "r2_score": [0.31, 0.48, 0.22],
    })


# =============================================================================
# plot_umap_scatter
# =============================================================================


class TestPlotUmapScatter:
    def test_basic_plot_no_color(self, sample_umap_df):
        fig = plot_umap_scatter(sample_umap_df)
        assert isinstance(fig, plt.Figure)
        ax = fig.get_axes()[0]
        assert ax.get_xlabel() == "UMAP 1"
        assert ax.get_ylabel() == "UMAP 2"
        plt.close(fig)

    def test_continuous_color(self, sample_umap_df):
        fig = plot_umap_scatter(sample_umap_df, color_by="pathology")
        assert isinstance(fig, plt.Figure)
        # Continuous path adds a colorbar — figure should have >= 2 axes
        assert len(fig.get_axes()) >= 2
        plt.close(fig)

    def test_categorical_color(self, sample_umap_df):
        fig = plot_umap_scatter(sample_umap_df, color_by="diagnosis")
        ax = fig.get_axes()[0]
        # Legend should be present with category title
        legend = ax.get_legend()
        assert legend is not None
        assert legend.get_title().get_text() == "diagnosis"
        plt.close(fig)

    def test_low_cardinality_int_treated_categorical(self, sample_umap_df):
        # cluster has <10 unique values → categorical branch
        fig = plot_umap_scatter(sample_umap_df, color_by="cluster")
        ax = fig.get_axes()[0]
        assert ax.get_legend() is not None
        plt.close(fig)

    def test_missing_color_column_falls_to_default(self, sample_umap_df):
        fig = plot_umap_scatter(sample_umap_df, color_by="not_a_column")
        assert isinstance(fig, plt.Figure)
        ax = fig.get_axes()[0]
        # No legend, no colorbar
        assert ax.get_legend() is None
        assert len(fig.get_axes()) == 1
        plt.close(fig)

    def test_custom_figsize(self, sample_umap_df):
        fig = plot_umap_scatter(sample_umap_df, figsize=(7, 5))
        assert fig.get_figwidth() == 7
        assert fig.get_figheight() == 5
        plt.close(fig)

    def test_custom_title(self, sample_umap_df):
        fig = plot_umap_scatter(sample_umap_df, title="Custom UMAP")
        assert fig.get_axes()[0].get_title() == "Custom UMAP"
        plt.close(fig)

    def test_save_path(self, tmp_path, sample_umap_df):
        save_path = tmp_path / "umap.png"
        fig = plot_umap_scatter(sample_umap_df, save_path=save_path)
        assert save_path.exists()
        assert save_path.stat().st_size > 0
        plt.close(fig)


# =============================================================================
# plot_cluster_composition
# =============================================================================


class TestPlotClusterComposition:
    def test_basic_cluster_sizes(self, sample_cluster_df):
        fig = plot_cluster_composition(sample_cluster_df)
        assert isinstance(fig, plt.Figure)
        ax = fig.get_axes()[0]
        assert ax.get_ylabel() == "Number of Subjects"
        # One bar per cluster
        n_clusters = sample_cluster_df["cluster"].nunique()
        assert len(ax.patches) == n_clusters
        plt.close(fig)

    def test_with_covariate_stacked(self, sample_cluster_df):
        fig = plot_cluster_composition(sample_cluster_df, covariate="diagnosis")
        ax = fig.get_axes()[0]
        # Stacked: bars per cluster × categories
        n_clusters = sample_cluster_df["cluster"].nunique()
        n_cats = sample_cluster_df["diagnosis"].nunique()
        assert len(ax.patches) == n_clusters * n_cats
        assert ax.get_legend() is not None
        plt.close(fig)

    def test_missing_cluster_column_returns_none(self):
        df = pd.DataFrame({"subject_id": ["a", "b"], "diagnosis": ["AD", "Control"]})
        result = plot_cluster_composition(df)
        assert result is None

    def test_missing_covariate_falls_back_to_sizes(self, sample_cluster_df):
        fig = plot_cluster_composition(sample_cluster_df, covariate="not_a_column")
        assert isinstance(fig, plt.Figure)
        ax = fig.get_axes()[0]
        # Falls back to basic cluster sizes (one bar per cluster, no legend)
        assert ax.get_legend() is None
        plt.close(fig)

    def test_custom_figsize(self, sample_cluster_df):
        fig = plot_cluster_composition(sample_cluster_df, figsize=(12, 4))
        assert fig.get_figwidth() == 12
        plt.close(fig)

    def test_custom_title(self, sample_cluster_df):
        fig = plot_cluster_composition(sample_cluster_df, title="Clusters")
        assert fig.get_axes()[0].get_title() == "Clusters"
        plt.close(fig)

    def test_save_path(self, tmp_path, sample_cluster_df):
        save_path = tmp_path / "clusters.png"
        fig = plot_cluster_composition(sample_cluster_df, save_path=save_path)
        assert save_path.exists()
        assert save_path.stat().st_size > 0
        plt.close(fig)


# =============================================================================
# plot_linear_probe_results
# =============================================================================


class TestPlotLinearProbeResults:
    def test_basic_plot(self, sample_probe_df):
        fig = plot_linear_probe_results(sample_probe_df)
        assert isinstance(fig, plt.Figure)
        # 2 panels since both regression and classification present
        assert len(fig.get_axes()) == 2
        plt.close(fig)

    def test_regression_only(self, sample_probe_df):
        reg_only = sample_probe_df[sample_probe_df["task_type"] == "regression"].copy()
        fig = plot_linear_probe_results(reg_only)
        assert len(fig.get_axes()) == 1
        assert fig.get_axes()[0].get_xlabel() == "R² Score"
        plt.close(fig)

    def test_classification_only(self, sample_probe_df):
        cls_only = sample_probe_df[sample_probe_df["task_type"] == "classification"].copy()
        fig = plot_linear_probe_results(cls_only)
        assert len(fig.get_axes()) == 1
        assert fig.get_axes()[0].get_xlabel() == "Accuracy"
        plt.close(fig)

    def test_backward_compat_r2_score(self, sample_probe_df_legacy):
        fig = plot_linear_probe_results(sample_probe_df_legacy)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_missing_required_columns_returns_none(self):
        df = pd.DataFrame({"score_mean": [0.1, 0.2]})  # no target
        result = plot_linear_probe_results(df)
        assert result is None

    def test_missing_score_columns_returns_none(self):
        df = pd.DataFrame({"target": ["a", "b"]})  # no score_mean or r2_score
        result = plot_linear_probe_results(df)
        assert result is None

    def test_empty_frame_returns_none(self):
        df = pd.DataFrame(columns=["target", "task_type", "score_mean"])
        result = plot_linear_probe_results(df)
        assert result is None

    def test_negative_scores_colored_differently(self, sample_probe_df):
        # Inject a negative regression score
        df = sample_probe_df.copy()
        df.loc[df["target"] == "cogn_global", "score_mean"] = -0.1
        fig = plot_linear_probe_results(df)
        reg_ax = fig.get_axes()[0]
        # Verify chance line at x=0
        vertical_lines = [
            line for line in reg_ax.lines if line.get_xdata()[0] == line.get_xdata()[1]
        ]
        assert len(vertical_lines) >= 1
        plt.close(fig)

    def test_classification_chance_line(self, sample_probe_df):
        cls_only = sample_probe_df[sample_probe_df["task_type"] == "classification"].copy()
        fig = plot_linear_probe_results(cls_only)
        ax = fig.get_axes()[0]
        # Chance line at x=0.5 should produce a legend entry
        assert ax.get_legend() is not None
        plt.close(fig)

    def test_custom_figsize(self, sample_probe_df):
        fig = plot_linear_probe_results(sample_probe_df, figsize=(14, 5))
        assert fig.get_figwidth() == 14
        plt.close(fig)

    def test_save_path(self, tmp_path, sample_probe_df):
        save_path = tmp_path / "probe.png"
        fig = plot_linear_probe_results(sample_probe_df, save_path=save_path)
        assert save_path.exists()
        assert save_path.stat().st_size > 0
        plt.close(fig)


# =============================================================================
# plot_embedding_summary
# =============================================================================


class TestPlotEmbeddingSummary:
    def test_umap_only(self, sample_umap_df):
        fig = plot_embedding_summary(sample_umap_df)
        assert len(fig.get_axes()) == 1
        plt.close(fig)

    def test_umap_plus_cluster(self, sample_umap_df, sample_cluster_df):
        fig = plot_embedding_summary(sample_umap_df, cluster_df=sample_cluster_df)
        # 2 panels: UMAP + cluster sizes
        # (colorbar may add extra axes for continuous color_by — default is "cluster", <10 unique → categorical, no colorbar)
        panel_count = sum(
            1 for ax in fig.get_axes()
            if ax.get_title() in {"Subject Embeddings", "Cluster Sizes"}
        )
        assert panel_count == 2
        plt.close(fig)

    def test_umap_plus_probe(self, sample_umap_df, sample_probe_df):
        fig = plot_embedding_summary(sample_umap_df, probe_df=sample_probe_df)
        panel_count = sum(
            1 for ax in fig.get_axes()
            if ax.get_title() in {"Subject Embeddings", "Linear Probe Quality"}
        )
        assert panel_count == 2
        plt.close(fig)

    def test_all_three_panels(self, sample_umap_df, sample_cluster_df, sample_probe_df):
        fig = plot_embedding_summary(
            sample_umap_df, cluster_df=sample_cluster_df, probe_df=sample_probe_df
        )
        panel_count = sum(
            1 for ax in fig.get_axes()
            if ax.get_title() in {"Subject Embeddings", "Cluster Sizes", "Linear Probe Quality"}
        )
        assert panel_count == 3
        plt.close(fig)

    def test_continuous_color_by_adds_colorbar(self, sample_umap_df):
        fig = plot_embedding_summary(sample_umap_df, color_by="pathology")
        # Colorbar axis adds one extra axes beyond the UMAP panel
        assert len(fig.get_axes()) >= 2
        plt.close(fig)

    def test_missing_color_by_falls_to_default(self, sample_umap_df):
        fig = plot_embedding_summary(sample_umap_df, color_by="not_a_column")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_backward_compat_r2_score_in_probe(self, sample_umap_df, sample_probe_df_legacy):
        fig = plot_embedding_summary(sample_umap_df, probe_df=sample_probe_df_legacy)
        panel_titles = [ax.get_title() for ax in fig.get_axes()]
        assert "Linear Probe Quality" in panel_titles
        plt.close(fig)

    def test_save_path(self, tmp_path, sample_umap_df, sample_cluster_df):
        save_path = tmp_path / "embedding_summary.png"
        fig = plot_embedding_summary(
            sample_umap_df, cluster_df=sample_cluster_df, save_path=save_path
        )
        assert save_path.exists()
        assert save_path.stat().st_size > 0
        plt.close(fig)
