"""Tests for importance visualization plots."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for testing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from src.visualization.importance_plots import (
    plot_top_genes_per_cell_type,
    plot_gene_importance_volcano,
    plot_ccc_network_summary,
    plot_top_interactions_heatmap,
    plot_regional_gene_importance,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_top_genes_df():
    """Sample top genes DataFrame."""
    data = []
    cell_types = ["Astrocyte", "Microglia", "Oligodendrocyte", "CGE interneuron"]
    for ct in cell_types:
        for rank in range(1, 11):
            data.append({
                "cell_type": ct,
                "rank": rank,
                "gene": f"GENE{rank + cell_types.index(ct) * 10}",
                "weight": np.random.rand() * 0.3 + 0.1,
            })
    return pd.DataFrame(data)


@pytest.fixture
def sample_gene_df():
    """Sample gene importance DataFrame."""
    np.random.seed(42)
    n_genes = 100
    return pd.DataFrame({
        "gene": [f"GENE{i}" for i in range(n_genes)],
        "weight": np.random.rand(n_genes) * 0.5,
        "p_value": np.random.rand(n_genes),
    })


@pytest.fixture
def sample_network_df():
    """Sample CCC network DataFrame."""
    return pd.DataFrame({
        "edge_type": ["Secreted_Signaling", "ECM_Receptor", "Cell_Cell_Contact"],
        "display_name": ["Secreted Signaling", "ECM-Receptor", "Cell-Cell Contact"],
        "mean_attention": [0.35, 0.25, 0.20],
        "std_attention": [0.05, 0.03, 0.04],
    })


@pytest.fixture
def sample_interactions_df():
    """Sample cell-cell interactions DataFrame."""
    data = []
    cell_types = ["Astrocyte", "Microglia", "Oligodendrocyte", "CGE interneuron"]
    for source in cell_types:
        for target in cell_types:
            data.append({
                "source": source,
                "target": target,
                "mean_attention": np.random.rand() * 0.3,
            })
    df = pd.DataFrame(data)
    return df.sort_values("mean_attention", ascending=False)


@pytest.fixture
def sample_regional_df():
    """Sample regional gene importance DataFrame."""
    data = []
    regions = ["DLPFC", "PCC", "AC"]
    cell_types = ["Astrocyte", "Microglia"]
    for region in regions:
        for ct in cell_types:
            for i in range(15):
                data.append({
                    "region": region,
                    "cell_type": ct,
                    "gene": f"GENE{i}_{region}",
                    "effective_weight": np.random.rand() * 0.5,
                })
    return pd.DataFrame(data)


# =============================================================================
# plot_top_genes_per_cell_type Tests
# =============================================================================


class TestPlotTopGenesPerCellType:
    """Test plot_top_genes_per_cell_type function."""

    def test_basic_plot(self, sample_top_genes_df):
        """Test basic faceted bar chart creation."""
        fig = plot_top_genes_per_cell_type(sample_top_genes_df)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_cell_types(self, sample_top_genes_df):
        """Test with specific cell types."""
        fig = plot_top_genes_per_cell_type(
            sample_top_genes_df,
            cell_types=["Astrocyte", "Microglia"],
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_n_genes(self, sample_top_genes_df):
        """Test with custom number of genes."""
        fig = plot_top_genes_per_cell_type(
            sample_top_genes_df,
            n_genes=5,
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_figsize(self, sample_top_genes_df):
        """Test custom figure size."""
        fig = plot_top_genes_per_cell_type(
            sample_top_genes_df,
            figsize=(16, 12),
        )

        assert fig.get_figwidth() == 16
        plt.close(fig)

    def test_save_path(self, tmp_path, sample_top_genes_df):
        """Test saving figure."""
        save_path = tmp_path / "top_genes.png"
        fig = plot_top_genes_per_cell_type(
            sample_top_genes_df,
            save_path=save_path,
        )

        assert save_path.exists()
        plt.close(fig)

    def test_single_cell_type(self, sample_top_genes_df):
        """Test with single cell type."""
        fig = plot_top_genes_per_cell_type(
            sample_top_genes_df,
            cell_types=["Astrocyte"],
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# =============================================================================
# plot_gene_importance_volcano Tests
# =============================================================================


class TestPlotGeneImportanceVolcano:
    """Test plot_gene_importance_volcano function."""

    def test_basic_plot_with_pvalues(self, sample_gene_df):
        """Test volcano plot with p-values."""
        fig = plot_gene_importance_volcano(sample_gene_df)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_without_pvalues(self, sample_gene_df):
        """Test fallback scatter when no p-values."""
        df = sample_gene_df.drop(columns=["p_value"])
        fig = plot_gene_importance_volcano(df)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_cell_type(self, sample_gene_df):
        """Test with cell type in title."""
        fig = plot_gene_importance_volcano(
            sample_gene_df,
            cell_type="Astrocyte",
        )

        ax = fig.get_axes()[0]
        assert "Astrocyte" in ax.get_title()
        plt.close(fig)

    def test_custom_significance(self, sample_gene_df):
        """Test custom significance threshold."""
        fig = plot_gene_importance_volcano(
            sample_gene_df,
            significance_threshold=0.01,
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_title(self, sample_gene_df):
        """Test custom title."""
        title = "Custom Volcano Title"
        fig = plot_gene_importance_volcano(
            sample_gene_df,
            title=title,
        )

        ax = fig.get_axes()[0]
        assert ax.get_title() == title
        plt.close(fig)

    def test_save_path(self, tmp_path, sample_gene_df):
        """Test saving figure."""
        save_path = tmp_path / "volcano.png"
        fig = plot_gene_importance_volcano(
            sample_gene_df,
            save_path=save_path,
        )

        assert save_path.exists()
        plt.close(fig)


# =============================================================================
# plot_ccc_network_summary Tests
# =============================================================================


class TestPlotCCCNetworkSummary:
    """Test plot_ccc_network_summary function."""

    def test_basic_plot(self, sample_network_df):
        """Test basic bar chart creation."""
        fig = plot_ccc_network_summary(sample_network_df)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_error_bars(self, sample_network_df):
        """Test with std_attention for error bars."""
        fig = plot_ccc_network_summary(sample_network_df)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_without_display_names(self, sample_network_df):
        """Test without display_name column."""
        df = sample_network_df.drop(columns=["display_name"])
        fig = plot_ccc_network_summary(df)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_without_error_bars(self, sample_network_df):
        """Test without std_attention column."""
        df = sample_network_df.drop(columns=["std_attention"])
        fig = plot_ccc_network_summary(df)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_figsize(self, sample_network_df):
        """Test custom figure size."""
        fig = plot_ccc_network_summary(
            sample_network_df,
            figsize=(12, 8),
        )

        assert fig.get_figwidth() == 12
        plt.close(fig)

    def test_save_path(self, tmp_path, sample_network_df):
        """Test saving figure."""
        save_path = tmp_path / "ccc_network.png"
        fig = plot_ccc_network_summary(
            sample_network_df,
            save_path=save_path,
        )

        assert save_path.exists()
        plt.close(fig)


# =============================================================================
# plot_top_interactions_heatmap Tests
# =============================================================================


class TestPlotTopInteractionsHeatmap:
    """Test plot_top_interactions_heatmap function."""

    def test_basic_plot(self, sample_interactions_df):
        """Test basic bar chart creation."""
        fig = plot_top_interactions_heatmap(sample_interactions_df)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_top_k(self, sample_interactions_df):
        """Test with custom top_k."""
        fig = plot_top_interactions_heatmap(
            sample_interactions_df,
            top_k=10,
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_figsize(self, sample_interactions_df):
        """Test custom figure size."""
        fig = plot_top_interactions_heatmap(
            sample_interactions_df,
            figsize=(12, 10),
        )

        assert fig.get_figwidth() == 12
        plt.close(fig)

    def test_custom_title(self, sample_interactions_df):
        """Test custom title."""
        title = "Custom Interactions Title"
        fig = plot_top_interactions_heatmap(
            sample_interactions_df,
            title=title,
        )

        ax = fig.get_axes()[0]
        assert title in ax.get_title()
        plt.close(fig)

    def test_save_path(self, tmp_path, sample_interactions_df):
        """Test saving figure."""
        save_path = tmp_path / "interactions.png"
        fig = plot_top_interactions_heatmap(
            sample_interactions_df,
            save_path=save_path,
        )

        assert save_path.exists()
        plt.close(fig)


# =============================================================================
# plot_regional_gene_importance Tests
# =============================================================================


class TestPlotRegionalGeneImportance:
    """Test plot_regional_gene_importance function."""

    def test_basic_plot(self, sample_regional_df):
        """Test basic faceted bar chart creation."""
        fig = plot_regional_gene_importance(sample_regional_df)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_regions(self, sample_regional_df):
        """Test with specific regions."""
        fig = plot_regional_gene_importance(
            sample_regional_df,
            regions=["DLPFC", "PCC"],
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_n_genes(self, sample_regional_df):
        """Test with custom number of genes."""
        fig = plot_regional_gene_importance(
            sample_regional_df,
            n_genes=5,
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_figsize(self, sample_regional_df):
        """Test custom figure size."""
        fig = plot_regional_gene_importance(
            sample_regional_df,
            figsize=(16, 10),
        )

        assert fig.get_figwidth() == 16
        plt.close(fig)

    def test_save_path(self, tmp_path, sample_regional_df):
        """Test saving figure."""
        save_path = tmp_path / "regional_importance.png"
        fig = plot_regional_gene_importance(
            sample_regional_df,
            save_path=save_path,
        )

        assert save_path.exists()
        plt.close(fig)

    def test_single_region(self, sample_regional_df):
        """Test with single region."""
        fig = plot_regional_gene_importance(
            sample_regional_df,
            regions=["DLPFC"],
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# =============================================================================
# Property-Based Tests
# =============================================================================


class TestImportancePlotsProperties:
    """Property-based tests for importance plots."""

    @given(
        n_cell_types=st.integers(min_value=1, max_value=6),
        n_genes=st.integers(min_value=5, max_value=15),
    )
    @settings(max_examples=10)
    def test_top_genes_various_sizes(self, n_cell_types, n_genes):
        """Test top genes plot with various cell type and gene counts."""
        data = []
        for ct_idx in range(n_cell_types):
            for rank in range(1, n_genes + 1):
                data.append({
                    "cell_type": f"Type_{ct_idx}",
                    "rank": rank,
                    "gene": f"GENE{rank}",
                    "weight": np.random.rand(),
                })
        df = pd.DataFrame(data)

        fig = plot_top_genes_per_cell_type(df, n_genes=n_genes)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    @given(n_genes=st.integers(min_value=10, max_value=200))
    @settings(max_examples=10)
    def test_volcano_various_gene_counts(self, n_genes):
        """Test volcano plot with various gene counts."""
        df = pd.DataFrame({
            "gene": [f"GENE{i}" for i in range(n_genes)],
            "weight": np.random.rand(n_genes),
            "p_value": np.random.rand(n_genes),
        })

        fig = plot_gene_importance_volcano(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# =============================================================================
# Edge Cases
# =============================================================================


class TestImportancePlotsEdgeCases:
    """Test edge cases for importance plots."""

    def test_volcano_all_significant(self):
        """Test volcano plot with all genes significant."""
        df = pd.DataFrame({
            "gene": [f"GENE{i}" for i in range(50)],
            "weight": np.random.rand(50),
            "p_value": np.random.rand(50) * 0.01,  # All < 0.05
        })

        fig = plot_gene_importance_volcano(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_volcano_none_significant(self):
        """Test volcano plot with no genes significant."""
        df = pd.DataFrame({
            "gene": [f"GENE{i}" for i in range(50)],
            "weight": np.random.rand(50),
            "p_value": np.random.rand(50) * 0.5 + 0.5,  # All > 0.5
        })

        fig = plot_gene_importance_volcano(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_single_interaction(self):
        """Test interactions plot with single interaction."""
        df = pd.DataFrame({
            "source": ["Astrocyte"],
            "target": ["Microglia"],
            "mean_attention": [0.5],
        })

        fig = plot_top_interactions_heatmap(df, top_k=1)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_single_edge_type(self):
        """Test network summary with single edge type."""
        df = pd.DataFrame({
            "edge_type": ["Secreted_Signaling"],
            "mean_attention": [0.35],
        })

        fig = plot_ccc_network_summary(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# =============================================================================
# Cleanup
# =============================================================================


@pytest.fixture(autouse=True)
def cleanup():
    """Cleanup matplotlib figures after each test."""
    yield
    plt.close("all")
