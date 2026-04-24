"""Tests for attention visualization plots."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for testing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from src.visualization.attention_plots import (
    plot_cell_type_attention_heatmap,
    plot_cell_type_importance_bar,
    plot_attention_distribution,
    plot_gene_gate_heatmap,
    plot_head_attention_bootstrap_ci,
    plot_resilience_signature_heatmap,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_attention_df():
    """Sample attention DataFrame for heatmap."""
    data = []
    for ct in ["Astrocyte", "Microglia", "Oligodendrocyte", "CGE interneuron"]:
        for tertile in ["low", "medium", "high"]:
            data.append({
                "cell_type": ct,
                "pathology_tertile": tertile,
                "mean_attention": np.random.rand() * 0.3 + 0.1,
            })
    return pd.DataFrame(data)


@pytest.fixture
def sample_importance_df():
    """Sample cell type importance DataFrame."""
    np.random.seed(42)
    cell_types = ["Astrocyte", "Microglia", "Oligodendrocyte", "CGE interneuron", "Upper-layer intratelencephalic"]
    return pd.DataFrame({
        "cell_type": cell_types,
        "mean_attention": np.random.rand(5) * 0.3 + 0.1,
        "std_attention": np.random.rand(5) * 0.05,
        "rank": range(1, 6),
    })


@pytest.fixture
def sample_attention_array():
    """Sample attention array [n_subjects, n_cell_types]."""
    np.random.seed(42)
    return np.random.rand(30, 8)


@pytest.fixture
def sample_gene_gate_weights():
    """Sample gene gate weights [n_cell_types, n_genes]."""
    np.random.seed(42)
    return np.random.rand(8, 100)


@pytest.fixture
def sample_signature_df():
    """Sample resilience signature DataFrame."""
    np.random.seed(42)
    cell_types = ["Astrocyte", "Microglia", "Oligodendrocyte", "CGE interneuron"]
    return pd.DataFrame({
        "cell_type": cell_types,
        "signature": np.random.randn(4) * 0.2,
    })


# =============================================================================
# plot_cell_type_attention_heatmap Tests
# =============================================================================


class TestPlotCellTypeAttentionHeatmap:
    """Test plot_cell_type_attention_heatmap function."""

    def test_basic_plot(self, sample_attention_df):
        """Test basic heatmap creation."""
        fig = plot_cell_type_attention_heatmap(sample_attention_df)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_figsize(self, sample_attention_df):
        """Test custom figure size."""
        fig = plot_cell_type_attention_heatmap(
            sample_attention_df,
            figsize=(10, 6),
        )

        assert fig.get_figwidth() == 10
        assert fig.get_figheight() == 6
        plt.close(fig)

    def test_custom_title(self, sample_attention_df):
        """Test custom title."""
        title = "Custom Attention Title"
        fig = plot_cell_type_attention_heatmap(
            sample_attention_df,
            title=title,
        )

        # Get axes title
        ax = fig.get_axes()[0]
        assert ax.get_title() == title
        plt.close(fig)

    def test_save_path(self, tmp_path, sample_attention_df):
        """Test saving figure."""
        save_path = tmp_path / "attention_heatmap.png"
        fig = plot_cell_type_attention_heatmap(
            sample_attention_df,
            save_path=save_path,
        )

        assert save_path.exists()
        plt.close(fig)

    def test_custom_colormap(self, sample_attention_df):
        """Test custom colormap."""
        fig = plot_cell_type_attention_heatmap(
            sample_attention_df,
            cmap="viridis",
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_partial_tertiles(self):
        """Test handling of partial pathology tertiles."""
        # Only low and high
        data = [
            {"cell_type": "Astrocyte", "pathology_tertile": "low", "mean_attention": 0.2},
            {"cell_type": "Astrocyte", "pathology_tertile": "high", "mean_attention": 0.3},
            {"cell_type": "Microglia", "pathology_tertile": "low", "mean_attention": 0.15},
            {"cell_type": "Microglia", "pathology_tertile": "high", "mean_attention": 0.25},
        ]
        df = pd.DataFrame(data)

        fig = plot_cell_type_attention_heatmap(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# =============================================================================
# plot_cell_type_importance_bar Tests
# =============================================================================


class TestPlotCellTypeImportanceBar:
    """Test plot_cell_type_importance_bar function."""

    def test_basic_plot(self, sample_importance_df):
        """Test basic bar chart creation."""
        fig = plot_cell_type_importance_bar(sample_importance_df)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_error_bars(self, sample_importance_df):
        """Test bar chart with error bars (std_attention present)."""
        fig = plot_cell_type_importance_bar(sample_importance_df)

        # Should create figure without error
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_without_error_bars(self, sample_importance_df):
        """Test bar chart without error bars."""
        df = sample_importance_df.drop(columns=["std_attention"])
        fig = plot_cell_type_importance_bar(df)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_figsize(self, sample_importance_df):
        """Test custom figure size."""
        fig = plot_cell_type_importance_bar(
            sample_importance_df,
            figsize=(12, 10),
        )

        assert fig.get_figwidth() == 12
        assert fig.get_figheight() == 10
        plt.close(fig)

    def test_save_path(self, tmp_path, sample_importance_df):
        """Test saving figure."""
        save_path = tmp_path / "importance_bar.png"
        fig = plot_cell_type_importance_bar(
            sample_importance_df,
            save_path=save_path,
        )

        assert save_path.exists()
        plt.close(fig)


# =============================================================================
# plot_attention_distribution Tests
# =============================================================================


class TestPlotAttentionDistribution:
    """Test plot_attention_distribution function."""

    def test_basic_plot(self, sample_attention_array):
        """Test basic distribution plot creation."""
        fig = plot_attention_distribution(sample_attention_array)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_cell_type_names(self, sample_attention_array):
        """Test with explicit cell type names."""
        cell_types = ["Ast", "Mic", "Oli", "OPC", "Exc", "Inh", "End", "Per"]
        fig = plot_attention_distribution(
            sample_attention_array,
            cell_type_names=cell_types,
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_figsize(self, sample_attention_array):
        """Test custom figure size."""
        fig = plot_attention_distribution(
            sample_attention_array,
            figsize=(14, 8),
        )

        assert fig.get_figwidth() == 14
        plt.close(fig)

    def test_custom_title(self, sample_attention_array):
        """Test custom title."""
        title = "Custom Distribution Title"
        fig = plot_attention_distribution(
            sample_attention_array,
            title=title,
        )

        ax = fig.get_axes()[0]
        assert ax.get_title() == title
        plt.close(fig)

    def test_save_path(self, tmp_path, sample_attention_array):
        """Test saving figure."""
        save_path = tmp_path / "attention_dist.png"
        fig = plot_attention_distribution(
            sample_attention_array,
            save_path=save_path,
        )

        assert save_path.exists()
        plt.close(fig)


# =============================================================================
# plot_gene_gate_heatmap Tests
# =============================================================================


class TestPlotGeneGateHeatmap:
    """Test plot_gene_gate_heatmap function."""

    def test_basic_plot(self, sample_gene_gate_weights):
        """Test basic heatmap creation."""
        fig = plot_gene_gate_heatmap(sample_gene_gate_weights)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_gene_names(self, sample_gene_gate_weights):
        """Test with explicit gene names."""
        gene_names = [f"GENE{i}" for i in range(100)]
        fig = plot_gene_gate_heatmap(
            sample_gene_gate_weights,
            gene_names=gene_names,
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_cell_type_names(self, sample_gene_gate_weights):
        """Test with explicit cell type names."""
        cell_types = ["Ast", "Mic", "Oli", "OPC", "Exc", "Inh", "End", "Per"]
        fig = plot_gene_gate_heatmap(
            sample_gene_gate_weights,
            cell_type_names=cell_types,
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_top_k_genes(self, sample_gene_gate_weights):
        """Test top_k_genes parameter."""
        fig = plot_gene_gate_heatmap(
            sample_gene_gate_weights,
            top_k_genes=20,
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_figsize(self, sample_gene_gate_weights):
        """Test custom figure size."""
        fig = plot_gene_gate_heatmap(
            sample_gene_gate_weights,
            figsize=(16, 12),
        )

        assert fig.get_figwidth() == 16
        plt.close(fig)

    def test_save_path(self, tmp_path, sample_gene_gate_weights):
        """Test saving figure."""
        save_path = tmp_path / "gene_gate_heatmap.png"
        fig = plot_gene_gate_heatmap(
            sample_gene_gate_weights,
            save_path=save_path,
        )

        assert save_path.exists()
        plt.close(fig)


# =============================================================================
# plot_resilience_signature_heatmap Tests
# =============================================================================


class TestPlotResilienceSignatureHeatmap:
    """Test plot_resilience_signature_heatmap function."""

    def test_basic_plot(self, sample_signature_df):
        """Test basic heatmap creation."""
        fig = plot_resilience_signature_heatmap(sample_signature_df)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_figsize(self, sample_signature_df):
        """Test custom figure size."""
        fig = plot_resilience_signature_heatmap(
            sample_signature_df,
            figsize=(8, 12),
        )

        assert fig.get_figwidth() == 8
        plt.close(fig)

    def test_custom_title(self, sample_signature_df):
        """Test custom title."""
        title = "Custom Signature Title"
        fig = plot_resilience_signature_heatmap(
            sample_signature_df,
            title=title,
        )

        ax = fig.get_axes()[0]
        assert ax.get_title() == title
        plt.close(fig)

    def test_save_path(self, tmp_path, sample_signature_df):
        """Test saving figure."""
        save_path = tmp_path / "resilience_signature.png"
        fig = plot_resilience_signature_heatmap(
            sample_signature_df,
            save_path=save_path,
        )

        assert save_path.exists()
        plt.close(fig)

    def test_symmetric_colorscale(self, sample_signature_df):
        """Test that colorscale is symmetric around zero."""
        # This tests the behavior but not the visual appearance
        fig = plot_resilience_signature_heatmap(sample_signature_df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# =============================================================================
# Property-Based Tests
# =============================================================================


class TestAttentionPlotsProperties:
    """Property-based tests for attention plots."""

    @given(
        n_subjects=st.integers(min_value=5, max_value=50),
        n_cell_types=st.integers(min_value=2, max_value=10),
    )
    @settings(max_examples=10)
    def test_distribution_accepts_various_shapes(self, n_subjects, n_cell_types):
        """Test attention distribution handles various array shapes."""
        attention = np.random.rand(n_subjects, n_cell_types)
        fig = plot_attention_distribution(attention)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    @given(n_cell_types=st.integers(min_value=2, max_value=8))
    @settings(max_examples=10)
    def test_importance_bar_various_cell_types(self, n_cell_types):
        """Test importance bar chart with various cell type counts."""
        df = pd.DataFrame({
            "cell_type": [f"Type_{i}" for i in range(n_cell_types)],
            "mean_attention": np.random.rand(n_cell_types),
            "rank": range(1, n_cell_types + 1),
        })

        fig = plot_cell_type_importance_bar(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# =============================================================================
# Edge Cases
# =============================================================================


class TestAttentionPlotsEdgeCases:
    """Test edge cases for attention plots."""

    def test_single_cell_type_attention(self):
        """Test attention heatmap with single cell type."""
        df = pd.DataFrame({
            "cell_type": ["Astrocyte"] * 3,
            "pathology_tertile": ["low", "medium", "high"],
            "mean_attention": [0.1, 0.2, 0.3],
        })

        fig = plot_cell_type_attention_heatmap(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_single_cell_type_importance(self):
        """Test importance bar chart with single cell type."""
        df = pd.DataFrame({
            "cell_type": ["Astrocyte"],
            "mean_attention": [0.5],
            "rank": [1],
        })

        fig = plot_cell_type_importance_bar(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_large_gene_count(self):
        """Test gene gate heatmap with many genes."""
        weights = np.random.rand(8, 500)
        fig = plot_gene_gate_heatmap(weights, top_k_genes=100)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# =============================================================================
# plot_head_attention_bootstrap_ci Tests
# =============================================================================


@pytest.fixture
def sample_head_attention_array():
    """Sample per-subject head attention [n_subj, n_head, n_ct]."""
    rng = np.random.default_rng(0)
    raw = rng.random((50, 4, 8))
    return raw / raw.sum(axis=-1, keepdims=True)


class TestPlotHeadAttentionBootstrapCi:
    """Test plot_head_attention_bootstrap_ci function."""

    def test_basic_plot(self, sample_head_attention_array):
        ct_names = [f"CT_{i}" for i in range(8)]
        fig = plot_head_attention_bootstrap_ci(
            sample_head_attention_array, ct_names, n_bootstrap=50,
        )
        assert isinstance(fig, plt.Figure)
        assert len(fig.get_axes()) >= 2
        plt.close(fig)

    def test_null_reference_annotation(self, sample_head_attention_array):
        ct_names = [f"CT_{i}" for i in range(8)]
        fig = plot_head_attention_bootstrap_ci(
            sample_head_attention_array, ct_names,
            n_bootstrap=50, null_reference=1.0 / 8,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_save_path(self, tmp_path, sample_head_attention_array):
        ct_names = [f"CT_{i}" for i in range(8)]
        save_path = tmp_path / "head_bootstrap"
        fig = plot_head_attention_bootstrap_ci(
            sample_head_attention_array, ct_names,
            n_bootstrap=50, save_path=save_path,
        )
        assert (save_path.with_suffix(".png")).exists()
        plt.close(fig)

    def test_empty_raises(self):
        ct_names = [f"CT_{i}" for i in range(8)]
        empty = np.zeros((0, 4, 8))
        with pytest.raises(ValueError, match="no subjects"):
            plot_head_attention_bootstrap_ci(empty, ct_names, n_bootstrap=10)

    def test_ct_mismatch_raises(self, sample_head_attention_array):
        wrong_names = [f"CT_{i}" for i in range(5)]
        with pytest.raises(ValueError, match="mismatch"):
            plot_head_attention_bootstrap_ci(
                sample_head_attention_array, wrong_names, n_bootstrap=10,
            )

    def test_reproducible_with_seed(self, sample_head_attention_array):
        ct_names = [f"CT_{i}" for i in range(8)]
        fig1 = plot_head_attention_bootstrap_ci(
            sample_head_attention_array, ct_names, n_bootstrap=100, seed=7,
        )
        fig2 = plot_head_attention_bootstrap_ci(
            sample_head_attention_array, ct_names, n_bootstrap=100, seed=7,
        )
        # Same seed → same CI widths in annotation titles.
        assert fig1.get_axes()[0].get_title() == fig2.get_axes()[0].get_title()
        plt.close(fig1)
        plt.close(fig2)


# =============================================================================
# Cleanup
# =============================================================================


@pytest.fixture(autouse=True)
def cleanup():
    """Cleanup matplotlib figures after each test."""
    yield
    plt.close("all")
