"""Tests for distributional-analysis visualization plots."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.visualization.distributional_plots import (
    plot_de_method_concordance_bar,
    plot_stability_selection_bar,
    plot_wasserstein_per_celltype_bar,
)


@pytest.fixture
def ct_names():
    return [f"CT_{i}" for i in range(8)]


class TestPlotWassersteinPerCelltypeBar:
    def test_basic(self, ct_names):
        w = [0.05, 0.03, 0.02, 0.04, 0.01, 0.06, 0.025, 0.015]
        fig = plot_wasserstein_per_celltype_bar(ct_names, w)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_top_genes(self, ct_names):
        w = np.random.default_rng(0).random(8).tolist()
        genes = [f"GENE_{i}" for i in range(8)]
        fig = plot_wasserstein_per_celltype_bar(ct_names, w, genes)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_save_path(self, tmp_path, ct_names):
        w = [0.01] * 8
        save_path = tmp_path / "wass"
        fig = plot_wasserstein_per_celltype_bar(
            ct_names, w, save_path=save_path,
        )
        assert save_path.with_suffix(".png").exists()
        plt.close(fig)

    def test_length_mismatch(self, ct_names):
        with pytest.raises(ValueError, match="length mismatch"):
            plot_wasserstein_per_celltype_bar(ct_names, [0.01] * 3)

    def test_empty(self):
        with pytest.raises(ValueError, match="no cell types"):
            plot_wasserstein_per_celltype_bar([], [])

    def test_gene_length_mismatch(self, ct_names):
        with pytest.raises(ValueError, match="top_gene_per_ct length"):
            plot_wasserstein_per_celltype_bar(
                ct_names, [0.01] * 8, ["g"] * 3,
            )


class TestPlotDeMethodConcordanceBar:
    def test_basic(self, ct_names):
        rho = [0.3, -0.1, 0.2, 0.0, -0.4, 0.5, 0.1, -0.2]
        fig = plot_de_method_concordance_bar(ct_names, rho)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_nan(self, ct_names):
        rho = [0.3, np.nan, 0.2, 0.0, np.nan, 0.5, 0.1, -0.2]
        fig = plot_de_method_concordance_bar(ct_names, rho)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_save_path(self, tmp_path, ct_names):
        rho = [0.1] * 8
        save_path = tmp_path / "conc"
        fig = plot_de_method_concordance_bar(
            ct_names, rho, save_path=save_path,
        )
        assert save_path.with_suffix(".png").exists()
        plt.close(fig)

    def test_custom_methods(self, ct_names):
        fig = plot_de_method_concordance_bar(
            ct_names, [0.1] * 8, method_labels=("A", "B"),
        )
        ax = fig.get_axes()[0]
        assert "A" in ax.get_xlabel() and "B" in ax.get_xlabel()
        plt.close(fig)


class TestPlotStabilitySelectionBar:
    def test_basic(self, ct_names):
        n_stable = [2, 0, 1, 0, 3, 0, 1, 0]
        fig = plot_stability_selection_bar(ct_names, n_stable)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_gene_annotations(self, ct_names):
        n_stable = [2, 0, 1, 0, 3, 0, 1, 0]
        stable_genes = [
            ["g1", "g2"], [], ["g3"], [], ["g4", "g5", "g6"],
            [], ["g7"], [],
        ]
        fig = plot_stability_selection_bar(
            ct_names, n_stable, stable_genes,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_save_path(self, tmp_path, ct_names):
        save_path = tmp_path / "stab"
        fig = plot_stability_selection_bar(
            ct_names, [0] * 8, save_path=save_path,
        )
        assert save_path.with_suffix(".png").exists()
        plt.close(fig)

    def test_gene_length_mismatch(self, ct_names):
        with pytest.raises(ValueError, match="stable_genes_per_ct length"):
            plot_stability_selection_bar(
                ct_names, [0] * 8, [["g"]] * 3,
            )


@pytest.fixture(autouse=True)
def cleanup():
    yield
    plt.close("all")
