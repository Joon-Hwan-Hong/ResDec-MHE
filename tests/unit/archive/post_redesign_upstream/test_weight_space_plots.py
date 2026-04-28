"""Tests for weight-space visualization plots."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.visualization.weight_space_plots import plot_checkpoint_weight_pca


@pytest.fixture
def small_weight_matrix():
    """5 checkpoints × 1000 flattened params."""
    rng = np.random.default_rng(0)
    return rng.normal(size=(5, 1000)).astype(np.float64)


class TestPlotCheckpointWeightPca:
    """Smoke tests for ``plot_checkpoint_weight_pca``."""

    def test_basic_plot(self, small_weight_matrix):
        fig = plot_checkpoint_weight_pca(small_weight_matrix)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_r2_annotations(self, small_weight_matrix):
        r2 = [0.44, 0.46, 0.39, 0.48, 0.41]
        fig = plot_checkpoint_weight_pca(
            small_weight_matrix,
            fold_labels=[f"fold {i}" for i in range(5)],
            r2_per_checkpoint=r2,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_save_path(self, tmp_path, small_weight_matrix):
        save_path = tmp_path / "landscape"
        fig = plot_checkpoint_weight_pca(
            small_weight_matrix, save_path=save_path,
        )
        assert save_path.with_suffix(".png").exists()
        plt.close(fig)

    def test_too_few_checkpoints_raises(self):
        single = np.zeros((1, 100))
        with pytest.raises(ValueError, match="≥2 checkpoints"):
            plot_checkpoint_weight_pca(single)

    def test_label_length_mismatch_raises(self, small_weight_matrix):
        with pytest.raises(ValueError, match="fold_labels length"):
            plot_checkpoint_weight_pca(
                small_weight_matrix, fold_labels=["a", "b"],
            )

    def test_r2_length_mismatch_raises(self, small_weight_matrix):
        with pytest.raises(ValueError, match="r2_per_checkpoint length"):
            plot_checkpoint_weight_pca(
                small_weight_matrix, r2_per_checkpoint=[0.1, 0.2],
            )

    def test_explained_variance_displayed(self, small_weight_matrix):
        """Axis labels should show explained variance percentages."""
        fig = plot_checkpoint_weight_pca(small_weight_matrix)
        ax = fig.get_axes()[0]
        assert "PC 1" in ax.get_xlabel()
        assert "var" in ax.get_xlabel()
        plt.close(fig)


@pytest.fixture(autouse=True)
def cleanup():
    yield
    plt.close("all")
