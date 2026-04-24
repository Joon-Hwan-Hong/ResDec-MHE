"""Tests for activation-cascade visualization."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.visualization.activation_plots import (
    plot_per_stage_activation_cascade,
)


@pytest.fixture
def cascade_norms():
    rng = np.random.default_rng(0)
    return {
        "input": rng.lognormal(2.0, 0.3, size=32),
        "gene_gate": rng.lognormal(1.5, 0.3, size=32),
        "hgt": rng.lognormal(1.8, 0.3, size=32),
        "pma": rng.lognormal(1.2, 0.3, size=32),
        "head": rng.lognormal(0.8, 0.3, size=32),
    }


class TestPlotPerStageActivationCascade:
    def test_basic(self, cascade_norms):
        fig = plot_per_stage_activation_cascade(cascade_norms)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_save_path(self, tmp_path, cascade_norms):
        save_path = tmp_path / "cascade"
        fig = plot_per_stage_activation_cascade(
            cascade_norms, save_path=save_path,
        )
        assert save_path.with_suffix(".png").exists()
        plt.close(fig)

    def test_linear_y(self, cascade_norms):
        fig = plot_per_stage_activation_cascade(
            cascade_norms, log_y=False,
        )
        ax = fig.get_axes()[0]
        assert ax.get_yscale() == "linear"
        plt.close(fig)

    def test_log_y_default(self, cascade_norms):
        fig = plot_per_stage_activation_cascade(cascade_norms)
        ax = fig.get_axes()[0]
        assert ax.get_yscale() == "log"
        plt.close(fig)

    def test_too_few_stages_raises(self):
        with pytest.raises(ValueError, match="≥2 stages"):
            plot_per_stage_activation_cascade({"only": np.array([1.0, 2.0])})

    def test_empty_stage_raises(self):
        with pytest.raises(ValueError, match="empty"):
            plot_per_stage_activation_cascade({
                "a": np.array([1.0, 2.0]),
                "b": np.array([]),
            })

    def test_all_nan_stage_raises(self):
        with pytest.raises(ValueError, match="no finite"):
            plot_per_stage_activation_cascade({
                "a": np.array([1.0, 2.0]),
                "b": np.array([np.nan, np.nan, np.nan]),
            })

    def test_mixed_stage_sizes(self):
        """Stages can have different numbers of subjects (e.g., masks differ)."""
        fig = plot_per_stage_activation_cascade({
            "a": np.array([1.0, 2.0, 3.0, 4.0]),
            "b": np.array([5.0, 6.0]),
            "c": np.array([7.0]),
        })
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


@pytest.fixture(autouse=True)
def cleanup():
    yield
    plt.close("all")
