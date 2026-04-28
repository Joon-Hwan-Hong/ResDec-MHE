"""Tests for src/visualization/learning_curve_plots.py."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from src.visualization.learning_curve_plots import plot_learning_curve_n_vs_r2


def _toy_results():
    return [
        {"N": 100, "rng_seed": 42, "mean_r2": 0.35, "std_r2": 0.10},
        {"N": 200, "rng_seed": 42, "mean_r2": 0.37, "std_r2": 0.10},
        {"N": 100, "rng_seed": 67, "mean_r2": 0.33, "std_r2": 0.11},
        {"N": 200, "rng_seed": 67, "mean_r2": 0.36, "std_r2": 0.10},
    ]


def test_basic_figure_returns_figure():
    fig = plot_learning_curve_n_vs_r2(_toy_results())
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_save_path_writes_png_and_pdf(tmp_path):
    save = tmp_path / "lc"
    fig = plot_learning_curve_n_vs_r2(_toy_results(), save_path=save)
    assert save.with_suffix(".png").exists()
    assert save.with_suffix(".pdf").exists()
    plt.close(fig)


def test_canonical_anchor_drawn(tmp_path):
    fig = plot_learning_curve_n_vs_r2(_toy_results(), canonical_r2=0.444)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        plot_learning_curve_n_vs_r2([])


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    plt.close("all")
