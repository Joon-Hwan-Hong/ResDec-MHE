"""Tests for plot_predicted_vs_actual: existing single-color path + new
KDE-marginal and color-by-attribute extensions.

Behavioural contract under test:
    1. Backward compatibility: calling with no new kwargs reproduces the
       prior behaviour (single ax, R^2 in legend, RMSE/MAE/R^2 annotation,
       N points scattered).
    2. add_marginals=True opens KDE marginal panels on top + right of the
       main scatter.
    3. color_by + color_label + color_palette adds a per-category legend
       AND results in more than one scatter color.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.visualization.prediction_plots import plot_predicted_vs_actual

# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_xy() -> dict[str, np.ndarray]:
    """Synthetic 50-point regression-like data."""
    rng = np.random.default_rng(42)
    n = 50
    actual = rng.normal(size=n) * 2.0
    predicted_mean = actual + rng.normal(size=n) * 0.5
    return {"actual": actual, "predicted_mean": predicted_mean}

@pytest.fixture
def sample_categorical(sample_xy):
    """Two-class color array + palette dict."""
    n = sample_xy["actual"].shape[0]
    rng = np.random.default_rng(0)
    classes = rng.choice(np.array(["F", "M"]), size=n)
    palette = {"F": "#E76A7B", "M": "#189584"}
    return {"color_by": classes, "palette": palette, "label": "Sex"}

@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")

# =============================================================================
# Backward compatibility — no new kwargs
# =============================================================================

def test_basic_call_returns_figure(sample_xy):
    fig = plot_predicted_vs_actual(
        predicted_mean=sample_xy["predicted_mean"],
        actual=sample_xy["actual"],
    )
    assert isinstance(fig, plt.Figure)

def test_basic_call_has_metrics_annotation(sample_xy):
    fig = plot_predicted_vs_actual(
        predicted_mean=sample_xy["predicted_mean"],
        actual=sample_xy["actual"],
    )
    ax = fig.get_axes()[0]
    text_content = " ".join(t.get_text() for t in ax.texts)
    assert "RMSE" in text_content
    assert "MAE" in text_content
    assert "R" in text_content  # mathtext "R²" or "R^2"

def test_basic_call_has_identity_and_fit_lines(sample_xy):
    fig = plot_predicted_vs_actual(
        predicted_mean=sample_xy["predicted_mean"],
        actual=sample_xy["actual"],
    )
    ax = fig.get_axes()[0]
    # identity + fit, both as Line2D
    assert len(ax.lines) >= 2

def test_basic_call_scatter_count_matches_n(sample_xy):
    fig = plot_predicted_vs_actual(
        predicted_mean=sample_xy["predicted_mean"],
        actual=sample_xy["actual"],
    )
    ax = fig.get_axes()[0]
    # PathCollection from ax.scatter
    n_pts = sum(c.get_offsets().shape[0] for c in ax.collections if hasattr(c, "get_offsets"))
    assert n_pts == sample_xy["actual"].shape[0]

def test_basic_call_axes_labels(sample_xy):
    fig = plot_predicted_vs_actual(
        predicted_mean=sample_xy["predicted_mean"],
        actual=sample_xy["actual"],
    )
    ax = fig.get_axes()[0]
    assert "Actual" in ax.get_xlabel()
    assert "Predicted" in ax.get_ylabel()

# =============================================================================
# add_marginals=True — KDE marginal panels
# =============================================================================

def test_marginals_creates_three_or_four_axes(sample_xy):
    """Main axis + top KDE + right KDE (corner spacer optional)."""
    fig = plot_predicted_vs_actual(
        predicted_mean=sample_xy["predicted_mean"],
        actual=sample_xy["actual"],
        add_marginals=True,
    )
    # Expect at least 3 axes (main + top + right). 4 if a corner spacer is created.
    assert len(fig.get_axes()) >= 3

def test_marginals_main_axis_still_has_metrics(sample_xy):
    fig = plot_predicted_vs_actual(
        predicted_mean=sample_xy["predicted_mean"],
        actual=sample_xy["actual"],
        add_marginals=True,
    )
    # The largest-area axis is the main scatter axis. Find it by bounding box.
    axes = fig.get_axes()
    main_ax = max(
        axes,
        key=lambda a: a.get_position().width * a.get_position().height,
    )
    text_content = " ".join(t.get_text() for t in main_ax.texts)
    assert "RMSE" in text_content
    assert "MAE" in text_content

# =============================================================================
# color_by — per-category coloring + legend
# =============================================================================

def test_color_by_adds_legend_label(sample_xy, sample_categorical):
    fig = plot_predicted_vs_actual(
        predicted_mean=sample_xy["predicted_mean"],
        actual=sample_xy["actual"],
        color_by=sample_categorical["color_by"],
        color_label=sample_categorical["label"],
        color_palette=sample_categorical["palette"],
    )
    axes = fig.get_axes()
    main_ax = max(
        axes,
        key=lambda a: a.get_position().width * a.get_position().height,
    )
    legend = main_ax.get_legend()
    assert legend is not None
    legend_str = " ".join(t.get_text() for t in legend.get_texts())
    # category labels should appear in the legend
    assert "F" in legend_str or "M" in legend_str

def test_color_by_uses_more_than_one_color(sample_xy, sample_categorical):
    fig = plot_predicted_vs_actual(
        predicted_mean=sample_xy["predicted_mean"],
        actual=sample_xy["actual"],
        color_by=sample_categorical["color_by"],
        color_label=sample_categorical["label"],
        color_palette=sample_categorical["palette"],
    )
    axes = fig.get_axes()
    main_ax = max(
        axes,
        key=lambda a: a.get_position().width * a.get_position().height,
    )
    # Each category creates a separate scatter call → multiple PathCollections.
    scatter_collections = [
        c for c in main_ax.collections if hasattr(c, "get_offsets") and c.get_offsets().shape[0] > 0
    ]
    assert len(scatter_collections) >= 2

def test_color_by_with_marginals_combined(sample_xy, sample_categorical):
    fig = plot_predicted_vs_actual(
        predicted_mean=sample_xy["predicted_mean"],
        actual=sample_xy["actual"],
        add_marginals=True,
        color_by=sample_categorical["color_by"],
        color_label=sample_categorical["label"],
        color_palette=sample_categorical["palette"],
    )
    assert isinstance(fig, plt.Figure)
    assert len(fig.get_axes()) >= 3

def test_color_by_save_path_writes_file(tmp_path, sample_xy, sample_categorical):
    out = tmp_path / "fig_color_by.png"
    plot_predicted_vs_actual(
        predicted_mean=sample_xy["predicted_mean"],
        actual=sample_xy["actual"],
        add_marginals=True,
        color_by=sample_categorical["color_by"],
        color_label=sample_categorical["label"],
        color_palette=sample_categorical["palette"],
        save_path=out,
    )
    assert out.exists()
    assert out.stat().st_size > 0
