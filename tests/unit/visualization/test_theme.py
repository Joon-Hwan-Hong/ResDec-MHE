"""Unit tests for src/visualization/theme.py."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from src.visualization.theme import (
    BASELINE_COLORS,
    PALETTES,
    apply_theme,
    baseline_color,
    errorbar_caps,
    errorbar_ribbon,
    fmt_axes,
    save_fig,
)


def test_palettes_have_required_keys():
    required = {"categorical", "categorical_paired", "sequential", "diverging", "fold_colors"}
    assert required.issubset(PALETTES.keys())


def test_fold_colors_count_matches_outer_folds():
    assert len(PALETTES["fold_colors"]) == 5


def test_apply_theme_sets_spines_off_by_default():
    apply_theme()
    assert matplotlib.rcParams["axes.spines.top"] is False
    assert matplotlib.rcParams["axes.spines.right"] is False


def test_apply_theme_sets_savefig_dpi_600():
    apply_theme()
    assert matplotlib.rcParams["savefig.dpi"] == 600


def test_apply_theme_sets_tick_direction_out():
    apply_theme()
    assert matplotlib.rcParams["xtick.direction"] == "out"
    assert matplotlib.rcParams["ytick.direction"] == "out"


def test_fmt_axes_hides_top_and_right_by_default():
    fig, ax = plt.subplots()
    fmt_axes(ax)
    assert ax.spines["top"].get_visible() is False
    assert ax.spines["right"].get_visible() is False
    assert ax.spines["bottom"].get_visible() is True
    assert ax.spines["left"].get_visible() is True
    plt.close(fig)


def test_fmt_axes_can_show_all_spines():
    fig, ax = plt.subplots()
    fmt_axes(ax, hide_spines=())
    for s in ("top", "right", "bottom", "left"):
        assert ax.spines[s].get_visible() is True
    plt.close(fig)


def test_save_fig_writes_both_png_and_pdf(tmp_path):
    apply_theme()
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    paths = save_fig(fig, tmp_path / "test_fig", dpi=72)
    assert len(paths) == 2
    assert (tmp_path / "test_fig.png").exists()
    assert (tmp_path / "test_fig.pdf").exists()
    plt.close(fig)


def test_save_fig_creates_parent_dirs(tmp_path):
    apply_theme()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    out_stem = tmp_path / "deeply" / "nested" / "fig"
    save_fig(fig, out_stem, dpi=72)
    assert (tmp_path / "deeply" / "nested" / "fig.png").exists()
    plt.close(fig)


def test_errorbar_caps_returns_artist(tmp_path):
    apply_theme()
    fig, ax = plt.subplots()
    artist = errorbar_caps(ax, [0, 1, 2], [1, 2, 3], yerr=[0.1, 0.2, 0.3], color="C0")
    assert artist is not None
    plt.close(fig)


def test_errorbar_ribbon_returns_line(tmp_path):
    apply_theme()
    fig, ax = plt.subplots()
    line = errorbar_ribbon(ax, [0, 1, 2], [1, 2, 3], yerr=[0.1, 0.2, 0.3], color="C0")
    assert line is not None
    # The fill_between adds at least one collection to ax.
    assert len(ax.collections) >= 1
    plt.close(fig)


def test_baseline_color_exact_match():
    assert baseline_color("TabPFN-2.6") == BASELINE_COLORS["TabPFN-2.6"]


def test_baseline_color_prefix_fallback():
    # "TabPFN-2.6 standalone (foo)" should still match TabPFN-2.6 prefix.
    color = baseline_color("TabPFN-2.6 standalone (foo)")
    assert color == BASELINE_COLORS["TabPFN-2.6 standalone"]


def test_baseline_color_unknown_returns_default():
    color = baseline_color("UnknownBaselineXYZ", default="#000000")
    assert color == "#000000"
