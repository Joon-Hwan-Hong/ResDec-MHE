"""Smoke tests for src/visualization/composite.py."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.visualization.composite import (
    auto_letter,
    make_facet_grid,
    make_panel,
)

def test_auto_letter_top_left():
    fig, ax = plt.subplots()
    auto_letter(ax, "A")
    texts = [t for t in ax.texts if t.get_text() == "A"]
    assert len(texts) == 1
    plt.close(fig)

def test_auto_letter_loc_options():
    fig, ax = plt.subplots()
    for letter, loc in [("A", "top-left"), ("B", "top-right"),
                        ("C", "bottom-left"), ("D", "bottom-right")]:
        auto_letter(ax, letter, loc=loc)
    assert sum(1 for t in ax.texts if t.get_text() in {"A", "B", "C", "D"}) == 4
    plt.close(fig)

def test_make_panel_tuple_layout_2x2():
    """4 panels, 2x2 layout, each panel just plots a line."""
    panels = [
        {"draw": (lambda ax: ax.plot([1, 2, 3], [1, 4, 9])), "title": f"Panel {i}"}
        for i in range(4)
    ]
    fig = make_panel(panels, layout=(2, 2), figsize=(6, 5))
    # 4 axes
    assert len(fig.axes) == 4
    # Letters A B C D should be present
    letters = {t.get_text() for ax in fig.axes for t in ax.texts}
    assert {"A", "B", "C", "D"}.issubset(letters)
    plt.close(fig)

def test_make_panel_string_layout_with_span():
    """String layout with ``"AAB;CCB"`` → A spans top-left 2 cols, B spans full right col."""
    panels = [
        {"draw": (lambda ax: ax.bar([0, 1, 2], [3, 1, 2])), "title": "wide"},
        {"draw": (lambda ax: ax.scatter([1, 2, 3], [3, 1, 2])), "title": "tall"},
        {"draw": (lambda ax: ax.hist([1, 1, 2, 2, 3])), "title": "small"},
    ]
    fig = make_panel(panels, layout="AAB;CCB", figsize=(7, 5))
    assert len(fig.axes) == 3
    plt.close(fig)

def test_make_panel_no_labels():
    panels = [
        {"draw": (lambda ax: ax.plot([0, 1])), "title": "test"},
    ]
    fig = make_panel(panels, layout=(1, 1), labels=False)
    # No A label
    letters = {t.get_text() for ax in fig.axes for t in ax.texts}
    assert "A" not in letters
    plt.close(fig)

def test_make_panel_custom_label_letters():
    panels = [{"draw": (lambda ax: ax.plot([0, 1])), "title": ""} for _ in range(2)]
    fig = make_panel(panels, layout=(1, 2), label_letters=["i", "ii"])
    letters = {t.get_text() for ax in fig.axes for t in ax.texts}
    assert "i" in letters and "ii" in letters
    plt.close(fig)

def test_make_facet_grid_col_only():
    """4 records grouped by 'ct' → 4 panels in 1 row of ncols=4."""
    records = [
        {"ct": "Splatter", "x": 1, "y": 1.0},
        {"ct": "Fibroblast", "x": 1, "y": 2.0},
        {"ct": "Vascular", "x": 1, "y": 3.0},
        {"ct": "MGE interneuron", "x": 1, "y": 4.0},
    ]

    def draw(ax, subset):
        ax.bar([0], [subset[0]["y"]])

    fig = make_facet_grid(records, draw, col="ct", ncols=4)
    # 4 axes (1x4)
    assert len(fig.axes) == 4
    plt.close(fig)

def test_make_facet_grid_unused_panels_hidden():
    """5 records, ncols=3 → 2x3=6 panels, last one hidden via axis('off')."""
    records = [{"ct": f"CT_{i}", "y": float(i)} for i in range(5)]

    def draw(ax, subset):
        ax.scatter([0], [subset[0]["y"]])

    fig = make_facet_grid(records, draw, col="ct", ncols=3)
    # 6 axes total (2 rows × 3 cols), 1 hidden via axis('off')
    assert len(fig.axes) == 6
    axes_off = [ax for ax in fig.axes if not ax.axison]
    assert len(axes_off) == 1
    plt.close(fig)

def test_make_facet_grid_with_row_and_col():
    """2x2 facet grid via row + col."""
    records = []
    for fold in [0, 1]:
        for ct in ["Splatter", "Fibroblast"]:
            records.append({"fold": fold, "ct": ct, "y": fold + 0.5})

    def draw(ax, subset):
        if subset:
            ax.bar([0], [subset[0]["y"]])

    fig = make_facet_grid(records, draw, col="ct", row="fold")
    assert len(fig.axes) == 4
    plt.close(fig)

def test_make_panel_callbacks_receive_axes():
    """Confirm that draw callbacks receive a matplotlib Axes."""
    received = []

    def make_capture(label):
        def _draw(ax):
            received.append((label, ax))
            ax.plot([0, 1], [0, 1])
        return _draw

    panels = [{"draw": make_capture(i), "title": str(i)} for i in range(3)]
    fig = make_panel(panels, layout=(1, 3))
    assert len(received) == 3
    for label, ax in received:
        assert hasattr(ax, "plot")
    plt.close(fig)
