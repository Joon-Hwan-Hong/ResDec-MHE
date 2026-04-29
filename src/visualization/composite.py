"""Composite-figure infrastructure for paper-quality multi-panel layouts.

Two layouts the project actually needs (verified from §23 use cases in MASTER-INFO):

1. **Hand-crafted multi-method panel** (matplotlib subfigures + GridSpec).
   Use case: headline result, Splatter convergence (5 methods), statistical
   rigor (Wilcoxon + perm null + bootstrap + variance components), CF rel-vs-abs
   comparison.  Each sub-panel is a different plot type, so we need explicit
   per-panel control.

2. **Same-plot-type × N instances** (seaborn-style FacetGrid via matplotlib).
   Use case: per-CT Wasserstein top-10 (16 well-covered CTs), per-fold
   prediction scatter (5 folds), per-subject CF top-features (20 subjects),
   attention-head specialization (4 heads).

This module also provides ``auto_letter`` (panel labels A, B, C, ...) and
``apply_theme``-aware defaults so all composite figures inherit the project
typography + palettes from ``src.visualization.theme``.

NO patchworklib — matplotlib + seaborn cover all our cases.

Public API:

    make_panel(panels, layout="ABCD" or (rows, cols),
               figsize=(W, H), labels=True, label_kwargs={"loc": "top-left", "size": 10},
               wspace=0.3, hspace=0.4) -> matplotlib.figure.Figure
        Hand-crafted layout. Each panel is a dict with at least ``draw`` (a
        callable taking ``ax``) and optional ``title``, ``slot`` (index into
        the layout grid).  Returns the figure for further customization or
        ``save_fig`` consumption.

    make_facet_grid(records, draw_fn, col, row=None, ncols=4, sharex=True,
                    sharey=True, figsize_per_panel=(2.5, 2.0), label_panels=False)
        ``records`` is an iterable of dicts; ``col`` (and optional ``row``)
        names the grouping key; for each unique value, call ``draw_fn(ax,
        records_subset)``.  Returns the figure.

    auto_letter(ax, letter, *, loc="top-left", offset=(-0.04, 1.05), **kwargs)
        Add a panel-label letter (e.g. "A") to an axes.  Use directly for
        custom layouts.

All functions assume ``apply_theme()`` has been called by the orchestrator
(or call it themselves at the top of ``make_panel`` / ``make_facet_grid``
to be safe).
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from src.visualization.theme import apply_theme, fmt_axes, style_paper_axes


def auto_letter(
    ax: plt.Axes,
    letter: str,
    *,
    loc: str = "top-left",
    offset: tuple[float, float] = (-0.04, 1.05),
    fontsize: float = 10.0,
    fontweight: str = "bold",
    **kwargs: Any,
) -> None:
    """Add a panel-label letter to ``ax``.

    Parameters
    ----------
    ax
        matplotlib Axes to label.
    letter
        Label text (e.g., ``"A"``, ``"(B)"``).
    loc
        Anchor location: ``"top-left"`` (default), ``"top-right"``,
        ``"bottom-left"``, ``"bottom-right"``.
    offset
        Tuple ``(dx, dy)`` in axes-fraction coordinates relative to the
        nominal anchor.
    """
    anchor = {
        "top-left": (0.0, 1.0),
        "top-right": (1.0, 1.0),
        "bottom-left": (0.0, 0.0),
        "bottom-right": (1.0, 0.0),
    }.get(loc, (0.0, 1.0))
    ax.text(
        anchor[0] + offset[0],
        anchor[1] + offset[1],
        letter,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight=fontweight,
        va="bottom" if anchor[1] > 0.5 else "top",
        ha="left" if anchor[0] < 0.5 else "right",
        **kwargs,
    )


def _resolve_layout(
    layout: str | tuple[int, int] | Sequence[Sequence[int | None]],
    n_panels: int,
) -> tuple[int, int, list[tuple[int, int, int, int]]]:
    """Translate a layout spec into (n_rows, n_cols, slot_bboxes).

    Returns slot_bboxes as ``[(row_start, row_end, col_start, col_end), ...]``.
    """
    # Tuple form: (rows, cols) — fill row-major
    if isinstance(layout, tuple):
        rows, cols = layout
        slots = []
        for i in range(n_panels):
            r, c = divmod(i, cols)
            if r >= rows:
                break
            slots.append((r, r + 1, c, c + 1))
        return rows, cols, slots
    # String form: each character is a slot label, e.g. "ABC;DEF" → 2x3
    if isinstance(layout, str):
        rows_strs = layout.split(";")
        rows = len(rows_strs)
        cols = max(len(r) for r in rows_strs)
        # Find bounding box for each unique character
        labels: dict[str, list[tuple[int, int]]] = {}
        for r, row_str in enumerate(rows_strs):
            for c, ch in enumerate(row_str):
                if ch == " " or ch == ".":
                    continue
                labels.setdefault(ch, []).append((r, c))
        slots = []
        for ch in sorted(labels):
            cells = labels[ch]
            rs = min(c[0] for c in cells)
            re = max(c[0] for c in cells) + 1
            cs = min(c[1] for c in cells)
            ce = max(c[1] for c in cells) + 1
            slots.append((rs, re, cs, ce))
        return rows, cols, slots[:n_panels]
    # Sequence-of-sequences: explicit grid
    raise NotImplementedError(
        "Sequence-of-sequences layout not supported; use string or tuple form."
    )


def make_panel(
    panels: Sequence[dict],
    layout: str | tuple[int, int],
    *,
    figsize: tuple[float, float] = (10.0, 7.0),
    labels: bool = True,
    label_letters: Sequence[str] | None = None,
    label_kwargs: dict | None = None,
    wspace: float = 0.30,
    hspace: float = 0.40,
    suptitle: str | None = None,
) -> Figure:
    """Hand-crafted multi-panel figure.

    Parameters
    ----------
    panels
        List of dicts. Each must contain ``draw`` (callable accepting one
        ``matplotlib.Axes``).  Optional keys: ``title`` (str), ``style``
        (dict of fmt_axes overrides).
    layout
        ``(rows, cols)`` for row-major fill, or a string like
        ``"ABCD;EEFF"`` where each character is a slot label and identical
        characters span multiple cells.
    labels
        If True, auto-letter A/B/C/... in top-left of each panel.
    label_letters
        Override the default A/B/C letters (e.g., ``["i", "ii"]``).
    label_kwargs
        Extra kwargs forwarded to ``auto_letter``.
    """
    apply_theme()
    rows, cols, slots = _resolve_layout(layout, len(panels))
    if len(slots) < len(panels):
        raise ValueError(
            f"Layout has {len(slots)} slots but {len(panels)} panels were "
            f"provided; string layouts must have at least one unique slot "
            f"character per panel (or use a tuple layout for row-major fill)."
        )
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(rows, cols, figure=fig, wspace=wspace, hspace=hspace)

    if label_letters is None:
        label_letters = [chr(ord("A") + i) for i in range(len(panels))]
    label_kwargs = label_kwargs or {}

    for panel_idx, (panel, slot) in enumerate(zip(panels, slots)):
        rs, re, cs, ce = slot
        ax = fig.add_subplot(gs[rs:re, cs:ce])
        draw = panel["draw"]
        draw(ax)
        title = panel.get("title")
        if title:
            ax.set_title(title)
        style = panel.get("style", {})
        # Image-axes (heatmaps) must never have a grid drawn over the data —
        # the rcParams default is grid-on, which produces white lines across
        # imshow cells. fmt_axes with grid_major=False suppresses it.
        has_image = len(ax.get_images()) > 0
        if has_image and "grid_major" not in style:
            style = {**style, "grid_major": False, "grid_minor": False}
        fmt_axes(ax, **style)
        if has_image:
            ax.minorticks_off()
        if labels:
            auto_letter(ax, label_letters[panel_idx], **label_kwargs)
    if suptitle:
        fig.suptitle(suptitle, y=0.99)
    # Final sweep — strip top/right ticks on all axes (user pref). Heatmap
    # axes keep their data-frame spines; non-image axes lose top/right
    # spines as well.
    style_paper_axes(fig)
    return fig


def make_facet_grid(
    records: Iterable[dict],
    draw_fn: Callable[[plt.Axes, list[dict]], None],
    *,
    col: str,
    row: str | None = None,
    ncols: int = 4,
    sharex: bool = True,
    sharey: bool = True,
    figsize_per_panel: tuple[float, float] = (2.5, 2.0),
    label_panels: bool = False,
    title_fn: Callable[[Any], str] | None = None,
) -> Figure:
    """Build a same-plot-type-per-instance panel grid.

    Parameters
    ----------
    records
        Iterable of dicts (e.g., one dict per (CT, gene) row).
    draw_fn
        Called once per (col-value [, row-value]) group; receives ``(ax,
        subset_records)``.
    col
        Key in each record whose unique values become panel columns.
    row
        Optional key for panel rows.  If None, all panels are laid out in
        ``ncols`` columns wrapped as needed.
    ncols
        Used only when ``row`` is None.
    sharex / sharey
        Forwarded to ``plt.subplots``.
    figsize_per_panel
        ``(w, h)`` per panel.
    title_fn
        Maps the col-value (or (row, col)) to a panel title.
    """
    apply_theme()
    records_list = list(records)
    if row is None:
        col_values = sorted({r[col] for r in records_list})
        n = len(col_values)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
            sharex=sharex, sharey=sharey,
            squeeze=False,
        )
        for i, cv in enumerate(col_values):
            r, c = divmod(i, ncols)
            ax = axes[r, c]
            subset = [rec for rec in records_list if rec[col] == cv]
            draw_fn(ax, subset)
            if title_fn:
                ax.set_title(title_fn(cv))
            else:
                ax.set_title(str(cv))
            fmt_axes(ax)
            if label_panels:
                auto_letter(ax, chr(ord("A") + i))
        # Hide unused axes
        for j in range(n, nrows * ncols):
            r, c = divmod(j, ncols)
            axes[r, c].axis("off")
    else:
        row_values = sorted({rec[row] for rec in records_list})
        col_values = sorted({rec[col] for rec in records_list})
        nrows, ncols_eff = len(row_values), len(col_values)
        fig, axes = plt.subplots(
            nrows, ncols_eff,
            figsize=(figsize_per_panel[0] * ncols_eff, figsize_per_panel[1] * nrows),
            sharex=sharex, sharey=sharey,
            squeeze=False,
        )
        for i, rv in enumerate(row_values):
            for j, cv in enumerate(col_values):
                ax = axes[i, j]
                subset = [
                    rec for rec in records_list if rec[row] == rv and rec[col] == cv
                ]
                draw_fn(ax, subset)
                if title_fn:
                    ax.set_title(title_fn((rv, cv)))
                else:
                    ax.set_title(f"{rv} / {cv}")
                fmt_axes(ax)
    style_paper_axes(fig)
    return fig
