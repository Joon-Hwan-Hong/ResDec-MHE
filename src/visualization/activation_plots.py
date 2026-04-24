"""Activation-cascade plots: per-subject activation distributions across forward-pass stages.

Functions:

  - ``plot_per_stage_activation_cascade`` — violin plot of per-subject
    activation L2 norms at each stage of the network forward pass. Medians
    are connected across stages to visualize how the signal magnitude
    evolves from input through to the final prediction.

The module takes pre-computed per-stage per-subject activation statistics
(a plain dict); orchestrators handle the model loading + forward-hook
registration + norm computation, keeping this module torch-free.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np

from src.visualization.theme import PALETTES, fmt_axes, save_fig


def plot_per_stage_activation_cascade(
    stage_norms: Mapping[str, np.ndarray],
    *,
    figsize: tuple[float, float] | None = None,
    log_y: bool = True,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Violin cascade of per-subject activation norms across forward-pass stages.

    Each stage's violin shows the distribution of per-subject L2 norms of
    that stage's output tensor (computed by the caller). Stages appear on
    the x-axis in insertion order — the caller is responsible for passing
    the dict in forward-pass order. Medians are connected across adjacent
    stages with a solid line to make the signal-magnitude trajectory easy
    to read.

    Y-axis is log-scaled by default since encoder stages typically span
    multiple orders of magnitude.

    Parameters
    ----------
    stage_norms
        Mapping ``stage_name`` → 1-D numpy array of per-subject norms
        (length ``n_subjects``; can vary across stages but typically
        identical). Must contain ≥2 stages.
    log_y
        Use log10 y-axis. Set ``False`` for linear y-axis.
    figsize
        Default ``(max(6.0, 0.8 * n_stages), 4.0)``.
    save_path
        If not ``None``, save figure stem to ``<save_path>.png`` and
        ``<save_path>.pdf``.

    Raises
    ------
    ValueError
        If fewer than 2 stages, any stage is empty, or a stage contains
        no finite values.
    """
    if len(stage_norms) < 2:
        raise ValueError(f"need ≥2 stages for cascade; got {len(stage_norms)}")
    stage_names = list(stage_norms.keys())
    arrays = [np.asarray(stage_norms[name], dtype=np.float64) for name in stage_names]
    for name, arr in zip(stage_names, arrays):
        if arr.size == 0:
            raise ValueError(f"stage '{name}' is empty")
        if not np.isfinite(arr).any():
            raise ValueError(f"stage '{name}' has no finite values")

    n_stages = len(stage_names)
    if figsize is None:
        figsize = (max(6.0, 0.8 * n_stages), 4.0)
    fig, ax = plt.subplots(figsize=figsize)

    positions = np.arange(1, n_stages + 1)
    violin_data = [arr[np.isfinite(arr)] for arr in arrays]
    parts = ax.violinplot(
        violin_data, positions=positions,
        showmeans=False, showmedians=True, showextrema=True,
        widths=0.75,
    )
    cmap = plt.get_cmap(PALETTES["sequential"])
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(cmap(0.25 + 0.65 * i / max(1, n_stages - 1)))
        body.set_edgecolor("black")
        body.set_alpha(0.85)
    for k in ("cmedians", "cmaxes", "cmins", "cbars"):
        if k in parts:
            parts[k].set_color("black")
            parts[k].set_linewidth(0.6)

    medians = np.array([float(np.median(arr)) for arr in violin_data])
    ax.plot(
        positions, medians,
        marker="o", markersize=5, linestyle="-", linewidth=1.2,
        color="black", zorder=4, label="median across subjects",
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(stage_names, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("Per-subject activation L2 norm" + (" (log₁₀)" if log_y else ""))
    if log_y:
        ax.set_yscale("log")
    fmt_axes(ax)
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    if save_path is not None:
        save_fig(fig, save_path)
    return fig
