"""Weight-space plots: geometry of model parameters across checkpoints.

Functions:

  - ``plot_checkpoint_weight_pca`` — 2D PCA projection of flattened
    checkpoint weights across folds / seeds. Annotates each point with
    its fold label and (optionally) its validation R².

The module operates on pre-flattened numpy matrices so it stays free of
``torch``. Orchestrators handle the state-dict → numpy conversion.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from src.visualization.theme import PALETTES, fmt_axes, save_fig


def plot_checkpoint_weight_pca(
    weight_matrix: np.ndarray,
    fold_labels: Sequence[str] | None = None,
    r2_per_checkpoint: Sequence[float] | None = None,
    *,
    figsize: tuple[float, float] = (5.5, 4.5),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """2D PCA projection of flattened checkpoint weights.

    Each row of ``weight_matrix`` is one checkpoint's flattened parameter
    vector (e.g. fold 0 through fold 4 final weights). The function
    centers the matrix, computes the first two principal components via
    SVD, and plots the 2D projection. Points are annotated with
    ``fold_labels`` and (if supplied) their validation R².

    Tightly clustered projections indicate consistent optimisation across
    folds; widely spread projections indicate multiple distinct minima —
    useful as a sanity check for the canonical ensemble's stability.

    Parameters
    ----------
    weight_matrix
        Shape ``(n_checkpoints, n_params)``. Rows are typically very long
        (millions of params); SVD on 5 × n_params is still cheap because
        the rank is bounded by the number of checkpoints.
    fold_labels
        Length ``n_checkpoints`` labels for each point. Defaults to
        ``["fold 0", ..., "fold N-1"]``.
    r2_per_checkpoint
        Optional per-checkpoint validation R² annotation.
    figsize, save_path
        Standard figure kwargs.
    """
    n_ckpt, n_params = weight_matrix.shape
    if n_ckpt < 2:
        raise ValueError(f"need ≥2 checkpoints for PCA; got {n_ckpt}")
    if fold_labels is None:
        fold_labels = [f"fold {i}" for i in range(n_ckpt)]
    if len(fold_labels) != n_ckpt:
        raise ValueError(
            f"fold_labels length {len(fold_labels)} != n_ckpt={n_ckpt}")
    if r2_per_checkpoint is not None and len(r2_per_checkpoint) != n_ckpt:
        raise ValueError(
            f"r2_per_checkpoint length {len(r2_per_checkpoint)} != n_ckpt={n_ckpt}")

    centered = weight_matrix - weight_matrix.mean(axis=0, keepdims=True)
    # SVD on (n_ckpt × n_params). n_ckpt is tiny (5), so this is cheap
    # even with millions of params.
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    proj = u[:, :2] * s[:2]
    explained = (s ** 2) / (s ** 2).sum() if s.sum() > 0 else np.zeros_like(s)

    fig, ax = plt.subplots(figsize=figsize)
    fold_colors = PALETTES["fold_colors"]
    for i, (x, y) in enumerate(proj):
        color = fold_colors[i % len(fold_colors)]
        ax.scatter(x, y, s=140, color=color, edgecolor="black",
                   linewidth=0.8, zorder=3, label=fold_labels[i])
        label_text = str(fold_labels[i])
        if r2_per_checkpoint is not None:
            label_text += f"\nR²={r2_per_checkpoint[i]:+.3f}"
        ax.annotate(
            label_text, (x, y), xytext=(6, 6),
            textcoords="offset points", fontsize=7, zorder=4,
        )
    ax.axhline(0, color="gray", linewidth=0.3, linestyle=":", zorder=1)
    ax.axvline(0, color="gray", linewidth=0.3, linestyle=":", zorder=1)
    ax.set_xlabel(f"PC 1 ({explained[0] * 100:.1f}% var)")
    ax.set_ylabel(f"PC 2 ({explained[1] * 100:.1f}% var)")
    fmt_axes(ax)
    ax.set_title("Checkpoint weight-space PCA", fontsize=9)
    fig.tight_layout()
    if save_path is not None:
        save_fig(fig, save_path)
    return fig
