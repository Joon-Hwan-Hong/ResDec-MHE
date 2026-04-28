"""Learning-curve figure: ResDec-MHE R² as a function of training-set size N.

Renders one figure with per-seed lines + cross-seed mean + shaded across-seed
std band per N, plus the canonical N=516 anchor as a horizontal reference.

Public API: ``plot_learning_curve_n_vs_r2(results, canonical_r2, save_path)``.

``results`` is the list-of-dicts schema written by
``scripts/resdec_mhe/training/run_learning_curve.py``: each entry has
``N``, ``rng_seed``, ``per_fold_r2`` (list of 5 floats), ``mean_r2``, ``std_r2``.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curve_n_vs_r2(
    results: Sequence[dict],
    canonical_r2: float | None = None,
    save_path: Path | str | None = None,
    *,
    figsize: tuple[float, float] = (6.4, 4.0),
) -> plt.Figure:
    """Plot per-seed N→R² curves + mean ± std band + canonical anchor.

    Parameters
    ----------
    results
        List of K=5 entries with keys ``N``, ``rng_seed``, ``mean_r2``,
        ``std_r2``.
    canonical_r2
        Optional reference R² for the full-cohort N=516 anchor.
    save_path
        If provided, save as PNG and PDF (suffix appended).
    figsize
        Matplotlib figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The rendered figure.
    """
    if not results:
        raise ValueError("results is empty")

    by_seed: dict[int, dict[int, float]] = defaultdict(dict)
    for r in results:
        if "mean_r2" not in r:
            continue
        by_seed[int(r["rng_seed"])][int(r["N"])] = float(r["mean_r2"])

    fig, ax = plt.subplots(figsize=figsize)
    Ns_all = sorted({int(r["N"]) for r in results if "mean_r2" in r})

    # Per-seed thin lines
    for seed, n_to_r in by_seed.items():
        xs = sorted(n_to_r.keys())
        ys = [n_to_r[n] for n in xs]
        ax.plot(xs, ys, lw=0.8, alpha=0.5, label=f"seed={seed}")

    # Mean and across-seed std per N
    means: list[float] = []
    stds: list[float] = []
    Ns_for_band = [n for n in Ns_all if sum(n in s for s in by_seed.values()) >= 2]
    for n in Ns_for_band:
        vals = np.array([s[n] for s in by_seed.values() if n in s])
        means.append(vals.mean())
        stds.append(vals.std(ddof=1))
    means_arr = np.array(means)
    stds_arr = np.array(stds)
    ax.plot(Ns_for_band, means_arr, lw=2.0, color="black", label="mean")
    ax.fill_between(
        Ns_for_band, means_arr - stds_arr, means_arr + stds_arr,
        alpha=0.2, color="gray",
    )

    if canonical_r2 is not None:
        ax.axhline(
            canonical_r2, ls="--", color="C3",
            label=f"canonical N=516: R²={canonical_r2:+.3f}",
        )

    ax.set_xlabel("Training-set size N")
    ax.set_ylabel("Mean 5-fold R² (across folds)")
    ax.set_title("Learning curve: ResDec-MHE R² vs N (K=5 seeds)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    if save_path is not None:
        sp = Path(save_path)
        sp.parent.mkdir(parents=True, exist_ok=True)
        for ext in (".png", ".pdf"):
            fig.savefig(sp.with_suffix(ext), dpi=300, bbox_inches="tight")
    return fig
