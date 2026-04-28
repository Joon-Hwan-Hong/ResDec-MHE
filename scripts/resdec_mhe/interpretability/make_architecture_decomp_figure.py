"""Figure 2: Architecture-cost decomposition.

Box-and-strip plot per condition, with paired Wilcoxon p-value annotations.

Conditions (left → right):
  - canonical SDPA: per_fold from `canonical`
  - sweep λ=0 einsum: per_fold from `sweep_per_lambda["0.0"]`
  - diff-test SDPA + no_grad-recompute: per_fold from `diff_test`
  - sweep λ=1.0: per_fold from `sweep_per_lambda["1.0"]`

Annotations:
  - Architecture cost: canonical − sweep λ=0 (Wilcoxon p_one_sided=0.03125 by data).
  - Diff-test cost: canonical − diff-test (p_one_sided=0.03125).
  - Regularization cost (max λ): sweep λ=0 − sweep λ=1.0 (p_one_sided=0.21875).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from src.visualization.theme import (
    PALETTES,
    apply_theme,
    fmt_axes,
    save_fig,
)

logger = logging.getLogger(__name__)


def _draw_box_strip(
    ax,
    per_condition: list[list[float]],
    labels: list[str],
    colors: list,
) -> None:
    """Draw boxplots + per-fold strip points overlay."""
    positions = np.arange(len(per_condition))

    # Boxplots
    bp = ax.boxplot(
        per_condition,
        positions=positions,
        widths=0.5,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.0},
        boxprops={"linewidth": 0.8},
        whiskerprops={"linewidth": 0.8},
        capprops={"linewidth": 0.8},
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)

    # Strip points (5 folds per condition); deterministic offset so the same
    # fold appears at the same horizontal jitter across conditions.
    rng = np.random.default_rng(0)
    n_folds = len(per_condition[0])
    fold_palette = list(PALETTES["fold_colors"])
    for j, vals in enumerate(per_condition):
        x_jitter = rng.uniform(-0.10, 0.10, size=n_folds)
        for k, (xj, v) in enumerate(zip(x_jitter, vals)):
            ax.scatter(
                positions[j] + xj, v,
                color=fold_palette[k % len(fold_palette)],
                s=24, zorder=3,
                edgecolors="white", linewidths=0.6,
                label=f"fold {k}" if j == 0 else None,
            )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("R²")
    ax.set_xlim(-0.6, len(per_condition) - 0.4)


def _annot_pair(
    ax,
    pos_a: int, pos_b: int,
    y_top: float, label: str, *,
    delta_h: float = 0.02,
    color: str = "#444444",
) -> None:
    """Draw a paired-comparison bracket with a label above it."""
    h = delta_h
    ax.plot(
        [pos_a, pos_a, pos_b, pos_b],
        [y_top, y_top + h, y_top + h, y_top],
        color=color, linewidth=0.8,
    )
    ax.text(
        (pos_a + pos_b) / 2, y_top + h * 1.1,
        label,
        ha="center", va="bottom",
        fontsize=7, color=color,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--decomp-json",
        default="outputs/redesign/interpretability/architecture_vs_regularization_decomposition.json",
    )
    p.add_argument(
        "--out-dir",
        default="outputs/redesign/interpretability/figures/architecture_decomp",
    )
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    apply_theme()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    decomp = json.loads(Path(args.decomp_json).read_text())

    canonical = decomp["canonical"]["per_fold"]
    lam0 = decomp["sweep_per_lambda"]["0.0"]["per_fold"]
    diff_test = decomp["diff_test"]["per_fold"]
    lam1 = decomp["sweep_per_lambda"]["1.0"]["per_fold"]

    per_condition = [canonical, lam0, diff_test, lam1]
    labels = [
        "canonical\nSDPA",
        "sweep λ=0\neinsum",
        "diff-test\nSDPA+no-grad",
        "sweep λ=1.0\neinsum+reg",
    ]
    palette = list(PALETTES["categorical"])
    cond_colors = [palette[0], palette[1], palette[2], palette[3]]

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    _draw_box_strip(ax, per_condition, labels, cond_colors)
    fmt_axes(ax)
    ax.legend(
        loc="lower left", fontsize=7, ncol=5,
        bbox_to_anchor=(0.0, -0.42), borderaxespad=0.0,
    )

    # Pull Wilcoxon p-values from JSON and annotate paired comparisons
    arch_p = decomp["decomposition"]["architecture_cost"]["wilcoxon_p_one_sided"]
    arch_mean = decomp["decomposition"]["architecture_cost"]["mean"]
    diff_p = decomp["decomposition"]["diff_test_cost"]["wilcoxon_p_one_sided"]
    diff_mean = decomp["decomposition"]["diff_test_cost"]["mean"]
    reg_p = decomp["decomposition"]["regularization_cost_max_lambda"]["wilcoxon_p_one_sided"]
    reg_mean = decomp["decomposition"]["regularization_cost_max_lambda"]["mean"]

    y_max = max(max(c) for c in per_condition)
    y_min = min(min(c) for c in per_condition)
    span = y_max - y_min
    base = y_max + span * 0.07

    _annot_pair(
        ax, 0, 1, base,
        f"Δ={arch_mean:+.3f}, p={arch_p:.4g}",
        delta_h=span * 0.02,
        color=palette[3],
    )
    _annot_pair(
        ax, 0, 2, base + span * 0.13,
        f"Δ={diff_mean:+.3f}, p={diff_p:.4g}",
        delta_h=span * 0.02,
        color=palette[4],
    )
    _annot_pair(
        ax, 1, 3, base + span * 0.26,
        f"Δ={reg_mean:+.3f}, p={reg_p:.4g}",
        delta_h=span * 0.02,
        color=palette[5],
    )

    ax.set_ylim(y_min - span * 0.05, base + span * 0.42)
    ax.set_title(
        "Architecture cost decomposition: 0.06 R² einsum penalty + non-significant regularization effect",
        fontsize=8.5, pad=10,
    )

    save_fig(fig, out_dir / "architecture_decomp")
    plt.close(fig)
    elapsed = time.perf_counter() - t0
    logger.info(
        "Rendered architecture_decomp.{png,pdf} in %.2fs (4 conditions × 5 folds)",
        elapsed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
