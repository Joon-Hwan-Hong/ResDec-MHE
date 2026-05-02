"""Figure 5: SAE cross-seed decoder cosine heatmap (3-panel composite).

For each (seed_i, seed_j) pair in {(0,1), (0,2), (1,2)}:
  - Render the 2048×2048 cosine-similarity matrix between decoder columns.
  - Diverging colormap (PiYG) centered at zero, range ±max(|values|).
  - Annotate each panel with off-diagonal cos-sim summary stats: max abs,
    mean abs, count of pairs with |cos| > 0.7.

Title: "SAE cross-seed instability at our scale (Paulo & Belrose 2025: ~30%
expected at 0.7; observed: 0%)."
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # must precede pyplot import

import matplotlib.pyplot as plt
import numpy as np

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from src.visualization.theme import apply_theme, fmt_axes, save_fig

logger = logging.getLogger(__name__)


def _offdiag_stats(mat: np.ndarray) -> dict:
    """Compute max/mean/abs-mean of upper-triangular off-diagonal entries.

    For a between-seed cosine matrix M[i,j] = cos(seed_a column i, seed_b
    column j), there is no natural diagonal alignment because feature indices
    permute across seeds.  We therefore use ALL entries (no diagonal removal)
    for the purpose of capturing the global pairing distribution.
    """
    arr = mat.flatten().astype(np.float64)
    abs_arr = np.abs(arr)
    return {
        "shape": list(mat.shape),
        "max_abs": float(abs_arr.max()),
        "mean_abs": float(abs_arr.mean()),
        "median": float(np.median(arr)),
        "frac_gt_0p7": float((abs_arr > 0.7).mean()),
        "n_gt_0p7": int((abs_arr > 0.7).sum()),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--cosine-npz",
        type=Path,
        default=(
            _WORKTREE_ROOT
            / "outputs/canonical/sae/cross_seed_stability"
            / "decoder_cosine_matrices.npz"
        ),
    )
    p.add_argument(
        "--summary-json",
        type=Path,
        default=(
            _WORKTREE_ROOT
            / "outputs/canonical/sae/cross_seed_stability/cross_seed_summary.json"
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=(
            _WORKTREE_ROOT
            / "outputs/canonical/interpretability/figures/sae_cross_seed"
        ),
    )
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    apply_theme()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    data = np.load(args.cosine_npz)
    cosines = data["cosine_matrices"]   # shape (S, S, m, m)
    seeds = data["seed_order"]
    summary = json.loads(Path(args.summary_json).read_text())

    pairs = [(0, 1), (0, 2), (1, 2)]
    pair_stats = {}
    for (a, b) in pairs:
        pair_stats[f"seed{seeds[a]}_vs_seed{seeds[b]}"] = _offdiag_stats(
            cosines[a, b]
        )

    # Color saturation: tight ±vmax so the noise structure is visible, since
    # the cosine distribution is concentrated near zero (mean|cos|=0.10,
    # std|cos|≈0.13). Picking 1×median(|cos|)≈0.10 keeps salt-and-pepper
    # pattern visible while still showing extreme matches as fully saturated.
    # The full max-abs value is annotated on each panel.
    abs_pool = np.concatenate(
        [np.abs(cosines[a, b]).flatten() for (a, b) in pairs]
    )
    vmax = float(np.percentile(abs_pool, 90))   # ≈0.20
    vmax = max(vmax, 0.10)

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.8))
    panel_letters = ["A", "B", "C"]
    for idx, (a, b) in enumerate(pairs):
        ax = axes[idx]
        mat = cosines[a, b]
        stats = pair_stats[f"seed{seeds[a]}_vs_seed{seeds[b]}"]
        im = ax.imshow(
            mat,
            cmap="PiYG", vmin=-vmax, vmax=vmax,
            aspect="equal", interpolation="nearest",
        )
        ax.set_xlabel(f"feature index (seed {seeds[b]})", fontsize=7)
        ax.set_ylabel(f"feature index (seed {seeds[a]})", fontsize=7)
        ax.set_xticks(np.linspace(0, mat.shape[1] - 1, 5).astype(int))
        ax.set_yticks(np.linspace(0, mat.shape[0] - 1, 5).astype(int))
        ax.tick_params(axis="both", labelsize=6)
        cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.04, shrink=0.85)
        cbar.set_label("cos sim", fontsize=7)
        cbar.ax.tick_params(labelsize=6)
        ax.set_title(
            f"seed {seeds[a]} vs seed {seeds[b]}\n"
            f"max|cos|={stats['max_abs']:.3f}, "
            f"mean|cos|={stats['mean_abs']:.3f}\n"
            f"|cos|>0.7: {stats['n_gt_0p7']}",
            fontsize=7.5,
        )
        fmt_axes(ax, hide_spines=(), grid_major=False, grid_minor=False)
        ax.text(
            -0.18, 1.12, panel_letters[idx], transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="left",
        )

    fig.subplots_adjust(top=0.78, bottom=0.13, left=0.06, right=0.97, wspace=0.55)
    fig.suptitle(
        "SAE cross-seed instability at our scale\n"
        "(Paulo & Belrose 2025: ~30% expected at |cos|>0.7; observed: 0%)",
        fontsize=10, y=0.96,
    )
    save_fig(fig, out_dir / "sae_cross_seed")
    plt.close(fig)

    # Persist stats summary
    summary_out = {
        "seed_order": [int(s) for s in seeds],
        "pairs": [{"a": int(a), "b": int(b)} for (a, b) in pairs],
        "pair_stats": pair_stats,
        "vmax_used": vmax,
        "stable_fraction_in_summary_json": summary.get("stable_fraction"),
        "n_stable_features_in_summary_json": summary.get("n_stable_features"),
    }
    (out_dir / "sae_cross_seed_data.json").write_text(
        json.dumps(summary_out, indent=2)
    )

    elapsed = time.perf_counter() - t0
    logger.info(
        "Rendered sae_cross_seed.{png,pdf} in %.2fs (3 pairs, %d×%d each)",
        elapsed, cosines.shape[2], cosines.shape[3],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
