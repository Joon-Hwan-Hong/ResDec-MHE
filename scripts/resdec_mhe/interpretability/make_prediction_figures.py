"""Orchestrator: render prediction figures from canonical artefacts.

Calls one function appended to ``src.visualization.prediction_plots``:
  - calibration overlay: TabPFN-only vs Composite reliability diagrams.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # must precede pyplot import

import matplotlib.pyplot as plt
import numpy as np

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from src.visualization.prediction_plots import plot_calibration_overlay
from src.visualization.theme import apply_theme

logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--canonical-dir", type=Path,
        default=_WORKTREE_ROOT / "outputs/canonical/p5_canonical_seed42",
    )
    p.add_argument(
        "--tabpfn-dir", type=Path,
        default=_WORKTREE_ROOT / "data/canonical",
    )
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument(
        "--out-dir", type=Path,
        default=(
            _WORKTREE_ROOT
            / "outputs/canonical/interpretability/figures/prediction"
        ),
    )
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    apply_theme()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    composite_per_fold = []
    for f in range(args.n_folds):
        p_path = Path(args.canonical_dir) / f"fold{f}/val_predictions_best.npz"
        if not p_path.exists():
            logger.warning("missing %s", p_path)
            continue
        d = np.load(p_path, allow_pickle=True)
        composite_per_fold.append(
            (np.asarray(d["targets"], dtype=np.float64),
             np.asarray(d["predictions"], dtype=np.float64)),
        )
    tabpfn_per_fold = []
    for f in range(args.n_folds):
        p_path = Path(args.tabpfn_dir) / f"tabpfn_outer_fold{f}.npz"
        if not p_path.exists():
            continue
        d = np.load(p_path, allow_pickle=True)
        tabpfn_per_fold.append(
            (np.asarray(d["y_true"], dtype=np.float64),
             np.asarray(d["y_tabpfn"], dtype=np.float64)),
        )
    if not tabpfn_per_fold or not composite_per_fold:
        logger.error("missing per-fold predictions; aborting")
        return 1

    fig = plot_calibration_overlay(
        tabpfn_per_fold, composite_per_fold,
        save_path=out_dir / "fig_calibration_overlay",
    )
    plt.close(fig)
    logger.info("rendered fig_calibration_overlay")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
