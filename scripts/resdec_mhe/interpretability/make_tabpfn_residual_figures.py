"""Render TabPFN-residual decomposition figures (4 candidates) for §4d.

Loads canonical 5-fold predictions:
  - outputs/canonical/p5_canonical_seed42/fold{0..4}/val_predictions_best.npz
    (provides predictions = f̂_residual, targets = y_true, subject_ids)
  - data/canonical/tabpfn_outer_fold{0..4}.npz
    (provides y_tabpfn = outer-fold TabPFN baseline)
  - outputs/canonical/interpretability/variance_decomposition.json (var components)
  - outputs/canonical/interpretability/residual_per_subject.csv (pathology covariates)

Renders 4 candidate figures to outputs/canonical/interpretability/figures/tabpfn_residual/:
  - fig_additive_3panel.{png,pdf}        — y vs TabPFN + y vs composite + residual hist
  - fig_variance_partition_bar.{png,pdf} — stacked-bar variance decomposition
  - fig_per_subject_delta_scatter.{png,pdf} — residual vs TabPFN colored by pathology
  - fig_residual_histogram_overlay.{png,pdf} — TabPFN-only vs composite error histogram

User picks visually after rendering.  No new GPU compute required.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from src.visualization.tabpfn_residual_plots import (
    plot_additive_3panel,
    plot_per_subject_delta_scatter,
    plot_residual_histogram_overlay,
    plot_variance_partition_bar,
)

logger = logging.getLogger(__name__)


def _load_composite_per_subject(
    pred_root: Path, tabpfn_root: Path, n_folds: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate y_true, y_tabpfn, y_residual across all val folds.

    Returns (y_true, y_tabpfn, y_residual, subject_ids) all length N=516.
    Aligns by val_subject_ids per fold.
    """
    y_true: list[np.ndarray] = []
    y_tabpfn: list[np.ndarray] = []
    y_residual: list[np.ndarray] = []
    subj: list[np.ndarray] = []
    for fold in range(n_folds):
        v = np.load(pred_root / f"fold{fold}/val_predictions_best.npz", allow_pickle=True)
        t = np.load(tabpfn_root / f"tabpfn_outer_fold{fold}.npz", allow_pickle=True)
        sids_v = list(v["subject_ids"])
        sids_t = list(t["val_subject_ids"])
        # Align: index of each subject in tabpfn outer
        idx_t = [sids_t.index(s) for s in sids_v]
        y_true.append(np.asarray(v["targets"], dtype=np.float64))
        y_tabpfn.append(np.asarray(t["y_tabpfn"], dtype=np.float64)[idx_t])
        y_residual.append(np.asarray(v["predictions"], dtype=np.float64))
        subj.append(np.asarray(sids_v, dtype=object))
    return (
        np.concatenate(y_true),
        np.concatenate(y_tabpfn),
        np.concatenate(y_residual),
        np.concatenate(subj),
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--pred-root",
        default="outputs/canonical/p5_canonical_seed42",
    )
    p.add_argument("--tabpfn-root", default="data/canonical")
    p.add_argument(
        "--variance-decomposition-json",
        default="outputs/canonical/interpretability/variance_decomposition.json",
    )
    p.add_argument(
        "--residual-csv",
        default="outputs/canonical/interpretability/residual_per_subject.csv",
        help="Provides per-subject pathology covariates (gpath, etc.).",
    )
    p.add_argument(
        "--out-dir",
        default="outputs/canonical/interpretability/figures/tabpfn_residual",
    )
    p.add_argument("--n-folds", type=int, default=5)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load composite predictions ───────────────────────────────────────────
    y_true, y_tabpfn, y_residual, subj_ids = _load_composite_per_subject(
        Path(args.pred_root), Path(args.tabpfn_root), n_folds=args.n_folds,
    )
    logger.info(
        "loaded n=%d subjects across %d folds; y_true mean=%.3f std=%.3f",
        len(y_true), args.n_folds, float(y_true.mean()), float(y_true.std()),
    )

    # ── Load pathology covariate ────────────────────────────────────────────
    res_df = pd.read_csv(args.residual_csv)
    id_col = "ROSMAP_IndividualID" if "ROSMAP_IndividualID" in res_df.columns else res_df.columns[0]
    res_df = res_df.rename(columns={id_col: "subject_id"})
    # Try common pathology column names
    path_col = None
    for c in ("gpath", "gpathsqrt", "amyloid", "tangles"):
        if c in res_df.columns:
            path_col = c
            break
    if path_col is None:
        logger.warning(
            "no pathology column found in %s; will fall back to ŷ_TabPFN as proxy color",
            args.residual_csv,
        )
        pathology = y_tabpfn
        path_label = "ŷ_TabPFN (pathology proxy)"
    else:
        m = res_df.set_index("subject_id")[path_col].to_dict()
        pathology = np.array([m.get(str(s), np.nan) for s in subj_ids], dtype=np.float64)
        path_label = f"Global pathology ({path_col})"

    # ── Render the 4 figures ─────────────────────────────────────────────────
    p1 = plot_additive_3panel(y_true, y_tabpfn, y_residual, out_dir / "fig_additive_3panel")
    logger.info("wrote %s", p1)

    with open(args.variance_decomposition_json) as f:
        vd = json.load(f)
    var_comp = vd["global"]
    p2 = plot_variance_partition_bar(var_comp, out_dir / "fig_variance_partition_bar")
    logger.info("wrote %s", p2)

    p3 = plot_per_subject_delta_scatter(
        y_tabpfn, y_residual, pathology,
        out_dir / "fig_per_subject_delta_scatter",
        pathology_label=path_label,
    )
    logger.info("wrote %s", p3)

    p4 = plot_residual_histogram_overlay(
        y_true, y_tabpfn, y_residual,
        out_dir / "fig_residual_histogram_overlay",
    )
    logger.info("wrote %s", p4)

    print("\nFigures written:")
    for path in (p1, p2, p3, p4):
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
