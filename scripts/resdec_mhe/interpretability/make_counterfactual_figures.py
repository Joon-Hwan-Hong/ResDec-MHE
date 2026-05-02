"""Orchestrator: render counterfactual figures from CF JSON outputs.

Loads (default) both ``counterfactuals_relative/`` and
``counterfactuals_absolute/`` outputs and renders the 3 plots from
``src.visualization.counterfactual_plots`` per target mode:

  - ``fig_cf_movement_{relative,absolute}`` — per-subject fraction-of-target
  - ``fig_cf_ct_aggregate_{relative,absolute}`` — per-CT feature-count bar
  - ``fig_cf_top_pairs_{relative,absolute}`` — top-20 (CT, gene) pairs

Output dir: ``outputs/canonical/interpretability/figures/counterfactual/``.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # must precede pyplot import

import matplotlib.pyplot as plt
import numpy as np

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from src.data.constants import CELL_TYPE_ORDER
from src.visualization.counterfactual_plots import (
    plot_counterfactual_ct_aggregate,
    plot_counterfactual_movement,
    plot_counterfactual_top_pairs,
)
from src.visualization.theme import apply_theme

logger = logging.getLogger(__name__)


# Floor for the |target - y_init| denominator in the fraction-of-target
# movement metric. Targets in {relative, absolute} modes can be arbitrarily
# close to y_init (e.g., a subject already at the cohort centile target);
# this floor avoids division-by-zero artefacts inflating ``frac`` to ±inf.
# 1e-9 is well below the smallest meaningful cognition-scale increment.
_EPS_FRAC_OF_TARGET: float = 1e-9


def _render_one(
    json_path: Path,
    out_dir: Path,
    label: str,
    *,
    gene_names_npy: Path,
) -> list[str]:
    """Render the 3-panel counterfactual figures for one target mode.

    ``gene_names_npy`` is required: it routes the ``data/precomputed/gene_names.npy``
    path through argparse instead of hardcoding it inside the function body.
    """
    d = json.loads(json_path.read_text())
    results = d["results"]
    n_features_per_subject = d["n_features_per_subject"]
    n_ct = len(CELL_TYPE_ORDER)
    gene_names = list(np.load(gene_names_npy, allow_pickle=True))
    n_gene = len(gene_names)

    # Movement plot
    sids = [r["subject_id"] for r in results]
    y_init = np.array([r["y_init"] for r in results])
    y_cf = np.array([r["y_cf"] for r in results])
    target = np.array([r["target_y"] for r in results])
    frac = np.abs(y_cf - y_init) / np.maximum(
        np.abs(target - y_init), _EPS_FRAC_OF_TARGET,
    )
    regime = [r["regime"] for r in results]
    rendered: list[str] = []
    try:
        fig = plot_counterfactual_movement(
            sids, frac, regime,
            save_path=out_dir / f"fig_cf_movement_{label}",
        )
        plt.close(fig)
        rendered.append(f"fig_cf_movement_{label}")
    except ValueError as exc:
        logger.warning("cf_movement (%s): %s", label, exc)

    # CT aggregate
    ct_counter: Counter = Counter()
    pair_counter: Counter = Counter()
    for r in results:
        for f in r["top_k_features"]:
            idx = f["feature_idx"]
            ct = idx // n_gene
            gene = idx % n_gene
            if 0 <= ct < n_ct and 0 <= gene < n_gene:
                ct_name = CELL_TYPE_ORDER[ct]
                gene_name = gene_names[gene]
                ct_counter[ct_name] += 1
                pair_counter[(ct_name, gene_name)] += 1
    total = sum(ct_counter.values())
    try:
        fig = plot_counterfactual_ct_aggregate(
            ct_counter, total=total,
            save_path=out_dir / f"fig_cf_ct_aggregate_{label}",
        )
        plt.close(fig)
        rendered.append(f"fig_cf_ct_aggregate_{label}")
    except ValueError as exc:
        logger.warning("cf_ct_aggregate (%s): %s", label, exc)

    # Top pairs
    try:
        fig = plot_counterfactual_top_pairs(
            pair_counter, top_n=20,
            save_path=out_dir / f"fig_cf_top_pairs_{label}",
        )
        plt.close(fig)
        rendered.append(f"fig_cf_top_pairs_{label}")
    except ValueError as exc:
        logger.warning("cf_top_pairs (%s): %s", label, exc)

    return rendered


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--relative-json",
        type=Path,
        default=(
            _WORKTREE_ROOT
            / "outputs/canonical/interpretability/counterfactuals_relative"
            / "counterfactuals_fold0.json"
        ),
        help="Path to the relative-mode CF JSON.",
    )
    p.add_argument(
        "--absolute-json",
        type=Path,
        default=(
            _WORKTREE_ROOT
            / "outputs/canonical/interpretability/counterfactuals_absolute"
            / "counterfactuals_fold0.json"
        ),
        help="Path to the absolute-mode CF JSON.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=(
            _WORKTREE_ROOT
            / "outputs/canonical/interpretability/figures/counterfactual"
        ),
        help="Output directory for the rendered figures.",
    )
    p.add_argument(
        "--gene-names-npy",
        type=Path,
        default=_WORKTREE_ROOT / "data/precomputed/gene_names.npy",
        help=(
            "Path to gene_names.npy (148_607-element object array). "
            "Routed here from argparse to avoid hardcoding under cwd-fragile "
            "relative paths."
        ),
    )
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    apply_theme()

    rel_path = Path(args.relative_json)
    abs_path = Path(args.absolute_json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rendered: list[str] = []
    for label, path in (("relative", rel_path), ("absolute", abs_path)):
        if not path.exists():
            logger.warning("missing %s", path)
            continue
        all_rendered.extend(
            _render_one(path, out_dir, label, gene_names_npy=args.gene_names_npy)
        )

    logger.info("rendered %d CF figures: %s", len(all_rendered), all_rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
