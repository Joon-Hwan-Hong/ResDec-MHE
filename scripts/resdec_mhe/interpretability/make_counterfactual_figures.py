"""Orchestrator: render counterfactual figures from CF v2 JSON outputs.

Loads either (default) both ``counterfactuals_v2_relative/`` and
``counterfactuals_v2_absolute/`` outputs and renders the 3 plots from
``src.visualization.counterfactual_plots`` per target mode:

  - ``fig_cf_movement_{relative,absolute}`` — per-subject fraction-of-target
  - ``fig_cf_ct_aggregate_{relative,absolute}`` — per-CT feature-count bar
  - ``fig_cf_top_pairs_{relative,absolute}`` — top-20 (CT, gene) pairs

Output dir: ``outputs/redesign/interpretability/figures/counterfactual/``.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

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


def _render_one(json_path: Path, out_dir: Path, label: str) -> list[str]:
    d = json.loads(json_path.read_text())
    results = d["results"]
    n_features_per_subject = d["n_features_per_subject"]
    n_ct = len(CELL_TYPE_ORDER)
    gene_names = list(np.load("data/precomputed/gene_names.npy", allow_pickle=True))
    n_gene = len(gene_names)

    # Movement plot
    sids = [r["subject_id"] for r in results]
    y_init = np.array([r["y_init"] for r in results])
    y_cf = np.array([r["y_cf"] for r in results])
    target = np.array([r["target_y"] for r in results])
    frac = np.abs(y_cf - y_init) / np.maximum(np.abs(target - y_init), 1e-9)
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
        "--cf-v2-relative",
        default="outputs/redesign/interpretability/counterfactuals_v2_relative/counterfactuals_fold0.json",
    )
    p.add_argument(
        "--cf-v2-absolute",
        default="outputs/redesign/interpretability/counterfactuals_v2_absolute/counterfactuals_fold0.json",
    )
    p.add_argument(
        "--out-dir",
        default="outputs/redesign/interpretability/figures/counterfactual",
    )
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    apply_theme()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rendered: list[str] = []
    for label, path in (("relative", Path(args.cf_v2_relative)),
                        ("absolute", Path(args.cf_v2_absolute))):
        if not path.exists():
            logger.warning("missing %s", path)
            continue
        all_rendered.extend(_render_one(path, out_dir, label))

    logger.info("rendered %d CF figures: %s", len(all_rendered), all_rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
