"""Orchestrator: render architecture figures from canonical artefacts.

Calls two functions from ``src.visualization.architecture_plots``:
  - architecture diagram (model schematic, no data dependency)
  - HGT cell-type interaction network (top-50 edges from CCC attention)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from src.visualization.architecture_plots import (
    plot_architecture_diagram,
    plot_hgt_celltype_network,
)
from src.visualization.theme import apply_theme

logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--ccc-edge-csv",
        default="outputs/canonical/interpretability/ccc/ccc_edge_attention.csv",
    )
    p.add_argument(
        "--out-dir",
        default="outputs/canonical/interpretability/figures/architecture",
    )
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    apply_theme()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered = []

    fig = plot_architecture_diagram(
        save_path=out_dir / "fig_architecture_diagram",
    )
    plt.close(fig)
    rendered.append("fig_architecture_diagram")

    edge_csv = Path(args.ccc_edge_csv)
    if edge_csv.exists():
        try:
            df = pd.read_csv(edge_csv)
            fig = plot_hgt_celltype_network(
                df, save_path=out_dir / "fig_hgt_celltype_network",
            )
            plt.close(fig)
            rendered.append("fig_hgt_celltype_network")
        except (ValueError, KeyError, ImportError) as exc:
            logger.warning("hgt network: %s", exc)
    logger.info("rendered %d architecture figures: %s", len(rendered), rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
