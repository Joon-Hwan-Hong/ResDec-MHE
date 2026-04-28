"""Orchestrator: render distributional-analysis figures from canonical artifacts.

Loads:
  - ``wasserstein_per_celltype_pseudobulk.json`` (pseudobulk Wasserstein-1)
  - ``stability_selection_pseudobulk.json`` (pseudobulk stability selection)
  - ``de_wilcoxon_vs_deseq2_topK.csv`` (per-CT method concordance)

Calls three functions from ``src.visualization.distributional_plots``:
  - ``plot_wasserstein_per_celltype_bar``
  - ``plot_de_method_concordance_bar``
  - ``plot_stability_selection_bar``

Outputs to ``--out-dir`` (default
``outputs/canonical/interpretability/figures/distributional``).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from src.visualization.distributional_plots import (
    plot_de_method_concordance_bar,
    plot_stability_selection_bar,
    plot_wasserstein_per_celltype_bar,
)
from src.visualization.theme import apply_theme

logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--wasserstein-json",
        default="outputs/canonical/interpretability/distributional_resilience/"
        "wasserstein_per_celltype_pseudobulk.json",
    )
    p.add_argument(
        "--stability-json",
        default="outputs/canonical/interpretability/distributional_resilience/"
        "stability_selection_pseudobulk.json",
    )
    p.add_argument(
        "--de-concordance-csv",
        default="outputs/canonical/interpretability/"
        "de_wilcoxon_vs_deseq2_topK.csv",
    )
    p.add_argument(
        "--out-dir",
        default="outputs/canonical/interpretability/figures/distributional",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    apply_theme()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[str] = []

    # 1. Wasserstein per CT
    wass_path = Path(args.wasserstein_json)
    if wass_path.exists():
        try:
            w = json.loads(wass_path.read_text())
            ct_names = [ct["cell_type"] for ct in w["per_cell_type"]]
            means = [ct["wasserstein_per_gene_mean"] for ct in w["per_cell_type"]]
            top_genes = [
                ct["wasserstein_per_gene_top10"][0][0]
                if ct["wasserstein_per_gene_top10"] else ""
                for ct in w["per_cell_type"]
            ]
            fig = plot_wasserstein_per_celltype_bar(
                ct_names, means, top_genes,
                save_path=out_dir / "fig_wasserstein_per_celltype",
            )
            plt.close(fig)
            rendered.append("fig_wasserstein_per_celltype")
        except (ValueError, KeyError) as exc:
            logger.warning("wasserstein: %s", exc)
    else:
        logger.warning("wasserstein JSON missing: %s", wass_path)

    # 2. DE method concordance
    conc_path = Path(args.de_concordance_csv)
    if conc_path.exists():
        try:
            df = pd.read_csv(conc_path)
            ct_labels = [f"CT_{int(i):02d}" for i in df["cell_type_index"]]
            rho = df["spearman_rho_pvalue"].tolist()
            fig = plot_de_method_concordance_bar(
                ct_labels, rho,
                method_labels=("Wilcoxon", "DESeq2"),
                save_path=out_dir / "fig_de_method_concordance",
            )
            plt.close(fig)
            rendered.append("fig_de_method_concordance")
        except (ValueError, KeyError) as exc:
            logger.warning("de-concordance: %s", exc)
    else:
        logger.warning("DE concordance CSV missing: %s", conc_path)

    # 3. Stability selection
    stab_path = Path(args.stability_json)
    if stab_path.exists():
        try:
            s = json.loads(stab_path.read_text())
            ct_names = [ct["cell_type"] for ct in s["per_cell_type"]]
            n_stable = [ct["n_stable"] for ct in s["per_cell_type"]]
            stable_genes = [ct["stable_genes"] for ct in s["per_cell_type"]]
            fig = plot_stability_selection_bar(
                ct_names, n_stable, stable_genes,
                save_path=out_dir / "fig_stability_selection_per_celltype",
            )
            plt.close(fig)
            rendered.append("fig_stability_selection_per_celltype")
        except (ValueError, KeyError) as exc:
            logger.warning("stability: %s", exc)
    else:
        logger.warning("stability JSON missing: %s", stab_path)

    logger.info(
        "rendered %d distributional figures: %s", len(rendered), rendered,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
