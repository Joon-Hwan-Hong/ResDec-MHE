"""Orchestrator: render attention figures from canonical artefacts.

Calls three functions appended to ``src.visualization.attention_plots``:
  - head-attention chord diagram (heads × top-K cell types)
  - head-fingerprint UMAP (subjects clustered by attention pattern,
    colored by residual quartile)
  - head-attention bootstrap CI heatmap (per-(head, CT) means with CI
    width panel; annotates cells whose CI excludes the uniform-attention
    null 1 / n_ct).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from src.visualization.attention_plots import (
    plot_head_attention_bootstrap_ci,
    plot_head_attention_chord,
    plot_head_fingerprint_umap,
)
from src.visualization.theme import apply_theme

logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--head-attention-npz",
        default="outputs/redesign/interpretability/pathology_attention_per_subject.npz",
    )
    p.add_argument(
        "--captum-summary-json",
        default="outputs/redesign/interpretability/captum_ig/composite_attribution_summary.json",
    )
    p.add_argument(
        "--residual-csv",
        default="outputs/redesign/interpretability/residual_per_subject.csv",
    )
    p.add_argument(
        "--out-dir",
        default="outputs/redesign/interpretability/figures/attention",
    )
    p.add_argument(
        "--bootstrap-n", type=int, default=1000,
        help="Bootstrap resamples for head-attention CI heatmap.",
    )
    p.add_argument(
        "--bootstrap-ci-level", type=float, default=0.95,
        help="Two-sided CI level for head-attention bootstrap.",
    )
    p.add_argument(
        "--bootstrap-seed", type=int, default=42,
        help="RNG seed for head-attention bootstrap resampling.",
    )
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    apply_theme()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered = []

    head_path = Path(args.head_attention_npz)
    if not head_path.exists():
        logger.error("head attention npz missing: %s", head_path)
        return 1
    d = np.load(head_path, allow_pickle=True)
    head_attention = None
    for key in ("attention", "per_subject", "head_attention", "per_head_attention"):
        if key in d.files:
            head_attention = np.asarray(d[key], dtype=np.float64)
            break
    if head_attention is None:
        logger.error("could not find head attention array in %s; keys=%s",
                     head_path, d.files)
        return 1

    # Axis-aligned CT names from CELL_TYPE_ORDER (NOT the attribution-ranked
    # list, which isn't axis-aligned). The attention tensor's last axis is
    # in CT-index order per the datamodule / encoder conventions.
    from src.data.constants import CELL_TYPE_ORDER
    n_ct_attn = head_attention.shape[-1]
    ct_names = list(CELL_TYPE_ORDER[:n_ct_attn])

    try:
        fig = plot_head_attention_chord(
            head_attention, ct_names,
            save_path=out_dir / "fig_head_attention_chord",
        )
        plt.close(fig)
        rendered.append("fig_head_attention_chord")
    except (ValueError, ImportError) as exc:
        logger.warning("chord: %s", exc)

    res_path = Path(args.residual_csv)
    if res_path.exists():
        residual_df = pd.read_csv(res_path)
        res_map = dict(zip(
            residual_df["ROSMAP_IndividualID"].astype(str),
            residual_df["residual"].astype(float),
        ))
        # Need per-subject residuals aligned to head_attention's subject axis.
        # Fallback: assume same order as residual_df. If npz has "subject_ids",
        # use it.
        if "subject_ids" in d.files:
            subj_ids = [str(s) for s in d["subject_ids"]]
        else:
            subj_ids = residual_df["ROSMAP_IndividualID"].astype(str).tolist()
        residuals = np.array([res_map.get(s, np.nan) for s in subj_ids])
        try:
            fig = plot_head_fingerprint_umap(
                head_attention, residuals,
                save_path=out_dir / "fig_head_fingerprint_umap",
            )
            plt.close(fig)
            rendered.append("fig_head_fingerprint_umap")
        except (ValueError, ImportError) as exc:
            logger.warning("UMAP: %s", exc)

    # Bootstrap CI heatmap — uses the uniform-attention null 1/n_ct as
    # the significance reference (softmax-normalized attention over
    # cell types sums to 1 per subject).
    try:
        n_ct = head_attention.shape[-1]
        fig = plot_head_attention_bootstrap_ci(
            head_attention, ct_names,
            n_bootstrap=args.bootstrap_n,
            ci_level=args.bootstrap_ci_level,
            null_reference=1.0 / n_ct,
            seed=args.bootstrap_seed,
            save_path=out_dir / "fig_head_attention_bootstrap_ci",
        )
        plt.close(fig)
        rendered.append("fig_head_attention_bootstrap_ci")
    except ValueError as exc:
        logger.warning("bootstrap CI: %s", exc)

    logger.info("rendered %d attention figures: %s", len(rendered), rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
