"""Figure 3: KSG-MI bootstrap bias-correction comparison.

For top-N CTs (default 10) by observed CMI:
  - Point: observed_cmi from `conditional_mi_per_celltype_raw_max.json`.
  - Gray error bar: with-replacement bootstrap [ci_2_5, ci_97_5] from
    `cmi_bootstrap_ci.json` — biased upward by KSG duplicate-point artifact.
  - Color (PiYG mid) error bar: Politis-Romano subsampling [ci_pr_lo, ci_pr_hi]
    from `cmi_subsample_bootstrap_ci.json` — bias-free.
  - Annotation per CT: bootstrap median offset (the bias).

Title: KSG-MI bootstrap bias: with-replacement CIs are upward-biased ~0.28 nat.
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

from src.visualization.theme import apply_theme, fmt_axes, save_fig

logger = logging.getLogger(__name__)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--cmi-point",
        default="outputs/redesign/interpretability/conditional_mi_per_celltype_raw_max.json",
    )
    p.add_argument(
        "--biased-bootstrap",
        default="outputs/redesign/interpretability/ct_ranking_nulls/cmi_bootstrap_ci.json",
    )
    p.add_argument(
        "--biasfree-bootstrap",
        default="outputs/redesign/interpretability/ct_ranking_nulls/cmi_subsample_bootstrap_ci.json",
    )
    p.add_argument(
        "--out-dir",
        default="outputs/redesign/interpretability/figures/cmi_bias",
    )
    p.add_argument("--top-n", type=int, default=10,
                   help="Number of CTs (sorted by observed CMI) to plot.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    apply_theme()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    # Load primary point estimates
    point_data = json.loads(Path(args.cmi_point).read_text())
    point_cmi = {
        item["cell_type"]: float(item["conditional_mi_given_pathology"])
        for item in point_data["per_cell_type"]
    }
    biased = json.loads(Path(args.biased_bootstrap).read_text())["per_ct"]
    biasfree = json.loads(Path(args.biasfree_bootstrap).read_text())["per_ct"]

    # Top-N CTs by observed CMI (intersection of all three sources)
    common = set(point_cmi.keys()) & set(biased.keys()) & set(biasfree.keys())
    sorted_cts = sorted(common, key=lambda c: -point_cmi[c])[: args.top_n]

    # Build vectors for plotting
    obs = np.array([point_cmi[c] for c in sorted_cts])
    biased_lo = np.array([biased[c]["ci_2_5"] for c in sorted_cts])
    biased_hi = np.array([biased[c]["ci_97_5"] for c in sorted_cts])
    biased_med = np.array([biased[c]["ci_50"] for c in sorted_cts])
    pr_lo = np.array([biasfree[c]["ci_pr_lo"] for c in sorted_cts])
    pr_hi = np.array([biasfree[c]["ci_pr_hi"] for c in sorted_cts])

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    x = np.arange(len(sorted_cts))
    dx = 0.18  # offset between biased / bias-free error bars

    # Biased (with-replacement): gray
    biased_color = "#888888"
    ax.errorbar(
        x - dx, biased_med,
        yerr=np.vstack([biased_med - biased_lo, biased_hi - biased_med]),
        fmt="s", color=biased_color, ecolor=biased_color,
        capsize=3.0, elinewidth=1.0, capthick=1.0,
        markeredgecolor="white", markeredgewidth=0.8, markersize=6,
        label="with-replacement bootstrap CI (biased; gray = bootstrap median)",
        zorder=3,
    )

    # Bias-free (Politis-Romano subsampling): tab10 green
    pr_color = "#2ca02c"
    pr_mid = (pr_lo + pr_hi) / 2.0
    ax.errorbar(
        x + dx, pr_mid,
        yerr=np.vstack([pr_mid - pr_lo, pr_hi - pr_mid]),
        fmt="D", color=pr_color, ecolor=pr_color,
        capsize=3.0, elinewidth=1.0, capthick=1.0,
        markeredgecolor="white", markeredgewidth=0.8, markersize=6,
        label="Politis–Romano subsample CI (bias-free; mid = midpoint)",
        zorder=3,
    )

    # Observed point
    ax.scatter(
        x, obs,
        marker="o", color="#d62728", s=42,
        edgecolors="white", linewidths=0.8, zorder=5,
        label="observed CMI (point estimate)",
    )

    # Annotation: bias = biased_median - observed_cmi
    bias_vals = biased_med - obs
    y_min = min(pr_lo.min(), obs.min())
    y_max = max(biased_hi.max(), pr_hi.max())
    span = y_max - y_min
    annot_y = y_max + span * 0.02
    for xi, b in zip(x, bias_vals):
        ax.text(
            xi, annot_y,
            f"+{b:.2f}",
            ha="center", va="bottom",
            fontsize=7, color=biased_color,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(sorted_cts, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("conditional MI (nats)")
    ax.set_xlabel("cell type (sorted by observed CMI)")
    ax.set_ylim(y_min - span * 0.05, annot_y + span * 0.06)

    ax.legend(loc="upper right", fontsize=7, framealpha=0.92)
    fmt_axes(ax)
    ax.set_title(
        "KSG-MI bootstrap bias: with-replacement CIs are upward-biased ~0.28 nat\n"
        "(duplicate-point KSG artifact; Politis–Romano subsampling avoids it)",
        fontsize=9, pad=8,
    )

    save_fig(fig, out_dir / "cmi_bias")
    plt.close(fig)

    # Persist summary numbers used
    summary = {
        "row_cts": sorted_cts,
        "observed_cmi": obs.tolist(),
        "biased_ci_2_5": biased_lo.tolist(),
        "biased_ci_97_5": biased_hi.tolist(),
        "biased_ci_50_median": biased_med.tolist(),
        "politis_romano_ci_lo": pr_lo.tolist(),
        "politis_romano_ci_hi": pr_hi.tolist(),
        "bias_estimate_per_ct": bias_vals.tolist(),
        "mean_bias": float(np.mean(bias_vals)),
    }
    (out_dir / "cmi_bias_data.json").write_text(json.dumps(summary, indent=2))

    elapsed = time.perf_counter() - t0
    logger.info(
        "Rendered cmi_bias.{png,pdf} (%d CTs) in %.2fs; mean bias=%.3f",
        len(sorted_cts), elapsed, summary["mean_bias"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
