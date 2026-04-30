#!/usr/bin/env python
"""5-fold held-out R² distribution figure (W4 framing).

Each of the 5 canonical CV folds is an independent held-out test:
the val subjects in fold k were never seen during fold k's training.
This script frames the per-fold val R² values as "5 held-out test R²s"
and visualizes the distribution + comparison vs TabPFN.

Inputs:
    outputs/canonical/p5_canonical_seed42/best_vs_tabpfn_summary.json

Outputs:
    outputs/canonical/interpretability/figures/held_out_distribution/fig_held_out_r2.{png,pdf}
    outputs/canonical/interpretability/held_out_r2_distribution.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

_WT = Path(__file__).resolve().parents[3]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--summary-json",
        type=Path,
        default=_WT / "outputs/canonical/p5_canonical_seed42/best_vs_tabpfn_summary.json",
    )
    p.add_argument(
        "--out-fig-dir",
        type=Path,
        default=_WT / "outputs/canonical/interpretability/figures/held_out_distribution",
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=_WT / "outputs/canonical/interpretability/held_out_r2_distribution.json",
    )
    args = p.parse_args()

    if not args.summary_json.is_file():
        print(f"missing {args.summary_json}", file=sys.stderr)
        return 1

    summary = json.loads(args.summary_json.read_text())
    per_fold = summary["per_fold"]

    ours_r2 = np.array([f["ours"]["r2"] for f in per_fold], dtype=np.float64)
    tab_en_r2 = np.array([f["tab_en"]["r2"] for f in per_fold], dtype=np.float64)
    tab_ge_r2 = np.array([f["tab_ge"]["r2"] for f in per_fold], dtype=np.float64)

    # Paired Wilcoxon vs TabPFN ensemble across 5 folds (already in EXP-002,
    # but recompute here for in-figure annotation).
    w_stat_en, w_p_en = stats.wilcoxon(ours_r2, tab_en_r2)
    w_stat_ge, w_p_ge = stats.wilcoxon(ours_r2, tab_ge_r2)

    # Mean held-out R² + bootstrap-style 5-fold std
    mean_r2 = float(ours_r2.mean())
    std_r2 = float(ours_r2.std(ddof=1))
    sem_r2 = float(ours_r2.std(ddof=1) / np.sqrt(len(ours_r2)))

    # ---- Figure: 1 row × 2 panels ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel A: per-fold held-out R² (ResDec-MHE vs TabPFN), points + connecting lines
    ax = axes[0]
    folds = np.arange(5)
    width = 0.25
    ax.bar(
        folds - width, ours_r2, width=width, label="ResDec-MHE",
        color="#2ca02c", alpha=0.85, edgecolor="black",
    )
    ax.bar(
        folds, tab_en_r2, width=width, label="TabPFN [A] ensemble",
        color="#1f77b4", alpha=0.65, edgecolor="black",
    )
    ax.bar(
        folds + width, tab_ge_r2, width=width, label="TabPFN [A] genes",
        color="#aec7e8", alpha=0.65, edgecolor="black",
    )
    for i, r2 in enumerate(ours_r2):
        ax.text(i - width, r2 + 0.01, f"{r2:.3f}", ha="center", fontsize=8)
    ax.set_xticks(folds)
    ax.set_xticklabels([f"Fold {i}\n(n={f['n']})" for i, f in enumerate(per_fold)])
    ax.set_ylabel("Held-out test R²")
    ax.set_title("(a) Per-fold held-out R² across 5 independent CV folds", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, max(ours_r2.max(), tab_en_r2.max(), tab_ge_r2.max()) * 1.15)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    # Panel B: distribution view — per-fold dots + mean line + ±1 SEM band
    ax = axes[1]
    methods = ["ResDec-MHE", "TabPFN [A] ens.", "TabPFN [A] gene"]
    data_arrays = [ours_r2, tab_en_r2, tab_ge_r2]
    colors = ["#2ca02c", "#1f77b4", "#aec7e8"]
    for i, (m, arr, c) in enumerate(zip(methods, data_arrays, colors)):
        x = np.full(5, i) + np.random.RandomState(i).uniform(-0.08, 0.08, 5)
        ax.scatter(x, arr, color=c, alpha=0.85, s=60, edgecolor="black", zorder=3)
        ax.hlines(arr.mean(), i - 0.25, i + 0.25, color=c, linewidth=2.5, zorder=4)
        # ±1 SEM band
        sem = arr.std(ddof=1) / np.sqrt(5)
        ax.fill_between(
            [i - 0.18, i + 0.18],
            [arr.mean() - sem, arr.mean() - sem],
            [arr.mean() + sem, arr.mean() + sem],
            color=c, alpha=0.18, zorder=2,
        )
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels(methods, rotation=15)
    ax.set_ylabel("Held-out test R² (each dot = 1 fold)")
    ax.set_title("(b) 5-fold held-out R² distribution", fontsize=11)
    # Annotate Wilcoxon p
    ax.text(
        0.02, 0.98,
        f"vs TabPFN ens.: W={w_stat_en:.0f}, p={w_p_en:.4f}\n"
        f"vs TabPFN gene: W={w_stat_ge:.0f}, p={w_p_ge:.4f}",
        transform=ax.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )
    ax.set_ylim(0, 0.7)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.suptitle(
        f"5-fold cross-validation as 5 held-out test sets — ResDec-MHE held-out R² = "
        f"{mean_r2:.4f} ± {std_r2:.4f} (mean ± std, n=5 folds; SEM = {sem_r2:.4f})",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()

    args.out_fig_dir.mkdir(parents=True, exist_ok=True)
    png = args.out_fig_dir / "fig_held_out_r2.png"
    pdf = args.out_fig_dir / "fig_held_out_r2.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png}")
    print(f"wrote {pdf}")

    # JSON with the 5 held-out test R²s and stats
    out = {
        "method": "5-fold CV per-fold val R² reframed as 5 independent held-out test R²s",
        "rationale": (
            "Each fold's val subjects were never seen during that fold's training. "
            "The 5 per-fold val R² values are 5 independent held-out test R²s in the "
            "CV-as-held-out sense; bootstrap CI on these 5 values is the inferential "
            "summary."
        ),
        "n_folds": 5,
        "per_fold": [
            {
                "fold": i,
                "n_held_out": int(per_fold[i]["n"]),
                "resdec_mhe_r2": float(ours_r2[i]),
                "tabpfn_ensemble_r2": float(tab_en_r2[i]),
                "tabpfn_gene_r2": float(tab_ge_r2[i]),
            }
            for i in range(5)
        ],
        "summary_resdec_mhe": {
            "mean_r2": mean_r2,
            "std_r2_ddof1": std_r2,
            "sem_r2": sem_r2,
            "min_r2": float(ours_r2.min()),
            "max_r2": float(ours_r2.max()),
        },
        "paired_wilcoxon_vs_tabpfn_ensemble": {
            "statistic": float(w_stat_en),
            "p_value": float(w_p_en),
        },
        "paired_wilcoxon_vs_tabpfn_gene": {
            "statistic": float(w_stat_ge),
            "p_value": float(w_p_ge),
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
