"""Post-chain figures for cogn-residual EXP-052..058.

Reads aggregate JSONs produced earlier in this analysis pass and writes 4
figures to outputs/canonical/cogn_residual/figures/:

  fig_post_chain_5seed_paired.png       - 5-seed paired stacked vs TabPFN-only
  fig_post_chain_cf_top5.png            - CF top-5 CTs (relative + absolute)
  fig_post_chain_distributional_top5.png - Wasserstein top-5 + CMI top-5 (2 panels)
  fig_post_chain_learning_curve.png     - N vs cross-seed mean R² with errors

Per feedback_no_default_protagonist.md:
  - Stable color palette via CELL_TYPE_COLORS for any per-CT coloring.
  - No red highlights, no special markers on a single CT.
  - All entries equal-billing.
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

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))

from src.data.constants import CELL_TYPE_ORDER  # noqa: E402
from src.visualization.config import CELL_TYPE_COLORS  # noqa: E402
from src.visualization.theme import apply_theme, save_fig  # noqa: E402


def _ct_color(ct: str) -> str:
    return CELL_TYPE_COLORS.get(ct, "#808080")


def fig_5seed_paired(stats_json: Path, out_stem: Path) -> None:
    """Per-seed paired boxplot: stacked-base vs TabPFN-only across 5 seeds."""
    d = json.loads(stats_json.read_text())
    seeds = d["seeds"]
    stacked = {int(k): v for k, v in d["stacked_per_fold_r2"].items()}
    tabpfn = {int(k): v for k, v in d["tabpfn_per_fold_r2"].items()}

    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    width = 0.32
    x = np.arange(len(seeds))
    bp_stacked = ax.boxplot(
        [stacked[s] for s in seeds],
        positions=x - width / 2 - 0.04, widths=width,
        patch_artist=True,
        boxprops=dict(facecolor="#1f77b4", alpha=0.6, edgecolor="black", linewidth=0.7),
        medianprops=dict(color="black", linewidth=1.0),
        whiskerprops=dict(color="black", linewidth=0.6),
        capprops=dict(color="black", linewidth=0.6),
        flierprops=dict(marker="o", markersize=2, markerfacecolor="#1f77b4", alpha=0.5),
    )
    bp_tabpfn = ax.boxplot(
        [tabpfn[s] for s in seeds],
        positions=x + width / 2 + 0.04, widths=width,
        patch_artist=True,
        boxprops=dict(facecolor="#ff7f0e", alpha=0.6, edgecolor="black", linewidth=0.7),
        medianprops=dict(color="black", linewidth=1.0),
        whiskerprops=dict(color="black", linewidth=0.6),
        capprops=dict(color="black", linewidth=0.6),
        flierprops=dict(marker="o", markersize=2, markerfacecolor="#ff7f0e", alpha=0.5),
    )
    # Cross-seed mean lines
    ax.axhline(d["cross_seed_mean_stacked"], color="#1f77b4",
               linestyle="--", linewidth=0.7, alpha=0.7, zorder=0)
    ax.axhline(d["cross_seed_mean_tabpfn"], color="#ff7f0e",
               linestyle="--", linewidth=0.7, alpha=0.7, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds])
    ax.set_xlabel("Training seed")
    ax.set_ylabel("Per-fold R² (composite, residualized cogn target)")
    ax.set_title(
        f"Stacked TabPFN+RF vs TabPFN-only (Δ = +{d['delta_paired_mean_n25']:.4f}, "
        f"n=25 paired, Wilcoxon p = {d['wilcoxon_greater_p_n25']:.1e})"
    )
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#1f77b4", alpha=0.6, edgecolor="black"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#ff7f0e", alpha=0.6, edgecolor="black"),
    ]
    ax.legend(legend_handles, ["Stacked base", "TabPFN-only"], loc="lower right",
              frameon=True, framealpha=0.9)
    save_fig(fig, out_stem, dpi=600)
    plt.close(fig)


def fig_cf_top5(cf_json: Path, out_stem: Path) -> None:
    """CF top-5 CTs side-by-side bar chart, relative + absolute."""
    d = json.loads(cf_json.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=False)
    for ax, mode in zip(axes, ["relative", "absolute"]):
        rows = d[f"{mode}_top5_CTs"]
        names = [r[0] for r in rows]
        mags = [r[1] for r in rows]
        colors = [_ct_color(n) for n in names]
        # Reverse for top-down reading
        y = np.arange(len(names))[::-1]
        ax.barh(y, mags, color=colors, edgecolor="black", linewidth=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.set_xlabel("Aggregate top-K |Δ| sum")
        elapsed = d[f"{mode}_elapsed_min"]
        n_succ = d[f"{mode}_n_success"]
        n_subj = d[f"{mode}_n_subjects"]
        ax.set_title(f"{mode}  ({n_succ}/{n_subj} converged, {elapsed:.0f} min)")
        ax.set_axisbelow(True)
    fig.suptitle(
        f"Variant CF top-5 CTs (intra-method top-5 overlap = "
        f"{d['top5_ct_overlap_relative_vs_absolute']}/5; top-10 (CT, gene) pair "
        f"overlap = {d['top10_pair_overlap_relative_vs_absolute']}/10)"
    )
    fig.tight_layout()
    save_fig(fig, out_stem, dpi=600)
    plt.close(fig)


def fig_distributional_top5(distr_json: Path, out_stem: Path) -> None:
    """Wasserstein top-5 (left) + CMI top-5 (right)."""
    d = json.loads(distr_json.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))

    # Left: Wasserstein top-5 by mean per-CT W₁
    w_top = d["wasserstein"]["top5_CTs_by_mean_W1"]
    names = [r["cell_type"] for r in w_top]
    means = [r["mean_W1"] for r in w_top]
    top_genes = [r["top3_genes"][0][0] if r["top3_genes"] else "?" for r in w_top]
    colors = [_ct_color(n) for n in names]
    y = np.arange(len(names))[::-1]
    axes[0].barh(y, means, color=colors, edgecolor="black", linewidth=0.6)
    for yi, gene in zip(y, top_genes):
        axes[0].text(0.001, yi, f"  top: {gene}", va="center", ha="left",
                     fontsize=6, color="white" if yi >= 0 else "black", weight="bold")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names)
    axes[0].set_xlabel("Mean Wasserstein-1 distance across genes")
    axes[0].set_title(
        f"Wasserstein top-5 CTs ("
        f"n_resilient = {d['wasserstein']['n_resilient']}, "
        f"n_vulnerable = {d['wasserstein']['n_vulnerable']})"
    )

    # Right: CMI top-5 (max aggregator)
    c_top = d["cmi_max_top5"]
    names_c = [r["cell_type"] for r in c_top]
    cmi = [r["conditional_mi"] for r in c_top]
    delta = [r["delta"] for r in c_top]
    colors_c = [_ct_color(n) for n in names_c]
    y = np.arange(len(names_c))[::-1]
    axes[1].barh(y, cmi, color=colors_c, edgecolor="black", linewidth=0.6)
    # Annotate Δ value on each bar
    for yi, d_val in zip(y, delta):
        sign = "+" if d_val > 0 else ""
        axes[1].text(0.002, yi, f"  Δ = {sign}{d_val:.3f}", va="center", ha="left",
                     fontsize=6, color="white" if yi >= 0 else "black", weight="bold")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(names_c)
    axes[1].set_xlabel("Conditional MI given pathology (raw pseudobulk, max-agg)")
    axes[1].set_title("Conditional MI top-5 CTs (Δ = unc − cond)")
    fig.tight_layout()
    save_fig(fig, out_stem, dpi=600)
    plt.close(fig)


def fig_learning_curve(lc_json: Path, out_stem: Path) -> None:
    """N vs cross-seed mean R² with error bars; canonical N=516 reference."""
    d = json.loads(lc_json.read_text())
    Ns = d["Ns"]
    seeds = d["seeds"]
    means = [d["cross_seed_mean_per_N"][str(N)] for N in Ns]
    stds = [d["cross_seed_std_per_N"][str(N)] for N in Ns]
    canon_r2 = d["canonical_N516_seed42_mean"]
    canon_per_fold = d["canonical_N516_seed42_per_fold_r2"]
    canon_std = float(np.std(canon_per_fold, ddof=1))

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ax.errorbar(Ns, means, yerr=stds, fmt="o-", capsize=3, capthick=0.8,
                elinewidth=0.8, color="#1f77b4", markersize=5, linewidth=1.0,
                label="Sub-N (cross-seed mean ± std)")
    # Canonical reference at N=516
    ax.errorbar([516], [canon_r2], yerr=[canon_std], fmt="s", capsize=3,
                capthick=0.8, elinewidth=0.8, color="#d62728", markersize=6,
                label=f"N=516 (seed 42, mean ± fold std = {canon_r2:.3f} ± {canon_std:.3f})")
    # Per-seed individual points to show seed spread
    for s in seeds:
        seed_means = [d["seed_means_per_N"][str(N)][str(s)] for N in Ns]
        ax.plot(Ns, seed_means, "o-", color="#1f77b4", alpha=0.18,
                markersize=2.5, linewidth=0.5)
    ax.set_xticks(Ns + [516])
    ax.set_xlabel("Training subset N (per fold)")
    ax.set_ylabel("Per-fold R² (variant residualized target)")
    ax.set_title(
        "Variant A learning curve (5 seeds × 4 sub-Ns + N=516 reference) — "
        "all sub-N pairwise p > 0.14; sub-N vs N=516 p = 0.0312 each"
    )
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)
    fig.tight_layout()
    save_fig(fig, out_stem, dpi=600)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    default_root = _ROOT / "outputs/canonical/cogn_residual/gpath_only"
    p.add_argument("--seed-stats", type=Path,
                   default=default_root / "seed_variation_paired_stats.json")
    p.add_argument("--cf-summary", type=Path,
                   default=default_root / "interpretability/cf_aggregate_summary.json")
    p.add_argument("--distr-summary", type=Path,
                   default=default_root / "interpretability/distributional_top_cts_summary.json")
    p.add_argument("--lc-stats", type=Path,
                   default=default_root / "learning_curve_k5/learning_curve_paired_stats.json")
    p.add_argument("--out-dir", type=Path,
                   default=_ROOT / "outputs/canonical/cogn_residual/figures")
    args = p.parse_args()

    apply_theme("paper")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fig_5seed_paired(args.seed_stats, args.out_dir / "fig_post_chain_5seed_paired")
    print(f"wrote {args.out_dir / 'fig_post_chain_5seed_paired.png'}")

    fig_cf_top5(args.cf_summary, args.out_dir / "fig_post_chain_cf_top5")
    print(f"wrote {args.out_dir / 'fig_post_chain_cf_top5.png'}")

    fig_distributional_top5(args.distr_summary,
                             args.out_dir / "fig_post_chain_distributional_top5")
    print(f"wrote {args.out_dir / 'fig_post_chain_distributional_top5.png'}")

    fig_learning_curve(args.lc_stats, args.out_dir / "fig_post_chain_learning_curve")
    print(f"wrote {args.out_dir / 'fig_post_chain_learning_curve.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
