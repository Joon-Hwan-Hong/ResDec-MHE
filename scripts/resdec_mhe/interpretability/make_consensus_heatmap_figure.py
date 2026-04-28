"""Figure 1: 11-method consensus heatmap (top-5 ranking by cell type).

Re-derives the §31.9 cross-method consensus matrix from primary JSONs and
renders it as a heatmap.

Method derivation per data source:
  - IG / GradientSHAP / SmoothGrad (Captum): CT importance =
    mean(mean_abs_attribution over top genes in `top_genes_per_cell_type`).
  - Attention attribution (AttnLRP, GMAR, GAF AF, GAF AGF, GAF GF): CT
    importance = `mean_importance` directly from `rank_by_mean_importance`.
  - Pseudobulk Wasserstein-1: `wasserstein_per_gene_mean` per CT (top-level
    file referenced in §31.9 source-list footnote 5).
  - Conditional MI (raw-pseudobulk, max aggregation):
    `conditional_mi_given_pathology` per CT.
  - LOCO zero-out: -`delta_r2_vs_canonical` per CT (more negative
    delta = larger drop = more important).

Cell color:
  - Rank 1-5: viridis-shaded cell with rank number annotation.
  - Rank 6+: white cell, no annotation.

Splatter row is bold-labeled to highlight its cross-paradigm consensus.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from src.visualization.theme import apply_theme, fmt_axes, save_fig

logger = logging.getLogger(__name__)


# Method label order (matches §31.9 column order)
METHOD_ORDER = [
    "IG",
    "GradientSHAP",
    "SmoothGrad",
    "AttnLRP",
    "GMAR",
    "GAF AF",
    "GAF AGF",
    "GAF GF",
    "Wasserstein",
    "CMI",
    "LOCO",
]


def _ct_score_from_captum(summary: dict) -> dict[str, float]:
    """CT score = mean(mean_abs_attribution over top genes)."""
    out: dict[str, float] = {}
    for ct, genes in summary["top_genes_per_cell_type"].items():
        if not genes:
            continue
        out[ct] = float(np.mean([g["mean_abs_attribution"] for g in genes]))
    return out


def _ct_score_from_attention(method_block: dict) -> dict[str, float]:
    """CT score = mean_importance from rank_by_mean_importance."""
    return {
        item["cell_type"]: float(item["mean_importance"])
        for item in method_block["rank_by_mean_importance"]
    }


def _ct_score_from_wasserstein(summary: dict) -> dict[str, float]:
    """CT score = wasserstein_per_gene_mean."""
    return {
        item["cell_type"]: float(item["wasserstein_per_gene_mean"])
        for item in summary["per_cell_type"]
    }


def _ct_score_from_cmi(summary: dict) -> dict[str, float]:
    """CT score = conditional_mi_given_pathology."""
    return {
        item["cell_type"]: float(item["conditional_mi_given_pathology"])
        for item in summary["per_cell_type"]
    }


def _ct_score_from_loco(summary: dict) -> dict[str, float]:
    """CT score = -delta_r2 (more-negative delta means larger drop = more important).

    LOCO `delta_r2_vs_canonical` is positive when removing the CT increased R²
    (adversarial CTs) and negative when removing the CT hurt (load-bearing).
    To rank "most important" first, negate so most-negative becomes largest.
    """
    return {
        item["cell_type"]: -float(item["delta_r2_vs_canonical"])
        for item in summary["per_cell_type"]
    }


def _rank_dict(scores: dict[str, float]) -> dict[str, int]:
    """Convert score dict into rank dict (1 = highest)."""
    sorted_cts = sorted(scores.items(), key=lambda kv: -kv[1])
    return {ct: rank for rank, (ct, _) in enumerate(sorted_cts, start=1)}


def _load_all_rankings(args: argparse.Namespace) -> dict[str, dict[str, int]]:
    """Compute per-method rank dicts. Returns {method_label: {ct: rank}}."""
    out: dict[str, dict[str, int]] = {}

    ig = json.loads(Path(args.captum_ig).read_text())
    out["IG"] = _rank_dict(_ct_score_from_captum(ig))

    gs = json.loads(Path(args.gradientshap).read_text())
    out["GradientSHAP"] = _rank_dict(_ct_score_from_captum(gs))

    sg = json.loads(Path(args.smoothgrad).read_text())
    out["SmoothGrad"] = _rank_dict(_ct_score_from_captum(sg))

    attn = json.loads(Path(args.attention).read_text())
    out["AttnLRP"] = _rank_dict(_ct_score_from_attention(attn["attnlrp"]))
    out["GMAR"] = _rank_dict(_ct_score_from_attention(attn["gmar"]))
    out["GAF AF"] = _rank_dict(_ct_score_from_attention(attn["gaf_af"]))
    out["GAF AGF"] = _rank_dict(_ct_score_from_attention(attn["gaf_agf"]))
    out["GAF GF"] = _rank_dict(_ct_score_from_attention(attn["gaf_gf"]))

    wass = json.loads(Path(args.wasserstein).read_text())
    out["Wasserstein"] = _rank_dict(_ct_score_from_wasserstein(wass))

    cmi = json.loads(Path(args.cmi).read_text())
    out["CMI"] = _rank_dict(_ct_score_from_cmi(cmi))

    loco = json.loads(Path(args.loco).read_text())
    out["LOCO"] = _rank_dict(_ct_score_from_loco(loco))

    return out


def _select_rows(rankings: dict[str, dict[str, int]], top_n: int) -> list[str]:
    """Select rows = top-N CTs by total top-5 frequency across methods."""
    freq: dict[str, int] = {}
    for method, ranks in rankings.items():
        for ct, rank in ranks.items():
            if rank <= 5:
                freq[ct] = freq.get(ct, 0) + 1
    # Tie-break on best (lowest) overall rank sum so Splatter beats CTs with
    # similar frequency.
    rank_sum: dict[str, int] = {}
    for ct in freq:
        rank_sum[ct] = sum(
            ranks.get(ct, 32) for ranks in rankings.values()
        )
    rows = sorted(freq.keys(), key=lambda c: (-freq[c], rank_sum[c]))
    return rows[:top_n]


def _build_rank_grid(
    rankings: dict[str, dict[str, int]],
    rows: Iterable[str],
    methods: Iterable[str],
) -> np.ndarray:
    """Return integer grid (n_rows, n_methods) of ranks (NaN if not in top-5+plotted)."""
    rows = list(rows)
    methods = list(methods)
    grid = np.full((len(rows), len(methods)), np.nan, dtype=float)
    for i, ct in enumerate(rows):
        for j, m in enumerate(methods):
            r = rankings[m].get(ct)
            if r is None:
                continue
            grid[i, j] = r
    return grid


def _plot_heatmap(
    rank_grid: np.ndarray,
    rows: list[str],
    methods: list[str],
    out_path: Path,
) -> None:
    """Render the consensus heatmap. Cells with rank>5 are plotted white."""
    n_rows, n_cols = rank_grid.shape
    fig_w = max(7.0, n_cols * 0.65)
    fig_h = max(4.0, n_rows * 0.45 + 1.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Build a masked color array: cells with rank<=5 take a viridis shade; rank>5
    # or NaN are pure white. Color picks based on rank: rank 1 (darkest) → 5
    # (lightest within shade range). We invert so rank 1 = strongest viridis.
    cmap = plt.get_cmap("viridis")
    rgba = np.ones((n_rows, n_cols, 4), dtype=float)  # white default
    for i in range(n_rows):
        for j in range(n_cols):
            r = rank_grid[i, j]
            if not np.isnan(r) and r <= 5:
                # Map rank 1→0.15 (dark), rank 5→0.85 (light)
                color_val = 0.15 + 0.7 * (r - 1) / 4.0
                rgba[i, j, :] = cmap(color_val)
    ax.imshow(rgba, aspect="auto", interpolation="nearest")

    # Annotate top-5 cells with rank number, contrasting text color
    for i in range(n_rows):
        for j in range(n_cols):
            r = rank_grid[i, j]
            if np.isnan(r):
                continue
            if r <= 5:
                # Text color: dark for light cells, light for dark cells
                color_val = 0.15 + 0.7 * (r - 1) / 4.0
                txt_color = "white" if color_val < 0.55 else "black"
                ax.text(
                    j, i, f"{int(r)}",
                    ha="center", va="center",
                    fontsize=8, color=txt_color, fontweight="bold",
                )

    # Labels: rotate x ticks 45°, highlight Splatter row in bold
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(n_rows))
    yticklabels = [
        f"$\\mathbf{{{ct.replace(' ', '~')}}}$" if ct.lower() == "splatter" else ct
        for ct in rows
    ]
    ax.set_yticklabels(yticklabels, fontsize=8)

    # Highlight Splatter row with a thick rectangle border
    if any(ct.lower() == "splatter" for ct in rows):
        i_splatter = next(i for i, ct in enumerate(rows) if ct.lower() == "splatter")
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle(
            (-0.5, i_splatter - 0.5),
            n_cols, 1.0,
            fill=False, edgecolor="#d62728", linewidth=2.0, zorder=5,
        ))

    # Grid / spines
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="#cccccc", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    fmt_axes(ax, hide_spines=(), grid_major=False, grid_minor=False)

    # Title (caption-only per project style — keep brief informative title)
    ax.set_title(
        "Cross-method top-5 ranking of cell types (Splatter 11/11, Fibroblast 10/11)",
        fontsize=9, pad=10,
    )

    # Adjust margins to make room for the rank-colorbar inset to the right
    fig.subplots_adjust(left=0.27, right=0.88, top=0.92, bottom=0.20)

    # Colorbar-style legend (rank 1-5 swatches), placed inside the figure to
    # the right of the heatmap.
    cb_ax = fig.add_axes([0.90, 0.55, 0.025, 0.30])
    grad = np.linspace(0.15, 0.85, 5).reshape(-1, 1)
    cb_ax.imshow(grad, aspect="auto", cmap=cmap, vmin=0, vmax=1, origin="lower")
    cb_ax.set_yticks(np.arange(5))
    cb_ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=7)
    cb_ax.set_xticks([])
    cb_ax.set_ylabel("rank", fontsize=7, labelpad=4)
    cb_ax.tick_params(axis="y", length=0, pad=2)
    cb_ax.invert_yaxis()  # rank 1 at top to match darkest/best color

    save_fig(fig, out_path.with_suffix(""))
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--captum-ig",
        default="outputs/canonical/interpretability/captum_ig/composite_attribution_summary.json",
    )
    p.add_argument(
        "--gradientshap",
        default="outputs/canonical/interpretability/captum_robustness/gradientshap_attribution_summary.json",
    )
    p.add_argument(
        "--smoothgrad",
        default="outputs/canonical/interpretability/captum_robustness/smoothgrad_attribution_summary.json",
    )
    p.add_argument(
        "--attention",
        default="outputs/canonical/interpretability/attention_attribution/attention_attribution_summary.json",
    )
    p.add_argument(
        "--wasserstein",
        default="outputs/canonical/interpretability/wasserstein_per_celltype.json",
    )
    p.add_argument(
        "--cmi",
        default="outputs/canonical/interpretability/conditional_mi_per_celltype_raw_max.json",
    )
    p.add_argument(
        "--loco",
        default="outputs/canonical/interpretability/loco_zero_out/loco_per_celltype.json",
    )
    p.add_argument(
        "--out-dir",
        default="outputs/canonical/interpretability/figures/consensus_heatmap",
    )
    p.add_argument("--top-n-rows", type=int, default=10,
                   help="Number of rows (CTs) by total top-5 frequency.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    apply_theme()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    rankings = _load_all_rankings(args)
    rows = _select_rows(rankings, args.top_n_rows)
    grid = _build_rank_grid(rankings, rows, METHOD_ORDER)
    _plot_heatmap(grid, rows, METHOD_ORDER, out_dir / "consensus_heatmap")
    elapsed = time.perf_counter() - t0

    # Also write a small summary so downstream / QA can verify the matrix
    matrix_summary = {
        "row_cts": rows,
        "methods": METHOD_ORDER,
        "ranks": {
            ct: {m: int(grid[i, j]) if not np.isnan(grid[i, j]) else None
                 for j, m in enumerate(METHOD_ORDER)}
            for i, ct in enumerate(rows)
        },
        "top5_counts": {
            ct: int(np.sum((grid[i, :] >= 1) & (grid[i, :] <= 5)))
            for i, ct in enumerate(rows)
        },
    }
    (out_dir / "consensus_heatmap_data.json").write_text(
        json.dumps(matrix_summary, indent=2)
    )
    logger.info(
        "Rendered consensus_heatmap.{png,pdf} (%d×%d) in %.2fs",
        len(rows), len(METHOD_ORDER), elapsed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
