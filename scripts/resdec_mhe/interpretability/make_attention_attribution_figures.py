"""Multi-panel figure for the AttnLRP / GMAR / GAF attention-attribution suite.

Produces one composite figure with four panels:
  A. CT × method z-scored importance heatmap (rows = CTs, cols = 5 methods).
     z-score per method (column-wise) so methods with different absolute
     scales are visually comparable. Annotates rank within each method.
  B. Cross-method consensus bar — for each CT, count of methods where it
     appears in the top-5. Sorted descending; horizontal bar chart.
  C. Per-method top-10 bar — 5 sub-panels, one per method, top-10 CTs by
     mean importance. Useful for the per-method narrative paragraph.
  D. Pairwise method-method Spearman rank correlation heatmap (5×5).
     Shows which methods agree on the per-CT ranking.

Inputs:
  outputs/redesign/interpretability/attention_attribution/per_subject_attribution.npz
  outputs/redesign/interpretability/attention_attribution/attention_attribution_summary.json

Outputs:
  outputs/redesign/interpretability/figures/attention_attribution/fig_attention_attribution_4panel.{png,pdf}
  outputs/redesign/interpretability/figures/attention_attribution/per_method_top10_bars.{png,pdf}

Usage:
    PYTHONPATH=<worktree-root> uv run python \\
        scripts/resdec_mhe/interpretability/make_attention_attribution_figures.py
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
from scipy.stats import spearmanr

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from src.data.constants import CELL_TYPE_ORDER
from src.visualization.composite import auto_letter, make_panel
from src.visualization.theme import apply_theme, fmt_axes, save_fig

logger = logging.getLogger(__name__)

METHODS = ["attnlrp", "gmar", "gaf_af", "gaf_gf", "gaf_agf"]
METHOD_LABELS = {
    "attnlrp": "AttnLRP",
    "gmar": "GMAR",
    "gaf_af": "GAF AF",
    "gaf_gf": "GAF GF",
    "gaf_agf": "GAF AGF",
}


def _load_inputs(npz_path: Path, summary_path: Path) -> tuple[dict[str, np.ndarray], dict, list[str]]:
    """Load per-subject attribution arrays and the summary JSON.

    Returns:
        per_method_per_ct: dict method → [C] mean importance across 516 subjects
        summary: parsed JSON
        ct_names: ordered list of CT names matching the array axis
    """
    d = np.load(npz_path, allow_pickle=True)
    summary = json.loads(summary_path.read_text())
    n_ct = int(d[METHODS[0]].shape[1])
    ct_names = list(CELL_TYPE_ORDER)[:n_ct]
    if len(ct_names) < n_ct:
        ct_names = ct_names + [f"ct_{i}" for i in range(len(ct_names), n_ct)]
    per_method_per_ct = {}
    for m in METHODS:
        arr = d[m]  # [N, C]
        # AttnLRP can be signed; take mean of absolute for ranking (matches summary)
        if m == "attnlrp":
            arr = np.abs(arr)
        per_method_per_ct[m] = arr.mean(axis=0)  # [C]
    return per_method_per_ct, summary, ct_names


def _z_score_per_method(per_method_per_ct: dict[str, np.ndarray]) -> np.ndarray:
    """Build [C, M] z-scored matrix with z-score along each column (method)."""
    M = len(METHODS)
    C = next(iter(per_method_per_ct.values())).shape[0]
    Z = np.zeros((C, M))
    for j, m in enumerate(METHODS):
        v = per_method_per_ct[m].astype(np.float64)
        mu = v.mean()
        sd = v.std(ddof=1) if v.std() > 0 else 1.0
        Z[:, j] = (v - mu) / sd
    return Z


def _rank_per_method(per_method_per_ct: dict[str, np.ndarray]) -> np.ndarray:
    """[C, M] rank within each method (1 = highest importance)."""
    M = len(METHODS)
    C = next(iter(per_method_per_ct.values())).shape[0]
    R = np.zeros((C, M), dtype=np.int64)
    for j, m in enumerate(METHODS):
        order = np.argsort(-per_method_per_ct[m])
        R[order, j] = np.arange(1, C + 1)
    return R


def _top5_consensus_count(rank_matrix: np.ndarray) -> np.ndarray:
    """For each CT, count of methods where it lands in top-5."""
    return ((rank_matrix >= 1) & (rank_matrix <= 5)).sum(axis=1)


def _draw_heatmap(ax: plt.Axes, Z: np.ndarray, rank: np.ndarray,
                  ct_names: list[str], top_k: int = 16) -> None:
    """Panel A: heatmap CT × method (z-scored), annotated with rank."""
    consensus = _top5_consensus_count(rank)
    order = np.argsort(-consensus)[:top_k]
    Z_sub = Z[order]
    rank_sub = rank[order]
    cts_sub = [ct_names[i] for i in order]

    im = ax.imshow(Z_sub, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], rotation=30, ha="right")
    ax.set_yticks(range(len(cts_sub)))
    ax.set_yticklabels(cts_sub, fontsize=8)
    for i in range(Z_sub.shape[0]):
        for j in range(Z_sub.shape[1]):
            ax.text(j, i, f"{rank_sub[i, j]}", ha="center", va="center",
                    fontsize=7, color="black" if abs(Z_sub[i, j]) < 1.0 else "white")
    plt.colorbar(im, ax=ax, label="z-score (per method)", fraction=0.04)
    ax.set_title("CT × method importance (top-16 by consensus)")


def _draw_consensus_bar(ax: plt.Axes, consensus: np.ndarray,
                        ct_names: list[str]) -> None:
    """Panel B: horizontal bar chart of #methods top-5 per CT."""
    nonzero = consensus > 0
    order = np.argsort(-consensus[nonzero])
    cts = np.array(ct_names)[nonzero][order]
    vals = consensus[nonzero][order]
    colors = plt.cm.viridis(vals / 5.0)
    ax.barh(range(len(cts)), vals, color=colors, edgecolor="#333333", linewidth=0.5)
    ax.set_yticks(range(len(cts)))
    ax.set_yticklabels(cts, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("# methods (top-5)")
    ax.set_xlim(0, 5)
    ax.set_title("Cross-method consensus")


def _draw_per_method_top10(ax: plt.Axes, per_method_per_ct: dict[str, np.ndarray],
                           ct_names: list[str]) -> None:
    """Panel C: grouped bar — top-10 from EACH method on a single axis.

    Uses unique color per method; CTs that appear in multiple methods'
    top-10 will have multiple bars stacked.
    """
    # Collect top-10 CTs per method
    union: dict[str, dict[str, float]] = {}
    for m in METHODS:
        v = per_method_per_ct[m]
        order = np.argsort(-v)[:10]
        for i in order:
            ct = ct_names[i]
            union.setdefault(ct, {})[m] = float(v[i])
    # Sort CTs by sum of normalized importance (z-scored)
    Z_full = _z_score_per_method(per_method_per_ct)
    rank_full = _rank_per_method(per_method_per_ct)
    consensus = _top5_consensus_count(rank_full)
    cts_sorted = sorted(
        union.keys(),
        key=lambda c: -consensus[ct_names.index(c)],
    )
    n_cts = len(cts_sorted)
    width = 0.16
    x = np.arange(n_cts)
    cmap = plt.cm.tab10
    for i, m in enumerate(METHODS):
        vals = np.array([union[ct].get(m, 0.0) for ct in cts_sorted])
        # Normalize per-method to [0, 1] so different scales are visible
        v_method = per_method_per_ct[m]
        if v_method.max() > 0:
            vals = vals / v_method.max()
        ax.bar(x + (i - 2) * width, vals, width=width,
               color=cmap(i), label=METHOD_LABELS[m], edgecolor="#333333",
               linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(cts_sorted, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("Importance / max(method)")
    ax.set_title("Per-method top-10 (CTs ranked by consensus)")
    ax.legend(fontsize=7, loc="upper right", ncol=5, frameon=True)


def _draw_method_correlation(ax: plt.Axes,
                             per_method_per_ct: dict[str, np.ndarray]) -> None:
    """Panel D: pairwise Spearman ρ between method rankings."""
    M = len(METHODS)
    rho = np.zeros((M, M))
    for i, mi in enumerate(METHODS):
        for j, mj in enumerate(METHODS):
            r, _ = spearmanr(per_method_per_ct[mi], per_method_per_ct[mj])
            rho[i, j] = r
    im = ax.imshow(rho, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(M))
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], rotation=30, ha="right")
    ax.set_yticks(range(M))
    ax.set_yticklabels([METHOD_LABELS[m] for m in METHODS])
    for i in range(M):
        for j in range(M):
            ax.text(j, i, f"{rho[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="black" if abs(rho[i, j]) < 0.7 else "white")
    plt.colorbar(im, ax=ax, label="Spearman ρ", fraction=0.04)
    ax.set_title("Pairwise method ranking agreement")


def make_4panel_figure(per_method_per_ct: dict[str, np.ndarray],
                       ct_names: list[str]) -> plt.Figure:
    """Hand-crafted 2×2 layout via composite.make_panel."""
    Z = _z_score_per_method(per_method_per_ct)
    rank = _rank_per_method(per_method_per_ct)
    consensus = _top5_consensus_count(rank)

    panels = [
        {
            "draw": (lambda ax: _draw_heatmap(ax, Z, rank, ct_names)),
            "title": "",
        },
        {
            "draw": (lambda ax: _draw_consensus_bar(ax, consensus, ct_names)),
            "title": "",
        },
        {
            "draw": (lambda ax: _draw_per_method_top10(ax, per_method_per_ct, ct_names)),
            "title": "",
        },
        {
            "draw": (lambda ax: _draw_method_correlation(ax, per_method_per_ct)),
            "title": "",
        },
    ]
    fig = make_panel(panels, layout=(2, 2), figsize=(14, 10),
                     wspace=0.4, hspace=0.5)
    fig.suptitle(
        "Attention attribution: AttnLRP / GMAR / GAF (AF, GF, AGF) on canonical 5-fold checkpoints (N=516)",
        fontsize=11, y=0.995,
    )
    return fig


def main(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    apply_theme()
    npz_path = Path(args.npz_path)
    summary_path = Path(args.summary_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_method_per_ct, summary, ct_names = _load_inputs(npz_path, summary_path)
    logger.info(
        "loaded %d subjects × %d cell types across %d methods",
        summary["cohort"]["n_subjects"], summary["cohort"]["n_cell_types"],
        len(METHODS),
    )
    fig = make_4panel_figure(per_method_per_ct, ct_names)
    save_fig(fig, out_dir / "fig_attention_attribution_4panel",
             formats=("png", "pdf"))
    logger.info("Wrote %s/fig_attention_attribution_4panel.{png,pdf}", out_dir)
    plt.close(fig)
    return 0


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--npz-path",
        default="outputs/redesign/interpretability/attention_attribution/per_subject_attribution.npz",
    )
    p.add_argument(
        "--summary-path",
        default="outputs/redesign/interpretability/attention_attribution/attention_attribution_summary.json",
    )
    p.add_argument(
        "--out-dir",
        default="outputs/redesign/interpretability/figures/attention_attribution",
    )
    return p


if __name__ == "__main__":
    sys.exit(main(_build_argparser().parse_args()))
