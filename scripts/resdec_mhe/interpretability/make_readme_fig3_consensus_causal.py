#!/usr/bin/env python
"""Render Figure 3 for the ResDec-MHE README redesign.

Two side-by-side panels that together summarize the *interpretability twist*:

  * Left:  11-method x 31-CT consensus heatmap. Each (method, CT) cell is
           binary -- 1 if that CT is in the method's top-5 ranking, else 0
           (rendered as viridis sequential). CTs are sorted by total top-5
           appearances across methods (most agreed-upon at the top). Two CTs
           (Splatter, Fibroblast) are top-5 in 11/11 methods; below those
           the agreement decays.
  * Right: SAE causal-patching null. Histogram of patch DeltaR^2 in the
           saturate mode for the lone Splatter-correlated SAE feature
           (5 fold values) overlaid with the random-feature noise floor
           (10 random control features x 5 folds = 50 samples). Annotates
           Splatter mean +/- std vs random mean +/- std and a dashed zero
           line. The two distributions overlap -- the Splatter-correlated
           feature is correlated, not causal.

Inputs
------
  Left panel (re-derived from primary JSONs to recover the *full* 31-CT
  universe; the published consensus_heatmap_data.json only stores the
  top-10 visualised rows):

  - outputs/canonical/interpretability/captum_ig/composite_attribution_summary.json
  - outputs/canonical/interpretability/captum_robustness/gradientshap_attribution_summary.json
  - outputs/canonical/interpretability/captum_robustness/smoothgrad_attribution_summary.json
  - outputs/canonical/interpretability/attention_attribution/attention_attribution_summary.json
  - outputs/canonical/interpretability/wasserstein_per_celltype.json
  - outputs/canonical/interpretability/conditional_mi_per_celltype_raw_max.json
  - outputs/canonical/interpretability/loco_zero_out/loco_per_celltype.json

  Right panel:

  - outputs/canonical/interpretability/sae_causal_patching.json

Outputs
-------
  - figures/fig3_consensus_causal.png  (~1400 x 600 at 600 dpi)
  - Verification numbers printed to stdout.

Usage
-----
  PYTHONPATH=<worktree-root> \\
    uv run python scripts/resdec_mhe/interpretability/make_readme_fig3_consensus_causal.py

Idempotence
-----------
The pipeline is fully deterministic (no sampling, no model inference --
only JSON I/O + numpy ranking). PYTHONHASHSEED is pinned defensively so
matplotlib's color paths are bit-identical across reruns.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Set PYTHONHASHSEED defensively so color-selection paths in matplotlib
# (which can hit Python set-iteration order in edge cases) are stable.
os.environ.setdefault("PYTHONHASHSEED", "42")

import matplotlib

matplotlib.use("Agg")  # must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from src.visualization.theme import (  # noqa: E402
    PALETTES,
    apply_theme,
    fmt_axes,
    save_fig,
)

logger = logging.getLogger(__name__)


# Method label order (matches the design doc and section 31.9 column order).
METHOD_ORDER: list[str] = [
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


# -----------------------------------------------------------------------------
# Per-method CT score helpers (shared with make_consensus_heatmap_figure.py).
# -----------------------------------------------------------------------------
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
    """CT score = -delta_r2_vs_canonical (more-negative delta = more important)."""
    return {
        item["cell_type"]: -float(item["delta_r2_vs_canonical"])
        for item in summary["per_cell_type"]
    }


def _rank_dict(scores: dict[str, float]) -> dict[str, int]:
    """Convert {ct: score} into {ct: rank} (rank 1 = highest score)."""
    sorted_cts = sorted(scores.items(), key=lambda kv: -kv[1])
    return {ct: rank for rank, (ct, _) in enumerate(sorted_cts, start=1)}


def _load_all_rankings(args: argparse.Namespace) -> dict[str, dict[str, int]]:
    """Compute per-method rank dicts. Returns {method_label: {ct: rank}}.

    Each per-method JSON enumerates CT-level scores; we rank them locally
    (rank 1 = highest score) so the top-K membership for each method is
    well-defined regardless of cross-method scale differences.
    """
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


def _ct_universe(rankings: dict[str, dict[str, int]]) -> list[str]:
    """Return the union of all CTs across all methods (expected: 31 CTs).

    The per-method JSONs each enumerate the same 31-CT taxonomy, but we
    take the union defensively to tolerate methods that omit CTs with all-
    zero scores (none currently do).
    """
    universe: set[str] = set()
    for ranks in rankings.values():
        universe.update(ranks.keys())
    return sorted(universe)


def _build_top5_grid(
    rankings: dict[str, dict[str, int]],
    cts: list[str],
    methods: list[str],
    *,
    top_k: int = 5,
) -> np.ndarray:
    """Return a binary (n_cts, n_methods) matrix where 1 = CT is top-K for the method.

    Cell coloring is binary (per design doc: "is this CT in this method's
    top-5?") -- per-fold fractions are not available in the source JSONs.
    """
    grid = np.zeros((len(cts), len(methods)), dtype=float)
    for i, ct in enumerate(cts):
        for j, m in enumerate(methods):
            r = rankings[m].get(ct)
            if r is not None and r <= top_k:
                grid[i, j] = 1.0
    return grid


def _sort_cts_by_top5_count(
    rankings: dict[str, dict[str, int]],
    cts: list[str],
    methods: list[str],
    *,
    top_k: int = 5,
) -> tuple[list[str], dict[str, int]]:
    """Sort CTs descending by total top-K appearances; tie-break on best rank-sum.

    Returns (sorted_cts, top5_counts).
    """
    counts: dict[str, int] = {}
    rank_sum: dict[str, int] = {}
    # Tie-breaker: lowest rank-sum wins among CTs with the same top-K count.
    # The "absent CT" rank fallback is one past the worst possible rank
    # (= max_rank + 1), computed from the rankings dict so the heatmap
    # tolerates expanded CT taxonomies (matches make_consensus_heatmap_figure.py).
    max_rank_per_method = max(
        (max(ranks.values(), default=0) for ranks in rankings.values()),
        default=0,
    )
    absent_rank_fallback = max_rank_per_method + 1

    for ct in cts:
        c = 0
        s = 0
        for m in methods:
            r = rankings[m].get(ct, absent_rank_fallback)
            if r <= top_k:
                c += 1
            s += r
        counts[ct] = c
        rank_sum[ct] = s

    # Most-appearances-at-top means descending count first; lower rank-sum
    # is "better" (smaller mean rank), so sort ascending on rank_sum within
    # equal counts. Final tie-break on alphabetical CT name for stability.
    sorted_cts = sorted(cts, key=lambda c: (-counts[c], rank_sum[c], c))
    return sorted_cts, counts


# -----------------------------------------------------------------------------
# Right panel: causal-patching null helpers.
# -----------------------------------------------------------------------------
def _load_causal_patching_saturate(
    payload: dict,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Extract Splatter (5,) + random (50,) DeltaR^2 arrays in saturate mode.

    Returns
    -------
    splatter_deltas : np.ndarray of shape (5,)
        Per-fold patch DeltaR^2 for the Splatter-correlated SAE feature
        (saturate mode -- the canonical p99 patch value).
    random_deltas : np.ndarray of shape (50,)
        Pooled per-fold patch DeltaR^2 across the 10 random control
        features x 5 folds = 50 samples.
    summary : dict[str, float]
        The source JSON's published summary statistics for round-trip
        verification (splatter_saturate_delta_r2_mean, ..._std,
        random_saturate_delta_r2_mean, ..._std, n_random_pooled).
    """
    sp_block = payload["splatter_feature_aggregate"]["saturate"]
    splatter_deltas = np.asarray(
        sp_block["delta_r2_per_fold"], dtype=np.float64,
    )
    if splatter_deltas.shape != (5,):
        raise ValueError(
            f"Expected 5 splatter deltas (one per fold); got shape "
            f"{splatter_deltas.shape}"
        )

    # Random features: payload["random_feature_aggregate"] is keyed by feature
    # index (str). Each value is a length-5 list of per-fold deltas in
    # saturate mode (the only mode persisted at the random-feature level
    # per run_sae_causal_patching.py output schema).
    random_block = payload["random_feature_aggregate"]
    random_lists: list[np.ndarray] = []
    for feat_idx_str, deltas in random_block.items():
        arr = np.asarray(deltas, dtype=np.float64)
        if arr.shape != (5,):
            raise ValueError(
                f"Random feature {feat_idx_str!r} has shape {arr.shape}; "
                f"expected (5,)"
            )
        random_lists.append(arr)
    if len(random_lists) != 10:
        raise ValueError(
            f"Expected 10 random control features; got {len(random_lists)}"
        )
    random_deltas = np.concatenate(random_lists, axis=0)
    if random_deltas.shape != (50,):
        raise ValueError(
            f"Expected pooled (50,) random deltas; got shape "
            f"{random_deltas.shape}"
        )

    summary = payload.get("summary_statistics", {})
    return splatter_deltas, random_deltas, summary


# -----------------------------------------------------------------------------
# Drawing.
# -----------------------------------------------------------------------------
def _draw_left_heatmap(
    ax: plt.Axes,
    grid: np.ndarray,
    rows: list[str],
    methods: list[str],
    counts: dict[str, int],
) -> None:
    """Render the binary "is this CT in top-5 for this method?" heatmap.

    Cells are colored on a viridis sequential scale (0 = light, 1 = dark)
    to match the design spec; white gridlines separate cells; CT names on
    Y, method names on X (rotated 45 degrees).
    """
    cmap = PALETTES["sequential"]
    im = ax.imshow(
        grid,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
    )

    n_rows, n_cols = grid.shape

    # Axis labels.
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(np.arange(n_rows))
    # Annotate each CT label with its top-5 count (e.g. "Splatter (11/11)")
    # so the agreement structure is readable from the y-axis alone.
    n_methods = len(methods)
    yticklabels = [f"{ct}  ({counts[ct]}/{n_methods})" for ct in rows]
    ax.set_yticklabels(yticklabels, fontsize=6)

    # White gridlines between cells (set as minor ticks at half-integer
    # positions; major-tick grid is disabled so gridlines don't double up).
    ax.set_xticks(np.arange(-0.5, n_cols, 1.0), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1.0), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Caption-only style; no in-figure title per project convention. Use
    # fmt_axes with grid disabled (handled manually above) and keep all
    # spines for the heatmap data-area frame.
    fmt_axes(ax, hide_spines=(), grid_major=False, grid_minor=False)

    # Compact colorbar to the right of the heatmap, labeled "in top-5"
    # so the binary semantics are explicit.
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.025, pad=0.02, ticks=[0, 1])
    cb.ax.set_yticklabels(["no", "yes"], fontsize=7)
    cb.set_label("in top-5", fontsize=7)
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(length=0)

    ax.set_xlabel("Method", fontsize=8)
    ax.set_ylabel("Cell type", fontsize=8)


def _draw_right_histogram(
    ax: plt.Axes,
    splatter_deltas: np.ndarray,
    random_deltas: np.ndarray,
) -> None:
    """Render the Splatter vs random-control DeltaR^2 histogram.

    Splatter (5 values) and random (50 values) are drawn as overlapping
    histograms with shared bin edges spanning the union of both samples.
    A dashed zero line, mean +/- std textboxes, and per-distribution
    labels are annotated.
    """
    # Distinguishable colors: viridis dark for random, viridis brighter
    # for Splatter (matches the task's "use cmap[0] / cmap[5]" hint by
    # sampling the sequential colormap at fixed positions).
    cmap = PALETTES["sequential"]
    splatter_color = cmap(0.75)   # brighter (yellow-green)
    random_color = cmap(0.20)     # darker (deep purple)

    # Shared bin edges spanning the union of both samples with a small
    # symmetric padding so distributions don't slam against the panel
    # edges. We force at least 12 bins so the 50-sample random
    # distribution shows visible structure.
    all_vals = np.concatenate([splatter_deltas, random_deltas])
    lo = float(np.min(all_vals))
    hi = float(np.max(all_vals))
    span = max(hi - lo, 1e-6)
    pad = 0.10 * span
    bin_edges = np.linspace(lo - pad, hi + pad, 13)

    ax.hist(
        random_deltas, bins=bin_edges,
        color=random_color, edgecolor="white", linewidth=0.5,
        alpha=0.75,
        label=f"Random-feature null (n=10x5={len(random_deltas)})",
        zorder=2,
    )
    ax.hist(
        splatter_deltas, bins=bin_edges,
        color=splatter_color, edgecolor="white", linewidth=0.6,
        alpha=0.85,
        label=f"Splatter feature (1/323, n={len(splatter_deltas)})",
        zorder=3,
    )

    # Dashed zero line.
    ax.axvline(
        0.0,
        color="#444444",
        linestyle="--",
        linewidth=1.0,
        zorder=4,
        label=r"$\Delta R^2$ = 0",
    )

    # Means as solid vertical ticks at the top of the panel + textbox
    # annotations summarizing mean +/- std for each distribution.
    sp_mean = float(splatter_deltas.mean())
    sp_std = float(splatter_deltas.std(ddof=1))
    rand_mean = float(random_deltas.mean())
    rand_std = float(random_deltas.std(ddof=1))

    # Mean lines for visual orientation (thin solid).
    ax.axvline(sp_mean, color=splatter_color, linewidth=1.4, alpha=0.95, zorder=5)
    ax.axvline(rand_mean, color=random_color, linewidth=1.4, alpha=0.95, zorder=5)

    # Top-left textbox: Splatter summary.
    ax.text(
        0.02, 0.97,
        (
            "Splatter feature (saturate)\n"
            r"mean $\pm$ std = "
            + f"{sp_mean:+.4f} $\\pm$ {sp_std:.4f}"
        ),
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=7,
        bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                  edgecolor=splatter_color, linewidth=0.8),
        zorder=6,
    )
    # Top-right textbox: Random summary.
    ax.text(
        0.98, 0.97,
        (
            "Random control (saturate)\n"
            r"mean $\pm$ std = "
            + f"{rand_mean:+.4f} $\\pm$ {rand_std:.4f}"
        ),
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=7,
        bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                  edgecolor=random_color, linewidth=0.8),
        zorder=6,
    )

    ax.set_xlabel(r"$\Delta R^2$ (val, per fold)")
    ax.set_ylabel("Count")
    ax.legend(loc="lower center", fontsize=6, frameon=True, ncol=1,
              bbox_to_anchor=(0.5, -0.02))
    fmt_axes(ax)


# -----------------------------------------------------------------------------
# Orchestrator.
# -----------------------------------------------------------------------------
def make_figure(
    *,
    rankings: dict[str, dict[str, int]],
    splatter_deltas: np.ndarray,
    random_deltas: np.ndarray,
) -> tuple[plt.Figure, list[str], dict[str, int]]:
    """Build the 2-panel figure. Returns (fig, sorted_cts, top5_counts)."""
    apply_theme("paper")

    cts = _ct_universe(rankings)
    sorted_cts, counts = _sort_cts_by_top5_count(rankings, cts, METHOD_ORDER)
    grid = _build_top5_grid(rankings, sorted_cts, METHOD_ORDER)

    # 14 x 6 figure -> at 100 dpi screen pre-render this is ~1400 x 600,
    # which matches the design spec; final raster is 600 dpi for paper.
    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, 6),
        gridspec_kw={"width_ratios": [1.0, 1.0]},
    )

    _draw_left_heatmap(axes[0], grid, sorted_cts, METHOD_ORDER, counts)
    _draw_right_histogram(axes[1], splatter_deltas, random_deltas)

    # Tighten subplot spacing so the heatmap labels + histogram legend
    # both fit without overlap. The default tight_layout would crop the
    # rotated x-tick labels, so we adjust manually.
    fig.subplots_adjust(left=0.18, right=0.96, top=0.96, bottom=0.16,
                        wspace=0.30)
    return fig, sorted_cts, counts


def _print_report(
    sorted_cts: list[str],
    counts: dict[str, int],
    splatter_deltas: np.ndarray,
    random_deltas: np.ndarray,
    summary_from_json: dict[str, float],
) -> None:
    """Echo verification numbers to stdout."""
    n_methods = len(METHOD_ORDER)
    n_cts = len(sorted_cts)
    print("=" * 72)
    print("README Figure 3 -- methods convergence + causal contradiction")
    print("=" * 72)
    print(f"  n_methods             : {n_methods}")
    print(f"  n_cts                 : {n_cts}")
    print(f"  top_2_cts_by_count    : {sorted_cts[:2]}")
    print(f"  top_2_counts          : "
          f"[{counts[sorted_cts[0]]}/{n_methods}, "
          f"{counts[sorted_cts[1]]}/{n_methods}]")

    # Per-distribution summary statistics (computed locally) + cross-check
    # against the producer JSON's summary block when present.
    sp_mean = float(splatter_deltas.mean())
    sp_std = float(splatter_deltas.std(ddof=1))
    rand_mean = float(random_deltas.mean())
    rand_std = float(random_deltas.std(ddof=1))
    print(f"  splatter_saturate_delta_r2_mean : {sp_mean:+.6e}")
    print(f"  splatter_saturate_delta_r2_std  : {sp_std:+.6e}")
    print(f"  random_saturate_delta_r2_mean   : {rand_mean:+.6e}")
    print(f"  random_saturate_delta_r2_std    : {rand_std:+.6e}")
    print(f"  n_splatter_samples              : {len(splatter_deltas)}")
    print(f"  n_random_samples                : {len(random_deltas)}")

    # Round-trip cross-check against the producer JSON's published summary.
    # If the JSON used ddof=0 (numpy default for ndarray.std()) and we use
    # ddof=1 (sample std), the numbers will differ slightly; we still print
    # the published values verbatim so the operator can confirm at a glance.
    if summary_from_json:
        print("  --- summary_from_json ---")
        for k in (
            "splatter_saturate_delta_r2_mean",
            "splatter_saturate_delta_r2_std",
            "random_saturate_delta_r2_mean",
            "random_saturate_delta_r2_std",
            "n_random_pooled",
        ):
            v = summary_from_json.get(k)
            if v is not None:
                if isinstance(v, float):
                    print(f"    {k:42s}: {v:+.6e}")
                else:
                    print(f"    {k:42s}: {v}")
    print("=" * 72)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])

    parser.add_argument(
        "--captum-ig", type=Path,
        default=_WORKTREE_ROOT / "outputs/canonical/interpretability/captum_ig/composite_attribution_summary.json",
    )
    parser.add_argument(
        "--gradientshap", type=Path,
        default=_WORKTREE_ROOT / "outputs/canonical/interpretability/captum_robustness/gradientshap_attribution_summary.json",
    )
    parser.add_argument(
        "--smoothgrad", type=Path,
        default=_WORKTREE_ROOT / "outputs/canonical/interpretability/captum_robustness/smoothgrad_attribution_summary.json",
    )
    parser.add_argument(
        "--attention", type=Path,
        default=_WORKTREE_ROOT / "outputs/canonical/interpretability/attention_attribution/attention_attribution_summary.json",
    )
    parser.add_argument(
        "--wasserstein", type=Path,
        default=_WORKTREE_ROOT / "outputs/canonical/interpretability/wasserstein_per_celltype.json",
    )
    parser.add_argument(
        "--cmi", type=Path,
        default=_WORKTREE_ROOT / "outputs/canonical/interpretability/conditional_mi_per_celltype_raw_max.json",
    )
    parser.add_argument(
        "--loco", type=Path,
        default=_WORKTREE_ROOT / "outputs/canonical/interpretability/loco_zero_out/loco_per_celltype.json",
    )
    parser.add_argument(
        "--causal-patching", type=Path,
        default=_WORKTREE_ROOT / "outputs/canonical/interpretability/sae_causal_patching.json",
    )
    parser.add_argument(
        "--out-stem", type=Path,
        default=_WORKTREE_ROOT / "figures/fig3_consensus_causal",
        help="Output path stem (no extension); save_fig appends .png.",
    )
    args = parser.parse_args()

    logger.info("[fig3] loading per-method rankings")
    rankings = _load_all_rankings(args)
    logger.info(
        "[fig3] loaded %d methods (each with %s CTs)",
        len(rankings),
        ", ".join(str(len(v)) for v in rankings.values()),
    )

    logger.info("[fig3] loading SAE causal-patching JSON: %s", args.causal_patching)
    patching_payload = json.loads(Path(args.causal_patching).read_text())
    splatter_deltas, random_deltas, summary_from_json = _load_causal_patching_saturate(
        patching_payload,
    )
    logger.info(
        "[fig3] splatter_deltas shape=%s  random_deltas shape=%s",
        splatter_deltas.shape, random_deltas.shape,
    )

    fig, sorted_cts, counts = make_figure(
        rankings=rankings,
        splatter_deltas=splatter_deltas,
        random_deltas=random_deltas,
    )

    out_png = args.out_stem.with_suffix(".png")
    if out_png.exists():
        logger.info("[fig3] removing preexisting %s", out_png)
        out_png.unlink()

    written = save_fig(fig, args.out_stem, formats=("png",))
    plt.close(fig)
    for w in written:
        logger.info("[fig3] wrote %s", w)

    _print_report(sorted_cts, counts, splatter_deltas, random_deltas,
                  summary_from_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
