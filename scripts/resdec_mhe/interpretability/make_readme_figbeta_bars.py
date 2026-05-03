#!/usr/bin/env python
"""README Fig appendix beta: 12-panel unified bar-chart grid for the ResDec-MHE README.

Each panel = one interpretability method, showing the top-5 cell types as
horizontal bars. Bar length = the method's actual value (mean abs attribution /
W1 / MI / |Delta R^2| / count). Bar color = the deterministic per-CT color
shared with Fig 3 main; bars belonging to sparse CTs (zero_frac >= 0.20 in
``ct_coverage_full_cohort.json``) are hatched + faded so the reader can tell
"low-coverage signals" apart from "well-covered signals" at a glance.

This is the "backup style" for readers who want a clean ranking view rather
than the method-faithful Fig 3 main.

Panels (2 rows x 6 cols = 12 panels):

  Row 1: IG, GradientSHAP, SmoothGrad, AttnLRP, GMAR, GAF AF
  Row 2: GAF AGF, GAF GF, Wasserstein, CMI, LOCO, Consensus

For each panel:

  Method                | source / value formula
  --------------------- | ------------------------------------------------------
  IG                    | captum_ig/composite_attribution_summary.json::
                        |   cell_types_ranked_by_total_attribution
                        |   value = total_abs_attribution
  GradientSHAP          | captum_robustness/gradientshap_attribution_summary.json
                        |   value = total_abs_attribution
  SmoothGrad            | captum_robustness/smoothgrad_attribution_summary.json
                        |   value = total_abs_attribution
  AttnLRP               | attention_attribution/attention_attribution_summary.json::
                        |   attnlrp.rank_by_mean_importance, value = mean_importance
  GMAR                  | (same file).gmar.rank_by_mean_importance
  GAF AF                | (same file).gaf_af.rank_by_mean_importance
  GAF AGF               | (same file).gaf_agf.rank_by_mean_importance
  GAF GF                | (same file).gaf_gf.rank_by_mean_importance
  Wasserstein           | distributional_resilience/
                        |   wasserstein_per_celltype_pseudobulk.json::per_cell_type
                        |   value = wasserstein_per_gene_mean (sort descending)
  CMI                   | conditional_mi_per_celltype_raw_max.json::per_cell_type
                        |   value = conditional_mi_given_pathology (sort descending)
  LOCO                  | loco_zero_out/loco_per_celltype.json::per_cell_type
                        |   sort by delta_r2_vs_canonical ASCENDING
                        |   bar value = abs(delta_r2_vs_canonical)
  Consensus             | union of the 11 method top-5 sets, count occurrences;
                        |   bar value = count (out of 11)

Coverage encoding (uses ``ct_coverage_full_cohort.json``):

  - well_covered (zero_frac < 0.20)  -> solid bar at full alpha
  - sparse       (zero_frac >= 0.20) -> hatched ('//') + reduced alpha (0.55)

Bar colors come from ``src.visualization.config.CELL_TYPE_COLORS`` (the same
deterministic per-CT dict used by Fig 3 main and other ResDec-MHE figures),
so a reader looking across all 12 panels can recognize CTs by hue.

Outputs
-------
  - figures/figbeta_bars.png  (~14 x 10 in at 600 dpi, target < 1.5 MB)
  - Verification: 12 method names + their top-5 CTs + values printed to stdout.

Usage
-----
  PYTHONPATH=<worktree-root> \\
    uv run python scripts/resdec_mhe/interpretability/make_readme_figbeta_bars.py
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path

# Pin PYTHONHASHSEED defensively so any matplotlib paths that touch dict
# iteration order remain bit-stable across reruns (mirrors fig3 main).
os.environ.setdefault("PYTHONHASHSEED", "42")

import matplotlib

matplotlib.use("Agg")  # must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from src.visualization.config import CELL_TYPE_COLORS, get_cell_type_color  # noqa: E402
from src.visualization.theme import (  # noqa: E402
    apply_theme,
    fmt_axes,
    save_fig,
)

logger = logging.getLogger(__name__)


# Panel order: 2 rows x 6 cols. Row 1 covers gradient/attribution + 3 attention
# variants; Row 2 covers the other attention variant + GAF GF + W1, CMI, LOCO,
# Consensus. The 11 "consensus-feeding" methods are the same set used by
# Fig 3 main and EXP-016.
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
    "Consensus",
]

# These 11 feed the consensus panel. "Consensus" itself is panel 12 and is
# *derived* from the union of these 11 top-5 sets.
CONSENSUS_METHODS: list[str] = [m for m in METHOD_ORDER if m != "Consensus"]

# x-axis label per panel (varies by method).
X_AXIS_LABELS: dict[str, str] = {
    "IG":           "total |attribution|",
    "GradientSHAP": "total |attribution|",
    "SmoothGrad":   "total |attribution|",
    "AttnLRP":      "mean importance",
    "GMAR":         "mean importance",
    "GAF AF":       "mean importance",
    "GAF AGF":      "mean importance",
    "GAF GF":       "mean importance",
    "Wasserstein":  "W$_1$ per gene (mean)",
    "CMI":          "MI(g; y | pathology)",
    "LOCO":         r"$|\Delta R^2|$",
    "Consensus":    "appearances (of 11)",
}


# -----------------------------------------------------------------------------
# Per-method (CT, value) extraction.
# -----------------------------------------------------------------------------
def _captum_ct_values(summary: dict) -> list[tuple[str, float]]:
    """Return [(ct, total_abs_attribution), ...] in descending value order.

    Source field: ``cell_types_ranked_by_total_attribution`` (already sorted
    descending in the producer JSON, but we re-sort defensively).
    """
    items = summary["cell_types_ranked_by_total_attribution"]
    pairs = [(e["cell_type"], float(e["total_abs_attribution"])) for e in items]
    pairs.sort(key=lambda kv: -kv[1])
    return pairs


def _attention_ct_values(method_block: dict) -> list[tuple[str, float]]:
    """Return [(ct, mean_importance), ...] in descending value order.

    Source field: ``rank_by_mean_importance`` under one of attnlrp / gmar /
    gaf_af / gaf_agf / gaf_gf.
    """
    items = method_block["rank_by_mean_importance"]
    pairs = [(e["cell_type"], float(e["mean_importance"])) for e in items]
    pairs.sort(key=lambda kv: -kv[1])
    return pairs


def _wasserstein_ct_values(summary: dict) -> list[tuple[str, float]]:
    """Return [(ct, wasserstein_per_gene_mean), ...] in descending order."""
    items = summary["per_cell_type"]
    pairs = [(e["cell_type"], float(e["wasserstein_per_gene_mean"])) for e in items]
    pairs.sort(key=lambda kv: -kv[1])
    return pairs


def _cmi_ct_values(summary: dict) -> list[tuple[str, float]]:
    """Return [(ct, conditional_mi_given_pathology), ...] in descending order."""
    items = summary["per_cell_type"]
    pairs = [
        (e["cell_type"], float(e["conditional_mi_given_pathology"]))
        for e in items
    ]
    pairs.sort(key=lambda kv: -kv[1])
    return pairs


def _loco_ct_values(summary: dict) -> list[tuple[str, float]]:
    """Return [(ct, |delta_r2_vs_canonical|), ...] sorted by ASCENDING delta.

    LOCO's "most load-bearing" CT has the *most negative* delta (zeroing it
    causes the biggest drop in R^2). The bar value is the absolute value, so
    longer bar == larger drop == more important. Sorting ASCENDING on the
    signed delta puts the most-load-bearing CT first.
    """
    items = summary["per_cell_type"]
    items_sorted = sorted(items, key=lambda e: e["delta_r2_vs_canonical"])
    return [
        (e["cell_type"], abs(float(e["delta_r2_vs_canonical"])))
        for e in items_sorted
    ]


# -----------------------------------------------------------------------------
# Loaders.
# -----------------------------------------------------------------------------
def load_method_top5(args: argparse.Namespace) -> dict[str, list[tuple[str, float]]]:
    """Load each method's top-5 (CT, value) pairs.

    Returns
    -------
    dict[str, list[tuple[str, float]]]
        ``{method_label: [(ct1, v1), ..., (ct5, v5)]}`` for the 11 base
        methods plus the ``"Consensus"`` panel (count out of 11).
    """
    top5: dict[str, list[tuple[str, float]]] = {}

    ig = json.loads(Path(args.captum_ig).read_text())
    top5["IG"] = _captum_ct_values(ig)[:5]

    gs = json.loads(Path(args.gradientshap).read_text())
    top5["GradientSHAP"] = _captum_ct_values(gs)[:5]

    sg = json.loads(Path(args.smoothgrad).read_text())
    top5["SmoothGrad"] = _captum_ct_values(sg)[:5]

    attn = json.loads(Path(args.attention).read_text())
    top5["AttnLRP"] = _attention_ct_values(attn["attnlrp"])[:5]
    top5["GMAR"]    = _attention_ct_values(attn["gmar"])[:5]
    top5["GAF AF"]  = _attention_ct_values(attn["gaf_af"])[:5]
    top5["GAF AGF"] = _attention_ct_values(attn["gaf_agf"])[:5]
    top5["GAF GF"]  = _attention_ct_values(attn["gaf_gf"])[:5]

    wass = json.loads(Path(args.wasserstein).read_text())
    top5["Wasserstein"] = _wasserstein_ct_values(wass)[:5]

    cmi = json.loads(Path(args.cmi).read_text())
    top5["CMI"] = _cmi_ct_values(cmi)[:5]

    loco = json.loads(Path(args.loco).read_text())
    top5["LOCO"] = _loco_ct_values(loco)[:5]

    # Consensus: count CT appearances across the 11 base method top-5 sets;
    # bar value is the count (so the consensus panel uses an integer x-axis).
    consensus_counter: Counter[str] = Counter()
    for m in CONSENSUS_METHODS:
        for ct, _ in top5[m]:
            consensus_counter[ct] += 1
    consensus_top5 = consensus_counter.most_common(5)
    # Cast counts to float for type uniformity with other methods.
    top5["Consensus"] = [(ct, float(c)) for ct, c in consensus_top5]

    return top5


def load_coverage(coverage_path: Path) -> dict[str, dict]:
    """Load per-CT coverage info: ``{ct_name: {zero_frac, well_covered, ...}}``.

    Source: ``ct_coverage_full_cohort.json::per_ct``. The producer file
    pre-computes the boolean ``well_covered`` flag (= ``zero_frac <
    zero_frac_threshold``); we double-check against the threshold reported
    in the same JSON to keep this script self-contained.
    """
    payload = json.loads(coverage_path.read_text())
    threshold = float(payload.get("zero_frac_threshold", 0.20))
    per_ct = payload["per_ct"]
    out: dict[str, dict] = {}
    for ct, info in per_ct.items():
        zf = float(info["zero_frac"])
        out[ct] = {
            "zero_frac": zf,
            "well_covered": zf < threshold,
            "median_cells": int(info.get("median_cells", 0)),
        }
    return out


# -----------------------------------------------------------------------------
# Drawing.
# -----------------------------------------------------------------------------
def _format_value(value: float, method: str) -> str:
    """Produce a compact textual annotation for the bar tip."""
    if method == "Consensus":
        # Bar value is a count out of 11.
        return f"{int(round(value))}/{len(CONSENSUS_METHODS)}"
    # Use 3 sig-figs scientific or fixed-point depending on magnitude.
    if value == 0.0:
        return "0"
    abs_v = abs(value)
    if abs_v >= 1e-2:
        return f"{value:.3f}"
    return f"{value:.2e}"


def _draw_method_panel(
    ax: plt.Axes,
    method: str,
    top5: list[tuple[str, float]],
    coverage: dict[str, dict],
) -> None:
    """Render a single method's top-5 horizontal bar chart.

    Bars are colored per-CT (via ``CELL_TYPE_COLORS``); bars whose CT is
    ``sparse`` (zero_frac >= 0.20) get a hatched fill at reduced alpha, so
    the reader can immediately tell which top-5 entries are driven by
    well-covered vs sparse cell types.
    """
    if not top5:
        # Defensive: if a method had < 5 entries, leave panel blank with title.
        ax.set_title(method, fontsize=9)
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color="#888888")
        ax.set_axis_off()
        return

    # Order so the highest-value bar is at the *top* of the panel (idiomatic
    # for ranked horizontal bar charts).
    cts = [t[0] for t in top5]
    values = [t[1] for t in top5]
    y_positions = np.arange(len(cts))[::-1]  # rank 1 at top

    for y, ct, v in zip(y_positions, cts, values):
        info = coverage.get(ct)
        well_covered = bool(info["well_covered"]) if info is not None else True
        color = get_cell_type_color(ct)
        if well_covered:
            ax.barh(
                y, v,
                color=color, edgecolor="#222222", linewidth=0.5,
                alpha=0.95, zorder=3,
            )
        else:
            # Sparse CT: hatched + faded so it visually reads as "low-coverage
            # signal". Hatch lines are ~7 pt thick, lighter color edge.
            ax.barh(
                y, v,
                color=color, edgecolor="#222222", linewidth=0.5,
                alpha=0.55, hatch="//", zorder=3,
            )
        # Tip annotation (bar value), placed just past the bar end. Use a
        # small fixed offset proportional to the panel xlim so the text never
        # collides with the bar end.
        # Note: xlim is unset here; we'll compute placement after seeing all
        # values via the post-loop x-axis sizing block below.

    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        [_short_ct(ct) for ct in cts],
        fontsize=6,
    )

    # x-axis: from 0 to slightly past the max value so tip annotations fit.
    vmax = max(values) if values else 1.0
    if vmax <= 0:
        vmax = 1.0
    pad = 0.18 * vmax  # leaves room for the tip annotation text
    ax.set_xlim(0, vmax + pad)

    # Now place tip annotations using the finalized xlim.
    for y, ct, v in zip(y_positions, cts, values):
        # Place text just past bar tip; a 1.5 % shift relative to xlim.
        x_text = v + 0.015 * (vmax + pad)
        ax.text(
            x_text, y,
            _format_value(v, method),
            ha="left", va="center", fontsize=6, color="#333333",
            zorder=4,
        )

    ax.set_xlabel(X_AXIS_LABELS.get(method, ""), fontsize=7)
    ax.set_title(method, fontsize=9, fontweight="bold")

    # Tighten panel spines and grid (paper style).
    fmt_axes(ax, hide_spines=("top", "right"), grid_major=True, grid_minor=False)
    ax.tick_params(axis="x", labelsize=6)
    ax.tick_params(axis="y", labelsize=6)


def _short_ct(ct: str, max_len: int = 28) -> str:
    """Truncate long CT names so y-tick labels stay legible in 12 small panels."""
    if len(ct) <= max_len:
        return ct
    # Common multi-word truncations: keep first word + last word's initial
    # so e.g. "Committed oligodendrocyte precursor" -> "Committed olig. precursor".
    return ct[: max_len - 1] + "..."


def _draw_coverage_legend(fig: plt.Figure) -> None:
    """Add a small legend at the bottom explaining the coverage encoding."""
    # Two proxy patches: one solid (well-covered), one hatched (sparse).
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor="#888888", edgecolor="#222222", linewidth=0.5,
              label="well-covered (zero_frac < 0.20)"),
        Patch(facecolor="#888888", edgecolor="#222222", linewidth=0.5,
              alpha=0.55, hatch="//",
              label="sparse (zero_frac $\\geq$ 0.20)"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.005),
        frameon=True,
        edgecolor="#cccccc",
    )


# -----------------------------------------------------------------------------
# Orchestrator.
# -----------------------------------------------------------------------------
def make_figure(
    method_top5: dict[str, list[tuple[str, float]]],
    coverage: dict[str, dict],
) -> plt.Figure:
    """Build the 12-panel figure (2 rows x 6 cols, ~14 x 10 in)."""
    apply_theme("paper")

    n_rows, n_cols = 2, 6
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(14, 10),
        gridspec_kw={"hspace": 0.55, "wspace": 0.85},
    )

    # Flatten in row-major order so panel index == METHOD_ORDER index.
    flat_axes = axes.ravel()

    for i, method in enumerate(METHOD_ORDER):
        ax = flat_axes[i]
        top5 = method_top5.get(method, [])
        _draw_method_panel(ax, method, top5, coverage)

    # Single shared legend at the bottom (proxy patches for solid vs hatched).
    _draw_coverage_legend(fig)

    # Margin tuning so y-tick CT labels + tip annotations both fit. The
    # bottom margin reserves room for the coverage legend.
    fig.subplots_adjust(left=0.08, right=0.98, top=0.94, bottom=0.10)
    return fig


# -----------------------------------------------------------------------------
# Verification print.
# -----------------------------------------------------------------------------
def _print_report(
    method_top5: dict[str, list[tuple[str, float]]],
    coverage: dict[str, dict],
) -> None:
    """Print 12 method names + their top-5 (CT, value, coverage_status)."""
    print("=" * 80)
    print("README Fig appendix beta -- 12-panel unified bar grid")
    print("=" * 80)
    for method in METHOD_ORDER:
        top5 = method_top5.get(method, [])
        x_label = X_AXIS_LABELS.get(method, "")
        print(f"\n{method}  [x = {x_label}]")
        for rank, (ct, v) in enumerate(top5, start=1):
            info = coverage.get(ct)
            covered = (
                "well-covered" if (info is not None and info["well_covered"])
                else "sparse"
            )
            zf = info["zero_frac"] if info is not None else float("nan")
            if method == "Consensus":
                v_text = f"{int(round(v))}/{len(CONSENSUS_METHODS)}"
            else:
                v_text = f"{v:.6e}" if abs(v) < 1e-2 else f"{v:.6f}"
            print(
                f"  {rank}. {ct:42s} value={v_text:>16s}  "
                f"zero_frac={zf:.3f} ({covered})"
            )
    print("=" * 80)


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
        default=_WORKTREE_ROOT / "outputs/canonical/interpretability/distributional_resilience/wasserstein_per_celltype_pseudobulk.json",
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
        "--coverage", type=Path,
        default=_WORKTREE_ROOT / "outputs/canonical/interpretability/ct_coverage_full_cohort.json",
    )
    parser.add_argument(
        "--out-stem", type=Path,
        default=_WORKTREE_ROOT / "figures/figbeta_bars",
        help="Output path stem (no extension); save_fig appends .png.",
    )
    args = parser.parse_args()

    logger.info("[figbeta] loading 11 method top-5 + consensus")
    method_top5 = load_method_top5(args)
    logger.info(
        "[figbeta] loaded %d panels (%s)",
        len(method_top5), ", ".join(method_top5.keys()),
    )

    logger.info("[figbeta] loading coverage from %s", args.coverage)
    coverage = load_coverage(args.coverage)
    logger.info(
        "[figbeta] loaded coverage for %d CTs (sparse: %d)",
        len(coverage),
        sum(1 for c in coverage.values() if not c["well_covered"]),
    )

    fig = make_figure(method_top5, coverage)

    out_png = args.out_stem.with_suffix(".png")
    if out_png.exists():
        logger.info("[figbeta] removing preexisting %s", out_png)
        out_png.unlink()

    written = save_fig(fig, args.out_stem, formats=("png",))
    plt.close(fig)
    for w in written:
        logger.info("[figbeta] wrote %s", w)
        try:
            size_kb = w.stat().st_size / 1024
            logger.info("[figbeta]   size = %.1f KB", size_kb)
        except OSError:
            pass

    _print_report(method_top5, coverage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
