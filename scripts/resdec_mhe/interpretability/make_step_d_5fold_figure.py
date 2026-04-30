#!/usr/bin/env python
"""Render Step D 5-fold counterfactual success-rate box-and-whisker figure.

Reads the Step D 5-fold aggregate produced by
``aggregate_step_d_5fold.py`` and renders one 4-panel figure with
per-fold success-rate distributions for the 4 (mode, delta) cells:

    relative δ=0.5  |  relative δ=0.3
    -----------------------------------
    absolute δ=0.5  |  absolute δ=0.3

Each panel shows two box-and-whisker boxes (resilient, vulnerable) over
the 5 per-fold success rates; per-fold dots are overlaid; the
across-folds aggregate success rate is drawn as a horizontal dashed line.

Output:
    outputs/canonical/interpretability/figures/step_d_5fold/
        fig_step_d_5fold_box_whisker.{png,pdf}    (600 DPI)

This is a CPU-only script: it consumes the already-aggregated JSON and
emits a static figure.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from src.visualization.theme import apply_theme  # noqa: E402

logger = logging.getLogger(__name__)

# Mode/delta cells in left-to-right, top-to-bottom panel order.
PANEL_KEYS: tuple[tuple[str, str, str], ...] = (
    ("relative_delta0.5", "Relative δ = 0.5", "(a)"),
    ("relative_delta0.3", "Relative δ = 0.3", "(b)"),
    ("absolute_delta0.5", "Absolute δ = 0.5", "(c)"),
    ("absolute_delta0.3", "Absolute δ = 0.3", "(d)"),
)
REGIME_ORDER: tuple[str, ...] = ("resilient", "vulnerable")
REGIME_COLORS: dict[str, str] = {
    "resilient": "#1f77b4",   # tab10 blue
    "vulnerable": "#d62728",  # tab10 red
}


def collect_per_fold_rates(summary: dict) -> dict[str, dict[str, list[float]]]:
    """Return ``{panel_key: {regime: [rate_per_fold...]}}``.

    Folds are read in numeric order (fold_0..fold_4). Missing folds yield
    ``np.nan`` so the box-and-whisker still draws over partial data.
    """
    out: dict[str, dict[str, list[float]]] = {}
    fold_keys = sorted(
        (k for k in summary.get("per_fold", {}).keys() if k.startswith("fold_")),
        key=lambda s: int(s.split("_")[1]),
    )
    for panel_key, _, _ in PANEL_KEYS:
        per_regime: dict[str, list[float]] = {r: [] for r in REGIME_ORDER}
        for fk in fold_keys:
            cell = summary["per_fold"][fk].get(panel_key, {})
            for regime in REGIME_ORDER:
                rec = cell.get(regime, {})
                rate = rec.get("success_rate")
                per_regime[regime].append(float(rate) if rate is not None else float("nan"))
        out[panel_key] = per_regime
    return out


def collect_across_folds(summary: dict) -> dict[str, dict[str, float]]:
    """Return ``{panel_key: {regime: across-folds success_rate}}``."""
    out: dict[str, dict[str, float]] = {}
    af = summary.get("across_folds", {})
    for panel_key, _, _ in PANEL_KEYS:
        per_regime: dict[str, float] = {}
        for regime in REGIME_ORDER:
            rec = af.get(panel_key, {}).get(regime, {})
            per_regime[regime] = float(rec.get("success_rate", float("nan")))
        out[panel_key] = per_regime
    return out


def make_figure(summary: dict) -> plt.Figure:
    """Render the 2x2 box-and-whisker figure."""
    apply_theme(style="paper", use_scienceplots=True)
    rates = collect_per_fold_rates(summary)
    across = collect_across_folds(summary)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 9.0))
    rng = np.random.default_rng(42)  # deterministic jitter

    for ax, (panel_key, title, sublabel) in zip(axes.flat, PANEL_KEYS):
        per_regime = rates[panel_key]
        af_per_regime = across[panel_key]

        positions = [1.0, 2.0]
        data = [per_regime[r] for r in REGIME_ORDER]
        # Drop NaN within each regime for the boxplot itself (boxplot
        # cannot handle NaN). Keep them as no-op dots.
        clean = [[v for v in dl if not np.isnan(v)] for dl in data]

        bp = ax.boxplot(
            clean,
            positions=positions,
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.4},
            whiskerprops={"color": "black", "linewidth": 1.0},
            capprops={"color": "black", "linewidth": 1.0},
            boxprops={"linewidth": 1.0},
        )
        for box, regime in zip(bp["boxes"], REGIME_ORDER):
            box.set_facecolor(REGIME_COLORS[regime])
            box.set_alpha(0.30)
            box.set_edgecolor("black")

        # Per-fold dots overlaid on the box; small horizontal jitter for
        # readability when multiple folds collide on the same value.
        for pos, regime, dl in zip(positions, REGIME_ORDER, data):
            color = REGIME_COLORS[regime]
            n = len(dl)
            jitter = rng.uniform(-0.10, 0.10, size=n)
            ax.scatter(
                np.full(n, pos) + jitter,
                dl,
                s=50,
                color=color,
                edgecolors="white",
                linewidth=0.8,
                zorder=3,
                alpha=0.95,
            )
            # Across-folds aggregate dashed line per regime over the box's
            # x-range only (no panel-spanning lines).
            agg = af_per_regime.get(regime, float("nan"))
            if not np.isnan(agg):
                ax.plot(
                    [pos - 0.32, pos + 0.32],
                    [agg, agg],
                    linestyle="--",
                    color=color,
                    linewidth=2.0,
                    alpha=0.85,
                    zorder=4,
                )
                ax.text(
                    pos + 0.34,
                    agg,
                    f"agg={agg:.2f}",
                    va="center",
                    fontsize=8,
                    color=color,
                )

        ax.set_xticks(positions)
        ax.set_xticklabels([r.capitalize() for r in REGIME_ORDER], fontsize=11)
        ax.set_ylim(-0.05, 1.10)
        ax.set_yticks(np.arange(0.0, 1.01, 0.2))
        ax.set_ylabel("Per-fold success rate", fontsize=11)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.set_title(f"{sublabel} {title}", fontsize=12)

    fig.suptitle(
        "Step D — 5-fold counterfactual success rate (per-fold dots + box, dashed "
        "line = across-folds aggregate)",
        fontsize=13,
        y=0.995,
    )
    fig.text(
        0.5,
        0.945,
        "Fold 0 was 100% in pre-chain phase 1; folds 1–4 expose mode-dependent asymmetry.",
        ha="center",
        fontsize=10,
        style="italic",
        color="dimgray",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    return fig


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=_WORKTREE_ROOT
        / "outputs/canonical/interpretability/f1_5fold_summary.json",
        help="Path to f1_5fold_summary.json (output of aggregate_step_d_5fold.py).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_WORKTREE_ROOT
        / "outputs/canonical/interpretability/figures/step_d_5fold",
        help="Output directory for the figure files.",
    )
    parser.add_argument(
        "--stem",
        type=str,
        default="fig_step_d_5fold_box_whisker",
        help="File stem (without extension) for PNG and PDF outputs.",
    )
    args = parser.parse_args()

    if not args.summary_json.is_file():
        logger.error("Summary JSON not found: %s", args.summary_json)
        return 1

    summary = json.loads(args.summary_json.read_text())
    fig = make_figure(summary)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.out_dir / f"{args.stem}.png"
    pdf_path = args.out_dir / f"{args.stem}.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", png_path)
    logger.info("Wrote %s", pdf_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
