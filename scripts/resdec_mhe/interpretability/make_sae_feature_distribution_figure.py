"""Figure 4: SAE feature-level CT distribution (two-panel composite).

Panel A: Histogram of all 2048 features' top-1 dominant CT, sorted by count.
Panel B: Same, filtered to relaxed-interpretable features (n=323).

Relaxed-interpretable filter (matches `feature_xref_consensus.json`):
  - feature is non-dead (no "dead" flag)
  - mw_p_cognition < 0.05
  - fraction_active in [1e-4, 0.5]

Splatter bar is highlighted in both panels with annotation showing its
absolute count.

Title: "SAE feature-level CT distribution: model uses distributed
representation (Splatter dominates only 1/323 interpretable features)."
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from src.visualization.theme import (
    PALETTES,
    apply_theme,
    fmt_axes,
    save_fig,
)

logger = logging.getLogger(__name__)


def _compute_top1_counts(
    feature_report: list[dict],
    relaxed: bool,
) -> Counter:
    """Return Counter of top-1 dominant CT per feature, optionally relaxed-filtered."""
    out: Counter = Counter()
    for f in feature_report:
        if relaxed:
            if "dead" in f.get("flags", []):
                continue
            mw = f.get("mw_p_cognition")
            if mw is None or mw >= 0.05:
                continue
            frac = f.get("fraction_active", 0.0)
            if not (1e-4 <= frac <= 0.5):
                continue
        if not f.get("top_cell_types"):
            continue
        out[f["top_cell_types"][0]["cell_type"]] += 1
    return out


def _draw_hist_panel(
    ax,
    counts: Counter,
    *,
    splatter_color: str = "#d62728",
    bar_color: str = "#1f77b4",
    label_n: int | None = None,
) -> None:
    """Render a sorted-bar histogram with Splatter highlighted."""
    items = sorted(counts.items(), key=lambda kv: -kv[1])
    cts = [kv[0] for kv in items]
    vals = [kv[1] for kv in items]

    colors = [splatter_color if c.lower() == "splatter" else bar_color for c in cts]
    x = np.arange(len(cts))
    ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.4, width=0.85)

    # Annotate Splatter bar
    if "Splatter" in counts:
        i_splatter = cts.index("Splatter")
        v = vals[i_splatter]
        ax.annotate(
            f"Splatter: {v}",
            xy=(i_splatter, v),
            xytext=(i_splatter + 4.0, max(vals) * 0.55),
            fontsize=7, color=splatter_color,
            arrowprops=dict(
                arrowstyle="->", color=splatter_color, linewidth=0.6,
                shrinkA=2, shrinkB=2,
            ),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(cts, rotation=70, ha="right", fontsize=6)
    ax.set_ylabel("# features (top-1 CT)")

    if label_n is not None:
        ax.text(
            0.99, 0.95,
            f"N={label_n}",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=8, color="#333333",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#cccccc"),
        )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--feature-report",
        default="outputs/canonical/sae/batch_topk/fused/exp32_k64_seed0/feature_report.json",
    )
    p.add_argument(
        "--xref-json",
        default="outputs/canonical/sae/feature_xref_consensus.json",
        help="Cross-reference summary; used only to verify N totals.",
    )
    p.add_argument(
        "--out-dir",
        default="outputs/canonical/interpretability/figures/sae_feature_dist",
    )
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    apply_theme()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    feature_report = json.loads(Path(args.feature_report).read_text())
    all_counts = _compute_top1_counts(feature_report, relaxed=False)
    relaxed_counts = _compute_top1_counts(feature_report, relaxed=True)

    # Sanity check vs xref_consensus when available
    if Path(args.xref_json).exists():
        xref = json.loads(Path(args.xref_json).read_text())
        n_relaxed_expected = xref["trained"]["relaxed"]["n_features"]
        if sum(relaxed_counts.values()) != n_relaxed_expected:
            logger.warning(
                "Relaxed feature count differs from xref expected: %d vs %d",
                sum(relaxed_counts.values()), n_relaxed_expected,
            )

    n_all = sum(all_counts.values())
    n_relaxed = sum(relaxed_counts.values())
    splatter_all = all_counts.get("Splatter", 0)
    splatter_relaxed = relaxed_counts.get("Splatter", 0)

    palette = list(PALETTES["categorical"])

    def _draw_a(ax):
        _draw_hist_panel(ax, all_counts, bar_color=palette[0], label_n=n_all)
        ax.set_title(
            f"All features (n={n_all}, Splatter={splatter_all})",
            fontsize=8.5,
        )

    def _draw_b(ax):
        _draw_hist_panel(ax, relaxed_counts, bar_color=palette[2], label_n=n_relaxed)
        ax.set_title(
            f"Relaxed-interpretable (n={n_relaxed}, Splatter={splatter_relaxed})",
            fontsize=8.5,
        )

    # Build the two-panel figure manually (rather than via make_panel) so we
    # can fine-tune the inter-panel spacing for rotated x-tick labels and a
    # multi-line suptitle. Panels share theme styling via fmt_axes inside
    # _draw_hist_panel.
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 9.5))
    _draw_a(axes[0])
    fmt_axes(axes[0])
    axes[0].text(-0.06, 1.10, "A", transform=axes[0].transAxes,
                 fontsize=10, fontweight="bold", va="bottom", ha="left")
    _draw_b(axes[1])
    fmt_axes(axes[1])
    axes[1].text(-0.06, 1.10, "B", transform=axes[1].transAxes,
                 fontsize=10, fontweight="bold", va="bottom", ha="left")

    # Manual layout: leave room at top for suptitle, stretch hspace so rotated
    # x-tick labels of panel A don't collide with panel B's title.
    fig.subplots_adjust(top=0.90, bottom=0.13, hspace=0.85,
                        left=0.10, right=0.97)
    fig.suptitle(
        "SAE feature-level CT distribution: model uses distributed representation\n"
        f"(Splatter dominates only {splatter_relaxed}/{n_relaxed} interpretable features)",
        fontsize=10, y=0.965,
    )
    save_fig(fig, out_dir / "sae_feature_dist")
    plt.close(fig)

    # Persist data summary
    summary = {
        "n_total_features": n_all,
        "n_relaxed_features": n_relaxed,
        "splatter_all": splatter_all,
        "splatter_relaxed": splatter_relaxed,
        "all_counts": dict(all_counts),
        "relaxed_counts": dict(relaxed_counts),
    }
    (out_dir / "sae_feature_dist_data.json").write_text(json.dumps(summary, indent=2))

    elapsed = time.perf_counter() - t0
    logger.info(
        "Rendered sae_feature_dist.{png,pdf} in %.2fs (n_all=%d, n_relaxed=%d)",
        elapsed, n_all, n_relaxed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
