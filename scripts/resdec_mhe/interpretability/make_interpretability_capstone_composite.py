"""Capstone interpretability figure: 2x2 composite for the closing slide.

Builds a single 4-panel figure that summarizes the interpretability story:

  - Panel A: 11-method consensus heatmap (CT x method, top-5 ranking).
    Source: ``consensus_heatmap_data.json`` (precomputed by
    ``make_consensus_heatmap_figure.py`` -- ranks dict + top5_counts).
  - Panel B: Pseudobulk Wasserstein-1 top-10 genes for the Splatter cell type
    (horizontal bar chart). Source:
    ``wasserstein_per_celltype_pseudobulk.json``.
  - Panel C: SAE relaxed-interpretable feature distribution (per-feature
    dominant CT count over 323 features). Splatter highlighted in a different
    color to visually emphasize 1/323 = 0.31%. Source:
    ``feature_xref_consensus.json`` -> ``trained.relaxed.per_ct_counts``.
  - Panel D: Permutation null distribution (N=10) with canonical R^2 marker.
    Source: ``permutation_summary.json`` -> ``null_mean_r2_per_perm`` array.

Output: ``outputs/canonical/interpretability/figures/composite/
fig_interpretability_capstone.{png,pdf}`` (default; override with ``--out-dir``).

Usage::

    uv run python scripts/resdec_mhe/interpretability/\
make_interpretability_capstone_composite.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from src.visualization.composite import make_panel
from src.visualization.theme import (
    PALETTES,
    apply_theme,
    fmt_axes,
    save_fig,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Panel draw callbacks
# ----------------------------------------------------------------------
def _draw_consensus_heatmap(ax: plt.Axes, data: dict) -> None:
    """Render the 11-method consensus heatmap (Panel A).

    `data` shape:
      - ``row_cts``: list[str] (cell types as rows)
      - ``methods``: list[str] (method labels as columns)
      - ``ranks``:   dict[ct, dict[method, int]] -- 1-indexed; missing entries
        are treated as rank > 5.
    """
    rows = list(data["row_cts"])
    methods = list(data["methods"])
    n_rows = len(rows)
    n_cols = len(methods)

    # Build NaN-padded rank grid (NaN means "not in top-5 / not present").
    grid = np.full((n_rows, n_cols), np.nan, dtype=float)
    for i, ct in enumerate(rows):
        for j, m in enumerate(methods):
            r = data["ranks"].get(ct, {}).get(m)
            if r is not None:
                grid[i, j] = r

    cmap = plt.get_cmap("viridis")
    rgba = np.ones((n_rows, n_cols, 4), dtype=float)  # white default
    for i in range(n_rows):
        for j in range(n_cols):
            r = grid[i, j]
            if not np.isnan(r) and r <= 5:
                color_val = 0.15 + 0.7 * (r - 1) / 4.0
                rgba[i, j, :] = cmap(color_val)
    ax.imshow(rgba, aspect="auto", interpolation="nearest")

    # Annotate top-5 cells with rank number.
    for i in range(n_rows):
        for j in range(n_cols):
            r = grid[i, j]
            if not np.isnan(r) and r <= 5:
                color_val = 0.15 + 0.7 * (r - 1) / 4.0
                txt_color = "white" if color_val < 0.55 else "black"
                ax.text(
                    j, i, f"{int(r)}",
                    ha="center", va="center",
                    fontsize=6, color=txt_color, fontweight="bold",
                )

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(methods, rotation=55, ha="right", fontsize=6)
    ax.set_yticks(np.arange(n_rows))
    yticklabels = [
        f"$\\mathbf{{{ct.replace(' ', '~')}}}$" if ct.lower() == "splatter" else ct
        for ct in rows
    ]
    ax.set_yticklabels(yticklabels, fontsize=6)

    # Highlight the Splatter row with a red border.
    if any(ct.lower() == "splatter" for ct in rows):
        i_splatter = next(i for i, ct in enumerate(rows) if ct.lower() == "splatter")
        ax.add_patch(mpatches.Rectangle(
            (-0.5, i_splatter - 0.5),
            n_cols, 1.0,
            fill=False, edgecolor="#d62728", linewidth=1.6, zorder=5,
        ))

    # Minor-tick grid for cell separation.
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="#cccccc", linewidth=0.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    fmt_axes(ax, hide_spines=(), grid_major=False, grid_minor=False)
    ax.set_xlabel("Method")


def _draw_splatter_wasserstein_top_genes(ax: plt.Axes, data: dict) -> None:
    """Render Splatter top-10 Wasserstein-1 genes as horizontal bars (Panel B)."""
    splatter_block = next(
        (ct for ct in data["per_cell_type"] if ct["cell_type"].lower() == "splatter"),
        None,
    )
    if splatter_block is None:
        ax.text(0.5, 0.5, "no Splatter row in Wasserstein JSON",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    pairs = splatter_block["wasserstein_per_gene_top10"]
    genes = [p[0] for p in pairs]
    values = [float(p[1]) for p in pairs]
    # Sort ascending (smallest at bottom; largest at top after barh).
    order = np.argsort(values)
    genes = [genes[i] for i in order]
    values = [values[i] for i in order]

    palette = list(PALETTES["categorical"])
    splatter_color = "#d62728"  # match red highlight elsewhere
    y = np.arange(len(genes))
    ax.barh(y, values, color=splatter_color, edgecolor="white", linewidth=0.4,
            height=0.78)
    ax.set_yticks(y)
    ax.set_yticklabels(genes, fontsize=7)
    ax.set_xlabel("Wasserstein-1 (resilient vs vulnerable, n=129/129)")
    fmt_axes(ax)
    # Mean line
    mean_val = float(splatter_block["wasserstein_per_gene_mean"])
    ax.axvline(mean_val, color="#444444", linestyle="--", linewidth=0.8,
               label=f"CT mean = {mean_val:.3f}")
    ax.legend(loc="lower right", fontsize=6, frameon=True)


def _draw_sae_feature_distribution(ax: plt.Axes, data: dict) -> None:
    """Render the SAE relaxed-interpretable feature CT distribution (Panel C).

    ``data`` is the parsed ``feature_xref_consensus.json``.  Reads
    ``trained.relaxed.per_ct_counts`` and renders a sorted bar chart with
    Splatter highlighted in red.
    """
    counts = data["trained"]["relaxed"]["per_ct_counts"]
    n_total = data["trained"]["relaxed"]["n_features"]
    items = sorted(counts.items(), key=lambda kv: -kv[1])
    cts = [k for k, _ in items]
    vals = [v for _, v in items]

    palette = list(PALETTES["categorical"])
    bar_color = palette[2]  # green-ish from tab10
    splatter_color = "#d62728"
    colors = [splatter_color if c.lower() == "splatter" else bar_color for c in cts]

    x = np.arange(len(cts))
    ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.4, width=0.85)

    # Annotate Splatter
    if "Splatter" in counts:
        i_sp = cts.index("Splatter")
        v_sp = vals[i_sp]
        # Place annotation away from the bar to avoid clipping.
        max_val = max(vals) if vals else 1
        ax.annotate(
            f"Splatter: {v_sp}/{n_total} ({100.0 * v_sp / max(n_total, 1):.1f}%)",
            xy=(i_sp, v_sp),
            xytext=(i_sp + 4.0, max_val * 0.55),
            fontsize=6, color=splatter_color,
            arrowprops=dict(
                arrowstyle="->", color=splatter_color, linewidth=0.6,
                shrinkA=2, shrinkB=2,
            ),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(cts, rotation=70, ha="right", fontsize=5)
    ax.set_ylabel(f"# SAE features (top-1 CT, n={n_total})")
    fmt_axes(ax)


def _draw_permutation_null(ax: plt.Axes, data: dict) -> None:
    """Render the permutation null distribution + canonical marker (Panel D)."""
    null_vals = np.asarray(data["null_mean_r2_per_perm"], dtype=float)
    canonical = float(data["canonical_mean_r2"])
    null_mean = float(data.get("null_mean", null_vals.mean()))
    null_std = float(data.get("null_std", null_vals.std(ddof=0)))
    z = float(data.get("z_under_null", (canonical - null_mean) / max(null_std, 1e-9)))
    p_one = float(data.get("p_value_one_sided", float("nan")))

    palette = list(PALETTES["categorical"])
    null_color = palette[7]   # tab10 gray
    canonical_color = palette[0]  # tab10 blue

    n_bins = max(5, min(8, len(null_vals)))
    ax.hist(
        null_vals, bins=n_bins,
        color=null_color, edgecolor="white", linewidth=0.6,
        label=f"Permutation null (N={len(null_vals)})",
    )
    # Mean +/- std band
    ax.axvline(null_mean, color="#444444", linestyle=":", linewidth=0.8,
               label=f"null mean = {null_mean:.3f}")
    ax.axvspan(null_mean - null_std, null_mean + null_std,
               color="#bbbbbb", alpha=0.3, zorder=0)
    # Canonical marker
    ax.axvline(canonical, color=canonical_color, linestyle="-", linewidth=1.6,
               label=f"canonical $R^2$ = {canonical:.4f}")
    # Annotation text in the corner with z + p
    ax.text(
        0.98, 0.98,
        f"$z$ = {z:.2f}\n$p_{{1\\text{{-sided}}}}$ = {p_one:.4f}",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=7,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                  edgecolor="#cccccc"),
    )

    ax.set_xlabel("$R^2$ (val, mean over folds)")
    ax.set_ylabel("Count")
    ax.legend(loc="upper left", fontsize=6, frameon=True)
    fmt_axes(ax)


# ----------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------
def build_figure(
    *,
    consensus_path: Path,
    wasserstein_path: Path,
    xref_path: Path,
    permutation_path: Path,
    figsize: tuple[float, float] = (12.0, 8.0),
) -> Figure:
    """Load all inputs and build the 2x2 composite figure.

    Returns the matplotlib Figure (not yet saved).
    """
    consensus = json.loads(Path(consensus_path).read_text())
    wasserstein = json.loads(Path(wasserstein_path).read_text())
    xref = json.loads(Path(xref_path).read_text())
    perm = json.loads(Path(permutation_path).read_text())

    panels = [
        {
            "draw": (lambda ax, d=consensus: _draw_consensus_heatmap(ax, d)),
            "title": "11-method consensus (top-5 ranks per CT)",
        },
        {
            "draw": (lambda ax, d=wasserstein: _draw_splatter_wasserstein_top_genes(ax, d)),
            "title": "Splatter pseudobulk Wasserstein-1: top-10 genes",
        },
        {
            "draw": (lambda ax, d=xref: _draw_sae_feature_distribution(ax, d)),
            "title": "SAE relaxed-interpretable feature CT distribution",
        },
        {
            "draw": (lambda ax, d=perm: _draw_permutation_null(ax, d)),
            "title": "Permutation null distribution (full-pipeline shuffle)",
        },
    ]
    fig = make_panel(
        panels,
        layout=(2, 2),
        figsize=figsize,
        wspace=0.45,
        hspace=0.65,
    )
    return fig


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--consensus-data",
        default="outputs/canonical/interpretability/figures/consensus_heatmap/"
                "consensus_heatmap_data.json",
        help="Pre-computed consensus heatmap rank matrix (Panel A).",
    )
    p.add_argument(
        "--wasserstein-json",
        default="outputs/canonical/interpretability/distributional_resilience/"
                "wasserstein_per_celltype_pseudobulk.json",
        help="Pseudobulk Wasserstein-1 per-CT top-genes (Panel B).",
    )
    p.add_argument(
        "--xref-json",
        default="outputs/canonical/sae/feature_xref_consensus.json",
        help="SAE feature cross-reference summary (Panel C).",
    )
    p.add_argument(
        "--permutation-summary",
        default="outputs/canonical/permutation_test/permutation_summary.json",
        help="Permutation null summary (Panel D).",
    )
    p.add_argument(
        "--out-dir",
        default="outputs/canonical/interpretability/figures/composite",
        help="Output directory; figure stem is fig_interpretability_capstone.",
    )
    p.add_argument(
        "--figsize",
        default="12,8",
        help="Comma-separated W,H in inches (default 12,8 ~ 16:9 talk slide).",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    apply_theme()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    w_str, h_str = args.figsize.split(",")
    figsize = (float(w_str), float(h_str))

    fig = build_figure(
        consensus_path=Path(args.consensus_data),
        wasserstein_path=Path(args.wasserstein_json),
        xref_path=Path(args.xref_json),
        permutation_path=Path(args.permutation_summary),
        figsize=figsize,
    )
    out_stem = out_dir / "fig_interpretability_capstone"
    written = save_fig(fig, out_stem)
    plt.close(fig)
    logger.info("Wrote %s", [str(p) for p in written])
    return 0


if __name__ == "__main__":
    sys.exit(main())
