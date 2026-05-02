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

import matplotlib

matplotlib.use("Agg")  # must precede pyplot import

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


def _load_canonical_pooled_r2(stat_rigor_path: Path) -> float:
    """Load the canonical pooled-bootstrap point R² (single source of truth).

    Reads ``bootstrap_r2_ci.point_r2`` from
    ``outputs/canonical/interpretability/statistical_rigor.json``.
    Falls back to 0.4493 (the historic lab-meeting display value) if the
    file is missing, but logs a warning so stale literals are visible.
    """
    if not stat_rigor_path.exists():
        logger.warning(
            "statistical_rigor.json not found at %s; falling back to 0.4493 "
            "lab-meeting display literal. Re-generate the file to remove this "
            "warning.", stat_rigor_path,
        )
        return 0.4493
    payload = json.loads(stat_rigor_path.read_text())
    return float(payload["bootstrap_r2_ci"]["point_r2"])


def _draw_permutation_null(
    ax: plt.Axes, data: dict, *, canonical_r2: float,
) -> None:
    """Render the permutation null distribution + canonical marker (Panel D).

    Lab-meeting unified-R² choice: ``canonical_r2`` is the pooled bootstrap
    point estimate (loaded from ``statistical_rigor.json``), which matches
    what the scatter in slot 4.1/4.2 shows. The per-fold mean R²=0.4436 is
    < 0.01 R² different — within fold variance — and the pooled value is
    used so a single number runs throughout the deck.

    Tolerates schemas missing ``null_mean_r2_per_perm`` (e.g., the N=50
    canonical aggregate prior to the post-CC1 fix): when the array is
    absent or empty, draws a Gaussian density around ``null_mean`` ± ``null_std``
    instead of a histogram and annotates the panel with a "summary-only"
    notice rather than silently producing an empty bar.
    """
    null_vals_raw = data.get("null_mean_r2_per_perm")
    null_vals = (
        np.asarray(null_vals_raw, dtype=float)
        if null_vals_raw is not None and len(null_vals_raw) > 0
        else np.zeros(0, dtype=float)
    )
    have_array = null_vals.size > 0

    null_mean = float(
        data.get("null_mean", null_vals.mean() if have_array else float("nan"))
    )
    null_std = float(
        data.get("null_std", null_vals.std(ddof=0) if have_array else float("nan"))
    )
    z = float(
        data.get(
            "z_under_null",
            (canonical_r2 - null_mean) / max(null_std, 1e-9)
            if have_array else float("nan"),
        )
    )
    p_one = float(data.get("p_value_one_sided", float("nan")))

    palette = list(PALETTES["categorical"])
    null_color = palette[7]   # tab10 gray
    canonical_color = palette[0]  # tab10 blue
    n_perms = int(data.get("n_permutations", null_vals.size))

    if have_array:
        n_bins = max(5, min(8, len(null_vals)))
        ax.hist(
            null_vals, bins=n_bins,
            color=null_color, edgecolor="white", linewidth=0.6,
            label=f"Permutation null (N={len(null_vals)})",
        )
    else:
        # Schema-fallback path: draw a Gaussian density around (null_mean, null_std)
        # so the panel still conveys the null distribution visually. Annotate
        # so the reader knows raw per-perm draws were not in the source JSON.
        if np.isfinite(null_mean) and np.isfinite(null_std) and null_std > 0:
            x_lo = null_mean - 4.0 * null_std
            x_hi = max(null_mean + 4.0 * null_std, canonical_r2 + 0.05)
            xs = np.linspace(x_lo, x_hi, 200)
            density = (
                np.exp(-0.5 * ((xs - null_mean) / null_std) ** 2)
                / (null_std * np.sqrt(2.0 * np.pi))
            )
            ax.fill_between(
                xs, density, color=null_color, alpha=0.5, edgecolor="white",
                linewidth=0.6,
                label=f"Permutation null (Gaussian, N={n_perms})",
            )
            ax.set_ylim(bottom=0)
        ax.text(
            0.5, 0.4,
            "summary-only\n(raw perm R² not in source)",
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=6, style="italic", color="#666666",
        )
    # Mean +/- std band
    ax.axvline(null_mean, color="#444444", linestyle=":", linewidth=0.8,
               label=f"null mean = {null_mean:.3f}")
    ax.axvspan(null_mean - null_std, null_mean + null_std,
               color="#bbbbbb", alpha=0.3, zorder=0)
    # Canonical marker
    ax.axvline(canonical_r2, color=canonical_color, linestyle="-", linewidth=1.6,
               label=f"canonical $R^2$ = {canonical_r2:.4f}")
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
    statistical_rigor_path: Path,
    figsize: tuple[float, float] = (12.0, 8.0),
) -> Figure:
    """Load all inputs and build the 2x2 composite figure.

    Returns the matplotlib Figure (not yet saved). The canonical R² used
    in Panel D is loaded from ``statistical_rigor_path`` (single source of
    truth, see ``_load_canonical_pooled_r2``).
    """
    consensus = json.loads(Path(consensus_path).read_text())
    wasserstein = json.loads(Path(wasserstein_path).read_text())
    xref = json.loads(Path(xref_path).read_text())
    perm = json.loads(Path(permutation_path).read_text())
    canonical_r2 = _load_canonical_pooled_r2(Path(statistical_rigor_path))

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
            "draw": (
                lambda ax, d=perm, c=canonical_r2:
                _draw_permutation_null(ax, d, canonical_r2=c)
            ),
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
        "--statistical-rigor",
        default="outputs/canonical/interpretability/statistical_rigor.json",
        help=(
            "Canonical pooled-bootstrap point R² source (Panel D). "
            "Reads bootstrap_r2_ci.point_r2 — single source of truth so "
            "the lab-meeting deck stays in sync if the canonical model "
            "retrains."
        ),
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
        statistical_rigor_path=Path(args.statistical_rigor),
        figsize=figsize,
    )
    out_stem = out_dir / "fig_interpretability_capstone"
    written = save_fig(fig, out_stem)
    plt.close(fig)
    logger.info("Wrote %s", [str(p) for p in written])
    return 0


if __name__ == "__main__":
    sys.exit(main())
