#!/usr/bin/env python
"""Render Figure 3 for the ResDec-MHE README revamp -- 12-panel methods grid.

Replaces the previous 2-panel ``fig3_consensus_causal.png`` with a tall
12-panel grid (2 rows x 6 cols, ~20x16 inches) that gives each
interpretability method its own native visualization rather than collapsing
all of them into a single binary heatmap.

Panel inventory
---------------
  Row 1 (gradient + attention methods)
    1.  IG sunburst                    (CT inner ring x gene outer ring)
    2.  GradientSHAP sunburst          (same schema)
    3.  SmoothGrad sunburst            (same schema)
    4.  AttnLRP radial violin          (31 CTs x 516-subj distributions)
    5.  GMAR radial violin             (same)
    6.  GAF AF radial violin           (same)

  Row 2 (causal + statistical methods + consensus strip)
    7.  GAF AGF radial violin
    8.  GAF GF radial violin
    9.  Wasserstein-1 ridge plot       (top-5 (CT, gene) pairs)
    10. CMI slope chart                (uncond MI -> cond MI per CT)
    11. LOCO tornado                   (signed deltaR2 horizontal bars)
    12. Consensus strip                (top-5 frequency dot strip)

CT identity colors are deterministic (alphabetical order -> tab20 + tab20b
extension) so the same CT lights up the same color across all panels.

Inputs
------
    outputs/canonical/interpretability/captum_ig/composite_attribution_summary.json
    outputs/canonical/interpretability/captum_robustness/{gradientshap,smoothgrad}_attribution_summary.json
    outputs/canonical/interpretability/attention_attribution/per_subject_attribution.npz
    outputs/canonical/interpretability/distributional_resilience/wasserstein_per_celltype_pseudobulk.json
    outputs/canonical/interpretability/conditional_mi_per_celltype_raw_max.json
    outputs/canonical/interpretability/loco_zero_out/loco_per_celltype.json
    outputs/canonical/interpretability/figures/consensus_heatmap/consensus_heatmap_data.json
    outputs/canonical/interpretability/ct_coverage_full_cohort.json
    outputs/canonical/interpretability/residual_per_subject.csv  (used to split
        resilient/vulnerable subjects for the W1 ridge plot - matches the
        producer convention in run_distributional_resilience.py)
    data/precomputed/{subject_id}.pt        (per-subject 31x4785 pseudobulks)
    data/precomputed/gene_names.npy

Outputs
-------
    figures/fig3_methods_grid.png
    Verification numbers printed to stdout.

Idempotence
-----------
All randomness is seeded (np.random.default_rng(42), PYTHONHASHSEED=42). No
sampling, no model inference -- pure JSON / .pt / .npz I/O + numpy. Re-running
should produce a bit-identical PNG.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Set PYTHONHASHSEED defensively so any color-selection paths that touch
# Python set iteration order are stable across reruns.
os.environ.setdefault("PYTHONHASHSEED", "42")

import matplotlib

matplotlib.use("Agg")  # must precede pyplot import

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import to_hex
from matplotlib.patches import Wedge

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from src.analysis.pseudobulk_io import load_pseudobulk_matrix  # noqa: E402
from src.visualization.theme import (  # noqa: E402
    PALETTES,
    apply_theme,
    fmt_axes,
    save_fig,
)

logger = logging.getLogger(__name__)


# Alphabetical CT order, used as the canonical "stable" ordering for the
# circle in radial panels and the deterministic color mapping.
def _alpha_ct_order(ct_coverage: dict) -> list[str]:
    return sorted(ct_coverage["per_ct"].keys())


def _build_ct_color_map(ct_order: list[str]) -> dict[str, str]:
    """Stable hex color per CT.

    31 CTs are colored using tab20 (20) extended with tab20b (11 more) so
    every CT has its own perceptually-distinct hue. The mapping is
    deterministic in the input ordering (alphabetical) so the same CT
    receives the same color across all panels.
    """
    base = list(plt.get_cmap("tab20").colors)        # 20 colors
    extra = list(plt.get_cmap("tab20b").colors)      # 20 colors
    palette = (base + extra)[: max(len(ct_order), 31)]
    return {ct: to_hex(palette[i]) for i, ct in enumerate(ct_order)}


# Method order matches the consensus heatmap convention.
SUNBURST_METHODS = ["IG", "GradientSHAP", "SmoothGrad"]
RADIAL_METHODS = ["AttnLRP", "GMAR", "GAF AF", "GAF AGF", "GAF GF"]
RADIAL_METHODS_ROW1 = ["AttnLRP", "GMAR", "GAF AF"]
RADIAL_METHODS_ROW2 = ["GAF AGF", "GAF GF"]
ATTN_KEY = {
    "AttnLRP": "attnlrp",
    "GMAR": "gmar",
    "GAF AF": "gaf_af",
    "GAF AGF": "gaf_agf",
    "GAF GF": "gaf_gf",
}


# -----------------------------------------------------------------------------
# Panels 1-3: sunburst (gradient-attribution methods)
# -----------------------------------------------------------------------------
def _draw_sunburst(
    ax: plt.Axes,
    summary: dict,
    *,
    ct_color_map: dict[str, str],
    method_label: str,
    top_k_cts: int = 10,
    top_n_genes: int = 4,
) -> tuple[list[str], list[float]]:
    """Render a two-ring sunburst for a captum-style method JSON.

    Inner ring sectors = top-K CTs by total_abs_attribution (size ~
    cell_types_ranked_by_total_attribution).
    Outer ring sectors = top-N genes within each CT, with size ~
    mean_abs_attribution.

    Sector ANGLES on the inner ring are proportional to the CT's total
    attribution. Outer ring sectors inherit the CT's angular range and are
    sub-divided proportional to per-gene mean_abs_attribution. Coloring:
    inner ring colored by CT (from ``ct_color_map``); outer ring shares
    the parent CT's color but with reduced alpha so the inner / outer
    rings are visually distinguishable.

    Returns
    -------
    top_cts : list[str]
        The top_k_cts CT names actually rendered (for verification logs).
    top_ct_values : list[float]
        Their total_abs_attribution values (for verification logs).
    """
    # Get top-K CTs by total_abs_attribution.
    ranked = summary["cell_types_ranked_by_total_attribution"]
    ranked_sorted = sorted(
        ranked, key=lambda d: -float(d["total_abs_attribution"])
    )[:top_k_cts]
    top_cts = [d["cell_type"] for d in ranked_sorted]
    top_ct_values = [float(d["total_abs_attribution"]) for d in ranked_sorted]

    total_ct_attr = sum(top_ct_values)
    if total_ct_attr <= 0.0:
        ax.text(0.5, 0.5, f"{method_label}: empty", ha="center", va="center",
                transform=ax.transAxes)
        return top_cts, top_ct_values

    # Inner ring: r in [0.0, 0.55]. Outer ring: r in [0.55, 1.0].
    R_INNER = 0.55
    R_OUTER = 1.00

    # Place sector angles starting from 90 degrees, going clockwise (so the
    # largest sector starts at top, mirroring conventional sunburst layout).
    theta_start = 90.0
    for ct, value in zip(top_cts, top_ct_values):
        sweep = 360.0 * (value / total_ct_attr)
        theta_end = theta_start - sweep   # clockwise

        ct_color = ct_color_map.get(ct, "#888888")
        # Inner ring sector for this CT.
        wedge = Wedge(
            (0.0, 0.0),
            R_INNER,
            theta_end, theta_start,        # matplotlib uses CCW; pass (smaller, larger)
            width=R_INNER,
            facecolor=ct_color,
            edgecolor="white",
            linewidth=0.6,
            zorder=2,
        )
        ax.add_patch(wedge)

        # Outer ring: top-N genes for this CT (from top_genes_per_cell_type).
        gene_list = summary.get("top_genes_per_cell_type", {}).get(ct, [])
        gene_list = gene_list[:top_n_genes]
        if not gene_list:
            theta_start = theta_end
            continue
        gene_total = sum(float(g["mean_abs_attribution"]) for g in gene_list)
        if gene_total <= 0.0:
            theta_start = theta_end
            continue

        gene_theta_start = theta_start
        for g in gene_list:
            g_val = float(g["mean_abs_attribution"])
            g_sweep = sweep * (g_val / gene_total)
            g_theta_end = gene_theta_start - g_sweep
            outer_wedge = Wedge(
                (0.0, 0.0),
                R_OUTER,
                g_theta_end, gene_theta_start,
                width=R_OUTER - R_INNER,
                facecolor=ct_color,
                edgecolor="white",
                linewidth=0.4,
                alpha=0.55,
                zorder=2,
            )
            ax.add_patch(outer_wedge)

            # Add a small gene label only if the wedge is wide enough (avoids
            # overlap on thin outer slices).
            if g_sweep >= 12.0:
                mid_angle_deg = (gene_theta_start + g_theta_end) / 2.0
                mid_angle = np.deg2rad(mid_angle_deg)
                lr = (R_INNER + R_OUTER) / 2.0
                tx = lr * np.cos(mid_angle)
                ty = lr * np.sin(mid_angle)
                # Rotate label tangentially for readability.
                rot = mid_angle_deg - 90.0
                if mid_angle_deg < -90.0 or mid_angle_deg > 90.0:
                    rot += 180.0
                ax.text(
                    tx, ty,
                    g["gene"],
                    ha="center", va="center",
                    fontsize=4.5, rotation=rot, color="#222222",
                    zorder=3,
                )

            gene_theta_start = g_theta_end

        theta_start = theta_end

    # Title + ring annotations.
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(method_label, fontsize=8, pad=3.0)
    return top_cts, top_ct_values


# -----------------------------------------------------------------------------
# Panels 4-8: radial violin (attention methods)
# -----------------------------------------------------------------------------
def _draw_radial_violin(
    ax: plt.Axes,
    method_label: str,
    method_arr: np.ndarray,        # shape (516, 31)
    ct_names: list[str],            # 31 entries; index aligned to method_arr columns
    *,
    ct_coverage: dict,
    ct_color_map: dict[str, str],
) -> tuple[list[str], list[float]]:
    """Render a polar plot with one half-violin per CT around a circle.

    Each spoke shows the per-subject distribution of ``method_arr[:, k]``.
    Spoke radius spans [r_min, r_max] of the per-CT data; the violin shape
    is generated from a kernel density estimate of the 516 subjects'
    values for that CT, normalized to a fixed angular width so spokes
    don't overlap.

    Coloring: spoke / violin face is the CT's deterministic color (so the
    same CT lights up the same color across all 5 attention panels);
    border is darker if ``well_covered`` else light gray.

    Returns top-5 CTs by mean(|attribution|) for verification.
    """
    n_cts = len(ct_names)
    if method_arr.shape[1] != n_cts:
        raise ValueError(
            f"radial violin: method_arr has {method_arr.shape[1]} columns "
            f"but ct_names has {n_cts}"
        )
    # Use absolute values (since attribution can be signed for some methods)
    # for the radial magnitude; this matches the consensus heatmap's
    # ranking-by-mean_importance convention.
    arr_abs = np.abs(method_arr)

    # Per-CT statistics. Use median for the dot reference (robust to outliers)
    # and {min, max} for the radial extent.
    per_ct_min = arr_abs.min(axis=0)
    per_ct_max = arr_abs.max(axis=0)
    per_ct_median = np.median(arr_abs, axis=0)
    per_ct_mean = arr_abs.mean(axis=0)

    # Angular layout: one spoke per CT.
    angles = np.linspace(0.0, 2.0 * np.pi, n_cts, endpoint=False)
    half_width = (2.0 * np.pi / n_cts) * 0.40   # angular half-width per violin

    # Normalize radii for visual clarity. We want all spokes to span [0.10, 1.0]
    # of the panel radius regardless of the absolute scale of the method's
    # attribution (different methods have very different magnitudes; e.g. AGF
    # is in 1e-3 range, AttnLRP in 1e-2). The dot at the per-CT mean uses the
    # same per-CT normalization for consistency.
    R_MIN = 0.10
    R_MAX = 1.00
    eps = 1e-12
    per_ct_lo = per_ct_min
    per_ct_hi = per_ct_max
    per_ct_span = np.maximum(per_ct_hi - per_ct_lo, eps)

    def _norm_r(values: np.ndarray, k: int) -> np.ndarray:
        return R_MIN + (values - per_ct_lo[k]) / per_ct_span[k] * (R_MAX - R_MIN)

    # Render each CT's violin shape (half violin, mirrored).
    for k in range(n_cts):
        ct = ct_names[k]
        well_covered = bool(
            ct_coverage["per_ct"].get(ct, {}).get("well_covered", False)
        )
        face = ct_color_map.get(ct, "#888888")
        edge = "#333333" if well_covered else "#cccccc"

        vals = arr_abs[:, k]
        # Kernel density estimate for the violin shape (Silverman bandwidth).
        # Skip degenerate columns (constant or all-zero).
        if vals.size < 2 or per_ct_span[k] < eps:
            continue

        from scipy.stats import gaussian_kde
        try:
            kde = gaussian_kde(vals)
        except (np.linalg.LinAlgError, ValueError):
            continue
        rs_vals = np.linspace(vals.min(), vals.max(), 60)
        density = kde(rs_vals)
        if density.max() <= 0.0:
            continue
        density = density / density.max() * half_width    # angular envelope

        # Convert radial values to plot radii (normalized [R_MIN, R_MAX]).
        rs = _norm_r(rs_vals, k)
        # Build the half-violin polygon: forward along rising edge, return
        # along the spoke axis.
        thetas_right = angles[k] + density
        thetas_left = angles[k] - density
        poly_thetas = np.concatenate([thetas_right, thetas_left[::-1]])
        poly_rs = np.concatenate([rs, rs[::-1]])
        ax.fill(
            poly_thetas, poly_rs,
            facecolor=face,
            edgecolor=edge,
            linewidth=0.4,
            alpha=0.85,
            zorder=3,
        )

        # Per-CT mean dot (for visual reference). Use _norm_r at per_ct_mean[k].
        mean_r = _norm_r(np.array([per_ct_mean[k]]), k)[0]
        ax.plot(
            angles[k], mean_r,
            marker="o", markersize=2.0,
            color="white", markeredgecolor="black",
            markeredgewidth=0.4, zorder=4,
        )
        # Faint median spoke for reference.
        median_r = _norm_r(np.array([per_ct_median[k]]), k)[0]
        ax.plot(
            [angles[k], angles[k]],
            [R_MIN * 0.5, median_r],
            color="#888888", linewidth=0.4, alpha=0.6, zorder=2,
        )

    # Polar styling: hide angular tick labels (too noisy for 31 CTs); keep
    # only a faint spoke at each CT angle.
    ax.set_xticks(angles)
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_ylim(0.0, R_MAX * 1.12)
    ax.set_title(method_label, fontsize=8, pad=4.0)
    ax.tick_params(pad=0.0)
    # Keep grid faint
    ax.grid(True, color="#e6e6e6", linewidth=0.3, alpha=0.7)
    ax.spines["polar"].set_linewidth(0.5)
    ax.spines["polar"].set_color("#888888")

    # Annotate the CT with the highest mean(|x|) on the rim (one label per
    # panel keeps the radial plot uncluttered).
    top_idx_full = np.argsort(-per_ct_mean)
    top5_cts = [ct_names[i] for i in top_idx_full[:5]]
    top5_values = [float(per_ct_mean[i]) for i in top_idx_full[:5]]

    # Place the #1 CT label at its angle, just outside the rim.
    top_k = int(top_idx_full[0])
    ax.text(
        angles[top_k], R_MAX * 1.10,
        ct_names[top_k],
        ha="center", va="center",
        fontsize=5.5, color=ct_color_map.get(ct_names[top_k], "#333333"),
        fontweight="bold", zorder=5,
    )
    return top5_cts, top5_values


# -----------------------------------------------------------------------------
# Panel 9: Wasserstein-1 ridge plot
# -----------------------------------------------------------------------------
def _identify_top_w1_pairs(w1_payload: dict, top_n: int = 5) -> list[tuple[str, str, float]]:
    """Pool wasserstein_per_gene_top10 across all 31 CTs and return top-N (CT, gene, W1)."""
    all_pairs = []
    for ct_entry in w1_payload["per_cell_type"]:
        ct = ct_entry["cell_type"]
        for gene, dist in ct_entry["wasserstein_per_gene_top10"]:
            all_pairs.append((ct, str(gene), float(dist)))
    all_pairs.sort(key=lambda t: -t[2])
    return all_pairs[:top_n]


def _split_resilient_vulnerable(residual_csv: Path, q_frac: float = 0.25) -> tuple[list[str], list[str]]:
    """Replicate run_distributional_resilience.py's split convention.

    residual = target - prediction (already pre-computed per-subject in
    residual_per_subject.csv). Top quartile of residual = resilient,
    bottom quartile = vulnerable. Returns
    (resilient_subject_ids, vulnerable_subject_ids).
    """
    df = pd.read_csv(residual_csv)
    id_col = (
        "ROSMAP_IndividualID"
        if "ROSMAP_IndividualID" in df.columns
        else df.columns[0]
    )
    df = df.rename(columns={id_col: "subject_id"})
    finite = np.isfinite(df["residual"])
    q_lo = df.loc[finite, "residual"].quantile(q_frac)
    q_hi = df.loc[finite, "residual"].quantile(1.0 - q_frac)
    resilient_ids = df.loc[df["residual"] >= q_hi, "subject_id"].astype(str).tolist()
    vulnerable_ids = df.loc[df["residual"] <= q_lo, "subject_id"].astype(str).tolist()
    return resilient_ids, vulnerable_ids


def _extract_pseudobulk_for_pair(
    precomputed_dir: Path,
    subject_ids: list[str],
    ct_idx: int,
    gene_idx: int,
) -> np.ndarray:
    """Return per-subject (n,) array of pseudobulk[ct_idx, gene_idx]."""
    pb = load_pseudobulk_matrix(precomputed_dir, subject_ids, n_jobs=1)
    return pb[:, ct_idx, gene_idx]


def _draw_w1_ridge(
    ax: plt.Axes,
    pairs: list[tuple[str, str, float]],
    res_vals_per_pair: list[np.ndarray],     # length-N list of (n_res,) arrays
    vul_vals_per_pair: list[np.ndarray],     # length-N list of (n_vul,) arrays
    *,
    ct_color_map: dict[str, str],
) -> None:
    """Render top-N (CT, gene) pairs as overlaid resilient/vulnerable ridges.

    Each pair gets its own horizontal lane; within a lane, two KDEs are
    overlaid (resilient blue / vulnerable red) and the W1 distance is
    annotated on the right side. Lane separation is achieved by adding
    a per-lane y-offset.

    Resilient color is a slightly darker variant of tab10 blue; vulnerable
    is the standard tab10 red (matches BASELINE_COLORS palette).
    """
    from scipy.stats import gaussian_kde

    n = len(pairs)
    res_color = "#1f77b4"   # tab10 blue
    vul_color = "#d62728"   # tab10 red
    LANE_HEIGHT = 1.0

    for i, ((ct, gene, w1), res_vals, vul_vals) in enumerate(
        zip(pairs, res_vals_per_pair, vul_vals_per_pair)
    ):
        y0 = (n - 1 - i) * LANE_HEIGHT      # top-most lane is the strongest pair

        all_vals = np.concatenate([res_vals, vul_vals])
        if all_vals.size < 2:
            continue
        x_min = float(np.min(all_vals))
        x_max = float(np.max(all_vals))
        span = max(x_max - x_min, 1e-9)
        x_pad = 0.05 * span
        xs = np.linspace(x_min - x_pad, x_max + x_pad, 200)

        # Resilient KDE.
        try:
            kde_r = gaussian_kde(res_vals)
            yr = kde_r(xs)
        except (np.linalg.LinAlgError, ValueError):
            yr = np.zeros_like(xs)
        # Vulnerable KDE.
        try:
            kde_v = gaussian_kde(vul_vals)
            yv = kde_v(xs)
        except (np.linalg.LinAlgError, ValueError):
            yv = np.zeros_like(xs)
        # Normalize each ridge to a fixed height (so visual comparison isn't
        # dominated by KDE bandwidth differences).
        peak = max(yr.max(), yv.max(), 1e-9)
        yr_n = yr / peak * 0.85
        yv_n = yv / peak * 0.85

        ax.fill_between(xs, y0, y0 + yr_n, color=res_color, alpha=0.45,
                        linewidth=0.0, zorder=2)
        ax.fill_between(xs, y0, y0 + yv_n, color=vul_color, alpha=0.45,
                        linewidth=0.0, zorder=3)
        ax.plot(xs, y0 + yr_n, color=res_color, linewidth=0.7, zorder=4)
        ax.plot(xs, y0 + yv_n, color=vul_color, linewidth=0.7, zorder=5)

        # Lane label on left side: "{CT} : {gene}".
        ax.text(
            x_min - x_pad - 0.02 * span, y0 + 0.5,
            f"{ct} : {gene}",
            ha="right", va="center",
            fontsize=5.5,
            color=ct_color_map.get(ct, "#333333"),
            fontweight="bold",
        )
        # W1 annotation on right side.
        ax.text(
            x_max + x_pad + 0.02 * span, y0 + 0.5,
            r"$W_1$=" + f"{w1:.3f}",
            ha="left", va="center",
            fontsize=6, color="#333333",
        )

    ax.set_ylim(-0.2, n * LANE_HEIGHT + 0.1)
    ax.set_yticks([])
    ax.set_xlabel("Pseudobulk expression", fontsize=7)
    ax.set_title(
        f"$W_1$ top-{n} (resilient n={len(res_vals_per_pair[0])} vs "
        f"vulnerable n={len(vul_vals_per_pair[0])})",
        fontsize=8,
    )
    fmt_axes(ax)
    # Legend.
    res_patch = mpatches.Patch(facecolor=res_color, alpha=0.45, label="Resilient")
    vul_patch = mpatches.Patch(facecolor=vul_color, alpha=0.45, label="Vulnerable")
    ax.legend(handles=[res_patch, vul_patch], loc="upper right",
              fontsize=6, frameon=True)


# -----------------------------------------------------------------------------
# Panel 10: CMI slope chart
# -----------------------------------------------------------------------------
def _draw_cmi_slope(
    ax: plt.Axes,
    cmi_payload: dict,
    *,
    ct_color_map: dict[str, str],
) -> tuple[list[str], list[float]]:
    """Slope chart: unconditional MI -> conditional MI per CT.

    CTs sorted by ``conditional_mi_given_pathology`` descending. Color slope
    by sign of delta = unconditional - conditional:
      delta > 0: pathology-conditioning DECREASES MI (red, "pathology contains info")
      delta < 0: pathology-conditioning INCREASES MI (blue, "pathology-orthogonal")
    """
    entries = list(cmi_payload["per_cell_type"])
    entries.sort(key=lambda d: -float(d["conditional_mi_given_pathology"]))

    n_cts = len(entries)
    y_positions = np.arange(n_cts)[::-1]   # top-most = highest cond MI

    pos_color = "#d62728"    # red: cond MI decreases (delta > 0; pathology contains info)
    neg_color = "#1f77b4"    # blue: cond MI increases (delta < 0; pathology-orthogonal)

    top5_cts, top5_values = [], []
    for y, d in zip(y_positions, entries):
        ct = d["cell_type"]
        x_uncond = float(d["unconditional_mi"])
        x_cond = float(d["conditional_mi_given_pathology"])
        # delta convention from JSON: delta = unconditional - conditional.
        # Use it directly so the color encoding matches the producer's sign.
        delta = float(d.get("delta", x_uncond - x_cond))
        slope_color = pos_color if delta > 0 else neg_color

        ax.plot(
            [x_uncond, x_cond],
            [y, y],
            color=slope_color, linewidth=0.9, alpha=0.75,
            zorder=2,
        )
        ax.plot(x_uncond, y, "o", color=slope_color, markersize=2.2,
                markeredgecolor="white", markeredgewidth=0.4, zorder=3)
        ax.plot(x_cond, y, "s", color=slope_color, markersize=2.5,
                markeredgecolor="white", markeredgewidth=0.4, zorder=3)

        if y >= n_cts - 5:
            top5_cts.append(ct)
            top5_values.append(x_cond)

    ax.set_yticks(y_positions[:n_cts])
    ax.set_yticklabels(
        [d["cell_type"] for d in entries], fontsize=4.5,
    )
    ax.set_xlabel("Mutual information", fontsize=7)
    ax.set_title(
        "CMI slope: unconditional -> conditional|pathology",
        fontsize=8,
    )
    fmt_axes(ax)
    # Legend.
    pos_handle = mpatches.Patch(color=pos_color, label=r"$\Delta>0$: pathology contains info")
    neg_handle = mpatches.Patch(color=neg_color, label=r"$\Delta<0$: pathology-orthogonal")
    ax.legend(handles=[pos_handle, neg_handle], loc="lower right",
              fontsize=5.5, frameon=True)
    # Marker legend explaining circle vs square.
    ax.plot([], [], "o", color="#555555", markersize=3, label="unconditional")
    ax.plot([], [], "s", color="#555555", markersize=3, label="conditional|patho")

    return top5_cts, top5_values


# -----------------------------------------------------------------------------
# Panel 11: LOCO tornado
# -----------------------------------------------------------------------------
def _draw_loco_tornado(
    ax: plt.Axes,
    loco_payload: dict,
    *,
    ct_color_map: dict[str, str],
) -> tuple[list[str], list[float]]:
    """Diverging horizontal bars: signed deltaR2 per CT.

    CTs sorted ascending by delta_r2_vs_canonical (most-negative = most
    load-bearing) so the bars at the top are the most predictively-
    important. Bars left of zero (negative delta) = zeroing this CT
    HURTS R2 (load-bearing); right of zero = adversarial.
    """
    entries = list(loco_payload["per_cell_type"])
    entries.sort(key=lambda d: float(d["delta_r2_vs_canonical"]))

    n_cts = len(entries)
    y_positions = np.arange(n_cts)
    deltas = np.array(
        [float(d["delta_r2_vs_canonical"]) for d in entries], dtype=np.float64,
    )
    cts = [d["cell_type"] for d in entries]
    colors = ["#1f77b4" if d <= 0 else "#d62728" for d in deltas]   # blue/red

    bars = ax.barh(
        y_positions, deltas,
        color=colors, edgecolor="white", linewidth=0.4,
        zorder=2,
    )
    # Per-CT marker dot in CT-color (so the tornado retains CT identity).
    for y, ct in zip(y_positions, cts):
        ax.plot(
            0.0, y, "o",
            markersize=2.2, markerfacecolor=ct_color_map.get(ct, "#888"),
            markeredgecolor="white", markeredgewidth=0.4, zorder=3,
        )

    ax.axvline(0.0, color="#444444", linewidth=0.6, linestyle="--", zorder=4)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(cts, fontsize=4.5)
    ax.set_xlabel(r"$\Delta R^2$ vs canonical (zero-out)", fontsize=7)
    ax.set_title("LOCO ranking (top = most load-bearing)", fontsize=8)
    fmt_axes(ax)

    # Top-5 most load-bearing (most negative delta).
    top5_cts = cts[:5]
    top5_values = [float(d) for d in deltas[:5]]
    return top5_cts, top5_values


# -----------------------------------------------------------------------------
# Panel 12: Consensus size-encoded strip
# -----------------------------------------------------------------------------
def _compute_top5_counts_full(
    consensus_payload: dict,
    ct_coverage: dict,
) -> dict[str, int]:
    """Return a {ct: top5_count} dict over all 31 CTs.

    The consensus_heatmap_data.json explicitly stores ``ranks`` for only 10
    CTs. For the remaining CTs we infer count = 0 (i.e., not in any
    method's top-5), which matches the source orchestrator's filtering
    (it persists only CTs that appeared in at least one method's top-5).
    """
    counts: dict[str, int] = {}
    n_methods = len(consensus_payload.get("methods", []))
    if n_methods == 0:
        n_methods = 11
    ranks = consensus_payload.get("ranks", {})
    all_cts = list(ct_coverage["per_ct"].keys())
    for ct in all_cts:
        method_ranks = ranks.get(ct, {})
        c = sum(1 for r in method_ranks.values() if isinstance(r, (int, float)) and r <= 5)
        counts[ct] = c
    return counts


def _draw_consensus_strip(
    ax: plt.Axes,
    consensus_payload: dict,
    ct_coverage: dict,
    *,
    ct_color_map: dict[str, str],
) -> tuple[list[str], list[int]]:
    """1D strip: 31 CTs sorted by top-5 count desc.

    Color = top-5 count (sequential viridis). Dot SIZE = (1 - zero_frac);
    well-covered CTs are large dots, sparsely-covered CTs are small.
    """
    counts = _compute_top5_counts_full(consensus_payload, ct_coverage)
    sorted_cts = sorted(counts.keys(), key=lambda c: (-counts[c], c))

    n = len(sorted_cts)
    xs = np.arange(n)
    count_vals = np.array([counts[c] for c in sorted_cts], dtype=float)
    zero_fracs = np.array(
        [float(ct_coverage["per_ct"][c]["zero_frac"]) for c in sorted_cts],
    )
    sizes = (1.0 - zero_fracs)              # in [0, 1]
    # Map size to marker area; floor at a small minimum so even zero-frac=1
    # CTs are visible.
    marker_areas = 30.0 + sizes * 200.0

    cmap = PALETTES["sequential"]
    n_methods = len(consensus_payload.get("methods", []))
    if n_methods == 0:
        n_methods = 11
    norm_counts = count_vals / max(n_methods, 1.0)
    colors = [cmap(min(0.95, 0.05 + nc)) for nc in norm_counts]

    ax.scatter(
        xs, np.zeros(n),
        s=marker_areas, c=colors, edgecolor="black", linewidth=0.5,
        zorder=3,
    )
    # Annotate each CT with its count above the dot.
    for i, (ct, c) in enumerate(zip(sorted_cts, count_vals.astype(int))):
        ax.text(
            xs[i], 0.35, str(c),
            ha="center", va="bottom", fontsize=5.5, color="#222222",
        )
    # CT names below dots.
    ax.set_xticks(xs)
    ax.set_xticklabels(sorted_cts, rotation=45, ha="right", fontsize=4.5)
    ax.set_yticks([])
    ax.set_ylim(-0.55, 0.65)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_title(
        f"Consensus: top-5 frequency (color) x coverage (dot size; n_methods={n_methods})",
        fontsize=8,
    )
    fmt_axes(ax)
    # Subtle scale annotation top-right.
    ax.text(
        n - 1, 0.55, "size = 1 - zero_frac",
        ha="right", va="top", fontsize=5.5, color="#666666",
    )
    return sorted_cts[:5], [int(counts[c]) for c in sorted_cts[:5]]


# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------
def _load_attention_npz(path: Path) -> dict:
    """Load per-subject attention attribution from npz; return dict of arrays."""
    arr = np.load(path, allow_pickle=True)
    return {k: arr[k] for k in arr.files}


def _attention_ct_order_from_summary(attn_summary_path: Path) -> list[str] | None:
    """If the summary JSON encodes the CT axis order, return it.

    The ``per_subject_attribution.npz`` file does NOT store the CT-axis
    ordering, but the summary JSON's ``rank_by_mean_importance`` is keyed
    by cell type. We can recover the index ordering by reading any per-CT
    field in the summary that retains source axis ordering. Fallback to
    the canonical CELL_TYPE_ORDER from the precomputed .pt files.
    """
    return None


def make_figure(args: argparse.Namespace) -> tuple[plt.Figure, dict[str, object]]:
    """Build the 12-panel figure. Returns (fig, verification_dict)."""
    rng = np.random.default_rng(42)
    apply_theme("paper")

    # --- Load core JSONs ---
    ig = json.loads(args.captum_ig.read_text())
    gs = json.loads(args.gradientshap.read_text())
    sg = json.loads(args.smoothgrad.read_text())
    attn_npz = _load_attention_npz(args.attention_npz)
    w1 = json.loads(args.wasserstein.read_text())
    cmi = json.loads(args.cmi.read_text())
    loco = json.loads(args.loco.read_text())
    consensus = json.loads(args.consensus.read_text())
    ct_coverage = json.loads(args.ct_coverage.read_text())

    # --- Stable CT ordering + coloring ---
    alpha_ct_order = _alpha_ct_order(ct_coverage)
    ct_color_map = _build_ct_color_map(alpha_ct_order)

    # --- CT order for attention .npz columns ---
    # The attention .npz per-subject arrays are stored in the same CT-axis
    # order as the precomputed ``cell_type_order`` list (canonical pipeline
    # order); load that explicitly from one of the .pt files.
    sample_pt = next(args.precomputed_dir.glob("R*.pt"))
    sample_data = torch.load(sample_pt, map_location="cpu", weights_only=False)
    canonical_ct_order = list(sample_data["cell_type_order"])
    if len(canonical_ct_order) != attn_npz["attnlrp"].shape[1]:
        raise ValueError(
            f"attention .npz has {attn_npz['attnlrp'].shape[1]} CT columns "
            f"but cell_type_order has {len(canonical_ct_order)}"
        )

    # --- Layout: 2 x 6 grid, tall ---
    # Polar (sunburst + radial violin) panels render an inner-circle inside
    # each cell, so we keep the per-cell aspect close to 1.0 by giving each
    # row equal height and using a tight wspace. The 20 x 14 figsize gives
    # each cell ~3.3 x 7 inches; the polar drawings naturally fit a 3.3 x 3.3
    # square, leaving headroom for titles + tick labels.
    fig = plt.figure(figsize=(20, 14))
    gs_grid = fig.add_gridspec(2, 6, hspace=0.30, wspace=0.30,
                               left=0.03, right=0.99, top=0.94, bottom=0.10)

    verify: dict[str, object] = {}

    # Row 1: panels 1-6.
    # Panels 1-3: sunbursts.
    summaries = {"IG": ig, "GradientSHAP": gs, "SmoothGrad": sg}
    for col, method in enumerate(SUNBURST_METHODS):
        ax = fig.add_subplot(gs_grid[0, col])
        top_cts, top_vals = _draw_sunburst(
            ax, summaries[method],
            ct_color_map=ct_color_map,
            method_label=method,
        )
        verify[f"{method}_top5"] = list(zip(top_cts[:5], top_vals[:5]))

    # Panels 4-6: radial violins (AttnLRP, GMAR, GAF AF).
    for col_idx, method in enumerate(RADIAL_METHODS_ROW1, start=3):
        ax = fig.add_subplot(gs_grid[0, col_idx], projection="polar")
        method_arr = attn_npz[ATTN_KEY[method]]
        top_cts, top_vals = _draw_radial_violin(
            ax, method, method_arr, canonical_ct_order,
            ct_coverage=ct_coverage, ct_color_map=ct_color_map,
        )
        verify[f"{method}_top5"] = list(zip(top_cts, top_vals))

    # Row 2: panels 7-12.
    # Panels 7-8: radial violins (GAF AGF, GAF GF).
    for col_idx, method in enumerate(RADIAL_METHODS_ROW2):
        ax = fig.add_subplot(gs_grid[1, col_idx], projection="polar")
        method_arr = attn_npz[ATTN_KEY[method]]
        top_cts, top_vals = _draw_radial_violin(
            ax, method, method_arr, canonical_ct_order,
            ct_coverage=ct_coverage, ct_color_map=ct_color_map,
        )
        verify[f"{method}_top5"] = list(zip(top_cts, top_vals))

    # Panel 9: W1 ridge.
    ax_w1 = fig.add_subplot(gs_grid[1, 2])
    top_pairs = _identify_top_w1_pairs(w1, top_n=5)
    res_ids, vul_ids = _split_resilient_vulnerable(args.residual_csv)
    logger.info(
        "[fig3] W1 ridge: %d resilient / %d vulnerable subjects",
        len(res_ids), len(vul_ids),
    )
    # Build a CT-name -> index map and gene-name -> index map.
    gene_names = list(np.load(args.gene_names, allow_pickle=True))
    gene_name_to_idx = {str(g): i for i, g in enumerate(gene_names)}
    ct_name_to_idx = {ct: i for i, ct in enumerate(canonical_ct_order)}

    # Pre-load pseudobulk for all resilient + vulnerable subjects ONCE
    # (avoids reloading the .pt files for each pair).
    pb_res = load_pseudobulk_matrix(args.precomputed_dir, res_ids)
    pb_vul = load_pseudobulk_matrix(args.precomputed_dir, vul_ids)

    res_vals_per_pair = []
    vul_vals_per_pair = []
    for ct, gene, _w1 in top_pairs:
        ct_idx = ct_name_to_idx[ct]
        gene_idx = gene_name_to_idx[gene]
        rv = pb_res[:, ct_idx, gene_idx]
        vv = pb_vul[:, ct_idx, gene_idx]
        # Drop NaNs (any subject with a missing .pt file would have NaN row).
        rv = rv[np.isfinite(rv)]
        vv = vv[np.isfinite(vv)]
        res_vals_per_pair.append(rv)
        vul_vals_per_pair.append(vv)
    _draw_w1_ridge(
        ax_w1, top_pairs, res_vals_per_pair, vul_vals_per_pair,
        ct_color_map=ct_color_map,
    )
    verify["W1_top5_pairs"] = [
        {"cell_type": ct, "gene": g, "w1": w, "n_res": int(rv.size),
         "n_vul": int(vv.size)}
        for (ct, g, w), rv, vv in zip(top_pairs, res_vals_per_pair, vul_vals_per_pair)
    ]

    # Panel 10: CMI slope.
    ax_cmi = fig.add_subplot(gs_grid[1, 3])
    top_cts_cmi, top_vals_cmi = _draw_cmi_slope(
        ax_cmi, cmi, ct_color_map=ct_color_map,
    )
    verify["CMI_top5"] = list(zip(top_cts_cmi, top_vals_cmi))

    # Panel 11: LOCO tornado.
    ax_loco = fig.add_subplot(gs_grid[1, 4])
    top_cts_loco, top_vals_loco = _draw_loco_tornado(
        ax_loco, loco, ct_color_map=ct_color_map,
    )
    verify["LOCO_top5_load_bearing"] = list(zip(top_cts_loco, top_vals_loco))

    # Panel 12: Consensus strip.
    ax_cons = fig.add_subplot(gs_grid[1, 5])
    top_cts_cons, top_vals_cons = _draw_consensus_strip(
        ax_cons, consensus, ct_coverage,
        ct_color_map=ct_color_map,
    )
    verify["Consensus_top5_count"] = list(zip(top_cts_cons, top_vals_cons))

    # --- Suptitle / legend ---
    fig.suptitle(
        "Methods grid: 11 interpretability methods, native visualization per family",
        fontsize=11, y=0.98,
    )

    # CT color legend along the bottom edge (compact, multi-row).
    legend_handles = [
        mpatches.Patch(facecolor=ct_color_map[ct], edgecolor="black",
                       linewidth=0.4, label=ct)
        for ct in alpha_ct_order
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=8,
        fontsize=5.5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )

    return fig, verify


def _print_verification(verify: dict[str, object]) -> None:
    print("=" * 78)
    print("README Figure 3 -- methods grid (12 panels)")
    print("=" * 78)
    for method in SUNBURST_METHODS:
        print(f"\n  {method} top-5 CTs by total_abs_attribution:")
        for ct, val in verify[f"{method}_top5"]:
            print(f"    - {ct:42s}: {val:.6e}")
    for method in RADIAL_METHODS:
        print(f"\n  {method} top-5 CTs by mean(|attribution|):")
        for ct, val in verify[f"{method}_top5"]:
            print(f"    - {ct:42s}: {val:.6e}")
    print("\n  Wasserstein-1 top-5 (CT, gene) pairs:")
    for d in verify["W1_top5_pairs"]:
        print(
            f"    - {d['cell_type']:32s} : {d['gene']:14s}  "
            f"W1={d['w1']:.4f}  n_res={d['n_res']}  n_vul={d['n_vul']}"
        )
    print("\n  CMI top-5 by conditional_mi_given_pathology:")
    for ct, val in verify["CMI_top5"]:
        print(f"    - {ct:42s}: {val:.6e}")
    print("\n  LOCO top-5 most load-bearing (most negative delta_r2_vs_canonical):")
    for ct, val in verify["LOCO_top5_load_bearing"]:
        print(f"    - {ct:42s}: {val:+.6e}")
    print("\n  Consensus top-5 by top-5 count:")
    for ct, c in verify["Consensus_top5_count"]:
        print(f"    - {ct:42s}: {c} methods")
    print("=" * 78)


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
        "--attention-npz", type=Path,
        default=_WORKTREE_ROOT / "outputs/canonical/interpretability/attention_attribution/per_subject_attribution.npz",
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
        "--consensus", type=Path,
        default=_WORKTREE_ROOT / "outputs/canonical/interpretability/figures/consensus_heatmap/consensus_heatmap_data.json",
    )
    parser.add_argument(
        "--ct-coverage", type=Path,
        default=_WORKTREE_ROOT / "outputs/canonical/interpretability/ct_coverage_full_cohort.json",
    )
    parser.add_argument(
        "--residual-csv", type=Path,
        default=_WORKTREE_ROOT / "outputs/canonical/interpretability/residual_per_subject.csv",
    )
    parser.add_argument(
        "--precomputed-dir", type=Path,
        default=_WORKTREE_ROOT / "data/precomputed",
    )
    parser.add_argument(
        "--gene-names", type=Path,
        default=_WORKTREE_ROOT / "data/precomputed/gene_names.npy",
    )
    parser.add_argument(
        "--out-stem", type=Path,
        default=_WORKTREE_ROOT / "figures/fig3_methods_grid",
    )
    parser.add_argument(
        "--old-figure", type=Path,
        default=_WORKTREE_ROOT / "figures/fig3_consensus_causal.png",
        help="Legacy figure to delete (replaced by this script).",
    )
    args = parser.parse_args()

    # --- Idempotent: delete old figure(s) if present ---
    out_png = args.out_stem.with_suffix(".png")
    if out_png.exists():
        logger.info("[fig3] removing existing %s", out_png)
        out_png.unlink()
    if args.old_figure.exists():
        logger.info("[fig3] removing legacy figure %s", args.old_figure)
        args.old_figure.unlink()

    fig, verify = make_figure(args)
    # Render at the project's standard 600 DPI (theme default). At 20x14
    # inches this produces a 12000x8400 pixel raster (~5-7 MB PNG), which is
    # the canonical resolution for paper / lab-meeting figures. File-size
    # cap was previously 2 MB but has been lifted by user authorization.
    written = save_fig(fig, args.out_stem, dpi=600, formats=("png",))
    plt.close(fig)
    for w in written:
        logger.info("[fig3] wrote %s (%.2f MB)", w, w.stat().st_size / 1e6)

    _print_verification(verify)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
