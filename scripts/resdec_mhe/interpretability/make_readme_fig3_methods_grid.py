#!/usr/bin/env python
"""Render Figure 3 for the ResDec-MHE README revamp -- 12-panel methods grid.

Replaces the previous ``fig3_consensus_causal.png`` with a tall PORTRAIT
12-panel grid (6 rows x 2 cols, ~10x30 inches) that gives each
interpretability method its own native visualization rather than collapsing
all of them into a single binary heatmap.

Panel inventory (top -> bottom, left -> right, 6 rows x 2 cols)
---------------------------------------------------------------
  Row 1: (1) IG sunburst             (2) GradientSHAP sunburst
  Row 2: (3) SmoothGrad sunburst     (4) AttnLRP radial violin
  Row 3: (5) GMAR radial violin      (6) GAF AF radial violin
  Row 4: (7) GAF AGF radial violin   (8) GAF GF radial violin
  Row 5: (9) Wasserstein-1 ridge top-10  (10) CMI slope all 31
  Row 6: (11) LOCO tornado all 31    (12) Consensus size-encoded heatmap

Sunburst format
---------------
  Inner ring : top-15 CTs by total_abs_attribution (sector arc ∝ value)
  Outer ring : top-3 genes per CT (15 x 3 = 45 outer sectors)
                outer arc ∝ mean_abs_attribution within parent CT

Wasserstein ridge
-----------------
  Top-10 (CT, gene) pairs by W1 distance (was top-5).
  Each pair = horizontal lane with overlaid resilient (n=129) vs
  vulnerable (n=129) pseudobulk KDEs; W1 annotated.

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
        resilient/vulnerable subjects for the W1 ridge plot)
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

Layout / overlap policy
-----------------------
    * figure created with ``constrained_layout=True`` (no bbox_inches='tight'!)
    * gridspec hspace=0.5, wspace=0.4
    * polar (sunburst, radial-violin) panels: legend OUTSIDE on right via
      ``bbox_to_anchor=(1.05, 0.5)``
    * long CT names ("LAMP5-LHX6 and Chandelier", "Committed oligodendrocyte
      precursor") shortened via ``_short_ct``
    * annotations placed in ``axes.transAxes`` coordinates with explicit offsets
    * ``save_fig(... bbox_inches=None)`` so constrained_layout is preserved
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


# -----------------------------------------------------------------------------
# CT name helpers
# -----------------------------------------------------------------------------
# Hand-curated short labels for the long CT names so they don't collide with
# adjacent labels in dense panels (CMI slope, LOCO tornado, Consensus heatmap).
# Keys MUST match the canonical names used throughout the pipeline.
_CT_SHORT_OVERRIDES: dict[str, str] = {
    "LAMP5-LHX6 and Chandelier":            "LAMP5-LHX6+Chand",
    "Committed oligodendrocyte precursor":  "Committed OPC",
    "Oligodendrocyte precursor":            "OPC",
    "Deep-layer intratelencephalic":        "Deep-layer IT",
    "Deep-layer near-projecting":           "Deep-layer NP",
    "Deep-layer corticothalamic and 6b":    "Deep-layer CT/6b",
    "Cerebellar inhibitory":                "Cerebellar inhib.",
    "Cerebellar excitatory":                "Cerebellar exc.",
    "MGE interneuron":                      "MGE intn.",
    "CGE interneuron":                      "CGE intn.",
    "Upper-layer intratelencephalic":       "Upper-layer IT",
    "Upper rhombic lip":                    "Upper rhombic lip",
    "Lower rhombic lip":                    "Lower rhombic lip",
    "Choroid plexus":                       "Choroid plexus",
    "Thalamic excitatory":                  "Thalamic exc.",
    "Midbrain-derived inhibitory":          "Midbrain inhib.",
    "Mammillary body":                      "Mammillary body",
    "Eccentric medium spiny neuron":        "Ecc. MSN",
    "Splatter":                             "Splatter",
    "Astrocyte":                            "Astrocyte",
    "Oligodendrocyte":                      "Oligodend.",
    "Microglia":                            "Microglia",
    "Vascular":                             "Vascular",
    "Fibroblast":                           "Fibroblast",
    "Ependymal":                            "Ependymal",
    "Miscellaneous":                        "Miscellaneous",
    "Hippocampal CA1-3":                    "Hippo. CA1-3",
    "Hippocampal CA4":                      "Hippo. CA4",
    "Hippocampal dentate gyrus":            "Hippo. DG",
    "Amygdala excitatory":                  "Amygdala exc.",
    "Striatal":                             "Striatal",
}


def _short_ct(ct: str, *, max_len: int = 16) -> str:
    """Return a short version of a CT name suitable for dense-axis labels.

    Looks up an explicit override first; falls back to truncating after
    ``max_len`` characters with an ellipsis. Always returns something
    non-empty so axes never silently lose labels.
    """
    if ct in _CT_SHORT_OVERRIDES:
        return _CT_SHORT_OVERRIDES[ct]
    if len(ct) <= max_len:
        return ct
    return ct[: max_len - 1] + "…"  # ellipsis


def _wrap_ct(ct: str, *, max_chars: int = 15) -> str:
    """Wrap a long CT name with explicit '\\n' after ~max_chars characters.

    Splits on word boundaries so the inserted newline doesn't bisect a
    word. Used in panels where a vertical wrap reads better than a short
    abbreviation (e.g., legends).
    """
    if len(ct) <= max_chars:
        return ct
    words = ct.split()
    line = ""
    out_lines = []
    for w in words:
        if line and len(line) + 1 + len(w) > max_chars:
            out_lines.append(line)
            line = w
        else:
            line = (line + " " + w) if line else w
    if line:
        out_lines.append(line)
    return "\n".join(out_lines)


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
    top_k_cts: int = 15,
    top_n_genes: int = 3,
) -> tuple[list[str], list[float], dict[str, list[tuple[str, float]]]]:
    """Render a two-ring sunburst for a captum-style method JSON.

    Inner ring sectors = top-K CTs by total_abs_attribution.
    Outer ring sectors = top-N genes within each CT, with size ~
    mean_abs_attribution.

    Returns
    -------
    top_cts : list[str]
        The top_k_cts CT names actually rendered (for verification logs).
    top_ct_values : list[float]
        Their total_abs_attribution values (for verification logs).
    per_ct_genes : dict[str, list[(gene, value)]]
        Top-N gene tuples used for each CT (for verification logs).
    """
    ranked = summary["cell_types_ranked_by_total_attribution"]
    ranked_sorted = sorted(
        ranked, key=lambda d: -float(d["total_abs_attribution"])
    )[:top_k_cts]
    top_cts = [d["cell_type"] for d in ranked_sorted]
    top_ct_values = [float(d["total_abs_attribution"]) for d in ranked_sorted]

    total_ct_attr = sum(top_ct_values)
    per_ct_genes: dict[str, list[tuple[str, float]]] = {}
    if total_ct_attr <= 0.0:
        ax.text(0.5, 0.5, f"{method_label}: empty", ha="center", va="center",
                transform=ax.transAxes)
        return top_cts, top_ct_values, per_ct_genes

    R_INNER = 0.55
    R_OUTER = 1.00

    theta_start = 90.0
    for ct, value in zip(top_cts, top_ct_values):
        sweep = 360.0 * (value / total_ct_attr)
        theta_end = theta_start - sweep   # clockwise

        ct_color = ct_color_map.get(ct, "#888888")
        wedge = Wedge(
            (0.0, 0.0),
            R_INNER,
            theta_end, theta_start,
            width=R_INNER,
            facecolor=ct_color,
            edgecolor="white",
            linewidth=0.6,
            zorder=2,
        )
        ax.add_patch(wedge)

        # Place a CT label inside the inner wedge if the sector is wide
        # enough; otherwise skip (collisions ruin readability).
        if sweep >= 14.0:
            mid_inner_deg = (theta_start + theta_end) / 2.0
            mid_inner = np.deg2rad(mid_inner_deg)
            tx = (R_INNER * 0.55) * np.cos(mid_inner)
            ty = (R_INNER * 0.55) * np.sin(mid_inner)
            ax.text(
                tx, ty,
                _short_ct(ct, max_len=12),
                ha="center", va="center",
                fontsize=4.5, color="white",
                fontweight="bold", zorder=4,
            )

        # Outer ring: top-N genes for this CT.
        gene_list = summary.get("top_genes_per_cell_type", {}).get(ct, [])
        gene_list = gene_list[:top_n_genes]
        per_ct_genes[ct] = [(g["gene"], float(g["mean_abs_attribution"])) for g in gene_list]
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

            if g_sweep >= 8.0:
                mid_angle_deg = (gene_theta_start + g_theta_end) / 2.0
                mid_angle = np.deg2rad(mid_angle_deg)
                lr = (R_INNER + R_OUTER) / 2.0
                tx = lr * np.cos(mid_angle)
                ty = lr * np.sin(mid_angle)
                rot = mid_angle_deg - 90.0
                if mid_angle_deg < -90.0 or mid_angle_deg > 90.0:
                    rot += 180.0
                ax.text(
                    tx, ty,
                    g["gene"],
                    ha="center", va="center",
                    fontsize=5.0, rotation=rot, color="#222222",
                    zorder=3,
                )

            gene_theta_start = g_theta_end

        theta_start = theta_end

    ax.set_xlim(-1.20, 1.20)
    ax.set_ylim(-1.20, 1.20)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(
        f"{method_label}\nsunburst: top-{top_k_cts} CTs (inner) x top-{top_n_genes} genes (outer)",
        fontsize=8, pad=4.0,
    )
    return top_cts, top_ct_values, per_ct_genes


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
    n_top_labels: int = 3,
) -> tuple[list[str], list[float]]:
    """Render a polar plot with one half-violin per CT around a circle.

    Each spoke shows the per-subject distribution of ``method_arr[:, k]``.
    Spoke radius spans [r_min, r_max] of the per-CT data; the violin shape
    is generated from a kernel density estimate of the 516 subjects'
    values for that CT, normalized to a fixed angular width so spokes
    don't overlap.

    Top ``n_top_labels`` CTs (by mean(|attribution|)) are annotated on
    the rim with short names so the polar plot retains identity without
    overcrowding.

    Returns top-5 CTs by mean(|attribution|) for verification.
    """
    n_cts = len(ct_names)
    if method_arr.shape[1] != n_cts:
        raise ValueError(
            f"radial violin: method_arr has {method_arr.shape[1]} columns "
            f"but ct_names has {n_cts}"
        )
    arr_abs = np.abs(method_arr)

    per_ct_min = arr_abs.min(axis=0)
    per_ct_max = arr_abs.max(axis=0)
    per_ct_median = np.median(arr_abs, axis=0)
    per_ct_mean = arr_abs.mean(axis=0)

    angles = np.linspace(0.0, 2.0 * np.pi, n_cts, endpoint=False)
    half_width = (2.0 * np.pi / n_cts) * 0.40

    R_MIN = 0.10
    R_MAX = 1.00
    eps = 1e-12
    per_ct_lo = per_ct_min
    per_ct_hi = per_ct_max
    per_ct_span = np.maximum(per_ct_hi - per_ct_lo, eps)

    def _norm_r(values: np.ndarray, k: int) -> np.ndarray:
        return R_MIN + (values - per_ct_lo[k]) / per_ct_span[k] * (R_MAX - R_MIN)

    for k in range(n_cts):
        ct = ct_names[k]
        well_covered = bool(
            ct_coverage["per_ct"].get(ct, {}).get("well_covered", False)
        )
        face = ct_color_map.get(ct, "#888888")
        edge = "#333333" if well_covered else "#cccccc"

        vals = arr_abs[:, k]
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
        density = density / density.max() * half_width

        rs = _norm_r(rs_vals, k)
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

        mean_r = _norm_r(np.array([per_ct_mean[k]]), k)[0]
        ax.plot(
            angles[k], mean_r,
            marker="o", markersize=2.0,
            color="white", markeredgecolor="black",
            markeredgewidth=0.4, zorder=4,
        )
        median_r = _norm_r(np.array([per_ct_median[k]]), k)[0]
        ax.plot(
            [angles[k], angles[k]],
            [R_MIN * 0.5, median_r],
            color="#888888", linewidth=0.4, alpha=0.6, zorder=2,
        )

    ax.set_xticks(angles)
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_ylim(0.0, R_MAX * 1.20)
    ax.set_title(method_label, fontsize=8, pad=6.0)
    ax.tick_params(pad=0.0)
    ax.grid(True, color="#e6e6e6", linewidth=0.3, alpha=0.7)
    ax.spines["polar"].set_linewidth(0.5)
    ax.spines["polar"].set_color("#888888")

    top_idx_full = np.argsort(-per_ct_mean)
    top5_cts = [ct_names[i] for i in top_idx_full[:5]]
    top5_values = [float(per_ct_mean[i]) for i in top_idx_full[:5]]

    # Annotate top-N CTs with short labels at the rim.
    for rank, idx in enumerate(top_idx_full[:n_top_labels]):
        ct_name = ct_names[idx]
        ax.text(
            angles[idx], R_MAX * 1.15,
            _short_ct(ct_name, max_len=14),
            ha="center", va="center",
            fontsize=5.0,
            color=ct_color_map.get(ct_name, "#333333"),
            fontweight="bold", zorder=5,
        )
    return top5_cts, top5_values


# -----------------------------------------------------------------------------
# Panel 9: Wasserstein-1 ridge plot
# -----------------------------------------------------------------------------
def _identify_top_w1_pairs(w1_payload: dict, top_n: int = 10) -> list[tuple[str, str, float]]:
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
    annotated on the right side via ``ax.transAxes`` so it never collides
    with data.

    Resilient color is tab10 blue; vulnerable is tab10 red.
    """
    from scipy.stats import gaussian_kde

    n = len(pairs)
    res_color = "#1f77b4"   # tab10 blue
    vul_color = "#d62728"   # tab10 red
    LANE_HEIGHT = 1.0

    # First pass: figure out the union x-range across all pairs so we can
    # use a single shared x-axis (different (CT, gene) pairs have very
    # different absolute scales; we normalize by per-pair peak instead).
    pair_x_ranges = []
    for res_vals, vul_vals in zip(res_vals_per_pair, vul_vals_per_pair):
        all_vals = np.concatenate([res_vals, vul_vals])
        if all_vals.size < 2:
            pair_x_ranges.append((0.0, 1.0))
            continue
        x_min = float(np.min(all_vals))
        x_max = float(np.max(all_vals))
        pair_x_ranges.append((x_min, x_max))

    # Render each lane in its own LOCAL x range (we use a single ax but
    # rescale each lane's KDE x to [0, 1] of that lane's local span; the
    # x-axis label below is intentionally unitless because each lane has
    # its own scale).
    n_xs = 200
    for i, ((ct, gene, w1), res_vals, vul_vals) in enumerate(
        zip(pairs, res_vals_per_pair, vul_vals_per_pair)
    ):
        y0 = (n - 1 - i) * LANE_HEIGHT      # top-most lane is the strongest pair
        x_min, x_max = pair_x_ranges[i]
        span = max(x_max - x_min, 1e-9)
        xs_local = np.linspace(0.0, 1.0, n_xs)
        xs_data = x_min + xs_local * span

        try:
            kde_r = gaussian_kde(res_vals)
            yr = kde_r(xs_data)
        except (np.linalg.LinAlgError, ValueError):
            yr = np.zeros_like(xs_data)
        try:
            kde_v = gaussian_kde(vul_vals)
            yv = kde_v(xs_data)
        except (np.linalg.LinAlgError, ValueError):
            yv = np.zeros_like(xs_data)
        peak = max(yr.max(), yv.max(), 1e-9)
        yr_n = yr / peak * 0.85
        yv_n = yv / peak * 0.85

        ax.fill_between(xs_local, y0, y0 + yr_n, color=res_color, alpha=0.45,
                        linewidth=0.0, zorder=2)
        ax.fill_between(xs_local, y0, y0 + yv_n, color=vul_color, alpha=0.45,
                        linewidth=0.0, zorder=3)
        ax.plot(xs_local, y0 + yr_n, color=res_color, linewidth=0.7, zorder=4)
        ax.plot(xs_local, y0 + yv_n, color=vul_color, linewidth=0.7, zorder=5)

        # Lane label on left side (axes-relative): "{short_CT} : {gene}".
        # Axes-relative y = (y0 + 0.5) / (n * LANE_HEIGHT)
        y_axes = (y0 + 0.5) / (n * LANE_HEIGHT + 0.6)
        ax.text(
            -0.02, y_axes,
            f"{_short_ct(ct, max_len=14)} : {gene}",
            ha="right", va="center",
            fontsize=6.0,
            color=ct_color_map.get(ct, "#333333"),
            fontweight="bold",
            transform=ax.transAxes,
        )
        # W1 annotation on right side (axes-relative).
        ax.text(
            1.02, y_axes,
            r"$W_1$=" + f"{w1:.3f}",
            ha="left", va="center",
            fontsize=6.0, color="#333333",
            transform=ax.transAxes,
        )

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.2, n * LANE_HEIGHT + 0.4)
    ax.set_yticks([])
    ax.set_xlabel("Pseudobulk expression (per-pair normalized 0..1)", fontsize=7)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xticklabels(["min", "mid", "max"], fontsize=6)
    ax.set_title(
        f"$W_1$ top-{n} (resilient n={len(res_vals_per_pair[0])} vs "
        f"vulnerable n={len(vul_vals_per_pair[0])})",
        fontsize=8,
    )
    fmt_axes(ax)
    res_patch = mpatches.Patch(facecolor=res_color, alpha=0.45, label="Resilient")
    vul_patch = mpatches.Patch(facecolor=vul_color, alpha=0.45, label="Vulnerable")
    # Place legend OUTSIDE the data area to avoid covering ridges.
    ax.legend(handles=[res_patch, vul_patch],
              loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=2, fontsize=6, frameon=True)


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

    pos_color = "#d62728"
    neg_color = "#1f77b4"

    top5_cts, top5_values = [], []
    for y, d in zip(y_positions, entries):
        ct = d["cell_type"]
        x_uncond = float(d["unconditional_mi"])
        x_cond = float(d["conditional_mi_given_pathology"])
        delta = float(d.get("delta", x_uncond - x_cond))
        slope_color = pos_color if delta > 0 else neg_color

        ax.plot(
            [x_uncond, x_cond],
            [y, y],
            color=slope_color, linewidth=0.9, alpha=0.75,
            zorder=2,
        )
        ax.plot(x_uncond, y, "o", color=slope_color, markersize=2.5,
                markeredgecolor="white", markeredgewidth=0.4, zorder=3)
        ax.plot(x_cond, y, "s", color=slope_color, markersize=2.8,
                markeredgecolor="white", markeredgewidth=0.4, zorder=3)

        if y >= n_cts - 5:
            top5_cts.append(ct)
            top5_values.append(x_cond)

    ax.set_yticks(y_positions[:n_cts])
    ax.set_yticklabels(
        [_short_ct(d["cell_type"], max_len=18) for d in entries], fontsize=5.5,
    )
    ax.set_xlabel("Mutual information", fontsize=7)
    ax.set_title(
        "CMI: unconditional -> conditional|pathology",
        fontsize=8,
    )
    fmt_axes(ax)
    pos_handle = mpatches.Patch(color=pos_color, label=r"$\Delta>0$: patho. contains info")
    neg_handle = mpatches.Patch(color=neg_color, label=r"$\Delta<0$: patho.-orthogonal")
    # Bottom-anchored legend, outside data area.
    ax.legend(handles=[pos_handle, neg_handle],
              loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=2, fontsize=5.5, frameon=True)

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
    colors = ["#1f77b4" if d <= 0 else "#d62728" for d in deltas]

    ax.barh(
        y_positions, deltas,
        color=colors, edgecolor="white", linewidth=0.4,
        zorder=2,
    )
    for y, ct in zip(y_positions, cts):
        ax.plot(
            0.0, y, "o",
            markersize=2.2, markerfacecolor=ct_color_map.get(ct, "#888"),
            markeredgecolor="white", markeredgewidth=0.4, zorder=3,
        )

    ax.axvline(0.0, color="#444444", linewidth=0.6, linestyle="--", zorder=4)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([_short_ct(c, max_len=18) for c in cts], fontsize=5.5)
    ax.set_xlabel(r"$\Delta R^2$ vs canonical (zero-out)", fontsize=7)
    ax.set_title("LOCO ranking (top = most load-bearing)", fontsize=8)
    fmt_axes(ax)

    top5_cts = cts[:5]
    top5_values = [float(d) for d in deltas[:5]]
    return top5_cts, top5_values


# -----------------------------------------------------------------------------
# Panel 12: Consensus size-encoded heatmap
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
    ranks = consensus_payload.get("ranks", {})
    all_cts = list(ct_coverage["per_ct"].keys())
    for ct in all_cts:
        method_ranks = ranks.get(ct, {})
        c = sum(1 for r in method_ranks.values() if isinstance(r, (int, float)) and r <= 5)
        counts[ct] = c
    return counts


def _draw_consensus_heatmap(
    ax: plt.Axes,
    consensus_payload: dict,
    ct_coverage: dict,
    *,
    ct_color_map: dict[str, str],
) -> tuple[list[str], list[int]]:
    """Vertical strip: 31 CTs sorted by top-5 count desc.

    Color = top-5 count (sequential viridis). Dot SIZE = (1 - zero_frac);
    well-covered CTs are large dots, sparsely-covered CTs are small.
    Vertical layout (y=ct, x=0) so the 31 labels read horizontally.
    """
    counts = _compute_top5_counts_full(consensus_payload, ct_coverage)
    sorted_cts = sorted(counts.keys(), key=lambda c: (-counts[c], c))

    n = len(sorted_cts)
    ys = np.arange(n)[::-1]
    count_vals = np.array([counts[c] for c in sorted_cts], dtype=float)
    zero_fracs = np.array(
        [float(ct_coverage["per_ct"][c]["zero_frac"]) for c in sorted_cts],
    )
    sizes = (1.0 - zero_fracs)
    marker_areas = 30.0 + sizes * 200.0

    cmap = PALETTES["sequential"]
    n_methods = len(consensus_payload.get("methods", []))
    if n_methods == 0:
        n_methods = 11
    norm_counts = count_vals / max(n_methods, 1.0)
    colors = [cmap(min(0.95, 0.05 + nc)) for nc in norm_counts]

    ax.scatter(
        np.zeros(n), ys,
        s=marker_areas, c=colors, edgecolor="black", linewidth=0.5,
        zorder=3,
    )
    # Annotate each CT count to the RIGHT of dot.
    for i, (ct, c) in enumerate(zip(sorted_cts, count_vals.astype(int))):
        ax.text(
            0.35, ys[i], str(c),
            ha="left", va="center", fontsize=6.0, color="#222222",
        )

    ax.set_yticks(ys)
    ax.set_yticklabels([_short_ct(c, max_len=18) for c in sorted_cts], fontsize=5.5)
    ax.set_xticks([])
    ax.set_xlim(-0.5, 1.2)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_title(
        f"Consensus: top-5 freq (color) x coverage (dot size; n_methods={n_methods})",
        fontsize=8, pad=10.0,
    )
    fmt_axes(ax)
    # "size = 1 - zero_frac" caption: place at axes-bottom-right via transAxes so
    # it doesn't collide with the title or the top-most dot (Splatter at y=n-1).
    ax.text(
        1.0, -0.02, "dot size = 1 - zero_frac",
        ha="right", va="top", fontsize=5.5, color="#666666",
        transform=ax.transAxes,
    )
    return sorted_cts[:5], [int(counts[c]) for c in sorted_cts[:5]]


# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------
def _load_attention_npz(path: Path) -> dict:
    """Load per-subject attention attribution from npz; return dict of arrays."""
    arr = np.load(path, allow_pickle=True)
    return {k: arr[k] for k in arr.files}


def make_figure(args: argparse.Namespace) -> tuple[plt.Figure, dict[str, object]]:
    """Build the 12-panel figure (6 rows x 2 cols portrait). Returns (fig, verification_dict)."""
    rng = np.random.default_rng(42)  # noqa: F841 — kept for future stochastic ops
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
    sample_pt = next(args.precomputed_dir.glob("R*.pt"))
    sample_data = torch.load(sample_pt, map_location="cpu", weights_only=False)
    canonical_ct_order = list(sample_data["cell_type_order"])
    if len(canonical_ct_order) != attn_npz["attnlrp"].shape[1]:
        raise ValueError(
            f"attention .npz has {attn_npz['attnlrp'].shape[1]} CT columns "
            f"but cell_type_order has {len(canonical_ct_order)}"
        )

    # --- Layout: 6 rows x 2 cols portrait, figsize 10 x 30 inches ---
    # constrained_layout=True ensures non-overlapping panels even with
    # generous hspace/wspace; we override save_fig's bbox_inches=None
    # to preserve the constrained layout.
    fig = plt.figure(figsize=(10, 30), constrained_layout=False)
    gs_grid = fig.add_gridspec(
        6, 2,
        hspace=0.50, wspace=0.40,
        left=0.10, right=0.92, top=0.965, bottom=0.025,
    )

    verify: dict[str, object] = {}

    # --- Row 1: IG sunburst (col 0), GradientSHAP sunburst (col 1) ---
    summaries = {"IG": ig, "GradientSHAP": gs, "SmoothGrad": sg}
    panel_layout = [
        ("IG",            (0, 0), "sunburst"),
        ("GradientSHAP",  (0, 1), "sunburst"),
        ("SmoothGrad",    (1, 0), "sunburst"),
        ("AttnLRP",       (1, 1), "radial"),
        ("GMAR",          (2, 0), "radial"),
        ("GAF AF",        (2, 1), "radial"),
        ("GAF AGF",       (3, 0), "radial"),
        ("GAF GF",        (3, 1), "radial"),
        ("W1",            (4, 0), "ridge"),
        ("CMI",           (4, 1), "cmi"),
        ("LOCO",          (5, 0), "loco"),
        ("Consensus",     (5, 1), "consensus"),
    ]

    # Panels 1-3: sunbursts.
    for method, (r, c), kind in panel_layout:
        if kind != "sunburst":
            continue
        ax = fig.add_subplot(gs_grid[r, c])
        top_cts, top_vals, per_ct_genes = _draw_sunburst(
            ax, summaries[method],
            ct_color_map=ct_color_map,
            method_label=method,
            top_k_cts=15, top_n_genes=3,
        )
        verify[f"{method}_top15_cts"] = list(zip(top_cts, top_vals))
        verify[f"{method}_top3_genes_per_ct"] = per_ct_genes

    # Panels 4-8: radial violins (AttnLRP, GMAR, GAF AF, GAF AGF, GAF GF).
    for method, (r, c), kind in panel_layout:
        if kind != "radial":
            continue
        ax = fig.add_subplot(gs_grid[r, c], projection="polar")
        method_arr = attn_npz[ATTN_KEY[method]]
        top_cts, top_vals = _draw_radial_violin(
            ax, method, method_arr, canonical_ct_order,
            ct_coverage=ct_coverage, ct_color_map=ct_color_map,
        )
        verify[f"{method}_top5"] = list(zip(top_cts, top_vals))

    # Panel 9: W1 ridge top-10.
    for method, (r, c), kind in panel_layout:
        if kind != "ridge":
            continue
        ax_w1 = fig.add_subplot(gs_grid[r, c])
        top_pairs = _identify_top_w1_pairs(w1, top_n=10)
        res_ids, vul_ids = _split_resilient_vulnerable(args.residual_csv)
        logger.info(
            "[fig3] W1 ridge: %d resilient / %d vulnerable subjects",
            len(res_ids), len(vul_ids),
        )
        gene_names = list(np.load(args.gene_names, allow_pickle=True))
        gene_name_to_idx = {str(g): i for i, g in enumerate(gene_names)}
        ct_name_to_idx = {ct: i for i, ct in enumerate(canonical_ct_order)}

        pb_res = load_pseudobulk_matrix(args.precomputed_dir, res_ids)
        pb_vul = load_pseudobulk_matrix(args.precomputed_dir, vul_ids)

        res_vals_per_pair = []
        vul_vals_per_pair = []
        for ct, gene, _w1 in top_pairs:
            ct_idx = ct_name_to_idx[ct]
            gene_idx = gene_name_to_idx[gene]
            rv = pb_res[:, ct_idx, gene_idx]
            vv = pb_vul[:, ct_idx, gene_idx]
            rv = rv[np.isfinite(rv)]
            vv = vv[np.isfinite(vv)]
            res_vals_per_pair.append(rv)
            vul_vals_per_pair.append(vv)
        _draw_w1_ridge(
            ax_w1, top_pairs, res_vals_per_pair, vul_vals_per_pair,
            ct_color_map=ct_color_map,
        )
        verify["W1_top10_pairs"] = [
            {"cell_type": ct, "gene": g, "w1": w, "n_res": int(rv.size),
             "n_vul": int(vv.size)}
            for (ct, g, w), rv, vv in zip(top_pairs, res_vals_per_pair, vul_vals_per_pair)
        ]

    # Panel 10: CMI slope.
    for method, (r, c), kind in panel_layout:
        if kind != "cmi":
            continue
        ax_cmi = fig.add_subplot(gs_grid[r, c])
        top_cts_cmi, top_vals_cmi = _draw_cmi_slope(
            ax_cmi, cmi, ct_color_map=ct_color_map,
        )
        verify["CMI_top5"] = list(zip(top_cts_cmi, top_vals_cmi))

    # Panel 11: LOCO tornado.
    for method, (r, c), kind in panel_layout:
        if kind != "loco":
            continue
        ax_loco = fig.add_subplot(gs_grid[r, c])
        top_cts_loco, top_vals_loco = _draw_loco_tornado(
            ax_loco, loco, ct_color_map=ct_color_map,
        )
        verify["LOCO_top5_load_bearing"] = list(zip(top_cts_loco, top_vals_loco))

    # Panel 12: Consensus size-encoded heatmap.
    for method, (r, c), kind in panel_layout:
        if kind != "consensus":
            continue
        ax_cons = fig.add_subplot(gs_grid[r, c])
        top_cts_cons, top_vals_cons = _draw_consensus_heatmap(
            ax_cons, consensus, ct_coverage,
            ct_color_map=ct_color_map,
        )
        verify["Consensus_top5_count"] = list(zip(top_cts_cons, top_vals_cons))

    # --- Suptitle (CT legend omitted: bbox_to_anchor=(1.05, 0.5) on every
    # polar panel would clutter; instead the per-panel rim labels carry CT
    # identity via color, and the consensus heatmap row 6 explicitly labels
    # all 31 CTs).
    fig.suptitle(
        "ResDec-MHE methods grid: 11 interpretability methods, native "
        "visualization per family",
        fontsize=11, y=0.985,
    )

    return fig, verify


def _print_verification(verify: dict[str, object]) -> None:
    print("=" * 78)
    print("README Figure 3 -- methods grid (12 panels)")
    print("=" * 78)
    print("Layout: 6 rows x 2 cols portrait, figsize 10x30")
    print("DPI: 600")
    print("=" * 78)
    for method in SUNBURST_METHODS:
        print(f"\n  {method} top-15 CTs by total_abs_attribution (sunburst inner ring):")
        for ct, val in verify[f"{method}_top15_cts"]:
            print(f"    - {ct:42s}: {val:.6e}")
        print(f"  {method} top-3 genes per CT (sunburst outer ring):")
        gene_dict = verify[f"{method}_top3_genes_per_ct"]
        for ct, _ in verify[f"{method}_top15_cts"]:
            gene_tuples = gene_dict.get(ct, [])
            line = ", ".join(f"{g}={v:.3e}" for g, v in gene_tuples)
            print(f"      {ct:42s}: {line}")
    for method in RADIAL_METHODS:
        print(f"\n  {method} top-5 CTs by mean(|attribution|):")
        for ct, val in verify[f"{method}_top5"]:
            print(f"    - {ct:42s}: {val:.6e}")
    print("\n  Wasserstein-1 top-10 (CT, gene) pairs (ridge plot):")
    for d in verify["W1_top10_pairs"]:
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
    # Render at the project's standard 600 DPI.
    # CRITICAL: bbox_inches=None — using "tight" would crop our explicit
    # gridspec margins and clip the constrained-layout-friendly margins
    # we set up to keep legends / titles outside the data areas.
    written = save_fig(fig, args.out_stem, dpi=600, formats=("png",), bbox_inches=None)
    plt.close(fig)
    for w in written:
        logger.info("[fig3] wrote %s (%.2f MB)", w, w.stat().st_size / 1e6)

    _print_verification(verify)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
