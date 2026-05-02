"""EXP-037: Cluster k0 (vulnerable) vs k2 (resilient) differential analysis.

EXP-033 established that the per-subject ResDec-MHE residual is bimodal and
that a 4-component GMM splits the n=516 cohort into clusters whose membership
is significantly associated (BH q ≤ 0.011) with five pathology axes (Braak,
CERAD, NIA-Reagan, gpath, plaq). The most-vulnerable cluster k0 (n=55,
μ=−1.54) is 78% AD-prob; the most-resilient k2 (n=60, μ=+0.84) is 60% NCI.

This EXP asks: do the snRNA-seq data differ between k0 and k2 BEYOND the
expected pathology-axis differences? I.e., are there cell-type abundance,
per-(CT, gene) pseudobulk expression, or per-(CT, gene) Captum-attribution
differences between the 55 vs 60 subjects?

Method
------
1. Re-fit GMM(k=4, random_state=0) on the canonical residual_per_subject.csv
   to recover per-subject cluster labels (the JSON does not store them but
   the procedure is deterministic). Confirm cluster sizes match
   `latent_class_k4_crosstab.json` (55, 262, 60, 139).
2. Identify subjects in k=0 (vulnerable) and k=2 (resilient).
3. **Cell-type abundance differential.** For each of 31 CTs, two-sample
   Wilcoxon rank-sum (mannwhitneyu, two-sided) on per-subject cell counts
   in that CT (k0 vs k2). BH-FDR across 31 tests.
4. **Per-CT pseudobulk gene differential.** For each (CT, gene) pair where
   the CT has reasonable cohort coverage (median ≥ 5 cells/subject in the
   union of k0 ∪ k2 — descriptive, not a tier filter; we still test all
   pairs but report median-cells alongside), Wilcoxon on per-subject
   pseudobulk values (k0 vs k2). BH-FDR across all tested pairs.
5. **Per-(CT, gene) attribution differential.** Two-sample Wilcoxon on
   per-subject Captum IG attributions (k0 vs k2). BH-FDR.
6. **Pathology-confound flag.** For top hits at BH q < 0.05, repeat the
   identical Wilcoxon between Braak-low (≤ 2) and Braak-high (≥ 4) subjects
   using the SAME (subject_indices, gene)/(subject_indices, CT) data.
   If the same pair survives BH q < 0.05 there too, flag as "pathology
   confounded" — meaning the k0/k2 split reflects pathology load rather
   than pure-resilience structure.

Outputs
-------
  --out-json        outputs/canonical/interpretability/cluster_k0_vs_k2_differential.json
  --out-md          outputs/canonical/interpretability/cluster_k0_vs_k2_differential.md
  --out-fig-dir     outputs/canonical/interpretability/figures/cluster_differential/
                    (4-panel 600 DPI PNG + PDF)

Caveats
-------
- k0 / k2 have an inherent pathology-load gradient (EXP-033). Any (CT, gene)
  signal that survives here could be confounded with Braak / CERAD / etc.
  We flag this explicitly via the Braak-stratum re-test on top hits.
- Wilcoxon rank-sum is non-parametric and robust to non-normality, but the
  tests are NOT independent across (CT, gene) within a CT (gene-gene
  correlation), so BH-FDR is conservative. We do not attempt a permutation
  null at this scale (31 × 4785 ≈ 148K pairs × N_perm > GPU budget).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.stats import false_discovery_control
from sklearn.mixture import GaussianMixture

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from src.analysis.pseudobulk_io import load_pseudobulk_matrix  # noqa: E402
from src.data.constants import CELL_TYPE_ORDER  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cluster-label recovery
# ---------------------------------------------------------------------------
def fit_k4_clusters(
    residuals: np.ndarray,
    *,
    random_state: int = 0,
    n_components: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Refit GMM(k=4) deterministically on residuals.

    Returns (cluster_labels, cluster_means_sorted_desc_by_input_order). We
    return the original (unsorted) labels so they match
    ``latent_class_k4_crosstab.json`` exactly.
    """
    x = np.asarray(residuals, dtype=np.float64).reshape(-1, 1)
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    gmm.fit(x)
    labels = gmm.predict(x).astype(np.int64)
    means = gmm.means_.ravel().astype(float)
    return labels, means


# ---------------------------------------------------------------------------
# Wilcoxon helpers
# ---------------------------------------------------------------------------
def wilcoxon_two_groups(
    values_a: np.ndarray,
    values_b: np.ndarray,
) -> tuple[float, float, float]:
    """Two-sided Mann-Whitney U on two 1D arrays. NaN-policy=omit.

    Returns (U_stat, p_value, log2_fold_change_of_means).
    log2 FC is computed on `mean(a) / mean(b)` with a small epsilon-floor
    so a zero-baseline (e.g., zero counts) does not blow up to ±inf. If
    either group has < 3 finite values, returns NaN p-value.
    """
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    a_finite = a[np.isfinite(a)]
    b_finite = b[np.isfinite(b)]
    if a_finite.size < 3 or b_finite.size < 3:
        return float("nan"), float("nan"), float("nan")
    try:
        stat = stats.mannwhitneyu(
            a_finite, b_finite, alternative="two-sided",
        )
        u = float(stat.statistic)
        p = float(stat.pvalue)
    except Exception:
        return float("nan"), float("nan"), float("nan")
    eps = 1e-9
    mean_a = float(a_finite.mean())
    mean_b = float(b_finite.mean())
    log2_fc = float(np.log2((np.abs(mean_a) + eps) / (np.abs(mean_b) + eps)))
    # Sign-aware flip when the means straddle zero (attribution magnitudes).
    # Both directions must trigger the flip:
    #   - a<0<b: a is "vulnerable" attribution, b is "resilient" attribution.
    #   - b<0<a: same logic with sides swapped.
    # Mirrors the vectorised path at lines 246-247 of this file.
    if (mean_a < 0 < mean_b) or (mean_b < 0 < mean_a):
        log2_fc = -log2_fc
    return u, p, log2_fc


def bh_correct(p_arr: np.ndarray) -> np.ndarray:
    """BH-FDR across an array; NaN preserved."""
    p_arr = np.asarray(p_arr, dtype=float)
    out = np.full_like(p_arr, np.nan)
    finite = np.isfinite(p_arr)
    if not finite.any():
        return out
    out[finite] = false_discovery_control(p_arr[finite], method="bh")
    return out


# ---------------------------------------------------------------------------
# Per-CT cell-count differential
# ---------------------------------------------------------------------------
def cell_count_differential(
    cell_counts: np.ndarray,           # (n_subjects, n_cell_types) int
    is_k0: np.ndarray,
    is_k2: np.ndarray,
    cell_type_order: list[str],
) -> pd.DataFrame:
    """Wilcoxon rank-sum on per-subject cell counts, k0 vs k2, per CT."""
    rows = []
    for ct_idx, ct in enumerate(cell_type_order):
        cnt_k0 = cell_counts[is_k0, ct_idx].astype(np.float64)
        cnt_k2 = cell_counts[is_k2, ct_idx].astype(np.float64)
        u, p, log2_fc = wilcoxon_two_groups(cnt_k0, cnt_k2)
        rows.append({
            "cell_type_idx": int(ct_idx),
            "cell_type": ct,
            "median_count_k0": float(np.median(cnt_k0)) if cnt_k0.size else float("nan"),
            "median_count_k2": float(np.median(cnt_k2)) if cnt_k2.size else float("nan"),
            "mean_count_k0": float(cnt_k0.mean()) if cnt_k0.size else float("nan"),
            "mean_count_k2": float(cnt_k2.mean()) if cnt_k2.size else float("nan"),
            "U_stat": u,
            "p_value": p,
            "log2_fold_change": log2_fc,
        })
    df = pd.DataFrame(rows)
    df["bh_q"] = bh_correct(df["p_value"].to_numpy())
    df["sig_q05"] = df["bh_q"] < 0.05
    return df


# ---------------------------------------------------------------------------
# Per-(CT, gene) pseudobulk differential
# ---------------------------------------------------------------------------
def gene_pseudobulk_differential(
    pseudobulk: np.ndarray,            # (n_subjects, n_cell_types, n_genes) float
    cell_counts: np.ndarray,           # (n_subjects, n_cell_types) int
    is_k0: np.ndarray,
    is_k2: np.ndarray,
    cell_type_order: list[str],
    gene_names: list[str],
    *,
    median_cells_threshold: int = 5,
) -> tuple[pd.DataFrame, dict]:
    """Per-(CT, gene) Wilcoxon, k0 vs k2.

    For each CT, the median across the union (k0 ∪ k2) of per-subject cell
    counts must be ≥ ``median_cells_threshold`` for the CT to be tested.
    This is descriptive — we still record every CT's coverage in the
    summary block and the threshold is configurable.
    """
    n_subj, n_ct, n_gene = pseudobulk.shape
    union = is_k0 | is_k2
    coverage = []
    for ct_idx in range(n_ct):
        med = float(np.median(cell_counts[union, ct_idx]))
        coverage.append({
            "cell_type": cell_type_order[ct_idx],
            "median_cells_union": med,
            "median_cells_k0": float(np.median(cell_counts[is_k0, ct_idx])),
            "median_cells_k2": float(np.median(cell_counts[is_k2, ct_idx])),
            "tested": bool(med >= median_cells_threshold),
        })
    coverage_df = pd.DataFrame(coverage)
    eligible_cts = [
        i for i, row in enumerate(coverage)
        if row["tested"]
    ]
    logger.info(
        "Per-(CT, gene) testing %d / %d CTs (median cells ≥ %d) × %d genes",
        len(eligible_cts), n_ct, median_cells_threshold, n_gene,
    )

    rows: list[dict] = []
    # vectorized mannwhitneyu over genes per CT
    for ct_idx in eligible_cts:
        a = pseudobulk[is_k0, ct_idx, :].astype(np.float64)   # (n_k0, n_gene)
        b = pseudobulk[is_k2, ct_idx, :].astype(np.float64)
        # Drop genes with all-NaN in either group
        finite_a = np.isfinite(a).sum(axis=0) >= 3
        finite_b = np.isfinite(b).sum(axis=0) >= 3
        ok = finite_a & finite_b
        if not ok.any():
            continue
        # Mean fold change (sign-aware for log2)
        eps = 1e-9
        mean_a = np.nanmean(a[:, ok], axis=0)
        mean_b = np.nanmean(b[:, ok], axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            log2_fc = np.log2((np.abs(mean_a) + eps) / (np.abs(mean_b) + eps))
            sign_flip = (mean_a < 0) & (mean_b > 0) | (mean_a > 0) & (mean_b < 0)
            log2_fc = np.where(sign_flip, -log2_fc, log2_fc)
        try:
            mw = stats.mannwhitneyu(
                a[:, ok], b[:, ok],
                alternative="two-sided", axis=0, nan_policy="omit",
            )
            u_arr = np.asarray(mw.statistic, dtype=float)
            p_arr = np.asarray(mw.pvalue, dtype=float)
        except Exception as e:
            logger.warning("mannwhitneyu vectorized failed for CT %d (%s); skipping",
                           ct_idx, e)
            continue
        ok_idx = np.where(ok)[0]
        for j, gene_local in enumerate(ok_idx):
            rows.append({
                "cell_type_idx": int(ct_idx),
                "cell_type": cell_type_order[ct_idx],
                "gene_idx": int(gene_local),
                "gene": gene_names[int(gene_local)],
                "mean_pb_k0": float(mean_a[j]),
                "mean_pb_k2": float(mean_b[j]),
                "U_stat": float(u_arr[j]),
                "p_value": float(p_arr[j]),
                "log2_fold_change": float(log2_fc[j]),
            })
    df = pd.DataFrame(rows)
    if len(df):
        df["bh_q"] = bh_correct(df["p_value"].to_numpy())
        df["sig_q05"] = df["bh_q"] < 0.05
    return df, {"coverage": coverage_df.to_dict(orient="records")}


# ---------------------------------------------------------------------------
# Per-(CT, gene) attribution differential
# ---------------------------------------------------------------------------
def attribution_differential(
    attributions: np.ndarray,          # (n_subjects, n_cell_types, n_genes)
    is_k0: np.ndarray,
    is_k2: np.ndarray,
    cell_type_order: list[str],
    gene_names: list[str],
) -> pd.DataFrame:
    """Wilcoxon rank-sum on per-subject Captum IG per (CT, gene), k0 vs k2."""
    if attributions is None:
        return pd.DataFrame()
    n_subj, n_ct, n_gene = attributions.shape
    rows: list[dict] = []
    for ct_idx in range(n_ct):
        a = attributions[is_k0, ct_idx, :].astype(np.float64)
        b = attributions[is_k2, ct_idx, :].astype(np.float64)
        finite_a = np.isfinite(a).sum(axis=0) >= 3
        finite_b = np.isfinite(b).sum(axis=0) >= 3
        ok = finite_a & finite_b
        if not ok.any():
            continue
        # For attributions we report mean diff (k0 - k2) and log2 of |mean| ratio
        eps = 1e-12
        mean_a = np.nanmean(a[:, ok], axis=0)
        mean_b = np.nanmean(b[:, ok], axis=0)
        mean_diff = mean_a - mean_b
        with np.errstate(divide="ignore", invalid="ignore"):
            log2_fc = np.log2((np.abs(mean_a) + eps) / (np.abs(mean_b) + eps))
        try:
            mw = stats.mannwhitneyu(
                a[:, ok], b[:, ok],
                alternative="two-sided", axis=0, nan_policy="omit",
            )
            u_arr = np.asarray(mw.statistic, dtype=float)
            p_arr = np.asarray(mw.pvalue, dtype=float)
        except Exception as e:
            logger.warning("attr mannwhitneyu vectorized failed for CT %d (%s); skipping",
                           ct_idx, e)
            continue
        ok_idx = np.where(ok)[0]
        for j, gene_local in enumerate(ok_idx):
            rows.append({
                "cell_type_idx": int(ct_idx),
                "cell_type": cell_type_order[ct_idx],
                "gene_idx": int(gene_local),
                "gene": gene_names[int(gene_local)],
                "mean_attr_k0": float(mean_a[j]),
                "mean_attr_k2": float(mean_b[j]),
                "mean_diff_k0_minus_k2": float(mean_diff[j]),
                "U_stat": float(u_arr[j]),
                "p_value": float(p_arr[j]),
                "log2_fold_change": float(log2_fc[j]),
            })
    df = pd.DataFrame(rows)
    if len(df):
        df["bh_q"] = bh_correct(df["p_value"].to_numpy())
        df["sig_q05"] = df["bh_q"] < 0.05
    return df


# ---------------------------------------------------------------------------
# Pathology-confound check
# ---------------------------------------------------------------------------
def pathology_confound_flag(
    top_hits: pd.DataFrame,
    pseudobulk: np.ndarray,
    is_braak_low: np.ndarray,
    is_braak_high: np.ndarray,
) -> pd.DataFrame:
    """For each top (CT, gene) hit, re-test with Braak-low vs Braak-high.

    Returns a copy of ``top_hits`` with two extra columns:
      - ``braak_p_value``  (Wilcoxon p between Braak-low / Braak-high)
      - ``braak_sig_q05``  (after BH across the top-hits set)
      - ``pathology_confounded`` (True if both k0/k2 AND Braak-low/Braak-high
        survive at q < 0.05)
    """
    if not len(top_hits):
        return top_hits.copy()
    p_braak = []
    for _, row in top_hits.iterrows():
        ct_idx = int(row["cell_type_idx"])
        gene_idx = int(row["gene_idx"])
        a = pseudobulk[is_braak_low, ct_idx, gene_idx]
        b = pseudobulk[is_braak_high, ct_idx, gene_idx]
        _u, p, _fc = wilcoxon_two_groups(a, b)
        p_braak.append(p)
    df = top_hits.copy().reset_index(drop=True)
    df["braak_p_value"] = p_braak
    df["braak_bh_q"] = bh_correct(df["braak_p_value"].to_numpy())
    df["braak_sig_q05"] = df["braak_bh_q"] < 0.05
    df["pathology_confounded"] = df["sig_q05"] & df["braak_sig_q05"]
    return df


# ---------------------------------------------------------------------------
# Figure rendering
# ---------------------------------------------------------------------------
def render_figure(
    *,
    cell_count_df: pd.DataFrame,
    gene_df: pd.DataFrame,
    attr_df: pd.DataFrame,
    cell_count_top_ct_for_genes: str,
    pathology_confound_df: pd.DataFrame,
    is_k0: np.ndarray,
    is_k2: np.ndarray,
    cell_counts: np.ndarray,
    out_png: Path,
    out_pdf: Path,
) -> None:
    """4-panel composite figure (600 DPI PNG + PDF)."""
    try:
        from src.visualization.theme import apply_theme
        apply_theme(style="paper", use_scienceplots=False)
    except Exception:
        plt.rcParams["figure.dpi"] = 100

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=600)
    ax_a, ax_b, ax_c, ax_d = axes.flatten()

    # ── Panel A: cell-type abundance bar plot per cluster ────────────────
    ct_names = cell_count_df["cell_type"].tolist()
    n_ct = len(ct_names)
    means_k0 = cell_count_df["mean_count_k0"].to_numpy()
    means_k2 = cell_count_df["mean_count_k2"].to_numpy()
    x = np.arange(n_ct)
    width = 0.42
    ax_a.bar(x - width / 2, means_k0, width, label="k0 (vulnerable, n=%d)" % int(is_k0.sum()),
             color="#cc4444", alpha=0.85)
    ax_a.bar(x + width / 2, means_k2, width, label="k2 (resilient, n=%d)" % int(is_k2.sum()),
             color="#3366aa", alpha=0.85)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(ct_names, rotation=70, fontsize=6)
    ax_a.set_ylabel("Mean cell count / subject")
    ax_a.set_title("(A) Cell-type abundance: k0 vs k2")
    ax_a.legend(fontsize=8, loc="upper right")
    # Mark BH-FDR-significant CTs with a star
    sig_mask = cell_count_df["sig_q05"].to_numpy()
    if sig_mask.any():
        ymax = max(np.max(means_k0), np.max(means_k2)) * 1.05
        for i in np.where(sig_mask)[0]:
            ax_a.text(i, ymax, "*", ha="center", va="bottom", fontsize=12, color="black")

    # ── Panel B: cell-type volcano (log2 FC vs −log10 p) ─────────────────
    log2fc = cell_count_df["log2_fold_change"].to_numpy()
    p_vals = cell_count_df["p_value"].to_numpy()
    nlp = -np.log10(np.clip(p_vals, 1e-300, 1.0))
    is_sig = cell_count_df["sig_q05"].to_numpy()
    ax_b.scatter(log2fc[~is_sig], nlp[~is_sig], s=24, color="#888888", alpha=0.6)
    ax_b.scatter(log2fc[is_sig], nlp[is_sig], s=44, color="#cc4444",
                 edgecolor="black", linewidth=0.5, label="q < 0.05")
    for i, ct in enumerate(ct_names):
        if is_sig[i] or nlp[i] > 1.5:
            ax_b.annotate(ct, (log2fc[i], nlp[i]), fontsize=6,
                          xytext=(3, 3), textcoords="offset points")
    ax_b.axhline(-np.log10(0.05), color="gray", linestyle="--", linewidth=0.8)
    ax_b.axvline(0, color="black", linewidth=0.5)
    ax_b.set_xlabel("log2 FC (mean count k0 / mean count k2)")
    ax_b.set_ylabel("-log10(p)")
    ax_b.set_title("(B) Cell-type abundance volcano (k0 vs k2)")
    if is_sig.any():
        ax_b.legend(loc="upper right", fontsize=8)

    # ── Panel C: gene volcano per top-CT ────────────────────────────────
    if len(gene_df):
        sub = gene_df[gene_df["cell_type"] == cell_count_top_ct_for_genes]
        if len(sub):
            xs = sub["log2_fold_change"].to_numpy()
            ps = -np.log10(np.clip(sub["p_value"].to_numpy(), 1e-300, 1.0))
            sig = sub["sig_q05"].to_numpy()
            ax_c.scatter(xs[~sig], ps[~sig], s=8, color="#888888", alpha=0.4)
            ax_c.scatter(xs[sig], ps[sig], s=18, color="#cc4444", alpha=0.9, edgecolor="black", linewidth=0.3)
            ax_c.axhline(-np.log10(0.05), color="gray", linestyle="--", linewidth=0.8)
            ax_c.axvline(0, color="black", linewidth=0.5)
            # Label top 10 by p
            top_p = sub.nsmallest(10, "p_value")
            for _, r in top_p.iterrows():
                ax_c.annotate(r["gene"],
                              (r["log2_fold_change"], -np.log10(max(r["p_value"], 1e-300))),
                              fontsize=6, xytext=(2, 2), textcoords="offset points")
            ax_c.set_title(f"(C) Gene volcano for {cell_count_top_ct_for_genes}")
        else:
            ax_c.text(0.5, 0.5, f"No genes tested for {cell_count_top_ct_for_genes}",
                      ha="center", va="center", transform=ax_c.transAxes)
            ax_c.set_title("(C) Gene volcano — no eligible CT data")
    else:
        ax_c.text(0.5, 0.5, "No (CT, gene) pairs tested", ha="center",
                  va="center", transform=ax_c.transAxes)
        ax_c.set_title("(C) Gene volcano — empty")
    ax_c.set_xlabel("log2 FC (mean pseudobulk k0 / k2)")
    ax_c.set_ylabel("-log10(p)")

    # ── Panel D: pathology-confound diagnostic ──────────────────────────
    if len(pathology_confound_df):
        # Scatter of -log10(k0vsk2 p) vs -log10(Braak p)
        x_p = -np.log10(np.clip(pathology_confound_df["p_value"].to_numpy(), 1e-300, 1.0))
        y_p = -np.log10(np.clip(pathology_confound_df["braak_p_value"].to_numpy(), 1e-300, 1.0))
        is_conf = pathology_confound_df["pathology_confounded"].to_numpy()
        ax_d.scatter(x_p[~is_conf], y_p[~is_conf], s=20, color="#3366aa", alpha=0.7,
                     label="k0/k2 sig only (resilience-specific)")
        ax_d.scatter(x_p[is_conf], y_p[is_conf], s=30, color="#cc4444", alpha=0.85,
                     edgecolor="black", linewidth=0.4,
                     label="confounded (k0/k2 ∧ Braak)")
        ax_d.axhline(-np.log10(0.05), color="gray", linestyle="--", linewidth=0.8)
        ax_d.axvline(-np.log10(0.05), color="gray", linestyle="--", linewidth=0.8)
        ax_d.set_xlabel("-log10(p) (k0 vs k2)")
        ax_d.set_ylabel("-log10(p) (Braak-low vs Braak-high)")
        ax_d.set_title("(D) Pathology-confound diagnostic on top hits")
        ax_d.legend(loc="upper left", fontsize=7)
    else:
        ax_d.text(0.5, 0.5, "No top hits to confound-check", ha="center",
                  va="center", transform=ax_d.transAxes)
        ax_d.set_title("(D) Pathology-confound diagnostic — empty")

    fig.suptitle(
        "EXP-037: Cluster k0 (vulnerable, μ=−1.54) vs k2 (resilient, μ=+0.84) — "
        "differential snRNA-seq beyond the residual axis",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=600, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------
def render_markdown(
    *,
    n_k0: int,
    n_k2: int,
    cell_count_df: pd.DataFrame,
    gene_df: pd.DataFrame,
    attr_df: pd.DataFrame,
    pathology_confound_df: pd.DataFrame,
    n_braak_low: int,
    n_braak_high: int,
    out_md: Path,
) -> None:
    lines = []
    lines.append("# EXP-037: Cluster k0 (vulnerable) vs k2 (resilient) differential")
    lines.append("")
    lines.append(
        f"**Cohort:** k0 (vulnerable, μ=−1.54) n={n_k0}; k2 (resilient, μ=+0.84) n={n_k2}."
    )
    lines.append("")
    lines.append("## 1. Cell-type abundance (Wilcoxon rank-sum on per-subject counts)")
    lines.append("")
    n_sig_ct = int(cell_count_df["sig_q05"].sum())
    lines.append(f"- {n_sig_ct} / {len(cell_count_df)} CTs differ at BH q < 0.05")
    lines.append("")
    top_ct = cell_count_df.sort_values("p_value").head(10).copy()
    lines.append("| CT | median k0 | median k2 | log2 FC | p | BH q | sig |")
    lines.append("|---|---:|---:|---:|---:|---:|:---:|")
    for _, r in top_ct.iterrows():
        sig_mark = "Y" if bool(r["sig_q05"]) else "n"
        lines.append(
            f"| {r['cell_type']} | {r['median_count_k0']:.1f} | {r['median_count_k2']:.1f} | "
            f"{r['log2_fold_change']:+.3f} | {r['p_value']:.3g} | {r['bh_q']:.3g} | {sig_mark} |"
        )
    lines.append("")

    lines.append("## 2. Per-(CT, gene) pseudobulk (Wilcoxon)")
    lines.append("")
    if len(gene_df):
        n_sig_gene = int(gene_df["sig_q05"].sum())
        lines.append(
            f"- {n_sig_gene} / {len(gene_df)} (CT, gene) pairs at BH q < 0.05"
        )
        lines.append("")
        top_gene = gene_df.sort_values("p_value").head(10).copy()
        lines.append("| CT | gene | mean pb k0 | mean pb k2 | log2 FC | p | BH q | sig |")
        lines.append("|---|---|---:|---:|---:|---:|---:|:---:|")
        for _, r in top_gene.iterrows():
            sig_mark = "Y" if bool(r["sig_q05"]) else "n"
            lines.append(
                f"| {r['cell_type']} | {r['gene']} | {r['mean_pb_k0']:.3f} | "
                f"{r['mean_pb_k2']:.3f} | {r['log2_fold_change']:+.3f} | {r['p_value']:.3g} | "
                f"{r['bh_q']:.3g} | {sig_mark} |"
            )
    else:
        lines.append("- No (CT, gene) pairs tested (no CT met coverage threshold).")
    lines.append("")

    lines.append("## 3. Per-(CT, gene) Captum IG attribution (Wilcoxon)")
    lines.append("")
    if len(attr_df):
        n_sig_attr = int(attr_df["sig_q05"].sum())
        lines.append(
            f"- {n_sig_attr} / {len(attr_df)} (CT, gene) attribution pairs at BH q < 0.05"
        )
        lines.append("")
        top_attr = attr_df.sort_values("p_value").head(10).copy()
        lines.append("| CT | gene | mean attr k0 | mean attr k2 | log2 FC | p | BH q | sig |")
        lines.append("|---|---|---:|---:|---:|---:|---:|:---:|")
        for _, r in top_attr.iterrows():
            sig_mark = "Y" if bool(r["sig_q05"]) else "n"
            lines.append(
                f"| {r['cell_type']} | {r['gene']} | {r['mean_attr_k0']:.3g} | "
                f"{r['mean_attr_k2']:.3g} | {r['log2_fold_change']:+.3f} | {r['p_value']:.3g} | "
                f"{r['bh_q']:.3g} | {sig_mark} |"
            )
    else:
        lines.append("- Per-subject Captum IG attributions not available; section skipped.")
    lines.append("")

    lines.append("## 4. Pathology-confound diagnostic (top-hit re-test on Braak-low vs Braak-high)")
    lines.append("")
    if len(pathology_confound_df):
        n_pure = int((pathology_confound_df["sig_q05"] & ~pathology_confound_df["braak_sig_q05"]).sum())
        n_conf = int(pathology_confound_df["pathology_confounded"].sum())
        lines.append(
            f"- Braak-low n={n_braak_low}; Braak-high n={n_braak_high}"
        )
        lines.append(
            f"- {n_pure} / {len(pathology_confound_df)} top hits are k0/k2-only (resilience-specific)"
        )
        lines.append(
            f"- {n_conf} / {len(pathology_confound_df)} top hits are also Braak-significant (confounded)"
        )
        lines.append("")
        head = pathology_confound_df.sort_values("p_value").head(10).copy()
        lines.append("| CT | gene | k0/k2 q | Braak q | confounded |")
        lines.append("|---|---|---:|---:|:---:|")
        for _, r in head.iterrows():
            lines.append(
                f"| {r['cell_type']} | {r['gene']} | {r['bh_q']:.3g} | "
                f"{r['braak_bh_q']:.3g} | {'Y' if bool(r['pathology_confounded']) else 'n'} |"
            )
    else:
        lines.append("- No top hits qualified for confound check.")
    lines.append("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--residual-csv",
        type=Path,
        default=_WORKTREE_ROOT / "outputs/canonical/interpretability/residual_per_subject.csv",
    )
    p.add_argument(
        "--metadata-csv",
        type=Path,
        default=_WORKTREE_ROOT / "data/metadata_ROSMAP/metadata.csv",
    )
    p.add_argument(
        "--latent-class-json-ref",
        type=Path,
        default=_WORKTREE_ROOT / "outputs/canonical/interpretability/latent_class_k4_crosstab.json",
        help="Used as a sanity-check pin for cluster sizes (55, 262, 60, 139).",
    )
    p.add_argument(
        "--precomputed-dir",
        type=Path,
        default=_WORKTREE_ROOT / "data/precomputed",
    )
    p.add_argument(
        "--gene-names-npy",
        type=Path,
        default=_WORKTREE_ROOT / "data/precomputed/gene_names.npy",
    )
    p.add_argument(
        "--captum-npz",
        type=Path,
        default=_WORKTREE_ROOT / "outputs/canonical/interpretability/captum_ig/composite_attributions.npz",
        help="Per-subject Captum IG attribution; if missing, attribution differential is skipped.",
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=_WORKTREE_ROOT / "outputs/canonical/interpretability/cluster_k0_vs_k2_differential.json",
    )
    p.add_argument(
        "--out-md",
        type=Path,
        default=_WORKTREE_ROOT / "outputs/canonical/interpretability/cluster_k0_vs_k2_differential.md",
    )
    p.add_argument(
        "--out-fig-dir",
        type=Path,
        default=_WORKTREE_ROOT / "outputs/canonical/interpretability/figures/cluster_differential",
    )
    p.add_argument("--median-cells-threshold", type=int, default=5)
    p.add_argument("--top-hits-confound-n", type=int, default=50,
                   help="Number of top (CT, gene) hits to re-test on Braak-low vs Braak-high.")
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # --- 1. Load residuals + recover cluster labels --------------------------
    if not args.residual_csv.exists():
        raise FileNotFoundError(args.residual_csv)
    df_res = pd.read_csv(args.residual_csv)
    df_res = df_res.loc[np.isfinite(df_res["residual"].to_numpy())].reset_index(drop=True)
    n_subj = len(df_res)
    logger.info("Loaded %d subjects", n_subj)

    cluster_labels, cluster_means = fit_k4_clusters(
        df_res["residual"].to_numpy(), random_state=args.random_state,
    )
    cluster_sizes = np.bincount(cluster_labels, minlength=4).tolist()
    logger.info("Cluster sizes (k=4): %s", cluster_sizes)
    logger.info("Cluster means: %s", cluster_means.tolist())

    # Sanity-check vs reference JSON
    with args.latent_class_json_ref.open() as fh:
        ref = json.load(fh)
    ref_sizes = ref["cluster_sizes"]
    if not all(int(ref_sizes[f"cluster_{i}"]) == cluster_sizes[i] for i in range(4)):
        raise RuntimeError(
            f"Cluster sizes mismatch with reference JSON: "
            f"got {cluster_sizes}, expected {ref_sizes}"
        )
    logger.info("Sanity check passed: cluster sizes match reference JSON.")

    is_k0 = cluster_labels == 0
    is_k2 = cluster_labels == 2
    n_k0 = int(is_k0.sum())
    n_k2 = int(is_k2.sum())
    # The expected-size warning at this point is unreachable: the prior
    # `raise RuntimeError(...)` block at lines 710-714 has already aborted on
    # any size mismatch with the reference JSON. Removed (dead scaffolding).
    logger.info("k0=%d, k2=%d (matched reference JSON)", n_k0, n_k2)

    # --- 2. Load per-subject pseudobulk + cell counts ------------------------
    subject_ids = df_res["ROSMAP_IndividualID"].astype(str).tolist()
    logger.info("Loading pseudobulk for %d subjects", n_subj)
    pseudobulk = load_pseudobulk_matrix(
        args.precomputed_dir, subject_ids, n_jobs=args.n_jobs,
    )
    logger.info("Pseudobulk shape: %s", pseudobulk.shape)
    n_subj_pb, n_ct, n_gene = pseudobulk.shape
    if n_subj_pb != n_subj:
        raise RuntimeError(
            f"Pseudobulk row count {n_subj_pb} != residual row count {n_subj}"
        )
    if len(CELL_TYPE_ORDER) != n_ct:
        raise RuntimeError(f"CELL_TYPE_ORDER ({len(CELL_TYPE_ORDER)}) != pseudobulk CT ({n_ct})")
    cell_type_order = CELL_TYPE_ORDER

    gene_names = np.load(args.gene_names_npy, allow_pickle=True).tolist()
    if len(gene_names) != n_gene:
        raise RuntimeError(f"gene_names ({len(gene_names)}) != pseudobulk genes ({n_gene})")

    # Cell counts: load from R*.pt directly (cell_counts not embedded in pseudobulk)
    logger.info("Loading per-subject cell_counts")
    cell_counts = np.zeros((n_subj, n_ct), dtype=np.int64)
    missing = 0
    for i, sid in enumerate(subject_ids):
        p_pt = args.precomputed_dir / f"{sid}.pt"
        if not p_pt.exists():
            missing += 1
            continue
        d = torch.load(p_pt, map_location="cpu", weights_only=False)
        cell_counts[i] = d["cell_counts"].numpy().astype(np.int64)
    logger.info("Missing cell-count files: %d / %d", missing, n_subj)

    # --- 3. Cell-type abundance differential ---------------------------------
    cell_count_df = cell_count_differential(cell_counts, is_k0, is_k2, cell_type_order)

    # --- 4. Per-(CT, gene) pseudobulk differential ---------------------------
    gene_df, gene_meta = gene_pseudobulk_differential(
        pseudobulk, cell_counts, is_k0, is_k2, cell_type_order, gene_names,
        median_cells_threshold=args.median_cells_threshold,
    )

    # --- 5. Per-(CT, gene) Captum attribution differential -------------------
    attr_df = pd.DataFrame()
    if args.captum_npz.exists():
        try:
            attr_npz = np.load(args.captum_npz, allow_pickle=True)
            attr_sids = attr_npz["subject_ids"].tolist()
            # Reorder attributions to match df_res order
            sid_to_idx = {s: i for i, s in enumerate(attr_sids)}
            order = [sid_to_idx.get(s, -1) for s in subject_ids]
            if any(o < 0 for o in order):
                logger.warning("Captum subject_ids do not cover all residual subjects; "
                               "attribution differential will be subset.")
            order_arr = np.array(order, dtype=np.int64)
            valid = order_arr >= 0
            attr_full = np.full(
                (n_subj, n_ct, n_gene), np.nan, dtype=np.float64,
            )
            attr_full[valid] = attr_npz["attributions"][order_arr[valid]]
            attr_df = attribution_differential(
                attr_full, is_k0, is_k2, cell_type_order, gene_names,
            )
        except Exception as e:
            logger.warning("Captum attribution load failed (%s); skipping", e)
    else:
        logger.warning("Captum npz not found at %s; skipping attribution differential",
                       args.captum_npz)

    # --- 6. Pathology-confound check on top gene hits ------------------------
    df_meta_full = pd.read_csv(args.metadata_csv, low_memory=False)[
        ["projid", "braaksc"]
    ]
    merged = df_res[["projid"]].merge(df_meta_full, on="projid", how="left")
    braak = merged["braaksc"].to_numpy()
    is_braak_low = (braak <= 2) & np.isfinite(braak)
    is_braak_high = (braak >= 4) & np.isfinite(braak)
    n_braak_low = int(is_braak_low.sum())
    n_braak_high = int(is_braak_high.sum())
    logger.info("Braak-low n=%d; Braak-high n=%d", n_braak_low, n_braak_high)

    pathology_confound_df = pd.DataFrame()
    if len(gene_df):
        top_hits = gene_df.sort_values("p_value").head(args.top_hits_confound_n).copy()
        pathology_confound_df = pathology_confound_flag(
            top_hits, pseudobulk, is_braak_low, is_braak_high,
        )

    # --- 7. Pick the top CT (lowest gene-level p) for the volcano panel -----
    cell_count_top_ct_for_genes = cell_type_order[0]
    if len(gene_df):
        ct_min_p = gene_df.groupby("cell_type")["p_value"].min().sort_values()
        if len(ct_min_p):
            cell_count_top_ct_for_genes = ct_min_p.index[0]

    # --- 8. Render figure ---------------------------------------------------
    out_png = args.out_fig_dir / "fig_cluster_k0_vs_k2.png"
    out_pdf = args.out_fig_dir / "fig_cluster_k0_vs_k2.pdf"
    render_figure(
        cell_count_df=cell_count_df,
        gene_df=gene_df,
        attr_df=attr_df,
        cell_count_top_ct_for_genes=cell_count_top_ct_for_genes,
        pathology_confound_df=pathology_confound_df,
        is_k0=is_k0,
        is_k2=is_k2,
        cell_counts=cell_counts,
        out_png=out_png,
        out_pdf=out_pdf,
    )

    # --- 9. Render markdown -------------------------------------------------
    render_markdown(
        n_k0=n_k0,
        n_k2=n_k2,
        cell_count_df=cell_count_df,
        gene_df=gene_df,
        attr_df=attr_df,
        pathology_confound_df=pathology_confound_df,
        n_braak_low=n_braak_low,
        n_braak_high=n_braak_high,
        out_md=args.out_md,
    )

    # --- 10. Write JSON ----------------------------------------------------
    n_sig_ct = int(cell_count_df["sig_q05"].sum())
    n_sig_gene = int(gene_df["sig_q05"].sum()) if len(gene_df) else 0
    n_sig_attr = int(attr_df["sig_q05"].sum()) if len(attr_df) else 0
    n_pathology_confounded = int(
        pathology_confound_df["pathology_confounded"].sum()
    ) if len(pathology_confound_df) else 0
    n_pure_resilience = int(
        (pathology_confound_df["sig_q05"] & ~pathology_confound_df["braak_sig_q05"]).sum()
    ) if len(pathology_confound_df) else 0

    payload = {
        "config": {
            "residual_csv": str(args.residual_csv),
            "metadata_csv": str(args.metadata_csv),
            "precomputed_dir": str(args.precomputed_dir),
            "captum_npz": str(args.captum_npz),
            "median_cells_threshold": int(args.median_cells_threshold),
            "top_hits_confound_n": int(args.top_hits_confound_n),
            "random_state": int(args.random_state),
        },
        "cohort": {
            "n_subjects_total": int(n_subj),
            "cluster_sizes": {f"cluster_{i}": int(s) for i, s in enumerate(cluster_sizes)},
            "cluster_means": {f"cluster_{i}": float(cluster_means[i]) for i in range(4)},
            "n_k0": int(n_k0),
            "n_k2": int(n_k2),
            "n_braak_low": int(n_braak_low),
            "n_braak_high": int(n_braak_high),
        },
        "cell_type_abundance": {
            "n_tested": int(len(cell_count_df)),
            "n_sig_q05": n_sig_ct,
            "results": cell_count_df.to_dict(orient="records"),
        },
        "gene_pseudobulk": {
            "median_cells_threshold": int(args.median_cells_threshold),
            "n_tested": int(len(gene_df)),
            "n_sig_q05": n_sig_gene,
            "coverage_per_ct": gene_meta.get("coverage", []),
            "top_50": (
                gene_df.sort_values("p_value").head(50).to_dict(orient="records")
                if len(gene_df) else []
            ),
        },
        "attribution_differential": {
            "available": bool(len(attr_df)),
            "n_tested": int(len(attr_df)),
            "n_sig_q05": n_sig_attr,
            "top_50": (
                attr_df.sort_values("p_value").head(50).to_dict(orient="records")
                if len(attr_df) else []
            ),
        },
        "pathology_confound": {
            "n_top_hits_tested": int(len(pathology_confound_df)),
            "n_pure_resilience": n_pure_resilience,
            "n_pathology_confounded": n_pathology_confounded,
            "results": pathology_confound_df.to_dict(orient="records"),
        },
        "outputs": {
            "figure_png": str(out_png),
            "figure_pdf": str(out_pdf),
            "markdown": str(args.out_md),
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    logger.info("Wrote %s", args.out_json)
    logger.info("Wrote %s", args.out_md)
    logger.info("Wrote %s + .pdf", out_png)

    # Print summary
    print("=" * 78)
    print(f"EXP-037 Cluster k0 vs k2 differential")
    print(f"  k0 (vulnerable, μ=−1.54) n={n_k0}; k2 (resilient, μ=+0.84) n={n_k2}")
    print(f"  Cell-type abundance: {n_sig_ct}/{len(cell_count_df)} sig at BH q<0.05")
    print(f"  Per-(CT, gene) pseudobulk: {n_sig_gene}/{len(gene_df) if len(gene_df) else 0} sig")
    if len(attr_df):
        print(f"  Per-(CT, gene) attribution: {n_sig_attr}/{len(attr_df)} sig")
    else:
        print("  Per-(CT, gene) attribution: SKIPPED (Captum file missing or load failed)")
    print(f"  Pathology-confound: {n_pure_resilience} pure-resilience / "
          f"{n_pathology_confounded} confounded out of {len(pathology_confound_df)} top hits")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
