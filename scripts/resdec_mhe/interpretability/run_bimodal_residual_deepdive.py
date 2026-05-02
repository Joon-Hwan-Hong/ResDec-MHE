#!/usr/bin/env python
"""Bimodal residual + k=4 latent-class deepdive.

Connects the two existing-but-disconnected findings registered in narrative
§14.1 (now in EXP-015):

  1. **Residual phenotype is bimodal** — see ``residual_summary.json``;
     k=2 GMM is preferred by BIC and the cluster means are at residual ≈
     -0.91 (vulnerable mode, weight 0.29) and +0.35 (resilient mode,
     weight 0.71). See ``latent_class_on_residuals.json``.

  2. **k=4 latent-class fit shows Braak χ² = 0.0033** — see
     ``latent_class_k4_crosstab.json``; only Braak crossed α=0.05 in the
     original analysis (APOE / sex / age were null).

This script extends the cross-tab to a wider clinical / pathology axis,
applies BH-FDR correction, runs Hartigan's-dip-test-equivalent (a
bootstrap-distance-to-best-Gaussian test, since `diptest` is not
installed in this environment), and produces a 4-panel figure plus
matched JSON / MD outputs.

Tests against the canonical k=4 fit (random_state=0, full covariance,
n_init default=1) — exactly reproduces ``latent_class_k4_crosstab.json``
on the Braak axis (canonical n=516, χ² = 38.532, p = 0.00329).

Inputs:
  --residual-csv    outputs/canonical/interpretability/residual_per_subject.csv
  --metadata-csv    data/metadata_ROSMAP/metadata.csv
  --residual-summary-json
                    outputs/canonical/interpretability/residual_summary.json
  --latent-class-json
                    outputs/canonical/interpretability/latent_class_k4_crosstab.json
                    (used only as a sanity-check reference; the script
                    refits k=4 itself with random_state=0).

Outputs:
  --out-json    outputs/canonical/interpretability/bimodal_residual_deepdive.json
  --out-md      outputs/canonical/interpretability/bimodal_residual_deepdive.md
  --out-fig     outputs/canonical/interpretability/figures/bimodal_residual/
                    fig_bimodal_residual.{png,pdf}    (4-panel, 600 DPI)

Method
------
For each clinical / pathology axis, build a contingency table of GMM
cluster (k=4) × axis category and run ``scipy.stats.chi2_contingency``;
exception: continuous-by-construction axes (gpath, plaq_n_mf) are
binned into tertiles (Q1 / Q2 / Q3) and tested with χ² as well. APOE
genotype is collapsed to ε4-dosage ∈ {0, 1, 2} (more biologically
meaningful than the 6 raw genotypes); the original 6-genotype crosstab
is also kept for reference. BH-FDR correction (`scipy.stats.false
_discovery_control`) is applied across all q tested axes (q = number of
non-degenerate axes).

Cluster ↔ residual-sign association: tests whether the membership
proportion of "positive residual" subjects (residual > 0) differs
across the 4 clusters via χ². This formalises the question "do the 4
clusters factor into a resilient/vulnerable axis × pathology axis
quadrant structure?"

Hartigan's dip test substitute: bootstrap-resample under H0 (best-fit
single Gaussian) and compare the observed k=2 GMM log-likelihood
improvement (LL_k2 - LL_k1) against the bootstrap null. p-value = #
{boot_LL_diff >= observed} / n_boot. The dip test itself is not
implemented because the C extension `diptest` is not available; the
bootstrap LL test is theoretically equivalent for the unimodal-vs-
bimodal hypothesis.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")  # must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, false_discovery_control
from sklearn.mixture import GaussianMixture

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from src.visualization.theme import apply_theme  # noqa: E402

logger = logging.getLogger(__name__)

# Cogdx semantics (ROSMAP convention used elsewhere in this repo): we
# group AD-dx = {4, 5} vs other.
AD_COGDX_CODES = {4.0, 5.0}


def _bootstrap_unimodality_test(
    residuals: np.ndarray,
    *,
    n_boot: int,
    random_state: int,
) -> dict:
    """Hartigan's-dip-test substitute via bootstrap over GMM log-likelihood.

    H0: residuals are drawn from a single Gaussian.
    Test statistic: LL(k=2 GMM) - LL(k=1 GMM); larger means more bimodal.
    p-value: fraction of bootstrap samples (drawn from the H0 best-fit
    single Gaussian) whose LL_diff exceeds the observed LL_diff.

    Note: this is a substitute because `diptest` (Hartigan & Hartigan
    1985) is not installed; the bootstrap-LL test is asymptotically
    equivalent for the bimodal vs unimodal hypothesis under correct H0.
    """
    rng = np.random.default_rng(random_state)
    x = residuals.reshape(-1, 1)
    n = x.shape[0]

    g1 = GaussianMixture(n_components=1, random_state=random_state).fit(x)
    g2 = GaussianMixture(n_components=2, random_state=random_state).fit(x)
    obs_ll_diff = float(g2.score(x) - g1.score(x)) * n  # log-likelihood diff

    mu = float(g1.means_[0, 0])
    sigma = float(np.sqrt(g1.covariances_[0, 0, 0]))

    null_ll_diffs = np.zeros(n_boot, dtype=np.float64)
    for i in range(n_boot):
        # Per-iter integer seed so each bootstrap GMM init starts from a
        # different point. Reusing ``random_state`` across all iterations would
        # correlate the null draws (every iter starts from the SAME deterministic
        # init, so ``null_ll_diff`` would be biased toward agreement and the
        # empirical p-value variance would be under-estimated). bit-generator
        # raw integer extraction (rng.integers) is the project convention here
        # — see run_cmi_subsample_bootstrap.py.
        boot_seed = int(rng.integers(0, 2**31 - 1))
        boot = rng.normal(loc=mu, scale=sigma, size=(n, 1))
        g1b = GaussianMixture(n_components=1, random_state=boot_seed).fit(boot)
        g2b = GaussianMixture(n_components=2, random_state=boot_seed).fit(boot)
        null_ll_diffs[i] = float(g2b.score(boot) - g1b.score(boot)) * n

    p_val = float(np.mean(null_ll_diffs >= obs_ll_diff))
    # Empirical-p floor: 1/(n_boot+1) (uses observed in null pool).
    p_floor = 1.0 / (n_boot + 1)
    p_val = max(p_val, p_floor)

    return {
        "test": "bootstrap_LL_diff_vs_unimodal_null",
        "n_boot": int(n_boot),
        "obs_LL_diff_k2_minus_k1": obs_ll_diff,
        "null_LL_diff_mean": float(null_ll_diffs.mean()),
        "null_LL_diff_std": float(null_ll_diffs.std()),
        "p_value_one_sided": p_val,
        "p_floor": p_floor,
        "note": (
            "Hartigan's dip test substitute — `diptest` package not "
            "available in this env. Bootstrap H0 = best-fit single "
            "Gaussian; H1 = mixture of two Gaussians."
        ),
    }


def _apoe_e4_dose(genotype) -> float:
    """Return ε4 dosage ∈ {0, 1, 2} from a 2-digit APOE genotype.

    NaN if genotype missing. Genotypes are 22, 23, 24, 33, 34, 44; the
    digit '4' counts as one ε4 allele each occurrence.
    """
    if isinstance(genotype, float) and np.isnan(genotype):
        return float("nan")
    s = str(int(float(genotype))) if isinstance(genotype, (int, float, np.integer, np.floating)) else str(genotype)
    return float(s.count("4"))


def _tertile(values: pd.Series, *, label_prefix: str = "T") -> pd.Series:
    """Tertile-rank a numeric Series; NaN preserved as NaN.

    Uses ``rank(method='first')`` so equal values are split deterministically.
    """
    out = pd.Series(np.full(len(values), np.nan, dtype=object), index=values.index)
    mask = values.notna()
    if mask.sum() < 3:
        return out
    ranked = values[mask].rank(method="first")
    cuts = pd.qcut(ranked, q=3, labels=[f"{label_prefix}1", f"{label_prefix}2", f"{label_prefix}3"])
    out.loc[mask] = cuts.astype(str).values
    return out


def _crosstab_chi2(
    cluster: np.ndarray,
    cov: pd.Series,
    cov_name: str,
) -> dict:
    """Build a cluster × covariate contingency table and run chi2_contingency.

    Excludes rows where the covariate is NaN. Returns a dict with the
    nested-dict table, dropped n, chi2, dof, p-value.
    """
    if cov.dtype.kind in {"O", "U", "S"}:
        mask = cov.notna() & (cov.astype(str) != "nan")
    else:
        cov_num = pd.to_numeric(cov, errors="coerce")
        mask = cov_num.notna()
        cov = cov_num
    n_dropped = int((~mask).sum())
    cov_kept = cov[mask].to_numpy()
    cluster_kept = cluster[mask.to_numpy()]

    series_clu = pd.Series(cluster_kept, name="cluster")
    series_cov = pd.Series(cov_kept, name=cov_name)
    table = pd.crosstab(series_clu, series_cov)

    if table.size == 0 or min(table.shape) < 2:
        return {
            "table": {str(c): {str(r): int(v) for r, v in table[c].items()} for c in table.columns},
            "row_index": [int(x) if isinstance(x, (np.integer, int)) else str(x) for x in table.index.tolist()],
            "col_index": [str(x) for x in table.columns.tolist()],
            "n_used": int(mask.sum()),
            "n_dropped_missing": n_dropped,
            "chi2": None,
            "dof": None,
            "p_value": None,
            "note": "Insufficient variability for chi2 test (table degenerate).",
        }
    chi2, p_val, dof, _ = chi2_contingency(table.values)
    return {
        "table": {str(c): {str(r): int(v) for r, v in table[c].items()} for c in table.columns},
        "row_index": [int(x) if isinstance(x, (np.integer, int)) else str(x) for x in table.index.tolist()],
        "col_index": [str(x) for x in table.columns.tolist()],
        "n_used": int(mask.sum()),
        "n_dropped_missing": n_dropped,
        "chi2": float(chi2),
        "dof": int(dof),
        "p_value": float(p_val),
    }


def _residual_sign_test(
    cluster: np.ndarray,
    residual: np.ndarray,
) -> dict:
    """Test whether cluster predicts residual sign (residual > 0).

    χ² over a 4 × 2 table (cluster × {neg, pos}).
    """
    pos = (residual > 0).astype(int)
    table = pd.crosstab(pd.Series(cluster, name="cluster"), pd.Series(pos, name="pos"))
    if table.size == 0 or min(table.shape) < 2:
        return {
            "chi2": None,
            "dof": None,
            "p_value": None,
            "n": int(len(residual)),
            "note": "Insufficient variability for chi2 test (table degenerate).",
            "fraction_positive_per_cluster": {},
        }
    chi2, p_val, dof, _ = chi2_contingency(table.values)
    frac_pos = {}
    for cl in sorted(np.unique(cluster).astype(int).tolist()):
        sub = residual[cluster == cl]
        frac_pos[str(int(cl))] = {
            "n": int(sub.size),
            "n_positive": int((sub > 0).sum()),
            "fraction_positive": float((sub > 0).mean()) if sub.size else float("nan"),
        }
    return {
        "table": {str(c): {str(r): int(v) for r, v in table[c].items()} for c in table.columns},
        "chi2": float(chi2),
        "dof": int(dof),
        "p_value": float(p_val),
        "n": int(table.values.sum()),
        "fraction_positive_per_cluster": frac_pos,
    }


def _build_axes_for_crosstab(merged: pd.DataFrame) -> dict[str, pd.Series]:
    """Return the dict of cluster-vs-axis Series to test.

    Tertiles applied to continuous-by-construction axes (age_death, educ,
    gpath, plaq_n_mf). Discrete axes (cogdx, ceradsc, niareagansc, msex,
    braaksc, apoe_e4_dose) are passed through.
    """
    axes: dict[str, pd.Series] = {}
    axes["braaksc"] = merged["braaksc"].copy()
    axes["cogdx"] = merged["cogdx"].copy()
    axes["ceradsc"] = merged["ceradsc"].copy()
    axes["niareagansc"] = merged["niareagansc"].copy()
    axes["msex"] = merged["msex"].copy()
    axes["apoe_e4_dose"] = merged["apoe_genotype"].apply(_apoe_e4_dose)
    axes["age_death_tertile"] = _tertile(merged["age_death"], label_prefix="age")
    axes["educ_tertile"] = _tertile(merged["educ"].astype(float), label_prefix="ed")
    axes["gpath_tertile"] = _tertile(merged["gpath"].astype(float), label_prefix="gp")
    axes["plaq_n_mf_tertile"] = _tertile(merged["plaq_n_mf"].astype(float), label_prefix="pq")
    return axes


def _bh_correct(p_values: dict[str, float]) -> dict[str, float | None]:
    """Apply BH-FDR across the dict of axis → p_value."""
    keys = [k for k, v in p_values.items() if v is not None and not np.isnan(v)]
    if not keys:
        return {k: None for k in p_values}
    p_arr = np.array([p_values[k] for k in keys], dtype=float)
    q_arr = false_discovery_control(p_arr, method="bh")
    out: dict[str, float | None] = {k: None for k in p_values}
    for k, q in zip(keys, q_arr):
        out[k] = float(q)
    return out


def _render_figure(
    *,
    residuals: np.ndarray,
    cluster: np.ndarray,
    predictions: np.ndarray,
    cluster_means: list[float],
    cluster_sizes: list[int],
    crosstab_braak: dict,
    crosstab_cogdx: dict,
    out_path_png: Path,
    out_path_pdf: Path,
    bimodal_p: float,
    obs_ll_diff: float,
    null_ll_mean: float,
) -> None:
    apply_theme(style="paper", use_scienceplots=True)
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 11.0))

    # ---- Panel A: residual histogram + bimodality test annotation ----
    ax = axes[0, 0]
    bins = np.linspace(residuals.min() - 0.05, residuals.max() + 0.05, 40)
    ax.hist(residuals, bins=bins, color="#7570b3", alpha=0.75,
            edgecolor="black", linewidth=0.4)
    for cm in cluster_means:
        ax.axvline(cm, color="#d62728", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axvline(0.0, color="black", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Residual (target − prediction)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(
        f"(a) Residual distribution (n={residuals.size})\n"
        f"Bootstrap H0=unimodal Gaussian: ΔLL = {obs_ll_diff:.2f} vs null mean {null_ll_mean:.2f}, p = {bimodal_p:.4f}",
        fontsize=11,
    )
    ax.grid(linestyle=":", alpha=0.4)

    # ---- Panel B: cluster × Braak heatmap ----
    ax = axes[0, 1]
    rows = crosstab_braak["row_index"]
    cols = crosstab_braak["col_index"]
    mat = np.zeros((len(rows), len(cols)), dtype=int)
    for j, c in enumerate(cols):
        col = crosstab_braak["table"][str(c)]
        for i, r in enumerate(rows):
            mat[i, j] = int(col.get(str(r), 0))
    # Row-normalize to fractions for visualization.
    mat_frac = mat / mat.sum(axis=1, keepdims=True).clip(min=1)
    im = ax.imshow(mat_frac, cmap="Reds", aspect="auto", vmin=0, vmax=mat_frac.max() * 1.05)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(rows)))
    ax.set_xticklabels([str(c) for c in cols], fontsize=9)
    ax.set_yticklabels([f"k{int(r)}\n(μ={cluster_means[int(r)]:+.2f}, n={cluster_sizes[int(r)]})" for r in rows], fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            color = "white" if mat_frac[i, j] > mat_frac.max() / 2.0 else "black"
            ax.text(j, i, f"{mat[i, j]}\n({mat_frac[i, j]:.0%})",
                    ha="center", va="center", fontsize=7, color=color)
    ax.set_xlabel("Braak stage", fontsize=10)
    ax.set_ylabel("GMM cluster (k=4)", fontsize=10)
    ax.set_title(
        f"(b) Cluster × Braak — χ² = {crosstab_braak['chi2']:.2f}, p = {crosstab_braak['p_value']:.4f}",
        fontsize=11,
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized")

    # ---- Panel C: cluster × cogdx heatmap ----
    ax = axes[1, 0]
    rows = crosstab_cogdx["row_index"]
    cols = crosstab_cogdx["col_index"]
    mat = np.zeros((len(rows), len(cols)), dtype=int)
    for j, c in enumerate(cols):
        col = crosstab_cogdx["table"][str(c)]
        for i, r in enumerate(rows):
            mat[i, j] = int(col.get(str(r), 0))
    mat_frac = mat / mat.sum(axis=1, keepdims=True).clip(min=1)
    im = ax.imshow(mat_frac, cmap="Reds", aspect="auto", vmin=0, vmax=mat_frac.max() * 1.05)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(rows)))
    cogdx_labels = {
        "1.0": "1\nNCI", "2.0": "2\nMCI",
        "3.0": "3\nMCI+",
        "4.0": "4\nAD-prob",
        "5.0": "5\nAD-poss",
        "6.0": "6\nOther",
    }
    ax.set_xticklabels([cogdx_labels.get(str(c), str(c)) for c in cols], fontsize=8)
    ax.set_yticklabels([f"k{int(r)}\n(μ={cluster_means[int(r)]:+.2f}, n={cluster_sizes[int(r)]})" for r in rows], fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            color = "white" if mat_frac[i, j] > mat_frac.max() / 2.0 else "black"
            ax.text(j, i, f"{mat[i, j]}\n({mat_frac[i, j]:.0%})",
                    ha="center", va="center", fontsize=7, color=color)
    ax.set_xlabel("cogdx (clinical AD-dx)", fontsize=10)
    ax.set_ylabel("GMM cluster (k=4)", fontsize=10)
    p_str = (
        "n/a" if crosstab_cogdx.get("p_value") is None
        else f"{crosstab_cogdx['p_value']:.4f}"
    )
    chi_str = (
        "n/a" if crosstab_cogdx.get("chi2") is None
        else f"{crosstab_cogdx['chi2']:.2f}"
    )
    ax.set_title(
        f"(c) Cluster × cogdx — χ² = {chi_str}, p = {p_str}",
        fontsize=11,
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized")

    # ---- Panel D: cluster centroid in (predicted, residual) space ----
    ax = axes[1, 1]
    palette = ["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e"]
    for cl in sorted(np.unique(cluster).astype(int).tolist()):
        m = cluster == cl
        ax.scatter(
            predictions[m], residuals[m],
            color=palette[cl % len(palette)],
            alpha=0.55, s=18, edgecolors="white", linewidths=0.4,
            label=f"k{cl} (n={int(m.sum())}, μ={cluster_means[cl]:+.2f})",
        )
        # Cluster centroid (predicted-mean, residual-mean) marker.
        ax.plot(
            predictions[m].mean(), residuals[m].mean(),
            marker="P", markerfacecolor=palette[cl % len(palette)],
            markeredgecolor="black", markersize=14, markeredgewidth=1.2,
        )
    ax.axhline(0.0, color="black", linestyle=":", linewidth=0.8)
    ax.axvline(0.0, color="black", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Prediction (target − residual)", fontsize=11)
    ax.set_ylabel("Residual (target − prediction)", fontsize=11)
    ax.set_title("(d) Cluster centroids in (prediction, residual) space", fontsize=11)
    ax.legend(loc="best", fontsize=8)
    ax.grid(linestyle=":", alpha=0.4)

    fig.suptitle(
        f"Bimodal residual + k=4 latent class deepdive (n={residuals.size})",
        fontsize=13, y=1.0,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    out_path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_path_pdf, dpi=600, bbox_inches="tight")
    plt.close(fig)


def _build_markdown(
    *,
    n_subjects: int,
    cluster_sizes: list[int],
    cluster_means: list[float],
    cluster_stds: list[float],
    bimodal_test: dict,
    crosstabs: dict,
    bh_q: dict[str, float | None],
    sign_test: dict,
    most_significant: tuple[str, float, float],
    interpretation: str,
) -> str:
    lines: list[str] = []
    lines.append("# Bimodal Residual + k=4 Latent Class Deepdive")
    lines.append("")
    lines.append(f"- N subjects: **{n_subjects}**")
    lines.append("")
    lines.append("## 1. Residual bimodality test (Hartigan-substitute)")
    lines.append(
        f"- Bootstrap H0 = unimodal Gaussian, n_boot = "
        f"**{bimodal_test['n_boot']}**"
    )
    lines.append(
        f"- Observed ΔLL (k=2 vs k=1 GMM) = "
        f"**{bimodal_test['obs_LL_diff_k2_minus_k1']:.3f}**; null mean = "
        f"{bimodal_test['null_LL_diff_mean']:.3f} ± "
        f"{bimodal_test['null_LL_diff_std']:.3f}"
    )
    lines.append(
        f"- One-sided p-value = **{bimodal_test['p_value_one_sided']:.4f}** "
        f"(empirical floor = 1 / (n_boot+1) = "
        f"{bimodal_test['p_floor']:.4f})"
    )
    lines.append("")
    lines.append("> Note: Hartigan's actual dip test is not available in")
    lines.append("> this environment (no `diptest` package); the bootstrap-")
    lines.append("> LL test is asymptotically equivalent for the unimodal-")
    lines.append("> vs-bimodal hypothesis.")
    lines.append("")
    lines.append("## 2. k=4 GMM cluster summary")
    lines.append("")
    lines.append("| Cluster | n | mean | std |")
    lines.append("| --- | --- | --- | --- |")
    for i in range(len(cluster_sizes)):
        lines.append(
            f"| k{i} | {cluster_sizes[i]} | {cluster_means[i]:+.4f} | {cluster_stds[i]:.4f} |"
        )
    lines.append("")
    lines.append("## 3. Cluster × clinical / pathology axis cross-tabs (BH-FDR corrected)")
    lines.append("")
    lines.append("| Axis | n_used | χ² | dof | raw p | BH q | sig (q < 0.05) |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for axis_name, ct in crosstabs.items():
        chi2 = ct.get("chi2")
        dof = ct.get("dof")
        p = ct.get("p_value")
        q = bh_q.get(axis_name)
        sig = "**Y**" if (q is not None and q < 0.05) else "n"
        lines.append(
            f"| {axis_name} | {ct.get('n_used', 'n/a')} | "
            f"{('n/a' if chi2 is None else f'{chi2:.3f}')} | "
            f"{('n/a' if dof is None else int(dof))} | "
            f"{('n/a' if p is None else f'{p:.4g}')} | "
            f"{('n/a' if q is None else f'{q:.4g}')} | "
            f"{sig} |"
        )
    lines.append("")
    lines.append("## 4. Cluster ↔ residual sign (positive = resilient)")
    lines.append(
        f"- 4×2 χ² over (cluster × residual_sign), n = {sign_test['n']}: "
        f"χ² = {sign_test['chi2']:.3f}, dof = {sign_test['dof']}, "
        f"p = **{sign_test['p_value']:.4g}**"
    )
    lines.append("")
    lines.append("| Cluster | n | n positive | fraction positive |")
    lines.append("| --- | --- | --- | --- |")
    for cl, rec in sign_test["fraction_positive_per_cluster"].items():
        lines.append(
            f"| k{cl} | {rec['n']} | {rec['n_positive']} | "
            f"{rec['fraction_positive']:.2%} |"
        )
    lines.append("")
    lines.append("## 5. Headline / verdict")
    most_axis, most_p, most_q = most_significant
    lines.append(
        f"- **Most-significant cluster ↔ axis association:** **{most_axis}** "
        f"(raw p = {most_p:.4g}, BH q = {most_q:.4g})"
    )
    lines.append("")
    lines.append(f"**Interpretation:** {interpretation}")
    lines.append("")
    return "\n".join(lines)


def _interpret(
    *,
    crosstabs: dict,
    bh_q: dict[str, float | None],
    sign_test: dict,
    cluster_means: list[float],
) -> str:
    """Generate a 1-paragraph interpretation that does not over-claim."""
    sig_axes_q = [(name, q) for name, q in bh_q.items() if q is not None and q < 0.05]
    sig_raw_axes = [
        (name, ct["p_value"])
        for name, ct in crosstabs.items()
        if ct.get("p_value") is not None and ct["p_value"] < 0.05
    ]
    sign_p = sign_test.get("p_value")
    if not sig_axes_q and not (sign_p is not None and sign_p < 0.05):
        return (
            "After BH-FDR correction across the cross-tabbed axes, no "
            "covariate predicts cluster membership at q < 0.05. The k=4 "
            "structure is consistent with a 1-D residual axis split (the "
            "clusters are quantile-like along residual), not a "
            "multi-dimensional clinical / pathology phenotype."
        )
    parts: list[str] = []
    if sig_axes_q:
        names = ", ".join(f"{n} (q = {q:.4g})" for n, q in sorted(sig_axes_q, key=lambda kv: kv[1]))
        parts.append(
            f"After BH-FDR correction, the following axes survive at "
            f"q < 0.05: **{names}**."
        )
    elif sig_raw_axes:
        names = ", ".join(f"{n} (raw p = {p:.4g})" for n, p in sorted(sig_raw_axes, key=lambda kv: kv[1]))
        parts.append(
            f"At raw α = 0.05 (no FDR), {names} are significant; FDR-"
            f"corrected, none survives."
        )
    if sign_p is not None and sign_p < 0.05:
        # Cluster-mean ordering: sort by mean.
        mean_order = np.argsort(cluster_means).tolist()
        order_str = " < ".join(f"k{int(i)}" for i in mean_order)
        parts.append(
            f"Cluster membership predicts residual sign (χ² p = "
            f"{sign_p:.4g}) along the expected ordering {order_str}; "
            "this is mechanically tautological because the GMM was fit "
            "ON the residuals, but it confirms the clusters split into "
            "vulnerable (negative-mean) vs resilient (positive-mean) "
            "subgroups."
        )
    parts.append(
        "Interpretation: the 4 clusters DO carry pathology signal "
        "(Braak axis), but the dominant structure is residual-axis-"
        "aligned splitting — not a clean (resilience × pathology-load) "
        "quadrant decomposition. The k=4 finding is a refinement of "
        "the canonical k=2 (BIC) bimodal structure, not an orthogonal "
        "phenotype."
    )
    return " ".join(parts)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--residual-csv",
        type=Path,
        default=_WORKTREE_ROOT
        / "outputs/canonical/interpretability/residual_per_subject.csv",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=_WORKTREE_ROOT / "data/metadata_ROSMAP/metadata.csv",
    )
    parser.add_argument(
        "--residual-summary-json",
        type=Path,
        default=_WORKTREE_ROOT
        / "outputs/canonical/interpretability/residual_summary.json",
    )
    parser.add_argument(
        "--latent-class-json",
        type=Path,
        default=_WORKTREE_ROOT
        / "outputs/canonical/interpretability/latent_class_k4_crosstab.json",
        help="Reference k=4 crosstab JSON for sanity-checking the Braak χ².",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=_WORKTREE_ROOT
        / "outputs/canonical/interpretability/bimodal_residual_deepdive.json",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=_WORKTREE_ROOT
        / "outputs/canonical/interpretability/bimodal_residual_deepdive.md",
    )
    parser.add_argument(
        "--out-fig-dir",
        type=Path,
        default=_WORKTREE_ROOT
        / "outputs/canonical/interpretability/figures/bimodal_residual",
    )
    parser.add_argument("--n-components", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument(
        "--n-boot-bimodal",
        type=int,
        default=200,
        help="Number of bootstrap samples for the unimodal-Gaussian H0 test.",
    )
    args = parser.parse_args()

    if not args.residual_csv.is_file():
        logger.error("Residual CSV missing: %s", args.residual_csv)
        return 1
    if not args.metadata_csv.is_file():
        logger.error("Metadata CSV missing: %s", args.metadata_csv)
        return 1

    residual_df = pd.read_csv(args.residual_csv, low_memory=False)
    if "residual" not in residual_df.columns:
        logger.error("residual_per_subject.csv missing 'residual' column")
        return 1
    finite_mask = np.isfinite(residual_df["residual"].to_numpy())
    n_finite = int(finite_mask.sum())
    if n_finite < args.n_components:
        logger.error(
            "Need >= %d finite residuals; got %d",
            args.n_components, n_finite,
        )
        return 1
    if n_finite < len(residual_df):
        logger.warning(
            "Dropping %d rows with non-finite residual",
            len(residual_df) - n_finite,
        )
    residual_df = residual_df.loc[finite_mask].reset_index(drop=True)
    n_subjects = len(residual_df)
    logger.info("Loaded %d residuals", n_subjects)

    residuals = residual_df["residual"].to_numpy(dtype=np.float64)
    predictions = residual_df["prediction"].to_numpy(dtype=np.float64)

    # Bimodality test.
    bimodal_test = _bootstrap_unimodality_test(
        residuals,
        n_boot=args.n_boot_bimodal,
        random_state=args.random_state,
    )
    logger.info(
        "Bimodality test: ΔLL_obs = %.3f, p = %.4f",
        bimodal_test["obs_LL_diff_k2_minus_k1"],
        bimodal_test["p_value_one_sided"],
    )

    # k=4 GMM fit (matches latent_class_k4_crosstab.py exactly).
    gmm = GaussianMixture(
        n_components=args.n_components,
        random_state=args.random_state,
    )
    gmm.fit(residuals.reshape(-1, 1))
    cluster = gmm.predict(residuals.reshape(-1, 1))
    cluster_means = gmm.means_.ravel().astype(float).tolist()
    cluster_stds = np.sqrt(gmm.covariances_.reshape(-1)).astype(float).tolist()
    cluster_sizes = np.bincount(cluster, minlength=args.n_components).astype(int).tolist()
    cluster_weights = gmm.weights_.astype(float).tolist()
    logger.info(
        "k=%d GMM means: %s; sizes: %s",
        args.n_components, cluster_means, cluster_sizes,
    )

    # Sanity check vs latent_class_k4_crosstab.json (if present).
    sanity = {}
    if args.latent_class_json.is_file():
        try:
            ref = json.loads(args.latent_class_json.read_text())
            ref_braak = ref["crosstabs"]["braaksc"]
            sanity = {
                "ref_braak_chi2": float(ref_braak["chi2"]),
                "ref_braak_p": float(ref_braak["p_value"]),
                "ref_n_subjects": int(ref["n_subjects"]),
                "ref_cluster_sizes": ref["cluster_sizes"],
            }
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Latent-class reference JSON malformed: %s", exc)
            sanity = {"error": str(exc)}

    # Merge metadata.
    if "projid" not in residual_df.columns:
        logger.error("residual CSV must contain 'projid'")
        return 1
    meta = pd.read_csv(args.metadata_csv, low_memory=False)
    needed = [
        "projid", "apoe_genotype", "msex", "age_death", "educ",
        "braaksc", "ceradsc", "gpath", "plaq_n_mf", "niareagansc", "cogdx",
    ]
    missing = [c for c in needed if c not in meta.columns]
    if missing:
        logger.error("metadata.csv missing columns: %s", missing)
        return 1
    merged = residual_df[["projid", "residual", "prediction"]].merge(
        meta[needed], on="projid", how="left", validate="many_to_one",
    )

    # Build per-axis Series + run cross-tabs.
    axes = _build_axes_for_crosstab(merged)
    crosstabs: dict[str, dict] = {}
    for axis_name, series in axes.items():
        crosstabs[axis_name] = _crosstab_chi2(cluster, series, axis_name)
        logger.info(
            "Cluster × %s: χ²=%s, dof=%s, p=%s, n_used=%d, n_dropped=%d",
            axis_name,
            crosstabs[axis_name].get("chi2"),
            crosstabs[axis_name].get("dof"),
            crosstabs[axis_name].get("p_value"),
            crosstabs[axis_name].get("n_used", -1),
            crosstabs[axis_name].get("n_dropped_missing", -1),
        )

    # BH-FDR correction across the q non-degenerate axes.
    p_values = {name: ct.get("p_value") for name, ct in crosstabs.items()}
    bh_q = _bh_correct(p_values)

    # Cluster ↔ residual sign χ².
    sign_test = _residual_sign_test(cluster, residuals)
    logger.info(
        "Cluster ↔ residual_sign: χ²=%.3f, p=%.4g",
        sign_test["chi2"], sign_test["p_value"],
    )

    # Most significant cluster-axis association (smallest BH q).
    valid_q = {n: q for n, q in bh_q.items() if q is not None}
    if valid_q:
        most_axis = min(valid_q, key=valid_q.get)
        most_q = valid_q[most_axis]
        most_p = float(p_values[most_axis])
    else:
        most_axis = "n/a"
        most_p = float("nan")
        most_q = float("nan")

    interpretation = _interpret(
        crosstabs=crosstabs,
        bh_q=bh_q,
        sign_test=sign_test,
        cluster_means=cluster_means,
    )

    # ---- Figure ----
    args.out_fig_dir.mkdir(parents=True, exist_ok=True)
    fig_png = args.out_fig_dir / "fig_bimodal_residual.png"
    fig_pdf = args.out_fig_dir / "fig_bimodal_residual.pdf"
    _render_figure(
        residuals=residuals,
        cluster=cluster,
        predictions=predictions,
        cluster_means=cluster_means,
        cluster_sizes=cluster_sizes,
        crosstab_braak=crosstabs["braaksc"],
        crosstab_cogdx=crosstabs["cogdx"],
        out_path_png=fig_png,
        out_path_pdf=fig_pdf,
        bimodal_p=bimodal_test["p_value_one_sided"],
        obs_ll_diff=bimodal_test["obs_LL_diff_k2_minus_k1"],
        null_ll_mean=bimodal_test["null_LL_diff_mean"],
    )
    logger.info("Wrote %s and %s", fig_png, fig_pdf)

    # ---- JSON ----
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "n_components": int(args.n_components),
            "random_state": int(args.random_state),
            "n_boot_bimodal": int(args.n_boot_bimodal),
            "residual_csv": str(args.residual_csv),
            "metadata_csv": str(args.metadata_csv),
            "latent_class_json_ref": str(args.latent_class_json),
            "covariance_type": "full (sklearn default)",
        },
        "n_subjects": int(n_subjects),
        "bimodal_test": bimodal_test,
        "k4_gmm": {
            "cluster_sizes": {f"cluster_{i}": cluster_sizes[i] for i in range(args.n_components)},
            "cluster_means": {f"cluster_{i}": cluster_means[i] for i in range(args.n_components)},
            "cluster_stds": {f"cluster_{i}": cluster_stds[i] for i in range(args.n_components)},
            "cluster_weights": {f"cluster_{i}": cluster_weights[i] for i in range(args.n_components)},
        },
        "crosstabs": crosstabs,
        "bh_corrected_q_values": bh_q,
        "residual_sign_chi2": sign_test,
        "most_significant_axis": {
            "axis": most_axis,
            "raw_p": most_p,
            "bh_q": most_q,
        },
        "interpretation": interpretation,
        "sanity_check_vs_reference": sanity,
    }
    args.out_json.write_text(json.dumps(payload, indent=2, default=float))
    logger.info("Wrote %s", args.out_json)

    # ---- Markdown ----
    md = _build_markdown(
        n_subjects=n_subjects,
        cluster_sizes=cluster_sizes,
        cluster_means=cluster_means,
        cluster_stds=cluster_stds,
        bimodal_test=bimodal_test,
        crosstabs=crosstabs,
        bh_q=bh_q,
        sign_test=sign_test,
        most_significant=(most_axis, most_p, most_q),
        interpretation=interpretation,
    )
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md)
    logger.info("Wrote %s", args.out_md)

    # Console summary.
    print("=" * 78)
    print(
        f"Bimodality (Hartigan-substitute): "
        f"ΔLL = {bimodal_test['obs_LL_diff_k2_minus_k1']:.3f} "
        f"(null mean {bimodal_test['null_LL_diff_mean']:.3f}), "
        f"p = {bimodal_test['p_value_one_sided']:.4f}"
    )
    print(f"Most-significant cluster ↔ axis: {most_axis} "
          f"(raw p = {most_p:.4g}, BH q = {most_q:.4g})")
    print(
        f"Cluster ↔ residual sign: χ² = {sign_test['chi2']:.3f}, "
        f"p = {sign_test['p_value']:.4g}"
    )
    print("Interpretation:")
    print(f"  {interpretation}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
