#!/usr/bin/env python
"""EXP-039: cross-link patient-stratification axes (cluster × sex × APOE × CCC × CF).

Builds a per-subject membership matrix combining four orthogonal stratification
axes from previous experiments and tests whether they are independent or share
overlap structure:

  1. **GMM cluster** (EXP-033)   — k=4 GMM on per-subject residual; reproduces
     `latent_class_k4_crosstab.json` byte-for-byte (`random_state=0`,
     sklearn-default covariance_type="full"). Cluster labels {0, 1, 2, 3}.
  2. **Sex**          (EXP-035)  — `msex ∈ {0=F, 1=M}` from ROSMAP metadata.
  3. **APOE-ε4 dose** (EXP-035)  — count of ε4 alleles in `apoe_genotype`
     (0 / 1 / 2). Coded directly to match the EXP-035 stratification.
  4. **CCC outlier**  (EXP-029)  — `Y` if subject is in the τ=0.01 outlier set
     (n=15) from `ccc_heterogeneity/threshold_sensitivity.json`, else `N`.
  5. **F1-CF success** (EXP-024-stepD) — `Y/N/N/A`: `Y` if the subject's
     fold-aware row in `counterfactuals_optimized_absolute_delta0p3*/...json`
     reports `success == True`; `N` if `success == False`; `N/A` otherwise.

For each pair of axes, runs Fisher exact (binary–binary) or χ² independence
(everything else), then BH-FDR-corrects the resulting p-value matrix.

Renders 4-panel figure:
  - Panel A: 4-cluster × sex (left) and 4-cluster × APOE-ε4 (right) crosstabs
    with cell counts overlaid.
  - Panel B: cluster scatter (residual on x, jitter on y) with CCC outliers
    highlighted; per-cluster outlier counts in the legend.
  - Panel C: per-fold R² stratified by (cluster × sex) — does the EXP-035
    sex disparity localize to specific clusters?
  - Panel D: F1 CF success rate per cluster — does the EXP-024-stepD success
    asymmetry track the GMM cluster (k0 vulnerable vs k2 resilient)?

Inputs (defaults; CLI-overridable):
  --residual-csv          outputs/canonical/interpretability/residual_per_subject.csv
  --metadata-csv          data/metadata_ROSMAP/metadata.csv
  --ccc-threshold-json    outputs/canonical/interpretability/ccc_heterogeneity/threshold_sensitivity.json
  --cf-fold0              outputs/canonical/interpretability/counterfactuals_optimized_absolute_delta0p3
  --cf-fold-template      outputs/canonical/interpretability/counterfactuals_optimized_absolute_delta0p3_fold{N}
  --pred-root             outputs/canonical/p5_canonical_seed42
  --tabpfn-dir            data/canonical

Outputs:
  --out-json              outputs/canonical/interpretability/patient_stratification_crosslink.json
  --out-md                outputs/canonical/interpretability/patient_stratification_crosslink.md
  --out-fig-dir           outputs/canonical/interpretability/figures/patient_stratification

Configuration matches the canonical EXP-033 GMM fit:
    GaussianMixture(n_components=4, random_state=0, covariance_type="full")
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, false_discovery_control, fisher_exact
from sklearn.metrics import r2_score
from sklearn.mixture import GaussianMixture

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from src.analysis.resdec_io import load_all_folds  # noqa: E402
from src.visualization.theme import apply_theme, fmt_axes, style_paper_axes  # noqa: E402

logger = logging.getLogger(__name__)

# Tau used for the canonical CCC-outlier set (τ=0.01 → 15 outliers per
# threshold_sensitivity.json, the cohort cited in EXP-028).
CANONICAL_CCC_TAU = 0.01

# Each ε4 allele in apoe_genotype contributes one "4" digit; e.g. 24 → 1, 44 → 2.
APOE_E4_DOSE_MAP = {22: 0, 23: 0, 33: 0, 24: 1, 34: 1, 44: 2}


# =============================================================================
# APOE-ε4 dose helper
# =============================================================================


def apoe_e4_dose(apoe_genotype: float) -> int | None:
    """Count number of ε4 alleles in a 2-digit APOE genotype.

    NaN / unknown → None (caller handles drop-or-keep).
    Matches the convention used in `subgroup_metrics.json` / EXP-035.
    """
    if apoe_genotype is None:
        return None
    try:
        f = float(apoe_genotype)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    code = int(round(f))
    return APOE_E4_DOSE_MAP.get(code)


# =============================================================================
# CCC-outlier extraction (from threshold_sensitivity.json)
# =============================================================================


def load_ccc_outliers(
    ccc_threshold_json: Path, tau: float = CANONICAL_CCC_TAU,
) -> set[str]:
    """Return the set of `subject_id`s flagged as CCC outliers at threshold τ.

    `threshold_sensitivity.json::per_threshold` is a list of records, one per
    τ value scanned. Each record has an `outlier_subjects` list of dicts with
    a `subject_id` key.
    """
    if not ccc_threshold_json.exists():
        raise FileNotFoundError(ccc_threshold_json)
    with ccc_threshold_json.open() as fh:
        d = json.load(fh)
    pt = d.get("per_threshold")
    if not isinstance(pt, list):
        raise ValueError(
            f"{ccc_threshold_json}: expected 'per_threshold' to be a list, "
            f"got {type(pt).__name__}"
        )
    for entry in pt:
        if abs(float(entry["threshold"]) - tau) < 1e-9:
            return {str(s["subject_id"]) for s in entry["outlier_subjects"]}
    raise KeyError(
        f"τ={tau} not found in {ccc_threshold_json}; available: "
        f"{[float(e['threshold']) for e in pt]}"
    )


# =============================================================================
# CF-success extraction (per-fold counterfactual JSONs)
# =============================================================================


def _resolve_cf_paths(
    cf_fold0: Path, cf_fold_template: str, n_folds: int = 5,
) -> dict[int, Path]:
    """Map fold idx → counterfactual JSON path.

    Fold-0 lives at the bare directory; folds 1..N-1 follow `fold_template`
    (with `{N}` substituted by the fold index).
    """
    paths: dict[int, Path] = {0: cf_fold0 / "counterfactuals_fold0.json"}
    for fold in range(1, n_folds):
        d = Path(cf_fold_template.format(N=fold))
        paths[fold] = d / f"counterfactuals_fold{fold}.json"
    return paths


def load_cf_success(
    cf_fold0: Path, cf_fold_template: str, n_folds: int = 5,
) -> dict[str, str]:
    """Build {subject_id → "Y" | "N"} from per-fold CF JSONs.

    Each fold JSON contains a `results` list with `subject_id` + `success`
    fields. Subjects that appear in *any* fold's CF run get a label; subjects
    in no fold's CF run will be labelled `"N/A"` later by the caller.
    """
    out: dict[str, str] = {}
    cf_paths = _resolve_cf_paths(cf_fold0, cf_fold_template, n_folds=n_folds)
    for fold, path in cf_paths.items():
        if not path.exists():
            logger.warning(
                "Fold-%d CF JSON missing: %s — subjects from this fold "
                "will be N/A", fold, path,
            )
            continue
        with path.open() as fh:
            d = json.load(fh)
        for r in d.get("results", []):
            sid = str(r["subject_id"])
            success = bool(r["success"])
            label = "Y" if success else "N"
            # If a subject appears in multiple folds (shouldn't happen with
            # 50 res + 50 vuln × 5 folds = 500 subjects covering 96.9% of the
            # 516 cohort, but each subject's val-fold is unique), keep the
            # first-seen label. The CF rep is by fold-of-evaluation, so a
            # given subject is only present in their own val fold.
            if sid not in out:
                out[sid] = label
    return out


# =============================================================================
# GMM cluster fit (matches latent_class_k4_crosstab.json)
# =============================================================================


def fit_residual_gmm(
    residual: np.ndarray, n_components: int = 4, random_state: int = 0,
) -> np.ndarray:
    """Fit `GaussianMixture(k=4, random_state=0)` on per-subject residuals.

    Matches EXP-033's canonical fit byte-for-byte. Returns the per-subject
    cluster label as int array of length len(residual).
    """
    finite_mask = np.isfinite(residual)
    if int(finite_mask.sum()) < n_components:
        raise ValueError(
            f"Need ≥ {n_components} finite residuals; got "
            f"{int(finite_mask.sum())}"
        )
    if not bool(np.all(finite_mask)):
        raise ValueError(
            f"residual contains {int((~finite_mask).sum())} non-finite values; "
            "drop these in the caller before fitting."
        )
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    gmm.fit(residual.reshape(-1, 1))
    return gmm.predict(residual.reshape(-1, 1)).astype(np.int64)


# =============================================================================
# Pairwise Fisher / χ² + BH-FDR
# =============================================================================


def _build_contingency(
    a: pd.Series, b: pd.Series,
) -> tuple[np.ndarray, list, list, int, int]:
    """Build a contingency table from two categorical Series.

    Drops rows where either series is NaN. Returns
    `(table, row_labels, col_labels, n_used, n_dropped)`.
    """
    mask = a.notna() & b.notna()
    a_kept = a[mask]
    b_kept = b[mask]
    if a_kept.empty or b_kept.empty:
        return np.zeros((0, 0)), [], [], 0, int((~mask).sum())
    tbl = pd.crosstab(a_kept, b_kept)
    return (
        tbl.values.astype(np.int64),
        [str(x) for x in tbl.index.tolist()],
        [str(x) for x in tbl.columns.tolist()],
        int(mask.sum()),
        int((~mask).sum()),
    )


def pair_test(
    a: pd.Series, b: pd.Series,
) -> dict:
    """Run Fisher exact (2x2) or χ² (otherwise) on the contingency of (a, b).

    Returns a dict with `table`, `n_used`, `n_dropped`, `test`, `statistic`,
    `dof`, `p_value`. Records `test='degenerate'` and p=1.0 if the table is
    too small for a test (any axis < 2 unique values).
    """
    table, rows, cols, n_used, n_dropped = _build_contingency(a, b)
    base = {
        "table": [list(map(int, r)) for r in table.tolist()],
        "row_labels": rows,
        "col_labels": cols,
        "n_used": n_used,
        "n_dropped": n_dropped,
    }
    if table.size == 0 or min(table.shape) < 2:
        return {**base, "test": "degenerate", "statistic": None, "dof": None, "p_value": 1.0}
    if table.shape == (2, 2):
        stat, pv = fisher_exact(table)
        return {**base, "test": "fisher_exact", "statistic": float(stat), "dof": None, "p_value": float(pv)}
    chi2_stat, pv, dof, _ = chi2_contingency(table)
    return {**base, "test": "chi2_contingency", "statistic": float(chi2_stat), "dof": int(dof), "p_value": float(pv)}


def bh_fdr(p_values: Sequence[float]) -> np.ndarray:
    """Benjamini–Hochberg q-values via `scipy.stats.false_discovery_control`."""
    p = np.asarray(p_values, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError(f"p_values must be 1-D, got shape {p.shape}")
    if np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError("p_values must be in [0, 1].")
    return false_discovery_control(p, method="bh")


def run_pairwise_tests(df: pd.DataFrame) -> dict:
    """Run pair_test on every (axis_i, axis_j) pair and BH-FDR-correct.

    Axes are: cluster, sex, apoe_e4, ccc_outlier, cf_success. Returns a dict
    with both the raw per-pair test results and the BH-corrected q-value
    matrix.
    """
    axes = ["cluster", "sex", "apoe_e4", "ccc_outlier", "cf_success"]
    tests: dict[str, dict] = {}
    pair_keys: list[tuple[str, str]] = []
    p_values: list[float] = []

    for i, a in enumerate(axes):
        for j, b in enumerate(axes):
            if j <= i:
                continue
            key = f"{a}__vs__{b}"
            res = pair_test(df[a], df[b])
            tests[key] = res
            pair_keys.append((a, b))
            p_values.append(res["p_value"])

    q_values = bh_fdr(p_values).tolist() if p_values else []
    for (a, b), q in zip(pair_keys, q_values):
        tests[f"{a}__vs__{b}"]["q_value_bh"] = float(q)

    # Square q-value matrix for display (diagonal = NaN).
    q_matrix = {a: {b: None for b in axes} for a in axes}
    for (a, b), q in zip(pair_keys, q_values):
        q_matrix[a][b] = float(q)
        q_matrix[b][a] = float(q)
    return {"per_pair": tests, "axis_order": axes, "q_matrix": q_matrix}


# =============================================================================
# Per-(cluster × sex) per-fold R²
# =============================================================================


def per_cluster_sex_r2(
    pred_df: pd.DataFrame, mem: pd.DataFrame, n_folds: int = 5,
) -> dict:
    """Per-fold R² stratified by (cluster, sex). Returns nested dict
    keyed by `f"k{c}_{sex}"` → {`per_fold_r2`, `n_per_fold`, `mean_r2`}.

    Subjects are merged on `ROSMAP_IndividualID`; rows where cluster or sex
    is missing are dropped.
    """
    merged = pred_df.merge(
        mem[["ROSMAP_IndividualID", "cluster", "sex"]],
        on="ROSMAP_IndividualID",
        how="inner",
    )
    out: dict[str, dict] = {}
    for c in sorted(merged["cluster"].dropna().unique().astype(int)):
        for sex in ["F", "M"]:
            mask = (merged["cluster"] == c) & (merged["sex"] == sex)
            sub = merged[mask]
            per_fold = []
            n_pf = []
            for f in range(n_folds):
                f_sub = sub[sub["fold"] == f]
                if len(f_sub) >= 3:
                    per_fold.append(float(r2_score(f_sub["y_true"], f_sub["y_composite"])))
                    n_pf.append(int(len(f_sub)))
                else:
                    per_fold.append(float("nan"))
                    n_pf.append(int(len(f_sub)))
            arr = np.asarray(per_fold)
            mean_r2 = float(np.nanmean(arr)) if np.any(np.isfinite(arr)) else float("nan")
            std_r2 = (
                float(np.nanstd(arr, ddof=1))
                if int(np.sum(np.isfinite(arr))) >= 2
                else float("nan")
            )
            out[f"k{c}_{sex}"] = {
                "cluster": int(c),
                "sex": sex,
                "per_fold_r2": per_fold,
                "n_per_fold": n_pf,
                "n_total": int(sum(n_pf)),
                "mean_r2": mean_r2,
                "std_r2": std_r2,
            }
    return out


# =============================================================================
# CF success rate per cluster
# =============================================================================


def cf_success_per_cluster(mem: pd.DataFrame) -> dict:
    """Returns {`f"k{c}"` → {n_cf_total, n_success, n_fail, success_rate}}.

    Subjects with cf_success == "N/A" (not in any CF run) are excluded from
    that cluster's denominator.
    """
    out: dict[str, dict] = {}
    for c in sorted(mem["cluster"].dropna().unique().astype(int)):
        sub = mem[(mem["cluster"] == c) & (mem["cf_success"] != "N/A")]
        n_total = int(len(sub))
        n_success = int((sub["cf_success"] == "Y").sum())
        n_fail = int((sub["cf_success"] == "N").sum())
        rate = n_success / n_total if n_total > 0 else float("nan")
        out[f"k{c}"] = {
            "cluster": int(c),
            "n_cf_total": n_total,
            "n_success": n_success,
            "n_fail": n_fail,
            "success_rate": float(rate),
        }
    return out


# =============================================================================
# Membership matrix builder
# =============================================================================


def build_membership_matrix(
    residual_csv: Path,
    metadata_csv: Path,
    ccc_threshold_json: Path,
    cf_fold0: Path,
    cf_fold_template: str,
    n_folds: int = 5,
    n_components: int = 4,
    random_state: int = 0,
    ccc_tau: float = CANONICAL_CCC_TAU,
) -> tuple[pd.DataFrame, dict]:
    """Build a per-subject membership matrix and return (df, gmm_metadata).

    Columns: `ROSMAP_IndividualID, projid, fold, residual, cluster, sex,
    apoe_e4, ccc_outlier, cf_success`. `gmm_metadata` records the fitted
    cluster centers / sizes for sanity-check vs `latent_class_k4_crosstab.json`.
    """
    if not residual_csv.exists():
        raise FileNotFoundError(residual_csv)
    if not metadata_csv.exists():
        raise FileNotFoundError(metadata_csv)

    rdf = pd.read_csv(residual_csv)
    finite_mask = np.isfinite(rdf["residual"].to_numpy())
    n_dropped = int((~finite_mask).sum())
    rdf = rdf.loc[finite_mask].reset_index(drop=True)

    cluster = fit_residual_gmm(
        rdf["residual"].to_numpy(),
        n_components=n_components,
        random_state=random_state,
    )

    # Compute cluster summary statistics for sanity-check.
    sizes = np.bincount(cluster, minlength=n_components).astype(int).tolist()
    means_per_cluster = [
        float(rdf.loc[cluster == c, "residual"].mean()) for c in range(n_components)
    ]

    ccc_outliers = load_ccc_outliers(ccc_threshold_json, tau=ccc_tau)
    cf_success = load_cf_success(cf_fold0, cf_fold_template, n_folds=n_folds)

    apoe_e4 = rdf["apoe_genotype"].apply(apoe_e4_dose).astype("Int64")
    sex = rdf["msex"].apply(
        lambda v: "F" if (np.isfinite(v) and int(v) == 0) else ("M" if np.isfinite(v) and int(v) == 1 else None)
    )

    mem = pd.DataFrame({
        "ROSMAP_IndividualID": rdf["ROSMAP_IndividualID"].astype(str),
        "projid": rdf["projid"].astype(int),
        "fold": rdf["fold"].astype(int),
        "residual": rdf["residual"].astype(float),
        "cluster": cluster.astype(int),
        "sex": sex,
        "apoe_e4": apoe_e4,
        "ccc_outlier": rdf["ROSMAP_IndividualID"].astype(str).map(
            lambda s: "Y" if s in ccc_outliers else "N"
        ),
        "cf_success": rdf["ROSMAP_IndividualID"].astype(str).map(
            lambda s: cf_success.get(s, "N/A")
        ),
    })

    gmm_meta = {
        "n_components": int(n_components),
        "random_state": int(random_state),
        "covariance_type": "full (sklearn default)",
        "cluster_sizes": {f"cluster_{i}": int(s) for i, s in enumerate(sizes)},
        "cluster_means": {f"cluster_{i}": float(m) for i, m in enumerate(means_per_cluster)},
        "n_subjects": int(len(rdf)),
        "n_dropped_nonfinite_residual": n_dropped,
    }
    return mem, gmm_meta


# =============================================================================
# Figure renderer
# =============================================================================


def _heatmap(ax, table: pd.DataFrame, title: str) -> None:
    """Render a 2-D contingency heatmap with cell counts overlaid."""
    arr = table.values.astype(np.float64)
    im = ax.imshow(arr, cmap="Blues", aspect="auto")
    ax.set_xticks(np.arange(arr.shape[1]))
    ax.set_xticklabels(table.columns)
    ax.set_yticks(np.arange(arr.shape[0]))
    ax.set_yticklabels(table.index)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = int(arr[i, j])
            color = "white" if arr[i, j] > arr.max() * 0.55 else "black"
            ax.text(j, i, f"{v}", ha="center", va="center", color=color, fontsize=8)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    fmt_axes(ax)


def render_figure(
    mem: pd.DataFrame, per_cluster_sex: dict, cf_per_cluster: dict,
    canonical_per_fold_r2: list[float] | None,
    out_fig_dir: Path,
) -> list[Path]:
    """Render the 4-panel cross-link figure."""
    apply_theme()

    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
    ax_a_left = fig.add_subplot(gs[0, 0])
    ax_a_right = fig.add_subplot(gs[0, 1])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    # Promote to 4-panel by splitting the bottom row into a 2x2 sub-grid.
    # Easier: redo as a 3-row 2-col grid with row 2 = panel B + panel C and
    # row 3 = panel D spanning both columns.
    fig.clear()
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.1, 0.9], hspace=0.45, wspace=0.3)
    ax_a_left = fig.add_subplot(gs[0, 0])
    ax_a_right = fig.add_subplot(gs[0, 1])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[2, :])

    # =========================================================================
    # Panel A: cluster × sex (left); cluster × APOE-ε4 (right)
    # =========================================================================
    cs_table = pd.crosstab(
        pd.Categorical(
            mem["cluster"].apply(lambda c: f"k{int(c)}"),
            categories=[f"k{i}" for i in range(int(mem["cluster"].max()) + 1)],
            ordered=True,
        ),
        mem["sex"].fillna("NA"),
    )
    _heatmap(ax_a_left, cs_table, "A. Cluster × Sex (counts)")
    ax_a_left.set_xlabel("Sex")
    ax_a_left.set_ylabel("GMM cluster (k=4)")

    apoe_str = mem["apoe_e4"].apply(
        lambda v: f"ε4={int(v)}" if pd.notna(v) else "NA"
    )
    ca_table = pd.crosstab(
        pd.Categorical(
            mem["cluster"].apply(lambda c: f"k{int(c)}"),
            categories=[f"k{i}" for i in range(int(mem["cluster"].max()) + 1)],
            ordered=True,
        ),
        apoe_str,
    )
    _heatmap(ax_a_right, ca_table, "A. Cluster × APOE-ε4 dose (counts)")
    ax_a_right.set_xlabel("APOE-ε4 dose")
    ax_a_right.set_ylabel("GMM cluster (k=4)")

    # =========================================================================
    # Panel B: residual scatter, cluster colour, CCC outliers highlighted
    # =========================================================================
    rng = np.random.default_rng(42)
    y_jitter = rng.uniform(-0.18, 0.18, size=len(mem))
    cluster_colors = {0: "#4C78A8", 1: "#F58518", 2: "#54A24B", 3: "#B279A2"}
    for c in sorted(mem["cluster"].unique()):
        sub_idx = mem[mem["cluster"] == c].index
        n_total = int(len(sub_idx))
        n_outl = int((mem.loc[sub_idx, "ccc_outlier"] == "Y").sum())
        ax_b.scatter(
            mem.loc[sub_idx, "residual"], y_jitter[sub_idx],
            color=cluster_colors.get(int(c), "0.5"),
            s=14, alpha=0.55, edgecolor="white", linewidth=0.3,
            label=f"k{int(c)} (n={n_total}, ccc-out={n_outl})",
            zorder=2,
        )
    out_idx = mem[mem["ccc_outlier"] == "Y"].index
    ax_b.scatter(
        mem.loc[out_idx, "residual"], y_jitter[out_idx],
        facecolors="none", edgecolor="crimson", s=70, linewidth=1.5,
        label=f"CCC outlier (τ={CANONICAL_CCC_TAU}, n={len(out_idx)})",
        zorder=4,
    )
    ax_b.axvline(0.0, color="0.4", linestyle="--", linewidth=0.7, zorder=0)
    ax_b.set_xlabel("per-subject residual (y_true − y_pred)")
    ax_b.set_yticks([])
    ax_b.set_title("B. CCC outliers overlaid on residual cluster scatter")
    ax_b.legend(fontsize=7, loc="upper right", frameon=True, framealpha=0.9)
    fmt_axes(ax_b)

    # =========================================================================
    # Panel C: per-fold R² stratified by (cluster × sex)
    # =========================================================================
    keys = sorted(per_cluster_sex.keys())
    means = [per_cluster_sex[k]["mean_r2"] for k in keys]
    stds = [per_cluster_sex[k]["std_r2"] for k in keys]
    plot_stds = [s if np.isfinite(s) else 0.0 for s in stds]
    ns = [per_cluster_sex[k]["n_total"] for k in keys]
    x_pos = np.arange(len(keys))
    bar_colors = [
        cluster_colors[per_cluster_sex[k]["cluster"]] for k in keys
    ]
    edge_colors = [
        "black" if per_cluster_sex[k]["sex"] == "F" else "0.2" for k in keys
    ]
    hatch_patterns = [
        "" if per_cluster_sex[k]["sex"] == "F" else "//" for k in keys
    ]
    # Clip the visible range to [-1.5, 1.0] so the within-cluster k0/k2
    # negative-R² strata (structural artefact of conditioning on a residual-
    # axis target — same regime as EXP-035 AD-dx subgroup) don't dominate
    # the y-axis. Off-scale bars get a "↓ R²=N.NN" annotation so the reader
    # still sees the underlying number.
    Y_LO, Y_HI = -1.5, 1.0
    for i, (k, color, hatch, edge) in enumerate(
        zip(keys, bar_colors, hatch_patterns, edge_colors)
    ):
        bar_height = means[i] if np.isfinite(means[i]) else 0.0
        clipped_height = float(np.clip(bar_height, Y_LO, Y_HI))
        clipped_err = (
            plot_stds[i] if np.isfinite(means[i]) and Y_LO <= means[i] <= Y_HI else 0.0
        )
        ax_c.bar(
            x_pos[i], clipped_height, yerr=clipped_err,
            color=color, edgecolor=edge, hatch=hatch,
            capsize=3, alpha=0.85, linewidth=1.0,
        )
        # Off-scale annotation for negative bars that hit the clip floor.
        # Place near the top of the panel for visibility (crimson text on white).
        if np.isfinite(means[i]) and means[i] < Y_LO:
            ax_c.text(
                x_pos[i], -0.4,
                f"R²={means[i]:.1f}\n↓",
                ha="center", va="center", fontsize=8, color="crimson",
                fontweight="bold",
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white",
                      "edgecolor": "crimson", "linewidth": 0.8},
                zorder=5,
            )
        for v in per_cluster_sex[k]["per_fold_r2"]:
            if np.isfinite(v) and Y_LO <= v <= Y_HI:
                ax_c.scatter(
                    x_pos[i] + rng.uniform(-0.1, 0.1), v,
                    color="0.2", s=9, zorder=3, alpha=0.75,
                )
    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels(
        [
            f"{k.replace('_', ' ')}\nn={ns[i]}"
            for i, k in enumerate(keys)
        ],
        fontsize=8,
    )
    ax_c.set_ylim(Y_LO, Y_HI)
    ax_c.axhline(0.0, color="0.7", linewidth=0.6)
    if canonical_per_fold_r2:
        ax_c.axhline(
            float(np.mean(canonical_per_fold_r2)),
            color="crimson", linestyle="--", linewidth=1.0,
            label=f"overall R²={np.mean(canonical_per_fold_r2):.3f}",
        )
        ax_c.legend(fontsize=7, frameon=False, loc="upper right")
    ax_c.set_ylabel("R² (per-fold mean ± std)")
    ax_c.set_title(
        "C. Per-fold R² by (cluster × sex)  [hatch = male; "
        "k0/k2 R² clipped at ↓; structural — see caveat]"
    )
    fmt_axes(ax_c)

    # =========================================================================
    # Panel D: F1 CF success rate per cluster
    # =========================================================================
    cf_keys = sorted(cf_per_cluster.keys())
    cf_rates = [cf_per_cluster[k]["success_rate"] for k in cf_keys]
    cf_n = [cf_per_cluster[k]["n_cf_total"] for k in cf_keys]
    cf_succ = [cf_per_cluster[k]["n_success"] for k in cf_keys]
    cf_fail = [cf_per_cluster[k]["n_fail"] for k in cf_keys]
    cf_x = np.arange(len(cf_keys))
    cf_colors = [cluster_colors[cf_per_cluster[k]["cluster"]] for k in cf_keys]

    ax_d.bar(
        cf_x, cf_rates, color=cf_colors,
        edgecolor="black", linewidth=1.0, alpha=0.85,
    )
    for i, (rate, n, succ, fail) in enumerate(zip(cf_rates, cf_n, cf_succ, cf_fail)):
        if n > 0:
            ax_d.text(
                cf_x[i], rate + 0.02,
                f"{succ}/{n}\n({rate*100:.0f}%)",
                ha="center", va="bottom", fontsize=8,
            )
        else:
            ax_d.text(
                cf_x[i], 0.02,
                "no CF subjects",
                ha="center", va="bottom", fontsize=8, color="0.4",
            )
    ax_d.set_xticks(cf_x)
    ax_d.set_xticklabels(
        [f"{k} (n_cf={cf_n[i]})" for i, k in enumerate(cf_keys)],
    )
    ax_d.set_ylabel("F1-CF success rate")
    ax_d.set_ylim(0.0, max(1.05, max(cf_rates + [0.0]) * 1.2 + 0.05))
    ax_d.set_title(
        "D. EXP-024-stepD F1 counterfactual success rate by GMM cluster "
        "(τ=δ0.3 absolute mode)"
    )
    fmt_axes(ax_d)

    fig.suptitle(
        "EXP-039 patient-stratification cross-link: "
        "GMM cluster × sex × APOE-ε4 × CCC outlier × F1-CF success",
        fontsize=11, y=0.995,
    )
    style_paper_axes(fig)

    out_fig_dir.mkdir(parents=True, exist_ok=True)
    stem = out_fig_dir / "fig_patient_stratification_crosslink"
    paths: list[Path] = []
    for ext in ("png", "pdf"):
        out = stem.with_suffix(f".{ext}")
        fig.savefig(out, dpi=600, bbox_inches="tight")
        paths.append(out)
    plt.close(fig)
    return paths


# =============================================================================
# Markdown summary renderer
# =============================================================================


def render_markdown(
    mem: pd.DataFrame,
    pairwise: dict,
    per_cluster_sex: dict,
    cf_per_cluster: dict,
    gmm_meta: dict,
    canonical_per_fold_r2: list[float] | None,
) -> str:
    lines: list[str] = []
    lines.append("# EXP-039 patient-stratification cross-link")
    lines.append("")
    lines.append(
        "Tests whether four orthogonal subject-stratification axes "
        "(GMM cluster, sex, APOE-ε4 dose, CCC outlier status, F1-CF success) "
        "share overlap structure or are independent."
    )
    lines.append("")
    lines.append("## GMM cluster fit (sanity check vs EXP-033)")
    lines.append("")
    lines.append(f"- n_subjects = {gmm_meta['n_subjects']}")
    lines.append(f"- cluster_sizes = {gmm_meta['cluster_sizes']}")
    lines.append(f"- cluster_means = {gmm_meta['cluster_means']}")
    lines.append("")
    lines.append("## Pairwise independence tests (BH-FDR-corrected)")
    lines.append("")
    lines.append("| Pair | Test | Statistic | p-value | q-value (BH) |")
    lines.append("|---|---|---:|---:|---:|")
    for key, res in pairwise["per_pair"].items():
        stat = res.get("statistic")
        stat_str = "—" if stat is None else f"{stat:.3f}"
        q = res.get("q_value_bh")
        q_str = "—" if q is None else f"{q:.4g}"
        lines.append(
            f"| {key} | {res['test']} | {stat_str} | "
            f"{res['p_value']:.4g} | {q_str} |"
        )
    lines.append("")
    lines.append("## q-value matrix (BH-FDR)")
    lines.append("")
    axes = pairwise["axis_order"]
    header = "| | " + " | ".join(axes) + " |"
    sep = "|" + "|".join(["---"] * (len(axes) + 1)) + "|"
    lines.append(header)
    lines.append(sep)
    for a in axes:
        row = [a]
        for b in axes:
            q = pairwise["q_matrix"][a][b]
            row.append("—" if q is None else f"{q:.4g}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## Per-fold R² by (cluster × sex)")
    lines.append("")
    lines.append("| Stratum | n | per-fold R² | mean ± std |")
    lines.append("|---|---:|---|---|")
    for k in sorted(per_cluster_sex.keys()):
        v = per_cluster_sex[k]
        per_fold = ", ".join(
            f"{x:.3f}" if np.isfinite(x) else "NaN" for x in v["per_fold_r2"]
        )
        std_str = (
            f"{v['std_r2']:.3f}" if np.isfinite(v["std_r2"]) else "NaN"
        )
        mean_str = (
            f"{v['mean_r2']:.3f}" if np.isfinite(v["mean_r2"]) else "NaN"
        )
        lines.append(
            f"| {k} | {v['n_total']} | [{per_fold}] | {mean_str} ± {std_str} |"
        )
    if canonical_per_fold_r2:
        cm = float(np.mean(canonical_per_fold_r2))
        cs = float(np.std(canonical_per_fold_r2, ddof=1))
        lines.append(
            f"| (overall) | {len(mem)} | "
            f"[{', '.join(f'{x:.3f}' for x in canonical_per_fold_r2)}] | "
            f"{cm:.3f} ± {cs:.3f} |"
        )
    lines.append("")
    lines.append("## F1 CF success rate by cluster")
    lines.append("")
    lines.append("| Cluster | n_cf_total | n_success | n_fail | success_rate |")
    lines.append("|---|---:|---:|---:|---:|")
    for k in sorted(cf_per_cluster.keys()):
        v = cf_per_cluster[k]
        rate = (
            f"{v['success_rate']*100:.1f}%"
            if np.isfinite(v["success_rate"])
            else "NaN"
        )
        lines.append(
            f"| {k} | {v['n_cf_total']} | {v['n_success']} | "
            f"{v['n_fail']} | {rate} |"
        )
    lines.append("")

    # Most-significant pairing (smallest q-value among real tests).
    finite = [
        (k, v) for k, v in pairwise["per_pair"].items()
        if v["test"] != "degenerate"
    ]
    if finite:
        ms = min(finite, key=lambda kv: kv[1]["q_value_bh"])
        lines.append("## Most-significant pairing")
        lines.append("")
        lines.append(
            f"- **{ms[0]}**: {ms[1]['test']}, "
            f"p={ms[1]['p_value']:.4g}, q(BH)={ms[1]['q_value_bh']:.4g}, "
            f"n_used={ms[1]['n_used']}"
        )
        lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "If most pairwise q-values are > 0.05, the four axes are **mostly "
        "independent** — patient subgroups defined by residual cluster, sex, "
        "APOE dose, CCC outlier status, and CF success are orthogonal "
        "axes of cohort heterogeneity, not a single 'vulnerable subgroup' "
        "axis driving multiple findings."
    )
    return "\n".join(lines) + "\n"


# =============================================================================
# Main entry
# =============================================================================


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--residual-csv",
        type=Path,
        default=_WORKTREE_ROOT
        / "outputs/canonical/interpretability/residual_per_subject.csv",
    )
    p.add_argument(
        "--metadata-csv",
        type=Path,
        default=_WORKTREE_ROOT / "data/metadata_ROSMAP/metadata.csv",
    )
    p.add_argument(
        "--ccc-threshold-json",
        type=Path,
        default=_WORKTREE_ROOT
        / "outputs/canonical/interpretability/ccc_heterogeneity/threshold_sensitivity.json",
    )
    p.add_argument(
        "--cf-fold0",
        type=Path,
        default=_WORKTREE_ROOT
        / "outputs/canonical/interpretability/counterfactuals_optimized_absolute_delta0p3",
        help="Directory containing fold-0 counterfactuals_fold0.json.",
    )
    p.add_argument(
        "--cf-fold-template",
        type=str,
        default=str(
            _WORKTREE_ROOT
            / "outputs/canonical/interpretability/counterfactuals_optimized_absolute_delta0p3_fold{N}"
        ),
        help="Template (with {N}) for folds 1..N-1 counterfactuals_fold{N}.json.",
    )
    p.add_argument(
        "--pred-root",
        type=Path,
        default=_WORKTREE_ROOT / "outputs/canonical/p5_canonical_seed42",
    )
    p.add_argument(
        "--tabpfn-dir",
        type=Path,
        default=_WORKTREE_ROOT / "data/canonical",
    )
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--n-components", type=int, default=4)
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument("--ccc-tau", type=float, default=CANONICAL_CCC_TAU)
    p.add_argument(
        "--out-json",
        type=Path,
        default=_WORKTREE_ROOT
        / "outputs/canonical/interpretability/patient_stratification_crosslink.json",
    )
    p.add_argument(
        "--out-md",
        type=Path,
        default=_WORKTREE_ROOT
        / "outputs/canonical/interpretability/patient_stratification_crosslink.md",
    )
    p.add_argument(
        "--out-fig-dir",
        type=Path,
        default=_WORKTREE_ROOT
        / "outputs/canonical/interpretability/figures/patient_stratification",
    )
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # ---------------------------------------------------------------- membership
    mem, gmm_meta = build_membership_matrix(
        residual_csv=args.residual_csv,
        metadata_csv=args.metadata_csv,
        ccc_threshold_json=args.ccc_threshold_json,
        cf_fold0=args.cf_fold0,
        cf_fold_template=args.cf_fold_template,
        n_folds=args.n_folds,
        n_components=args.n_components,
        random_state=args.random_state,
        ccc_tau=args.ccc_tau,
    )
    logger.info("Membership matrix: %d subjects, %d columns", len(mem), len(mem.columns))
    logger.info(
        "CCC outliers (τ=%g): %d subjects", args.ccc_tau,
        int((mem["ccc_outlier"] == "Y").sum()),
    )
    logger.info(
        "F1-CF labels: Y=%d, N=%d, N/A=%d",
        int((mem["cf_success"] == "Y").sum()),
        int((mem["cf_success"] == "N").sum()),
        int((mem["cf_success"] == "N/A").sum()),
    )

    # --------------------------------------------------------------- pair tests
    pairwise = run_pairwise_tests(mem)

    # ---------------------------------------------------------- per-(c×sex) R²
    pred_df = load_all_folds(
        args.pred_root, args.tabpfn_dir, n_folds=args.n_folds,
    )
    canonical_per_fold = []
    for f in range(args.n_folds):
        sub = pred_df[pred_df["fold"] == f]
        if len(sub):
            canonical_per_fold.append(
                float(r2_score(sub["y_true"], sub["y_composite"]))
            )

    per_cluster_sex = per_cluster_sex_r2(pred_df, mem, n_folds=args.n_folds)
    cf_per_cluster = cf_success_per_cluster(mem)

    # ---------------------------------------------------------------- write JSON
    out = {
        "config": {
            "n_subjects": int(len(mem)),
            "n_folds": int(args.n_folds),
            "n_components": int(args.n_components),
            "random_state": int(args.random_state),
            "ccc_tau": float(args.ccc_tau),
            "residual_csv": str(args.residual_csv),
            "metadata_csv": str(args.metadata_csv),
            "ccc_threshold_json": str(args.ccc_threshold_json),
            "cf_fold0": str(args.cf_fold0),
            "cf_fold_template": str(args.cf_fold_template),
            "pred_root": str(args.pred_root),
            "tabpfn_dir": str(args.tabpfn_dir),
        },
        "gmm_metadata": gmm_meta,
        "membership_counts": {
            "cluster": {
                f"k{int(c)}": int((mem["cluster"] == c).sum())
                for c in sorted(mem["cluster"].unique())
            },
            "sex": {
                "F": int((mem["sex"] == "F").sum()),
                "M": int((mem["sex"] == "M").sum()),
                "NA": int(mem["sex"].isna().sum()),
            },
            "apoe_e4": {
                f"ε4={int(d)}": int((mem["apoe_e4"] == d).sum())
                for d in [0, 1, 2]
            },
            "apoe_e4_NA": int(mem["apoe_e4"].isna().sum()),
            "ccc_outlier": {
                "Y": int((mem["ccc_outlier"] == "Y").sum()),
                "N": int((mem["ccc_outlier"] == "N").sum()),
            },
            "cf_success": {
                "Y": int((mem["cf_success"] == "Y").sum()),
                "N": int((mem["cf_success"] == "N").sum()),
                "N/A": int((mem["cf_success"] == "N/A").sum()),
            },
        },
        "pairwise_tests": pairwise,
        "per_cluster_sex_r2": per_cluster_sex,
        "cf_success_per_cluster": cf_per_cluster,
        "canonical_per_fold_r2": [float(x) for x in canonical_per_fold],
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w") as fh:
        json.dump(out, fh, indent=2)
    logger.info("Wrote %s", args.out_json)

    # ---------------------------------------------------------- write Markdown
    md = render_markdown(
        mem=mem,
        pairwise=pairwise,
        per_cluster_sex=per_cluster_sex,
        cf_per_cluster=cf_per_cluster,
        gmm_meta=gmm_meta,
        canonical_per_fold_r2=canonical_per_fold,
    )
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md)
    logger.info("Wrote %s", args.out_md)

    # --------------------------------------------------------------- write fig
    fig_paths = render_figure(
        mem=mem,
        per_cluster_sex=per_cluster_sex,
        cf_per_cluster=cf_per_cluster,
        canonical_per_fold_r2=canonical_per_fold,
        out_fig_dir=args.out_fig_dir,
    )
    for fp in fig_paths:
        logger.info("Wrote %s", fp)

    # --------------------------------------------------------------- stdout
    print("=" * 78)
    print("EXP-039 patient-stratification cross-link")
    print("=" * 78)
    print(f"n_subjects = {len(mem)}")
    print(f"GMM k=4 cluster sizes: {gmm_meta['cluster_sizes']}")
    print(f"CCC outliers (τ={args.ccc_tau}): {int((mem['ccc_outlier']=='Y').sum())}")
    print(
        f"CF labels: Y={int((mem['cf_success']=='Y').sum())} "
        f"N={int((mem['cf_success']=='N').sum())} "
        f"N/A={int((mem['cf_success']=='N/A').sum())}"
    )
    print("-" * 78)
    print("Pairwise tests (BH-FDR-corrected):")
    for key, res in pairwise["per_pair"].items():
        q = res.get("q_value_bh")
        print(
            f"  {key:<35s} test={res['test']:<18s}  "
            f"p={res['p_value']:.4g}  q={q if q is None else f'{q:.4g}'}"
        )
    print("-" * 78)
    print("Per-(cluster × sex) mean R²:")
    for k in sorted(per_cluster_sex.keys()):
        v = per_cluster_sex[k]
        std_str = f"{v['std_r2']:.3f}" if np.isfinite(v["std_r2"]) else "NaN"
        print(
            f"  {k:<10s}  n={v['n_total']:>3d}  mean R²={v['mean_r2']:7.3f} "
            f"± {std_str}"
        )
    print("-" * 78)
    print("F1-CF success rate per cluster:")
    for k in sorted(cf_per_cluster.keys()):
        v = cf_per_cluster[k]
        rate = (
            f"{v['success_rate']*100:.1f}%"
            if np.isfinite(v["success_rate"])
            else "NaN"
        )
        print(
            f"  {k:<5s}  n_cf={v['n_cf_total']:>3d}  "
            f"success={v['n_success']:>3d}  fail={v['n_fail']:>3d}  rate={rate}"
        )
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
