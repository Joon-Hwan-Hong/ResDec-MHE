"""F7 follow-up — fit GMM with k=4 on per-subject residuals and cross-tabulate
the resulting cluster assignments against pathology / APOE / sex / age.

The canonical latent-class analysis (`latent_class_on_residuals.json`) selects
k=2 by BIC, but AIC marginally favors k=4 (ΔAIC < 3 vs k=2 — substantially
supported alternative per Burnham & Anderson 2002). This script tests whether
the k=4 substructure is biologically meaningful by stratifying the n=516
subjects against:

  - Braak stage (braaksc, 0..6)
  - APOE genotype (apoe_genotype, 23/33/34/44/etc.)
  - Sex (msex, 0=F / 1=M)
  - Age band (age_death, <80 / 80-90 / 90+)

For each cross-tab a chi-square independence test (`scipy.stats.chi2_contingency`)
returns chi2, dof, p-value; a covariate with p < 0.05 indicates non-trivial
dependence between the GMM-assigned latent class and that covariate.

Inputs (defaults; CLI-overridable):
    --residual-csv  outputs/canonical/interpretability/residual_per_subject.csv
    --metadata-csv  data/metadata_ROSMAP/metadata.csv

Output:
    --out-json      outputs/canonical/interpretability/latent_class_k4_crosstab.json

Cluster fit reproduces the user-procedure exactly:
    GaussianMixture(n_components=4, random_state=0)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.mixture import GaussianMixture

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

logger = logging.getLogger(__name__)


def _age_band(age: float):
    """Bucket age_death into <80 / 80-90 / 90+. Returns ``np.nan`` if missing
    so the cross-tab routine drops the row rather than treating "NA" as a
    valid category."""
    if not np.isfinite(age):
        return np.nan
    if age < 80:
        return "<80"
    if age < 90:
        return "80-90"
    return "90+"


def _crosstab_chi2(
    cluster: np.ndarray,
    cov: np.ndarray,
    cov_name: str,
) -> dict:
    """Build a cluster x covariate contingency table and run chi2_contingency.

    Excludes rows where the covariate is NaN. Returns a dict with the table
    (as nested dict of int counts), dropped n, chi2, dof, p-value.
    """
    if cov.dtype.kind in {"i", "u"} or cov_name in {"braaksc"}:
        # treat numeric category labels (Braak, msex) as discrete
        mask = np.isfinite(cov.astype(float))
    else:
        mask = pd.notna(cov)
    n_dropped = int(np.sum(~mask))
    cluster_kept = cluster[mask]
    cov_kept = cov[mask]
    table = pd.crosstab(
        pd.Series(cluster_kept, name="cluster"),
        pd.Series(cov_kept, name=cov_name),
    )
    if table.size == 0 or min(table.shape) < 2:
        return {
            "table": table.to_dict(),
            "row_index": [int(x) if isinstance(x, (np.integer, int)) else str(x) for x in table.index.tolist()],
            "col_index": [str(x) for x in table.columns.tolist()],
            "n_used": int(np.sum(mask)),
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
        "n_used": int(np.sum(mask)),
        "n_dropped_missing": n_dropped,
        "chi2": float(chi2),
        "dof": int(dof),
        "p_value": float(p_val),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--residual-csv",
        type=Path,
        default=_WORKTREE_ROOT
        / "outputs/canonical/interpretability/residual_per_subject.csv",
        help="Per-subject residuals + projid (axis convention from canonical pipeline).",
    )
    p.add_argument(
        "--metadata-csv",
        type=Path,
        default=_WORKTREE_ROOT / "data/metadata_ROSMAP/metadata.csv",
        help="ROSMAP metadata CSV with apoe_genotype, msex, age_death, braaksc.",
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=_WORKTREE_ROOT
        / "outputs/canonical/interpretability/latent_class_k4_crosstab.json",
    )
    p.add_argument("--n-components", type=int, default=4)
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.residual_csv.exists():
        raise FileNotFoundError(f"--residual-csv not found: {args.residual_csv}")
    if not args.metadata_csv.exists():
        raise FileNotFoundError(f"--metadata-csv not found: {args.metadata_csv}")

    logger.info("Loading residuals from %s", args.residual_csv)
    residual_df = pd.read_csv(args.residual_csv)
    if "residual" not in residual_df.columns:
        raise ValueError(
            f"residual_per_subject.csv missing 'residual' column; got {residual_df.columns.tolist()}"
        )
    if "projid" not in residual_df.columns:
        raise ValueError(
            f"residual_per_subject.csv missing 'projid' column; got {residual_df.columns.tolist()}"
        )
    n_total = len(residual_df)
    logger.info("Loaded %d rows from residual CSV", n_total)

    finite_mask = np.isfinite(residual_df["residual"].to_numpy())
    n_finite = int(finite_mask.sum())
    if n_finite < args.n_components:
        raise ValueError(
            f"Need at least n_components={args.n_components} finite residuals; got {n_finite}"
        )
    if n_finite < n_total:
        logger.warning(
            "Dropping %d rows with non-finite residual (kept %d/%d)",
            n_total - n_finite, n_finite, n_total,
        )
    residual_df = residual_df.loc[finite_mask].reset_index(drop=True)

    residuals = residual_df["residual"].to_numpy(dtype=np.float64).reshape(-1, 1)

    logger.info(
        "Fitting GaussianMixture(n_components=%d, random_state=%d)",
        args.n_components, args.random_state,
    )
    gmm = GaussianMixture(
        n_components=args.n_components,
        random_state=args.random_state,
    )
    gmm.fit(residuals)
    cluster = gmm.predict(residuals)
    means = gmm.means_.ravel().astype(float).tolist()
    stds = np.sqrt(gmm.covariances_.reshape(-1)).astype(float).tolist()
    weights = gmm.weights_.astype(float).tolist()
    logger.info("Cluster means: %s", means)
    logger.info("Cluster sizes (k=%d): %s", args.n_components, np.bincount(cluster).tolist())

    # Merge metadata by projid (residual CSV records int projid; metadata also int).
    logger.info("Loading metadata from %s", args.metadata_csv)
    meta_df = pd.read_csv(args.metadata_csv, low_memory=False)
    needed = ["projid", "apoe_genotype", "msex", "age_death", "braaksc"]
    missing = [c for c in needed if c not in meta_df.columns]
    if missing:
        raise ValueError(f"metadata.csv missing columns: {missing}")

    merged = residual_df[["projid", "residual"]].merge(
        meta_df[needed],
        on="projid",
        how="left",
        validate="many_to_one",
    )
    merged["cluster"] = cluster

    # Quick coverage check
    for col in ("apoe_genotype", "msex", "age_death", "braaksc"):
        n_missing = int(merged[col].isna().sum())
        if n_missing:
            logger.warning("%d/%d subjects missing %s", n_missing, len(merged), col)

    # Build covariate arrays. APOE: keep NaN as NaN so _crosstab_chi2 drops
    # those rows rather than introducing a degenerate "NA" category.
    braak = merged["braaksc"].to_numpy()

    def _apoe_to_str(v):
        if isinstance(v, float) and np.isnan(v):
            return np.nan
        if isinstance(v, (int, float, np.integer, np.floating)):
            f = float(v)
            return str(int(f)) if f.is_integer() else str(f)
        return str(v)

    apoe_str = merged["apoe_genotype"].apply(_apoe_to_str).to_numpy(dtype=object)
    sex = merged["msex"].to_numpy()
    age_band = merged["age_death"].apply(_age_band).to_numpy()

    crosstab_braak = _crosstab_chi2(cluster, braak, "braaksc")
    crosstab_apoe = _crosstab_chi2(cluster, apoe_str, "apoe_genotype")
    crosstab_sex = _crosstab_chi2(cluster, sex, "msex")
    crosstab_age = _crosstab_chi2(cluster, age_band, "age_band")

    output = {
        "config": {
            "n_components": int(args.n_components),
            "random_state": int(args.random_state),
            "covariance_type": "full (sklearn default)",
            "n_init_default": 1,
        },
        "n_subjects": int(len(merged)),
        "n_subjects_dropped_nonfinite_residual": int(n_total - n_finite),
        "cluster_sizes": {
            f"cluster_{i}": int(cnt)
            for i, cnt in enumerate(np.bincount(cluster, minlength=args.n_components).tolist())
        },
        "cluster_centers": {f"cluster_{i}": float(m) for i, m in enumerate(means)},
        "cluster_stds": {f"cluster_{i}": float(s) for i, s in enumerate(stds)},
        "cluster_weights": {f"cluster_{i}": float(w) for i, w in enumerate(weights)},
        "crosstabs": {
            "braaksc": crosstab_braak,
            "apoe_genotype": crosstab_apoe,
            "msex": crosstab_sex,
            "age_band": crosstab_age,
        },
    }

    # Verdict: list covariates with p < 0.05.
    sig_covariates = []
    for cov_name, ct in output["crosstabs"].items():
        pv = ct.get("p_value")
        if pv is not None and pv < 0.05:
            sig_covariates.append({"covariate": cov_name, "p_value": float(pv), "chi2": float(ct["chi2"]), "dof": int(ct["dof"])})
    output["significant_covariates_p_lt_0p05"] = sig_covariates
    output["any_significant"] = bool(sig_covariates)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w") as fh:
        json.dump(output, fh, indent=2)
    logger.info("Wrote %s", args.out_json)

    print("=" * 78)
    print(f"GMM k={args.n_components} on n={len(merged)} residuals (random_state={args.random_state})")
    print(f"Cluster sizes: {output['cluster_sizes']}")
    print(f"Cluster centers (means): {output['cluster_centers']}")
    print(f"Cluster stds: {output['cluster_stds']}")
    print(f"Cluster weights: {output['cluster_weights']}")
    print("-" * 78)
    for cov_name, ct in output["crosstabs"].items():
        pv = ct.get("p_value")
        chi2_val = ct.get("chi2")
        dof = ct.get("dof")
        print(
            f"  {cov_name:16s}  chi2={chi2_val if chi2_val is None else f'{chi2_val:.3f}':>10s}  "
            f"dof={dof}  p={pv if pv is None else f'{pv:.4g}'}  n_used={ct['n_used']}  n_dropped={ct['n_dropped_missing']}"
        )
    print("-" * 78)
    if sig_covariates:
        print(f"VERDICT: {len(sig_covariates)} covariate(s) with p<0.05:")
        for s in sig_covariates:
            print(f"  - {s['covariate']}: chi2={s['chi2']:.3f}, dof={s['dof']}, p={s['p_value']:.4g}")
    else:
        print("VERDICT: NO covariate has p<0.05 — k=4 substructure is NOT meaningfully")
        print("         tied to pathology / APOE / sex / age.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
