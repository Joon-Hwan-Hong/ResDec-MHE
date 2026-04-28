"""
Per-subject Splatter heterogeneity analysis.

For each of 516 subjects, computes Splatter (CT idx=30) cell counts (total +
PFC) and per-marker mean expression on the SAMPLED cell-level data
(`cell_data` in the precomputed .pt files). Marker genes: SST, CHODL, NPY,
NOS1, TAC1, LHX6.

Notes on data semantics (verified against `src/data/datasets.py`):
- `cell_counts[ct]`: TRUE total per-CT cell count across all available
  regions for that subject (no cap, no min-threshold filter).
- `cell_data[offsets[ct]:offsets[ct+1]]`: SAMPLED cells used for cell-level
  branches. Capped at `max_cells_per_type=1000`; CTs with fewer than
  `min_cells_threshold=50` cells are zeroed out (slice has length 0).
- `region_0_pseudobulk[ct]`: PFC-only mean expression. Non-zero values
  indicate Splatter is present in PFC for that subject. Per-region cell
  counts are NOT stored, so we cannot exactly recover them for multi-region
  subjects.

Strategy (chosen to be faithful — no silent simplification):
- `n_splatter_cells_total`: from `cell_counts[30]` (true global count).
- `n_splatter_cells_pfc`: equals `cell_counts[30]` when the subject only
  has PFC (`available_regions == [0]`); otherwise NaN, with a separate
  binary indicator `splatter_present_in_pfc` derived from
  `region_0_pseudobulk[30]` having any non-zero gene.
- Marker means: average across `cell_data[offsets[30]:offsets[31]]`. NaN if
  no Splatter cells in `cell_data` (slice length 0 — happens when global
  count < min_cells_threshold = 50).

Outputs:
- `outputs/canonical/interpretability/splatter_per_subject_heterogeneity.json`
- `outputs/canonical/interpretability/splatter_per_subject_features.csv`
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.mixture import GaussianMixture

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PRECOMPUTED_DIR = PROJECT_ROOT / "data" / "precomputed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "redesign" / "interpretability"
RESIDUAL_CSV = OUTPUTS_DIR / "residual_per_subject.csv"
GENE_NAMES_PATH = PRECOMPUTED_DIR / "gene_names.npy"

SPLATTER_IDX = 30  # last entry in CELL_TYPE_ORDER
PFC_REGION_IDX = 0
MARKERS = ["SST", "CHODL", "NPY", "NOS1", "TAC1", "LHX6"]
FEATURES = [
    "n_splatter_cells_total",
    "n_splatter_cells_pfc",
    "mean_SST",
    "mean_CHODL",
    "mean_NPY",
    "mean_NOS1",
    "mean_TAC1",
    "mean_LHX6",
]


def load_gene_indices(gene_names_path: Path, markers: list[str]) -> dict[str, int]:
    """Map marker gene names to their column indices in the HVG list."""
    gene_names = np.load(gene_names_path, allow_pickle=True)
    indices: dict[str, int] = {}
    for marker in markers:
        hits = np.where(gene_names == marker)[0]
        if len(hits) == 0:
            raise ValueError(f"Marker {marker!r} not found in HVG list")
        indices[marker] = int(hits[0])
    return indices


def compute_subject_features(
    pt_path: Path,
    marker_idx: dict[str, int],
) -> dict[str, float | int | bool]:
    """Compute Splatter features for a single subject .pt file."""
    d = torch.load(pt_path, weights_only=False, map_location="cpu")

    cell_counts = d["cell_counts"]
    cell_offsets = d["cell_offsets"]
    cell_data = d["cell_data"]
    available_regions: list[int] = list(d["available_regions"])
    region_0_pb = d.get("region_0_pseudobulk", None)

    n_total = int(cell_counts[SPLATTER_IDX].item())

    # PFC-only count: exact only when subject is PFC-only
    if available_regions == [PFC_REGION_IDX]:
        n_pfc: float = float(n_total)
    else:
        n_pfc = float("nan")

    # Indicator: does Splatter appear in PFC region for this subject?
    if region_0_pb is not None:
        splatter_present_in_pfc = bool((region_0_pb[SPLATTER_IDX] != 0).any().item())
    else:
        splatter_present_in_pfc = False

    # Marker means over SAMPLED Splatter cells
    start = int(cell_offsets[SPLATTER_IDX].item())
    end = int(cell_offsets[SPLATTER_IDX + 1].item())
    n_sampled = end - start

    marker_means: dict[str, float] = {}
    if n_sampled > 0:
        slc = cell_data[start:end]
        for m in MARKERS:
            marker_means[f"mean_{m}"] = float(slc[:, marker_idx[m]].mean().item())
    else:
        for m in MARKERS:
            marker_means[f"mean_{m}"] = float("nan")

    return {
        "n_splatter_cells_total": n_total,
        "n_splatter_cells_pfc": n_pfc,
        "n_splatter_cells_sampled": int(n_sampled),
        "splatter_present_in_pfc": splatter_present_in_pfc,
        "available_regions": ";".join(str(r) for r in available_regions),
        **marker_means,
    }


def safe_corr(x: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
    """Pearson + Spearman with NaN/zero-variance guards."""
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < 3 or np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
        return {
            "pearson_r": float("nan"),
            "pearson_p": float("nan"),
            "spearman_r": float("nan"),
            "spearman_p": float("nan"),
            "n": n,
        }
    pr, pp = stats.pearsonr(x[mask], y[mask])
    sr, sp = stats.spearmanr(x[mask], y[mask])
    return {
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_r": float(sr),
        "spearman_p": float(sp),
        "n": n,
    }


def summary_stats(arr: np.ndarray) -> dict[str, float | int]:
    """Descriptive stats with NaN handling."""
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {k: float("nan") for k in ("mean", "std", "median", "q25", "q75", "min", "max")} | {"n_finite": 0}
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan"),
        "median": float(np.median(finite)),
        "q25": float(np.quantile(finite, 0.25)),
        "q75": float(np.quantile(finite, 0.75)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "n_finite": int(finite.size),
    }


def fit_gmm_bic(X: np.ndarray, k_range: range, seed: int = 42) -> dict:
    """Fit GMM at each k and return BIC + means.

    Z-scores X (drops rows with any NaN) before fitting. Reports per-k
    BIC, per-k component means in z-score space, and per-k component
    means back-transformed to original feature scale.
    """
    valid_mask = np.all(np.isfinite(X), axis=1)
    X_valid = X[valid_mask]
    n_used = X_valid.shape[0]

    mu = X_valid.mean(axis=0)
    sd = X_valid.std(axis=0, ddof=1)
    sd_safe = np.where(sd > 0, sd, 1.0)
    Z = (X_valid - mu) / sd_safe

    results = {
        "n_subjects_used": int(n_used),
        "n_subjects_dropped_nan": int((~valid_mask).sum()),
        "feature_means_for_zscore": mu.tolist(),
        "feature_stds_for_zscore": sd.tolist(),
        "per_k": {},
    }

    best_k = None
    best_bic = float("inf")
    for k in k_range:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=seed,
            max_iter=500,
            n_init=5,
            reg_covar=1e-6,
        )
        gmm.fit(Z)
        bic = float(gmm.bic(Z))
        # Back-transform component means (z-space) → original scale
        z_means = gmm.means_  # (k, d)
        orig_means = z_means * sd_safe + mu
        # Hard cluster assignments
        labels = gmm.predict(Z)
        counts = np.bincount(labels, minlength=k).tolist()

        results["per_k"][str(k)] = {
            "bic": bic,
            "aic": float(gmm.aic(Z)),
            "log_likelihood": float(gmm.score(Z) * n_used),
            "weights": gmm.weights_.tolist(),
            "means_zscore": z_means.tolist(),
            "means_original_scale": orig_means.tolist(),
            "cluster_sizes": counts,
            "converged": bool(gmm.converged_),
        }
        if bic < best_bic:
            best_bic = bic
            best_k = int(k)

    results["best_k_by_bic"] = best_k
    results["best_bic"] = best_bic
    return results


def get_git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True
        )
        return out.strip()
    except subprocess.CalledProcessError:
        return "unknown"


def main() -> None:
    print(f"[{datetime.now().isoformat()}] Loading inputs...")

    # Marker gene indices
    marker_idx = load_gene_indices(GENE_NAMES_PATH, MARKERS)
    print(f"  Marker indices: {marker_idx}")

    # Residual CSV (column 'ROSMAP_IndividualID' for subject ID)
    residual_df = pd.read_csv(RESIDUAL_CSV)
    if "ROSMAP_IndividualID" not in residual_df.columns or "residual" not in residual_df.columns:
        raise KeyError(
            f"Expected columns ROSMAP_IndividualID, residual in {RESIDUAL_CSV}; got {list(residual_df.columns)}"
        )
    residual_df = residual_df[["ROSMAP_IndividualID", "residual", "fold"]].rename(
        columns={"ROSMAP_IndividualID": "subject"}
    )
    print(f"  Residual rows: {len(residual_df)}")

    # All subject .pt files
    pt_files = sorted(PRECOMPUTED_DIR.glob("R*.pt"))
    print(f"  Found {len(pt_files)} subject .pt files")

    # Per-subject features
    rows = []
    for i, pt in enumerate(pt_files):
        sid = pt.stem
        feats = compute_subject_features(pt, marker_idx)
        feats["subject"] = sid
        rows.append(feats)
        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{len(pt_files)}] processed")

    feat_df = pd.DataFrame(rows)

    # Merge with residual on subject
    merged = residual_df.merge(feat_df, on="subject", how="inner")
    print(f"\nMerged rows (subject ∩ residual): {len(merged)}")
    if len(merged) != len(residual_df):
        missing = set(residual_df["subject"]) - set(feat_df["subject"])
        print(f"  WARNING: {len(missing)} residual subjects without .pt: {sorted(missing)[:5]}...")

    n_with_splatter = int((merged["n_splatter_cells_total"] > 0).sum())
    n_with_sampled = int((merged["n_splatter_cells_sampled"] > 0).sum())
    print(f"  Subjects with Splatter (count > 0): {n_with_splatter}")
    print(f"  Subjects with Splatter cells in cell_data (≥ min_cells_threshold=50): {n_with_sampled}")

    # CSV output (canonical column order)
    csv_path = OUTPUTS_DIR / "splatter_per_subject_features.csv"
    csv_cols = ["subject", "residual", "fold"] + FEATURES + [
        "n_splatter_cells_sampled",
        "splatter_present_in_pfc",
        "available_regions",
    ]
    merged[csv_cols].to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    # Correlations vs residual
    residual_arr = merged["residual"].to_numpy()
    corr_results: dict[str, dict] = {}
    for f in FEATURES:
        feat_arr = merged[f].to_numpy(dtype=float)
        corr_results[f] = safe_corr(feat_arr, residual_arr)

    # Summary stats per feature
    summary: dict[str, dict] = {f: summary_stats(merged[f].to_numpy(dtype=float)) for f in FEATURES}

    # GMM heterogeneity test (drop n_splatter_cells_pfc due to multi-region NaNs;
    # keep all other features). Also fit a "non-zero Splatter only" GMM since
    # subjects with 0 cells will have NaN marker means.
    all_features_no_pfc = [f for f in FEATURES if f != "n_splatter_cells_pfc"]
    X_all = merged[all_features_no_pfc].to_numpy(dtype=float)
    gmm_all = fit_gmm_bic(X_all, range(1, 6))
    gmm_all["features_used"] = all_features_no_pfc

    # Subset to subjects with sampled Splatter cells (no NaN markers)
    has_sampled = merged["n_splatter_cells_sampled"] > 0
    X_sampled = merged.loc[has_sampled, all_features_no_pfc].to_numpy(dtype=float)
    gmm_sampled = fit_gmm_bic(X_sampled, range(1, 6))
    gmm_sampled["features_used"] = all_features_no_pfc

    json_path = OUTPUTS_DIR / "splatter_per_subject_heterogeneity.json"
    out_json = {
        "n_subjects": int(len(merged)),
        "n_subjects_with_splatter": n_with_splatter,
        "n_subjects_with_splatter_sampled": n_with_sampled,
        "features": FEATURES,
        "marker_gene_hvg_indices": marker_idx,
        "data_semantics_note": (
            "n_splatter_cells_total uses cell_counts[30] (global). "
            "n_splatter_cells_pfc equals cell_counts[30] iff available_regions==[0]; NaN otherwise. "
            "Marker means are over cell_data[offsets[30]:offsets[31]] (sampled, cap=1000, "
            "min_cells_threshold=50). Subjects with global count < 50 have empty slices "
            "and NaN marker means."
        ),
        "correlation_with_residual": corr_results,
        "summary_stats_per_feature": summary,
        "gmm_results": {
            "gmm_all_subjects": gmm_all,
            "gmm_subjects_with_sampled_splatter": gmm_sampled,
        },
        "provenance": {
            "git_commit": get_git_commit(),
            "timestamp": datetime.now().isoformat(),
            "script": str(Path(__file__).resolve()),
        },
    }

    with json_path.open("w") as fh:
        json.dump(out_json, fh, indent=2, default=lambda o: None if (isinstance(o, float) and np.isnan(o)) else o)
    print(f"Wrote {json_path}")

    # Quick text summary printed to stdout
    print("\n=== TOP CORRELATIONS BY |pearson_r| ===")
    sorted_feats = sorted(
        FEATURES,
        key=lambda f: abs(corr_results[f]["pearson_r"]) if not np.isnan(corr_results[f]["pearson_r"]) else -1,
        reverse=True,
    )
    for f in sorted_feats:
        c = corr_results[f]
        print(
            f"  {f:30s}  pearson r={c['pearson_r']:+.4f} (p={c['pearson_p']:.4g}, n={c['n']}) "
            f"spearman r={c['spearman_r']:+.4f} (p={c['spearman_p']:.4g})"
        )

    print(f"\n=== GMM BIC (all 516 subjects, NaN rows dropped) ===")
    for k in range(1, 6):
        rec = gmm_all["per_k"][str(k)]
        print(f"  k={k}: BIC={rec['bic']:.2f}  AIC={rec['aic']:.2f}  weights={[f'{w:.2f}' for w in rec['weights']]}")
    print(f"  best k by BIC: {gmm_all['best_k_by_bic']} (n_used={gmm_all['n_subjects_used']})")

    print(f"\n=== GMM BIC (subjects with sampled Splatter cells only) ===")
    for k in range(1, 6):
        rec = gmm_sampled["per_k"][str(k)]
        print(f"  k={k}: BIC={rec['bic']:.2f}  AIC={rec['aic']:.2f}  weights={[f'{w:.2f}' for w in rec['weights']]}")
    print(f"  best k by BIC: {gmm_sampled['best_k_by_bic']} (n_used={gmm_sampled['n_subjects_used']})")


if __name__ == "__main__":
    main()
