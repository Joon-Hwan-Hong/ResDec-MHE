"""Inference-only permutation test (strategy (a) — no model retraining).

Standard permutation-test interpretation: the trained model and its predictions
are FIXED; only the cognition labels (cogn_global) are permuted. Per
permutation seed k:

  1. Permute the finite cogn_global values across all finite-label
     subjects in metadata (NaN positions preserved per the existing
     ``generate_shuffled_metadata`` contract).
  2. For each fold ``f`` in {0, ..., n_folds-1}, load the canonical
     ``val_predictions_best.npz`` (FIXED predictions + subject IDs),
     look up each subject's SHUFFLED cogn_global, and compute
     ``r2_score(shuffled_y_fold, canonical_predictions_fold)``.
  3. Mean across folds → ``mean_r2_per_perm[k]``.

Aggregate across N perms:
  - ``null_mean_r2_per_perm`` — list of per-perm mean R²
  - ``null_mean``, ``null_std`` — distribution moments
  - ``z_under_null`` — (canonical - null_mean) / null_std
  - ``n_perms_ge_canonical`` — count of perms with mean R² >= canonical
  - ``p_value_one_sided`` — (1 + #perms >= canonical) / (N + 1)

Cost: ~1 sec per perm × N perms; orders of magnitude cheaper than the full
re-training permutation test in ``run_permutation_test.py``.

Outputs:
  <output-base>/permutation_results.json  — per-perm records
  <output-base>/permutation_summary.json  — aggregate moments + p-value

Usage:
  uv run python scripts/resdec_mhe/training/run_permutation_test_inference_only.py \\
      --num-perms 100 \\
      --canonical-dir outputs/canonical/p5_canonical_seed42 \\
      --output-base outputs/canonical/permutation_test_inference_only
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TARGET = "cogn_global"
DEFAULT_ID_COL = "ROSMAP_IndividualID"


def load_canonical_predictions(
    canonical_dir: Path,
    n_folds: int,
    pred_filename: str = "val_predictions_best.npz",
) -> list[dict]:
    """Load per-fold canonical predictions.

    Returns
    -------
    list of length ``n_folds``; each entry is a dict
    ``{"subject_ids": list[str], "predictions": np.ndarray, "targets": np.ndarray}``.
    ``targets`` is the cogn_global value the canonical run actually scored
    against (used for the canonical-mean-R² recomputation).
    """
    folds: list[dict] = []
    for f in range(n_folds):
        npz_path = canonical_dir / f"fold{f}" / pred_filename
        if not npz_path.exists():
            raise FileNotFoundError(
                f"Canonical predictions not found at {npz_path}. "
                "Inference-only permutation test requires per-fold "
                f"{pred_filename} from a completed canonical run."
            )
        npz = np.load(npz_path, allow_pickle=True)
        folds.append({
            "subject_ids": [str(s) for s in npz["subject_ids"]],
            "predictions": np.asarray(npz["predictions"], dtype=np.float64),
            "targets": np.asarray(npz["targets"], dtype=np.float64),
        })
    return folds


def shuffle_finite_labels(
    base_metadata_csv: Path,
    perm_seed: int,
    target_col: str,
    id_col: str,
) -> dict[str, float]:
    """Permute finite values of ``target_col`` across non-NaN subjects.

    Mirrors the contract of ``run_permutation_test.generate_shuffled_metadata``:
    NaN positions stay NaN; only finite values are permuted among non-NaN
    positions. Returns a dict ``{subject_id: shuffled_y}`` for downstream
    R² lookup; NaN subjects retain ``np.nan`` in the dict.
    """
    df = pd.read_csv(base_metadata_csv)
    rng = np.random.default_rng(perm_seed)
    vals = df[target_col].values.astype(np.float64)
    finite_mask = np.isfinite(vals)
    permuted = vals.copy()
    permuted[finite_mask] = rng.permutation(vals[finite_mask])
    return dict(zip(df[id_col].astype(str), permuted))


def compute_per_fold_r2(
    folds: list[dict],
    y_lookup: dict[str, float],
) -> tuple[list[float], list[int]]:
    """For each fold, lookup ``y_lookup[subject_id]`` and compute R²(y, canonical_pred).

    Subjects whose shuffled y is NaN are dropped from that fold's
    R² calculation (consistent with how the canonical pipeline excludes
    NaN-target subjects from validation).
    """
    per_fold_r2: list[float] = []
    per_fold_n: list[int] = []
    for fold in folds:
        sids = fold["subject_ids"]
        preds = fold["predictions"]
        y_shuffled = np.array(
            [y_lookup.get(s, np.nan) for s in sids], dtype=np.float64,
        )
        keep = np.isfinite(y_shuffled)
        if not keep.any():
            per_fold_r2.append(np.nan)
            per_fold_n.append(0)
            continue
        r2 = float(r2_score(y_shuffled[keep], preds[keep]))
        per_fold_r2.append(r2)
        per_fold_n.append(int(keep.sum()))
    return per_fold_r2, per_fold_n


def compute_canonical_mean_r2(folds: list[dict]) -> float:
    """Re-compute canonical mean R² from each fold's stored ``targets``+``predictions``."""
    per_fold = []
    for fold in folds:
        per_fold.append(r2_score(fold["targets"], fold["predictions"]))
    return float(np.mean(per_fold))


def run_permutation_test_inference_only(
    canonical_dir: Path,
    base_metadata_csv: Path,
    output_base: Path,
    num_perms: int,
    start_perm: int,
    n_folds: int,
    target_col: str,
    id_col: str,
    pred_filename: str,
) -> dict:
    """Run the inference-only permutation test and write per-perm + summary JSON."""
    output_base.mkdir(parents=True, exist_ok=True)

    folds = load_canonical_predictions(canonical_dir, n_folds, pred_filename)
    canonical_mean_r2 = compute_canonical_mean_r2(folds)

    perm_records: list[dict] = []
    aggregate_path = output_base / "permutation_results.json"
    if aggregate_path.exists():
        try:
            perm_records = json.loads(aggregate_path.read_text())
        except json.JSONDecodeError:
            perm_records = []

    # De-dupe on resume: skip seeds already in the persisted records so an
    # overlapping --start-perm range doesn't double-count.
    existing_seeds = {r["perm_seed"] for r in perm_records}

    t_total = time.time()
    for k in range(start_perm, start_perm + num_perms):
        if k in existing_seeds:
            continue
        t0 = time.time()
        y_lookup = shuffle_finite_labels(
            base_metadata_csv, perm_seed=k,
            target_col=target_col, id_col=id_col,
        )
        per_fold_r2, per_fold_n = compute_per_fold_r2(folds, y_lookup)
        # Compute fold-mean ignoring NaN folds (no kept subjects).
        finite_r2 = [r for r in per_fold_r2 if np.isfinite(r)]
        mean_r2 = float(np.mean(finite_r2)) if finite_r2 else float("nan")
        elapsed_s = time.time() - t0
        perm_records.append({
            "perm_seed": k,
            "per_fold_r2": per_fold_r2,
            "per_fold_n": per_fold_n,
            "mean_r2": mean_r2,
            "elapsed_s": round(elapsed_s, 4),
        })
        # Persist incrementally so a long sweep can be resumed.
        aggregate_path.write_text(json.dumps(perm_records, indent=2))

    # Build summary using the records that were produced this invocation
    # (matches the schema of outputs/canonical/permutation_test/permutation_summary.json).
    null_means = np.array(
        [r["mean_r2"] for r in perm_records if np.isfinite(r["mean_r2"])],
        dtype=np.float64,
    )
    n_perms = int(null_means.size)
    null_mean = float(null_means.mean()) if n_perms > 0 else float("nan")
    # Population std (ddof=0) — matches existing canonical N=10 perm summary
    # convention (verified: (0.4436 - (-0.2944)) / 0.0845 = 8.73).
    null_std = float(null_means.std(ddof=0)) if n_perms > 1 else 0.0
    n_perms_ge_canonical = int((null_means >= canonical_mean_r2).sum())
    p_value_one_sided = (1 + n_perms_ge_canonical) / (n_perms + 1) if n_perms > 0 else float("nan")
    z_under_null = (
        (canonical_mean_r2 - null_mean) / null_std
        if null_std > 0 else float("nan")
    )

    summary = {
        "method": "inference-only permutation test (strategy (a))",
        "description": (
            "Canonical model predictions are FIXED; cogn_global labels are "
            "permuted across finite-label subjects per perm seed. R² is "
            "computed against shuffled labels per fold, then averaged."
        ),
        "canonical_mean_r2": canonical_mean_r2,
        "n_permutations": n_perms,
        "perm_seeds": [r["perm_seed"] for r in perm_records],
        "null_mean_r2_per_perm": [r["mean_r2"] for r in perm_records],
        "null_mean": null_mean,
        "null_std": null_std,
        "null_min": float(null_means.min()) if n_perms > 0 else float("nan"),
        "null_max": float(null_means.max()) if n_perms > 0 else float("nan"),
        "n_perms_ge_canonical": n_perms_ge_canonical,
        "p_value_one_sided": p_value_one_sided,
        "p_value_formula": "(1 + #perms≥obs) / (N + 1)",
        "z_under_null": z_under_null,
        "n_folds": n_folds,
        "pred_filename": pred_filename,
        "elapsed_total_s": round(time.time() - t_total, 2),
    }
    summary_path = output_base / "permutation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--num-perms", type=int, default=100)
    p.add_argument("--start-perm", type=int, default=0)
    p.add_argument(
        "--output-base",
        default=str(ROOT / "outputs" / "canonical" / "permutation_test_inference_only"),
    )
    p.add_argument(
        "--canonical-dir",
        default=str(ROOT / "outputs" / "canonical" / "p5_canonical_seed42"),
        help="Directory holding fold{0..n_folds-1}/<pred-filename>",
    )
    p.add_argument(
        "--base-metadata-csv",
        default=str(ROOT / "data" / "metadata_ROSMAP" / "metadata.csv"),
    )
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--target-col", default=DEFAULT_TARGET)
    p.add_argument("--id-col", default=DEFAULT_ID_COL)
    p.add_argument(
        "--pred-filename", default="val_predictions_best.npz",
        help="Per-fold prediction filename (default val_predictions_best.npz).",
    )
    args = p.parse_args()

    summary = run_permutation_test_inference_only(
        canonical_dir=Path(args.canonical_dir),
        base_metadata_csv=Path(args.base_metadata_csv),
        output_base=Path(args.output_base),
        num_perms=args.num_perms,
        start_perm=args.start_perm,
        n_folds=args.n_folds,
        target_col=args.target_col,
        id_col=args.id_col,
        pred_filename=args.pred_filename,
    )
    print(
        f"\ninference-only permutation test: N={summary['n_permutations']}, "
        f"canonical R²={summary['canonical_mean_r2']:+.4f}, "
        f"null mean={summary['null_mean']:+.4f} ± {summary['null_std']:.4f}, "
        f"z={summary['z_under_null']:.2f}, "
        f"p_one_sided={summary['p_value_one_sided']:.4f}"
    )
    print(f"summary → {Path(args.output_base) / 'permutation_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
