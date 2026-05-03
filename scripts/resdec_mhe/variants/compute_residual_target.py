"""Compute per-fold residualized cognition target.

Per-fold OLS fit on training subjects only, applied to all 516 subjects.
Writes one .npz per fold + a summary.json with per-fold and aggregate
beta + alpha stats.

USAGE
-----
uv run python scripts/resdec_mhe/variants/compute_residual_target.py \\
    --variant-name gpath_only --axes gpath \\
    --out-dir outputs/canonical/variants/gpath_only/cache

uv run python scripts/resdec_mhe/variants/compute_residual_target.py \\
    --variant-name multi_axis --axes gpath tangsqrt amylsqrt \\
    --out-dir outputs/canonical/variants/multi_axis/cache
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow `uv run python scripts/...` invocation by ensuring repo root is on
# sys.path before importing src.data.* (mirrors sibling scripts/resdec_mhe/).
_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))
from src.data.residualization import apply_residual, fit_pathology_residual  # noqa: E402


def _fold_subjects(splits: dict, fold_idx: int) -> tuple[list[str], list[str]]:
    """Return (train_subjects, val_subjects) for the given fold."""
    folds = splits.get("folds", splits.get("splits", []))
    f = folds[fold_idx]
    if isinstance(f, dict):
        return list(f.get("train", [])), list(f.get("val", []))
    raise ValueError(f"unknown splits format: {type(f)}")


def _stdev_or_zero(vals: list[float]) -> float:
    return statistics.stdev(vals) if len(vals) >= 2 else 0.0


def main() -> int:
    p = argparse.ArgumentParser(
        description="Compute per-fold residualized cognition target."
    )
    p.add_argument("--variant-name", required=True,
                   help="Tag used in output filenames (e.g. gpath_only).")
    p.add_argument("--axes", nargs="+", required=True,
                   help="Pathology variable names to residualize against.")
    p.add_argument("--target", default="cogn_global")
    p.add_argument("--metadata-path", type=Path,
                   default=_ROOT / "data/metadata_ROSMAP")
    p.add_argument("--splits-path", type=Path,
                   default=_ROOT / "outputs/splits.json")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(args.metadata_path / "metadata.csv")
    splits = json.loads(args.splits_path.read_text())

    folds = splits.get("folds", splits.get("splits", []))
    cohort_list = splits.get("train_val_pool")
    if cohort_list is None:
        cohort: set[str] = set()
        for f in folds:
            if isinstance(f, dict):
                for k in ("train", "val", "test"):
                    if k in f:
                        cohort.update(f[k])
        cohort_list = sorted(cohort)
    cohort = set(cohort_list)

    df_cohort = (
        metadata[metadata["ROSMAP_IndividualID"].isin(cohort)]
        .reset_index(drop=True)
    )
    if df_cohort["ROSMAP_IndividualID"].duplicated().any():
        raise RuntimeError(
            "duplicate ROSMAP_IndividualID rows in metadata.csv; "
            "resolve upstream rather than dropping silently."
        )
    cohort_ids = df_cohort["ROSMAP_IndividualID"].tolist()
    if len(cohort_ids) != len(cohort):
        raise RuntimeError(
            f"metadata/cohort mismatch: {len(cohort_ids)} vs {len(cohort)}"
        )

    per_fold = []
    for fold in range(len(folds)):
        train_ids, val_ids = _fold_subjects(splits, fold)
        df_train = df_cohort[df_cohort["ROSMAP_IndividualID"].isin(train_ids)]
        fit = fit_pathology_residual(
            df_train, target=args.target, axes=args.axes,
        )
        target_all = apply_residual(df_cohort, target=args.target, fit=fit)

        npz_path = args.out_dir / f"residual_target_fold{fold}.npz"
        np.savez(
            npz_path,
            fold=fold,
            subject_ids=np.array(cohort_ids, dtype=object),
            target=target_all.astype(np.float32),
            alpha=fit["alpha"],
            **{f"beta_{a}": fit["beta"][a] for a in args.axes},
        )
        per_fold.append({
            "fold": fold,
            "alpha": fit["alpha"],
            "beta": fit["beta"],
            "axes": fit["axes"],
            "n_train": fit["n_train"],
            "n_val": len(val_ids),
        })

    aggregate: dict = {"axes": list(args.axes)}
    for axis in args.axes:
        vals = [pf["beta"][axis] for pf in per_fold]
        aggregate[f"beta_{axis}_mean"] = statistics.fmean(vals)
        aggregate[f"beta_{axis}_std"] = _stdev_or_zero(vals)
    alpha_vals = [pf["alpha"] for pf in per_fold]
    aggregate["alpha_mean"] = statistics.fmean(alpha_vals)
    aggregate["alpha_std"] = _stdev_or_zero(alpha_vals)

    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps({
        "variant_name": args.variant_name,
        "target": args.target,
        "axes": list(args.axes),
        "per_fold": per_fold,
        "aggregate": aggregate,
        "n_subjects_cohort": len(cohort_ids),
    }, indent=2))
    print(f"wrote {len(per_fold)} fold .npz files + summary.json to {args.out_dir}")
    print(json.dumps(aggregate, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
