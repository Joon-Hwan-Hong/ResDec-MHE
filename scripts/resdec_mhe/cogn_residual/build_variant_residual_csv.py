"""Pool per-fold residualized cognition targets into a single per-subject CSV.

For each subject, picks the residualized target value from the fold where the
subject is in the VALIDATION set (each subject appears in exactly one val fold
under disjoint K-fold CV). This gives one residual value per subject, suitable
as input to ``run_counterfactuals.py --residual-csv`` and other downstream
analyses that expect canonical's residual_per_subject.csv schema.

Output schema mirrors canonical's residual_per_subject.csv:
  ROSMAP_IndividualID, residual, fold

"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--residual-cache-dir", type=Path, required=True,
                   help="Variant residual cache (residual_target_fold{0..4}.npz).")
    p.add_argument("--splits-path", type=Path, required=True)
    p.add_argument("--out-csv", type=Path, required=True)
    args = p.parse_args()

    splits = json.loads(args.splits_path.read_text())
    fold_val_ids: dict[int, set[str]] = {}
    for fi, fold in enumerate(splits["folds"]):
        fold_val_ids[fi] = {str(s) for s in fold["val"]}

    rows: list[dict] = []
    for fi in range(len(splits["folds"])):
        npz_path = args.residual_cache_dir / f"residual_target_fold{fi}.npz"
        if not npz_path.is_file():
            raise FileNotFoundError(f"residual cache missing: {npz_path}")
        d = np.load(npz_path, allow_pickle=True)
        subj_ids = [str(s) for s in d["subject_ids"]]
        target = np.asarray(d["target"], dtype=float)
        for sid, t in zip(subj_ids, target):
            if sid in fold_val_ids[fi] and np.isfinite(t):
                rows.append({"ROSMAP_IndividualID": sid, "residual": float(t),
                             "fold": fi})

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset="ROSMAP_IndividualID", keep="first")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"wrote {args.out_csv} (n={len(df)} subjects across {df['fold'].nunique()} folds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
