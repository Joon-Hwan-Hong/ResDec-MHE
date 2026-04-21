"""Compute top-K feature indices per CV fold using XGBoost importance on the
flat pseudobulk (148,335 features = 31 cell types * 4785 genes).
Output used as input selector for TabPFN-2.6 residual-base predictions."""
import json
import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import xgboost as xgb
import pandas as pd

# Ensure the worktree root is on sys.path so `src.data.tabpfn_input` resolves
# from this worktree (uv run adds the main-repo path, which may lack newer
# modules introduced in this branch).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.splits import load_splits  # noqa: E402
from src.data.tabpfn_input import flatten_pseudobulk  # noqa: E402


def _load_all_flat_features(precomputed_dir: Path, subject_ids: list[str]) -> dict:
    out = {}
    for sid in subject_ids:
        pt_path = precomputed_dir / f"{sid}.pt"
        if not pt_path.exists():
            continue
        pt = torch.load(pt_path, weights_only=False)
        out[sid] = flatten_pseudobulk(pt).numpy()
    return out


def _load_targets(meta_csv: Path, subject_ids: list[str]) -> dict:
    """Load cogn_global per subject via ROSMAP_IndividualID (NOT projid)."""
    df = pd.read_csv(meta_csv)
    wanted = set(subject_ids)
    return {
        r["ROSMAP_IndividualID"]: float(r["cogn_global"])
        for _, r in df.iterrows()
        if r["ROSMAP_IndividualID"] in wanted and not pd.isna(r["cogn_global"])
    }


def main(args):
    splits = load_splits(args.splits_path)
    precomputed_dir = Path(args.precomputed_dir)
    meta_csv = Path(args.metadata_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_ids = sorted({sid for fold in splits["folds"] for sid in fold["train"] + fold["val"]})
    features = _load_all_flat_features(precomputed_dir, all_ids)
    targets = _load_targets(meta_csv, all_ids)

    for fold_idx, fold_split in enumerate(splits["folds"]):
        train_ids = [s for s in fold_split["train"] if s in features and s in targets]
        X_train = np.stack([features[s] for s in train_ids])
        y_train = np.array([targets[s] for s in train_ids], dtype=np.float32)

        reg = xgb.XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            n_jobs=-1, tree_method="hist",
            random_state=args.seed,
        )
        reg.fit(X_train, y_train)
        imp = reg.feature_importances_
        top_k_idx = np.argsort(imp)[::-1][: args.top_k].tolist()

        out_path = output_dir / f"top_{args.top_k}_features_fold{fold_idx}.json"
        out_path.write_text(json.dumps({
            "fold": fold_idx,
            "top_k": args.top_k,
            "n_features_total": int(X_train.shape[1]),
            "indices": top_k_idx,
            "seed": args.seed,
        }))
        print(f"fold {fold_idx}: wrote {out_path} (top {args.top_k} of {X_train.shape[1]})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--splits-path", default="outputs/splits.json")
    p.add_argument("--precomputed-dir", default="data/precomputed")
    p.add_argument("--metadata-csv", default="data/metadata_ROSMAP/metadata.csv")
    p.add_argument("--output-dir", default="data/redesign")
    p.add_argument("--top-k", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
