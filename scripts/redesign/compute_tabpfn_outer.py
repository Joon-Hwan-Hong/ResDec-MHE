"""Pre-compute TabPFN-2.6 OUTER-fold predictions (standalone baseline + Phase-2
residual-base at val time).

For each of the 5 CV folds, fits TabPFN on all 412 training subjects' top-2K
features and predicts on the 104 validation subjects. Writes per-fold .npz.

Complements compute_tabpfn_oof.py (inner-OOF predictions used for stage-1
training residual targets). The OUTER-fold predictions here are:
  - The apples-to-apples TabPFN-2.6 standalone baseline R² for the paper table
    (vs XGBoost R²=0.358 on the same outer folds)
  - The cached `y_tabpfn_val` used by Phase-2 training at validation time
    (so we don't re-fit TabPFN per validation epoch)
"""
from __future__ import annotations
import json
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tabpfn import TabPFNRegressor

from src.data.splits import load_splits
from src.data.tabpfn_input import flatten_pseudobulk


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
    df = pd.read_csv(meta_csv)
    wanted = set(subject_ids)
    return {
        r["ROSMAP_IndividualID"]: float(r["cogn_global"])
        for _, r in df.iterrows()
        if r["ROSMAP_IndividualID"] in wanted and not pd.isna(r["cogn_global"])
    }


def _predict_with_sigma(reg: TabPFNRegressor, X: np.ndarray):
    median = reg.predict(X, output_type="median")
    q = reg.predict(X, output_type="quantiles", quantiles=[0.16, 0.84])
    lower = np.asarray(q[0])
    upper = np.asarray(q[1])
    sigma = np.clip((upper - lower) / 2.0, a_min=1e-3, a_max=None)
    return np.asarray(median), sigma


def main(args):
    os.environ.setdefault(
        "TABPFN_MODEL_CACHE_DIR",
        "/host/milan/tank/Joon/__external_programs/tabpfn",
    )
    splits = load_splits(args.splits_path)
    precomputed_dir = Path(args.precomputed_dir)
    meta_csv = Path(args.metadata_csv)
    top_k_dir = Path(args.top_k_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_ids = sorted({sid for fold in splits["folds"] for sid in fold["train"] + fold["val"]})
    features = _load_all_flat_features(precomputed_dir, all_ids)
    targets = _load_targets(meta_csv, all_ids)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = []
    for fold_idx, fold_split in enumerate(splits["folds"]):
        print(f"\n=== Fold {fold_idx} ===")
        train_ids = [s for s in fold_split["train"] if s in features and s in targets]
        val_ids = [s for s in fold_split["val"] if s in features and s in targets]
        top_k = json.loads(
            (top_k_dir / f"top_{args.top_k}_features_fold{fold_idx}.json").read_text()
        )["indices"]

        X_train = np.stack([features[s] for s in train_ids])[:, top_k].astype(np.float32)
        y_train = np.array([targets[s] for s in train_ids], dtype=np.float32)
        X_val = np.stack([features[s] for s in val_ids])[:, top_k].astype(np.float32)
        y_val = np.array([targets[s] for s in val_ids], dtype=np.float32)
        print(f"  train {X_train.shape}, val {X_val.shape}")

        reg = TabPFNRegressor(device=device, random_state=args.seed)
        reg.fit(X_train, y_train)
        mean, sigma = _predict_with_sigma(reg, X_val)

        out_path = output_dir / f"tabpfn_outer_fold{fold_idx}.npz"
        np.savez(
            out_path,
            val_subject_ids=np.array(val_ids, dtype=object),
            y_true=y_val,
            y_tabpfn=mean.astype(np.float32),
            sigma_tabpfn=sigma.astype(np.float32),
            train_n=len(train_ids),
        )

        from sklearn.metrics import r2_score
        r2 = r2_score(y_val, mean)
        results.append({"fold": fold_idx, "r2": r2, "n_val": len(val_ids),
                        "n_train": len(train_ids), "mean_sigma": float(sigma.mean())})
        print(f"fold {fold_idx}: R²={r2:+.4f}, mean σ={sigma.mean():.4f}  (wrote {out_path})")

    # Summary
    r2s = [r["r2"] for r in results]
    print(f"\n==============================")
    print(f"TabPFN-2.6 OUTER-fold standalone baseline:")
    print(f"  mean R² = {np.mean(r2s):+.4f} ± {np.std(r2s):.4f}")
    print(f"  per-fold R² = {[f'{r:+.4f}' for r in r2s]}")
    print(f"==============================")

    # Write CSV for paper table
    outputs_pipeline = Path("outputs/pipeline")
    outputs_pipeline.mkdir(parents=True, exist_ok=True)
    csv_path = outputs_pipeline / "baseline_results_tabpfn.csv"
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"Wrote baseline CSV: {csv_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--splits-path", default="outputs/splits.json")
    p.add_argument("--precomputed-dir", default="data/precomputed")
    p.add_argument("--metadata-csv", default="data/metadata_ROSMAP/metadata.csv")
    p.add_argument("--top-k-dir", default="data/redesign")
    p.add_argument("--output-dir", default="data/redesign")
    p.add_argument("--top-k", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
