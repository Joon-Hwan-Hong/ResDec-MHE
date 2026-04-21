"""Pre-compute TabPFN-2.6 out-of-fold predictions on training subjects per CV fold.
Uses 5-fold-within-train OOF.

Outputs .npz files per CV fold with:
  subject_ids:      [N] str array of training subject IDs
  y_true:           [N] cognition target
  y_tabpfn_oof:     [N] TabPFN median prediction for each subject (OOF)
  sigma_tabpfn_oof: [N] TabPFN per-prediction std (derived from 0.16/0.84 quantiles)

These residuals (y_true - y_tabpfn_oof) become the stage-1 regression target in
the ResDec-H3 head during training. sigma_tabpfn_oof is used by aug-U to weight
the stage-k auxiliary loss by 1/σ².

Model version: TabPFN-2.6 (tabpfn==7.1.1 default ModelVersion.V2_6).
Weights cache: /host/milan/tank/Joon/__external_programs/tabpfn/ (set via
TABPFN_MODEL_CACHE_DIR env var).
"""
from __future__ import annotations
import json
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
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
    """Load cogn_global per subject via ROSMAP_IndividualID (NOT projid)."""
    df = pd.read_csv(meta_csv)
    wanted = set(subject_ids)
    return {
        r["ROSMAP_IndividualID"]: float(r["cogn_global"])
        for _, r in df.iterrows()
        if r["ROSMAP_IndividualID"] in wanted and not pd.isna(r["cogn_global"])
    }


def _predict_with_sigma(reg: TabPFNRegressor, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (median, std) per row. std approximated via (q84 - q16) / 2."""
    median = reg.predict(X, output_type="median")
    q = reg.predict(X, output_type="quantiles", quantiles=[0.16, 0.84])
    # q is a list of two arrays [lower, upper], each of length N
    lower = np.asarray(q[0])
    upper = np.asarray(q[1])
    sigma = (upper - lower) / 2.0
    # Clamp away from zero to avoid div-by-zero downstream
    sigma = np.clip(sigma, a_min=1e-3, a_max=None)
    return np.asarray(median), sigma


def main(args):
    # TabPFN cache path must be set before any TabPFN calls
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
    print(f"Loading features for {len(all_ids)} subjects...")
    features = _load_all_flat_features(precomputed_dir, all_ids)
    targets = _load_targets(meta_csv, all_ids)
    print(f"Features ready: {len(features)} subjects with pseudobulk; {len(targets)} with cogn_global.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    for fold_idx, fold_split in enumerate(splits["folds"]):
        print(f"\n=== Fold {fold_idx} ===")
        train_ids = [s for s in fold_split["train"] if s in features and s in targets]
        top_k = json.loads(
            (top_k_dir / f"top_{args.top_k}_features_fold{fold_idx}.json").read_text()
        )["indices"]

        X_train_full = np.stack([features[s] for s in train_ids])[:, top_k].astype(np.float32)
        y_train = np.array([targets[s] for s in train_ids], dtype=np.float32)
        print(f"  X shape: {X_train_full.shape}, y shape: {y_train.shape}")

        oof_mean = np.zeros_like(y_train, dtype=np.float32)
        oof_std = np.ones_like(y_train, dtype=np.float32)

        inner_kf = KFold(n_splits=args.n_inner_folds, shuffle=True, random_state=args.seed)
        for inner_fold, (tr_idx, va_idx) in enumerate(inner_kf.split(X_train_full)):
            reg = TabPFNRegressor(device=device, random_state=args.seed)
            reg.fit(X_train_full[tr_idx], y_train[tr_idx])
            try:
                mean, sigma = _predict_with_sigma(reg, X_train_full[va_idx])
            except Exception as e:
                print(f"  inner {inner_fold}: quantile path failed ({e}); fallback to mean-only")
                mean = reg.predict(X_train_full[va_idx])
                sigma = np.ones_like(mean, dtype=np.float32)
            oof_mean[va_idx] = mean.astype(np.float32)
            oof_std[va_idx] = sigma.astype(np.float32)
            print(f"  inner {inner_fold}: {len(va_idx)} val preds done")

        out_path = output_dir / f"tabpfn_oof_fold{fold_idx}.npz"
        np.savez(
            out_path,
            subject_ids=np.array(train_ids, dtype=object),
            y_true=y_train,
            y_tabpfn_oof=oof_mean,
            sigma_tabpfn_oof=oof_std,
        )

        from sklearn.metrics import r2_score
        r2 = r2_score(y_train, oof_mean)
        print(f"fold {fold_idx}: wrote {out_path}  OOF R² = {r2:.4f}  "
              f"mean σ = {oof_std.mean():.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--splits-path", default="outputs/splits.json")
    p.add_argument("--precomputed-dir", default="data/precomputed")
    p.add_argument("--metadata-csv", default="data/metadata_ROSMAP/metadata.csv")
    p.add_argument("--top-k-dir", default="data/redesign")
    p.add_argument("--output-dir", default="data/redesign")
    p.add_argument("--top-k", type=int, default=2000)
    p.add_argument("--n-inner-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
