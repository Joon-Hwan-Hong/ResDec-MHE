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

Model version: TabPFN-2.6 (pinned explicitly via ModelVersion.V2_6).
Weights cache: /host/milan/tank/Joon/__external_programs/tabpfn/ (set via
TABPFN_MODEL_CACHE_DIR env var).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

from src.data.feature_loaders import load_flat_features, load_targets
from src.data.splits import load_splits

logger = logging.getLogger(__name__)


def _predict_with_sigma(
    reg: TabPFNRegressor, X: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (median, std) per row via a SINGLE predict() call.

    Uses output_type="full" with quantiles=[0.16, 0.84] to retrieve the
    median and both quantiles in one forward pass; std = (q84 - q16) / 2.

    Falls back to a two-call path only if the dict schema differs from the
    tabpfn 7.1.1 contract (mean/median/mode/quantiles/criterion/logits).
    """
    result = reg.predict(X, output_type="full", quantiles=[0.16, 0.84])
    if isinstance(result, dict) and "median" in result and "quantiles" in result:
        median = np.asarray(result["median"])
        q = result["quantiles"]  # list[np.ndarray], length 2 -> [q16, q84]
        lower = np.asarray(q[0])
        upper = np.asarray(q[1])
        sigma = np.clip((upper - lower) / 2.0, a_min=1e-3, a_max=None)
        return median, sigma

    # Fallback: legacy two-call path (should not trigger on tabpfn 7.1.1)
    logger.warning(
        "TabPFN predict(output_type='full') returned unexpected schema; "
        "falling back to two-call path."
    )
    median = np.asarray(reg.predict(X, output_type="median"))
    q_arr = reg.predict(X, output_type="quantiles", quantiles=[0.16, 0.84])
    lower = np.asarray(q_arr[0])
    upper = np.asarray(q_arr[1])
    sigma = np.clip((upper - lower) / 2.0, a_min=1e-3, a_max=None)
    return median, sigma


def main(args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

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

    all_ids = sorted(
        {sid for fold in splits["folds"] for sid in fold["train"] + fold["val"]}
    )
    logger.info("Loading features for %d subjects...", len(all_ids))
    features = load_flat_features(precomputed_dir, all_ids)
    targets = load_targets(meta_csv, all_ids)
    logger.info(
        "Features ready: %d with pseudobulk; %d with cogn_global.",
        len(features), len(targets),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s  (TabPFN model_version=ModelVersion.V2_6)", device)

    for fold_idx, fold_split in enumerate(splits["folds"]):
        logger.info("=== Fold %d ===", fold_idx)
        n_train_raw = len(fold_split["train"])
        train_ids = [
            s for s in fold_split["train"] if s in features and s in targets
        ]
        dropped_no_feat = sum(
            1 for s in fold_split["train"] if s not in features
        )
        dropped_no_tgt = sum(
            1 for s in fold_split["train"]
            if s in features and s not in targets
        )
        logger.info(
            "fold %d: train usable=%d/%d (dropped no_features=%d, no_target=%d)",
            fold_idx, len(train_ids), n_train_raw,
            dropped_no_feat, dropped_no_tgt,
        )
        top_k = json.loads(
            (top_k_dir / f"top_{args.top_k}_features_fold{fold_idx}.json").read_text()
        )["indices"]

        X_train_full = np.stack(
            [features[s] for s in train_ids]
        )[:, top_k].astype(np.float32)
        y_train = np.array([targets[s] for s in train_ids], dtype=np.float32)
        logger.info(
            "  X shape: %s, y shape: %s", X_train_full.shape, y_train.shape
        )

        oof_mean = np.zeros_like(y_train, dtype=np.float32)
        oof_std = np.ones_like(y_train, dtype=np.float32)

        inner_kf = KFold(
            n_splits=args.n_inner_folds, shuffle=True, random_state=args.seed
        )
        for inner_fold, (tr_idx, va_idx) in enumerate(inner_kf.split(X_train_full)):
            reg = TabPFNRegressor(
                device=device,
                random_state=args.seed,
                model_version=ModelVersion.V2_6,
            )
            reg.fit(X_train_full[tr_idx], y_train[tr_idx])
            try:
                mean, sigma = _predict_with_sigma(reg, X_train_full[va_idx])
            except Exception as e:
                logger.warning(
                    "  inner %d: quantile path failed (%s); fallback to mean-only",
                    inner_fold, e,
                )
                mean = reg.predict(X_train_full[va_idx])
                sigma = np.ones_like(mean, dtype=np.float32)
            oof_mean[va_idx] = mean.astype(np.float32)
            oof_std[va_idx] = sigma.astype(np.float32)
            logger.info("  inner %d: %d val preds done", inner_fold, len(va_idx))

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
        logger.info(
            "fold %d: wrote %s  OOF R² = %.4f  mean σ = %.4f",
            fold_idx, out_path, r2, oof_std.mean(),
        )


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
