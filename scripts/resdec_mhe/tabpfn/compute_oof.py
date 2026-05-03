"""Pre-compute TabPFN-2.6 out-of-fold predictions on training subjects per CV fold.
Uses 5-fold-within-train OOF.

Outputs .npz files per CV fold with:
  subject_ids:      [N] str array of training subject IDs
  y_true:           [N] cognition target
  y_tabpfn_oof:     [N] TabPFN median prediction for each subject (OOF)
  sigma_tabpfn_oof: [N] TabPFN per-prediction std (derived from 0.16/0.84 quantiles)

These residuals (y_true - y_tabpfn_oof) become the stage-1 regression target in
the ResDec-MHE head during training. sigma_tabpfn_oof is used by aug-U to weight
the stage-k auxiliary loss by 1/σ².

Model version: TabPFN-2.6 (pinned explicitly via ModelVersion.V2_6).
Weights cache: set via TABPFN_MODEL_CACHE_DIR env var (resolved by
``scripts.resdec_mhe.tabpfn._helpers.resolve_tabpfn_cache_dir``).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from src.analysis.tabpfn_preprocessing import apply_zscore_train_only
from src.data.enriched_features import (
    FEATURE_SETS,
    FEATURE_SET_SIZES,
    load_enriched_features,
    load_pathology,
)
from src.data.feature_loaders import load_flat_features, load_targets
from src.data.splits import load_splits

# Shared TabPFN helpers (CC4: ``predict_with_sigma`` / ``build_regressor`` /
# ``filter_usable_subjects`` / ``top_k_filename`` / ``resolve_tabpfn_cache_dir``
# previously duplicated across compute_top_k_features.py / compute_oof.py /
# compute_outer.py).
from scripts.resdec_mhe.tabpfn._helpers import (
    build_regressor,
    filter_usable_subjects,
    predict_with_sigma,
    resolve_tabpfn_cache_dir,
    top_k_filename,
)

logger = logging.getLogger(__name__)


def process_oof_fold(
    fold_idx: int,
    fold_split: dict,
    features: dict,
    targets: dict[str, float],
    args,
    device: str,
    output_dir: Path,
    top_k_dir: Path,
) -> dict:
    """Run inner-OOF TabPFN on one fold.

    Reusable per-fold callable extracted from main() so variant pipelines
    (residualized targets) can inject a custom ``targets`` map per fold.
    Writes ``tabpfn_oof_fold{fold_idx}.npz`` and returns a summary dict.
    """
    logger.info("=== Fold %d ===", fold_idx)
    n_train_raw = len(fold_split["train"])
    train_ids, dropped_no_feat, dropped_no_tgt = filter_usable_subjects(
        fold_split["train"], features, targets,
    )
    logger.info(
        "fold %d: train usable=%d/%d (dropped no_features=%d, no_target=%d)",
        fold_idx, len(train_ids), n_train_raw,
        dropped_no_feat, dropped_no_tgt,
    )
    top_k_path = top_k_filename(
        top_k_dir, args.top_k, fold_idx, args.feature_set,
    )
    top_k = json.loads(top_k_path.read_text())["indices"]

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
        X_tr = X_train_full[tr_idx]
        X_va = X_train_full[va_idx]
        if args.zscore:
            # Per-feature z-score fit on INNER-fold train ONLY; transform
            # both inner-train and inner-val with those train stats. No
            # pooled stats across inner splits.
            X_tr, X_va = apply_zscore_train_only(X_tr, X_va)
        reg = build_regressor(
            device=device,
            seed=args.seed,
            ignore_pretraining_limits=args.ignore_pretraining_limits,
        )
        reg.fit(X_tr, y_train[tr_idx])
        try:
            mean, sigma = predict_with_sigma(reg, X_va)
        except Exception as e:
            logger.warning(
                "  inner %d: quantile path failed (%s); fallback to mean-only",
                inner_fold, e,
            )
            mean = reg.predict(X_va)
            sigma = np.ones_like(mean, dtype=np.float32)
        oof_mean[va_idx] = mean.astype(np.float32)
        oof_std[va_idx] = sigma.astype(np.float32)
        logger.info("  inner %d: %d val preds done", inner_fold, len(va_idx))

    if args.feature_set == "A":
        out_path = output_dir / f"tabpfn_oof_fold{fold_idx}.npz"
    else:
        out_path = output_dir / f"tabpfn_oof_fold{fold_idx}_{args.feature_set}.npz"
    np.savez(
        out_path,
        subject_ids=np.array(train_ids, dtype=object),
        y_true=y_train,
        y_tabpfn_oof=oof_mean,
        sigma_tabpfn_oof=oof_std,
    )

    r2 = r2_score(y_train, oof_mean)
    logger.info(
        "fold %d: wrote %s  OOF R² = %.4f  mean σ = %.4f",
        fold_idx, out_path, r2, oof_std.mean(),
    )
    return {"fold": fold_idx, "out_path": str(out_path), "r2": float(r2),
            "n_train": len(train_ids), "mean_sigma": float(oof_std.mean())}


def main(args):
    # See ``build_regressor`` in _helpers.py for --ignore-pretraining-limits
    # override semantics.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # TabPFN cache path must be set before any TabPFN calls. Falls back to a
    # host-specific path with a warning if TABPFN_MODEL_CACHE_DIR is unset
    # (CC8: avoids silent reliance on an absolute path that won't exist on
    # other machines).
    resolve_tabpfn_cache_dir(
        default="/host/milan/tank/Joon/__external_programs/tabpfn",
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
    logger.info(
        "Loading features for %d subjects... feature_set=%s (dim=%d)",
        len(all_ids), args.feature_set, FEATURE_SET_SIZES[args.feature_set],
    )
    if args.feature_set == "A":
        features = load_flat_features(precomputed_dir, all_ids)
    else:
        pathology = None
        if args.feature_set == "A+C+E+P+R":
            pathology = load_pathology(meta_csv, all_ids)
        features = load_enriched_features(
            precomputed_dir, all_ids, args.feature_set, pathology=pathology,
        )
    targets = load_targets(meta_csv, all_ids)
    logger.info(
        "Features ready: %d with pseudobulk; %d with cogn_global.",
        len(features), len(targets),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s  (TabPFN model_version=ModelVersion.V2_6)", device)
    if args.ignore_pretraining_limits:
        logger.warning(
            "ignore_pretraining_limits=True: TabPFN-2.6's 2000-feature safety "
            "check is DISABLED. User-approved override for ablation studies of "
            ">2000-feature behavior. TabPFN's prior was trained on ≤2000 features; "
            "predictions beyond that regime are distributional extrapolation."
        )

    for fold_idx, fold_split in enumerate(splits["folds"]):
        process_oof_fold(
            fold_idx=fold_idx, fold_split=fold_split,
            features=features, targets=targets,
            args=args, device=device,
            output_dir=output_dir, top_k_dir=top_k_dir,
        )


if __name__ == "__main__":
    # Anchor defaults to the worktree root so callers in arbitrary cwds get
    # consistent paths (mirrors run_clinical_baseline.py:54-65).
    REPO_ROOT = Path(__file__).resolve().parents[3]
    p = argparse.ArgumentParser()
    p.add_argument("--splits-path", default=str(REPO_ROOT / "outputs/splits.json"))
    p.add_argument("--precomputed-dir", default=str(REPO_ROOT / "data/precomputed"))
    p.add_argument(
        "--metadata-csv",
        default=str(REPO_ROOT / "data/metadata_ROSMAP/metadata.csv"),
    )
    p.add_argument("--top-k-dir", default=str(REPO_ROOT / "data/canonical"))
    p.add_argument("--output-dir", default=str(REPO_ROOT / "data/canonical"))
    p.add_argument("--top-k", type=int, default=2000)
    p.add_argument("--n-inner-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--feature-set",
        default="A",
        choices=list(FEATURE_SETS),
        help=(
            "Feature set consumed by TabPFN. Must match the --feature-set "
            "that compute_top_k_features.py used to produce the top-K JSON. "
            "Default A=pseudobulk only (backwards-compatible with the "
            "canonical residual-base pipeline)."
        ),
    )
    p.add_argument(
        "--ignore-pretraining-limits",
        action="store_true",
        default=False,
        help=(
            "Override TabPFN-2.6's 2000-feature safety check. Use ONLY when "
            "deliberately testing >2000-feature behavior (e.g., top-k > 2000 "
            "ablations). Accepts the distributional-extrapolation risk; "
            "TabPFN's prior was trained on ≤2000 features. Default: False."
        ),
    )
    p.add_argument(
        "--zscore",
        action="store_true",
        default=False,
        help=(
            "Per-feature z-score the TabPFN input. Stats computed from "
            "TRAIN-ONLY subjects (inner-fold train for OOF; outer-fold train "
            "for outer). Critical: no pooled stats. Default: False."
        ),
    )
    main(p.parse_args())
