"""Pre-compute TabPFN-2.6 OUTER-fold predictions (standalone baseline + cached
residual-base at val time).

For each outer CV fold, fits TabPFN on the training subjects' top-k features and
predicts on the validation subjects. Writes per-fold .npz.

Complements compute_tabpfn_oof.py (inner-OOF predictions used for stage-1
training residual targets). The OUTER-fold predictions here are:
  - The apples-to-apples TabPFN-2.6 standalone baseline R² for the paper table
  - The cached ``y_tabpfn_val`` used by the ResDec Lightning module at
    validation time (so we don't re-fit TabPFN per validation epoch)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tabpfn import TabPFNRegressor

from src.analysis.tabpfn_preprocessing import apply_zscore_train_only
from src.data.enriched_features import (
    FEATURE_SETS,
    FEATURE_SET_SIZES,
    load_enriched_features,
    load_pathology,
)
from src.data.feature_loaders import load_flat_features, load_targets
from src.data.splits import load_splits

logger = logging.getLogger(__name__)


def _top_k_filename(top_k_dir: Path, top_k: int, fold_idx: int, feature_set: str) -> Path:
    if feature_set == "A":
        return top_k_dir / f"top_{top_k}_features_fold{fold_idx}.json"
    return top_k_dir / f"top_{top_k}_features_fold{fold_idx}_{feature_set}.json"


def _outer_output_filename(output_dir: Path, fold_idx: int, feature_set: str) -> Path:
    if feature_set == "A":
        return output_dir / f"tabpfn_outer_fold{fold_idx}.npz"
    return output_dir / f"tabpfn_outer_fold{fold_idx}_{feature_set}.npz"


def _predict_with_sigma(reg: TabPFNRegressor, X: np.ndarray):
    """Return (median, std) via a SINGLE predict() call (output_type='full').

    Falls back to legacy two-call path if dict schema differs.
    """
    result = reg.predict(X, output_type="full", quantiles=[0.16, 0.84])
    if isinstance(result, dict) and "median" in result and "quantiles" in result:
        median = np.asarray(result["median"])
        q = result["quantiles"]  # list[np.ndarray], length 2 -> [q16, q84]
        lower = np.asarray(q[0])
        upper = np.asarray(q[1])
        sigma = np.clip((upper - lower) / 2.0, a_min=1e-3, a_max=None)
        return median, sigma

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


def _build_regressor(
    device: str, seed: int, ignore_pretraining_limits: bool
) -> TabPFNRegressor:
    """Construct a TabPFNRegressor with the ablation safety-override flag.

    ``ignore_pretraining_limits=True`` is a DELIBERATE override of TabPFN-2.6's
    2000-feature safety check. Use ONLY when deliberately testing
    >2000-feature behavior (e.g., top-k > 2000 ablations). Accepts the
    distributional-extrapolation risk; TabPFN's prior was trained on ≤2000
    features. Default MUST be False everywhere upstream.

    model_version is NOT a TabPFNRegressor constructor kwarg — set via
    tabpfn.settings (default is ModelVersion.V2_6 per tabpfn/settings.py:36).
    """
    return TabPFNRegressor(
        device=device,
        random_state=seed,
        ignore_pretraining_limits=ignore_pretraining_limits,
    )


def main(args):
    # See _build_regressor() for --ignore-pretraining-limits override semantics.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s  (TabPFN model_version=ModelVersion.V2_6)", device)
    if args.ignore_pretraining_limits:
        logger.warning(
            "ignore_pretraining_limits=True: TabPFN-2.6's 2000-feature safety "
            "check is DISABLED. User-approved override for ablation studies of "
            ">2000-feature behavior. TabPFN's prior was trained on ≤2000 features; "
            "predictions beyond that regime are distributional extrapolation."
        )

    results = []
    for fold_idx, fold_split in enumerate(splits["folds"]):
        logger.info("=== Fold %d ===", fold_idx)
        n_train_raw = len(fold_split["train"])
        n_val_raw = len(fold_split["val"])
        train_ids = [
            s for s in fold_split["train"] if s in features and s in targets
        ]
        val_ids = [
            s for s in fold_split["val"] if s in features and s in targets
        ]
        dropped_train_no_feat = sum(
            1 for s in fold_split["train"] if s not in features
        )
        dropped_train_no_tgt = sum(
            1 for s in fold_split["train"]
            if s in features and s not in targets
        )
        dropped_val_no_feat = sum(
            1 for s in fold_split["val"] if s not in features
        )
        dropped_val_no_tgt = sum(
            1 for s in fold_split["val"]
            if s in features and s not in targets
        )
        logger.info(
            "fold %d: train usable=%d/%d (dropped no_features=%d, no_target=%d)"
            " | val usable=%d/%d (dropped no_features=%d, no_target=%d)",
            fold_idx,
            len(train_ids), n_train_raw, dropped_train_no_feat, dropped_train_no_tgt,
            len(val_ids), n_val_raw, dropped_val_no_feat, dropped_val_no_tgt,
        )
        top_k_path = _top_k_filename(top_k_dir, args.top_k, fold_idx, args.feature_set)
        top_k = json.loads(top_k_path.read_text())["indices"]

        X_train = np.stack(
            [features[s] for s in train_ids]
        )[:, top_k].astype(np.float32)
        y_train = np.array([targets[s] for s in train_ids], dtype=np.float32)
        X_val = np.stack(
            [features[s] for s in val_ids]
        )[:, top_k].astype(np.float32)
        y_val = np.array([targets[s] for s in val_ids], dtype=np.float32)
        logger.info("  train %s, val %s", X_train.shape, X_val.shape)

        if args.zscore:
            # Per-feature z-score fit on OUTER-fold train ONLY; transform
            # both outer-train and outer-val with those train stats. No
            # pooled stats.
            X_train, X_val = apply_zscore_train_only(X_train, X_val)

        reg = _build_regressor(
            device=device,
            seed=args.seed,
            ignore_pretraining_limits=args.ignore_pretraining_limits,
        )
        reg.fit(X_train, y_train)
        mean, sigma = _predict_with_sigma(reg, X_val)

        out_path = _outer_output_filename(output_dir, fold_idx, args.feature_set)
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
        results.append({
            "fold": fold_idx, "r2": r2, "n_val": len(val_ids),
            "n_train": len(train_ids), "mean_sigma": float(sigma.mean()),
        })
        logger.info(
            "fold %d: R²=%+.4f, mean σ=%.4f  (wrote %s)",
            fold_idx, r2, sigma.mean(), out_path,
        )

    # Summary
    r2s = [r["r2"] for r in results]
    logger.info("==============================")
    logger.info("TabPFN-2.6 OUTER-fold standalone baseline:")
    logger.info(
        "  mean R² = %+.4f ± %.4f", np.mean(r2s), np.std(r2s)
    )
    logger.info("  per-fold R² = %s", [f"{r:+.4f}" for r in r2s])
    logger.info("==============================")

    # Write CSV for paper table
    outputs_pipeline = Path("outputs/pipeline")
    outputs_pipeline.mkdir(parents=True, exist_ok=True)
    if args.feature_set == "A":
        csv_path = outputs_pipeline / "baseline_results_tabpfn.csv"
    else:
        csv_path = outputs_pipeline / f"baseline_results_tabpfn_{args.feature_set}.csv"
    pd.DataFrame(results).to_csv(csv_path, index=False)
    logger.info("Wrote baseline CSV: %s", csv_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--splits-path", default="outputs/splits.json")
    p.add_argument("--precomputed-dir", default="data/precomputed")
    p.add_argument("--metadata-csv", default="data/metadata_ROSMAP/metadata.csv")
    p.add_argument("--top-k-dir", default="data/canonical")
    p.add_argument("--output-dir", default="data/canonical")
    p.add_argument("--top-k", type=int, default=2000)
    p.add_argument(
        "--feature-set",
        default="A",
        choices=list(FEATURE_SETS),
        help=(
            "Feature set consumed by TabPFN. Must match the --feature-set "
            "that compute_top_k_features.py used to produce the top-K JSON."
        ),
    )
    p.add_argument("--seed", type=int, default=42)
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
            "outer-fold TRAIN-ONLY subjects. Critical: no pooled stats. "
            "Default: False."
        ),
    )
    main(p.parse_args())
