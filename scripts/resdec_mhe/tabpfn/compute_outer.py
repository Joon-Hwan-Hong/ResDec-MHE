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
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score

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
# ``filter_usable_subjects`` / ``top_k_filename`` / ``outer_output_filename`` /
# ``resolve_tabpfn_cache_dir`` previously duplicated across
# compute_top_k_features.py / compute_oof.py / compute_outer.py).
from scripts.resdec_mhe.tabpfn._helpers import (
    build_regressor,
    filter_usable_subjects,
    outer_output_filename,
    predict_with_sigma,
    resolve_tabpfn_cache_dir,
    top_k_filename,
)

logger = logging.getLogger(__name__)


def main(args):
    # See ``build_regressor`` in _helpers.py for --ignore-pretraining-limits
    # override semantics.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # CC8: resolve TABPFN_MODEL_CACHE_DIR via the shared helper. Falls back to
    # a host-specific path with a logged warning rather than silent setdefault.
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
        train_ids, dropped_train_no_feat, dropped_train_no_tgt = filter_usable_subjects(
            fold_split["train"], features, targets,
        )
        val_ids, dropped_val_no_feat, dropped_val_no_tgt = filter_usable_subjects(
            fold_split["val"], features, targets,
        )
        logger.info(
            "fold %d: train usable=%d/%d (dropped no_features=%d, no_target=%d)"
            " | val usable=%d/%d (dropped no_features=%d, no_target=%d)",
            fold_idx,
            len(train_ids), n_train_raw, dropped_train_no_feat, dropped_train_no_tgt,
            len(val_ids), n_val_raw, dropped_val_no_feat, dropped_val_no_tgt,
        )
        top_k_path = top_k_filename(top_k_dir, args.top_k, fold_idx, args.feature_set)
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

        reg = build_regressor(
            device=device,
            seed=args.seed,
            ignore_pretraining_limits=args.ignore_pretraining_limits,
        )
        reg.fit(X_train, y_train)
        mean, sigma = predict_with_sigma(reg, X_val)

        out_path = outer_output_filename(output_dir, fold_idx, args.feature_set)
        np.savez(
            out_path,
            val_subject_ids=np.array(val_ids, dtype=object),
            y_true=y_val,
            y_tabpfn=mean.astype(np.float32),
            sigma_tabpfn=sigma.astype(np.float32),
            train_n=len(train_ids),
        )

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

    # Write CSV for paper table. CC7: writable dir is configurable via
    # --csv-output-dir (defaults to outputs/pipeline alongside output-dir).
    csv_dir = Path(args.csv_output_dir)
    csv_dir.mkdir(parents=True, exist_ok=True)
    if args.feature_set == "A":
        csv_path = csv_dir / "baseline_results_tabpfn.csv"
    else:
        csv_path = csv_dir / f"baseline_results_tabpfn_{args.feature_set}.csv"
    pd.DataFrame(results).to_csv(csv_path, index=False)
    logger.info("Wrote baseline CSV: %s", csv_path)


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
    p.add_argument(
        "--csv-output-dir",
        default=str(REPO_ROOT / "outputs/pipeline"),
        help=(
            "Directory for per-feature-set baseline CSV (paper table). "
            "Separate from --output-dir which holds per-fold .npz."
        ),
    )
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
