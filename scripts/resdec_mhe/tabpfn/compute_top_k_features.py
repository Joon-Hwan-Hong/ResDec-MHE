"""Compute top-K feature indices per CV fold using XGBoost importance.

Default feature set ``A`` is the flat pseudobulk (148,335 features = 31 cell
types * 4785 genes). Other feature sets add CCC + composition + pathology +
region_mask for the enriched-TabPFN scoping experiment; see
``src/data/enriched_features.py`` for the definitions.

Output used as input selector for TabPFN-2.6 residual-base predictions."""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

# Ensure the worktree root is on sys.path so the modules imported below
# (`src.data.enriched_features`, `src.data.feature_loaders`, `src.data.splits`)
# resolve from THIS worktree. `uv run` adds the main-repo path, which may
# lack newer modules introduced in this branch.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data.enriched_features import (
    FEATURE_SETS,
    FEATURE_SET_SIZES,
    load_enriched_features,
    load_pathology,
)
from src.data.feature_loaders import load_flat_features, load_targets
from src.data.splits import load_splits

# Shared helpers for TabPFN compute scripts (CC4: previously duplicated across
# compute_top_k_features.py / compute_oof.py / compute_outer.py).
from scripts.resdec_mhe.tabpfn._helpers import (
    DEFAULT_XGB_LEARNING_RATE,
    DEFAULT_XGB_MAX_DEPTH,
    DEFAULT_XGB_N_ESTIMATORS,
    build_xgb_regressor,
    filter_usable_subjects,
    top_k_filename,
)

logger = logging.getLogger(__name__)


def main(args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    splits = load_splits(args.splits_path)
    precomputed_dir = Path(args.precomputed_dir)
    meta_csv = Path(args.metadata_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_ids = sorted(
        {sid for fold in splits["folds"] for sid in fold["train"] + fold["val"]}
    )
    logger.info(
        "Union of all split subjects: %d | feature_set=%s (dim=%d)",
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

    for fold_idx, fold_split in enumerate(splits["folds"]):
        n_train_raw = len(fold_split["train"])
        train_ids, dropped_no_feat, dropped_no_tgt = filter_usable_subjects(
            fold_split["train"], features, targets,
        )
        logger.info(
            "fold %d: train subjects usable=%d/%d "
            "(dropped: no_features=%d, no_target=%d)",
            fold_idx, len(train_ids), n_train_raw,
            dropped_no_feat, dropped_no_tgt,
        )
        X_train = np.stack([features[s] for s in train_ids])
        y_train = np.array([targets[s] for s in train_ids], dtype=np.float32)

        reg = build_xgb_regressor(
            n_estimators=args.xgb_n_estimators,
            max_depth=args.xgb_max_depth,
            learning_rate=args.xgb_learning_rate,
            n_jobs=-1,
            seed=args.seed,
        )
        reg.fit(X_train, y_train)
        imp = reg.feature_importances_
        top_k_idx = np.argsort(imp)[::-1][: args.top_k].tolist()

        out_path = top_k_filename(output_dir, args.top_k, fold_idx, args.feature_set)
        out_path.write_text(json.dumps({
            "fold": fold_idx,
            "top_k": args.top_k,
            "feature_set": args.feature_set,
            "n_features_total": int(X_train.shape[1]),
            "indices": top_k_idx,
            "seed": args.seed,
        }))
        logger.info(
            "fold %d: wrote %s (top %d of %d, feature_set=%s)",
            fold_idx, out_path, args.top_k, X_train.shape[1], args.feature_set,
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
    p.add_argument("--output-dir", default=str(REPO_ROOT / "data/canonical"))
    p.add_argument("--top-k", type=int, default=2000)
    p.add_argument(
        "--feature-set",
        default="A",
        choices=list(FEATURE_SETS),
        help=(
            "Feature set to build per subject before XGBoost importance ranking. "
            "A=pseudobulk only (default, backwards-compatible). "
            "A+C+E+P+R adds CCC dense + CCC aggregate + composition + pathology + region_mask."
        ),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--xgb-n-estimators", type=int, default=DEFAULT_XGB_N_ESTIMATORS,
        help=f"XGBoost n_estimators (default: {DEFAULT_XGB_N_ESTIMATORS}).",
    )
    p.add_argument(
        "--xgb-max-depth", type=int, default=DEFAULT_XGB_MAX_DEPTH,
        help=f"XGBoost max_depth (default: {DEFAULT_XGB_MAX_DEPTH}).",
    )
    p.add_argument(
        "--xgb-learning-rate", type=float, default=DEFAULT_XGB_LEARNING_RATE,
        help=f"XGBoost learning_rate (default: {DEFAULT_XGB_LEARNING_RATE}).",
    )
    main(p.parse_args())
