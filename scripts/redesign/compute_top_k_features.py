"""Compute top-K feature indices per CV fold using XGBoost importance on the
flat pseudobulk (148,335 features = 31 cell types * 4785 genes).
Output used as input selector for TabPFN-2.6 residual-base predictions."""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb

# Ensure the worktree root is on sys.path so `src.data.tabpfn_input` resolves
# from this worktree (uv run adds the main-repo path, which may lack newer
# modules introduced in this branch).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.feature_loaders import load_flat_features, load_targets  # noqa: E402
from src.data.splits import load_splits  # noqa: E402

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
    logger.info("Union of all split subjects: %d", len(all_ids))
    features = load_flat_features(precomputed_dir, all_ids)
    targets = load_targets(meta_csv, all_ids)

    for fold_idx, fold_split in enumerate(splits["folds"]):
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
            "fold %d: train subjects usable=%d/%d "
            "(dropped: no_features=%d, no_target=%d)",
            fold_idx, len(train_ids), n_train_raw,
            dropped_no_feat, dropped_no_tgt,
        )
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
        logger.info(
            "fold %d: wrote %s (top %d of %d)",
            fold_idx, out_path, args.top_k, X_train.shape[1],
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--splits-path", default="outputs/splits.json")
    p.add_argument("--precomputed-dir", default="data/precomputed")
    p.add_argument("--metadata-csv", default="data/metadata_ROSMAP/metadata.csv")
    p.add_argument("--output-dir", default="data/redesign")
    p.add_argument("--top-k", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
