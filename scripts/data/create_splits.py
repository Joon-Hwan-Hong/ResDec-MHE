#!/usr/bin/env python3
"""Create and save stratified subject-level splits.

Generates a splits JSON file used by all downstream scripts (training/train.py,
training/hpo.py, data/precompute_features.py, profiling/profile_training.py).

Stratification: joint pathology × cognition tertiles (3×3 = 9 strata).
Each subject appears in exactly one validation fold (disjoint).

Usage:
    .venv/bin/python scripts/data/create_splits.py \
        --config configs/default.yaml \
        --output outputs/splits.json

    # Custom split parameters:
    .venv/bin/python scripts/data/create_splits.py \
        --config configs/default.yaml \
        --output outputs/splits.json \
        --test-frac 0.15 --n-folds 3 --seed 123
"""

import argparse
import logging

from src.data.splits import create_stratified_splits, save_splits, validate_no_leakage
from src.utils.config import load_config, validate_config

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create stratified subject-level splits"
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to config YAML file (reads metadata path, column names, split params)",
    )
    parser.add_argument(
        "--output", type=str, default="outputs/splits.json",
        help="Output path for splits JSON",
    )
    parser.add_argument(
        "--test-frac", type=float, default=None,
        help="Holdout test fraction (default: from config, typically 0.1)",
    )
    parser.add_argument(
        "--n-folds", type=int, default=None,
        help="Number of CV folds (default: from config, typically 5)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (default: from config experiment.seed)",
    )
    parser.add_argument(
        "--precomputed-dir", type=str, default=None,
        help="Path to precomputed .pt dir — restricts splits to subjects with features",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(args.config)
    validate_config(config, required_keys=["data", "experiment"])

    # Load metadata
    import pandas as pd
    from pathlib import Path

    metadata_csv = Path(config.data.metadata_path) / "metadata.csv"
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_csv}")
    metadata = pd.read_csv(metadata_csv)
    logger.info("Loaded metadata: %d subjects", len(metadata))

    # Filter to subjects with precomputed features
    if args.precomputed_dir:
        precomputed_subjects = {
            p.stem for p in Path(args.precomputed_dir).glob("*.pt")
        }
        subject_col = config.data.get("subject_column", "ROSMAP_IndividualID")
        before = len(metadata)
        metadata = metadata[metadata[subject_col].isin(precomputed_subjects)]
        logger.info(
            "Filtered to %d subjects with precomputed features (from %d)",
            len(metadata), before,
        )

    # Split parameters (CLI overrides > config)
    data_cfg = config.data
    test_frac = args.test_frac if args.test_frac is not None else data_cfg.splits.test_frac
    n_folds = args.n_folds or data_cfg.splits.n_folds
    seed = args.seed or config.experiment.get("seed", 42)

    splits = create_stratified_splits(
        metadata,
        subject_column=data_cfg.get("subject_column", "ROSMAP_IndividualID"),
        pathology_column=data_cfg.splits.stratify_by[0] if data_cfg.splits.get("stratify_by") else "gpath",
        cognition_column=data_cfg.get("target_column", "cogn_global"),
        test_frac=test_frac,
        n_folds=n_folds,
        random_state=seed,
    )

    # Validate no leakage
    if not validate_no_leakage(splits):
        raise RuntimeError("Data leakage detected in splits — aborting")

    save_splits(splits, args.output)

    # Summary
    meta = splits["metadata"]
    print(f"\nSplits saved to {args.output}")
    print(f"  Subjects:     {meta['n_subjects']}")
    print(f"  Holdout test: {meta['n_test']} ({meta['test_frac']:.0%})")
    print(f"  Train/val:    {meta['n_train_val']} ({n_folds} folds)")
    print(f"  Seed:         {meta['random_state']}")


if __name__ == "__main__":
    main()
