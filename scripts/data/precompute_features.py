#!/usr/bin/env python3
"""Pre-compute per-subject .npz feature files from AnnData.

Creates PrecomputedDataset-compatible .npz files for each subject,
avoiding the need to load the full AnnData during training. This is
the recommended approach for large datasets (e.g., full ROSMAP with
3.9M cells) where on-the-fly processing is too slow or memory-intensive.

Usage:
    # Basic (no LIANA — CCC edges will be empty):
    uv run python scripts/data/precompute_features.py \
        --config configs/default.yaml \
        --output-dir data/precomputed/

    # With LIANA results:
    uv run python scripts/data/precompute_features.py \
        --config configs/default.yaml \
        --output-dir data/precomputed/ \
        --liana-dir data/liana_cache/

    # Subset of subjects:
    uv run python scripts/data/precompute_features.py \
        --config configs/default.yaml \
        --output-dir data/precomputed/ \
        --splits-path outputs/splits.json
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute per-subject .npz features from AnnData"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory to save .npz files (one per subject)",
    )
    parser.add_argument(
        "--liana-dir", type=str, default=None,
        help="Directory with LIANA result parquet files (optional, CCC edges empty if omitted)",
    )
    parser.add_argument(
        "--splits-path", type=str, default=None,
        help="Path to splits JSON — only precompute subjects in splits (optional, default: all subjects)",
    )
    parser.add_argument(
        "--adata", type=str, default=None,
        help="Path to AnnData .h5ad file (overrides config adata_path). "
             "Use this to point to a preprocessed AnnData.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing .npz files (default: skip existing)",
    )
    args = parser.parse_args()

    # Load config
    from src.utils.config import load_config, validate_config
    config = load_config(args.config)
    validate_config(config, required_keys=["data", "model"])
    data_cfg = config.data

    # Load metadata
    metadata_dir = Path(data_cfg.metadata_path)
    metadata_csv = metadata_dir / "metadata.csv"
    if not metadata_csv.exists():
        logger.error("Metadata file not found: %s", metadata_csv)
        sys.exit(1)
    metadata = pd.read_csv(metadata_csv)
    logger.info("Loaded metadata: %d subjects", len(metadata))

    # Load AnnData
    import scanpy as sc
    adata_path = args.adata or data_cfg.adata_path
    logger.info("Loading AnnData from %s (this may take several minutes)...", adata_path)
    t0 = time.time()
    adata = sc.read_h5ad(adata_path)
    logger.info("Loaded AnnData: %s in %.1fs", adata.shape, time.time() - t0)

    # Determine subject list
    subject_col = data_cfg.get("subject_column", "ROSMAP_IndividualID")
    all_subjects = sorted(adata.obs[subject_col].unique().tolist())

    if args.splits_path:
        with open(args.splits_path) as f:
            splits = json.load(f)
        # Collect all unique subjects across all splits
        split_subjects = set()
        for key in ("holdout_test", "train_val_pool"):
            if key in splits:
                split_subjects.update(splits[key])
        for fold in splits.get("folds", []):
            split_subjects.update(fold.get("train", []))
            split_subjects.update(fold.get("val", []))
        subject_ids = sorted(split_subjects & set(all_subjects))
        logger.info("Filtered to %d subjects from splits", len(subject_ids))
    else:
        subject_ids = all_subjects
        logger.info("Using all %d subjects", len(subject_ids))

    # Filter metadata to subjects in AnnData
    if subject_col in metadata.columns:
        metadata = metadata[metadata[subject_col].isin(subject_ids)]
        metadata = metadata.set_index(subject_col)
    logger.info("Metadata filtered: %d subjects", len(metadata))

    # Drop subjects with NaN in target or pathology columns (can't train on them)
    target_col = data_cfg.get("target_column", "cogn_global")
    pathology_cols = list(data_cfg.get("pathology_columns", []))
    required_cols = [target_col] + pathology_cols
    nan_subjects = set()
    for col in required_cols:
        if col in metadata.columns:
            nans = metadata[metadata[col].isna()].index.tolist()
            if nans:
                logger.warning(
                    "%d subjects have NaN in '%s', excluding: %s",
                    len(nans), col, nans,
                )
                nan_subjects.update(nans)
    if nan_subjects:
        subject_ids = [s for s in subject_ids if s not in nan_subjects]
        logger.info("Excluded %d subjects with NaN values, %d remaining", len(nan_subjects), len(subject_ids))

    # Log subjects missing from metadata (helps debug preprocessing issues)
    missing_from_metadata = set(subject_ids) - set(metadata.index)
    if missing_from_metadata:
        logger.warning(
            "%d subjects in splits but not in metadata: %s",
            len(missing_from_metadata),
            sorted(missing_from_metadata)[:10],  # show first 10
        )

    # Load LIANA results (optional)
    liana_results = {}
    if args.liana_dir:
        liana_dir = Path(args.liana_dir)
        for sid in subject_ids:
            # Check both naming conventions: liana_{sid}.parquet and {sid}.parquet
            parquet_file = liana_dir / f"liana_{sid}.parquet"
            if not parquet_file.exists():
                parquet_file = liana_dir / f"{sid}.parquet"
            if parquet_file.exists():
                liana_results[sid] = pd.read_parquet(parquet_file)
        logger.info("Loaded LIANA results for %d/%d subjects", len(liana_results), len(subject_ids))
    else:
        logger.info("No LIANA directory provided — CCC edges will be empty")

    # Seed all RNGs for reproducible precomputation
    from src.utils.reproducibility import set_seed
    seed = config.experiment.get("seed", 42)
    set_seed(seed, deterministic=False, benchmark=False)

    # Create dataset
    from src.data.datasets import CognitiveResilienceDataset, save_precomputed_features

    dataset = CognitiveResilienceDataset(
        adata=adata,
        metadata=metadata,
        subject_ids=subject_ids,
        subject_column=subject_col,
        cell_type_column=data_cfg.get("cell_type_column", "supercluster_name"),
        target_column=data_cfg.get("target_column", "cogn_global"),
        pathology_columns=list(data_cfg.get("pathology_columns", [])),
        max_cells_per_type=data_cfg.cell_sampling.max_cells_per_type,
        min_cells_threshold=data_cfg.cell_sampling.min_cells_threshold,
        sampling_strategy=data_cfg.cell_sampling.sampling_strategy,
        sampling_seed=seed,
        region_column=data_cfg.get("region_column", "BrainRegion"),
        liana_results=liana_results if liana_results else None,
    )

    # Check for existing files and skip if not overwriting
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    skip_existing = set()
    if not args.overwrite:
        existing = set(p.stem for p in output_dir.glob("*.npz"))
        skip_existing = existing & set(subject_ids)
        if skip_existing:
            logger.info(
                "%d subjects already precomputed — skipping (use --overwrite to redo).",
                len(skip_existing),
            )

    # Precompute
    logger.info("Starting precomputation to %s...", output_dir)
    t0 = time.time()
    save_precomputed_features(dataset, output_dir, verbose=True, skip_subjects=skip_existing)
    elapsed = time.time() - t0
    logger.info(
        "Done! Precomputed %d subjects in %.1fs (%.2fs/subject)",
        len(dataset), elapsed, elapsed / max(len(dataset), 1),
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
