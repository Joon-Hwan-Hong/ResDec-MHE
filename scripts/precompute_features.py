#!/usr/bin/env python3
"""Pre-compute per-subject .npz feature files from AnnData.

Creates PrecomputedDataset-compatible .npz files for each subject,
avoiding the need to load the full AnnData during training. This is
the recommended approach for large datasets (e.g., full ROSMAP with
3.9M cells) where on-the-fly processing is too slow or memory-intensive.

Usage:
    # Basic (no LIANA — CCC edges will be empty):
    uv run python scripts/precompute_features.py \
        --config configs/default.yaml \
        --output-dir data/precomputed/

    # With LIANA results:
    uv run python scripts/precompute_features.py \
        --config configs/default.yaml \
        --output-dir data/precomputed/ \
        --liana-dir data/liana_cache/

    # Subset of subjects:
    uv run python scripts/precompute_features.py \
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
        "--overwrite", action="store_true",
        help="Overwrite existing .npz files (default: skip existing)",
    )
    args = parser.parse_args()

    # Load config
    from src.utils.config import load_config
    config = load_config(args.config)
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
    adata_path = data_cfg.adata_path
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

    # Create dataset
    from src.data.datasets import CognitiveResilienceDataset, save_precomputed_features
    from src.data.cell_sampling import CellSampler

    sampler = CellSampler(
        max_cells_per_type=data_cfg.cell_sampling.max_cells_per_type,
        min_cells_threshold=data_cfg.cell_sampling.min_cells_threshold,
        strategy=data_cfg.cell_sampling.sampling_strategy,
    )

    dataset = CognitiveResilienceDataset(
        adata=adata,
        metadata=metadata,
        subject_ids=subject_ids,
        sampler=sampler,
        n_genes=config.model.n_genes,
        subject_column=subject_col,
        cell_type_column=data_cfg.get("cell_type_column", "supercluster_name"),
        target_column=data_cfg.get("target_column", "cogn_global"),
        pathology_columns=list(data_cfg.get("pathology_columns", [])),
        liana_results=liana_results if liana_results else None,
    )

    # Check for existing files and skip if not overwriting
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.overwrite:
        existing = set(p.stem for p in output_dir.glob("*.npz"))
        n_existing = len(existing & set(subject_ids))
        if n_existing > 0:
            logger.info(
                "%d subjects already precomputed (use --overwrite to redo). "
                "Skipping existing.",
                n_existing,
            )

    # Precompute
    logger.info("Starting precomputation to %s...", output_dir)
    t0 = time.time()
    save_precomputed_features(dataset, output_dir, verbose=True)
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
