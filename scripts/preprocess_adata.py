#!/usr/bin/env python3
"""Preprocess raw ROSMAP AnnData for downstream ML and LIANA+ pipelines.

Applies the full preprocessing pipeline from src/data/preprocessing.py:
  1. Gene QC (filter by min_cells)
  2. Round CellBender-corrected counts
  3. HVG selection on raw counts (seurat_v3)
  4. Force-include CellChatDB L-R genes
  5. Store raw counts in .raw
  6. Normalize (target_sum=1e4) + log1p
  7. Filter to HVG + L-R gene set

Saves the preprocessed AnnData to disk for use by:
  - scripts/run_liana.py (CCC analysis)
  - scripts/precompute_features.py (PrecomputedDataset generation)

Usage:
    # With splits (recommended — prevents HVG leakage from test/val subjects):
    uv run python scripts/preprocess_adata.py \
        --config configs/default.yaml \
        --output data/snRNAseq/adata_ROSMAP_preprocessed.h5ad \
        --splits-path outputs/splits.json

    # Without splits (uses all subjects for HVG — acceptable for exploration):
    uv run python scripts/preprocess_adata.py \
        --config configs/default.yaml \
        --output data/snRNAseq/adata_ROSMAP_preprocessed.h5ad
"""

import argparse
import json
import logging
import time

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess raw ROSMAP AnnData for ML and LIANA+",
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path to save preprocessed AnnData (.h5ad)",
    )
    parser.add_argument(
        "--splits-path", type=str, default=None,
        help="Path to splits JSON — restricts HVG selection to training subjects "
             "only, preventing data leakage. Recommended for final preprocessing.",
    )
    args = parser.parse_args()

    from src.utils.config import load_config, validate_config
    config = load_config(args.config)
    validate_config(config, required_keys=["data"])
    data_cfg = config.data

    from src.data.preprocessing import preprocess_adata

    # Load training subject IDs from splits if provided
    training_subject_ids = None
    if args.splits_path:
        with open(args.splits_path) as f:
            splits = json.load(f)
        training_subject_ids = splits.get("train_val_pool")
        if training_subject_ids:
            logger.info(
                "HVG selection restricted to %d training subjects (leak-free)",
                len(training_subject_ids),
            )
        else:
            logger.warning(
                "Splits file has no 'train_val_pool' key — using all subjects for HVG"
            )

    logger.info("Starting preprocessing pipeline...")
    t0 = time.time()

    subject_col = data_cfg.get("subject_column", "ROSMAP_IndividualID")

    adata = preprocess_adata(
        adata_path=data_cfg.adata_path,
        cellchatdb_path=data_cfg.cellchatdb_path,
        n_hvg=data_cfg.preprocessing.n_hvg,
        target_sum=data_cfg.preprocessing.target_sum,
        min_cells_per_gene=data_cfg.preprocessing.min_cells_per_gene,
        hvg_flavor="seurat_v3",
        copy=False,  # No need to copy — we're saving to a new file
        training_subject_ids=training_subject_ids,
        subject_column=subject_col,
        seed=config.experiment.get("seed", 42),
    )

    elapsed = time.time() - t0
    logger.info(
        "Preprocessing complete in %.1fs: %d cells x %d genes",
        elapsed, adata.n_obs, adata.n_vars,
    )

    logger.info("Saving to %s ...", args.output)
    t1 = time.time()
    adata.write_h5ad(args.output)
    logger.info("Saved in %.1fs", time.time() - t1)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
