"""
Inference script for cognitive resilience model.

Usage:
    uv run python scripts/run_inference.py --checkpoint path/to/best.ckpt --config configs/default.yaml
    uv run python scripts/run_inference.py --checkpoint path/to/best.ckpt --config configs/default.yaml --output-dir outputs/analysis/
    uv run python scripts/run_inference.py --checkpoint path/to/best.ckpt --config configs/default.yaml --extract-hgt-attention --extract-embeddings

Workflow:
1. Load config from YAML with optional CLI overrides
2. Load checkpoint and initialize Predictor
3. Build DataLoader for inference
4. Run inference with requested extraction options
5. Save predictions (parquet, csv) and attention weights (HDF5)
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.data.collate import create_dataloader
from src.data.datasets import CognitiveResilienceDataset, PrecomputedDataset
from src.inference.predict import Predictor
from src.utils.config import load_config

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with a trained cognitive resilience model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML (optional if config is in checkpoint)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: checkpoint parent / analysis)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to preprocessed data directory (overrides config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference (default: 32)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for inference: auto, cuda, cpu (default: auto)",
    )
    parser.add_argument(
        "--extract-hgt-attention",
        action="store_true",
        help="Extract HGT cell-cell communication attention",
    )
    parser.add_argument(
        "--extract-pma-attention",
        action="store_true",
        help="Extract PMA per-cell attention weights",
    )
    parser.add_argument(
        "--extract-region-attention",
        action="store_true",
        help="Extract per-subject region attention",
    )
    parser.add_argument(
        "--extract-embeddings",
        action="store_true",
        help="Extract branch/fused/attended embeddings",
    )
    parser.add_argument(
        "--extract-all",
        action="store_true",
        help="Extract all attention types and embeddings",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip CSV output (only parquet + HDF5)",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Config overrides in dotlist format (e.g., model.n_cell_types=31)",
    )
    return parser.parse_args()


def build_dataloader(
    config, data_path: str | None, batch_size: int
) -> DataLoader:
    """Build inference DataLoader from config and data path.

    Supports two modes:
    - Precomputed: load from .npz feature directory (fast, recommended)
    - On-the-fly: load from AnnData + metadata (slower, for debugging)

    Args:
        config: OmegaConf config with data section
        data_path: Path to precomputed directory or None to use config
        batch_size: Inference batch size

    Returns:
        DataLoader configured for inference (no shuffle, HGT multiregion format)
    """
    data_cfg = config.data

    # Resolve data path
    precomputed_dir = data_path or data_cfg.get("precomputed_dir")

    # Load metadata (always needed for targets/pathology)
    metadata_path = Path(data_cfg.metadata_path)
    metadata_csv = metadata_path / "metadata.csv"
    if not metadata_csv.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_csv}. "
            "Ensure the metadata CSV exists at the configured path."
        )
    metadata = pd.read_csv(metadata_csv)

    subject_column = data_cfg.get("subject_column", "ROSMAP_IndividualID")
    target_column = data_cfg.get("target_column", "cogn_global")
    pathology_columns = list(data_cfg.get("pathology_columns", []))

    if precomputed_dir is not None:
        precomputed_dir = Path(precomputed_dir)
        if not precomputed_dir.exists():
            raise FileNotFoundError(
                f"Precomputed data directory not found: {precomputed_dir}"
            )
        # Discover subject IDs from .npz files on disk
        subject_ids = sorted(
            p.stem for p in precomputed_dir.glob("*.npz")
        )
        if not subject_ids:
            raise ValueError(
                f"No .npz files found in {precomputed_dir}"
            )
        logger.info(
            "Found %d subjects in precomputed directory: %s",
            len(subject_ids), precomputed_dir,
        )
        dataset = PrecomputedDataset(
            feature_dir=precomputed_dir,
            subject_ids=subject_ids,
            metadata=metadata,
            subject_column=subject_column,
            target_column=target_column,
            pathology_columns=pathology_columns,
        )
    else:
        # On-the-fly mode: load AnnData
        adata_path = data_cfg.get("adata_path")
        if adata_path is None:
            raise ValueError(
                "No data source found. Provide --data-path for precomputed features, "
                "or set data.precomputed_dir or data.adata_path in config."
            )
        logger.info("Loading AnnData from %s (on-the-fly mode)", adata_path)
        import scanpy as sc  # Lazy import: only needed for AnnData mode
        adata = sc.read_h5ad(adata_path)
        # Use all subjects present in metadata
        subject_ids = metadata[subject_column].unique().tolist()
        dataset = CognitiveResilienceDataset(
            adata=adata,
            metadata=metadata,
            subject_ids=subject_ids,
            cell_type_column=data_cfg.get("cell_type_column", "supercluster_name"),
            subject_column=subject_column,
            target_column=target_column,
            pathology_columns=pathology_columns,
            max_cells_per_type=data_cfg.cell_sampling.get("max_cells_per_type", 1000),
            min_cells_threshold=data_cfg.cell_sampling.get("min_cells_threshold", 50),
        )

    dataloader = create_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        multiregion=True,
        use_hgt_format=True,
        prefetch_factor=None,
    )
    return dataloader


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = parse_args()

    # Load config if provided
    config = None
    if args.config:
        config = load_config(args.config, overrides=args.overrides if args.overrides else None)

    # Resolve checkpoint and output paths
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = checkpoint_path.parent.parent / "analysis"

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Load predictor
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    predictor = Predictor.from_checkpoint(
        checkpoint_path,
        device=args.device,
        config=config,
    )

    # Build DataLoader
    # If no config provided via CLI, use config recovered from checkpoint
    if config is None:
        config = predictor.config
        logger.info("Using config recovered from checkpoint")

    if config is None:
        raise ValueError(
            "No config available. Provide --config path/to/config.yaml "
            "or use a checkpoint that contains model_config."
        )

    # Validate config has required sections for data loading
    if not hasattr(config, "data") or config.data is None:
        raise ValueError(
            "Config recovered from checkpoint is missing 'data' section. "
            "Provide --config with a full configuration YAML that includes data paths."
        )

    from src.utils.config import validate_config
    validate_config(config, required_keys=["model"])

    data_path = args.data_path
    logger.info("Building inference DataLoader...")
    dataloader = build_dataloader(config, data_path, args.batch_size)
    logger.info(f"DataLoader ready: {len(dataloader.dataset)} subjects, batch_size={args.batch_size}")

    # Determine extraction flags
    # If no CLI flags explicitly set, check config for defaults
    extract_all = args.extract_all
    cli_flags_set = any([
        args.extract_hgt_attention,
        args.extract_pma_attention,
        args.extract_region_attention,
        args.extract_embeddings,
        extract_all,
    ])

    if not cli_flags_set and config is not None:
        # Use config inference defaults when no CLI flags provided
        config_extract = config.get("inference", {}).get("extract_attention", False)
        if config_extract:
            logger.info("Using config inference.extract_attention=true as default")
            extract_all = True

    extract_hgt = args.extract_hgt_attention or extract_all
    extract_pma = args.extract_pma_attention or extract_all
    extract_region = args.extract_region_attention or extract_all
    extract_emb = args.extract_embeddings or extract_all

    if not any([extract_hgt, extract_pma, extract_region, extract_emb]):
        logger.warning(
            "No extraction flags set — only predictions will be saved. "
            "Use --extract-all to extract all attention weights and embeddings, "
            "or use individual flags (--extract-hgt-attention, --extract-pma-attention, etc.)."
        )

    # Run inference
    logger.info("Running inference...")
    results = predictor.predict(
        dataloader,
        extract_hgt_attention=extract_hgt,
        extract_pma_attention=extract_pma,
        extract_region_attention=extract_region,
        extract_embeddings=extract_emb,
    )
    logger.info(f"Inference complete: {len(results.subject_ids)} subjects")

    # Save results
    saved = predictor.save_predictions(
        results,
        output_dir,
        save_csv=not args.no_csv,
    )

    logger.info("Saved files:")
    for fmt, path in saved.items():
        logger.info(f"  {fmt}: {path}")


if __name__ == "__main__":
    main()
