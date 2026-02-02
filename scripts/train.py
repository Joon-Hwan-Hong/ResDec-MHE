"""
Training script for cognitive resilience model.

Usage:
    uv run python scripts/train.py --config configs/default.yaml
    uv run python scripts/train.py --config configs/default.yaml training.max_epochs=50
    uv run python scripts/train.py --config configs/default.yaml --fold 0

Workflow:
1. Load config from YAML with optional CLI overrides
2. Set seed for reproducibility
3. Load preprocessed data and create stratified splits
4. Build DataLoaders for the specified fold
5. Instantiate Lightning module and Trainer with callbacks
6. Train with early stopping on val_loss
7. Save best checkpoint and experiment config
"""

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf

from src.training.callbacks import GradientNormLogger, ResilienceModelCheckpoint, TemperatureAnnealing
from src.training.lightning_module import CognitiveResilienceLightningModule

logger = logging.getLogger(__name__)


def load_config(
    config_path: str,
    overrides: list[str] | None = None,
) -> DictConfig:
    """
    Load configuration from YAML file with optional CLI overrides.

    Args:
        config_path: Path to YAML configuration file
        overrides: List of dotlist overrides (e.g., ["training.max_epochs=50"])

    Returns:
        Merged OmegaConf DictConfig

    Raises:
        FileNotFoundError: If config_path does not exist
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = OmegaConf.load(str(path))

    if overrides:
        override_conf = OmegaConf.from_dotlist(overrides)
        config = OmegaConf.merge(config, override_conf)

    return config


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across all frameworks.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)


def setup_callbacks(config: DictConfig) -> list[pl.Callback]:
    """
    Create training callbacks from config.

    Returns list of:
    - ModelCheckpoint: save best model by val_loss
    - EarlyStopping: stop unpromising runs
    - LearningRateMonitor: log LR to TensorBoard
    - TemperatureAnnealing: anneal gene gate temperature
    - GradientNormLogger: monitor per-branch gradient health
    - ResilienceModelCheckpoint: save custom metadata (version, hash, RNG states)

    Args:
        config: Full experiment config

    Returns:
        List of Lightning callbacks
    """
    train_cfg = config.training
    callbacks = []

    # ModelCheckpoint
    ckpt_cfg = train_cfg.checkpoint
    callbacks.append(
        ModelCheckpoint(
            dirpath=config.paths.get("checkpoint_dir", "outputs/checkpoints/"),
            monitor=ckpt_cfg.monitor,
            mode=ckpt_cfg.mode,
            save_top_k=ckpt_cfg.save_top_k,
            save_last=ckpt_cfg.get("save_last", True),
            filename="epoch={epoch}-val_loss={val_loss:.4f}",
            auto_insert_metric_name=False,
        )
    )

    # EarlyStopping
    es_cfg = train_cfg.early_stopping
    callbacks.append(
        EarlyStopping(
            monitor=es_cfg.monitor,
            patience=es_cfg.patience,
            min_delta=es_cfg.min_delta,
            mode=es_cfg.mode,
        )
    )

    # LearningRateMonitor
    callbacks.append(LearningRateMonitor(logging_interval="step"))

    # TemperatureAnnealing
    ta_cfg = train_cfg.temperature_annealing
    callbacks.append(
        TemperatureAnnealing(
            tau_max=ta_cfg.tau_max,
            tau_min=ta_cfg.tau_min,
            warmup_epochs=ta_cfg.warmup_epochs,
            anneal_epochs=ta_cfg.anneal_epochs,
            schedule=ta_cfg.schedule,
        )
    )

    # GradientNormLogger
    log_cfg = train_cfg.get("logging", {})
    callbacks.append(
        GradientNormLogger(
            log_every_n_steps=log_cfg.get("log_every_n_steps", 10),
        )
    )

    # ResilienceModelCheckpoint (custom metadata)
    callbacks.append(ResilienceModelCheckpoint())

    return callbacks


def setup_trainer(
    config: DictConfig,
    callbacks: list[pl.Callback] | None = None,
) -> pl.Trainer:
    """
    Create Lightning Trainer from config.

    Args:
        config: Full experiment config
        callbacks: Optional list of callbacks (if None, created from config)

    Returns:
        Configured Lightning Trainer
    """
    train_cfg = config.training

    if callbacks is None:
        callbacks = setup_callbacks(config)

    # Logger
    tb_logger = TensorBoardLogger(
        save_dir=config.paths.get("logs_dir", "outputs/logs/"),
        name=config.experiment.get("name", "cognitive_resilience"),
    )

    # Accelerator
    device_cfg = config.experiment.get("device", "auto")
    if device_cfg == "auto":
        accelerator = "auto"
    elif device_cfg == "cuda":
        accelerator = "gpu"
    else:
        accelerator = "cpu"

    # Read reproducibility settings from config
    repro_cfg = config.get("reproducibility", {})

    trainer = pl.Trainer(
        max_epochs=train_cfg.max_epochs,
        min_epochs=train_cfg.early_stopping.get("min_epochs", 1),
        accelerator=accelerator,
        precision=train_cfg.get("precision", "32"),
        gradient_clip_val=train_cfg.get("gradient_clip_val", None),
        callbacks=callbacks,
        logger=tb_logger,
        log_every_n_steps=train_cfg.get("logging", {}).get("log_every_n_steps", 10),
        val_check_interval=train_cfg.get("logging", {}).get("val_check_interval", 1.0),
        deterministic=repro_cfg.get("deterministic", True),
        benchmark=repro_cfg.get("benchmark", False),
        enable_progress_bar=True,
    )

    return trainer


def main() -> None:
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train cognitive resilience model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Cross-validation fold index (0-indexed)",
    )
    parser.add_argument(
        "--splits-path",
        type=str,
        default=None,
        help="Path to pre-computed splits JSON file",
    )
    parser.add_argument(
        "--precomputed-dir",
        type=str,
        default=None,
        help="Path to precomputed feature directory (skip on-the-fly preprocessing)",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Config overrides in dotlist format (e.g., training.max_epochs=50)",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config, overrides=args.overrides)
    logger.info("Config loaded from %s", args.config)

    # Set seed
    seed = config.experiment.get("seed", 42)
    set_seed(seed)
    logger.info("Seed set to %d", seed)

    # Save config for reproducibility
    output_dir = Path(config.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output_dir / "config.yaml")

    # Build Lightning module
    module = CognitiveResilienceLightningModule(config)
    logger.info("Model built: %s", type(module.model).__name__)

    # Setup trainer
    trainer = setup_trainer(config)
    logger.info("Trainer configured: max_epochs=%d", trainer.max_epochs)

    # Data loading
    from src.data.splits import create_stratified_splits, get_fold_subjects, load_splits
    from src.data.datasets import CognitiveResilienceDataset, PrecomputedDataset
    from src.data.collate import create_dataloader

    data_cfg = config.data
    dl_cfg = data_cfg.dataloader

    adata = None
    metadata = None

    # Load or create splits
    if args.splits_path:
        splits = load_splits(args.splits_path)
        logger.info("Loaded splits from %s", args.splits_path)
    else:
        import scanpy as sc
        import pandas as pd
        adata = sc.read_h5ad(data_cfg.adata_path)
        metadata_path = Path(data_cfg.metadata_path)
        metadata = pd.read_csv(metadata_path / "metadata.csv") if (metadata_path / "metadata.csv").exists() else None
        splits = create_stratified_splits(
            metadata,
            subject_column=data_cfg.get("subject_column", "ROSMAP_IndividualID"),
            pathology_column=data_cfg.splits.stratify_by[0] if data_cfg.splits.get("stratify_by") else "gpath",
            cognition_column=data_cfg.get("target_column", "cogn_global"),
            test_frac=data_cfg.splits.test_frac,
            n_folds=data_cfg.splits.n_folds,
            random_state=seed,
        )
        logger.info("Created stratified splits: %d folds", data_cfg.splits.n_folds)

    # Get fold subjects
    train_subjects = get_fold_subjects(splits, fold_idx=args.fold, split_type="train")
    val_subjects = get_fold_subjects(splits, fold_idx=args.fold, split_type="val")
    logger.info("Fold %d: %d train, %d val subjects", args.fold, len(train_subjects), len(val_subjects))

    # Ensure metadata is loaded (needed for both dataset types)
    if metadata is None:
        import pandas as pd
        metadata_path = Path(data_cfg.metadata_path)
        metadata = pd.read_csv(metadata_path / "metadata.csv")

    # Create datasets
    if args.precomputed_dir:
        train_ds = PrecomputedDataset(
            feature_dir=args.precomputed_dir,
            subject_ids=train_subjects,
            metadata=metadata,
            subject_column=data_cfg.get("subject_column", "ROSMAP_IndividualID"),
            target_column=data_cfg.get("target_column", "cogn_global"),
            pathology_columns=list(data_cfg.get("pathology_columns", [])),
        )
        val_ds = PrecomputedDataset(
            feature_dir=args.precomputed_dir,
            subject_ids=val_subjects,
            metadata=metadata,
            subject_column=data_cfg.get("subject_column", "ROSMAP_IndividualID"),
            target_column=data_cfg.get("target_column", "cogn_global"),
            pathology_columns=list(data_cfg.get("pathology_columns", [])),
        )
    else:
        if adata is None:
            import scanpy as sc
            adata = sc.read_h5ad(data_cfg.adata_path)

        train_ds = CognitiveResilienceDataset(
            adata, metadata, train_subjects,
            cell_type_column=data_cfg.get("cell_type_column", "supercluster_name"),
            subject_column=data_cfg.get("subject_column", "ROSMAP_IndividualID"),
            target_column=data_cfg.get("target_column", "cogn_global"),
            pathology_columns=list(data_cfg.get("pathology_columns", [])),
            max_cells_per_type=data_cfg.cell_sampling.get("max_cells_per_type", 1000),
            min_cells_threshold=data_cfg.cell_sampling.get("min_cells_threshold", 50),
        )
        val_ds = CognitiveResilienceDataset(
            adata, metadata, val_subjects,
            cell_type_column=data_cfg.get("cell_type_column", "supercluster_name"),
            subject_column=data_cfg.get("subject_column", "ROSMAP_IndividualID"),
            target_column=data_cfg.get("target_column", "cogn_global"),
            pathology_columns=list(data_cfg.get("pathology_columns", [])),
            max_cells_per_type=data_cfg.cell_sampling.get("max_cells_per_type", 1000),
            min_cells_threshold=data_cfg.cell_sampling.get("min_cells_threshold", 50),
        )

    # Create dataloaders
    train_loader = create_dataloader(
        train_ds, batch_size=dl_cfg.batch_size, shuffle=True,
        num_workers=dl_cfg.get("num_workers", 4),
        pin_memory=dl_cfg.get("pin_memory", True),
        multiregion=True, use_hgt_format=True,
    )
    val_loader = create_dataloader(
        val_ds, batch_size=dl_cfg.batch_size, shuffle=False,
        num_workers=dl_cfg.get("num_workers", 4),
        pin_memory=dl_cfg.get("pin_memory", True),
        multiregion=True, use_hgt_format=True,
    )

    # Train
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    logger.info("Training complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
