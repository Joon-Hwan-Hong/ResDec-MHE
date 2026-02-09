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
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf

from src.training.callbacks import (
    GradientNormLogger,
    MinEpochEarlyStopping,
    ResilienceModelCheckpoint,
    TemperatureAnnealing,
)
from src.training.lightning_module import CognitiveResilienceLightningModule
from src.utils.config import load_config
from src.utils.experiment import ExperimentManager
from src.utils.reproducibility import set_seed

logger = logging.getLogger(__name__)


def _export_weights(module, model_dir: Path, is_bayesian: bool) -> None:
    """Export inference-ready weights artifact.

    For deterministic heads: full model state_dict as weights.pt.
    For Bayesian heads: backbone-only (excluding prediction_head.*) as backbone_weights.pt.
    Bayesian inference requires the full .ckpt with guide and param store.
    """
    if is_bayesian:
        backbone_state = {
            k: v for k, v in module.model.state_dict().items()
            if not k.startswith("prediction_head.")
        }
        path = model_dir / "backbone_weights.pt"
        torch.save(backbone_state, path)
        logger.info("Saved backbone-only weights to %s (Bayesian model — use .ckpt for inference)", path)
    else:
        path = model_dir / "weights.pt"
        torch.save(module.model.state_dict(), path)
        logger.info("Saved weights-only artifact to %s", path)


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

    # EarlyStopping with min_epochs enforcement
    es_cfg = train_cfg.early_stopping
    callbacks.append(
        MinEpochEarlyStopping(
            min_epochs=es_cfg.get("min_epochs", 20),
            monitor=es_cfg.monitor,
            patience=es_cfg.patience,
            min_delta=es_cfg.min_delta,
            mode=es_cfg.mode,
        )
    )

    # LearningRateMonitor — skip in Bayesian mode (SVI uses Pyro's internal
    # optimizer with lrd decay; Lightning's LR monitor can't see Pyro's LR)
    head_type = config.model.head.get("type", "deterministic")
    if head_type != "bayesian":
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

    # Distributed training settings
    devices = train_cfg.get("devices", "auto")
    strategy = train_cfg.get("strategy", "auto")

    trainer = pl.Trainer(
        max_epochs=train_cfg.max_epochs,
        min_epochs=train_cfg.early_stopping.get("min_epochs", 1),
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
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
        "--final",
        action="store_true",
        default=False,
        help="Final training mode: train on full train_val_pool, evaluate on holdout_test. "
             "Requires --splits-path. Ignores --fold.",
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

    from src.utils.config import validate_config
    validate_config(config, required_keys=["experiment", "data", "model", "training", "paths"])

    # Set seed
    seed = config.experiment.get("seed", 42)
    repro_cfg = config.get("reproducibility", {})
    set_seed(
        seed,
        deterministic=repro_cfg.get("deterministic", True),
        benchmark=repro_cfg.get("benchmark", False),
    )
    logger.info("Seed set to %d", seed)

    # Create experiment directory structure via ExperimentManager
    base_dir = config.paths.get("output_dir", "outputs/")
    exp_manager = ExperimentManager(base_dir=base_dir)
    config_dict = OmegaConf.to_container(config, resolve=True)
    experiment = exp_manager.create_experiment(config_dict)
    logger.info("Experiment created: %s", experiment.exp_hash)

    # Override paths in config to use experiment-specific directories
    OmegaConf.update(config, "paths.output_dir", str(experiment.exp_dir))
    OmegaConf.update(config, "paths.checkpoint_dir", str(experiment.checkpoints_dir))
    OmegaConf.update(config, "paths.logs_dir", str(experiment.tensorboard_dir))

    # Build Lightning module
    module = CognitiveResilienceLightningModule(config)
    logger.info("Model built: %s", type(module.model).__name__)

    # Setup trainer
    trainer = setup_trainer(config)
    logger.info("Trainer configured: max_epochs=%d", trainer.max_epochs)

    # Data loading
    from src.data.splits import create_stratified_splits, load_splits
    from src.data.loaders import create_fold_dataloaders

    data_cfg = config.data
    adata = None
    metadata = None

    if args.precomputed_dir and not args.splits_path:
        raise ValueError(
            "When using --precomputed-dir, you must also provide --splits-path. "
            "Pre-compute splits first with a separate run, then pass both. "
            "This avoids loading the full AnnData file unnecessarily."
        )

    if args.splits_path:
        splits = load_splits(args.splits_path)
        logger.info("Loaded splits from %s", args.splits_path)
    else:
        adata = sc.read_h5ad(data_cfg.adata_path)
        metadata_path = Path(data_cfg.metadata_path)
        metadata_csv = metadata_path / "metadata.csv"
        if not metadata_csv.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_csv}. "
                "Provide --splits-path to skip metadata loading, or ensure "
                "the metadata CSV exists at the configured path."
            )
        metadata = pd.read_csv(metadata_csv)
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

    if metadata is None:
        metadata_path = Path(data_cfg.metadata_path)
        metadata_csv = metadata_path / "metadata.csv"
        if not metadata_csv.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_csv}. "
                "Ensure the metadata CSV exists at the configured path."
            )
        metadata = pd.read_csv(metadata_csv)

    if adata is None and not args.precomputed_dir:
        adata = sc.read_h5ad(data_cfg.adata_path)

    if args.final:
        if splits is None:
            raise ValueError(
                "Final training mode requires pre-computed splits. "
                "Provide --splits-path /path/to/splits.json."
            )

        from src.data.splits import get_final_train_subjects

        train_subjects = get_final_train_subjects(splits)
        test_subjects = splits["holdout_test"]

        logger.info(
            f"Final training mode: {len(train_subjects)} train subjects, "
            f"{len(test_subjects)} holdout test subjects"
        )

        # Create train-only loader: synthetic fold with all train_val_pool as train,
        # and train subjects also as "val" (required by create_fold_dataloaders
        # return signature) -- but we will NOT pass it to trainer.fit.
        train_fold = {"train": train_subjects, "val": train_subjects}
        splits_for_train = dict(splits)
        splits_for_train["folds"] = [train_fold]

        train_loader, _ = create_fold_dataloaders(
            config, adata, metadata, splits_for_train, fold_idx=0,
            precomputed_dir=args.precomputed_dir,
        )

        # Create test-only loader for post-training evaluation
        test_fold = {"train": test_subjects, "val": test_subjects}
        splits_for_test = dict(splits)
        splits_for_test["folds"] = [test_fold]

        _, test_loader = create_fold_dataloaders(
            config, adata, metadata, splits_for_test, fold_idx=0,
            precomputed_dir=args.precomputed_dir,
        )

        logger.info(
            "Final-mode dataloaders created: %d train, %d test subjects",
            len(train_subjects), len(test_subjects),
        )

        # Override callbacks: remove early stopping and val_loss-based checkpointing
        # to prevent any holdout data from influencing training decisions.
        default_callbacks = setup_callbacks(config)
        final_callbacks = [
            cb for cb in default_callbacks
            if not isinstance(cb, (MinEpochEarlyStopping, ModelCheckpoint,
                                   ResilienceModelCheckpoint))
        ]

        # Add last-epoch-only checkpoint (no metric selection)
        final_callbacks.append(
            ModelCheckpoint(
                dirpath=config.paths.get("checkpoint_dir", "outputs/checkpoints/"),
                save_last=True,
                save_top_k=0,
                filename="final-epoch={epoch}",
                auto_insert_metric_name=False,
            )
        )

        # Re-add ResilienceModelCheckpoint for custom metadata
        final_callbacks.append(ResilienceModelCheckpoint())

        # Override trainer with final-mode callbacks
        trainer = setup_trainer(config, callbacks=final_callbacks)

        logger.info(
            "Final mode: using fixed max_epochs=%d with no early stopping",
            trainer.max_epochs,
        )

        # Train without val_dataloaders -- holdout is never seen during training
        trainer.fit(module, train_dataloaders=train_loader)
        logger.info("Final training complete.")

        # Export weights-only artifact for inference (lighter than full Lightning checkpoint)
        _export_weights(module, experiment.model_dir, is_bayesian=config.model.head.type == "bayesian")

        # Single unbiased evaluation on holdout test set
        trainer.test(module, dataloaders=test_loader)
        logger.info("Holdout test evaluation complete.")
    else:
        # Standard fold-based training
        train_loader, val_loader = create_fold_dataloaders(
            config, adata, metadata, splits, args.fold,
            precomputed_dir=args.precomputed_dir,
        )
        logger.info("Fold %d dataloaders created", args.fold)

        # Train
        trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
        logger.info("Training complete.")

        # Export weights-only artifact for inference (lighter than full Lightning checkpoint)
        _export_weights(module, experiment.model_dir, is_bayesian=config.model.head.type == "bayesian")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
