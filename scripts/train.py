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


def _export_weights(module, model_dir: Path, is_bayesian: bool, best_ckpt_path: str | None = None) -> None:
    """Export inference-ready weights artifact.

    For deterministic heads: full model state_dict as weights.pt.
    For Bayesian heads: backbone-only (excluding prediction_head.*) as backbone_weights.pt.
    Bayesian inference requires the full .ckpt with guide and param store.

    Args:
        module: Lightning module (used if best_ckpt_path is None)
        model_dir: Directory to save weights
        is_bayesian: Whether model uses Bayesian head
        best_ckpt_path: If provided, load weights from this checkpoint instead of module
    """
    if best_ckpt_path:
        checkpoint = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", {})
        # Strip 'model.' prefix from Lightning state dict
        model_state = {k[6:]: v for k, v in state_dict.items() if k.startswith("model.")}
        if not model_state:
            raise RuntimeError(
                f"No 'model.*' keys found in checkpoint state_dict from {best_ckpt_path}. "
                f"Keys present: {list(state_dict.keys())[:5]}"
            )
        logger.info("Loading best checkpoint weights from %s", best_ckpt_path)
    else:
        model_state = module.model.state_dict()

    if is_bayesian:
        backbone_state = {k: v for k, v in model_state.items() if not k.startswith("prediction_head.")}
        path = model_dir / "backbone_weights.pt"
        torch.save(backbone_state, path)
        logger.info("Saved backbone-only weights to %s (Bayesian model — use .ckpt for inference)", path)
    else:
        path = model_dir / "weights.pt"
        torch.save(model_state, path)
        logger.info("Saved weights-only artifact to %s", path)


def setup_callbacks(config: DictConfig) -> list[pl.Callback]:
    """
    Create training callbacks from config.

    Returns list of:
    - ModelCheckpoint: save best model by val_loss (ELBO for Bayesian, MSE for deterministic)
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

    # LearningRateMonitor — works for both deterministic (CosineAnnealingLR)
    # and Bayesian (ExponentialLR) paths since both use standard torch optimizers.
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

    # Profiler setup
    profiler = None
    profiler_cfg = train_cfg.get("profiler", None)
    if profiler_cfg is not None and profiler_cfg.get("type", None):
        profiler_type = profiler_cfg.type
        if profiler_type == "simple":
            from lightning.pytorch.profilers import SimpleProfiler
            profiler = SimpleProfiler(
                dirpath=config.paths.get("logs_dir", "outputs/logs"),
                filename="profiler_simple",
            )
        elif profiler_type == "pytorch":
            from lightning.pytorch.profilers import PyTorchProfiler
            profiler = PyTorchProfiler(
                dirpath=config.paths.get("logs_dir", "outputs/logs"),
                filename="profiler_pytorch",
                emit_nvtx=profiler_cfg.get("emit_nvtx", False),
                export_to_chrome=profiler_cfg.get("export_to_chrome", True),
                row_limit=profiler_cfg.get("row_limit", 20),
                schedule=torch.profiler.schedule(
                    wait=profiler_cfg.get("wait", 1),
                    warmup=profiler_cfg.get("warmup", 1),
                    active=profiler_cfg.get("active", 3),
                    repeat=profiler_cfg.get("repeat", 1),
                ) if profiler_cfg.get("use_schedule", False) else None,
            )
            logger.info(f"PyTorch profiler enabled, output to {config.paths.get('logs_dir', 'outputs/logs')}")

    trainer = pl.Trainer(
        max_epochs=train_cfg.max_epochs,
        min_epochs=train_cfg.early_stopping.get("min_epochs", 1),
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=train_cfg.get("precision", "32-true"),
        gradient_clip_val=train_cfg.get("gradient_clip_val", None),
        callbacks=callbacks,
        logger=tb_logger,
        log_every_n_steps=train_cfg.get("logging", {}).get("log_every_n_steps", 10),
        val_check_interval=train_cfg.get("logging", {}).get("val_check_interval", 1.0),
        deterministic=repro_cfg.get("deterministic", True),
        benchmark=repro_cfg.get("benchmark", False),
        enable_progress_bar=True,
        profiler=profiler,
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
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from (e.g., outputs/.../last.ckpt). "
             "Restores model weights, optimizer state, epoch, scheduler state, and "
             "global RNG states. Note: not bit-reproducible vs uninterrupted run "
             "(DataLoader position and per-worker CellSampler RNG not restored).",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Config overrides in dotlist format (e.g., training.max_epochs=50)",
    )

    args = parser.parse_args()

    if args.final and not args.splits_path:
        raise ValueError(
            "Final training mode (--final) requires --splits-path to ensure "
            "the holdout test set matches the one used during HP optimization."
        )

    if args.resume_from and not Path(args.resume_from).exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {args.resume_from}")

    if args.resume_from:
        logger.info("Will resume training from checkpoint: %s", args.resume_from)

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

    # Data loading
    from src.data.splits import create_stratified_splits, load_splits
    from src.data.datamodule import CognitiveResilienceDataModule

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
        import scanpy as sc
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
        import scanpy as sc
        adata = sc.read_h5ad(data_cfg.adata_path)

    if args.final:
        # Defensive guard: splits should always be non-None here because
        # --final requires --splits-path (validated above), but kept as
        # defense-in-depth for future refactors that might reorder validation.
        if splits is None:
            raise ValueError(
                "Final training mode requires pre-computed splits. "
                "Provide --splits-path /path/to/splits.json."
            )

        dm = CognitiveResilienceDataModule(
            config=config, metadata=metadata, splits=splits,
            fold_idx=0, precomputed_dir=args.precomputed_dir,
            adata=adata, final_mode=True,
        )
        logger.info("Final mode DataModule created")

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
        trainer.fit(module, datamodule=dm, ckpt_path=args.resume_from)
        logger.info("Final training complete.")

        # Export weights-only artifact for inference (lighter than full Lightning checkpoint).
        # Final mode uses last-epoch weights (not "best") because there is no validation
        # set for model selection — all data is used for training. Compare with the
        # fold-based path below which loads from best_model_path.
        _export_weights(module, experiment.model_dir, is_bayesian=config.model.head.type == "bayesian")

        # Single unbiased evaluation on holdout test set
        trainer.test(module, datamodule=dm)
        logger.info("Holdout test evaluation complete.")
    else:
        # Standard fold-based training
        dm = CognitiveResilienceDataModule(
            config=config, metadata=metadata, splits=splits,
            fold_idx=args.fold, precomputed_dir=args.precomputed_dir,
            adata=adata,
        )
        logger.info("Fold %d DataModule created", args.fold)

        trainer = setup_trainer(config)
        logger.info("Trainer configured: max_epochs=%d", trainer.max_epochs)

        # Train
        trainer.fit(module, datamodule=dm, ckpt_path=args.resume_from)
        logger.info("Training complete.")

        # Export best checkpoint weights (not last epoch)
        best_ckpt = getattr(trainer.checkpoint_callback, "best_model_path", None)
        _export_weights(module, experiment.model_dir,
                        is_bayesian=config.model.head.type == "bayesian",
                        best_ckpt_path=best_ckpt if best_ckpt else None)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
