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
6. Train with early stopping on val_nll
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
    SerialCompilationWarmup,
    TemperatureAnnealing,
)
from src.training.lightning_module import CognitiveResilienceLightningModule
from src.utils.config import load_config
from src.utils.experiment import ExperimentManager
from src.utils.hashing import hash_config
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
    - ModelCheckpoint: save best model by val_nll (predictive quality for Bayesian, MSE for deterministic)
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
            filename="epoch={epoch}-val_nll={val_nll:.4f}",
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

    # KL Annealing (Bayesian head only)
    kl_cfg = train_cfg.get("kl_annealing", {})
    if kl_cfg.get("enabled", False):
        from src.training.callbacks import KLAnnealingCallback
        callbacks.append(
            KLAnnealingCallback(
                alpha_min=kl_cfg.get("alpha_min", 0.01),
                warmup_epochs=kl_cfg.get("warmup_epochs", 5),
                schedule=kl_cfg.get("schedule", "linear"),
            )
        )

    # GradientNormLogger must be appended BEFORE GradientModulationCallback.
    # Both use on_before_optimizer_step; Lightning calls callbacks in list order.
    # GradientNormLogger needs to see raw (unmodified) gradient norms.
    log_cfg = train_cfg.get("logging", {})
    callbacks.append(
        GradientNormLogger(
            log_every_n_steps=log_cfg.get("log_every_n_steps", 10),
        )
    )

    # OGM-GE gradient modulation (Peng et al., CVPR 2022)
    gm_cfg = train_cfg.get("gradient_modulation", {})
    if gm_cfg.get("enabled", False):
        method = gm_cfg.get("method", "ogm_ge")
        if method != "ogm_ge":
            raise ValueError(
                f"Unknown gradient_modulation.method='{method}'. "
                f"Only 'ogm_ge' is currently supported."
            )
        from src.training.gradient_modulation import GradientModulationCallback
        callbacks.append(
            GradientModulationCallback(
                alpha=gm_cfg.get("alpha", 1.0),
                ge_enabled=gm_cfg.get("ge_enabled", True),
                log_modulation=gm_cfg.get("log_modulation", True),
                log_every_n_steps=log_cfg.get("log_every_n_steps", 10),
            )
        )

    # ResilienceModelCheckpoint (custom metadata)
    callbacks.append(ResilienceModelCheckpoint())

    # Serial torch.compile warmup (DDP only — serializes compilation
    # across ranks to avoid exceeding commit limit under overcommit_memory=2)
    if config.model.get("use_torch_compile", False):
        callbacks.append(SerialCompilationWarmup())

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
        gradient_clip_algorithm="norm",
        callbacks=callbacks,
        logger=tb_logger,
        log_every_n_steps=train_cfg.get("logging", {}).get("log_every_n_steps", 10),
        val_check_interval=train_cfg.get("logging", {}).get("val_check_interval", 1.0),
        deterministic=repro_cfg.get("deterministic", True),
        benchmark=repro_cfg.get("benchmark", False),
        enable_progress_bar=True,
        profiler=profiler,
        use_distributed_sampler=True,
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

    # Clean up stale /dev/shm caches from previous HPO or DDP runs.
    from src.utils.shm import cleanup_stale_shm
    cleanup_stale_shm()

    # Enable TF32 Tensor Cores for float32 matmuls on Ada GPUs.
    # Same exponent range as FP32, shorter mantissa (10 vs 23 bits).
    # Negligible precision impact for training; ~2-3x speedup on FP32 ops
    # not already covered by bf16-mixed.
    torch.set_float32_matmul_precision("high")

    # Set global seed (same on all DDP ranks for identical model initialization).
    # Rank-specific seeding for data augmentation is handled at the DataLoader
    # worker level (see CognitiveResilienceDataModule._make_worker_init_fn).
    seed = config.experiment.get("seed", 42)
    repro_cfg = config.get("reproducibility", {})
    set_seed(
        seed,
        deterministic=repro_cfg.get("deterministic", True),
        benchmark=repro_cfg.get("benchmark", False),
    )
    logger.info("Seed set to %d", seed)

    # Pyro DDP: store parameters on nn.Modules (not global ParamStore).
    # Required for DDP gradient sync — DDP synchronizes module.parameters(),
    # not Pyro's global dict.  Must be set before guide construction.
    # See: pyro.ai/examples/svi_lightning.html
    import pyro
    pyro.settings.set(module_local_params=True)

    # Create experiment directory structure via ExperimentManager.
    # Under DDP, generate_experiment_hash includes a timestamp, so ranks
    # would create different directories.  Rank 0 creates the experiment;
    # other ranks receive the path via a shared temp file.
    #
    # We check config.training.devices (not os.environ WORLD_SIZE) to
    # determine DDP intent, because Lightning's DDPStrategy doesn't set
    # WORLD_SIZE until trainer.fit() spawns worker processes.  The launcher
    # process (which becomes rank 0) runs this code BEFORE trainer.fit(),
    # so WORLD_SIZE is still 1.  Using config.training.devices ensures rank 0
    # writes .ddp_exp_path before rank 1 is spawned and polls for it.
    import os
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    base_dir = config.paths.get("output_dir", "outputs/")

    # Inject fold index into config so the experiment hash is fold-specific.
    # Without this, concurrent folds with identical config could race to
    # write checkpoints to the same directory (hash collision).
    if not args.final:
        OmegaConf.update(config, "experiment.fold_idx", args.fold)

    config_dict = OmegaConf.to_container(config, resolve=True)

    # Determine intended device count from config (not env var)
    devices_cfg = config.training.get("devices", 1)
    if devices_cfg == "auto":
        import torch as _torch
        n_devices = _torch.cuda.device_count()
    elif isinstance(devices_cfg, (list, tuple)):
        n_devices = len(devices_cfg)
    else:
        n_devices = int(devices_cfg)

    if n_devices > 1:
        exp_path_file = Path(base_dir) / ".ddp_exp_path"
        if local_rank == 0:
            # Delete stale file from previous runs BEFORE creating experiment.
            # Without this, rank 1 reads the stale path immediately (race condition).
            if exp_path_file.exists():
                exp_path_file.unlink()
            exp_manager = ExperimentManager(base_dir=base_dir)
            experiment = exp_manager.create_experiment(config_dict)
            # Atomic write: write to temp file then os.rename() so non-zero
            # ranks polling never read a truncated/partial path.
            import tempfile as _tempfile
            tmp_fd, tmp_path = _tempfile.mkstemp(dir=str(exp_path_file.parent))
            os.write(tmp_fd, str(experiment.exp_dir).encode())
            os.close(tmp_fd)
            os.rename(tmp_path, str(exp_path_file))
            # Clean up coordination file on exit so it doesn't confuse future runs.
            import atexit
            atexit.register(lambda f=exp_path_file: f.unlink(missing_ok=True))
            logger.info("Experiment created: %s", experiment.exp_hash)
        else:
            # Wait for rank 0 to create the experiment and write the path file.
            # The file was deleted by rank 0 at startup, so existence means
            # rank 0 has written the NEW path (not a stale one).
            import time as _time
            for _ in range(300):  # up to 30s
                if exp_path_file.exists():
                    break
                _time.sleep(0.1)
            else:
                raise TimeoutError(
                    f"Rank {local_rank}: timed out waiting for rank 0 to create experiment. "
                    f"Check rank 0 logs for errors."
                )
            exp_dir = Path(exp_path_file.read_text().strip())
            # Validate: experiment directory name contains full config hash.
            config_hash = hash_config(config_dict)
            if config_hash not in exp_dir.name:
                raise ValueError(
                    f"Rank {local_rank}: experiment path '{exp_dir.name}' does not contain "
                    f"expected config hash '{config_hash}'. Possible stale .ddp_exp_path file."
                )
            from src.utils.experiment import Experiment
            experiment = Experiment(
                exp_dir=exp_dir,
                config=config_dict,
                exp_hash=exp_dir.name,
            )
            logger.info("Joined experiment: %s", exp_dir.name)
    else:
        exp_manager = ExperimentManager(base_dir=base_dir)
        experiment = exp_manager.create_experiment(config_dict)
        logger.info("Experiment created: %s", experiment.exp_hash)

    # Override paths in config to use experiment-specific directories
    OmegaConf.update(config, "paths.output_dir", str(experiment.exp_dir))
    OmegaConf.update(config, "paths.checkpoint_dir", str(experiment.checkpoints_dir))
    OmegaConf.update(config, "paths.logs_dir", str(experiment.tensorboard_dir))

    # Data loading
    from src.data.splits import create_stratified_splits, load_splits
    from src.data.datamodule import CognitiveResilienceDataModule

    data_cfg = config.data
    adata = None
    metadata = None

    # Fall back to config values when CLI args are not provided
    if args.precomputed_dir is None:
        args.precomputed_dir = data_cfg.get("precomputed_dir", None)
    if args.splits_path is None:
        # Check default location
        default_splits = Path("outputs/splits.json")
        if default_splits.exists():
            args.splits_path = str(default_splits)
            logger.info("Using default splits path: %s", args.splits_path)

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
            # stratify_by[0] is the pathology column, cognition_column is from target_column.
            # Both are used for stratification (pathology bins × cognition bins).
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

    # Validate n_genes matches AnnData if available. Update config before
    # building the model so the correct value is used on first construction.
    if adata is not None:
        actual_n_genes = adata.n_vars
        if config.model.n_genes != actual_n_genes:
            logger.warning(
                "Config model.n_genes=%d but AnnData has %d genes. Updating config.",
                config.model.n_genes, actual_n_genes,
            )
            OmegaConf.update(config, "model.n_genes", actual_n_genes)

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

        # Auto-compute target_mean from training data for Bayesian prior centering
        if config.model.head.get("target_mean") is None and config.model.head.type == "bayesian":
            dm.setup("fit")
            target_mean = dm.train_target_mean
            OmegaConf.update(config, "model.head.target_mean", target_mean)
            logger.info("Auto-computed target_mean=%.4f from training set", target_mean)

        # Re-seed immediately before model construction to ensure identical
        # weight initialization regardless of preceding data-loading stochasticity.
        set_seed(
            seed,
            deterministic=repro_cfg.get("deterministic", True),
            benchmark=repro_cfg.get("benchmark", False),
        )

        # Build Lightning module (after target_mean is injected into config)
        module = CognitiveResilienceLightningModule(config)
        logger.info("Model built: %s", type(module.model).__name__)

        # Override callbacks: remove early stopping and val_loss-based checkpointing
        # to prevent any holdout data from influencing training decisions.
        default_callbacks = setup_callbacks(config)
        final_callbacks = [
            cb for cb in default_callbacks
            if not isinstance(cb, (MinEpochEarlyStopping, ModelCheckpoint))
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
        # Guard with is_global_zero: under DDP (not ddp_spawn), only the main
        # process continues after trainer.fit(), but this guard is defensive.
        if trainer.is_global_zero:
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

        # Auto-compute target_mean from training data for Bayesian prior centering
        if config.model.head.get("target_mean") is None and config.model.head.type == "bayesian":
            dm.setup("fit")
            target_mean = dm.train_target_mean
            OmegaConf.update(config, "model.head.target_mean", target_mean)
            logger.info("Auto-computed target_mean=%.4f from training set", target_mean)

        # Re-seed immediately before model construction to ensure identical
        # weight initialization regardless of preceding data-loading stochasticity.
        set_seed(
            seed,
            deterministic=repro_cfg.get("deterministic", True),
            benchmark=repro_cfg.get("benchmark", False),
        )

        # Build Lightning module (after target_mean is injected into config)
        module = CognitiveResilienceLightningModule(config)
        logger.info("Model built: %s", type(module.model).__name__)

        trainer = setup_trainer(config)
        logger.info("Trainer configured: max_epochs=%d", trainer.max_epochs)

        # Train
        trainer.fit(module, datamodule=dm, ckpt_path=args.resume_from)
        logger.info("Training complete.")

        # Export best checkpoint weights (not last epoch)
        if trainer.is_global_zero:
            best_ckpt = getattr(trainer.checkpoint_callback, "best_model_path", None)
            _export_weights(module, experiment.model_dir,
                            is_bayesian=config.model.head.type == "bayesian",
                            best_ckpt_path=best_ckpt if best_ckpt else None)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
