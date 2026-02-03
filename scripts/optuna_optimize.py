"""
Optuna hyperparameter optimization for cognitive resilience model.

Usage:
    uv run python scripts/optuna_optimize.py --config configs/default.yaml --n-trials 100
    uv run python scripts/optuna_optimize.py --config configs/default.yaml --n-trials 50 --timeout 3600

    # Monitor with dashboard:
    optuna-dashboard sqlite:///outputs/optuna.db

Workflow:
1. Create Optuna study with TPE sampler + MedianPruner
2. For each trial:
   a. Sample hyperparameters from search space
   b. Build trial config (override base config with sampled params)
   c. Train on each CV fold with pruning callback
   d. Return mean val_loss across all folds
3. Save study to SQLite database
"""

import argparse
import logging
import warnings
from pathlib import Path

import optuna
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def create_study(config: DictConfig, storage: str | None = None) -> optuna.Study:
    """
    Create Optuna study with configured sampler and pruner.

    Uses TPE sampler (Tree-structured Parzen Estimator) and configured pruner
    (MedianPruner or HyperbandPruner) for efficient search with early
    termination of unpromising trials.

    Args:
        config: Full experiment config with optuna section
        storage: Optional Optuna storage URL (e.g., sqlite:///outputs/optuna.db)

    Returns:
        Configured Optuna Study (direction=minimize)
    """
    optuna_cfg = config.optuna

    # Suppress Optuna experimental warnings (e.g., multivariate TPE)
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

    # Sampler
    sampler_cfg = optuna_cfg.sampler
    sampler = optuna.samplers.TPESampler(
        seed=sampler_cfg.get("seed", 42),
        n_startup_trials=sampler_cfg.get("n_startup_trials", 10),
        multivariate=True,
    )

    # Pruner
    pruner_cfg = optuna_cfg.pruner
    if pruner_cfg.type == "hyperband":
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=pruner_cfg.get("min_resource", 5),
            max_resource=pruner_cfg.get("max_resource", 100),
            reduction_factor=pruner_cfg.get("reduction_factor", 3),
        )
    elif pruner_cfg.type == "median":
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=pruner_cfg.get("n_startup_trials", 5),
            n_warmup_steps=pruner_cfg.get("n_warmup_steps", 25),
            interval_steps=pruner_cfg.get("interval_steps", 5),
        )
    else:
        raise ValueError(f"Unknown pruner type: {pruner_cfg.type}")

    study = optuna.create_study(
        study_name=config.experiment.get("name", "cognitive_resilience"),
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    return study


def sample_hyperparameters(
    trial: optuna.Trial,
    config: DictConfig,
) -> dict:
    """
    Sample hyperparameters from Optuna trial using configured search space.

    Supports parameter types:
    - loguniform: Log-uniform distribution (e.g., learning rate)
    - uniform: Uniform distribution (e.g., dropout)
    - categorical: Categorical choices (e.g., d_embed)
    - int: Integer range (e.g., n_hgt_layers)

    Args:
        trial: Optuna trial object
        config: Full config with optuna.search_space section

    Returns:
        Dict mapping parameter name to sampled value
    """
    search_space = config.optuna.search_space
    params = {}

    for name, spec in search_space.items():
        param_type = spec.type

        if param_type == "loguniform":
            params[name] = trial.suggest_float(name, spec.low, spec.high, log=True)
        elif param_type == "uniform":
            params[name] = trial.suggest_float(name, spec.low, spec.high)
        elif param_type == "categorical":
            choices = list(spec.choices)
            params[name] = trial.suggest_categorical(name, choices)
        elif param_type == "int":
            params[name] = trial.suggest_int(name, spec.low, spec.high)
        else:
            raise ValueError(f"Unknown parameter type '{param_type}' for '{name}'")

    return params


def build_trial_config(
    base_config: DictConfig,
    params: dict,
) -> DictConfig:
    """
    Override base config with trial hyperparameters.

    Maps sampled parameters to their config locations:
    - lr -> training.optimizer.lr
    - weight_decay -> training.optimizer.weight_decay
    - d_embed -> model.d_embed + model.d_fused
    - dropout -> model.dropout + model.pseudobulk.dropout + model.hgt.dropout + model.set_transformer.dropout
    - n_hgt_layers -> model.hgt.n_layers
    - beta -> training.loss.beta
    - batch_size -> data.dataloader.batch_size
    - n_heads -> model.hgt.n_heads + model.pathology_attention.n_heads + model.set_transformer.n_heads
    - n_inducing -> model.set_transformer.n_inducing_points
    - gene_gate_temp -> model.gene_gate.initial_temperature

    Args:
        base_config: Base experiment config (not modified)
        params: Dict of sampled hyperparameters

    Returns:
        New DictConfig with overrides applied
    """
    config = OmegaConf.create(OmegaConf.to_container(base_config, resolve=True))

    # Parameter mapping (simple 1:1 mappings)
    param_map = {
        "lr": "training.optimizer.lr",
        "weight_decay": "training.optimizer.weight_decay",
        "n_hgt_layers": "model.hgt.n_layers",
        "beta": "training.loss.beta",
        "batch_size": "data.dataloader.batch_size",
        "n_inducing": "model.set_transformer.n_inducing_points",
        "gene_gate_temp": "model.gene_gate.initial_temperature",
    }

    for name, value in params.items():
        if name in param_map:
            OmegaConf.update(config, param_map[name], value)
        elif name == "d_embed":
            # d_embed also updates d_fused to match
            OmegaConf.update(config, "model.d_embed", value)
            OmegaConf.update(config, "model.d_fused", value)
        elif name == "n_heads":
            # Shared head count across attention mechanisms
            OmegaConf.update(config, "model.hgt.n_heads", value)
            OmegaConf.update(config, "model.pathology_attention.n_heads", value)
            OmegaConf.update(config, "model.set_transformer.n_heads", value)
        elif name == "dropout":
            # Dropout is shared across all branches
            OmegaConf.update(config, "model.dropout", value)
            OmegaConf.update(config, "model.pseudobulk.dropout", value)
            OmegaConf.update(config, "model.hgt.dropout", value)
            OmegaConf.update(config, "model.set_transformer.dropout", value)
        else:
            logger.warning("Unknown parameter '%s' — skipping", name)

    return config


def objective(
    trial: optuna.Trial,
    base_config: DictConfig,
    gpu_id: int | None = None,
    adata=None,
    metadata=None,
    splits: dict | None = None,
) -> float:
    """
    Optuna objective function: train model and return mean val_loss across folds.

    This function:
    1. Samples hyperparameters
    2. Builds trial config
    3. Trains on each CV fold (with pruning callback)
    4. Returns mean validation loss across folds

    Args:
        trial: Optuna trial object
        base_config: Base experiment config
        gpu_id: Optional GPU device index for training
        adata: Pre-loaded AnnData object (optional, for data loading)
        metadata: Pre-loaded metadata DataFrame (optional, for data loading)
        splits: Pre-computed splits dict (optional, for data loading)

    Returns:
        Mean validation loss across CV folds (lower is better)
    """
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

    from scripts.train import setup_callbacks
    from src.utils.reproducibility import set_seed
    from src.training.callbacks import MinEpochEarlyStopping, ResilienceModelCheckpoint
    from src.training.lightning_module import CognitiveResilienceLightningModule

    # Sample and build config
    params = sample_hyperparameters(trial, base_config)
    config = build_trial_config(base_config, params)

    seed = config.experiment.get("seed", 42)
    set_seed(seed)

    n_folds = config.data.splits.get("n_folds", 5)
    fold_val_losses = []

    # GPU configuration
    if gpu_id is not None:
        accelerator = "gpu"
        devices = [gpu_id]
    else:
        accelerator = "auto"
        devices = "auto"

    # Callback types to exclude for Optuna trials:
    # - ModelCheckpoint: trials don't save checkpoints
    # - ResilienceModelCheckpoint: same reason
    # - LearningRateMonitor: no logger in trial trainers
    #
    # Note: We do NOT use PyTorchLightningPruningCallback because:
    # 1. Each fold creates a new Trainer, causing epoch counter resets
    # 2. This creates step collisions (fold 0 epoch 5 vs fold 1 epoch 5)
    # 3. Cross-fold epoch comparison is statistically invalid
    #
    # Instead, we report only at fold boundaries (see trial.report below).
    # Pruning semantics: n_warmup_steps=1 means complete 1 fold before pruning.
    # Within-fold warmup protection is handled by MinEpochEarlyStopping.
    _EXCLUDED_TRIAL_CALLBACKS = (ModelCheckpoint, ResilienceModelCheckpoint, LearningRateMonitor)

    for fold_idx in range(n_folds):
        # Build model
        module = CognitiveResilienceLightningModule(config)

        # Setup callbacks, filtering those inappropriate for trials
        callbacks = [
            cb for cb in setup_callbacks(config)
            if not isinstance(cb, _EXCLUDED_TRIAL_CALLBACKS)
        ]

        # Trainer for this fold (no checkpointing for trials)
        trainer = pl.Trainer(
            max_epochs=config.training.max_epochs,
            min_epochs=config.training.early_stopping.get("min_epochs", 1),
            accelerator=accelerator,
            devices=devices,
            precision=config.training.get("precision", "32"),
            gradient_clip_val=config.training.get("gradient_clip_val", None),
            callbacks=callbacks,
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
            logger=False,
            deterministic=config.get("reproducibility", {}).get("deterministic", True),
            benchmark=config.get("reproducibility", {}).get("benchmark", False),
        )

        # Data loading for this fold
        if splits is None or adata is None or metadata is None:
            logger.warning(
                "Data not provided to objective() — returning inf. "
                "Pass --splits-path to enable real training."
            )
            return float("inf")

        from src.data.loaders import create_fold_dataloaders
        train_loader, val_loader = create_fold_dataloaders(
            config, adata, metadata, splits, fold_idx,
        )
        trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

        val_loss = trainer.callback_metrics.get("val_loss")
        if val_loss is not None:
            fold_val_losses.append(val_loss.item())

        # Report intermediate value for pruning
        trial.report(fold_val_losses[-1] if fold_val_losses else float("inf"), fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    # Return mean val_loss across folds
    if fold_val_losses:
        return sum(fold_val_losses) / len(fold_val_losses)
    return float("inf")  # Placeholder when data not provided


def main() -> None:
    """Main optimization entry point."""
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter optimization for cognitive resilience model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Number of optimization trials (overrides config)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout in seconds (overrides config)",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL (e.g., sqlite:///outputs/optuna.db)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU device index for training (e.g., 0, 1)",
    )
    parser.add_argument(
        "--splits-path",
        type=str,
        default=None,
        help="Path to pre-computed splits JSON file",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Config overrides in dotlist format",
    )

    args = parser.parse_args()

    # Load config
    from scripts.train import load_config

    config = load_config(args.config, overrides=args.overrides)

    optuna_cfg = config.optuna
    n_trials = args.n_trials or optuna_cfg.get("n_trials", 100)
    timeout = args.timeout or optuna_cfg.get("timeout", None)

    # Load data once (if splits path provided)
    adata = None
    metadata = None
    splits = None
    if args.splits_path:
        from src.data.splits import load_splits
        splits = load_splits(args.splits_path)
        logger.info("Loaded splits from %s", args.splits_path)

        # Load adata and metadata
        import scanpy as sc
        import pandas as pd
        adata = sc.read_h5ad(config.data.adata_path)
        metadata_path = Path(config.data.metadata_path)
        metadata = pd.read_csv(metadata_path / "metadata.csv") if (metadata_path / "metadata.csv").exists() else None
        logger.info("Loaded adata (%d cells) and metadata", adata.n_obs)

    # Create study (always use create_study to respect config pruner type)
    study = create_study(config, storage=args.storage)

    logger.info(
        "Starting optimization: n_trials=%d, timeout=%s",
        n_trials,
        timeout,
    )

    gpu_id = args.gpu
    study.optimize(
        lambda trial: objective(
            trial, config, gpu_id=gpu_id,
            adata=adata, metadata=metadata, splits=splits,
        ),
        n_trials=n_trials,
        timeout=timeout,
    )

    # Report results
    logger.info("Best trial: %s", study.best_trial.params)
    logger.info("Best value: %.6f", study.best_value)

    # Save best config
    best_params = study.best_trial.params
    best_config = build_trial_config(config, best_params)
    output_dir = Path(config.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(best_config, output_dir / "best_config.yaml")
    logger.info("Best config saved to %s", output_dir / "best_config.yaml")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    main()
