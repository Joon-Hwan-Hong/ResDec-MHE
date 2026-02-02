"""
Optuna hyperparameter optimization for cognitive resilience model.

Usage:
    uv run python scripts/optuna_optimize.py --config configs/default.yaml --n-trials 100
    uv run python scripts/optuna_optimize.py --config configs/default.yaml --n-trials 50 --timeout 3600

    # Monitor with dashboard:
    optuna-dashboard sqlite:///outputs/optuna.db

Workflow:
1. Create Optuna study with TPE sampler + Hyperband pruner
2. For each trial:
   a. Sample hyperparameters from search space
   b. Build trial config (override base config with sampled params)
   c. Train on each CV fold with pruning callback
   d. Return mean val_loss across all folds
3. Save study to SQLite database
"""

import argparse
import logging
from copy import deepcopy
from pathlib import Path

import optuna
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def create_study(config: DictConfig) -> optuna.Study:
    """
    Create Optuna study with configured sampler and pruner.

    Uses TPE sampler (Tree-structured Parzen Estimator) and Hyperband pruner
    for efficient search with early termination of unpromising trials.

    Args:
        config: Full experiment config with optuna section

    Returns:
        Configured Optuna Study (direction=minimize)
    """
    optuna_cfg = config.optuna

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
            n_warmup_steps=pruner_cfg.get("n_warmup_steps", 10),
        )
    else:
        raise ValueError(f"Unknown pruner type: {pruner_cfg.type}")

    study = optuna.create_study(
        study_name=config.experiment.get("name", "cognitive_resilience"),
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
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
    - dropout -> model.dropout
    - n_hgt_layers -> model.hgt.n_layers
    - beta -> training.loss.beta
    - batch_size -> data.dataloader.batch_size
    - n_heads -> model.hgt.n_heads + model.pathology_attention.n_heads
    - n_inducing -> model.set_transformer.n_inducing_points
    - gene_gate_temp -> model.gene_gate.initial_temperature

    Args:
        base_config: Base experiment config (not modified)
        params: Dict of sampled hyperparameters

    Returns:
        New DictConfig with overrides applied
    """
    config = OmegaConf.create(OmegaConf.to_container(base_config, resolve=True))

    # Parameter mapping
    param_map = {
        "lr": "training.optimizer.lr",
        "weight_decay": "training.optimizer.weight_decay",
        "dropout": "model.dropout",
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
        else:
            logger.warning("Unknown parameter '%s' — skipping", name)

    return config


def objective(
    trial: optuna.Trial,
    base_config: DictConfig,
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

    Returns:
        Mean validation loss across CV folds (lower is better)
    """
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping

    from scripts.train import set_seed, setup_callbacks
    from src.training.lightning_module import CognitiveResilienceLightningModule

    # Sample and build config
    params = sample_hyperparameters(trial, base_config)
    config = build_trial_config(base_config, params)

    seed = config.experiment.get("seed", 42)
    set_seed(seed)

    n_folds = config.data.splits.get("n_folds", 5)
    fold_val_losses = []

    for fold_idx in range(n_folds):
        # Build model
        module = CognitiveResilienceLightningModule(config)

        # Setup callbacks with Optuna pruning
        callbacks = setup_callbacks(config)
        callbacks.append(
            optuna.integration.PyTorchLightningPruningCallback(
                trial, monitor="val_loss"
            )
        )

        # Trainer for this fold
        trainer = pl.Trainer(
            max_epochs=config.training.max_epochs,
            accelerator="auto",
            precision=config.training.get("precision", "32"),
            gradient_clip_val=config.training.get("gradient_clip_val", None),
            callbacks=callbacks,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            deterministic=True,
        )

        # Note: In production, load data and create fold-specific dataloaders here.
        # trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # For now, return the trial's best val_loss from the trainer
        # In production: fold_val_losses.append(trainer.callback_metrics["val_loss"].item())

    # Return mean val_loss across folds
    # In production: return sum(fold_val_losses) / len(fold_val_losses)
    return float("inf")  # Placeholder until data loading is connected


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

    # Create study
    if args.storage:
        # With persistent storage
        study = optuna.create_study(
            study_name=config.experiment.get("name", "cognitive_resilience"),
            storage=args.storage,
            direction="minimize",
            sampler=optuna.samplers.TPESampler(
                seed=optuna_cfg.sampler.get("seed", 42),
                n_startup_trials=optuna_cfg.sampler.get("n_startup_trials", 10),
                multivariate=True,
            ),
            pruner=optuna.pruners.HyperbandPruner(
                min_resource=optuna_cfg.pruner.get("min_resource", 5),
                max_resource=optuna_cfg.pruner.get("max_resource", 100),
                reduction_factor=optuna_cfg.pruner.get("reduction_factor", 3),
            ),
            load_if_exists=True,
        )
    else:
        study = create_study(config)

    logger.info(
        "Starting optimization: n_trials=%d, timeout=%s",
        n_trials,
        timeout,
    )

    study.optimize(
        lambda trial: objective(trial, config),
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
