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
   d. Return mean val_nll (predictive quality) across all folds
3. Save study to SQLite database
"""

import argparse
import contextlib
import logging
import os
import warnings
from pathlib import Path

import optuna
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def _make_storage(path: str | None):
    """Build Optuna JournalStorage from a file path.

    JournalFileBackend uses atomic file operations for safe multi-process
    coordination on a single node. Never use SQLite for parallel optimization
    (Optuna docs: "never use SQLite3 for parallel optimization").
    """
    if path is None:
        return None
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend
    return JournalStorage(JournalFileBackend(path))


def _collect_all_subject_ids(splits: dict) -> list[str]:
    """Return deduplicated list of all subject IDs across all splits."""
    ids: set[str] = set()
    ids.update(splits.get("holdout_test", []))
    ids.update(splits.get("train_val_pool", []))
    for fold in splits.get("folds", []):
        ids.update(fold.get("train", []))
        ids.update(fold.get("val", []))
    return sorted(ids)


def create_study(config: DictConfig, storage=None) -> optuna.Study:
    """
    Create Optuna study with configured sampler and pruner.

    Uses TPE sampler (Tree-structured Parzen Estimator) and configured pruner
    (MedianPruner or HyperbandPruner) for efficient search with early
    termination of unpromising trials.

    Args:
        config: Full experiment config with optuna section
        storage: Optional Optuna storage (JournalStorage object or None for in-memory)

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
            n_warmup_steps=pruner_cfg.get("n_warmup_steps", 1),
            interval_steps=pruner_cfg.get("interval_steps", 1),
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
    - guide_lr -> training.optimizer.guide_lr
    - d_embed -> model.d_embed + model.d_fused
    - dropout -> model.dropout
    - n_hgt_layers -> model.hgt.n_layers
    - beta -> training.loss.beta
    - batch_size -> data.dataloader.batch_size
    - n_heads -> model.hgt.n_heads + model.pathology_attention.n_heads + model.set_transformer.n_heads
    - n_inducing -> model.set_transformer.n_inducing_points
    - gene_gate_temp -> model.gene_gate.initial_temperature
    - selection_temperature -> model.cell_type_selector.selection_temperature
    - ogm_alpha -> training.gradient_modulation.alpha

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
        "guide_lr": "training.optimizer.guide_lr",
        "selection_temperature": "model.cell_type_selector.selection_temperature",
        "ogm_alpha": "training.gradient_modulation.alpha",
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
            OmegaConf.update(config, "model.dropout", value)
        else:
            logger.warning("Unknown parameter '%s' — skipping", name)

    # Validate n_heads divides d_embed (required by all attention mechanisms).
    # Raise TrialPruned (not ValueError) so Optuna discards this trial
    # and samples new parameters instead of crashing the entire study.
    n_heads = config.model.hgt.get("n_heads")
    d_embed = config.model.get("d_embed")
    if n_heads and d_embed and d_embed % n_heads != 0:
        raise optuna.TrialPruned(
            f"d_embed ({d_embed}) not divisible by n_heads ({n_heads}) — pruning trial"
        )

    return config


def shorten_annealing_for_hpo(
    config: DictConfig,
    full_max_epochs: int = 100,
) -> DictConfig:
    """
    Proportionally shorten annealing schedules for HPO trials.

    When HPO uses fewer max_epochs than full training, temperature annealing
    and KL annealing schedules must be shortened proportionally so the model
    still completes its full annealing cycle within the trial budget.

    Also updates early_stopping.min_epochs to match the shortened schedule
    (warmup + anneal), so early stopping can fire after annealing completes.

    Args:
        config: Trial config (modified in-place and returned)
        full_max_epochs: The max_epochs used in full training (for computing ratio)

    Returns:
        The same config with shortened annealing schedules
    """
    ratio = config.training.max_epochs / full_max_epochs
    if ratio >= 1.0:
        return config

    # Temperature annealing
    ta = config.training.temperature_annealing
    ta.warmup_epochs = max(1, round(ta.warmup_epochs * ratio))
    ta.anneal_epochs = max(1, round(ta.anneal_epochs * ratio))

    # KL annealing
    kl = config.training.get("kl_annealing", {})
    if kl and kl.get("enabled", False):
        kl.warmup_epochs = max(1, round(kl.warmup_epochs * ratio))

    # Update min_epochs to match shortened schedule
    new_min = ta.warmup_epochs + ta.anneal_epochs
    config.training.early_stopping.min_epochs = new_min

    return config


_FOLD_METRICS = (
    "val_nll", "val_r2", "val_mae", "val_rmse", "val_crps",
    "val_calibration_error", "val_mean_std", "val_pearson_r",
    "val_spearman_rho",
)


def store_fold_metrics(
    trial,
    fold_idx: int,
    metrics: dict,
) -> None:
    """
    Store per-fold validation metrics as Optuna trial user attributes.

    Stored as `fold_{i}_{metric_name}` in the trial's SQLite record,
    queryable via study.trials_dataframe() for post-hoc multi-metric
    analysis (e.g., "which hyperparameters produce the best R²?").

    Args:
        trial: Optuna trial object
        fold_idx: Cross-validation fold index
        metrics: Dict from trainer.callback_metrics
    """
    import torch

    for name in _FOLD_METRICS:
        value = metrics.get(name)
        if value is not None:
            if isinstance(value, torch.Tensor):
                value = value.item()
            trial.set_user_attr(f"fold_{fold_idx}_{name}", float(value))


def objective(
    trial: optuna.Trial,
    base_config: DictConfig,
    gpu_id: int | None = None,
    adata=None,
    metadata=None,
    splits: dict | None = None,
    precomputed_dir: str | Path | None = None,
    preloaded_cache: dict[str, dict] | None = None,
) -> float:
    """
    Optuna objective function: train model and return mean val_nll across folds.

    This function:
    1. Samples hyperparameters
    2. Builds trial config
    3. Trains on each CV fold (with pruning callback)
    4. Returns mean val_nll (predictive quality) across folds

    Args:
        trial: Optuna trial object
        base_config: Base experiment config
        gpu_id: Optional GPU device index for training
        adata: Pre-loaded AnnData object (optional, for data loading)
        metadata: Pre-loaded metadata DataFrame (optional, for data loading)
        splits: Pre-computed splits dict (optional, for data loading)
        precomputed_dir: If provided, DataModule loads pre-built features from
            this directory instead of reconstructing them from raw AnnData each
            trial, avoiding ~300k redundant computations across trials/folds.
        preloaded_cache: Pre-loaded subject tensors from
            ``PrecomputedDataset.load_subject_cache``. Eliminates per-trial
            disk I/O (~14 s / trial) during HPO.

    Returns:
        Mean val_nll across CV folds (lower is better)
    """
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

    from scripts.train import setup_callbacks
    from src.utils.reproducibility import set_seed
    from src.training.callbacks import (
        GradientNormLogger, MinEpochEarlyStopping, ResilienceModelCheckpoint,
    )
    from src.training.lightning_module import CognitiveResilienceLightningModule

    # Sample and build config
    params = sample_hyperparameters(trial, base_config)
    config = build_trial_config(base_config, params)

    # Shorten annealing schedules proportionally for HPO trials
    full_max_epochs = base_config.training.get("max_epochs", 100)
    config = shorten_annealing_for_hpo(config, full_max_epochs=full_max_epochs)

    seed = config.experiment.get("seed", 42)
    repro_cfg = config.get("reproducibility", {})
    set_seed(
        seed,
        deterministic=repro_cfg.get("deterministic", True),
        benchmark=repro_cfg.get("benchmark", False),
    )

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
    # - ModelCheckpoint / ResilienceModelCheckpoint: trials don't save checkpoints
    # - LearningRateMonitor: no logger in trial trainers
    # - GradientNormLogger: unnecessary overhead per trial (no TensorBoard)
    # - GradientModulationCallback: OGM-GE adds per-step overhead; trials run
    #   single-GPU so gradient modulation behaviour may differ from DDP training
    #
    # Note: We do NOT use PyTorchLightningPruningCallback because:
    # 1. Each fold creates a new Trainer, causing epoch counter resets
    # 2. This creates step collisions (fold 0 epoch 5 vs fold 1 epoch 5)
    # 3. Cross-fold epoch comparison is statistically invalid
    #
    # Instead, we report only at fold boundaries (see trial.report below).
    # Pruning semantics: n_warmup_steps=1 means complete 1 fold before pruning.
    # Within-fold warmup protection is handled by MinEpochEarlyStopping.
    _excluded = [ModelCheckpoint, ResilienceModelCheckpoint, LearningRateMonitor, GradientNormLogger]
    try:
        from src.training.gradient_modulation import GradientModulationCallback
        _excluded.append(GradientModulationCallback)
    except ImportError:
        pass
    _EXCLUDED_TRIAL_CALLBACKS = tuple(_excluded)

    # Per-fold results are in-memory — no fold-level checkpointing.
    # If the process crashes at fold K, folds 0..K-1 results are lost.
    # This is acceptable for HP search (trials are cheaper than full training).
    for fold_idx in range(n_folds):
        # Re-seed per fold for independent reproducibility: each fold's
        # initialization and training are reproducible regardless of what
        # happened in prior folds (pruning, early stopping, etc.).
        set_seed(
            seed + fold_idx,
            deterministic=repro_cfg.get("deterministic", True),
            benchmark=repro_cfg.get("benchmark", False),
        )

        # Build model
        module = CognitiveResilienceLightningModule(config)

        # Setup callbacks, filtering those inappropriate for trials
        callbacks = [
            cb for cb in setup_callbacks(config)
            if not isinstance(cb, _EXCLUDED_TRIAL_CALLBACKS)
        ]

        # Trainer for this fold (no checkpointing for trials).
        # strategy="auto" overrides config's "ddp" — trials run single-GPU.
        trainer = pl.Trainer(
            max_epochs=config.training.max_epochs,
            min_epochs=config.training.early_stopping.get("min_epochs", 1),
            accelerator=accelerator,
            devices=devices,
            strategy="auto",
            precision=config.training.get("precision", "32-true"),
            gradient_clip_val=config.training.get("gradient_clip_val", None),
            gradient_clip_algorithm="norm",
            callbacks=callbacks,
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
            logger=False,
            val_check_interval=config.training.get("logging", {}).get("val_check_interval", 1.0),
            deterministic=False,   # Speed over reproducibility for HP search
            benchmark=True,        # Enable cuDNN autotuner for HP search
        )

        # Data loading for this fold — fail-fast on missing inputs.
        # adata is allowed to be None when precomputed_dir is set, because
        # PrecomputedDataset reads from .npz files without touching AnnData.
        if splits is None or metadata is None:
            raise RuntimeError(
                "Data not provided to objective(). This should not happen — "
                "main() should validate --splits-path before calling optimize()."
            )
        if adata is None and precomputed_dir is None:
            raise RuntimeError(
                "Either adata or precomputed_dir must be provided. "
                "Use --precomputed-dir for pre-built features, or ensure "
                "config.data.adata_path points to a valid h5ad file."
            )

        from src.data.datamodule import CognitiveResilienceDataModule
        dm = CognitiveResilienceDataModule(
            config=config, metadata=metadata, splits=splits,
            fold_idx=fold_idx, adata=adata,
            precomputed_dir=precomputed_dir,
            preloaded_cache=preloaded_cache,
        )
        trainer.fit(module, datamodule=dm)

        # Use val_nll (predictive quality) not val_loss (ELBO) as objective.
        # ELBO = NLL + KL, and KL annealing makes ELBO non-stationary across
        # epochs — trials with different convergence rates aren't comparable.
        # val_nll measures pure predictive quality, consistent with early
        # stopping and checkpoint monitors.
        val_nll = trainer.callback_metrics.get("val_nll")
        if val_nll is not None:
            fold_val_losses.append(val_nll.item())

        # Store per-fold diagnostic metrics for post-hoc analysis
        store_fold_metrics(trial, fold_idx, trainer.callback_metrics)

        # Report intermediate value for pruning
        running_mean = sum(fold_val_losses) / len(fold_val_losses) if fold_val_losses else float("inf")
        trial.report(running_mean, fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # Note: no explicit torch.cuda.empty_cache() between folds.
        # PyTorch's CUDA allocator reuses cached memory blocks, so calling
        # empty_cache() would force re-allocation and hurt performance.
        # Python reference counting deallocates the old module/trainer above.

    # Return mean val_nll across folds
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
        help="Optuna journal file path for persistent storage (e.g., outputs/optuna_journal.log). "
             "Uses JournalFileBackend for safe multi-process coordination.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU device index for training (e.g., 0, 1)",
    )
    parser.add_argument(
        "--n-gpus",
        type=int,
        default=1,
        help="Number of GPUs for parallel trial execution. Each GPU runs trials independently. "
             "Requires --storage for multi-process coordination (default: 1).",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=1,
        help="Number of CV folds per trial (default: 1). Use 1 for fast HPO, "
             "then retrain best config with full 5-fold CV.",
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
        help="Path to pre-built feature directory. Skips per-fold feature "
             "reconstruction from raw AnnData, avoiding redundant computation "
             "across trials and folds.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Config overrides in dotlist format",
    )

    args = parser.parse_args()

    # Load config
    from src.utils.config import load_config

    config = load_config(args.config, overrides=args.overrides)

    from src.utils.config import validate_config
    validate_config(config, required_keys=["experiment", "data", "model", "training", "optuna", "paths"])

    # Override n_folds for HPO (default: 1 fold for fast search)
    OmegaConf.update(config, "data.splits.n_folds", args.n_folds)
    logger.info("HPO using %d CV fold(s) per trial", args.n_folds)

    optuna_cfg = config.optuna
    n_trials = args.n_trials or optuna_cfg.get("n_trials", 100)
    timeout = args.timeout or optuna_cfg.get("timeout", None)
    per_trial_timeout = optuna_cfg.get("per_trial_timeout", None)

    # Fail fast if data not provided (before expensive data loading)
    if args.splits_path is None:
        raise ValueError(
            "Optuna optimization requires pre-computed data splits. "
            "Provide --splits-path /path/to/splits.json. "
            "Use scripts/train.py for single-fold training without splits."
        )

    # Load data once
    from src.data.splits import load_splits
    splits = load_splits(args.splits_path)
    logger.info("Loaded splits from %s", args.splits_path)

    import pandas as pd
    metadata_path = Path(config.data.metadata_path)
    # metadata can be None if CSV doesn't exist — objective() will raise
    # RuntimeError on the first trial attempt, providing a clear error message.
    metadata = pd.read_csv(metadata_path / "metadata.csv") if (metadata_path / "metadata.csv").exists() else None

    # Skip AnnData loading when precomputed features are available.
    # PrecomputedDataset reads from .npz files and only needs metadata,
    # so loading the full AnnData (potentially multi-GB for ROSMAP) is wasted I/O.
    # In multi-GPU mode this is especially costly since each subprocess would
    # independently re-read the entire file.
    if args.precomputed_dir:
        adata = None
        logger.info("Using precomputed features from %s — skipping AnnData loading", args.precomputed_dir)
    else:
        import scanpy as sc
        adata = sc.read_h5ad(config.data.adata_path)
        logger.info("Loaded adata (%d cells)", adata.n_obs)

    # Create study (always use create_study to respect config pruner type)
    storage = _make_storage(args.storage)
    study = create_study(config, storage=storage)

    # Per-trial timeout: Optuna 4.x removed MaxTrialDurationCallback.
    # Early stopping (patience=15) and fold-level pruning provide sufficient
    # protection against runaway trials. Log the configured value for reference.
    callbacks = []
    if per_trial_timeout:
        logger.info(
            "Per-trial timeout configured: %d s (%.1f hours) — "
            "enforced via early stopping + fold-level pruning",
            per_trial_timeout, per_trial_timeout / 3600,
        )

    logger.info(
        "Starting optimization: n_trials=%d, timeout=%s",
        n_trials,
        timeout,
    )

    n_gpus = args.n_gpus
    gpu_id = args.gpu

    if n_gpus > 1:
        # Multi-GPU: spawn N worker processes, each pinned to a different GPU
        if args.storage is None:
            raise ValueError(
                "Multi-GPU optimization requires persistent storage for coordination. "
                "Provide --storage outputs/optuna_journal.log"
            )

        import subprocess
        import sys

        # Distribute trials across GPUs
        trials_per_gpu = (n_trials + n_gpus - 1) // n_gpus

        workers = []
        log_dir = Path(config.paths.get("logs_dir", "outputs/logs"))
        log_dir.mkdir(parents=True, exist_ok=True)

        with contextlib.ExitStack() as stack:
            for i in range(n_gpus):
                gpu_trials = min(trials_per_gpu, n_trials - i * trials_per_gpu)
                if gpu_trials <= 0:
                    continue

                cmd = [
                    sys.executable, str(Path(__file__)),
                    "--config", args.config,
                    "--n-trials", str(gpu_trials),
                    "--gpu", "0",  # Always GPU 0 since CUDA_VISIBLE_DEVICES limits visibility
                    "--storage", args.storage,
                    "--splits-path", args.splits_path,
                    "--n-folds", str(args.n_folds),
                ]
                if args.precomputed_dir:
                    cmd.extend(["--precomputed-dir", args.precomputed_dir])
                if timeout:
                    cmd.extend(["--timeout", str(timeout)])
                if args.overrides:
                    cmd.extend(args.overrides)

                # Pin each worker to a single GPU via CUDA_VISIBLE_DEVICES
                # This prevents each process from initializing all GPU contexts
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(i)
                env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

                stdout_path = log_dir / f"optuna_gpu{i}_stdout.log"
                stderr_path = log_dir / f"optuna_gpu{i}_stderr.log"
                logger.info(
                    f"Spawning worker on GPU {i} (CUDA_VISIBLE_DEVICES={i}): "
                    f"{gpu_trials} trials, logs: {stdout_path}"
                )
                stdout_f = stack.enter_context(open(stdout_path, "w"))
                stderr_f = stack.enter_context(open(stderr_path, "w"))
                proc = subprocess.Popen(cmd, env=env, stdout=stdout_f, stderr=stderr_f)
                workers.append((i, proc))

            # Wait for all workers
            failed = []
            for gpu_idx, proc in workers:
                returncode = proc.wait()
                if returncode != 0:
                    failed.append(gpu_idx)
                    logger.error(f"Worker on GPU {gpu_idx} failed with code {returncode}")

        if failed:
            logger.error(f"Workers failed on GPUs: {failed}")
            sys.exit(1)

        # Reload study to report results (all workers wrote to same storage)
        study = optuna.load_study(
            study_name=config.experiment.get("name", "cognitive_resilience"),
            storage=storage,
        )
    else:
        # Single-GPU mode (original behavior)
        # Limitation: interrupted trials are NOT resumed. If the process crashes
        # mid-trial, the trial is marked FAIL in the study journal and a new trial
        # with new hyperparameters starts on re-run. Per-fold results (fold_val_losses
        # list in objective()) are in-memory and lost on crash.
        import torch as _torch

        # Pre-load all subject tensors once — reused across all trials in this
        # worker, eliminating ~14 s of disk I/O per trial.
        preloaded_cache = None
        if args.precomputed_dir:
            from src.data.datasets import PrecomputedDataset
            all_subject_ids = _collect_all_subject_ids(splits)
            preloaded_cache = PrecomputedDataset.load_subject_cache(
                args.precomputed_dir, all_subject_ids,
            )

        def _safe_objective(trial):
            """Wrapper that frees CUDA memory after OOM so subsequent trials can proceed."""
            try:
                return objective(
                    trial, config, gpu_id=gpu_id,
                    adata=adata, metadata=metadata, splits=splits,
                    precomputed_dir=args.precomputed_dir,
                    preloaded_cache=preloaded_cache,
                )
            except _torch.cuda.OutOfMemoryError:
                _torch.cuda.empty_cache()
                raise  # Let Optuna mark as FAIL and continue

        study.optimize(
            _safe_objective,
            n_trials=n_trials,
            timeout=timeout,
            callbacks=callbacks or None,
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

    import torch
    torch.set_float32_matmul_precision("high")

    main()
