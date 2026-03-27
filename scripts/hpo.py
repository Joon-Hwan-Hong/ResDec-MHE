"""
Ray Tune + Optuna TPE hyperparameter optimization for cognitive resilience model.

- Per-trial process isolation via Ray Tune (fresh GPU context per trial)
- Optuna TPE search + MedianStoppingRule cross-trial pruning
- Object store data sharing (zero-copy via ray.put)

Usage:
    uv run python scripts/hpo.py --config configs/hpo_round6.yaml \\
        --splits-path data/splits.json --precomputed-dir data/precomputed/ \\
        --n-gpus 2 --n-trials 120 --n-folds 2
"""

import argparse
import gc
import logging
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pyro
import torch
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Search space translation
# ---------------------------------------------------------------------------

def _yaml_to_search_space(config: DictConfig) -> dict:
    """Translate YAML search_space section to Ray Tune search space primitives.

    Args:
        config: Full experiment config.

    Returns:
        Dict mapping parameter names to Ray Tune distributions.
    """
    from ray import tune

    space = {}
    search_cfg = config.get("hpo", {}).get("search_space", {})
    for name, spec in search_cfg.items():
        spec_type = spec.get("type", "")
        if spec_type == "loguniform":
            space[name] = tune.loguniform(spec.low, spec.high)
        elif spec_type == "uniform":
            space[name] = tune.uniform(spec.low, spec.high)
        elif spec_type == "categorical":
            space[name] = tune.choice(list(spec.choices))
        elif spec_type == "int":
            # randint is exclusive on the upper bound
            space[name] = tune.randint(spec.low, spec.high + 1)
        else:
            raise ValueError(f"Unknown search space type '{spec_type}' for param '{name}'")
    return space


# ---------------------------------------------------------------------------
# Warm-start from prior HPO run
# ---------------------------------------------------------------------------

def load_warm_start_data(ray_dir: str, search_space_keys: list[str]) -> tuple[list[dict], list[float]]:
    """Load prior HPO trials as warm-start data for OptunaSearch.

    Parses result.json files from the most recent experiment in a Ray Tune
    directory and extracts (config, best_val_nll) pairs. Only HPs present
    in the current search space are included in each point. Trials from
    older experiment runs in the same directory are filtered out.

    Args:
        ray_dir: Path to prior Ray Tune experiment directory.
        search_space_keys: List of HP names in the current search space.

    Returns:
        Tuple of (points_to_evaluate, evaluated_rewards) ready for
        OptunaSearch constructor.
    """
    import json
    import re
    from pathlib import Path

    ray_dir = Path(ray_dir)

    # Find latest experiment start time to filter trials
    state_files = sorted(ray_dir.glob("experiment_state-*.json"))
    latest_ts = None
    if state_files:
        latest_ts = state_files[-1].stem.replace("experiment_state-", "")

    points = []
    rewards = []

    for trial_dir in sorted(ray_dir.iterdir()):
        if not trial_dir.is_dir() or not trial_dir.name.startswith("train_fn_"):
            continue

        # Filter to latest experiment if timestamp available
        if latest_ts:
            ts_match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})$', trial_dir.name)
            if ts_match and ts_match.group(1) < latest_ts:
                continue

        result_file = trial_dir / "result.json"
        if not result_file.exists():
            continue

        best_nll = float("inf")
        config = {}
        with open(result_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                nll = record.get("val_nll", float("inf"))
                if nll < best_nll:
                    best_nll = nll
                if not config and "config" in record:
                    config = record["config"]

        if not config or best_nll == float("inf"):
            continue

        # Only include HPs that exist in the current search space
        point = {k: config[k] for k in search_space_keys if k in config}
        if len(point) == len(search_space_keys):
            points.append(point)
            rewards.append(best_nll)

    logger.info(
        "Loaded %d warm-start trials from %s (best val_nll=%.4f, worst=%.4f)",
        len(points), ray_dir,
        min(rewards) if rewards else float("nan"),
        max(rewards) if rewards else float("nan"),
    )
    return points, rewards


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def build_config_from_ray(ray_config: dict, base_config: DictConfig) -> DictConfig | None:
    """Apply Ray-sampled hyperparameters to a base OmegaConf config.

    Returns ``None`` when the sampled HP combination is invalid (e.g.,
    d_embed not divisible by n_heads), signalling the caller to skip
    this trial.

    Args:
        ray_config: Dict of sampled hyperparameters from Ray Tune.
        base_config: Base experiment config (not modified).

    Returns:
        New DictConfig with overrides applied, or ``None`` if the HP
        combination is invalid.
    """
    config = OmegaConf.create(OmegaConf.to_container(base_config, resolve=True))

    # 1:1 parameter mappings
    param_map = {
        "lr": "training.optimizer.lr",
        "weight_decay": "training.optimizer.weight_decay",
        "n_hgt_layers": "model.hgt.n_layers",
        "beta": "training.loss.beta",
        "batch_size": "data.dataloader.batch_size",
        "n_inducing": "model.set_transformer.n_inducing_points",
        "gene_gate_temp": "model.gene_gate.initial_temperature",
        "guide_lr": "training.optimizer.guide_lr",
        "fusion_type": "model.fusion.type",
        "fusion_n_heads": "model.fusion.n_heads",
    }

    for name, value in ray_config.items():
        # BOHB's ConfigSpace returns numpy types (np.int64, np.str_) that
        # OmegaConf rejects. Convert to native Python types.
        if hasattr(value, 'item'):
            value = value.item()  # np scalar → Python scalar
        elif isinstance(value, np.generic):
            value = value.item()

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
        elif name == "tau_min":
            OmegaConf.update(config, "training.temperature_annealing.tau_min", value)
        elif name == "anneal_epochs":
            OmegaConf.update(config, "training.temperature_annealing.anneal_epochs", int(value))
        else:
            logger.warning("Unknown parameter '%s' — skipping", name)

    # concat_normalized doesn't use attention heads — reset to default so
    # Ax/BO doesn't model fusion_n_heads as relevant for this fusion type.
    if config.model.fusion.type == "concat_normalized":
        OmegaConf.update(config, "model.fusion.n_heads", 4)

    # Validate n_heads divides d_embed (required by all attention mechanisms).
    # Return None (caller skips trial) for invalid HP combos.
    n_heads = config.model.hgt.get("n_heads")
    d_embed = config.model.get("d_embed")
    if n_heads and d_embed and d_embed % n_heads != 0:
        logger.warning(
            "d_embed (%d) not divisible by n_heads (%d) — skipping trial",
            d_embed, n_heads,
        )
        return None
    # Same check for fusion n_heads
    fusion_n_heads = config.model.get("fusion", {}).get("n_heads")
    if fusion_n_heads and d_embed and d_embed % fusion_n_heads != 0:
        logger.warning(
            "d_embed (%d) not divisible by fusion_n_heads (%d) — skipping trial",
            d_embed, fusion_n_heads,
        )
        return None

    return config


# ---------------------------------------------------------------------------
# Ray Tune reporting callback
# ---------------------------------------------------------------------------

def _make_tune_report_callback():
    """Create Ray Tune's official Lightning callback for per-epoch metric reporting.

    Uses TuneReportCheckpointCallback with save_checkpoints=False (HPO trials
    don't save checkpoints). Handles sanity-check skipping internally.
    """
    from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
    return TuneReportCheckpointCallback(
        metrics={
            "val_nll": "val_nll",
            "val_r2": "val_r2",
            "val_pearson_r": "val_pearson_r",
            "val_spearman_rho": "val_spearman_rho",
            "val_rmse": "val_rmse",
        },
        save_checkpoints=False,
        on="validation_end",
    )


# ---------------------------------------------------------------------------
# Fold cleanup
# ---------------------------------------------------------------------------

def _cleanup_fold(trainer, module, dm):
    """Aggressively free CUDA memory and Pyro state between folds/trials.

    Shuts down ThreadedPrefetcher daemon threads (which hold GPU-resident
    batch tensors in their closures, ~8-10 GB per fold) via the DataModule's
    explicit tracking, then frees all references.
    """
    dm.shutdown_prefetchers()
    del trainer, module, dm
    pyro.clear_param_store()
    for _ in range(3):
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # Debug: warn if CUDA tensors survived cleanup (reference cycle detector)
        post_cleanup = torch.cuda.memory_allocated()
        if post_cleanup > 500 * 1024 * 1024:  # > 500 MB remaining = likely leak
            logger.warning(
                "VRAM leak detected: %.2f GB still allocated after fold cleanup. "
                "Enable torch.cuda.memory._record_memory_history() to diagnose.",
                post_cleanup / 1024**3,
            )


# ---------------------------------------------------------------------------
# Annealing shortening
# ---------------------------------------------------------------------------

def shorten_annealing_for_hpo(
    config: DictConfig,
    full_max_epochs: int = 100,
) -> DictConfig:
    """Proportionally shorten annealing schedules for HPO trials.

    When HPO uses fewer max_epochs than full training, temperature annealing
    and KL annealing schedules must be shortened proportionally so the model
    still completes its full annealing cycle within the trial budget.

    Also updates early_stopping.min_epochs to match the shortened schedule
    (warmup + anneal), so early stopping can fire after annealing completes.

    Args:
        config: Trial config (modified in-place and returned).
        full_max_epochs: The max_epochs used in full training (for computing ratio).

    Returns:
        The same config with shortened annealing schedules.
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_all_subject_ids(splits: dict) -> list[str]:
    """Return deduplicated list of all subject IDs across all splits."""
    ids: set[str] = set()
    ids.update(splits.get("holdout_test", []))
    ids.update(splits.get("train_val_pool", []))
    for fold in splits.get("folds", []):
        ids.update(fold.get("train", []))
        ids.update(fold.get("val", []))
    return sorted(ids)


# ---------------------------------------------------------------------------
# Per-trial training function
# ---------------------------------------------------------------------------

def train_fn(ray_config: dict, base_config: dict, splits: dict, metadata, preloaded_cache, n_folds: int):
    """Per-trial training function invoked by Ray Tune.

    Trains on N folds, reports per-epoch val_nll to HyperBand via
    ``TuneReportCheckpointCallback``, and reports mean val_nll at the end.

    Data is passed via tune.with_parameters (uses ray.put internally).
    With RAY_ENABLE_ZERO_COPY_TORCH_TENSORS=1, tensors in the cache are
    backed by shared object store memory — no per-worker copy.

    Args:
        ray_config: Dict of sampled hyperparameters from Ray Tune.
        base_config: Base experiment config (plain dict, reconstructed to DictConfig).
        splits: CV splits dict.
        metadata: Metadata DataFrame.
        preloaded_cache: Pre-loaded subject cache dict (or None).
        n_folds: Number of CV folds per trial.
    """
    from ray import tune
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger

    from scripts.train import setup_callbacks
    from src.data.datamodule import CognitiveResilienceDataModule
    from src.training.callbacks import GradientNormLogger, ResilienceModelCheckpoint
    from src.training.lightning_module import CognitiveResilienceLightningModule
    from src.utils.reproducibility import set_seed

    # Reconstruct DictConfig from plain dict (serialized by tune.with_parameters)
    base_config = OmegaConf.create(base_config)

    # Worker-level setup (each trial runs in a fresh process)
    torch.set_float32_matmul_precision("high")
    pyro.settings.set(module_local_params=True)

    # Build trial config from Ray-sampled HPs
    config = build_config_from_ray(ray_config, base_config)
    if config is None:
        # Invalid HP combination (e.g., d_embed % n_heads != 0)
        tune.report({"val_nll": float("inf")})
        return

    # Override n_folds in config
    OmegaConf.update(config, "data.splits.n_folds", n_folds)

    # Annealing schedule: when tau_min and anneal_epochs are HPO-controlled
    # (present in search space), skip proportional shortening — the optimizer
    # sets them directly. Update min_epochs to match the sampled schedule so
    # early stopping can't fire before annealing completes.
    search_cfg = base_config.get("hpo", {})
    search_space = search_cfg.get("search_space", {})
    if "tau_min" in search_space or "anneal_epochs" in search_space:
        ta = config.training.temperature_annealing
        # When tau_min == tau_max, annealing is disabled — no min_epochs constraint
        if ta.tau_min >= ta.tau_max:
            OmegaConf.update(config, "training.early_stopping.min_epochs", 1)
        else:
            new_min = ta.warmup_epochs + ta.anneal_epochs
            OmegaConf.update(config, "training.early_stopping.min_epochs", new_min)
    else:
        config = shorten_annealing_for_hpo(config, full_max_epochs=100)

    seed = config.experiment.get("seed", 42)
    # HPO overrides: disable deterministic algorithms and enable cuDNN autotuner.
    # Reproducibility is already broken by scatter_add in HGT and FlashAttention
    # in SetTransformer. Deterministic mode forces slower cuBLAS kernels for no benefit.
    set_seed(seed, deterministic=False, benchmark=True)

    # Data is already available — passed via tune.with_parameters
    # (ray.put/ray.get handled internally by Ray)

    # Callback types to exclude for HPO trials:
    # - ModelCheckpoint / ResilienceModelCheckpoint: trials don't save checkpoints
    # - LearningRateMonitor: logged via CSVLogger instead
    # - GradientNormLogger: unnecessary overhead per trial
    _EXCLUDED_TRIAL_CALLBACKS = (
        ModelCheckpoint, ResilienceModelCheckpoint,
        LearningRateMonitor, GradientNormLogger,
    )

    max_epochs = config.training.max_epochs
    fold_val_losses = []

    for fold_idx in range(n_folds):
        # Re-seed per fold for independent reproducibility
        set_seed(seed + fold_idx, deterministic=False, benchmark=True)

        # Build model
        module = CognitiveResilienceLightningModule(config)

        # Setup callbacks, filtering those inappropriate for trials
        callbacks = [
            cb for cb in setup_callbacks(config)
            if not isinstance(cb, _EXCLUDED_TRIAL_CALLBACKS)
        ]
        # Add Ray Tune reporting callback for per-epoch HyperBand feedback
        callbacks.append(_make_tune_report_callback())

        # CSVLogger for per-epoch val_nll curves (negligible overhead)
        trial_logger = CSVLogger(
            save_dir=str(Path(config.paths.get("logs_dir", "outputs/logs")) / "hpo_trials"),
            name="ray_trial",
            version=f"fold_{fold_idx}",
        )

        # Trainer: strategy="auto" overrides config's "ddp" — trials run single-GPU.
        # GPU pinning is handled by Ray (no manual CUDA_VISIBLE_DEVICES).
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            min_epochs=config.training.early_stopping.get("min_epochs", 1),
            accelerator="auto",
            devices="auto",
            strategy="auto",
            precision=config.training.get("precision", "32-true"),
            gradient_clip_val=config.training.get("gradient_clip_val", None),
            gradient_clip_algorithm="norm",
            callbacks=callbacks,
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
            logger=trial_logger,
            log_every_n_steps=config.training.get("logging", {}).get("log_every_n_steps", 10),
            val_check_interval=config.training.get("logging", {}).get("val_check_interval", 1.0),
            deterministic=False,
            benchmark=True,
        )

        # Data module for this fold
        dm = CognitiveResilienceDataModule(
            config=config,
            metadata=metadata,
            splits=splits,
            fold_idx=fold_idx,
            adata=None,  # PrecomputedDataset path — no AnnData needed
            precomputed_dir=config.data.get("precomputed_dir"),
            preloaded_cache=preloaded_cache,
        )

        try:
            trainer.fit(module, datamodule=dm)
        except torch.cuda.OutOfMemoryError:
            logger.error("OOM at fold %d/%d — reporting inf and terminating trial", fold_idx, n_folds)
            torch.cuda.empty_cache()
            _cleanup_fold(trainer, module, dm)
            tune.report({"val_nll": float("inf")})
            return

        # Use val_nll (predictive quality) not val_loss (ELBO) as objective.
        # ELBO = NLL + KL, and KL annealing makes ELBO non-stationary.
        val_nll = trainer.callback_metrics.get("val_nll")
        if val_nll is not None:
            fold_val_losses.append(val_nll.item())

        _cleanup_fold(trainer, module, dm)

    # Report mean val_nll across folds
    mean_val_nll = sum(fold_val_losses) / len(fold_val_losses) if fold_val_losses else float("inf")
    tune.report({"val_nll": mean_val_nll})


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main() -> None:
    """Main HPO entry point using Ray Tune + Ax + HyperBand."""
    parser = argparse.ArgumentParser(
        description="Ray Tune HPO for cognitive resilience model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--n-gpus",
        type=int,
        default=1,
        help="Number of GPUs for parallel trial execution (default: 1)",
    )
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=16,
        help="Number of CPUs for Ray (default: 16). Set explicitly to avoid "
             "Ray auto-detecting all system CPUs on shared servers.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Number of optimization trials (overrides config)",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=1,
        help="Number of CV folds per trial (default: 1). Use 1 for fast HPO, "
             "then retrain best config with full 5-fold CV.",
    )
    parser.add_argument(
        "--precomputed-dir",
        type=str,
        default=None,
        help="Path to pre-built feature directory (overrides config).",
    )
    parser.add_argument(
        "--splits-path",
        type=str,
        default=None,
        help="Path to pre-computed splits JSON file",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing HPO run (uses stored experiment state)",
    )
    parser.add_argument(
        "--warm-start",
        type=str,
        default=None,
        help="Path to Ray results directory from a prior HPO run. "
             "Injects all prior trials into Optuna's TPE as evaluated points, "
             "so the new search starts with knowledge of which HP regions are "
             "good (l(x)) and bad (g(x)). Example: "
             "outputs/ray_results/cognitive_resilience_hpo6/",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Config overrides in dotlist format",
    )

    args = parser.parse_args()

    # Load config
    from src.utils.config import load_config, validate_config

    config = load_config(args.config, overrides=args.overrides)
    required_keys = ["experiment", "data", "model", "training", "paths"]
    if "hpo" in config:
        required_keys.append("hpo")
    validate_config(config, required_keys=required_keys)

    # Override n_folds for HPO (default: 1 fold for fast search)
    OmegaConf.update(config, "data.splits.n_folds", args.n_folds)
    logger.info("HPO using %d CV fold(s) per trial", args.n_folds)

    # Resolve n_trials from CLI or config
    hpo_cfg = config.get("hpo", {})
    n_trials = args.n_trials or hpo_cfg.get("n_trials", 100)

    # Fail fast if data not provided
    if args.splits_path is None:
        raise ValueError(
            "HPO requires pre-computed data splits. "
            "Provide --splits-path /path/to/splits.json."
        )

    # Override precomputed_dir if provided via CLI
    precomputed_dir = args.precomputed_dir or config.data.get("precomputed_dir")
    if precomputed_dir:
        OmegaConf.update(config, "data.precomputed_dir", precomputed_dir)

    # ---- Load data once (orchestrator process) ----

    from src.data.splits import load_splits
    splits = load_splits(args.splits_path)
    logger.info("Loaded splits from %s", args.splits_path)

    import pandas as pd
    metadata_path = Path(config.data.metadata_path)
    metadata_csv = metadata_path / "metadata.csv"
    if not metadata_csv.exists():
        raise FileNotFoundError(
            f"Metadata CSV not found at {metadata_csv}. "
            "Ensure config.data.metadata_path points to a directory containing metadata.csv."
        )
    metadata = pd.read_csv(metadata_csv)

    # Pre-load subject cache for zero-copy sharing via Ray object store.
    # RAY_ENABLE_ZERO_COPY_TORCH_TENSORS=1 makes ray.put() serialize
    # PyTorch tensors as pickle5 out-of-band buffers. ray.get() returns
    # tensors backed by shared object store memory — no per-worker copy.
    preloaded_cache = None
    if precomputed_dir:
        from src.data.datasets import PrecomputedDataset
        all_subject_ids = _collect_all_subject_ids(splits)
        preloaded_cache = PrecomputedDataset.load_subject_cache(
            precomputed_dir, all_subject_ids,
        )
        logger.info("Pre-loaded %d subject tensors from %s", len(all_subject_ids), precomputed_dir)

        # Ensure all tensors are contiguous — Ray's zero-copy serialization
        # requires stride(-1)==1, which fails on expanded/non-contiguous tensors.
        for sid, tensors in preloaded_cache.items():
            preloaded_cache[sid] = {
                k: v.contiguous() if isinstance(v, torch.Tensor) and not v.is_contiguous() else v
                for k, v in tensors.items()
            }

    # ---- Initialize Ray ----

    import os as _os
    _os.environ["RAY_ENABLE_ZERO_COPY_TORCH_TENSORS"] = "1"

    import ray
    from ray import tune
    from ray.tune.schedulers import MedianStoppingRule
    from ray.tune.search.optuna import OptunaSearch

    ray.init(num_gpus=args.n_gpus, num_cpus=args.num_cpus)

    try:
        # Data is passed to workers via tune.with_parameters, which calls
        # ray.put() internally. With RAY_ENABLE_ZERO_COPY_TORCH_TENSORS=1,
        # tensors in the cache are stored as zero-copy shared-memory buffers.

        # ---- Build search space (Ray Tune distributions for Optuna) ----
        search_space = _yaml_to_search_space(config)

        # ---- Search algorithm: Optuna TPE ----
        # Same algorithm as HPO4. Optuna's TPE updates from pruned trials
        # (uses last intermediate value), compatible with MedianStoppingRule.
        #
        # Warm-start: if --warm-start is provided, inject prior trial results
        # into TPE via points_to_evaluate + evaluated_rewards. TPE uses these
        # to build l(x) (good region) and g(x) (bad region) models from trial 1.
        warm_start_kwargs = {}
        if args.warm_start:
            points, rewards = load_warm_start_data(
                args.warm_start, search_space_keys=list(search_space.keys()),
            )
            if points:
                warm_start_kwargs["points_to_evaluate"] = points
                warm_start_kwargs["evaluated_rewards"] = rewards

        optuna_search = OptunaSearch(
            metric="val_nll",
            mode="min",
            seed=config.experiment.get("seed", 42),
            **warm_start_kwargs,
        )

        # ---- Scheduler: MedianStoppingRule ----
        # Equivalent to Optuna's MedianPruner from HPO4.
        # grace_period=10: don't prune before epoch 10 (past warmup)
        # min_samples_required=10: need 10 completed trials before comparing
        scheduler_cfg = hpo_cfg.get("scheduler", {})
        scheduler = MedianStoppingRule(
            time_attr="training_iteration",
            metric="val_nll",
            mode="min",
            grace_period=scheduler_cfg.get("grace_period", 10),
            min_samples_required=scheduler_cfg.get("min_samples_required", 10),
            hard_stop=True,
        )

        # ---- Configure and run Tuner ----
        # Convert DictConfig to plain dict for cloudpickle serialization
        base_config_dict = OmegaConf.to_container(config, resolve=True)
        trainable = tune.with_parameters(
            train_fn,
            base_config=base_config_dict,
            splits=splits,
            metadata=metadata,
            preloaded_cache=preloaded_cache,
            n_folds=args.n_folds,
        )
        trainable = tune.with_resources(trainable, {"gpu": 1})

        exp_name = config.experiment.get("name", "hpo_cognitive_resilience")
        storage = str(Path(config.paths.output_dir).resolve() / "ray_results")

        if args.resume:
            import os
            tuner = tune.Tuner.restore(
                os.path.join(storage, exp_name),
                trainable=trainable,
                resume_errored=True,
            )
            logger.info("Resuming HPO from %s/%s", storage, exp_name)
        else:
            tuner = tune.Tuner(
                trainable,
                param_space=search_space,
                tune_config=tune.TuneConfig(
                    num_samples=n_trials,
                    search_alg=optuna_search,
                    scheduler=scheduler,
                    max_concurrent_trials=args.n_gpus,
                ),
                run_config=tune.RunConfig(
                    name=exp_name,
                    storage_path=storage,
                ),
            )

        results = tuner.fit()

        # ---- Report results ----
        best_result = results.get_best_result(metric="val_nll", mode="min")
        logger.info("Best trial config: %s", best_result.config)
        logger.info("Best val_nll: %.6f", best_result.metrics["val_nll"])

        # Save best config
        best_config = build_config_from_ray(best_result.config, config)
        if best_config is not None:
            output_dir = Path(config.paths.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(best_config, output_dir / "best_config.yaml")
            logger.info("Best config saved to %s", output_dir / "best_config.yaml")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    torch.set_float32_matmul_precision("high")
    main()
