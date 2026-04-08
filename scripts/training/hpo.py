"""
Ray Tune + Optuna TPE hyperparameter optimization for cognitive resilience model.

- Per-trial process isolation via Ray Tune (fresh GPU context per trial)
- Optuna TPE search + MedianStoppingRule cross-trial pruning
- Object store data sharing (zero-copy via ray.put)

Usage:
    uv run python scripts/training/hpo.py --config configs/hpo_round6.yaml \\
        --splits-path data/splits.json --precomputed-dir data/precomputed/ \\
        --n-gpus 2 --n-trials 120 --n-folds 2
"""

import argparse
import gc
import logging
from pathlib import Path

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

def load_warm_start_data(
    ray_dir: str,
    search_space_keys: list[str],
    defaults: dict | None = None,
) -> tuple[list[dict], list[float]]:
    """Load prior HPO trials as warm-start data for OptunaSearch.

    Parses result.json files from the most recent experiment in a Ray Tune
    directory and extracts (config, best_val_nll) pairs. Only HPs present
    in the current search space are included in each point. Trials from
    older experiment runs in the same directory are filtered out.

    Args:
        ray_dir: Path to prior Ray Tune experiment directory.
        search_space_keys: List of HP names in the current search space.
        defaults: Optional mapping from HP name to default value. When a trial
            config is missing a key in ``search_space_keys``, the default is
            used instead of dropping the trial. Lets warm-start survive when
            the search space is expanded with new dimensions (e.g., adding
            ``d_embed`` to a search where prior trials had it hardcoded).

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

    # Normalize defaults: None becomes empty dict (no fallbacks). Use explicit
    # `is not None` check rather than `or {}` so that accidentally-falsy values
    # (e.g., False, []) raise a TypeError on the first key lookup instead of
    # being silently coerced.
    defaults_dict = defaults if defaults is not None else {}

    points = []
    rewards = []
    # Track diagnostics for logging at the end (see end-of-function logger.warning)
    dropped_count = 0
    dropped_missing_keys: set[str] = set()

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

        # Use the LAST result line's val_nll (the final cross-fold mean),
        # not the per-epoch minimum. Per-epoch reports may be single-fold
        # values (old sequential code) or fold-averaged (interleaved code);
        # the last line is always the authoritative final metric.
        final_nll = float("inf")
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
                final_nll = record.get("val_nll", final_nll)
                if not config and "config" in record:
                    config = record["config"]

        if not config or final_nll == float("inf"):
            continue

        # Build the point by looking up each search-space key in the trial
        # config, falling back to ``defaults_dict`` if missing. This lets
        # warm-start survive expansion of the search space (e.g., adding
        # d_embed to a search where prior trials had it hardcoded).
        point = {}
        for k in search_space_keys:
            if k in config:
                point[k] = config[k]
            elif k in defaults_dict:
                point[k] = defaults_dict[k]
            else:
                # Key has no prior value and no default — drop this trial
                dropped_count += 1
                dropped_missing_keys.add(k)
                break
        else:
            # for...else: ran when the for-loop completed WITHOUT a break,
            # i.e., every key resolved to a value. Keep the trial.
            points.append(point)
            rewards.append(final_nll)

    logger.info(
        "Loaded %d warm-start trials from %s (best val_nll=%.4f, worst=%.4f)",
        len(points), ray_dir,
        min(rewards) if rewards else float("nan"),
        max(rewards) if rewards else float("nan"),
    )
    if dropped_count > 0:
        logger.warning(
            "Dropped %d warm-start trial(s) due to missing search-space keys "
            "with no defaults: %s. Add these to the `defaults` kwarg to "
            "preserve the trials.",
            dropped_count, sorted(dropped_missing_keys),
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
        "gene_gate_l1": "training.regularization.gene_gate_l1",
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

    Trains N folds interleaved epoch-by-epoch using a manual training loop
    with fold-swap: only 1 model lives on GPU at a time, K-1 fold states
    are kept in CPU RAM as state_dicts. Each epoch trains all folds for one
    epoch, then reports the mean val_nll to Ray. This gives
    MedianStoppingRule clean, comparable metrics without fold-boundary spikes.

    Early stopping: if mean val_nll across folds doesn't improve for
    ``patience`` epochs after ``min_epochs``, the trial stops.

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
    from contextlib import nullcontext

    from ray import tune

    from src.data.datamodule import CognitiveResilienceDataModule
    from src.training.callbacks import TemperatureAnnealing, KLAnnealingCallback
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
        tune.report({"val_nll": float("inf")})
        return

    OmegaConf.update(config, "data.splits.n_folds", n_folds)

    # Annealing schedule adjustment
    search_cfg = base_config.get("hpo", {})
    search_space = search_cfg.get("search_space", {})
    if "tau_min" in search_space or "anneal_epochs" in search_space:
        ta = config.training.temperature_annealing
        if ta.tau_min >= ta.tau_max:
            OmegaConf.update(config, "training.early_stopping.min_epochs", 1)
        else:
            new_min = ta.warmup_epochs + ta.anneal_epochs
            OmegaConf.update(config, "training.early_stopping.min_epochs", new_min)
    else:
        config = shorten_annealing_for_hpo(config, full_max_epochs=100)

    seed = config.experiment.get("seed", 42)
    set_seed(seed, deterministic=False, benchmark=True)

    # --- Config extraction ---
    train_cfg = config.training
    max_epochs = train_cfg.max_epochs
    min_epochs = train_cfg.early_stopping.get("min_epochs", 1)
    es_patience = train_cfg.early_stopping.get("patience", 15)
    es_min_delta = train_cfg.early_stopping.get("min_delta", 0.0001)
    grad_clip_val = train_cfg.get("gradient_clip_val", None)
    use_bf16 = train_cfg.get("precision", "32-true") == "bf16-mixed"

    # Temperature annealing (pure function — no Trainer needed)
    ta_cfg = train_cfg.temperature_annealing
    temp_annealer = TemperatureAnnealing(
        tau_max=ta_cfg.tau_max, tau_min=ta_cfg.tau_min,
        warmup_epochs=ta_cfg.warmup_epochs, anneal_epochs=ta_cfg.anneal_epochs,
        schedule=ta_cfg.schedule,
    )

    # KL annealing
    kl_cfg = train_cfg.get("kl_annealing", {})
    kl_annealer = None
    if kl_cfg.get("enabled", False):
        kl_annealer = KLAnnealingCallback(
            alpha_min=kl_cfg.get("alpha_min", 0.01),
            warmup_epochs=kl_cfg.get("warmup_epochs", 5),
        )

    # --- Helper: build optimizer + scheduler for a module ---
    def _build_optimizer(module):
        opt_cfg = train_cfg.optimizer
        lr = opt_cfg.lr
        wd = opt_cfg.get("weight_decay", 0)
        betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))
        # Split decay/no-decay (biases, LayerNorm, gate_logits excluded)
        split_fn = CognitiveResilienceLightningModule._split_weight_decay_params

        if module._use_bayesian_svi:
            guide_lr = opt_cfg.get("guide_lr", lr)
            model_decay, model_no_decay = split_fn(module.model)
            optimizer = torch.optim.Adam([
                {"params": model_decay, "lr": lr, "weight_decay": wd},
                {"params": model_no_decay, "lr": lr, "weight_decay": 0.0},
                {"params": list(module.guide.parameters()), "lr": guide_lr, "weight_decay": 0.0},
            ], betas=betas)
            lrd = opt_cfg.get("lrd", 1.0)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lrd)
            sched_interval = "step"
        else:
            decay_params, no_decay_params = split_fn(module)
            param_groups = [
                {"params": decay_params, "weight_decay": wd},
                {"params": no_decay_params, "weight_decay": 0.0},
            ]
            if opt_cfg.type == "adamw":
                optimizer = torch.optim.AdamW(
                    param_groups, lr=lr, betas=betas,
                )
            else:
                optimizer = torch.optim.Adam(
                    param_groups, lr=lr, betas=betas,
                )
            sched_cfg = train_cfg.scheduler
            warmup_epochs = sched_cfg.get("warmup_epochs", 0)
            eta_min = sched_cfg.get("eta_min", 1e-6)
            t_max = sched_cfg.get("T_max", max_epochs - warmup_epochs)
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=t_max, eta_min=eta_min,
            )
            if warmup_epochs > 0:
                warmup = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.01, end_factor=1.0,
                    total_iters=warmup_epochs,
                )
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer, schedulers=[warmup, cosine],
                    milestones=[warmup_epochs],
                )
            else:
                scheduler = cosine
            sched_interval = "epoch"
        return optimizer, scheduler, sched_interval

    # --- Helper: train one epoch ---
    def _train_one_epoch(module, optimizer, scheduler, sched_interval, train_dl):
        module.train()
        autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()
        n_steps = 0
        for batch in train_dl:
            # Move batch to GPU
            batch = {
                k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx:
                if module._use_bayesian_svi:
                    loss = module._svi_forward(batch)
                    if module._gene_gate_l1_lambda > 0:
                        loss = loss + module._gene_gate_l1_penalty()
                else:
                    output = module._forward_batch(batch)
                    loss = module._compute_loss(output, batch["cognition"])

            if torch.isnan(loss):
                continue  # skip NaN batches

            loss.backward()
            if grad_clip_val is not None:
                torch.nn.utils.clip_grad_norm_(module.parameters(), grad_clip_val)
            optimizer.step()
            n_steps += 1
            if sched_interval == "step":
                scheduler.step()

        if sched_interval == "epoch":
            scheduler.step()

    # --- Helper: validate one epoch, return val_nll and metrics dict ---
    def _validate(module, val_dl):
        module.eval()
        all_means = []
        all_targets = []
        all_nlls = []
        autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()

        with torch.no_grad():
            for batch in val_dl:
                batch = {
                    k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                with autocast_ctx:
                    if module._use_bayesian_svi:
                        output = module._forward_batch_posterior(batch)
                    else:
                        output = module._forward_batch(batch)
                    nll = module._compute_loss(output, batch["cognition"])

                all_nlls.append(nll.item() * batch["cognition"].shape[0])
                all_means.append(output["mean"].cpu())
                all_targets.append(batch["cognition"].cpu())

        # Weighted mean NLL
        total_samples = sum(t.shape[0] for t in all_targets)
        val_nll = sum(all_nlls) / total_samples if total_samples > 0 else float("inf")

        # Correlation metrics
        metrics = {"val_nll": val_nll}
        if all_means:
            means_cat = torch.cat(all_means, dim=0)
            targets_cat = torch.cat(all_targets, dim=0)
            result = module.metrics.compute(means_cat, None, targets_cat)
            for name, value in result.items():
                metrics[f"val_{name}"] = float(value)

        return metrics

    # --- Setup K fold modules (CPU-resident) + dataloaders ---
    fold_modules = []  # CPU-resident nn.Modules (swapped to GPU per fold per epoch)
    fold_optimizers = []  # CPU-resident optimizers
    fold_schedulers = []  # schedulers
    fold_sched_intervals = []
    fold_dataloaders = []  # (train_dl, val_dl, dm) per fold

    for fold_idx in range(n_folds):
        set_seed(seed + fold_idx, deterministic=False, benchmark=True)

        module = CognitiveResilienceLightningModule(config)

        dm = CognitiveResilienceDataModule(
            config=config, metadata=metadata, splits=splits,
            fold_idx=fold_idx, adata=None,
            precomputed_dir=config.data.get("precomputed_dir"),
            preloaded_cache=preloaded_cache,
        )
        dm.setup("fit")

        if module._use_bayesian_svi:
            module._prototype_guide_if_needed(caller="hpo_train_fn")
            module.elbo.n_train = len(dm.train_dataset)

        optimizer, scheduler, sched_interval = _build_optimizer(module)

        # Module stays on CPU — will be swapped to GPU during training
        fold_modules.append(module)
        fold_optimizers.append(optimizer)
        fold_schedulers.append(scheduler)
        fold_sched_intervals.append(sched_interval)

        train_dl = dm.train_dataloader()
        val_dl = dm.val_dataloader()
        fold_dataloaders.append((train_dl, val_dl, dm))

    # --- Interleaved epoch loop with fold-swap ---
    fold_last_nll = [float("inf")] * n_folds
    best_mean_nll = float("inf")
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        epoch_fold_nlls = []
        epoch_fold_metrics = [{} for _ in range(n_folds)]

        for fold_idx in range(n_folds):
            set_seed(seed + fold_idx + epoch * 1000, deterministic=False, benchmark=True)

            module = fold_modules[fold_idx]
            optimizer = fold_optimizers[fold_idx]
            scheduler = fold_schedulers[fold_idx]
            sched_interval = fold_sched_intervals[fold_idx]

            # --- Swap fold to GPU ---
            module.cuda()

            # Apply temperature annealing
            tau = temp_annealer.get_temperature(epoch)
            for gate in [
                getattr(module.model, "hgt_gene_gate", None),
                getattr(module.model.cell_transformer, "gene_gate", None)
                if hasattr(module.model, "cell_transformer") else None,
            ]:
                if gate is not None:
                    gate.temperature = tau

            # Apply KL annealing
            if kl_annealer is not None and hasattr(module, "elbo"):
                kl_weight = kl_annealer.get_kl_weight(epoch)
                if hasattr(module.elbo, "kl_weight"):
                    module.elbo.kl_weight = kl_weight

            # --- Train + validate one epoch ---
            train_dl, val_dl, _ = fold_dataloaders[fold_idx]

            try:
                _train_one_epoch(module, optimizer, scheduler, sched_interval, train_dl)
                metrics = _validate(module, val_dl)
            except torch.cuda.OutOfMemoryError:
                logger.error("OOM at fold %d epoch %d", fold_idx, epoch)
                torch.cuda.empty_cache()
                epoch_fold_nlls.append(float("inf"))
                fold_last_nll[fold_idx] = float("inf")
                module.cpu()
                gc.collect()
                torch.cuda.empty_cache()
                continue

            val_nll = metrics.get("val_nll", float("inf"))
            fold_last_nll[fold_idx] = val_nll
            epoch_fold_nlls.append(val_nll)
            epoch_fold_metrics[fold_idx] = metrics

            # --- Swap fold back to CPU ---
            module.cpu()
            pyro.clear_param_store()
            gc.collect()
            torch.cuda.empty_cache()

        # --- Report mean across folds to Ray ---
        mean_nll = sum(epoch_fold_nlls) / len(epoch_fold_nlls)
        report_dict = {
            "val_nll": mean_nll,
            "val_nll_std": float(np.std(epoch_fold_nlls)),
        }
        for metric_name in ("val_r2", "val_pearson_r", "val_spearman_rho"):
            vals = [m.get(metric_name) for m in epoch_fold_metrics]
            vals = [v for v in vals if v is not None and not np.isnan(v)]
            if vals:
                report_dict[metric_name] = float(np.mean(vals))

        tune.report(report_dict)

        # --- Mean-level early stopping ---
        if epoch >= min_epochs:
            if mean_nll < best_mean_nll - es_min_delta:
                best_mean_nll = mean_nll
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= es_patience:
                    logger.info(
                        "Mean val_nll early stopping at epoch %d (patience %d)",
                        epoch, es_patience,
                    )
                    break

    # Cleanup
    for _, _, dm in fold_dataloaders:
        dm.shutdown_prefetchers()
    del fold_modules, fold_optimizers, fold_schedulers, fold_dataloaders
    pyro.clear_param_store()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
        # grace_period=20: don't prune before iter 20. HPO8 top-10 trials by
        #   best val_r2 peaked at iter 10-27 (mean ~17). The previous
        #   grace_period=10 risked pruning trials before they reached their
        #   peak — combined with the (now-fixed) scope='last' selection bug,
        #   it doubly penalized fast-then-degrading trials. Iter 20 is past
        #   the peak window for early-converging trials and gives slow ones
        #   one more chance before pruning.
        # min_samples_required=10: need 10 completed trials before comparing
        scheduler_cfg = hpo_cfg.get("scheduler", {})
        scheduler = MedianStoppingRule(
            time_attr="training_iteration",
            metric="val_nll",
            mode="min",
            grace_period=scheduler_cfg.get("grace_period", 20),
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
        # scope='all' picks the trial whose MIN val_nll across all reported
        # iters is smallest, instead of comparing trials by their LAST iter
        # value (default scope='last'). Top trials in HPO8 peaked at iter
        # 10-27 then degraded — scope='last' was selecting trials by their
        # degraded final value, not their actual best epoch.
        best_result = results.get_best_result(metric="val_nll", mode="min", scope="all")
        logger.info("Best trial config: %s", best_result.config)
        # Log both the true min val_nll (selection metric) and last-iter
        # val_nll (which is what best_result.metrics defaults to).
        try:
            best_nll_min = float(best_result.metrics_dataframe["val_nll"].min())
        except Exception:
            best_nll_min = best_result.metrics.get("val_nll", float("nan"))
        logger.info("Best val_nll (min over iters): %.6f", best_nll_min)
        logger.info("Best val_nll (last iter, for reference): %.6f", best_result.metrics.get("val_nll", float("nan")))

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
