"""
Configuration loading and management utilities.
"""

from pathlib import Path
from typing import Any

import yaml
from omegaconf import OmegaConf, DictConfig
from omegaconf.errors import (
    ConfigKeyError,
    ConfigAttributeError,
    MissingMandatoryValue,
    InterpolationKeyError,
    InterpolationResolutionError,
)


def load_config(
    path: str | Path,
    overrides: dict[str, Any] | list[str] | None = None,
) -> DictConfig:
    """
    Load configuration from YAML file.

    Supports:
    - YAML with OmegaConf interpolation
    - Optional runtime overrides (dict or dotlist format)

    Args:
        path: Path to YAML configuration file
        overrides: Optional overrides — either a dict of values or a list
            of dotlist strings (e.g., ["training.max_epochs=50"])

    Returns:
        OmegaConf DictConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)
        if raw is None:
            raw = {}
        config = OmegaConf.create(raw)

    if overrides:
        if isinstance(overrides, list):
            override_config = OmegaConf.from_dotlist(overrides)
        else:
            override_config = OmegaConf.create(overrides)
        config = OmegaConf.merge(config, override_config)

    return config


def save_config(config: DictConfig | dict, path: str | Path) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration to save (DictConfig or dict)
        path: Output path for YAML file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict if OmegaConf
    if isinstance(config, DictConfig):
        config_dict = OmegaConf.to_container(config, resolve=True)
    else:
        config_dict = config

    with open(path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)



def validate_config(config: DictConfig, required_keys: list[str]) -> None:
    """
    Validate configuration structure, types, ranges, and enumerations.

    Level 3 validation:
    - Required key presence
    - Type checks for critical fields
    - Range checks for numeric parameters
    - Enum checks for categorical parameters

    Args:
        config: Configuration to validate
        required_keys: List of required top-level keys

    Raises:
        ValueError: If validation fails
    """
    errors = []

    # Level 1: Required keys
    missing = [key for key in required_keys if key not in config]
    if missing:
        errors.append(f"Missing required keys: {missing}")

    # Level 2+3: Type, range, enum checks for known fields
    _FIELD_RULES = {
        "training.max_epochs": (int, lambda v: v > 0),
        "training.optimizer.lr": ((int, float), lambda v: v > 0),
        "training.optimizer.weight_decay": ((int, float), lambda v: v >= 0),
        # gradient_clip_val: when present, must be a positive number.
        # To disable clipping, omit the key entirely (Lightning defaults to None).
        # Setting it to null in YAML will fail validation — use omission instead.
        "training.gradient_clip_val": ((int, float), lambda v: v > 0),
        "training.loss.type": (str, lambda v: v in ("beta_nll", "mse")),
        "training.loss.beta": ((int, float), lambda v: 0 <= v <= 1),
        "training.precision": (str, lambda v: v in ("32-true", "16-mixed", "bf16-mixed")),
        "model.dropout": ((int, float), lambda v: 0 <= v < 1),
        "model.d_embed": (int, lambda v: v > 0),
        "model.d_fused": (int, lambda v: v > 0),
        "model.n_genes": (int, lambda v: v > 0),
        "model.n_cell_types": (int, lambda v: v > 0),
        "model.head.type": (str, lambda v: v in ("bayesian", "deterministic")),
        "model.head.d_hidden": (int, lambda v: v > 0),
        "model.hgt.n_layers": (int, lambda v: v > 0),
        "model.hgt.n_heads": (int, lambda v: v > 0),
        "data.dataloader.batch_size": (int, lambda v: v > 0),
        "data.dataloader.num_workers": (int, lambda v: v >= 0),
        # Logging
        "training.logging.log_every_n_steps": (int, lambda v: v > 0),
        # Scheduler
        "training.scheduler.type": (str, lambda v: v in ("cosine",)),
        "training.scheduler.warmup_epochs": (int, lambda v: v >= 0),
        "training.scheduler.eta_min": ((int, float), lambda v: v >= 0),
        # Optimizer type
        "training.optimizer.type": (str, lambda v: v in ("adamw", "adam")),
        # Per-step LR decay for Bayesian SVI (ignored in deterministic mode)
        "training.optimizer.lrd": ((int, float), lambda v: 0 < v <= 1),
        # Sampling
        "data.cell_sampling.sampling_strategy": (str, lambda v: v in ("random", "stratified", "importance")),
        "data.cell_sampling.max_cells_per_type": (int, lambda v: v > 0),
        "data.cell_sampling.min_cells_threshold": (int, lambda v: v >= 0),
        # Splits
        "data.splits.test_frac": ((int, float), lambda v: 0 < v < 1),
        "data.splits.n_folds": (int, lambda v: v >= 2),
        # Temperature annealing
        "training.temperature_annealing.tau_max": ((int, float), lambda v: v > 0),
        "training.temperature_annealing.tau_min": ((int, float), lambda v: v > 0),
        "training.temperature_annealing.schedule": (str, lambda v: v in ("exponential", "linear", "cosine")),
        # Inference
        "inference.num_posterior_samples": (int, lambda v: v > 0),
        "inference.batch_size": (int, lambda v: v > 0),
        # Visualization
        "visualization.format": (str, lambda v: v in ("png", "svg", "pdf")),
    }

    for dotpath, (expected_type, validator) in _FIELD_RULES.items():
        keys = dotpath.split(".")
        value = config
        try:
            for k in keys:
                value = value[k]
        except (KeyError, TypeError, ConfigKeyError, ConfigAttributeError,
            MissingMandatoryValue, InterpolationKeyError, InterpolationResolutionError):
            continue  # Field not present — only required_keys are mandatory

        # OmegaConf returns int for YAML integers, but check robustly
        if not isinstance(value, expected_type):
            type_name = expected_type.__name__ if isinstance(expected_type, type) else str(expected_type)
            errors.append(
                f"{dotpath}: expected {type_name}, got {type(value).__name__} ({value!r})"
            )
        elif not validator(value):
            errors.append(f"{dotpath}: invalid value {value!r}")

    # Cross-field validation
    # 1. tau_min < tau_max
    try:
        tau_max = config.training.temperature_annealing.tau_max
        tau_min = config.training.temperature_annealing.tau_min
        if tau_min >= tau_max:
            errors.append(
                f"Temperature annealing: tau_min ({tau_min}) must be < tau_max ({tau_max})"
            )
    except (KeyError, TypeError, ConfigKeyError, ConfigAttributeError,
            MissingMandatoryValue, InterpolationKeyError, InterpolationResolutionError):
        pass

    # 2. LR > eta_min
    try:
        lr = config.training.optimizer.lr
        eta_min = config.training.scheduler.eta_min
        if eta_min >= lr:
            errors.append(
                f"Scheduler eta_min ({eta_min}) must be < optimizer lr ({lr})"
            )
    except (KeyError, TypeError, ConfigKeyError, ConfigAttributeError,
            MissingMandatoryValue, InterpolationKeyError, InterpolationResolutionError):
        pass

    # 3. Bayesian SVI uses Adam + ExponentialLR (Pyro convention).
    #    Warn if user configured non-default scheduler/optimizer, since those
    #    settings are silently ignored in Bayesian mode.
    try:
        head_type = config.model.head.type
        if head_type == "bayesian":
            import logging as _logging
            _cfg_logger = _logging.getLogger(__name__)

            opt_type = config.training.optimizer.type
            if opt_type != "adamw":
                _cfg_logger.warning(
                    "Bayesian SVI ignores training.optimizer.type='%s'. "
                    "SVI uses Adam (Pyro convention). "
                    "See: https://pyro.ai/examples/svi_part_iv.html",
                    opt_type,
                )

            sched_type = config.training.scheduler.type
            warmup = config.training.scheduler.get("warmup_epochs", 0)
            if sched_type != "cosine" or warmup > 0:
                _cfg_logger.warning(
                    "Bayesian SVI ignores training.scheduler (type='%s', warmup_epochs=%d). "
                    "SVI uses per-step ExponentialLR via training.optimizer.lrd. "
                    "Configure lrd (0 < lrd <= 1) to control LR decay in Bayesian mode.",
                    sched_type, warmup,
                )

            wd = config.training.optimizer.get("weight_decay", 0)
            if wd > 0:
                _cfg_logger.warning(
                    "Bayesian SVI ignores training.optimizer.weight_decay=%.4f. "
                    "Pyro's Adam does not support decoupled weight decay (AdamW). "
                    "Set weight_decay=0 to silence this warning.",
                    wd,
                )
    except (KeyError, TypeError, ConfigKeyError, ConfigAttributeError,
            MissingMandatoryValue, InterpolationKeyError, InterpolationResolutionError):
        pass

    # 4. n_pathology_features must match len(pathology_columns)
    try:
        n_path_cols = len(config.data.pathology_columns)
        n_path_model = config.model.pathology_attention.n_pathology_features
        if n_path_cols != n_path_model:
            errors.append(
                f"data.pathology_columns has {n_path_cols} entries but "
                f"model.pathology_attention.n_pathology_features={n_path_model}. "
                f"These must match."
            )
    except (KeyError, TypeError, ConfigKeyError, ConfigAttributeError,
            MissingMandatoryValue, InterpolationKeyError, InterpolationResolutionError):
        pass

    # 5. n_regions must match dataset constant if specified
    try:
        from src.data.constants import N_REGIONS
        cfg_n_regions = config.model.n_regions
        if cfg_n_regions != N_REGIONS:
            errors.append(
                f"model.n_regions={cfg_n_regions} but data constant N_REGIONS={N_REGIONS}. "
                f"Region count is fixed by dataset schema — remove n_regions from config."
            )
    except (KeyError, TypeError, ConfigKeyError, ConfigAttributeError,
            MissingMandatoryValue, InterpolationKeyError, InterpolationResolutionError):
        pass  # n_regions not in config — correct behavior

    # 6. DDP strategy must use find_unused_parameters=True for HGT edge-specific params
    try:
        strategy = config.training.strategy
        if isinstance(strategy, str) and "ddp" in strategy and "find_unused" not in strategy:
            import logging as _logging
            _cfg_logger = _logging.getLogger(__name__)
            _cfg_logger.warning(
                "training.strategy='%s' with HGT model requires find_unused_parameters=True "
                "because edge-type-specific parameters may be unused in some batches. "
                "Use 'ddp_find_unused_parameters_true' instead.",
                strategy,
            )
    except (KeyError, TypeError, ConfigKeyError, ConfigAttributeError,
            MissingMandatoryValue, InterpolationKeyError, InterpolationResolutionError):
        pass

    # 7. n_cell_types must match dataset constant if specified
    try:
        from src.data.constants import N_CELL_TYPES
        cfg_n_ct = config.model.n_cell_types
        if cfg_n_ct != N_CELL_TYPES:
            errors.append(
                f"model.n_cell_types={cfg_n_ct} but data constant N_CELL_TYPES={N_CELL_TYPES}. "
                f"Cell type count is fixed by dataset schema."
            )
    except (KeyError, TypeError, ConfigKeyError, ConfigAttributeError,
            MissingMandatoryValue, InterpolationKeyError, InterpolationResolutionError):
        pass

    if errors:
        raise ValueError(
            "Configuration validation failed:\n" +
            "\n".join(f"  - {e}" for e in errors)
        )


