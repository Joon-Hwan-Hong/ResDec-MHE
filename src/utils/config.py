"""
Configuration loading and management utilities.
"""

from pathlib import Path
from typing import Any

import yaml
from omegaconf import OmegaConf, DictConfig
from omegaconf.errors import ConfigKeyError, ConfigAttributeError


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
        config = OmegaConf.create(yaml.safe_load(f))

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


def merge_configs(*configs: DictConfig | dict) -> DictConfig:
    """
    Merge multiple configurations (later configs override earlier).

    Args:
        *configs: Configuration dictionaries to merge

    Returns:
        Merged configuration
    """
    result = OmegaConf.create({})
    for config in configs:
        if isinstance(config, dict):
            config = OmegaConf.create(config)
        result = OmegaConf.merge(result, config)
    return result


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
        "training.gradient_clip_val": ((int, float), lambda v: v > 0),
        "training.loss.type": (str, lambda v: v in ("beta_nll", "mse", "huber")),
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
    }

    for dotpath, (expected_type, validator) in _FIELD_RULES.items():
        keys = dotpath.split(".")
        value = config
        try:
            for k in keys:
                value = value[k]
        except (KeyError, TypeError, ConfigKeyError, ConfigAttributeError):
            continue  # Field not present — only required_keys are mandatory

        # OmegaConf returns int for YAML integers, but check robustly
        if not isinstance(value, expected_type):
            type_name = expected_type.__name__ if isinstance(expected_type, type) else str(expected_type)
            errors.append(
                f"{dotpath}: expected {type_name}, got {type(value).__name__} ({value!r})"
            )
        elif not validator(value):
            errors.append(f"{dotpath}: invalid value {value!r}")

    if errors:
        raise ValueError(
            "Configuration validation failed:\n" +
            "\n".join(f"  - {e}" for e in errors)
        )


def flatten_config(config: DictConfig | dict, parent_key: str = "", sep: str = ".") -> dict:
    """
    Flatten nested configuration to dot-notation keys.

    Args:
        config: Configuration to flatten
        parent_key: Parent key prefix
        sep: Separator for nested keys

    Returns:
        Flattened dictionary with dot-notation keys

    Example:
        {"model": {"d_embed": 128}} -> {"model.d_embed": 128}
    """
    items = []
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)

    for k, v in config.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))

    return dict(items)