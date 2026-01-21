"""
Configuration loading and management utilities.
"""

from pathlib import Path
from typing import Any

import yaml
from omegaconf import OmegaConf, DictConfig


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> DictConfig:
    """
    Load configuration from YAML file.

    Supports:
    - YAML with OmegaConf interpolation
    - Optional runtime overrides
    - Configuration validation

    Args:
        path: Path to YAML configuration file
        overrides: Optional dictionary of overrides to apply

    Returns:
        OmegaConf DictConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    # Load base config
    with open(path) as f:
        config = OmegaConf.create(yaml.safe_load(f))

    # Apply overrides if provided
    if overrides:
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
    Validate that configuration contains required keys.

    Args:
        config: Configuration to validate
        required_keys: List of required top-level keys

    Raises:
        ValueError: If any required key is missing
    """
    missing = [key for key in required_keys if key not in config]
    if missing:
        raise ValueError(f"Missing required configuration keys: {missing}")


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