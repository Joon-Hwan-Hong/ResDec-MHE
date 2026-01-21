"""
Experiment hashing utilities for reproducibility tracking.
"""

import hashlib
import json
from datetime import datetime
from typing import Any


def generate_experiment_hash(config: dict[str, Any], length: int = 8) -> str:
    """
    Generate unique experiment hash from configuration.

    Format: YYYYMMDD_HHMMSS_{config_hash}
    Example: 20260113_143052_a3f7b2c1

    Args:
        config: Experiment configuration dictionary
        length: Length of the hash suffix (default 8)

    Returns:
        Unique experiment identifier string
    """
    # Serialize config to a deterministic string
    config_str = json.dumps(config, sort_keys=True, default=str)

    # Generate SHA256 hash
    full_hash = hashlib.sha256(config_str.encode()).hexdigest()
    short_hash = full_hash[:length]

    # Prepend timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    return f"{timestamp}_{short_hash}"


def hash_config(config: dict[str, Any]) -> str:
    """
    Generate a deterministic hash of a configuration.

    Useful for comparing if two configs are identical.

    Args:
        config: Configuration dictionary

    Returns:
        Full SHA256 hash string
    """
    config_str = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(config_str.encode()).hexdigest()


def hash_file(filepath: str) -> str:
    """
    Generate SHA256 hash of a file.

    Useful for tracking data file versions.

    Args:
        filepath: Path to file

    Returns:
        SHA256 hash of file contents
    """
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()