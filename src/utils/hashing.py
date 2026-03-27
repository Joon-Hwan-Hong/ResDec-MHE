"""
Experiment hashing utilities for reproducibility tracking.
"""

import hashlib
import json
import re
from datetime import datetime
from typing import Any


def generate_experiment_hash(config: dict[str, Any], length: int = 8) -> str:
    """
    Generate unique experiment hash from configuration.

    Format: YYYYMMDD_HHMMSS_ffffff_{config_hash}
    or:     YYYYMMDD_HHMMSS_ffffff_{name}_{config_hash}
    Example: 20260113_143052_123456_a3f7b2c1
             20260113_143052_123456_baseline-5fold_a3f7b2c1

    If config["experiment"]["run_name"] is set, it is included as a
    human-readable tag between the timestamp and the hash.

    The microsecond precision prevents collisions when identical configs
    are started within the same second.

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

    # Prepend timestamp with microseconds for uniqueness
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    # Optional human-readable tag
    run_name = config.get("experiment", {}).get("run_name")
    if run_name:
        # Sanitize: lowercase, replace spaces/special chars with hyphens
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "-", run_name).strip("-")
        return f"{timestamp}_{safe_name}_{short_hash}"

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
