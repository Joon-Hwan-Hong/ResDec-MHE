"""
Reproducibility utilities for deterministic experiments.
"""

import random
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True, benchmark: bool = False) -> None:
    """
    Set random seeds for reproducibility across all relevant libraries.

    Args:
        seed: Random seed value
        deterministic: If True, use deterministic algorithms (slower but reproducible)
        benchmark: If True, enable cuDNN autotuner (faster but non-deterministic)
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Deterministic operations
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        # Required for some operations
        import os
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # cuDNN settings
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark


def get_rng_states() -> dict[str, Any]:
    """
    Capture current RNG states for all relevant libraries.

    Returns:
        Dictionary containing RNG states for Python, NumPy, and PyTorch
    """
    states = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        states["cuda"] = torch.cuda.get_rng_state_all()

    return states


def set_rng_states(states: dict[str, Any]) -> None:
    """
    Restore RNG states from a previous capture.

    Args:
        states: Dictionary of RNG states (from get_rng_states)
    """
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch"])

    if "cuda" in states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["cuda"])
