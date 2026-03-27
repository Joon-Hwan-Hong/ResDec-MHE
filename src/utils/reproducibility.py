"""
Reproducibility utilities for deterministic experiments.

Reproducibility guarantee levels:
- CPU: bit-reproducible given same seed, config, and data order.
- CUDA: statistically reproducible (same convergence, not bit-for-bit).
  Non-deterministic operations: scatter_add in HGT message passing (atomicAdd),
  scatter_reduce_ in HGT softmax (amax), and nn.MultiheadAttention with
  dropout via FlashAttention kernel. warn_only=True permits all three.
- Checkpoint resume: converges to same quality but exact gradient trajectory
  differs from uninterrupted run. See ResilienceModelCheckpoint.on_load_checkpoint()
  for limitations (DataLoader position, CellSampler per-worker RNG).
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

    # Pyro (Bayesian inference) — separate RNG from PyTorch
    try:
        import pyro
        pyro.set_rng_seed(seed)
    except ImportError:
        pass

    # Deterministic operations.
    # warn_only=True because scatter_add (used in HGT message passing) has no
    # deterministic CUDA implementation. Setting warn_only=False would error on
    # every HGT forward pass. On CPU, scatter ops are deterministic.
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

    Note: Pyro state is NOT captured separately because Pyro delegates
    entirely to PyTorch/Python/NumPy RNG (confirmed on Pyro 1.9.1:
    pyro.set_rng_seed calls torch.manual_seed + random.seed + np.random.seed).
    Capturing torch + python + numpy state covers Pyro completely.

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
