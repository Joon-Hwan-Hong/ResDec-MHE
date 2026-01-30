"""
GPU utilities for memory management and device selection.
"""

import os
from typing import Any

import torch

from src.utils.device import move_batch_to_device


def get_available_gpus() -> list[int]:
    """
    Get list of available GPU device IDs.

    Returns:
        List of GPU device indices
    """
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


def get_gpu_memory_info(device_id: int = 0) -> dict[str, float]:
    """
    Get GPU memory information in GB.

    Args:
        device_id: GPU device ID

    Returns:
        Dictionary with total, allocated, cached, and free memory
    """
    if not torch.cuda.is_available():
        return {"total": 0, "allocated": 0, "cached": 0, "free": 0}

    total = torch.cuda.get_device_properties(device_id).total_memory / 1e9
    allocated = torch.cuda.memory_allocated(device_id) / 1e9
    cached = torch.cuda.memory_reserved(device_id) / 1e9
    free = total - allocated

    return {
        "total": total,
        "allocated": allocated,
        "cached": cached,
        "free": free,
    }


def clear_gpu_memory() -> None:
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def select_device(
    preferred: str = "auto",
    gpu_id: int | None = None,
) -> torch.device:
    """
    Select compute device based on availability and preference.

    Args:
        preferred: Device preference ("auto", "cuda", "cpu")
        gpu_id: Specific GPU ID to use (if preferred="cuda")

    Returns:
        torch.device object
    """
    if preferred == "cpu":
        return torch.device("cpu")

    if preferred == "auto" or preferred == "cuda":
        if torch.cuda.is_available():
            if gpu_id is not None:
                return torch.device(f"cuda:{gpu_id}")
            return torch.device("cuda")

    return torch.device("cpu")


def set_visible_gpus(gpu_ids: list[int] | None) -> None:
    """
    Set which GPUs are visible to PyTorch.

    Must be called before any CUDA operations.

    Args:
        gpu_ids: List of GPU IDs to make visible, or None for all
    """
    if gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_ids)


def estimate_batch_size(
    model: torch.nn.Module,
    sample_input: dict[str, Any],
    target_memory_fraction: float = 0.8,
    device: torch.device | str = "cuda",
) -> int:
    """
    Estimate optimal batch size based on available GPU memory.

    Args:
        model: Model to estimate for
        sample_input: Sample input dictionary (batch size 1). Supports nested
            dict/list structures (e.g., HGT format with x_dict_list).

            IMPORTANT: This dict is unpacked as **kwargs to the model's forward()
            method. Only include keys that match the model's forward signature.
            Do NOT pass raw dataset samples or collated batches directly, as they
            contain metadata keys (subject_id, cell_type_order, etc.) that will
            cause TypeError. Filter to model-relevant keys first.
        target_memory_fraction: Target fraction of GPU memory to use
        device: Device to estimate for

    Returns:
        Estimated optimal batch size
    """
    if not torch.cuda.is_available():
        return 32  # Default for CPU

    device = torch.device(device)
    model = model.to(device)

    # Clear cache and get baseline
    clear_gpu_memory()
    baseline_memory = torch.cuda.memory_allocated(device)

    # Move sample to device (handles nested dict/list structures)
    sample = move_batch_to_device(sample_input, device)

    # Forward pass to measure memory
    model.eval()
    with torch.no_grad():
        _ = model(**sample)

    forward_memory = torch.cuda.memory_allocated(device) - baseline_memory

    # Estimate memory for training (roughly 3x forward for gradients)
    training_memory_per_sample = forward_memory * 3

    # Calculate batch size
    total_memory = torch.cuda.get_device_properties(device).total_memory
    available_memory = total_memory * target_memory_fraction

    estimated_batch_size = int(available_memory / training_memory_per_sample)

    # Clamp to reasonable range
    return max(1, min(estimated_batch_size, 128))
