"""
GPU utilities for memory management.
"""

import os

import torch


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


def set_visible_gpus(gpu_ids: list[int] | None) -> None:
    """
    Set which GPUs are visible to PyTorch.

    Must be called before any CUDA operations.

    Args:
        gpu_ids: List of GPU IDs to make visible, or None for all
    """
    if gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_ids)
