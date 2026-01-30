"""
Device movement utilities for moving batch data between devices.

This module is the canonical home for device-transfer logic, keeping the
dependency graph clean:  utils.device has NO imports from the data layer.

Both ``src.data.collate`` and ``src.utils.gpu`` import from here.
"""

from typing import Any

import torch

# PyG types are optional -- guard the import so the module stays usable
# in environments that do not have torch_geometric installed.
try:
    from torch_geometric.data import Batch, HeteroData
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False


def _move_to_device(value: Any, device: torch.device) -> Any:
    """Recursively move tensors to device, handling nested lists and dicts."""
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    elif _HAS_PYG and isinstance(value, (Batch, HeteroData)):
        return value.to(device)
    elif isinstance(value, dict):
        return {k: _move_to_device(v, device) for k, v in value.items()}
    elif isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    else:
        return value


def move_batch_to_device(
    batch: dict[str, Any],
    device: torch.device | str,
) -> dict[str, Any]:
    """
    Move batch tensors to specified device.

    WARNING: This function is for manual/debugging use only.
    When using PyTorch Lightning, device placement is handled automatically
    by the Trainer. Do NOT call this in LightningModule.training_step().

    Handles nested structures including:
    - torch.Tensor
    - PyG Batch and HeteroData
    - list[dict[str, Tensor]] (edge_*_dict_list from collate_for_hgt)
    - list[dict[tuple, Tensor]] (edge dicts with triplet keys)

    For multi-GPU setups:
    - With DDP: Each process sees only its GPU as device 0
    - The Trainer handles data distribution and device placement
    - Effective batch size = batch_size * num_gpus

    Args:
        batch: Batch dictionary from collate_fn or collate_for_hgt
        device: Target device (e.g., "cuda:0", "cuda:1", torch.device("cuda"))

    Returns:
        Batch with tensors on device
    """
    if isinstance(device, str):
        device = torch.device(device)

    moved = {}

    # Keys that should remain on CPU (non-tensor metadata)
    cpu_keys = {"subject_ids", "batch_size", "n_nodes_per_graph", "node_types",
                "edge_types", "cell_type_order"}

    for key, value in batch.items():
        if key in cpu_keys:
            moved[key] = value
        else:
            moved[key] = _move_to_device(value, device)

    return moved
