"""
Custom collate functions for batching heterogeneous graph data.

Handles batching of:
- Variable-sized graphs (different number of CCC edges per subject)
- Flat cell-level data (cell_data + cell_offsets)
- Multi-region data (optional)

Collate Format:
    This module outputs **flat concatenated tensors** for HGT graph data:
    ccc_edge_index [2, E_total], ccc_edge_type [E_total],
    ccc_edge_attr [E_total, 1] with node indices offset per sample.
    These are directly compatible with HGTEncoderTensor. Key functions:
    - collate_for_hgt_multiregion: Primary collate for training (recommended)
    - collate_for_hgt: Single-region variant
    - collate_fn: Basic collate without HGT format

Note on Multi-GPU:
    Device allocation is handled by PyTorch Lightning's Trainer, NOT by manual
    move_batch_to_device(). When using DDP (recommended), Lightning spawns
    separate processes per GPU - each process only sees its assigned GPU as
    device 0. The Trainer handles:
    - Distributing data across GPUs (via DistributedSampler)
    - Moving batches to the correct device
    - Gradient synchronization

    See: https://lightning.ai/docs/pytorch/stable/accelerators/gpu_intermediate.html
"""

import logging
import warnings
from typing import Any

import numpy as np
import torch

from src.data.constants import N_REGIONS, PFC_REGION_IDX

logger = logging.getLogger(__name__)


def _derive_available_regions_from_keys(sample: dict[str, Any]) -> list[int]:
    """Derive available regions from region_{idx}_pseudobulk keys in sample."""
    regions = []
    for key in sample.keys():
        if key.startswith("region_") and key.endswith("_pseudobulk"):
            try:
                idx = int(key[7:-11])  # len("region_") = 7, len("_pseudobulk") = 11
                regions.append(idx)
            except ValueError:
                pass
    return sorted(regions)



def _assemble_region_tensors(
    batch: list[dict[str, Any]],
    batch_size: int,
    n_cell_types: int,
    n_genes: int,
    n_regions: int = N_REGIONS,
    auto_derive_regions: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Assemble region pseudobulk tensors from per-sample data.

    Shared logic for collate_multiregion and collate_for_hgt_multiregion.

    Args:
        batch: List of sample dictionaries
        batch_size: Number of samples
        n_cell_types: Number of cell types per region
        n_genes: Number of genes per cell type
        n_regions: Number of brain regions
        auto_derive_regions: If True, derive available_regions from
            region_{idx}_pseudobulk keys when not explicitly provided,
            and emit a warning. If False, fall back to PFC only.

    Returns:
        region_pseudobulk: [batch_size, n_regions, n_cell_types, n_genes]
        region_mask: [batch_size, n_regions] bool mask
    """
    region_pseudobulk = torch.zeros(batch_size, n_regions, n_cell_types, n_genes)
    region_mask = torch.zeros(batch_size, n_regions, dtype=torch.bool)

    # Pre-compute region key strings to avoid f-string allocation in inner loop
    region_keys = [f"region_{idx}_pseudobulk" for idx in range(n_regions)]

    missing_count = 0

    for i, s in enumerate(batch):
        # Determine available regions for this sample
        if "available_regions" in s:
            available_regions = s["available_regions"]
        elif auto_derive_regions:
            derived_regions = _derive_available_regions_from_keys(s)
            if derived_regions:
                available_regions = derived_regions
                if i == 0:  # Warn once per batch
                    warnings.warn(
                        f"Sample missing 'available_regions' key but has "
                        f"region_*_pseudobulk keys for regions {derived_regions}. "
                        f"Deriving available_regions from keys. Consider adding "
                        f"'available_regions' explicitly to samples.",
                        UserWarning,
                        stacklevel=3,
                    )
            else:
                available_regions = [PFC_REGION_IDX]
        else:
            available_regions = [PFC_REGION_IDX]

        for region_idx in available_regions:
            if region_idx < n_regions:
                region_key = region_keys[region_idx]
                if region_key in s:
                    region_pseudobulk[i, region_idx] = s[region_key]
                    region_mask[i, region_idx] = True
                elif region_idx == PFC_REGION_IDX:
                    # Use main pseudobulk for PFC
                    region_pseudobulk[i, PFC_REGION_IDX] = s["pseudobulk"]
                    region_mask[i, PFC_REGION_IDX] = True
                else:
                    # Non-PFC region key missing — zero-filled, mask stays False
                    missing_count += 1

    if missing_count > 0:
        warnings.warn(
            f"{missing_count} region entries missing across batch "
            f"(zero-filled, region_mask=False). "
            f"This is expected for single-region subjects.",
            UserWarning,
            stacklevel=2,
        )

    return region_pseudobulk, region_mask


def _collate_cell_data(
    batch: list[dict[str, Any]], n_genes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate per-sample cell_data and offset cell_offsets for the batch.

    Shared helper for collate_fn and collate_for_hgt.

    Returns:
        cell_data: [total_cells, n_genes] concatenated cell expressions
        cell_offsets: [batch_size, n_cell_types + 1] batch-adjusted offsets
    """
    all_data: list[torch.Tensor] = []
    batch_offsets: list[torch.Tensor] = []
    cumulative = 0
    for s in batch:
        sample_offsets = s["cell_offsets"]  # [n_types + 1]
        batch_offsets.append(sample_offsets + cumulative)
        cumulative += int(sample_offsets[-1])
        if s["cell_data"].shape[0] > 0:
            all_data.append(s["cell_data"])

    cell_data = torch.cat(all_data) if all_data else torch.empty(0, n_genes)
    cell_offsets = torch.stack(batch_offsets)  # [B, n_types + 1]
    return cell_data, cell_offsets


def _collate_edges(
    batch: list[dict[str, Any]], n_nodes_per_graph: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Concatenate per-sample edge tensors with node index offsets.

    Shared helper for collate_fn and collate_for_hgt.

    Returns:
        ccc_edge_index: [2, E_total] with per-sample node offsets
        ccc_edge_type: [E_total]
        ccc_edge_attr: [E_total, edge_dim]
    """
    all_edge_indices: list[torch.Tensor] = []
    all_edge_types: list[torch.Tensor] = []
    all_edge_attrs: list[torch.Tensor] = []

    for i, s in enumerate(batch):
        ei = s["ccc_edge_index"]  # [2, n_edges_i]
        n_edges = ei.shape[1] if ei.numel() > 0 else 0
        if n_edges > 0:
            all_edge_indices.append(ei + i * n_nodes_per_graph)
            all_edge_types.append(s["ccc_edge_type"])
            all_edge_attrs.append(s["ccc_edge_attr"])

    if all_edge_indices:
        ccc_edge_index = torch.cat(all_edge_indices, dim=1)
        ccc_edge_type = torch.cat(all_edge_types)
        ccc_edge_attr = torch.cat(all_edge_attrs)
    else:
        ccc_edge_index = torch.zeros(2, 0, dtype=torch.long)
        ccc_edge_type = torch.zeros(0, dtype=torch.long)
        ccc_edge_attr = torch.zeros(0, 1)

    return ccc_edge_index, ccc_edge_type, ccc_edge_attr


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Collate function for CognitiveResilienceDataset.

    Handles:
    - Standard tensor stacking (pseudobulk, pathology, cognition)
    - Graph batching for variable-sized CCC graphs
    - Flat cell-level data (cell_data + cell_offsets)
    - Subject IDs as list

    Args:
        batch: List of sample dictionaries from Dataset.__getitem__

    Returns:
        Batched dictionary with:
        - pseudobulk: [batch, n_cell_types, n_genes]
        - cell_type_mask: [batch, n_cell_types]
        - cell_counts: [batch, n_cell_types] number of cells per type
        - pathology: [batch, n_pathology]
        - cognition: [batch, 1]
        - ccc_edge_index: [2, total_edges] batched edge indices (with node offsets)
        - ccc_edge_type: [total_edges] edge type indices
        - ccc_edge_attr: [total_edges, 1] edge attributes
        - graph_batch: [total_nodes] mapping each node to its graph index
        - graph_ptr: [batch+1] pointers to graph boundaries
        - cell_data: [total_cells, n_genes] concatenated cell expressions
        - cell_offsets: [batch, n_cell_types + 1] cumulative offsets
        - region_mask: [batch, n_regions] bool mask for available regions
        - subject_ids: list of strings
        - batch_size: int
        - n_nodes_per_graph: int (31 cell types)
    """
    batch_size = len(batch)

    # ─────────────────────────────────────────────────────────────────────────
    # Pre-allocate and fill in single pass (avoids 6 separate list comprehensions)
    # ─────────────────────────────────────────────────────────────────────────
    s0 = batch[0]
    pseudobulk = torch.empty(batch_size, *s0["pseudobulk"].shape, dtype=torch.float32)
    cell_type_mask = torch.empty(batch_size, *s0["cell_type_mask"].shape, dtype=torch.bool)
    cell_counts = torch.empty(batch_size, *s0["cell_counts"].shape, dtype=torch.long)
    pathology = torch.empty(batch_size, *s0["pathology"].shape, dtype=torch.float32)
    cognition = torch.empty(batch_size, *s0["cognition"].shape, dtype=torch.float32)
    region_mask = torch.empty(batch_size, *s0["region_mask"].shape, dtype=torch.bool)

    for i, s in enumerate(batch):
        pseudobulk[i] = s["pseudobulk"]
        cell_type_mask[i] = s["cell_type_mask"]
        cell_counts[i] = s["cell_counts"]
        pathology[i] = s["pathology"]
        cognition[i] = s["cognition"]
        region_mask[i] = s["region_mask"]

    # ─────────────────────────────────────────────────────────────────────────
    # Cell-level data: flat format (cell_data + cell_offsets)
    # ─────────────────────────────────────────────────────────────────────────
    n_genes_flat = s0["pseudobulk"].shape[1]
    cell_data, cell_offsets = _collate_cell_data(batch, n_genes_flat)

    # ─────────────────────────────────────────────────────────────────────────
    # Graph batching - Manual approach for homogeneous treatment
    # ─────────────────────────────────────────────────────────────────────────
    n_nodes_per_graph = batch[0]["pseudobulk"].shape[0]  # 31 cell types
    batched_ccc_edge_index, batched_ccc_edge_type, batched_ccc_edge_attr = (
        _collate_edges(batch, n_nodes_per_graph)
    )

    # Create batch vector: [0,0,...,0, 1,1,...,1, 2,2,...,2, ...]
    graph_batch = torch.arange(batch_size).repeat_interleave(n_nodes_per_graph)

    # Create pointer tensor for graph boundaries
    graph_ptr = torch.arange(0, (batch_size + 1) * n_nodes_per_graph, n_nodes_per_graph)

    # ─────────────────────────────────────────────────────────────────────────
    # Subject IDs
    # ─────────────────────────────────────────────────────────────────────────
    subject_ids = [s["subject_id"] for s in batch]

    result = {
        "pseudobulk": pseudobulk,
        "cell_type_mask": cell_type_mask,
        "cell_counts": cell_counts,
        "pathology": pathology,
        "cognition": cognition,
        # Graph data (flat representation for efficiency)
        "ccc_edge_index": batched_ccc_edge_index,
        "ccc_edge_type": batched_ccc_edge_type,
        "ccc_edge_attr": batched_ccc_edge_attr,
        "graph_batch": graph_batch,
        "graph_ptr": graph_ptr,
        "n_nodes_per_graph": n_nodes_per_graph,
        # Cell-level data (flat format)
        "cell_data": cell_data,
        "cell_offsets": cell_offsets,
        # Region mask
        "region_mask": region_mask,
        # Metadata
        "subject_ids": subject_ids,
        "batch_size": batch_size,
    }

    return result


def collate_for_hgt(batch: list[dict[str, Any]], *, skip_region_mask: bool = False) -> dict[str, Any]:
    """
    Collate function returning flat concatenated tensors for HGTEncoderTensor.

    This is the RECOMMENDED collate function for our model because:
    - We have 31 distinct node types (cell types)
    - We have 5 distinct edge/relation types (CellChatDB categories)
    - HGTEncoderTensor expects flat edge tensors (no padding needed)

    Returns flat concatenated edge tensors directly compatible with
    HGTEncoderTensor, with node indices offset per sample.

    Returns:
        Batched dictionary with:
        - pseudobulk: [batch, n_cell_types, n_genes] (for PseudobulkEncoder)
        - cell_type_mask: [batch, n_cell_types] bool mask for available cell types
        - cell_counts: [batch, n_cell_types] number of cells per type
        - ccc_edge_index: [2, E_total] flat edge indices with per-sample node offsets
        - ccc_edge_type: [E_total] edge type indices
        - ccc_edge_attr: [E_total, edge_dim] edge attributes
        - pathology: [batch, n_pathology]
        - cognition: [batch, 1]
        - cell_data: [total_cells, n_genes] concatenated cell expressions
        - cell_offsets: [batch, n_cell_types + 1] cumulative offsets
        - region_mask: [batch, n_regions] bool mask for available regions
        - subject_ids: list of strings
        - batch_size: int
    """
    batch_size = len(batch)

    # Pre-allocate and fill in single pass (avoids 6 separate list comprehensions)
    s0 = batch[0]
    pseudobulk = torch.empty(batch_size, *s0["pseudobulk"].shape, dtype=torch.float32)
    cell_type_mask = torch.empty(batch_size, *s0["cell_type_mask"].shape, dtype=torch.bool)
    cell_counts = torch.empty(batch_size, *s0["cell_counts"].shape, dtype=torch.long)
    pathology = torch.empty(batch_size, *s0["pathology"].shape, dtype=torch.float32)
    cognition = torch.empty(batch_size, *s0["cognition"].shape, dtype=torch.float32)
    if not skip_region_mask:
        region_mask = torch.empty(batch_size, *s0["region_mask"].shape, dtype=torch.bool)

    for i, s in enumerate(batch):
        pseudobulk[i] = s["pseudobulk"]
        cell_type_mask[i] = s["cell_type_mask"]
        cell_counts[i] = s["cell_counts"]
        pathology[i] = s["pathology"]
        cognition[i] = s["cognition"]
        if not skip_region_mask:
            region_mask[i] = s["region_mask"]

    # ─────────────────────────────────────────────────────────────────────────
    # Cell-level data: flat format (cell_data + cell_offsets)
    # ─────────────────────────────────────────────────────────────────────────
    n_genes_flat = s0["pseudobulk"].shape[1]
    cell_data, cell_offsets = _collate_cell_data(batch, n_genes_flat)

    # ─────────────────────────────────────────────────────────────────────────
    # Build flat edge tensors for HGTEncoderTensor
    # ─────────────────────────────────────────────────────────────────────────
    N = s0["pseudobulk"].shape[0]  # n_cell_types (fixed at 31)
    ccc_edge_index, ccc_edge_type, ccc_edge_attr = _collate_edges(batch, N)

    subject_ids = [s["subject_id"] for s in batch]

    result = {
        "pseudobulk": pseudobulk,
        "cell_type_mask": cell_type_mask,
        "cell_counts": cell_counts,
        "pathology": pathology,
        "cognition": cognition,
        # HGT inputs (flat concatenated tensors)
        "ccc_edge_index": ccc_edge_index,
        "ccc_edge_type": ccc_edge_type,
        "ccc_edge_attr": ccc_edge_attr,
        # Cell-level data (flat format)
        "cell_data": cell_data,
        "cell_offsets": cell_offsets,
        # Metadata
        "subject_ids": subject_ids,
        "batch_size": batch_size,
    }

    if not skip_region_mask:
        result["region_mask"] = region_mask

    return result


def collate_multiregion(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Collate function for multi-region data (non-HGT format).

    Handles subjects with variable numbers of brain regions.

    Note:
        Multi-region support: detects region data via "region_pseudobulk"
        sentinel key OR by auto-deriving from region_{idx}_pseudobulk keys
        (via _derive_available_regions_from_keys). When region data is
        detected, assembles region tensors with auto_derive_regions=True.

        For full multi-region support with HGT graph format, use
        collate_for_hgt_multiregion() instead, which additionally handles
        per-region edge_index/edge_attr dicts for graph construction.

    Args:
        batch: List of sample dictionaries with per-region data

    Returns:
        Batched dictionary with region dimension
    """
    batch_size = len(batch)
    n_regions = N_REGIONS

    # Check if multi-region data is present
    has_regions = "region_pseudobulk" in batch[0] or bool(_derive_available_regions_from_keys(batch[0]))

    if not has_regions:
        # Fall back to standard collate
        return collate_fn(batch)

    # ─────────────────────────────────────────────────────────────────────────
    # Multi-region tensor handling
    # ─────────────────────────────────────────────────────────────────────────
    n_cell_types = batch[0]["pseudobulk"].shape[0]
    n_genes = batch[0]["pseudobulk"].shape[1]

    region_pseudobulk, region_mask = _assemble_region_tensors(
        batch, batch_size, n_cell_types, n_genes,
        n_regions=n_regions, auto_derive_regions=True,
    )

    # Standard collate for non-region data
    base_batch = collate_fn(batch)

    # Add region-specific data
    base_batch["region_pseudobulk"] = region_pseudobulk
    base_batch["region_mask"] = region_mask

    return base_batch


def collate_for_hgt_multiregion(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Collate function combining HGT flat tensor format with multi-region support.

    This combines:
    - Flat concatenated edge tensors for HGTEncoderTensor (from collate_for_hgt)
    - Region-specific pseudobulk tensors (from collate_multiregion)

    Use this when you need both HGT graph structure AND multi-region data.

    Args:
        batch: List of sample dictionaries

    Returns:
        Batched dictionary with:
        - All keys from collate_for_hgt (edge dicts, pseudobulk, etc.)
        - region_pseudobulk: [batch, n_regions, n_cell_types, n_genes]
        - region_mask: [batch, n_regions] bool mask (overrides inherited version
          with actual data presence check)
    """
    # Check if samples have pre-stacked region_pseudobulk [n_regions, C, G]
    # (produced by PrecomputedDataset mmap templates).
    # If so, torch.stack is ~100x faster than _assemble_region_tensors
    # (which allocates 36 MB zeros + nested fill loop).
    s0 = batch[0]
    pre_stacked = (
        "region_pseudobulk" in s0
        and isinstance(s0["region_pseudobulk"], torch.Tensor)
        and s0["region_pseudobulk"].ndim == 3  # [n_regions, C, G]
    )

    if pre_stacked:
        # Fast path: all region data pre-stacked in templates
        result = collate_for_hgt(batch, skip_region_mask=True)
        result["region_pseudobulk"] = torch.stack(
            [s["region_pseudobulk"] for s in batch]
        )
        result["region_mask"] = torch.stack(
            [s["region_mask"] for s in batch]
        )
    else:
        # Slow path: assemble from per-region keys
        has_regions = (
            "region_pseudobulk" in s0
            or bool(_derive_available_regions_from_keys(s0))
        )
        result = collate_for_hgt(batch, skip_region_mask=has_regions)

        if has_regions:
            batch_size = len(batch)
            n_cell_types = s0["pseudobulk"].shape[0]
            n_genes = s0["pseudobulk"].shape[1]

            region_pseudobulk, region_mask = _assemble_region_tensors(
                batch, batch_size, n_cell_types, n_genes,
                n_regions=N_REGIONS, auto_derive_regions=True,
            )

            result["region_pseudobulk"] = region_pseudobulk
            result["region_mask"] = region_mask

    return result


def _worker_init_fn(worker_id: int) -> None:
    """
    Re-seed each DataLoader worker for reproducible cell sampling.

    When num_workers > 0, each worker process gets a copy of the dataset.
    Without re-seeding, all workers share the same CellSampler RNG state,
    producing identical samples. This function seeds each worker's
    CellSampler.rng with (base_seed + worker_id) so each worker produces
    unique but reproducible samples.

    Also seeds numpy and stdlib random for any other stochastic operations
    in the data pipeline.

    Args:
        worker_id: Worker process index (0 to num_workers-1)
    """
    import random

    # Use PyTorch's built-in worker seed (set from the global seed + worker_id)
    worker_info = torch.utils.data.get_worker_info()
    worker_seed = worker_info.seed % (2**32)

    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

    # Re-seed CellSampler's RNG if the dataset has one.
    # PrecomputedDataset has no sampler (no cell sampling) — hasattr is
    # intentionally a no-op for it.
    dataset = worker_info.dataset
    if hasattr(dataset, "sampler") and hasattr(dataset.sampler, "rng"):
        dataset.sampler.rng = np.random.default_rng(worker_seed)


def create_dataloader(
    dataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
    multiregion: bool = False,
    use_hgt_format: bool = True,  # Default True - tensor format for HGTEncoderTensor
    prefetch_factor: int | None = 2,
    worker_init_fn=None,
    sampler=None,
    seed: int | None = None,
) -> torch.utils.data.DataLoader:
    """
    Create DataLoader with appropriate collate function.

    Note on Multi-GPU:
        For DDP training, use CognitiveResilienceDataModule instead of calling
        this function directly. The DataModule handles DistributedSampler setup
        and rank-aware worker seeding. This function is used internally by the
        DataModule and is safe for single-GPU usage.

    Worker Reproducibility:
        Uses _worker_init_fn to re-seed each worker's CellSampler RNG
        with a unique seed derived from (global_seed + worker_id). This
        ensures each worker produces unique but reproducible cell samples
        across runs.

    Args:
        dataset: CognitiveResilienceDataset or PrecomputedDataset
        batch_size: Batch size (per GPU when using DDP)
        shuffle: Whether to shuffle (ignored when using DistributedSampler)
        num_workers: Number of worker processes
        pin_memory: Pin memory for faster GPU transfer
        drop_last: Drop incomplete last batch
        multiregion: Use multi-region collate function
        use_hgt_format: Use collate_for_hgt which returns padded edge tensors
                        compatible with HGTEncoderTensor (default: True, recommended)
        prefetch_factor: Number of batches to prefetch per worker (None when num_workers=0)
        worker_init_fn: Optional custom worker init function. When provided,
            overrides the default _worker_init_fn. Used by CognitiveResilienceDataModule
            to inject rank-aware seeding for DDP.
        seed: Optional seed for DataLoader's internal Generator. Controls
            shuffling order when shuffle=True and no external sampler is used.
            Pass experiment seed for reproducible batch ordering.

    Returns:
        Configured DataLoader
    """
    # Create a seeded Generator for reproducible shuffling
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    # Select collate function based on flags
    if use_hgt_format and multiregion:
        collate = collate_for_hgt_multiregion
    elif use_hgt_format:
        collate = collate_for_hgt
    elif multiregion:
        collate = collate_multiregion
    else:
        collate = collate_fn

    # prefetch_factor requires num_workers > 0
    effective_prefetch = prefetch_factor if num_workers > 0 else None

    # sampler and shuffle are mutually exclusive in DataLoader
    effective_shuffle = False if sampler is not None else shuffle

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=effective_shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate,
        persistent_workers=num_workers > 0,
        worker_init_fn=(worker_init_fn or _worker_init_fn) if num_workers > 0 else None,
        prefetch_factor=effective_prefetch,
        generator=generator,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Device utilities (for non-Lightning usage only)
# ─────────────────────────────────────────────────────────────────────────────
# Canonical implementations live in src.utils.device.  Re-exported here for
from src.utils.device import move_batch_to_device


def get_effective_batch_size(batch_size: int, num_gpus: int, strategy: str = "ddp") -> int:
    """
    Calculate effective batch size for distributed training.

    Args:
        batch_size: Per-GPU batch size
        num_gpus: Number of GPUs
        strategy: Training strategy ("ddp", "dp", "ddp_spawn")

    Returns:
        Effective batch size across all GPUs
    """
    if strategy in ("ddp", "ddp_spawn", "deepspeed"):
        # Each GPU processes batch_size samples independently
        return batch_size * num_gpus
    elif strategy == "dp":
        # DataParallel splits batch across GPUs
        return batch_size
    else:
        return batch_size
