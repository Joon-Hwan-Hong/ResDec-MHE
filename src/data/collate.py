"""
Custom collate functions for batching heterogeneous graph data.

Handles batching of:
- Variable-sized graphs (different number of CCC edges per subject)
- Padded cell-level data
- Multi-region data (optional)

Collate Format:
    This module uses **dict lists** format for HGT graph data, NOT PyG HeteroData.
    The dict format is directly compatible with HGTEncoderBatched and avoids
    runtime conversion overhead. Key functions:
    - collate_for_hgt_multiregion: Primary collate for training (recommended)
    - collate_for_hgt: Single-region variant
    - collate_fn: Basic collate without HGT dict format

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
import os
import warnings
from typing import Any

import numpy as np
import torch

from src.data.constants import (
    CELL_TYPE_ORDER, ALL_EDGE_TYPES, N_REGIONS, PFC_REGION_IDX,
    sanitize_key, SANITIZED_CELL_TYPE_ORDER, SANITIZED_EDGE_TYPES,
)

logger = logging.getLogger(__name__)

# Cache for edge dict grouping — PrecomputedDataset produces identical
# edge tensors per subject every epoch, so caching avoids repeated argsort +
# boundary detection. Keyed by (subject_id, n_edges).
# Note: this is module-level (global) state. With persistent_workers=True,
# each worker retains its cache across epochs. Call clear_edge_dict_cache()
# between dataset switches or in test teardown.
_edge_dict_cache: dict[tuple[str, int], tuple[dict, dict]] = {}
_EDGE_DICT_CACHE_MAX = 1024  # Limit memory usage


def clear_edge_dict_cache() -> None:
    """Clear the module-level edge dict cache.

    Call between test cases, dataset switches, or when edge data changes
    for the same subject IDs.
    """
    _edge_dict_cache.clear()


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


def _pad_and_stack_cells(
    batch: list[dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad cells/cell_mask to batch actual max and stack.

    Pre-computes actual max valid cell index to avoid over-allocation
    and eliminate the separate trim pass.
    """
    n_cell_types = batch[0]["cells"].shape[0]
    n_genes = batch[0]["cells"].shape[2]
    batch_size = len(batch)

    # Compute actual max valid cells across all samples to avoid
    # over-allocating and then trimming.
    actual_max = 0
    for s in batch:
        mask = s["cell_mask"]  # [n_cell_types, nc]
        if mask.any():
            # Find highest valid cell index across all cell types
            valid_per_col = mask.any(dim=0)  # [nc]
            if valid_per_col.any():
                last_valid = valid_per_col.nonzero()[-1].item() + 1
                actual_max = max(actual_max, last_valid)
    actual_max = max(actual_max, 1)  # At least 1

    # Allocate directly to actual_max (zero-filled = correct padding)
    cells = torch.zeros(batch_size, n_cell_types, actual_max, n_genes)
    cell_mask = torch.zeros(batch_size, n_cell_types, actual_max, dtype=torch.bool)

    for i, s in enumerate(batch):
        nc = min(s["cells"].shape[1], actual_max)
        cells[i, :, :nc, :] = s["cells"][:, :nc, :]
        cell_mask[i, :, :nc] = s["cell_mask"][:, :nc]

    return cells, cell_mask


def _trim_cells_to_actual_max(
    cells: torch.Tensor,
    cell_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Trim cells and cell_mask to the actual max cell count used in the batch.

    After padding to batch max, many trailing columns may be entirely unused
    (all-False in mask). This trims them to save memory.

    Args:
        cells: [B, n_cell_types, max_cells, n_genes]
        cell_mask: [B, n_cell_types, max_cells]

    Returns:
        Trimmed (cells, cell_mask) — same tensors if no trim needed.
    """
    if not cell_mask.any():
        return cells, cell_mask
    # Find highest valid cell index across all samples and cell types
    max_valid = cell_mask.any(dim=0).any(dim=0).long()  # [max_cells] bool
    if max_valid.any():
        actual_max = max_valid.nonzero()[-1].item() + 1
        if actual_max < cell_mask.shape[2]:
            cells = cells[:, :, :actual_max, :]
            cell_mask = cell_mask[:, :, :actual_max]
    return cells, cell_mask


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
                region_key = f"region_{region_idx}_pseudobulk"
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


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Collate function for CognitiveResilienceDataset.

    Handles:
    - Standard tensor stacking (pseudobulk, pathology, cognition)
    - Graph batching for variable-sized CCC graphs
    - Padded tensors (cells, cell_mask)
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
        - cells: [batch, n_cell_types, max_cells, n_genes]
        - cell_mask: [batch, n_cell_types, max_cells]
        - region_mask: [batch, n_regions] bool mask for available regions
        - subject_ids: list of strings
        - batch_size: int
        - n_nodes_per_graph: int (31 cell types)
    """
    batch_size = len(batch)

    # ─────────────────────────────────────────────────────────────────────────
    # Standard tensor stacking
    # ─────────────────────────────────────────────────────────────────────────
    pseudobulk = torch.stack([s["pseudobulk"] for s in batch], dim=0)
    cell_type_mask = torch.stack([s["cell_type_mask"] for s in batch], dim=0)
    cell_counts = torch.stack([s["cell_counts"] for s in batch], dim=0)
    pathology = torch.stack([s["pathology"] for s in batch], dim=0)
    cognition = torch.stack([s["cognition"] for s in batch], dim=0)
    cells, cell_mask = _pad_and_stack_cells(batch)
    region_mask = torch.stack([s["region_mask"] for s in batch], dim=0)

    # ─────────────────────────────────────────────────────────────────────────
    # Graph batching - Manual approach for homogeneous treatment
    # ─────────────────────────────────────────────────────────────────────────
    # For HGT, we have a fixed number of nodes (31 cell types) per graph,
    # but variable edges. We batch by:
    # 1. Concatenating edge_index with node offsets
    # 2. Concatenating edge_type and edge_attr
    # 3. Creating batch vector mapping nodes to graphs

    n_nodes_per_graph = batch[0]["pseudobulk"].shape[0]  # 31 cell types

    edge_indices = []
    edge_types = []
    edge_attrs = []
    node_offset = 0

    for s in batch:
        ccc_edge_index = s["ccc_edge_index"]
        n_edges = ccc_edge_index.shape[1] if ccc_edge_index.numel() > 0 else 0

        if n_edges > 0:
            # Add node offset for batching
            edge_indices.append(ccc_edge_index + node_offset)
            edge_types.append(s["ccc_edge_type"])
            edge_attrs.append(s["ccc_edge_attr"])

        node_offset += n_nodes_per_graph

    # Concatenate all edges
    if edge_indices:
        batched_ccc_edge_index = torch.cat(edge_indices, dim=1)
        batched_ccc_edge_type = torch.cat(edge_types, dim=0)
        batched_ccc_edge_attr = torch.cat(edge_attrs, dim=0)
    else:
        batched_ccc_edge_index = torch.zeros((2, 0), dtype=torch.long)
        batched_ccc_edge_type = torch.zeros((0,), dtype=torch.long)
        batched_ccc_edge_attr = torch.zeros((0, 1), dtype=torch.float)

    # Create batch vector: [0,0,...,0, 1,1,...,1, 2,2,...,2, ...]
    graph_batch = torch.arange(batch_size).repeat_interleave(n_nodes_per_graph)

    # Create pointer tensor for graph boundaries
    graph_ptr = torch.arange(0, (batch_size + 1) * n_nodes_per_graph, n_nodes_per_graph)

    # ─────────────────────────────────────────────────────────────────────────
    # Subject IDs
    # ─────────────────────────────────────────────────────────────────────────
    subject_ids = [s["subject_id"] for s in batch]

    return {
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
        # Cell-level data (all 31 cell types)
        "cells": cells,
        "cell_mask": cell_mask,
        # Region mask
        "region_mask": region_mask,
        # Metadata
        "subject_ids": subject_ids,
        "cell_barcodes": [s.get("cell_barcodes") for s in batch],
        "batch_size": batch_size,
    }


def collate_for_hgt(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Collate function returning dict format for HGTEncoderBatched.

    This is the RECOMMENDED collate function for our model because:
    - We have 31 distinct node types (cell types)
    - We have 5 distinct edge/relation types (CellChatDB categories)
    - HGTEncoderBatched expects lists of dicts (one per sample)

    Returns dicts directly compatible with HGTEncoderBatched, avoiding the need
    for HeteroData → dict conversion at runtime.

    Note:
        This collate does NOT produce x_dict_list for HGT node features.
        The full model builds its own x_dict_list from encoded embeddings via
        build_x_dict_list_from_embeddings(). For standalone HGT testing,
        use build_x_dict_list_from_embeddings() with raw pseudobulk or
        dummy embeddings.

    Returns:
        Batched dictionary with:
        - pseudobulk: [batch, n_cell_types, n_genes] (for PseudobulkEncoder)
        - cell_type_mask: [batch, n_cell_types] bool mask for available cell types
        - cell_counts: [batch, n_cell_types] number of cells per type
        - edge_index_dict_list: List of {(src, rel, dst): (2, n_edges)} per sample
        - edge_attr_dict_list: List of {(src, rel, dst): (n_edges, 1)} per sample
        - pathology: [batch, n_pathology]
        - cognition: [batch, 1]
        - cells: [batch, n_cell_types, max_cells, n_genes]
        - cell_mask: [batch, n_cell_types, max_cells]
        - region_mask: [batch, n_regions] bool mask for available regions
        - subject_ids: list of strings
        - batch_size: int
        - node_types: list of sanitized cell type names
        - edge_types: list of sanitized edge type names
    """
    batch_size = len(batch)

    # Standard tensor stacking (same as collate_fn)
    pseudobulk = torch.stack([s["pseudobulk"] for s in batch], dim=0)
    cell_type_mask = torch.stack([s["cell_type_mask"] for s in batch], dim=0)
    cell_counts = torch.stack([s["cell_counts"] for s in batch], dim=0)
    pathology = torch.stack([s["pathology"] for s in batch], dim=0)
    cognition = torch.stack([s["cognition"] for s in batch], dim=0)
    cells, cell_mask = _pad_and_stack_cells(batch)

    region_mask = torch.stack([s["region_mask"] for s in batch], dim=0)

    # ─────────────────────────────────────────────────────────────────────────
    # Build dict lists for HGTEncoderBatched
    # ─────────────────────────────────────────────────────────────────────────
    # Use cell_type_order from dataset, with fallback to global constant for
    # backward compatibility with samples that don't include it
    cell_type_names = batch[0].get("cell_type_order", CELL_TYPE_ORDER)
    edge_type_names = ALL_EDGE_TYPES

    # Validate cell_type_order consistency (structural invariant from dataset construction).
    # Gated behind RESILIENCE_DEBUG because this is a hot path and the invariant is
    # enforced by dataset construction; the check is useful for debugging only.
    if os.environ.get("RESILIENCE_DEBUG"):
        for s in batch[1:]:
            if s.get("cell_type_order", CELL_TYPE_ORDER) != cell_type_names:
                raise RuntimeError(
                    "cell_type_order mismatch within batch — dataset construction bug. "
                    f"Expected {cell_type_names[:3]}..., got {s.get('cell_type_order', 'N/A')[:3]}..."
                )

    # Sanitize names for PyG compatibility (uses shared sanitize_key from constants)
    # Use pre-computed constants when cell_type_names matches global order (common case)
    if cell_type_names is CELL_TYPE_ORDER or cell_type_names == CELL_TYPE_ORDER:
        sanitized_cell_types = SANITIZED_CELL_TYPE_ORDER
    else:
        sanitized_cell_types = [sanitize_key(ct) for ct in cell_type_names]
    if edge_type_names is ALL_EDGE_TYPES:
        sanitized_edge_types = SANITIZED_EDGE_TYPES
    else:
        sanitized_edge_types = [sanitize_key(et) for et in edge_type_names]

    edge_index_dict_list = []
    edge_attr_dict_list = []

    for s in batch:
        subject_id = s.get("subject_id")
        edge_index = s["ccc_edge_index"]
        n_edges = edge_index.shape[1] if edge_index.numel() > 0 else 0

        # Check cache for precomputed edge dicts (valid for PrecomputedDataset
        # where edge tensors are identical across epochs for each subject).
        # Key includes n_edges to avoid stale hits when the same subject_id
        # appears with different edge counts (e.g., across test scenarios).
        cache_key = (subject_id, n_edges) if subject_id is not None else None
        if cache_key is not None and cache_key in _edge_dict_cache:
            cached_ei, cached_ea = _edge_dict_cache[cache_key]
            edge_index_dict_list.append(cached_ei)
            edge_attr_dict_list.append(cached_ea)
            continue

        # Build edge dicts: {(src, rel, dst): tensor}
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor] = {}
        edge_attr_dict: dict[tuple[str, str, str], torch.Tensor] = {}

        edge_type_indices = s["ccc_edge_type"]
        edge_attr = s["ccc_edge_attr"]

        if n_edges > 0:
            # Vectorized grouping: compute a composite key per edge encoding
            # (src_type, edge_type, dst_type) as a single integer, then use
            # torch.unique to find distinct triplets in one pass.
            n_ct = len(sanitized_cell_types)
            n_et = len(sanitized_edge_types)

            src_indices = edge_index[0]       # [n_edges]
            dst_indices = edge_index[1]       # [n_edges]

            # Guard against silent integer overflow in composite key arithmetic.
            # With int64 this is safe for current constants (31 cell types, 5 edge types)
            # but would fail loudly if constants ever grow beyond safe limits.
            max_key = (n_ct - 1) * n_ct * n_et + (n_ct - 1) * n_et + (n_et - 1)
            if max_key >= torch.iinfo(torch.int64).max:
                raise ValueError(
                    f"Composite key overflow: max_key={max_key} exceeds int64 max "
                    f"({torch.iinfo(torch.int64).max}). n_cell_types={n_ct}, n_edge_types={n_et}"
                )

            # Composite key: src * (n_ct * n_et) + dst * n_et + edge_type
            composite = src_indices * (n_ct * n_et) + dst_indices * n_et + edge_type_indices

            sorted_order = torch.argsort(composite)
            sorted_composite = composite[sorted_order]

            # Find group boundaries in single pass (pre-allocated, no sentinel tensors)
            changes = torch.empty(len(sorted_composite) + 1, dtype=torch.bool,
                                  device=sorted_composite.device)
            changes[0] = True
            changes[1:-1] = sorted_composite[1:] != sorted_composite[:-1]
            changes[-1] = True
            boundaries = torch.where(changes)[0]

            for g in range(len(boundaries) - 1):
                start, end = boundaries[g].item(), boundaries[g + 1].item()
                k = sorted_composite[start].item()
                src_idx = k // (n_ct * n_et)
                remaining = k % (n_ct * n_et)
                dst_idx = remaining // n_et
                et_idx = remaining % n_et

                triplet = (
                    sanitized_cell_types[src_idx],
                    sanitized_edge_types[et_idx],
                    sanitized_cell_types[dst_idx],
                )
                n_triplet_edges = end - start

                # All edges are node 0 → node 0 because each cell type has
                # exactly 1 node per subject (the pseudobulk embedding).
                # This invariant is set by build_x_dict_list_from_embeddings
                # (unsqueeze(0) → 1 node per type). If multi-node-per-type
                # is ever added, edge indices must be updated.
                edge_index_dict[triplet] = torch.zeros(
                    2, n_triplet_edges, dtype=torch.long
                )
                # Contiguous slice from sorted order instead of boolean mask
                edge_attr_dict[triplet] = edge_attr[sorted_order[start:end]]

        # Cache result for this subject
        if cache_key is not None and len(_edge_dict_cache) < _EDGE_DICT_CACHE_MAX:
            _edge_dict_cache[cache_key] = (edge_index_dict, edge_attr_dict)

        edge_index_dict_list.append(edge_index_dict)
        edge_attr_dict_list.append(edge_attr_dict)

    subject_ids = [s["subject_id"] for s in batch]

    return {
        "pseudobulk": pseudobulk,
        "cell_type_mask": cell_type_mask,
        "cell_counts": cell_counts,
        "pathology": pathology,
        "cognition": cognition,
        # HGT inputs (dict lists for HGTEncoderBatched)
        "edge_index_dict_list": edge_index_dict_list,
        "edge_attr_dict_list": edge_attr_dict_list,
        # Cell-level data (all 31 cell types)
        "cells": cells,
        "cell_mask": cell_mask,
        # Region mask
        "region_mask": region_mask,
        # Metadata
        "subject_ids": subject_ids,
        "cell_barcodes": [s.get("cell_barcodes") for s in batch],
        "batch_size": batch_size,
        # Include metadata for model
        "node_types": sanitized_cell_types,
        "edge_types": sanitized_edge_types,
        # Raw cell type order (unsanitized) for reference
        "cell_type_order": cell_type_names,
    }


def build_x_dict_list_from_embeddings(
    pseudobulk_embeddings: torch.Tensor,
    node_types: list[str],
) -> list[dict[str, torch.Tensor]]:
    """
    Build x_dict_list from encoded pseudobulk embeddings for HGTEncoderBatched.

    This is the RECOMMENDED way to prepare HGT node features in the full model.
    Call this in model.forward() after running PseudobulkEncoder:

        pseudobulk_emb = self.pseudobulk_encoder(batch["pseudobulk"])
        x_dict_list = build_x_dict_list_from_embeddings(
            pseudobulk_emb, batch["node_types"]
        )
        hgt_out, _ = self.hgt_encoder(
            x_dict_list,
            batch["edge_index_dict_list"],
            batch["edge_attr_dict_list"],
        )

    Args:
        pseudobulk_embeddings: [batch, n_cell_types, d_embed] encoded features
            from PseudobulkEncoder
        node_types: List of sanitized cell type names (from batch["node_types"])

    Returns:
        List of {cell_type: (1, d_embed)} dicts, one per sample in batch.
        Ready to pass to HGTEncoderBatched.
    """
    batch_size = pseudobulk_embeddings.size(0)
    # n_cell_types is always 31 (Allen Brain Cell Atlas mapping invariant from snRNAseq data)
    n_cell_types = pseudobulk_embeddings.size(1)

    # Pre-split along cell type dim in one C++ call, then use slice views
    ct_slices = pseudobulk_embeddings.unbind(dim=1)  # tuple of [B, d_embed]

    x_dict_list = []
    for b in range(batch_size):
        x_dict = {
            node_types[ct_idx]: ct_slices[ct_idx][b:b+1]  # [1, d_embed] slice view
            for ct_idx in range(n_cell_types)
        }
        x_dict_list.append(x_dict)

    return x_dict_list


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
    Collate function combining HGT format with multi-region support.

    This combines:
    - Dict lists for HGTEncoderBatched (from collate_for_hgt)
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
    # Start with HGT format collate
    result = collate_for_hgt(batch)

    # Check only first sample — all samples in a batch come from the same
    # dataset and should have the same key structure.
    has_regions = (
        "region_pseudobulk" in batch[0]
        or bool(_derive_available_regions_from_keys(batch[0]))
    )

    if has_regions:
        batch_size = len(batch)
        n_cell_types = batch[0]["pseudobulk"].shape[0]
        n_genes = batch[0]["pseudobulk"].shape[1]

        region_pseudobulk, region_mask = _assemble_region_tensors(
            batch, batch_size, n_cell_types, n_genes,
            n_regions=N_REGIONS, auto_derive_regions=True,
        )

        result["region_pseudobulk"] = region_pseudobulk
        # Override inherited region_mask with the computed version based on
        # actual data presence. This is more accurate than the sample-level
        # region_mask which may differ in edge cases.
        # Assert the computed mask is a subset of the per-sample mask: a region
        # should never appear in the computed mask if the sample said it's absent.
        if "region_mask" in result:
            sample_mask = result["region_mask"].bool()
            computed_mask = region_mask.bool()
            if (computed_mask & ~sample_mask).any():
                logger.warning(
                    "Computed region_mask has regions marked present that per-sample "
                    "mask marks absent. This may indicate a precomputation mismatch."
                )
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
    use_hgt_format: bool = True,  # Default True - dict format for HGTEncoderBatched
    prefetch_factor: int | None = 2,
    worker_init_fn=None,
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
        use_hgt_format: Use collate_for_hgt which returns dict lists compatible
                        with HGTEncoderBatched (default: True, recommended)
        prefetch_factor: Number of batches to prefetch per worker (None when num_workers=0)
        worker_init_fn: Optional custom worker init function. When provided,
            overrides the default _worker_init_fn. Used by CognitiveResilienceDataModule
            to inject rank-aware seeding for DDP.

    Returns:
        Configured DataLoader
    """
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

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate,
        persistent_workers=num_workers > 0,
        worker_init_fn=(worker_init_fn or _worker_init_fn) if num_workers > 0 else None,
        prefetch_factor=effective_prefetch,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Device utilities (for non-Lightning usage only)
# ─────────────────────────────────────────────────────────────────────────────
# Canonical implementations live in src.utils.device.  Re-exported here for
# backward-compatibility so that existing ``from src.data.collate import
# move_batch_to_device`` statements continue to work.
from src.utils.device import move_batch_to_device  # noqa: F401


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
