"""
Custom collate functions for batching heterogeneous graph data.

Handles batching of:
- Variable-sized graphs (different number of CCC edges per subject)
- Padded cell-level data
- Multi-region data (optional)

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

from typing import Any

import torch
from torch_geometric.data import Batch, Data, HeteroData

from src.data.constants import CELL_TYPE_ORDER, ALL_EDGE_TYPES


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
    cells = torch.stack([s["cells"] for s in batch], dim=0)
    cell_mask = torch.stack([s["cell_mask"] for s in batch], dim=0)
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
        "batch_size": batch_size,
    }


def collate_to_heterodata(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Collate function creating proper HeteroData for HGTConv.

    This is the RECOMMENDED collate function for our model because:
    - We have 31 distinct node types (cell types)
    - We have 5 distinct edge/relation types (CellChatDB categories)
    - HGTConv learns type-specific projection matrices

    See: https://pytorch-geometric.readthedocs.io/en/latest/notes/heterogeneous.html

    Returns:
        Batched dictionary with:
        - pseudobulk: [batch, n_cell_types, n_genes] (for non-graph branches)
        - cell_type_mask: [batch, n_cell_types] bool mask for available cell types
        - cell_counts: [batch, n_cell_types] number of cells per type
        - hetero_batch: PyG Batch of HeteroData objects
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
    cells = torch.stack([s["cells"] for s in batch], dim=0)
    cell_mask = torch.stack([s["cell_mask"] for s in batch], dim=0)
    region_mask = torch.stack([s["region_mask"] for s in batch], dim=0)

    # ─────────────────────────────────────────────────────────────────────────
    # Build HeteroData for each sample
    # ─────────────────────────────────────────────────────────────────────────
    hetero_graphs = []
    cell_type_names = CELL_TYPE_ORDER
    edge_type_names = ALL_EDGE_TYPES

    # Sanitize edge type names for PyG (no spaces, slashes)
    def sanitize_name(name: str) -> str:
        return name.replace(" ", "_").replace("/", "_").replace("-", "_")

    sanitized_edge_types = [sanitize_name(et) for et in edge_type_names]

    for s in batch:
        data = HeteroData()

        # Add node features for each cell type
        # Each cell type is a separate node type with 1 node per subject
        for ct_idx, ct_name in enumerate(cell_type_names):
            safe_ct = sanitize_name(ct_name)
            data[safe_ct].x = s["pseudobulk"][ct_idx].unsqueeze(0)  # [1, n_genes]

        # Add edges - group by (src_type, relation, dst_type) triplet
        edge_index = s["ccc_edge_index"]
        edge_type_indices = s["ccc_edge_type"]
        edge_attr = s["ccc_edge_attr"]

        if edge_index.numel() > 0:
            # Build a dictionary to accumulate edges per triplet
            edge_dict: dict[tuple, dict] = {}

            n_edges = edge_index.shape[1]
            for e in range(n_edges):
                src_ct_idx = edge_index[0, e].item()
                dst_ct_idx = edge_index[1, e].item()
                et_idx = edge_type_indices[e].item()

                src_ct = sanitize_name(cell_type_names[src_ct_idx])
                dst_ct = sanitize_name(cell_type_names[dst_ct_idx])
                relation = sanitized_edge_types[et_idx]

                triplet = (src_ct, relation, dst_ct)

                if triplet not in edge_dict:
                    edge_dict[triplet] = {"src": [], "dst": [], "attr": []}

                # All edges go from node 0 to node 0 (single node per type per subject)
                edge_dict[triplet]["src"].append(0)
                edge_dict[triplet]["dst"].append(0)
                edge_dict[triplet]["attr"].append(edge_attr[e])

            # Convert accumulated edges to tensors
            for triplet, edges in edge_dict.items():
                data[triplet].edge_index = torch.tensor(
                    [edges["src"], edges["dst"]], dtype=torch.long
                )
                data[triplet].edge_attr = torch.stack(edges["attr"], dim=0)

        hetero_graphs.append(data)

    # Batch heterogeneous graphs
    # PyG handles variable schemas by only batching edges that exist
    hetero_batch = Batch.from_data_list(hetero_graphs)

    subject_ids = [s["subject_id"] for s in batch]

    return {
        "pseudobulk": pseudobulk,
        "cell_type_mask": cell_type_mask,
        "cell_counts": cell_counts,
        "pathology": pathology,
        "cognition": cognition,
        "hetero_batch": hetero_batch,
        # Cell-level data (all 31 cell types)
        "cells": cells,
        "cell_mask": cell_mask,
        # Region mask
        "region_mask": region_mask,
        # Metadata
        "subject_ids": subject_ids,
        "batch_size": batch_size,
        # Include metadata for model
        "node_types": [sanitize_name(ct) for ct in cell_type_names],
        "edge_types": sanitized_edge_types,
    }


def collate_multiregion(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Collate function for multi-region data.

    Handles subjects with variable numbers of brain regions.

    Args:
        batch: List of sample dictionaries with per-region data

    Returns:
        Batched dictionary with region dimension
    """
    batch_size = len(batch)
    n_regions = 6  # Fixed number of regions

    # Check if multi-region data is present
    has_regions = "region_pseudobulk" in batch[0]

    if not has_regions:
        # Fall back to standard collate
        return collate_fn(batch)

    # ─────────────────────────────────────────────────────────────────────────
    # Multi-region tensor handling
    # ─────────────────────────────────────────────────────────────────────────

    # Get dimensions from first sample
    n_cell_types = batch[0]["pseudobulk"].shape[0]
    n_genes = batch[0]["pseudobulk"].shape[1]

    # Initialize tensors
    region_pseudobulk = torch.zeros(batch_size, n_regions, n_cell_types, n_genes)
    region_mask = torch.zeros(batch_size, n_regions, dtype=torch.bool)

    for i, s in enumerate(batch):
        # Get available regions for this subject
        available_regions = s.get("available_regions", [0])  # Default: only DLPFC

        for region_idx in available_regions:
            if region_idx < n_regions:
                region_key = f"region_{region_idx}_pseudobulk"
                if region_key in s:
                    region_pseudobulk[i, region_idx] = s[region_key]
                    region_mask[i, region_idx] = True
                elif region_idx == 0:
                    # Use main pseudobulk for DLPFC
                    region_pseudobulk[i, 0] = s["pseudobulk"]
                    region_mask[i, 0] = True

    # Standard collate for non-region data
    base_batch = collate_fn(batch)

    # Add region-specific data
    base_batch["region_pseudobulk"] = region_pseudobulk
    base_batch["region_mask"] = region_mask

    return base_batch


def create_dataloader(
    dataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
    multiregion: bool = False,
    use_heterodata: bool = True,  # Default True - proper heterogeneous graphs
) -> torch.utils.data.DataLoader:
    """
    Create DataLoader with appropriate collate function.

    Note on Multi-GPU:
        When using PyTorch Lightning with multiple GPUs, do NOT use this
        function directly. Instead, let Lightning's Trainer handle DataLoader
        creation via LightningDataModule, which properly sets up
        DistributedSampler for DDP training.

    Args:
        dataset: CognitiveResilienceDataset or PrecomputedDataset
        batch_size: Batch size (per GPU when using DDP)
        shuffle: Whether to shuffle (ignored when using DistributedSampler)
        num_workers: Number of worker processes
        pin_memory: Pin memory for faster GPU transfer
        drop_last: Drop incomplete last batch
        multiregion: Use multi-region collate function
        use_heterodata: Use HeteroData collate for proper heterogeneous HGT
                        (default: True, recommended for our model)

    Returns:
        Configured DataLoader
    """
    if use_heterodata:
        collate = collate_to_heterodata
    elif multiregion:
        collate = collate_multiregion
    else:
        collate = collate_fn

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate,
        persistent_workers=num_workers > 0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Device utilities (for non-Lightning usage only)
# ─────────────────────────────────────────────────────────────────────────────

def move_batch_to_device(
    batch: dict[str, Any],
    device: torch.device | str,
) -> dict[str, Any]:
    """
    Move batch tensors to specified device.

    WARNING: This function is for manual/debugging use only.
    When using PyTorch Lightning, device placement is handled automatically
    by the Trainer. Do NOT call this in LightningModule.training_step().

    For multi-GPU setups:
    - With DDP: Each process sees only its GPU as device 0
    - The Trainer handles data distribution and device placement
    - Effective batch size = batch_size * num_gpus

    Args:
        batch: Batch dictionary from collate_fn
        device: Target device (e.g., "cuda:0", "cuda:1", torch.device("cuda"))

    Returns:
        Batch with tensors on device
    """
    if isinstance(device, str):
        device = torch.device(device)

    moved = {}

    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=True)
        elif isinstance(value, (Batch, HeteroData)):
            moved[key] = value.to(device)
        elif key in ("subject_ids", "batch_size", "n_nodes_per_graph"):
            moved[key] = value  # Keep as-is
        else:
            moved[key] = value

    return moved


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
