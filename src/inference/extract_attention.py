"""
HGT attention aggregation for CCC importance analysis.

Aggregates per-layer HGT attention weights into summary statistics
suitable for downstream cell communication analysis.

Current model output format (HGTEncoderTensor):
    hgt_attention: list[Tensor] — one [total_edges_in_batch, n_heads] per layer
    Edges are flat-concatenated across subjects in the batch, with node indices
    offset by 31 (n_cell_types) per subject.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import torch

from src.data.constants import N_CELL_TYPES

logger = logging.getLogger(__name__)


def split_batch_attention_by_subject(
    hgt_attention: list[torch.Tensor],
    ccc_edge_type: torch.Tensor,
    batch_size: int,
    ccc_edge_index: torch.Tensor,
    n_cell_types: int = N_CELL_TYPES,
) -> list[list[dict[int, np.ndarray]]]:
    """Split flat batch attention into per-subject, per-layer, per-edge-type arrays.

    Args:
        hgt_attention: list[Tensor [total_edges, n_heads]] per layer
        ccc_edge_type: Tensor [total_edges] — edge type indices
        batch_size: Number of subjects in batch
        ccc_edge_index: Tensor [2, total_edges] — source/target node indices
        n_cell_types: Nodes per subject (31)

    Returns:
        list[list[dict[int, ndarray]]] — [n_subjects][n_layers]{edge_type: [n_edges, n_heads]}
    """
    n_layers = len(hgt_attention)
    src_nodes = ccc_edge_index[0]

    # Determine which subject each edge belongs to from source node index
    subject_ids = src_nodes // n_cell_types  # [total_edges]

    result = []
    for s in range(batch_size):
        subject_mask = (subject_ids == s)
        subject_edge_types = ccc_edge_type[subject_mask]
        subject_layers = []
        for layer_idx in range(n_layers):
            layer_attn = hgt_attention[layer_idx][subject_mask]  # [n_subject_edges, n_heads]
            layer_attn_np = layer_attn.cpu().numpy() if isinstance(layer_attn, torch.Tensor) else layer_attn
            edge_types_np = subject_edge_types.cpu().numpy() if isinstance(subject_edge_types, torch.Tensor) else subject_edge_types

            edge_type_dict = {}
            for et in np.unique(edge_types_np):
                et_mask = edge_types_np == et
                edge_type_dict[int(et)] = layer_attn_np[et_mask]
            subject_layers.append(edge_type_dict)
        result.append(subject_layers)
    return result


def aggregate_hgt_attention(
    per_subject_attention: list[list[dict[int, np.ndarray]]],
    include_per_sample: bool = True,
) -> dict[str, np.ndarray | list]:
    """
    Aggregate HGT attention across subjects.

    Args:
        per_subject_attention: [n_subjects][n_layers]{edge_type_int: [n_edges, n_heads]}
            Output of split_batch_attention_by_subject, accumulated across batches.
        include_per_sample: Whether to include per-sample summaries

    Returns:
        Dict with:
            - 'edge_type_ids': sorted list of integer edge type indices
            - 'mean_by_edge_type': [n_edge_types, n_heads]
            - 'std_by_edge_type': [n_edge_types, n_heads]
            - 'per_sample': [n_samples, n_edge_types, n_layers, n_heads] (if include_per_sample)
            - 'n_samples', 'n_layers'
    """
    if not per_subject_attention:
        return {
            "edge_type_ids": [],
            "mean_by_edge_type": np.array([]),
            "std_by_edge_type": np.array([]),
            "per_sample": np.array([]) if include_per_sample else None,
            "n_samples": 0,
            "n_layers": 0,
        }

    n_samples = len(per_subject_attention)
    n_layers = len(per_subject_attention[0])

    # Discover all edge types
    edge_type_set = set()
    for subject_attn in per_subject_attention:
        for layer_attn in subject_attn:
            edge_type_set.update(layer_attn.keys())
    edge_type_ids = sorted(edge_type_set)

    if not edge_type_ids:
        return {
            "edge_type_ids": [],
            "mean_by_edge_type": np.array([]),
            "std_by_edge_type": np.array([]),
            "per_sample": np.array([]) if include_per_sample else None,
            "n_samples": n_samples,
            "n_layers": n_layers,
        }

    # Determine n_heads
    n_heads = None
    for subject_attn in per_subject_attention:
        for layer_attn in subject_attn:
            for attn in layer_attn.values():
                n_heads = attn.shape[-1]
                break
            if n_heads is not None:
                break
        if n_heads is not None:
            break

    if n_heads is None:
        return {
            "edge_type_ids": edge_type_ids,
            "mean_by_edge_type": np.array([]),
            "std_by_edge_type": np.array([]),
            "per_sample": np.array([]) if include_per_sample else None,
            "n_samples": n_samples,
            "n_layers": n_layers,
        }

    # Collect per-sample, per-edge-type, per-layer mean attention
    # Shape: [n_samples, n_edge_types, n_layers, n_heads]
    per_sample_array = np.full((n_samples, len(edge_type_ids), n_layers, n_heads), np.nan)

    for s_idx, subject_attn in enumerate(per_subject_attention):
        for l_idx, layer_attn in enumerate(subject_attn):
            for et_idx, et in enumerate(edge_type_ids):
                if et in layer_attn:
                    attn = layer_attn[et]
                    per_sample_array[s_idx, et_idx, l_idx, :] = attn.mean(axis=0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        attention_per_sample = np.nanmean(per_sample_array, axis=2)  # [n_samples, n_edge_types, n_heads]
        mean_by_edge_type = np.nanmean(attention_per_sample, axis=0)  # [n_edge_types, n_heads]
        std_by_edge_type = np.nanstd(attention_per_sample, axis=0)

    return {
        "edge_type_ids": edge_type_ids,
        "mean_by_edge_type": mean_by_edge_type,
        "std_by_edge_type": std_by_edge_type,
        "per_sample": per_sample_array if include_per_sample else None,
        "n_samples": n_samples,
        "n_layers": n_layers,
    }
