"""
HGT attention aggregation for CCC importance analysis.

Aggregates per-sample, per-layer HGT attention weights into summary statistics
suitable for downstream cell communication analysis.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def aggregate_hgt_attention(
    hgt_attention: list[list[dict]],
    edge_types: list[tuple[str, str, str]] | None = None,
    include_per_sample: bool = True,
) -> dict[str, np.ndarray | list[str]]:
    """
    Aggregate HGT attention across samples.

    Computes mean and std of attention weights for each edge type across all samples.
    Optionally includes per-sample summaries to preserve subject-level variation.

    Design Note (2026-02):
        Full per-edge attention storage was not implemented because edge counts vary
        per subject (different CCC graphs), making rectangular tensor storage infeasible
        without padding or ragged arrays. Instead, we store per-sample summaries that
        aggregate within each sample while preserving between-sample variation.

    Args:
        hgt_attention: List of per-sample attention, where each sample contains
                      a list of per-layer dicts mapping edge_type -> [n_edges, n_heads]
        edge_types: Optional list of edge types to include. If None, discovers
                    from data with deterministic sorted ordering.
        include_per_sample: Whether to include per-sample per-layer summaries (default: True)

    Returns:
        Dict with:
            - 'edge_type_names': List of string representations of edge types
            - 'mean_by_edge_type': [n_edge_types, n_heads] mean attention per edge type
            - 'std_by_edge_type': [n_edge_types, n_heads] std attention per edge type
            - 'per_sample': [n_samples, n_edge_types, n_layers, n_heads] per-sample summaries (if include_per_sample)
            - 'n_samples': Number of samples
            - 'n_layers': Number of HGT layers
    """
    if not hgt_attention or len(hgt_attention) == 0:
        return {
            "edge_type_names": [],
            "mean_by_edge_type": np.array([]),
            "std_by_edge_type": np.array([]),
            "per_sample": np.array([]) if include_per_sample else None,
            "n_samples": 0,
            "n_layers": 0,
        }

    n_samples = len(hgt_attention)
    n_layers = len(hgt_attention[0]) if hgt_attention[0] else 0

    # Discover edge types from data with deterministic sort for reproducibility
    if edge_types is None:
        edge_type_set: set[tuple[str, str, str]] = set()
        for sample_attn in hgt_attention:
            for layer_attn in sample_attn:
                edge_type_set.update(layer_attn.keys())
        edge_types = sorted(edge_type_set, key=str)

    if len(edge_types) == 0:
        return {
            "edge_type_names": [],
            "mean_by_edge_type": np.array([]),
            "std_by_edge_type": np.array([]),
            "per_sample": np.array([]) if include_per_sample else None,
            "n_samples": n_samples,
            "n_layers": n_layers,
        }

    # For each edge type, aggregate attention across samples and layers
    # We take the mean across edges within each sample, then aggregate across samples
    edge_type_names = [f"{et[0]}|{et[1]}|{et[2]}" for et in edge_types]

    # Determine n_heads from first available attention
    n_heads = None
    for sample_attn in hgt_attention:
        for layer_attn in sample_attn:
            for attn in layer_attn.values():
                if isinstance(attn, torch.Tensor):
                    n_heads = attn.shape[-1]
                else:
                    n_heads = attn.shape[-1]
                break
            if n_heads is not None:
                break
        if n_heads is not None:
            break

    if n_heads is None:
        n_heads = 4  # Default

    # Collect per-sample per-layer attention summaries
    # Shape: [n_samples, n_edge_types, n_layers, n_heads]
    # NaN init: absent edge types stay NaN instead of biasing means to zero
    per_sample_per_layer = np.full((n_samples, len(edge_types), n_layers, n_heads), np.nan)

    for sample_idx, sample_attn in enumerate(hgt_attention):
        for layer_idx, layer_attn in enumerate(sample_attn):
            for et_idx, et in enumerate(edge_types):
                if et in layer_attn:
                    attn = layer_attn[et]
                    if isinstance(attn, torch.Tensor):
                        attn = attn.cpu().numpy()
                    # Mean across edges for this edge type in this layer
                    per_sample_per_layer[sample_idx, et_idx, layer_idx, :] = attn.mean(axis=0)

    # Aggregate: nanmean across layers, then across samples
    # (absent edge types excluded from aggregation, not treated as zero)
    # Suppress "Mean of empty slice" warning — expected when edge type is
    # absent from all layers for a given sample (NaN output is correct).
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        attention_per_sample = np.nanmean(per_sample_per_layer, axis=2)  # [n_samples, n_edge_types, n_heads]
        mean_by_edge_type = np.nanmean(attention_per_sample, axis=0)  # [n_edge_types, n_heads]
        std_by_edge_type = np.nanstd(attention_per_sample, axis=0)    # [n_edge_types, n_heads]

    # Count non-NaN samples per edge type (for downstream transparency)
    n_samples_per_edge_type = np.sum(
        ~np.isnan(attention_per_sample[:, :, 0]), axis=0
    )  # [n_edge_types]

    return {
        "edge_type_names": edge_type_names,
        "mean_by_edge_type": mean_by_edge_type,
        "std_by_edge_type": std_by_edge_type,
        "per_sample": per_sample_per_layer if include_per_sample else None,
        "n_samples": n_samples,
        "n_layers": n_layers,
        "n_samples_per_edge_type": n_samples_per_edge_type,
    }
