"""
Attention extraction utilities for interpretability analysis.

Extracts attention weights from all model components:
- Gene Gate: [n_cell_types, n_genes] - static learned weights
- HGT: [n_layers, n_heads, n_edges] per sample - CCC attention
- PMA (Set Transformer): [k_seeds, n_cells] per cell type - cell-level attention
- Pathology Attention: [n_heads, n_cell_types] per sample - cell type importance

Output formats:
- HDF5 for large tensors with compression
- DataFrames for analysis-ready formats
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch

from src.data.constants import CELL_TYPE_ORDER, ALL_EDGE_TYPES
from src.models.full_model import CognitiveResilienceModel

logger = logging.getLogger(__name__)


@dataclass
class AttentionWeights:
    """
    Container for all extracted attention weights.

    Attributes:
        gene_gate: [n_cell_types, n_genes] - static gene attention per cell type
        pathology_attention: [n_subjects, n_heads, n_cell_types] - cell type attention
        cell_type_selection: [n_cell_types] - learned cell type importance
        region_weights: [n_regions] - learned region importance
        hgt_layer_scales: Dict mapping cell type to layer scale values
        subject_ids: List of subject identifiers
        cell_type_names: List of cell type names
    """
    gene_gate: np.ndarray
    pathology_attention: np.ndarray | None = None
    cell_type_selection: np.ndarray | None = None
    region_weights: np.ndarray | None = None
    hgt_layer_scales: dict[str, np.ndarray] | None = None
    subject_ids: list[str] | None = None
    cell_type_names: list[str] | None = None

    def __post_init__(self):
        if self.cell_type_names is None:
            self.cell_type_names = list(CELL_TYPE_ORDER)


class AttentionExtractor:
    """
    Extract and format attention weights from trained model.

    Extracts both static weights (gene gate, cell type selection, region weights)
    and per-sample weights (pathology attention, HGT attention).

    Example:
        >>> extractor = AttentionExtractor(model)
        >>> weights = extractor.extract_static_weights()
        >>> extractor.save_gene_gate_weights(weights.gene_gate, "gene_gate.h5")
    """

    def __init__(self, model: CognitiveResilienceModel):
        """
        Initialize extractor with model.

        Args:
            model: Trained CognitiveResilienceModel
        """
        self.model = model
        self.model.eval()

        # Extract configuration
        self.n_cell_types = model.n_cell_types
        self.n_genes = model.n_genes
        self.cell_type_names = list(CELL_TYPE_ORDER)
        self.edge_types = list(ALL_EDGE_TYPES)

    @torch.no_grad()
    def extract_static_weights(self) -> AttentionWeights:
        """
        Extract static (learned) attention weights.

        These weights don't depend on input data - they're model parameters.

        Returns:
            AttentionWeights with gene_gate, cell_type_selection, region_weights
        """
        # Gene gate weights: [n_cell_types, n_genes]
        gene_gate = self.model.pseudobulk_encoder.gene_gate.get_gate_weights()
        gene_gate = gene_gate.cpu().numpy()

        # Cell type selection weights: [n_cell_types]
        cell_type_selection = self.model.cell_transformer.get_selection_weights()
        cell_type_selection = cell_type_selection.cpu().numpy()

        # Region weights: [n_regions]
        region_importance = self.model.get_region_importance()
        region_weights = np.array(list(region_importance.values()))

        # HGT layer scales (per cell type, per layer)
        hgt_layer_scales = self.model.get_hgt_layer_scales()
        hgt_layer_scales = {
            k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
            for k, v in hgt_layer_scales.items()
        }

        return AttentionWeights(
            gene_gate=gene_gate,
            cell_type_selection=cell_type_selection,
            region_weights=region_weights,
            hgt_layer_scales=hgt_layer_scales,
            cell_type_names=self.cell_type_names,
        )

    def gene_gate_to_dataframe(
        self,
        gene_gate: np.ndarray,
        gene_names: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Convert gene gate weights to tidy DataFrame.

        Args:
            gene_gate: [n_cell_types, n_genes] weight matrix
            gene_names: Optional list of gene names

        Returns:
            DataFrame with columns: cell_type, gene, gene_idx, weight
        """
        n_cell_types, n_genes = gene_gate.shape

        if gene_names is None:
            gene_names = [f"gene_{i}" for i in range(n_genes)]

        rows = []
        for ct_idx, ct_name in enumerate(self.cell_type_names[:n_cell_types]):
            for gene_idx, gene_name in enumerate(gene_names[:n_genes]):
                rows.append({
                    "cell_type": ct_name,
                    "gene": gene_name,
                    "gene_idx": gene_idx,
                    "weight": gene_gate[ct_idx, gene_idx],
                })

        return pd.DataFrame(rows)

    def get_top_genes_per_cell_type(
        self,
        gene_gate: np.ndarray,
        gene_names: list[str] | None = None,
        top_k: int = 100,
    ) -> pd.DataFrame:
        """
        Get top-k genes per cell type by attention weight.

        Args:
            gene_gate: [n_cell_types, n_genes] weight matrix
            gene_names: Optional list of gene names
            top_k: Number of top genes per cell type

        Returns:
            DataFrame with columns: cell_type, rank, gene, weight
        """
        n_cell_types, n_genes = gene_gate.shape

        if gene_names is None:
            gene_names = [f"gene_{i}" for i in range(n_genes)]

        rows = []
        for ct_idx, ct_name in enumerate(self.cell_type_names[:n_cell_types]):
            weights = gene_gate[ct_idx]
            top_indices = np.argsort(weights)[::-1][:top_k]

            for rank, gene_idx in enumerate(top_indices, 1):
                rows.append({
                    "cell_type": ct_name,
                    "rank": rank,
                    "gene": gene_names[gene_idx] if gene_idx < len(gene_names) else f"gene_{gene_idx}",
                    "gene_idx": int(gene_idx),
                    "weight": float(weights[gene_idx]),
                })

        return pd.DataFrame(rows)

    def cell_type_selection_to_dataframe(
        self,
        weights: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """
        Convert cell type selection weights to DataFrame.

        Args:
            weights: Optional [n_cell_types] weights, extracts from model if None

        Returns:
            DataFrame with columns: cell_type, weight, rank
        """
        if weights is None:
            weights = self.model.cell_transformer.get_selection_weights().cpu().numpy()

        df = pd.DataFrame({
            "cell_type": self.cell_type_names[:len(weights)],
            "weight": weights,
        })
        df = df.sort_values("weight", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)

        return df[["cell_type", "weight", "rank"]]

    def region_weights_to_dataframe(
        self,
        weights: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """
        Convert region weights to DataFrame.

        Args:
            weights: Optional [n_regions] weights, extracts from model if None

        Returns:
            DataFrame with columns: region, weight, rank
        """
        if weights is None:
            region_importance = self.model.get_region_importance()
            regions = list(region_importance.keys())
            weights = np.array(list(region_importance.values()))
        else:
            from src.data.constants import REGION_ORDER
            regions = list(REGION_ORDER)[:len(weights)]

        df = pd.DataFrame({
            "region": regions,
            "weight": weights,
        })
        df = df.sort_values("weight", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)

        return df[["region", "weight", "rank"]]

    def pathology_attention_to_dataframe(
        self,
        attention: np.ndarray,
        subject_ids: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Convert pathology attention weights to tidy DataFrame.

        Args:
            attention: [n_subjects, n_heads, n_cell_types] attention weights
            subject_ids: Optional list of subject identifiers

        Returns:
            DataFrame with columns: subject_id, head, cell_type, weight
        """
        n_subjects, n_heads, n_cell_types = attention.shape

        if subject_ids is None:
            subject_ids = [f"subject_{i}" for i in range(n_subjects)]

        rows = []
        for subj_idx, subj_id in enumerate(subject_ids):
            for head_idx in range(n_heads):
                for ct_idx, ct_name in enumerate(self.cell_type_names[:n_cell_types]):
                    rows.append({
                        "subject_id": subj_id,
                        "head": head_idx,
                        "cell_type": ct_name,
                        "weight": float(attention[subj_idx, head_idx, ct_idx]),
                    })

        return pd.DataFrame(rows)

    def aggregate_pathology_attention(
        self,
        attention: np.ndarray,
        subject_ids: list[str] | None = None,
        aggregation: str = "mean",
    ) -> pd.DataFrame:
        """
        Aggregate pathology attention across heads and/or subjects.

        Args:
            attention: [n_subjects, n_heads, n_cell_types] attention weights
            subject_ids: Optional list of subject identifiers
            aggregation: "mean" (across heads) or "per_head"

        Returns:
            DataFrame with aggregated attention per cell type
        """
        n_subjects, n_heads, n_cell_types = attention.shape

        if subject_ids is None:
            subject_ids = [f"subject_{i}" for i in range(n_subjects)]

        if aggregation == "mean":
            # Mean across heads: [n_subjects, n_cell_types]
            mean_attention = attention.mean(axis=1)

            rows = []
            for subj_idx, subj_id in enumerate(subject_ids):
                for ct_idx, ct_name in enumerate(self.cell_type_names[:n_cell_types]):
                    rows.append({
                        "subject_id": subj_id,
                        "cell_type": ct_name,
                        "mean_attention": float(mean_attention[subj_idx, ct_idx]),
                    })

            return pd.DataFrame(rows)

        else:
            # Keep per-head, return tidy format
            return self.pathology_attention_to_dataframe(attention, subject_ids)


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
        edge_types: Optional list of edge types to include. If None, extracts from data.
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

    # Collect all edge types from the data if not provided
    if edge_types is None:
        edge_type_set = set()
        for sample_attn in hgt_attention:
            for layer_attn in sample_attn:
                edge_type_set.update(layer_attn.keys())
        edge_types = sorted(list(edge_type_set))

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
    per_sample_per_layer = np.zeros((n_samples, len(edge_types), n_layers, n_heads))

    for sample_idx, sample_attn in enumerate(hgt_attention):
        for layer_idx, layer_attn in enumerate(sample_attn):
            for et_idx, et in enumerate(edge_types):
                if et in layer_attn:
                    attn = layer_attn[et]
                    if isinstance(attn, torch.Tensor):
                        attn = attn.cpu().numpy()
                    # Mean across edges for this edge type in this layer
                    per_sample_per_layer[sample_idx, et_idx, layer_idx, :] = attn.mean(axis=0)

    # Aggregate: mean across layers, then across samples
    attention_per_sample = per_sample_per_layer.mean(axis=2)  # [n_samples, n_edge_types, n_heads]
    mean_by_edge_type = attention_per_sample.mean(axis=0)  # [n_edge_types, n_heads]
    std_by_edge_type = attention_per_sample.std(axis=0)    # [n_edge_types, n_heads]

    return {
        "edge_type_names": edge_type_names,
        "mean_by_edge_type": mean_by_edge_type,
        "std_by_edge_type": std_by_edge_type,
        "per_sample": per_sample_per_layer if include_per_sample else None,
        "n_samples": n_samples,
        "n_layers": n_layers,
    }
