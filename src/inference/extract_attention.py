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


def save_attention_weights_hdf5(
    weights: AttentionWeights,
    path: str | Path,
    gene_names: list[str] | None = None,
) -> None:
    """
    Save all attention weights to HDF5 file.

    Args:
        weights: AttentionWeights container
        path: Output HDF5 path
        gene_names: Optional gene names to include
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        # Schema version
        f.attrs["schema_version"] = "1.0"

        # Gene gate weights
        f.create_dataset(
            "gene_gate",
            data=weights.gene_gate,
            compression="gzip",
            compression_opts=4,
        )
        f["gene_gate"].attrs["shape"] = "[n_cell_types, n_genes]"

        # Cell type selection weights
        if weights.cell_type_selection is not None:
            f.create_dataset("cell_type_selection", data=weights.cell_type_selection)
            f["cell_type_selection"].attrs["shape"] = "[n_cell_types]"

        # Region weights
        if weights.region_weights is not None:
            f.create_dataset("region_weights", data=weights.region_weights)
            f["region_weights"].attrs["shape"] = "[n_regions]"

        # Pathology attention (if available)
        if weights.pathology_attention is not None:
            f.create_dataset(
                "pathology_attention",
                data=weights.pathology_attention,
                compression="gzip",
                compression_opts=4,
            )
            f["pathology_attention"].attrs["shape"] = "[n_subjects, n_heads, n_cell_types]"

        # Cell type names
        cell_types_encoded = np.array(weights.cell_type_names, dtype="S64")
        f.create_dataset("cell_type_names", data=cell_types_encoded)

        # Gene names (if provided)
        if gene_names is not None:
            gene_names_encoded = np.array(gene_names, dtype="S64")
            f.create_dataset("gene_names", data=gene_names_encoded)

        # Subject IDs (if available)
        if weights.subject_ids is not None:
            subject_ids_encoded = np.array(weights.subject_ids, dtype="S64")
            f.create_dataset("subject_ids", data=subject_ids_encoded)

        # HGT layer scales
        if weights.hgt_layer_scales is not None:
            hgt_group = f.create_group("hgt_layer_scales")
            for key, value in weights.hgt_layer_scales.items():
                if isinstance(value, np.ndarray):
                    hgt_group.create_dataset(key, data=value)

    logger.info(f"Saved attention weights to {path}")


def load_attention_weights_hdf5(path: str | Path) -> AttentionWeights:
    """
    Load attention weights from HDF5 file.

    Args:
        path: Path to HDF5 file

    Returns:
        AttentionWeights container
    """
    with h5py.File(path, "r") as f:
        gene_gate = f["gene_gate"][:]

        cell_type_selection = None
        if "cell_type_selection" in f:
            cell_type_selection = f["cell_type_selection"][:]

        region_weights = None
        if "region_weights" in f:
            region_weights = f["region_weights"][:]

        pathology_attention = None
        if "pathology_attention" in f:
            pathology_attention = f["pathology_attention"][:]

        cell_type_names = None
        if "cell_type_names" in f:
            cell_type_names = [x.decode("utf-8") for x in f["cell_type_names"][:]]

        subject_ids = None
        if "subject_ids" in f:
            subject_ids = [x.decode("utf-8") for x in f["subject_ids"][:]]

        hgt_layer_scales = None
        if "hgt_layer_scales" in f:
            hgt_layer_scales = {}
            for key in f["hgt_layer_scales"].keys():
                hgt_layer_scales[key] = f["hgt_layer_scales"][key][:]

    return AttentionWeights(
        gene_gate=gene_gate,
        cell_type_selection=cell_type_selection,
        region_weights=region_weights,
        pathology_attention=pathology_attention,
        hgt_layer_scales=hgt_layer_scales,
        subject_ids=subject_ids,
        cell_type_names=cell_type_names,
    )
