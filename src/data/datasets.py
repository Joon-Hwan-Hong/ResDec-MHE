"""
PyTorch Dataset for cognitive resilience model.

Handles loading and batching of:
- Pseudobulk expression [31 cell types × n_genes]
- Cell-cell communication graph features
- Cell-level data for Set Transformer
- Pathology scores and cognition targets
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from anndata import AnnData

from src.visualization.config import CELL_TYPE_ORDER


class CognitiveResilienceDataset(Dataset):
    """
    Dataset for cognitive resilience prediction from snRNA-seq data.

    Each sample is a subject with:
    - Pseudobulk expression per cell type
    - CCC graph features (from LIANA+)
    - Cell-level data for selected cell types
    - Pathology scores (amyloid, tau, global)
    - Cognition target
    """

    def __init__(
        self,
        adata: AnnData,
        metadata: pd.DataFrame,
        subject_ids: list[str],
        liana_results: dict[str, pd.DataFrame] | None = None,
        cell_type_column: str = "supercluster_name",
        subject_column: str = "ROSMAP_IndividualID",
        target_column: str = "cogn_global",
        pathology_columns: list[str] | None = None,
        cell_type_order: list[str] | None = None,
        max_cells_per_type: int = 1000,
        min_cells_threshold: int = 50,
        selected_cell_types: list[str] | None = None,
        transform: Any = None,
    ):
        """
        Initialize dataset.

        Args:
            adata: Preprocessed AnnData (normalized, log-transformed, filtered to HVG+LR)
            metadata: Subject-level metadata with phenotypes
            subject_ids: List of subject IDs to include in this dataset
            liana_results: Dict mapping subject_id -> LIANA+ results DataFrame
            cell_type_column: Column in adata.obs for cell type labels
            subject_column: Column in adata.obs for subject IDs
            target_column: Column in metadata for cognition target
            pathology_columns: Columns in metadata for pathology scores
            cell_type_order: Ordered list of cell types (default: CELL_TYPE_ORDER)
            max_cells_per_type: Maximum cells to sample per cell type
            min_cells_threshold: Minimum cells needed to include cell type
            selected_cell_types: Cell types for Set Transformer (subset of cell_type_order)
            transform: Optional transform to apply to samples
        """
        self.adata = adata
        self.metadata = metadata.set_index(subject_column) if subject_column in metadata.columns else metadata
        self.subject_ids = list(subject_ids)
        self.liana_results = liana_results or {}

        self.cell_type_column = cell_type_column
        self.subject_column = subject_column
        self.target_column = target_column
        self.pathology_columns = pathology_columns or ["gpath", "amylsqrt", "tangsqrt"]

        self.cell_type_order = cell_type_order or CELL_TYPE_ORDER
        self.n_cell_types = len(self.cell_type_order)
        self.ct_to_idx = {ct: idx for idx, ct in enumerate(self.cell_type_order)}

        self.max_cells_per_type = max_cells_per_type
        self.min_cells_threshold = min_cells_threshold
        self.selected_cell_types = selected_cell_types or self.cell_type_order[:8]
        self.selected_ct_indices = [self.ct_to_idx[ct] for ct in self.selected_cell_types if ct in self.ct_to_idx]

        self.transform = transform

        # Cache gene count
        self.n_genes = adata.n_vars

        # Validate subjects exist in both adata and metadata
        self._validate_subjects()

    def _validate_subjects(self):
        """Validate that all subject IDs exist in data."""
        adata_subjects = set(self.adata.obs[self.subject_column].unique())
        metadata_subjects = set(self.metadata.index)

        valid_subjects = []
        for sid in self.subject_ids:
            if sid in adata_subjects and sid in metadata_subjects:
                valid_subjects.append(sid)

        if len(valid_subjects) < len(self.subject_ids):
            n_removed = len(self.subject_ids) - len(valid_subjects)
            print(f"Warning: Removed {n_removed} subjects not found in adata or metadata")

        self.subject_ids = valid_subjects

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a single subject's data.

        Returns:
            Dictionary with:
            - pseudobulk: [n_cell_types, n_genes] expression tensor
            - cell_type_mask: [n_cell_types] bool tensor (True if cell type present)
            - pathology: [n_pathology] pathology scores
            - target: [1] cognition score
            - edge_index: [2, n_edges] graph edges (if LIANA available)
            - edge_type: [n_edges] edge type indices
            - edge_attr: [n_edges, 1] edge attributes
            - cells: [n_selected_types, max_cells, n_genes] cell-level data
            - cell_mask: [n_selected_types, max_cells] valid cell mask
            - subject_id: string identifier
        """
        subject_id = self.subject_ids[idx]

        # Get subject's cells
        subject_mask = self.adata.obs[self.subject_column] == subject_id
        adata_subject = self.adata[subject_mask]

        # ─────────────────────────────────────────────────────────────────────
        # Pseudobulk expression
        # ─────────────────────────────────────────────────────────────────────
        pseudobulk, cell_type_mask = self._compute_pseudobulk(adata_subject)

        # ─────────────────────────────────────────────────────────────────────
        # Cell-level data for Set Transformer
        # ─────────────────────────────────────────────────────────────────────
        cells, cell_mask = self._get_cell_level_data(adata_subject)

        # ─────────────────────────────────────────────────────────────────────
        # CCC graph features
        # ─────────────────────────────────────────────────────────────────────
        edge_index, edge_type, edge_attr = self._get_graph_features(subject_id)

        # ─────────────────────────────────────────────────────────────────────
        # Phenotypes
        # ─────────────────────────────────────────────────────────────────────
        pathology = self._get_pathology(subject_id)
        target = self._get_target(subject_id)

        sample = {
            "pseudobulk": torch.from_numpy(pseudobulk).float(),
            "cell_type_mask": torch.from_numpy(cell_type_mask).bool(),
            "pathology": torch.from_numpy(pathology).float(),
            "target": torch.tensor([target], dtype=torch.float32),
            "edge_index": torch.from_numpy(edge_index).long(),
            "edge_type": torch.from_numpy(edge_type).long(),
            "edge_attr": torch.from_numpy(edge_attr).float(),
            "cells": torch.from_numpy(cells).float(),
            "cell_mask": torch.from_numpy(cell_mask).bool(),
            "subject_id": subject_id,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _compute_pseudobulk(self, adata_subject: AnnData) -> tuple[np.ndarray, np.ndarray]:
        """Compute pseudobulk expression for each cell type."""
        pseudobulk = np.zeros((self.n_cell_types, self.n_genes), dtype=np.float32)
        cell_type_mask = np.zeros(self.n_cell_types, dtype=bool)

        # Get expression matrix
        X = adata_subject.X
        if hasattr(X, "toarray"):
            X = X.toarray()

        for ct_idx, ct_name in enumerate(self.cell_type_order):
            ct_mask = adata_subject.obs[self.cell_type_column] == ct_name
            n_cells = ct_mask.sum()

            if n_cells > 0:
                pseudobulk[ct_idx] = X[ct_mask.values].mean(axis=0)
                cell_type_mask[ct_idx] = True

        return pseudobulk, cell_type_mask

    def _get_cell_level_data(self, adata_subject: AnnData) -> tuple[np.ndarray, np.ndarray]:
        """Get cell-level expression for selected cell types."""
        n_selected = len(self.selected_cell_types)
        cells = np.zeros((n_selected, self.max_cells_per_type, self.n_genes), dtype=np.float32)
        cell_mask = np.zeros((n_selected, self.max_cells_per_type), dtype=bool)

        # Get expression matrix
        X = adata_subject.X
        if hasattr(X, "toarray"):
            X = X.toarray()

        for i, ct_name in enumerate(self.selected_cell_types):
            ct_mask = adata_subject.obs[self.cell_type_column] == ct_name
            ct_indices = np.where(ct_mask.values)[0]
            n_cells = len(ct_indices)

            if n_cells < self.min_cells_threshold:
                # Not enough cells, leave as zeros
                continue

            # Sample or take all cells
            if n_cells > self.max_cells_per_type:
                sampled_indices = np.random.choice(ct_indices, self.max_cells_per_type, replace=False)
            else:
                sampled_indices = ct_indices

            n_sampled = len(sampled_indices)
            cells[i, :n_sampled] = X[sampled_indices]
            cell_mask[i, :n_sampled] = True

        return cells, cell_mask

    def _get_graph_features(self, subject_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get CCC graph features from LIANA+ results."""
        if subject_id not in self.liana_results or self.liana_results[subject_id].empty:
            # Return empty graph
            return (
                np.zeros((2, 0), dtype=np.int64),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0, 1), dtype=np.float32),
            )

        from src.data.liana_processing import build_subject_ccc_features

        liana_df = self.liana_results[subject_id]
        features = build_subject_ccc_features(liana_df, self.cell_type_order)

        return features["edge_index"], features["edge_type"], features["edge_attr"]

    def _get_pathology(self, subject_id: str) -> np.ndarray:
        """Get pathology scores for subject."""
        pathology = np.zeros(len(self.pathology_columns), dtype=np.float32)

        if subject_id in self.metadata.index:
            for i, col in enumerate(self.pathology_columns):
                if col in self.metadata.columns:
                    val = self.metadata.loc[subject_id, col]
                    pathology[i] = 0.0 if pd.isna(val) else float(val)

        return pathology

    def _get_target(self, subject_id: str) -> float:
        """Get cognition target for subject."""
        if subject_id in self.metadata.index:
            val = self.metadata.loc[subject_id, self.target_column]
            return 0.0 if pd.isna(val) else float(val)
        return 0.0

    def get_gene_names(self) -> list[str]:
        """Get gene names in order."""
        return list(self.adata.var_names)

    def get_cell_type_names(self) -> list[str]:
        """Get cell type names in order."""
        return self.cell_type_order

    def get_metadata_for_subjects(self) -> pd.DataFrame:
        """Get metadata subset for subjects in this dataset."""
        return self.metadata.loc[self.metadata.index.isin(self.subject_ids)]


class PrecomputedDataset(Dataset):
    """
    Dataset loading precomputed features from disk.

    Use this for faster training after initial preprocessing.
    """

    def __init__(
        self,
        feature_dir: str | Path,
        subject_ids: list[str],
        metadata: pd.DataFrame,
        subject_column: str = "ROSMAP_IndividualID",
        target_column: str = "cogn_global",
        pathology_columns: list[str] | None = None,
    ):
        """
        Initialize from precomputed features.

        Args:
            feature_dir: Directory containing precomputed .npz files
            subject_ids: List of subject IDs
            metadata: Subject metadata
            subject_column: Column for subject IDs
            target_column: Column for cognition target
            pathology_columns: Columns for pathology scores
        """
        self.feature_dir = Path(feature_dir)
        self.subject_ids = list(subject_ids)
        self.metadata = metadata.set_index(subject_column) if subject_column in metadata.columns else metadata

        self.target_column = target_column
        self.pathology_columns = pathology_columns or ["gpath", "amylsqrt", "tangsqrt"]

        # Validate files exist
        self._validate_files()

    def _validate_files(self):
        """Check that feature files exist for all subjects."""
        valid_subjects = []
        for sid in self.subject_ids:
            feature_file = self.feature_dir / f"{sid}.npz"
            if feature_file.exists() and sid in self.metadata.index:
                valid_subjects.append(sid)

        if len(valid_subjects) < len(self.subject_ids):
            n_removed = len(self.subject_ids) - len(valid_subjects)
            print(f"Warning: Removed {n_removed} subjects without feature files")

        self.subject_ids = valid_subjects

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Load precomputed features for a subject."""
        subject_id = self.subject_ids[idx]
        feature_file = self.feature_dir / f"{subject_id}.npz"

        # Load features
        data = np.load(feature_file, allow_pickle=True)

        # Get phenotypes from metadata
        pathology = np.zeros(len(self.pathology_columns), dtype=np.float32)
        for i, col in enumerate(self.pathology_columns):
            if col in self.metadata.columns:
                val = self.metadata.loc[subject_id, col]
                pathology[i] = 0.0 if pd.isna(val) else float(val)

        target = self.metadata.loc[subject_id, self.target_column]
        target = 0.0 if pd.isna(target) else float(target)

        return {
            "pseudobulk": torch.from_numpy(data["pseudobulk"]).float(),
            "cell_type_mask": torch.from_numpy(data["cell_type_mask"]).bool(),
            "pathology": torch.from_numpy(pathology).float(),
            "target": torch.tensor([target], dtype=torch.float32),
            "edge_index": torch.from_numpy(data["edge_index"]).long(),
            "edge_type": torch.from_numpy(data["edge_type"]).long(),
            "edge_attr": torch.from_numpy(data["edge_attr"]).float(),
            "cells": torch.from_numpy(data["cells"]).float(),
            "cell_mask": torch.from_numpy(data["cell_mask"]).bool(),
            "subject_id": subject_id,
        }


def save_precomputed_features(
    dataset: CognitiveResilienceDataset,
    output_dir: str | Path,
    verbose: bool = True,
) -> None:
    """
    Save precomputed features to disk for faster loading.

    Args:
        dataset: CognitiveResilienceDataset to precompute
        output_dir: Directory to save .npz files
        verbose: Print progress
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(len(dataset)):
        sample = dataset[i]
        subject_id = sample["subject_id"]

        output_file = output_dir / f"{subject_id}.npz"

        np.savez_compressed(
            output_file,
            pseudobulk=sample["pseudobulk"].numpy(),
            cell_type_mask=sample["cell_type_mask"].numpy(),
            edge_index=sample["edge_index"].numpy(),
            edge_type=sample["edge_type"].numpy(),
            edge_attr=sample["edge_attr"].numpy(),
            cells=sample["cells"].numpy(),
            cell_mask=sample["cell_mask"].numpy(),
        )

        if verbose and (i + 1) % 50 == 0:
            print(f"Saved {i + 1}/{len(dataset)} subjects")

    if verbose:
        print(f"Saved all {len(dataset)} subjects to {output_dir}")