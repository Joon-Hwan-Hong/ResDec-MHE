"""
PyTorch Dataset for cognitive resilience model.

Handles loading and batching of:
- Pseudobulk expression [31 cell types × n_genes]
- Cell-cell communication graph features
- Cell-level data for ALL 31 cell types (model learns which are important)
- Pathology scores and cognition targets

Design Decisions:

1. Cell type selection (2026-01-26): Dataset provides cells for ALL 31 cell types.
   CellTypeSelector in the model learns which types are most relevant for
   predicting cognitive resilience. This enables end-to-end learning of cell
   type importance rather than requiring a priori biological assumptions.

2. Mask semantics (2026-01-27): Two distinct masks serve different purposes:

   - cell_type_mask [n_cell_types]: True if ANY cells exist for this type (>0).
     Used by Pseudobulk and HGT branches. Even 1 cell provides meaningful
     pseudobulk (mean expression) and allows HGT message passing.

   - cell_mask [n_cell_types, max_cells]: True for valid sampled cells.
     Used by SetTransformer branch. Cell types with fewer than min_cells_threshold
     (default: 50) get all-False masks because modeling within-cell-type
     heterogeneity requires sufficient cell counts.

   This design allows each branch to use data appropriate to its requirements:
   - Pseudobulk branch: Uses all cell types with any cells
   - HGT branch: Uses pseudobulk as node features for all present cell types
   - SetTransformer branch: Only processes cell types with ≥50 cells;
     others receive learned empty_embedding (no NaN, fully differentiable)
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from anndata import AnnData

from src.data.constants import CELL_TYPE_ORDER, REGION_ORDER
from src.data.cell_sampling import CellSampler


class CognitiveResilienceDataset(Dataset):
    """
    Dataset for cognitive resilience prediction from snRNA-seq data.

    Each sample is a subject with:
    - Pseudobulk expression per cell type
    - CCC graph features (from LIANA+)
    - Cell-level data for all 31 cell types
    - Pathology scores (amyloid, tau, global)
    - Cognition target

    Mask semantics (see module docstring for rationale):
    - cell_type_mask: True if cell type has >0 cells (for Pseudobulk/HGT)
    - cell_mask: True for valid cells, requires ≥min_cells_threshold (for SetTransformer)
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
        sampling_strategy: str = "random",
        sampling_seed: int = 42,
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
            cell_type_order: Ordered list of cell types (default: CELL_TYPE_ORDER, 31 types)
            max_cells_per_type: Maximum cells to sample per cell type
            min_cells_threshold: Minimum cells needed for valid cell-level data
            sampling_strategy: Strategy for cell sampling ("random", "stratified", "importance")
            sampling_seed: Random seed for reproducible cell sampling
            transform: Optional transform to apply to samples

        Note:
            Cell-level data is provided for ALL cell types. The model's CellTypeSelector
            learns which types are most relevant for prediction. This is a design decision
            from 2026-01-26 to enable end-to-end learning of cell type importance.
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

        self.transform = transform

        # Initialize cell sampler for reproducible sampling
        self.sampler = CellSampler(
            max_cells_per_type=max_cells_per_type,
            min_cells_threshold=min_cells_threshold,
            strategy=sampling_strategy,
            seed=sampling_seed,
        )

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
            - subject_id: string identifier
            - pseudobulk: [n_cell_types, n_genes] expression tensor (31 types)
            - cell_type_mask: [n_cell_types] bool tensor (True if cell type present)
            - cell_counts: [n_cell_types] long tensor (cell count per type)
            - cells: [n_cell_types, max_cells, n_genes] cell-level data for ALL 31 types
            - cell_mask: [n_cell_types, max_cells] valid cell mask for ALL 31 types
            - ccc_edge_index: [2, n_edges] graph edges (if LIANA available)
            - ccc_edge_type: [n_edges] edge type indices
            - ccc_edge_attr: [n_edges, 1] edge attributes (LIANA magnitude)
            - pathology: [n_pathology] pathology scores
            - cognition: [1] cognition score
            - region_mask: [n_regions] bool tensor (True if region has cells)

        Note:
            Cell-level data is provided for ALL 31 cell types. Cell types with
            fewer cells than min_cells_threshold will have all-False masks.
            The model's CellTypeSelector learns which types to use.
        """
        subject_id = self.subject_ids[idx]

        # Get subject's cells
        subject_mask = self.adata.obs[self.subject_column] == subject_id
        adata_subject = self.adata[subject_mask]

        # ─────────────────────────────────────────────────────────────────────
        # Pseudobulk expression
        # ─────────────────────────────────────────────────────────────────────
        pseudobulk, cell_type_mask, cell_counts = self._compute_pseudobulk(adata_subject)

        # ─────────────────────────────────────────────────────────────────────
        # Cell-level data for Set Transformer
        # ─────────────────────────────────────────────────────────────────────
        cells, cell_mask = self._get_cell_level_data(adata_subject)

        # ─────────────────────────────────────────────────────────────────────
        # CCC graph features
        # ─────────────────────────────────────────────────────────────────────
        edge_index, edge_type, edge_attr = self._get_graph_features(subject_id)

        # ─────────────────────────────────────────────────────────────────────
        # Region mask
        # ─────────────────────────────────────────────────────────────────────
        region_mask = self._get_region_mask(adata_subject)

        # ─────────────────────────────────────────────────────────────────────
        # Phenotypes
        # ─────────────────────────────────────────────────────────────────────
        pathology = self._get_pathology(subject_id)
        target = self._get_target(subject_id)

        sample = {
            "subject_id": subject_id,
            "pseudobulk": torch.from_numpy(pseudobulk).float(),
            "cell_type_mask": torch.from_numpy(cell_type_mask).bool(),
            "cell_counts": torch.from_numpy(cell_counts).long(),
            "region_mask": torch.from_numpy(region_mask).bool(),
            # Cell-level data for ALL 31 types (model selects which to use)
            "cells": torch.from_numpy(cells).float(),
            "cell_mask": torch.from_numpy(cell_mask).bool(),
            # Graph features (CCC = cell-cell communication)
            "ccc_edge_index": torch.from_numpy(edge_index).long(),
            "ccc_edge_type": torch.from_numpy(edge_type).long(),
            "ccc_edge_attr": torch.from_numpy(edge_attr).float(),
            # Phenotypes
            "pathology": torch.from_numpy(pathology).float(),
            "cognition": torch.tensor([target], dtype=torch.float32),
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _compute_pseudobulk(self, adata_subject: AnnData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute pseudobulk expression and cell counts for each cell type."""
        pseudobulk = np.zeros((self.n_cell_types, self.n_genes), dtype=np.float32)
        cell_type_mask = np.zeros(self.n_cell_types, dtype=bool)
        cell_counts = np.zeros(self.n_cell_types, dtype=np.int64)

        # Get expression matrix
        X = adata_subject.X
        if hasattr(X, "toarray"):
            X = X.toarray()

        for ct_idx, ct_name in enumerate(self.cell_type_order):
            ct_mask = adata_subject.obs[self.cell_type_column] == ct_name
            n_cells = ct_mask.sum()

            cell_counts[ct_idx] = n_cells

            if n_cells > 0:
                pseudobulk[ct_idx] = X[ct_mask.values].mean(axis=0)
                cell_type_mask[ct_idx] = True

        return pseudobulk, cell_type_mask, cell_counts

    def _get_region_mask(self, adata_subject: AnnData) -> np.ndarray:
        """Create boolean mask indicating which brain regions are present for this subject.

        If BrainRegion column is missing, defaults to first region (PFC) only.
        This handles single-region datasets like DLPFC-only data.
        """
        region_mask = np.zeros(len(REGION_ORDER), dtype=bool)

        # Check if BrainRegion column exists
        if "BrainRegion" not in adata_subject.obs.columns:
            # Default to first region (PFC) if column missing
            region_mask[0] = True
            return region_mask

        # Get the unique regions present in this subject's cells
        present_regions = set(adata_subject.obs["BrainRegion"].unique())

        for idx, region in enumerate(REGION_ORDER):
            if region in present_regions:
                region_mask[idx] = True

        # If no regions matched (e.g., different naming convention), default to first
        if not region_mask.any():
            region_mask[0] = True

        return region_mask

    def _get_cell_level_data(self, adata_subject: AnnData) -> tuple[np.ndarray, np.ndarray]:
        """
        Get cell-level expression for ALL cell types using CellSampler.

        Returns data for all 31 cell types. Cell types with fewer cells than
        min_cells_threshold will have empty data (all-False mask). The model's
        CellTypeSelector learns which types to use for prediction.
        """
        # Allocate for ALL cell types (not just a subset)
        cells = np.zeros((self.n_cell_types, self.max_cells_per_type, self.n_genes), dtype=np.float32)
        cell_mask = np.zeros((self.n_cell_types, self.max_cells_per_type), dtype=bool)

        # Use CellSampler for reproducible sampling across ALL cell types
        sampled_indices = self.sampler.sample(
            adata_subject,
            cell_type_column=self.cell_type_column,
            cell_types=self.cell_type_order,  # ALL 31 types
        )

        # Get expression matrix
        X = adata_subject.X
        if hasattr(X, "toarray"):
            X = X.toarray()

        for i, ct_name in enumerate(self.cell_type_order):
            indices = sampled_indices.get(ct_name, np.array([], dtype=np.int64))
            n_sampled = len(indices)

            if n_sampled > 0:
                cells[i, :n_sampled] = X[indices]
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
            "subject_id": subject_id,
            "pseudobulk": torch.from_numpy(data["pseudobulk"]).float(),
            "cell_type_mask": torch.from_numpy(data["cell_type_mask"]).bool(),
            "cells": torch.from_numpy(data["cells"]).float(),
            "cell_mask": torch.from_numpy(data["cell_mask"]).bool(),
            # Graph features (CCC = cell-cell communication)
            "ccc_edge_index": torch.from_numpy(data["edge_index"]).long(),
            "ccc_edge_type": torch.from_numpy(data["edge_type"]).long(),
            "ccc_edge_attr": torch.from_numpy(data["edge_attr"]).float(),
            # Phenotypes
            "pathology": torch.from_numpy(pathology).float(),
            "cognition": torch.tensor([target], dtype=torch.float32),
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
            # Note: We keep the npz keys as edge_* for backward compatibility
            # with existing precomputed files. The PrecomputedDataset maps
            # these to ccc_edge_* when loading.
            edge_index=sample["ccc_edge_index"].numpy(),
            edge_type=sample["ccc_edge_type"].numpy(),
            edge_attr=sample["ccc_edge_attr"].numpy(),
            cells=sample["cells"].numpy(),
            cell_mask=sample["cell_mask"].numpy(),
        )

        if verbose and (i + 1) % 50 == 0:
            print(f"Saved {i + 1}/{len(dataset)} subjects")

    if verbose:
        print(f"Saved all {len(dataset)} subjects to {output_dir}")