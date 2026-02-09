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

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from anndata import AnnData

from src.data.constants import CELL_TYPE_ORDER, REGION_ORDER
from src.data.cell_sampling import CellSampler


def _validate_no_nan_columns(
    metadata: pd.DataFrame,
    subject_ids: list,
    columns: list[str],
    column_type: str,
) -> None:
    """Validate that specified columns have no NaN for included subjects.

    Args:
        metadata: Metadata DataFrame indexed by subject ID.
        subject_ids: List of subject IDs to check.
        columns: Column names to validate.
        column_type: Label for error messages (e.g., "target", "pathology").
    """
    for col in columns:
        if col not in metadata.columns:
            continue
        values = metadata.loc[metadata.index.isin(subject_ids), col]
        nan_values = values[values.isna()]
        if len(nan_values) > 0:
            nan_ids = list(nan_values.index[:10])
            suffix = f" (and {len(nan_values) - 10} more)" if len(nan_values) > 10 else ""
            raise ValueError(
                f"{len(nan_values)} subject(s) have NaN in {column_type} column "
                f"'{col}': {nan_ids}{suffix}. "
                f"Filter these subjects before constructing the dataset."
            )


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
        region_column: str = "BrainRegion",
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
            region_column: Column in adata.obs for brain region labels (for multi-region)
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
        self.region_column = region_column
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
        """Validate that all subject IDs exist in data and have valid targets/pathology."""
        adata_subjects = set(self.adata.obs[self.subject_column].unique())
        metadata_subjects = set(self.metadata.index)

        valid_subjects = []
        for sid in self.subject_ids:
            if sid in adata_subjects and sid in metadata_subjects:
                valid_subjects.append(sid)

        if len(valid_subjects) < len(self.subject_ids):
            n_removed = len(self.subject_ids) - len(valid_subjects)
            warnings.warn(
                f"Removed {n_removed} subjects not found in adata or metadata",
                UserWarning,
                stacklevel=2,
            )

        self.subject_ids = valid_subjects

        _validate_no_nan_columns(
            self.metadata, self.subject_ids,
            [self.target_column], "target",
        )
        _validate_no_nan_columns(
            self.metadata, self.subject_ids,
            self.pathology_columns, "pathology",
        )

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
        cells, cell_mask, cell_barcodes = self._get_cell_level_data(adata_subject)

        # ─────────────────────────────────────────────────────────────────────
        # CCC graph features
        # ─────────────────────────────────────────────────────────────────────
        edge_index, edge_type, edge_attr = self._get_graph_features(subject_id)

        # ─────────────────────────────────────────────────────────────────────
        # Region mask and multi-region pseudobulk
        # ─────────────────────────────────────────────────────────────────────
        region_mask = self._get_region_mask(adata_subject)
        region_pseudobulks, available_regions = self._compute_pseudobulk_by_region(adata_subject)

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
            "cell_barcodes": cell_barcodes,
            # Graph features (CCC = cell-cell communication)
            "ccc_edge_index": torch.from_numpy(edge_index).long(),
            "ccc_edge_type": torch.from_numpy(edge_type).long(),
            "ccc_edge_attr": torch.from_numpy(edge_attr).float(),
            # Phenotypes
            "pathology": torch.from_numpy(pathology).float(),
            "cognition": torch.tensor([target], dtype=torch.float32),
            # Metadata for collate - cell type ordering for edge index mapping
            "cell_type_order": self.cell_type_order,
        }

        # Add multi-region pseudobulk data (if BrainRegion column exists)
        for key, value in region_pseudobulks.items():
            sample[key] = torch.from_numpy(value).float()
        if available_regions:
            sample["available_regions"] = available_regions

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

    def _compute_pseudobulk_by_region(
        self, adata_subject: AnnData
    ) -> tuple[dict[str, np.ndarray], list[int]]:
        """
        Compute per-region pseudobulk for multi-region data.

        For subjects with cells from multiple brain regions, computes separate
        pseudobulk aggregations per region. This enables the model to learn
        region-specific expression patterns.

        Args:
            adata_subject: AnnData subset for one subject

        Returns:
            region_pseudobulks: Dict mapping "region_{idx}_pseudobulk" -> [n_cell_types, n_genes]
            available_regions: Sorted list of region indices with data
        """
        region_pseudobulks = {}
        available_regions = []

        # If no region column, return empty (single-region fallback)
        if self.region_column not in adata_subject.obs.columns:
            return region_pseudobulks, available_regions

        # Get expression matrix once
        X = adata_subject.X
        if hasattr(X, "toarray"):
            X = X.toarray()

        for region_idx, region_name in enumerate(REGION_ORDER):
            # Filter cells for this region
            region_mask = adata_subject.obs[self.region_column] == region_name
            if not region_mask.any():
                continue

            # Compute pseudobulk for this region
            pseudobulk = np.zeros((self.n_cell_types, self.n_genes), dtype=np.float32)
            X_region = X[region_mask.values]
            obs_region = adata_subject.obs[region_mask]

            for ct_idx, ct_name in enumerate(self.cell_type_order):
                ct_mask = obs_region[self.cell_type_column] == ct_name
                if ct_mask.sum() > 0:
                    pseudobulk[ct_idx] = X_region[ct_mask.values].mean(axis=0)

            region_pseudobulks[f"region_{region_idx}_pseudobulk"] = pseudobulk
            available_regions.append(region_idx)

        return region_pseudobulks, sorted(available_regions)

    def _get_region_mask(self, adata_subject: AnnData) -> np.ndarray:
        """Create boolean mask indicating which brain regions are present for this subject.

        If the region column is missing, defaults to first region (PFC) only.
        This handles single-region datasets like PFC-only data.
        """
        region_mask = np.zeros(len(REGION_ORDER), dtype=bool)

        # Check if region column exists
        if self.region_column not in adata_subject.obs.columns:
            # Default to first region (PFC) if column missing
            region_mask[0] = True
            return region_mask

        # Get the unique regions present in this subject's cells
        present_regions = set(adata_subject.obs[self.region_column].unique())

        for idx, region in enumerate(REGION_ORDER):
            if region in present_regions:
                region_mask[idx] = True

        # If no regions matched (e.g., different naming convention), default to first
        if not region_mask.any():
            region_mask[0] = True

        return region_mask

    def _get_cell_level_data(
        self, adata_subject: AnnData,
    ) -> tuple[np.ndarray, np.ndarray, list[list[str]]]:
        """
        Get cell-level expression for ALL cell types using CellSampler.

        Returns data for all 31 cell types. Cell types with fewer cells than
        min_cells_threshold will have empty data (all-False mask). The model's
        CellTypeSelector learns which types to use for prediction.

        Returns:
            cells: [n_cell_types, max_cells, n_genes] expression data
            cell_mask: [n_cell_types, max_cells] valid cell mask
            cell_barcodes: list of lists of barcode strings per cell type
        """
        # Allocate for ALL cell types (not just a subset)
        cells = np.zeros((self.n_cell_types, self.max_cells_per_type, self.n_genes), dtype=np.float32)
        cell_mask = np.zeros((self.n_cell_types, self.max_cells_per_type), dtype=bool)
        cell_barcodes: list[list[str]] = [[] for _ in range(self.n_cell_types)]

        # Use CellSampler for reproducible sampling across ALL cell types
        sampled_indices = self.sampler.sample(
            adata_subject,
            cell_type_column=self.cell_type_column,
            cell_types=self.cell_type_order,  # ALL 31 types
        )

        # Get expression matrix and obs index for barcodes
        X = adata_subject.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        obs_index = adata_subject.obs.index

        for i, ct_name in enumerate(self.cell_type_order):
            indices = sampled_indices.get(ct_name, np.array([], dtype=np.int64))
            n_sampled = len(indices)

            if n_sampled > 0:
                cells[i, :n_sampled] = X[indices]
                cell_mask[i, :n_sampled] = True
                cell_barcodes[i] = [str(obs_index[idx]) for idx in indices]

        return cells, cell_mask, cell_barcodes

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
        """Get pathology scores for subject.

        NaN values are validated at __init__ time and should never be encountered here.
        """
        pathology = np.zeros(len(self.pathology_columns), dtype=np.float32)

        if subject_id in self.metadata.index:
            for i, col in enumerate(self.pathology_columns):
                if col in self.metadata.columns:
                    val = self.metadata.loc[subject_id, col]
                    if pd.isna(val):
                        raise RuntimeError(
                            f"NaN in pathology column '{col}' for subject '{subject_id}'. "
                            f"This should have been caught at __init__ validation."
                        )
                    pathology[i] = float(val)

        return pathology

    def _get_target(self, subject_id: str) -> float:
        """Get cognition target for subject.

        NaN values are validated at __init__ time and should never be encountered here.
        """
        if subject_id in self.metadata.index:
            val = self.metadata.loc[subject_id, self.target_column]
            if pd.isna(val):
                raise RuntimeError(
                    f"NaN in target column '{self.target_column}' for subject '{subject_id}'. "
                    f"This should have been caught at __init__ validation."
                )
            return float(val)
        raise RuntimeError(
            f"Subject '{subject_id}' not found in metadata. "
            f"This should have been caught at __init__ validation."
        )

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
        cell_type_order: list[str] | None = None,
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
            cell_type_order: Order of cell types for edge index mapping.
                Must match the order used when precomputing features.
                Defaults to CELL_TYPE_ORDER from constants.
        """
        self.feature_dir = Path(feature_dir)
        self.subject_ids = list(subject_ids)
        self.metadata = metadata.set_index(subject_column) if subject_column in metadata.columns else metadata

        self.target_column = target_column
        self.pathology_columns = pathology_columns or ["gpath", "amylsqrt", "tangsqrt"]
        self.cell_type_order = cell_type_order or CELL_TYPE_ORDER

        # Validate files exist
        self._validate_files()

        # Validate no NaN in target or pathology columns
        self._validate_metadata()

    def _validate_files(self):
        """Check that feature files exist for all subjects."""
        valid_subjects = []
        for sid in self.subject_ids:
            feature_file = self.feature_dir / f"{sid}.npz"
            if feature_file.exists() and sid in self.metadata.index:
                valid_subjects.append(sid)

        if len(valid_subjects) < len(self.subject_ids):
            n_removed = len(self.subject_ids) - len(valid_subjects)
            warnings.warn(f"Removed {n_removed} subjects without feature files")

        self.subject_ids = valid_subjects

    def _validate_metadata(self):
        """Validate that target and pathology columns have no NaN for included subjects."""
        _validate_no_nan_columns(
            self.metadata, self.subject_ids,
            [self.target_column], "target",
        )
        _validate_no_nan_columns(
            self.metadata, self.subject_ids,
            self.pathology_columns, "pathology",
        )

    def get_gene_names(self) -> list[str] | None:
        """Get gene names from sidecar file if available.

        Looks for gene_names.npy in feature_dir. Returns None if not found,
        which causes downstream analysis to use synthetic gene_i labels.
        """
        gene_names_path = self.feature_dir / "gene_names.npy"
        if gene_names_path.exists():
            names = np.load(gene_names_path, allow_pickle=True)
            return [str(n) for n in names]
        return None

    def get_cell_type_names(self) -> list[str]:
        """Get cell type names in order."""
        return self.cell_type_order

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Load precomputed features for a subject."""
        subject_id = self.subject_ids[idx]
        feature_file = self.feature_dir / f"{subject_id}.npz"

        # Load features (context manager ensures file handle is closed)
        with np.load(feature_file, allow_pickle=True) as npz_data:
            # Validate cell_type_order matches (if stored in file)
            if "cell_type_order" in npz_data:
                saved_order = list(npz_data["cell_type_order"])
                if saved_order != self.cell_type_order:
                    raise ValueError(
                        f"Precomputed file {feature_file} was saved with different "
                        f"cell_type_order than expected. Edge indices may be incorrectly "
                        f"mapped. Saved: {saved_order[:3]}... Expected: {self.cell_type_order[:3]}..."
                    )

            # BACKWARD COMPAT: Pre-multi-region .npz files may lack cell_counts and
            # region_mask. Safe to remove after re-running precompute_features on all
            # subjects with multi-region support enabled.
            if "cell_counts" in npz_data:
                cell_counts = torch.from_numpy(npz_data["cell_counts"]).long()
            else:
                # Derive cell_counts from cell_mask: sum valid cells per cell type
                # cell_mask shape: [n_cell_types, max_cells] -> sum along dim=1
                cell_mask_np = npz_data["cell_mask"]  # [n_cell_types, max_cells] bool
                cell_counts = torch.from_numpy(cell_mask_np.sum(axis=1).astype(np.int64))

            if "region_mask" in npz_data:
                region_mask = torch.from_numpy(npz_data["region_mask"]).bool()
            else:
                # Default to first region only (PFC) for backward compatibility
                from src.data.constants import REGION_ORDER
                region_mask = torch.zeros(len(REGION_ORDER), dtype=torch.bool)
                region_mask[0] = True

            # Extract all arrays into tensors while file is open
            pseudobulk = torch.from_numpy(npz_data["pseudobulk"]).float()
            cell_type_mask = torch.from_numpy(npz_data["cell_type_mask"]).bool()
            cells = torch.from_numpy(npz_data["cells"]).float()
            cell_mask = torch.from_numpy(npz_data["cell_mask"]).bool()
            ccc_edge_index = torch.from_numpy(npz_data["edge_index"]).long()
            ccc_edge_type = torch.from_numpy(npz_data["edge_type"]).long()
            ccc_edge_attr = torch.from_numpy(npz_data["edge_attr"]).float()

            # Load multi-region pseudobulk data (if present in file)
            region_pseudobulks = {}
            for key in npz_data.files:
                if key.startswith("region_") and key.endswith("_pseudobulk"):
                    region_pseudobulks[key] = torch.from_numpy(npz_data[key]).float()

            available_regions = (
                list(npz_data["available_regions"])
                if "available_regions" in npz_data
                else None
            )

        # Get phenotypes from metadata (no npz access needed)
        pathology = np.zeros(len(self.pathology_columns), dtype=np.float32)
        for i, col in enumerate(self.pathology_columns):
            if col in self.metadata.columns:
                val = self.metadata.loc[subject_id, col]
                if pd.isna(val):
                    raise RuntimeError(
                        f"NaN in pathology column '{col}' for subject '{subject_id}'. "
                        f"This should have been caught at __init__ validation."
                    )
                pathology[i] = float(val)

        target = self.metadata.loc[subject_id, self.target_column]
        if pd.isna(target):
            raise RuntimeError(
                f"NaN in target column '{self.target_column}' for subject '{subject_id}'. "
                f"This should have been caught at __init__ validation."
            )
        target = float(target)

        sample = {
            "subject_id": subject_id,
            "pseudobulk": pseudobulk,
            "cell_type_mask": cell_type_mask,
            "cell_counts": cell_counts,
            "region_mask": region_mask,
            "cells": cells,
            "cell_mask": cell_mask,
            # Graph features (CCC = cell-cell communication)
            "ccc_edge_index": ccc_edge_index,
            "ccc_edge_type": ccc_edge_type,
            "ccc_edge_attr": ccc_edge_attr,
            # Phenotypes
            "pathology": torch.from_numpy(pathology).float(),
            "cognition": torch.tensor([target], dtype=torch.float32),
            # Metadata for collate - cell type ordering for edge index mapping
            "cell_type_order": self.cell_type_order,
            **region_pseudobulks,
        }

        if available_regions is not None:
            sample["available_regions"] = available_regions

        return sample


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

        # Build save dict with core features
        save_data = {
            "pseudobulk": sample["pseudobulk"].numpy(),
            "cell_type_mask": sample["cell_type_mask"].numpy(),
            "cell_counts": sample["cell_counts"].numpy(),
            "region_mask": sample["region_mask"].numpy(),
            # Note: We keep the npz keys as edge_* for backward compatibility
            # with existing precomputed files. The PrecomputedDataset maps
            # these to ccc_edge_* when loading.
            "edge_index": sample["ccc_edge_index"].numpy(),
            "edge_type": sample["ccc_edge_type"].numpy(),
            "edge_attr": sample["ccc_edge_attr"].numpy(),
            "cells": sample["cells"].numpy(),
            "cell_mask": sample["cell_mask"].numpy(),
            # Store ordering metadata for validation on load
            "cell_type_order": np.array(sample["cell_type_order"], dtype=object),
        }

        # Add multi-region pseudobulk data (if present)
        for key in sample:
            if key.startswith("region_") and key.endswith("_pseudobulk"):
                save_data[key] = sample[key].numpy()

        if "available_regions" in sample:
            save_data["available_regions"] = np.array(sample["available_regions"])

        np.savez_compressed(output_file, **save_data)

        if verbose and (i + 1) % 50 == 0:
            print(f"Saved {i + 1}/{len(dataset)} subjects")

    if verbose:
        print(f"Saved all {len(dataset)} subjects to {output_dir}")