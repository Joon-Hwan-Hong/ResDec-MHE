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

import logging
import tempfile
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

logger = logging.getLogger(__name__)


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
            available = sorted(metadata.columns.tolist())
            if column_type == "target":
                raise ValueError(
                    f"Target column '{col}' not found in metadata. "
                    f"Available columns: {available}"
                )
            warnings.warn(
                f"{column_type} column '{col}' not found in metadata. "
                f"Values will default to 0.0 for all subjects. "
                f"Available columns: {available}",
                UserWarning,
                stacklevel=3,
            )
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
        max_missing_subject_fraction: float = 0.1,
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
            max_missing_subject_fraction: Maximum fraction of subjects allowed to be
                missing from adata/metadata before raising an error (default: 0.1 = 10%).
                Set via data.max_missing_subject_fraction in config.
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
        self.max_missing_subject_fraction = max_missing_subject_fraction

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

        # Pre-compute subject-to-row index mapping to avoid O(n_cells) string
        # scan in __getitem__. Single-pass O(n_cells) groupby instead of
        # O(n_subjects * n_cells) loop of np.where calls.
        obs_values = self.adata.obs[self.subject_column].values

        # Vectorized: use pd.Categorical for O(n_cells) integer encoding
        cat = pd.Categorical(obs_values, categories=self.subject_ids)
        codes = cat.codes  # -1 for cells not in subject_ids
        valid_mask = codes >= 0
        valid_indices = np.where(valid_mask)[0]
        valid_codes = codes[valid_mask]

        # Sort by subject code for grouped extraction
        order = np.argsort(valid_codes, kind='stable')
        sorted_indices = valid_indices[order]
        sorted_codes = valid_codes[order]

        # Split into per-subject arrays using searchsorted
        boundaries = np.searchsorted(sorted_codes, np.arange(len(self.subject_ids) + 1))
        self._subject_indices = {}
        for i, sid in enumerate(self.subject_ids):
            self._subject_indices[sid] = sorted_indices[boundaries[i]:boundaries[i + 1]]

        # Warn if using on-the-fly dataset with large AnnData — PrecomputedDataset
        # is the recommended path for training at scale.
        if adata.n_obs > 1_000_000:
            warnings.warn(
                f"CognitiveResilienceDataset initialized with {adata.n_obs:,} cells. "
                f"With num_workers > 0, each DataLoader worker forks the full AnnData "
                f"object (~{adata.n_obs * adata.n_vars * 4 / 1e9:.1f} GB if dense). "
                f"Use PrecomputedDataset with precomputed .npz files for training at scale.",
                UserWarning,
                stacklevel=2,
            )

    def _validate_subjects(self):
        """Validate that all subject IDs exist in data and have valid targets/pathology.

        Raises ValueError if too many subjects are missing (> max_missing_subject_fraction
        of the original list). This catches bulk data pipeline failures while allowing
        small expected mismatches from metadata/AnnData filtering.
        """
        adata_subjects = set(self.adata.obs[self.subject_column].unique())
        metadata_subjects = set(self.metadata.index)

        valid_subjects = []
        for sid in self.subject_ids:
            if sid in adata_subjects and sid in metadata_subjects:
                valid_subjects.append(sid)

        if len(valid_subjects) < len(self.subject_ids):
            n_removed = len(self.subject_ids) - len(valid_subjects)
            missing_fraction = n_removed / len(self.subject_ids) if self.subject_ids else 0.0
            max_fraction = self.max_missing_subject_fraction

            if missing_fraction > max_fraction:
                raise ValueError(
                    f"Too many subjects missing: {n_removed}/{len(self.subject_ids)} "
                    f"({missing_fraction:.1%}) exceeds threshold ({max_fraction:.0%}). "
                    f"Check that adata and metadata contain the expected subjects. "
                    f"Adjust data.max_missing_subject_fraction in config to override."
                )
            warnings.warn(
                f"Removed {n_removed} subjects not found in adata or metadata "
                f"({missing_fraction:.1%} of {len(self.subject_ids)})",
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

        # Pre-extract phenotypes to numpy arrays for O(1) __getitem__ access.
        self._sid_to_idx = {sid: i for i, sid in enumerate(self.subject_ids)}
        self._pathology_array = np.zeros(
            (len(self.subject_ids), len(self.pathology_columns)), dtype=np.float32
        )
        self._target_array = np.zeros(len(self.subject_ids), dtype=np.float32)
        for i, sid in enumerate(self.subject_ids):
            if sid in self.metadata.index:
                for j, col in enumerate(self.pathology_columns):
                    if col in self.metadata.columns:
                        val = self.metadata.loc[sid, col]
                        if not pd.isna(val):
                            self._pathology_array[i, j] = float(val)
                if self.target_column in self.metadata.columns:
                    val = self.metadata.loc[sid, self.target_column]
                    if not pd.isna(val):
                        self._target_array[i] = float(val)

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

        # Get subject's cells using pre-computed index for O(1) lookup
        if subject_id not in self._subject_indices:
            raise RuntimeError(
                f"Subject '{subject_id}' not in pre-computed index. "
                f"This indicates a bug — all subjects should be indexed at __init__."
            )
        adata_subject = self.adata[self._subject_indices[subject_id]]

        # ─────────────────────────────────────────────────────────────────────
        # Expression data — keep sparse for pseudobulk, densify only sampled cells.
        # A subject with 20K+ cells would produce ~305MB dense array; sampling
        # only ~1K cells per type keeps the dense allocation bounded.
        # ─────────────────────────────────────────────────────────────────────
        X_sparse = adata_subject.X
        X_is_sparse = hasattr(X_sparse, "toarray")

        # ─────────────────────────────────────────────────────────────────────
        # Single-pass cell-type grouping: compute once, reuse across
        # _compute_pseudobulk, _get_cell_level_data, _compute_pseudobulk_by_region.
        # Avoids ~248 redundant string comparisons for a 20K-cell subject.
        # ─────────────────────────────────────────────────────────────────────
        ct_values = adata_subject.obs[self.cell_type_column].values
        ct_grouped: dict[str, np.ndarray] = {}
        for ct_name in self.cell_type_order:
            ct_grouped[ct_name] = np.where(ct_values == ct_name)[0]

        # ─────────────────────────────────────────────────────────────────────
        # Pseudobulk expression (works on sparse or dense)
        # ─────────────────────────────────────────────────────────────────────
        pseudobulk, cell_type_mask, cell_counts = self._compute_pseudobulk(
            adata_subject, X_sparse, ct_grouped=ct_grouped,
        )

        # ─────────────────────────────────────────────────────────────────────
        # Cell-level data for Set Transformer (densifies only sampled rows)
        # ─────────────────────────────────────────────────────────────────────
        cells, cell_mask, cell_barcodes = self._get_cell_level_data(
            adata_subject, X_sparse, ct_grouped=ct_grouped,
        )

        # ─────────────────────────────────────────────────────────────────────
        # CCC graph features
        # ─────────────────────────────────────────────────────────────────────
        edge_index, edge_type, edge_attr = self._get_graph_features(subject_id)

        # ─────────────────────────────────────────────────────────────────────
        # Region mask and multi-region pseudobulk
        # ─────────────────────────────────────────────────────────────────────
        region_mask = self._get_region_mask(adata_subject)
        region_pseudobulks, available_regions = self._compute_pseudobulk_by_region(
            adata_subject, X_sparse, ct_grouped=ct_grouped,
        )

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

        # Add multi-region pseudobulk data (if BrainRegion column exists)
        for key, value in region_pseudobulks.items():
            sample[key] = torch.from_numpy(value).float()
        if available_regions:
            sample["available_regions"] = available_regions

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _compute_pseudobulk(
        self, adata_subject: AnnData, X,
        ct_grouped: dict[str, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute pseudobulk expression and cell counts for each cell type.

        Works with both sparse (scipy CSR) and dense numpy arrays. Using
        sparse avoids densifying the full subject expression matrix (~305MB
        for subjects with 20K+ cells).

        Args:
            adata_subject: AnnData subset for one subject
            X: Expression matrix [n_cells, n_genes] — sparse or dense
            ct_grouped: Optional pre-computed dict mapping cell_type_name
                -> np.ndarray of positional indices. When provided, skips the
                per-cell-type np.where scan (perf optimization from __getitem__).
        """
        pseudobulk = np.zeros((self.n_cell_types, self.n_genes), dtype=np.float32)
        cell_type_mask = np.zeros(self.n_cell_types, dtype=bool)
        cell_counts = np.zeros(self.n_cell_types, dtype=np.int64)

        # Use pre-computed grouping if available; otherwise compute on the fly.
        # Positional indexing via np.where — avoids pandas label-based
        # get_indexer which fails silently on duplicate obs indices
        # (common in merged multi-region AnnData).
        if ct_grouped is None:
            ct_values = adata_subject.obs[self.cell_type_column].values
            ct_grouped = {ct: np.where(ct_values == ct)[0] for ct in self.cell_type_order}

        for ct_idx, ct_name in enumerate(self.cell_type_order):
            pos = ct_grouped[ct_name]
            n_cells = len(pos)
            cell_counts[ct_idx] = n_cells
            if n_cells > 0:
                # scipy sparse .mean() returns a matrix; squeeze to 1D
                row_mean = X[pos].mean(axis=0)
                pseudobulk[ct_idx] = np.asarray(row_mean, dtype=np.float32).ravel()
                cell_type_mask[ct_idx] = True

        return pseudobulk, cell_type_mask, cell_counts

    def _compute_pseudobulk_by_region(
        self, adata_subject: AnnData, X,
        ct_grouped: dict[str, np.ndarray] | None = None,
    ) -> tuple[dict[str, np.ndarray], list[int]]:
        """
        Compute per-region pseudobulk for multi-region data.

        For subjects with cells from multiple brain regions, computes separate
        pseudobulk aggregations per region. This enables the model to learn
        region-specific expression patterns.

        Note: per-region cell_type_mask is not computed. RegionHandler pools
        across regions using region_mask only (region present/absent).
        If per-region cell-type masking is needed, add it here.

        Args:
            adata_subject: AnnData subset for one subject
            X: Expression matrix [n_cells, n_genes] — sparse or dense
            ct_grouped: Optional pre-computed dict mapping cell_type_name
                -> np.ndarray of positional indices. When provided, intersects
                with region positions instead of doing inner np.where scans
                (perf optimization from __getitem__).

        Returns:
            region_pseudobulks: Dict mapping "region_{idx}_pseudobulk" -> [n_cell_types, n_genes]
            available_regions: Sorted list of region indices with data
        """
        region_pseudobulks = {}
        available_regions = []

        # If no region column, return empty (single-region fallback)
        if self.region_column not in adata_subject.obs.columns:
            return region_pseudobulks, available_regions

        # Build region name → index lookup
        region_to_idx = {name: idx for idx, name in enumerate(REGION_ORDER)}

        # Group cells by (region, cell_type) using positional indexing —
        # avoids pandas label-based get_indexer which fails on duplicate obs indices.
        obs = adata_subject.obs
        regions_in_data = set()
        region_values = obs[self.region_column].values

        # Use pre-computed grouping if available; otherwise compute on the fly.
        if ct_grouped is None:
            ct_values = obs[self.cell_type_column].values
            ct_grouped = {ct: np.where(ct_values == ct)[0] for ct in self.cell_type_order}

        for region_name, region_idx in region_to_idx.items():
            region_pos = np.where(region_values == region_name)[0]
            if len(region_pos) == 0:
                continue

            key = f"region_{region_idx}_pseudobulk"
            region_pseudobulks[key] = np.zeros(
                (self.n_cell_types, self.n_genes), dtype=np.float32
            )
            regions_in_data.add(region_idx)

            # Boolean mask for O(1) membership test — avoids np.isin
            # re-sorting region_pos for each of 31 cell types.
            region_mask_arr = np.zeros(len(region_values), dtype=bool)
            region_mask_arr[region_pos] = True

            for ct_idx, ct_name in enumerate(self.cell_type_order):
                ct_pos = ct_grouped[ct_name]
                # Intersect: cells that are both this cell type AND in this region
                pos = ct_pos[region_mask_arr[ct_pos]]
                if len(pos) == 0:
                    continue
                row_mean = X[pos].mean(axis=0)
                region_pseudobulks[key][ct_idx] = np.asarray(row_mean, dtype=np.float32).ravel()

        available_regions = sorted(regions_in_data)
        return region_pseudobulks, available_regions

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
        self, adata_subject: AnnData, X,
        ct_grouped: dict[str, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[list[str]]]:
        """
        Get cell-level expression for ALL cell types using CellSampler.

        Only densifies the sampled rows (bounded by max_cells_per_type × 31),
        not the entire subject expression matrix. This keeps memory bounded
        even for subjects with 20K+ cells.

        Returns data for all 31 cell types. Cell types with fewer cells than
        min_cells_threshold will have empty data (all-False mask). The model's
        CellTypeSelector learns which types to use for prediction.

        Args:
            adata_subject: AnnData subset for one subject
            X: Expression matrix [n_cells, n_genes] — sparse or dense
            ct_grouped: Optional pre-computed dict mapping cell_type_name
                -> np.ndarray of positional indices. Passed through to
                CellSampler.sample as precomputed_indices (perf optimization
                from __getitem__).

        Returns:
            cells: [n_cell_types, max_cells, n_genes] expression data
            cell_mask: [n_cell_types, max_cells] valid cell mask
            cell_barcodes: list of lists of barcode strings per cell type
        """
        cell_barcodes: list[list[str]] = [[] for _ in range(self.n_cell_types)]

        # Use CellSampler for reproducible sampling across ALL cell types
        sampled_indices = self.sampler.sample(
            adata_subject,
            cell_type_column=self.cell_type_column,
            cell_types=self.cell_type_order,  # ALL 31 types
            precomputed_indices=ct_grouped,
        )

        # Determine actual max cells across all types for this subject
        actual_max = max(
            (len(sampled_indices.get(ct, [])) for ct in self.cell_type_order),
            default=0,
        )
        # Clamp to at least 1 to avoid zero-sized dimension
        actual_max = max(actual_max, 1)

        cells = np.zeros((self.n_cell_types, actual_max, self.n_genes), dtype=np.float32)
        cell_mask = np.zeros((self.n_cell_types, actual_max), dtype=bool)

        obs_index = adata_subject.obs.index

        for i, ct_name in enumerate(self.cell_type_order):
            indices = sampled_indices.get(ct_name, np.array([], dtype=np.int64))
            n_sampled = len(indices)

            if n_sampled > 0:
                # Densify only the sampled rows (bounded, not full subject)
                rows = X[indices]
                if hasattr(rows, "toarray"):
                    rows = rows.toarray()
                cells[i, :n_sampled] = np.asarray(rows, dtype=np.float32)
                cell_mask[i, :n_sampled] = True
                cell_barcodes[i] = [str(x) for x in obs_index[indices].tolist()]

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

        # Lazy import: liana_processing has heavy pandas/scipy dependencies that
        # are only needed when CCC edges are used. Avoids import-time overhead for
        # the common case (PrecomputedDataset) where LIANA is not needed.
        from src.data.liana_processing import build_subject_ccc_features

        liana_df = self.liana_results[subject_id]
        features = build_subject_ccc_features(liana_df, self.cell_type_order, compute_adjacency=False)

        return features["edge_index"], features["edge_type"], features["edge_attr"]

    def _get_pathology(self, subject_id: str) -> np.ndarray:
        """Get pathology scores (O(1) numpy lookup)."""
        return self._pathology_array[self._sid_to_idx[subject_id]]

    def _get_target(self, subject_id: str) -> float:
        """Get cognition target (O(1) numpy lookup)."""
        return float(self._target_array[self._sid_to_idx[subject_id]])

    def get_gene_names(self) -> list[str]:
        """Get gene names in order."""
        return list(self.adata.var_names)

    def get_cell_type_names(self) -> list[str]:
        """Get cell type names in order."""
        return self.cell_type_order


class PrecomputedDataset(Dataset):
    """
    Dataset loading precomputed features from disk.

    Use this for faster training after initial preprocessing.

    When ``preload_to_ram=True``, all .npz files are decompressed and
    converted to torch tensors at init time.  Subsequent ``__getitem__``
    calls return from the in-memory cache (microseconds) instead of
    hitting disk (~1.6 s per subject).  Total RAM cost is ~20 GB for the
    full ROSMAP dataset (517 subjects), which fits easily on machines with
    64+ GB RAM.
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
        max_missing_subject_fraction: float = 0.1,
        preload_to_ram: bool = False,
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
            max_missing_subject_fraction: Maximum fraction of subjects allowed to be
                missing .npz files before raising an error (default: 0.1 = 10%).
            preload_to_ram: If True, load all .npz files into memory at init time.
                Eliminates disk I/O during training (~20 GB RAM for full ROSMAP dataset).
        """
        self.feature_dir = Path(feature_dir)
        self.subject_ids = list(subject_ids)
        self.metadata = metadata.set_index(subject_column) if subject_column in metadata.columns else metadata

        self.target_column = target_column
        self.pathology_columns = pathology_columns or ["gpath", "amylsqrt", "tangsqrt"]
        self.cell_type_order = cell_type_order or CELL_TYPE_ORDER
        self.max_missing_subject_fraction = max_missing_subject_fraction

        # Validate files exist
        self._validate_files()
        self._validate_gene_names()

        # Validate no NaN in target or pathology columns
        self._validate_metadata()

        # Pre-extract phenotypes to numpy arrays for O(1) __getitem__ access.
        # Avoids 4 pandas .loc lookups per sample (3 pathology + 1 target).
        self._sid_to_idx = {sid: i for i, sid in enumerate(self.subject_ids)}
        self._pathology_array = np.zeros(
            (len(self.subject_ids), len(self.pathology_columns)), dtype=np.float32
        )
        self._target_array = np.zeros(len(self.subject_ids), dtype=np.float32)
        for i, sid in enumerate(self.subject_ids):
            for j, col in enumerate(self.pathology_columns):
                if col in self.metadata.columns:
                    self._pathology_array[i, j] = float(self.metadata.loc[sid, col])
            self._target_array[i] = float(self.metadata.loc[sid, self.target_column])

        # Pre-compute cognition tensors to avoid per-sample torch.tensor allocation
        self._cognition_tensors = torch.tensor(
            self._target_array, dtype=torch.float32
        ).unsqueeze(1)  # [N_subjects, 1]

        # Pre-load small arrays to RAM (optional).  The cells array is too
        # large to fit all subjects in memory (~545 MB/subject × 465 = 254 GB),
        # so only metadata/masks/edges/pseudobulk are cached (~1.7 MB/subject).
        # The cells array is still loaded from disk per __getitem__ call;
        # use num_workers > 0 to overlap cells decompression with GPU compute.
        self._small_cache: dict[str, dict[str, Any]] | None = None
        if preload_to_ram:
            self._preload_small_arrays()

        # Pre-build sample templates for cached mode (avoids dict construction per __getitem__)
        if self._small_cache is not None:
            self._sample_templates = {}
            for i, subject_id in enumerate(self.subject_ids):
                cached = self._small_cache[subject_id]
                template = {
                    "subject_id": subject_id,
                    "pseudobulk": cached["pseudobulk"],
                    "cell_type_mask": cached["cell_type_mask"],
                    "cell_counts": cached["cell_counts"],
                    "region_mask": cached["region_mask"],
                    "cell_offsets": cached["cell_offsets"],
                    "ccc_edge_index": cached["ccc_edge_index"],
                    "ccc_edge_type": cached["ccc_edge_type"],
                    "ccc_edge_attr": cached["ccc_edge_attr"],
                    "pathology": torch.from_numpy(self._pathology_array[i]).float().clone(),
                    "cognition": self._cognition_tensors[i],
                }
                # Add region pseudobulks
                for key, val in cached.items():
                    if key.startswith("region_") and key.endswith("_pseudobulk"):
                        template[key] = val
                if "available_regions" in cached:
                    template["available_regions"] = cached["available_regions"]
                self._sample_templates[subject_id] = template
        else:
            self._sample_templates = None

    def _preload_small_arrays(self) -> None:
        """Pre-load everything except the cells array into memory."""
        import time as _time

        t0 = _time.monotonic()
        self._small_cache = {}
        for sid in self.subject_ids:
            feature_file = self.feature_dir / f"{sid}.npz"
            with np.load(feature_file, allow_pickle=True) as npz_data:
                entry: dict[str, Any] = {}
                entry["pseudobulk"] = torch.from_numpy(npz_data["pseudobulk"]).float()
                entry["cell_type_mask"] = torch.from_numpy(npz_data["cell_type_mask"]).bool()
                if "cell_offsets" in npz_data:
                    entry["cell_offsets"] = torch.from_numpy(npz_data["cell_offsets"]).long()
                else:
                    # Backward compat: convert old cell_mask to offsets
                    cell_mask = npz_data["cell_mask"]  # [n_types, max_cells]
                    n_types = cell_mask.shape[0]
                    offsets = np.zeros(n_types + 1, dtype=np.int64)
                    for ct in range(n_types):
                        offsets[ct + 1] = offsets[ct] + int(cell_mask[ct].sum())
                    entry["cell_offsets"] = torch.from_numpy(offsets).long()
                entry["ccc_edge_index"] = torch.from_numpy(npz_data["edge_index"]).long()
                entry["ccc_edge_type"] = torch.from_numpy(npz_data["edge_type"]).long()
                entry["ccc_edge_attr"] = torch.from_numpy(npz_data["edge_attr"]).float()
                if "cell_counts" in npz_data:
                    entry["cell_counts"] = torch.from_numpy(npz_data["cell_counts"]).long()
                elif "cell_offsets" in npz_data:
                    offsets = npz_data["cell_offsets"]
                    entry["cell_counts"] = torch.from_numpy(
                        (offsets[1:] - offsets[:-1]).astype(np.int64)
                    )
                else:
                    entry["cell_counts"] = torch.from_numpy(
                        npz_data["cell_mask"].sum(axis=1).astype(np.int64)
                    )
                if "region_mask" in npz_data:
                    entry["region_mask"] = torch.from_numpy(npz_data["region_mask"]).bool()
                else:
                    from src.data.constants import REGION_ORDER
                    rm = torch.zeros(len(REGION_ORDER), dtype=torch.bool)
                    rm[0] = True
                    entry["region_mask"] = rm
                # Region pseudobulks
                for key in npz_data.files:
                    if key.startswith("region_") and key.endswith("_pseudobulk"):
                        entry[key] = torch.from_numpy(npz_data[key]).float()
                if "available_regions" in npz_data:
                    entry["available_regions"] = npz_data["available_regions"].tolist()
                self._small_cache[sid] = entry
        elapsed = _time.monotonic() - t0
        logger.info(
            "Pre-loaded small arrays for %d subjects to RAM in %.1f s",
            len(self.subject_ids), elapsed,
        )

    def _validate_files(self):
        """Check that feature files exist for all subjects.

        Also removes degenerate subjects with fewer than 2 active cell types,
        since CCC edges require at least 2 cell types interacting and the
        model cannot learn from subjects with no inter-type communication.
        """
        valid_subjects = []
        degenerate = []
        edge_counts = {}
        for sid in self.subject_ids:
            feature_file = self.feature_dir / f"{sid}.npz"
            if not (feature_file.exists() and sid in self.metadata.index):
                continue
            # Check for degenerate subjects (< 2 active cell types)
            with np.load(feature_file, allow_pickle=True) as npz:
                n_active = int(npz["cell_type_mask"].sum())
                n_edges = int(npz["edge_index"].shape[1])
            if n_active < 2:
                degenerate.append(sid)
                continue
            valid_subjects.append(sid)
            edge_counts[sid] = n_edges

        if degenerate:
            warnings.warn(
                f"Removed {len(degenerate)} degenerate subjects with <2 active "
                f"cell types (no CCC edges possible): {degenerate}",
                stacklevel=2,
            )

        if len(valid_subjects) < len(self.subject_ids):
            n_removed = len(self.subject_ids) - len(valid_subjects)
            missing_fraction = n_removed / len(self.subject_ids) if self.subject_ids else 0.0

            if missing_fraction > self.max_missing_subject_fraction:
                raise ValueError(
                    f"Too many subjects missing .npz files or metadata: "
                    f"{n_removed}/{len(self.subject_ids)} ({missing_fraction:.1%}) "
                    f"exceeds threshold ({self.max_missing_subject_fraction:.0%}). "
                    f"Check feature_dir path: {self.feature_dir}"
                )

            missing = [s for s in self.subject_ids if s not in valid_subjects]
            preview = missing[:10]
            suffix = f" (and {len(missing) - 10} more)" if len(missing) > 10 else ""
            warnings.warn(
                f"Removed {n_removed} subjects without feature files or metadata: "
                f"{preview}{suffix}"
            )

        self.subject_ids = valid_subjects
        self._edge_counts = edge_counts

    def get_edge_counts(self) -> list[int]:
        """Return edge counts in subject_ids order (for bucket batching)."""
        return [self._edge_counts[sid] for sid in self.subject_ids]

    def _validate_gene_names(self):
        """Validate gene_names.npy gene count matches pseudobulk gene dimension."""
        gene_names = self.get_gene_names()
        if gene_names is None or len(self.subject_ids) == 0:
            return
        # Check against first subject's pseudobulk
        sample_file = self.feature_dir / f"{self.subject_ids[0]}.npz"
        with np.load(sample_file) as npz:
            n_genes_data = npz["pseudobulk"].shape[1]
        if len(gene_names) != n_genes_data:
            raise ValueError(
                f"gene_names.npy has {len(gene_names)} genes but pseudobulk "
                f"has {n_genes_data} genes. Re-run precompute_features.py to "
                f"regenerate the sidecar file."
            )

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
            try:
                # allow_pickle=True required for object-dtype arrays (string gene names).
                # Safe here: .npy files are self-generated by save_precomputed_features().
                names = np.load(gene_names_path, allow_pickle=True)
                return [str(n) for n in names]
            except Exception as e:
                warnings.warn(f"Could not load gene names from {gene_names_path}: {e}")
        return None

    def get_cell_type_names(self) -> list[str]:
        """Get cell type names in order."""
        return self.cell_type_order

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Load precomputed features for a subject.

        When preload_to_ram is enabled, small arrays (pseudobulk, masks, edges)
        are returned from the in-memory cache.  Only the cells array (~545 MB)
        is loaded from disk.
        """
        subject_id = self.subject_ids[idx]
        feature_file = self.feature_dir / f"{subject_id}.npz"

        try:
            if self._small_cache is not None:
                return self._load_npz_with_cache(subject_id, feature_file, idx)
            return self._load_npz(subject_id, feature_file, idx)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load precomputed features for subject '{subject_id}' "
                f"from {feature_file}: {type(e).__name__}: {e}"
            ) from e

    def _load_npz_with_cache(
        self, subject_id: str, feature_file: Path, idx: int
    ) -> dict[str, torch.Tensor]:
        """Load cell data from disk, combine with pre-built template."""
        sample = self._sample_templates[subject_id].copy()  # shallow copy
        # Only load the cell data array from disk (the expensive part)
        with np.load(feature_file, allow_pickle=True) as npz_data:
            if "cell_data" in npz_data:
                sample["cell_data"] = torch.from_numpy(npz_data["cell_data"]).float()
            else:
                # Backward compat: convert old padded format on-the-fly
                cells = npz_data["cells"]
                cell_mask = npz_data["cell_mask"]
                n_types = cells.shape[0]
                offsets = np.zeros(n_types + 1, dtype=np.int64)
                flat_parts = []
                for ct in range(n_types):
                    n_cells = int(cell_mask[ct].sum())
                    if n_cells > 0:
                        flat_parts.append(cells[ct, :n_cells])
                    offsets[ct + 1] = offsets[ct] + n_cells
                sample["cell_data"] = torch.from_numpy(
                    np.concatenate(flat_parts) if flat_parts else np.empty((0, cells.shape[2]), dtype=np.float32)
                ).float()
                sample["cell_offsets"] = torch.from_numpy(offsets).long()
        return sample

    def _load_npz(self, subject_id: str, feature_file: Path, idx: int) -> dict[str, torch.Tensor]:
        """Load and validate a single .npz file. Called by __getitem__."""
        # Load features (context manager ensures file handle is closed)
        # allow_pickle=True required for object-dtype arrays (cell_type_order).
        # Safe here: .npz files are self-generated by save_precomputed_features().
        with np.load(feature_file, allow_pickle=True) as npz_data:
            file_keys = set(npz_data.files)
            has_flat = "cell_data" in file_keys and "cell_offsets" in file_keys
            has_padded = "cells" in file_keys and "cell_mask" in file_keys

            # Validate required keys (accept either flat or padded cell format)
            base_required = {"pseudobulk", "cell_type_mask", "edge_index", "edge_type", "edge_attr"}
            missing_base = base_required - file_keys
            if missing_base:
                raise KeyError(
                    f"Missing required keys: {missing_base}. "
                    f"Available: {list(npz_data.files)}"
                )
            if not has_flat and not has_padded:
                raise KeyError(
                    f"Missing cell data keys: need either {{cell_data, cell_offsets}} "
                    f"or {{cells, cell_mask}}. Available: {list(npz_data.files)}"
                )

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
                # Derive cell_counts from cell_mask (only available in padded format)
                cell_mask_np = npz_data["cell_mask"]  # [n_cell_types, max_cells] bool
                cell_counts = torch.from_numpy(cell_mask_np.sum(axis=1).astype(np.int64))

            if "region_mask" in npz_data:
                region_mask = torch.from_numpy(npz_data["region_mask"]).bool()
            else:
                # Default to first region only (PFC) for backward compatibility
                from src.data.constants import REGION_ORDER
                region_mask = torch.zeros(len(REGION_ORDER), dtype=torch.bool)
                region_mask[0] = True

            # Extract common arrays into tensors while file is open
            pseudobulk = torch.from_numpy(npz_data["pseudobulk"]).float()
            cell_type_mask = torch.from_numpy(npz_data["cell_type_mask"]).bool()
            ccc_edge_index = torch.from_numpy(npz_data["edge_index"]).long()
            ccc_edge_type = torch.from_numpy(npz_data["edge_type"]).long()
            ccc_edge_attr = torch.from_numpy(npz_data["edge_attr"]).float()

            # Load cell data (flat format preferred, padded converted on-the-fly)
            if has_flat:
                cell_data = torch.from_numpy(npz_data["cell_data"]).float()
                cell_offsets = torch.from_numpy(npz_data["cell_offsets"]).long()
            else:
                # Backward compat: convert old padded format on-the-fly
                cells_np = npz_data["cells"]
                cell_mask_np = npz_data["cell_mask"]
                n_types = cells_np.shape[0]
                offsets = np.zeros(n_types + 1, dtype=np.int64)
                flat_parts = []
                for ct in range(n_types):
                    n_cells = int(cell_mask_np[ct].sum())
                    if n_cells > 0:
                        flat_parts.append(cells_np[ct, :n_cells])
                    offsets[ct + 1] = offsets[ct] + n_cells
                cell_data = torch.from_numpy(
                    np.concatenate(flat_parts) if flat_parts
                    else np.empty((0, cells_np.shape[2]), dtype=np.float32)
                ).float()
                cell_offsets = torch.from_numpy(offsets).long()

            # Shape validation
            n_ct = len(self.cell_type_order)
            if pseudobulk.ndim != 2 or pseudobulk.shape[0] != n_ct:
                raise ValueError(
                    f"pseudobulk shape {tuple(pseudobulk.shape)}: expected "
                    f"[{n_ct}, n_genes], got wrong n_cell_types dim"
                )
            if cell_data.ndim != 2:
                raise ValueError(
                    f"cell_data should be 2D [total_cells, n_genes], "
                    f"got {cell_data.ndim}D shape {tuple(cell_data.shape)}"
                )
            if cell_offsets.shape[0] != n_ct + 1:
                raise ValueError(
                    f"cell_offsets length {cell_offsets.shape[0]}: expected "
                    f"{n_ct + 1} (n_cell_types + 1)"
                )
            if int(cell_offsets[-1].item()) != cell_data.shape[0]:
                raise ValueError(
                    f"cell_offsets[-1]={int(cell_offsets[-1].item())} != "
                    f"cell_data.shape[0]={cell_data.shape[0]}: offset/data mismatch"
                )
            if cell_data.shape[0] > 0 and pseudobulk.shape[1] != cell_data.shape[1]:
                raise ValueError(
                    f"Gene dimension mismatch: pseudobulk has {pseudobulk.shape[1]} genes "
                    f"but cell_data has {cell_data.shape[1]} genes"
                )

            # Load multi-region pseudobulk data (if present in file)
            region_pseudobulks = {}
            for key in npz_data.files:
                if key.startswith("region_") and key.endswith("_pseudobulk"):
                    region_pseudobulks[key] = torch.from_numpy(npz_data[key]).float()

            available_regions = (
                npz_data["available_regions"].tolist()
                if "available_regions" in npz_data
                else None
            )

        # Use pre-extracted phenotype arrays (O(1) numpy indexing vs pandas .loc)
        pathology = torch.from_numpy(self._pathology_array[idx]).float()

        sample = {
            "subject_id": subject_id,
            "pseudobulk": pseudobulk,
            "cell_type_mask": cell_type_mask,
            "cell_counts": cell_counts,
            "region_mask": region_mask,
            "cell_data": cell_data,
            "cell_offsets": cell_offsets,
            # Graph features (CCC = cell-cell communication)
            "ccc_edge_index": ccc_edge_index,
            "ccc_edge_type": ccc_edge_type,
            "ccc_edge_attr": ccc_edge_attr,
            # Phenotypes
            "pathology": pathology,
            "cognition": self._cognition_tensors[idx],
            **region_pseudobulks,
        }

        if available_regions is not None:
            sample["available_regions"] = available_regions

        return sample


def save_precomputed_features(
    dataset: CognitiveResilienceDataset,
    output_dir: str | Path,
    verbose: bool = True,
    skip_subjects: set[str] | None = None,
) -> None:
    """
    Save precomputed features to disk for faster loading.

    Note: pathology and cognition values are NOT saved in .npz files — they
    are read from metadata at training time by PrecomputedDataset. This allows
    updating targets (e.g., different cognition measures) without re-precomputing
    the expensive cell-level features.

    Note: cell_barcodes are not saved in .npz files. For barcode-level
    interpretability analysis, use CognitiveResilienceDataset directly.

    Args:
        dataset: CognitiveResilienceDataset to precompute
        output_dir: Directory to save .npz files
        verbose: Print progress
        skip_subjects: Subject IDs to skip (already precomputed)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save gene names sidecar for downstream interpretability.
    # Only save when dataset is non-empty (skip_subjects may contain stale IDs
    # not in dataset, so counting effective_subjects from set difference is fragile).
    if len(dataset) > 0:
        gene_names = dataset.get_gene_names()
        if gene_names is not None:
            np.save(output_dir / "gene_names.npy", np.array(gene_names, dtype=object))

    n_skipped = 0
    n_saved = 0
    for i in range(len(dataset)):
        sample = dataset[i]
        subject_id = sample["subject_id"]

        if skip_subjects and subject_id in skip_subjects:
            n_skipped += 1
            continue

        output_file = output_dir / f"{subject_id}.npz"

        # Convert padded cells [n_types, max_cells, n_genes] to flat format
        # cell_data [total_real_cells, n_genes] + cell_offsets [n_types+1]
        cells_padded = sample["cells"].numpy()
        cell_mask_padded = sample["cell_mask"].numpy()
        n_types = cells_padded.shape[0]

        cell_offsets = np.zeros(n_types + 1, dtype=np.int64)
        flat_parts = []
        for ct in range(n_types):
            n = int(cell_mask_padded[ct].sum())
            if n > 0:
                flat_parts.append(cells_padded[ct, :n])
            cell_offsets[ct + 1] = cell_offsets[ct] + n

        cell_data = (
            np.concatenate(flat_parts, axis=0)
            if flat_parts
            else np.empty((0, cells_padded.shape[2]), dtype=np.float32)
        )

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
            "cell_data": cell_data,
            "cell_offsets": cell_offsets,
            # Store ordering metadata for validation on load
            "cell_type_order": np.array(dataset.cell_type_order, dtype=object),
        }

        # Add multi-region pseudobulk data (if present)
        for key in sample:
            if key.startswith("region_") and key.endswith("_pseudobulk"):
                save_data[key] = sample[key].numpy()

        if "available_regions" in sample:
            save_data["available_regions"] = np.array(sample["available_regions"])

        # Atomic write: save to temp file then rename to prevent partial files on crash.
        # Flat cell format eliminates ~87% zero padding vs the old 3D padded layout,
        # so the uncompressed size is already much smaller.  We still use
        # savez_compressed for the remaining arrays (pseudobulk, edges, etc.).
        with tempfile.NamedTemporaryFile(
            dir=output_dir, suffix=".npz", delete=False
        ) as tmp_f:
            tmp_path = Path(tmp_f.name)
        np.savez_compressed(tmp_path, **save_data)
        tmp_path.rename(output_file)
        n_saved += 1

        if verbose and n_saved % 50 == 0:
            print(f"Saved {n_saved}/{len(dataset) - n_skipped} subjects")
    if verbose:
        print(f"Saved {n_saved} subjects to {output_dir}" +
              (f" (skipped {n_skipped} existing)" if n_skipped else ""))