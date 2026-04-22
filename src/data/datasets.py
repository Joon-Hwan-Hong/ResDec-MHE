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

2. Cell data format: Both CognitiveResilienceDataset and PrecomputedDataset
   return flat cell representations:

   - cell_data [total_cells, n_genes]: concatenated expression for all cell types
   - cell_offsets [n_cell_types + 1]: cumulative offsets into cell_data

   This avoids the ~87% zero padding of the old 3D [n_types, max_cells, n_genes]
   format.

3. Mask semantics:

   - cell_type_mask [n_cell_types]: True if ANY cells exist for this type (>0).
     Used by Pseudobulk and HGT branches. Even 1 cell provides meaningful
     pseudobulk (mean expression) and allows HGT message passing.

   Cell types with fewer than min_cells_threshold (default: 50) have zero cells
   in the flat representation (offsets are equal). The model's CellTransformer
   handles empty types via zero embeddings.
"""

import contextlib
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
from src.data.tabpfn_input import METADATA_FIELDS, load_metadata_vector

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _shared_mmap():
    """Temporarily set torch mmap default to MAP_SHARED, restoring MAP_PRIVATE on exit.

    MAP_SHARED lets all processes/ranks share physical pages via the OS page
    cache instead of getting private copies (MAP_PRIVATE). This is critical
    for DDP and multi-process HPO to avoid duplicating multi-GB data per rank.
    """
    import mmap as _mmap

    torch.serialization.set_default_mmap_options(_mmap.MAP_SHARED)
    try:
        yield
    finally:
        torch.serialization.set_default_mmap_options(_mmap.MAP_PRIVATE)


def _build_metadata_vectors(
    subject_ids: list[str],
    meta_csv: Path | None,
    age_mean: float | None,
    age_std: float | None,
) -> torch.Tensor | None:
    """Build a [N, 8] tensor of FiLM metadata vectors for the given subjects.

    Returns None when meta_csv is None so callers can fall back to zero-metadata
    training. Age stats are forwarded to load_metadata_vector only when both are
    provided; otherwise the loader falls back to cohort-wide constants.
    """
    if meta_csv is None:
        return None
    kwargs: dict = {}
    if age_mean is not None:
        kwargs["age_mean"] = float(age_mean)
    if age_std is not None:
        kwargs["age_std"] = float(age_std)
    vecs = [
        load_metadata_vector(sid, meta_csv, **kwargs)[0]
        for sid in subject_ids
    ]
    if not vecs:
        return torch.zeros(0, len(METADATA_FIELDS), dtype=torch.float32)
    return torch.stack(vecs).float()


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
        meta_csv: Path | None = None,
        age_mean: float | None = None,
        age_std: float | None = None,
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
            meta_csv: Path to metadata.csv for FiLM metadata vectors. When
                provided, per-subject 8-dim metadata tensors are precomputed
                at __init__ (APOE / sex / age + missingness bits) and added to
                each sample under the "metadata" key. Required together with
                age_mean/age_std to avoid val leakage via age z-scoring.
            age_mean: Mean of age_death on the TRAIN fold only, used for
                z-scoring. Must come from the train split (not pooled) to
                prevent leakage from val into train statistics.
            age_std: Std of age_death on the TRAIN fold only, used for
                z-scoring. Same leakage guard as age_mean.

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

        # FiLM metadata: precompute per-subject 8-dim vectors once using the
        # fold's train-only age_mean/age_std (passed in from the datamodule)
        # to avoid val leakage via z-scoring statistics.
        self.meta_csv = Path(meta_csv) if meta_csv is not None else None
        self.age_mean = age_mean
        self.age_std = age_std
        self._metadata_vectors = _build_metadata_vectors(
            self.subject_ids, self.meta_csv, self.age_mean, self.age_std,
        )

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
                f"Use PrecomputedDataset with precomputed .pt files for training at scale.",
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
        # Returns flat format: cell_data + cell_offsets
        # ─────────────────────────────────────────────────────────────────────
        cell_data, cell_offsets, cell_barcodes = self._get_cell_level_data(
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
            # Cell-level data in flat format: cell_data + cell_offsets
            "cell_data": torch.from_numpy(cell_data).float(),
            "cell_offsets": torch.from_numpy(cell_offsets).long(),
            # Graph features (CCC = cell-cell communication)
            "ccc_edge_index": torch.from_numpy(edge_index).long(),
            "ccc_edge_type": torch.from_numpy(edge_type).long(),
            "ccc_edge_attr": torch.from_numpy(edge_attr).float(),
            # Phenotypes
            "pathology": torch.from_numpy(pathology).float(),
            "cognition": torch.tensor([target], dtype=torch.float32),
        }

        # FiLM metadata vector (APOE + sex + age + missingness bits). Only
        # attached when meta_csv was provided at __init__ — the lightning
        # module's None→zeros fallback covers the unconfigured case.
        if self._metadata_vectors is not None:
            sample["metadata"] = self._metadata_vectors[idx]

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

        Returns flat format: cell_data [total_cells, n_genes] + cell_offsets
        [n_cell_types + 1] (cumulative offsets). Only densifies the sampled
        rows (bounded by max_cells_per_type * 31), not the entire subject
        expression matrix. This keeps memory bounded even for subjects with
        20K+ cells.

        Returns data for all 31 cell types. Cell types with fewer cells than
        min_cells_threshold will have empty data (zero cells in offsets). The
        model's CellTypeSelector learns which types to use for prediction.

        Args:
            adata_subject: AnnData subset for one subject
            X: Expression matrix [n_cells, n_genes] — sparse or dense
            ct_grouped: Optional pre-computed dict mapping cell_type_name
                -> np.ndarray of positional indices. Passed through to
                CellSampler.sample as precomputed_indices (perf optimization
                from __getitem__).

        Returns:
            cell_data: [total_cells, n_genes] flat concatenated expression data
            cell_offsets: [n_cell_types + 1] cumulative offsets into cell_data
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

        flat_parts: list[np.ndarray] = []
        cell_offsets = np.zeros(self.n_cell_types + 1, dtype=np.int64)

        obs_index = adata_subject.obs.index

        for i, ct_name in enumerate(self.cell_type_order):
            indices = sampled_indices.get(ct_name, np.array([], dtype=np.int64))
            n_sampled = len(indices)

            if n_sampled > 0:
                # Densify only the sampled rows (bounded, not full subject)
                rows = X[indices]
                if hasattr(rows, "toarray"):
                    rows = rows.toarray()
                flat_parts.append(np.asarray(rows, dtype=np.float32))
                cell_barcodes[i] = [str(x) for x in obs_index[indices].tolist()]

            cell_offsets[i + 1] = cell_offsets[i] + n_sampled

        if flat_parts:
            cell_data = np.concatenate(flat_parts, axis=0)
        else:
            cell_data = np.empty((0, self.n_genes), dtype=np.float32)

        return cell_data, cell_offsets, cell_barcodes

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
    Dataset loading precomputed features from .pt files into RAM.

    All subjects are loaded into process-local heap memory at init time.
    ``__getitem__`` returns from pre-built templates with zero I/O.

    Each DDP rank loads its own copy (~38 GB for full ROSMAP dataset).
    This avoids Linux ``mmap_lock`` contention that causes 4x data-loading
    slowdown under DDP on kernels < 6.4 (no per-VMA locking).

    For HPO, pass ``preloaded_cache`` to skip per-trial disk I/O. Use
    :meth:`load_subject_cache` to build the cache once per worker process.
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
        preloaded_cache: dict[str, dict[str, Any]] | None = None,
        meta_csv: Path | None = None,
        age_mean: float | None = None,
        age_std: float | None = None,
    ):
        """
        Initialize from precomputed .pt features (loaded into RAM).

        Args:
            feature_dir: Directory containing precomputed .pt files
            subject_ids: List of subject IDs
            metadata: Subject metadata
            subject_column: Column for subject IDs
            target_column: Column for cognition target
            pathology_columns: Columns for pathology scores
            cell_type_order: Order of cell types for edge index mapping.
                Must match the order used when precomputing features.
                Defaults to CELL_TYPE_ORDER from constants.
            max_missing_subject_fraction: Maximum fraction of subjects allowed to be
                missing .pt files before raising an error (default: 0.1 = 10%).
            preloaded_cache: Pre-loaded subject data from :meth:`load_subject_cache`.
                When provided, validation runs from cache with zero disk I/O
                (``_validate_from_cache``) instead of loading from disk.
            meta_csv: Path to metadata.csv for FiLM metadata vectors. When
                provided, per-subject 8-dim metadata tensors are precomputed
                and baked into each sample template under the "metadata" key.
                Required together with age_mean/age_std to avoid val leakage.
            age_mean: Mean of age_death on the TRAIN fold only, used for
                z-scoring. Must come from the train split (not pooled) to
                prevent leakage from val into train statistics.
            age_std: Std of age_death on the TRAIN fold only, used for
                z-scoring. Same leakage guard as age_mean.
        """
        self.feature_dir = Path(feature_dir)
        self.subject_ids = list(subject_ids)
        self.metadata = metadata.set_index(subject_column) if subject_column in metadata.columns else metadata

        self.target_column = target_column
        self.pathology_columns = pathology_columns or ["gpath", "amylsqrt", "tangsqrt"]
        self.max_missing_subject_fraction = max_missing_subject_fraction
        self.meta_csv = Path(meta_csv) if meta_csv is not None else None
        self.age_mean = age_mean
        self.age_std = age_std

        # Single-pass load + validate: loads each .pt file exactly once,
        # detecting cell_type_order, filtering degenerate subjects, and
        # checking gene_names in the same pass.  When preloaded_cache is
        # provided (HPO fast path), all validation is done from the cache
        # with zero disk I/O.
        if preloaded_cache is not None:
            self._small_cache, detected_order = self._validate_from_cache(
                preloaded_cache, cell_type_order,
            )
        else:
            self._small_cache, detected_order = self._load_and_validate_all(
                cell_type_order,
            )

        if cell_type_order is not None:
            self.cell_type_order = cell_type_order
        else:
            self.cell_type_order = detected_order or CELL_TYPE_ORDER

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

        # FiLM metadata: precompute per-subject 8-dim vectors once using the
        # fold's train-only age_mean/age_std. Subjects use the same train-only
        # stats for both train and val datasets to avoid leakage.
        self._metadata_vectors = _build_metadata_vectors(
            self.subject_ids, self.meta_csv, self.age_mean, self.age_std,
        )

        # Pre-build sample templates (avoids dict construction per __getitem__)
        from src.data.constants import N_REGIONS, PFC_REGION_IDX

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
                "cell_data": cached["cell_data"],
                "pathology": torch.from_numpy(self._pathology_array[i]).float(),
                "cognition": self._cognition_tensors[i],
            }
            if self._metadata_vectors is not None:
                template["metadata"] = self._metadata_vectors[i]
            # Pre-stack region pseudobulks into [n_regions, n_cell_types, n_genes]
            # so collation can torch.stack directly instead of allocating zeros
            # + nested fill loop over the 36 MB region_pseudobulk tensor.
            n_ct, n_genes = cached["pseudobulk"].shape
            region_pb = torch.zeros(N_REGIONS, n_ct, n_genes)
            region_msk = torch.zeros(N_REGIONS, dtype=torch.bool)
            avail = cached.get("available_regions", [PFC_REGION_IDX])
            for ridx in avail:
                rkey = f"region_{ridx}_pseudobulk"
                if rkey in cached:
                    region_pb[ridx] = cached[rkey]
                    region_msk[ridx] = True
                elif ridx == PFC_REGION_IDX:
                    region_pb[ridx] = cached["pseudobulk"]
                    region_msk[ridx] = True
            template["region_pseudobulk"] = region_pb
            template["region_mask"] = region_msk
            self._sample_templates[subject_id] = template

    def _load_and_validate_all(
        self,
        cell_type_order: list[str] | None,
    ) -> tuple[dict[str, dict[str, Any]], list[str] | None]:
        """Single-pass load + validate: loads each .pt file exactly once.

        Merges the work previously split across ``_detect_cell_type_order``,
        ``_validate_files``, ``_validate_gene_names``, and ``_load_all`` into
        one pass over the .pt files. Each file is loaded once via mmap and
        validated in-line.

        Under DDP, automatically caches data to ``/dev/shm`` (RAM-backed
        tmpfs) for near-instant loading across all ranks. The staggered
        loading pattern (rank 0 first, barrier, then other ranks) ensures
        rank 0's ``madvise(MADV_WILLNEED)`` pre-faults all pages before
        other ranks touch them.

        Returns:
            (cache, detected_cell_type_order) where cache maps subject_id
            to its tensor dict and detected_cell_type_order is the order
            read from the first .pt file (or None).
        """
        import time as _time
        import torch.distributed as dist

        ddp_active = dist.is_initialized() and dist.get_world_size() > 1
        rank = dist.get_rank() if ddp_active else 0
        world_size = dist.get_world_size() if ddp_active else 1

        # Under DDP, cache data to /dev/shm for fast loading across ranks.
        load_dir = self.feature_dir
        if ddp_active:
            load_dir = self._ensure_shm_cache(rank)

        # These will be populated by the loading rank and used after barriers.
        cache: dict[str, dict[str, Any]] = {}
        detected_order: list[str] | None = None
        degenerate: list[str] = []
        edge_counts: dict[str, int] = {}

        # Gene-name validation state (checked on first loaded subject).
        gene_names = self.get_gene_names()
        gene_names_checked = False

        # Enable MAP_SHARED so all ranks share physical pages via the
        # OS page cache instead of getting private copies. Context manager
        # restores MAP_PRIVATE on exit (normal or exceptional).
        #
        # DDP deadlock prevention: if any rank throws during loading, other
        # ranks would hang forever at dist.barrier(). We use all_reduce to
        # communicate failure before the barrier so all ranks can raise.
        load_error: Exception | None = None
        with _shared_mmap():
            # Stagger loading: each rank loads sequentially to avoid
            # thundering-herd page faults on the process-level mmap_lock.
            for load_rank in range(world_size):
                if rank == load_rank:
                    try:
                        t0 = _time.monotonic()
                        total_cell_bytes = 0
                        is_first_loaded = True

                        for sid in self.subject_ids:
                            # --- file existence check (was in _validate_files) ---
                            feature_file = load_dir / f"{sid}.pt"
                            if not feature_file.exists():
                                feature_file = self._resolve_feature_file(sid)
                            if feature_file is None:
                                continue
                            if sid not in self.metadata.index:
                                continue

                            pt_data = torch.load(
                                feature_file, weights_only=False, mmap=True,
                            )

                            # --- detect cell_type_order from first file (was _detect_cell_type_order) ---
                            if is_first_loaded:
                                if "cell_type_order" in pt_data:
                                    detected_order = list(pt_data["cell_type_order"])

                                # --- validate gene_names against first pseudobulk (was _validate_gene_names) ---
                                if gene_names is not None and not gene_names_checked:
                                    n_genes_data = pt_data["pseudobulk"].shape[1]
                                    if len(gene_names) != n_genes_data:
                                        raise ValueError(
                                            f"gene_names.npy has {len(gene_names)} genes but pseudobulk "
                                            f"has {n_genes_data} genes. Re-run precompute_features.py to "
                                            f"regenerate the sidecar file."
                                        )
                                    gene_names_checked = True
                                is_first_loaded = False

                            # --- validate cell_type_order consistency (was in old _load_all) ---
                            # Use the effective order: explicit > detected > default
                            effective_order = cell_type_order or detected_order or CELL_TYPE_ORDER
                            if "cell_type_order" in pt_data:
                                saved_order = list(pt_data["cell_type_order"])
                                if saved_order != list(effective_order):
                                    raise RuntimeError(
                                        f"Subject {sid} was precomputed with different cell_type_order. "
                                        f"Saved: {saved_order[:5]}... vs current: {list(effective_order)[:5]}... "
                                        f"Re-run precompute_features.py with the correct cell_type_order."
                                    )

                            # --- degenerate subject check (was in _validate_files) ---
                            ct_mask = pt_data["cell_type_mask"]
                            n_active = int(
                                ct_mask.sum()
                                if isinstance(ct_mask, (torch.Tensor, np.ndarray))
                                else np.array(ct_mask).sum()
                            )
                            if n_active < 2:
                                degenerate.append(sid)
                                continue

                            edge_val = pt_data["ccc_edge_index"]
                            n_edges = int(
                                edge_val.shape[1]
                                if isinstance(edge_val, (torch.Tensor, np.ndarray))
                                else np.array(edge_val).shape[1]
                            )
                            edge_counts[sid] = n_edges

                            # --- build cache entry (was in old _load_all) ---
                            entry: dict[str, Any] = {
                                "pseudobulk": pt_data["pseudobulk"],
                                "cell_type_mask": pt_data["cell_type_mask"],
                                "cell_offsets": pt_data["cell_offsets"],
                                "ccc_edge_index": pt_data["ccc_edge_index"],
                                "ccc_edge_type": pt_data["ccc_edge_type"],
                                "ccc_edge_attr": pt_data["ccc_edge_attr"],
                                "cell_counts": pt_data["cell_counts"],
                                "region_mask": pt_data["region_mask"],
                                "cell_data": pt_data["cell_data"],
                            }
                            for key in pt_data:
                                if key.startswith("region_") and key.endswith("_pseudobulk"):
                                    entry[key] = pt_data[key]
                            if "available_regions" in pt_data:
                                entry["available_regions"] = list(pt_data["available_regions"])
                            total_cell_bytes += entry["cell_data"].nelement() * entry["cell_data"].element_size()
                            cache[sid] = entry

                        elapsed = _time.monotonic() - t0
                        logger.info(
                            "Rank %d: loaded %d subjects via mmap (MAP_SHARED) in %.1f s "
                            "(cell_data: %.1f GB, source: %s)",
                            rank, len(cache), elapsed,
                            total_cell_bytes / (1024**3),
                            "shm" if str(load_dir).startswith("/dev/shm") else "disk",
                        )

                        # Pre-fault all mmap'd pages so subsequent ranks find them
                        # resident in the page cache (minor faults only).
                        if ddp_active and rank == 0:
                            self._prefault_cache(cache)

                    except Exception as exc:
                        load_error = exc
                        logger.error("Rank %d: loading failed: %s", rank, exc)

                # DDP deadlock prevention: use all_reduce to check if any
                # rank failed before the barrier. Without this, a failed rank
                # skips the barrier while healthy ranks wait forever.
                if ddp_active:
                    device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")
                    load_ok = torch.tensor([1 if load_error is None else 0], dtype=torch.int32, device=device)
                    dist.all_reduce(load_ok, op=dist.ReduceOp.MIN)
                    if load_ok.item() == 0:
                        if load_error is not None:
                            raise load_error
                        raise RuntimeError(
                            f"Rank {rank}: a DDP rank failed during data loading; "
                            "aborting to prevent barrier deadlock"
                        )
                    dist.barrier()

        # Re-raise outside DDP context for single-GPU runs
        if load_error is not None:
            raise load_error

        # --- post-load validation (was in _validate_files) ---
        if degenerate:
            warnings.warn(
                f"Removed {len(degenerate)} degenerate subjects with <2 active "
                f"cell types (no CCC edges possible): {degenerate}",
                stacklevel=2,
            )

        valid_subjects = list(cache.keys())
        if len(valid_subjects) < len(self.subject_ids):
            n_removed = len(self.subject_ids) - len(valid_subjects)
            missing_fraction = n_removed / len(self.subject_ids) if self.subject_ids else 0.0

            if missing_fraction > self.max_missing_subject_fraction:
                raise ValueError(
                    f"Too many subjects missing feature files or metadata: "
                    f"{n_removed}/{len(self.subject_ids)} ({missing_fraction:.1%}) "
                    f"exceeds threshold ({self.max_missing_subject_fraction:.0%}). "
                    f"Check feature_dir path: {self.feature_dir}"
                )

            missing = [s for s in self.subject_ids if s not in cache]
            preview = missing[:10]
            suffix = f" (and {len(missing) - 10} more)" if len(missing) > 10 else ""
            warnings.warn(
                f"Removed {n_removed} subjects without feature files or metadata: "
                f"{preview}{suffix}"
            )

        self.subject_ids = valid_subjects
        self._edge_counts = edge_counts
        return cache, detected_order

    def _validate_from_cache(
        self,
        preloaded_cache: dict[str, dict[str, Any]],
        cell_type_order: list[str] | None,
    ) -> tuple[dict[str, dict[str, Any]], list[str] | None]:
        """Validate and filter subjects from a preloaded cache (zero disk I/O).

        Performs the same validation as ``_load_and_validate_all`` but reads
        all data from the in-memory cache instead of .pt files on disk.
        Used on the HPO fast path where ``load_subject_cache()`` has already
        loaded everything.

        Returns:
            (cache, detected_cell_type_order)
        """
        detected_order: list[str] | None = None
        degenerate: list[str] = []
        edge_counts: dict[str, int] = {}
        cache: dict[str, dict[str, Any]] = {}

        gene_names = self.get_gene_names()
        gene_names_checked = False
        is_first = True

        for sid in self.subject_ids:
            if sid not in preloaded_cache or sid not in self.metadata.index:
                continue
            data = preloaded_cache[sid]

            if is_first:
                # Detect cell_type_order from the cached data if a
                # "cell_type_order" key was preserved (load_subject_cache
                # doesn't store it, but the pseudobulk shape is available).
                # For the HPO path, cell_type_order is typically passed
                # explicitly, so detected_order is a fallback.
                # We check the first subject's pseudobulk for gene_names.
                if gene_names is not None and not gene_names_checked:
                    n_genes_data = data["pseudobulk"].shape[1]
                    if len(gene_names) != n_genes_data:
                        raise ValueError(
                            f"gene_names.npy has {len(gene_names)} genes but pseudobulk "
                            f"has {n_genes_data} genes. Re-run precompute_features.py to "
                            f"regenerate the sidecar file."
                        )
                    gene_names_checked = True
                is_first = False

            # Degenerate subject check
            ct_mask = data["cell_type_mask"]
            n_active = int(
                ct_mask.sum()
                if isinstance(ct_mask, (torch.Tensor, np.ndarray))
                else np.array(ct_mask).sum()
            )
            if n_active < 2:
                degenerate.append(sid)
                continue

            edge_val = data["ccc_edge_index"]
            n_edges = int(
                edge_val.shape[1]
                if isinstance(edge_val, (torch.Tensor, np.ndarray))
                else np.array(edge_val).shape[1]
            )
            edge_counts[sid] = n_edges
            cache[sid] = data

        if degenerate:
            warnings.warn(
                f"Removed {len(degenerate)} degenerate subjects with <2 active "
                f"cell types (no CCC edges possible): {degenerate}",
                stacklevel=2,
            )

        valid_subjects = list(cache.keys())
        if len(valid_subjects) < len(self.subject_ids):
            n_removed = len(self.subject_ids) - len(valid_subjects)
            missing_fraction = n_removed / len(self.subject_ids) if self.subject_ids else 0.0

            if missing_fraction > self.max_missing_subject_fraction:
                raise ValueError(
                    f"Too many subjects missing feature files or metadata: "
                    f"{n_removed}/{len(self.subject_ids)} ({missing_fraction:.1%}) "
                    f"exceeds threshold ({self.max_missing_subject_fraction:.0%}). "
                    f"Check feature_dir path: {self.feature_dir}"
                )

            missing = [s for s in self.subject_ids if s not in cache]
            preview = missing[:10]
            suffix = f" (and {len(missing) - 10} more)" if len(missing) > 10 else ""
            warnings.warn(
                f"Removed {n_removed} subjects without feature files or metadata: "
                f"{preview}{suffix}"
            )

        self.subject_ids = valid_subjects
        self._edge_counts = edge_counts
        return cache, detected_order

    def _ensure_shm_cache(self, rank: int) -> Path:
        """Copy feature files to /dev/shm for fast DDP loading.

        Rank 0 copies files; other ranks wait at a barrier.
        Returns the /dev/shm cache directory path.
        """
        import shutil
        import torch.distributed as dist

        shm_dir = Path("/dev/shm") / f"precomputed_{self.feature_dir.name}"

        if rank == 0:
            # Check if cache is valid (exists and has at least as many files)
            if shm_dir.exists() and len(list(shm_dir.glob("*.pt"))) >= len(self.subject_ids):
                logger.info("Using existing /dev/shm cache: %s", shm_dir)
                self._register_shm_cleanup(shm_dir)
            else:
                # Check space: need ~1.2x data size for safety margin
                shm_stat = shutil.disk_usage("/dev/shm")
                data_size = sum(f.stat().st_size for f in self.feature_dir.glob("*.pt"))
                if shm_stat.free > data_size * 1.2:
                    logger.info(
                        "Caching %d .pt files (%.1f GB) to /dev/shm for DDP...",
                        len(self.subject_ids), data_size / (1024**3),
                    )
                    shm_dir.mkdir(exist_ok=True)
                    for sid in self.subject_ids:
                        src = self._resolve_feature_file(sid)
                        if src is not None:
                            shutil.copy2(src, shm_dir / f"{sid}.pt")
                    logger.info("Cached to %s", shm_dir)
                    self._register_shm_cleanup(shm_dir)
                else:
                    logger.warning(
                        "/dev/shm has %.1f GB free but data needs %.1f GB. "
                        "Falling back to disk loading.",
                        shm_stat.free / (1024**3), data_size / (1024**3),
                    )
                    shm_dir = self.feature_dir

        # Broadcast the resolved shm_dir from rank 0 so all ranks agree.
        # Without this, if rank 0 fell back to self.feature_dir (not enough
        # /dev/shm space), non-zero ranks might load stale data from a
        # pre-existing /dev/shm/precomputed_* directory from a previous run.
        resolved = [str(shm_dir)]
        dist.broadcast_object_list(resolved, src=0)
        shm_dir = Path(resolved[0])

        dist.barrier()
        return shm_dir if shm_dir.exists() else self.feature_dir

    def _prefault_cache(self, cache: dict[str, dict[str, Any]] | None = None) -> None:
        """Advise kernel to pre-fault all mmap'd tensor pages.

        Uses ``madvise(MADV_WILLNEED)`` to trigger readahead on all pages
        backing the mmap'd tensors. After this, the pages are resident in
        the page cache and other ranks' MAP_SHARED mappings will only
        trigger minor faults (page table setup, no disk I/O).

        Args:
            cache: Cache dict to pre-fault. If None, uses self._small_cache.
        """
        import ctypes
        import ctypes.util

        if cache is None:
            cache = self._small_cache
        libc_name = ctypes.util.find_library("c")
        if libc_name is None:
            logger.warning("Could not find libc for madvise pre-fault; skipping")
            return
        libc = ctypes.CDLL(libc_name)
        MADV_WILLNEED = 3
        n_advised = 0
        for entry in cache.values():
            for val in entry.values():
                if isinstance(val, torch.Tensor) and val.is_contiguous():
                    nbytes = val.nelement() * val.element_size()
                    if nbytes > 0 and val.data_ptr() != 0:
                        libc.madvise(
                            ctypes.c_void_p(val.data_ptr()),
                            ctypes.c_size_t(nbytes),
                            MADV_WILLNEED,
                        )
                        n_advised += 1
        logger.info("Pre-faulted %d tensors via madvise(MADV_WILLNEED)", n_advised)

    @staticmethod
    def load_subject_cache(
        feature_dir: str | Path,
        subject_ids: list[str],
        use_mmap: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """Load .pt files for all subjects into a reusable cache dict.

        Call once per worker process, then pass the result as
        ``preloaded_cache`` to avoid per-trial disk I/O during HPO.

        Args:
            feature_dir: Directory containing .pt files.
            subject_ids: Subject IDs to load.
            use_mmap: If True, use mmap with MAP_SHARED for cross-process
                page sharing. Requires files on a tmpfs (e.g. /dev/shm)
                for best performance. Default False for backward compat.

        Returns:
            Mapping of subject_id -> dict of tensors (same structure as
            ``_small_cache``).
        """
        import time as _time

        # Use _shared_mmap() context manager to set MAP_SHARED during loading
        # and restore MAP_PRIVATE on exit (normal or exceptional).
        cm = _shared_mmap() if use_mmap else contextlib.nullcontext()

        feature_dir = Path(feature_dir)
        t0 = _time.monotonic()
        cache: dict[str, dict[str, Any]] = {}
        total_cell_bytes = 0
        with cm:
            for sid in subject_ids:
                pt_path = feature_dir / f"{sid}.pt"
                if not pt_path.exists():
                    continue
                pt_data = torch.load(pt_path, weights_only=False, mmap=use_mmap)
                entry: dict[str, Any] = {
                    "pseudobulk": pt_data["pseudobulk"],
                    "cell_type_mask": pt_data["cell_type_mask"],
                    "cell_offsets": pt_data["cell_offsets"],
                    "ccc_edge_index": pt_data["ccc_edge_index"],
                    "ccc_edge_type": pt_data["ccc_edge_type"],
                    "ccc_edge_attr": pt_data["ccc_edge_attr"],
                    "cell_counts": pt_data["cell_counts"],
                    "region_mask": pt_data["region_mask"],
                    "cell_data": pt_data["cell_data"],
                }
                for key in pt_data:
                    if key.startswith("region_") and key.endswith("_pseudobulk"):
                        entry[key] = pt_data[key]
                if "available_regions" in pt_data:
                    entry["available_regions"] = list(pt_data["available_regions"])
                total_cell_bytes += entry["cell_data"].nelement() * entry["cell_data"].element_size()
                cache[sid] = entry
        elapsed = _time.monotonic() - t0
        logger.info(
            "Pre-loaded %d subjects into cache in %.1f s (cell_data: %.1f GB, mmap=%s)",
            len(cache), elapsed, total_cell_bytes / (1024**3), use_mmap,
        )
        return cache

    @staticmethod
    def share_cache_memory(cache: dict[str, dict[str, Any]]) -> None:
        """Move all tensors in a preloaded cache to shared memory (in-place).

        Call before ``fork()``-based DDP so child ranks share the same
        physical pages instead of duplicating ~37 GB per rank.
        """
        n_shared = 0
        for sid, entry in cache.items():
            for key, val in entry.items():
                if isinstance(val, torch.Tensor) and not val.is_shared():
                    entry[key] = val.share_memory_()
                    n_shared += 1
        logger.info("Moved %d tensors to shared memory", n_shared)

    @staticmethod
    def cleanup_shm_cache(shm_dir: Path) -> None:
        """Remove a /dev/shm cache directory.

        Safe to call from atexit or signal handlers. Only deletes paths
        under /dev/shm or /tmp (safety guard against accidental deletion).
        """
        import shutil

        if not shm_dir.exists():
            return
        # Safety: only remove dirs under /dev/shm or /tmp
        parent = str(shm_dir.resolve())
        if not (parent.startswith("/dev/shm/") or parent.startswith("/tmp/")):
            logger.warning("Refusing to delete non-shm path: %s", shm_dir)
            return
        shutil.rmtree(shm_dir, ignore_errors=True)
        logger.info("Cleaned up shm cache: %s", shm_dir)

    @staticmethod
    def _register_shm_cleanup(shm_dir: Path) -> None:
        """Register atexit + SIGTERM/SIGINT handlers to clean up a /dev/shm cache dir."""
        import atexit
        import signal

        _cleaning_up = False

        def _cleanup():
            nonlocal _cleaning_up
            if _cleaning_up:
                return  # Re-entrancy guard: signal during atexit cleanup
            _cleaning_up = True
            try:
                PrecomputedDataset.cleanup_shm_cache(shm_dir)
            finally:
                _cleaning_up = False

        atexit.register(_cleanup)
        prev_sigterm = signal.getsignal(signal.SIGTERM)
        prev_sigint = signal.getsignal(signal.SIGINT)

        def _signal_handler(signum, frame):
            _cleanup()
            # Chain to previous handler
            prev = prev_sigterm if signum == signal.SIGTERM else prev_sigint
            if callable(prev) and prev not in (signal.SIG_DFL, signal.SIG_IGN):
                prev(signum, frame)
            else:
                raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, _signal_handler)
        signal.signal(signal.SIGINT, _signal_handler)
        logger.info("Registered atexit + SIGTERM/SIGINT cleanup for %s", shm_dir)

    def _resolve_feature_file(self, sid: str) -> Path | None:
        """Return the .pt feature file path for a subject.

        Returns None if the file does not exist.
        """
        pt_path = self.feature_dir / f"{sid}.pt"
        if pt_path.exists():
            return pt_path
        return None

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
        """Return pre-built template for a subject (heap-backed, zero I/O)."""
        subject_id = self.subject_ids[idx]
        return self._sample_templates[subject_id]


def save_precomputed_features(
    dataset: CognitiveResilienceDataset,
    output_dir: str | Path,
    verbose: bool = True,
    skip_subjects: set[str] | None = None,
) -> None:
    """
    Save precomputed features to disk for faster loading.

    Note: pathology and cognition values are NOT saved in .pt files — they
    are read from metadata at training time by PrecomputedDataset. This allows
    updating targets (e.g., different cognition measures) without re-precomputing
    the expensive cell-level features.

    Note: cell_barcodes are not saved in .pt files. For barcode-level
    interpretability analysis, use CognitiveResilienceDataset directly.

    Args:
        dataset: CognitiveResilienceDataset to precompute
        output_dir: Directory to save .pt files
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

        # Skip degenerate subjects: 0 cells or <2 active cell types
        # (no CCC edges possible, HGT branch gets empty graph)
        n_cells = sample["cell_data"].shape[0] if isinstance(sample["cell_data"], torch.Tensor) else len(sample["cell_data"])
        n_active_types = int(sample["cell_type_mask"].sum()) if isinstance(sample["cell_type_mask"], torch.Tensor) else int(sum(sample["cell_type_mask"]))
        if n_cells == 0 or n_active_types < 2:
            logger.warning(
                "Skipping degenerate subject %s: %d cells, %d active types",
                subject_id, n_cells, n_active_types,
            )
            n_skipped += 1
            continue

        output_file = output_dir / f"{subject_id}.pt"

        # Build save dict with core features.
        # Dataset already returns flat format (cell_data + cell_offsets).
        # Keys use ccc_edge_* names (matching PrecomputedDataset expectations).
        # All tensor values are saved as torch.Tensor directly.
        save_data: dict = {
            "pseudobulk": sample["pseudobulk"],
            "cell_type_mask": sample["cell_type_mask"],
            "cell_counts": sample["cell_counts"],
            "region_mask": sample["region_mask"],
            "ccc_edge_index": sample["ccc_edge_index"],
            "ccc_edge_type": sample["ccc_edge_type"],
            "ccc_edge_attr": sample["ccc_edge_attr"],
            "cell_data": sample["cell_data"],
            "cell_offsets": sample["cell_offsets"],
            # Store ordering metadata for validation on load (Python list)
            "cell_type_order": list(dataset.cell_type_order),
        }

        # Add multi-region pseudobulk data (if present)
        for key in sample:
            if key.startswith("region_") and key.endswith("_pseudobulk"):
                save_data[key] = sample[key]

        if "available_regions" in sample:
            save_data["available_regions"] = list(sample["available_regions"])

        # Ensure all tensors are contiguous (stride-0 empty tensors crash
        # Ray's zero-copy serialization during HPO).
        for k, v in save_data.items():
            if isinstance(v, torch.Tensor) and v.nelement() == 0:
                # Replace empty tensors with properly-strided empty tensors
                save_data[k] = torch.empty(v.shape, dtype=v.dtype)
            elif isinstance(v, torch.Tensor) and not v.is_contiguous():
                save_data[k] = v.contiguous()

        # Atomic write: save to temp file then rename to prevent partial files on crash.
        with tempfile.NamedTemporaryFile(
            dir=output_dir, suffix=".pt", delete=False
        ) as tmp_f:
            tmp_path = Path(tmp_f.name)
        try:
            torch.save(save_data, tmp_path)
            tmp_path.rename(output_file)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
        n_saved += 1

        if verbose and n_saved % 50 == 0:
            print(f"Saved {n_saved}/{len(dataset) - n_skipped} subjects")
    if verbose:
        print(f"Saved {n_saved} subjects to {output_dir}" +
              (f" (skipped {n_skipped} existing)" if n_skipped else ""))