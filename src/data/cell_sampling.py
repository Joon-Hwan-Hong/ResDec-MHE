"""
Cell sampling strategies for Set Transformer branch.

Handles sampling cells from each cell type for cell-level modeling,
with various strategies for handling variable cell counts.
"""

from typing import Literal

import numpy as np
from anndata import AnnData

from src.data.constants import EPSILON_POSITIVE_FLOOR


class CellSampler:
    """
    Sample cells from each cell type for Set Transformer input.

    Strategies:
    - random: Uniform random sampling
    - stratified: Sample proportionally to preserve distribution
    - importance: Use metadata to prioritize certain cells (future)

    Note on Worker Reproducibility:
        When using DataLoader with num_workers > 0, each worker process gets a
        copy of the dataset (including this sampler) with the same RNG state.
        Worker-level re-seeding is handled by _worker_init_fn() in
        src.data.collate, which seeds each worker's CellSampler.rng with
        (base_seed + worker_id) for unique but reproducible samples per worker.
    """

    def __init__(
        self,
        max_cells_per_type: int = 1000,
        min_cells_threshold: int = 50,
        strategy: Literal["random", "stratified", "importance"] = "random",
        seed: int | None = None,
    ):
        """
        Initialize sampler.

        Args:
            max_cells_per_type: Maximum cells to sample per cell type
            min_cells_threshold: Minimum cells required to include cell type
            strategy: Sampling strategy
            seed: Random seed for reproducibility
        """
        self.max_cells_per_type = max_cells_per_type
        self.min_cells_threshold = min_cells_threshold
        self.strategy = strategy
        self.rng = np.random.default_rng(seed)

    def sample(
        self,
        adata: AnnData,
        cell_type_column: str = "supercluster_name",
        cell_types: list[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Sample cells from each cell type.

        Args:
            adata: AnnData object (single subject)
            cell_type_column: Column containing cell type labels
            cell_types: List of cell types to sample (in order)

        Returns:
            Dictionary mapping cell_type -> array of cell indices
        """
        if cell_types is None:
            cell_types = adata.obs[cell_type_column].unique().tolist()

        sampled_indices = {}

        for ct_name in cell_types:
            ct_mask = adata.obs[cell_type_column] == ct_name
            ct_indices = np.where(ct_mask.values)[0]
            n_cells = len(ct_indices)

            if n_cells < self.min_cells_threshold:
                # Not enough cells, skip this cell type
                sampled_indices[ct_name] = np.array([], dtype=np.int64)
                continue

            if n_cells <= self.max_cells_per_type:
                # Take all cells
                sampled_indices[ct_name] = ct_indices
            else:
                # Sample based on strategy
                if self.strategy == "random":
                    sampled = self._random_sample(ct_indices)
                elif self.strategy == "stratified":
                    sampled = self._stratified_sample(adata, ct_indices, ct_mask)
                elif self.strategy == "importance":
                    sampled = self._importance_sample(adata, ct_indices, ct_mask)
                else:
                    raise ValueError(f"Unknown strategy: {self.strategy}")

                sampled_indices[ct_name] = sampled

        return sampled_indices

    def _random_sample(self, indices: np.ndarray) -> np.ndarray:
        """Uniform random sampling without replacement."""
        return self.rng.choice(indices, self.max_cells_per_type, replace=False)

    def _stratified_sample(
        self,
        adata: AnnData,
        indices: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Stratified sampling to preserve within-type distribution.

        Uses clustering or metadata to define strata.
        """
        # Check if clustering exists
        if "leiden" in adata.obs.columns:
            strata_col = "leiden"
        elif "louvain" in adata.obs.columns:
            strata_col = "louvain"
        else:
            # Fall back to random sampling
            return self._random_sample(indices)

        # Get strata for these cells.
        # Alignment note: `indices` (from np.where(mask.values)[0]) and
        # `adata.obs.loc[mask, strata_col].values` iterate True positions
        # in the same positional order by definition of boolean indexing,
        # so strata[i] always corresponds to indices[i].
        strata = adata.obs.loc[mask, strata_col].values

        # Sample proportionally from each stratum
        unique_strata, counts = np.unique(strata, return_counts=True)
        total_cells = len(indices)
        samples_per_stratum = (counts / total_cells * self.max_cells_per_type).astype(int)

        # Ensure we get exactly max_cells_per_type
        diff = self.max_cells_per_type - samples_per_stratum.sum()
        if diff > 0:
            # Add remaining samples to largest strata
            largest_strata_idx = np.argsort(counts)[-diff:]
            samples_per_stratum[largest_strata_idx] += 1
        elif diff < 0:
            # Rounding produced more than max_cells_per_type; reduce largest strata
            excess = -diff
            largest_first = np.argsort(counts)[::-1]
            for idx in largest_first:
                reduce = min(excess, samples_per_stratum[idx])
                samples_per_stratum[idx] -= reduce
                excess -= reduce
                if excess == 0:
                    break

        # Note: if a stratum has fewer cells than its allocated quota, we take
        # all available cells. The shortfall is NOT redistributed to other strata,
        # so the final count may be less than max_cells_per_type. This is acceptable
        # because downstream padding handles variable-length cell arrays.
        sampled = []
        for stratum, n_samples in zip(unique_strata, samples_per_stratum):
            stratum_mask = strata == stratum
            stratum_indices = indices[stratum_mask]

            if len(stratum_indices) <= n_samples:
                sampled.extend(stratum_indices)
            else:
                sampled.extend(self.rng.choice(stratum_indices, n_samples, replace=False))

        return np.array(sampled[:self.max_cells_per_type])

    def _importance_sample(
        self,
        adata: AnnData,
        indices: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Importance sampling based on cell-level scores.

        Can use QC metrics, expression of marker genes, etc.
        """
        # Derive indices from mask to guarantee alignment between mask and indices.
        # This prevents silent bugs if the caller ever passes non-aligned arguments.
        derived_indices = np.where(mask.values if hasattr(mask, 'values') else mask)[0]
        if len(derived_indices) != len(indices):
            raise ValueError(
                f"mask/indices mismatch: mask selects {len(derived_indices)} cells "
                f"but indices has {len(indices)} entries"
            )
        indices = derived_indices

        # Check for importance scores
        if "importance_score" in adata.obs.columns:
            scores = adata.obs.loc[mask, "importance_score"].values
        elif "n_genes" in adata.obs.columns:
            # Use number of expressed genes as proxy for quality
            scores = adata.obs.loc[mask, "n_genes"].values
        else:
            # Fall back to random
            return self._random_sample(indices)

        # Guard against NaN in user-provided importance_score column
        if np.isnan(scores).any():
            return self._random_sample(indices)

        # Convert to probabilities (higher score = higher probability)
        scores = scores - scores.min() + EPSILON_POSITIVE_FLOOR  # Ensure positive
        probs = scores / scores.sum()

        # Sample without replacement using probabilities
        sampled = self.rng.choice(
            indices,
            size=self.max_cells_per_type,
            replace=False,
            p=probs,
        )

        return sampled


def subsample_adata(
    adata: AnnData,
    max_cells_per_type: int = 1000,
    cell_type_column: str = "supercluster_name",
    seed: int = 42,
) -> AnnData:
    """
    Subsample AnnData to have at most max_cells per cell type.

    Useful for reducing memory during preprocessing.

    Args:
        adata: Full AnnData
        max_cells_per_type: Maximum cells per cell type
        cell_type_column: Column for cell type labels
        seed: Random seed

    Returns:
        Subsampled AnnData
    """
    rng = np.random.default_rng(seed)

    keep_indices = []

    for ct_name in adata.obs[cell_type_column].unique():
        ct_mask = adata.obs[cell_type_column] == ct_name
        ct_indices = np.where(ct_mask.values)[0]

        if len(ct_indices) <= max_cells_per_type:
            keep_indices.extend(ct_indices)
        else:
            sampled = rng.choice(ct_indices, max_cells_per_type, replace=False)
            keep_indices.extend(sampled)

    keep_indices = np.array(sorted(keep_indices))

    return adata[keep_indices].copy()


def get_cell_type_counts(
    adata: AnnData,
    subject_column: str = "ROSMAP_IndividualID",
    cell_type_column: str = "supercluster_name",
) -> dict[str, dict[str, int]]:
    """
    Get cell counts per cell type per subject.

    Args:
        adata: AnnData object
        subject_column: Column for subject IDs
        cell_type_column: Column for cell type labels

    Returns:
        Nested dict: subject_id -> cell_type -> count
    """
    counts = {}

    for subject_id in adata.obs[subject_column].unique():
        subject_mask = adata.obs[subject_column] == subject_id
        ct_counts = adata.obs.loc[subject_mask, cell_type_column].value_counts()
        counts[subject_id] = ct_counts.to_dict()

    return counts


def filter_subjects_by_cell_coverage(
    adata: AnnData,
    required_cell_types: list[str],
    min_cells_per_type: int = 50,
    subject_column: str = "ROSMAP_IndividualID",
    cell_type_column: str = "supercluster_name",
) -> list[str]:
    """
    Filter to subjects having minimum cells in required cell types.

    Args:
        adata: AnnData object
        required_cell_types: Cell types that must be present
        min_cells_per_type: Minimum cells required per type
        subject_column: Column for subject IDs
        cell_type_column: Column for cell type labels

    Returns:
        List of subject IDs meeting criteria
    """
    counts = get_cell_type_counts(adata, subject_column, cell_type_column)

    valid_subjects = []
    for subject_id, ct_counts in counts.items():
        has_all = True
        for ct in required_cell_types:
            if ct_counts.get(ct, 0) < min_cells_per_type:
                has_all = False
                break

        if has_all:
            valid_subjects.append(subject_id)

    return valid_subjects
