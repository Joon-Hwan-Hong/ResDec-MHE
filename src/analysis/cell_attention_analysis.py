"""
Cell-level attention analysis from PMA (Pooling by Multihead Attention).

Analyzes individual cell importance by aggregating attention weights across
inducing points, identifying high-attention cells, and characterizing their
transcriptomic profiles.

Primary Method: Aggregate Across Inducing Points
1. PMA attention: [n_subjects, k_inducing, n_cells]
2. Aggregate via mean/max: [n_subjects, n_cells]
3. Identify cells with consistently high attention across subjects
4. Link high-attention cells to cell type and gene expression

Output format: Tidy DataFrames saved as Parquet (primary) and CSV (human-readable).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import pandas as pd
from scipy import stats

from src.utils.io import save_dataframe
from src.utils.statistics import gini_coefficient, attention_entropy

logger = logging.getLogger(__name__)


@dataclass
class CellAttentionResult:
    """
    Container for cell attention analysis results.

    Attributes:
        cell_summary: Per-cell attention summary across all subjects
        high_attention_cells: Cells with attention above threshold
        per_subject_summary: Per-subject cell attention statistics
        attention_by_celltype: Attention aggregated by cell type
        metadata: Additional analysis metadata
    """

    cell_summary: pd.DataFrame
    high_attention_cells: pd.DataFrame | None = None
    per_subject_summary: pd.DataFrame | None = None
    attention_by_celltype: pd.DataFrame | None = None
    metadata: dict = field(default_factory=dict)


class CellAttentionAnalyzer:
    """
    Analyze cell-level attention from PMA pooling.

    Aggregates attention across inducing points to identify which individual
    cells receive the most attention during pooling. This provides finer
    resolution than cell-type-level analysis.

    Example:
        >>> analyzer = CellAttentionAnalyzer(
        ...     pma_attention=attention,  # [n_subjects, k_inducing, n_cells]
        ...     cell_ids=cell_ids,        # [n_cells]
        ...     cell_types=cell_types,    # [n_cells]
        ... )
        >>> result = analyzer.analyze()
        >>> analyzer.save(result, output_dir)
    """

    def __init__(
        self,
        pma_attention: np.ndarray,
        cell_ids: list[str] | np.ndarray | None = None,
        cell_types: list[str] | np.ndarray | None = None,
        subject_ids: list[str] | None = None,
        aggregation: Literal["mean", "max"] = "mean",
    ):
        """
        Initialize analyzer with PMA attention weights.

        Args:
            pma_attention: PMA attention [n_subjects, k_inducing, n_cells]
            cell_ids: Cell identifiers [n_cells]
            cell_types: Cell type labels [n_cells]
            subject_ids: Subject identifiers [n_subjects]
            aggregation: Method to aggregate across inducing points ("mean" or "max")
        """
        self.pma_attention = np.asarray(pma_attention)
        self.aggregation = aggregation

        # Validate shape
        if self.pma_attention.ndim != 3:
            raise ValueError(
                f"pma_attention must be 3D [n_subjects, k_inducing, n_cells], "
                f"got shape {self.pma_attention.shape}"
            )

        self.n_subjects, self.k_inducing, self.n_cells = self.pma_attention.shape

        # Set default identifiers
        self.cell_ids = (
            list(cell_ids) if cell_ids is not None
            else [f"cell_{i}" for i in range(self.n_cells)]
        )
        self.cell_types = (
            list(cell_types) if cell_types is not None
            else ["unknown"] * self.n_cells
        )
        self.subject_ids = (
            list(subject_ids) if subject_ids is not None
            else [f"subject_{i}" for i in range(self.n_subjects)]
        )

        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """Validate input shapes."""
        if len(self.cell_ids) != self.n_cells:
            raise ValueError(
                f"cell_ids has {len(self.cell_ids)} entries "
                f"but attention has {self.n_cells} cells"
            )
        if len(self.cell_types) != self.n_cells:
            raise ValueError(
                f"cell_types has {len(self.cell_types)} entries "
                f"but attention has {self.n_cells} cells"
            )
        if len(self.subject_ids) != self.n_subjects:
            raise ValueError(
                f"subject_ids has {len(self.subject_ids)} entries "
                f"but attention has {self.n_subjects} subjects"
            )

    def analyze(
        self,
        high_attention_percentile: float = 95.0,
        min_subjects_fraction: float = 0.1,
    ) -> CellAttentionResult:
        """
        Run cell attention analysis.

        Args:
            high_attention_percentile: Percentile threshold for high-attention cells
            min_subjects_fraction: Minimum fraction of subjects where cell must have
                                  high attention to be considered consistently important

        Returns:
            CellAttentionResult with analysis results
        """
        # Aggregate across inducing points
        aggregated = self._aggregate_attention()

        # Compute cell summary
        cell_summary = self._compute_cell_summary(aggregated)

        # Identify high-attention cells
        high_attention = self._identify_high_attention_cells(
            aggregated,
            percentile=high_attention_percentile,
            min_subjects_fraction=min_subjects_fraction,
        )

        # Per-subject summary
        per_subject = self._compute_per_subject_summary(aggregated)

        # Attention by cell type
        by_celltype = self._compute_attention_by_celltype(aggregated)

        metadata = {
            "n_subjects": self.n_subjects,
            "n_cells": self.n_cells,
            "k_inducing": self.k_inducing,
            "aggregation": self.aggregation,
            "high_attention_percentile": high_attention_percentile,
            "min_subjects_fraction": min_subjects_fraction,
            "n_high_attention_cells": len(high_attention) if high_attention is not None else 0,
        }

        return CellAttentionResult(
            cell_summary=cell_summary,
            high_attention_cells=high_attention,
            per_subject_summary=per_subject,
            attention_by_celltype=by_celltype,
            metadata=metadata,
        )

    def _aggregate_attention(self) -> np.ndarray:
        """
        Aggregate attention across inducing points.

        Returns:
            Aggregated attention [n_subjects, n_cells]
        """
        if self.aggregation == "mean":
            return self.pma_attention.mean(axis=1)
        elif self.aggregation == "max":
            return self.pma_attention.max(axis=1)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

    def _compute_cell_summary(self, aggregated: np.ndarray) -> pd.DataFrame:
        """
        Compute per-cell attention summary across all subjects.

        Args:
            aggregated: [n_subjects, n_cells]

        Returns:
            DataFrame with cell-level statistics
        """
        # Statistics across subjects
        mean_attention = aggregated.mean(axis=0)
        std_attention = aggregated.std(axis=0, ddof=1)
        median_attention = np.median(aggregated, axis=0)
        max_attention = aggregated.max(axis=0)
        min_attention = aggregated.min(axis=0)

        df = pd.DataFrame({
            "cell_id": self.cell_ids,
            "cell_type": self.cell_types,
            "mean_attention": mean_attention,
            "std_attention": std_attention,
            "median_attention": median_attention,
            "max_attention": max_attention,
            "min_attention": min_attention,
            "cv": std_attention / (mean_attention + 1e-10),  # Coefficient of variation
        })

        # Rank by mean attention
        df = df.sort_values("mean_attention", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)

        return df

    def _identify_high_attention_cells(
        self,
        aggregated: np.ndarray,
        percentile: float,
        min_subjects_fraction: float,
    ) -> pd.DataFrame | None:
        """
        Identify cells with consistently high attention.

        Args:
            aggregated: [n_subjects, n_cells]
            percentile: Percentile threshold for "high" attention
            min_subjects_fraction: Minimum fraction of subjects

        Returns:
            DataFrame with high-attention cells
        """
        # Compute threshold per subject
        thresholds = np.percentile(aggregated, percentile, axis=1)  # [n_subjects]

        # For each cell, count subjects where it's above threshold
        above_threshold = aggregated >= thresholds[:, np.newaxis]  # [n_subjects, n_cells]
        subject_counts = above_threshold.sum(axis=0)  # [n_cells]

        min_subjects = int(self.n_subjects * min_subjects_fraction)
        high_attention_mask = subject_counts >= min_subjects

        if not high_attention_mask.any():
            logger.warning("No cells meet high-attention criteria")
            return None

        # Extract high-attention cells
        high_indices = np.where(high_attention_mask)[0]

        rows = []
        for idx in high_indices:
            rows.append({
                "cell_id": self.cell_ids[idx],
                "cell_type": self.cell_types[idx],
                "mean_attention": aggregated[:, idx].mean(),
                "n_subjects_high": int(subject_counts[idx]),
                "fraction_subjects_high": subject_counts[idx] / self.n_subjects,
            })

        df = pd.DataFrame(rows)
        df = df.sort_values("mean_attention", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)

        return df

    def _compute_per_subject_summary(self, aggregated: np.ndarray) -> pd.DataFrame:
        """
        Compute per-subject attention statistics.

        Args:
            aggregated: [n_subjects, n_cells]

        Returns:
            DataFrame with subject-level statistics
        """
        rows = []
        for i, subject_id in enumerate(self.subject_ids):
            attention = aggregated[i]
            rows.append({
                "subject_id": subject_id,
                "mean_attention": float(attention.mean()),
                "std_attention": float(attention.std()),
                "max_attention": float(attention.max()),
                "gini": gini_coefficient(attention),
                "entropy": attention_entropy(attention),
            })

        return pd.DataFrame(rows)

    def _compute_attention_by_celltype(self, aggregated: np.ndarray) -> pd.DataFrame:
        """
        Aggregate attention by cell type.

        Args:
            aggregated: [n_subjects, n_cells]

        Returns:
            DataFrame with cell-type-level statistics
        """
        cell_types = np.array(self.cell_types)
        unique_types = np.unique(cell_types)

        rows = []
        for ct in unique_types:
            mask = cell_types == ct
            ct_attention = aggregated[:, mask]  # [n_subjects, n_cells_of_type]

            # Mean across cells of this type, then across subjects
            mean_per_subject = ct_attention.mean(axis=1)

            rows.append({
                "cell_type": ct,
                "n_cells": int(mask.sum()),
                "mean_attention": float(mean_per_subject.mean()),
                "std_attention": float(mean_per_subject.std()),
                "median_attention": float(np.median(mean_per_subject)),
            })

        df = pd.DataFrame(rows)
        df = df.sort_values("mean_attention", ascending=False).reset_index(drop=True)

        return df

    def save(
        self,
        result: CellAttentionResult,
        output_dir: str | Path,
        formats: list[Literal["parquet", "csv"]] | None = None,
        save_hdf5: bool = True,
    ) -> dict[str, Path]:
        """
        Save analysis results to files.

        Args:
            result: CellAttentionResult to save
            output_dir: Directory for output files
            formats: Output formats (default: ["parquet", "csv"])
            save_hdf5: Whether to save raw attention to HDF5

        Returns:
            Dict mapping output name to file path
        """
        if formats is None:
            formats = ["parquet", "csv"]

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        # Save cell summary
        for fmt in formats:
            path = output_dir / f"cell_attention_summary.{fmt}"
            save_dataframe(result.cell_summary, path, fmt)
            saved_files[f"cell_summary_{fmt}"] = path

        # Save high-attention cells (if available)
        if result.high_attention_cells is not None:
            for fmt in formats:
                path = output_dir / f"high_attention_cells.{fmt}"
                save_dataframe(result.high_attention_cells, path, fmt)
                saved_files[f"high_attention_{fmt}"] = path

        # Save per-subject summary
        if result.per_subject_summary is not None:
            for fmt in formats:
                path = output_dir / f"cell_attention_per_subject.{fmt}"
                save_dataframe(result.per_subject_summary, path, fmt)
                saved_files[f"per_subject_{fmt}"] = path

        # Save attention by cell type
        if result.attention_by_celltype is not None:
            for fmt in formats:
                path = output_dir / f"cell_attention_by_celltype.{fmt}"
                save_dataframe(result.attention_by_celltype, path, fmt)
                saved_files[f"by_celltype_{fmt}"] = path

        # Save raw attention to HDF5
        if save_hdf5:
            h5_path = output_dir / "cell_attention.h5"
            self._save_hdf5(h5_path)
            saved_files["hdf5"] = h5_path

        logger.info(f"Saved cell attention analysis to {output_dir}")
        return saved_files

    def _save_hdf5(self, path: Path) -> None:
        """Save attention weights and metadata to HDF5."""
        with h5py.File(path, "w") as f:
            f.attrs["schema_version"] = "2.0"

            # Save raw attention
            f.create_dataset(
                "pma_attention",
                data=self.pma_attention,
                compression="gzip",
                compression_opts=4,
            )

            # Save aggregated attention
            aggregated = self._aggregate_attention()
            f.create_dataset(
                "aggregated_attention",
                data=aggregated,
                compression="gzip",
                compression_opts=4,
            )

            # Save metadata
            f.attrs["n_subjects"] = self.n_subjects
            f.attrs["k_inducing"] = self.k_inducing
            f.attrs["n_cells"] = self.n_cells
            f.attrs["aggregation"] = self.aggregation

            # Save identifiers (as variable-length strings)
            dt = h5py.special_dtype(vlen=str)
            f.create_dataset("cell_ids", data=self.cell_ids, dtype=dt)
            f.create_dataset("cell_types", data=self.cell_types, dtype=dt)
            f.create_dataset("subject_ids", data=self.subject_ids, dtype=dt)


def compute_cell_attention(
    pma_attention: np.ndarray,
    cell_ids: list[str] | None = None,
    cell_types: list[str] | None = None,
    subject_ids: list[str] | None = None,
    aggregation: Literal["mean", "max"] = "mean",
    output_dir: str | Path | None = None,
) -> CellAttentionResult:
    """
    Convenience function to compute and optionally save cell attention analysis.

    Args:
        pma_attention: PMA attention [n_subjects, k_inducing, n_cells]
        cell_ids: Cell identifiers
        cell_types: Cell type labels
        subject_ids: Subject identifiers
        aggregation: Aggregation method ("mean" or "max")
        output_dir: If provided, save results to this directory

    Returns:
        CellAttentionResult with analysis results
    """
    analyzer = CellAttentionAnalyzer(
        pma_attention=pma_attention,
        cell_ids=cell_ids,
        cell_types=cell_types,
        subject_ids=subject_ids,
        aggregation=aggregation,
    )

    result = analyzer.analyze()

    if output_dir is not None:
        analyzer.save(result, output_dir)

    return result
