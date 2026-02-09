"""
Within-cell-type heterogeneity analysis using PMA attention weights.

Identifies which individual cells within each cell type receive high attention
from the Set Transformer's PMA mechanism, highlighting disease-relevant
cellular subpopulations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import pandas as pd

from src.utils.io import save_dataframe
from src.utils.statistics import gini_coefficient, attention_entropy

logger = logging.getLogger(__name__)


@dataclass
class CellHeterogeneityResult:
    """
    Container for cell heterogeneity analysis results.

    Attributes:
        summary: Per-cell-type heterogeneity statistics (Gini, entropy, etc.)
        high_attention_cells: Cell barcodes/indices with high attention scores
        all_scores: All cells with their attention scores
        metadata: Additional analysis metadata
    """

    summary: pd.DataFrame
    high_attention_cells: pd.DataFrame
    all_scores: pd.DataFrame
    metadata: dict = field(default_factory=dict)


class CellHeterogeneityAnalyzer:
    """
    Analyze within-cell-type heterogeneity from PMA attention.

    Evaluates which individual cells within each cell type receive
    disproportionate attention, identifying disease-relevant subpopulations.

    Example:
        >>> analyzer = CellHeterogeneityAnalyzer(
        ...     pma_attention=pma_3d,       # [n_subjects, n_cell_types, n_cells]
        ...     cell_type_names=ct_names,
        ...     subject_ids=subject_ids,
        ... )
        >>> result = analyzer.analyze()
        >>> analyzer.save(result, output_dir)
    """

    def __init__(
        self,
        pma_attention: np.ndarray,
        cell_type_names: list[str],
        subject_ids: list[str] | None = None,
        cell_barcodes: dict | None = None,
        cell_metadata: pd.DataFrame | None = None,
        top_percentile: float = 10.0,
        min_cells_per_type: int = 10,
    ):
        """
        Initialize analyzer with PMA attention data.

        Args:
            pma_attention: PMA attention weights [n_subjects, n_cell_types, n_cells]
            cell_type_names: Names of cell types
            subject_ids: Subject identifiers
            cell_barcodes: Dict mapping "subject_idx_celltype_idx" to cell barcodes
            cell_metadata: DataFrame with cell-level metadata (index=barcodes)
            top_percentile: Percentile threshold for high-attention cells
            min_cells_per_type: Minimum cells per type to analyze
        """
        self.pma_attention = np.asarray(pma_attention)
        self.cell_type_names = cell_type_names
        self.n_subjects, self.n_cell_types, self.max_cells = self.pma_attention.shape
        self.subject_ids = subject_ids or [f"subject_{i}" for i in range(self.n_subjects)]
        self.cell_barcodes = cell_barcodes
        self.cell_metadata = cell_metadata
        self.top_percentile = top_percentile
        self.min_cells_per_type = min_cells_per_type

    def analyze(self) -> CellHeterogeneityResult:
        """
        Run cell heterogeneity analysis.

        Returns:
            CellHeterogeneityResult with summary, high_attention_cells, all_scores
        """
        summary_rows = []
        high_attention_rows = []
        all_scores_rows = []

        for ct_idx, ct_name in enumerate(self.cell_type_names):
            ct_attention = self.pma_attention[:, ct_idx, :]  # [n_subjects, n_cells]

            # Flatten across subjects for global statistics
            valid_mask = ct_attention > 0
            valid_attention = ct_attention[valid_mask]

            if len(valid_attention) < self.min_cells_per_type:
                logger.warning(f"Cell type '{ct_name}' has only {len(valid_attention)} cells, skipping")
                continue

            # Compute threshold for high attention
            threshold = np.percentile(valid_attention, 100 - self.top_percentile)

            # Statistics
            summary_rows.append({
                "cell_type": ct_name,
                "n_cells_total": int(valid_mask.sum()),
                "n_high_attention": int((valid_attention >= threshold).sum()),
                "attention_mean": float(valid_attention.mean()),
                "attention_std": float(valid_attention.std()),
                "attention_median": float(np.median(valid_attention)),
                "attention_threshold": float(threshold),
                "attention_entropy": float(attention_entropy(valid_attention)),
                "gini_coefficient": float(gini_coefficient(valid_attention)),
            })

            # Identify high-attention cells per subject
            for subj_idx, subj_id in enumerate(self.subject_ids):
                subj_attention = ct_attention[subj_idx]
                subj_valid = subj_attention > 0

                if subj_valid.sum() == 0:
                    continue

                high_mask = subj_attention >= threshold

                for cell_idx in np.where(high_mask)[0]:
                    row = {
                        "subject_id": subj_id,
                        "cell_type": ct_name,
                        "cell_idx": int(cell_idx),
                        "attention_score": float(subj_attention[cell_idx]),
                    }
                    barcode_key = f"{subj_idx}_{ct_idx}"
                    if self.cell_barcodes and barcode_key in self.cell_barcodes:
                        barcodes = self.cell_barcodes[barcode_key]
                        if cell_idx < len(barcodes):
                            row["cell_barcode"] = barcodes[cell_idx]
                    high_attention_rows.append(row)

                # All scores for this subject-celltype
                for cell_idx in np.where(subj_valid)[0]:
                    row = {
                        "subject_id": subj_id,
                        "cell_type": ct_name,
                        "cell_idx": int(cell_idx),
                        "attention_score": float(subj_attention[cell_idx]),
                        "is_high_attention": bool(subj_attention[cell_idx] >= threshold),
                    }
                    barcode_key = f"{subj_idx}_{ct_idx}"
                    if self.cell_barcodes and barcode_key in self.cell_barcodes:
                        barcodes = self.cell_barcodes[barcode_key]
                        if cell_idx < len(barcodes):
                            row["cell_barcode"] = barcodes[cell_idx]
                    all_scores_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        high_attention_df = pd.DataFrame(high_attention_rows)
        all_scores_df = pd.DataFrame(all_scores_rows)

        # Sort summary by Gini coefficient (higher = more heterogeneous)
        if len(summary_df) > 0:
            summary_df = summary_df.sort_values("gini_coefficient", ascending=False).reset_index(drop=True)

        # Sort high attention by score
        if len(high_attention_df) > 0:
            high_attention_df = high_attention_df.sort_values(
                ["cell_type", "attention_score"],
                ascending=[True, False],
            ).reset_index(drop=True)

        # Enrich with cell metadata if available
        if self.cell_metadata is not None and "cell_barcode" in high_attention_df.columns:
            meta_cols = [c for c in self.cell_metadata.columns if c not in high_attention_df.columns]
            if meta_cols:
                high_attention_df = high_attention_df.merge(
                    self.cell_metadata[meta_cols], left_on="cell_barcode", right_index=True, how="left"
                )
        if self.cell_metadata is not None and "cell_barcode" in all_scores_df.columns:
            meta_cols = [c for c in self.cell_metadata.columns if c not in all_scores_df.columns]
            if meta_cols:
                all_scores_df = all_scores_df.merge(
                    self.cell_metadata[meta_cols], left_on="cell_barcode", right_index=True, how="left"
                )

        return CellHeterogeneityResult(
            summary=summary_df,
            high_attention_cells=high_attention_df,
            all_scores=all_scores_df,
            metadata={
                "n_subjects": self.n_subjects,
                "n_cell_types": self.n_cell_types,
                "max_cells": self.max_cells,
                "top_percentile": self.top_percentile,
                "min_cells_per_type": self.min_cells_per_type,
            },
        )

    def save(
        self,
        result: CellHeterogeneityResult,
        output_dir: str | Path,
        formats: list[Literal["parquet", "csv"]] | None = None,
    ) -> dict[str, Path]:
        """
        Save analysis results to files.

        Args:
            result: CellHeterogeneityResult to save
            output_dir: Directory for output files
            formats: Output formats (default: ["parquet", "csv"])

        Returns:
            Dict mapping output name to file path
        """
        if formats is None:
            formats = ["parquet", "csv"]

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        # Save DataFrames
        for fmt in formats:
            path = output_dir / f"cell_attention_summary.{fmt}"
            save_dataframe(result.summary, path, fmt)
            saved_files[f"summary_{fmt}"] = path

            path = output_dir / f"high_attention_cells.{fmt}"
            save_dataframe(result.high_attention_cells, path, fmt)
            saved_files[f"high_attention_{fmt}"] = path

            path = output_dir / f"cell_attention_scores.{fmt}"
            save_dataframe(result.all_scores, path, fmt)
            saved_files[f"all_scores_{fmt}"] = path

        # Save HDF5 with raw attention data
        h5_path = output_dir / "cell_attention.h5"
        vlen_str = h5py.special_dtype(vlen=str)

        with h5py.File(h5_path, "w") as f:
            f.attrs["schema_version"] = "2.0"

            f.create_dataset(
                "pma_attention",
                data=self.pma_attention,
                compression="gzip",
                compression_opts=4,
            )

            f.attrs["n_subjects"] = self.n_subjects
            f.attrs["n_cell_types"] = self.n_cell_types
            f.attrs["max_cells"] = self.max_cells

            f.create_dataset(
                "cell_type_names",
                data=np.array(self.cell_type_names, dtype=object),
                dtype=vlen_str,
            )
            f.create_dataset(
                "subject_ids",
                data=np.array(self.subject_ids, dtype=object),
                dtype=vlen_str,
            )

        saved_files["h5"] = h5_path
        logger.info(f"Saved cell heterogeneity analysis to {output_dir}")
        return saved_files


def compute_cell_heterogeneity(
    pma_attention: np.ndarray,
    cell_type_names: list[str],
    subject_ids: list[str] | None = None,
    cell_barcodes: dict | None = None,
    cell_metadata: pd.DataFrame | None = None,
    top_percentile: float = 10.0,
    min_cells_per_type: int = 10,
    output_dir: str | Path | None = None,
    formats: list[Literal["parquet", "csv"]] | None = None,
) -> CellHeterogeneityResult:
    """
    Convenience function to compute and optionally save cell heterogeneity analysis.

    Args:
        pma_attention: PMA attention weights [n_subjects, n_cell_types, n_cells]
        cell_type_names: Names of cell types
        subject_ids: Subject identifiers
        cell_barcodes: Dict mapping "subject_idx_celltype_idx" to cell barcodes
        cell_metadata: DataFrame with cell-level metadata (index=barcodes)
        top_percentile: Percentile threshold for high-attention cells
        min_cells_per_type: Minimum cells per type to analyze
        output_dir: If provided, save results to this directory
        formats: Output formats (default: ["parquet", "csv"])

    Returns:
        CellHeterogeneityResult with analysis results
    """
    analyzer = CellHeterogeneityAnalyzer(
        pma_attention=pma_attention,
        cell_type_names=cell_type_names,
        subject_ids=subject_ids,
        cell_barcodes=cell_barcodes,
        cell_metadata=cell_metadata,
        top_percentile=top_percentile,
        min_cells_per_type=min_cells_per_type,
    )
    result = analyzer.analyze()

    if output_dir is not None:
        analyzer.save(result, output_dir, formats=formats)

    return result
