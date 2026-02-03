"""
Cell type importance analysis from pathology attention weights.

Produces three analysis outputs:
1. cell_type_importance.csv - Overall importance (mean attention, std, rank)
2. cell_type_importance_by_pathology.csv - Stratified by pathology tertile
3. cell_type_importance_by_region.csv - Stratified by brain region

Output format: Tidy DataFrames saved as Parquet (primary) and CSV (human-readable).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from src.data.constants import CELL_TYPE_ORDER, REGION_ORDER
from src.utils.io import save_dataframe

logger = logging.getLogger(__name__)


@dataclass
class CellTypeImportanceResult:
    """
    Container for cell type importance analysis results.

    Attributes:
        overall: DataFrame with columns [cell_type, mean_attention, std_attention, rank]
        by_pathology: DataFrame with columns [cell_type, pathology_tertile, mean_attention, std_attention, n_subjects]
        by_region: DataFrame with columns [cell_type, region, mean_attention, std_attention, n_subjects]
        metadata: Additional analysis metadata
    """

    overall: pd.DataFrame
    by_pathology: pd.DataFrame | None = None
    by_region: pd.DataFrame | None = None
    metadata: dict = field(default_factory=dict)


class CellTypeImportanceAnalyzer:
    """
    Analyze cell type importance from pathology attention weights.

    The pathology attention mechanism produces per-subject attention over cell types,
    which indicates which cell types the model deems important for cognition prediction
    conditioned on pathology level.

    Example:
        >>> analyzer = CellTypeImportanceAnalyzer(
        ...     attention=attention_weights,  # [n_subjects, n_heads, n_cell_types]
        ...     pathology_scores=pathology,   # [n_subjects]
        ...     subject_ids=subject_ids,
        ... )
        >>> result = analyzer.analyze()
        >>> analyzer.save(result, output_dir)
    """

    def __init__(
        self,
        attention: np.ndarray,
        pathology_scores: np.ndarray | None = None,
        region_labels: np.ndarray | None = None,
        subject_ids: list[str] | None = None,
        cell_type_names: list[str] | None = None,
    ):
        """
        Initialize analyzer with attention weights and optional stratification data.

        Args:
            attention: Pathology attention weights [n_subjects, n_heads, n_cell_types]
            pathology_scores: Pathology scores for stratification [n_subjects]
            region_labels: Region labels for stratification [n_subjects]
            subject_ids: Subject identifiers
            cell_type_names: Cell type names (defaults to CELL_TYPE_ORDER)
        """
        self.attention = attention
        self.pathology_scores = pathology_scores
        self.region_labels = region_labels
        self.subject_ids = subject_ids
        self.cell_type_names = cell_type_names or list(CELL_TYPE_ORDER)

        # Validate shapes
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """Validate input array shapes and consistency."""
        if self.attention.ndim != 3:
            raise ValueError(
                f"attention must be 3D [n_subjects, n_heads, n_cell_types], "
                f"got shape {self.attention.shape}"
            )

        n_subjects, n_heads, n_cell_types = self.attention.shape

        if n_cell_types != len(self.cell_type_names):
            raise ValueError(
                f"attention has {n_cell_types} cell types but "
                f"{len(self.cell_type_names)} names provided"
            )

        if self.pathology_scores is not None:
            if len(self.pathology_scores) != n_subjects:
                raise ValueError(
                    f"pathology_scores has {len(self.pathology_scores)} entries "
                    f"but attention has {n_subjects} subjects"
                )

        if self.region_labels is not None:
            if len(self.region_labels) != n_subjects:
                raise ValueError(
                    f"region_labels has {len(self.region_labels)} entries "
                    f"but attention has {n_subjects} subjects"
                )

        if self.subject_ids is not None:
            if len(self.subject_ids) != n_subjects:
                raise ValueError(
                    f"subject_ids has {len(self.subject_ids)} entries "
                    f"but attention has {n_subjects} subjects"
                )

    def analyze(self) -> CellTypeImportanceResult:
        """
        Run all cell type importance analyses.

        Returns:
            CellTypeImportanceResult with overall and stratified analyses
        """
        overall = self._compute_overall_importance()

        by_pathology = None
        if self.pathology_scores is not None:
            by_pathology = self._compute_importance_by_pathology()

        by_region = None
        if self.region_labels is not None:
            by_region = self._compute_importance_by_region()

        n_subjects, n_heads, n_cell_types = self.attention.shape
        metadata = {
            "n_subjects": n_subjects,
            "n_heads": n_heads,
            "n_cell_types": n_cell_types,
            "has_pathology_stratification": by_pathology is not None,
            "has_region_stratification": by_region is not None,
        }

        return CellTypeImportanceResult(
            overall=overall,
            by_pathology=by_pathology,
            by_region=by_region,
            metadata=metadata,
        )

    def _compute_overall_importance(self) -> pd.DataFrame:
        """
        Compute overall cell type importance across all subjects.

        Aggregates attention across heads (mean) then computes statistics
        across subjects for each cell type.

        Returns:
            DataFrame with columns: cell_type, mean_attention, std_attention, rank
        """
        # Average across heads: [n_subjects, n_cell_types]
        attention_per_subject = self.attention.mean(axis=1)

        # Statistics across subjects
        mean_attention = attention_per_subject.mean(axis=0)
        std_attention = attention_per_subject.std(axis=0)

        df = pd.DataFrame({
            "cell_type": self.cell_type_names,
            "mean_attention": mean_attention,
            "std_attention": std_attention,
        })

        # Rank by mean attention (1 = highest)
        df = df.sort_values("mean_attention", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)

        return df[["cell_type", "mean_attention", "std_attention", "rank"]]

    def _compute_importance_by_pathology(
        self,
        n_tertiles: int = 3,
    ) -> pd.DataFrame:
        """
        Compute cell type importance stratified by pathology tertiles.

        Subjects are grouped into low/medium/high pathology based on tertiles.

        Args:
            n_tertiles: Number of tertile groups (default 3)

        Returns:
            DataFrame with columns: cell_type, pathology_tertile, mean_attention,
                                   std_attention, n_subjects
        """
        if self.pathology_scores is None:
            raise ValueError("pathology_scores required for pathology stratification")

        # Compute tertile boundaries
        tertile_edges = np.percentile(
            self.pathology_scores,
            [100 * i / n_tertiles for i in range(n_tertiles + 1)]
        )
        tertile_labels = ["low", "medium", "high"][:n_tertiles]

        # Assign tertiles
        tertile_assignments = np.digitize(self.pathology_scores, tertile_edges[1:-1])

        # Average attention across heads: [n_subjects, n_cell_types]
        attention_per_subject = self.attention.mean(axis=1)

        rows = []
        for tertile_idx, tertile_label in enumerate(tertile_labels):
            mask = tertile_assignments == tertile_idx
            n_in_group = mask.sum()

            if n_in_group == 0:
                continue

            group_attention = attention_per_subject[mask]
            mean_attention = group_attention.mean(axis=0)
            std_attention = group_attention.std(axis=0) if n_in_group > 1 else np.zeros_like(mean_attention)

            for ct_idx, ct_name in enumerate(self.cell_type_names):
                rows.append({
                    "cell_type": ct_name,
                    "pathology_tertile": tertile_label,
                    "mean_attention": float(mean_attention[ct_idx]),
                    "std_attention": float(std_attention[ct_idx]),
                    "n_subjects": int(n_in_group),
                })

        return pd.DataFrame(rows)

    def _compute_importance_by_region(self) -> pd.DataFrame:
        """
        Compute cell type importance stratified by brain region.

        Each subject may have data from multiple regions; this aggregates
        by the primary region label if provided.

        Returns:
            DataFrame with columns: cell_type, region, mean_attention,
                                   std_attention, n_subjects
        """
        if self.region_labels is None:
            raise ValueError("region_labels required for region stratification")

        # Average attention across heads: [n_subjects, n_cell_types]
        attention_per_subject = self.attention.mean(axis=1)

        # Get unique regions
        unique_regions = np.unique(self.region_labels)

        rows = []
        for region in unique_regions:
            # Convert numpy string to Python string if needed
            region_str = str(region)
            mask = self.region_labels == region
            n_in_group = mask.sum()

            if n_in_group == 0:
                continue

            group_attention = attention_per_subject[mask]
            mean_attention = group_attention.mean(axis=0)
            std_attention = group_attention.std(axis=0) if n_in_group > 1 else np.zeros_like(mean_attention)

            for ct_idx, ct_name in enumerate(self.cell_type_names):
                rows.append({
                    "cell_type": ct_name,
                    "region": region_str,
                    "mean_attention": float(mean_attention[ct_idx]),
                    "std_attention": float(std_attention[ct_idx]),
                    "n_subjects": int(n_in_group),
                })

        return pd.DataFrame(rows)

    def save(
        self,
        result: CellTypeImportanceResult,
        output_dir: str | Path,
        formats: list[Literal["parquet", "csv"]] | None = None,
    ) -> dict[str, Path]:
        """
        Save analysis results to files.

        Args:
            result: CellTypeImportanceResult to save
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

        # Save overall importance
        for fmt in formats:
            path = output_dir / f"cell_type_importance.{fmt}"
            save_dataframe(result.overall, path, fmt)
            saved_files[f"overall_{fmt}"] = path

        # Save pathology-stratified (if available)
        if result.by_pathology is not None:
            for fmt in formats:
                path = output_dir / f"cell_type_importance_by_pathology.{fmt}"
                save_dataframe(result.by_pathology, path, fmt)
                saved_files[f"by_pathology_{fmt}"] = path

        # Save region-stratified (if available)
        if result.by_region is not None:
            for fmt in formats:
                path = output_dir / f"cell_type_importance_by_region.{fmt}"
                save_dataframe(result.by_region, path, fmt)
                saved_files[f"by_region_{fmt}"] = path

        logger.info(f"Saved cell type importance analysis to {output_dir}")
        return saved_files


def compute_cell_type_importance(
    attention: np.ndarray,
    pathology_scores: np.ndarray | None = None,
    region_labels: np.ndarray | None = None,
    subject_ids: list[str] | None = None,
    cell_type_names: list[str] | None = None,
    output_dir: str | Path | None = None,
) -> CellTypeImportanceResult:
    """
    Convenience function to compute and optionally save cell type importance.

    Args:
        attention: Pathology attention weights [n_subjects, n_heads, n_cell_types]
        pathology_scores: Pathology scores for stratification
        region_labels: Region labels for stratification
        subject_ids: Subject identifiers
        cell_type_names: Cell type names
        output_dir: If provided, save results to this directory

    Returns:
        CellTypeImportanceResult with analysis results

    Example:
        >>> result = compute_cell_type_importance(
        ...     attention=weights.pathology_attention,
        ...     pathology_scores=metadata["pathology"],
        ...     output_dir="outputs/analysis/cell_type/",
        ... )
    """
    analyzer = CellTypeImportanceAnalyzer(
        attention=attention,
        pathology_scores=pathology_scores,
        region_labels=region_labels,
        subject_ids=subject_ids,
        cell_type_names=cell_type_names,
    )

    result = analyzer.analyze()

    if output_dir is not None:
        analyzer.save(result, output_dir)

    return result


def load_cell_type_importance(
    path: str | Path,
) -> pd.DataFrame:
    """
    Load cell type importance results from file.

    Supports both Parquet and CSV formats (auto-detected from extension).

    Args:
        path: Path to saved analysis file

    Returns:
        DataFrame with loaded results
    """
    path = Path(path)

    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    elif path.suffix == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
