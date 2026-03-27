"""
Cell-cell communication (CCC) importance analysis from HGT attention weights.

Produces analysis outputs:
1. ccc_importance.csv - Source cell type, target cell type, edge type, attention score
2. ccc_importance_by_region.csv - CCC importance stratified by brain region
3. top_interactions.csv - Top 100 LR interactions by attention, with gene names
4. ccc_network_summary.csv - Aggregated by edge type category

Note: Raw HGT attention tensors are stored in the main attention_weights.h5 file
(saved by the predictor), not by this analyzer.

Output format: Tidy DataFrames saved as Parquet (primary) and CSV (human-readable).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from src.utils.io import save_dataframe
from src.data.constants import (
    CELL_TYPE_ORDER,
    ALL_EDGE_TYPES,
    EDGE_TYPE_DISPLAY_NAMES,
    N_CELL_TYPES,
)

logger = logging.getLogger(__name__)


@dataclass
class CCCImportanceResult:
    """
    Container for CCC importance analysis results.

    Attributes:
        edge_importance: DataFrame with columns [source, target, edge_type, mean_attention, std_attention]
        top_interactions: DataFrame with top-k interactions
        by_region: DataFrame stratified by region (if available)
        network_summary: DataFrame aggregated by edge type category
        raw_attention: Raw HGT attention tensors (if retained)
        metadata: Additional analysis metadata
    """

    edge_importance: pd.DataFrame
    top_interactions: pd.DataFrame
    by_region: pd.DataFrame | None = None
    network_summary: pd.DataFrame | None = None
    metadata: dict = field(default_factory=dict)


class CCCImportanceAnalyzer:
    """
    Analyze cell-cell communication importance from HGT attention weights.

    The HGT learns attention over edges (cell type pairs × edge types) which
    indicates which intercellular communication channels are important for
    cognition prediction.

    Example:
        >>> analyzer = CCCImportanceAnalyzer(
        ...     hgt_attention=attention_dict,  # Per-subject attention tensors
        ...     edge_index_dict=edge_index,    # Edge connectivity
        ... )
        >>> result = analyzer.analyze(top_k=100)
        >>> analyzer.save(result, output_dir)
    """

    def __init__(
        self,
        edge_attention_scores: np.ndarray | None = None,
        edge_metadata: pd.DataFrame | None = None,
        cell_type_names: list[str] | None = None,
        edge_types: list[str] | None = None,
        region_labels: np.ndarray | None = None,
        subject_ids: list[str] | None = None,
        lr_pair_mapping: dict[str, list[str]] | None = None,
    ):
        """
        Initialize analyzer with edge attention scores.

        Args:
            edge_attention_scores: Aggregated edge attention [n_subjects, n_edges] or
                                  [n_edges] if pre-aggregated across subjects
            edge_metadata: DataFrame with edge info (source, target, edge_type, etc.)
            cell_type_names: Cell type names (defaults to CELL_TYPE_ORDER)
            edge_types: Edge type names (defaults to ALL_EDGE_TYPES)
            region_labels: Region labels for stratification [n_subjects]
            subject_ids: Subject identifiers
            lr_pair_mapping: Mapping from "source|target|edge_type" to list of L-R pairs.
                           Use extract_lr_pairs_by_edge() to create this mapping.
        """
        self.edge_attention_scores = edge_attention_scores
        self.edge_metadata = edge_metadata
        self.cell_type_names = cell_type_names or list(CELL_TYPE_ORDER)
        self.edge_types = edge_types or list(ALL_EDGE_TYPES)
        self.region_labels = region_labels
        self.subject_ids = subject_ids
        self.lr_pair_mapping = lr_pair_mapping or {}

    def analyze(self, top_k: int = 100) -> CCCImportanceResult:
        """
        Run all CCC importance analyses.

        Args:
            top_k: Number of top interactions to extract

        Returns:
            CCCImportanceResult with all analyses
        """
        # Compute edge importance aggregated across subjects
        edge_importance = self._compute_edge_importance()

        # Get top interactions
        top_interactions = self._compute_top_interactions(edge_importance, top_k=top_k)

        # Region stratification
        by_region = None
        if self.region_labels is not None and self.edge_attention_scores is not None:
            if self.edge_attention_scores.ndim == 2:
                by_region = self._compute_importance_by_region()

        # Network summary by edge type
        network_summary = self._compute_network_summary(edge_importance)

        n_edges = len(edge_importance) if edge_importance is not None else 0
        metadata = {
            "n_edges": n_edges,
            "top_k": top_k,
            "has_region_analysis": by_region is not None,
            "n_edge_types": len(self.edge_types),
            "attention_data_source": getattr(self, '_attention_data_source', 'real'),
        }

        return CCCImportanceResult(
            edge_importance=edge_importance,
            top_interactions=top_interactions,
            by_region=by_region,
            network_summary=network_summary,
            metadata=metadata,
        )

    def _compute_edge_importance(self) -> pd.DataFrame:
        """
        Compute edge importance aggregated across subjects.

        Returns:
            DataFrame with columns: source, target, edge_type, mean_attention, std_attention
        """
        if self.edge_attention_scores is None:
            if self.edge_metadata is not None:
                return self.edge_metadata.copy()
            return self._generate_placeholder_edge_importance()

        # edge_attention_scores is available — use it
        self._attention_data_source = "real"
        if self.edge_attention_scores.ndim == 2:
            # [n_subjects, n_edges]
            mean_attention = np.nanmean(self.edge_attention_scores, axis=0)
            std_attention = np.nanstd(self.edge_attention_scores, axis=0)
        else:
            # [n_edges] - already aggregated
            mean_attention = self.edge_attention_scores
            std_attention = np.zeros_like(mean_attention)

        if self.edge_metadata is not None:
            result = self.edge_metadata.copy()
            n_attention = len(mean_attention)
            n_metadata = len(result)
            if n_attention != n_metadata:
                raise ValueError(
                    f"Attention length ({n_attention}) does not match edge metadata "
                    f"length ({n_metadata}). Ensure edge_metadata was computed from "
                    f"the same HGT attention data."
                )
            result["mean_attention"] = mean_attention
            result["std_attention"] = std_attention
        else:
            # No metadata but have scores — create numbered edge DataFrame
            logger.warning(
                "Edge attention scores available but no edge metadata — "
                "output will lack source/target cell type labels."
            )
            result = pd.DataFrame({
                "edge_idx": range(len(mean_attention)),
                "mean_attention": mean_attention,
                "std_attention": std_attention,
            })

        return result

    def _generate_placeholder_edge_importance(self) -> pd.DataFrame:
        """Generate placeholder edge importance for all cell type pairs.

        Internal safety mechanism for direct analyzer usage (notebooks, tests).
        The orchestrator (run_analysis.py) gates CCC execution on data
        availability, so this path is not reached from the standard pipeline.
        Returns zero-filled DataFrame when no edge attention data is available.
        """
        self._attention_data_source = "placeholder"
        rows = []
        for edge_type in self.edge_types:
            for src_idx, src_name in enumerate(self.cell_type_names):
                for tgt_idx, tgt_name in enumerate(self.cell_type_names):
                    rows.append({
                        "source": src_name,
                        "target": tgt_name,
                        "edge_type": edge_type,
                        "source_idx": src_idx,
                        "target_idx": tgt_idx,
                        "mean_attention": 0.0,
                        "std_attention": 0.0,
                    })
        return pd.DataFrame(rows)

    def _compute_top_interactions(
        self,
        edge_importance: pd.DataFrame,
        top_k: int = 100,
    ) -> pd.DataFrame:
        """
        Get top-k interactions by attention weight.

        If lr_pair_mapping is provided, annotates each edge with the contributing
        ligand-receptor pairs. Note that attention is computed at the edge-type
        level, not per-L-R pair, so these are the L-R pairs that contribute to
        each edge rather than individual L-R attention scores.

        Args:
            edge_importance: DataFrame with edge importance
            top_k: Number of top interactions

        Returns:
            DataFrame with top interactions ranked, including 'lr_pairs' column
            if lr_pair_mapping was provided
        """
        if "mean_attention" not in edge_importance.columns:
            # No attention scores, return empty
            return pd.DataFrame(columns=["rank", "source", "target", "edge_type", "mean_attention"])

        df = edge_importance.sort_values("mean_attention", ascending=False).head(top_k).copy()
        df["rank"] = range(1, len(df) + 1)

        # Build column list from what's actually available
        cols = ["rank"]
        for col in ("source", "target", "edge_type", "edge_idx"):
            if col in df.columns:
                cols.append(col)
        cols.append("mean_attention")
        if "std_attention" in df.columns:
            cols.append("std_attention")

        # Add L-R pair annotations if mapping is available
        if self.lr_pair_mapping and "source" in df.columns:
            lr_pairs_col = []
            for _, row in df.iterrows():
                edge_key = f"{row['source']}|{row['target']}|{row['edge_type']}"
                lr_pairs = self.lr_pair_mapping.get(edge_key, [])
                # Join with semicolon for CSV compatibility
                lr_pairs_col.append(";".join(lr_pairs) if lr_pairs else "")
            df["lr_pairs"] = lr_pairs_col
            cols.append("lr_pairs")

        return df[cols].reset_index(drop=True)

    def _compute_importance_by_region(self) -> pd.DataFrame:
        """
        Compute edge importance stratified by brain region.

        Returns:
            DataFrame with columns: region, source, target, edge_type, mean_attention, n_subjects
        """
        if self.region_labels is None or self.edge_attention_scores is None:
            raise ValueError("region_labels and edge_attention_scores required")

        if self.edge_attention_scores.ndim != 2:
            raise ValueError("edge_attention_scores must be 2D for region stratification")

        # Filter out empty strings (subjects with no region label)
        unique_regions = [r for r in np.unique(self.region_labels) if r]

        rows = []
        for region in unique_regions:
            region_str = str(region)
            mask = self.region_labels == region
            n_in_group = mask.sum()

            if n_in_group == 0:
                continue

            group_attention = self.edge_attention_scores[mask]
            mean_attention = np.nanmean(group_attention, axis=0)

            # Combine with edge metadata
            if self.edge_metadata is not None:
                n_attention = len(mean_attention)
                n_metadata = len(self.edge_metadata)
                if n_attention != n_metadata:
                    raise ValueError(
                        f"Region '{region_str}' attention length ({n_attention}) does not match "
                        f"edge metadata length ({n_metadata}). Ensure edge_metadata was computed "
                        f"from the same HGT attention data."
                    )
                for pos, (_idx, row) in enumerate(self.edge_metadata.iterrows()):
                    rows.append({
                        "region": region_str,
                        "source": row.get("source", f"edge_{pos}"),
                        "target": row.get("target", f"edge_{pos}"),
                        "edge_type": row.get("edge_type", "unknown"),
                        "mean_attention": float(mean_attention[pos]),
                        "n_subjects": int(n_in_group),
                    })

        return pd.DataFrame(rows)

    def _compute_network_summary(self, edge_importance: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate importance by edge type category.

        Returns:
            DataFrame with columns: edge_type, display_name, mean_attention, std_attention, n_edges
        """
        if "edge_type" not in edge_importance.columns or "mean_attention" not in edge_importance.columns:
            return pd.DataFrame(columns=["edge_type", "display_name", "mean_attention", "std_attention", "n_edges"])

        summary = edge_importance.groupby("edge_type").agg({
            "mean_attention": ["mean", "std", "count"],
        }).reset_index()

        summary.columns = ["edge_type", "mean_attention", "std_attention", "n_edges"]

        # Add display names
        summary["display_name"] = summary["edge_type"].map(
            lambda x: EDGE_TYPE_DISPLAY_NAMES.get(x, x)
        )

        # Sort by mean attention
        summary = summary.sort_values("mean_attention", ascending=False).reset_index(drop=True)

        return summary[["edge_type", "display_name", "mean_attention", "std_attention", "n_edges"]]

    def save(
        self,
        result: CCCImportanceResult,
        output_dir: str | Path,
        formats: list[Literal["parquet", "csv"]] | None = None,
    ) -> dict[str, Path]:
        """
        Save analysis results to files.

        Args:
            result: CCCImportanceResult to save
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

        # Save edge importance
        for fmt in formats:
            path = output_dir / f"ccc_importance.{fmt}"
            save_dataframe(result.edge_importance, path, fmt)
            saved_files[f"edge_importance_{fmt}"] = path

        # Save top interactions
        for fmt in formats:
            path = output_dir / f"top_interactions.{fmt}"
            save_dataframe(result.top_interactions, path, fmt)
            saved_files[f"top_interactions_{fmt}"] = path

        # Save region-stratified (if available)
        if result.by_region is not None:
            for fmt in formats:
                path = output_dir / f"ccc_importance_by_region.{fmt}"
                save_dataframe(result.by_region, path, fmt)
                saved_files[f"by_region_{fmt}"] = path

        # Save network summary
        if result.network_summary is not None:
            for fmt in formats:
                path = output_dir / f"ccc_network_summary.{fmt}"
                save_dataframe(result.network_summary, path, fmt)
                saved_files[f"network_summary_{fmt}"] = path

        logger.info(f"Saved CCC importance analysis to {output_dir}")
        return saved_files


def compute_ccc_importance(
    edge_attention_scores: np.ndarray | None = None,
    edge_metadata: pd.DataFrame | None = None,
    cell_type_names: list[str] | None = None,
    edge_types: list[str] | None = None,
    region_labels: np.ndarray | None = None,
    top_k: int = 100,
    output_dir: str | Path | None = None,
) -> CCCImportanceResult:
    """
    Convenience function to compute and optionally save CCC importance.

    Args:
        edge_attention_scores: Edge attention [n_subjects, n_edges] or [n_edges]
        edge_metadata: DataFrame with edge info
        cell_type_names: Cell type names
        edge_types: Edge type names
        region_labels: Region labels for stratification
        top_k: Number of top interactions
        output_dir: If provided, save results to this directory

    Returns:
        CCCImportanceResult with analysis results
    """
    analyzer = CCCImportanceAnalyzer(
        edge_attention_scores=edge_attention_scores,
        edge_metadata=edge_metadata,
        cell_type_names=cell_type_names,
        edge_types=edge_types,
        region_labels=region_labels,
    )

    result = analyzer.analyze(top_k=top_k)

    if output_dir is not None:
        analyzer.save(result, output_dir)

    return result


def create_edge_metadata_from_graph(
    edge_index_dict: dict,
    cell_type_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Create edge metadata DataFrame from graph edge index dict.

    Args:
        edge_index_dict: Dict mapping (src_type, edge_type, tgt_type) to edge indices
        cell_type_names: Cell type names

    Returns:
        DataFrame with columns: source, target, edge_type, source_idx, target_idx
    """
    cell_type_names = cell_type_names or list(CELL_TYPE_ORDER)

    rows = []
    for (src_type, edge_type, tgt_type), edge_index in edge_index_dict.items():
        # edge_index is [2, n_edges]
        if hasattr(edge_index, 'numpy'):
            edge_index = edge_index.numpy()

        for i in range(edge_index.shape[1]):
            src_idx = int(edge_index[0, i])
            tgt_idx = int(edge_index[1, i])
            rows.append({
                "source": src_type,
                "target": tgt_type,
                "edge_type": edge_type,
                "source_idx": src_idx,
                "target_idx": tgt_idx,
            })

    return pd.DataFrame(rows)
