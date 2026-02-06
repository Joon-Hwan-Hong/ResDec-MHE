"""
Regional analysis of attention patterns and gene importance.

Produces analysis outputs:
1. regional_attention_summary.csv - Attention patterns per brain region
2. regional_gene_importance.csv - Top genes per region per cell type
3. region_contribution.csv - Learned region weights from RegionHandler

Output format: Tidy DataFrames saved as Parquet (primary) and CSV (human-readable).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from src.analysis.gene_importance import compute_effective_gene_importance
from src.data.constants import CELL_TYPE_ORDER, REGION_ORDER, N_CELL_TYPES, N_REGIONS
from src.utils.io import save_dataframe

logger = logging.getLogger(__name__)


@dataclass
class RegionalAnalysisResult:
    """
    Container for regional analysis results.

    Attributes:
        attention_summary: DataFrame with attention patterns per region
        gene_importance: DataFrame with top genes per region per cell type
        region_contribution: DataFrame with learned region weights
        metadata: Additional analysis metadata
    """

    attention_summary: pd.DataFrame
    per_subject_attention: pd.DataFrame | None = None
    gene_importance: pd.DataFrame | None = None
    region_contribution: pd.DataFrame | None = None
    metadata: dict = field(default_factory=dict)


class RegionalAnalyzer:
    """
    Analyze attention patterns and importance across brain regions.

    Examines how the model weights different brain regions and how
    gene/cell type importance varies by region.

    Example:
        >>> analyzer = RegionalAnalyzer(
        ...     region_attention=attention,       # [n_subjects, n_regions, ...]
        ...     region_weights=region_weights,    # [n_regions] from RegionHandler
        ...     gene_gate_weights=gene_gate,      # [n_cell_types, n_genes]
        ...     region_pseudobulk=region_data,    # Dict[region, [n_cell_types, n_genes]]
        ... )
        >>> result = analyzer.analyze()
        >>> analyzer.save(result, output_dir)
    """

    def __init__(
        self,
        region_attention: np.ndarray | None = None,
        region_weights: np.ndarray | None = None,
        gene_gate_weights: np.ndarray | None = None,
        region_pseudobulk: dict[str, np.ndarray] | None = None,
        region_names: list[str] | None = None,
        cell_type_names: list[str] | None = None,
        gene_names: list[str] | None = None,
        subject_ids: list[str] | None = None,
    ):
        """
        Initialize analyzer with regional data.

        Args:
            region_attention: Per-region attention [n_subjects, n_regions, n_cell_types]
                             or [n_subjects, n_regions] if aggregated
            region_weights: Learned region weights from RegionHandler [n_regions]
            gene_gate_weights: Gene gate weights [n_cell_types, n_genes]
            region_pseudobulk: Dict mapping region names to mean pseudobulk [n_cell_types, n_genes]
            region_names: Region names (defaults to REGION_ORDER)
            cell_type_names: Cell type names (defaults to CELL_TYPE_ORDER)
            gene_names: Gene names
            subject_ids: Subject identifiers
        """
        self.region_attention = region_attention
        self.region_weights = region_weights
        self.gene_gate_weights = gene_gate_weights
        self.region_pseudobulk = region_pseudobulk
        self.region_names = region_names or list(REGION_ORDER)
        self.cell_type_names = cell_type_names or list(CELL_TYPE_ORDER)
        self.gene_names = gene_names
        self.subject_ids = subject_ids

    def analyze(self, top_k_genes: int = 100) -> RegionalAnalysisResult:
        """
        Run all regional analyses.

        Args:
            top_k_genes: Number of top genes per region per cell type

        Returns:
            RegionalAnalysisResult with all analyses
        """
        # Compute attention summary
        attention_summary = self._compute_attention_summary()

        # Compute per-subject attention table
        per_subject_attention = self._compute_per_subject_attention()

        # Compute region contribution (from learned weights)
        region_contribution = self._compute_region_contribution()

        # Compute gene importance per region (if data available)
        gene_importance = None
        if (self.gene_gate_weights is not None and
            self.region_pseudobulk is not None and
            self.gene_names is not None):
            gene_importance = self._compute_gene_importance_by_region(top_k=top_k_genes)

        metadata = {
            "n_regions": len(self.region_names),
            "has_attention_data": self.region_attention is not None,
            "has_region_weights": self.region_weights is not None,
            "has_gene_importance": gene_importance is not None,
            "top_k_genes": top_k_genes,
        }

        return RegionalAnalysisResult(
            attention_summary=attention_summary,
            per_subject_attention=per_subject_attention,
            gene_importance=gene_importance,
            region_contribution=region_contribution,
            metadata=metadata,
        )

    def _compute_attention_summary(self) -> pd.DataFrame:
        """
        Compute attention summary statistics per region.

        Returns:
            DataFrame with columns: region, mean_attention, std_attention, n_subjects
        """
        if self.region_attention is None:
            # Return placeholder with region names only
            return pd.DataFrame({
                "region": self.region_names,
                "mean_attention": [np.nan] * len(self.region_names),
                "std_attention": [np.nan] * len(self.region_names),
                "n_subjects": [0] * len(self.region_names),
            })

        # Handle different attention shapes
        if self.region_attention.ndim == 2:
            # [n_subjects, n_regions] - already aggregated
            attention_per_region = self.region_attention
        elif self.region_attention.ndim == 3:
            # [n_subjects, n_regions, n_cell_types] - aggregate over cell types
            attention_per_region = self.region_attention.mean(axis=2)
        else:
            raise ValueError(f"Unexpected attention shape: {self.region_attention.shape}")

        n_subjects = attention_per_region.shape[0]
        n_regions = min(attention_per_region.shape[1], len(self.region_names))

        rows = []
        for region_idx in range(n_regions):
            region_attention = attention_per_region[:, region_idx]
            rows.append({
                "region": self.region_names[region_idx],
                "mean_attention": float(region_attention.mean()),
                "std_attention": float(region_attention.std()),
                "n_subjects": n_subjects,
            })

        df = pd.DataFrame(rows)
        df = df.sort_values("mean_attention", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)

        return df

    def _compute_per_subject_attention(self) -> pd.DataFrame | None:
        """
        Compute per-subject regional attention table.

        Returns:
            DataFrame with columns: subject_id, region, attention_weight
        """
        if self.region_attention is None:
            return None

        # Handle different shapes
        if self.region_attention.ndim == 2:
            attention_per_region = self.region_attention
        elif self.region_attention.ndim == 3:
            attention_per_region = self.region_attention.mean(axis=2)
        else:
            return None

        n_subjects = attention_per_region.shape[0]
        n_regions = min(attention_per_region.shape[1], len(self.region_names))
        subject_ids = self.subject_ids or [f"subject_{i}" for i in range(n_subjects)]

        rows = []
        for subj_idx, subj_id in enumerate(subject_ids):
            for region_idx in range(n_regions):
                rows.append({
                    "subject_id": subj_id,
                    "region": self.region_names[region_idx],
                    "attention_weight": float(attention_per_region[subj_idx, region_idx]),
                })

        return pd.DataFrame(rows)

    def _compute_region_contribution(self) -> pd.DataFrame | None:
        """
        Compute region contribution from learned weights.

        Returns:
            DataFrame with columns: region, weight, rank, normalized_weight
        """
        if self.region_weights is None:
            return None

        n_regions = min(len(self.region_weights), len(self.region_names))

        df = pd.DataFrame({
            "region": self.region_names[:n_regions],
            "weight": self.region_weights[:n_regions],
        })

        # Sort by weight
        df = df.sort_values("weight", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)

        # Normalize weights (softmax-like interpretation)
        total = df["weight"].sum()
        df["normalized_weight"] = df["weight"] / total if total > 0 else 0.0

        return df[["region", "weight", "normalized_weight", "rank"]]

    def _compute_gene_importance_by_region(self, top_k: int = 100) -> pd.DataFrame:
        """
        Compute effective gene importance per region.

        Effective importance = gate_weight × mean_expression_in_region

        Args:
            top_k: Number of top genes per region per cell type

        Returns:
            DataFrame with columns: region, cell_type, rank, gene, gate_weight,
                                   mean_expression, effective_weight
        """
        if self.gene_gate_weights is None or self.region_pseudobulk is None:
            raise ValueError("gene_gate_weights and region_pseudobulk required")

        if self.gene_names is None:
            n_genes = self.gene_gate_weights.shape[1]
            self.gene_names = [f"gene_{i}" for i in range(n_genes)]

        rows = []
        for region, region_data in self.region_pseudobulk.items():
            if region not in self.region_names:
                continue

            # Effective importance = gate_weight × mean_expression
            effective_importance = compute_effective_gene_importance(self.gene_gate_weights, region_data)

            n_cell_types = min(effective_importance.shape[0], len(self.cell_type_names))
            for ct_idx in range(n_cell_types):
                ct_name = self.cell_type_names[ct_idx]
                eff_weights = effective_importance[ct_idx]
                top_indices = np.argsort(eff_weights)[::-1][:top_k]

                for rank, gene_idx in enumerate(top_indices, 1):
                    rows.append({
                        "region": region,
                        "cell_type": ct_name,
                        "rank": rank,
                        "gene": self.gene_names[gene_idx] if gene_idx < len(self.gene_names) else f"gene_{gene_idx}",
                        "gene_idx": int(gene_idx),
                        "gate_weight": float(self.gene_gate_weights[ct_idx, gene_idx]),
                        "mean_expression": float(region_data[ct_idx, gene_idx]),
                        "effective_weight": float(eff_weights[gene_idx]),
                    })

        return pd.DataFrame(rows)

    def save(
        self,
        result: RegionalAnalysisResult,
        output_dir: str | Path,
        formats: list[Literal["parquet", "csv"]] | None = None,
    ) -> dict[str, Path]:
        """
        Save analysis results to files.

        Args:
            result: RegionalAnalysisResult to save
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

        # Save attention summary
        for fmt in formats:
            path = output_dir / f"regional_attention_summary.{fmt}"
            save_dataframe(result.attention_summary, path, fmt)
            saved_files[f"attention_summary_{fmt}"] = path

        # Save per-subject attention (if available)
        if result.per_subject_attention is not None:
            for fmt in formats:
                path = output_dir / f"regional_attention_per_subject.{fmt}"
                save_dataframe(result.per_subject_attention, path, fmt)
                saved_files[f"per_subject_attention_{fmt}"] = path

        # Save region contribution (if available)
        if result.region_contribution is not None:
            for fmt in formats:
                path = output_dir / f"region_contribution.{fmt}"
                save_dataframe(result.region_contribution, path, fmt)
                saved_files[f"region_contribution_{fmt}"] = path

        # Save gene importance (if available)
        if result.gene_importance is not None:
            for fmt in formats:
                path = output_dir / f"regional_gene_importance.{fmt}"
                save_dataframe(result.gene_importance, path, fmt)
                saved_files[f"gene_importance_{fmt}"] = path

        logger.info(f"Saved regional analysis to {output_dir}")
        return saved_files


def compute_regional_analysis(
    region_attention: np.ndarray | None = None,
    region_weights: np.ndarray | None = None,
    gene_gate_weights: np.ndarray | None = None,
    region_pseudobulk: dict[str, np.ndarray] | None = None,
    region_names: list[str] | None = None,
    cell_type_names: list[str] | None = None,
    gene_names: list[str] | None = None,
    subject_ids: list[str] | None = None,
    top_k_genes: int = 100,
    output_dir: str | Path | None = None,
) -> RegionalAnalysisResult:
    """
    Convenience function to compute and optionally save regional analysis.

    Args:
        region_attention: Per-region attention weights
        region_weights: Learned region weights from RegionHandler
        gene_gate_weights: Gene gate weights
        region_pseudobulk: Dict mapping region names to mean pseudobulk
        region_names: Region names
        cell_type_names: Cell type names
        gene_names: Gene names
        subject_ids: Subject identifiers
        top_k_genes: Number of top genes per region per cell type
        output_dir: If provided, save results to this directory

    Returns:
        RegionalAnalysisResult with analysis results
    """
    analyzer = RegionalAnalyzer(
        region_attention=region_attention,
        region_weights=region_weights,
        gene_gate_weights=gene_gate_weights,
        region_pseudobulk=region_pseudobulk,
        region_names=region_names,
        cell_type_names=cell_type_names,
        gene_names=gene_names,
        subject_ids=subject_ids,
    )

    result = analyzer.analyze(top_k_genes=top_k_genes)

    if output_dir is not None:
        analyzer.save(result, output_dir)

    return result
