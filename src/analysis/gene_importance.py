"""
Gene importance analysis from gene gate attention weights.

Produces analysis outputs:
1. gene_importance_by_celltype.csv - Gene × cell type attention weights matrix
2. top_genes_per_celltype.csv - Top 100 genes per cell type with attention scores
3. gene_importance_by_region.csv - Gene importance stratified by brain region
4. gene_gate_weights.h5 - Full [n_cell_types, n_genes] matrix for programmatic access

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

from src.data.constants import CELL_TYPE_ORDER, REGION_ORDER, N_CELL_TYPES, EPSILON_DIVISION
from src.utils.io import save_dataframe

logger = logging.getLogger(__name__)


@dataclass
class GeneImportanceResult:
    """
    Container for gene importance analysis results.

    Attributes:
        by_celltype: DataFrame with columns [cell_type, gene, gene_idx, weight]
        top_genes: DataFrame with columns [cell_type, rank, gene, gene_idx, weight]
        by_region: DataFrame with columns [region, cell_type, gene, gene_idx, effective_weight] or None
        differential_expression: DataFrame with DE results [cell_type, gene, gene_idx, gate_weight,
            log2_fold_change, pvalue, padj, mean_resilient, mean_vulnerable] or None
        gene_gate_weights: Raw gene gate weights [n_cell_types, n_genes]
        gene_names: List of gene names
        metadata: Additional analysis metadata
    """

    by_celltype: pd.DataFrame
    top_genes: pd.DataFrame
    by_region: pd.DataFrame | None = None
    differential_expression: pd.DataFrame | None = None
    gene_gate_weights: np.ndarray | None = None
    gene_names: list[str] | None = None
    metadata: dict = field(default_factory=dict)


def compute_effective_gene_importance(
    gene_gate_weights: np.ndarray,
    region_expression: np.ndarray,
) -> np.ndarray:
    """
    Compute effective gene importance: gate_weight x mean_expression.

    Args:
        gene_gate_weights: [n_cell_types, n_genes]
        region_expression: [n_cell_types, n_genes] mean expression in region

    Returns:
        [n_cell_types, n_genes] effective importance scores
    """
    return gene_gate_weights * region_expression


class GeneImportanceAnalyzer:
    """
    Analyze gene importance from gene gate attention weights.

    The gene gate learns which genes are important per cell type for cognition
    prediction. This module extracts and analyzes those weights.

    Example:
        >>> analyzer = GeneImportanceAnalyzer(
        ...     gene_gate_weights=weights,  # [n_cell_types, n_genes]
        ...     gene_names=gene_names,
        ... )
        >>> result = analyzer.analyze(top_k=100)
        >>> analyzer.save(result, output_dir)
    """

    def __init__(
        self,
        gene_gate_weights: np.ndarray,
        gene_names: list[str] | None = None,
        cell_type_names: list[str] | None = None,
        region_pseudobulk: dict[str, np.ndarray] | None = None,
    ):
        """
        Initialize analyzer with gene gate weights.

        Args:
            gene_gate_weights: Gene gate attention weights [n_cell_types, n_genes]
            gene_names: List of gene names (generates default names if None)
            cell_type_names: Cell type names (defaults to CELL_TYPE_ORDER)
            region_pseudobulk: Optional dict mapping region names to mean pseudobulk
                              arrays [n_cell_types, n_genes] for effective importance
        """
        self.gene_gate_weights = gene_gate_weights

        # Validate dimensions first
        if gene_gate_weights.ndim != 2:
            raise ValueError(
                f"gene_gate_weights must be 2D [n_cell_types, n_genes], "
                f"got shape {gene_gate_weights.shape}"
            )

        self.n_cell_types, self.n_genes = gene_gate_weights.shape

        self.gene_names = gene_names or [f"gene_{i}" for i in range(self.n_genes)]
        self.cell_type_names = cell_type_names or list(CELL_TYPE_ORDER)[:self.n_cell_types]
        self.region_pseudobulk = region_pseudobulk

        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """Validate input array shapes and consistency."""
        # Note: ndim validation done in __init__ before unpacking

        if len(self.gene_names) != self.n_genes:
            raise ValueError(
                f"gene_names has {len(self.gene_names)} entries but "
                f"gene_gate_weights has {self.n_genes} genes"
            )

        if len(self.cell_type_names) != self.n_cell_types:
            raise ValueError(
                f"cell_type_names has {len(self.cell_type_names)} entries but "
                f"gene_gate_weights has {self.n_cell_types} cell types"
            )

        if self.region_pseudobulk is not None:
            for region, data in self.region_pseudobulk.items():
                if data.shape != self.gene_gate_weights.shape:
                    raise ValueError(
                        f"region_pseudobulk['{region}'] has shape {data.shape} but "
                        f"gene_gate_weights has shape {self.gene_gate_weights.shape}"
                    )

    def analyze(
        self,
        top_k: int = 100,
        group_labels: np.ndarray | None = None,
        subject_expression: np.ndarray | None = None,
        group_a: str = "resilient",
        group_b: str = "vulnerable",
        gate_threshold: float = 0.01,
        apply_fdr: bool = True,
    ) -> GeneImportanceResult:
        """
        Run all gene importance analyses.

        Args:
            top_k: Number of top genes per cell type to extract
            group_labels: Optional [n_subjects] array of group labels for
                differential expression analysis
            subject_expression: Optional [n_subjects, n_cell_types, n_genes]
                per-subject pseudobulk for differential expression analysis
            group_a: Label for first group (numerator in fold change)
            group_b: Label for second group (denominator in fold change)
            gate_threshold: Minimum gate weight to include gene (default: 0.01)
            apply_fdr: Whether to apply Benjamini-Hochberg FDR correction (default: True)

        Returns:
            GeneImportanceResult with all analyses
        """
        by_celltype = self._compute_importance_by_celltype()
        top_genes = self._compute_top_genes(top_k=top_k)

        by_region = None
        if self.region_pseudobulk is not None:
            by_region = self._compute_importance_by_region(top_k=top_k)

        differential = None
        if group_labels is not None and subject_expression is not None:
            differential = self._compute_differential_expression(
                group_labels=group_labels,
                subject_expression=subject_expression,
                group_a=group_a,
                group_b=group_b,
                gate_threshold=gate_threshold,
                apply_fdr=apply_fdr,
            )

        metadata = {
            "n_cell_types": self.n_cell_types,
            "n_genes": self.n_genes,
            "top_k": top_k,
            "has_region_analysis": by_region is not None,
            "has_differential_analysis": differential is not None,
        }

        return GeneImportanceResult(
            by_celltype=by_celltype,
            top_genes=top_genes,
            by_region=by_region,
            differential_expression=differential,
            gene_gate_weights=self.gene_gate_weights,
            gene_names=self.gene_names,
            metadata=metadata,
        )

    def _compute_importance_by_celltype(self) -> pd.DataFrame:
        """
        Compute gene importance for all genes across all cell types.

        Returns:
            DataFrame with columns: cell_type, gene, gene_idx, weight
        """
        n_ct = len(self.cell_type_names)
        n_genes = len(self.gene_names)

        return pd.DataFrame({
            "cell_type": np.repeat(self.cell_type_names, n_genes),
            "gene": np.tile(self.gene_names, n_ct),
            "gene_idx": np.tile(np.arange(n_genes), n_ct),
            "weight": self.gene_gate_weights.ravel(),
        })

    def _compute_top_genes(self, top_k: int = 100) -> pd.DataFrame:
        """
        Get top-k genes per cell type by attention weight.

        Args:
            top_k: Number of top genes per cell type

        Returns:
            DataFrame with columns: cell_type, rank, gene, gene_idx, weight
        """
        rows = []
        for ct_idx, ct_name in enumerate(self.cell_type_names):
            weights = self.gene_gate_weights[ct_idx]
            top_indices = np.argsort(weights)[::-1][:top_k]

            for rank, gene_idx in enumerate(top_indices, 1):
                rows.append({
                    "cell_type": ct_name,
                    "rank": rank,
                    "gene": self.gene_names[gene_idx],
                    "gene_idx": int(gene_idx),
                    "weight": float(weights[gene_idx]),
                })

        return pd.DataFrame(rows)

    def _compute_differential_expression(
        self,
        group_labels: np.ndarray,
        subject_expression: np.ndarray,
        group_a: str = "resilient",
        group_b: str = "vulnerable",
        gate_threshold: float = 0.01,
        apply_fdr: bool = True,
    ) -> pd.DataFrame:
        """
        Compute differential expression between two subject groups for gate-selected genes.

        The gene gate weight acts as a feature selector: only genes with gate weight
        above threshold are tested. This reduces the multiple testing burden and focuses
        on model-relevant genes. The statistical test is Mann-Whitney U on raw expression.

        Args:
            group_labels: [n_subjects] array of group labels
            subject_expression: [n_subjects, n_cell_types, n_genes] per-subject pseudobulk
            group_a: Label for first group (numerator in fold change)
            group_b: Label for second group (denominator in fold change)
            gate_threshold: Minimum gate weight to include gene (default: 0.01)
            apply_fdr: Whether to apply Benjamini-Hochberg FDR correction (default: True)

        Returns:
            DataFrame with columns: cell_type, gene, gene_idx, gate_weight,
                log2_fold_change, pvalue, padj, mean_resilient, mean_vulnerable
        """
        from scipy.stats import mannwhitneyu

        mask_a = group_labels == group_a
        mask_b = group_labels == group_b

        if mask_a.sum() < 5 or mask_b.sum() < 5:
            logger.warning(
                f"Insufficient samples for differential expression: "
                f"{group_a}={mask_a.sum()}, {group_b}={mask_b.sum()} (minimum: 5)"
            )
            return pd.DataFrame()

        expr_a = subject_expression[mask_a]
        expr_b = subject_expression[mask_b]

        # Gate-based feature selection
        gene_mask = self.gene_gate_weights >= gate_threshold
        n_tested = gene_mask.sum()
        n_total = gene_mask.size
        logger.info(
            f"Gate-filtered DE: testing {n_tested}/{n_total} (cell_type, gene) pairs "
            f"(gate_threshold={gate_threshold})"
        )

        if n_tested == 0:
            logger.warning("No genes pass gate threshold — returning empty DataFrame")
            return pd.DataFrame()

        rows = []
        for ct_idx, ct_name in enumerate(self.cell_type_names):
            for gene_idx, gene_name in enumerate(self.gene_names):
                if not gene_mask[ct_idx, gene_idx]:
                    continue

                vals_a = expr_a[:, ct_idx, gene_idx]
                vals_b = expr_b[:, ct_idx, gene_idx]

                mean_a = float(np.mean(vals_a))
                mean_b = float(np.mean(vals_b))

                pseudo = EPSILON_DIVISION
                log2fc = float(np.log2((mean_a + pseudo) / (mean_b + pseudo)))

                try:
                    _, pval = mannwhitneyu(vals_a, vals_b, alternative="two-sided")
                except ValueError:
                    pval = 1.0

                rows.append({
                    "cell_type": ct_name,
                    "gene": gene_name,
                    "gene_idx": gene_idx,
                    "gate_weight": float(self.gene_gate_weights[ct_idx, gene_idx]),
                    "log2_fold_change": log2fc,
                    "pvalue": float(pval),
                    "mean_resilient": mean_a,
                    "mean_vulnerable": mean_b,
                })

        result_df = pd.DataFrame(rows)

        if len(result_df) > 0 and apply_fdr:
            from src.utils.statistics import benjamini_hochberg
            padj, _ = benjamini_hochberg(result_df["pvalue"].values)
            result_df["padj"] = padj
        elif len(result_df) > 0:
            result_df["padj"] = result_df["pvalue"]

        return result_df

    def _compute_importance_by_region(self, top_k: int = 100) -> pd.DataFrame:
        """
        Compute effective gene importance per region.

        Effective importance = gate_weight × mean_expression

        Args:
            top_k: Number of top genes per region per cell type

        Returns:
            DataFrame with columns: region, cell_type, rank, gene, gene_idx,
                                   gate_weight, mean_expression, effective_weight
        """
        if self.region_pseudobulk is None:
            raise ValueError("region_pseudobulk required for region analysis")

        rows = []
        for region, region_data in self.region_pseudobulk.items():
            # Effective importance = gate_weight × mean_expression
            effective_importance = compute_effective_gene_importance(self.gene_gate_weights, region_data)

            for ct_idx, ct_name in enumerate(self.cell_type_names):
                eff_weights = effective_importance[ct_idx]
                top_indices = np.argsort(eff_weights)[::-1][:top_k]

                for rank, gene_idx in enumerate(top_indices, 1):
                    rows.append({
                        "region": region,
                        "cell_type": ct_name,
                        "rank": rank,
                        "gene": self.gene_names[gene_idx],
                        "gene_idx": int(gene_idx),
                        "gate_weight": float(self.gene_gate_weights[ct_idx, gene_idx]),
                        "mean_expression": float(region_data[ct_idx, gene_idx]),
                        "effective_weight": float(eff_weights[gene_idx]),
                    })

        return pd.DataFrame(rows)

    def save(
        self,
        result: GeneImportanceResult,
        output_dir: str | Path,
        formats: list[Literal["parquet", "csv"]] | None = None,
    ) -> dict[str, Path]:
        """
        Save analysis results to files.

        Args:
            result: GeneImportanceResult to save
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

        # Save gene importance by cell type
        for fmt in formats:
            path = output_dir / f"gene_importance_by_celltype.{fmt}"
            save_dataframe(result.by_celltype, path, fmt)
            saved_files[f"by_celltype_{fmt}"] = path

        # Save top genes per cell type
        for fmt in formats:
            path = output_dir / f"top_genes_per_celltype.{fmt}"
            save_dataframe(result.top_genes, path, fmt)
            saved_files[f"top_genes_{fmt}"] = path

        # Save region-stratified (if available)
        if result.by_region is not None:
            for fmt in formats:
                path = output_dir / f"gene_importance_by_region.{fmt}"
                save_dataframe(result.by_region, path, fmt)
                saved_files[f"by_region_{fmt}"] = path

        # Save differential expression (if available)
        if result.differential_expression is not None and len(result.differential_expression) > 0:
            for fmt in formats:
                path = output_dir / f"differential_expression.{fmt}"
                save_dataframe(result.differential_expression, path, fmt)
                saved_files[f"differential_expression_{fmt}"] = path

        # Save raw weights as HDF5
        if result.gene_gate_weights is not None:
            h5_path = output_dir / "gene_gate_weights.h5"
            self._save_hdf5(result, h5_path)
            saved_files["hdf5"] = h5_path

        logger.info(f"Saved gene importance analysis to {output_dir}")
        return saved_files

    def _save_hdf5(self, result: GeneImportanceResult, path: Path) -> None:
        """Save gene gate weights to HDF5."""
        with h5py.File(path, "w") as f:
            f.attrs["schema_version"] = "2.0"
            f.attrs["n_cell_types"] = self.n_cell_types
            f.attrs["n_genes"] = self.n_genes

            # Gene gate weights
            f.create_dataset(
                "gene_gate",
                data=result.gene_gate_weights,
                compression="gzip",
                compression_opts=4,
            )
            f["gene_gate"].attrs["shape"] = "[n_cell_types, n_genes]"

            # Variable-length string type
            vlen_str = h5py.special_dtype(vlen=str)

            # Gene names
            f.create_dataset("gene_names", data=np.array(result.gene_names, dtype=object), dtype=vlen_str)

            # Cell type names
            f.create_dataset("cell_type_names", data=np.array(self.cell_type_names, dtype=object), dtype=vlen_str)


def compute_gene_importance(
    gene_gate_weights: np.ndarray,
    gene_names: list[str] | None = None,
    cell_type_names: list[str] | None = None,
    region_pseudobulk: dict[str, np.ndarray] | None = None,
    top_k: int = 100,
    output_dir: str | Path | None = None,
    # Differential expression parameters
    group_labels: np.ndarray | None = None,
    subject_expression: np.ndarray | None = None,
    group_a: str = "resilient",
    group_b: str = "vulnerable",
    gate_threshold: float = 0.01,
    apply_fdr: bool = True,
) -> GeneImportanceResult:
    """
    Convenience function to compute and optionally save gene importance.

    Args:
        gene_gate_weights: Gene gate attention weights [n_cell_types, n_genes]
        gene_names: List of gene names
        cell_type_names: Cell type names
        region_pseudobulk: Dict mapping region names to mean pseudobulk for
                          effective importance computation
        top_k: Number of top genes per cell type
        output_dir: If provided, save results to this directory
        group_labels: Optional [n_subjects] array of group labels for
            differential expression analysis
        subject_expression: Optional [n_subjects, n_cell_types, n_genes]
            per-subject pseudobulk for differential expression analysis
        group_a: Label for first group (numerator in fold change)
        group_b: Label for second group (denominator in fold change)
        gate_threshold: Minimum gate weight to include gene (default: 0.01)
        apply_fdr: Whether to apply Benjamini-Hochberg FDR correction (default: True)

    Returns:
        GeneImportanceResult with analysis results
    """
    analyzer = GeneImportanceAnalyzer(
        gene_gate_weights=gene_gate_weights,
        gene_names=gene_names,
        cell_type_names=cell_type_names,
        region_pseudobulk=region_pseudobulk,
    )

    result = analyzer.analyze(
        top_k=top_k,
        group_labels=group_labels,
        subject_expression=subject_expression,
        group_a=group_a,
        group_b=group_b,
        gate_threshold=gate_threshold,
        apply_fdr=apply_fdr,
    )

    if output_dir is not None:
        analyzer.save(result, output_dir)

    return result


def load_gene_gate_weights_hdf5(path: str | Path) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Load gene gate weights from HDF5 file.

    Args:
        path: Path to HDF5 file

    Returns:
        Tuple of (gene_gate_weights, gene_names, cell_type_names)
    """
    def _safe_decode(x):
        return x.decode("utf-8") if isinstance(x, bytes) else str(x)

    with h5py.File(path, "r") as f:
        gene_gate_weights = f["gene_gate"][:]
        gene_names = [_safe_decode(x) for x in f["gene_names"][:]]
        cell_type_names = [_safe_decode(x) for x in f["cell_type_names"][:]]

    return gene_gate_weights, gene_names, cell_type_names
