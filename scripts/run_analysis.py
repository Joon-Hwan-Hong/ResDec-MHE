"""
Run post-hoc analysis pipeline on trained model outputs.

Usage:
    uv run python scripts/run_analysis.py --experiment-dir experiments/20260113_143052_a3f7b2c1
    uv run python scripts/run_analysis.py --experiment-dir experiments/20260113_143052_a3f7b2c1 --skip-ccc
    uv run python scripts/run_analysis.py --predictions-path analysis/predictions.parquet --attention-path analysis/attention.h5

Workflow:
1. Load predictions and attention weights from experiment directory (or explicit paths)
2. Run cell type importance analysis
3. Run gene importance analysis
4. Run CCC importance analysis (if HGT attention available)
5. Run resilience signature analysis (if pathology data available)
6. Run regional analysis (if region data available)
7. Run uncertainty analysis (if std predictions available)
8. Run cell attention analysis (if PMA attention available)
9. Run embedding analysis (if embeddings available)
10. Save all results to analysis output directory

Outputs saved to: {experiment_dir}/analysis/ or --output-dir
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import load_dataframe, load_attention_weights, save_dataframe, unpack_hgt_for_ccc
from src.utils.statistics import derive_resilience_groups
from src.analysis import (
    CellTypeImportanceAnalyzer,
    GeneImportanceAnalyzer,
    CCCImportanceAnalyzer,
    ResilienceSignatureAnalyzer,
    RegionalAnalyzer,
    UncertaintyAnalyzer,
    compute_expected_calibration_error,
    EmbeddingAnalyzer,
)
from src.data.constants import CELL_TYPE_ORDER, REGION_ORDER

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run post-hoc analysis on trained model outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input sources (either experiment-dir or explicit paths)
    input_group = parser.add_argument_group("Input Sources")
    input_group.add_argument(
        "--experiment-dir",
        type=str,
        help="Path to experiment directory containing analysis/ subdirectory",
    )
    input_group.add_argument(
        "--predictions-path",
        type=str,
        help="Explicit path to predictions parquet file",
    )
    input_group.add_argument(
        "--attention-path",
        type=str,
        help="Explicit path to attention weights HDF5 file",
    )
    input_group.add_argument(
        "--metadata-path",
        type=str,
        help="Path to subject metadata CSV/parquet with pathology info",
    )
    input_group.add_argument(
        "--liana-dir",
        type=str,
        default=None,
        help="Directory containing per-subject LIANA results (parquet files)",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for analysis results (default: {experiment-dir}/analysis/)",
    )

    # Analysis options
    analysis_group = parser.add_argument_group("Analysis Options")
    analysis_group.add_argument(
        "--skip-cell-type",
        action="store_true",
        help="Skip cell type importance analysis",
    )
    analysis_group.add_argument(
        "--skip-gene",
        action="store_true",
        help="Skip gene importance analysis",
    )
    analysis_group.add_argument(
        "--skip-ccc",
        action="store_true",
        help="Skip cell-cell communication importance analysis",
    )
    analysis_group.add_argument(
        "--skip-resilience",
        action="store_true",
        help="Skip resilience signature analysis",
    )
    analysis_group.add_argument(
        "--skip-regional",
        action="store_true",
        help="Skip regional analysis",
    )
    analysis_group.add_argument(
        "--skip-uncertainty",
        action="store_true",
        help="Skip uncertainty analysis",
    )
    analysis_group.add_argument(
        "--skip-embedding",
        action="store_true",
        help="Skip embedding analysis (UMAP, clustering, linear probes)",
    )
    analysis_group.add_argument(
        "--skip-cell-heterogeneity",
        action="store_true",
        help="Skip within-cell-type heterogeneity analysis (PMA attention)",
    )

    # Parameters
    param_group = parser.add_argument_group("Parameters")
    param_group.add_argument(
        "--top-k-genes",
        type=int,
        default=100,
        help="Number of top genes per cell type (default: 100)",
    )
    param_group.add_argument(
        "--n-permutations",
        type=int,
        default=1000,
        help="Number of permutations for resilience signature test (default: 1000)",
    )
    param_group.add_argument(
        "--fdr-threshold",
        type=float,
        default=0.05,
        help="FDR threshold for significance (default: 0.05)",
    )
    param_group.add_argument(
        "--no-fdr",
        action="store_true",
        help="Disable FDR correction (report uncorrected p-values)",
    )
    param_group.add_argument(
        "--top-percentile",
        type=float,
        default=10.0,
        help="Top percentile for high-attention cells in heterogeneity analysis (default: 10.0)",
    )
    param_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible permutation tests (default: 42)",
    )
    param_group.add_argument(
        "--gate-threshold", type=float, default=0.01,
        help="Minimum gene gate weight to include in differential expression (default: 0.01)",
    )
    param_group.add_argument(
        "--no-fdr-correction", action="store_true", default=False,
        help="Disable Benjamini-Hochberg FDR correction for differential expression",
    )
    param_group.add_argument(
        "--run-ablation",
        action="store_true",
        help="Run ablation study as part of resilience signature analysis",
    )
    param_group.add_argument(
        "--ablation-method",
        type=str,
        default="both",
        choices=["both", "zero_embedding", "node_removal"],
        help="Ablation method for resilience analysis (default: both)",
    )

    # Output formats
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["parquet", "csv"],
        choices=["parquet", "csv"],
        help="Output formats (default: parquet csv)",
    )

    return parser.parse_args()


def load_predictions(path: Path) -> pd.DataFrame:
    """Load predictions from parquet or CSV file."""
    path = Path(path)
    # Check for supported formats
    if path.suffix not in (".parquet", ".csv", ""):
        raise ValueError(f"Unsupported predictions format: {path.suffix}")
    df = load_dataframe(path)
    if df is None:
        raise ValueError(f"Could not load predictions from: {path}")
    return df




def _align_array_by_subject_id(
    source_df: pd.DataFrame,
    target_subject_ids: list[str],
    column: str,
    source_id_column: str = "subject_id",
) -> np.ndarray | None:
    """Align a column from source_df to match target_subject_ids order.

    Args:
        source_df: DataFrame containing the data column and subject ID column.
        target_subject_ids: Subject IDs in the desired order (from attention weights).
        column: Data column to extract and align.
        source_id_column: Column name for subject ID in source_df.

    Returns:
        np.ndarray aligned to target_subject_ids order, or None if column/ID not found.
    """
    if column not in source_df.columns:
        return None
    if source_id_column not in source_df.columns:
        return None

    meta_indexed = source_df.set_index(source_id_column)
    aligned = []
    n_missing = 0
    for sid in target_subject_ids:
        if sid in meta_indexed.index:
            aligned.append(meta_indexed.loc[sid, column])
        else:
            aligned.append(np.nan)
            n_missing += 1

    if n_missing > 0:
        logger.warning(
            f"  {n_missing}/{len(target_subject_ids)} subjects missing from "
            f"metadata for column '{column}'"
        )

    return np.array(aligned, dtype=float)


def _align_predictions_to_subjects(
    predictions_df: pd.DataFrame,
    target_subject_ids: list[str],
) -> pd.DataFrame:
    """Reindex predictions_df to match target_subject_ids order.

    If predictions_df has a 'subject_id' column, rows are reordered (and
    potentially subset/extended with NaN) to match target_subject_ids.
    If no 'subject_id' column, returns unchanged with a warning.
    """
    if "subject_id" not in predictions_df.columns:
        logger.warning(
            "predictions_df has no 'subject_id' column — cannot align "
            "to attention_weights subject order. Assuming positional match."
        )
        return predictions_df

    aligned = predictions_df.set_index("subject_id").reindex(target_subject_ids)
    n_missing = aligned.isna().all(axis=1).sum()
    aligned = aligned.reset_index()
    if n_missing > 0:
        logger.warning(
            f"{n_missing}/{len(target_subject_ids)} attention subjects not found "
            "in predictions_df"
        )
    return aligned


def run_cell_type_importance(
    pathology_attention: np.ndarray,
    cell_type_names: list[str],
    pathology_levels: np.ndarray | None = None,
    region_labels: np.ndarray | None = None,
    output_dir: Path = None,
    formats: list[str] = None,
) -> None:
    """Run cell type importance analysis."""
    logger.info("Running cell type importance analysis...")

    analyzer = CellTypeImportanceAnalyzer(
        attention=pathology_attention,
        cell_type_names=cell_type_names,
        pathology_scores=pathology_levels,
        region_labels=region_labels,
    )

    result = analyzer.analyze()

    if output_dir:
        analyzer.save(result, output_dir, formats=formats)
        logger.info(f"  Saved cell type importance to {output_dir}")


def run_gene_importance(
    gene_gate_weights: np.ndarray,
    cell_type_names: list[str],
    gene_names: list[str] | None = None,
    region_pseudobulk: dict[str, np.ndarray] | None = None,
    top_k: int = 100,
    output_dir: Path = None,
    formats: list[str] = None,
    group_labels: np.ndarray | None = None,
    subject_expression: np.ndarray | None = None,
    gate_threshold: float = 0.01,
    apply_fdr: bool = True,
) -> None:
    """Run gene importance analysis."""
    logger.info("Running gene importance analysis...")

    analyzer = GeneImportanceAnalyzer(
        gene_gate_weights=gene_gate_weights,
        cell_type_names=cell_type_names,
        gene_names=gene_names,
        region_pseudobulk=region_pseudobulk,
    )

    result = analyzer.analyze(
        top_k=top_k,
        group_labels=group_labels,
        subject_expression=subject_expression,
        gate_threshold=gate_threshold,
        apply_fdr=apply_fdr,
    )

    if output_dir:
        analyzer.save(result, output_dir, formats=formats)
        logger.info(f"  Saved gene importance to {output_dir}")


def run_ccc_importance(
    edge_attention_scores: np.ndarray | None = None,
    edge_metadata: pd.DataFrame | None = None,
    cell_type_names: list[str] | None = None,
    edge_types: list[str] | None = None,
    lr_pair_mapping: dict[str, list[str]] | None = None,
    region_labels: np.ndarray | None = None,
    subject_ids: list[str] | None = None,
    output_dir: Path = None,
    formats: list[str] = None,
) -> None:
    """Run cell-cell communication importance analysis."""
    logger.info("Running CCC importance analysis...")

    analyzer = CCCImportanceAnalyzer(
        edge_attention_scores=edge_attention_scores,
        edge_metadata=edge_metadata,
        cell_type_names=cell_type_names,
        edge_types=edge_types,
        lr_pair_mapping=lr_pair_mapping,
        region_labels=region_labels,
        subject_ids=subject_ids,
    )

    result = analyzer.analyze()

    if output_dir:
        analyzer.save(result, output_dir, formats=formats)
        logger.info(f"  Saved CCC importance to {output_dir}")


def run_resilience_signature(
    attention: np.ndarray,
    pathology_scores: np.ndarray,
    cognition_scores: np.ndarray,
    cell_type_names: list[str],
    n_permutations: int = 1000,
    random_seed: int | None = None,
    apply_fdr_correction: bool = True,
    fdr_threshold: float = 0.05,
    run_ablation: bool = False,
    ablation_method: str = "both",
    embeddings: np.ndarray | None = None,
    region_labels: np.ndarray | None = None,
    subject_ids: list[str] | None = None,
    output_dir: Path = None,
    formats: list[str] = None,
) -> None:
    """Run resilience signature analysis."""
    logger.info("Running resilience signature analysis...")

    analyzer = ResilienceSignatureAnalyzer(
        attention=attention,
        pathology_scores=pathology_scores,
        cognition_scores=cognition_scores,
        cell_type_names=cell_type_names,
        region_labels=region_labels,
        subject_ids=subject_ids,
    )

    result = analyzer.analyze(
        n_permutations=n_permutations,
        random_seed=random_seed,
        apply_fdr_correction=apply_fdr_correction,
        fdr_threshold=fdr_threshold,
        run_ablation=run_ablation,
        ablation_method=ablation_method,
        embeddings=embeddings,
    )

    if output_dir:
        analyzer.save(result, output_dir, formats=formats)
        logger.info(f"  Saved resilience signature to {output_dir}")


def run_regional_analysis(
    region_attention: np.ndarray | None = None,
    region_weights: np.ndarray | None = None,
    gene_gate_weights: np.ndarray | None = None,
    region_pseudobulk: dict | None = None,
    region_names: list[str] | None = None,
    cell_type_names: list[str] | None = None,
    gene_names: list[str] | None = None,
    subject_ids: list[str] | None = None,
    top_k_genes: int = 100,
    output_dir: Path = None,
    formats: list[str] = None,
) -> None:
    """Run regional analysis."""
    logger.info("Running regional analysis...")

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

    if output_dir:
        analyzer.save(result, output_dir, formats=formats)
        logger.info(f"  Saved regional analysis to {output_dir}")


def run_uncertainty_analysis(
    predicted_mean: np.ndarray,
    predicted_std: np.ndarray,
    actual: np.ndarray | None = None,
    subject_ids: list[str] | None = None,
    covariates: pd.DataFrame | None = None,
    output_dir: Path = None,
    formats: list[str] = None,
) -> None:
    """Run uncertainty analysis."""
    logger.info("Running uncertainty analysis...")

    analyzer = UncertaintyAnalyzer(
        predicted_mean=predicted_mean,
        predicted_std=predicted_std,
        actual=actual,
        subject_ids=subject_ids,
        covariates=covariates,
    )

    result = analyzer.analyze()

    if output_dir:
        analyzer.save(result, output_dir, formats=formats)
        logger.info(f"  Saved uncertainty analysis to {output_dir}")

    # Log ECE if actual values available
    if actual is not None:
        ece = compute_expected_calibration_error(predicted_mean, predicted_std, actual)
        logger.info(f"  Expected Calibration Error (ECE): {ece:.4f}")



def run_embedding_analysis(
    embeddings: np.ndarray,
    subject_ids: list[str] | None = None,
    covariates: pd.DataFrame | None = None,
    batch_labels: np.ndarray | None = None,
    output_dir: Path = None,
    formats: list[str] = None,
) -> None:
    """Run embedding analysis (UMAP, clustering, linear probes)."""
    logger.info("Running embedding analysis...")

    analyzer = EmbeddingAnalyzer(
        embeddings=embeddings,
        subject_ids=subject_ids,
        covariates=covariates,
        batch_labels=batch_labels,
    )

    result = analyzer.analyze()

    if output_dir:
        analyzer.save(result, output_dir, formats=formats)
        logger.info(f"  Saved embedding analysis to {output_dir}")


def main():
    """Main entry point."""
    args = parse_args()

    # Resolve input paths
    if args.experiment_dir:
        exp_dir = Path(args.experiment_dir)
        analysis_dir = exp_dir / "analysis"

        predictions_path = args.predictions_path or analysis_dir / "predictions.parquet"
        attention_path = args.attention_path or analysis_dir / "attention_weights.h5"
        output_dir = Path(args.output_dir) if args.output_dir else analysis_dir
    else:
        if not args.predictions_path and not args.attention_path:
            raise ValueError(
                "Must provide either --experiment-dir or explicit paths "
                "(--predictions-path, --attention-path)"
            )
        predictions_path = Path(args.predictions_path) if args.predictions_path else None
        attention_path = Path(args.attention_path) if args.attention_path else None
        output_dir = Path(args.output_dir) if args.output_dir else Path("analysis_output")

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Analysis output directory: {output_dir}")

    # Load predictions if available
    predictions_df = None
    if predictions_path and predictions_path.exists():
        logger.info(f"Loading predictions from {predictions_path}")
        predictions_df = load_predictions(predictions_path)
        logger.info(f"  Loaded {len(predictions_df)} predictions")

    # Load attention weights if available
    attention_weights = {}
    if attention_path and attention_path.exists():
        logger.info(f"Loading attention weights from {attention_path}")
        attention_weights = load_attention_weights(attention_path)
        logger.info(f"  Loaded keys: {list(attention_weights.keys())}")

    # Get metadata from attention weights (stored in "metadata" key by io.load_attention_weights)
    metadata = attention_weights.get("metadata", {})
    cell_type_names = list(metadata.get("cell_type_names", CELL_TYPE_ORDER))
    gene_names = metadata.get("gene_names")
    if gene_names is not None:
        gene_names = list(gene_names)
    subject_ids = metadata.get("subject_ids")
    if subject_ids is not None:
        subject_ids = list(subject_ids)

    # Align predictions_df to attention_weights subject order
    if predictions_df is not None and subject_ids is not None:
        predictions_df = _align_predictions_to_subjects(predictions_df, subject_ids)
        logger.info("  Aligned predictions_df to attention_weights subject order")

    # Load metadata if provided
    metadata_df = None
    if args.metadata_path:
        metadata_path = Path(args.metadata_path)
        if metadata_path.suffix == ".parquet":
            metadata_df = pd.read_parquet(metadata_path)
        else:
            metadata_df = pd.read_csv(metadata_path)
        logger.info(f"Loaded metadata with {len(metadata_df)} subjects")

    # Extract arrays for analysis
    pathology_attention = attention_weights.get("pathology_attention")
    gene_gate = attention_weights.get("gene_gate")
    hgt_raw = attention_weights.get("hgt_attention")
    region_weights = attention_weights.get("region_weights")
    region_attention = attention_weights.get("region_attention")
    region_pseudobulk_raw = attention_weights.get("region_pseudobulk")
    embeddings = attention_weights.get("embeddings")
    cell_counts = attention_weights.get("cell_counts")  # [n_subjects, n_cell_types] or None

    # Convert region_pseudobulk array to dict format for analyzers
    # Prefer explicit region names from HDF5; fall back to REGION_ORDER
    stored_region_names = attention_weights.get("region_names")
    if stored_region_names is not None:
        region_names_list = stored_region_names
    else:
        region_names_list = list(REGION_ORDER)
        if attention_weights:
            logger.warning(
                "No region_names in HDF5 — assuming REGION_ORDER. "
                "Re-run inference with latest code to store region names explicitly."
            )

    region_pseudobulk_dict = None
    if region_pseudobulk_raw is not None:
        region_pseudobulk_dict = {}
        for i, rname in enumerate(region_names_list):
            if i < region_pseudobulk_raw.shape[0]:
                region_pseudobulk_dict[rname] = region_pseudobulk_raw[i]
        logger.info(f"Loaded region pseudobulk for {len(region_pseudobulk_dict)} regions")

    # Unpack HGT for CCC analyzer
    edge_attention_scores = None
    hgt_edge_metadata = None
    edge_type_names = None
    if hgt_raw is not None and isinstance(hgt_raw, dict):
        edge_attention_scores, hgt_edge_metadata, edge_type_names = unpack_hgt_for_ccc(hgt_raw)

    # PMA attention is unpacked later for cell heterogeneity analysis (if enabled).

    # Load edge metadata for CCC analysis if available.
    # NOTE: edge_metadata.parquet is NOT produced by the current pipeline.
    # It can be generated externally from CellChatDB or LIANA+ output
    # to provide source/target cell type labels for edge types.
    # If absent, edge metadata is derived from HGT edge_type_names.
    edge_metadata = None
    edge_metadata_path = output_dir / "edge_metadata.parquet"
    if edge_metadata_path.exists():
        edge_metadata = pd.read_parquet(edge_metadata_path)
        logger.info(f"Loaded edge metadata with {len(edge_metadata)} edges")
        # Validate against current attention data to catch stale metadata
        if edge_attention_scores is not None:
            n_expected = edge_attention_scores.shape[-1]
            if len(edge_metadata) != n_expected:
                logger.warning(
                    f"Stale edge_metadata.parquet ({len(edge_metadata)} edges) "
                    f"doesn't match attention scores ({n_expected} edges). "
                    f"Discarding file-based metadata."
                )
                edge_metadata = None

    # Source region labels from metadata, aligned to subject order
    region_labels = None
    if metadata_df is not None and "region" in metadata_df.columns:
        if subject_ids is not None and "subject_id" in metadata_df.columns:
            meta_indexed = metadata_df.set_index("subject_id")
            aligned_regions = []
            for sid in subject_ids:
                if sid in meta_indexed.index:
                    aligned_regions.append(meta_indexed.loc[sid, "region"])
                else:
                    aligned_regions.append(None)
            # Replace None with empty string for safe array creation
            # (np.unique on mixed str/None raises TypeError)
            region_labels = np.array(
                [r if r is not None else "" for r in aligned_regions],
                dtype=str,
            )
            n_missing_regions = sum(1 for r in aligned_regions if r is None)
            if n_missing_regions > 0:
                logger.warning(
                    f"{n_missing_regions} subjects have no region label in metadata"
                )
            logger.info(f"Aligned region labels to {len(subject_ids)} subjects")
        else:
            region_labels = metadata_df["region"].values
            logger.warning("Region labels taken without alignment — metadata may not match subject order")

    # Build LR-pair mapping from LIANA results if provided
    lr_pair_mapping = None
    if args.liana_dir:
        from src.data.liana_processing import extract_lr_pairs_by_edge, aggregate_lr_mapping_across_subjects
        liana_dir = Path(args.liana_dir)
        subject_mappings = {}
        for liana_file in sorted(liana_dir.glob("*.parquet")):
            subject_id = liana_file.stem
            liana_df = pd.read_parquet(liana_file)
            subject_mappings[subject_id] = extract_lr_pairs_by_edge(
                liana_df, cell_types=cell_type_names
            )
        if subject_mappings:
            lr_pair_mapping = aggregate_lr_mapping_across_subjects(subject_mappings, min_subjects=2)
            logger.info(f"Built LR mapping from {len(subject_mappings)} subjects, {len(lr_pair_mapping)} edges")

    # Run analyses
    analyses_run = 0

    # Cell type importance
    if not args.skip_cell_type and pathology_attention is not None:
        # Get pathology levels for stratification (prefer gpath as primary)
        pathology_levels = None
        if predictions_df is not None:
            for col in ["gpath", "amylsqrt", "tangsqrt", "pathology"]:
                if col in predictions_df.columns:
                    pathology_levels = predictions_df[col].values
                    logger.info(f"  Using {col} for pathology stratification")
                    break

        run_cell_type_importance(
            pathology_attention=pathology_attention,
            cell_type_names=cell_type_names,
            pathology_levels=pathology_levels,
            region_labels=region_labels,
            output_dir=output_dir,
            formats=args.formats,
        )
        analyses_run += 1

    # Acquire cognition scores early (needed for both gene DE and resilience analysis)
    cognition_scores = None
    cognition_source = None

    if predictions_df is not None and "actual" in predictions_df.columns:
        cognition_scores = predictions_df["actual"].values
        cognition_source = "predictions_df['actual']"

    if cognition_scores is None and metadata_df is not None and subject_ids is not None:
        for cog_col in ["cogn_global", "cognition", "cognitive_score"]:
            aligned = _align_array_by_subject_id(metadata_df, subject_ids, cog_col)
            if aligned is not None:
                cognition_scores = aligned
                cognition_source = f"metadata_df['{cog_col}']"
                break

    if cognition_scores is not None:
        logger.info(f"  Using cognition from {cognition_source}")

    # Gene importance
    if not args.skip_gene and gene_gate is not None:
        # Extract per-subject pseudobulk from attention weights (if available)
        per_subject_pb = attention_weights.get("per_subject_pseudobulk")

        # Derive resilient/vulnerable group labels (if cognition + pathology available)
        gene_group_labels = None
        if cognition_scores is not None and per_subject_pb is not None:
            # Use the first available pathology measure for group derivation
            gene_pathology = None
            for col in ["gpath", "amylsqrt", "tangsqrt", "pathology"]:
                if predictions_df is not None and col in predictions_df.columns:
                    gene_pathology = predictions_df[col].values
                    break
                elif metadata_df is not None and col in metadata_df.columns:
                    if subject_ids is not None:
                        aligned = _align_array_by_subject_id(metadata_df, subject_ids, col)
                        if aligned is not None:
                            gene_pathology = aligned
                            break

            if gene_pathology is not None:
                gene_group_labels = derive_resilience_groups(
                    cognition_scores=cognition_scores,
                    pathology_scores=gene_pathology,
                )
                n_resilient = (gene_group_labels == "resilient").sum()
                n_vulnerable = (gene_group_labels == "vulnerable").sum()
                logger.info(
                    f"  Derived resilience groups for DE: "
                    f"{n_resilient} resilient, {n_vulnerable} vulnerable"
                )

        run_gene_importance(
            gene_gate_weights=gene_gate,
            cell_type_names=cell_type_names,
            gene_names=gene_names,
            region_pseudobulk=region_pseudobulk_dict,
            top_k=args.top_k_genes,
            output_dir=output_dir,
            formats=args.formats,
            group_labels=gene_group_labels,
            subject_expression=per_subject_pb,
            gate_threshold=args.gate_threshold,
            apply_fdr=not args.no_fdr_correction,
        )
        analyses_run += 1

    # CCC importance (analyzer handles None attention gracefully)
    if not args.skip_ccc and (edge_attention_scores is not None or hgt_edge_metadata is not None or edge_metadata is not None):
        # Merge HGT-derived edge_metadata with any file-based edge_metadata
        effective_edge_metadata = edge_metadata if edge_metadata is not None else hgt_edge_metadata
        run_ccc_importance(
            edge_attention_scores=edge_attention_scores,
            edge_metadata=effective_edge_metadata,
            cell_type_names=cell_type_names,
            edge_types=edge_type_names,
            lr_pair_mapping=lr_pair_mapping,
            region_labels=region_labels,
            subject_ids=subject_ids,
            output_dir=output_dir,
            formats=args.formats,
        )
        analyses_run += 1

    # Resilience signatures (requires pathology and cognition data)
    # Runs separately for each available pathology measure (gpath, amylsqrt, tangsqrt)
    # cognition_scores already derived above (shared with gene DE)
    if not args.skip_resilience and pathology_attention is not None:
        if cognition_scores is not None:

            # Acquire pathology scores
            pathology_columns = ["gpath", "amylsqrt", "tangsqrt"]
            available_pathology = []

            for col in pathology_columns:
                if predictions_df is not None and col in predictions_df.columns:
                    available_pathology.append((col, predictions_df[col].values))
                elif metadata_df is not None and col in metadata_df.columns:
                    if subject_ids is not None:
                        aligned = _align_array_by_subject_id(metadata_df, subject_ids, col)
                        if aligned is not None:
                            available_pathology.append((col, aligned))
                    else:
                        available_pathology.append((col, metadata_df[col].values))
                        logger.warning(f"Pathology '{col}' taken without alignment — no subject_ids")

            # Also check for legacy "pathology" column
            if not available_pathology:
                if metadata_df is not None and "pathology" in metadata_df.columns:
                    if subject_ids is not None:
                        aligned = _align_array_by_subject_id(metadata_df, subject_ids, "pathology")
                        if aligned is not None:
                            available_pathology.append(("pathology", aligned))
                    else:
                        available_pathology.append(("pathology", metadata_df["pathology"].values))
                elif predictions_df is not None and "pathology" in predictions_df.columns:
                    available_pathology.append(("pathology", predictions_df["pathology"].values))

            if available_pathology:
                # Prepare ablation embeddings (prefer fused, fall back to attended)
                ablation_embeddings = None
                if embeddings is not None:
                    if isinstance(embeddings, dict):
                        ablation_embeddings = embeddings.get("fused")
                        if ablation_embeddings is None:
                            candidate = embeddings.get("attended")
                            if candidate is not None and candidate.ndim == 3:
                                ablation_embeddings = candidate
                            elif candidate is not None:
                                logger.warning(
                                    f"Attended embeddings are {candidate.ndim}D, need 3D for ablation. Skipping."
                                )
                    elif isinstance(embeddings, np.ndarray):
                        ablation_embeddings = embeddings

                for pathology_name, pathology_scores in available_pathology:
                    logger.info(f"Running resilience signature for {pathology_name}...")
                    resilience_output_dir = output_dir / f"resilience_{pathology_name}"
                    run_resilience_signature(
                        attention=pathology_attention,
                        cognition_scores=cognition_scores,
                        pathology_scores=pathology_scores,
                        cell_type_names=cell_type_names,
                        n_permutations=args.n_permutations,
                        random_seed=args.seed,
                        apply_fdr_correction=not args.no_fdr,
                        fdr_threshold=args.fdr_threshold,
                        run_ablation=args.run_ablation,
                        ablation_method=args.ablation_method,
                        embeddings=ablation_embeddings,
                        region_labels=region_labels,
                        subject_ids=subject_ids,
                        output_dir=resilience_output_dir,
                        formats=args.formats,
                    )
                    analyses_run += 1
            else:
                logger.warning("Skipping resilience signature: no pathology columns found (gpath, amylsqrt, tangsqrt, or pathology)")
        else:
            logger.warning(
                "Skipping resilience signature: no cognition scores found "
                "(checked predictions_df['actual'], metadata cogn_global/cognition/cognitive_score)"
            )

    # Regional analysis
    if not args.skip_regional and region_weights is not None and (
        region_attention is not None or region_pseudobulk_dict
    ):
        run_regional_analysis(
            region_attention=region_attention,
            region_weights=region_weights,
            gene_gate_weights=gene_gate,
            region_pseudobulk=region_pseudobulk_dict,
            region_names=region_names_list,
            cell_type_names=cell_type_names,
            gene_names=gene_names,
            subject_ids=subject_ids,
            top_k_genes=args.top_k_genes,
            output_dir=output_dir,
            formats=args.formats,
        )
        analyses_run += 1

    # Uncertainty analysis
    if not args.skip_uncertainty and predictions_df is not None:
        if "predicted_std" in predictions_df.columns:
            actual = predictions_df.get("actual")
            actual = actual.values if actual is not None else None

            # Build covariates from predictions_df if available
            covariates = None
            cov_cols = [c for c in predictions_df.columns
                       if c not in ["subject_id", "predicted_mean", "predicted_std", "actual"]]
            if cov_cols:
                covariates = predictions_df[cov_cols].copy()

            # Merge metadata numeric columns (aligned by subject_id) for richer correlates
            if metadata_df is not None and subject_ids is not None:
                exclude_cols = {"subject_id", "region"}
                if covariates is not None:
                    exclude_cols.update(covariates.columns)

                numeric_cols = metadata_df.select_dtypes(include=[np.number]).columns
                meta_cov_cols = [c for c in numeric_cols if c not in exclude_cols]

                if meta_cov_cols:
                    for col in meta_cov_cols:
                        aligned = _align_array_by_subject_id(metadata_df, subject_ids, col)
                        if aligned is not None:
                            if covariates is None:
                                covariates = pd.DataFrame(index=range(len(subject_ids)))
                            covariates[col] = aligned
                    logger.info(f"  Added {len(meta_cov_cols)} metadata covariates: {meta_cov_cols}")

            # Add cell counts as covariates (per-cell-type counts + total)
            if cell_counts is not None and cell_counts.shape[0] == len(predictions_df):
                if covariates is None:
                    covariates = pd.DataFrame(index=range(len(predictions_df)))
                for ct_idx, ct_name in enumerate(cell_type_names):
                    covariates[f"cell_count_{ct_name}"] = cell_counts[:, ct_idx]
                covariates["total_cell_count"] = cell_counts.sum(axis=1)
                logger.info(f"  Added {len(cell_type_names) + 1} cell count covariates")

            run_uncertainty_analysis(
                predicted_mean=predictions_df["predicted_mean"].values,
                predicted_std=predictions_df["predicted_std"].values,
                actual=actual,
                subject_ids=predictions_df.get("subject_id", pd.Series(subject_ids)).tolist() if subject_ids else None,
                covariates=covariates,
                output_dir=output_dir,
                formats=args.formats,
            )
            analyses_run += 1
        else:
            logger.warning("Skipping uncertainty analysis: no predicted_std in predictions")

    # Cell heterogeneity analysis (PMA attention)
    if not args.skip_cell_heterogeneity:
        pma_raw = attention_weights.get("pma_attention")
        if pma_raw is not None and isinstance(pma_raw, dict):
            from src.utils.io import unpack_pma_attention
            pma_3d = unpack_pma_attention(pma_raw, cell_type_names=cell_type_names)
            if pma_3d is not None:
                from src.analysis.cell_heterogeneity import CellHeterogeneityAnalyzer

                cell_barcodes = attention_weights.get("cell_barcodes")
                het_output_dir = output_dir / "cell_heterogeneity"

                logger.info("Running cell heterogeneity analysis...")
                het_analyzer = CellHeterogeneityAnalyzer(
                    pma_attention=pma_3d,
                    cell_type_names=cell_type_names,
                    subject_ids=subject_ids,
                    cell_barcodes=cell_barcodes,
                    top_percentile=args.top_percentile,
                )
                het_result = het_analyzer.analyze()
                het_analyzer.save(het_result, het_output_dir, formats=args.formats)

                analyses_run += 1
                logger.info(f"Cell heterogeneity analysis saved to {het_output_dir}")
            else:
                logger.warning("Skipping cell heterogeneity: could not unpack PMA attention")
        else:
            logger.warning("Skipping cell heterogeneity: no pma_attention in HDF5")

    # Embedding analysis (requires subject embeddings)
    if not args.skip_embedding and embeddings is not None:
        # Build covariates for linear probe analysis
        covariates = None
        if predictions_df is not None:
            cov_cols = [c for c in predictions_df.columns
                       if c not in ["subject_id", "predicted_mean", "predicted_std"]]
            if cov_cols:
                covariates = predictions_df[cov_cols]

        if isinstance(embeddings, dict):
            # Multiple embedding types (pseudobulk, hgt, cell, fused, attended)
            for emb_name, emb_array in embeddings.items():
                if emb_array is None:
                    continue
                # Mean-pool 3D branch embeddings to 2D for analysis
                if emb_array.ndim == 3:
                    emb_2d = emb_array.mean(axis=1)
                elif emb_array.ndim == 2:
                    emb_2d = emb_array
                else:
                    logger.warning(f"Skipping embedding '{emb_name}' with {emb_array.ndim}D shape")
                    continue

                emb_output_dir = output_dir / f"embedding_{emb_name}"
                logger.info(f"  Running embedding analysis for '{emb_name}' "
                           f"(shape {emb_2d.shape}) → {emb_output_dir}")
                run_embedding_analysis(
                    embeddings=emb_2d,
                    subject_ids=subject_ids,
                    covariates=covariates,
                    output_dir=emb_output_dir,
                    formats=args.formats,
                )
                analyses_run += 1
        else:
            # Single embedding array (backward compat)
            run_embedding_analysis(
                embeddings=embeddings,
                subject_ids=subject_ids,
                covariates=covariates,
                output_dir=output_dir,
                formats=args.formats,
            )
            analyses_run += 1

    logger.info(f"Completed {analyses_run} analyses")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
