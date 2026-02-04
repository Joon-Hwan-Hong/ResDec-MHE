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
8. Save all results to analysis output directory

Outputs saved to: {experiment_dir}/analysis/ or --output-dir
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.utils.io import load_dataframe, load_attention_weights
from src.analysis import (
    CellTypeImportanceAnalyzer,
    GeneImportanceAnalyzer,
    CCCImportanceAnalyzer,
    ResilienceSignatureAnalyzer,
    RegionalAnalyzer,
    UncertaintyAnalyzer,
    compute_expected_calibration_error,
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

    # Parameters
    param_group = parser.add_argument_group("Parameters")
    param_group.add_argument(
        "--top-k-genes",
        type=int,
        default=50,
        help="Number of top genes per cell type (default: 50)",
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




def run_cell_type_importance(
    pathology_attention: np.ndarray,
    cell_type_names: list[str],
    pathology_levels: np.ndarray | None = None,
    region_names: list[str] | None = None,
    output_dir: Path = None,
    formats: list[str] = None,
) -> None:
    """Run cell type importance analysis."""
    logger.info("Running cell type importance analysis...")

    analyzer = CellTypeImportanceAnalyzer(
        attention=pathology_attention,
        cell_type_names=cell_type_names,
        pathology_scores=pathology_levels,
    )

    result = analyzer.analyze()

    if output_dir:
        analyzer.save(result, output_dir, formats=formats)
        logger.info(f"  Saved cell type importance to {output_dir}")


def run_gene_importance(
    gene_gate_weights: np.ndarray,
    cell_type_names: list[str],
    gene_names: list[str] | None = None,
    top_k: int = 50,
    output_dir: Path = None,
    formats: list[str] = None,
) -> None:
    """Run gene importance analysis."""
    logger.info("Running gene importance analysis...")

    analyzer = GeneImportanceAnalyzer(
        gene_gate_weights=gene_gate_weights,
        cell_type_names=cell_type_names,
        gene_names=gene_names,
    )

    result = analyzer.analyze(top_k=top_k)

    if output_dir:
        analyzer.save(result, output_dir, formats=formats)
        logger.info(f"  Saved gene importance to {output_dir}")


def run_ccc_importance(
    edge_attention_scores: np.ndarray,
    edge_metadata: pd.DataFrame | None = None,
    cell_type_names: list[str] | None = None,
    edge_types: list[str] | None = None,
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
    )

    result = analyzer.analyze(
        n_permutations=n_permutations,
        random_seed=random_seed,
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
    top_k_genes: int = 50,
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
    hgt_attention = attention_weights.get("hgt_attention")
    region_weights = attention_weights.get("region_weights")

    # Run analyses
    analyses_run = 0

    # Cell type importance
    if not args.skip_cell_type and pathology_attention is not None:
        run_cell_type_importance(
            pathology_attention=pathology_attention,
            cell_type_names=cell_type_names,
            output_dir=output_dir,
            formats=args.formats,
        )
        analyses_run += 1

    # Gene importance
    if not args.skip_gene and gene_gate is not None:
        run_gene_importance(
            gene_gate_weights=gene_gate,
            cell_type_names=cell_type_names,
            gene_names=gene_names,
            top_k=args.top_k_genes,
            output_dir=output_dir,
            formats=args.formats,
        )
        analyses_run += 1

    # CCC importance
    if not args.skip_ccc and hgt_attention is not None:
        run_ccc_importance(
            edge_attention_scores=hgt_attention,
            output_dir=output_dir,
            formats=args.formats,
        )
        analyses_run += 1

    # Resilience signatures (requires pathology and cognition data)
    if not args.skip_resilience and pathology_attention is not None:
        if predictions_df is not None and "actual" in predictions_df.columns:
            if metadata_df is not None and "pathology" in metadata_df.columns:
                run_resilience_signature(
                    attention=pathology_attention,
                    cognition_scores=predictions_df["actual"].values,
                    pathology_scores=metadata_df["pathology"].values,
                    cell_type_names=cell_type_names,
                    n_permutations=args.n_permutations,
                    output_dir=output_dir,
                    formats=args.formats,
                )
                analyses_run += 1
            else:
                logger.warning("Skipping resilience signature: no pathology levels in metadata")
        else:
            logger.warning("Skipping resilience signature: no actual cognition scores")

    # Regional analysis
    if not args.skip_regional and region_weights is not None:
        run_regional_analysis(
            region_weights=region_weights,
            gene_gate_weights=gene_gate,
            region_names=list(REGION_ORDER),
            cell_type_names=cell_type_names,
            gene_names=gene_names,
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
                covariates = predictions_df[cov_cols]

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

    logger.info(f"Completed {analyses_run} analyses")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
