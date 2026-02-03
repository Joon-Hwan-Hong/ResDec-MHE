"""
Generate publication-quality plots from analysis outputs.

Usage:
    uv run python scripts/generate_plots.py --experiment-dir experiments/20260113_143052_a3f7b2c1
    uv run python scripts/generate_plots.py --analysis-dir analysis_output --plots-dir plots
    uv run python scripts/generate_plots.py --experiment-dir experiments/20260113_143052_a3f7b2c1 --only attention importance

Workflow:
1. Load analysis results from experiment directory (or explicit paths)
2. Generate attention visualization plots
3. Generate importance visualization plots
4. Generate prediction quality plots
5. Save all plots to output directory at 600 DPI

Outputs saved to: data/plots/{experiment_hash}/ or --plots-dir
with nested subdirectories:
    attention/      - Cell type and gene attention heatmaps
    importance/     - Gene and CCC importance plots
    regional/       - Regional analysis plots
    prediction/     - Predicted vs actual, residuals
    uncertainty/    - Calibration, uncertainty correlates
"""

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from src.utils.io import load_dataframe
from src.visualization import (
    setup_seaborn_style,
    setup_matplotlib_defaults,
    FIGURE_DPI,
    # Attention plots
    plot_cell_type_attention_heatmap,
    plot_cell_type_importance_bar,
    plot_attention_distribution,
    plot_gene_gate_heatmap,
    plot_resilience_signature_heatmap,
    # Importance plots
    plot_top_genes_per_cell_type,
    plot_gene_importance_volcano,
    plot_ccc_network_summary,
    plot_top_interactions_heatmap,
    plot_regional_gene_importance,
    # Prediction plots
    plot_predicted_vs_actual,
    plot_calibration_curve,
    plot_residuals,
    plot_uncertainty_vs_error,
    plot_uncertainty_correlates,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


PLOT_CATEGORIES = ["attention", "importance", "prediction", "uncertainty", "regional"]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate publication-quality plots from analysis outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input sources
    input_group = parser.add_argument_group("Input Sources")
    input_group.add_argument(
        "--experiment-dir",
        type=str,
        help="Path to experiment directory containing analysis/ subdirectory",
    )
    input_group.add_argument(
        "--analysis-dir",
        type=str,
        help="Explicit path to analysis output directory",
    )
    input_group.add_argument(
        "--attention-path",
        type=str,
        help="Explicit path to attention weights HDF5 file",
    )

    # Output
    parser.add_argument(
        "--plots-dir",
        type=str,
        help="Output directory for plots (default: data/plots/{experiment_hash}/)",
    )

    # Plot selection
    parser.add_argument(
        "--only",
        nargs="+",
        choices=PLOT_CATEGORIES,
        help=f"Only generate specific plot categories: {PLOT_CATEGORIES}",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        choices=PLOT_CATEGORIES,
        help="Skip specific plot categories",
    )

    # Plot options
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output format (default: png)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=FIGURE_DPI,
        help=f"Output DPI (default: {FIGURE_DPI})",
    )
    parser.add_argument(
        "--top-k-genes",
        type=int,
        default=50,
        help="Number of top genes for gene heatmaps (default: 50)",
    )
    parser.add_argument(
        "--n-genes-per-cell-type",
        type=int,
        default=10,
        help="Number of genes per cell type in bar plots (default: 10)",
    )

    return parser.parse_args()


def load_attention_weights(path: Path) -> dict:
    """Load attention weights from HDF5 file."""
    weights = {}
    if not path.exists():
        return weights

    with h5py.File(path, "r") as f:
        for key in f.keys():
            weights[key] = f[key][:]

        # Load metadata from attributes
        for attr in ["cell_type_names", "gene_names", "subject_ids"]:
            if attr in f.attrs:
                weights[attr] = list(f.attrs[attr])

    return weights


def generate_attention_plots(
    analysis_dir: Path,
    attention_weights: dict,
    plots_dir: Path,
    fmt: str = "png",
    top_k_genes: int = 50,
) -> int:
    """Generate attention visualization plots."""
    logger.info("Generating attention plots...")
    count = 0

    # Create subdirectory for attention plots
    attention_dir = plots_dir / "attention"
    attention_dir.mkdir(parents=True, exist_ok=True)

    # Cell type importance bar chart
    importance_df = load_dataframe(analysis_dir / "cell_type_importance")
    if importance_df is not None:
        plot_cell_type_importance_bar(
            importance_df,
            title="Cell Type Importance Ranking",
            save_path=attention_dir / f"cell_type_importance_bar.{fmt}",
        )
        count += 1
        logger.info("  Created attention/cell_type_importance_bar")

    # Cell type attention heatmap (by pathology level)
    attention_by_pathology = load_dataframe(analysis_dir / "cell_type_importance_by_pathology")
    if attention_by_pathology is not None:
        plot_cell_type_attention_heatmap(
            attention_by_pathology,
            title="Cell Type Attention by Pathology Level",
            save_path=attention_dir / f"cell_type_attention_heatmap.{fmt}",
        )
        count += 1
        logger.info("  Created attention/cell_type_attention_heatmap")

    # Gene gate heatmap
    gene_gate = attention_weights.get("gene_gate")
    if gene_gate is not None:
        gene_names = attention_weights.get("gene_names")
        cell_type_names = attention_weights.get("cell_type_names")
        plot_gene_gate_heatmap(
            gene_gate,
            gene_names=gene_names,
            cell_type_names=cell_type_names,
            top_k_genes=top_k_genes,
            title="Gene Attention Weights",
            save_path=attention_dir / f"gene_gate_heatmap.{fmt}",
        )
        count += 1
        logger.info("  Created attention/gene_gate_heatmap")

    # Attention distribution (if pathology attention available)
    pathology_attention = attention_weights.get("pathology_attention")
    if pathology_attention is not None and pathology_attention.ndim >= 2:
        # Average over heads if present
        if pathology_attention.ndim == 3:
            attention_2d = pathology_attention.mean(axis=1)
        else:
            attention_2d = pathology_attention

        cell_type_names = attention_weights.get("cell_type_names")
        plot_attention_distribution(
            attention_2d,
            cell_type_names=cell_type_names,
            title="Attention Weight Distribution by Cell Type",
            save_path=attention_dir / f"attention_distribution.{fmt}",
        )
        count += 1
        logger.info("  Created attention/attention_distribution")

    # Resilience signature heatmap
    signature_df = load_dataframe(analysis_dir / "resilience_signature")
    if signature_df is not None:
        plot_resilience_signature_heatmap(
            signature_df,
            title="Resilience Signature",
            save_path=attention_dir / f"resilience_signature_heatmap.{fmt}",
        )
        count += 1
        logger.info("  Created attention/resilience_signature_heatmap")

    return count


def generate_importance_plots(
    analysis_dir: Path,
    plots_dir: Path,
    fmt: str = "png",
    n_genes: int = 10,
) -> int:
    """Generate gene and CCC importance plots."""
    logger.info("Generating importance plots...")
    count = 0

    # Create subdirectory for importance plots
    importance_dir = plots_dir / "importance"
    importance_dir.mkdir(parents=True, exist_ok=True)

    # Top genes per cell type
    top_genes_df = load_dataframe(analysis_dir / "gene_importance_top_genes")
    if top_genes_df is not None:
        plot_top_genes_per_cell_type(
            top_genes_df,
            n_genes=n_genes,
            title="Top Genes per Cell Type",
            save_path=importance_dir / f"top_genes_per_cell_type.{fmt}",
        )
        count += 1
        logger.info("  Created importance/top_genes_per_cell_type")

    # Gene importance by cell type (full ranking)
    gene_importance_df = load_dataframe(analysis_dir / "gene_importance_by_celltype")
    if gene_importance_df is not None:
        # Create volcano plot for first cell type as example
        cell_types = gene_importance_df["cell_type"].unique()
        if len(cell_types) > 0:
            for ct in cell_types[:3]:  # Top 3 cell types
                ct_df = gene_importance_df[gene_importance_df["cell_type"] == ct].copy()
                if len(ct_df) > 0:
                    plot_gene_importance_volcano(
                        ct_df,
                        cell_type=ct,
                        save_path=importance_dir / f"gene_importance_volcano_{ct.replace(' ', '_')}.{fmt}",
                    )
                    count += 1
            logger.info(f"  Created {min(3, len(cell_types))} importance/gene_importance_volcano plots")

    # CCC network summary
    network_df = load_dataframe(analysis_dir / "ccc_by_category")
    if network_df is not None:
        plot_ccc_network_summary(
            network_df,
            title="Cell-Cell Communication by Category",
            save_path=importance_dir / f"ccc_network_summary.{fmt}",
        )
        count += 1
        logger.info("  Created importance/ccc_network_summary")

    # Top interactions
    interactions_df = load_dataframe(analysis_dir / "ccc_top_interactions")
    if interactions_df is not None:
        plot_top_interactions_heatmap(
            interactions_df,
            top_k=20,
            title="Top Cell-Cell Interactions",
            save_path=importance_dir / f"top_interactions.{fmt}",
        )
        count += 1
        logger.info("  Created importance/top_interactions")

    return count


def generate_regional_plots(
    analysis_dir: Path,
    plots_dir: Path,
    fmt: str = "png",
    n_genes: int = 10,
) -> int:
    """Generate regional analysis plots."""
    logger.info("Generating regional plots...")
    count = 0

    # Create subdirectory for regional plots
    regional_dir = plots_dir / "regional"
    regional_dir.mkdir(parents=True, exist_ok=True)

    # Regional gene importance
    regional_df = load_dataframe(analysis_dir / "regional_gene_importance")
    if regional_df is not None:
        plot_regional_gene_importance(
            regional_df,
            n_genes=n_genes,
            title="Top Genes by Region",
            save_path=regional_dir / f"regional_gene_importance.{fmt}",
        )
        count += 1
        logger.info("  Created regional/regional_gene_importance")

    return count


def generate_prediction_plots(
    analysis_dir: Path,
    plots_dir: Path,
    fmt: str = "png",
) -> int:
    """Generate prediction quality plots."""
    logger.info("Generating prediction plots...")
    count = 0

    # Create subdirectory for prediction plots
    prediction_dir = plots_dir / "prediction"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    # Load predictions
    predictions_df = load_dataframe(analysis_dir / "predictions")
    if predictions_df is None:
        predictions_df = load_dataframe(analysis_dir / "prediction_uncertainty")

    if predictions_df is not None and "predicted_mean" in predictions_df.columns:
        predicted_mean = predictions_df["predicted_mean"].values
        actual = predictions_df.get("actual")
        actual = actual.values if actual is not None else None
        predicted_std = predictions_df.get("predicted_std")
        predicted_std = predicted_std.values if predicted_std is not None else None

        # Predicted vs actual
        if actual is not None:
            plot_predicted_vs_actual(
                predicted_mean=predicted_mean,
                actual=actual,
                predicted_std=predicted_std,
                title="Predicted vs Actual Cognition",
                save_path=prediction_dir / f"predicted_vs_actual.{fmt}",
            )
            count += 1
            logger.info("  Created prediction/predicted_vs_actual")

            # Residuals
            plot_residuals(
                predicted_mean=predicted_mean,
                actual=actual,
                predicted_std=predicted_std,
                title="Residual Analysis",
                save_path=prediction_dir / f"residuals.{fmt}",
            )
            count += 1
            logger.info("  Created prediction/residuals")

    return count


def generate_uncertainty_plots(
    analysis_dir: Path,
    plots_dir: Path,
    fmt: str = "png",
) -> int:
    """Generate uncertainty analysis plots."""
    logger.info("Generating uncertainty plots...")
    count = 0

    # Create subdirectory for uncertainty plots
    uncertainty_dir = plots_dir / "uncertainty"
    uncertainty_dir.mkdir(parents=True, exist_ok=True)

    # Load predictions for uncertainty vs error
    predictions_df = load_dataframe(analysis_dir / "predictions")
    if predictions_df is None:
        predictions_df = load_dataframe(analysis_dir / "prediction_uncertainty")

    if predictions_df is not None:
        if all(col in predictions_df.columns for col in ["predicted_mean", "predicted_std", "actual"]):
            plot_uncertainty_vs_error(
                predicted_mean=predictions_df["predicted_mean"].values,
                actual=predictions_df["actual"].values,
                predicted_std=predictions_df["predicted_std"].values,
                title="Uncertainty vs Prediction Error",
                save_path=uncertainty_dir / f"uncertainty_vs_error.{fmt}",
            )
            count += 1
            logger.info("  Created uncertainty/uncertainty_vs_error")

    # Calibration curve
    calibration_df = load_dataframe(analysis_dir / "calibration_summary")
    if calibration_df is not None:
        plot_calibration_curve(
            calibration_df,
            title="Uncertainty Calibration",
            save_path=uncertainty_dir / f"calibration_curve.{fmt}",
        )
        count += 1
        logger.info("  Created uncertainty/calibration_curve")

    # Uncertainty correlates
    correlates_df = load_dataframe(analysis_dir / "uncertainty_correlates")
    if correlates_df is not None and len(correlates_df) > 0:
        plot_uncertainty_correlates(
            correlates_df,
            title="Uncertainty Correlates",
            save_path=uncertainty_dir / f"uncertainty_correlates.{fmt}",
        )
        count += 1
        logger.info("  Created uncertainty/uncertainty_correlates")

    return count


def main():
    """Main entry point."""
    args = parse_args()

    # Setup plotting style
    setup_seaborn_style()
    setup_matplotlib_defaults()

    # Resolve paths
    if args.experiment_dir:
        exp_dir = Path(args.experiment_dir)
        exp_hash = exp_dir.name
        analysis_dir = Path(args.analysis_dir) if args.analysis_dir else exp_dir / "analysis"
        plots_dir = Path(args.plots_dir) if args.plots_dir else Path("data/plots") / exp_hash
        attention_path = Path(args.attention_path) if args.attention_path else analysis_dir / "attention_weights.h5"
    else:
        if not args.analysis_dir:
            raise ValueError("Must provide either --experiment-dir or --analysis-dir")
        analysis_dir = Path(args.analysis_dir)
        plots_dir = Path(args.plots_dir) if args.plots_dir else Path("plots")
        attention_path = Path(args.attention_path) if args.attention_path else analysis_dir / "attention_weights.h5"

    plots_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Plots output directory: {plots_dir}")

    # Determine which plot categories to generate
    categories = set(PLOT_CATEGORIES)
    if args.only:
        categories = set(args.only)
    if args.skip:
        categories -= set(args.skip)

    logger.info(f"Generating plot categories: {categories}")

    # Load attention weights
    attention_weights = load_attention_weights(attention_path)
    if attention_weights:
        logger.info(f"Loaded attention weights: {list(attention_weights.keys())}")

    # Generate plots
    total_plots = 0

    if "attention" in categories:
        total_plots += generate_attention_plots(
            analysis_dir=analysis_dir,
            attention_weights=attention_weights,
            plots_dir=plots_dir,
            fmt=args.format,
            top_k_genes=args.top_k_genes,
        )

    if "importance" in categories:
        total_plots += generate_importance_plots(
            analysis_dir=analysis_dir,
            plots_dir=plots_dir,
            fmt=args.format,
            n_genes=args.n_genes_per_cell_type,
        )

    if "regional" in categories:
        total_plots += generate_regional_plots(
            analysis_dir=analysis_dir,
            plots_dir=plots_dir,
            fmt=args.format,
            n_genes=args.n_genes_per_cell_type,
        )

    if "prediction" in categories:
        total_plots += generate_prediction_plots(
            analysis_dir=analysis_dir,
            plots_dir=plots_dir,
            fmt=args.format,
        )

    if "uncertainty" in categories:
        total_plots += generate_uncertainty_plots(
            analysis_dir=analysis_dir,
            plots_dir=plots_dir,
            fmt=args.format,
        )

    logger.info(f"Generated {total_plots} plots")
    logger.info(f"Plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
