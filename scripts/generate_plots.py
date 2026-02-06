"""
Generate publication-quality plots from analysis results.

Usage:
    uv run python scripts/generate_plots.py --experiment-dir experiments/20260113_143052_a3f7b2c1
    uv run python scripts/generate_plots.py --experiment-dir experiments/20260113 --plot-types attention resilience
    uv run python scripts/generate_plots.py --analysis-dir analysis_output --output-dir figures/

Workflow:
1. Load analysis results from experiment (or explicit paths)
2. Generate publication-quality plots using src.visualization
3. Save plots to figures/ subdirectory

Outputs saved to: {experiment_dir}/figures/ or --output-dir
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import load_dataframe, load_attention_weights
from src.visualization import (
    # Config
    setup_seaborn_style,
    setup_matplotlib_defaults,
    save_figure,
    FIGURE_FORMAT,
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
    # Embedding plots
    plot_umap_scatter,
    plot_cluster_composition,
    plot_linear_probe_results,
    plot_embedding_summary,
    # Training curves
    plot_training_summary,
)
from src.data.constants import CELL_TYPE_ORDER

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Available plot categories
PLOT_TYPES = {
    "attention": [
        "cell_type_attention_heatmap",
        "cell_type_importance_bar",
        "attention_distribution",
        "gene_gate_heatmap",
    ],
    "resilience": [
        "resilience_signature_heatmap",
    ],
    "importance": [
        "top_genes_per_cell_type",
        "differential_expression_volcano",
        "ccc_network_summary",
        "top_interactions_heatmap",
        "regional_gene_importance",
    ],
    "prediction": [
        "predicted_vs_actual",
        "calibration_curve",
        "residuals",
        "uncertainty_vs_error",
        "uncertainty_correlates",
    ],
    "embedding": [
        "umap_scatter",
        "cluster_composition",
        "linear_probe_results",
        "embedding_summary",
    ],
    "training": [
        "loss_curves",
        "learning_rate",
    ],
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate publication-quality plots from analysis results",
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
        help="Explicit path to analysis results directory",
    )
    input_group.add_argument(
        "--attention-path",
        type=str,
        help="Explicit path to attention weights HDF5 file",
    )
    input_group.add_argument(
        "--predictions-path",
        type=str,
        help="Explicit path to predictions parquet file",
    )
    input_group.add_argument(
        "--training-log-dir",
        type=str,
        help="Path to training logs directory (for training curve plots)",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for figures (default: {experiment-dir}/figures/)",
    )

    # Plot selection
    plot_group = parser.add_argument_group("Plot Selection")
    plot_group.add_argument(
        "--plot-types",
        nargs="+",
        choices=list(PLOT_TYPES.keys()) + ["all"],
        default=["all"],
        help="Categories of plots to generate (default: all)",
    )
    plot_group.add_argument(
        "--skip-plots",
        nargs="+",
        default=[],
        help="Specific plots to skip (by name)",
    )

    # Output format
    format_group = parser.add_argument_group("Output Format")
    format_group.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output figure format (default: png)",
    )
    format_group.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="Output DPI for raster formats (default: 600)",
    )

    return parser.parse_args()


def load_analysis_data(analysis_dir: Path) -> dict:
    """
    Load analysis results from directory.

    Args:
        analysis_dir: Path to analysis results directory

    Returns:
        Dictionary of loaded DataFrames and arrays
    """
    data = {}

    # Cell type importance
    cell_type_path = analysis_dir / "cell_type_importance.parquet"
    if cell_type_path.exists():
        data["cell_type_importance"] = load_dataframe(cell_type_path)
        logger.info(f"  Loaded cell_type_importance: {len(data['cell_type_importance'])} rows")

    # Gene importance (use top_genes_per_celltype which has rank column for plotting)
    gene_path = analysis_dir / "top_genes_per_celltype.parquet"
    if gene_path.exists():
        data["gene_importance"] = load_dataframe(gene_path)
        logger.info(f"  Loaded gene_importance: {len(data['gene_importance'])} rows")
    else:
        # Fallback to gene_importance_by_celltype (may lack rank column)
        alt_path = analysis_dir / "gene_importance_by_celltype.parquet"
        if alt_path.exists():
            data["gene_importance"] = load_dataframe(alt_path)
            logger.info(f"  Loaded gene_importance (alt): {len(data['gene_importance'])} rows")

    # Differential expression analysis
    diff_expr_path = analysis_dir / "differential_expression.parquet"
    if diff_expr_path.exists():
        data["differential_expression"] = pd.read_parquet(diff_expr_path)
        logger.info(f"  Loaded differential_expression: {len(data['differential_expression'])} rows")

    # Cell type importance by pathology (for heatmap)
    pathology_ct_path = analysis_dir / "cell_type_importance_by_pathology.parquet"
    if pathology_ct_path.exists():
        data["cell_type_importance_by_pathology"] = load_dataframe(pathology_ct_path)
        logger.info(f"  Loaded cell_type_importance_by_pathology: {len(data['cell_type_importance_by_pathology'])} rows")

    # CCC importance
    ccc_path = analysis_dir / "ccc_importance.parquet"
    if ccc_path.exists():
        data["ccc_importance"] = load_dataframe(ccc_path)
        logger.info(f"  Loaded ccc_importance: {len(data['ccc_importance'])} rows")

    # CCC network summary (aggregated by edge type)
    ccc_summary_path = analysis_dir / "ccc_network_summary.parquet"
    if ccc_summary_path.exists():
        data["ccc_network_summary"] = load_dataframe(ccc_summary_path)
        logger.info(f"  Loaded ccc_network_summary: {len(data['ccc_network_summary'])} rows")

    # Top interactions (ranked by attention)
    top_interactions_path = analysis_dir / "top_interactions.parquet"
    if top_interactions_path.exists():
        data["top_interactions"] = load_dataframe(top_interactions_path)
        logger.info(f"  Loaded top_interactions: {len(data['top_interactions'])} rows")

    # Resilience signature — scan subdirectories (resilience_gpath/, resilience_amylsqrt/, etc.)
    resilience_data = {}
    for subdir in sorted(analysis_dir.glob("resilience_*/")):
        sig_path = subdir / "resilience_signature.parquet"
        if sig_path.exists():
            pathology_name = subdir.name.replace("resilience_", "")
            resilience_data[pathology_name] = load_dataframe(sig_path)
            logger.info(f"  Loaded resilience_signature ({pathology_name}): {len(resilience_data[pathology_name])} rows")

    # Also check root for backward compat
    root_resilience = analysis_dir / "resilience_signature.parquet"
    if root_resilience.exists() and not resilience_data:
        resilience_data["combined"] = load_dataframe(root_resilience)
        logger.info(f"  Loaded resilience_signature (root): {len(resilience_data['combined'])} rows")

    if resilience_data:
        data["resilience_signatures"] = resilience_data
        # Also store first one as "resilience_signature" for backward compat
        first_key = next(iter(resilience_data))
        data["resilience_signature"] = resilience_data[first_key]

    # Regional analysis
    regional_path = analysis_dir / "regional_gene_importance.parquet"
    if regional_path.exists():
        data["regional_gene_importance"] = load_dataframe(regional_path)
        logger.info(f"  Loaded regional_gene_importance: {len(data['regional_gene_importance'])} rows")

    # Predictions
    predictions_path = analysis_dir / "predictions.parquet"
    if predictions_path.exists():
        data["predictions"] = load_dataframe(predictions_path)
        logger.info(f"  Loaded predictions: {len(data['predictions'])} rows")

    # Uncertainty analysis
    uncertainty_path = analysis_dir / "prediction_uncertainty.parquet"
    if uncertainty_path.exists():
        data["uncertainty"] = load_dataframe(uncertainty_path)
        logger.info(f"  Loaded uncertainty: {len(data['uncertainty'])} rows")

    # Uncertainty correlates (for correlation bar chart)
    unc_correlates_path = analysis_dir / "uncertainty_correlates.parquet"
    if unc_correlates_path.exists():
        data["uncertainty_correlates"] = load_dataframe(unc_correlates_path)
        logger.info(f"  Loaded uncertainty_correlates: {len(data['uncertainty_correlates'])} rows")

    # Calibration
    calibration_path = analysis_dir / "calibration_summary.parquet"
    if calibration_path.exists():
        data["calibration"] = load_dataframe(calibration_path)
        logger.info(f"  Loaded calibration: {len(data['calibration'])} rows")

    # Embedding analysis — check for embedding_* subdirectories (multi-embedding)
    # or top-level files (legacy single-embedding)
    emb_dirs = sorted(analysis_dir.glob("embedding_*"))
    if emb_dirs:
        data["embedding_dirs"] = {}
        for emb_dir in emb_dirs:
            if not emb_dir.is_dir():
                continue
            emb_name = emb_dir.name.replace("embedding_", "")
            emb_data = {}
            for fname in ("umap_projection", "cluster_assignments", "linear_probe_results"):
                fpath = emb_dir / f"{fname}.parquet"
                if fpath.exists():
                    emb_data[fname] = load_dataframe(fpath)
            if emb_data:
                data["embedding_dirs"][emb_name] = emb_data
                logger.info(f"  Loaded embedding '{emb_name}': {list(emb_data.keys())}")
    else:
        # Legacy: top-level embedding files
        umap_path = analysis_dir / "umap_projection.parquet"
        if umap_path.exists():
            data["umap_projection"] = load_dataframe(umap_path)
            logger.info(f"  Loaded umap_projection: {len(data['umap_projection'])} rows")

        cluster_path = analysis_dir / "cluster_assignments.parquet"
        if cluster_path.exists():
            data["cluster_assignments"] = load_dataframe(cluster_path)
            logger.info(f"  Loaded cluster_assignments: {len(data['cluster_assignments'])} rows")

        probe_path = analysis_dir / "linear_probe_results.parquet"
        if probe_path.exists():
            data["linear_probe_results"] = load_dataframe(probe_path)
            logger.info(f"  Loaded linear_probe_results: {len(data['linear_probe_results'])} rows")

    return data


def generate_attention_plots(
    data: dict,
    attention: dict,
    output_dir: Path,
    skip_plots: list[str],
    fmt: str = "png",
    dpi: int = 600,
    cell_type_names: list[str] | None = None,
    gene_names: list[str] | None = None,
) -> list[str]:
    """
    Generate attention-related plots.

    Returns:
        List of generated plot paths
    """
    generated = []

    # Cell type attention heatmap
    if "cell_type_attention_heatmap" not in skip_plots:
        heatmap_df = None
        if "cell_type_importance" in data:
            df = data["cell_type_importance"]
            if "pathology_tertile" in df.columns:
                heatmap_df = df
        if heatmap_df is None and "cell_type_importance_by_pathology" in data:
            heatmap_df = data["cell_type_importance_by_pathology"]
        if heatmap_df is not None:
            try:
                fig = plot_cell_type_attention_heatmap(heatmap_df)
                path = output_dir / f"cell_type_attention_heatmap.{fmt}"
                save_figure(fig, str(path), format=fmt, dpi=dpi)
                generated.append(str(path))
                logger.info(f"  Generated: {path.name}")
            except Exception as e:
                logger.warning(f"  Failed cell_type_attention_heatmap: {e}")

    # Cell type importance bar chart
    if "cell_type_importance_bar" not in skip_plots:
        if "cell_type_importance" in data:
            df = data["cell_type_importance"]
            try:
                fig = plot_cell_type_importance_bar(df)
                path = output_dir / f"cell_type_importance_bar.{fmt}"
                save_figure(fig, str(path), format=fmt, dpi=dpi)
                generated.append(str(path))
                logger.info(f"  Generated: {path.name}")
            except Exception as e:
                logger.warning(f"  Failed cell_type_importance_bar: {e}")

    # Attention distribution
    if "attention_distribution" not in skip_plots:
        if "pathology_attention" in attention:
            try:
                # Average over heads: [n_subjects, n_heads, n_cell_types] -> [n_subjects, n_cell_types]
                patho_attn = attention["pathology_attention"]
                if patho_attn.ndim == 3:
                    patho_attn = patho_attn.mean(axis=1)
                fig = plot_attention_distribution(
                    patho_attn,
                    cell_type_names=cell_type_names,
                )
                path = output_dir / f"attention_distribution.{fmt}"
                save_figure(fig, str(path), format=fmt, dpi=dpi)
                generated.append(str(path))
                logger.info(f"  Generated: {path.name}")
            except Exception as e:
                logger.warning(f"  Failed attention_distribution: {e}")

    # Gene gate heatmap
    if "gene_gate_heatmap" not in skip_plots:
        if "gene_gate" in attention:
            try:
                fig = plot_gene_gate_heatmap(
                    attention["gene_gate"],
                    gene_names=gene_names,
                    cell_type_names=cell_type_names,
                )
                path = output_dir / f"gene_gate_heatmap.{fmt}"
                save_figure(fig, str(path), format=fmt, dpi=dpi)
                generated.append(str(path))
                logger.info(f"  Generated: {path.name}")
            except Exception as e:
                logger.warning(f"  Failed gene_gate_heatmap: {e}")

    return generated


def generate_resilience_plots(
    data: dict,
    output_dir: Path,
    skip_plots: list[str],
    fmt: str = "png",
    dpi: int = 600,
) -> list[str]:
    """Generate resilience signature plots."""
    generated = []

    if "resilience_signature_heatmap" not in skip_plots:
        resilience_data = data.get("resilience_signatures", {})
        # Fallback to single resilience_signature
        if not resilience_data and "resilience_signature" in data:
            resilience_data = {"combined": data["resilience_signature"]}

        for pathology_name, df in resilience_data.items():
            try:
                fig = plot_resilience_signature_heatmap(df)
                suffix = f"_{pathology_name}" if pathology_name != "combined" else ""
                path = output_dir / f"resilience_signature_heatmap{suffix}.{fmt}"
                save_figure(fig, str(path), format=fmt, dpi=dpi)
                generated.append(str(path))
                logger.info(f"  Generated: {path.name}")
            except Exception as e:
                logger.warning(f"  Failed resilience_signature_heatmap ({pathology_name}): {e}")

    return generated


def generate_importance_plots(
    data: dict,
    output_dir: Path,
    skip_plots: list[str],
    fmt: str = "png",
    dpi: int = 600,
) -> list[str]:
    """Generate gene/CCC importance plots."""
    generated = []

    # Top genes per cell type
    if "top_genes_per_cell_type" not in skip_plots:
        if "gene_importance" in data:
            df = data["gene_importance"]
            try:
                fig = plot_top_genes_per_cell_type(df, n_genes=10)
                path = output_dir / f"top_genes_per_cell_type.{fmt}"
                save_figure(fig, str(path), format=fmt, dpi=dpi)
                generated.append(str(path))
                logger.info(f"  Generated: {path.name}")
            except Exception as e:
                logger.warning(f"  Failed top_genes_per_cell_type: {e}")

    # Differential expression analysis (volcano-style)
    if "differential_expression_volcano" not in skip_plots:
        df = data.get("differential_expression")
        if df is None:
            gi = data.get("gene_importance")
            if gi is not None and "log2_fold_change" in gi.columns and "pvalue" in gi.columns:
                df = gi
        if df is not None and "log2_fold_change" in df.columns and "pvalue" in df.columns:
            try:
                fig = plot_gene_importance_volcano(df)
                path = output_dir / f"differential_expression_volcano.{fmt}"
                save_figure(fig, str(path), format=fmt, dpi=dpi)
                generated.append(str(path))
                logger.info(f"  Generated: {path.name}")
            except Exception as e:
                logger.warning(f"  Failed differential_expression_volcano: {e}")

    # CCC network summary (prefer aggregated summary, fall back to raw importance)
    if "ccc_network_summary" not in skip_plots:
        df = data.get("ccc_network_summary")
        if df is None:
            df = data.get("ccc_importance")
        if df is not None:
            try:
                fig = plot_ccc_network_summary(df)
                path = output_dir / f"ccc_network_summary.{fmt}"
                save_figure(fig, str(path), format=fmt, dpi=dpi)
                generated.append(str(path))
                logger.info(f"  Generated: {path.name}")
            except Exception as e:
                logger.warning(f"  Failed ccc_network_summary: {e}")

    # Top interactions heatmap (prefer ranked top_interactions, fall back to raw importance)
    if "top_interactions_heatmap" not in skip_plots:
        df = data.get("top_interactions")
        if df is None:
            df = data.get("ccc_importance")
        if df is not None:
            try:
                fig = plot_top_interactions_heatmap(df, top_k=20)
                path = output_dir / f"top_interactions_heatmap.{fmt}"
                save_figure(fig, str(path), format=fmt, dpi=dpi)
                generated.append(str(path))
                logger.info(f"  Generated: {path.name}")
            except Exception as e:
                logger.warning(f"  Failed top_interactions_heatmap: {e}")

    # Regional gene importance
    if "regional_gene_importance" not in skip_plots:
        if "regional_gene_importance" in data:
            df = data["regional_gene_importance"]
            try:
                fig = plot_regional_gene_importance(df)
                path = output_dir / f"regional_gene_importance.{fmt}"
                save_figure(fig, str(path), format=fmt, dpi=dpi)
                generated.append(str(path))
                logger.info(f"  Generated: {path.name}")
            except Exception as e:
                logger.warning(f"  Failed regional_gene_importance: {e}")

    return generated


def generate_prediction_plots(
    data: dict,
    output_dir: Path,
    skip_plots: list[str],
    fmt: str = "png",
    dpi: int = 600,
) -> list[str]:
    """Generate prediction and uncertainty plots."""
    generated = []

    predictions = data.get("predictions")
    if predictions is None:
        return generated

    # Predicted vs actual
    if "predicted_vs_actual" not in skip_plots:
        if "predicted_mean" in predictions.columns and "actual" in predictions.columns:
            try:
                fig = plot_predicted_vs_actual(
                    predictions["predicted_mean"].values,
                    predictions["actual"].values,
                )
                path = output_dir / f"predicted_vs_actual.{fmt}"
                save_figure(fig, str(path), format=fmt, dpi=dpi)
                generated.append(str(path))
                logger.info(f"  Generated: {path.name}")
            except Exception as e:
                logger.warning(f"  Failed predicted_vs_actual: {e}")

    # Calibration curve
    if "calibration_curve" not in skip_plots:
        if "calibration" in data:
            cal_df = data["calibration"]
            try:
                fig = plot_calibration_curve(cal_df)
                path = output_dir / f"calibration_curve.{fmt}"
                save_figure(fig, str(path), format=fmt, dpi=dpi)
                generated.append(str(path))
                logger.info(f"  Generated: {path.name}")
            except Exception as e:
                logger.warning(f"  Failed calibration_curve: {e}")

    # Residuals
    if "residuals" not in skip_plots:
        if "predicted_mean" in predictions.columns and "actual" in predictions.columns:
            try:
                fig = plot_residuals(
                    predictions["predicted_mean"].values,
                    predictions["actual"].values,
                )
                path = output_dir / f"residuals.{fmt}"
                save_figure(fig, str(path), format=fmt, dpi=dpi)
                generated.append(str(path))
                logger.info(f"  Generated: {path.name}")
            except Exception as e:
                logger.warning(f"  Failed residuals: {e}")

    # Uncertainty vs error
    if "uncertainty_vs_error" not in skip_plots:
        if all(col in predictions.columns for col in ["predicted_mean", "actual", "predicted_std"]):
            try:
                fig = plot_uncertainty_vs_error(
                    predictions["predicted_mean"].values,
                    predictions["actual"].values,
                    predictions["predicted_std"].values,
                )
                path = output_dir / f"uncertainty_vs_error.{fmt}"
                save_figure(fig, str(path), format=fmt, dpi=dpi)
                generated.append(str(path))
                logger.info(f"  Generated: {path.name}")
            except Exception as e:
                logger.warning(f"  Failed uncertainty_vs_error: {e}")

    # Uncertainty correlates
    if "uncertainty_correlates" not in skip_plots:
        if "uncertainty_correlates" in data:
            unc_df = data["uncertainty_correlates"]
            try:
                fig = plot_uncertainty_correlates(unc_df)
                path = output_dir / f"uncertainty_correlates.{fmt}"
                save_figure(fig, str(path), format=fmt, dpi=dpi)
                generated.append(str(path))
                logger.info(f"  Generated: {path.name}")
            except Exception as e:
                logger.warning(f"  Failed uncertainty_correlates: {e}")

    return generated


def generate_embedding_plots(
    data: dict,
    output_dir: Path,
    skip_plots: list[str],
    fmt: str = "png",
    dpi: int = 600,
) -> list[str]:
    """Generate embedding analysis plots."""
    generated = []

    umap_df = data.get("umap_projection")
    cluster_df = data.get("cluster_assignments")
    probe_df = data.get("linear_probe_results")

    if umap_df is None:
        return generated

    # UMAP scatter
    if "umap_scatter" not in skip_plots:
        try:
            # Try to color by cluster if available
            color_by = None
            if cluster_df is not None and "cluster" in cluster_df.columns:
                # Merge cluster info into umap_df
                if "subject_id" in umap_df.columns and "subject_id" in cluster_df.columns:
                    umap_with_cluster = umap_df.merge(
                        cluster_df[["subject_id", "cluster"]],
                        on="subject_id",
                        how="left"
                    )
                    color_by = "cluster"
                else:
                    umap_with_cluster = umap_df
            else:
                umap_with_cluster = umap_df

            fig = plot_umap_scatter(umap_with_cluster, color_by=color_by)
            path = output_dir / f"umap_scatter.{fmt}"
            save_figure(fig, str(path), format=fmt, dpi=dpi)
            generated.append(str(path))
            logger.info(f"  Generated: {path.name}")
        except Exception as e:
            logger.warning(f"  Failed umap_scatter: {e}")

    # Cluster composition
    if "cluster_composition" not in skip_plots:
        if cluster_df is not None and "cluster" in cluster_df.columns:
            try:
                fig = plot_cluster_composition(cluster_df)
                if fig is not None:
                    path = output_dir / f"cluster_composition.{fmt}"
                    save_figure(fig, str(path), format=fmt, dpi=dpi)
                    generated.append(str(path))
                    logger.info(f"  Generated: {path.name}")
            except Exception as e:
                logger.warning(f"  Failed cluster_composition: {e}")

    # Linear probe results
    if "linear_probe_results" not in skip_plots:
        if probe_df is not None and "r2_score" in probe_df.columns:
            try:
                fig = plot_linear_probe_results(probe_df)
                if fig is not None:
                    path = output_dir / f"linear_probe_results.{fmt}"
                    save_figure(fig, str(path), format=fmt, dpi=dpi)
                    generated.append(str(path))
                    logger.info(f"  Generated: {path.name}")
            except Exception as e:
                logger.warning(f"  Failed linear_probe_results: {e}")

    # Embedding summary (multi-panel)
    if "embedding_summary" not in skip_plots:
        try:
            fig = plot_embedding_summary(
                umap_df=umap_df,
                cluster_df=cluster_df,
                probe_df=probe_df,
            )
            path = output_dir / f"embedding_summary.{fmt}"
            save_figure(fig, str(path), format=fmt, dpi=dpi)
            generated.append(str(path))
            logger.info(f"  Generated: {path.name}")
        except Exception as e:
            logger.warning(f"  Failed embedding_summary: {e}")

    return generated


def generate_training_plots(
    log_dir: Path,
    output_dir: Path,
    skip_plots: list[str],
    fmt: str = "png",
    dpi: int = 600,
) -> list[str]:
    """Generate training curve plots."""
    generated = []

    if not log_dir or not log_dir.exists():
        logger.info("  No training log directory provided, skipping training plots")
        return generated

    # Use the plot_training_summary function which handles the full pipeline
    if "loss_curves" not in skip_plots or "learning_rate" not in skip_plots:
        try:
            paths = plot_training_summary(
                log_dir=log_dir,
                output_dir=output_dir,
                fmt=fmt,
                dpi=dpi,
            )
            for p in paths:
                generated.append(str(p))
                logger.info(f"  Generated: {Path(p).name}")
        except Exception as e:
            logger.warning(f"  Failed training curves: {e}")

    return generated


def main():
    """Main entry point."""
    args = parse_args()

    # Setup plotting
    setup_seaborn_style()
    setup_matplotlib_defaults()

    # Resolve paths
    if args.experiment_dir:
        exp_dir = Path(args.experiment_dir)
        analysis_dir = Path(args.analysis_dir) if args.analysis_dir else exp_dir / "analysis"
        attention_path = Path(args.attention_path) if args.attention_path else analysis_dir / "attention_weights.h5"
        output_dir = Path(args.output_dir) if args.output_dir else exp_dir / "figures"
        training_log_dir = Path(args.training_log_dir) if args.training_log_dir else exp_dir / "logs"
    else:
        if not args.analysis_dir:
            raise ValueError("Must provide either --experiment-dir or --analysis-dir")
        analysis_dir = Path(args.analysis_dir)
        attention_path = Path(args.attention_path) if args.attention_path else analysis_dir / "attention_weights.h5"
        output_dir = Path(args.output_dir) if args.output_dir else Path("figures")
        training_log_dir = Path(args.training_log_dir) if args.training_log_dir else None

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Load data
    logger.info(f"Loading analysis results from {analysis_dir}...")
    data = load_analysis_data(analysis_dir)

    attention = {}
    if attention_path.exists():
        logger.info(f"Loading attention weights from {attention_path}...")
        attention = load_attention_weights(attention_path)

    if not data and not attention:
        logger.error("No data found to plot. Check input paths.")
        return

    # Extract metadata from HDF5 attention for cell type / gene names
    cell_type_names = list(CELL_TYPE_ORDER)  # Default fallback
    gene_names = None
    if "metadata" in attention:
        loaded_ct = attention["metadata"].get("cell_type_names")
        if loaded_ct is not None and len(loaded_ct) > 0:
            cell_type_names = list(loaded_ct)
        loaded_gn = attention["metadata"].get("gene_names")
        if loaded_gn is not None and len(loaded_gn) > 0:
            gene_names = list(loaded_gn)

    dpi = args.dpi

    # Determine which plot types to generate
    if "all" in args.plot_types:
        plot_categories = list(PLOT_TYPES.keys())
    else:
        plot_categories = args.plot_types

    skip_plots = set(args.skip_plots)
    fmt = args.format

    # Generate plots by category
    all_generated = []

    if "attention" in plot_categories:
        logger.info("Generating attention plots...")
        attention_dir = output_dir / "attention"
        attention_dir.mkdir(parents=True, exist_ok=True)
        generated = generate_attention_plots(
            data, attention, attention_dir, skip_plots, fmt,
            dpi=dpi, cell_type_names=cell_type_names, gene_names=gene_names,
        )
        all_generated.extend(generated)

    if "resilience" in plot_categories:
        logger.info("Generating resilience plots...")
        resilience_dir = output_dir / "attention"
        resilience_dir.mkdir(parents=True, exist_ok=True)
        generated = generate_resilience_plots(data, resilience_dir, skip_plots, fmt, dpi=dpi)
        all_generated.extend(generated)

    if "importance" in plot_categories:
        logger.info("Generating importance plots...")
        importance_dir = output_dir / "importance"
        importance_dir.mkdir(parents=True, exist_ok=True)
        generated = generate_importance_plots(data, importance_dir, skip_plots, fmt, dpi=dpi)
        all_generated.extend(generated)

    if "prediction" in plot_categories:
        logger.info("Generating prediction plots...")
        prediction_dir = output_dir / "prediction"
        prediction_dir.mkdir(parents=True, exist_ok=True)
        generated = generate_prediction_plots(data, prediction_dir, skip_plots, fmt, dpi=dpi)
        all_generated.extend(generated)

    if "embedding" in plot_categories:
        logger.info("Generating embedding plots...")
        embedding_base_dir = output_dir / "embedding"
        embedding_base_dir.mkdir(parents=True, exist_ok=True)
        if "embedding_dirs" in data:
            for emb_name, emb_data in data["embedding_dirs"].items():
                logger.info(f"  Embedding type: {emb_name}")
                emb_output = embedding_base_dir / emb_name
                emb_output.mkdir(parents=True, exist_ok=True)
                generated = generate_embedding_plots(emb_data, emb_output, skip_plots, fmt, dpi=dpi)
                all_generated.extend(generated)
        else:
            generated = generate_embedding_plots(data, embedding_base_dir, skip_plots, fmt, dpi=dpi)
            all_generated.extend(generated)

    if "training" in plot_categories:
        logger.info("Generating training curve plots...")
        training_dir = output_dir / "training"
        training_dir.mkdir(parents=True, exist_ok=True)
        generated = generate_training_plots(training_log_dir, training_dir, skip_plots, fmt, dpi=dpi)
        all_generated.extend(generated)

    # Summary
    logger.info(f"\nGenerated {len(all_generated)} plots:")
    for path in all_generated:
        logger.info(f"  {Path(path).name}")

    if not all_generated:
        logger.warning("No plots were generated. Check that analysis data exists.")

    logger.info(f"\nAll figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
