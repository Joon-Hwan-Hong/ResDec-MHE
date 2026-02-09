"""
Run within-cell-type heterogeneity analysis using PMA attention weights.

This script analyzes which individual cells within each cell type receive
high attention from the Set Transformer's PMA (Pooling by Multihead Attention)
mechanism. This identifies disease-relevant cellular subpopulations.

Usage:
    uv run python scripts/run_cell_heterogeneity.py --experiment-dir experiments/20260113_143052_a3f7b2c1
    uv run python scripts/run_cell_heterogeneity.py --attention-path analysis/attention_weights.h5 --adata-path data/snRNAseq/adata.h5ad
    uv run python scripts/run_cell_heterogeneity.py --experiment-dir experiments/20260113 --top-percentile 10

Workflow:
1. Load PMA attention weights from experiment (or explicit path)
2. Load original anndata to map attention to cell barcodes
3. For each cell type, identify high-attention cells
4. Compute statistics on high-attention vs low-attention cells
5. Save results for downstream analysis (e.g., DEG analysis)

Outputs saved to: {experiment_dir}/analysis/ or --output-dir
- cell_attention_summary.{parquet,csv}: Per cell-type heterogeneity statistics
- high_attention_cells.{parquet,csv}: Cell barcodes with high attention
- cell_attention_scores.{parquet,csv}: All cells with their attention scores
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.constants import CELL_TYPE_ORDER

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run within-cell-type heterogeneity analysis using PMA attention",
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
        "--attention-path",
        type=str,
        help="Explicit path to attention weights HDF5 file",
    )
    input_group.add_argument(
        "--adata-path",
        type=str,
        help="Path to anndata file with cell-level data",
    )
    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for analysis results (default: {experiment-dir}/analysis/)",
    )

    # Analysis parameters
    param_group = parser.add_argument_group("Parameters")
    param_group.add_argument(
        "--top-percentile",
        type=float,
        default=10.0,
        help="Top percentile of cells to consider high-attention (default: 10)",
    )
    param_group.add_argument(
        "--min-cells-per-type",
        type=int,
        default=10,
        help="Minimum cells per cell type to include (default: 10)",
    )
    param_group.add_argument(
        "--cell-type-column",
        type=str,
        default="cell_type",
        help="Column name for cell type in adata.obs (default: cell_type)",
    )
    param_group.add_argument(
        "--subject-column",
        type=str,
        default="subject_id",
        help="Column name for subject ID in adata.obs (default: subject_id)",
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


def load_pma_attention(path: Path) -> dict:
    """
    Load PMA attention weights from HDF5 file.

    Uses io.load_attention_weights() for consistent schema handling, then
    unpacks PMA attention into the 3D format expected by CellHeterogeneityAnalyzer.

    Returns:
        Dict with 'pma_attention' [n_subjects, n_cell_types, n_cells] and metadata
    """
    from src.utils.io import load_attention_weights, unpack_pma_attention

    if not path.exists():
        raise FileNotFoundError(f"Attention file not found: {path}")

    raw = load_attention_weights(path)
    weights = {}

    pma_raw = raw.get("pma_attention")
    cell_type_names = raw.get("cell_type_names")
    if isinstance(cell_type_names, list):
        weights["cell_type_names"] = cell_type_names
    elif "metadata" in raw and "cell_type_names" in raw["metadata"]:
        weights["cell_type_names"] = list(raw["metadata"]["cell_type_names"])

    subject_ids = raw.get("subject_ids")
    if isinstance(subject_ids, list):
        weights["subject_ids"] = subject_ids
    elif "metadata" in raw and "subject_ids" in raw["metadata"]:
        weights["subject_ids"] = list(raw["metadata"]["subject_ids"])

    # Cell barcodes — already decoded by load_attention_weights()
    cell_barcodes = raw.get("cell_barcodes")
    if cell_barcodes is not None and isinstance(cell_barcodes, dict):
        # Strip 'attrs' key from HDF5 group metadata
        weights["cell_barcodes"] = {
            k: v for k, v in cell_barcodes.items() if k != "attrs"
        }

    if pma_raw is not None:
        if isinstance(pma_raw, dict):
            # Nested group from predictor — unpack to 3D
            pma_3d = unpack_pma_attention(pma_raw, weights.get("cell_type_names"))
            if pma_3d is not None:
                weights["pma_attention"] = pma_3d
                logger.info(f"  Loaded pma_attention (unpacked) with shape {pma_3d.shape}")
            else:
                logger.warning("PMA attention group found but could not unpack")
        elif isinstance(pma_raw, np.ndarray):
            # Legacy flat format
            weights["pma_attention"] = pma_raw
            logger.info(f"  Loaded pma_attention with shape {pma_raw.shape}")
    else:
        logger.warning("No PMA attention found in HDF5 file")

    return weights


def load_anndata_obs(path: Path, columns: list[str] | None = None) -> pd.DataFrame | None:
    """
    Load observation metadata from anndata file.

    Args:
        path: Path to .h5ad file
        columns: Columns to load (None = all)

    Returns:
        DataFrame with cell metadata, index is cell barcodes
    """
    try:
        import anndata
    except ImportError:
        logger.warning("anndata not installed. Cell metadata not available.")
        return None

    if not path.exists():
        logger.warning(f"Anndata file not found: {path}")
        return None

    logger.info(f"Loading cell metadata from {path}...")
    adata = anndata.read_h5ad(path, backed="r")
    obs = adata.obs

    if columns:
        available_cols = [c for c in columns if c in obs.columns]
        obs = obs[available_cols]

    return obs.copy()


def main():
    """Main entry point."""
    args = parse_args()

    # Resolve input paths
    if args.experiment_dir:
        exp_dir = Path(args.experiment_dir)
        analysis_dir = exp_dir / "analysis"
        attention_path = Path(args.attention_path) if args.attention_path else analysis_dir / "attention_weights.h5"
        output_dir = Path(args.output_dir) if args.output_dir else analysis_dir
    else:
        if not args.attention_path:
            raise ValueError("Must provide either --experiment-dir or --attention-path")
        attention_path = Path(args.attention_path)
        output_dir = Path(args.output_dir) if args.output_dir else Path("analysis_output")

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Load PMA attention
    logger.info(f"Loading PMA attention from {attention_path}")
    attention_data = load_pma_attention(attention_path)

    if "pma_attention" not in attention_data:
        logger.error("No PMA attention found. Ensure the model saved Set Transformer attention.")
        return

    # Get cell type names
    cell_type_names = attention_data.get("cell_type_names", list(CELL_TYPE_ORDER))
    subject_ids = attention_data.get("subject_ids")
    cell_barcodes = attention_data.get("cell_barcodes")

    # Load cell metadata if anndata provided
    cell_metadata = None
    if args.adata_path:
        cell_metadata = load_anndata_obs(
            Path(args.adata_path),
            columns=[args.cell_type_column, args.subject_column],
        )

    # Run analysis using Analyzer class
    from src.analysis.cell_heterogeneity import CellHeterogeneityAnalyzer

    logger.info("Running cell heterogeneity analysis...")
    analyzer = CellHeterogeneityAnalyzer(
        pma_attention=attention_data["pma_attention"],
        cell_type_names=cell_type_names,
        subject_ids=subject_ids,
        cell_barcodes=cell_barcodes,
        cell_metadata=cell_metadata,
        top_percentile=args.top_percentile,
        min_cells_per_type=args.min_cells_per_type,
    )
    result = analyzer.analyze()
    analyzer.save(result, output_dir, formats=args.formats)

    # Log summary
    summary_df = result.summary
    logger.info(f"\nCell Heterogeneity Summary:")
    logger.info(f"  Cell types analyzed: {len(summary_df)}")
    logger.info(f"  High-attention cells identified: {len(result.high_attention_cells)}")
    logger.info(f"  Total cells with scores: {len(result.all_scores)}")

    if len(summary_df) > 0:
        logger.info(f"\nMost heterogeneous cell types (by Gini coefficient):")
        for _, row in summary_df.head(5).iterrows():
            logger.info(f"  {row['cell_type']}: Gini={row['gini_coefficient']:.3f}, "
                       f"n_cells={row['n_cells_total']}, n_high={row['n_high_attention']}")

    logger.info(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
