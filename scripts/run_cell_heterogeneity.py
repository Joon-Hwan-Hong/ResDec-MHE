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
- cell_heterogeneity_summary.{parquet,csv}: Per cell-type heterogeneity statistics
- high_attention_cells.{parquet,csv}: Cell barcodes with high attention
- cell_attention_scores.{parquet,csv}: All cells with their attention scores
"""

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from src.data.constants import CELL_TYPE_ORDER
from src.utils.io import save_dataframe
from src.utils.statistics import gini_coefficient

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
    input_group.add_argument(
        "--metadata-path",
        type=str,
        help="Path to subject metadata CSV/parquet",
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
    unpacks PMA attention into the 3D format expected by analyze_cell_heterogeneity().

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


def analyze_cell_heterogeneity(
    pma_attention: np.ndarray,
    cell_type_names: list[str],
    subject_ids: list[str] | None = None,
    cell_barcodes: dict | None = None,
    cell_metadata: pd.DataFrame | None = None,
    top_percentile: float = 10.0,
    min_cells_per_type: int = 10,
    cell_type_column: str = "cell_type",
    subject_column: str = "subject_id",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Analyze within-cell-type heterogeneity from PMA attention.

    Args:
        pma_attention: PMA attention weights [n_subjects, n_cell_types, n_cells]
        cell_type_names: Names of cell types
        subject_ids: Subject identifiers
        cell_barcodes: Dict mapping "subject_idx_celltype_idx" to cell barcodes
        cell_metadata: DataFrame with cell-level metadata (index=barcodes)
        top_percentile: Percentile threshold for high-attention cells
        min_cells_per_type: Minimum cells per type to analyze
        cell_type_column: Column name for cell type in metadata
        subject_column: Column name for subject ID in metadata

    Returns:
        Tuple of (summary_df, high_attention_df, all_scores_df)
    """
    n_subjects, n_cell_types, max_cells = pma_attention.shape

    if subject_ids is None:
        subject_ids = [f"subject_{i}" for i in range(n_subjects)]

    # Summary statistics per cell type
    summary_rows = []

    # High attention cells
    high_attention_rows = []

    # All cell scores
    all_scores_rows = []

    for ct_idx, ct_name in enumerate(cell_type_names):
        ct_attention = pma_attention[:, ct_idx, :]  # [n_subjects, n_cells]

        # Flatten across subjects for global statistics
        # Note: Many slots may be padding (zeros)
        valid_mask = ct_attention > 0
        valid_attention = ct_attention[valid_mask]

        if len(valid_attention) < min_cells_per_type:
            logger.warning(f"Cell type '{ct_name}' has only {len(valid_attention)} cells, skipping")
            continue

        # Compute threshold for high attention
        threshold = np.percentile(valid_attention, 100 - top_percentile)

        # Statistics
        summary_rows.append({
            "cell_type": ct_name,
            "n_cells_total": int(valid_mask.sum()),
            "n_high_attention": int((valid_attention >= threshold).sum()),
            "attention_mean": float(valid_attention.mean()),
            "attention_std": float(valid_attention.std()),
            "attention_median": float(np.median(valid_attention)),
            "attention_threshold": float(threshold),
            "attention_entropy": float(-np.sum(
                valid_attention / valid_attention.sum() *
                np.log(valid_attention / valid_attention.sum() + 1e-10)
            )),
            "gini_coefficient": float(gini_coefficient(valid_attention)),
        })

        # Identify high-attention cells per subject
        for subj_idx, subj_id in enumerate(subject_ids):
            subj_attention = ct_attention[subj_idx]
            subj_valid = subj_attention > 0

            if subj_valid.sum() == 0:
                continue

            # Get indices of high attention cells
            high_mask = subj_attention >= threshold

            for cell_idx in np.where(high_mask)[0]:
                row = {
                    "subject_id": subj_id,
                    "cell_type": ct_name,
                    "cell_idx": int(cell_idx),
                    "attention_score": float(subj_attention[cell_idx]),
                }

                # Add barcode if available
                barcode_key = f"{subj_idx}_{ct_idx}"
                if cell_barcodes and barcode_key in cell_barcodes:
                    barcodes = cell_barcodes[barcode_key]
                    if cell_idx < len(barcodes):
                        row["cell_barcode"] = barcodes[cell_idx]

                high_attention_rows.append(row)

            # Also collect all scores for this subject-celltype
            for cell_idx in np.where(subj_valid)[0]:
                row = {
                    "subject_id": subj_id,
                    "cell_type": ct_name,
                    "cell_idx": int(cell_idx),
                    "attention_score": float(subj_attention[cell_idx]),
                    "is_high_attention": bool(subj_attention[cell_idx] >= threshold),
                }

                barcode_key = f"{subj_idx}_{ct_idx}"
                if cell_barcodes and barcode_key in cell_barcodes:
                    barcodes = cell_barcodes[barcode_key]
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
            ascending=[True, False]
        ).reset_index(drop=True)

    return summary_df, high_attention_df, all_scores_df


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

    # Run analysis
    logger.info("Running cell heterogeneity analysis...")
    summary_df, high_attention_df, all_scores_df = analyze_cell_heterogeneity(
        pma_attention=attention_data["pma_attention"],
        cell_type_names=cell_type_names,
        subject_ids=subject_ids,
        cell_barcodes=cell_barcodes,
        cell_metadata=cell_metadata,
        top_percentile=args.top_percentile,
        min_cells_per_type=args.min_cells_per_type,
        cell_type_column=args.cell_type_column,
        subject_column=args.subject_column,
    )

    # Save results
    for fmt in args.formats:
        save_dataframe(summary_df, output_dir / f"cell_heterogeneity_summary.{fmt}", fmt)
        save_dataframe(high_attention_df, output_dir / f"high_attention_cells.{fmt}", fmt)
        # All scores can be large, only save parquet by default
        if fmt == "parquet" or len(all_scores_df) < 100000:
            save_dataframe(all_scores_df, output_dir / f"cell_attention_scores.{fmt}", fmt)

    # Log summary
    logger.info(f"\nCell Heterogeneity Summary:")
    logger.info(f"  Cell types analyzed: {len(summary_df)}")
    logger.info(f"  High-attention cells identified: {len(high_attention_df)}")
    logger.info(f"  Total cells with scores: {len(all_scores_df)}")

    if len(summary_df) > 0:
        logger.info(f"\nMost heterogeneous cell types (by Gini coefficient):")
        for _, row in summary_df.head(5).iterrows():
            logger.info(f"  {row['cell_type']}: Gini={row['gini_coefficient']:.3f}, "
                       f"n_cells={row['n_cells_total']}, n_high={row['n_high_attention']}")

    logger.info(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
