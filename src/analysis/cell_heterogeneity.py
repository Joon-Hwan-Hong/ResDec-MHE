"""
Within-cell-type heterogeneity analysis using PMA attention weights.

Identifies which individual cells within each cell type receive high attention
from the Set Transformer's PMA mechanism, highlighting disease-relevant
cellular subpopulations.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.utils.statistics import gini_coefficient

logger = logging.getLogger(__name__)


def analyze_cell_heterogeneity(
    pma_attention: np.ndarray,
    cell_type_names: list[str],
    subject_ids: list[str] | None = None,
    cell_barcodes: dict | None = None,
    cell_metadata: pd.DataFrame | None = None,
    top_percentile: float = 10.0,
    min_cells_per_type: int = 10,
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

    # Enrich with cell metadata if available
    if cell_metadata is not None and "cell_barcode" in high_attention_df.columns:
        meta_cols = [c for c in cell_metadata.columns if c not in high_attention_df.columns]
        if meta_cols:
            high_attention_df = high_attention_df.merge(
                cell_metadata[meta_cols], left_on="cell_barcode", right_index=True, how="left"
            )
    if cell_metadata is not None and "cell_barcode" in all_scores_df.columns:
        meta_cols = [c for c in cell_metadata.columns if c not in all_scores_df.columns]
        if meta_cols:
            all_scores_df = all_scores_df.merge(
                cell_metadata[meta_cols], left_on="cell_barcode", right_index=True, how="left"
            )

    return summary_df, high_attention_df, all_scores_df
