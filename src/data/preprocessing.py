"""
Preprocessing pipeline for ROSMAP snRNA-seq data.

Workflow: Raw counts → HVG (on raw) + L-R forcing → Normalize + log1p → Filter → Ready for LIANA+ & ML

Note: seurat_v3 HVG selection requires RAW COUNTS, not log-normalized data.
"""

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

logger = logging.getLogger(__name__)

from src.data.constants import CELLCHATDB_PATH


def get_lr_genes_from_cellchatdb(cellchatdb_path: str | Path) -> set[str]:
    """
    Extract all ligand and receptor genes from CellChatDB.

    Handles complex names like "TGFB1_TGFBR1_TGFBR2" by splitting on underscore.

    Args:
        cellchatdb_path: Path to CellChatDB_human_interaction.csv

    Returns:
        Set of gene symbols
    """
    cellchatdb = pd.read_csv(cellchatdb_path)
    lr_genes = set()

    for col in ["ligand.symbol", "receptor.symbol"]:
        if col not in cellchatdb.columns:
            # Try alternative column names
            alt_col = col.replace(".", "_")
            if alt_col in cellchatdb.columns:
                col = alt_col
            else:
                continue

        # Split complex names and collect all genes
        genes = cellchatdb[col].dropna().str.split("_").explode().unique()
        lr_genes.update(genes)

    return lr_genes


def preprocess_adata(
    adata_path: str | Path,
    cellchatdb_path: str | Path = CELLCHATDB_PATH,
    n_hvg: int = 4000,
    target_sum: float = 1e4,
    min_cells_per_gene: int = 10,
    hvg_flavor: Literal["seurat_v3", "cell_ranger", "seurat"] = "seurat_v3",
    copy: bool = True,
) -> AnnData:
    """
    Preprocess merged ROSMAP snRNA-seq data for LIANA+ and ML model.

    Pipeline:
    1. Basic QC (filter genes by min_cells)
    2. HVG selection on RAW COUNTS (seurat_v3 requires raw counts)
    3. Force include L-R genes from CellChatDB
    4. Store raw counts in .raw
    5. Normalize (target_sum) + log1p
    6. Filter to final gene set

    Args:
        adata_path: Path to adata_ROSMAP_merged.h5ad
        cellchatdb_path: Path to CellChatDB interaction database
        n_hvg: Number of highly variable genes to select
        target_sum: Target sum for normalization
        min_cells_per_gene: Minimum cells expressing a gene
        hvg_flavor: Method for HVG selection
        copy: Whether to return a copy (recommended)

    Returns:
        Preprocessed AnnData with HVG + L-R genes, normalized and log-transformed
    """
    logger.info(f"Loading data from {adata_path}...")
    adata = sc.read_h5ad(adata_path)

    if copy:
        adata = adata.copy()

    logger.info(f"Initial shape: {adata.shape[0]:,} cells x {adata.shape[1]:,} genes")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: Basic QC - Filter genes by minimum cells
    # ─────────────────────────────────────────────────────────────────────────
    n_genes_before = adata.n_vars
    sc.pp.filter_genes(adata, min_cells=min_cells_per_gene)
    logger.info(f"After gene filter (min_cells={min_cells_per_gene}): {adata.n_vars:,} genes "
                f"(removed {n_genes_before - adata.n_vars:,})")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1b: Round counts for CellBender-corrected subjects
    # ─────────────────────────────────────────────────────────────────────────
    # 20 subjects in the "DLPFC" batch have fractional counts from CellBender
    # ambient RNA correction. Batch B32 (200709-B32-A/B) is entirely composed
    # of decontaminated subjects (all 8), while the other 12 are scattered
    # across 7 additional batches that also contain unaffected subjects —
    # the correction was applied per-subject, not per-batch. Round to nearest
    # integer so seurat_v3 HVG selection operates on proper count data.
    from scipy import sparse
    if sparse.issparse(adata.X):
        np.round(adata.X.data, out=adata.X.data)
    else:
        np.round(adata.X, out=adata.X)
    logger.info("Rounded counts to integers (handles CellBender-corrected subjects)")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2: HVG selection on RAW COUNTS
    # ─────────────────────────────────────────────────────────────────────────
    # IMPORTANT: seurat_v3 requires raw counts, NOT log-normalized data
    # Other flavors (cell_ranger, seurat) expect log-normalized data
    if hvg_flavor == "seurat_v3":
        # seurat_v3 expects raw counts - run BEFORE normalization
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_hvg,
            flavor=hvg_flavor,
        )
        logger.info(f"HVG selection ({hvg_flavor}) on raw counts: {adata.var['highly_variable'].sum():,} genes")
    # For other flavors, we'll run HVG after normalization (handled below)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3: Force include L-R genes from CellChatDB
    # ─────────────────────────────────────────────────────────────────────────
    cellchatdb_path = Path(cellchatdb_path)
    if cellchatdb_path.exists():
        lr_genes = get_lr_genes_from_cellchatdb(cellchatdb_path)
        logger.info(f"Loaded {len(lr_genes):,} L-R genes from CellChatDB")

        # Find L-R genes present in our data
        lr_in_data = adata.var_names.isin(lr_genes)
        n_lr_in_data = lr_in_data.sum()
        logger.info(f"L-R genes present in data: {n_lr_in_data:,}")

        # Mark L-R genes
        adata.var["lr_gene"] = lr_in_data

        if hvg_flavor == "seurat_v3":
            # HVG already computed, add L-R genes
            n_hvg_only = adata.var["highly_variable"].sum()
            adata.var["highly_variable"] = adata.var["highly_variable"] | lr_in_data
            n_lr_added = adata.var["highly_variable"].sum() - n_hvg_only
            logger.info(f"L-R genes added (not in HVG): {n_lr_added:,}")
    else:
        logger.warning(f"CellChatDB not found at {cellchatdb_path}, skipping L-R forcing")
        adata.var["lr_gene"] = False

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4: Store raw counts BEFORE normalization
    # ─────────────────────────────────────────────────────────────────────────
    adata.raw = adata.copy()
    logger.info("Stored raw counts in adata.raw")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 5: Normalize + Log transform
    # ─────────────────────────────────────────────────────────────────────────
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    logger.info(f"Normalized (target_sum={target_sum:.0e}) and log1p transformed")

    # For non-seurat_v3 flavors, run HVG on normalized data
    if hvg_flavor != "seurat_v3":
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_hvg,
            flavor=hvg_flavor,
        )
        logger.info(f"HVG selection ({hvg_flavor}) on normalized data: {adata.var['highly_variable'].sum():,} genes")

        # Add L-R genes if CellChatDB was loaded
        if "lr_gene" in adata.var.columns and adata.var["lr_gene"].any():
            n_hvg_only = adata.var["highly_variable"].sum()
            adata.var["highly_variable"] = adata.var["highly_variable"] | adata.var["lr_gene"]
            n_lr_added = adata.var["highly_variable"].sum() - n_hvg_only
            logger.info(f"L-R genes added (not in HVG): {n_lr_added:,}")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 6: Filter to final gene set
    # ─────────────────────────────────────────────────────────────────────────
    n_final = adata.var["highly_variable"].sum()
    adata = adata[:, adata.var["highly_variable"]].copy()

    logger.info(f"Final gene set: {n_final:,} genes")
    logger.info(f"Final shape: {adata.shape[0]:,} cells x {adata.shape[1]:,} genes")

    return adata


def get_subjects_with_min_cells(
    adata: AnnData,
    subject_column: str = "ROSMAP_IndividualID",
    min_cells: int = 100,
) -> list[str]:
    """
    Get list of subjects with at least min_cells.

    Args:
        adata: AnnData object
        subject_column: Column containing subject IDs
        min_cells: Minimum number of cells required

    Returns:
        List of subject IDs meeting the threshold
    """
    cell_counts = adata.obs[subject_column].value_counts()
    valid_subjects = cell_counts[cell_counts >= min_cells].index.tolist()
    return valid_subjects
