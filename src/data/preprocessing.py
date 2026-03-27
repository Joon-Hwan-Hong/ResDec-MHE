"""
Preprocessing pipeline for ROSMAP snRNA-seq data.

Workflow: Raw counts → HVG (on raw) + L-R forcing → Normalize + log1p → Filter → Ready for LIANA+ & ML

Note: seurat_v3 HVG selection requires RAW COUNTS, not log-normalized data.

Memory-efficient design:
  The full ROSMAP AnnData (3.9M cells × 20K genes, 13B nnz) takes ~106GB in
  memory as a sparse CSR matrix. Additionally, the file has a sparse dtype
  mismatch (indices=int32, indptr=int64) that causes scipy sort_indices() to
  fail. The pipeline avoids intermediate full-matrix copies by:
  - Using bincount for gene filtering (avoids sort_indices)
  - Subsampling cells for HVG selection (avoids scanpy's internal copy)
  - Performing ONE column subset at the end (20K → ~4K genes)
"""

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy import sparse

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
    hvg_subsample_n: int = 100_000,
    copy: bool = True,
    training_subject_ids: list[str] | None = None,
    subject_column: str = "ROSMAP_IndividualID",
    seed: int = 42,
) -> AnnData:
    """
    Preprocess merged ROSMAP snRNA-seq data for LIANA+ and ML model.

    Memory-efficient pipeline for large datasets (~100GB+):
    1. Round CellBender counts (in-place)
    2. Compute gene QC mask (bincount, no copy)
    3. Subsample cells for HVG selection (small copy)
    4. Force include L-R genes from CellChatDB
    5. One column subset to final gene set (~4K genes)
    6. Normalize (target_sum) + log1p on small matrix
    7. Store raw counts in .raw

    Args:
        adata_path: Path to adata_ROSMAP_merged.h5ad
        cellchatdb_path: Path to CellChatDB interaction database
        n_hvg: Number of highly variable genes to select
        target_sum: Target sum for normalization
        min_cells_per_gene: Minimum cells expressing a gene
        hvg_flavor: Method for HVG selection
        hvg_subsample_n: Number of cells to subsample for HVG selection.
            100K cells gives stable variance estimates while using ~3GB
            instead of ~106GB for the full matrix.
        copy: Whether to return a copy (recommended for small datasets,
            set False for large datasets to avoid doubling memory)
        training_subject_ids: Subject IDs to restrict HVG selection to.
            Prevents data leakage by ensuring gene variance estimates come
            only from training subjects, not test/val. If None, uses all
            cells (legacy behavior, suitable when splits are unavailable).
        subject_column: Column in adata.obs containing subject IDs.
        seed: Random seed for HVG subsampling reproducibility.

    Returns:
        Preprocessed AnnData with HVG + L-R genes, normalized and log-transformed
    """
    logger.info(f"Loading data from {adata_path}...")
    adata = sc.read_h5ad(adata_path)

    if copy and adata.n_obs < 500_000:
        # Only copy for small datasets — for large ones it would OOM
        adata = adata.copy()

    logger.info(f"Initial shape: {adata.shape[0]:,} cells x {adata.shape[1]:,} genes")

    if sparse.issparse(adata.X):
        logger.info(
            f"Sparse matrix: {adata.X.nnz:,} nnz, "
            f"indices={adata.X.indices.dtype}, indptr={adata.X.indptr.dtype}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: Round counts for CellBender-corrected subjects (IN-PLACE)
    # ─────────────────────────────────────────────────────────────────────────
    # 20 subjects in the "DLPFC" batch have fractional counts from CellBender
    # ambient RNA correction. Round to nearest integer so seurat_v3 HVG
    # selection operates on proper count data.
    if sparse.issparse(adata.X):
        np.round(adata.X.data, out=adata.X.data)
    else:
        np.round(adata.X, out=adata.X)
    logger.info("Rounded counts to integers (handles CellBender-corrected subjects)")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2: Compute gene QC mask (NO COPY — just compute which genes to keep)
    # ─────────────────────────────────────────────────────────────────────────
    # Use bincount instead of sc.pp.filter_genes to avoid triggering scipy's
    # sort_indices (which fails on the int32/int64 dtype mismatch).
    if sparse.issparse(adata.X):
        cells_per_gene = np.bincount(adata.X.indices, minlength=adata.shape[1])
    else:
        cells_per_gene = np.count_nonzero(adata.X, axis=0)
    min_cells_mask = cells_per_gene >= min_cells_per_gene
    n_passing = min_cells_mask.sum()
    logger.info(
        f"Gene QC mask (min_cells={min_cells_per_gene}): {n_passing:,} pass, "
        f"{adata.n_vars - n_passing:,} fail"
    )

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3: HVG selection via cell subsampling
    # ─────────────────────────────────────────────────────────────────────────
    # scanpy's seurat_v3 internally copies the full matrix. For a ~106GB
    # matrix this OOMs. Instead, subsample cells for HVG estimation —
    # 100K cells gives stable mean/variance estimates.
    #
    # When training_subject_ids is provided, subsample ONLY from training
    # subjects to prevent gene variance information from test/val subjects
    # leaking into the feature set.
    hvg_mask = np.zeros(adata.n_vars, dtype=bool)

    if hvg_flavor == "seurat_v3":
        rng = np.random.default_rng(seed)

        if training_subject_ids is not None:
            train_mask = adata.obs[subject_column].isin(training_subject_ids).values
            train_cell_idx = np.where(train_mask)[0]
            n_sub = min(hvg_subsample_n, len(train_cell_idx))
            sub_idx = rng.choice(train_cell_idx, size=n_sub, replace=False)
            logger.info(
                f"Subsampling {n_sub:,} cells from {len(training_subject_ids)} "
                f"training subjects for HVG selection (leak-free)..."
            )
        else:
            n_sub = min(hvg_subsample_n, adata.n_obs)
            sub_idx = rng.choice(adata.n_obs, size=n_sub, replace=False)
            logger.info(f"Subsampling {n_sub:,} cells for HVG selection...")

        sub_idx.sort()  # Sorted for efficient CSR row slicing

        # Chain row + column slicing before copying — avoids materializing at full
        # gene count. Both adata[sub_idx] and [:, min_cells_mask] return views;
        # .copy() materializes only the final [n_sub × n_filtered_genes] matrix.
        adata_sub = adata[sub_idx][:, min_cells_mask].copy()

        logger.info(f"Subsample shape: {adata_sub.shape}")

        sc.pp.highly_variable_genes(
            adata_sub,
            n_top_genes=n_hvg,
            flavor=hvg_flavor,
        )

        # Map HVG results back to full gene indices
        sub_gene_names = set(adata_sub.var_names[adata_sub.var["highly_variable"]])
        hvg_mask = adata.var_names.isin(sub_gene_names)

        del adata_sub
        logger.info(f"HVG selection ({hvg_flavor}) on {n_sub:,}-cell subsample: {hvg_mask.sum():,} genes")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4: Force include L-R genes from CellChatDB
    # ─────────────────────────────────────────────────────────────────────────
    cellchatdb_path = Path(cellchatdb_path)
    lr_genes: set[str] = set()
    lr_mask = np.zeros(adata.n_vars, dtype=bool)

    if cellchatdb_path.exists():
        lr_genes = get_lr_genes_from_cellchatdb(cellchatdb_path)
        logger.info(f"Loaded {len(lr_genes):,} L-R genes from CellChatDB")

        lr_mask = adata.var_names.isin(lr_genes)
        n_lr_in_data = lr_mask.sum()
        n_lr_added = (lr_mask & ~hvg_mask & min_cells_mask).sum()
        logger.info(f"L-R genes in data: {n_lr_in_data:,}, added (not in HVG): {n_lr_added:,}")
    else:
        logger.warning(f"CellChatDB not found at {cellchatdb_path}, skipping L-R forcing")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 5: ONE column subset to final gene set
    # ─────────────────────────────────────────────────────────────────────────
    # Combine: must pass min_cells AND (be HVG OR be L-R gene)
    final_mask = min_cells_mask & (hvg_mask | lr_mask)
    n_final = final_mask.sum()
    logger.info(f"Final gene set: {n_final:,} genes (from {adata.n_vars:,})")

    # This is the ONE copy — from 20K genes to ~4K genes
    logger.info("Subsetting to final gene set (this may take a few minutes)...")
    adata = adata[:, final_mask].copy()
    logger.info(f"After subset: {adata.shape[0]:,} cells x {adata.shape[1]:,} genes")

    if sparse.issparse(adata.X):
        logger.info(f"Subset nnz: {adata.X.nnz:,}")

    # Store gene metadata — all genes in the final set are "selected" (HVG or L-R forced)
    adata.var["highly_variable"] = True
    adata.var["lr_gene"] = adata.var_names.isin(lr_genes) if lr_genes else False

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 6: Normalize + Log transform
    # ─────────────────────────────────────────────────────────────────────────
    # Store raw counts in .raw BEFORE normalization
    adata.raw = adata.copy()
    logger.info("Stored raw counts in adata.raw")

    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    logger.info(f"Normalized (target_sum={target_sum:.0e}) and log1p transformed")

    # For non-seurat_v3 flavors, run HVG on normalized data.
    # WARNING: This path runs HVG on already-subsetted data (~4K genes).
    # The design doc specifies seurat_v3 exclusively for production use.
    # Non-seurat_v3 flavors may produce unexpected results at this stage.
    if hvg_flavor != "seurat_v3":
        logger.warning(
            "Non-seurat_v3 HVG running on already-subsetted data (%d genes). "
            "This path is not validated at scale. seurat_v3 is recommended.",
            adata.n_vars,
        )
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_hvg,
            flavor=hvg_flavor,
        )
        logger.info(f"HVG selection ({hvg_flavor}) on normalized data: {adata.var['highly_variable'].sum():,} genes")

        # Add L-R genes
        if lr_mask.any():
            adata.var["highly_variable"] = adata.var["highly_variable"] | adata.var["lr_gene"]

        # Filter to final set
        adata = adata[:, adata.var["highly_variable"]].copy()

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
