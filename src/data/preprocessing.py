"""
Preprocessing pipeline for ROSMAP snRNA-seq data.

Two entry points:
  1. merge_h5ads() — merge multiple source h5ads into one file (backed, chunked, concat_on_disk)
  2. preprocess_adata() — HVG + CellChatDB gene selection, normalize, store .raw

CLI usage (merge + preprocess in one shot):
    uv run python -m src.data.preprocessing \
        --dlpfc /path/to/ROSMAP_DLPFC_ABC_mapped.h5ad \
        --multiregion /path/to/ROSMAP_111.h5ad \
        --output data/snRNAseq/adata_ROSMAP_merged.h5ad

    uv run python -m src.data.preprocessing \
        --preprocess-only data/snRNAseq/adata_ROSMAP_merged_raw.h5ad \
        --output data/snRNAseq/adata_ROSMAP_merged.h5ad

Notes:
  - seurat_v3 HVG selection requires RAW COUNTS, not log-normalized data
  - scanpy's seurat_v3 internally copies the full matrix; for ~106GB this OOMs,
    so we subsample 100K cells for HVG estimation (stable mean/variance estimates)
  - The merged raw h5ad has a sparse dtype mismatch (indices=int32, indptr=int64)
    that causes scipy sort_indices() to fail; we use bincount to avoid this
"""

import argparse
import gc
import logging
import time
from pathlib import Path
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy import sparse
from scipy.sparse import csr_matrix, vstack

logger = logging.getLogger(__name__)

from src.data.constants import CELLCHATDB_PATH


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

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
            alt_col = col.replace(".", "_")
            if alt_col in cellchatdb.columns:
                col = alt_col
            else:
                continue

        # Split on both underscore and comma (CellChatDB uses both as delimiters
        # for complex receptor names like "ACVR1_ACVR2A" or "ACVR1, ACVR2A")
        genes = cellchatdb[col].dropna().str.replace(",", "_", regex=False).str.split("_").explode().str.strip()
        lr_genes.update(g for g in genes.unique() if g)

    return lr_genes


def get_subjects_with_min_cells(
    adata: AnnData,
    subject_column: str = "ROSMAP_IndividualID",
    min_cells: int = 100,
) -> list[str]:
    """Get list of subjects with at least min_cells."""
    cell_counts = adata.obs[subject_column].value_counts()
    valid_subjects = cell_counts[cell_counts >= min_cells].index.tolist()
    return valid_subjects


# ─────────────────────────────────────────────────────────────────────────────
# Merge
# ─────────────────────────────────────────────────────────────────────────────

def _extract_genes_backed(
    h5ad_path: str,
    gene_names: list[str],
    output_path: str,
    obs_renames: dict | None = None,
    obs_additions: dict | None = None,
    chunk_size: int = 200_000,
) -> None:
    """Backed-read h5ad, extract only specified genes in chunks, save."""
    logger.info("  Extracting %d genes from %s...", len(gene_names), Path(h5ad_path).name)
    adata = ad.read_h5ad(h5ad_path, backed="r")
    n_cells = adata.n_obs

    gene_locs = [adata.var_names.get_loc(g) for g in gene_names if g in adata.var_names]
    gene_locs_sorted = sorted(gene_locs)
    gene_order = np.argsort(np.argsort(gene_locs))
    actual_genes = [g for g in gene_names if g in adata.var_names]

    obs_df = adata.obs.copy()
    if obs_renames:
        obs_df = obs_df.rename(columns=obs_renames)
    if obs_additions:
        for k, v in obs_additions.items():
            obs_df[k] = v

    var_df = pd.DataFrame(index=actual_genes)

    chunks = []
    for start in range(0, n_cells, chunk_size):
        end = min(start + chunk_size, n_cells)
        chunk = adata.X[start:end][:, gene_locs_sorted]
        if not isinstance(chunk, csr_matrix):
            chunk = csr_matrix(chunk, dtype=np.float32)
        elif chunk.dtype != np.float32:
            chunk = chunk.astype(np.float32)
        chunk = chunk[:, gene_order]
        chunks.append(chunk)
        if end % (chunk_size * 5) == 0 or end >= n_cells:
            logger.info("    %d / %d cells", end, n_cells)

    X = vstack(chunks, format="csr")
    X.indices = X.indices.astype(np.int64)
    X.indptr = X.indptr.astype(np.int64)
    del chunks

    adata.file.close()
    del adata

    out = ad.AnnData(X=X, obs=obs_df, var=var_df)
    out.write_h5ad(output_path)
    logger.info("  Saved: %d cells x %d genes", out.n_obs, out.n_vars)
    del out, X


def merge_h5ads(
    dlpfc_path: str | Path,
    multiregion_path: str | Path,
    output_path: str | Path,
    chunk_size: int = 200_000,
) -> Path:
    """
    Merge DLPFC and multiregion h5ads into one file with common genes.

    Uses backed reading + chunked extraction + concat_on_disk to stay
    within ~250GB RAM. Output has raw counts, all ~20K common genes,
    and harmonized obs columns.

    Args:
        dlpfc_path: Path to ROSMAP DLPFC h5ad
        multiregion_path: Path to ROSMAP multiregion h5ad
        output_path: Where to write the merged h5ad
        chunk_size: Cells per chunk for backed extraction

    Returns:
        Path to the merged h5ad
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Find common genes
    logger.info("Finding common genes (backed)...")
    dlpfc = ad.read_h5ad(str(dlpfc_path), backed="r")
    dlpfc_genes = set(dlpfc.var_names)
    dlpfc.file.close()

    mr = ad.read_h5ad(str(multiregion_path), backed="r")
    mr_genes = set(mr.var_names)
    mr.file.close()

    common_genes = sorted(dlpfc_genes & mr_genes)
    logger.info("Common genes: %d", len(common_genes))

    # Step 2: Extract common genes from each source
    tmp_dlpfc = str(output_path.parent / "_tmp_dlpfc.h5ad")
    tmp_mr = str(output_path.parent / "_tmp_mr.h5ad")

    _extract_genes_backed(
        str(dlpfc_path), common_genes, tmp_dlpfc,
        obs_renames={"individualID": "ROSMAP_IndividualID"},
        obs_additions={"BrainRegion": "PFC", "dataset": "DLPFC"},
        chunk_size=chunk_size,
    )

    _extract_genes_backed(
        str(multiregion_path), common_genes, tmp_mr,
        obs_additions={"dataset": "multiregion"},
        chunk_size=chunk_size,
    )

    # Step 3: Concat on disk
    logger.info("Concatenating on disk...")
    ad.experimental.concat_on_disk(
        in_files=[tmp_dlpfc, tmp_mr],
        out_file=str(output_path),
        join="inner",
        merge="same",
    )
    Path(tmp_dlpfc).unlink(missing_ok=True)
    Path(tmp_mr).unlink(missing_ok=True)
    logger.info("Merge complete → %s", output_path)

    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Preprocess
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_adata(
    adata_path: str | Path,
    cellchatdb_path: str | Path = CELLCHATDB_PATH,
    n_hvg: int = 4000,
    target_sum: float = 1e4,
    min_cells_per_gene: int = 10,
    hvg_flavor: Literal["seurat_v3", "cell_ranger", "seurat", "blocked"] = "blocked",
    hvg_subsample_n: int = 100_000,
    hvg_per_type_n: int = 5_000,
    copy: bool = True,
    training_subject_ids: list[str] | None = None,
    subject_column: str = "ROSMAP_IndividualID",
    cell_type_column: str = "supercluster_name",
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
        adata_path: Path to merged h5ad with raw counts (~20K genes)
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
        adata = adata.copy()

    logger.info(f"Initial shape: {adata.shape[0]:,} cells x {adata.shape[1]:,} genes")

    if sparse.issparse(adata.X):
        logger.info(
            f"Sparse matrix: {adata.X.nnz:,} nnz, "
            f"indices={adata.X.indices.dtype}, indptr={adata.X.indptr.dtype}"
        )

    # ── STEP 1: Round counts for CellBender-corrected subjects (IN-PLACE) ──
    # 20 subjects in the "DLPFC" batch have fractional counts from CellBender
    # ambient RNA correction. Round to nearest integer so seurat_v3 HVG
    # selection operates on proper count data.
    if sparse.issparse(adata.X):
        np.round(adata.X.data, out=adata.X.data)
    else:
        np.round(adata.X, out=adata.X)
    logger.info("Rounded counts to integers (handles CellBender-corrected subjects)")

    # ── STEP 2: Compute gene QC mask (NO COPY — just compute which genes to keep) ──
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

    # ── STEP 3: HVG selection via cell subsampling ──
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

    elif hvg_flavor == "blocked":
        # scran-style blocked HVG: equal-weight per-cell-type variance,
        # loess-normalized to remove mean-expression bias.
        rng = np.random.default_rng(seed)

        # Determine which cells are eligible
        if training_subject_ids is not None:
            eligible_mask = adata.obs[subject_column].isin(training_subject_ids).values
        else:
            eligible_mask = np.ones(adata.n_obs, dtype=bool)

        # Also filter to genes passing min_cells QC
        gene_mask = min_cells_mask
        eligible_idx = np.where(eligible_mask)[0]

        # Stratified subsample: equal N per cell type
        ct_values = adata.obs[cell_type_column].values[eligible_idx]
        unique_types = np.unique(ct_values)
        sub_idx = []
        type_sample_counts = {}
        for ct in unique_types:
            ct_positions = eligible_idx[ct_values == ct]
            n = min(hvg_per_type_n, len(ct_positions))
            if n > 0:
                chosen = rng.choice(ct_positions, size=n, replace=False)
                sub_idx.extend(chosen)
                type_sample_counts[ct] = n
        sub_idx = np.array(sorted(sub_idx))
        logger.info(
            f"Blocked HVG: stratified subsample {len(sub_idx):,} cells "
            f"from {len(unique_types)} types (up to {hvg_per_type_n} per type)"
        )
        for ct, n in sorted(type_sample_counts.items(), key=lambda x: -x[1])[:5]:
            logger.info(f"  {ct}: {n:,} cells")

        # Subset to QC-passing genes and subsample
        X_sub = adata[sub_idx][:, gene_mask].X
        if sparse.issparse(X_sub):
            X_sub = X_sub.toarray()
        X_sub = X_sub.astype(np.float32)
        gene_names_filtered = adata.var_names[gene_mask]

        # Build one-hot cell type indicator [n_sub, n_types]
        ct_sub = adata.obs[cell_type_column].values[sub_idx]
        ct_categories = np.unique(ct_sub)
        ct_to_idx = {ct: i for i, ct in enumerate(ct_categories)}
        ct_codes = np.array([ct_to_idx[c] for c in ct_sub])
        n_types = len(ct_categories)

        C = sparse.csc_matrix(
            (np.ones(len(ct_codes)), (np.arange(len(ct_codes)), ct_codes)),
            shape=(len(ct_codes), n_types),
        )

        # Per-type variance via E[X^2] - E[X]^2 (no dense intermediates beyond X_sub)
        type_counts = np.asarray(C.sum(axis=0)).ravel().astype(np.float64)
        type_counts_safe = np.maximum(type_counts, 1)

        type_means = (C.T @ X_sub) / type_counts_safe[:, None]       # [n_types, n_genes]
        type_mean_sq = (C.T @ (X_sub ** 2)) / type_counts_safe[:, None]
        variances = type_mean_sq - type_means ** 2                     # [n_types, n_genes]

        # Mask out types with too few cells
        for t in range(n_types):
            if type_counts[t] < 10:
                variances[t] = np.nan

        # Equal-weight average across types (scran-style)
        mean_var = np.nanmean(variances, axis=0)  # [n_genes]

        # Gene-level mean expression (for loess normalization)
        gene_means = X_sub.mean(axis=0)  # [n_genes]

        # Loess normalization: fit mean-variance trend, divide out
        from statsmodels.nonparametric.smoothers_lowess import lowess
        finite_mask = np.isfinite(mean_var) & (gene_means > 0)
        loess_result = lowess(
            np.log1p(mean_var[finite_mask]),
            np.log1p(gene_means[finite_mask]),
            frac=0.3,
            return_sorted=False,
        )
        # Map back: expected log-variance for each gene
        expected_log_var = np.full(len(mean_var), np.nan)
        expected_log_var[finite_mask] = loess_result
        normalized_var = np.log1p(mean_var) - expected_log_var
        normalized_var = np.nan_to_num(normalized_var, nan=-np.inf)

        # Select top n_hvg by normalized within-type variance
        top_idx = np.argsort(normalized_var)[-n_hvg:]
        selected_genes = set(gene_names_filtered[top_idx])

        hvg_mask = adata.var_names.isin(selected_genes)
        del X_sub
        logger.info(
            f"Blocked HVG: {hvg_mask.sum():,} genes selected "
            f"(from {gene_mask.sum():,} QC-passing, {n_types} types)"
        )

    # ── STEP 4: Force include L-R genes from CellChatDB ──
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

    # ── STEP 5: ONE column subset to final gene set ──
    # Combine: must pass min_cells AND (be HVG OR be L-R gene)
    final_mask = min_cells_mask & (hvg_mask | lr_mask)
    n_final = final_mask.sum()
    logger.info(f"Final gene set: {n_final:,} genes (from {adata.n_vars:,})")

    # This is the ONE copy — from 20K genes to ~4K genes
    logger.info("Subsetting to final gene set (this may take a few minutes)...")
    adata_subset = adata[:, final_mask].copy()
    del adata
    gc.collect()
    adata = adata_subset
    del adata_subset
    logger.info(f"After subset: {adata.shape[0]:,} cells x {adata.shape[1]:,} genes")

    if sparse.issparse(adata.X):
        logger.info(f"Subset nnz: {adata.X.nnz:,}")

    # Store gene metadata
    adata.var["highly_variable"] = True
    adata.var["lr_gene"] = adata.var_names.isin(lr_genes) if lr_genes else False

    # ── STEP 6: Normalize + Log transform ──
    # Store raw counts in .raw BEFORE normalization
    adata.raw = adata.copy()
    logger.info("Stored raw counts in adata.raw")

    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    logger.info(f"Normalized (target_sum={target_sum:.0e}) and log1p transformed")

    # For non-seurat_v3 flavors that don't pre-select HVGs, run HVG on normalized data.
    # "blocked" selects genes on raw counts (like seurat_v3), so skip this path.
    if hvg_flavor not in ("seurat_v3", "blocked"):
        logger.warning(
            "Non-seurat_v3 HVG running on already-subsetted data (%d genes). "
            "This path is not validated at scale. seurat_v3 or blocked is recommended.",
            adata.n_vars,
        )
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_hvg,
            flavor=hvg_flavor,
        )
        logger.info(f"HVG selection ({hvg_flavor}) on normalized data: {adata.var['highly_variable'].sum():,} genes")

        if lr_mask.any():
            adata.var["highly_variable"] = adata.var["highly_variable"] | adata.var["lr_gene"]

        adata = adata[:, adata.var["highly_variable"]].copy()

    logger.info(f"Final shape: {adata.shape[0]:,} cells x {adata.shape[1]:,} genes")

    return adata


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Merge and/or preprocess ROSMAP snRNA-seq h5ads",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- merge-and-preprocess ---
    mp = sub.add_parser("merge-and-preprocess",
                        help="Merge two h5ads then run full preprocessing")
    mp.add_argument("--dlpfc", type=str, required=True)
    mp.add_argument("--multiregion", type=str, required=True)
    mp.add_argument("--output", type=str, default="data/snRNAseq/adata_ROSMAP_merged.h5ad")
    mp.add_argument("--cellchatdb", type=str, default=str(CELLCHATDB_PATH))
    mp.add_argument("--n-hvg", type=int, default=4000)
    mp.add_argument("--target-sum", type=float, default=1e4)
    mp.add_argument("--min-cells-per-gene", type=int, default=10)
    mp.add_argument("--hvg-subsample", type=int, default=100_000)
    mp.add_argument("--hvg-flavor", default="blocked",
                    choices=["seurat_v3", "cell_ranger", "seurat", "blocked"])
    mp.add_argument("--hvg-per-type-n", type=int, default=5000,
                    help="Cells per type for blocked HVG (only used with --hvg-flavor blocked)")
    mp.add_argument("--cell-type-column", default="supercluster_name")
    mp.add_argument("--seed", type=int, default=42)
    mp.add_argument("--chunk-size", type=int, default=200_000)
    mp.add_argument("--keep-raw-merged", action="store_true",
                    help="Keep the intermediate 20K-gene merged h5ad")

    # --- preprocess-only ---
    pp = sub.add_parser("preprocess-only",
                        help="Run preprocessing on an existing merged h5ad")
    pp.add_argument("--input", type=str, required=True,
                    help="Path to merged h5ad with raw counts")
    pp.add_argument("--output", type=str, required=True)
    pp.add_argument("--cellchatdb", type=str, default=str(CELLCHATDB_PATH))
    pp.add_argument("--n-hvg", type=int, default=4000)
    pp.add_argument("--target-sum", type=float, default=1e4)
    pp.add_argument("--min-cells-per-gene", type=int, default=10)
    pp.add_argument("--hvg-subsample", type=int, default=100_000)
    pp.add_argument("--hvg-flavor", default="blocked",
                    choices=["seurat_v3", "cell_ranger", "seurat", "blocked"])
    pp.add_argument("--hvg-per-type-n", type=int, default=5000,
                    help="Cells per type for blocked HVG (only used with --hvg-flavor blocked)")
    pp.add_argument("--cell-type-column", default="supercluster_name")
    pp.add_argument("--seed", type=int, default=42)
    pp.add_argument("--splits-path", type=str, default=None,
                    help="Path to splits JSON for leak-free HVG selection")

    args = parser.parse_args()
    t0 = time.time()

    if args.command == "merge-and-preprocess":
        # Step 1: Merge
        raw_path = Path(args.output).with_suffix(".raw.h5ad")
        merge_h5ads(
            dlpfc_path=args.dlpfc,
            multiregion_path=args.multiregion,
            output_path=raw_path,
            chunk_size=args.chunk_size,
        )

        # Step 2: Preprocess
        logger.info("Starting preprocessing...")
        adata = preprocess_adata(
            adata_path=raw_path,
            cellchatdb_path=args.cellchatdb,
            n_hvg=args.n_hvg,
            target_sum=args.target_sum,
            min_cells_per_gene=args.min_cells_per_gene,
            hvg_flavor=args.hvg_flavor,
            hvg_subsample_n=args.hvg_subsample,
            hvg_per_type_n=args.hvg_per_type_n,
            copy=False,
            cell_type_column=args.cell_type_column,
            seed=args.seed,
        )

        logger.info("Saving preprocessed → %s", args.output)
        adata.write_h5ad(args.output)

        if not args.keep_raw_merged:
            raw_path.unlink(missing_ok=True)
            logger.info("Removed intermediate raw merged file")

    elif args.command == "preprocess-only":
        import json
        training_subject_ids = None
        if args.splits_path:
            with open(args.splits_path) as f:
                splits = json.load(f)
            training_subject_ids = splits.get("train_val_pool")
            if training_subject_ids:
                logger.info("HVG restricted to %d training subjects", len(training_subject_ids))

        adata = preprocess_adata(
            adata_path=args.input,
            cellchatdb_path=args.cellchatdb,
            n_hvg=args.n_hvg,
            target_sum=args.target_sum,
            min_cells_per_gene=args.min_cells_per_gene,
            hvg_flavor=args.hvg_flavor,
            hvg_subsample_n=args.hvg_subsample,
            hvg_per_type_n=args.hvg_per_type_n,
            copy=False,
            training_subject_ids=training_subject_ids,
            cell_type_column=args.cell_type_column,
            seed=args.seed,
        )

        logger.info("Saving → %s", args.output)
        adata.write_h5ad(args.output)

    elapsed = time.time() - t0
    logger.info("Done in %.0fs (%.1f min)", elapsed, elapsed / 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
