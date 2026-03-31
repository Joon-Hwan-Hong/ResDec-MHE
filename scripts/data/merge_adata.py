#!/usr/bin/env python3
"""Merge ROSMAP h5ad datasets with HVG + CellChatDB gene selection.

1. Backed-read both h5ads to get gene names and subsample for HVG
2. Determine final gene set (~4800: 4000 HVGs + CellChatDB L-R genes)
3. Chunked-extract ONLY final genes from each h5ad (small intermediates)
4. On-disk concatenation
5. Load small merged result, add subject IDs, normalize, save

Usage:
    .venv/bin/python scripts/merge_adata.py \
        --dlpfc /path/to/ROSMAP_DLPFC_ABC_mapped.h5ad \
        --multiregion /path/to/ROSMAP_111.h5ad \
        --cellchatdb data/database/CellChatDB_human_interaction.csv \
        --output data/snRNAseq/adata_ROSMAP_merged.h5ad
"""

import argparse
import logging
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix, vstack

logger = logging.getLogger(__name__)


def get_cellchatdb_genes(path: str | Path) -> set[str]:
    db = pd.read_csv(path)
    genes = set()
    for col in ["ligand.symbol", "receptor.symbol"]:
        if col not in db.columns:
            col = col.replace(".", "_")
        if col not in db.columns:
            continue
        for val in db[col].dropna():
            for g in str(val).replace(",", "_").split("_"):
                g = g.strip()
                if g:
                    genes.add(g)
    return genes


def select_hvg_backed(h5ad_paths: list[str], common_genes: list[str],
                      n_hvg: int, chunk_size: int = 200_000) -> set[str]:
    """Run seurat_v3 HVG on ALL cells from backed h5ads."""
    chunks = []

    for path in h5ad_paths:
        adata = ad.read_h5ad(path, backed="r")
        n_cells = adata.n_obs

        # Get gene indices for common genes
        gene_locs = [adata.var_names.get_loc(g) for g in common_genes if g in adata.var_names]
        gene_locs_sorted = sorted(gene_locs)
        gene_order = np.argsort(np.argsort(gene_locs))

        # Read all cells in chunks, subset to common genes
        for start in range(0, n_cells, chunk_size):
            end = min(start + chunk_size, n_cells)
            X_chunk = adata.X[start:end][:, gene_locs_sorted]
            if not isinstance(X_chunk, csr_matrix):
                X_chunk = csr_matrix(X_chunk, dtype=np.float32)
            X_chunk = X_chunk[:, gene_order]
            chunks.append(X_chunk)
            if end % (chunk_size * 5) == 0 or end >= n_cells:
                logger.info("    HVG read: %d / %d cells from %s", end, n_cells, Path(path).name)

        adata.file.close()
        del adata
        logger.info("  Loaded %d cells from %s", n_cells, Path(path).name)

    X = vstack(chunks, format="csr")
    del chunks

    # Build minimal AnnData for HVG
    var_df = pd.DataFrame(index=common_genes)
    adata_sub = ad.AnnData(X=X, var=var_df)
    sc.pp.highly_variable_genes(adata_sub, n_top_genes=n_hvg, flavor="seurat_v3")
    hvg = set(adata_sub.var_names[adata_sub.var["highly_variable"]])
    del adata_sub, X
    return hvg


def extract_genes_backed(h5ad_path: str, gene_names: list[str], output_path: str,
                         obs_renames: dict | None = None,
                         obs_additions: dict | None = None,
                         chunk_size: int = 200_000) -> None:
    """Backed-read h5ad, extract only specified genes in chunks, save."""
    logger.info("  Extracting %d genes from %s...", len(gene_names), Path(h5ad_path).name)
    adata = ad.read_h5ad(h5ad_path, backed="r")
    n_cells = adata.n_obs

    # Gene index mapping
    gene_locs = [adata.var_names.get_loc(g) for g in gene_names if g in adata.var_names]
    gene_locs_sorted = sorted(gene_locs)
    gene_order = np.argsort(np.argsort(gene_locs))
    actual_genes = [gene_names[i] for i in range(len(gene_names)) if gene_names[i] in adata.var_names]

    # Obs
    obs_df = adata.obs.copy()
    if obs_renames:
        obs_df = obs_df.rename(columns=obs_renames)
    if obs_additions:
        for k, v in obs_additions.items():
            obs_df[k] = v

    # Var
    var_df = pd.DataFrame(index=actual_genes)

    # Chunks
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dlpfc", type=str, required=True)
    parser.add_argument("--multiregion", type=str, required=True)
    parser.add_argument("--cellchatdb", type=str, default="data/database/CellChatDB_human_interaction.csv")
    parser.add_argument("--output", type=str, default="data/snRNAseq/adata_ROSMAP_merged.h5ad")
    parser.add_argument("--n-hvg", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=200_000)
    args = parser.parse_args()

    t0 = time.time()
    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: Find common genes ----
    logger.info("Finding common genes (backed)...")
    dlpfc = ad.read_h5ad(args.dlpfc, backed="r")
    dlpfc_genes = set(dlpfc.var_names)
    dlpfc.file.close()

    mr = ad.read_h5ad(args.multiregion, backed="r")
    mr_genes = set(mr.var_names)
    mr.file.close()

    common_genes = sorted(dlpfc_genes & mr_genes)
    logger.info("Common genes: %d", len(common_genes))

    # ---- Step 2: HVG selection on subsampled backed data ----
    lr_genes = get_cellchatdb_genes(args.cellchatdb)
    logger.info("CellChatDB genes: %d", len(lr_genes))

    logger.info("HVG selection (seurat_v3, n=%d, all cells)...", args.n_hvg)
    hvg = select_hvg_backed(
        [args.dlpfc, args.multiregion], common_genes,
        n_hvg=args.n_hvg, chunk_size=args.chunk_size,
    )
    logger.info("HVGs: %d", len(hvg))

    # ---- Step 3: Final gene set ----
    lr_in_common = lr_genes & set(common_genes)
    lr_new = lr_in_common - hvg
    final_genes = sorted(hvg | lr_in_common)
    logger.info("Final genes: %d (HVG=%d, CellChatDB=%d, new from CellChatDB=%d)",
                len(final_genes), len(hvg), len(lr_in_common), len(lr_new))

    # ---- Step 4: Extract only final genes from each h5ad ----
    tmp_dlpfc = str(out_dir / "_tmp_dlpfc.h5ad")
    tmp_mr = str(out_dir / "_tmp_mr.h5ad")

    extract_genes_backed(
        args.dlpfc, final_genes, tmp_dlpfc,
        obs_renames={"individualID": "ROSMAP_IndividualID"},
        obs_additions={"BrainRegion": "PFC", "dataset": "DLPFC"},
        chunk_size=args.chunk_size,
    )

    extract_genes_backed(
        args.multiregion, final_genes, tmp_mr,
        obs_additions={"dataset": "111"},
        chunk_size=args.chunk_size,
    )

    # ---- Step 5: Concat on disk ----
    logger.info("Concatenating on disk...")
    ad.experimental.concat_on_disk(
        in_files=[tmp_dlpfc, tmp_mr],
        out_file=args.output,
        join="inner",
        merge="same",
    )
    Path(tmp_dlpfc).unlink(missing_ok=True)
    Path(tmp_mr).unlink(missing_ok=True)
    logger.info("Concat done, temp files cleaned.")

    # ---- Step 6: Load small merged, normalize, save ----
    logger.info("Loading merged for normalization...")
    adata = sc.read_h5ad(args.output)
    n_subjects = adata.obs["ROSMAP_IndividualID"].nunique() if "ROSMAP_IndividualID" in adata.obs.columns else 0
    logger.info("Merged: %d cells x %d genes, %d subjects", adata.n_obs, adata.n_vars, n_subjects)

    # Gene metadata
    adata.var["highly_variable"] = adata.var_names.isin(hvg)
    adata.var["lr_gene"] = adata.var_names.isin(lr_genes)

    # Raw + normalize
    logger.info("Storing raw in .raw, normalizing (target_sum=1e4 + log1p)...")
    adata.raw = adata.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Save
    logger.info("Saving...")
    adata.write_h5ad(args.output)

    elapsed = time.time() - t0
    logger.info("Done in %.0fs (%.1f hours). Final: %d cells x %d genes, %d subjects",
                elapsed, elapsed / 3600, adata.n_obs, adata.n_vars, n_subjects)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    main()
