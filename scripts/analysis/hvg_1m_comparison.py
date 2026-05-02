"""Compute HVG on 1M-cell subsample using raw counts (seurat_v3) and compare with prior gene sets.

Run with::

    python -m scripts.analysis.hvg_1m_comparison \
        --raw-h5ad data/snRNAseq/adata_ROSMAP_merged.raw.h5ad \
        --old-gene-names-npy /path/to/archive/precomputed/gene_names.npy \
        --blocked-gene-names-npy data/precomputed/gene_names.npy \
        --output-dir outputs/pipeline

The ``--old-gene-names-npy`` flag is required: it is the .npy gene-names file
from a prior project archive (e.g. the 2026-03-31 pre-HVG archive). No default
is provided because the path is workstation-specific.
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import scanpy as sc

from src.data.preprocessing import CELLCHATDB_PATH, get_lr_genes_from_cellchatdb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-h5ad",
        type=Path,
        default=Path("data/snRNAseq/adata_ROSMAP_merged.raw.h5ad"),
        help="Path to raw merged ROSMAP h5ad.",
    )
    parser.add_argument(
        "--old-gene-names-npy",
        type=Path,
        required=True,
        help=(
            "Path to old project's gene_names.npy (e.g. from a pre-HVG archive). "
            "Workstation-specific — no default."
        ),
    )
    parser.add_argument(
        "--blocked-gene-names-npy",
        type=Path,
        default=Path("data/precomputed/gene_names.npy"),
        help="Path to current blocked-HVG gene_names.npy.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/pipeline"),
        help="Where to write hvg_1M_raw_seed42_4000.npy.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for the 1M subsample.",
    )
    parser.add_argument(
        "--n-cells",
        type=int,
        default=1_000_000,
        help="Number of cells to subsample for HVG computation.",
    )
    parser.add_argument(
        "--n-top-genes",
        type=int,
        default=4000,
        help="Top-N HVGs to keep (seurat_v3 flavor).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load full adata into memory, subsample, delete full copy ---
    print("Loading full raw adata into memory...", flush=True)
    adata = sc.read_h5ad(args.raw_h5ad)
    print(f"Loaded: {adata.shape}", flush=True)

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(adata.n_obs, size=args.n_cells, replace=False)
    sub = adata[idx].copy()
    del adata
    gc.collect()
    print(f"{args.n_cells:,}-cell subsample: {sub.shape}", flush=True)

    # --- seurat_v3 HVG on RAW counts (no normalization) ---
    print(
        f"Computing seurat_v3 HVG on raw counts ({args.n_cells:,} cells)...",
        flush=True,
    )
    sc.pp.highly_variable_genes(sub, n_top_genes=args.n_top_genes, flavor="seurat_v3")
    hvg_1m = sorted(sub.var_names[sub.var["highly_variable"]].tolist())
    print(f"Done: {len(hvg_1m)} HVGs", flush=True)

    out_npy = args.output_dir / f"hvg_1M_raw_seed{args.seed}_{args.n_top_genes}.npy"
    np.save(out_npy, hvg_1m)
    print(f"Saved to {out_npy}", flush=True)

    del sub
    gc.collect()

    # --- Load prior gene sets for comparison ---
    ccc = get_lr_genes_from_cellchatdb(CELLCHATDB_PATH)
    hvg_1m_set = set(hvg_1m)

    # allow_pickle=True required for object-dtype string arrays.
    old_genes = set(
        np.load(args.old_gene_names_npy, allow_pickle=True).tolist()
    )
    old_hvg = old_genes - ccc  # HVG-only portion of old set

    # allow_pickle=True required for object-dtype string arrays.
    blocked_genes = set(
        np.load(args.blocked_gene_names_npy, allow_pickle=True).tolist()
    )
    blocked_hvg = blocked_genes - ccc

    # --- Comparisons ---
    print(f"\n{'=' * 60}")
    print(f"1M raw HVG (seed {args.seed}):  {len(hvg_1m_set)} genes")
    print(f"Old 100K HVG (excl CCC):    {len(old_hvg)} genes")
    print(f"Blocked HVG (excl CCC):     {len(blocked_hvg)} genes")

    print("\n--- 1M raw vs Old 100K ---")
    print(f"  Overlap:      {len(hvg_1m_set & old_hvg)}")
    print(f"  Only in 1M:   {len(hvg_1m_set - old_hvg)}")
    print(f"  Only in old:  {len(old_hvg - hvg_1m_set)}")

    print("\n--- 1M raw vs Blocked ---")
    print(f"  Overlap:      {len(hvg_1m_set & blocked_hvg)}")
    print(f"  Only in 1M:   {len(hvg_1m_set - blocked_hvg)}")
    print(f"  Only in blk:  {len(blocked_hvg - hvg_1m_set)}")

    print("\n--- 1M raw ∪ CCC ---")
    hvg_1m_full = hvg_1m_set | ccc
    print(f"  Total genes:  {len(hvg_1m_full)}")
    print(f"  CCC overlap:  {len(hvg_1m_set & ccc)} (CCC genes already in HVG)")
    print(f"  CCC added:    {len(ccc - hvg_1m_set)} (new from CCC union)")

    # --- Key neuronal genes ---
    print("\nKey neuronal genes:")
    print(f"{'Gene':12} {'1M_raw':>7} {'Old100K':>8} {'Blocked':>8}")
    key_genes = [
        "CAMK2A", "CAMK2B", "SNAP25", "SYN1", "SYN2", "SYP",
        "RBFOX3", "SCN1A", "SCN8A", "NEUROD2", "NEUROD6",
        "BCL11B", "CUX2", "TBR1", "FEZF2", "LHX2",
        "CACNA1B", "CACNA1C", "GRIN3A", "DLGAP1",
    ]
    for g in key_genes:
        a = "Y" if g in hvg_1m_set else "N"
        b = "Y" if g in old_hvg else "N"
        c = "Y" if g in blocked_hvg else "N"
        print(f"  {g:12} {a:>7} {b:>8} {c:>8}")

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
