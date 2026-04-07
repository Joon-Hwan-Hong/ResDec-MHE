"""Compute HVG on 1M-cell subsample using raw counts (seurat_v3) and compare with prior gene sets."""

import gc
import numpy as np
import scanpy as sc
from pathlib import Path

# Paths
RAW_H5AD = "data/snRNAseq/adata_ROSMAP_merged.raw.h5ad"
OLD_GENE_NAMES = "/host/milan/tank/Joon/proj_ml_snrna_archive_2026-03-31_pre-hvg/precomputed/gene_names.npy"
BLOCKED_GENE_NAMES = "data/precomputed/gene_names.npy"
OUTPUT_DIR = Path("outputs/pipeline")

# --- Load full adata into memory, subsample 1M, delete full copy ---
print("Loading full raw adata into memory...", flush=True)
adata = sc.read_h5ad(RAW_H5AD)
print(f"Loaded: {adata.shape}", flush=True)

rng = np.random.default_rng(42)
idx = rng.choice(adata.n_obs, size=1_000_000, replace=False)
sub = adata[idx].copy()
del adata
gc.collect()
print(f"1M subsample: {sub.shape}", flush=True)

# --- seurat_v3 HVG on RAW counts (no normalization) ---
print("Computing seurat_v3 HVG on raw counts (1M cells)...", flush=True)
sc.pp.highly_variable_genes(sub, n_top_genes=4000, flavor="seurat_v3")
hvg_1m = sorted(sub.var_names[sub.var["highly_variable"]].tolist())
print(f"Done: {len(hvg_1m)} HVGs", flush=True)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
np.save(OUTPUT_DIR / "hvg_1M_raw_seed42_4000.npy", hvg_1m)
print(f"Saved to {OUTPUT_DIR / 'hvg_1M_raw_seed42_4000.npy'}", flush=True)

del sub
gc.collect()

# --- Load prior gene sets for comparison ---
from src.data.preprocessing import get_lr_genes_from_cellchatdb, CELLCHATDB_PATH

ccc = get_lr_genes_from_cellchatdb(CELLCHATDB_PATH)
hvg_1m_set = set(hvg_1m)

old_genes = set(np.load(OLD_GENE_NAMES, allow_pickle=True).tolist())
old_hvg = old_genes - ccc  # HVG-only portion of old set

blocked_genes = set(np.load(BLOCKED_GENE_NAMES, allow_pickle=True).tolist())
blocked_hvg = blocked_genes - ccc

# --- Comparisons ---
print(f"\n{'='*60}")
print(f"1M raw HVG (seed 42):       {len(hvg_1m_set)} genes")
print(f"Old 100K HVG (excl CCC):    {len(old_hvg)} genes")
print(f"Blocked HVG (excl CCC):     {len(blocked_hvg)} genes")

print(f"\n--- 1M raw vs Old 100K ---")
print(f"  Overlap:      {len(hvg_1m_set & old_hvg)}")
print(f"  Only in 1M:   {len(hvg_1m_set - old_hvg)}")
print(f"  Only in old:  {len(old_hvg - hvg_1m_set)}")

print(f"\n--- 1M raw vs Blocked ---")
print(f"  Overlap:      {len(hvg_1m_set & blocked_hvg)}")
print(f"  Only in 1M:   {len(hvg_1m_set - blocked_hvg)}")
print(f"  Only in blk:  {len(blocked_hvg - hvg_1m_set)}")

print(f"\n--- 1M raw ∪ CCC ---")
hvg_1m_full = hvg_1m_set | ccc
print(f"  Total genes:  {len(hvg_1m_full)}")
print(f"  CCC overlap:  {len(hvg_1m_set & ccc)} (CCC genes already in HVG)")
print(f"  CCC added:    {len(ccc - hvg_1m_set)} (new from CCC union)")

# --- Key neuronal genes ---
print(f"\nKey neuronal genes:")
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
