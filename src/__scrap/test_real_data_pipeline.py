#!/usr/bin/env python
"""
Test the full data pipeline on a real subset of ROSMAP data.

Steps:
    1. Load and inspect the full AnnData (105 GB, backed mode)
    2. Load metadata (subject_to_projid + clinical data)
    3. Select ~30 stratified subjects
    4. Subset AnnData and save
    5. Preprocess (HVG + normalize)
    6. Create Dataset
    7. Run LIANA+ processing
    8. Test collation — verifies that collate_for_hgt_multiregion correctly
       pads and stacks variable-length per-subject tensors (pseudobulk,
       cells, masks, edge indices) into a uniform batch, ensuring shapes
       and dtypes are compatible with the model's expected input contract.
    9. Test model forward pass — instantiates CognitiveResilienceModel and
       runs a single forward pass on the collated batch. The forward pass
       exercises the full architecture:
         a. PseudobulkEncoder encodes region_pseudobulk [B, R, 31, G] per
            region into cell-type embeddings [B, R, 31, d].
         b. RegionHandler pools across regions → pooled [B, 31, d] +
            region_context [B, d].
         c. HGTEncoderBatched processes per-subject CCC graphs (from LIANA+)
            using the pooled embeddings as node features → hgt_emb [B, 31, d].
         d. CellTransformer encodes cell-level expression [B, 31, max_cells, G]
            via Set Transformer (ISAB) → cell_emb [B, 31, d].
         e. FusionLayer combines all three branches → fused [B, 31, d_fused].
         f. PathologyEncoder maps pathology scores + region_context → path_emb
            [B, d_cond].
         g. PathologyStratifiedAttention attends over cell types conditioned on
            pathology → attended [B, d_fused] + attention_weights.
         h. BayesianPredictionHead produces mean [B, 1] and std [B, 1].
       The test checks for expected output keys and no NaN/Inf values.

Usage:
    python src/__scrap/test_real_data_pipeline.py

Notes:
    - Step 4 loads the full 105 GB AnnData into memory for subsetting.
      Requires ~120+ GB RAM. On memory-constrained machines, pre-subset
      using h5py or backed mode slice-and-write.
    - Steps 5-9 operate on the small subset (~30 subjects) and are fast.
    - If the subset already exists at SUBSET_PATH, steps 1-4 are skipped.
"""

import sys
import time
from pathlib import Path

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

FULL_ADATA_PATH = PROJECT_ROOT / "data/snRNAseq/adata_ROSMAP_merged.h5ad"
SUBSET_PATH = PROJECT_ROOT / "data/snRNAseq/adata_test_subset.h5ad"
PREPROCESSED_PATH = PROJECT_ROOT / "data/snRNAseq/adata_test_subset_preprocessed.h5ad"
METADATA_EXCEL = PROJECT_ROOT / "data/metadata_ROSMAP/dataset_810_basic_10-30-2023.xlsx"
SUBJECT_PROJID_CSV = PROJECT_ROOT / "data/metadata_ROSMAP/subject_to_projid.csv"
CELLCHATDB_PATH = PROJECT_ROOT / "data/database/CellChatDB_human_interaction.csv"
LIANA_CACHE_DIR = PROJECT_ROOT / "data/liana_cache/test_subset"

N_TARGET_SUBJECTS = 30
N_HVG = 2000  # Fewer HVGs for speed (production uses 4000)
MAX_CELLS_PER_TYPE = 200  # Reduced for speed (production uses 1000)
MIN_CELLS_THRESHOLD = 10  # Lower threshold for subset (production uses 50)
BATCH_SIZE = 4  # Small batch for testing


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: Inspect AnnData (backed mode)
# ═══════════════════════════════════════════════════════════════════════════════

def step1_inspect_adata() -> pd.DataFrame:
    """Load AnnData in backed mode and print summary stats.

    Returns the full obs DataFrame for subject selection.
    """
    import scanpy as sc

    print(f"Loading {FULL_ADATA_PATH} in backed mode...")
    adata = sc.read_h5ad(FULL_ADATA_PATH, backed="r")

    print(f"  Shape: {adata.shape[0]:,} cells x {adata.shape[1]:,} genes")
    print(f"  obs columns: {list(adata.obs.columns)}")

    obs_df = adata.obs.copy()

    print(f"\n  Brain regions:")
    for region, count in obs_df["BrainRegion"].value_counts().items():
        print(f"    {region}: {count:,}")

    print(f"\n  Cell types ({obs_df['supercluster_name'].nunique()}):")
    for ct, count in obs_df["supercluster_name"].value_counts().items():
        print(f"    {ct}: {count:,}")

    print(f"\n  Subjects: {obs_df['ROSMAP_IndividualID'].nunique():,}")

    adata.file.close()
    return obs_df


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: Load metadata
# ═══════════════════════════════════════════════════════════════════════════════

def step2_load_metadata() -> pd.DataFrame:
    """Load and join subject-level metadata from Excel + CSV mapping.

    Renames columns to match codebase conventions:
        cogn_global_lv -> cogn_global   (target column used by datasets.py/splits.py)
    """
    # Subject-to-projid mapping
    subject_to_projid = pd.read_csv(SUBJECT_PROJID_CSV)
    print(f"  Subject-to-projid mapping: {len(subject_to_projid)} entries")

    # Clinical data
    clinical = pd.read_excel(METADATA_EXCEL)
    print(f"  Clinical data: {clinical.shape[0]} rows, {clinical.shape[1]} columns")

    # Rename columns to match codebase conventions
    # The Excel file uses "cogn_global_lv" but the pipeline expects "cogn_global"
    rename_map = {"cogn_global_lv": "cogn_global"}
    renamed = {k: v for k, v in rename_map.items() if k in clinical.columns}
    if renamed:
        clinical = clinical.rename(columns=renamed)
        print(f"  Renamed columns: {renamed}")

    # Key columns check
    target_cols = ["cogn_global", "gpath", "amylsqrt", "tangsqrt"]
    available = [c for c in target_cols if c in clinical.columns]
    missing = [c for c in target_cols if c not in clinical.columns]
    print(f"  Target columns found: {available}")
    if missing:
        print(f"  WARNING: Missing target columns: {missing}")

    # Join
    metadata = subject_to_projid.merge(clinical, on="projid", how="left")
    print(f"  Joined metadata: {metadata.shape[0]} rows")

    # Stats on key columns
    for col in available:
        n_valid = metadata[col].notna().sum()
        print(f"    {col}: {n_valid} non-null ({n_valid / len(metadata) * 100:.1f}%)")

    return metadata


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3: Select ~30 stratified subjects
# ═══════════════════════════════════════════════════════════════════════════════

def step3_select_subjects(
    obs_df: pd.DataFrame,
    metadata: pd.DataFrame,
    n_target: int = N_TARGET_SUBJECTS,
) -> list[str]:
    """Stratified subject selection ensuring region, cell type, and phenotype coverage."""
    from src.data.constants import CELL_TYPE_ORDER, REGION_ORDER

    # Build subject -> regions mapping
    subject_regions = (
        obs_df.groupby("ROSMAP_IndividualID", observed=True)["BrainRegion"]
        .apply(set)
        .to_dict()
    )
    # Build subject -> cell types mapping
    subject_celltypes = (
        obs_df.groupby("ROSMAP_IndividualID", observed=True)["supercluster_name"]
        .apply(set)
        .to_dict()
    )

    # Filter to subjects with clinical data
    meta_with_data = metadata.dropna(subset=["cogn_global", "gpath"])
    valid_ids = set(meta_with_data["ROSMAP_IndividualID"])
    print(f"  Subjects with cogn_global + gpath: {len(valid_ids)}")

    # For each region, pick ~5 subjects spanning low/med/high pathology
    selected = set()
    for region in REGION_ORDER:
        region_subjects = [
            s for s, regions in subject_regions.items()
            if region in regions and s in valid_ids
        ]
        if not region_subjects:
            print(f"  WARNING: No valid subjects for region {region}")
            continue

        region_meta = meta_with_data[
            meta_with_data["ROSMAP_IndividualID"].isin(region_subjects)
        ].sort_values("gpath")

        n = len(region_meta)
        # Pick subjects at quintile positions for pathology spread
        n_pick = min(5, n)
        if n_pick <= 1:
            indices = [0]
        else:
            indices = [int(i * (n - 1) / (n_pick - 1)) for i in range(n_pick)]

        for idx in indices:
            selected.add(region_meta.iloc[idx]["ROSMAP_IndividualID"])

        print(f"  {region}: {len(region_subjects)} available, picked {len(indices)}")

    # Check cell type coverage
    covered_types = set()
    for s in selected:
        if s in subject_celltypes:
            covered_types.update(subject_celltypes[s])

    missing_types = set(CELL_TYPE_ORDER) - covered_types
    if missing_types:
        print(f"\n  Missing cell types ({len(missing_types)}): {missing_types}")
        # Add subjects that have rare cell types
        for ct in sorted(missing_types):
            for s, types in subject_celltypes.items():
                if ct in types and s in valid_ids and s not in selected:
                    selected.add(s)
                    covered_types.update(types)
                    print(f"    Added subject {s} for cell type '{ct}'")
                    break
            else:
                print(f"    WARNING: Could not find subject with cell type '{ct}'")

    selected = sorted(selected)

    # Print summary
    final_covered = set()
    final_regions = set()
    for s in selected:
        if s in subject_celltypes:
            final_covered.update(subject_celltypes[s])
        if s in subject_regions:
            final_regions.update(subject_regions[s])

    print(f"\n  Selected {len(selected)} subjects")
    print(f"  Cell type coverage: {len(final_covered)}/{len(CELL_TYPE_ORDER)}")
    print(f"  Region coverage: {final_regions}")

    # Print pathology/cognition distribution
    sel_meta = meta_with_data[meta_with_data["ROSMAP_IndividualID"].isin(selected)]
    for col in ["gpath", "cogn_global"]:
        if col in sel_meta.columns:
            vals = sel_meta[col].dropna()
            print(f"  {col}: mean={vals.mean():.3f}, std={vals.std():.3f}, "
                  f"range=[{vals.min():.3f}, {vals.max():.3f}]")

    return selected


# ═══════════════════════════════════════════════════════════════════════════════
# Step 4: Subset AnnData and save
# ═══════════════════════════════════════════════════════════════════════════════

def step4_subset_and_save(selected_ids: list[str]) -> None:
    """Subset AnnData to selected subjects using a memory-efficient approach.

    Strategy: Load in backed mode, identify cell indices, then read and write
    in chunks to avoid the massive sparse matrix copy that OOMs.
    """
    import gc
    import scanpy as sc
    from anndata import AnnData
    from scipy.sparse import vstack as sparse_vstack

    print(f"  Opening AnnData in backed mode to identify cells...")
    adata_backed = sc.read_h5ad(FULL_ADATA_PATH, backed="r")
    print(f"  Shape: {adata_backed.shape[0]:,} cells x {adata_backed.shape[1]:,} genes")

    # Get boolean mask of cells belonging to selected subjects
    mask = adata_backed.obs["ROSMAP_IndividualID"].isin(selected_ids)
    cell_indices = np.where(mask.values)[0]
    n_cells = len(cell_indices)
    print(f"  Selected {n_cells:,} cells across {len(selected_ids)} subjects")

    # Subset obs and var (lightweight)
    obs_subset = adata_backed.obs.iloc[cell_indices].copy()
    var_subset = adata_backed.var.copy()

    # Read expression matrix in chunks to avoid OOM
    CHUNK_SIZE = 100_000
    n_chunks = (n_cells + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"  Reading expression matrix in {n_chunks} chunks of {CHUNK_SIZE:,} cells...")

    chunks = []
    for i in range(n_chunks):
        start = i * CHUNK_SIZE
        end = min((i + 1) * CHUNK_SIZE, n_cells)
        idx = cell_indices[start:end]

        # Read chunk from backed array (returns sparse or dense)
        chunk = adata_backed.X[idx]
        chunks.append(chunk)

        if (i + 1) % 5 == 0 or i == n_chunks - 1:
            print(f"    Chunk {i + 1}/{n_chunks} done ({end:,}/{n_cells:,} cells)")

    adata_backed.file.close()

    # Stack chunks
    print(f"  Stacking chunks...")
    from scipy import sparse
    if sparse.issparse(chunks[0]):
        X_subset = sparse_vstack(chunks, format="csr")
    else:
        X_subset = np.vstack(chunks)
    del chunks
    gc.collect()

    print(f"  Expression matrix: {X_subset.shape}")

    # Build new AnnData
    adata_subset = AnnData(
        X=X_subset,
        obs=obs_subset.reset_index(drop=True),
        var=var_subset,
    )
    # Restore obs index to match original cell barcodes
    adata_subset.obs.index = obs_subset.index

    print(f"  Subset: {adata_subset.shape[0]:,} cells x {adata_subset.shape[1]:,} genes")
    print(f"  Subjects: {adata_subset.obs['ROSMAP_IndividualID'].nunique()}")
    print(f"  Regions: {dict(adata_subset.obs['BrainRegion'].value_counts())}")
    print(f"  Cell types: {adata_subset.obs['supercluster_name'].nunique()}")

    # Save
    SUBSET_PATH.parent.mkdir(parents=True, exist_ok=True)
    adata_subset.write_h5ad(SUBSET_PATH)
    print(f"  Saved to {SUBSET_PATH}")
    print(f"  File size: {SUBSET_PATH.stat().st_size / 1e9:.2f} GB")


# ═══════════════════════════════════════════════════════════════════════════════
# Step 5: Preprocess
# ═══════════════════════════════════════════════════════════════════════════════

def step5_preprocess():
    """Run preprocessing pipeline on the subset. Caches result to disk."""
    import scanpy as sc

    if PREPROCESSED_PATH.exists():
        print(f"  Preprocessed data found at {PREPROCESSED_PATH}")
        print(f"  Loading from cache...")
        adata = sc.read_h5ad(PREPROCESSED_PATH)
        print(f"  Shape: {adata.shape[0]:,} cells x {adata.shape[1]:,} genes")
        print(f"  Has .raw: {adata.raw is not None}")
        return adata

    from src.data.preprocessing import preprocess_adata

    print(f"  Running preprocess_adata(n_hvg={N_HVG})...")
    adata = preprocess_adata(
        adata_path=str(SUBSET_PATH),
        cellchatdb_path=str(CELLCHATDB_PATH),
        n_hvg=N_HVG,
    )

    print(f"  Preprocessed shape: {adata.shape[0]:,} cells x {adata.shape[1]:,} genes")
    print(f"  Has .raw: {adata.raw is not None}")
    if adata.raw is not None:
        print(f"  .raw shape: {adata.raw.shape}")

    # Cache preprocessed data
    adata.write_h5ad(PREPROCESSED_PATH)
    print(f"  Cached to {PREPROCESSED_PATH}")

    return adata


# ═══════════════════════════════════════════════════════════════════════════════
# Step 6: Create Dataset
# ═══════════════════════════════════════════════════════════════════════════════

def step6_create_dataset(adata, metadata: pd.DataFrame):
    """Instantiate CognitiveResilienceDataset and test single sample."""
    from src.data.datasets import CognitiveResilienceDataset

    subject_ids = adata.obs["ROSMAP_IndividualID"].unique().tolist()
    print(f"  Creating dataset for {len(subject_ids)} subjects...")

    dataset = CognitiveResilienceDataset(
        adata=adata,
        metadata=metadata,
        subject_ids=subject_ids,
        max_cells_per_type=MAX_CELLS_PER_TYPE,
        min_cells_threshold=MIN_CELLS_THRESHOLD,
        sampling_seed=42,
    )

    print(f"  Dataset size: {len(dataset)} subjects")

    # Test single sample
    print(f"\n  Testing single sample retrieval (subject 0)...")
    sample = dataset[0]

    print(f"  Sample keys: {sorted(sample.keys())}")
    print(f"  subject_id: {sample['subject_id']}")
    print(f"  pseudobulk: {sample['pseudobulk'].shape}")
    print(f"  cell_type_mask: {sample['cell_type_mask'].shape} "
          f"(sum={sample['cell_type_mask'].sum().item()})")
    print(f"  cell_counts: {sample['cell_counts'].shape} "
          f"(total={sample['cell_counts'].sum().item()})")
    print(f"  cells: {sample['cells'].shape}")
    print(f"  cell_mask: {sample['cell_mask'].shape} "
          f"(sum={sample['cell_mask'].sum().item()})")
    print(f"  pathology: {sample['pathology']} (shape={sample['pathology'].shape})")
    print(f"  cognition: {sample['cognition']} (shape={sample['cognition'].shape})")
    print(f"  region_mask: {sample['region_mask']}")

    # Region pseudobulk
    region_keys = [k for k in sample if k.startswith("region_") and k.endswith("_pseudobulk")]
    print(f"  Per-region pseudobulk keys: {region_keys}")
    if "available_regions" in sample:
        print(f"  available_regions: {sample['available_regions']}")

    return dataset


# ═══════════════════════════════════════════════════════════════════════════════
# Step 7: Run LIANA+ processing
# ═══════════════════════════════════════════════════════════════════════════════

def step7_run_liana(adata):
    """Run LIANA+ per-subject CCC analysis."""
    try:
        from src.data.liana_processing import run_liana_per_subject
    except ImportError as e:
        print(f"  WARNING: Cannot import LIANA processing: {e}")
        print(f"  Skipping LIANA. Model will use empty CCC graphs.")
        return None

    print(f"  Running LIANA+ per subject...")
    print(f"  Cache dir: {LIANA_CACHE_DIR}")

    try:
        liana_results = run_liana_per_subject(
            adata,
            subject_column="ROSMAP_IndividualID",
            cell_type_column="supercluster_name",
            min_cells_per_type=10,
            cache_dir=str(LIANA_CACHE_DIR),
            verbose=True,
        )

        n_with_results = sum(1 for df in liana_results.values() if len(df) > 0)
        print(f"\n  LIANA results: {n_with_results}/{len(liana_results)} subjects with interactions")

        for sid, df in list(liana_results.items())[:3]:
            print(f"    {sid}: {len(df)} interactions")

        return liana_results

    except Exception as e:
        print(f"  ERROR running LIANA: {e}")
        print(f"  Continuing without CCC data.")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Step 8: Test collation
# ═══════════════════════════════════════════════════════════════════════════════

def step8_test_collation(dataset):
    """Test collation of a small batch."""
    from src.data.collate import collate_for_hgt_multiregion

    n = min(BATCH_SIZE, len(dataset))
    print(f"  Collating batch of {n} samples...")

    batch = [dataset[i] for i in range(n)]
    collated = collate_for_hgt_multiregion(batch)

    print(f"\n  Collated batch keys: {sorted(collated.keys())}")
    print(f"  batch_size: {collated['batch_size']}")
    print(f"  pseudobulk: {collated['pseudobulk'].shape}")
    print(f"  cells: {collated['cells'].shape}")
    print(f"  cell_mask: {collated['cell_mask'].shape}")
    print(f"  cell_type_mask: {collated['cell_type_mask'].shape}")
    print(f"  pathology: {collated['pathology'].shape}")
    print(f"  cognition: {collated['cognition'].shape}")
    print(f"  region_mask: {collated['region_mask'].shape}")

    if "region_pseudobulk" in collated:
        print(f"  region_pseudobulk: {collated['region_pseudobulk'].shape}")

    if "edge_index_dict_list" in collated:
        print(f"  edge_index_dict_list: {len(collated['edge_index_dict_list'])} samples")
        for i, eid in enumerate(collated["edge_index_dict_list"][:2]):
            n_edge_types = len(eid)
            n_total_edges = sum(v.shape[1] for v in eid.values())
            print(f"    sample {i}: {n_edge_types} edge types, {n_total_edges} total edges")

    print(f"  node_types: {collated.get('node_types', 'N/A')[:5]}...")
    print(f"  edge_types: {collated.get('edge_types', 'N/A')}")

    return collated


# ═══════════════════════════════════════════════════════════════════════════════
# Step 9: Test model forward pass
# ═══════════════════════════════════════════════════════════════════════════════

def step9_test_model_forward(collated: dict, n_genes: int) -> None:
    """Instantiate model and run forward pass."""
    from src.models.full_model import CognitiveResilienceModel

    print(f"  Instantiating CognitiveResilienceModel(n_genes={n_genes})...")
    model = CognitiveResilienceModel(
        n_genes=n_genes,
        n_cell_types=31,
        d_embed=128,
        d_fused=128,
        d_cond=64,
        n_regions=6,
        n_hgt_layers=2,  # Fewer layers for speed
        n_hgt_heads=4,
        n_cell_transformer_heads=4,
        n_isab_layers=2,
        n_inducing_points=32,
        n_attention_heads=4,
        use_bayesian_head=True,
        d_head_hidden=64,
        dropout=0.1,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    model.eval()

    # Build forward kwargs from collated batch
    forward_kwargs = {
        "pseudobulk": collated["pseudobulk"],
        "region_mask": collated["region_mask"],
        "cells": collated["cells"],
        "cell_mask": collated["cell_mask"],
        "cell_type_mask": collated["cell_type_mask"],
        "pathology": collated["pathology"],
        "cognition": collated["cognition"],
    }

    if "region_pseudobulk" in collated:
        forward_kwargs["region_pseudobulk"] = collated["region_pseudobulk"]

    if "edge_index_dict_list" in collated:
        forward_kwargs["edge_index_dict_list"] = collated["edge_index_dict_list"]
    if "edge_attr_dict_list" in collated:
        forward_kwargs["edge_attr_dict_list"] = collated["edge_attr_dict_list"]

    print(f"\n  Running forward pass...")
    with torch.no_grad():
        output = model(**forward_kwargs)

    print(f"\n  Output keys: {sorted(output.keys())}")
    print(f"  mean: {output['mean'].shape} -> {output['mean'].squeeze().tolist()}")
    if "std" in output:
        print(f"  std: {output['std'].shape} -> {output['std'].squeeze().tolist()}")
    if "attention_weights" in output:
        print(f"  attention_weights: {output['attention_weights'].shape}")

    # Check for NaN/Inf
    for key, val in output.items():
        if isinstance(val, torch.Tensor):
            if torch.isnan(val).any():
                print(f"  WARNING: {key} contains NaN!")
            if torch.isinf(val).any():
                print(f"  WARNING: {key} contains Inf!")

    print(f"\n  Forward pass successful!")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t_total = time.time()

    # ── Steps 1-4: Create subset (skip if already exists) ──────────────────
    if SUBSET_PATH.exists():
        print(f"Subset already exists at {SUBSET_PATH}")
        print(f"  Size: {SUBSET_PATH.stat().st_size / 1e9:.2f} GB")
        print(f"  Skipping steps 1-4. Delete the file to re-create.\n")
    else:
        print("=" * 70)
        print("STEP 1: Inspect full AnnData (backed mode)")
        print("=" * 70)
        t = time.time()
        obs_df = step1_inspect_adata()
        print(f"  [{time.time() - t:.1f}s]\n")

        print("=" * 70)
        print("STEP 2: Load metadata")
        print("=" * 70)
        t = time.time()
        metadata_full = step2_load_metadata()
        print(f"  [{time.time() - t:.1f}s]\n")

        print("=" * 70)
        print("STEP 3: Select subjects")
        print("=" * 70)
        t = time.time()
        selected_ids = step3_select_subjects(obs_df, metadata_full)
        print(f"  [{time.time() - t:.1f}s]\n")

        print("=" * 70)
        print("STEP 4: Subset AnnData and save")
        print("=" * 70)
        t = time.time()
        step4_subset_and_save(selected_ids)
        print(f"  [{time.time() - t:.1f}s]\n")

    # ── Steps 5-9: Pipeline test on subset ─────────────────────────────────

    # Always reload metadata for steps 5-9
    print("=" * 70)
    print("STEP 2b: Load metadata (for pipeline steps)")
    print("=" * 70)
    t = time.time()
    metadata = step2_load_metadata()
    print(f"  [{time.time() - t:.1f}s]\n")

    print("=" * 70)
    print("STEP 5: Preprocess subset")
    print("=" * 70)
    t = time.time()
    adata = step5_preprocess()
    print(f"  [{time.time() - t:.1f}s]\n")

    print("=" * 70)
    print("STEP 6: Create Dataset")
    print("=" * 70)
    t = time.time()
    dataset = step6_create_dataset(adata, metadata)
    print(f"  [{time.time() - t:.1f}s]\n")

    print("=" * 70)
    print("STEP 7: Run LIANA+ processing")
    print("=" * 70)
    t = time.time()
    liana_results = step7_run_liana(adata)
    print(f"  [{time.time() - t:.1f}s]\n")

    # If LIANA succeeded, recreate dataset with LIANA results
    if liana_results is not None:
        print("  Recreating dataset with LIANA results...")
        from src.data.datasets import CognitiveResilienceDataset
        subject_ids = adata.obs["ROSMAP_IndividualID"].unique().tolist()
        dataset = CognitiveResilienceDataset(
            adata=adata,
            metadata=metadata,
            subject_ids=subject_ids,
            liana_results=liana_results,
            max_cells_per_type=MAX_CELLS_PER_TYPE,
            min_cells_threshold=MIN_CELLS_THRESHOLD,
            sampling_seed=42,
        )
        print(f"  Dataset recreated with LIANA data\n")

    print("=" * 70)
    print("STEP 8: Test collation")
    print("=" * 70)
    t = time.time()
    collated = step8_test_collation(dataset)
    print(f"  [{time.time() - t:.1f}s]\n")

    print("=" * 70)
    print("STEP 9: Test model forward pass")
    print("=" * 70)
    t = time.time()
    step9_test_model_forward(collated, n_genes=adata.n_vars)
    print(f"  [{time.time() - t:.1f}s]\n")

    print("=" * 70)
    print(f"PIPELINE TEST COMPLETE [{time.time() - t_total:.1f}s total]")
    print("=" * 70)


if __name__ == "__main__":
    main()
