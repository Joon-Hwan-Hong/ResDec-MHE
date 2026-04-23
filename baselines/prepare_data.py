"""Prepare input data for DL baseline methods (scPhase, MixMIL).

Creates method-specific .h5ad files from our preprocessed AnnData,
using the same 465 subjects and 4797 genes as our model.

Outputs:
    baselines/shared/scphase_input.h5ad  — cells x 4797 genes, sparse
        .obs: ROSMAP_IndividualID, cogn_global, batch, supercluster_name
    baselines/shared/mixmil_input.h5ad   — cells x 30 scVI latent dims
        .obs: ROSMAP_IndividualID, cogn_global, supercluster_name
        .obsm['X_scVI']: 30-dim scVI embeddings

Usage:
    uv run python baselines/prepare_data.py \
        --adata data/snRNAseq/adata_ROSMAP_preprocessed.h5ad \
        --splits outputs/splits.json \
        --metadata data/metadata_ROSMAP/metadata.csv \
        --output-dir baselines/shared/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def prepare_scphase_input(
    adata: sc.AnnData,
    metadata: pd.DataFrame,
    subject_ids: list[str],
    subject_col: str,
    target_col: str,
    output_path: Path,
    max_cells_per_subject: int | None = None,
    rng_seed: int = 42,
) -> None:
    """Create scPhase-compatible .h5ad.

    scPhase requires:
    - .X: expression matrix (sparse OK, 5000 HVGs recommended; we use our 4797)
    - .obs['sample_id']: subject identifier per cell
    - .obs['phenotype']: continuous target value per cell (same for all cells of a subject)
    - .obs['batch']: batch/cohort label (single cohort → 'ROSMAP' constant)
    - .obs['cell_type']: cell type annotations (for interpretability)

    max_cells_per_subject: if set, deterministically subsample each subject's cells to at most
        this many. Prevents OOM when per-subject cell counts are large; scPhase's own
        max_instances=20000 training-time cap makes pre-capping at or below 20000 effectively
        equivalent for training while dramatically shrinking the h5ad + in-memory data object.
    """
    logger.info("Preparing scPhase input...")

    # Subset to our subjects
    mask = adata.obs[subject_col].isin(subject_ids)
    logger.info(f"Subsetting to {mask.sum():,} cells from {len(subject_ids)} subjects")
    adata_sub = adata[mask].copy()

    # Add target column (cogn_global) — map from metadata to each cell
    meta_target = metadata.set_index(subject_col)[target_col].to_dict()
    adata_sub.obs["phenotype"] = adata_sub.obs[subject_col].map(meta_target).astype(np.float32)

    # Drop cells where target is missing
    valid_mask = adata_sub.obs["phenotype"].notna()
    if (~valid_mask).sum() > 0:
        logger.warning(f"Dropping {(~valid_mask).sum():,} cells with missing target")
        adata_sub = adata_sub[valid_mask].copy()

    # Per-subject cell cap (deterministic subsample). Writes a reproducible subset sized for
    # memory fit at our ROSMAP scale; matches scPhase's own training-time max_instances cap.
    if max_cells_per_subject is not None:
        logger.info(
            f"Applying per-subject cell cap: max {max_cells_per_subject} cells/subject "
            f"(rng_seed={rng_seed})"
        )
        rng = np.random.default_rng(rng_seed)
        subj_labels = adata_sub.obs[subject_col].to_numpy()
        keep_chunks: list[np.ndarray] = []
        for sid in subject_ids:
            subj_indices = np.where(subj_labels == sid)[0]
            if subj_indices.size > max_cells_per_subject:
                chosen = rng.choice(subj_indices, size=max_cells_per_subject, replace=False)
                keep_chunks.append(chosen)
            else:
                keep_chunks.append(subj_indices)
        keep_indices = np.sort(np.concatenate(keep_chunks)) if keep_chunks else np.array([], dtype=int)
        n_before = adata_sub.shape[0]
        adata_sub = adata_sub[keep_indices].copy()
        n_after = adata_sub.shape[0]
        logger.info(
            f"Per-subject cap: {n_before:,} cells → {n_after:,} cells "
            f"({n_after / max(n_before, 1) * 100:.1f}% retained)"
        )

    # Rename columns for scPhase conventions
    adata_sub.obs["sample_id"] = adata_sub.obs[subject_col]
    adata_sub.obs["batch"] = "ROSMAP"  # single cohort
    adata_sub.obs["cell_type"] = adata_sub.obs["supercluster_name"]

    # Keep only the columns scPhase needs
    adata_sub.obs = adata_sub.obs[["sample_id", "phenotype", "batch", "cell_type"]].copy()

    # Ensure sparse format for storage efficiency
    from scipy.sparse import issparse, csr_matrix
    if not issparse(adata_sub.X):
        logger.info("Converting to sparse format...")
        adata_sub.X = csr_matrix(adata_sub.X)

    logger.info(f"scPhase input: {adata_sub.shape[0]:,} cells x {adata_sub.shape[1]:,} genes")
    logger.info(f"Subjects: {adata_sub.obs['sample_id'].nunique()}")
    logger.info(f"Writing to {output_path}")
    adata_sub.write_h5ad(output_path)
    logger.info(f"Done. File size: {output_path.stat().st_size / 1e9:.1f} GB")


def prepare_mixmil_input(
    adata: sc.AnnData,
    metadata: pd.DataFrame,
    subject_ids: list[str],
    subject_col: str,
    target_col: str,
    output_path: Path,
    n_latent: int = 30,
    max_epochs: int = 50,
    num_workers: int = 8,
    devices: int = 1,
) -> None:
    """Create MixMIL-compatible .h5ad with scVI embeddings.

    MixMIL requires:
    - Precomputed cell embeddings (paper used scVI with 30 latent factors)
    - Patient/bag labels
    - .obsm['X_scVI']: 30-dim embeddings
    """
    import scvi

    logger.info("Preparing MixMIL input (scVI embeddings)...")

    # Subset to our subjects
    mask = adata.obs[subject_col].isin(subject_ids)
    logger.info(f"Subsetting to {mask.sum():,} cells from {len(subject_ids)} subjects")
    adata_sub = adata[mask].copy()

    # Add metadata
    meta_target = metadata.set_index(subject_col)[target_col].to_dict()
    adata_sub.obs["phenotype"] = adata_sub.obs[subject_col].map(meta_target).astype(np.float32)
    adata_sub.obs["sample_id"] = adata_sub.obs[subject_col]
    adata_sub.obs["cell_type"] = adata_sub.obs["supercluster_name"]
    adata_sub.obs["batch"] = "ROSMAP"

    # Drop cells where target is missing
    valid_mask = adata_sub.obs["phenotype"].notna()
    if (~valid_mask).sum() > 0:
        logger.warning(f"Dropping {(~valid_mask).sum():,} cells with missing target")
        adata_sub = adata_sub[valid_mask].copy()

    # scVI expects raw counts, not normalized expression.
    # The preprocessed h5ad stores normalized data in .X and raw counts in .raw.
    if adata_sub.raw is not None:
        logger.info("Swapping .X with .raw counts for scVI (same 4797 genes, integer counts)")
        adata_sub.X = adata_sub.raw[adata_sub.obs_names, adata_sub.var_names].X.copy()
        adata_sub.raw = None  # avoid confusion
    else:
        sample_vals = adata_sub.X[:100].toarray() if hasattr(adata_sub.X, "toarray") else adata_sub.X[:100]
        if not np.allclose(sample_vals, np.round(sample_vals), atol=0.01, equal_nan=True):
            logger.warning(
                "Expression matrix appears normalized but .raw not available. "
                "scVI expects raw counts. Proceeding but results may be suboptimal."
            )

    # Enable Tensor Core acceleration (suppresses warning on Ada GPUs)
    torch.set_float32_matmul_precision("medium")

    # Set up and train scVI
    logger.info(f"Setting up scVI model (n_latent={n_latent})...")
    scvi.model.SCVI.setup_anndata(
        adata_sub,
        batch_key="batch" if "batch" in adata_sub.obs.columns else None,
    )

    model = scvi.model.SCVI(
        adata_sub,
        n_latent=n_latent,
        n_layers=2,
        gene_likelihood="nb",  # negative binomial for count data
    )

    train_kwargs = dict(
        max_epochs=max_epochs,
        early_stopping=True,
    )
    if devices > 1:
        train_kwargs.update(accelerator="gpu", devices=devices, strategy="ddp")
        logger.info(f"Training scVI for {max_epochs} epochs (num_workers={num_workers}, devices={devices}, strategy=ddp)...")
    else:
        logger.info(f"Training scVI for {max_epochs} epochs (num_workers={num_workers})...")
    model.train(**train_kwargs)

    # Extract latent representation
    logger.info("Extracting latent representations...")
    adata_sub.obsm["X_scVI"] = model.get_latent_representation()
    logger.info(f"scVI embeddings shape: {adata_sub.obsm['X_scVI'].shape}")

    # For MixMIL, we only need the embeddings + metadata, not the full expression
    # Create a lightweight AnnData with just embeddings
    import anndata
    adata_out = anndata.AnnData(
        X=adata_sub.obsm["X_scVI"],  # [n_cells, n_latent]
        obs=adata_sub.obs[["sample_id", "phenotype", "cell_type"]].copy(),
    )

    logger.info(f"MixMIL input: {adata_out.shape[0]:,} cells x {adata_out.shape[1]} latent dims")
    logger.info(f"Subjects: {adata_out.obs['sample_id'].nunique()}")
    logger.info(f"Writing to {output_path}")
    adata_out.write_h5ad(output_path)
    logger.info(f"Done. File size: {output_path.stat().st_size / 1e9:.2f} GB")


def main():
    parser = argparse.ArgumentParser(description="Prepare data for DL baseline methods")
    parser.add_argument("--adata", required=True, help="Path to preprocessed .h5ad file")
    parser.add_argument("--splits", required=True, help="Path to splits.json")
    parser.add_argument("--metadata", required=True, help="Path to metadata.csv")
    parser.add_argument("--output-dir", default="baselines/shared/", help="Output directory")
    parser.add_argument("--subject-col", default="ROSMAP_IndividualID")
    parser.add_argument("--target-col", default="cogn_global")
    parser.add_argument("--scvi-epochs", type=int, default=50, help="scVI training epochs")
    parser.add_argument("--scvi-latent", type=int, default=30, help="scVI latent dimensions")
    parser.add_argument("--scvi-num-workers", type=int, default=8, help="DataLoader workers for scVI")
    parser.add_argument("--scvi-devices", type=int, default=1, help="Number of GPUs for scVI (>1 uses DDP)")
    parser.add_argument(
        "--methods", nargs="+", default=["scphase", "mixmil"],
        choices=["scphase", "mixmil"],
        help="Which methods to prepare data for",
    )
    parser.add_argument(
        "--scphase-max-cells-per-subject", type=int, default=10000,
        help=(
            "Per-subject cell cap for scPhase (deterministic subsample, seed=42). "
            "Defaults to 10000 which is below scPhase's own training-time max_instances=20000 "
            "and keeps the h5ad + in-memory data object within our 251 GB RAM budget. "
            "Set to 0 to disable (warning: may OOM on large cohorts)."
        ),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load splits
    with open(args.splits) as f:
        splits = json.load(f)
    subject_ids = splits["train_val_pool"]
    logger.info(f"Loaded {len(subject_ids)} subjects from splits")

    # Load metadata
    metadata = pd.read_csv(args.metadata)
    logger.info(f"Loaded metadata: {len(metadata)} rows")

    # Load AnnData (this is the expensive step — 66 GB)
    logger.info(f"Loading AnnData from {args.adata} (this may take a few minutes)...")
    adata = sc.read_h5ad(args.adata)
    logger.info(f"Loaded: {adata.shape[0]:,} cells x {adata.shape[1]:,} genes")

    # Verify gene identity with our precomputed features
    gene_names_file = Path("data/precomputed/rosmap/gene_names.npy")
    if gene_names_file.exists():
        our_genes = list(np.load(gene_names_file, allow_pickle=True))
        h5ad_genes = list(adata.var_names)
        if our_genes != h5ad_genes:
            raise ValueError(
                f"Gene mismatch: h5ad has {len(h5ad_genes)} genes, our model uses {len(our_genes)}. "
                f"First difference at index {next(i for i, (a, b) in enumerate(zip(our_genes, h5ad_genes)) if a != b)}. "
                f"Baselines must use the exact same genes as our model for fair comparison."
            )
        logger.info(f"Gene identity verified: {len(our_genes)} genes match precomputed features")

    if "scphase" in args.methods:
        prepare_scphase_input(
            adata, metadata, subject_ids,
            args.subject_col, args.target_col,
            output_dir / "scphase_input.h5ad",
            max_cells_per_subject=(
                args.scphase_max_cells_per_subject
                if args.scphase_max_cells_per_subject > 0
                else None
            ),
        )

    if "mixmil" in args.methods:
        prepare_mixmil_input(
            adata, metadata, subject_ids,
            args.subject_col, args.target_col,
            output_dir / "mixmil_input.h5ad",
            n_latent=args.scvi_latent,
            max_epochs=args.scvi_epochs,
            num_workers=args.scvi_num_workers,
            devices=args.scvi_devices,
        )

    logger.info("All data preparation complete.")


if __name__ == "__main__":
    main()
