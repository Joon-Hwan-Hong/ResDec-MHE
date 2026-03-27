#!/usr/bin/env python3
"""Run LIANA+ cell-cell communication analysis per subject.

Runs li.mt.rank_aggregate() on each subject's cells separately, producing
per-subject parquet files with ligand-receptor interaction scores. These
are consumed by precompute_features.py to build CCC graph edges for the
HGT encoder branch.

The AnnData must have raw counts in adata.raw (set by merge_adata.py).
LIANA+ reads from adata.raw when use_raw=True.

Usage:
    # Full ROSMAP dataset (all subjects):
    uv run python scripts/run_liana.py \
        --config configs/default.yaml \
        --output-dir data/liana_cache/rosmap/

    # Resume after interruption (skips cached subjects):
    uv run python scripts/run_liana.py \
        --config configs/default.yaml \
        --output-dir data/liana_cache/rosmap/

    # Specific subjects only:
    uv run python scripts/run_liana.py \
        --config configs/default.yaml \
        --output-dir data/liana_cache/rosmap/ \
        --subjects R1234567 R7654321

    # With custom parameters:
    uv run python scripts/run_liana.py \
        --config configs/default.yaml \
        --output-dir data/liana_cache/rosmap/ \
        --n-perms 1000 \
        --min-cells 10 \
        --n-jobs 4

Requirements:
    uv pip install liana  (already installed: v1.7.1)

LIANA+ reference:
    Dimitrov et al., Nature Cell Biology (2024)
    https://github.com/saezlab/liana-py
    https://liana-py.readthedocs.io/
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def run_liana_single_subject(
    adata_subject,
    cell_type_column: str,
    resource_name: str,
    min_cells: int,
    n_perms: int,
    seed: int,
    n_jobs: int,
) -> pd.DataFrame:
    """Run LIANA+ rank_aggregate on a single subject's AnnData.

    Args:
        adata_subject: AnnData subset for one subject (preprocessed)
        cell_type_column: Column with cell type labels
        resource_name: L-R resource (e.g., "CellChatDB")
        min_cells: Minimum cells per cell type
        n_perms: Permutations for significance
        seed: Random seed
        n_jobs: Parallel jobs for LIANA internals

    Returns:
        DataFrame with LIANA+ results, or empty DataFrame on failure
    """
    import liana as li

    # Filter to cell types with enough cells
    ct_counts = adata_subject.obs[cell_type_column].value_counts()
    valid_cts = ct_counts[ct_counts >= min_cells].index.tolist()

    if len(valid_cts) < 2:
        logger.info("  Skipping: only %d cell types with >= %d cells", len(valid_cts), min_cells)
        return pd.DataFrame()

    # Filter to valid cell types (copy avoids view-to-copy promotion in LIANA)
    adata_filtered = adata_subject[
        adata_subject.obs[cell_type_column].isin(valid_cts)
    ].copy()

    logger.info(
        "  %d cells, %d cell types (of %d with >= %d cells)",
        adata_filtered.n_obs, len(valid_cts),
        adata_subject.obs[cell_type_column].nunique(), min_cells,
    )

    li.mt.rank_aggregate(
        adata_filtered,
        groupby=cell_type_column,
        resource_name=resource_name,
        expr_prop=0.1,
        min_cells=min_cells,
        use_raw=True,  # Use raw counts from adata.raw (set by merge_adata.py)
        n_perms=n_perms,
        seed=seed,
        n_jobs=n_jobs,
        verbose=False,
    )

    results = adata_filtered.uns["liana_res"].copy()
    logger.info("  Found %d interactions", len(results))
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run LIANA+ CCC analysis per subject on ROSMAP snRNA-seq data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory to save per-subject parquet files",
    )
    parser.add_argument(
        "--subjects", nargs="+", default=None,
        help="Specific subject IDs to process (default: all)",
    )
    parser.add_argument(
        "--resource", type=str, default="CellChatDB",
        help="LIANA L-R resource name (default: CellChatDB)",
    )
    parser.add_argument(
        "--min-cells", type=int, default=10,
        help="Minimum cells per cell type to include (default: 10)",
    )
    parser.add_argument(
        "--n-perms", type=int, default=100,
        help="Permutations for significance testing (default: 100, increase to 1000 for publication)",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=1,
        help="Parallel jobs for LIANA internals (default: 1)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--adata", type=str, default=None,
        help="Path to AnnData .h5ad file (overrides config adata_path). "
             "Use this to point to a preprocessed AnnData.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-run subjects that already have cached results",
    )
    args = parser.parse_args()

    # Load config
    from src.utils.config import load_config
    config = load_config(args.config)
    data_cfg = config.data

    subject_col = data_cfg.get("subject_column", "ROSMAP_IndividualID")
    cell_type_col = data_cfg.get("cell_type_column", "supercluster_name")

    # Load AnnData
    adata_path = args.adata or data_cfg.adata_path
    logger.info("Loading AnnData from %s ...", adata_path)
    t0 = time.time()

    import scanpy as sc
    adata = sc.read_h5ad(adata_path)
    logger.info(
        "Loaded AnnData: %d cells x %d genes in %.1fs",
        adata.n_obs, adata.n_vars, time.time() - t0,
    )

    # Check if data is preprocessed (normalized + log-transformed)
    # Heuristic: if max value > 50, data is likely raw counts
    max_val = adata.X[:1000].max() if hasattr(adata.X, 'max') else np.max(adata.X[:1000].toarray())
    if max_val > 50:
        logger.warning(
            "AnnData max expression value is %.1f — this may be raw counts. "
            "LIANA+ expects normalized, log-transformed data (use_raw=False). "
            "Consider running sc.pp.normalize_total() + sc.pp.log1p() first.",
            max_val,
        )

    # Determine subjects to process
    all_subjects = sorted(adata.obs[subject_col].unique().tolist())
    if args.subjects:
        subjects = [s for s in args.subjects if s in all_subjects]
        missing = set(args.subjects) - set(subjects)
        if missing:
            logger.warning("Subjects not found in AnnData: %s", missing)
    else:
        subjects = all_subjects
    logger.info("Will process %d subjects", len(subjects))

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing results
    if not args.overwrite:
        existing = set()
        for sid in subjects:
            # Check both naming conventions
            if (output_dir / f"liana_{sid}.parquet").exists() or \
               (output_dir / f"{sid}.parquet").exists():
                existing.add(sid)
        if existing:
            logger.info(
                "%d subjects already cached (use --overwrite to redo)",
                len(existing),
            )
            subjects = [s for s in subjects if s not in existing]
            logger.info("%d subjects remaining to process", len(subjects))

    if not subjects:
        logger.info("All subjects already processed. Nothing to do.")
        return

    # Process each subject
    t_start = time.time()
    n_success = 0
    n_skipped = 0
    n_failed = 0

    for i, subject_id in enumerate(subjects):
        logger.info(
            "[%d/%d] Processing subject %s ...",
            i + 1, len(subjects), subject_id,
        )
        t_subj = time.time()

        # Subset AnnData to this subject
        subject_mask = adata.obs[subject_col] == subject_id
        n_cells = subject_mask.sum()
        if n_cells == 0:
            logger.warning("  No cells found for subject %s, skipping", subject_id)
            n_skipped += 1
            continue

        adata_subject = adata[subject_mask].copy()

        try:
            results = run_liana_single_subject(
                adata_subject,
                cell_type_column=cell_type_col,
                resource_name=args.resource,
                min_cells=args.min_cells,
                n_perms=args.n_perms,
                seed=args.seed,
                n_jobs=args.n_jobs,
            )

            if results.empty:
                n_skipped += 1
            else:
                # Add subject_id column and save
                results["subject_id"] = subject_id

                # Use liana_{subject_id}.parquet naming (matches run_liana_per_subject)
                out_file = output_dir / f"liana_{subject_id}.parquet"
                results.to_parquet(out_file)
                n_success += 1

            elapsed = time.time() - t_subj
            logger.info("  Done in %.1fs", elapsed)

        except Exception as e:
            logger.error("  Failed: %s", e)
            n_failed += 1

        # Free memory
        del adata_subject

        # Progress summary every 50 subjects
        if (i + 1) % 50 == 0:
            elapsed_total = time.time() - t_start
            rate = (i + 1) / elapsed_total * 60
            remaining = (len(subjects) - i - 1) / rate if rate > 0 else 0
            logger.info(
                "Progress: %d/%d (%.1f subj/min, ~%.0f min remaining). "
                "Success: %d, Skipped: %d, Failed: %d",
                i + 1, len(subjects), rate, remaining,
                n_success, n_skipped, n_failed,
            )

    # Final summary
    elapsed_total = time.time() - t_start
    logger.info(
        "Completed in %.1fs. Success: %d, Skipped: %d, Failed: %d (of %d total)",
        elapsed_total, n_success, n_skipped, n_failed, len(subjects),
    )
    logger.info("Results saved to %s", output_dir)

    if n_failed > 0:
        logger.warning(
            "%d subjects failed. Re-run without --overwrite to retry only failed subjects.",
            n_failed,
        )
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
