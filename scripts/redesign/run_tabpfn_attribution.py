"""Run Captum FeatureAblation attribution on TabPFN-2.6 for a single fold.

Fits TabPFN on outer-train (412 subjects × top-2K features), then computes
per-subject FeatureAblation attributions on the first N val subjects, and
re-hydrates flat feature indices back to ``(cell_type, gene)``.

Outputs:
  - ``outputs/redesign/tabpfn_attribution_fold{k}.npz`` with arrays:
      * ``attributions`` [N, 2000]
      * ``mean_abs_attrib`` [2000]
      * ``val_subject_ids`` [N]
      * ``top_k_feature_indices`` [2000]
      * ``top_k_ct_ids`` [2000]
      * ``top_k_gene_ids`` [2000]
  - Cohort-level top-20 ``(ct, gene)`` pairs by mean |attribution| printed
    to stdout (and logged via ``logging``).

Usage:
    uv run python scripts/redesign/run_tabpfn_attribution.py \\
        CUDA_VISIBLE_DEVICES=1 ... --fold 0 --n-val 20 --device cuda
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

# Ensure the worktree root is on sys.path so `src.analysis.tabpfn_attribution`
# resolves from this worktree when invoked as a script (matches pattern used
# by scripts/redesign/compute_top_k_features.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _load_cell_type_names() -> list[str]:
    """Load human-readable cell type names (31 entries)."""
    from src.data.constants import CELL_TYPE_ORDER

    return list(CELL_TYPE_ORDER)


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    os.environ.setdefault(
        "TABPFN_MODEL_CACHE_DIR",
        "/host/milan/tank/Joon/__external_programs/tabpfn",
    )

    # Import here (after env setup, to avoid CUDA init races with logging).
    from src.analysis.tabpfn_attribution import attribute_tabpfn_fold

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    result = attribute_tabpfn_fold(
        fold_idx=args.fold,
        precomputed_dir=Path(args.precomputed_dir),
        meta_csv=Path(args.metadata_csv),
        splits_path=Path(args.splits_path),
        top_k_dir=Path(args.top_k_dir),
        n_val_subjects=args.n_val,
        device=args.device,
        method="feature_ablation",
        seed=args.seed,
    )
    elapsed = time.perf_counter() - start

    n_val = len(result["val_subject_ids"])
    attributions = result["attributions"]
    mean_abs = result["mean_abs_attrib"]
    schema = result["feature_schema"]
    logger.info(
        "fold %d: attributed %d val subjects in %.1fs (%.2fs/subject)",
        args.fold, n_val, elapsed, elapsed / max(n_val, 1),
    )

    # Save NPZ (numpy-friendly — keep feature_schema as 3 parallel arrays).
    out_path = output_dir / f"tabpfn_attribution_fold{args.fold}.npz"
    np.savez(
        out_path,
        attributions=attributions,
        mean_abs_attrib=mean_abs,
        val_subject_ids=np.array(result["val_subject_ids"], dtype=object),
        top_k_feature_indices=schema["feature_idx"].to_numpy(),
        top_k_ct_ids=schema["ct_id"].to_numpy(),
        top_k_gene_ids=schema["gene_id"].to_numpy(),
    )
    logger.info("Wrote %s", out_path)

    # Cohort-level top-20 by mean |attribution|
    ct_names = _load_cell_type_names()
    order = np.argsort(-mean_abs)[:20]
    logger.info("=" * 72)
    logger.info("Cohort top-20 (cell_type, gene_id) by mean |attribution|, fold %d", args.fold)
    logger.info("=" * 72)
    logger.info(
        "%3s  %-38s  %-8s  %10s",
        "rk", "cell_type", "gene_id", "mean|attr|",
    )
    for rank, j in enumerate(order, start=1):
        ct_id = int(schema.iloc[j]["ct_id"])
        gene_id = int(schema.iloc[j]["gene_id"])
        ct_name = ct_names[ct_id] if 0 <= ct_id < len(ct_names) else f"ct_{ct_id}"
        logger.info(
            "%3d  %-38s  %-8d  %10.4f",
            rank, ct_name[:38], gene_id, float(mean_abs[j]),
        )
    logger.info("=" * 72)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fold", type=int, default=0, help="Outer CV fold index (0..4)")
    p.add_argument("--n-val", type=int, default=20,
                   help="Number of val subjects to attribute (None = all).")
    p.add_argument("--device", type=str, default="cuda",
                   help="Torch device string. Default 'cuda' defers to CUDA_VISIBLE_DEVICES "
                        "(launch with CUDA_VISIBLE_DEVICES=1 to use GPU 1). Avoid hard-coding "
                        "':1' — it conflicts with CUDA_VISIBLE_DEVICES remapping (internal idx 0).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="outputs/redesign")
    p.add_argument("--splits-path", default="outputs/splits.json")
    p.add_argument("--precomputed-dir", default="data/precomputed")
    p.add_argument("--metadata-csv", default="data/metadata_ROSMAP/metadata.csv")
    p.add_argument("--top-k-dir", default="data/redesign")
    return p


if __name__ == "__main__":
    main(_build_parser().parse_args())
