"""Build TabPFN OOF + outer caches on a per-fold residualized target.

Wraps the canonical TabPFN per-fold callables (process_oof_fold + process_outer_fold)
exposed by Task 5's refactor of compute_oof.py / compute_outer.py. For each fold,
loads the residualized target NPZ produced by compute_residual_target.py and injects
it as the targets dict, so TabPFN trains on the residualized target rather than raw
cogn_global.

USAGE
-----
uv run python scripts/resdec_mhe/variants/build_tabpfn_cache_variant.py \\
    --variant-name gpath_only \\
    --residual-cache-dir outputs/canonical/variants/gpath_only/cache \\
    --out-dir outputs/canonical/variants/gpath_only/tabpfn_cache \\
    --folds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

# Allow `uv run python scripts/...` invocation by ensuring repo root is on
# sys.path before importing src.* (mirrors sibling scripts/resdec_mhe/).
_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))

from src.data.enriched_features import (  # noqa: E402
    FEATURE_SETS,
    FEATURE_SET_SIZES,
    load_enriched_features,
    load_pathology,
)
from src.data.feature_loaders import (  # noqa: E402
    load_flat_features,
    load_residualized_targets,
)
from src.data.splits import load_splits  # noqa: E402

from scripts.resdec_mhe.tabpfn._helpers import (  # noqa: E402
    TabPFNFoldArgs,
    resolve_tabpfn_cache_dir,
)
from scripts.resdec_mhe.tabpfn.compute_oof import process_oof_fold  # noqa: E402
from scripts.resdec_mhe.tabpfn.compute_outer import process_outer_fold  # noqa: E402

logger = logging.getLogger(__name__)


def _load_full_fold_targets(
    cache_dir: Path, fold_idx: int,
) -> dict[str, float]:
    """Load all per-subject residualized targets for a fold (NaN-skipped).

    Thin wrapper around load_residualized_targets that requests every subject
    in the NPZ — the variant builder needs the full cohort map because
    filter_usable_subjects (called inside process_*_fold) trims to fold splits.
    """
    npz = np.load(
        cache_dir / f"residual_target_fold{fold_idx}.npz", allow_pickle=True,
    )
    all_sids = npz["subject_ids"].tolist()
    return load_residualized_targets(
        subject_ids=all_sids, cache_dir=cache_dir, fold_idx=fold_idx,
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description="Build variant TabPFN OOF + outer caches on residualized target."
    )
    p.add_argument("--variant-name", required=True)
    p.add_argument("--residual-cache-dir", type=Path, required=True,
                   help="Output of compute_residual_target.py "
                        "(per-fold residual_target_fold{F}.npz files).")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Where to write variant tabpfn_oof_fold{F}.npz + "
                        "tabpfn_outer_fold{F}.npz files.")
    p.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])

    p.add_argument("--splits-path", type=Path,
                   default=_ROOT / "outputs/splits.json")
    p.add_argument("--precomputed-dir", type=Path,
                   default=_ROOT / "data/precomputed")
    p.add_argument("--metadata-csv", type=Path,
                   default=_ROOT / "data/metadata_ROSMAP/metadata.csv")
    p.add_argument("--top-k-dir", type=Path,
                   default=_ROOT / "data/canonical")

    p.add_argument("--top-k", type=int, default=2000)
    p.add_argument("--n-inner-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--feature-set", default="A", choices=list(FEATURE_SETS))
    p.add_argument("--ignore-pretraining-limits",
                   action="store_true", default=False)
    p.add_argument("--zscore", action="store_true", default=False)
    p.add_argument(
        "--device", default=None,
        help=("Device override for TabPFN. Default: 'cuda' if available else "
              "'cpu'. Use e.g. 'cuda:1' for explicit second-GPU pinning."),
    )

    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    resolve_tabpfn_cache_dir(
        default="/host/milan/tank/Joon/__external_programs/tabpfn",
    )

    splits = load_splits(str(args.splits_path))
    all_ids = sorted(
        {sid for fold in splits["folds"] for sid in fold["train"] + fold["val"]}
    )
    logger.info(
        "variant=%s: loading features for %d subjects (feature_set=%s, dim=%d)",
        args.variant_name, len(all_ids),
        args.feature_set, FEATURE_SET_SIZES[args.feature_set],
    )

    if args.feature_set == "A":
        features = load_flat_features(args.precomputed_dir, all_ids)
    else:
        pathology = None
        if args.feature_set == "A+C+E+P+R":
            pathology = load_pathology(args.metadata_csv, all_ids)
        features = load_enriched_features(
            args.precomputed_dir, all_ids, args.feature_set, pathology=pathology,
        )
    logger.info(
        "variant=%s: features ready: %d subjects",
        args.variant_name, len(features),
    )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("variant=%s: device=%s", args.variant_name, device)

    # Wrap argparse args in TabPFNFoldArgs for explicit-contract documentation
    # of the fields the per-fold callables read (top_k, feature_set, seed,
    # zscore, ignore_pretraining_limits, n_inner_folds).
    fold_args = TabPFNFoldArgs.from_argparse(args)
    for fold_idx in args.folds:
        targets = _load_full_fold_targets(args.residual_cache_dir, fold_idx)
        logger.info(
            "variant=%s fold=%d: loaded %d residualized targets",
            args.variant_name, fold_idx, len(targets),
        )
        fold_split = splits["folds"][fold_idx]

        process_oof_fold(
            fold_idx=fold_idx, fold_split=fold_split,
            features=features, targets=targets,
            args=fold_args, device=device,
            output_dir=args.out_dir, top_k_dir=args.top_k_dir,
        )
        process_outer_fold(
            fold_idx=fold_idx, fold_split=fold_split,
            features=features, targets=targets,
            args=fold_args, device=device,
            output_dir=args.out_dir, top_k_dir=args.top_k_dir,
        )
        logger.info(
            "variant=%s fold=%d: oof + outer caches written",
            args.variant_name, fold_idx,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
