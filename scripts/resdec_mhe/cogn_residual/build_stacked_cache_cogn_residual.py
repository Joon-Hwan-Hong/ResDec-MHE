"""Build stacked TabPFN+RF residual-base cache by averaging two existing caches.

  y_stacked     = (y_tabpfn + y_rf) / 2
  sigma_stacked = max(sigma_tabpfn, sigma_rf)

Output schema matches TabPFN cache schema. TabPFN sigma is a clipped IQR-based
predictive interval; RF sigma is per-tree disagreement; these are not
homogeneous quantities, so element-wise max is the conservative choice rather
than the independent-Gaussian closed form.

USAGE
-----
PYTHONPATH=. uv run python scripts/resdec_mhe/cogn_residual/build_stacked_cache_cogn_residual.py \\
    --tabpfn-cache-dir outputs/canonical/cogn_residual/gpath_only/tabpfn_cache \\
    --rf-cache-dir     outputs/canonical/cogn_residual/gpath_only/rf_cache \\
    --out-dir          outputs/canonical/cogn_residual/gpath_only/stacked_cache \\
    --folds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _stack_cache(
    tab_path: Path,
    rf_path: Path,
    out_path: Path,
    *,
    sids_key: str,
    y_key: str,
    sigma_key: str,
    extra_keys: tuple[str, ...] = (),
) -> None:
    """Average TabPFN + RF cache files; write stacked cache to ``out_path``.

    ``sids_key`` is the subject-ID array key (``subject_ids`` for OOF,
    ``val_subject_ids`` for outer). ``y_key`` is the prediction array key
    (``y_tabpfn_oof`` for OOF, ``y_tabpfn`` for outer). Same for ``sigma_key``.
    ``extra_keys`` are passthrough keys taken from the TabPFN cache only
    (e.g. ``train_n``).

    Reorders RF rows to match TabPFN row order if subject SETS are equal but
    arrays are permuted. Raises with a per-set diff if the sets differ.
    """
    tab = np.load(tab_path, allow_pickle=True)
    rf = np.load(rf_path, allow_pickle=True)
    tab_sids = tab[sids_key].tolist()
    rf_sids = rf[sids_key].tolist()
    if tab_sids != rf_sids:
        if set(tab_sids) != set(rf_sids):
            raise ValueError(
                f"Subject SET mismatch ({sids_key}): "
                f"tab={len(tab_sids)} vs rf={len(rf_sids)}; "
                f"tab\\rf={sorted(set(tab_sids) - set(rf_sids))[:10]}; "
                f"rf\\tab={sorted(set(rf_sids) - set(tab_sids))[:10]} "
                f"(showing first 10 of each)"
            )
        rf_idx = {s: i for i, s in enumerate(rf_sids)}
        order = np.array([rf_idx[s] for s in tab_sids])
        y_rf = rf[y_key][order]
        sigma_rf = rf[sigma_key][order]
        y_true_rf = rf["y_true"][order]
    else:
        y_rf = rf[y_key]
        sigma_rf = rf[sigma_key]
        y_true_rf = rf["y_true"]

    if not np.allclose(tab["y_true"], y_true_rf, atol=1e-5):
        raise ValueError(
            f"y_true mismatch ({sids_key}): "
            f"max abs diff={np.max(np.abs(tab['y_true'] - y_true_rf)):.6f}"
        )

    y_stacked = (
        (tab[y_key].astype(np.float32) + y_rf.astype(np.float32)) / 2.0
    ).astype(np.float32)
    # Conservative sigma: take element-wise max — see SIGMA NOTE in module
    # docstring. This avoids the unjustified independent-Gaussian assumption.
    sigma_stacked = np.maximum(
        tab[sigma_key].astype(np.float32),
        sigma_rf.astype(np.float32),
    ).astype(np.float32)

    out_dict = {
        sids_key: tab[sids_key],
        "y_true": tab["y_true"],
        y_key: y_stacked,
        sigma_key: sigma_stacked,
    }
    for k in extra_keys:
        out_dict[k] = tab[k]
    np.savez(out_path, **out_dict)
    logger.info("  wrote %s (%s)", out_path, sids_key)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tabpfn-cache-dir", type=Path, required=True)
    p.add_argument("--rf-cache-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing stacked cache files. Default refuses.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for fold_idx in args.folds:
        logger.info("fold %d:", fold_idx)
        oof_path = args.out_dir / f"tabpfn_oof_fold{fold_idx}.npz"
        outer_path = args.out_dir / f"tabpfn_outer_fold{fold_idx}.npz"
        if (oof_path.is_file() or outer_path.is_file()) and not args.force:
            raise FileExistsError(
                f"Refusing to overwrite existing cache: {oof_path} or {outer_path}. "
                "Pass --force to clobber, or remove the files first."
            )
        _stack_cache(
            args.tabpfn_cache_dir / f"tabpfn_oof_fold{fold_idx}.npz",
            args.rf_cache_dir / f"tabpfn_oof_fold{fold_idx}.npz",
            oof_path,
            sids_key="subject_ids",
            y_key="y_tabpfn_oof",
            sigma_key="sigma_tabpfn_oof",
        )
        _stack_cache(
            args.tabpfn_cache_dir / f"tabpfn_outer_fold{fold_idx}.npz",
            args.rf_cache_dir / f"tabpfn_outer_fold{fold_idx}.npz",
            outer_path,
            sids_key="val_subject_ids",
            y_key="y_tabpfn",
            sigma_key="sigma_tabpfn",
            extra_keys=("train_n",),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
