"""Build RandomForest OOF + outer caches on a per-fold residualized target.

Mirrors the schema of TabPFN's compute_oof.py / compute_outer.py outputs so
the Lightning module loads RF residual-base predictions through the same
code path. Schema:

  tabpfn_oof_fold{F}.npz   (RF-on-residual OOF predictions)
    subject_ids:        train pool ids
    y_true:             residualized target on train
    y_tabpfn_oof:       5-fold inner-OOF RF mean prediction
    sigma_tabpfn_oof:   per-tree std of RF prediction (clipped at 1e-3)

  tabpfn_outer_fold{F}.npz (RF trained on full train pool, predicted on val)
    val_subject_ids:    val fold ids
    y_true:             residualized target on val
    y_tabpfn:           RF mean prediction on val
    sigma_tabpfn:       per-tree std on val (clipped at 1e-3)
    train_n:            train pool size

USAGE
-----
PYTHONPATH=. uv run python scripts/resdec_mhe/cogn_residual/build_rf_cache_cogn_residual.py \\
    --residual-cache-dir outputs/canonical/cogn_residual/gpath_only/cache \\
    --out-dir outputs/canonical/cogn_residual/gpath_only/rf_cache \\
    --folds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))

from src.baselines.rf_defaults import INNER_OOF_KFOLDS, RF_KWARGS, TOP_K  # noqa: E402
from src.data.feature_loaders import load_flat_features, load_residualized_targets  # noqa: E402
from src.data.splits import load_splits  # noqa: E402

from scripts.resdec_mhe.tabpfn._helpers import (  # noqa: E402
    filter_usable_subjects,
    top_k_filename,
)

logger = logging.getLogger(__name__)


def _rf_predict_with_sigma(
    rf: RandomForestRegressor, X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """RF mean + per-tree-disagreement sigma. Sigma clipped at 1e-3.

    Streams sums-of-squares per tree to avoid materializing the
    ``(n_estimators, n_samples)`` matrix.
    """
    mean = rf.predict(X).astype(np.float32)
    n_trees = len(rf.estimators_)
    sumsq = np.zeros(X.shape[0], dtype=np.float64)
    sum_p = np.zeros(X.shape[0], dtype=np.float64)
    for est in rf.estimators_:
        p = est.predict(X)
        sum_p += p
        sumsq += p * p
    mean64 = sum_p / n_trees
    var = np.maximum(sumsq / n_trees - mean64 * mean64, 0.0)
    sigma = np.clip(np.sqrt(var), 1e-3, None).astype(np.float32)
    return mean, sigma


def _make_rf(rf_kwargs: dict) -> RandomForestRegressor:
    return RandomForestRegressor(**rf_kwargs)


def _process_oof_fold(
    fold_idx: int, fold_split: dict, features: dict, targets: dict[str, float],
    rf_kwargs: dict, n_inner_folds: int,
    top_k: int, top_k_dir: Path, output_dir: Path,
) -> None:
    train_ids, dropped_no_feat, dropped_no_tgt = filter_usable_subjects(
        fold_split["train"], features, targets,
    )
    logger.info(
        "fold %d OOF: train usable=%d/%d (dropped no_features=%d, no_target=%d)",
        fold_idx, len(train_ids), len(fold_split["train"]),
        dropped_no_feat, dropped_no_tgt,
    )
    top_k_path = top_k_filename(top_k_dir, top_k, fold_idx, "A")
    top_k_indices = json.loads(top_k_path.read_text())["indices"]

    X_full = np.stack([features[s] for s in train_ids])[:, top_k_indices].astype(np.float32)
    y_full = np.array([targets[s] for s in train_ids], dtype=np.float32)

    oof_mean = np.zeros_like(y_full)
    oof_std = np.ones_like(y_full)

    inner_kf = KFold(n_splits=n_inner_folds, shuffle=True,
                     random_state=rf_kwargs["random_state"])
    for inner_fold, (tr_idx, va_idx) in enumerate(inner_kf.split(X_full)):
        rf = _make_rf(rf_kwargs)
        rf.fit(X_full[tr_idx], y_full[tr_idx])
        mean, sigma = _rf_predict_with_sigma(rf, X_full[va_idx])
        oof_mean[va_idx] = mean
        oof_std[va_idx] = sigma
        logger.info("  inner %d: %d preds done", inner_fold, len(va_idx))

    out_path = output_dir / f"tabpfn_oof_fold{fold_idx}.npz"
    np.savez(
        out_path,
        subject_ids=np.array(train_ids, dtype=object),
        y_true=y_full,
        y_tabpfn_oof=oof_mean,
        sigma_tabpfn_oof=oof_std,
    )
    logger.info("  wrote %s", out_path)


def _process_outer_fold(
    fold_idx: int, fold_split: dict, features: dict, targets: dict[str, float],
    rf_kwargs: dict,
    top_k: int, top_k_dir: Path, output_dir: Path,
) -> None:
    train_ids, dropped_train_no_feat, dropped_train_no_tgt = filter_usable_subjects(
        fold_split["train"], features, targets,
    )
    val_ids, dropped_val_no_feat, dropped_val_no_tgt = filter_usable_subjects(
        fold_split["val"], features, targets,
    )
    logger.info(
        "fold %d OUTER: train=%d/%d (-%d -%d); val=%d/%d (-%d -%d)",
        fold_idx, len(train_ids), len(fold_split["train"]),
        dropped_train_no_feat, dropped_train_no_tgt,
        len(val_ids), len(fold_split["val"]),
        dropped_val_no_feat, dropped_val_no_tgt,
    )
    top_k_path = top_k_filename(top_k_dir, top_k, fold_idx, "A")
    top_k_indices = json.loads(top_k_path.read_text())["indices"]

    X_train = np.stack([features[s] for s in train_ids])[:, top_k_indices].astype(np.float32)
    y_train = np.array([targets[s] for s in train_ids], dtype=np.float32)
    X_val = np.stack([features[s] for s in val_ids])[:, top_k_indices].astype(np.float32)
    y_val = np.array([targets[s] for s in val_ids], dtype=np.float32)

    rf = _make_rf(rf_kwargs)
    rf.fit(X_train, y_train)
    mean, sigma = _rf_predict_with_sigma(rf, X_val)

    out_path = output_dir / f"tabpfn_outer_fold{fold_idx}.npz"
    np.savez(
        out_path,
        val_subject_ids=np.array(val_ids, dtype=object),
        y_true=y_val,
        y_tabpfn=mean,
        sigma_tabpfn=sigma,
        train_n=len(train_ids),
    )
    logger.info("  wrote %s", out_path)


def main() -> int:
    p = argparse.ArgumentParser(description="Build RF residual-base cache (TabPFN-cache schema).")
    p.add_argument("--residual-cache-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--splits-path", type=Path, default=_ROOT / "outputs/splits.json")
    p.add_argument("--precomputed-dir", type=Path, default=_ROOT / "data/precomputed")
    p.add_argument("--top-k-dir", type=Path, default=_ROOT / "data/canonical")
    p.add_argument("--top-k", type=int, default=TOP_K)
    p.add_argument("--n-estimators", type=int, default=RF_KWARGS["n_estimators"])
    p.add_argument("--max-depth", type=int, default=RF_KWARGS["max_depth"])
    p.add_argument("--n-inner-folds", type=int, default=INNER_OOF_KFOLDS)
    p.add_argument("--seed", type=int, default=RF_KWARGS["random_state"])
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing per-fold cache files. Default refuses.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    splits = load_splits(str(args.splits_path))
    all_ids = sorted({sid for fold in splits["folds"] for sid in fold["train"] + fold["val"]})
    logger.info("loading flat features for %d subjects", len(all_ids))
    features = load_flat_features(args.precomputed_dir, all_ids)

    rf_kwargs = {
        **RF_KWARGS,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "random_state": args.seed,
    }

    for fold_idx in args.folds:
        oof_path = args.out_dir / f"tabpfn_oof_fold{fold_idx}.npz"
        outer_path = args.out_dir / f"tabpfn_outer_fold{fold_idx}.npz"
        if (oof_path.is_file() or outer_path.is_file()) and not args.force:
            raise FileExistsError(
                f"Refusing to overwrite existing cache: {oof_path} or {outer_path}. "
                "Pass --force to clobber, or remove the files first."
            )
        npz = np.load(args.residual_cache_dir / f"residual_target_fold{fold_idx}.npz", allow_pickle=True)
        targets = load_residualized_targets(
            subject_ids=npz["subject_ids"].tolist(),
            cache_dir=args.residual_cache_dir, fold_idx=fold_idx,
        )
        fold_split = splits["folds"][fold_idx]
        _process_oof_fold(
            fold_idx=fold_idx, fold_split=fold_split, features=features,
            targets=targets, rf_kwargs=rf_kwargs,
            n_inner_folds=args.n_inner_folds,
            top_k=args.top_k, top_k_dir=args.top_k_dir, output_dir=args.out_dir,
        )
        _process_outer_fold(
            fold_idx=fold_idx, fold_split=fold_split, features=features,
            targets=targets, rf_kwargs=rf_kwargs,
            top_k=args.top_k, top_k_dir=args.top_k_dir, output_dir=args.out_dir,
        )
        logger.info("fold %d: oof + outer caches written", fold_idx)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
