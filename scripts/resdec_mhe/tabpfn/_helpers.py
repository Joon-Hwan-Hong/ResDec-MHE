"""Shared helpers for the TabPFN compute scripts.

Single source of truth for:
- ``predict_with_sigma`` — TabPFN-2.6 quantile-based (median, std) extraction.
- ``build_regressor`` — TabPFNRegressor constructor with safety-override flag.
- ``filter_usable_subjects`` — drop subjects missing features or targets.
- ``top_k_filename`` — feature-set-aware top-K JSON filename.
- ``outer_output_filename`` — feature-set-aware outer-fold NPZ filename.
- ``build_xgb_regressor`` — XGBoost regressor with paper-canonical hyper-params.
- ``resolve_tabpfn_cache_dir`` — TABPFN_MODEL_CACHE_DIR resolution with a
  loud-fail when the env var is unset (instead of silently using a host-
  specific path that won't exist on other machines).

These functions previously lived as duplicate ``_predict_with_sigma`` /
``_build_regressor`` / inline subject-filter code blocks across
``compute_top_k_features.py`` / ``compute_oof.py`` / ``compute_outer.py``;
extracted per code-review CC4.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xgboost as xgb
from tabpfn import TabPFNRegressor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TabPFNFoldArgs:
    """Per-fold TabPFN-2.6 hyperparameters consumed by process_*_fold.

    Single source of truth for the duck-typed `args` object historically passed
    through subprocess.Popen-style argparse Namespaces. Build via
    `TabPFNFoldArgs.from_argparse(args)` at the entry point.

    Attributes
    ----------
    top_k : int
        Top-K HVG count used by both compute_top_k_features and the per-fold
        TabPFN feature selection. Default 2000 (canonical).
    feature_set : str
        Feature-set tag (canonical "A" = pseudobulk-only). Must match the tag
        used to produce the top-K JSON.
    seed : int
        RNG seed for KFold splits and TabPFN initializer.
    zscore : bool
        If True, per-feature z-score the TabPFN input fit on inner/outer-fold
        TRAIN ONLY (no leakage).
    ignore_pretraining_limits : bool
        Override TabPFN-2.6's 2000-feature safety check (only when explicitly
        testing >2000-feature behavior; distributional extrapolation risk).
    n_inner_folds : int
        Inner-OOF fold count for compute_oof's KFold (default 5).
    """
    top_k: int = 2000
    feature_set: str = "A"
    seed: int = 42
    zscore: bool = False
    ignore_pretraining_limits: bool = False
    n_inner_folds: int = 5

    @classmethod
    def from_argparse(cls, args) -> "TabPFNFoldArgs":
        """Build from an argparse Namespace, picking up only the relevant fields."""
        return cls(
            top_k=int(getattr(args, "top_k", 2000)),
            feature_set=str(getattr(args, "feature_set", "A")),
            seed=int(getattr(args, "seed", 42)),
            zscore=bool(getattr(args, "zscore", False)),
            ignore_pretraining_limits=bool(
                getattr(args, "ignore_pretraining_limits", False)
            ),
            n_inner_folds=int(getattr(args, "n_inner_folds", 5)),
        )


# Default XGBoost hyper-params used by compute_top_k_features. Centralised so
# ablation studies can override via CLI without re-deriving the canonical
# baseline values from inline literals.
DEFAULT_XGB_N_ESTIMATORS = 200
DEFAULT_XGB_MAX_DEPTH = 6
DEFAULT_XGB_LEARNING_RATE = 0.1


def predict_with_sigma(
    reg: TabPFNRegressor, X: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (median, std) per row via a SINGLE TabPFN.predict() call.

    Uses ``output_type="full"`` with ``quantiles=[0.16, 0.84]`` to retrieve
    the median and both quantiles in one forward pass; std = (q84 - q16) / 2.

    Falls back to a two-call path only if the dict schema differs from the
    tabpfn 7.1.1 contract (mean/median/mode/quantiles/criterion/logits).
    """
    result = reg.predict(X, output_type="full", quantiles=[0.16, 0.84])
    if isinstance(result, dict) and "median" in result and "quantiles" in result:
        median = np.asarray(result["median"])
        q = result["quantiles"]  # list[np.ndarray], length 2 -> [q16, q84]
        lower = np.asarray(q[0])
        upper = np.asarray(q[1])
        sigma = np.clip((upper - lower) / 2.0, a_min=1e-3, a_max=None)
        return median, sigma

    logger.warning(
        "TabPFN predict(output_type='full') returned unexpected schema; "
        "falling back to two-call path."
    )
    median = np.asarray(reg.predict(X, output_type="median"))
    q_arr = reg.predict(X, output_type="quantiles", quantiles=[0.16, 0.84])
    lower = np.asarray(q_arr[0])
    upper = np.asarray(q_arr[1])
    sigma = np.clip((upper - lower) / 2.0, a_min=1e-3, a_max=None)
    return median, sigma


def build_regressor(
    device: str, seed: int, ignore_pretraining_limits: bool
) -> TabPFNRegressor:
    """Construct a TabPFNRegressor with the ablation safety-override flag.

    ``ignore_pretraining_limits=True`` is a DELIBERATE override of TabPFN-2.6's
    2000-feature safety check. Use ONLY when deliberately testing
    >2000-feature behavior (e.g., top-k > 2000 ablations). Accepts the
    distributional-extrapolation risk; TabPFN's prior was trained on ≤2000
    features. Default MUST be False everywhere upstream.

    model_version is NOT a TabPFNRegressor constructor kwarg — it's set via
    tabpfn.settings (env var TABPFN_MODEL_VERSION or the settings default,
    which is ``ModelVersion.V2_6`` per ``tabpfn/settings.py:36``).
    """
    return TabPFNRegressor(
        device=device,
        random_state=seed,
        ignore_pretraining_limits=ignore_pretraining_limits,
    )


def build_xgb_regressor(
    *,
    n_estimators: int = DEFAULT_XGB_N_ESTIMATORS,
    max_depth: int = DEFAULT_XGB_MAX_DEPTH,
    learning_rate: float = DEFAULT_XGB_LEARNING_RATE,
    n_jobs: int = -1,
    seed: int = 42,
) -> xgb.XGBRegressor:
    """Construct the XGBoost regressor used by compute_top_k_features.

    Hyperparameters mirror the canonical paper baseline; pass overrides via
    CLI flags (compute_top_k_features.py --xgb-n-estimators / --xgb-max-depth /
    --xgb-learning-rate).
    """
    return xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_jobs=n_jobs,
        tree_method="hist",
        random_state=seed,
    )


def filter_usable_subjects(
    fold_subject_ids: list[str],
    features: dict,
    targets: dict,
) -> tuple[list[str], int, int]:
    """Drop subjects missing features or targets.

    Returns
    -------
    kept_ids : list[str]
        Subjects with both feature vectors and finite targets, preserving
        the input order.
    dropped_no_feat : int
    dropped_no_tgt : int
    """
    kept = [s for s in fold_subject_ids if s in features and s in targets]
    dropped_no_feat = sum(1 for s in fold_subject_ids if s not in features)
    dropped_no_tgt = sum(
        1 for s in fold_subject_ids if s in features and s not in targets
    )
    return kept, dropped_no_feat, dropped_no_tgt


def top_k_filename(
    top_k_dir: Path, top_k: int, fold_idx: int, feature_set: str
) -> Path:
    """Return the per-fold top-K JSON filename, branching on feature set.

    The ``A`` default keeps the original ``top_{k}_features_fold{f}.json``
    layout for backwards compatibility; non-A sets append ``_<feature_set>``.
    """
    if feature_set == "A":
        return top_k_dir / f"top_{top_k}_features_fold{fold_idx}.json"
    return top_k_dir / f"top_{top_k}_features_fold{fold_idx}_{feature_set}.json"


def outer_output_filename(
    output_dir: Path, fold_idx: int, feature_set: str
) -> Path:
    """Return the per-fold outer-prediction NPZ filename, by feature set."""
    if feature_set == "A":
        return output_dir / f"tabpfn_outer_fold{fold_idx}.npz"
    return output_dir / f"tabpfn_outer_fold{fold_idx}_{feature_set}.npz"


def resolve_tabpfn_cache_dir(default: str | None = None) -> str:
    """Return ``$TABPFN_MODEL_CACHE_DIR`` or apply ``default`` (with a warn).

    If ``default`` is None and the env var is unset, raises EnvironmentError
    with an actionable message. Avoids silently falling back to a host-
    specific path that won't exist on other machines.
    """
    cur = os.environ.get("TABPFN_MODEL_CACHE_DIR")
    if cur:
        return cur
    if default is not None:
        logger.warning(
            "TABPFN_MODEL_CACHE_DIR unset; falling back to host-specific "
            "default %r. Set TABPFN_MODEL_CACHE_DIR explicitly to silence "
            "this warning.", default,
        )
        os.environ["TABPFN_MODEL_CACHE_DIR"] = default
        return default
    raise EnvironmentError(
        "TABPFN_MODEL_CACHE_DIR is not set and no default was provided. "
        "Export TABPFN_MODEL_CACHE_DIR to a directory holding the TabPFN "
        "model weights before running this script."
    )
