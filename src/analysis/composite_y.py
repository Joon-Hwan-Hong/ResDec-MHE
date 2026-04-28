"""Shared composite-Y loader + Y-distribution sanity guard.

Consolidates the per-fold ``val_predictions_best.npz`` reading + Y-mean/std
guard that was duplicated across ``run_ct_ranking_nulls.py`` and
``run_cmi_subsample_bootstrap.py``.

Background
----------
``val_predictions_best.npz["predictions"]`` IS already the composite Y
(``ŷ_tabpfn + residual``) — see
``src/training/resdec_lightning_module.py:498`` where the per-fold writer
does ``pred = pred + y_tabpfn`` before serialisation. Adding ``y_tabpfn``
again would double-count.

The 2026-04-28 CMI bootstrap bug (memory rule
``feedback_verify_y_semantics.md``) exhibited this exact failure mode and
was caught only after the fact. This module provides:

  - :func:`load_composite_y_with_sanity_check` — reads the per-fold npz +
    runs the heuristic guard (|mean| < 1.5, std < 1.3) AND the stronger
    metadata-correlation guard (|Pearson(y, target_col)| > 0.3 for at least
    one of ``cogng_random_slope`` / ``cogn_global``).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)


# Memory rule feedback_verify_y_semantics.md heuristic bounds:
# canonical composite Y has |mean| < 1.0 and std < 1.0 on the n=516 cohort;
# corrupted double-composite Y has |mean| > 1.5 and std > 1.3 (~2× inflation).
_Y_MEAN_ABS_LIMIT: float = 1.5
_Y_STD_LIMIT: float = 1.3
_Y_CORR_LOWER_BOUND: float = 0.3


def load_composite_y_with_sanity_check(
    pred_root: Path,
    all_ids: Iterable[str],
    metadata_path: Path | None = None,
    *,
    n_folds: int = 5,
) -> np.ndarray:
    """Load the canonical composite-Y vector for the ``all_ids`` cohort.

    Reads ``<pred_root>/fold{0..n_folds-1}/val_predictions_best.npz``,
    builds a ``subject_id → predictions`` lookup, projects to the
    ``all_ids`` order (NaN for missing subjects), and runs two safety
    guards:

      1. **Heuristic mean/std guard** (memory rule
         ``feedback_verify_y_semantics.md``): refuses values consistent
         with the double-composite bug class (``|mean| > 1.5`` OR
         ``std > 1.3``).
      2. **Metadata-correlation guard** (when ``metadata_path`` is
         provided): aligns Y against ``cogng_random_slope`` or
         ``cogn_global`` and asserts ``|Pearson(y, target)| > 0.3``.
         Catches scale bugs that the heuristic guard misses.

    Parameters
    ----------
    pred_root
        Directory containing ``fold{0..N}/val_predictions_best.npz``.
    all_ids
        Subject IDs to extract Y for, in order. Missing subjects → NaN.
    metadata_path
        Optional path to ROSMAP metadata.csv. When provided, the Pearson
        correlation guard runs against the metadata target column.
    n_folds
        Number of fold directories to read (default 5 for the canonical
        ResDec-MHE seed42 split).

    Returns
    -------
    np.ndarray
        ``[len(all_ids)]`` float64 composite-Y vector. Entries for
        subjects with no fold prediction are NaN.

    Raises
    ------
    RuntimeError
        If either Y-distribution guard fails. Failures indicate a
        producer-side bug (likely double-composite or scale corruption);
        do not silently coerce.
    """
    pred_root = Path(pred_root)
    composite_y: dict[str, float] = {}
    for fold in range(int(n_folds)):
        fold_npz = pred_root / f"fold{fold}/val_predictions_best.npz"
        if not fold_npz.exists():
            raise FileNotFoundError(
                f"Missing per-fold predictions file: {fold_npz}"
            )
        v = np.load(fold_npz, allow_pickle=True)
        for sid, p in zip(v["subject_ids"], v["predictions"]):
            composite_y[str(sid)] = float(p)

    all_ids_list = [str(s) for s in all_ids]
    y = np.array([composite_y.get(sid, np.nan) for sid in all_ids_list],
                 dtype=np.float64)

    # ── Guard 1: heuristic mean/std band (memory rule). ────────────────
    finite = np.isfinite(y)
    if finite.sum() < 2:
        raise RuntimeError(
            f"composite Y has fewer than 2 finite entries (n_finite="
            f"{int(finite.sum())}); cannot validate."
        )
    y_mean = float(y[finite].mean())
    y_std = float(y[finite].std(ddof=1))
    logger.info("composite Y stats: mean=%.4f, std=%.4f, n=%d",
                y_mean, y_std, int(finite.sum()))
    if abs(y_mean) > _Y_MEAN_ABS_LIMIT or y_std > _Y_STD_LIMIT:
        raise RuntimeError(
            f"composite Y looks corrupted: mean={y_mean:.4f}, std={y_std:.4f}. "
            f"Canonical composite Y has |mean| < 1.0 and std < 1.0; values "
            f"outside [{_Y_MEAN_ABS_LIMIT}, {_Y_STD_LIMIT}] indicate a "
            f"double-add of y_tabpfn or similar producer bug. "
            f"See feedback_verify_y_semantics.md."
        )

    # ── Guard 2: Pearson correlation against metadata target. ──────────
    if metadata_path is not None:
        import pandas as pd

        meta = pd.read_csv(metadata_path)
        # Subject-ID column resolution: prefer ROSMAP_IndividualID.
        sid_col = None
        for cand in ("ROSMAP_IndividualID", "projid", "subject_id", "subject"):
            if cand in meta.columns:
                sid_col = cand
                break
        if sid_col is None:
            raise ValueError(
                f"metadata CSV missing subject-ID column; expected one of "
                f"['ROSMAP_IndividualID', 'projid', 'subject_id', 'subject']; "
                f"got {list(meta.columns)}"
            )
        meta_index = meta.set_index(meta[sid_col].astype(str))
        # Cognition target candidates: cogng_random_slope first (canonical
        # ResDec target), then cogn_global as fallback. Whichever the
        # metadata file actually carries is used to validate Y.
        target_col = None
        for cand in ("cogng_random_slope", "cogn_global"):
            if cand in meta.columns:
                target_col = cand
                break
        if target_col is None:
            logger.warning(
                "metadata CSV %s has neither 'cogng_random_slope' nor "
                "'cogn_global' — skipping correlation guard.",
                metadata_path,
            )
            return y
        target_lookup: dict[str, float] = {}
        for sid_idx, val in meta_index[target_col].items():
            if pd.notna(val):
                target_lookup[str(sid_idx)] = float(val)
        target_vec = np.array(
            [target_lookup.get(sid, np.nan) for sid in all_ids_list],
            dtype=np.float64,
        )
        ok = np.isfinite(y) & np.isfinite(target_vec)
        if ok.sum() < 10:
            logger.warning(
                "Y correlation guard: only %d aligned (Y, %s) pairs — too "
                "few for a stable Pearson r; skipping correlation guard.",
                int(ok.sum()), target_col,
            )
            return y
        r = float(np.corrcoef(y[ok], target_vec[ok])[0, 1])
        logger.info(
            "composite Y vs metadata %s correlation: r=%.4f over n=%d",
            target_col, r, int(ok.sum()),
        )
        if not (abs(r) > _Y_CORR_LOWER_BOUND):
            raise RuntimeError(
                f"composite Y is uncorrelated with metadata target "
                f"'{target_col}' (Pearson r={r:.4f}, threshold="
                f"{_Y_CORR_LOWER_BOUND}). This catches scale bugs that the "
                f"mean/std guard misses (e.g., wrong fold alignment, sign "
                f"flip, scaler mis-application). See "
                f"feedback_verify_y_semantics.md."
            )
    return y
