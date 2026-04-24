"""Shared TabPFN input preprocessing helpers.

This module centralises preprocessing steps used by the TabPFN pre-compute
scripts (``scripts/resdec_mhe/tabpfn/compute_oof.py`` and ``compute_outer.py``)
so that the two call sites stay in lockstep.
"""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler


def apply_zscore_train_only(
    X_train: np.ndarray, X_val: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature z-score using TRAIN-ONLY stats (NO POOLED STATS -> no val leakage).

    For inner-fold OOF callers, ``X_train`` should be the inner-fold train split
    (X_train_full[tr_idx]). For outer-fold callers, ``X_train`` is the outer train
    split. Fits ``sklearn.preprocessing.StandardScaler`` on X_train only and
    applies the fitted transform to both arrays.

    Zero-variance features -> scaler sets scale_=1, so they mean-center to 0 without
    divide-by-zero. Returns freshly-allocated float32 arrays; inputs are not mutated.
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32, copy=False)
    X_val_s = scaler.transform(X_val).astype(np.float32, copy=False)
    return X_train_s, X_val_s
