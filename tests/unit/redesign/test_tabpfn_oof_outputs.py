"""Sanity-check tests for TabPFN OOF output files produced by
scripts/redesign/compute_tabpfn_oof.py.

Skipped if the files haven't been generated yet (first-run setup).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest
from sklearn.metrics import r2_score


OOF_DIR = Path("data/redesign")


def _have_all_folds() -> bool:
    return all((OOF_DIR / f"tabpfn_oof_fold{f}.npz").exists() for f in range(5))


pytestmark = pytest.mark.skipif(
    not _have_all_folds(),
    reason="TabPFN OOF files not yet generated (run scripts/redesign/compute_tabpfn_oof.py)",
)


def test_tabpfn_oof_shapes_consistent():
    for fold_idx in range(5):
        d = np.load(OOF_DIR / f"tabpfn_oof_fold{fold_idx}.npz", allow_pickle=True)
        n = len(d["subject_ids"])
        assert len(d["y_true"]) == n
        assert len(d["y_tabpfn_oof"]) == n
        assert len(d["sigma_tabpfn_oof"]) == n
        assert n > 0


def test_tabpfn_oof_no_nan_and_sensible_range():
    for fold_idx in range(5):
        d = np.load(OOF_DIR / f"tabpfn_oof_fold{fold_idx}.npz", allow_pickle=True)
        assert not np.any(np.isnan(d["y_tabpfn_oof"])), f"NaN in fold {fold_idx} predictions"
        assert not np.any(np.isnan(d["sigma_tabpfn_oof"])), f"NaN in fold {fold_idx} sigmas"
        # Predictions should fall within a few std of training-target range
        y_min, y_max = d["y_true"].min(), d["y_true"].max()
        assert d["y_tabpfn_oof"].min() >= y_min - 3, f"fold {fold_idx}: suspiciously low"
        assert d["y_tabpfn_oof"].max() <= y_max + 3, f"fold {fold_idx}: suspiciously high"


def test_tabpfn_oof_sigma_positive():
    for fold_idx in range(5):
        d = np.load(OOF_DIR / f"tabpfn_oof_fold{fold_idx}.npz", allow_pickle=True)
        assert (d["sigma_tabpfn_oof"] > 0).all(), f"fold {fold_idx}: non-positive sigma"


def test_tabpfn_oof_r2_sensible_range():
    """Mean OOF R² across folds should be roughly 0.15-0.55 (sanity window)."""
    r2s = []
    for fold_idx in range(5):
        d = np.load(OOF_DIR / f"tabpfn_oof_fold{fold_idx}.npz", allow_pickle=True)
        r2s.append(r2_score(d["y_true"], d["y_tabpfn_oof"]))
    mean_r2 = float(np.mean(r2s))
    assert 0.15 < mean_r2 < 0.55, f"TabPFN OOF mean R²={mean_r2:.4f} outside expected range"
