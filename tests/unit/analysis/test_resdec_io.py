"""Smoke tests for :mod:`src.analysis.resdec_io`.

Contract (from C.3 review I-3):

- ``load_fold_predictions`` reads ``val_predictions_best.npz`` and the matching
  ``tabpfn_outer_fold{f}.npz``, merges them on ``ROSMAP_IndividualID``, and
  returns a DataFrame with the canonical six columns used by every downstream
  interpretability analysis.
- ``compute_per_fold_r2_ours`` agrees with :func:`sklearn.metrics.r2_score` on
  per-fold subsets of the concatenated DataFrame.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import r2_score

from src.analysis.resdec_io import (
    compute_per_fold_r2_ours,
    compute_per_fold_r2_tabpfn,
    load_all_folds,
    load_fold_predictions,
)


def _write_fold_npz(
    pred_root,
    tabpfn_dir,
    fold,
    subject_ids,
    y_true,
    y_composite,
    y_tabpfn,
    sigma_tabpfn=None,
):
    """Write the per-fold val_predictions_best.npz + tabpfn_outer_fold{f}.npz."""
    fold_dir = pred_root / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        fold_dir / "val_predictions_best.npz",
        subject_ids=np.asarray(subject_ids, dtype=object),
        targets=np.asarray(y_true, dtype=np.float64),
        predictions=np.asarray(y_composite, dtype=np.float64),
    )
    if sigma_tabpfn is None:
        sigma_tabpfn = np.full(len(subject_ids), 0.5, dtype=np.float64)
    tabpfn_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        tabpfn_dir / f"tabpfn_outer_fold{fold}.npz",
        val_subject_ids=np.asarray(subject_ids, dtype=object),
        y_true=np.asarray(y_true, dtype=np.float64),
        y_tabpfn=np.asarray(y_tabpfn, dtype=np.float64),
        sigma_tabpfn=np.asarray(sigma_tabpfn, dtype=np.float64),
    )


def test_load_fold_predictions_schema(tmp_path):
    """Mock npz with expected keys → returns DataFrame with expected columns."""
    pred_root = tmp_path / "pred"
    tabpfn_dir = tmp_path / "tabpfn"
    subject_ids = ["S1", "S2", "S3"]
    y_true = np.array([1.0, 2.0, 3.0])
    y_composite = np.array([1.1, 1.9, 3.2])
    y_tabpfn = np.array([0.9, 2.1, 2.8])
    _write_fold_npz(
        pred_root, tabpfn_dir, fold=0,
        subject_ids=subject_ids,
        y_true=y_true, y_composite=y_composite, y_tabpfn=y_tabpfn,
    )
    df = load_fold_predictions(pred_root, tabpfn_dir, fold=0)
    expected_cols = [
        "ROSMAP_IndividualID", "fold", "y_true",
        "y_composite", "y_tabpfn", "f1_residual",
    ]
    assert list(df.columns) == expected_cols
    assert len(df) == 3
    # f1_residual = y_composite - y_tabpfn
    np.testing.assert_allclose(
        df["f1_residual"].to_numpy(), y_composite - y_tabpfn, rtol=0, atol=1e-12,
    )
    # fold column populated
    assert (df["fold"] == 0).all()


def test_load_fold_predictions_raises_on_missing_predictions(tmp_path):
    pred_root = tmp_path / "pred"
    tabpfn_dir = tmp_path / "tabpfn"
    # Write only the TabPFN side; predictions side is missing.
    tabpfn_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        tabpfn_dir / "tabpfn_outer_fold0.npz",
        val_subject_ids=np.array(["S1"], dtype=object),
        y_true=np.array([1.0]),
        y_tabpfn=np.array([1.1]),
        sigma_tabpfn=np.array([0.5]),
    )
    with pytest.raises(FileNotFoundError, match="per-fold predictions"):
        load_fold_predictions(pred_root, tabpfn_dir, fold=0)


def test_load_fold_predictions_raises_on_ytrue_mismatch(tmp_path):
    pred_root = tmp_path / "pred"
    tabpfn_dir = tmp_path / "tabpfn"
    subject_ids = ["S1", "S2"]
    # y_true in predictions differs from y_true in tabpfn npz → raise
    _write_fold_npz(
        pred_root, tabpfn_dir, fold=0,
        subject_ids=subject_ids,
        y_true=np.array([1.0, 2.0]),
        y_composite=np.array([1.1, 1.9]),
        y_tabpfn=np.array([0.9, 2.1]),
    )
    # Overwrite the TabPFN npz with a mismatched y_true.
    np.savez(
        tabpfn_dir / "tabpfn_outer_fold0.npz",
        val_subject_ids=np.asarray(subject_ids, dtype=object),
        y_true=np.array([1.5, 2.0]),  # first entry differs
        y_tabpfn=np.array([0.9, 2.1]),
        sigma_tabpfn=np.array([0.5, 0.5]),
    )
    with pytest.raises(RuntimeError, match="y_true mismatch"):
        load_fold_predictions(pred_root, tabpfn_dir, fold=0)


def test_load_all_folds_concatenates(tmp_path):
    pred_root = tmp_path / "pred"
    tabpfn_dir = tmp_path / "tabpfn"
    for f in range(3):
        _write_fold_npz(
            pred_root, tabpfn_dir, fold=f,
            subject_ids=[f"F{f}_S{i}" for i in range(4)],
            y_true=np.arange(4, dtype=np.float64) + f,
            y_composite=np.arange(4, dtype=np.float64) + f + 0.1,
            y_tabpfn=np.arange(4, dtype=np.float64) + f - 0.1,
        )
    df = load_all_folds(pred_root, tabpfn_dir, n_folds=3)
    assert len(df) == 12  # 3 folds × 4 subjects
    assert set(df["fold"].unique()) == {0, 1, 2}


def test_compute_per_fold_r2_ours_agrees_with_sklearn(tmp_path):
    """Per-fold R² matches sklearn.metrics.r2_score within float precision."""
    rng = np.random.default_rng(0)
    records = []
    for f in range(5):
        n = 30
        y_true = rng.standard_normal(n)
        y_composite = y_true + rng.standard_normal(n) * 0.3
        y_tabpfn = y_true + rng.standard_normal(n) * 0.4
        for i in range(n):
            records.append({
                "ROSMAP_IndividualID": f"F{f}_S{i}",
                "fold": f,
                "y_true": y_true[i],
                "y_composite": y_composite[i],
                "y_tabpfn": y_tabpfn[i],
                "f1_residual": y_composite[i] - y_tabpfn[i],
            })
    df = pd.DataFrame.from_records(records)
    r2s = compute_per_fold_r2_ours(df, n_folds=5)
    assert r2s.shape == (5,)
    # Direct sklearn comparison per fold.
    for f in range(5):
        sub = df[df["fold"] == f]
        expected = r2_score(sub["y_true"].to_numpy(), sub["y_composite"].to_numpy())
        assert r2s[f] == pytest.approx(expected, abs=1e-12)


def test_compute_per_fold_r2_ours_raises_on_missing_fold():
    df = pd.DataFrame({
        "fold": [0, 0, 1, 1],
        "y_true": [1.0, 2.0, 3.0, 4.0],
        "y_composite": [1.1, 1.9, 3.1, 3.9],
    })
    # n_folds=3 but fold=2 is absent → raise
    with pytest.raises(RuntimeError, match="fold 2"):
        compute_per_fold_r2_ours(df, n_folds=3)


def test_compute_per_fold_r2_tabpfn_missing_npz_raises(tmp_path):
    tabpfn_dir = tmp_path / "tabpfn"
    tabpfn_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="TabPFN-2.6 standalone baseline"):
        compute_per_fold_r2_tabpfn(tabpfn_dir, n_folds=5)


def test_compute_per_fold_r2_tabpfn_agrees_with_sklearn(tmp_path):
    tabpfn_dir = tmp_path / "tabpfn"
    tabpfn_dir.mkdir()
    rng = np.random.default_rng(1)
    n_folds = 3
    expected_r2s = []
    for f in range(n_folds):
        n = 25
        y_true = rng.standard_normal(n)
        y_tabpfn = y_true + rng.standard_normal(n) * 0.3
        sigma_tabpfn = np.full(n, 0.5)
        np.savez(
            tabpfn_dir / f"tabpfn_outer_fold{f}.npz",
            val_subject_ids=np.asarray([f"F{f}_S{i}" for i in range(n)], dtype=object),
            y_true=y_true.astype(np.float64),
            y_tabpfn=y_tabpfn.astype(np.float64),
            sigma_tabpfn=sigma_tabpfn,
        )
        expected_r2s.append(r2_score(y_true, y_tabpfn))
    out = compute_per_fold_r2_tabpfn(tabpfn_dir, n_folds=n_folds)
    np.testing.assert_allclose(out, np.array(expected_r2s), rtol=0, atol=1e-12)
