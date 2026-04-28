"""Unit tests for baselines/mixmil/results_canonical.write_dl_results_csv.

The MixMIL training adapter writes:
  - ``AllFolds_MixMIL_ROSMAP.csv`` with columns
    ``r2, mae, pearson_r, spearman_rho, fold, train_time_s`` (no rmse).
  - Per-fold ``fold_<N>/predictions.csv`` with columns
    ``sample_id, y_true, y_pred`` (1-indexed folds).

The paper-table aggregator's DL-baseline path expects ``results.csv`` with
columns ``fold, r2, mae, rmse, pearson_r, spearman_rho``. ``write_dl_results_csv``
synthesizes that file by passing through r2/mae/pearson_r/spearman_rho from
AllFolds and computing rmse per fold from the per-fold predictions CSVs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# baselines/mixmil/ is a script dir, not a package. Add to sys.path so we can
# import results_canonical as a standalone module.
_MIXMIL_DIR = (
    Path(__file__).resolve().parents[3] / "baselines" / "mixmil"
)
if str(_MIXMIL_DIR) not in sys.path:
    sys.path.insert(0, str(_MIXMIL_DIR))

from results_canonical import write_dl_results_csv


def _make_allfolds_csv(
    results_dir: Path,
    per_fold_rows: list[dict],
    model_name: str = "MixMIL_ROSMAP",
) -> Path:
    """Write a synthetic AllFolds CSV matching MixMIL's schema.

    Columns: r2, mae, pearson_r, spearman_rho, fold, train_time_s.
    """
    rows = [
        {**r, "fold": i + 1, "train_time_s": 100.0}
        for i, r in enumerate(per_fold_rows)
    ]
    df = pd.DataFrame(rows)[
        ["r2", "mae", "pearson_r", "spearman_rho", "fold", "train_time_s"]
    ]
    path = results_dir / f"AllFolds_{model_name}.csv"
    df.to_csv(path, index=False)
    return path


def _make_fold_predictions_csv(
    results_dir: Path,
    fold: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Path:
    """Write a synthetic per-fold predictions.csv (1-indexed folds)."""
    fold_dir = results_dir / f"fold_{fold}"
    fold_dir.mkdir(exist_ok=True)
    df = pd.DataFrame({
        "sample_id": [f"R{i}" for i in range(len(y_true))],
        "y_true": y_true,
        "y_pred": y_pred,
    })
    path = fold_dir / "predictions.csv"
    df.to_csv(path, index=False)
    return path


class TestWriteDLResultsCsv:

    def _make_two_folds(self, tmp_path: Path) -> None:
        _make_allfolds_csv(tmp_path, [
            {"r2": 0.30, "mae": 0.50, "pearson_r": 0.60, "spearman_rho": 0.55},
            {"r2": 0.10, "mae": 1.00, "pearson_r": 0.40, "spearman_rho": 0.35},
        ])
        # Fold 1: y_true=[0,2], y_pred=[0,1] → mse = (0+1)/2 = 0.5, rmse = sqrt(0.5)
        _make_fold_predictions_csv(
            tmp_path, fold=1,
            y_true=np.array([0.0, 2.0]),
            y_pred=np.array([0.0, 1.0]),
        )
        # Fold 2: y_true=[1,3], y_pred=[2,5] → mse = (1+4)/2 = 2.5, rmse = sqrt(2.5)
        _make_fold_predictions_csv(
            tmp_path, fold=2,
            y_true=np.array([1.0, 3.0]),
            y_pred=np.array([2.0, 5.0]),
        )

    def test_emits_canonical_dl_schema(self, tmp_path):
        self._make_two_folds(tmp_path)
        out = write_dl_results_csv(tmp_path)
        assert out == tmp_path / "results.csv"
        assert out.exists()
        df = pd.read_csv(out)
        assert list(df.columns) == [
            "fold", "r2", "mae", "rmse", "pearson_r", "spearman_rho",
        ]
        assert len(df) == 2
        assert sorted(df["fold"].tolist()) == [1, 2]

    def test_passthrough_metrics(self, tmp_path):
        self._make_two_folds(tmp_path)
        df = pd.read_csv(write_dl_results_csv(tmp_path)).set_index("fold")
        assert df.loc[1, "r2"] == pytest.approx(0.30)
        assert df.loc[1, "mae"] == pytest.approx(0.50)
        assert df.loc[1, "pearson_r"] == pytest.approx(0.60)
        assert df.loc[1, "spearman_rho"] == pytest.approx(0.55)
        assert df.loc[2, "r2"] == pytest.approx(0.10)

    def test_rmse_computed_from_predictions(self, tmp_path):
        self._make_two_folds(tmp_path)
        df = pd.read_csv(write_dl_results_csv(tmp_path)).set_index("fold")
        assert df.loc[1, "rmse"] == pytest.approx(np.sqrt(0.5))
        assert df.loc[2, "rmse"] == pytest.approx(np.sqrt(2.5))

    def test_raises_if_allfolds_missing(self, tmp_path):
        _make_fold_predictions_csv(
            tmp_path, fold=1,
            y_true=np.array([0.0, 1.0]), y_pred=np.array([0.0, 1.0]),
        )
        with pytest.raises(FileNotFoundError):
            write_dl_results_csv(tmp_path)

    def test_raises_if_predictions_missing_for_a_fold(self, tmp_path):
        _make_allfolds_csv(tmp_path, [
            {"r2": 0.30, "mae": 0.50, "pearson_r": 0.60, "spearman_rho": 0.55},
            {"r2": 0.10, "mae": 1.00, "pearson_r": 0.40, "spearman_rho": 0.35},
        ])
        # Only fold 1 predictions; fold 2 missing.
        _make_fold_predictions_csv(
            tmp_path, fold=1,
            y_true=np.array([0.0, 2.0]), y_pred=np.array([0.0, 1.0]),
        )
        with pytest.raises(FileNotFoundError):
            write_dl_results_csv(tmp_path)
