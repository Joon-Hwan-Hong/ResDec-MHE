"""Unit tests for baselines/scPhase/summary_canonical.write_canonical_summary.

The scPhase adapter writes AllFolds_*.csv (one row per fold: ``mse, mae, r2,
person``) + a per-sample predictions CSV. The canonical Summary CSV schema
expected by the paper-table aggregator (matching MixMIL's output) is long
format: columns ``metric, mean, std``; rows ``r2, mae, rmse, pearson_r,
spearman_rho``.

``write_canonical_summary`` synthesizes that canonical Summary by:
  - passing through r2, mae
  - renaming ``person`` → ``pearson_r``
  - computing ``rmse = sqrt(mse)`` per fold
  - computing ``spearman_rho`` per fold from the predictions CSV
  - aggregating mean + std (ddof=1) across folds
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

# baselines/scPhase/ is a script dir, not a package. Add it to sys.path so we
# can import ``summary_canonical`` as a standalone module.
_SCPHASE_DIR = (
    Path(__file__).resolve().parents[3] / "baselines" / "scPhase"
)
if str(_SCPHASE_DIR) not in sys.path:
    sys.path.insert(0, str(_SCPHASE_DIR))

from summary_canonical import (  # noqa: E402
    write_canonical_summary,
    write_dl_results_csv,
)


def _make_allfolds_csv(
    results_dir: Path,
    per_fold_rows: list[dict],
    model_name: str = "scPhase_ROSMAP",
) -> Path:
    rows = [
        {"model_name": model_name, "fold": i + 1, **r}
        for i, r in enumerate(per_fold_rows)
    ]
    path = results_dir / f"AllFolds_{model_name}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _make_predictions_csv(
    results_dir: Path,
    per_fold_y: list[tuple[np.ndarray, np.ndarray]],
    model_name: str = "scPhase_ROSMAP",
) -> Path:
    preds_dir = results_dir / "predictions"
    preds_dir.mkdir(exist_ok=True)
    rows: list[dict] = []
    for fold_idx, (y_true, y_pred) in enumerate(per_fold_y, start=1):
        for i, (t, p) in enumerate(zip(y_true, y_pred)):
            rows.append({
                "model_name": model_name,
                "fold": fold_idx,
                "test_group": "[0]",
                "sample_idx": i,
                "y_true": float(t),
                "y_pred": float(p),
                "auc_score": 0.0,
            })
    path = preds_dir / f"{model_name}_predictions.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


class TestWriteCanonicalSummary:

    def _make_two_folds(self, tmp_path: Path) -> None:
        _make_allfolds_csv(tmp_path, [
            {"mse": 1.0, "mae": 0.5, "r2": 0.3, "person": 0.6},
            {"mse": 4.0, "mae": 1.0, "r2": 0.1, "person": 0.4},
        ])
        _make_predictions_csv(tmp_path, [
            (np.array([0.1, 0.5, 0.9, 0.3, 0.7]),
             np.array([0.2, 0.4, 0.8, 0.5, 0.6])),
            (np.array([0.9, 0.5, 0.1, 0.7, 0.3]),
             np.array([0.1, 0.5, 0.9, 0.3, 0.7])),
        ])

    def test_emits_long_format_with_five_metrics(self, tmp_path):
        self._make_two_folds(tmp_path)
        out = write_canonical_summary(tmp_path)
        assert out == tmp_path / "Summary_scPhase_ROSMAP.csv"
        assert out.exists()
        df = pd.read_csv(out)
        assert list(df.columns) == ["metric", "mean", "std"]
        assert set(df["metric"]) == {
            "r2", "mae", "rmse", "pearson_r", "spearman_rho",
        }

    def test_passthrough_r2_and_mae(self, tmp_path):
        self._make_two_folds(tmp_path)
        out = write_canonical_summary(tmp_path)
        df = pd.read_csv(out).set_index("metric")
        assert df.loc["r2", "mean"] == pytest.approx(0.2)
        assert df.loc["r2", "std"] == pytest.approx(
            float(np.std([0.3, 0.1], ddof=1)),
        )
        assert df.loc["mae", "mean"] == pytest.approx(0.75)
        assert df.loc["mae", "std"] == pytest.approx(
            float(np.std([0.5, 1.0], ddof=1)),
        )

    def test_rmse_equals_sqrt_mse_per_fold(self, tmp_path):
        self._make_two_folds(tmp_path)
        out = write_canonical_summary(tmp_path)
        df = pd.read_csv(out).set_index("metric")
        expected = [1.0, 2.0]
        assert df.loc["rmse", "mean"] == pytest.approx(float(np.mean(expected)))
        assert df.loc["rmse", "std"] == pytest.approx(
            float(np.std(expected, ddof=1)),
        )

    def test_pearson_renamed_from_person(self, tmp_path):
        self._make_two_folds(tmp_path)
        out = write_canonical_summary(tmp_path)
        df = pd.read_csv(out).set_index("metric")
        assert df.loc["pearson_r", "mean"] == pytest.approx(0.5)
        assert df.loc["pearson_r", "std"] == pytest.approx(
            float(np.std([0.6, 0.4], ddof=1)),
        )

    def test_spearman_computed_per_fold_from_predictions(self, tmp_path):
        self._make_two_folds(tmp_path)
        out = write_canonical_summary(tmp_path)
        df = pd.read_csv(out).set_index("metric")
        y1t = np.array([0.1, 0.5, 0.9, 0.3, 0.7])
        y1p = np.array([0.2, 0.4, 0.8, 0.5, 0.6])
        y2t = np.array([0.9, 0.5, 0.1, 0.7, 0.3])
        y2p = np.array([0.1, 0.5, 0.9, 0.3, 0.7])
        sp1 = float(spearmanr(y1t, y1p).statistic)
        sp2 = float(spearmanr(y2t, y2p).statistic)
        assert df.loc["spearman_rho", "mean"] == pytest.approx(
            float(np.mean([sp1, sp2])),
        )
        assert df.loc["spearman_rho", "std"] == pytest.approx(
            float(np.std([sp1, sp2], ddof=1)),
        )

    def test_single_fold_std_is_zero(self, tmp_path):
        _make_allfolds_csv(tmp_path, [
            {"mse": 1.0, "mae": 0.5, "r2": 0.3, "person": 0.6},
        ])
        _make_predictions_csv(tmp_path, [
            (np.array([0.1, 0.5, 0.9]), np.array([0.2, 0.4, 0.8])),
        ])
        out = write_canonical_summary(tmp_path)
        df = pd.read_csv(out).set_index("metric")
        for metric in ("r2", "mae", "rmse", "pearson_r", "spearman_rho"):
            assert df.loc[metric, "std"] == 0.0, f"{metric} std should be 0 for n=1"

    def test_raises_if_allfolds_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            write_canonical_summary(tmp_path)

    def test_raises_if_predictions_missing(self, tmp_path):
        _make_allfolds_csv(tmp_path, [
            {"mse": 1.0, "mae": 0.5, "r2": 0.3, "person": 0.6},
        ])
        with pytest.raises(FileNotFoundError):
            write_canonical_summary(tmp_path)


class TestWriteDLResultsCsv:
    """Per-fold results.csv (DL baseline schema: fold, r2, mae, rmse, pearson_r,
    spearman_rho) — matches the schema of cloudpred/gpio/perceiver_io baselines
    so the paper-table aggregator's ``collect_dl_baseline_rows`` picks scPhase
    up via the same path as the other DL baselines (no warning, no special
    case)."""

    def _make_two_folds(self, tmp_path: Path) -> None:
        _make_allfolds_csv(tmp_path, [
            {"mse": 1.0, "mae": 0.5, "r2": 0.3, "person": 0.6},
            {"mse": 4.0, "mae": 1.0, "r2": 0.1, "person": 0.4},
        ])
        _make_predictions_csv(tmp_path, [
            (np.array([0.1, 0.5, 0.9, 0.3, 0.7]),
             np.array([0.2, 0.4, 0.8, 0.5, 0.6])),
            (np.array([0.9, 0.5, 0.1, 0.7, 0.3]),
             np.array([0.1, 0.5, 0.9, 0.3, 0.7])),
        ])

    def test_emits_per_fold_dl_schema(self, tmp_path):
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

    def test_passthrough_r2_mae_per_fold(self, tmp_path):
        self._make_two_folds(tmp_path)
        df = pd.read_csv(write_dl_results_csv(tmp_path)).set_index("fold")
        assert df.loc[1, "r2"] == pytest.approx(0.3)
        assert df.loc[2, "r2"] == pytest.approx(0.1)
        assert df.loc[1, "mae"] == pytest.approx(0.5)
        assert df.loc[2, "mae"] == pytest.approx(1.0)

    def test_rmse_equals_sqrt_mse(self, tmp_path):
        self._make_two_folds(tmp_path)
        df = pd.read_csv(write_dl_results_csv(tmp_path)).set_index("fold")
        assert df.loc[1, "rmse"] == pytest.approx(1.0)
        assert df.loc[2, "rmse"] == pytest.approx(2.0)

    def test_pearson_renamed_from_person(self, tmp_path):
        self._make_two_folds(tmp_path)
        df = pd.read_csv(write_dl_results_csv(tmp_path)).set_index("fold")
        assert df.loc[1, "pearson_r"] == pytest.approx(0.6)
        assert df.loc[2, "pearson_r"] == pytest.approx(0.4)

    def test_spearman_computed_per_fold(self, tmp_path):
        self._make_two_folds(tmp_path)
        df = pd.read_csv(write_dl_results_csv(tmp_path)).set_index("fold")
        y1t = np.array([0.1, 0.5, 0.9, 0.3, 0.7])
        y1p = np.array([0.2, 0.4, 0.8, 0.5, 0.6])
        y2t = np.array([0.9, 0.5, 0.1, 0.7, 0.3])
        y2p = np.array([0.1, 0.5, 0.9, 0.3, 0.7])
        sp1 = float(spearmanr(y1t, y1p).statistic)
        sp2 = float(spearmanr(y2t, y2p).statistic)
        assert df.loc[1, "spearman_rho"] == pytest.approx(sp1)
        assert df.loc[2, "spearman_rho"] == pytest.approx(sp2)

    def test_single_fold_emits_one_row(self, tmp_path):
        _make_allfolds_csv(tmp_path, [
            {"mse": 1.0, "mae": 0.5, "r2": 0.3, "person": 0.6},
        ])
        _make_predictions_csv(tmp_path, [
            (np.array([0.1, 0.5, 0.9]), np.array([0.2, 0.4, 0.8])),
        ])
        df = pd.read_csv(write_dl_results_csv(tmp_path))
        assert len(df) == 1
        assert df["fold"].tolist() == [1]

    def test_raises_if_allfolds_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            write_dl_results_csv(tmp_path)

    def test_raises_if_predictions_missing(self, tmp_path):
        _make_allfolds_csv(tmp_path, [
            {"mse": 1.0, "mae": 0.5, "r2": 0.3, "person": 0.6},
        ])
        with pytest.raises(FileNotFoundError):
            write_dl_results_csv(tmp_path)
