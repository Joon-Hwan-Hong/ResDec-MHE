"""Canonical per-fold results.csv writer for the MixMIL baseline.

The MixMIL training adapter writes:
  - ``AllFolds_<MODEL_NAME>.csv`` with columns
    ``r2, mae, pearson_r, spearman_rho, fold, train_time_s`` (no rmse).
  - Per-fold ``fold_<N>/predictions.csv`` with columns
    ``sample_id, y_true, y_pred`` (1-indexed folds).

The paper-table aggregator's DL-baseline path
(``scripts/resdec_mhe/interpretability/make_baseline_table.py::collect_dl_baseline_rows``)
expects a per-baseline ``results.csv`` with columns
``fold, r2, mae, rmse, pearson_r, spearman_rho``. ``write_dl_results_csv``
synthesizes that file by passing through r2/mae/pearson_r/spearman_rho from
AllFolds and computing rmse per fold from the per-fold predictions CSVs
(MixMIL doesn't emit an rmse column or an mse column from which it could be
derived; the per-fold predictions file is the only source).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_DL_COLUMNS: tuple[str, ...] = (
    "fold", "r2", "mae", "rmse", "pearson_r", "spearman_rho",
)


def write_dl_results_csv(
    results_dir: Path | str,
    model_name: str = "MixMIL_ROSMAP",
) -> Path:
    """Emit ``results.csv`` in per-fold DL baseline schema.

    Parameters
    ----------
    results_dir
        Directory containing ``AllFolds_<model_name>.csv`` and
        ``fold_<N>/predictions.csv`` (1-indexed folds).
    model_name
        Used to construct the AllFolds filename.

    Returns
    -------
    Path
        Path to the written ``results.csv``.

    Raises
    ------
    FileNotFoundError
        If AllFolds CSV is missing, or any fold's predictions CSV is missing.
    """
    results_dir = Path(results_dir)
    allfolds_path = results_dir / f"AllFolds_{model_name}.csv"
    if not allfolds_path.exists():
        raise FileNotFoundError(f"AllFolds CSV missing: {allfolds_path}")

    allfolds = pd.read_csv(allfolds_path).sort_values("fold")

    rows: list[dict] = []
    for _, row in allfolds.iterrows():
        fold_id = int(row["fold"])
        preds_path = results_dir / f"fold_{fold_id}" / "predictions.csv"
        if not preds_path.exists():
            raise FileNotFoundError(
                f"Predictions CSV missing for fold {fold_id}: {preds_path}"
            )
        preds_df = pd.read_csv(preds_path)
        y_true = preds_df["y_true"].to_numpy(dtype=np.float64)
        y_pred = preds_df["y_pred"].to_numpy(dtype=np.float64)
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        rows.append({
            "fold": fold_id,
            "r2": float(row["r2"]),
            "mae": float(row["mae"]),
            "rmse": rmse,
            "pearson_r": float(row["pearson_r"]),
            "spearman_rho": float(row["spearman_rho"]),
        })

    out_path = results_dir / "results.csv"
    pd.DataFrame(rows)[list(_DL_COLUMNS)].to_csv(out_path, index=False)
    return out_path
