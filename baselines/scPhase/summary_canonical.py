"""Canonical Summary CSV writer for the scPhase baseline.

The scPhase training adapter writes two CSVs per run:
  - ``AllFolds_<MODEL_NAME>.csv`` — per-fold metrics with columns
    ``model_name, fold, mse, mae, r2, person`` (``person`` is an upstream
    typo for ``pearson``; not spearman).
  - ``predictions/<MODEL_NAME>_predictions.csv`` — per-sample predictions
    with columns ``model_name, fold, test_group, sample_idx, y_true,
    y_pred, auc_score``.

The paper baseline-table aggregator expects a long-format Summary CSV
matching the MixMIL schema: columns ``metric, mean, std``; rows
``r2, mae, rmse, pearson_r, spearman_rho``.

``write_canonical_summary`` reads the two inputs above and writes the
Summary CSV in the canonical schema. It renames ``person`` → ``pearson_r``,
adds ``rmse = sqrt(mse)``, and computes ``spearman_rho`` per fold from the
predictions CSV (which scPhase does not aggregate otherwise).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_CANONICAL_METRICS: tuple[str, ...] = (
    "r2", "mae", "rmse", "pearson_r", "spearman_rho",
)


def write_canonical_summary(
    results_dir: Path | str,
    model_name: str = "scPhase_ROSMAP",
) -> Path:
    """Emit ``Summary_<model_name>.csv`` in long format from AllFolds + predictions.

    Parameters
    ----------
    results_dir
        Directory containing ``AllFolds_<model_name>.csv`` and
        ``predictions/<model_name>_predictions.csv``.
    model_name
        Used to construct the input/output filenames.

    Returns
    -------
    Path
        Path to the written Summary CSV.

    Raises
    ------
    FileNotFoundError
        If AllFolds or predictions CSV is missing.
    ValueError
        If predictions CSV has no rows for a fold referenced in AllFolds.
    """
    results_dir = Path(results_dir)
    allfolds_path = results_dir / f"AllFolds_{model_name}.csv"
    predictions_path = (
        results_dir / "predictions" / f"{model_name}_predictions.csv"
    )

    if not allfolds_path.exists():
        raise FileNotFoundError(f"AllFolds CSV missing: {allfolds_path}")
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions CSV missing: {predictions_path}")

    allfolds = pd.read_csv(allfolds_path).sort_values("fold")
    preds = pd.read_csv(predictions_path)

    per_fold: dict[str, list[float]] = {k: [] for k in _CANONICAL_METRICS}
    for _, row in allfolds.iterrows():
        fold_id = int(row["fold"])
        per_fold["r2"].append(float(row["r2"]))
        per_fold["mae"].append(float(row["mae"]))
        per_fold["rmse"].append(float(np.sqrt(float(row["mse"]))))
        per_fold["pearson_r"].append(float(row["person"]))

        fold_preds = preds[preds["fold"] == fold_id]
        if len(fold_preds) == 0:
            raise ValueError(
                f"Predictions CSV has no rows for fold {fold_id} "
                f"({predictions_path})"
            )
        sp = spearmanr(
            fold_preds["y_true"].to_numpy(),
            fold_preds["y_pred"].to_numpy(),
        ).statistic
        per_fold["spearman_rho"].append(float(sp))

    rows: list[dict] = []
    for metric in _CANONICAL_METRICS:
        vals = np.asarray(per_fold[metric], dtype=np.float64)
        mean_v = float(vals.mean())
        std_v = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
        rows.append({"metric": metric, "mean": mean_v, "std": std_v})

    out_path = results_dir / f"Summary_{model_name}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path
