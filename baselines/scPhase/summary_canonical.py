"""Canonical Summary + per-fold results CSV writers for the scPhase baseline.

The scPhase training adapter writes two CSVs per run:
  - ``AllFolds_<MODEL_NAME>.csv`` — per-fold metrics with columns
    ``model_name, fold, mse, mae, r2, person`` (``person`` is an upstream
    typo for ``pearson``; not spearman).
  - ``predictions/<MODEL_NAME>_predictions.csv`` — per-sample predictions
    with columns ``model_name, fold, test_group, sample_idx, y_true,
    y_pred, auc_score``.

The paper baseline-table aggregator consumes two CSV shapes:
  - Long-format Summary CSV (``metric, mean, std``) for ROSMAP-style
    baselines (matches MixMIL's output).
  - Per-fold ``results.csv`` (``fold, r2, mae, rmse, pearson_r,
    spearman_rho``) for the DL baseline parser
    (cloudpred/gpio/perceiver_io schema).

This module emits both, so scPhase shows up cleanly under whichever
aggregator path the table builder uses. Both writers normalize the
upstream ``person`` typo to ``pearson_r``, add ``rmse = sqrt(mse)``, and
compute ``spearman_rho`` per fold from the predictions CSV (which
scPhase does not emit otherwise).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_CANONICAL_METRICS: tuple[str, ...] = (
    "r2", "mae", "rmse", "pearson_r", "spearman_rho",
)


def _compute_per_fold_metrics(
    results_dir: Path,
    model_name: str,
) -> pd.DataFrame:
    """Read AllFolds + predictions; return per-fold DataFrame in canonical schema.

    Output columns: ``fold, r2, mae, rmse, pearson_r, spearman_rho`` —
    one row per fold, sorted by fold id.
    """
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

    rows: list[dict] = []
    for _, row in allfolds.iterrows():
        fold_id = int(row["fold"])
        fold_preds = preds[preds["fold"] == fold_id]
        if len(fold_preds) == 0:
            raise ValueError(
                f"Predictions CSV has no rows for fold {fold_id} "
                f"({predictions_path})"
            )
        sp = float(spearmanr(
            fold_preds["y_true"].to_numpy(),
            fold_preds["y_pred"].to_numpy(),
        ).statistic)
        rows.append({
            "fold": fold_id,
            "r2": float(row["r2"]),
            "mae": float(row["mae"]),
            "rmse": float(np.sqrt(float(row["mse"]))),
            "pearson_r": float(row["person"]),
            "spearman_rho": sp,
        })
    return pd.DataFrame(rows)


def write_canonical_summary(
    results_dir: Path | str,
    model_name: str = "scPhase_ROSMAP",
) -> Path:
    """Emit ``Summary_<model_name>.csv`` in long format from AllFolds + predictions.

    Output schema: columns ``metric, mean, std``; one row per metric in
    :data:`_CANONICAL_METRICS`. Aggregates per-fold values with
    ``ddof=1`` (sample std); single-fold ``std = 0.0``.

    Raises
    ------
    FileNotFoundError
        If AllFolds or predictions CSV is missing.
    ValueError
        If predictions CSV has no rows for a fold referenced in AllFolds.
    """
    results_dir = Path(results_dir)
    df = _compute_per_fold_metrics(results_dir, model_name)

    rows: list[dict] = []
    for metric in _CANONICAL_METRICS:
        vals = df[metric].to_numpy(dtype=np.float64)
        mean_v = float(vals.mean())
        std_v = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
        rows.append({"metric": metric, "mean": mean_v, "std": std_v})

    out_path = results_dir / f"Summary_{model_name}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


def write_dl_results_csv(
    results_dir: Path | str,
    model_name: str = "scPhase_ROSMAP",
) -> Path:
    """Emit ``results.csv`` in per-fold DL baseline schema.

    Output schema: columns ``fold, r2, mae, rmse, pearson_r,
    spearman_rho`` — one row per fold. Matches the schema of
    cloudpred/gpio/perceiver_io ``results.csv`` files so the
    paper-table aggregator's ``collect_dl_baseline_rows`` picks scPhase
    up via the same path as the other DL baselines.

    Raises
    ------
    FileNotFoundError
        If AllFolds or predictions CSV is missing.
    ValueError
        If predictions CSV has no rows for a fold referenced in AllFolds.
    """
    results_dir = Path(results_dir)
    df = _compute_per_fold_metrics(results_dir, model_name)
    out_path = results_dir / "results.csv"
    df.to_csv(out_path, index=False)
    return out_path
