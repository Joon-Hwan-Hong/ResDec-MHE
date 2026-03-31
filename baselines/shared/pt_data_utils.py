"""Shared data loading and metrics for .pt-based baselines (CloudPred, Perceiver IO, GPIO)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_splits(splits_path: str | Path) -> dict:
    """Load 5-fold CV splits."""
    with open(splits_path) as f:
        return json.load(f)


def load_metadata(metadata_dir: str | Path, subject_column: str = "ROSMAP_IndividualID") -> dict[str, float]:
    """Load cogn_global targets from metadata.csv. Returns {subject_id: cogn_global}."""
    meta = pd.read_csv(Path(metadata_dir) / "metadata.csv")
    meta = meta.dropna(subset=["cogn_global"])
    return dict(zip(meta[subject_column], meta["cogn_global"]))


def load_subject_pt(data_dir: str | Path, subject_id: str) -> dict:
    """Load a single subject's precomputed .pt file."""
    return torch.load(Path(data_dir) / f"{subject_id}.pt", weights_only=False)


def extract_ccc_summary(pt_data: dict, n_ccc_types: int = 5, n_cell_types: int = 31) -> np.ndarray:
    """Extract CCC graph summary features [18] — same as run_baselines.py extract_features_e().

    Per-type (5 types): edge count, mean edge attr, std edge attr.
    Global node-degree: mean, std, max.
    """
    edge_index = pt_data["ccc_edge_index"]
    edge_type = pt_data["ccc_edge_type"]
    edge_attr = pt_data["ccc_edge_attr"]
    n_edges = edge_index.shape[1]

    counts = np.zeros(n_ccc_types, dtype=np.float32)
    mean_attrs = np.zeros(n_ccc_types, dtype=np.float32)
    std_attrs = np.zeros(n_ccc_types, dtype=np.float32)

    for t in range(n_ccc_types):
        mask = edge_type == t
        c = mask.sum().item()
        counts[t] = c
        if c > 0:
            attrs_t = edge_attr[mask]
            mean_attrs[t] = attrs_t.mean().item()
            std_attrs[t] = attrs_t.std().item() if c > 1 else 0.0

    degrees = torch.zeros(n_cell_types, dtype=torch.float32)
    if n_edges > 0:
        src_nodes = edge_index[0]
        for i in range(n_edges):
            degrees[src_nodes[i]] += 1

    return np.concatenate([
        counts, mean_attrs, std_attrs,
        np.array([degrees.mean().item(), degrees.std().item(), degrees.max().item()], dtype=np.float32),
    ])


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute regression metrics: R2, MAE, RMSE, Pearson r, Spearman rho."""
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "pearson_r": float(pearsonr(y_true.ravel(), y_pred.ravel())[0]),
        "spearman_rho": float(spearmanr(y_true.ravel(), y_pred.ravel())[0]),
    }


def save_results(results: list[dict], results_dir: Path, model_name: str) -> None:
    """Save per-fold results CSV and print summary."""
    results_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(results_dir / "results.csv", index=False)

    metric_cols = ["r2", "mae", "rmse", "pearson_r", "spearman_rho"]
    print(f"\n{'='*60}")
    print(f"  {model_name} — Summary across folds")
    print(f"{'='*60}")
    for col in metric_cols:
        vals = df[col]
        print(f"  {col:15s}  {vals.mean():.4f} +/- {vals.std():.4f}")
    print(f"\nResults saved to {results_dir}")
