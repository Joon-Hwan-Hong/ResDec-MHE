"""Shared per-fold prediction + TabPFN outer-fold loaders.

Both C.1 (variance decomposition) and C.3 (statistical rigor) reuse these
loaders to maintain subject-set consistency across analyses. Kept in
``src/analysis`` (not the scripts subdir) so any future analysis can import
without crossing into orchestration-script territory.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


def load_fold_predictions(
    pred_root: Path, tabpfn_dir: Path, fold: int,
) -> pd.DataFrame:
    """Load predictions + TabPFN for a single fold and align by subject_id.

    Returns a long DataFrame with columns
    ``ROSMAP_IndividualID, fold, y_true, y_composite, y_tabpfn, f1_residual``.
    """
    pred_path = pred_root / f"fold{fold}/val_predictions_best.npz"
    tabpfn_path = tabpfn_dir / f"tabpfn_outer_fold{fold}.npz"

    if not pred_path.exists():
        raise FileNotFoundError(f"Missing per-fold predictions: {pred_path}")
    if not tabpfn_path.exists():
        raise FileNotFoundError(f"Missing outer TabPFN file: {tabpfn_path}")

    pred = np.load(pred_path, allow_pickle=True)
    tab = np.load(tabpfn_path, allow_pickle=True)

    pred_df = pd.DataFrame({
        "ROSMAP_IndividualID": pred["subject_ids"].astype(str),
        "y_true": pred["targets"].astype(np.float64),
        "y_composite": pred["predictions"].astype(np.float64),
    })
    tab_df = pd.DataFrame({
        "ROSMAP_IndividualID": tab["val_subject_ids"].astype(str),
        "y_true_tabpfn": tab["y_true"].astype(np.float64),
        "y_tabpfn": tab["y_tabpfn"].astype(np.float64),
    })
    merged = pred_df.merge(tab_df, on="ROSMAP_IndividualID", how="inner")

    missing_in_tabpfn = set(pred_df["ROSMAP_IndividualID"]) - set(tab_df["ROSMAP_IndividualID"])
    if missing_in_tabpfn:
        raise RuntimeError(
            f"Fold {fold}: {len(missing_in_tabpfn)} predicted subjects absent from "
            f"TabPFN outer-fold file ({tabpfn_path.name}). "
            f"First few: {sorted(missing_in_tabpfn)[:5]}"
        )

    # Sanity: y_true in predictions file should match y_true in TabPFN file.
    delta = np.max(np.abs(merged["y_true"].values - merged["y_true_tabpfn"].values))
    if delta > 1e-5:
        raise RuntimeError(
            f"Fold {fold}: y_true mismatch between val_predictions_best.npz and "
            f"tabpfn_outer_fold{fold}.npz (max |Δ| = {delta:.3e}). Refusing to proceed."
        )

    merged["f1_residual"] = merged["y_composite"].values - merged["y_tabpfn"].values
    merged["fold"] = fold
    return merged[
        ["ROSMAP_IndividualID", "fold", "y_true", "y_composite", "y_tabpfn", "f1_residual"]
    ]


def load_all_folds(
    pred_root: Path, tabpfn_dir: Path, n_folds: int = 5,
) -> pd.DataFrame:
    """Concatenate all folds' val predictions into a single long DataFrame."""
    return pd.concat(
        [load_fold_predictions(pred_root, tabpfn_dir, f) for f in range(n_folds)],
        ignore_index=True,
    )


def compute_per_fold_r2_ours(df: pd.DataFrame, n_folds: int) -> np.ndarray:
    """Per-fold R²(y_true, y_composite) on our concatenated predictions."""
    r2s = np.empty(n_folds, dtype=np.float64)
    for f in range(n_folds):
        sub = df[df["fold"] == f]
        if len(sub) == 0:
            raise RuntimeError(f"Ours: fold {f} has 0 subjects in df.")
        r2s[f] = r2_score(sub["y_true"].to_numpy(), sub["y_composite"].to_numpy())
    return r2s


def load_tabpfn_outer_fold(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(y_true, y_tabpfn)`` as float64 arrays from an outer-fold npz.

    Centralises the read-and-cast so every downstream TabPFN analysis pulls
    the same canonical tensors from the same canonical file layout.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    d = np.load(path, allow_pickle=True)
    return d["y_true"].astype(np.float64), d["y_tabpfn"].astype(np.float64)


def compute_per_fold_r2_tabpfn(tabpfn_dir: Path, n_folds: int) -> np.ndarray:
    """Per-fold R²(y_true, y_tabpfn) from ``tabpfn_outer_fold{f}.npz``.

    Fails loud if any fold's npz is missing — TabPFN-2.6 standalone is the
    required strongest baseline for this study and cannot be skipped.
    """
    r2s = np.empty(n_folds, dtype=np.float64)
    for f in range(n_folds):
        path = tabpfn_dir / f"tabpfn_outer_fold{f}.npz"
        try:
            y_true, y_tabpfn = load_tabpfn_outer_fold(path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"TabPFN-2.6 standalone baseline is required; missing {path}"
            )
        r2s[f] = r2_score(y_true, y_tabpfn)
    return r2s


__all__ = [
    "compute_per_fold_r2_ours",
    "compute_per_fold_r2_tabpfn",
    "load_all_folds",
    "load_fold_predictions",
    "load_tabpfn_outer_fold",
]
