"""Shared RandomForest hyperparameters for the cogn-residual baseline + RF residual base."""
from __future__ import annotations

RF_KWARGS = {
    "n_estimators": 100,
    "max_depth": 16,
    "random_state": 42,
    "n_jobs": -1,
}

INNER_OOF_KFOLDS = 5
TOP_K = 2000
