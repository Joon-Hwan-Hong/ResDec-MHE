"""Shared loaders for flat pseudobulk features + cognition targets.

Used by scripts/redesign/compute_* and downstream redesign tooling. Replaces
duplicated copies previously in each script. All loaders emit rich logging
(counts of loaded / skipped / total).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from src.data.tabpfn_input import flatten_pseudobulk

logger = logging.getLogger(__name__)


def load_flat_features(
    precomputed_dir: Path,
    subject_ids: Iterable[str],
) -> dict[str, np.ndarray]:
    """Flatten pseudobulk per subject. Returns {subject_id -> [F]} numpy arrays.

    Logs: total requested, loaded, missing (by filename). Does NOT error on
    missing files — skips with a warning, so callers can intersect with
    target-available subjects explicitly.
    """
    subject_ids = list(subject_ids)
    out: dict[str, np.ndarray] = {}
    missing: list[str] = []
    for sid in subject_ids:
        pt_path = precomputed_dir / f"{sid}.pt"
        if not pt_path.exists():
            missing.append(sid)
            continue
        pt = torch.load(pt_path, weights_only=False)
        out[sid] = flatten_pseudobulk(pt).numpy()
    logger.info(
        "load_flat_features: %d/%d subjects loaded (%d missing .pt files)",
        len(out), len(subject_ids), len(missing),
    )
    if missing and len(missing) <= 10:
        logger.warning("Missing .pt files: %s", missing)
    elif missing:
        logger.warning(
            "Missing .pt files (showing first 10 of %d): %s",
            len(missing), missing[:10],
        )
    return out


def load_targets(
    meta_csv: Path,
    subject_ids: Iterable[str],
    target_col: str = "cogn_global",
    id_col: str = "ROSMAP_IndividualID",
) -> dict[str, float]:
    """Load a numeric target per subject via ROSMAP_IndividualID. NOT `projid`.

    Logs: total requested, found-with-non-null-target, not-in-metadata, null-target.
    """
    subject_ids = list(subject_ids)
    df = pd.read_csv(meta_csv)
    wanted = set(subject_ids)
    df = df[df[id_col].isin(wanted)]

    out: dict[str, float] = {}
    null_target = 0
    for _, r in df.iterrows():
        if pd.isna(r[target_col]):
            null_target += 1
            continue
        out[r[id_col]] = float(r[target_col])

    not_in_meta = len(wanted) - len(df)
    logger.info(
        "load_targets (%s): %d/%d subjects with non-null target "
        "(not_in_meta=%d, null_target=%d)",
        target_col, len(out), len(subject_ids), not_in_meta, null_target,
    )
    return out


def compute_age_stats_from_training(
    meta_csv: Path,
    train_subject_ids: Iterable[str],
) -> tuple[float, float]:
    """Compute (mean, std) of age_death over training subjects only (per-fold).

    Used by FiLM metadata loader for per-fold z-scoring (avoids val-set leakage).
    """
    df = pd.read_csv(meta_csv)
    wanted = set(train_subject_ids)
    df = df[df["ROSMAP_IndividualID"].isin(wanted)]
    ages = df["age_death"].dropna().values
    if len(ages) == 0:
        raise ValueError(
            "No non-null age_death values for given training subjects"
        )
    mean, std = float(np.mean(ages)), float(np.std(ages))
    logger.info(
        "age stats (train subjects): mean=%.4f, std=%.4f (n=%d)",
        mean, std, len(ages),
    )
    return mean, std
