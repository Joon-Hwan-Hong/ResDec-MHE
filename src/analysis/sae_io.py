"""Shared I/O helpers for SAE training scripts.

Consolidates the persistence + metadata-loading + per-(subject, CT)
fold-index expansion logic that was duplicated across
``run_sae_train.py`` / ``run_sae_random_null.py`` (and is referenced by
the cross-seed / feature-xref orchestrators).

Functions
---------
save_sae_model
    Persist a trained :class:`SAEModel` to ``<run_dir>/sae_model.npz``
    using the canonical key layout (``W_enc`` / ``W_dec`` / ``b_enc`` /
    ``b_dec`` + ``stat_*`` arrays + ``config_json``).
load_metadata_lookup
    Read ROSMAP metadata.csv and emit the per-subject dict expected by
    :func:`src.analysis.sparse_autoencoder.interpret_features`.
expand_fold_idx_to_rows
    For ``layer == "fused"``, expand ``[N]`` per-subject fold indices to
    ``[N * C]`` per-(subject, CT) rows so per-fold reconstruction-metric
    masks line up with the flattened activation matrix.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.analysis.sparse_autoencoder import SAEModel


def save_sae_model(sae: SAEModel, path: Path) -> None:
    """Persist a trained :class:`SAEModel` as a single ``.npz`` file.

    Schema (matches ``run_sae_train.py`` / ``run_sae_random_null.py``):

      - ``W_enc``, ``b_enc``, ``W_dec``, ``b_dec`` — float32 weight arrays.
      - ``config_json`` — JSON string of :func:`dataclasses.asdict(sae.config)`.
      - ``stat_mean`` / ``stat_std`` / ``stat_fraction_active`` /
        ``stat_is_dead`` / ``stat_threshold`` — per-feature stats from
        :attr:`SAEModel.activation_stats` (zero-length array when absent).
    """
    cfg_dict = asdict(sae.config)
    np.savez(
        path,
        W_enc=sae.W_enc,
        b_enc=sae.b_enc,
        W_dec=sae.W_dec,
        b_dec=sae.b_dec,
        config_json=np.array(json.dumps(cfg_dict), dtype=object),
        stat_mean=sae.activation_stats.get("mean", np.zeros(0)),
        stat_std=sae.activation_stats.get("std", np.zeros(0)),
        stat_fraction_active=sae.activation_stats.get(
            "fraction_active", np.zeros(0)
        ),
        stat_is_dead=sae.activation_stats.get("is_dead", np.zeros(0, dtype=bool)),
        stat_threshold=sae.activation_stats.get("threshold", np.zeros(0)),
    )


def load_metadata_lookup(metadata_csv: Path, subject_ids: np.ndarray) -> dict:
    """Build the metadata dict expected by :func:`interpret_features`.

    Resolves ROSMAP subject-ID column from the canonical candidates
    (``ROSMAP_IndividualID`` → ``projid`` → ``subject_id``), reindexes the
    metadata frame onto the SAE-bundle subject order, and emits a per-key
    float64 array (NaN where missing).

    Cognition column resolution prefers ``cogng_random_slope`` first
    (canonical project target — see :class:`ResDecLightningModule`), then
    ``cogn_global`` as fallback, then the literal ``"cognition"`` column.
    """
    import pandas as pd

    df = pd.read_csv(metadata_csv)
    sid_col = None
    for c in ("ROSMAP_IndividualID", "projid", "subject_id"):
        if c in df.columns:
            sid_col = c
            break
    if sid_col is None:
        raise KeyError(
            f"No subject-id column found in {metadata_csv}; expected one of "
            "ROSMAP_IndividualID / projid / subject_id."
        )
    df[sid_col] = df[sid_col].astype(str)
    df = df.set_index(sid_col)

    sids_str = [str(s) for s in subject_ids]
    df_subj = df.reindex(sids_str)

    out: dict[str, np.ndarray] = {
        "subject_ids": np.asarray(sids_str, dtype=object),
    }
    # Cognition: cogng_random_slope is the canonical project target;
    # cogn_global is the fallback for legacy metadata files.
    for canonical_key, candidates in [
        ("cognition", ["cogng_random_slope", "cogn_global", "cognition"]),
        ("global_pathology", ["gpath", "global_pathology", "global_path"]),
        ("amyloid", ["amyloid", "amyloid_sqrt"]),
        ("tau", ["tangles_sqrt", "tau", "tangles"]),
    ]:
        col = next((c for c in candidates if c in df_subj.columns), None)
        if col is None:
            out[canonical_key] = np.full(len(sids_str), np.nan, dtype=np.float64)
        else:
            out[canonical_key] = df_subj[col].astype(np.float64).values
    return out


def expand_fold_idx_to_rows(
    fold_indices: np.ndarray,
    layer: str,
    n_celltypes: int | None,
) -> np.ndarray:
    """Per-row fold indices for the flattened activation matrix.

    For ``layer == "attended"``, returns ``fold_indices`` unchanged
    (one row per subject). For ``layer == "fused"``, repeats each
    subject's fold index ``n_celltypes`` times so the resulting array
    aligns with the ``[N * C, n]`` flattened activation matrix produced
    by ``activations.reshape(N * C, n)``.
    """
    if layer == "attended":
        return np.asarray(fold_indices)
    if layer == "fused":
        if n_celltypes is None:
            raise ValueError(
                "n_celltypes is required when layer='fused' to expand "
                "per-subject fold indices to per-(subject, CT) rows."
            )
        return np.repeat(np.asarray(fold_indices), int(n_celltypes))
    raise ValueError(f"Unsupported layer: {layer!r}")
