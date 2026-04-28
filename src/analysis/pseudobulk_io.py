"""Loaders for per-subject precomputed pseudobulk matrices.

Single source of truth for reading the per-subject ``{subject_id}.pt`` files
produced by the data pipeline. The downstream interpretability orchestrators
(distributional, DE, conditional MI on raw pseudobulk) used to each carry a
near-identical local copy of this loader; they should now import from here.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)


def _load_one_subject(
    sid: str, precomputed_dir: Path,
) -> tuple[str, np.ndarray | None]:
    """Worker for the F13 threaded loader. Returns ``(sid, pb_or_None)``.

    ``None`` indicates a missing ``.pt`` file (caller fills the row with NaN).
    """
    p = precomputed_dir / f"{sid}.pt"
    if not p.exists():
        logger.warning("missing %s; row will be NaN", p)
        return sid, None
    d = torch.load(p, map_location="cpu", weights_only=False)
    pb = d["pseudobulk"].numpy().astype(np.float64)
    return sid, pb


def load_pseudobulk_matrix(
    precomputed_dir: Path,
    subject_ids: list[str],
    *,
    log_every: int = 50,
    n_jobs: int | None = None,
) -> np.ndarray:
    """Load per-subject pseudobulk into shape ``(n_subjects, n_cell_types, n_genes)``.

    Subjects whose ``.pt`` file is missing get an all-NaN row in the output.
    A warning is emitted per missing subject so that callers can spot
    cohort-vs-precomputed mismatches.

    Parameters
    ----------
    precomputed_dir
        Directory containing ``{subject_id}.pt`` files; each is a torch dict
        with a ``"pseudobulk"`` tensor of shape ``(n_cell_types, n_genes)``.
    subject_ids
        Subject IDs to assemble the output around, in row order.
    log_every
        Emit a progress info log every this many subjects (default 50;
        set to 0 to silence).
    n_jobs
        Number of threads for the per-subject ``torch.load`` loop. Default
        ``None`` resolves to ``min(8, os.cpu_count())``; pass ``1`` to force
        the legacy serial path. ``torch.load`` releases the GIL during disk
        I/O and tensor construction, so threading materially reduces wall
        time on warm-cache cohorts (verified bit-equivalent vs serial in the
        unit test suite — F13).

    Returns
    -------
    np.ndarray
        Shape ``(len(subject_ids), n_cell_types, n_genes)``, dtype float64.

    Raises
    ------
    FileNotFoundError
        If NO ``.pt`` files are loadable in ``precomputed_dir``.
    """
    n = len(subject_ids)
    if n_jobs is None:
        n_jobs = min(8, os.cpu_count() or 1)
    n_jobs = max(1, int(n_jobs))

    if n_jobs == 1:
        # Legacy serial path retained for back-compat (and fast unit tests).
        out: np.ndarray | None = None
        for i, sid in enumerate(subject_ids):
            _, pb = _load_one_subject(sid, precomputed_dir)
            if pb is None:
                if out is not None:
                    out[i] = np.nan
                continue
            if out is None:
                out = np.full((n,) + pb.shape, np.nan, dtype=np.float64)
            out[i] = pb
            if log_every and (i + 1) % log_every == 0:
                logger.info("loaded %d/%d subjects", i + 1, n)
        if out is None:
            raise FileNotFoundError(f"no .pt files loadable from {precomputed_dir}")
        return out

    # F13: parallel path. Threading is safe because torch.load is read-only
    # and fills new tensors per call; ordering of the result is preserved by
    # iterating over the joblib output in submission order.
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_load_one_subject)(sid, precomputed_dir) for sid in subject_ids
    )

    out_arr: np.ndarray | None = None
    for i, (_sid, pb) in enumerate(results):
        if pb is None:
            if out_arr is not None:
                out_arr[i] = np.nan
            continue
        if out_arr is None:
            out_arr = np.full((n,) + pb.shape, np.nan, dtype=np.float64)
        out_arr[i] = pb
        if log_every and (i + 1) % log_every == 0:
            logger.info("loaded %d/%d subjects", i + 1, n)
    if out_arr is None:
        raise FileNotFoundError(f"no .pt files loadable from {precomputed_dir}")
    return out_arr
