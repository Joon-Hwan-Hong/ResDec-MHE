"""Loaders for per-subject precomputed pseudobulk matrices.

Single source of truth for reading the per-subject ``{subject_id}.pt`` files
produced by the data pipeline. The downstream interpretability orchestrators
(distributional, DE, conditional MI on raw pseudobulk) used to each carry a
near-identical local copy of this loader; they should now import from here.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def load_pseudobulk_matrix(
    precomputed_dir: Path,
    subject_ids: list[str],
    *,
    log_every: int = 50,
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
    out: np.ndarray | None = None
    for i, sid in enumerate(subject_ids):
        p = precomputed_dir / f"{sid}.pt"
        if not p.exists():
            logger.warning("missing %s; row will be NaN", p)
            if out is not None:
                out[i] = np.nan
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        pb = d["pseudobulk"].numpy().astype(np.float64)
        if out is None:
            out = np.full((n,) + pb.shape, np.nan, dtype=np.float64)
        out[i] = pb
        if log_every and (i + 1) % log_every == 0:
            logger.info("loaded %d/%d subjects", i + 1, n)
    if out is None:
        raise FileNotFoundError(f"no .pt files loadable from {precomputed_dir}")
    return out
