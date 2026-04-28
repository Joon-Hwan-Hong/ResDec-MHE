"""Tests for src/analysis/pseudobulk_io.py.

The F13 optimisation parallelises the per-subject ``torch.load`` loop with
``joblib.Parallel(prefer="threads")``. The threaded result must match the
serial baseline element-for-element on every subject (no bit-rot, no row
re-ordering, missing-file rows fall back to NaN).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.analysis.pseudobulk_io import load_pseudobulk_matrix


def _write_subject(tmp_dir: Path, sid: str, pb: np.ndarray) -> None:
    torch.save({"pseudobulk": torch.from_numpy(pb)}, tmp_dir / f"{sid}.pt")


def test_load_pseudobulk_matrix_parallel_matches_serial(tmp_path: Path):
    rng = np.random.default_rng(0)
    sids = [f"R{i:04d}" for i in range(10)]
    payloads = {sid: rng.standard_normal(size=(3, 5)).astype(np.float64) for sid in sids}
    for sid, pb in payloads.items():
        _write_subject(tmp_path, sid, pb)

    serial = load_pseudobulk_matrix(tmp_path, sids, n_jobs=1, log_every=0)
    parallel = load_pseudobulk_matrix(tmp_path, sids, n_jobs=4, log_every=0)

    assert serial.shape == (10, 3, 5)
    np.testing.assert_array_equal(serial, parallel)


def test_load_pseudobulk_matrix_missing_subject_fills_nan(tmp_path: Path):
    rng = np.random.default_rng(0)
    sids = [f"R{i:04d}" for i in range(4)]
    # Write only 3 of 4 subjects.
    for sid in sids[:3]:
        _write_subject(tmp_path, sid, rng.standard_normal(size=(2, 3)).astype(np.float64))

    out = load_pseudobulk_matrix(tmp_path, sids, n_jobs=2, log_every=0)
    assert out.shape == (4, 2, 3)
    assert np.isnan(out[3]).all()
    assert np.isfinite(out[:3]).all()


def test_load_pseudobulk_matrix_no_files_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_pseudobulk_matrix(tmp_path, ["R0000"], n_jobs=2, log_every=0)
