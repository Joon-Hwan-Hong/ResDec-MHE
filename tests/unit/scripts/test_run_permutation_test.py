"""Tests for scripts/resdec_mhe/training/run_permutation_test.py.

Currently exercises the NaN-preservation contract on
``generate_shuffled_metadata``: NaN positions are invariant under shuffle;
only finite values are permuted among the non-NaN positions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add the repo root so we can import the script's helpers.
ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "scripts" / "resdec_mhe" / "training"
sys.path.insert(0, str(SCRIPTS))

import importlib.util


def _import_script():
    spec = importlib.util.spec_from_file_location(
        "run_permutation_test", SCRIPTS / "run_permutation_test.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_generate_shuffled_metadata_preserves_nan_positions(tmp_path):
    """NaN rows in the target column must stay NaN after the shuffle."""
    mod = _import_script()
    base = tmp_path / "metadata.csv"
    df = pd.DataFrame({
        "ROSMAP_IndividualID": [f"R{i}" for i in range(20)],
        "cogn_global": [
            1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 7.0, 8.0, 9.0, 10.0,
            11.0, 12.0, np.nan, 14.0, 15.0, 16.0, 17.0, np.nan, 19.0, 20.0,
        ],
    })
    df.to_csv(base, index=False)
    out = tmp_path / "permuted.csv"
    mod.generate_shuffled_metadata(
        perm_seed=42, base_csv=base, target_col="cogn_global", out_csv=out,
    )
    permuted = pd.read_csv(out)
    expected_nan_idx = [2, 5, 12, 17]
    nan_mask = permuted["cogn_global"].isna().values
    assert list(np.flatnonzero(nan_mask)) == expected_nan_idx
    # Non-NaN values must be a permutation of the original non-NaN values.
    orig_finite = sorted(df["cogn_global"].dropna().tolist())
    perm_finite = sorted(permuted["cogn_global"].dropna().tolist())
    assert orig_finite == perm_finite


def test_generate_shuffled_metadata_actually_shuffles(tmp_path):
    """The non-NaN entries must not all be in their original positions."""
    mod = _import_script()
    base = tmp_path / "metadata.csv"
    df = pd.DataFrame({
        "ROSMAP_IndividualID": [f"R{i}" for i in range(50)],
        "cogn_global": np.arange(50, dtype=float),
    })
    df.to_csv(base, index=False)
    out = tmp_path / "permuted.csv"
    mod.generate_shuffled_metadata(
        perm_seed=42, base_csv=base, target_col="cogn_global", out_csv=out,
    )
    permuted = pd.read_csv(out)
    # With seed 42 and 50 elements the probability that even one stays in
    # place is high, so we just check at least one moved.
    assert not (permuted["cogn_global"].values == df["cogn_global"].values).all()


def test_generate_shuffled_metadata_deterministic_under_seed(tmp_path):
    """Same seed must produce identical permutation."""
    mod = _import_script()
    base = tmp_path / "metadata.csv"
    df = pd.DataFrame({
        "ROSMAP_IndividualID": [f"R{i}" for i in range(20)],
        "cogn_global": np.arange(20, dtype=float),
    })
    df.to_csv(base, index=False)
    out_a = tmp_path / "a.csv"
    out_b = tmp_path / "b.csv"
    mod.generate_shuffled_metadata(
        perm_seed=42, base_csv=base, target_col="cogn_global", out_csv=out_a,
    )
    mod.generate_shuffled_metadata(
        perm_seed=42, base_csv=base, target_col="cogn_global", out_csv=out_b,
    )
    a = pd.read_csv(out_a)
    b = pd.read_csv(out_b)
    np.testing.assert_array_equal(a["cogn_global"].values, b["cogn_global"].values)
