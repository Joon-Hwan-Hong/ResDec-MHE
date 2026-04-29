"""Tests for scripts/resdec_mhe/training/run_permutation_test_inference_only.py.

Strategy (a) — inference-only permutation test. The model and its predictions
are FIXED; only cogn_global labels are permuted. Verifies:

  1. ``shuffle_finite_labels`` permutes only finite values (NaN preserved).
  2. ``compute_per_fold_r2`` returns r² + n per fold.
  3. End-to-end ``run_permutation_test_inference_only`` writes summary JSON
     with the canonical schema fields:
       null_mean_r2_per_perm, canonical_mean_r2, null_mean, null_std,
       z_under_null, n_perms_ge_canonical, p_value_one_sided.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = (
    _WORKTREE_ROOT
    / "scripts"
    / "resdec_mhe"
    / "training"
    / "run_permutation_test_inference_only.py"
)


def _import_script():
    if str(_WORKTREE_ROOT) not in sys.path:
        sys.path.insert(0, str(_WORKTREE_ROOT))
    spec = importlib.util.spec_from_file_location(
        "run_permutation_test_inference_only", SCRIPT_PATH,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_synthetic_canonical_dir(tmp_path: Path, n_folds: int = 3, n_per_fold: int = 8):
    """Create a fake canonical run with per-fold val_predictions_best.npz.

    Subjects S00 … S{n_folds*n_per_fold-1}; each fold gets a contiguous slice.
    Predictions are ~ targets + noise so canonical R² > 0.
    """
    canonical = tmp_path / "p5_canonical_synthetic"
    canonical.mkdir()
    rng = np.random.default_rng(42)
    all_subjects = [f"S{i:03d}" for i in range(n_folds * n_per_fold)]
    all_targets = rng.standard_normal(len(all_subjects))
    fold_subjects: list[list[str]] = []
    for f in range(n_folds):
        fold_dir = canonical / f"fold{f}"
        fold_dir.mkdir()
        sids = all_subjects[f * n_per_fold:(f + 1) * n_per_fold]
        targets = all_targets[f * n_per_fold:(f + 1) * n_per_fold]
        # Predictions: targets + small noise → R² ~ 0.9
        preds = targets + 0.1 * rng.standard_normal(len(targets))
        np.savez(
            fold_dir / "val_predictions_best.npz",
            subject_ids=np.array(sids, dtype=object),
            predictions=preds.astype(np.float32),
            targets=targets.astype(np.float32),
        )
        fold_subjects.append(sids)
    # metadata.csv with the same subjects (and a few extra NaN-target ones)
    extra_ids = ["NX01", "NX02", "NX03"]  # 3 NaN-target subjects (not in any fold)
    meta = pd.DataFrame({
        "ROSMAP_IndividualID": all_subjects + extra_ids,
        "cogn_global": list(all_targets) + [np.nan] * len(extra_ids),
    })
    meta_csv = tmp_path / "metadata.csv"
    meta.to_csv(meta_csv, index=False)
    return canonical, meta_csv, all_subjects, all_targets


def test_shuffle_finite_labels_preserves_nan(tmp_path):
    """NaN positions must remain NaN; finite values must be a permutation."""
    mod = _import_script()
    df = pd.DataFrame({
        "ROSMAP_IndividualID": ["S0", "S1", "S2", "S3", "S4"],
        "cogn_global": [1.0, np.nan, 3.0, 4.0, np.nan],
    })
    base = tmp_path / "metadata.csv"
    df.to_csv(base, index=False)

    lookup = mod.shuffle_finite_labels(
        base, perm_seed=0, target_col="cogn_global", id_col="ROSMAP_IndividualID",
    )
    # NaN positions must remain NaN.
    assert np.isnan(lookup["S1"])
    assert np.isnan(lookup["S4"])
    # Finite values: a permutation of {1, 3, 4}.
    finite_vals = sorted(
        v for k, v in lookup.items() if np.isfinite(v)
    )
    assert finite_vals == [1.0, 3.0, 4.0]


def test_compute_per_fold_r2_drops_nan_subjects(tmp_path):
    """Subjects with NaN shuffled labels are excluded from R² calculation."""
    mod = _import_script()
    folds = [{
        "subject_ids": ["A", "B", "C", "D"],
        "predictions": np.array([0.1, 0.2, 0.3, 0.4]),
        "targets": np.array([0.1, 0.2, 0.3, 0.4]),
    }]
    y_lookup = {"A": 0.1, "B": np.nan, "C": 0.3, "D": 0.4}
    r2s, ns = mod.compute_per_fold_r2(folds, y_lookup)
    assert ns[0] == 3  # 4 subjects but B has NaN
    assert np.isfinite(r2s[0])


def test_smoke_run_inference_only_n5(tmp_path):
    """Smoke: N=5 perms produce summary JSON with all required schema fields."""
    mod = _import_script()
    canonical_dir, meta_csv, _, _ = _build_synthetic_canonical_dir(
        tmp_path, n_folds=3, n_per_fold=8,
    )
    out_base = tmp_path / "perm_out"

    summary = mod.run_permutation_test_inference_only(
        canonical_dir=canonical_dir,
        base_metadata_csv=meta_csv,
        output_base=out_base,
        num_perms=5,
        start_perm=0,
        n_folds=3,
        target_col="cogn_global",
        id_col="ROSMAP_IndividualID",
        pred_filename="val_predictions_best.npz",
    )

    # Required schema fields per the prereq.
    required = [
        "null_mean_r2_per_perm",
        "canonical_mean_r2",
        "null_mean",
        "null_std",
        "z_under_null",
        "n_perms_ge_canonical",
        "p_value_one_sided",
    ]
    for k in required:
        assert k in summary, f"missing field: {k}"

    # Specific shape checks.
    assert len(summary["null_mean_r2_per_perm"]) == 5
    assert all(isinstance(x, float) for x in summary["null_mean_r2_per_perm"])
    assert summary["n_permutations"] == 5
    # Files written.
    assert (out_base / "permutation_summary.json").exists()
    assert (out_base / "permutation_results.json").exists()
    # Reload + sanity-check summary JSON.
    summary_disk = json.loads((out_base / "permutation_summary.json").read_text())
    assert summary_disk["canonical_mean_r2"] == summary["canonical_mean_r2"]
    assert summary_disk["n_permutations"] == 5

    # Canonical R² should be high (since predictions ≈ targets in the fixture).
    assert summary["canonical_mean_r2"] > 0.5
    # Null mean should be near zero or negative (random shuffling kills signal).
    assert summary["null_mean"] < summary["canonical_mean_r2"]


def test_canonical_dir_missing_files_raises(tmp_path):
    """Non-existent canonical predictions raise a clear error."""
    mod = _import_script()
    with pytest.raises(FileNotFoundError):
        mod.load_canonical_predictions(
            canonical_dir=tmp_path / "does_not_exist",
            n_folds=5,
            pred_filename="val_predictions_best.npz",
        )


def test_p_value_floor_at_one_over_n_plus_one():
    """Empirical p-value floor is 1/(N+1) when no perm beats canonical."""
    # Inline math check, no I/O.
    n_perms = 5
    n_ge = 0
    p = (1 + n_ge) / (n_perms + 1)
    assert p == pytest.approx(1 / 6)
