"""Unit + integration tests for ``run_coverage_stratified_loco.py``.

The integration smoke test runs the full script for a single fold against
real canonical artefacts (precomputed .pt caches, splits, checkpoint,
TabPFN outer NPZs). Pure-Python helpers are tested with synthetic
fold payloads that exercise the full-cohort vs restricted-cohort
ΔR² semantics, rank-shift detection, and the JSON/MD writers.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

# Import the module under test using importlib (script lives in scripts/).
import importlib.util  # noqa: E402

_SCRIPT_PATH = (
    _WORKTREE_ROOT
    / "scripts/resdec_mhe/interpretability/run_coverage_stratified_loco.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_coverage_stratified_loco", _SCRIPT_PATH,
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# -----------------------------------------------------------------------
# Pure-Python helper tests (no GPU / disk required)
# -----------------------------------------------------------------------


def _make_synthetic_fold_payloads(
    n_folds: int = 2, n_val: int = 4, n_ct: int = 3, seed: int = 0,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Build a synthetic 2-fold payload + matching subject_counts dict.

    Subject layout (8 unique SIDs, 4 per fold):

      fold0: S0, S1, S2, S3
      fold1: S4, S5, S6, S7

    cell_counts (shape [n_ct]) per subject:
      S0: [10, 0, 5]   — has CT0, CT2, no CT1
      S1: [0, 8, 4]    — has CT1, CT2, no CT0
      S2: [12, 9, 0]   — has CT0, CT1, no CT2
      S3: [3, 0, 0]    — has only CT0
      S4: [0, 7, 0]    — has only CT1
      S5: [0, 0, 11]   — has only CT2
      S6: [4, 5, 0]    — has CT0, CT1
      S7: [0, 0, 0]    — has none

    Predictions (composite = residual + tabpfn) are constructed so that:
      * canonical preds nearly match true_y (high R²)
      * zero-out of CT0 hurts in fold0 only (subjects with CT0 cells)
    """
    rng = np.random.default_rng(seed)
    sids_f0 = ["S0", "S1", "S2", "S3"]
    sids_f1 = ["S4", "S5", "S6", "S7"]
    counts_dict = {
        "S0": np.array([10, 0, 5], dtype=np.int64),
        "S1": np.array([0, 8, 4], dtype=np.int64),
        "S2": np.array([12, 9, 0], dtype=np.int64),
        "S3": np.array([3, 0, 0], dtype=np.int64),
        "S4": np.array([0, 7, 0], dtype=np.int64),
        "S5": np.array([0, 0, 11], dtype=np.int64),
        "S6": np.array([4, 5, 0], dtype=np.int64),
        "S7": np.array([0, 0, 0], dtype=np.int64),
    }

    payloads: list[dict] = []
    for f, sids in enumerate([sids_f0, sids_f1]):
        true_y = rng.normal(loc=0.0, scale=1.0, size=n_val)
        # canonical near-perfect: small noise.
        comp_canon = true_y + rng.normal(0, 0.05, size=n_val)
        # LOCO: per CT, perturb only subjects that have that CT.
        comp_loco = np.zeros((n_ct, n_val), dtype=np.float64)
        for ct in range(n_ct):
            comp_loco[ct] = comp_canon.copy()
            for i, sid in enumerate(sids):
                if counts_dict[sid][ct] > 0:
                    # Inject a 0.5 magnitude residual error.
                    comp_loco[ct, i] += 0.5
        payloads.append({
            "fold": f,
            "subject_ids": np.asarray(sids, dtype=object),
            "true_y": true_y.astype(np.float64),
            "comp_canon": comp_canon.astype(np.float64),
            "comp_loco": comp_loco,
        })
    return payloads, counts_dict


def test_aggregate_loco_full_vs_restricted_basic():
    mod = _load_module()
    payloads, counts = _make_synthetic_fold_payloads()
    out = mod.aggregate_loco(payloads, counts, n_cell_types=3, min_cells_threshold=1)
    assert out["n_folds"] == 2
    assert len(out["per_cell_type"]) == 3

    canon_pf = out["canonical_per_fold_full"]
    assert len(canon_pf) == 2
    # Canonical R² should be very high (preds = true + 0.05 noise).
    for v in canon_pf:
        assert v > 0.9, v

    for row in out["per_cell_type"]:
        # Full-cohort always defined.
        assert isinstance(row["delta_r2_full"], float)
        # Restricted may be None for tiny CT subsets — but with our data
        # every CT has >= 2 subjects (in at least 1 fold), so the
        # aggregate restricted ΔR² should be defined.
        assert row["delta_r2_restricted"] is not None
        # Each per-fold restricted entry is None or a float.
        assert len(row["delta_r2_restricted_per_fold"]) == 2
        for v in row["delta_r2_restricted_per_fold"]:
            assert v is None or isinstance(v, float)


def test_aggregate_loco_canonical_recovers_per_fold_r2():
    """Canonical R² should equal r2_score(true_y, comp_canon) per fold."""
    mod = _load_module()
    from sklearn.metrics import r2_score
    payloads, counts = _make_synthetic_fold_payloads()
    out = mod.aggregate_loco(
        payloads, counts, n_cell_types=3, min_cells_threshold=1,
    )
    for f, fp in enumerate(payloads):
        expect = float(r2_score(fp["true_y"], fp["comp_canon"]))
        got = out["canonical_per_fold_full"][f]
        assert abs(got - expect) < 1e-9, (got, expect)


def test_aggregate_loco_restricted_paired_semantics():
    """Restricted ΔR² for fold f and CT ct equals r2(loco) - r2(canon) on
    the subset of fold-f val subjects with cell_counts[ct] >= 1."""
    mod = _load_module()
    from sklearn.metrics import r2_score
    payloads, counts = _make_synthetic_fold_payloads()
    out = mod.aggregate_loco(
        payloads, counts, n_cell_types=3, min_cells_threshold=1,
    )
    for ct_row in out["per_cell_type"]:
        ct = ct_row["cell_type_index"]
        for f, fp in enumerate(payloads):
            keep = np.array(
                [counts[sid][ct] >= 1 for sid in fp["subject_ids"]],
                dtype=bool,
            )
            n_keep = int(keep.sum())
            assert ct_row["n_val_restricted"][f] == n_keep
            if n_keep < 2:
                assert ct_row["delta_r2_restricted_per_fold"][f] is None
                continue
            r2_canon_r = r2_score(
                fp["true_y"][keep], fp["comp_canon"][keep],
            )
            r2_loco_r = r2_score(
                fp["true_y"][keep], fp["comp_loco"][ct][keep],
            )
            expect = r2_loco_r - r2_canon_r
            got = ct_row["delta_r2_restricted_per_fold"][f]
            assert got is not None and abs(got - expect) < 1e-9


def test_compute_rank_shift_handles_none_restricted():
    """CTs whose restricted ΔR² is None should not appear in restricted ranking."""
    mod = _load_module()
    aggregated = {
        "per_cell_type": [
            {"cell_type_index": 0, "cell_type": "A",
             "delta_r2_full": -0.05, "delta_r2_restricted": None},
            {"cell_type_index": 1, "cell_type": "B",
             "delta_r2_full": -0.02, "delta_r2_restricted": -0.03},
            {"cell_type_index": 2, "cell_type": "C",
             "delta_r2_full": +0.01, "delta_r2_restricted": +0.02},
        ],
    }
    mod.compute_rank_shift(aggregated)
    rows = aggregated["per_cell_type"]
    # full ranks: A=1 (most negative), B=2, C=3.
    assert rows[0]["full_rank"] == 1
    assert rows[1]["full_rank"] == 2
    assert rows[2]["full_rank"] == 3
    # restricted ranks: A skipped, B=1, C=2.
    assert rows[0]["restricted_rank"] is None
    assert rows[0]["rank_shift"] is None
    assert rows[1]["restricted_rank"] == 1
    assert rows[1]["rank_shift"] == 2 - 1  # full_rank - rest_rank = 1
    assert rows[2]["restricted_rank"] == 2
    assert aggregated["n_cell_types_with_valid_restricted"] == 2


def test_merge_coverage_stats_attaches_fields(tmp_path: Path):
    mod = _load_module()
    cov_path = tmp_path / "cov.json"
    cov_path.write_text(json.dumps({
        "n_subjects": 100,
        "per_ct": {
            "A": {"median_cells": 50, "n_subj_with_cells": 100,
                  "zero_frac": 0.0, "q90_cells": 200,
                  "total_cells": 5000},
            "B": {"median_cells": 0, "n_subj_with_cells": 5,
                  "zero_frac": 0.95, "q90_cells": 0,
                  "total_cells": 12},
        },
    }))
    aggregated = {
        "per_cell_type": [
            {"cell_type_index": 0, "cell_type": "A",
             "delta_r2_full": -0.01, "delta_r2_restricted": -0.01},
            {"cell_type_index": 1, "cell_type": "B",
             "delta_r2_full": +0.001, "delta_r2_restricted": None},
        ],
    }
    mod.merge_coverage_stats(aggregated, cov_path)
    assert aggregated["per_cell_type"][0]["median_cells"] == 50
    assert aggregated["per_cell_type"][0]["zero_frac"] == 0.0
    assert aggregated["per_cell_type"][1]["zero_frac"] == 0.95
    assert aggregated["coverage_n_subjects_total"] == 100


def test_write_json_and_md_round_trip(tmp_path: Path):
    mod = _load_module()
    aggregated = {
        "n_folds": 5,
        "canonical_per_fold_full": [0.4, 0.5, 0.6, 0.45, 0.5],
        "canonical_mean_full": 0.49,
        "n_cell_types_with_valid_restricted": 2,
        "coverage_n_subjects_total": 516,
        "per_cell_type": [
            {
                "cell_type_index": 0, "cell_type": "Splatter",
                "loco_per_fold_full": [0.39, 0.49, 0.59, 0.44, 0.49],
                "loco_mean_full": 0.48,
                "delta_r2_full": -0.01,
                "delta_r2_full_per_fold": [-0.01]*5,
                "n_val_full": [104]*5,
                "loco_per_fold_restricted": [0.5, 0.6, 0.7, 0.5, 0.6],
                "canonical_per_fold_restricted": [0.51, 0.61, 0.71, 0.51, 0.61],
                "delta_r2_restricted": -0.01,
                "delta_r2_restricted_per_fold": [-0.01]*5,
                "n_val_restricted": [80]*5,
                "n_val_restricted_total": 400,
                "median_cells": 2, "n_subj_with_cells": 437,
                "zero_frac": 0.153,
                "q90_cells": 103, "total_cells": 18595,
                "full_rank": 1, "restricted_rank": 1, "rank_shift": 0,
            },
            {
                "cell_type_index": 1, "cell_type": "Cerebellar inhibitory",
                "loco_per_fold_full": [0.42, 0.52, 0.62, 0.47, 0.52],
                "loco_mean_full": 0.51,
                "delta_r2_full": 0.02,
                "delta_r2_full_per_fold": [0.02]*5,
                "n_val_full": [104]*5,
                "loco_per_fold_restricted": [None]*5,
                "canonical_per_fold_restricted": [None]*5,
                "delta_r2_restricted": None,
                "delta_r2_restricted_per_fold": [None]*5,
                "n_val_restricted": [0]*5,
                "n_val_restricted_total": 0,
                "median_cells": 0, "n_subj_with_cells": 66,
                "zero_frac": 0.872,
                "q90_cells": 1, "total_cells": 182,
                "full_rank": 2, "restricted_rank": None, "rank_shift": None,
            },
        ],
    }
    json_path = tmp_path / "out.json"
    md_path = tmp_path / "out.md"
    mod.write_json(aggregated, json_path)
    mod.write_md(aggregated, md_path)
    assert json_path.is_file() and json_path.stat().st_size > 100
    payload = json.loads(json_path.read_text())
    assert payload["n_folds"] == 5
    assert len(payload["per_cell_type"]) == 2
    assert md_path.is_file() and md_path.stat().st_size > 100
    md_text = md_path.read_text()
    assert "Splatter" in md_text
    assert "Cerebellar inhibitory" in md_text


def test_build_subject_cell_counts_smoke(tmp_path: Path):
    """Build subject cell_counts map from real precomputed dir (smoke)."""
    mod = _load_module()
    precomp = _WORKTREE_ROOT / "data/precomputed"
    if not precomp.is_dir() or not any(precomp.glob("R*.pt")):
        pytest.skip("data/precomputed/R*.pt not available")
    sids, counts = mod.build_subject_cell_counts(precomp, n_cell_types=31)
    assert len(sids) == 516
    assert counts.shape == (516, 31)
    assert counts.dtype == np.int64
    # Sanity: total cells should match sum across CTs (with cell_type_mask
    # — but cell_counts are already int64 totals per CT).
    assert counts.sum() > 0


# -----------------------------------------------------------------------
# Integration smoke test — single fold via the script CLI.
# -----------------------------------------------------------------------


@pytest.mark.slow
def test_script_runs_smoke_one_fold(tmp_path: Path):
    pred_root = _WORKTREE_ROOT / "outputs/canonical/p5_canonical_seed42"
    precomp_dir = _WORKTREE_ROOT / "data/precomputed"
    cov_json = (
        _WORKTREE_ROOT
        / "outputs/canonical/interpretability/ct_coverage_full_cohort.json"
    )
    splits_json = _WORKTREE_ROOT / "outputs/splits.json"
    fold0_ckpt = list((pred_root / "fold0/checkpoints").glob("best-*.ckpt"))
    if not fold0_ckpt or not precomp_dir.is_dir() or not cov_json.is_file():
        pytest.skip("Required canonical artefacts missing.")

    out_data = tmp_path / "data"
    out_fig = tmp_path / "fig"

    cmd = [
        sys.executable,
        str(_SCRIPT_PATH),
        "--precomputed-dir", str(precomp_dir),
        "--coverage-json", str(cov_json),
        "--splits-path", str(splits_json),
        "--canonical-dir", str(pred_root),
        "--out-data-dir", str(out_data),
        "--out-fig-dir", str(out_fig),
        "--smoke-fold-only", "0",
        "--device", "cuda:1",
    ]
    res = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(_WORKTREE_ROOT),
        timeout=900,
    )
    assert res.returncode == 0, (
        f"Script failed.\nstdout: {res.stdout}\nstderr: {res.stderr}"
    )
    json_path = out_data / "coverage_stratified_loco.json"
    md_path = out_data / "coverage_stratified_loco.md"
    png_path = out_fig / "fig_coverage_stratified_loco.png"
    pdf_path = out_fig / "fig_coverage_stratified_loco.pdf"
    assert json_path.is_file() and json_path.stat().st_size > 1000
    assert md_path.is_file() and md_path.stat().st_size > 200
    assert png_path.is_file() and png_path.stat().st_size > 1000
    assert pdf_path.is_file() and pdf_path.stat().st_size > 1000

    payload = json.loads(json_path.read_text())
    # smoke-fold-only=0 means we iterate exactly one fold in the result.
    assert payload["n_folds"] == 1
    assert len(payload["per_cell_type"]) == 31
    assert payload["canonical_mean_full"] is not None
    assert len(payload["canonical_per_fold_full"]) == 1
