"""Unit tests for scripts/redesign/interpretability/make_baseline_table.py.

Tests cover:
- Discovery of redesign ablation dirs via glob
- Parsing of best_vs_tabpfn_summary.json schema (per-fold ours metrics)
- Parsing of classical baseline CSV (Ridge/ElasticNet/PLS/RF/XGBoost)
- Parsing of DL baseline CSVs (cloudpred/gpio/perceiver_io; 1-indexed folds)
- Missing baselines produce NaN rows with a "missing" note (not crash)
- Missing ablations (e.g. D.2 pending) produce a NaN row with a "pending" note
- Aggregation computes mean/std with ddof=1 across available folds
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# Import the script as a module without having to pip install; matches
# the approach used by other interpretability scripts' tests.
_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT_PATH = (
    _WORKTREE_ROOT
    / "scripts" / "redesign" / "interpretability" / "make_baseline_table.py"
)


def _import_script_module():
    """Dynamically load make_baseline_table.py without installing the package."""
    if str(_WORKTREE_ROOT) not in sys.path:
        sys.path.insert(0, str(_WORKTREE_ROOT))
    spec = importlib.util.spec_from_file_location(
        "make_baseline_table", _SCRIPT_PATH,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_summary_json(path: Path, per_fold_r2s: list[float]) -> None:
    """Write a minimal best_vs_tabpfn_summary.json with the given per-fold R²s."""
    per_fold = []
    for i, r2 in enumerate(per_fold_r2s):
        per_fold.append({
            "fold": i,
            "n": 100 + i,
            "ours": {
                "r2": r2,
                "mae": 0.7 + 0.01 * i,
                "rmse": 0.9 + 0.01 * i,
                "pearson_r": 0.6 + 0.01 * i,
                "spearman_rho": 0.55 + 0.01 * i,
            },
            "tab_ge": {
                "r2": 0.4, "mae": 0.7, "rmse": 0.9,
                "pearson_r": 0.6, "spearman_rho": 0.55,
            },
            "tab_en": {
                "r2": 0.4, "mae": 0.7, "rmse": 0.9,
                "pearson_r": 0.6, "spearman_rho": 0.55,
            },
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "per_fold": per_fold,
        "outroot": str(path.parent),
        "tabpfn_dir": "data/redesign",
    }))


# ---------------------------------------------------------------------------
# Summary-JSON parsing
# ---------------------------------------------------------------------------

def test_parse_summary_json_extracts_all_five_metrics(tmp_path: Path) -> None:
    """A well-formed summary JSON → 5 per-fold arrays, one per metric."""
    mod = _import_script_module()
    path = tmp_path / "summary.json"
    _write_summary_json(path, [0.1, 0.2, 0.3, 0.4, 0.5])

    metrics = mod.parse_summary_json(path)

    assert set(metrics) == {"r2", "mae", "rmse", "pearson_r", "spearman_rho"}
    np.testing.assert_allclose(metrics["r2"], [0.1, 0.2, 0.3, 0.4, 0.5])
    # MAE starts at 0.70 and increments by 0.01 per fold.
    np.testing.assert_allclose(
        metrics["mae"], [0.70, 0.71, 0.72, 0.73, 0.74], rtol=0, atol=1e-9,
    )


def test_parse_summary_json_missing_file_returns_none(tmp_path: Path) -> None:
    """A missing summary JSON should return None (not raise)."""
    mod = _import_script_module()
    out = mod.parse_summary_json(tmp_path / "does_not_exist.json")
    assert out is None


def test_parse_summary_json_malformed_returns_none(tmp_path: Path) -> None:
    """A malformed JSON should return None (caller logs WARNING)."""
    mod = _import_script_module()
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json]}")
    out = mod.parse_summary_json(bad)
    assert out is None


# ---------------------------------------------------------------------------
# Ablation discovery
# ---------------------------------------------------------------------------

def test_discover_ablations_picks_up_all_p5_dirs(tmp_path: Path) -> None:
    """Glob should discover all p5_* dirs with a best_vs_tabpfn_summary.json."""
    mod = _import_script_module()
    # Layout: three redesign dirs, one with summary, one without, one unrelated.
    (tmp_path / "p5_canonical_seed42").mkdir()
    (tmp_path / "p5_ablation_k1").mkdir()
    (tmp_path / "p5_ablation_no_film").mkdir()
    (tmp_path / "unrelated_dir").mkdir()

    _write_summary_json(
        tmp_path / "p5_canonical_seed42" / "best_vs_tabpfn_summary.json",
        [0.4, 0.4, 0.4, 0.4, 0.4],
    )
    _write_summary_json(
        tmp_path / "p5_ablation_k1" / "best_vs_tabpfn_summary.json",
        [0.3, 0.3, 0.3, 0.3, 0.3],
    )
    # p5_ablation_no_film has no summary → should NOT appear.
    # unrelated_dir is filtered by the p5_* prefix → should NOT appear.

    found = mod.discover_ablation_dirs(tmp_path)
    found_names = sorted(p.name for p in found)

    assert "p5_canonical_seed42" in found_names
    assert "p5_ablation_k1" in found_names
    assert "p5_ablation_no_film" not in found_names
    assert "unrelated_dir" not in found_names


# ---------------------------------------------------------------------------
# Classical baseline CSV parsing
# ---------------------------------------------------------------------------

def _write_classical_csv(path: Path) -> None:
    """Minimal classical_baseline CSV: Ridge + XGBoost on C + A+C+E."""
    rows = []
    for model in ("Ridge", "XGBoost"):
        for feature_set in ("C", "A+C+E"):
            for fold in range(5):
                rows.append({
                    "model": model,
                    "feature_set": feature_set,
                    "fold": fold,
                    # Use a deterministic function so the test is data-robust.
                    "r2": 0.1 * fold + (0.1 if model == "XGBoost" else 0.0),
                    "mae": 0.9 - 0.01 * fold,
                    "rmse": 1.0 - 0.01 * fold,
                    "pearson_r": 0.3 + 0.02 * fold,
                    "spearman_rho": 0.25 + 0.02 * fold,
                    "best_params": "{}",
                })
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_parse_classical_csv_one_row_per_model_featureset(tmp_path: Path) -> None:
    """Classical CSV parser → one row per (model, feature_set) with all metrics."""
    mod = _import_script_module()
    csv_path = tmp_path / "classical.csv"
    _write_classical_csv(csv_path)

    rows = mod.parse_classical_csv(csv_path)

    # 2 models × 2 feature sets = 4 rows.
    assert len(rows) == 4
    keys = sorted((r["model"], r["feature_set"]) for r in rows)
    assert keys == [
        ("Ridge", "A+C+E"),
        ("Ridge", "C"),
        ("XGBoost", "A+C+E"),
        ("XGBoost", "C"),
    ]
    # All rows should have 5 folds of metrics.
    for r in rows:
        for k in ("r2", "mae", "rmse", "pearson_r", "spearman_rho"):
            assert len(r["metrics"][k]) == 5
            assert all(math.isfinite(v) for v in r["metrics"][k])


def test_parse_classical_csv_missing_returns_empty(tmp_path: Path) -> None:
    """Missing classical CSV → empty list, not crash."""
    mod = _import_script_module()
    rows = mod.parse_classical_csv(tmp_path / "does_not_exist.csv")
    assert rows == []


# ---------------------------------------------------------------------------
# DL baseline CSV parsing (cloudpred/gpio/perceiver_io)
# ---------------------------------------------------------------------------

def _write_dl_baseline_csv(path: Path, folds_one_indexed: bool = True) -> None:
    """Minimal DL baseline results CSV (1-indexed folds by default)."""
    rows = []
    fold_ids = range(1, 6) if folds_one_indexed else range(5)
    for i, fold in enumerate(fold_ids):
        rows.append({
            "r2": 0.05 + 0.01 * i,
            "mae": 0.9 - 0.01 * i,
            "rmse": 1.1 - 0.01 * i,
            "pearson_r": 0.25 + 0.02 * i,
            "spearman_rho": 0.2 + 0.02 * i,
            "fold": fold,
            "train_time_s": 500 + 10 * i,
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_parse_dl_baseline_csv_five_folds(tmp_path: Path) -> None:
    """DL baseline → one dict with 5 per-fold metric arrays."""
    mod = _import_script_module()
    csv_path = tmp_path / "results.csv"
    _write_dl_baseline_csv(csv_path)

    out = mod.parse_dl_baseline_csv(csv_path)
    assert out is not None
    for k in ("r2", "mae", "rmse", "pearson_r", "spearman_rho"):
        assert len(out[k]) == 5


def test_parse_dl_baseline_csv_missing_returns_none(tmp_path: Path) -> None:
    mod = _import_script_module()
    out = mod.parse_dl_baseline_csv(tmp_path / "missing.csv")
    assert out is None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def test_summarise_row_computes_mean_std() -> None:
    """summarise_row → mean + std (ddof=1) for each metric."""
    mod = _import_script_module()
    metrics = {
        "r2": [0.1, 0.2, 0.3, 0.4, 0.5],
        "mae": [1.0, 1.0, 1.0, 1.0, 1.0],
        "rmse": [0.9, 0.9, 0.9, 0.9, 0.9],
        "pearson_r": [0.5, 0.5, 0.5, 0.5, 0.5],
        "spearman_rho": [0.4, 0.4, 0.4, 0.4, 0.4],
    }
    s = mod.summarise_row(metrics)
    assert s["n_folds"] == 5
    assert math.isclose(s["r2_mean"], 0.3)
    # Std (ddof=1) of [0.1, 0.2, 0.3, 0.4, 0.5] ≈ 0.1581139.
    assert math.isclose(s["r2_std"], float(np.std([0.1, 0.2, 0.3, 0.4, 0.5], ddof=1)))
    # Constant metrics have std 0.
    assert math.isclose(s["mae_std"], 0.0)


def test_summarise_row_empty_returns_nans() -> None:
    """summarise_row on empty metrics → all NaN, n_folds=0."""
    mod = _import_script_module()
    metrics = {k: [] for k in ("r2", "mae", "rmse", "pearson_r", "spearman_rho")}
    s = mod.summarise_row(metrics)
    assert s["n_folds"] == 0
    assert math.isnan(s["r2_mean"])
    assert math.isnan(s["r2_std"])


# ---------------------------------------------------------------------------
# End-to-end row assembly with missing ablations
# ---------------------------------------------------------------------------

def test_ablation_row_falls_back_to_per_fold_npz(tmp_path: Path) -> None:
    """When summary.json is absent but per-fold npz files exist, use them."""
    mod = _import_script_module()
    subdir = tmp_path / "p5_phase3_3stage"
    # No best_vs_tabpfn_summary.json — only per-fold npz files.
    for f in range(5):
        fold_dir = subdir / f"fold{f}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            fold_dir / "val_predictions_best.npz",
            subject_ids=np.array(["s0", "s1", "s2"], dtype=object),
            predictions=np.array([0.0, 1.0, 2.0], dtype=np.float32),
            targets=np.array([0.0, 1.0, 2.0], dtype=np.float32),
            epoch=np.array(10),
            mse=np.array(0.0),
            mae=np.array(0.01 * f),
            rmse=np.array(0.02 * f),
            r2=np.array(0.1 + 0.1 * f),
            pearson_r=np.array(0.5 + 0.05 * f),
            spearman_rho=np.array(0.4 + 0.05 * f),
        )

    rows = mod.collect_ablation_rows(
        ablation_root=tmp_path,
        requested=[("p5_phase3_3stage", "n=3 ablation")],
    )
    assert len(rows) == 1
    r = rows[0]
    assert r["n_folds"] == 5
    # r2 = [0.1, 0.2, 0.3, 0.4, 0.5] → mean 0.3.
    assert math.isclose(r["r2_mean"], 0.3)
    assert "val_predictions_best.npz" in r["source_path"]


def test_missing_ablation_produces_nan_row_with_pending_note(tmp_path: Path) -> None:
    """A requested ablation dir that doesn't exist → NaN row + 'pending' note."""
    mod = _import_script_module()
    # Only canonical exists; request an ablation that does not exist on disk.
    (tmp_path / "p5_canonical_seed42").mkdir()
    _write_summary_json(
        tmp_path / "p5_canonical_seed42" / "best_vs_tabpfn_summary.json",
        [0.4, 0.4, 0.4, 0.4, 0.4],
    )

    rows = mod.collect_ablation_rows(
        ablation_root=tmp_path,
        requested=[
            ("p5_canonical_seed42", "ResDec-H3 (canonical)"),
            ("p5_ablation_topk_4000", "top-k=4000 (D.2)"),  # not on disk
        ],
    )
    by_model = {r["model"]: r for r in rows}

    # Canonical row is fully populated.
    assert by_model["p5_canonical_seed42"]["n_folds"] == 5
    assert math.isclose(by_model["p5_canonical_seed42"]["r2_mean"], 0.4)

    # Missing ablation row exists but has NaN metrics + "pending" note.
    missing = by_model["p5_ablation_topk_4000"]
    assert missing["n_folds"] == 0
    assert math.isnan(missing["r2_mean"])
    assert "pending" in missing["notes"].lower()
