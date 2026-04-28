"""Unit tests for scripts/resdec_mhe/interpretability/make_baseline_table.py.

Tests cover:
- Discovery of resdec_mhe ablation dirs via glob
- Parsing of best_vs_tabpfn_summary.json schema (per-fold ours metrics)
- Parsing of classical baseline CSV (Ridge/ElasticNet/PLS/RF/XGBoost)
- Parsing of DL baseline CSVs (cloudpred/gpio/perceiver_io; 1-indexed folds)
- Parsing of TabPFN-2.6 outer-fold npz (per-fold R² computed on the fly)
- Missing baselines produce NaN rows with a "missing" note (not crash)
- Missing ablations (not-yet-launched runs) produce a NaN row with a "pending" note
- Aggregation computes mean/std with ddof=1 across available folds
- Sort places baselines (by r2 desc) first, ours (by r2 desc) second, NaN last
- ``_fmt_pair`` handles NaN, regular mean±std, and size-1 mean-only cases
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import r2_score


# Standard-import the script as a module. Requires scripts/__init__.py and
# scripts/resdec_mhe/__init__.py + scripts/resdec_mhe/interpretability/__init__.py
# to exist in the worktree.
_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from scripts.resdec_mhe.interpretability import make_baseline_table as mod


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
        "tabpfn_dir": "data/canonical",
    }))


# ---------------------------------------------------------------------------
# Summary-JSON parsing
# ---------------------------------------------------------------------------

def test_parse_summary_json_extracts_all_five_metrics(tmp_path: Path) -> None:
    """A well-formed summary JSON → 5 per-fold arrays, one per metric."""
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
    out = mod.parse_summary_json(tmp_path / "does_not_exist.json")
    assert out is None


def test_parse_summary_json_malformed_returns_none(tmp_path: Path) -> None:
    """A malformed JSON should return None (caller logs WARNING)."""
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json]}")
    out = mod.parse_summary_json(bad)
    assert out is None


# ---------------------------------------------------------------------------
# Ablation discovery
# ---------------------------------------------------------------------------

def test_discover_ablations_picks_up_all_p5_dirs(tmp_path: Path) -> None:
    """Glob should discover all p5_* dirs with a best_vs_tabpfn_summary.json."""
    # Layout: three resdec_mhe dirs, one with summary, one without, one unrelated.
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
    csv_path = tmp_path / "results.csv"
    _write_dl_baseline_csv(csv_path)

    out = mod.parse_dl_baseline_csv(csv_path)
    assert out is not None
    for k in ("r2", "mae", "rmse", "pearson_r", "spearman_rho"):
        assert len(out[k]) == 5


def test_parse_dl_baseline_csv_missing_returns_none(tmp_path: Path) -> None:
    out = mod.parse_dl_baseline_csv(tmp_path / "missing.csv")
    assert out is None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def test_summarise_row_computes_mean_std() -> None:
    """summarise_row → mean + std (ddof=1) for each metric."""
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
    metrics = {k: [] for k in ("r2", "mae", "rmse", "pearson_r", "spearman_rho")}
    s = mod.summarise_row(metrics)
    assert s["n_folds"] == 0
    assert math.isnan(s["r2_mean"])
    assert math.isnan(s["r2_std"])


# ---------------------------------------------------------------------------
# TabPFN standalone parsing
# ---------------------------------------------------------------------------

def test_parse_tabpfn_standalone_computes_r2_from_npz(tmp_path: Path) -> None:
    """Synthetic tabpfn_outer_fold{f}.npz → per-fold R² computed on the fly."""
    n_folds = 5
    rng = np.random.default_rng(42)
    expected_r2s: list[float] = []
    for f in range(n_folds):
        n = 30
        y_true = rng.standard_normal(n)
        # Known noise level → finite R² bounded < 1.
        y_tabpfn = y_true + rng.standard_normal(n) * 0.4
        np.savez(
            tmp_path / f"tabpfn_outer_fold{f}.npz",
            val_subject_ids=np.asarray([f"F{f}_S{i}" for i in range(n)], dtype=object),
            y_true=y_true.astype(np.float64),
            y_tabpfn=y_tabpfn.astype(np.float64),
            sigma_tabpfn=np.full(n, 0.5, dtype=np.float64),
        )
        expected_r2s.append(float(r2_score(y_true, y_tabpfn)))

    metrics = mod.parse_tabpfn_standalone(tmp_path, n_folds=n_folds)

    assert metrics is not None
    assert set(metrics) == {"r2", "mae", "rmse", "pearson_r", "spearman_rho"}
    # 5 R²s, all finite, and agree with sklearn to float precision.
    assert len(metrics["r2"]) == n_folds
    assert all(math.isfinite(v) for v in metrics["r2"])
    np.testing.assert_allclose(
        metrics["r2"], expected_r2s, rtol=0, atol=1e-12,
    )
    # Pearson and Spearman should also be finite on non-degenerate data.
    assert all(math.isfinite(v) for v in metrics["pearson_r"])
    assert all(math.isfinite(v) for v in metrics["spearman_rho"])


def test_parse_tabpfn_standalone_missing_fold_returns_none(tmp_path: Path) -> None:
    """Any missing outer-fold npz → None (entire row dropped)."""
    # Only write 3 of the 5 expected folds.
    for f in range(3):
        np.savez(
            tmp_path / f"tabpfn_outer_fold{f}.npz",
            val_subject_ids=np.asarray(["S0", "S1", "S2"], dtype=object),
            y_true=np.array([1.0, 2.0, 3.0]),
            y_tabpfn=np.array([0.9, 2.1, 2.8]),
            sigma_tabpfn=np.array([0.5, 0.5, 0.5]),
        )
    out = mod.parse_tabpfn_standalone(tmp_path, n_folds=5)
    assert out is None


# ---------------------------------------------------------------------------
# Sort order for the markdown render
# ---------------------------------------------------------------------------

def test_sort_for_md_places_ours_at_bottom_and_nan_last() -> None:
    """_sort_for_md: baselines first (r2 desc), ours second (r2 desc), NaN last."""
    rows: list[dict] = [
        {
            "model": "p5_canonical_seed42",
            "display_name": "Ours",
            "r2_mean": 0.44,
            "_is_ours": True,
        },
        {
            "model": "tabpfn",
            "display_name": "TabPFN",
            "r2_mean": 0.40,
            "_is_ours": False,
        },
        {
            "model": "xgboost",
            "display_name": "XGBoost",
            "r2_mean": 0.36,
            "_is_ours": False,
        },
        {
            "model": "pending",
            "display_name": "Pending",
            "r2_mean": float("nan"),
            "_is_ours": False,
        },
    ]
    sorted_rows = mod._sort_for_md(rows)
    # Baseline block comes first, highest r2 first, NaN last within block.
    assert sorted_rows[0]["model"] == "tabpfn"
    assert sorted_rows[1]["model"] == "xgboost"
    assert sorted_rows[2]["model"] == "pending"  # NaN last within baseline block
    # Ours row at the bottom.
    assert sorted_rows[-1]["model"] == "p5_canonical_seed42"


# ---------------------------------------------------------------------------
# _fmt_pair formatting rules
# ---------------------------------------------------------------------------

def test_fmt_pair_handles_nan_and_size_one() -> None:
    """_fmt_pair: NaN mean → '—'; NaN std → mean-only; std == 0 → mean-only."""
    # Both NaN → em-dash (pending/missing row).
    assert mod._fmt_pair(float("nan"), float("nan")) == "—"
    # NaN mean alone → also em-dash.
    assert mod._fmt_pair(float("nan"), 0.1) == "—"
    # Regular mean + std → mean ± std.
    assert mod._fmt_pair(0.4436, 0.0996) == "0.4436 ± 0.0996"
    # Size-1 / reference row: std == 0.0 → mean-only.
    assert mod._fmt_pair(0.286, 0.0) == "0.2860"
    # NaN std only → mean-only (current-encoder-alone reference).
    assert mod._fmt_pair(0.286, float("nan")) == "0.2860"


# ---------------------------------------------------------------------------
# End-to-end row assembly with missing ablations
# ---------------------------------------------------------------------------

def test_ablation_row_falls_back_to_per_fold_npz(tmp_path: Path) -> None:
    """When summary.json is absent but per-fold npz files exist, use them."""
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
    # Only canonical exists; request an ablation that does not exist on disk.
    (tmp_path / "p5_canonical_seed42").mkdir()
    _write_summary_json(
        tmp_path / "p5_canonical_seed42" / "best_vs_tabpfn_summary.json",
        [0.4, 0.4, 0.4, 0.4, 0.4],
    )

    rows = mod.collect_ablation_rows(
        ablation_root=tmp_path,
        requested=[
            ("p5_canonical_seed42", "ResDec-MHE (canonical)"),
            ("p5_ablation_topk_4000", "top-k=4000 ablation"),  # not on disk
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
