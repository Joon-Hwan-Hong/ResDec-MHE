"""Tests for make_calibration_figure.py.

Covers:
1. End-to-end smoke test against the canonical p5_canonical_seed42 +
   tabpfn outer caches (skipped if either is unavailable).
2. ``compute_calibration_metrics`` shape + invariant checks under a
   synthetic well-calibrated Gaussian residual.
3. ``compute_calibration_metrics`` correctly detects over-confidence
   (sigma too small -> empirical coverage < nominal).
4. ``load_calibration_data`` joins TabPFN sigma + cogdx metadata onto
   composite predictions without dropping subjects.
5. ``make_figure`` returns a Figure with 4 axes (panels A, B, C, D).
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


@pytest.fixture
def synthetic_well_calibrated() -> dict:
    """500 well-calibrated Gaussian residuals (sigma=1, residual~N(0,1))."""
    rng = np.random.default_rng(42)
    n = 500
    sigma = np.full(n, 1.0)
    y_true = rng.normal(0.0, 1.0, size=n)
    y_pred = np.zeros(n)  # so residual = y_true ~ N(0, 1) = sigma
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "sigma": sigma,
        "is_ad": rng.uniform(size=n) < 0.3,
    }


@pytest.fixture
def synthetic_overconfident() -> dict:
    """Sigma too small (0.5) but residual~N(0,1) -> empirical < nominal."""
    rng = np.random.default_rng(43)
    n = 500
    sigma = np.full(n, 0.5)  # report sigma=0.5 but residual std=1.0
    y_true = rng.normal(0.0, 1.0, size=n)
    y_pred = np.zeros(n)
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "sigma": sigma,
        "is_ad": np.zeros(n, dtype=bool),
    }


def _build_data(d: dict):
    from scripts.resdec_mhe.interpretability.make_calibration_figure import (
        CalibrationData,
    )
    n = len(d["y_true"])
    return CalibrationData(
        subject_id=np.array([f"R{i}" for i in range(n)]),
        fold=np.zeros(n, dtype=np.int64),
        y_true=d["y_true"],
        y_pred=d["y_pred"],
        sigma=d["sigma"],
        is_ad=d["is_ad"],
    )


def test_compute_calibration_metrics_well_calibrated(synthetic_well_calibrated):
    """Well-calibrated Gaussian -> empirical coverage matches nominal +-0.05."""
    from scripts.resdec_mhe.interpretability.make_calibration_figure import (
        compute_calibration_metrics,
    )
    data = _build_data(synthetic_well_calibrated)
    metrics = compute_calibration_metrics(data)

    # Each canonical level should be within ~5 pp of nominal at n=500.
    for p in (0.5, 0.68, 0.8, 0.95):
        emp = metrics["coverage_by_nominal"][f"coverage_at_{p}"]
        assert abs(emp - p) < 0.05, (
            f"Well-calibrated synthetic at p={p}: empirical={emp}, expected ~{p}"
        )

    # PIT must be approximately uniform -> KS p-value > 0.05.
    assert metrics["pit_ks_pvalue"] > 0.05, (
        f"PIT KS rejected uniformity for well-calibrated data: "
        f"p={metrics['pit_ks_pvalue']}"
    )

    # Required keys.
    for key in (
        "n", "n_ad", "pooled_r2", "mean_sigma", "mean_abs_residual",
        "coverage_by_nominal", "pit_ks_statistic", "pit_ks_pvalue",
        "abs_residual_vs_sigma_spearman_rho",
        "abs_residual_vs_sigma_spearman_pvalue",
        "per_sigma_quartile_residual", "nominal_levels",
    ):
        assert key in metrics, f"missing key: {key}"

    # Per-sigma-quartile structure.
    assert len(metrics["per_sigma_quartile_residual"]) == 4
    for label, rec in metrics["per_sigma_quartile_residual"].items():
        assert {"n", "mean_abs_residual", "median_abs_residual", "mean_sigma"} <= rec.keys()


def test_compute_calibration_metrics_overconfident(synthetic_overconfident):
    """Over-confident sigma -> empirical coverage < nominal."""
    from scripts.resdec_mhe.interpretability.make_calibration_figure import (
        compute_calibration_metrics,
    )
    data = _build_data(synthetic_overconfident)
    metrics = compute_calibration_metrics(data)
    # When sigma is reported half its true value, every coverage level
    # under-shoots its nominal level by a wide margin (e.g. nominal 0.95
    # collapses well below 0.95).
    for p in (0.5, 0.68, 0.8, 0.95):
        emp = metrics["coverage_by_nominal"][f"coverage_at_{p}"]
        assert emp < p, f"Expected over-confident empirical < nominal at p={p}; got {emp}"
    # PIT KS should reject uniformity.
    assert metrics["pit_ks_pvalue"] < 0.05


def test_load_calibration_data_canonical():
    """End-to-end load against canonical artifacts (skip if missing)."""
    pred_root = _WORKTREE_ROOT / "outputs/canonical/p5_canonical_seed42"
    tabpfn_dir = _WORKTREE_ROOT / "data/canonical"
    metadata_csv = _WORKTREE_ROOT / "data/metadata_ROSMAP/metadata.csv"
    if not (pred_root / "fold0/val_predictions_best.npz").is_file():
        pytest.skip(f"Canonical predictions not available under {pred_root}")
    if not (tabpfn_dir / "tabpfn_outer_fold0.npz").is_file():
        pytest.skip(f"TabPFN outer caches not available under {tabpfn_dir}")
    if not metadata_csv.is_file():
        pytest.skip(f"Metadata CSV not available at {metadata_csv}")

    from scripts.resdec_mhe.interpretability.make_calibration_figure import (
        load_calibration_data,
    )
    data = load_calibration_data(pred_root, tabpfn_dir, metadata_csv)
    # Canonical 5-fold p5_canonical_seed42 is 516 subjects.
    assert len(data.y_true) > 500, f"unexpectedly few subjects: {len(data.y_true)}"
    assert data.y_true.shape == data.y_pred.shape == data.sigma.shape
    assert data.sigma.min() > 0.0
    assert data.is_ad.dtype == bool


def test_make_figure_returns_4_axes(synthetic_well_calibrated):
    """make_figure returns a Figure with at least 4 axes (the 4 panels)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scripts.resdec_mhe.interpretability.make_calibration_figure import (
        compute_calibration_metrics, make_figure,
    )
    data = _build_data(synthetic_well_calibrated)
    metrics = compute_calibration_metrics(data)
    fig = make_figure(data, metrics)
    try:
        assert fig is not None
        assert len(fig.axes) >= 4
    finally:
        plt.close(fig)


def test_main_smoke(tmp_path: Path):
    """End-to-end smoke test: script runs and writes PNG/PDF/JSON."""
    pred_root = _WORKTREE_ROOT / "outputs/canonical/p5_canonical_seed42"
    tabpfn_dir = _WORKTREE_ROOT / "data/canonical"
    metadata_csv = _WORKTREE_ROOT / "data/metadata_ROSMAP/metadata.csv"
    if not (pred_root / "fold0/val_predictions_best.npz").is_file():
        pytest.skip(f"Canonical predictions not available under {pred_root}")
    if not (tabpfn_dir / "tabpfn_outer_fold0.npz").is_file():
        pytest.skip(f"TabPFN outer caches not available under {tabpfn_dir}")
    if not metadata_csv.is_file():
        pytest.skip(f"Metadata CSV not available at {metadata_csv}")

    out_dir = tmp_path / "calibration"
    summary_json = tmp_path / "calibration_summary.json"
    script = (
        _WORKTREE_ROOT
        / "scripts/resdec_mhe/interpretability/make_calibration_figure.py"
    )
    cmd = [
        sys.executable, str(script),
        "--pred-root", str(pred_root),
        "--tabpfn-dir", str(tabpfn_dir),
        "--metadata-csv", str(metadata_csv),
        "--out-dir", str(out_dir),
        "--summary-json", str(summary_json),
        "--stem", "fig_calibration",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_WORKTREE_ROOT))
    assert result.returncode == 0, (
        f"Script failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    png = out_dir / "fig_calibration.png"
    pdf = out_dir / "fig_calibration.pdf"
    assert png.is_file() and png.stat().st_size > 1000, png
    assert pdf.is_file() and pdf.stat().st_size > 1000, pdf
    assert summary_json.is_file() and summary_json.stat().st_size > 100, summary_json

    payload = json.loads(summary_json.read_text())
    for key in ("n", "pooled_r2", "coverage_by_nominal", "pit_ks_pvalue", "provenance"):
        assert key in payload
    # Canonical 4 levels must be present in the JSON.
    for p in (0.5, 0.68, 0.8, 0.95):
        assert f"coverage_at_{p}" in payload["coverage_by_nominal"]
