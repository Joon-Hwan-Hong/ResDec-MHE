"""Smoke test for run_ccc_outlier_threshold_sweep.py.

Verifies the script runs end-to-end against the actual CCC raw NPZ
tensor + ROSMAP metadata CSV and produces non-empty PNG + PDF + JSON +
MD outputs across the default 4-threshold sweep.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]


def test_run_ccc_outlier_threshold_sweep_runs(tmp_path: Path) -> None:
    ccc_npz = (
        _WORKTREE_ROOT
        / "outputs/canonical/interpretability/ccc/per_subject_ccc_attention.npz"
    )
    metadata_csv = _WORKTREE_ROOT / "data/metadata_ROSMAP/metadata.csv"
    if not ccc_npz.is_file() or not metadata_csv.is_file():
        pytest.skip("Required CCC NPZ tensor or metadata CSV missing.")

    out_fig_dir = tmp_path / "ccc_heterogeneity_fig"
    out_data_dir = tmp_path / "ccc_heterogeneity_data"
    script = (
        _WORKTREE_ROOT
        / "scripts/resdec_mhe/interpretability/run_ccc_outlier_threshold_sweep.py"
    )
    cmd = [
        sys.executable,
        str(script),
        "--ccc-npz",
        str(ccc_npz),
        "--metadata-csv",
        str(metadata_csv),
        "--out-fig-dir",
        str(out_fig_dir),
        "--out-data-dir",
        str(out_data_dir),
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(_WORKTREE_ROOT)
    )
    assert result.returncode == 0, (
        f"Script failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )

    png = out_fig_dir / "fig_threshold_sensitivity.png"
    pdf = out_fig_dir / "fig_threshold_sensitivity.pdf"
    out_json = out_data_dir / "threshold_sensitivity.json"
    out_md = out_data_dir / "threshold_sensitivity.md"
    assert png.is_file() and png.stat().st_size > 1000, png
    assert pdf.is_file() and pdf.stat().st_size > 1000, pdf
    assert out_json.is_file() and out_json.stat().st_size > 100, out_json
    assert out_md.is_file() and out_md.stat().st_size > 100, out_md

    # Spot-check JSON content.
    payload = json.loads(out_json.read_text())
    assert "config" in payload and "per_threshold" in payload
    assert payload["config"]["thresholds"] == [0.005, 0.01, 0.02, 0.05]
    assert len(payload["per_threshold"]) == 4
    for rec in payload["per_threshold"]:
        assert "threshold" in rec
        assert "n_outliers" in rec
        assert "enrichment" in rec
        assert "cogn_global" in rec["enrichment"]
        assert "ad_dx_cogdx_4_or_5" in rec["enrichment"]
        assert "sex_msex_male_eq_1" in rec["enrichment"]
        assert "top_n_dominant_edges_outliers" in rec
    # Stability list should have N-1 entries.
    assert len(payload["stability_consecutive"]) == 3
    for s in payload["stability_consecutive"]:
        assert "jaccard_top_n" in s
        assert "intersection_size" in s
    # n_outliers should be monotonically non-increasing as τ rises (a
    # stricter threshold cannot create new outliers).
    n_out = [r["n_outliers"] for r in payload["per_threshold"]]
    assert all(n_out[i] >= n_out[i + 1] for i in range(len(n_out) - 1)), n_out


def test_per_subject_outlier_metrics_basic() -> None:
    """``_per_subject_outlier_metrics`` correctly counts edges and max."""
    sys.path.insert(0, str(_WORKTREE_ROOT))
    from scripts.resdec_mhe.interpretability import (  # noqa: E402
        run_ccc_outlier_threshold_sweep as mod,
    )
    import numpy as np

    # 3 subjects × 2 CT × 2 CT × 1 edge type.
    att = np.array(
        [
            [[[0.05], [0.001]], [[0.01], [0.0001]]],   # subj 0
            [[[np.nan], [0.02]], [[0.001], [0.001]]],  # subj 1
            [[[0.0001], [0.0001]], [[0.0001], [0.0001]]],  # subj 2 (all small)
        ],
        dtype=np.float32,
    )
    max_above, n_edges = mod._per_subject_outlier_metrics(att, threshold=0.01)
    assert n_edges.tolist() == [2, 1, 0]
    assert max_above[0] == pytest.approx(0.05, rel=1e-5)
    assert max_above[1] == pytest.approx(0.02, rel=1e-5)
    assert np.isnan(max_above[2])


def test_jaccard_overlap() -> None:
    sys.path.insert(0, str(_WORKTREE_ROOT))
    from scripts.resdec_mhe.interpretability import (  # noqa: E402
        run_ccc_outlier_threshold_sweep as mod,
    )

    a = [("X", "Y", "T1"), ("X", "Z", "T1"), ("A", "B", "T2")]
    b = [("X", "Y", "T1"), ("A", "B", "T2"), ("M", "N", "T1")]
    j = mod._jaccard(a, b)
    # |∩|=2, |∪|=4 → 0.5
    assert j == pytest.approx(0.5, rel=1e-6)
    assert mod._jaccard([], []) != mod._jaccard([], []) or True  # NaN handling
    import math

    j_empty = mod._jaccard([], [])
    assert math.isnan(j_empty)


def test_dominant_edge_counter_dedups_per_subject() -> None:
    sys.path.insert(0, str(_WORKTREE_ROOT))
    from scripts.resdec_mhe.interpretability import (  # noqa: E402
        run_ccc_outlier_threshold_sweep as mod,
    )

    # Subject 0 has the same (X,Y,T1) tuple twice (different attention vals)
    # — should count once.
    edges = [
        [("X", "Y", "T1", 0.05), ("X", "Y", "T1", 0.04), ("A", "B", "T2", 0.03)],
        [("X", "Y", "T1", 0.06)],
    ]
    counter = mod._dominant_edge_counter(edges)
    assert counter[("X", "Y", "T1")] == 2
    assert counter[("A", "B", "T2")] == 1


def test_parse_thresholds_sorted() -> None:
    sys.path.insert(0, str(_WORKTREE_ROOT))
    from scripts.resdec_mhe.interpretability import (  # noqa: E402
        run_ccc_outlier_threshold_sweep as mod,
    )

    assert mod._parse_thresholds("0.05,0.005,0.01") == [0.005, 0.01, 0.05]
    assert mod._parse_thresholds("0.01") == [0.01]
    with pytest.raises(ValueError):
        mod._parse_thresholds("")
