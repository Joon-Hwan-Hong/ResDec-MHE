"""Smoke test for run_ccc_outlier_deepdive.py.

Verifies the script runs end-to-end against the actual CCC summary
JSON + ROSMAP metadata CSV and produces non-empty PNG + PDF + JSON + MD
outputs.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]


def test_run_ccc_outlier_deepdive_runs(tmp_path: Path) -> None:
    summary_json = (
        _WORKTREE_ROOT
        / "outputs/canonical/interpretability/ccc/per_subject_ccc_attention_summary.json"
    )
    metadata_csv = _WORKTREE_ROOT / "data/metadata_ROSMAP/metadata.csv"
    if not summary_json.is_file() or not metadata_csv.is_file():
        pytest.skip("Required CCC summary or metadata CSV missing.")

    out_fig_dir = tmp_path / "ccc_heterogeneity_fig"
    out_data_dir = tmp_path / "ccc_heterogeneity_data"
    script = (
        _WORKTREE_ROOT
        / "scripts/resdec_mhe/interpretability/run_ccc_outlier_deepdive.py"
    )
    cmd = [
        sys.executable,
        str(script),
        "--ccc-summary-json",
        str(summary_json),
        "--metadata-csv",
        str(metadata_csv),
        "--out-fig-dir",
        str(out_fig_dir),
        "--out-data-dir",
        str(out_data_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_WORKTREE_ROOT))
    assert result.returncode == 0, (
        f"Script failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    png = out_fig_dir / "fig_ccc_outlier_demographics.png"
    pdf = out_fig_dir / "fig_ccc_outlier_demographics.pdf"
    out_json = out_data_dir / "per_subject_outlier_analysis.json"
    out_md = out_data_dir / "per_subject_outlier_analysis.md"
    assert png.is_file() and png.stat().st_size > 1000, png
    assert pdf.is_file() and pdf.stat().st_size > 1000, pdf
    assert out_json.is_file() and out_json.stat().st_size > 100, out_json
    assert out_md.is_file() and out_md.stat().st_size > 100, out_md

    # Spot-check JSON content
    payload = json.loads(out_json.read_text())
    assert "config" in payload and "enrichment" in payload
    assert payload["config"]["n_outliers"] >= 1
    assert "cogn_global" in payload["enrichment"]
    assert "ad_dx_cogdx_4_or_5" in payload["enrichment"]
    assert "outlier_subjects" in payload
    assert isinstance(payload["outlier_subjects"], list)
    assert len(payload["outlier_subjects"]) == payload["config"]["n_outliers"]


def test_split_outlier_typical() -> None:
    sys.path.insert(0, str(_WORKTREE_ROOT))
    from scripts.resdec_mhe.interpretability import run_ccc_outlier_deepdive as mod  # noqa: E402

    summary = {
        "per_subject": [
            {"subject_id": "A", "n_high_attention_edges": 0, "top_edges": []},
            {"subject_id": "B", "n_high_attention_edges": 3, "top_edges": []},
            {"subject_id": "C", "n_high_attention_edges": 1, "top_edges": []},
        ]
    }
    out, typ = mod._split_outlier_typical(summary)
    assert {s["subject_id"] for s in out} == {"B", "C"}
    assert {s["subject_id"] for s in typ} == {"A"}


def test_top_pair_frequency() -> None:
    sys.path.insert(0, str(_WORKTREE_ROOT))
    from scripts.resdec_mhe.interpretability import run_ccc_outlier_deepdive as mod  # noqa: E402

    records = [
        {
            "top_edges": [
                {"source_ct": "X", "target_ct": "Y", "attention": 0.05},
                {"source_ct": "X", "target_ct": "Z", "attention": 0.04},
                {"source_ct": "Y", "target_ct": "Z", "attention": 0.03},
                {"source_ct": "tail", "target_ct": "tail", "attention": 0.001},
            ],
        },
        {
            "top_edges": [
                {"source_ct": "X", "target_ct": "Y", "attention": 0.06},
                {"source_ct": "Y", "target_ct": "Z", "attention": 0.05},
                {"source_ct": "A", "target_ct": "B", "attention": 0.02},
            ],
        },
    ]
    counter = mod._top_pair_frequency(records, top_k=3)
    assert counter[("X", "Y")] == 2
    assert counter[("Y", "Z")] == 2
    assert counter[("X", "Z")] == 1
    assert counter[("A", "B")] == 1
    assert ("tail", "tail") not in counter
