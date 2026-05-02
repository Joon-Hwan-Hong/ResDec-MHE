"""Smoke test for make_step_b_trajectory_figure.py.

Verifies:
  - the script runs end-to-end against the actual fold-0 trajectory JSONs
    and produces non-empty PNG + PDF outputs (when the trajectory files
    exist),
  - the placeholder branch produces non-empty PNG + PDF if both inputs
    are absent,
  - the trajectory-to-arrays helper squares the signed-gap column.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import WORKTREE_ROOT as _WORKTREE_ROOT

def test_make_step_b_trajectory_runs(tmp_path: Path) -> None:
    rel_json = (
        _WORKTREE_ROOT
        / "outputs/canonical/interpretability/counterfactuals_trajectory_relative_delta0p5/counterfactuals_fold0.json"
    )
    abs_json = (
        _WORKTREE_ROOT
        / "outputs/canonical/interpretability/counterfactuals_trajectory_absolute_delta0p5/counterfactuals_fold0.json"
    )
    if not rel_json.is_file() and not abs_json.is_file():
        pytest.skip("No trajectory JSONs present (placeholder will run instead).")

    out_dir = tmp_path / "step_b_trajectory"
    script = (
        _WORKTREE_ROOT
        / "scripts/resdec_mhe/interpretability/make_step_b_trajectory_figure.py"
    )
    cmd = [
        sys.executable,
        str(script),
        "--relative-json",
        str(rel_json),
        "--absolute-json",
        str(abs_json),
        "--out-dir",
        str(out_dir),
        "--stem",
        "fig_step_b_trajectory",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_WORKTREE_ROOT))
    assert result.returncode == 0, (
        f"Script failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    png = out_dir / "fig_step_b_trajectory.png"
    pdf = out_dir / "fig_step_b_trajectory.pdf"
    assert png.is_file() and png.stat().st_size > 1000, png
    assert pdf.is_file() and pdf.stat().st_size > 1000, pdf

def test_placeholder_branch_runs(tmp_path: Path) -> None:
    """If both inputs are missing, script still emits PNG+PDF placeholder."""
    out_dir = tmp_path / "step_b_trajectory_placeholder"
    script = (
        _WORKTREE_ROOT
        / "scripts/resdec_mhe/interpretability/make_step_b_trajectory_figure.py"
    )
    bogus_rel = tmp_path / "no_such_relative.json"
    bogus_abs = tmp_path / "no_such_absolute.json"
    cmd = [
        sys.executable,
        str(script),
        "--relative-json",
        str(bogus_rel),
        "--absolute-json",
        str(bogus_abs),
        "--out-dir",
        str(out_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_WORKTREE_ROOT))
    assert result.returncode == 0
    png = out_dir / "fig_step_b_trajectory.png"
    pdf = out_dir / "fig_step_b_trajectory.pdf"
    assert png.is_file() and png.stat().st_size > 500, png
    assert pdf.is_file() and pdf.stat().st_size > 500, pdf

def test_trajectory_to_arrays_squares() -> None:
    """The helper must return ``signed_gap ** 2`` to match the spec."""
    from scripts.resdec_mhe.interpretability import make_step_b_trajectory_figure as mod  # noqa: E402

    traj = [[1e-3, 0.5], [2e-3, -0.4], [4e-3, -0.1], [8e-3, 0.0]]
    steps, vals = mod._trajectory_to_arrays(traj)
    assert steps.tolist() == [0, 1, 2, 3]
    np.testing.assert_allclose(vals, np.array([0.25, 0.16, 0.01, 0.0]))

def test_trajectory_to_arrays_empty() -> None:
    from scripts.resdec_mhe.interpretability import make_step_b_trajectory_figure as mod  # noqa: E402

    steps, vals = mod._trajectory_to_arrays([])
    assert steps.size == 0 and vals.size == 0
