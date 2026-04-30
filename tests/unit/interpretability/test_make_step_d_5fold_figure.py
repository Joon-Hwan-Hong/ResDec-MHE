"""Smoke test for make_step_d_5fold_figure.py.

Verifies the script runs end-to-end against the actual canonical
``f1_5fold_summary.json`` and produces non-empty PNG + PDF outputs.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]


def test_make_step_d_5fold_runs(tmp_path: Path) -> None:
    summary_path = (
        _WORKTREE_ROOT
        / "outputs/canonical/interpretability/f1_5fold_summary.json"
    )
    if not summary_path.is_file():
        pytest.skip(f"Summary JSON not present: {summary_path}")

    out_dir = tmp_path / "step_d_5fold"
    script = (
        _WORKTREE_ROOT
        / "scripts/resdec_mhe/interpretability/make_step_d_5fold_figure.py"
    )
    cmd = [
        sys.executable,
        str(script),
        "--summary-json",
        str(summary_path),
        "--out-dir",
        str(out_dir),
        "--stem",
        "fig_step_d_5fold_box_whisker",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_WORKTREE_ROOT))
    assert result.returncode == 0, (
        f"Script failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    png = out_dir / "fig_step_d_5fold_box_whisker.png"
    pdf = out_dir / "fig_step_d_5fold_box_whisker.pdf"
    assert png.is_file() and png.stat().st_size > 1000, png
    assert pdf.is_file() and pdf.stat().st_size > 1000, pdf


def test_collect_per_fold_rates_round_trip(tmp_path: Path) -> None:
    """Synthetic 1-fold summary still parses; rates are a 1-element list."""
    sys.path.insert(0, str(_WORKTREE_ROOT))
    from scripts.resdec_mhe.interpretability import make_step_d_5fold_figure as mod  # noqa: E402

    fake_summary = {
        "per_fold": {
            "fold_0": {
                "relative_delta0.5": {
                    "resilient": {"success_rate": 0.8},
                    "vulnerable": {"success_rate": 0.6},
                },
                "relative_delta0.3": {
                    "resilient": {"success_rate": 0.9},
                    "vulnerable": {"success_rate": 0.7},
                },
                "absolute_delta0.5": {
                    "resilient": {"success_rate": 0.5},
                    "vulnerable": {"success_rate": 0.3},
                },
                "absolute_delta0.3": {
                    "resilient": {"success_rate": 0.95},
                    "vulnerable": {"success_rate": 0.85},
                },
            },
        },
        "across_folds": {
            "relative_delta0.5": {
                "resilient": {"success_rate": 0.8},
                "vulnerable": {"success_rate": 0.6},
            },
            "relative_delta0.3": {
                "resilient": {"success_rate": 0.9},
                "vulnerable": {"success_rate": 0.7},
            },
            "absolute_delta0.5": {
                "resilient": {"success_rate": 0.5},
                "vulnerable": {"success_rate": 0.3},
            },
            "absolute_delta0.3": {
                "resilient": {"success_rate": 0.95},
                "vulnerable": {"success_rate": 0.85},
            },
        },
    }
    rates = mod.collect_per_fold_rates(fake_summary)
    for panel_key, _, _ in mod.PANEL_KEYS:
        assert panel_key in rates
        assert "resilient" in rates[panel_key] and "vulnerable" in rates[panel_key]
        assert len(rates[panel_key]["resilient"]) == 1


def test_make_figure_returns_figure() -> None:
    """make_figure(summary) returns a matplotlib Figure with 4 axes."""
    sys.path.insert(0, str(_WORKTREE_ROOT))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scripts.resdec_mhe.interpretability import make_step_d_5fold_figure as mod  # noqa: E402

    summary_path = (
        _WORKTREE_ROOT
        / "outputs/canonical/interpretability/f1_5fold_summary.json"
    )
    if not summary_path.is_file():
        pytest.skip(f"Summary JSON not present: {summary_path}")
    summary = json.loads(summary_path.read_text())
    fig = mod.make_figure(summary)
    assert fig is not None
    assert len(fig.axes) >= 4
    plt.close(fig)
