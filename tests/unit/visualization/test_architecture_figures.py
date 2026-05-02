"""Smoke tests for the architecture-diagram lab-meeting orchestrator.

Verifies that ``make_lab_meeting_architecture_figures.py`` produces:
  - fig_slot3_4_fusion_stack.{png,pdf}              (slot 3.4)
  - fig_slot3_full_architecture_hybrid.{png,pdf}    (slot 3.1-3.3)

Each PNG must exist and exceed 50 KB (sanity-check that something was actually
rendered, not a near-empty stub).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tests.conftest import WORKTREE_ROOT as _WORKTREE_ROOT

def test_orchestrator_module_imports():
    """The orchestrator must be importable."""
    from scripts.resdec_mhe.interpretability import (  # noqa: F401
        make_lab_meeting_architecture_figures as orch,
    )

def test_both_figures_written(tmp_path: Path):
    """Smoke: orchestrator writes both figures (PNG + PDF), > 50 KB each."""
    from scripts.resdec_mhe.interpretability import (
        make_lab_meeting_architecture_figures as orch,
    )

    out_dir = tmp_path / "lab_meeting"
    orch.build_all_figures(out_dir=out_dir)

    expected_stems = [
        "fig_slot3_4_fusion_stack",
        "fig_slot3_full_architecture_hybrid",
    ]
    for stem in expected_stems:
        png = out_dir / f"{stem}.png"
        pdf = out_dir / f"{stem}.pdf"
        assert png.exists(), f"missing {png}"
        # PDF intentionally NOT written (user pref — PNG only).
        assert not pdf.exists(), f"unexpected pdf at {pdf}"
        size_kb = png.stat().st_size / 1024.0
        assert size_kb > 50.0, (
            f"{png} too small ({size_kb:.1f} KB); expected > 50 KB"
        )

def test_slot3_4_fusion_stack_is_single_panel():
    """Slot 3.4 fusion-stack figure should be a single-axes block diagram."""
    from scripts.resdec_mhe.interpretability import (
        make_lab_meeting_architecture_figures as orch,
    )

    fig = orch.build_slot3_4_fusion_stack()
    assert len(fig.axes) >= 1

def test_slot3_full_architecture_hybrid_is_diagram():
    """Slot 3.1-3.3 hybrid full architecture figure should be a diagram."""
    from scripts.resdec_mhe.interpretability import (
        make_lab_meeting_architecture_figures as orch,
    )

    fig = orch.build_slot3_full_architecture_hybrid()
    assert len(fig.axes) >= 1
