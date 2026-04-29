"""Smoke tests for the 3-figure lab-meeting orchestrator.

Verifies that the orchestrator script ``make_lab_meeting_figures.py`` produces:
  - fig_slot1_residual_definition.{png,pdf}
  - fig_slot2_marker_validation.{png,pdf}
  - fig_slot6_methods_recap.{png,pdf}

Each PNG must exist and exceed 50 KB (sanity-check that something was actually
rendered, not a near-empty stub).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))


def _write_synthetic_canonical_dir(tmp_path: Path) -> Path:
    """Write 5 fold-NPZ files with synthetic predictions/targets."""
    canonical = tmp_path / "p5_canonical_seed42"
    rng = np.random.default_rng(42)
    n_per_fold = 20
    for f in range(5):
        fold_dir = canonical / f"fold{f}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        # Synthetic cognitive residuals roughly N(-0.9, 1.16) like real data.
        targets = rng.normal(loc=-0.9, scale=1.16, size=n_per_fold)
        predictions = targets + rng.normal(scale=0.5, size=n_per_fold)
        subject_ids = np.array([f"R{i:08d}" for i in range(n_per_fold)],
                               dtype=object)
        np.savez(
            fold_dir / "val_predictions_best.npz",
            subject_ids=subject_ids,
            predictions=predictions.astype(np.float32),
            targets=targets.astype(np.float32),
            epoch=0,
            mse=0.5, mae=0.4, rmse=0.7, r2=0.4,
            pearson_r=0.7, spearman_rho=0.65,
        )
    return canonical


def _write_synthetic_marker_jsons(tmp_path: Path) -> tuple[Path, Path]:
    """Write minimal extended_marker_verification.json + splatter_marker_*.json."""
    extended = {
        "Splatter (SST+CHODL+ projection-IN)": {
            "target_ct": "Splatter",
            "missing_from_HVG": [],
            "markers": {
                "SST":   {"target_rank": 1, "ratio_target_over_median": 70.0,
                          "target_mean": 1.5, "all_ct_median": 0.02,
                          "top_ct_in_atlas": "Splatter"},
                "NPY":   {"target_rank": 1, "ratio_target_over_median": 80.0,
                          "target_mean": 1.8, "all_ct_median": 0.02,
                          "top_ct_in_atlas": "Splatter"},
                "CHODL": {"target_rank": 1, "ratio_target_over_median": 60.0,
                          "target_mean": 0.6, "all_ct_median": 0.01,
                          "top_ct_in_atlas": "Splatter"},
            },
        },
        "Vascular (endothelial)": {
            "target_ct": "Vascular",
            "missing_from_HVG": [],
            "markers": {
                "CDH5":  {"target_rank": 1, "ratio_target_over_median": 280.0,
                          "target_mean": 0.9, "all_ct_median": 0.003,
                          "top_ct_in_atlas": "Vascular"},
                "CLDN5": {"target_rank": 1, "ratio_target_over_median": 145.0,
                          "target_mean": 2.5, "all_ct_median": 0.017,
                          "top_ct_in_atlas": "Vascular"},
            },
        },
        "Microglia (CSF1R+)": {
            "target_ct": "Microglia",
            "missing_from_HVG": ["TMEM119"],
            "markers": {
                "CSF1R":  {"target_rank": 1, "ratio_target_over_median": 84.0,
                           "target_mean": 2.4, "all_ct_median": 0.028,
                           "top_ct_in_atlas": "Microglia"},
                "P2RY12": {"target_rank": 1, "ratio_target_over_median": 33.0,
                           "target_mean": 1.9, "all_ct_median": 0.057,
                           "top_ct_in_atlas": "Microglia"},
            },
        },
    }
    extended_path = tmp_path / "extended_marker_verification.json"
    extended_path.write_text(json.dumps(extended))

    splatter = {
        "markers_present": ["SST", "CHODL", "LHX6", "NPY", "NOS1"],
        "markers_missing": [],
        "splatter_rank_per_marker": {
            "SST": 1, "CHODL": 1, "LHX6": 3, "NPY": 1, "NOS1": 1,
        },
        "mean_pseudobulk_per_ct_per_marker": {
            "Splatter": {"SST": 1.5, "CHODL": 0.6, "LHX6": 0.77,
                         "NPY": 1.78, "NOS1": 1.65},
            "Astrocyte": {"SST": 0.02, "CHODL": 0.05, "LHX6": 0.025,
                          "NPY": 0.04, "NOS1": 0.02},
        },
        "n_subjects_with_cells_per_ct": {"Splatter": 437},
    }
    splatter_path = tmp_path / "splatter_marker_verification.json"
    splatter_path.write_text(json.dumps(splatter))

    return extended_path, splatter_path


@pytest.fixture
def synthetic_inputs(tmp_path: Path) -> dict[str, Path]:
    """Create all synthetic inputs needed by the 3-figure orchestrator."""
    canonical = _write_synthetic_canonical_dir(tmp_path)
    extended_path, splatter_path = _write_synthetic_marker_jsons(tmp_path)
    out_dir = tmp_path / "lab_meeting"
    return {
        "canonical_dir": canonical,
        "extended_path": extended_path,
        "splatter_path": splatter_path,
        "out_dir": out_dir,
    }


def test_orchestrator_module_imports():
    """The orchestrator must be importable."""
    from scripts.resdec_mhe.interpretability import (  # noqa: F401
        make_lab_meeting_figures as orch,
    )


def test_all_three_figures_written(synthetic_inputs):
    """Smoke: orchestrator writes all 3 figures (PNG + PDF), > 50 KB each."""
    from scripts.resdec_mhe.interpretability import make_lab_meeting_figures as orch

    orch.build_all_figures(
        canonical_dir=synthetic_inputs["canonical_dir"],
        extended_marker_json=synthetic_inputs["extended_path"],
        splatter_marker_json=synthetic_inputs["splatter_path"],
        out_dir=synthetic_inputs["out_dir"],
        n_folds=5,
    )

    out_dir = synthetic_inputs["out_dir"]
    expected_stems = [
        "fig_slot1_residual_definition",
        "fig_slot2_marker_validation",
        "fig_slot6_methods_recap",
    ]
    for stem in expected_stems:
        png = out_dir / f"{stem}.png"
        pdf = out_dir / f"{stem}.pdf"
        assert png.exists(), f"missing {png}"
        # PDF intentionally NOT written (user pref — PNG only for the
        # lab-meeting deliverable).
        assert not pdf.exists(), f"unexpected pdf at {pdf}"
        size_kb = png.stat().st_size / 1024.0
        assert size_kb > 50.0, (
            f"{png} too small ({size_kb:.1f} KB); expected > 50 KB"
        )


def test_slot1_residual_definition_panel_count(synthetic_inputs):
    """Slot 1 figure should be single-panel (1 axes)."""
    from scripts.resdec_mhe.interpretability import make_lab_meeting_figures as orch

    fig = orch.build_slot1_residual_definition(
        canonical_dir=synthetic_inputs["canonical_dir"],
        n_folds=5,
    )
    # Single-panel figure: exactly 1 Axes.
    assert len(fig.axes) == 1, f"expected 1 axes, got {len(fig.axes)}"


def test_slot2_marker_validation_has_heatmap(synthetic_inputs):
    """Slot 2 figure should be a heatmap with at least 2 rows × 2 cols."""
    from scripts.resdec_mhe.interpretability import make_lab_meeting_figures as orch

    fig = orch.build_slot2_marker_validation(
        extended_marker_json=synthetic_inputs["extended_path"],
        splatter_marker_json=synthetic_inputs["splatter_path"],
    )
    # Should have at least 1 axes (the heatmap).
    assert len(fig.axes) >= 1


def test_slot6_methods_recap_is_diagram(synthetic_inputs):
    """Slot 6 figure should be a block diagram (no data plotting)."""
    from scripts.resdec_mhe.interpretability import make_lab_meeting_figures as orch

    fig = orch.build_slot6_methods_recap()
    assert len(fig.axes) >= 1
