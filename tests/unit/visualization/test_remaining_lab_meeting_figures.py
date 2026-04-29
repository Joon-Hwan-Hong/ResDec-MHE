"""Smoke tests for the 3 remaining lab-meeting figures (Slot 4.3, S1, S8).

Verifies that ``make_remaining_lab_meeting_figures.py`` produces:
  - fig_slot4_3_statistical_rigor.{png,pdf}
  - fig_S1_per_fold_r2_strip.{png,pdf}
  - fig_S8_published_marker_concordance.{png,pdf}

Each PNG must exist and exceed 50 KB.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pytest

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))


def _write_synthetic_perm_summary(path: Path) -> None:
    payload = {
        "canonical_mean_r2": 0.4436,
        "n_permutations": 10,
        "null_mean_r2_per_perm": [
            -0.337, -0.241, -0.145, -0.365, -0.322,
            -0.453, -0.298, -0.250, -0.229, -0.304,
        ],
        "null_mean": -0.294,
        "null_std": 0.085,
        "n_perms_ge_canonical": 0,
        "p_value_one_sided": 0.0909,
        "z_under_null": 8.73,
    }
    path.write_text(json.dumps(payload))


def _write_synthetic_statistical_rigor(path: Path) -> None:
    payload = {
        "bootstrap_r2_ci": {
            "point_r2": 0.449,
            "ci_lower": 0.373,
            "ci_upper": 0.507,
            "n_boot": 1000,
            "conf": 0.95,
            "n": 516,
        },
        "provenance": {
            "per_fold_r2": {
                "ours": [0.4842, 0.4171, 0.5717, 0.2990, 0.4461],
                "tabpfn_2_6_standalone": [0.474, 0.383, 0.526, 0.276, 0.338],
            },
        },
    }
    path.write_text(json.dumps(payload))


def _write_synthetic_seed_variation(path: Path) -> None:
    payload = {
        "seeds": [42, 67, 21, 2000, 426],
        "per_baseline": {
            "TabPFN-2.6": {
                "per_seed": {
                    "42":   {"wilcoxon_p_one_sided_greater": 0.03125},
                    "67":   {"wilcoxon_p_one_sided_greater": 0.03125},
                    "21":   {"wilcoxon_p_one_sided_greater": 0.03125},
                    "2000": {"wilcoxon_p_one_sided_greater": 0.0625},
                    "426":  {"wilcoxon_p_one_sided_greater": 0.03125},
                },
                "stouffer_p_one_sided": 2.93e-05,
            },
            "MixMIL": {
                "per_seed": {
                    "42":   {"wilcoxon_p_one_sided_greater": 0.03125},
                    "67":   {"wilcoxon_p_one_sided_greater": 0.03125},
                    "21":   {"wilcoxon_p_one_sided_greater": 0.03125},
                    "2000": {"wilcoxon_p_one_sided_greater": 0.03125},
                    "426":  {"wilcoxon_p_one_sided_greater": 0.03125},
                },
                "stouffer_p_one_sided": 1.55e-05,
            },
            "scPhase": {
                "per_seed": {
                    "42":   {"wilcoxon_p_one_sided_greater": 0.03125},
                    "67":   {"wilcoxon_p_one_sided_greater": 0.03125},
                    "21":   {"wilcoxon_p_one_sided_greater": 0.03125},
                    "2000": {"wilcoxon_p_one_sided_greater": 0.03125},
                    "426":  {"wilcoxon_p_one_sided_greater": 0.03125},
                },
                "stouffer_p_one_sided": 1.55e-05,
            },
        },
    }
    path.write_text(json.dumps(payload))


def _write_synthetic_baseline_table(path: Path) -> None:
    rows = [
        "model,display_name,n_folds,r2_mean,r2_std",
        "tabpfn_2_6_standalone,TabPFN-2.6 standalone,5,0.399,0.10",
        "ridge_A,Ridge [A],5,0.270,0.08",
        "xgboost_A,XGBoost [A],5,0.352,0.05",
        "randomforest_A,RandomForest [A],5,0.308,0.07",
        "mixmil,MixMIL (Engelmann),5,0.157,0.07",
        "scphase,scPhase (Berson),5,-0.074,0.05",
        "p5_canonical_seed42,ResDec-MHE (canonical),5,0.4436,0.0996",
    ]
    path.write_text("\n".join(rows) + "\n")


@pytest.fixture
def synthetic_inputs(tmp_path: Path) -> dict[str, Path]:
    perm = tmp_path / "permutation_summary.json"
    rigor = tmp_path / "statistical_rigor.json"
    wilcoxon = tmp_path / "seed_variation_wilcoxon_all_baselines.json"
    baselines = tmp_path / "paper_baseline_table.csv"
    marker_md = tmp_path / "published_marker_concordance.md"
    out_dir = tmp_path / "lab_meeting_remaining"
    _write_synthetic_perm_summary(perm)
    _write_synthetic_statistical_rigor(rigor)
    _write_synthetic_seed_variation(wilcoxon)
    _write_synthetic_baseline_table(baselines)
    # marker MD only used for source-name annotation; content does not need to be parsed.
    marker_md.write_text("# Stub\n")
    return {
        "perm": perm,
        "rigor": rigor,
        "wilcoxon": wilcoxon,
        "baselines": baselines,
        "marker_md": marker_md,
        "out_dir": out_dir,
    }


def test_orchestrator_module_imports():
    from scripts.resdec_mhe.interpretability import (  # noqa: F401
        make_remaining_lab_meeting_figures as orch,
    )


def test_all_three_figures_written(synthetic_inputs):
    from scripts.resdec_mhe.interpretability import (
        make_remaining_lab_meeting_figures as orch,
    )

    orch.build_all_figures(
        permutation_summary=synthetic_inputs["perm"],
        statistical_rigor=synthetic_inputs["rigor"],
        seed_wilcoxon=synthetic_inputs["wilcoxon"],
        baseline_table=synthetic_inputs["baselines"],
        marker_md=synthetic_inputs["marker_md"],
        out_dir=synthetic_inputs["out_dir"],
    )

    out_dir = synthetic_inputs["out_dir"]
    expected_stems = [
        "fig_slot4_3_statistical_rigor",
        "fig_S1_per_fold_r2_strip",
        "fig_S8_published_marker_concordance",
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


def test_slot4_3_has_three_panels(synthetic_inputs):
    from scripts.resdec_mhe.interpretability import (
        make_remaining_lab_meeting_figures as orch,
    )

    fig = orch.build_slot4_3_statistical_rigor(
        permutation_summary=synthetic_inputs["perm"],
        statistical_rigor=synthetic_inputs["rigor"],
        seed_wilcoxon=synthetic_inputs["wilcoxon"],
        baseline_table=synthetic_inputs["baselines"],
    )
    # 3-panel composite via make_panel
    assert len(fig.axes) >= 3, f"expected 3+ axes, got {len(fig.axes)}"


def test_S1_per_fold_strip_single_panel(synthetic_inputs):
    from scripts.resdec_mhe.interpretability import (
        make_remaining_lab_meeting_figures as orch,
    )

    fig = orch.build_S1_per_fold_r2_strip(
        statistical_rigor=synthetic_inputs["rigor"],
    )
    assert len(fig.axes) == 1, f"expected single axes, got {len(fig.axes)}"


def test_S8_published_marker_single_panel(synthetic_inputs):
    from scripts.resdec_mhe.interpretability import (
        make_remaining_lab_meeting_figures as orch,
    )

    fig = orch.build_S8_published_marker_concordance(
        marker_md=synthetic_inputs["marker_md"],
    )
    assert len(fig.axes) >= 1
