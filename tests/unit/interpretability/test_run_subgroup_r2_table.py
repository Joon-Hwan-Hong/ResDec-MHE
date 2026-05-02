"""Tests for run_subgroup_r2_table.py.

A mix of unit tests on the small pure helpers (tertile labelling,
strata assignment, per-fold metric computation, markdown rendering)
plus a single end-to-end smoke test that exercises the orchestrator
against the canonical predictions on disk and asserts the JSON / MD /
figure artefacts have plausible contents.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.conftest import WORKTREE_ROOT as _WORKTREE_ROOT

from scripts.resdec_mhe.interpretability import (  # noqa: E402
    run_subgroup_r2_table as mod,
)

# =============================================================================
# Tertile labelling
# =============================================================================

class TestTertileLabel:
    """``_tertile_label`` correctly buckets values against ``(q1, q2)``."""

    def test_below_q1_is_t1(self) -> None:
        assert mod._tertile_label(1.0, (5.0, 10.0)) == "T1"

    def test_between_q1_q2_is_t2(self) -> None:
        assert mod._tertile_label(7.5, (5.0, 10.0)) == "T2"
        # Equality with q1 lands in T2 (upper bucket on the boundary).
        assert mod._tertile_label(5.0, (5.0, 10.0)) == "T2"

    def test_above_q2_is_t3(self) -> None:
        assert mod._tertile_label(15.0, (5.0, 10.0)) == "T3"
        # Equality with q2 lands in T3.
        assert mod._tertile_label(10.0, (5.0, 10.0)) == "T3"

    def test_nan_returns_none(self) -> None:
        assert mod._tertile_label(float("nan"), (5.0, 10.0)) is None
        assert mod._tertile_label(float("inf"), (5.0, 10.0)) is None

# =============================================================================
# AD-dx labelling
# =============================================================================

class TestAdDxLabel:
    """``_ad_dx_label`` binarizes cogdx codes correctly."""

    @pytest.mark.parametrize("code", [4.0, 5.0])
    def test_ad_codes(self, code: float) -> None:
        assert mod._ad_dx_label(code) == "AD"

    @pytest.mark.parametrize("code", [1.0, 2.0, 3.0])
    def test_nonad_codes(self, code: float) -> None:
        assert mod._ad_dx_label(code) == "non-AD"

    def test_other_dementia_excluded(self) -> None:
        assert mod._ad_dx_label(6.0) is None

    def test_nan_excluded(self) -> None:
        assert mod._ad_dx_label(float("nan")) is None

# =============================================================================
# Tertile cut computation
# =============================================================================

class TestComputeTertileCuts:
    def test_uniform_distribution_split_evenly(self) -> None:
        s = pd.Series(np.arange(1, 10))  # 1..9 → tertiles at ~3.67 and ~6.33
        q1, q2 = mod._compute_tertile_cuts(s)
        # np.quantile(1..9, 1/3) = 3.6666; (..., 2/3) = 6.3333
        assert q1 == pytest.approx(3.6666666, rel=1e-3)
        assert q2 == pytest.approx(6.3333333, rel=1e-3)

    def test_drops_nans(self) -> None:
        s = pd.Series([1.0, 2.0, float("nan"), 3.0, 4.0, float("nan"), 5.0, 6.0])
        # On non-null [1..6]: q1=2.6667, q2=4.3333
        q1, q2 = mod._compute_tertile_cuts(s)
        assert q1 == pytest.approx(2.6666666, rel=1e-3)
        assert q2 == pytest.approx(4.3333333, rel=1e-3)

    def test_too_few_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot compute tertiles"):
            mod._compute_tertile_cuts(pd.Series([1.0, 2.0]))

# =============================================================================
# assign_strata
# =============================================================================

class TestAssignStrata:
    """Synthetic 12-row frame covering every stratum value."""

    @staticmethod
    def _make_df() -> pd.DataFrame:
        return pd.DataFrame({
            # APOE: 33→0, 34→1, 44→2, 23→0
            "apoe_genotype": [33, 33, 34, 34, 44, 44, 23, 23, 33, 34, 44, 33],
            "msex": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            # Age range so tertiles split predictably.
            "age_death": [70, 75, 80, 85, 90, 72, 78, 82, 88, 92, 76, 86],
            "educ": [10, 12, 14, 16, 18, 11, 13, 15, 17, 19, 12, 16],
            "cogdx": [1, 2, 3, 4, 5, 6, 1, 2, 4, 5, 3, 4],
        })

    def test_apoe_dosage(self) -> None:
        df = self._make_df()
        out, _ = mod.assign_strata(df)
        # First two rows are 33 (zero ε4); two of the 44 rows are dosage 2.
        assert out.loc[0, "stratum_apoe"] == "0"
        assert out.loc[2, "stratum_apoe"] == "1"
        assert out.loc[4, "stratum_apoe"] == "2"

    def test_sex_strings(self) -> None:
        df = self._make_df()
        out, _ = mod.assign_strata(df)
        assert out.loc[0, "stratum_sex"] == "female"
        assert out.loc[1, "stratum_sex"] == "male"

    def test_age_tertiles_are_balanced(self) -> None:
        df = self._make_df()
        out, _ = mod.assign_strata(df)
        counts = out["stratum_age"].value_counts()
        # Balanced ~ 4 per tertile across n=12 (modulo boundaries).
        assert counts.get("T1", 0) >= 3
        assert counts.get("T3", 0) >= 3

    def test_educ_tertiles_present(self) -> None:
        df = self._make_df()
        out, meta = mod.assign_strata(df)
        assert set(out["stratum_educ"].dropna().unique()).issubset({"T1", "T2", "T3"})
        assert "tertile_cuts_q1_q2" in meta["educ"]

    def test_cogdx_excludes_six(self) -> None:
        df = self._make_df()
        out, _ = mod.assign_strata(df)
        # cogdx=6 → exclude.
        cogdx6_mask = (df["cogdx"] == 6)
        assert out.loc[cogdx6_mask, "stratum_addx"].isna().all()
        # cogdx=4 → AD; cogdx=1 → non-AD.
        assert out.loc[df["cogdx"] == 4, "stratum_addx"].eq("AD").all()
        assert out.loc[df["cogdx"] == 1, "stratum_addx"].eq("non-AD").all()

# =============================================================================
# _per_fold_metrics + _aggregate
# =============================================================================

class TestPerFoldMetrics:
    @staticmethod
    def _two_fold_df() -> pd.DataFrame:
        # fold 0 = 5 perfect predictions in stratum X; fold 1 = 5 noisy ones.
        rng = np.random.default_rng(0)
        y_true_f0 = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        y_pred_f0 = y_true_f0.copy()  # perfect → r²=1, mae=0
        y_true_f1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        # Add a deterministic noise pattern.
        y_pred_f1 = y_true_f1 + np.array([0.05, -0.05, 0.0, -0.1, 0.1])
        return pd.DataFrame({
            "fold": [0] * 5 + [1] * 5,
            "y_true": np.concatenate([y_true_f0, y_true_f1]),
            "y_composite": np.concatenate([y_pred_f0, y_pred_f1]),
            "stratum_x": ["X"] * 10,
        })

    def test_perfect_fold_r2_one(self) -> None:
        df = self._two_fold_df()
        m = mod._per_fold_metrics(df, n_folds=2, stratum_col="stratum_x", stratum_value="X")
        assert m["r2"][0] == pytest.approx(1.0)
        assert m["mae"][0] == pytest.approx(0.0)

    def test_noisy_fold_r2_less_than_one(self) -> None:
        df = self._two_fold_df()
        m = mod._per_fold_metrics(df, n_folds=2, stratum_col="stratum_x", stratum_value="X")
        assert m["r2"][1] < 1.0
        assert m["mae"][1] > 0.0

    def test_too_few_subjects_yields_nan(self) -> None:
        # Single-subject fold → metrics NaN.
        df = pd.DataFrame({
            "fold": [0, 0, 1],
            "y_true": [0.1, 0.2, 0.5],
            "y_composite": [0.1, 0.2, 0.5],
            "stratum_x": ["X", "X", "X"],
        })
        m = mod._per_fold_metrics(
            df, n_folds=2, stratum_col="stratum_x", stratum_value="X",
        )
        assert math.isnan(m["r2"][1])
        assert math.isnan(m["pearson_r"][1])
        assert math.isnan(m["mae"][1])
        assert m["n"][1] == 1.0

    def test_aggregate_ignores_nans(self) -> None:
        mean, std = mod._aggregate([0.5, 0.4, float("nan"), 0.6])
        assert mean == pytest.approx((0.5 + 0.4 + 0.6) / 3, rel=1e-6)
        # ddof=1 sample std on 3 finite values.
        assert std == pytest.approx(np.std([0.5, 0.4, 0.6], ddof=1), rel=1e-6)

    def test_aggregate_single_finite_returns_nan_std(self) -> None:
        mean, std = mod._aggregate([0.5, float("nan"), float("nan")])
        assert mean == pytest.approx(0.5)
        assert math.isnan(std)

    def test_aggregate_all_nan(self) -> None:
        mean, std = mod._aggregate([float("nan"), float("nan")])
        assert math.isnan(mean)
        assert math.isnan(std)

# =============================================================================
# compute_subgroup_r2_table
# =============================================================================

class TestComputeSubgroupR2Table:
    def test_table_shape_and_keys(self) -> None:
        rng = np.random.default_rng(42)
        n = 60
        df = pd.DataFrame({
            "fold": np.repeat(np.arange(5), 12),
            "y_true": rng.standard_normal(n),
            "y_composite": rng.standard_normal(n),
            "stratum_apoe": ["0", "1", "2"] * 20,
            "stratum_sex": ["female", "male"] * 30,
            "stratum_age": ["T1", "T2", "T3"] * 20,
            "stratum_educ": ["T1", "T2", "T3"] * 20,
            "stratum_addx": (["AD", "non-AD"] * 30),
        })
        table = mod.compute_subgroup_r2_table(df, n_folds=5)
        assert set(table.keys()) == {"apoe", "sex", "age", "educ", "addx"}
        assert set(table["apoe"].keys()) == {"0", "1", "2"}
        assert set(table["sex"].keys()) == {"female", "male"}
        assert set(table["age"].keys()) == {"T1", "T2", "T3"}
        assert set(table["educ"].keys()) == {"T1", "T2", "T3"}
        assert set(table["addx"].keys()) == {"non-AD", "AD"}
        for fam, strata in table.items():
            for stratum, stats in strata.items():
                assert "n_total" in stats
                assert len(stats["per_fold_r2"]) == 5
                assert len(stats["per_fold_pearson_r"]) == 5
                assert len(stats["per_fold_mae"]) == 5
                assert len(stats["n_per_fold"]) == 5

# =============================================================================
# Markdown rendering
# =============================================================================

class TestRenderMarkdownTable:
    def test_produces_header_and_rows(self) -> None:
        table = {
            "apoe": {
                "0": {
                    "n_total": 100,
                    "n_per_fold": [20, 20, 20, 20, 20],
                    "per_fold_r2": [0.4, 0.41, 0.42, 0.43, 0.44],
                    "mean_r2": 0.42,
                    "std_r2": 0.015,
                    "per_fold_pearson_r": [0.65, 0.66, 0.67, 0.68, 0.69],
                    "mean_pearson_r": 0.67,
                    "std_pearson_r": 0.014,
                    "per_fold_mae": [0.8, 0.82, 0.81, 0.83, 0.79],
                    "mean_mae": 0.81,
                    "std_mae": 0.015,
                },
            },
        }
        md = mod.render_markdown_table(table, canonical_r2_per_fold=[0.4, 0.45, 0.42, 0.43, 0.46])
        assert "| Subgroup | Stratum | n |" in md
        assert "APOE-ε4 dosage" in md
        assert "ε4=0" in md
        assert "0.420" in md  # mean R²
        assert "Overall" in md  # canonical row present
        # Notes section also present.
        assert "Smallest stratum" in md
        assert "Highest mean R²" in md
        assert "Lowest mean R²" in md

# =============================================================================
# End-to-end smoke (uses real predictions if available)
# =============================================================================

def test_orchestrator_runs_against_canonical(tmp_path: Path) -> None:
    """End-to-end: invoke the script as a subprocess and inspect the JSON / MD / figure."""
    pred_root = _WORKTREE_ROOT / "outputs/canonical/p5_canonical_seed42"
    tabpfn_dir = _WORKTREE_ROOT / "data/canonical"
    metadata_csv = _WORKTREE_ROOT / "data/metadata_ROSMAP/metadata.csv"
    if not pred_root.is_dir() or not tabpfn_dir.is_dir() or not metadata_csv.is_file():
        pytest.skip("Canonical predictions / metadata missing.")
    if not (pred_root / "fold0/val_predictions_best.npz").is_file():
        pytest.skip("fold0/val_predictions_best.npz missing.")
    if not (tabpfn_dir / "tabpfn_outer_fold0.npz").is_file():
        pytest.skip("tabpfn_outer_fold0.npz missing.")

    out_json = tmp_path / "subgroup_r2_unified.json"
    out_md = tmp_path / "subgroup_r2_unified.md"
    out_fig_dir = tmp_path / "fig_subgroup_r2"
    script = (
        _WORKTREE_ROOT
        / "scripts/resdec_mhe/interpretability/run_subgroup_r2_table.py"
    )
    cmd = [
        sys.executable,
        str(script),
        "--pred-root", str(pred_root),
        "--tabpfn-dir", str(tabpfn_dir),
        "--metadata-csv", str(metadata_csv),
        "--out-json", str(out_json),
        "--out-md", str(out_md),
        "--out-fig-dir", str(out_fig_dir),
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(_WORKTREE_ROOT),
    )
    assert result.returncode == 0, (
        f"orchestrator failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    assert out_json.is_file() and out_json.stat().st_size > 200
    assert out_md.is_file() and out_md.stat().st_size > 200
    png = out_fig_dir / "fig_subgroup_r2.png"
    pdf = out_fig_dir / "fig_subgroup_r2.pdf"
    assert png.is_file() and png.stat().st_size > 1000
    assert pdf.is_file() and pdf.stat().st_size > 1000

    payload = json.loads(out_json.read_text())
    assert "subgroup_table" in payload
    table = payload["subgroup_table"]
    assert set(table.keys()) == {"apoe", "sex", "age", "educ", "addx"}
    # APOE: three strata, totals must roughly equal cohort N (~516 minus
    # subjects missing apoe_genotype).
    apoe_total = sum(table["apoe"][k]["n_total"] for k in ("0", "1", "2"))
    assert apoe_total >= 400
    # AD-dx is binary with no overlap.
    addx_total = (
        table["addx"]["AD"]["n_total"] + table["addx"]["non-AD"]["n_total"]
    )
    assert addx_total >= 400

    # Per-fold R² lists must each have length 5.
    for fam, strata in table.items():
        for stratum, stats in strata.items():
            assert len(stats["per_fold_r2"]) == 5
            assert isinstance(stats["mean_r2"], float)

    assert "overall_per_fold_r2" in payload
    assert len(payload["overall_per_fold_r2"]) == 5
    # Overall R² should be in the documented neighbourhood ~0.44.
    overall = np.asarray(payload["overall_per_fold_r2"], dtype=np.float64)
    assert np.nanmean(overall) > 0.3
