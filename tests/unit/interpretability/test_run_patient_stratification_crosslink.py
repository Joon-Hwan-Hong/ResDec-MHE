"""Tests for run_patient_stratification_crosslink.py (EXP-039).

Unit tests on the small helpers (apoe_e4_dose, load_ccc_outliers,
fit_residual_gmm, pair_test, bh_fdr) plus an end-to-end smoke test that
invokes ``main()`` against the canonical artefacts on disk.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from scripts.resdec_mhe.interpretability import (  # noqa: E402
    run_patient_stratification_crosslink as mod,
)


# =============================================================================
# apoe_e4_dose
# =============================================================================


class TestApoeE4Dose:

    @pytest.mark.parametrize(
        "geno,expected",
        [
            (22.0, 0),
            (23.0, 0),
            (33.0, 0),
            (24.0, 1),
            (34.0, 1),
            (44.0, 2),
        ],
    )
    def test_known_genotypes(self, geno, expected):
        assert mod.apoe_e4_dose(geno) == expected

    def test_nan_is_none(self):
        assert mod.apoe_e4_dose(float("nan")) is None

    def test_inf_is_none(self):
        assert mod.apoe_e4_dose(float("inf")) is None

    def test_unknown_code_is_none(self):
        # APOE only has alleles 2/3/4; e.g. "55" is not a real human genotype.
        assert mod.apoe_e4_dose(55.0) is None

    def test_int_input_works(self):
        assert mod.apoe_e4_dose(34) == 1

    def test_string_unparseable_is_none(self):
        assert mod.apoe_e4_dose("garbage") is None


# =============================================================================
# load_ccc_outliers
# =============================================================================


class TestLoadCccOutliers:
    """Round-trip a synthetic threshold_sensitivity.json structure."""

    def test_extracts_outlier_subjects_at_tau(self, tmp_path):
        synth = {
            "config": {"thresholds": [0.005, 0.01]},
            "per_threshold": [
                {
                    "threshold": 0.005,
                    "outlier_subjects": [
                        {"subject_id": "R1"},
                        {"subject_id": "R2"},
                    ],
                },
                {
                    "threshold": 0.01,
                    "outlier_subjects": [
                        {"subject_id": "R3"},
                        {"subject_id": "R4"},
                        {"subject_id": "R5"},
                    ],
                },
            ],
        }
        path = tmp_path / "ts.json"
        path.write_text(json.dumps(synth))
        ids = mod.load_ccc_outliers(path, tau=0.01)
        assert ids == {"R3", "R4", "R5"}

    def test_missing_tau_raises(self, tmp_path):
        path = tmp_path / "ts.json"
        path.write_text(json.dumps({
            "per_threshold": [
                {"threshold": 0.005, "outlier_subjects": []},
            ],
        }))
        with pytest.raises(KeyError, match="τ=0.05"):
            mod.load_ccc_outliers(path, tau=0.05)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            mod.load_ccc_outliers(tmp_path / "nope.json", tau=0.01)


# =============================================================================
# fit_residual_gmm
# =============================================================================


class TestFitResidualGmm:

    def test_reproduces_canonical_k4(self):
        """Re-fit the GMM on the canonical residual CSV and verify cluster
        sizes match `latent_class_k4_crosstab.json` byte-for-byte."""
        residual_csv = (
            _WORKTREE_ROOT
            / "outputs/canonical/interpretability/residual_per_subject.csv"
        )
        if not residual_csv.exists():
            pytest.skip(f"canonical residual CSV missing: {residual_csv}")
        ref_json = (
            _WORKTREE_ROOT
            / "outputs/canonical/interpretability/latent_class_k4_crosstab.json"
        )
        if not ref_json.exists():
            pytest.skip(f"reference crosstab JSON missing: {ref_json}")

        rdf = pd.read_csv(residual_csv)
        rdf = rdf.loc[np.isfinite(rdf["residual"])].reset_index(drop=True)
        cluster = mod.fit_residual_gmm(
            rdf["residual"].to_numpy(), n_components=4, random_state=0,
        )
        sizes = np.bincount(cluster, minlength=4).tolist()

        with ref_json.open() as fh:
            ref = json.load(fh)
        ref_sizes = [
            ref["cluster_sizes"][f"cluster_{i}"] for i in range(4)
        ]
        assert sizes == ref_sizes, (
            f"GMM cluster sizes drift from EXP-033 reference: "
            f"got {sizes}, ref {ref_sizes}"
        )

    def test_too_few_finite_raises(self):
        with pytest.raises(ValueError, match="finite residuals"):
            mod.fit_residual_gmm(
                np.array([1.0, 2.0]), n_components=4,
            )

    def test_nonfinite_raises(self):
        with pytest.raises(ValueError, match="non-finite"):
            mod.fit_residual_gmm(
                np.array([1.0, 2.0, np.nan, 3.0, 4.0]),
                n_components=2,
            )


# =============================================================================
# pair_test
# =============================================================================


class TestPairTest:

    def test_2x2_uses_fisher_exact(self):
        a = pd.Series(["X"] * 5 + ["Y"] * 5)
        b = pd.Series(["a"] * 5 + ["b"] * 5)
        res = mod.pair_test(a, b)
        # Perfectly aligned 2x2 → Fisher p ≈ small.
        assert res["test"] == "fisher_exact"
        assert res["p_value"] < 0.05
        assert res["dof"] is None

    def test_3x2_uses_chi2(self):
        a = pd.Series(["A", "A", "B", "B", "C", "C"] * 4)
        b = pd.Series(["x", "y"] * 12)
        res = mod.pair_test(a, b)
        assert res["test"] == "chi2_contingency"
        assert res["dof"] == 2

    def test_degenerate_returns_p_one(self):
        a = pd.Series(["X"] * 10)  # only one unique value
        b = pd.Series(["a"] * 10)
        res = mod.pair_test(a, b)
        assert res["test"] == "degenerate"
        assert res["p_value"] == 1.0

    def test_drops_nan_rows(self):
        a = pd.Series(["X", "X", None, None, "Y", "Y"])
        b = pd.Series(["a", "a", "b", "b", "b", "b"])
        res = mod.pair_test(a, b)
        assert res["n_dropped"] == 2
        assert res["n_used"] == 4


# =============================================================================
# bh_fdr
# =============================================================================


class TestBhFdr:

    def test_monotone(self):
        # If all p-values are 0.5 → all q = 0.5; if mixed, q ≥ p in worst case.
        p = [0.001, 0.05, 0.5, 0.7]
        q = mod.bh_fdr(p)
        # BH q-values are non-decreasing in sorted-p order.
        assert q[0] <= q[1] <= q[2] <= q[3]

    def test_p_in_range(self):
        with pytest.raises(ValueError):
            mod.bh_fdr([-0.1, 0.5])
        with pytest.raises(ValueError):
            mod.bh_fdr([0.5, 1.5])


# =============================================================================
# End-to-end smoke (canonical artefacts on disk)
# =============================================================================


class TestEndToEnd:
    """One full subprocess invocation to verify the orchestrator runs and
    writes JSON / MD / PNG / PDF with the documented schema."""

    def test_full_pipeline_smoke(self, tmp_path):
        residual_csv = (
            _WORKTREE_ROOT
            / "outputs/canonical/interpretability/residual_per_subject.csv"
        )
        if not residual_csv.exists():
            pytest.skip(f"canonical residual CSV missing: {residual_csv}")
        ccc_threshold_json = (
            _WORKTREE_ROOT
            / "outputs/canonical/interpretability/ccc_heterogeneity/threshold_sensitivity.json"
        )
        if not ccc_threshold_json.exists():
            pytest.skip(
                f"canonical threshold sensitivity JSON missing: {ccc_threshold_json}"
            )
        cf_fold0 = (
            _WORKTREE_ROOT
            / "outputs/canonical/interpretability/counterfactuals_optimized_absolute_delta0p3"
        )
        if not (cf_fold0 / "counterfactuals_fold0.json").exists():
            pytest.skip(f"fold0 CF JSON missing under {cf_fold0}")
        pred_root = _WORKTREE_ROOT / "outputs/canonical/p5_canonical_seed42"
        if not (pred_root / "fold0/val_predictions_best.npz").exists():
            pytest.skip("canonical fold0 predictions missing")

        out_json = tmp_path / "out.json"
        out_md = tmp_path / "out.md"
        out_fig_dir = tmp_path / "fig"

        rc = mod.main([
            "--out-json", str(out_json),
            "--out-md", str(out_md),
            "--out-fig-dir", str(out_fig_dir),
            "--log-level", "WARNING",
        ])
        assert rc == 0
        assert out_json.exists()
        assert out_md.exists()
        assert (out_fig_dir / "fig_patient_stratification_crosslink.png").exists()
        assert (out_fig_dir / "fig_patient_stratification_crosslink.pdf").exists()

        with out_json.open() as fh:
            d = json.load(fh)

        # Schema checks.
        for k in [
            "config", "gmm_metadata", "membership_counts", "pairwise_tests",
            "per_cluster_sex_r2", "cf_success_per_cluster",
            "canonical_per_fold_r2",
        ]:
            assert k in d, f"missing key: {k}"

        # Sanity: subject count.
        assert d["config"]["n_subjects"] == 516

        # Sanity: all 4 clusters present.
        assert set(d["membership_counts"]["cluster"].keys()) == {
            "k0", "k1", "k2", "k3",
        }

        # 10 unique unordered pairs from 5 axes.
        assert len(d["pairwise_tests"]["per_pair"]) == 10
        for r in d["pairwise_tests"]["per_pair"].values():
            assert "p_value" in r
            assert "q_value_bh" in r

        # CF labels: 100 Y/N labelled, ~416 N/A. Just check the totals add up.
        cf = d["membership_counts"]["cf_success"]
        assert cf["Y"] + cf["N"] + cf["N/A"] == 516

        # CCC outliers: τ=0.01 has 15.
        assert d["membership_counts"]["ccc_outlier"]["Y"] == 15
        assert d["membership_counts"]["ccc_outlier"]["N"] == 501
