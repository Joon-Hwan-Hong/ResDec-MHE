"""Tests for run_sex_disparity_attribution.py.

Pure-helper unit tests on the small statistical primitives (per-subject
aggregation, ranking, Wilcoxon w/ BH-FDR, classifier heuristic) plus a
single end-to-end smoke test that runs the orchestrator on a synthetic
mini-cohort and checks the JSON / MD / figure artefacts have plausible
contents.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.conftest import WORKTREE_ROOT as _WORKTREE_ROOT

from scripts.resdec_mhe.interpretability import (  # noqa: E402
    run_sex_disparity_attribution as mod,
)

# =============================================================================
# Per-subject aggregation
# =============================================================================

class TestPerSubjectCTMagnitude:
    """``per_subject_ct_magnitude`` returns mean(|attr|) over genes per CT."""

    def test_basic_shape_and_value(self) -> None:
        # 2 subj × 3 CT × 4 genes; CT0 row {-2, -1, 0, 1} → mean|.| = 1.0
        attr = np.zeros((2, 3, 4), dtype=np.float32)
        attr[0, 0, :] = [-2, -1, 0, 1]
        attr[0, 1, :] = [3, 3, 3, 3]
        attr[0, 2, :] = [-1, -1, -1, -1]
        attr[1, 0, :] = [0, 0, 0, 0]
        attr[1, 1, :] = [1, -1, 1, -1]
        attr[1, 2, :] = [10, 0, 0, 0]
        mag = mod.per_subject_ct_magnitude(attr)
        assert mag.shape == (2, 3)
        assert mag[0, 0] == pytest.approx(1.0)
        assert mag[0, 1] == pytest.approx(3.0)
        assert mag[0, 2] == pytest.approx(1.0)
        assert mag[1, 0] == pytest.approx(0.0)
        assert mag[1, 1] == pytest.approx(1.0)
        assert mag[1, 2] == pytest.approx(2.5)

    def test_rejects_non_3d(self) -> None:
        with pytest.raises(ValueError, match="3D"):
            mod.per_subject_ct_magnitude(np.zeros((4, 5)))

class TestPerSubjectPairMagnitude:
    """``per_subject_pair_magnitude`` indexes specific (CT, gene) pairs."""

    def test_basic_indexing(self) -> None:
        attr = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
        pairs = [(0, 0), (1, 2), (2, 3)]
        mag = mod.per_subject_pair_magnitude(attr, pairs)
        assert mag.shape == (2, 3)
        # Subj 0: attr[0, 0, 0] = 0, attr[0, 1, 2] = 6, attr[0, 2, 3] = 11.
        assert mag[0, 0] == pytest.approx(0.0)
        assert mag[0, 1] == pytest.approx(6.0)
        assert mag[0, 2] == pytest.approx(11.0)

    def test_negative_takes_abs(self) -> None:
        attr = np.array([[[-3.0, 5.0]]], dtype=np.float32)  # (1, 1, 2)
        out = mod.per_subject_pair_magnitude(attr, [(0, 0)])
        assert out[0, 0] == pytest.approx(3.0)

    def test_empty_pairs_returns_zero_columns(self) -> None:
        attr = np.zeros((4, 2, 3), dtype=np.float32)
        out = mod.per_subject_pair_magnitude(attr, [])
        assert out.shape == (4, 0)

# =============================================================================
# Ranking helpers
# =============================================================================

class TestRankTopCTPerSex:
    """``rank_top_ct_per_sex`` selects the top-N CTs by per-sex mean."""

    def test_ranks_by_sex_mean(self) -> None:
        # 4 subjects: 2 F, 2 M. F prefers CT0 (mean=10), M prefers CT2.
        ct_mag = np.array(
            [
                [10.0, 1.0, 0.0],
                [10.0, 1.0, 0.0],
                [0.0, 1.0, 10.0],
                [0.0, 1.0, 10.0],
            ],
            dtype=np.float32,
        )
        sex = np.array(["female", "female", "male", "male"])
        ct_names = ["CT0", "CT1", "CT2"]
        out = mod.rank_top_ct_per_sex(ct_mag, sex, ct_names, top_n=2)
        assert out["female"][0]["cell_type"] == "CT0"
        assert out["female"][1]["cell_type"] == "CT1"
        assert out["male"][0]["cell_type"] == "CT2"
        assert out["male"][1]["cell_type"] == "CT1"
        assert out["female"][0]["rank"] == 1
        assert out["female"][1]["rank"] == 2

    def test_handles_empty_sex(self) -> None:
        ct_mag = np.zeros((3, 2), dtype=np.float32)
        sex = np.array(["female", "female", "female"])
        out = mod.rank_top_ct_per_sex(ct_mag, sex, ["A", "B"])
        assert out["female"]
        assert out["male"] == []

    def test_rejects_name_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="ct_names"):
            mod.rank_top_ct_per_sex(
                np.zeros((1, 3), dtype=np.float32),
                np.array(["female"]),
                ct_names=["A", "B"],  # wrong length
            )

class TestRankTopGenesPerCTPerSex:
    """``rank_top_genes_per_ct_per_sex`` ranks genes within each CT and sex."""

    def test_picks_high_mean_genes(self) -> None:
        # 2 F, 2 M, 1 CT, 4 genes. F: gene0 dominates. M: gene3 dominates.
        attr = np.zeros((4, 1, 4), dtype=np.float32)
        attr[0, 0, :] = [10, 1, 0, 0]
        attr[1, 0, :] = [10, 0, 0, 0]
        attr[2, 0, :] = [0, 0, 0, 10]
        attr[3, 0, :] = [0, 0, 0, 10]
        sex = np.array(["female", "female", "male", "male"])
        out = mod.rank_top_genes_per_ct_per_sex(
            attr, sex, ["CT0"], ["g0", "g1", "g2", "g3"], top_n=2,
        )
        f0 = out["CT0"]["female"][0]
        m0 = out["CT0"]["male"][0]
        assert f0["gene"] == "g0"
        assert m0["gene"] == "g3"
        assert f0["rank"] == 1

# =============================================================================
# Wilcoxon + BH-FDR
# =============================================================================

class TestWilcoxonPerCT:
    """``wilcoxon_per_ct`` returns p / q / means; q is BH-adjusted."""

    def test_clear_difference_yields_low_p(self) -> None:
        # 50 F all 1.0, 50 M all 0.0 → ranksums separates perfectly.
        rng = np.random.default_rng(0)
        f = rng.normal(loc=1.0, scale=0.01, size=50)
        m = rng.normal(loc=0.0, scale=0.01, size=50)
        ct_mag = np.concatenate([f[:, None], m[:, None]], axis=0)
        sex = np.array(["female"] * 50 + ["male"] * 50)
        out = mod.wilcoxon_per_ct(ct_mag, sex, ["CT0"])
        assert out["CT0"]["p_value"] < 1e-10
        assert out["CT0"]["q_value"] < 1e-10
        # Direction: mean F > mean M (statistic > 0 by ranksums convention).
        assert out["CT0"]["mean_diff_F_minus_M"] > 0.5

    def test_no_difference_yields_high_p(self) -> None:
        rng = np.random.default_rng(1)
        f = rng.normal(loc=0.0, scale=1.0, size=50)
        m = rng.normal(loc=0.0, scale=1.0, size=50)
        ct_mag = np.concatenate([f[:, None], m[:, None]], axis=0)
        sex = np.array(["female"] * 50 + ["male"] * 50)
        out = mod.wilcoxon_per_ct(ct_mag, sex, ["CT0"])
        assert out["CT0"]["p_value"] > 0.05

    def test_bh_fdr_q_ge_p(self) -> None:
        # When all CTs have low p, q should be approximately p × (M / rank).
        # Easiest invariant: q ≥ p for the smallest-p entry across multiple CTs.
        rng = np.random.default_rng(2)
        n_ct = 5
        ct_mag = np.zeros((40, n_ct), dtype=np.float32)
        # All CTs separate perfectly → all p ≈ 0.
        ct_mag[:20, :] = rng.normal(loc=1.0, scale=0.01, size=(20, n_ct))
        ct_mag[20:, :] = rng.normal(loc=0.0, scale=0.01, size=(20, n_ct))
        sex = np.array(["female"] * 20 + ["male"] * 20)
        ct_names = [f"CT{c}" for c in range(n_ct)]
        out = mod.wilcoxon_per_ct(ct_mag, sex, ct_names)
        for ct in ct_names:
            assert out[ct]["q_value"] >= out[ct]["p_value"] - 1e-12

class TestWilcoxonPerPair:
    """``wilcoxon_per_pair`` mirrors ``wilcoxon_per_ct`` for (CT, gene) pairs."""

    def test_pair_separation_low_p(self) -> None:
        rng = np.random.default_rng(3)
        attr = np.zeros((40, 1, 2), dtype=np.float32)
        attr[:20, 0, 0] = rng.normal(loc=1.0, scale=0.01, size=20)
        attr[20:, 0, 0] = rng.normal(loc=0.0, scale=0.01, size=20)
        attr[:, 0, 1] = rng.normal(scale=0.5, size=40)
        sex = np.array(["female"] * 20 + ["male"] * 20)
        pairs = [(0, 0, "CT0", "g0", 1.0), (0, 1, "CT0", "g1", 0.0)]
        out = mod.wilcoxon_per_pair(attr, sex, pairs)
        assert out[0]["p_value"] < 1e-5
        assert out[0]["mean_diff_F_minus_M"] > 0.5
        # Second pair has no signal → p should NOT be very small.
        assert out[1]["p_value"] > 0.01

# =============================================================================
# Spearman
# =============================================================================

class TestSpearmanResidualVsCTPerSex:
    """``spearman_residual_vs_ct_per_sex`` returns sex-split rho/p tables."""

    def test_perfect_correlation_returns_high_rho(self) -> None:
        rng = np.random.default_rng(4)
        n = 40
        ct_mag = rng.uniform(size=(n, 1)).astype(np.float32)
        # |residual| = ct_mag (perfect correlation by construction).
        abs_residual = ct_mag[:, 0].astype(np.float64)
        sex = np.array(["female"] * (n // 2) + ["male"] * (n // 2))
        out = mod.spearman_residual_vs_ct_per_sex(
            abs_residual, ct_mag, sex, ["CT0"],
        )
        assert out["female"][0]["rho"] > 0.95
        assert out["male"][0]["rho"] > 0.95

    def test_small_n_returns_empty(self) -> None:
        attr = np.zeros((2, 1), dtype=np.float32)
        sex = np.array(["female", "male"])
        out = mod.spearman_residual_vs_ct_per_sex(
            np.zeros(2), attr, sex, ["CT0"],
        )
        # Each sex has n=1; Spearman undefined → empty list.
        assert out["female"] == []
        assert out["male"] == []

# =============================================================================
# Top-K cohort pairs
# =============================================================================

class TestTopKPairsOverall:
    """``topk_pairs_overall`` returns the top-K (CT, gene) by cohort mean |attr|."""

    def test_picks_largest_means(self) -> None:
        attr = np.zeros((10, 2, 3), dtype=np.float32)
        attr[:, 0, 0] = 5.0  # cohort mean = 5.0
        attr[:, 1, 2] = 3.0  # cohort mean = 3.0
        attr[:, 0, 1] = 1.0  # cohort mean = 1.0
        ct_names = ["CT0", "CT1"]
        gene_names = ["g0", "g1", "g2"]
        out = mod.topk_pairs_overall(attr, ct_names, gene_names, top_k=2)
        assert len(out) == 2
        assert out[0][2] == "CT0"
        assert out[0][3] == "g0"
        assert out[0][4] == pytest.approx(5.0)
        assert out[1][2] == "CT1"
        assert out[1][3] == "g2"

# =============================================================================
# Classifier heuristic
# =============================================================================

class TestClassifyExplanation:
    """``classify_explanation`` picks one of (a) / (b) / (a+c) given inputs."""

    def _r2(self, female: list[float], male: list[float]) -> dict:
        return {
            "female": {
                "per_fold_r2": female,
                "mean_r2": float(np.nanmean(female)),
                "std_r2": float(np.nanstd(female, ddof=1)),
            },
            "male": {
                "per_fold_r2": male,
                "mean_r2": float(np.nanmean(male)),
                "std_r2": float(np.nanstd(male, ddof=1)),
            },
        }

    def test_signature_explanation_when_multiple_sig_ct(self) -> None:
        out = mod.classify_explanation(
            n_sig_ct=3,
            n_sig_pair=0,
            sex_per_fold_r2=self._r2([0.5, 0.5, 0.5], [0.4, 0.4, 0.4]),
            top_ct_overlap=5,
            top_ct_total=5,
            mean_abs_residual_female=0.6,
            mean_abs_residual_male=0.6,
        )
        assert "(b)" in out

    def test_signature_explanation_when_top_lists_differ(self) -> None:
        # Top-CT membership disagreement → (b) regardless of n_sig.
        out = mod.classify_explanation(
            n_sig_ct=0,
            n_sig_pair=0,
            sex_per_fold_r2=self._r2([0.5, 0.5, 0.5], [0.4, 0.4, 0.4]),
            top_ct_overlap=2,
            top_ct_total=5,
            mean_abs_residual_female=0.6,
            mean_abs_residual_male=0.6,
        )
        assert "(b)" in out

    def test_single_sig_ct_does_not_force_signature(self) -> None:
        # n_sig_CT=1 alone should NOT trigger (b) — too marginal given the
        # 31-CT test universe; classifier should fall through to (a) or (c).
        out = mod.classify_explanation(
            n_sig_ct=1,
            n_sig_pair=0,
            sex_per_fold_r2=self._r2(
                [0.5, 0.5, 0.5, 0.5], [0.45, 0.5, 0.5, 0.45],
            ),
            top_ct_overlap=5,
            top_ct_total=5,
            mean_abs_residual_female=0.6,
            mean_abs_residual_male=0.6,
        )
        assert "(b)" not in out

    def test_male_heterogeneity_with_similar_residuals(self) -> None:
        out = mod.classify_explanation(
            n_sig_ct=1,
            n_sig_pair=0,
            sex_per_fold_r2=self._r2(
                [0.5, 0.5, 0.5, 0.5], [0.6, -0.3, 0.5, 0.1],
            ),
            top_ct_overlap=5,
            top_ct_total=5,
            mean_abs_residual_female=0.7,
            mean_abs_residual_male=0.65,
        )
        # Similar residuals + high male variance → (a)+(c).
        assert "(c)" in out
        assert "(a)" in out

    def test_noise_when_lists_overlap_and_no_sig(self) -> None:
        out = mod.classify_explanation(
            n_sig_ct=0,
            n_sig_pair=0,
            sex_per_fold_r2=self._r2(
                [0.5, 0.5, 0.5, 0.5], [0.45, 0.5, 0.5, 0.45],
            ),
            top_ct_overlap=5,
            top_ct_total=5,
            mean_abs_residual_female=0.6,
            mean_abs_residual_male=0.6,
        )
        assert "(a)" in out and "noise" in out

# =============================================================================
# Per-fold R² by sex
# =============================================================================

class TestPerFoldR2BySex:
    """``per_fold_r2_by_sex`` computes per-fold R² over each sex slice."""

    def test_perfect_predictions_yield_r2_1(self) -> None:
        pred_df = pd.DataFrame(
            {
                "ROSMAP_IndividualID": [f"R{i}" for i in range(20)],
                "fold": [0] * 10 + [1] * 10,
                "y_true": np.linspace(0, 1, 20),
                "y_composite": np.linspace(0, 1, 20),
            }
        )
        sex_df = pd.DataFrame(
            {
                "ROSMAP_IndividualID": [f"R{i}" for i in range(20)],
                "sex": ["female"] * 10 + ["male"] * 10,
            }
        )
        out = mod.per_fold_r2_by_sex(pred_df, sex_df, n_folds=2)
        # Each fold has a single sex (10 subj), R² should be 1.0.
        assert all(np.isfinite(v) and v == pytest.approx(1.0) for v in out["female"]["per_fold_r2"][:1])
        assert all(np.isfinite(v) and v == pytest.approx(1.0) for v in out["male"]["per_fold_r2"][1:])

    def test_small_fold_returns_nan(self) -> None:
        pred_df = pd.DataFrame(
            {
                "ROSMAP_IndividualID": ["R0", "R1"],
                "fold": [0, 0],
                "y_true": [0.0, 1.0],
                "y_composite": [0.0, 1.0],
            }
        )
        sex_df = pd.DataFrame(
            {"ROSMAP_IndividualID": ["R0", "R1"], "sex": ["female", "female"]},
        )
        out = mod.per_fold_r2_by_sex(pred_df, sex_df, n_folds=1)
        # n=2 < 3 → NaN for that fold.
        assert not np.isfinite(out["female"]["per_fold_r2"][0])

# =============================================================================
# End-to-end smoke test (synthetic)
# =============================================================================

def _write_synthetic_inputs(tmp_path: Path, n_ct: int = 4, n_genes: int = 6) -> dict:
    """Build a tiny self-consistent fixture: attributions, predictions, metadata,
    gene-name sidecar.

    Returns a dict of paths suitable for passing to ``main``.
    """
    rng = np.random.default_rng(123)
    n_per_fold = 8
    n_folds = 2
    n = n_per_fold * n_folds  # 16
    sids = np.array([f"R{i:04d}" for i in range(n)])
    folds = np.repeat(np.arange(n_folds), n_per_fold)

    # Synthetic attributions: female-dominated CT0 in gene0, male-dominated
    # CT1 in gene1, plus background noise.
    attr = rng.normal(scale=0.01, size=(n, n_ct, n_genes)).astype(np.float32)
    is_female_idx = np.arange(n) < n // 2  # first half female
    attr[is_female_idx, 0, 0] += 1.0
    attr[~is_female_idx, 1, 1] += 1.0
    sex = np.where(is_female_idx, "female", "male")

    # Composite predictions: female perfect, male noisy → high disparity.
    y_true = rng.normal(size=n).astype(np.float32)
    y_pred = y_true.copy()
    y_pred[~is_female_idx] += rng.normal(scale=0.5, size=(~is_female_idx).sum()).astype(
        np.float32
    )

    # Write per-fold val_predictions_best.npz under pred_root/foldX/.
    pred_root = tmp_path / "pred"
    for f in range(n_folds):
        d = pred_root / f"fold{f}"
        d.mkdir(parents=True, exist_ok=True)
        mask = folds == f
        np.savez(
            d / "val_predictions_best.npz",
            subject_ids=sids[mask],
            predictions=y_pred[mask],
            targets=y_true[mask],
            epoch=np.array(0),
            mse=np.array(0.0),
            mae=np.array(0.0),
            rmse=np.array(0.0),
            r2=np.array(0.0),
            pearson_r=np.array(0.0),
            spearman_rho=np.array(0.0),
        )

    # Write attribution npz (concatenated across folds).
    attr_root = tmp_path / "attr"
    attr_root.mkdir()
    attr_npz = attr_root / "composite_attributions.npz"
    np.savez(
        attr_npz,
        subject_ids=sids,
        attributions=attr,
        predictions_residual=y_pred - y_true,
        fold=folds.astype(np.int32),
    )

    # Write minimal metadata CSV.
    meta = pd.DataFrame(
        {
            "ROSMAP_IndividualID": sids,
            "msex": np.where(sex == "female", 0, 1).astype(int),
        }
    )
    meta_path = tmp_path / "metadata.csv"
    meta.to_csv(meta_path, index=False)

    # Gene names sidecar.
    pre_dir = tmp_path / "precomputed"
    pre_dir.mkdir()
    np.save(pre_dir / "gene_names.npy", np.array(
        [f"g{i}" for i in range(n_genes)], dtype=object,
    ))

    return {
        "attr_npz": attr_npz,
        "pred_root": pred_root,
        "metadata_csv": meta_path,
        "precomputed_dir": pre_dir,
        "n_ct": n_ct,
        "n_genes": n_genes,
        "n_folds": n_folds,
    }

def test_end_to_end_synthetic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run ``main()`` on a synthetic fixture; verify JSON/MD/figure exist and
    the differential-CT panel is detected (we built signal in CT0 vs CT1)."""
    fx = _write_synthetic_inputs(tmp_path)

    # Patch CELL_TYPE_ORDER so we can use a 4-CT cohort without altering
    # the global constant.
    monkeypatch.setattr(
        mod, "CELL_TYPE_ORDER",
        [f"CT{i}" for i in range(fx["n_ct"])],
    )

    out_json = tmp_path / "out.json"
    out_md = tmp_path / "out.md"
    out_fig = tmp_path / "fig"
    rc = mod.main(
        [
            "--attr-npz", str(fx["attr_npz"]),
            "--pred-root", str(fx["pred_root"]),
            "--metadata-csv", str(fx["metadata_csv"]),
            "--precomputed-dir", str(fx["precomputed_dir"]),
            "--out-json", str(out_json),
            "--out-md", str(out_md),
            "--out-fig-dir", str(out_fig),
            "--n-folds", str(fx["n_folds"]),
            "--top-n-ct", "3",
            "--top-n-genes-per-ct", "3",
            "--top-n-pairs", "5",
        ]
    )
    assert rc == 0
    assert out_json.exists()
    assert out_md.exists()
    assert (out_fig / "fig_sex_disparity.png").exists()
    assert (out_fig / "fig_sex_disparity.pdf").exists()

    payload = json.loads(out_json.read_text())
    # Cohort: 8 F, 8 M.
    assert payload["cohort"]["n_female"] == 8
    assert payload["cohort"]["n_male"] == 8
    # Female top-1 should be CT0 (we injected gene0 signal).
    assert payload["top_ct_per_sex"]["female"][0]["cell_type"] == "CT0"
    assert payload["top_ct_per_sex"]["male"][0]["cell_type"] == "CT1"
    # Headline: at least one of (b)/(c) in the explanation since we built signal.
    expl = payload["headline"]["most_plausible_explanation"]
    assert ("(b)" in expl) or ("(c)" in expl)
