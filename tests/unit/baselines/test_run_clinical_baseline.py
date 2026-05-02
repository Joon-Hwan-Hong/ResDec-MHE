"""Unit tests for clinical-only baseline script.

Covers APOE-e4 dosage parsing, predictor frame assembly, fold preparation
(z-score on train applied to val, NaN imputation), the LinReg/ElasticNet
fits, paired Wilcoxon, and the end-to-end driver on a synthetic dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.resdec_mhe.baselines.run_clinical_baseline import (
    ALL_PREDICTORS,
    CONTINUOUS_PREDICTORS,
    SUBJECT_COLUMN,
    TARGET_COLUMN,
    build_predictor_frame,
    fit_predict_elasticnet,
    fit_predict_linreg,
    load_reference_per_fold,
    paired_wilcoxon,
    parse_apoe4_dosage,
    prepare_fold,
    run_clinical_baseline,
)

# ─── parse_apoe4_dosage ─────────────────────────────────────────────────

@pytest.mark.parametrize(
    ("genotype", "expected"),
    [
        (33.0, 0),
        (22.0, 0),
        (23.0, 0),
        (32, 0),
        (34.0, 1),
        (43, 1),
        (24.0, 1),
        (42, 1),
        (44.0, 2),
        ("34", 1),
        ("44", 2),
    ],
)
def test_parse_apoe4_dosage_matches_allele_count(genotype, expected):
    assert parse_apoe4_dosage(genotype) == expected

def test_parse_apoe4_dosage_nan_passthrough():
    assert np.isnan(parse_apoe4_dosage(float("nan")))
    assert np.isnan(parse_apoe4_dosage(None))

def test_parse_apoe4_dosage_strict_rejects_invalid_range():
    """In strict mode, out-of-range genotypes raise ValueError (legacy behaviour)."""
    with pytest.raises(ValueError):
        parse_apoe4_dosage(11, strict=True)
    with pytest.raises(ValueError):
        parse_apoe4_dosage(50, strict=True)

def test_parse_apoe4_dosage_lenient_returns_nan_on_invalid_range():
    """Default (lenient) mode returns NaN on out-of-range encodings."""
    assert np.isnan(parse_apoe4_dosage(11))
    assert np.isnan(parse_apoe4_dosage(50))

def test_parse_apoe4_dosage_strict_rejects_unparseable():
    """In strict mode, unparseable genotypes raise ValueError."""
    with pytest.raises(ValueError):
        parse_apoe4_dosage("not-a-genotype", strict=True)

def test_parse_apoe4_dosage_lenient_returns_nan_on_unparseable():
    """Default (lenient) mode returns NaN on unparseable input."""
    assert np.isnan(parse_apoe4_dosage("not-a-genotype"))

def test_parse_apoe4_dosage_lenient_returns_nan_on_non_234_alleles():
    """Default (lenient) mode returns NaN on non-{2,3,4} alleles in-range.

    The range check ([22, 44]) admits values like 25 / 35 / 45 whose digits
    include non-{2,3,4} alleles (the 5). Those should return NaN, not raise.
    """
    assert np.isnan(parse_apoe4_dosage(25))
    assert np.isnan(parse_apoe4_dosage(35))

# ─── build_predictor_frame ──────────────────────────────────────────────

def _toy_metadata(n: int = 12, *, seed: int = 0) -> pd.DataFrame:
    """Synthetic ROSMAP-shaped metadata with the columns the script needs."""
    rng = np.random.default_rng(seed)
    apoe_options = [33, 34, 44, 23, 24, 22]
    rows = []
    for i in range(n):
        rows.append(
            {
                SUBJECT_COLUMN: f"R{i:06d}",
                TARGET_COLUMN: float(rng.normal(loc=-0.5, scale=1.2)),
                "apoe_genotype": float(apoe_options[i % len(apoe_options)]),
                "age_death": float(rng.uniform(70, 95)),
                "educ": int(rng.integers(8, 22)),
                "msex": int(rng.integers(0, 2)),
                "braaksc": float(rng.integers(0, 7)),
            }
        )
    df = pd.DataFrame(rows)
    # Force a controlled NaN for apoe_genotype on one subject.
    df.loc[3, "apoe_genotype"] = np.nan
    return df

def test_build_predictor_frame_columns():
    df = _toy_metadata()
    out = build_predictor_frame(df)
    expected = [SUBJECT_COLUMN, TARGET_COLUMN, *ALL_PREDICTORS]
    assert list(out.columns) == expected
    # APOE-4 dosage parsed from genotype digits, 1 NaN preserved.
    assert int(out["apoe4_dosage"].isna().sum()) == 1

def test_build_predictor_frame_missing_column_raises():
    df = _toy_metadata().drop(columns=["braaksc"])
    with pytest.raises(KeyError):
        build_predictor_frame(df)

# ─── prepare_fold ───────────────────────────────────────────────────────

def test_prepare_fold_zscores_continuous_on_train_only():
    df = _toy_metadata(n=20)
    predictors = build_predictor_frame(df)
    train_ids = [f"R{i:06d}" for i in range(0, 14)]
    val_ids = [f"R{i:06d}" for i in range(14, 20)]
    fold = prepare_fold(predictors, train_ids, val_ids, fold_idx=0)

    n_cont = len(CONTINUOUS_PREDICTORS)
    train_cont = fold.X_train[:, :n_cont]
    # z-score on train -> mean ~0, std ~1 across all continuous predictors
    assert np.allclose(train_cont.mean(axis=0), 0.0, atol=1e-8)
    assert np.allclose(train_cont.std(axis=0), 1.0, atol=1e-6)

    # passthrough columns unchanged (modulo train-mean imputation for one
    # NaN apoe4_dosage in the toy data)
    train_pass = fold.X_train[:, n_cont:]
    assert set(np.unique(train_pass[:, 0])).issubset({0.0, 1.0})  # msex
    apoe_vals = train_pass[:, 1]
    assert apoe_vals.min() >= 0.0
    assert apoe_vals.max() <= 2.0
    # Non-imputed dosages are integer in {0, 1, 2}; at most one fractional
    # value (the imputed mean) is allowed.
    fractional = np.abs(apoe_vals - np.round(apoe_vals)) > 1e-9
    assert int(fractional.sum()) <= 1

def test_prepare_fold_imputes_apoe4_to_train_mean():
    """Setting subject 3 (in train) NaN -> imputed with train mean apoe4_dosage."""
    df = _toy_metadata(n=20)
    predictors = build_predictor_frame(df)
    train_ids = [f"R{i:06d}" for i in range(0, 14)]
    val_ids = [f"R{i:06d}" for i in range(14, 20)]
    fold = prepare_fold(predictors, train_ids, val_ids, fold_idx=0)
    # subject 3 (index in train arrays) was the NaN in toy data; index 3 in
    # the train frame corresponds to subject "R000003".
    apoe_col_idx = len(CONTINUOUS_PREDICTORS) + 1  # last column
    apoe_train = fold.X_train[:, apoe_col_idx]
    # No NaNs after impute.
    assert not np.isnan(apoe_train).any()
    assert not np.isnan(fold.X_val[:, apoe_col_idx]).any()

# ─── estimators ──────────────────────────────────────────────────────────

def _toy_fold(n_train: int = 60, n_val: int = 20, *, seed: int = 7):
    """Build a fold where y is a noisy linear combination of predictors."""
    df = _toy_metadata(n=n_train + n_val, seed=seed)
    predictors = build_predictor_frame(df)
    # Inject genuine signal: y = 0.5 * z(braaksc) - 0.4 * apoe4 + noise
    rng = np.random.default_rng(seed)
    bz = (df["braaksc"] - df["braaksc"].mean()) / df["braaksc"].std()
    apoe = predictors["apoe4_dosage"].fillna(predictors["apoe4_dosage"].mean())
    df[TARGET_COLUMN] = 0.5 * bz.to_numpy() - 0.4 * apoe.to_numpy() + rng.normal(0, 0.3, len(df))
    predictors = build_predictor_frame(df)
    train_ids = [f"R{i:06d}" for i in range(n_train)]
    val_ids = [f"R{i:06d}" for i in range(n_train, n_train + n_val)]
    return prepare_fold(predictors, train_ids, val_ids, fold_idx=0)

def test_fit_predict_linreg_returns_valid_metrics():
    fold = _toy_fold()
    metrics = fit_predict_linreg(fold)
    assert set(metrics).issuperset({"r2", "mae", "rmse", "pearson_r", "spearman_rho", "coef"})
    assert -1.0 <= metrics["r2"] <= 1.0
    assert set(metrics["coef"].keys()) == set(fold.feature_names)

def test_fit_predict_elasticnet_returns_valid_metrics():
    fold = _toy_fold()
    metrics = fit_predict_elasticnet(fold)
    assert "alpha" in metrics and "l1_ratio" in metrics
    assert metrics["alpha"] > 0
    assert metrics["l1_ratio"] in (0.1, 0.3, 0.5, 0.7, 0.9)

# ─── paired Wilcoxon ─────────────────────────────────────────────────────

def test_paired_wilcoxon_detects_difference():
    ours = [0.10, 0.12, 0.08, 0.14, 0.09]
    theirs = [0.40, 0.42, 0.38, 0.44, 0.39]
    res = paired_wilcoxon(ours, theirs)
    # mean diff ours - theirs is strongly negative -> p ≈ 0.03 for n=5
    assert res["mean_diff"] < 0
    assert res["p_value"] < 0.1
    assert res["n"] == 5

def test_paired_wilcoxon_zero_diff_returns_p_one():
    res = paired_wilcoxon([0.3, 0.3], [0.3, 0.3])
    assert res["p_value"] == 1.0

# ─── load_reference_per_fold ─────────────────────────────────────────────

def test_load_reference_per_fold_parses_canonical_shape(tmp_path: Path):
    payload = {
        "per_fold": [
            {"fold": 0, "ours": {"r2": 0.4}, "tab_ge": {"r2": 0.3}, "tab_en": {"r2": 0.32}},
            {"fold": 1, "ours": {"r2": 0.5}, "tab_ge": {"r2": 0.42}, "tab_en": {"r2": 0.41}},
        ]
    }
    p = tmp_path / "ref.json"
    p.write_text(json.dumps(payload))
    out = load_reference_per_fold(p)
    assert out["resdec_mhe"] == [0.4, 0.5]
    assert out["tabpfn"] == [0.3, 0.42]

def test_load_reference_missing_returns_empty(tmp_path: Path):
    out = load_reference_per_fold(tmp_path / "does_not_exist.json")
    assert out == {}

# ─── end-to-end driver ──────────────────────────────────────────────────

def test_run_clinical_baseline_end_to_end(tmp_path: Path):
    """Synthetic-data smoke test: writes JSON + MD, contains expected keys."""
    n = 200
    df = _toy_metadata(n=n, seed=2026)
    # Inject a strong signal so the regressors do better than chance.
    rng = np.random.default_rng(2026)
    bz = (df["braaksc"] - df["braaksc"].mean()) / df["braaksc"].std()
    apoe_dosage = df["apoe_genotype"].apply(
        lambda g: float("nan") if pd.isna(g)
        else (int(int(g) // 10 == 4) + int(int(g) % 10 == 4))
    ).fillna(0)
    df[TARGET_COLUMN] = (
        0.7 * bz.to_numpy() - 0.5 * apoe_dosage.to_numpy() + rng.normal(0, 0.4, n)
    )

    metadata_path = tmp_path / "metadata.csv"
    df.to_csv(metadata_path, index=False)

    # 5 folds over 200 subjects (40 val each).
    fold_size = n // 5
    folds = []
    for k in range(5):
        val = [f"R{i:06d}" for i in range(k * fold_size, (k + 1) * fold_size)]
        train = [sid for sid in (f"R{i:06d}" for i in range(n)) if sid not in set(val)]
        folds.append({"train": train, "val": val})
    splits_path = tmp_path / "splits.json"
    splits_path.write_text(
        json.dumps(
            {
                "holdout_test": [],
                "train_val_pool": [f"R{i:06d}" for i in range(n)],
                "folds": folds,
            }
        )
    )

    # Reference JSON: pretend our snRNA model crushes clinical
    ref_path = tmp_path / "ref.json"
    ref_path.write_text(
        json.dumps(
            {
                "per_fold": [
                    {
                        "fold": k,
                        "ours": {"r2": 0.55},
                        "tab_ge": {"r2": 0.50},
                        "tab_en": {"r2": 0.49},
                    }
                    for k in range(5)
                ]
            }
        )
    )

    out_dir = tmp_path / "out"
    summary = run_clinical_baseline(
        metadata_path=metadata_path,
        splits_path=splits_path,
        output_dir=out_dir,
        reference_path=ref_path,
        random_state=42,
    )

    json_path = out_dir / "clinical_baseline_summary.json"
    md_path = out_dir / "clinical_baseline_summary.md"
    assert json_path.exists() and md_path.exists()

    payload = json.loads(json_path.read_text())
    assert payload["n_pool"] == n
    assert len(payload["per_fold"]) == 5
    # Per-fold structure
    for rec in payload["per_fold"]:
        assert {"fold", "n_train", "n_val", "linreg", "elasticnet"}.issubset(rec.keys())
        assert -1.0 <= rec["linreg"]["r2"] <= 1.0
        assert -1.0 <= rec["elasticnet"]["r2"] <= 1.0
    # Wilcoxon comparisons present
    pw = payload["paired_wilcoxon"]
    assert "linreg_vs_resdec_mhe" in pw
    assert "elasticnet_vs_tabpfn" in pw
    # Summary block
    s = payload["summary"]
    assert "linreg" in s and "elasticnet" in s
    assert "r2_mean" in s["linreg"]
    # Markdown sanity
    md = md_path.read_text()
    assert "Clinical-only Baseline" in md
    assert "Paired Wilcoxon" in md

def test_summary_predictor_list_is_complete():
    """The summary predictors block must include all five required predictors."""
    expected = {"age_death", "educ", "msex", "braaksc", "apoe4_dosage"}
    assert set(ALL_PREDICTORS) == expected
