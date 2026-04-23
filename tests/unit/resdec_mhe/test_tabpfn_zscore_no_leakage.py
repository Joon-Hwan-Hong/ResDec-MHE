"""Verify that --zscore uses TRAIN-ONLY stats, not pooled, for TabPFN input.

Contract: when ``--zscore`` is set, the per-feature z-score stats MUST be fit
on train-fold X only (inner-fold train for OOF; outer-fold train for outer)
and then applied to both train and val. Pooling stats over train+val would
leak label-correlated information at val-time.

Tests:
  1. Direct helper correctness: val-transformed mean diverges from zero when
     train/val have different true means (proof: stats are train-only).
  2. Helper does not mutate the input arrays.
  3. Backward compatibility: when --zscore is absent, no scaler is applied
     (verified via a mocked main() — the X fed to regressor.fit must equal
     the raw X, not the z-scored X).
  4. When --zscore is set, the X fed to regressor.fit is scaled such that its
     mean is close to zero and std close to one (train-fit sanity).
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helper-level tests: verify the shared apply_zscore_train_only is leakage-free.
# ---------------------------------------------------------------------------
def test_apply_zscore_train_only_stats():
    """Helper: val-transformed mean reflects TRAIN stats, not val stats."""
    from src.analysis.tabpfn_preprocessing import apply_zscore_train_only

    rng = np.random.default_rng(0)
    # Train ~ N(0, 1). Val ~ N(10, 1) — 10σ offset from train mean.
    X_train = rng.standard_normal((100, 3)).astype(np.float32)
    X_val = (rng.standard_normal((20, 3)) + 10.0).astype(np.float32)

    X_train_s, X_val_s = apply_zscore_train_only(X_train, X_val)

    # Train after transform: ~mean=0, ~std=1 (sample-size dependent).
    assert abs(X_train_s.mean()) < 0.2, (
        f"train-transformed mean {X_train_s.mean():.3f} too large; "
        "scaler did not fit on train"
    )
    assert abs(X_train_s.std() - 1.0) < 0.2, (
        f"train-transformed std {X_train_s.std():.3f} too far from 1"
    )
    # Val after transform: mean should be ~10 (val's true mean is 10σ above
    # train mean, so train-stat-transformed val mean stays ~10). If stats
    # were POOLED, val mean after transform would be much closer to zero.
    assert X_val_s.mean() > 5.0, (
        f"val-transformed mean {X_val_s.mean():.3f} is too close to 0 — "
        "likely pooled stats, not train-only"
    )


def test_apply_zscore_train_only_reproducibility_second_rng():
    """Second RNG seed sanity check on train-only stats (same contract)."""
    from src.analysis.tabpfn_preprocessing import apply_zscore_train_only

    rng = np.random.default_rng(1)
    X_train = rng.standard_normal((100, 3)).astype(np.float32)
    X_val = (rng.standard_normal((20, 3)) + 10.0).astype(np.float32)

    X_train_s, X_val_s = apply_zscore_train_only(X_train, X_val)

    assert abs(X_train_s.mean()) < 0.2
    assert abs(X_train_s.std() - 1.0) < 0.2
    assert X_val_s.mean() > 5.0, (
        f"val-transformed mean {X_val_s.mean():.3f} is too close to 0 — "
        "likely pooled stats, not train-only"
    )


def test_apply_zscore_preserves_input_shape_and_dtype():
    """The helper must not change shape; dtype must stay float32."""
    from src.analysis.tabpfn_preprocessing import apply_zscore_train_only

    rng = np.random.default_rng(2)
    X_train = rng.standard_normal((50, 7)).astype(np.float32)
    X_val = rng.standard_normal((10, 7)).astype(np.float32)

    X_train_s, X_val_s = apply_zscore_train_only(X_train, X_val)
    assert X_train_s.shape == X_train.shape
    assert X_val_s.shape == X_val.shape
    assert X_train_s.dtype == np.float32
    assert X_val_s.dtype == np.float32


def test_apply_zscore_does_not_mutate_inputs():
    """Inputs must not be modified in place; outputs are fresh arrays."""
    from src.analysis.tabpfn_preprocessing import apply_zscore_train_only

    rng = np.random.default_rng(4)
    X_train = (rng.standard_normal((30, 5)) + 7.0).astype(np.float32)
    X_val = (rng.standard_normal((10, 5)) + 7.0).astype(np.float32)
    X_train_orig = X_train.copy()
    X_val_orig = X_val.copy()

    _ = apply_zscore_train_only(X_train, X_val)

    assert np.array_equal(X_train, X_train_orig), "X_train was mutated"
    assert np.array_equal(X_val, X_val_orig), "X_val was mutated"


def test_apply_zscore_handles_zero_variance_feature():
    """sklearn StandardScaler must NOT divide-by-zero on constant features."""
    from src.analysis.tabpfn_preprocessing import apply_zscore_train_only

    rng = np.random.default_rng(3)
    X_train = rng.standard_normal((50, 4)).astype(np.float32)
    X_train[:, 1] = 5.0  # constant column
    X_val = rng.standard_normal((10, 4)).astype(np.float32)
    X_val[:, 1] = 5.0

    X_train_s, X_val_s = apply_zscore_train_only(X_train, X_val)
    # Constant column should become all-zeros (mean-centered, std=1 fallback).
    assert np.all(np.isfinite(X_train_s))
    assert np.all(np.isfinite(X_val_s))
    assert np.allclose(X_train_s[:, 1], 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# CLI-default backward compatibility: when --zscore is omitted, default=False.
# ---------------------------------------------------------------------------
def _parse_oof_cli(argv: list[str]):
    """Mirror the OOF script argparse block for default-value testing."""
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--splits-path", default="outputs/splits.json")
    p.add_argument("--precomputed-dir", default="data/precomputed")
    p.add_argument("--metadata-csv", default="data/metadata_ROSMAP/metadata.csv")
    p.add_argument("--top-k-dir", default="data/redesign")
    p.add_argument("--output-dir", default="data/redesign")
    p.add_argument("--top-k", type=int, default=2000)
    p.add_argument("--n-inner-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ignore-pretraining-limits", action="store_true", default=False)
    p.add_argument("--zscore", action="store_true", default=False)
    return p.parse_args(argv)


def test_zscore_cli_defaults_to_false_when_omitted():
    args = _parse_oof_cli(argv=[])
    assert args.zscore is False


def test_zscore_cli_enabled_when_passed():
    args = _parse_oof_cli(argv=["--zscore"])
    assert args.zscore is True


# ---------------------------------------------------------------------------
# End-to-end: when --zscore is absent, behavior must be identical to before
# (X fed to regressor.fit == raw X, not z-scored).
# When --zscore is set, X fed to regressor.fit is z-scored with train stats.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("zscore_on", [True, False])
def test_main_oof_zscore_behavior(monkeypatch, tmp_path, zscore_on):
    """Verify X passed to regressor.fit is z-scored iff --zscore is set."""
    from scripts.resdec_mhe.tabpfn import compute_oof as compute_tabpfn_oof

    n_train = 20
    n_feats_total = 50
    top_k = 5
    subject_ids = [f"s{i:03d}" for i in range(n_train)]
    rng = np.random.default_rng(0)
    # Shift features so the mean is far from zero — that way we can detect
    # whether z-score was applied: zscored features will have ~mean=0.
    features = {
        s: (rng.standard_normal(n_feats_total) + 100.0).astype(np.float32)
        for s in subject_ids
    }
    targets = {s: float(rng.standard_normal()) for s in subject_ids}

    monkeypatch.setattr(
        compute_tabpfn_oof, "load_flat_features", lambda d, ids: features
    )
    monkeypatch.setattr(
        compute_tabpfn_oof, "load_targets", lambda csv, ids: targets
    )
    monkeypatch.setattr(
        compute_tabpfn_oof, "load_splits",
        lambda p: {"folds": [{"train": subject_ids, "val": []}]},
    )

    # Capture X passed to fit()
    captured_fit_X: list[np.ndarray] = []
    mock_cls = MagicMock()
    mock_instance = MagicMock()

    def fit_capture(X, y):
        captured_fit_X.append(np.asarray(X).copy())

    def predict_full(X, output_type=None, quantiles=None):
        n = len(X)
        return {
            "median": np.zeros(n, dtype=np.float32),
            "quantiles": [
                np.full(n, -0.1, dtype=np.float32),
                np.full(n, 0.1, dtype=np.float32),
            ],
        }

    mock_instance.fit.side_effect = fit_capture
    mock_instance.predict.side_effect = predict_full
    mock_cls.return_value = mock_instance
    monkeypatch.setattr(compute_tabpfn_oof, "TabPFNRegressor", mock_cls)

    # Provide top-k JSON
    top_k_dir = tmp_path / "topk"
    top_k_dir.mkdir()
    (top_k_dir / f"top_{top_k}_features_fold0.json").write_text(
        json.dumps({"indices": list(range(top_k))})
    )
    output_dir = tmp_path / "out"

    class _Args:
        pass

    a = _Args()
    a.splits_path = "dummy.json"
    a.precomputed_dir = "dummy"
    a.metadata_csv = "dummy.csv"
    a.top_k_dir = str(top_k_dir)
    a.output_dir = str(output_dir)
    a.top_k = top_k
    a.n_inner_folds = 2
    a.seed = 42
    a.ignore_pretraining_limits = False
    a.zscore = zscore_on

    monkeypatch.setattr(
        compute_tabpfn_oof.torch.cuda, "is_available", lambda: False
    )

    compute_tabpfn_oof.main(a)

    assert len(captured_fit_X) == 2, (
        f"expected 2 fit() calls (n_inner_folds=2), got {len(captured_fit_X)}"
    )
    # Raw features are centered at 100. Z-scored features would be ~mean 0.
    for X_fit in captured_fit_X:
        if zscore_on:
            # Train stats subtract their own mean -> ~0.
            assert abs(X_fit.mean()) < 1.0, (
                f"zscore was requested but fit() saw mean={X_fit.mean():.3f} "
                "(far from 0) — z-score not applied"
            )
        else:
            # Raw shifted features — mean should still be ~100.
            assert X_fit.mean() > 50.0, (
                f"zscore was NOT requested but fit() saw mean={X_fit.mean():.3f} "
                "(far from original 100) — something unexpectedly transformed X"
            )


@pytest.mark.parametrize("zscore_on", [True, False])
def test_main_outer_zscore_behavior(monkeypatch, tmp_path, zscore_on):
    """Verify X passed to outer regressor.fit is z-scored iff --zscore is set."""
    from scripts.resdec_mhe.tabpfn import compute_outer as compute_tabpfn_outer

    n_train = 16
    n_val = 4
    n_feats_total = 50
    top_k = 5
    train_ids = [f"t{i:03d}" for i in range(n_train)]
    val_ids = [f"v{i:03d}" for i in range(n_val)]
    rng = np.random.default_rng(1)
    # Train features drawn from mean=100, val features drawn from mean=110:
    # a 10-unit offset that lets us directly distinguish train-only vs pooled
    # stats in the X_pred leakage-direction assertion below.
    features = {
        s: (rng.standard_normal(n_feats_total) + 100.0).astype(np.float32)
        for s in train_ids
    }
    features.update({
        s: (rng.standard_normal(n_feats_total) + 110.0).astype(np.float32)
        for s in val_ids
    })
    targets = {
        s: float(rng.standard_normal()) for s in train_ids + val_ids
    }

    monkeypatch.setattr(
        compute_tabpfn_outer, "load_flat_features", lambda d, ids: features
    )
    monkeypatch.setattr(
        compute_tabpfn_outer, "load_targets", lambda csv, ids: targets
    )
    monkeypatch.setattr(
        compute_tabpfn_outer, "load_splits",
        lambda p: {"folds": [{"train": train_ids, "val": val_ids}]},
    )

    captured_fit_X: list[np.ndarray] = []
    captured_predict_X: list[np.ndarray] = []
    mock_cls = MagicMock()
    mock_instance = MagicMock()

    def fit_capture(X, y):
        captured_fit_X.append(np.asarray(X).copy())

    def predict_full(X, output_type=None, quantiles=None):
        captured_predict_X.append(np.asarray(X).copy())
        n = len(X)
        return {
            "median": np.zeros(n, dtype=np.float32),
            "quantiles": [
                np.full(n, -0.1, dtype=np.float32),
                np.full(n, 0.1, dtype=np.float32),
            ],
        }

    mock_instance.fit.side_effect = fit_capture
    mock_instance.predict.side_effect = predict_full
    mock_cls.return_value = mock_instance
    monkeypatch.setattr(compute_tabpfn_outer, "TabPFNRegressor", mock_cls)

    top_k_dir = tmp_path / "topk"
    top_k_dir.mkdir()
    (top_k_dir / f"top_{top_k}_features_fold0.json").write_text(
        json.dumps({"indices": list(range(top_k))})
    )
    output_dir = tmp_path / "out"

    class _Args:
        pass

    a = _Args()
    a.splits_path = "dummy.json"
    a.precomputed_dir = "dummy"
    a.metadata_csv = "dummy.csv"
    a.top_k_dir = str(top_k_dir)
    a.output_dir = str(output_dir)
    a.top_k = top_k
    a.feature_set = "A"
    a.seed = 42
    a.ignore_pretraining_limits = False
    a.zscore = zscore_on

    monkeypatch.setattr(
        compute_tabpfn_outer.torch.cuda, "is_available", lambda: False
    )

    compute_tabpfn_outer.main(a)

    assert len(captured_fit_X) == 1
    X_fit = captured_fit_X[0]
    X_pred = captured_predict_X[0]

    if zscore_on:
        assert abs(X_fit.mean()) < 1.0, (
            f"zscore on but fit() saw mean={X_fit.mean():.3f}"
        )
        # Direction-of-leakage test: train features have mean=100, val has
        # mean=110. Under TRAIN-ONLY stats, scaler subtracts ~100 and divides
        # by ~1; val mean shifts to ~+10. Under POOLED stats (leakage), both
        # train and val would be centered at ~0 (pooled mean ≈ 102) and
        # X_pred.mean() would collapse to near-0. Requiring >3.0 catches a
        # pooled-stats regression that would otherwise slip through.
        assert X_pred.mean() > 3.0, (
            f"X_pred.mean()={X_pred.mean():.3f} — too close to 0, suggests "
            "pooled stats were used instead of train-only (val mean should "
            "remain offset since val is drawn from mean=110, train from mean=100)"
        )
    else:
        assert X_fit.mean() > 50.0, (
            f"zscore off but fit() saw mean={X_fit.mean():.3f} (expected ~100)"
        )
        assert X_pred.mean() > 50.0, (
            f"zscore off but predict() saw mean={X_pred.mean():.3f}"
        )


# ---------------------------------------------------------------------------
# Combinability: --zscore and --ignore-pretraining-limits are orthogonal.
# Each must reach its own sink (scaler vs TabPFNRegressor kwarg) when set,
# regardless of the other's value. Skip the (False, False) case because it is
# already exercised by the zscore_on=False branch of the backward-compat tests.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "zscore_on,ignore_on",
    [(True, True), (True, False), (False, True)],
)
def test_oof_flags_coexist(monkeypatch, tmp_path, zscore_on, ignore_on):
    """--zscore and --ignore-pretraining-limits are orthogonal in OOF main()."""
    from scripts.resdec_mhe.tabpfn import compute_oof as compute_tabpfn_oof

    n_train = 20
    n_feats_total = 50
    top_k = 5
    subject_ids = [f"s{i:03d}" for i in range(n_train)]
    rng = np.random.default_rng(0)
    features = {
        s: (rng.standard_normal(n_feats_total) + 100.0).astype(np.float32)
        for s in subject_ids
    }
    targets = {s: float(rng.standard_normal()) for s in subject_ids}

    monkeypatch.setattr(
        compute_tabpfn_oof, "load_flat_features", lambda d, ids: features
    )
    monkeypatch.setattr(
        compute_tabpfn_oof, "load_targets", lambda csv, ids: targets
    )
    monkeypatch.setattr(
        compute_tabpfn_oof, "load_splits",
        lambda p: {"folds": [{"train": subject_ids, "val": []}]},
    )

    captured_fit_X: list[np.ndarray] = []
    mock_cls = MagicMock()
    mock_instance = MagicMock()

    def fit_capture(X, y):
        captured_fit_X.append(np.asarray(X).copy())

    def predict_full(X, output_type=None, quantiles=None):
        n = len(X)
        return {
            "median": np.zeros(n, dtype=np.float32),
            "quantiles": [
                np.full(n, -0.1, dtype=np.float32),
                np.full(n, 0.1, dtype=np.float32),
            ],
        }

    mock_instance.fit.side_effect = fit_capture
    mock_instance.predict.side_effect = predict_full
    mock_cls.return_value = mock_instance
    monkeypatch.setattr(compute_tabpfn_oof, "TabPFNRegressor", mock_cls)

    top_k_dir = tmp_path / "topk"
    top_k_dir.mkdir()
    (top_k_dir / f"top_{top_k}_features_fold0.json").write_text(
        json.dumps({"indices": list(range(top_k))})
    )
    output_dir = tmp_path / "out"

    class _Args:
        pass

    a = _Args()
    a.splits_path = "dummy.json"
    a.precomputed_dir = "dummy"
    a.metadata_csv = "dummy.csv"
    a.top_k_dir = str(top_k_dir)
    a.output_dir = str(output_dir)
    a.top_k = top_k
    a.n_inner_folds = 2
    a.seed = 42
    a.ignore_pretraining_limits = ignore_on
    a.zscore = zscore_on

    monkeypatch.setattr(
        compute_tabpfn_oof.torch.cuda, "is_available", lambda: False
    )

    compute_tabpfn_oof.main(a)

    # zscore sink: check X passed to fit() was scaled iff zscore_on
    assert len(captured_fit_X) == 2
    for X_fit in captured_fit_X:
        if zscore_on:
            assert abs(X_fit.mean()) < 1.0, (
                f"zscore_on={zscore_on}, ignore_on={ignore_on}: fit() saw "
                f"mean={X_fit.mean():.3f} (expected ~0 after z-score)"
            )
        else:
            assert X_fit.mean() > 50.0, (
                f"zscore_on={zscore_on}, ignore_on={ignore_on}: fit() saw "
                f"mean={X_fit.mean():.3f} (expected ~100, raw features)"
            )

    # ignore sink: every TabPFNRegressor ctor must see the flag value
    assert mock_cls.call_count == 2
    for call in mock_cls.call_args_list:
        assert call.kwargs["ignore_pretraining_limits"] is ignore_on, (
            f"zscore_on={zscore_on}, ignore_on={ignore_on}: ctor kwargs="
            f"{call.kwargs}"
        )


@pytest.mark.parametrize(
    "zscore_on,ignore_on",
    [(True, True), (True, False), (False, True)],
)
def test_outer_flags_coexist(monkeypatch, tmp_path, zscore_on, ignore_on):
    """--zscore and --ignore-pretraining-limits are orthogonal in outer main()."""
    from scripts.resdec_mhe.tabpfn import compute_outer as compute_tabpfn_outer

    n_train = 16
    n_val = 4
    n_feats_total = 50
    top_k = 5
    train_ids = [f"t{i:03d}" for i in range(n_train)]
    val_ids = [f"v{i:03d}" for i in range(n_val)]
    all_ids = train_ids + val_ids
    rng = np.random.default_rng(1)
    features = {
        s: (rng.standard_normal(n_feats_total) + 100.0).astype(np.float32)
        for s in all_ids
    }
    targets = {s: float(rng.standard_normal()) for s in all_ids}

    monkeypatch.setattr(
        compute_tabpfn_outer, "load_flat_features", lambda d, ids: features
    )
    monkeypatch.setattr(
        compute_tabpfn_outer, "load_targets", lambda csv, ids: targets
    )
    monkeypatch.setattr(
        compute_tabpfn_outer, "load_splits",
        lambda p: {"folds": [{"train": train_ids, "val": val_ids}]},
    )

    captured_fit_X: list[np.ndarray] = []
    mock_cls = MagicMock()
    mock_instance = MagicMock()

    def fit_capture(X, y):
        captured_fit_X.append(np.asarray(X).copy())

    def predict_full(X, output_type=None, quantiles=None):
        n = len(X)
        return {
            "median": np.zeros(n, dtype=np.float32),
            "quantiles": [
                np.full(n, -0.1, dtype=np.float32),
                np.full(n, 0.1, dtype=np.float32),
            ],
        }

    mock_instance.fit.side_effect = fit_capture
    mock_instance.predict.side_effect = predict_full
    mock_cls.return_value = mock_instance
    monkeypatch.setattr(compute_tabpfn_outer, "TabPFNRegressor", mock_cls)

    top_k_dir = tmp_path / "topk"
    top_k_dir.mkdir()
    (top_k_dir / f"top_{top_k}_features_fold0.json").write_text(
        json.dumps({"indices": list(range(top_k))})
    )
    output_dir = tmp_path / "out"

    class _Args:
        pass

    a = _Args()
    a.splits_path = "dummy.json"
    a.precomputed_dir = "dummy"
    a.metadata_csv = "dummy.csv"
    a.top_k_dir = str(top_k_dir)
    a.output_dir = str(output_dir)
    a.top_k = top_k
    a.feature_set = "A"
    a.seed = 42
    a.ignore_pretraining_limits = ignore_on
    a.zscore = zscore_on

    monkeypatch.setattr(
        compute_tabpfn_outer.torch.cuda, "is_available", lambda: False
    )

    compute_tabpfn_outer.main(a)

    assert len(captured_fit_X) == 1
    X_fit = captured_fit_X[0]
    if zscore_on:
        assert abs(X_fit.mean()) < 1.0, (
            f"zscore_on={zscore_on}, ignore_on={ignore_on}: fit() saw "
            f"mean={X_fit.mean():.3f} (expected ~0 after z-score)"
        )
    else:
        assert X_fit.mean() > 50.0, (
            f"zscore_on={zscore_on}, ignore_on={ignore_on}: fit() saw "
            f"mean={X_fit.mean():.3f} (expected ~100, raw features)"
        )

    assert mock_cls.call_count == 1
    for call in mock_cls.call_args_list:
        assert call.kwargs["ignore_pretraining_limits"] is ignore_on, (
            f"zscore_on={zscore_on}, ignore_on={ignore_on}: ctor kwargs="
            f"{call.kwargs}"
        )
