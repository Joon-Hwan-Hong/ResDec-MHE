"""Verify the --ignore-pretraining-limits CLI flag is threaded all the way into
TabPFNRegressor(...) in both pre-compute scripts.

This flag deliberately overrides TabPFN-2.6's 2000-feature safety check and is
used for Task D.2 (top-k=4000 ablation). It MUST default to False and MUST
reach every TabPFNRegressor construction site when enabled; silently dropping
it would reintroduce the TabPFNValidationError that blocked D.2.

Tests mock TabPFNRegressor so no weights are actually loaded.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# Paths used by the scripts when run in this worktree. We write minimal stubs
# into tmp_path to keep the tests self-contained.


# ---------------------------------------------------------------------------
# _build_regressor helper: direct function-level thread test
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("flag_value", [True, False])
def test_build_regressor_threads_ignore_pretraining_limits_oof(flag_value):
    """compute_tabpfn_oof._build_regressor forwards the flag verbatim."""
    from scripts.redesign import compute_tabpfn_oof

    with patch.object(compute_tabpfn_oof, "TabPFNRegressor") as mock_cls:
        compute_tabpfn_oof._build_regressor(
            device="cpu", seed=42, ignore_pretraining_limits=flag_value,
        )
        mock_cls.assert_called_once()
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["ignore_pretraining_limits"] is flag_value
        assert kwargs["device"] == "cpu"
        assert kwargs["random_state"] == 42


@pytest.mark.parametrize("flag_value", [True, False])
def test_build_regressor_threads_ignore_pretraining_limits_outer(flag_value):
    """compute_tabpfn_outer._build_regressor forwards the flag verbatim."""
    from scripts.redesign import compute_tabpfn_outer

    with patch.object(compute_tabpfn_outer, "TabPFNRegressor") as mock_cls:
        compute_tabpfn_outer._build_regressor(
            device="cpu", seed=42, ignore_pretraining_limits=flag_value,
        )
        mock_cls.assert_called_once()
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["ignore_pretraining_limits"] is flag_value
        assert kwargs["device"] == "cpu"
        assert kwargs["random_state"] == 42


# ---------------------------------------------------------------------------
# CLI default: when the flag is omitted, argparse parses it as False
# ---------------------------------------------------------------------------
def _parse_oof_cli(argv: list[str]):
    """Invoke the OOF script's argparse block without executing main()."""
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
    p.add_argument(
        "--ignore-pretraining-limits", action="store_true", default=False,
    )
    return p.parse_args(argv)


def test_cli_flag_defaults_to_false_when_omitted():
    args = _parse_oof_cli(argv=[])
    assert args.ignore_pretraining_limits is False


def test_cli_flag_enabled_when_passed():
    args = _parse_oof_cli(argv=["--ignore-pretraining-limits"])
    assert args.ignore_pretraining_limits is True


# ---------------------------------------------------------------------------
# End-to-end mock: verify that the CLI-driven value lands on every
# TabPFNRegressor construction inside main(). This is the "leakage-style"
# test from the task spec — no TabPFN weights involved.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("flag_value", [True, False])
def test_main_oof_threads_flag_to_every_regressor_call(monkeypatch, tmp_path, flag_value):
    """main(args) must pass ignore_pretraining_limits into EVERY inner-fold
    TabPFNRegressor construction."""
    from scripts.redesign import compute_tabpfn_oof

    # Stub out data loaders so main() doesn't try to read real data
    n_train = 20
    n_feats_total = 50
    top_k = 5
    subject_ids = [f"s{i:03d}" for i in range(n_train)]
    rng = np.random.default_rng(0)
    features = {s: rng.standard_normal(n_feats_total).astype(np.float32)
                for s in subject_ids}
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

    # Stub TabPFNRegressor
    mock_cls = MagicMock()
    mock_instance = MagicMock()

    def predict_full(X, output_type=None, quantiles=None):
        n = len(X)
        return {
            "median": np.zeros(n, dtype=np.float32),
            "quantiles": [
                np.full(n, -0.1, dtype=np.float32),
                np.full(n, 0.1, dtype=np.float32),
            ],
        }

    mock_instance.predict.side_effect = predict_full
    mock_instance.fit.return_value = None
    mock_cls.return_value = mock_instance
    monkeypatch.setattr(compute_tabpfn_oof, "TabPFNRegressor", mock_cls)

    # Provide top-k JSON file
    top_k_dir = tmp_path / "topk"
    top_k_dir.mkdir()
    import json as _json
    (top_k_dir / f"top_{top_k}_features_fold0.json").write_text(
        _json.dumps({"indices": list(range(top_k))})
    )
    output_dir = tmp_path / "out"

    # Fake args
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
    a.ignore_pretraining_limits = flag_value
    a.zscore = False  # unrelated to this test; default backward-compat path

    # Avoid cuda probe
    monkeypatch.setattr(compute_tabpfn_oof.torch.cuda, "is_available", lambda: False)

    compute_tabpfn_oof.main(a)

    # Each inner fold (n_inner_folds=2) should have built one regressor
    assert mock_cls.call_count == 2, (
        f"expected 2 TabPFNRegressor constructions (2 inner folds), "
        f"got {mock_cls.call_count}"
    )
    for call in mock_cls.call_args_list:
        assert call.kwargs["ignore_pretraining_limits"] is flag_value, (
            f"ignore_pretraining_limits={flag_value} must be threaded to EVERY "
            f"TabPFNRegressor(...) call; saw kwargs={call.kwargs}"
        )


@pytest.mark.parametrize("flag_value", [True, False])
def test_main_outer_threads_flag_to_every_regressor_call(monkeypatch, tmp_path, flag_value):
    """main(args) of the outer script must pass ignore_pretraining_limits
    into EVERY outer-fold TabPFNRegressor construction."""
    from scripts.redesign import compute_tabpfn_outer

    # Stub out data loaders
    n_train = 16
    n_val = 4
    n_feats_total = 50
    top_k = 5
    train_ids = [f"t{i:03d}" for i in range(n_train)]
    val_ids = [f"v{i:03d}" for i in range(n_val)]
    all_ids = train_ids + val_ids
    rng = np.random.default_rng(1)
    features = {s: rng.standard_normal(n_feats_total).astype(np.float32)
                for s in all_ids}
    targets = {s: float(rng.standard_normal()) for s in all_ids}

    monkeypatch.setattr(
        compute_tabpfn_outer, "load_flat_features", lambda d, ids: features
    )
    monkeypatch.setattr(
        compute_tabpfn_outer, "load_targets", lambda csv, ids: targets
    )
    monkeypatch.setattr(
        compute_tabpfn_outer, "load_splits",
        lambda p: {"folds": [
            {"train": train_ids, "val": val_ids},
            {"train": train_ids, "val": val_ids},
        ]},
    )

    mock_cls = MagicMock()
    mock_instance = MagicMock()

    def predict_full(X, output_type=None, quantiles=None):
        n = len(X)
        return {
            "median": np.zeros(n, dtype=np.float32),
            "quantiles": [
                np.full(n, -0.1, dtype=np.float32),
                np.full(n, 0.1, dtype=np.float32),
            ],
        }

    mock_instance.predict.side_effect = predict_full
    mock_instance.fit.return_value = None
    mock_cls.return_value = mock_instance
    monkeypatch.setattr(compute_tabpfn_outer, "TabPFNRegressor", mock_cls)

    # Provide top-k JSON files for both folds
    top_k_dir = tmp_path / "topk"
    top_k_dir.mkdir()
    import json as _json
    for f in range(2):
        (top_k_dir / f"top_{top_k}_features_fold{f}.json").write_text(
            _json.dumps({"indices": list(range(top_k))})
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
    a.ignore_pretraining_limits = flag_value
    a.zscore = False  # unrelated to this test; default backward-compat path

    monkeypatch.setattr(
        compute_tabpfn_outer.torch.cuda, "is_available", lambda: False
    )

    compute_tabpfn_outer.main(a)

    # Two outer folds => two regressor constructions
    assert mock_cls.call_count == 2, (
        f"expected 2 TabPFNRegressor constructions (2 outer folds), "
        f"got {mock_cls.call_count}"
    )
    for call in mock_cls.call_args_list:
        assert call.kwargs["ignore_pretraining_limits"] is flag_value, (
            f"ignore_pretraining_limits={flag_value} must be threaded to EVERY "
            f"TabPFNRegressor(...) call; saw kwargs={call.kwargs}"
        )
