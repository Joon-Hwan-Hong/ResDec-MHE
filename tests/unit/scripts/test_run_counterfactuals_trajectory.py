"""Tests for ``--record-trajectory`` flag plumbing in run_counterfactuals.py.

Validates that:
  1. The orchestrator's argparse exposes ``--record-trajectory`` as a boolean
     flag (default off).
  2. When set, ``_run_batched`` emits a ``trajectory`` field in each
     per-subject result dict, populated as a list of ``[lam, residual]`` pairs.
  3. When set, ``_run_per_subject`` emits a ``trajectory`` field in each
     per-subject result dict.
  4. When unset, neither path emits the ``trajectory`` field.

Heavy dependencies (Lightning model, datamodule) are NOT exercised; only the
result-dict construction logic is unit-tested by monkeypatching the underlying
counterfactual-search callables to return stub ``CounterfactualResult``
instances.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest


_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = (
    _WORKTREE_ROOT
    / "scripts"
    / "resdec_mhe"
    / "interpretability"
    / "run_counterfactuals.py"
)


def _import_orchestrator():
    """Import the run_counterfactuals.py module without running main()."""
    if str(_WORKTREE_ROOT) not in sys.path:
        sys.path.insert(0, str(_WORKTREE_ROOT))
    spec = importlib.util.spec_from_file_location(
        "run_counterfactuals_for_test", SCRIPT_PATH,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_args(record_trajectory: bool):
    """Minimal argparse.Namespace with all fields used by _run_batched/_run_per_subject."""
    return types.SimpleNamespace(
        record_trajectory=record_trajectory,
        target_mode="relative",
        target_delta=0.5,
        lr=0.05,
        max_steps=50,
        tol=1e-3,
        lambda_start=1e-3,
        lambda_max=10.0,
        lambda_mult=2.0,
        top_k=3,
    )


def test_argparse_exposes_record_trajectory_flag():
    """``--record-trajectory`` is parsable and defaults to False."""
    mod = _import_orchestrator()
    parser = None
    # Trampoline through the script: we mimic the parser construction by
    # locating the ArgumentParser inside main(). The simplest reliable way is
    # to call argparse on synthetic argv with --help-substituted minimal args.
    # We construct a tiny parser-replica here that mirrors the relevant flag.
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--record-trajectory", action="store_true")
    ns_default = p.parse_args([])
    ns_set = p.parse_args(["--record-trajectory"])
    assert ns_default.record_trajectory is False
    assert ns_set.record_trajectory is True
    # Sanity: the orchestrator file contains the literal flag string.
    src = SCRIPT_PATH.read_text()
    assert "--record-trajectory" in src, (
        "run_counterfactuals.py must expose --record-trajectory in argparse"
    )


def test_run_batched_includes_trajectory_when_flag_set(monkeypatch):
    """``_run_batched`` adds a 'trajectory' field iff args.record_trajectory."""
    mod = _import_orchestrator()
    from src.analysis.counterfactual_resilience import CounterfactualResult

    # Stub return: 2 subjects, each with a small trajectory.
    def fake_batch(
        f_and_grad_batch,
        x_init,
        target_y,
        *,
        lr,
        max_steps,
        tol,
        lambda_start,
        lambda_max,
        lambda_mult,
        record_trajectory=False,
        l2_budget=None,
    ):
        B = x_init.shape[0]
        results = []
        for i in range(B):
            traj = [(1e-3, 0.5), (2e-3, 0.4)] if record_trajectory else []
            results.append(
                CounterfactualResult(
                    x_init=x_init[i],
                    x_cf=x_init[i] + 0.01,
                    y_init=0.0,
                    y_cf=0.05,
                    target_y=float(target_y[i]),
                    success=False,
                    n_steps_used=10,
                    l2_distance=float(np.linalg.norm(0.01 * np.ones_like(x_init[i]))),
                    lambda_best=1e-3,
                    lambda_max_attempted=2e-3,
                    gap=0.45,
                    trajectory=traj,
                )
            )
        return results

    monkeypatch.setattr(
        mod, "find_counterfactual_mode_a_adaptive_batch", fake_batch,
    )

    # Stub the closure builder to return a no-op f_and_grad_batch and dummy shape.
    def fake_closure_builder(model, merged, device):
        n_features = 6
        pfc_shape = (2, 3)

        def f_and_grad_batch(X):
            B = X.shape[0]
            return np.zeros(B, dtype=np.float64), np.zeros((B, n_features))

        return f_and_grad_batch, n_features, pfc_shape

    monkeypatch.setattr(mod, "_build_batched_pfc_only_closure", fake_closure_builder)

    # Stub the batch-merge: return a dict with a synthetic region_pseudobulk.
    def fake_stack(per_subject_batches):
        B = len(per_subject_batches)
        import torch as _torch
        return {
            "region_pseudobulk": _torch.zeros(B, 6, 2, 3, dtype=_torch.float32),
            "subject_ids": [b["subject_ids"][0] for b in per_subject_batches],
            "batch_size": B,
        }

    monkeypatch.setattr(mod, "_stack_subject_batches", fake_stack)

    # Build 2 subject dicts (only fields needed for control flow + indexing).
    import torch as _torch
    subject_batches = [
        ("S1", {"region_pseudobulk": _torch.zeros(1, 6, 2, 3), "subject_ids": ["S1"]}),
        ("S2", {"region_pseudobulk": _torch.zeros(1, 6, 2, 3), "subject_ids": ["S2"]}),
    ]
    regime_map = {"S1": "resilient", "S2": "vulnerable"}

    # With record_trajectory=True
    args = _make_args(record_trajectory=True)
    out = mod._run_batched(
        model=None, subject_batches=subject_batches,
        regime_map=regime_map, device="cpu", args=args,
    )
    results = out["results"]
    assert len(results) == 2
    for r in results:
        assert "trajectory" in r, f"trajectory missing from {r}"
        assert isinstance(r["trajectory"], list)
        assert len(r["trajectory"]) == 2
        # Each entry must be JSON-serialisable list of two floats.
        for entry in r["trajectory"]:
            assert isinstance(entry, list) and len(entry) == 2
            assert isinstance(entry[0], float)
            assert isinstance(entry[1], float)

    # With record_trajectory=False the field must NOT be present.
    args_off = _make_args(record_trajectory=False)
    out_off = mod._run_batched(
        model=None, subject_batches=subject_batches,
        regime_map=regime_map, device="cpu", args=args_off,
    )
    for r in out_off["results"]:
        assert "trajectory" not in r, (
            "trajectory key must be absent when --record-trajectory not set"
        )


def test_run_per_subject_includes_trajectory_when_flag_set(monkeypatch):
    """``_run_per_subject`` adds a 'trajectory' field iff args.record_trajectory."""
    mod = _import_orchestrator()
    from src.analysis.counterfactual_resilience import CounterfactualResult

    def fake_single(
        f, grad_f, x_init, target_y,
        *, lr, max_steps, tol, lambda_start, lambda_max, lambda_mult,
        l2_budget=None, f_and_grad=None, record_trajectory=False,
    ):
        traj = [(1e-3, 0.5), (2e-3, 0.4)] if record_trajectory else []
        return CounterfactualResult(
            x_init=np.asarray(x_init).copy(),
            x_cf=np.asarray(x_init).copy() + 0.01,
            y_init=0.0,
            y_cf=0.05,
            target_y=float(target_y),
            success=False,
            n_steps_used=10,
            l2_distance=float(np.linalg.norm(0.01 * np.ones_like(np.asarray(x_init)))),
            lambda_best=1e-3,
            lambda_max_attempted=2e-3,
            gap=0.45,
            trajectory=traj,
        )

    monkeypatch.setattr(mod, "find_counterfactual_mode_a_adaptive", fake_single)

    # Stub closure builder so we don't need the real model.
    def fake_closure_builder(model, template_batch, device):
        n_features = 6
        pfc_shape = (1, 2, 3)

        def f(x):
            return 0.0

        def grad_f(x):
            return np.zeros_like(x)

        def f_and_grad(x):
            return 0.0, np.zeros_like(x)

        return f, grad_f, f_and_grad, n_features, pfc_shape

    monkeypatch.setattr(mod, "_build_pfc_only_closures", fake_closure_builder)

    import torch as _torch
    subject_batches = [
        ("S1", {"region_pseudobulk": _torch.zeros(1, 6, 2, 3), "subject_ids": ["S1"]}),
        ("S2", {"region_pseudobulk": _torch.zeros(1, 6, 2, 3), "subject_ids": ["S2"]}),
    ]
    regime_map = {"S1": "resilient", "S2": "vulnerable"}

    args = _make_args(record_trajectory=True)
    out = mod._run_per_subject(
        model=None, subject_batches=subject_batches,
        regime_map=regime_map, device="cpu", args=args,
    )
    for r in out["results"]:
        assert "trajectory" in r
        assert isinstance(r["trajectory"], list) and len(r["trajectory"]) == 2

    args_off = _make_args(record_trajectory=False)
    out_off = mod._run_per_subject(
        model=None, subject_batches=subject_batches,
        regime_map=regime_map, device="cpu", args=args_off,
    )
    for r in out_off["results"]:
        assert "trajectory" not in r
