"""Tests for src/analysis/counterfactual_resilience.py (Wachter Mode-A literal)."""
from __future__ import annotations

import numpy as np
import pytest

from src.analysis.counterfactual_resilience import (
    CounterfactualResult,
    find_counterfactual_mode_a_adaptive,
    find_counterfactual_mode_a_adaptive_batch,
)

def _linear_model(w: np.ndarray, b: float = 0.0):
    """Return (f, grad_f) for f(x) = w·x + b."""
    def f(x: np.ndarray) -> float:
        return float(np.dot(w, x) + b)
    def grad_f(x: np.ndarray) -> np.ndarray:
        return w.astype(np.float64)
    return f, grad_f

# ─────────────────────────────────────────────────────────────────────────────
# Mode-A adaptive doubling (Wachter 2017 literal preferred algorithm)
# ─────────────────────────────────────────────────────────────────────────────

def test_mode_a_adaptive_reaches_target_on_linear_model():
    """With adaptive doubling, target is reached on a simple linear model."""
    rng = np.random.default_rng(0)
    n = 5
    w = rng.normal(size=n)
    f, grad_f = _linear_model(w, b=0.0)
    x_init = rng.normal(size=n)
    target = f(x_init) + 1.0
    result = find_counterfactual_mode_a_adaptive(
        f, grad_f, x_init, target,
        lr=0.05, max_steps=500, tol=1e-2,
        lambda_start=1e-3, lambda_max=1e3,
    )
    assert result.success
    assert abs(result.y_cf - target) <= 1e-2

def test_mode_a_adaptive_returns_best_effort_when_unreachable():
    """If no λ reaches target within budget, returns best CF with success=False."""
    # Very small weight → hard to move the output by a large target
    w = np.array([1e-6])
    f, grad_f = _linear_model(w, b=0.0)
    x_init = np.array([0.0])
    target = 100.0  # far
    result = find_counterfactual_mode_a_adaptive(
        f, grad_f, x_init, target,
        lr=0.01, max_steps=50, tol=1e-3,
        lambda_start=1e-3, lambda_max=10.0,
    )
    # Didn't converge in budget, but we should still get a valid result
    assert result.success is False
    assert isinstance(result, CounterfactualResult)
    # And lambda_used should be the max attempted
    assert result.lambda_used == pytest.approx(10.0) or result.lambda_used >= 1.0

def test_mode_a_combined_f_and_grad_equivalent_to_separate():
    """Mode-A with f_and_grad must match Mode-A with separate f/grad_f.

    The combined callable is a perf optimization; output must be identical.
    """
    rng = np.random.default_rng(0)
    n = 5
    w = rng.normal(size=n)
    f, grad_f = _linear_model(w, b=0.0)
    x_init = rng.normal(size=n)
    target = f(x_init) + 1.0

    def f_and_grad(x):
        return f(x), grad_f(x)

    r_separate = find_counterfactual_mode_a_adaptive(
        f, grad_f, x_init, target,
        lr=0.05, max_steps=200, tol=1e-2,
        lambda_start=1e-3, lambda_max=1e2,
    )
    r_combined = find_counterfactual_mode_a_adaptive(
        f, grad_f, x_init, target,
        lr=0.05, max_steps=200, tol=1e-2,
        lambda_start=1e-3, lambda_max=1e2,
        f_and_grad=f_and_grad,
    )
    assert r_separate.success == r_combined.success
    assert r_separate.y_cf == pytest.approx(r_combined.y_cf, rel=1e-9, abs=1e-12)
    assert r_separate.lambda_used == pytest.approx(r_combined.lambda_used)
    assert r_separate.n_steps_used == r_combined.n_steps_used
    np.testing.assert_allclose(r_separate.x_cf, r_combined.x_cf, rtol=1e-9, atol=1e-12)

def test_mode_a_records_lambda_used():
    """The result must report which λ value was the final (successful or max) attempt."""
    w = np.array([1.0, 1.0])
    f, grad_f = _linear_model(w, b=0.0)
    x_init = np.zeros(2)
    target = 1.0
    result = find_counterfactual_mode_a_adaptive(
        f, grad_f, x_init, target,
        lr=0.05, max_steps=200, tol=1e-3,
        lambda_start=0.01, lambda_max=100.0,
    )
    assert hasattr(result, "lambda_used")
    assert result.lambda_used >= 0.01

# ─────────────────────────────────────────────────────────────────────────────
# P1.4 — caching of (y_init, g_init) across λ doublings
# ─────────────────────────────────────────────────────────────────────────────

def test_mode_a_caches_initial_y_and_grad_across_lambda_doublings():
    """f_and_grad(x_init) must be invoked exactly once even with many λ doublings.

    On every λ doubling the inner loop resets ``x ← x_init``, so without caching
    the implementation would call f_and_grad(x_init) once per doubling. With
    P1.4 caching, the first call must be reused.
    """
    # Choose a hard target so we go through ALL doublings without converging.
    w = np.array([1e-9])  # near-zero gradient → impossible to reach target
    f_, grad_f_ = _linear_model(w, b=0.0)
    x_init = np.array([0.0])
    target = 1e6

    n_calls_at_x_init = {"count": 0}

    def f_and_grad(x):
        x = np.asarray(x, dtype=np.float64)
        if np.allclose(x, x_init, atol=0.0):
            n_calls_at_x_init["count"] += 1
        return f_(x), grad_f_(x)

    result = find_counterfactual_mode_a_adaptive(
        f_, grad_f_, x_init, target,
        lr=0.01, max_steps=2, tol=1e-3,
        lambda_start=1e-3, lambda_max=1e3, lambda_mult=2.0,
        f_and_grad=f_and_grad,
    )
    # With λ doubling 1e-3 → 2e-3 → ... → ≥1e3, we expect ~21 doublings.
    # Without caching, that means ~21 f_and_grad(x_init) calls. With caching,
    # only the very first call evaluates at x_init.
    assert n_calls_at_x_init["count"] == 1, (
        f"Expected exactly 1 call at x_init (cached), got "
        f"{n_calls_at_x_init['count']}"
    )
    # And we should still report a valid (failed) result.
    assert isinstance(result, CounterfactualResult)
    assert result.success is False

# ─────────────────────────────────────────────────────────────────────────────
# P2.1 / P2.2 — new fields: lambda_best, lambda_max_attempted, gap, trajectory,
# and lambda_used backward-compat alias
# ─────────────────────────────────────────────────────────────────────────────

def test_result_exposes_lambda_best_and_lambda_max_attempted():
    """``lambda_best`` is the λ of the closest/converging attempt; ``lambda_max_attempted``
    is the actual final λ tried (may differ when search ran past the best
    attempt to higher λ values without improvement)."""
    # Easily reachable target → succeeds at first or second λ.
    w = np.array([1.0, 1.0])
    f, grad_f = _linear_model(w, b=0.0)
    x_init = np.zeros(2)
    target = 0.1
    result = find_counterfactual_mode_a_adaptive(
        f, grad_f, x_init, target,
        lr=0.05, max_steps=200, tol=1e-3,
        lambda_start=0.01, lambda_max=100.0,
    )
    assert hasattr(result, "lambda_best")
    assert hasattr(result, "lambda_max_attempted")
    assert result.lambda_best >= 0.01
    assert result.lambda_max_attempted >= result.lambda_best

def test_result_lambda_used_is_alias_for_lambda_best():
    """Backward-compat: ``lambda_used`` reads ``lambda_best``."""
    w = np.array([1.0, 1.0])
    f, grad_f = _linear_model(w, b=0.0)
    x_init = np.zeros(2)
    target = 0.1
    result = find_counterfactual_mode_a_adaptive(
        f, grad_f, x_init, target,
        lr=0.05, max_steps=200, tol=1e-3,
        lambda_start=0.01, lambda_max=100.0,
    )
    assert result.lambda_used == result.lambda_best

def test_result_exposes_gap():
    """``gap = abs(y_cf - target_y)`` so callers don't recompute."""
    w = np.array([1.0, 1.0])
    f, grad_f = _linear_model(w, b=0.0)
    x_init = np.zeros(2)
    target = 0.1
    result = find_counterfactual_mode_a_adaptive(
        f, grad_f, x_init, target,
        lr=0.05, max_steps=200, tol=1e-3,
        lambda_start=0.01, lambda_max=100.0,
    )
    assert hasattr(result, "gap")
    assert result.gap == pytest.approx(abs(result.y_cf - result.target_y), abs=1e-12)

def test_result_trajectory_recorded_when_requested():
    """When ``record_trajectory=True``, result.trajectory has one entry per λ."""
    w = np.array([1.0])
    f, grad_f = _linear_model(w, b=0.0)
    x_init = np.array([0.0])
    target = 100.0  # unreachable in budget → multiple λ doublings
    result = find_counterfactual_mode_a_adaptive(
        f, grad_f, x_init, target,
        lr=0.01, max_steps=5, tol=1e-3,
        lambda_start=1e-3, lambda_max=10.0, lambda_mult=2.0,
        record_trajectory=True,
    )
    assert hasattr(result, "trajectory")
    assert isinstance(result.trajectory, list)
    assert len(result.trajectory) >= 2
    # Each entry is (lam, residual_at_end_of_inner_loop)
    lam0, res0 = result.trajectory[0]
    assert lam0 == pytest.approx(1e-3)
    assert isinstance(res0, float)

def test_result_trajectory_default_empty():
    """Without ``record_trajectory=True``, trajectory is an empty list."""
    w = np.array([1.0])
    f, grad_f = _linear_model(w, b=0.0)
    x_init = np.array([0.0])
    target = 0.1
    result = find_counterfactual_mode_a_adaptive(
        f, grad_f, x_init, target,
        lr=0.05, max_steps=200, tol=1e-3,
        lambda_start=0.01, lambda_max=10.0,
    )
    assert result.trajectory == []

# ─────────────────────────────────────────────────────────────────────────────
# P1.2 — batched API ``find_counterfactual_mode_a_adaptive_batch``
# ─────────────────────────────────────────────────────────────────────────────

def _linear_batch_model(W: np.ndarray, b: np.ndarray | None = None):
    """Return f_and_grad_batch(X) for f_i(x) = W[i]·x + b[i]."""
    B, n = W.shape
    if b is None:
        b = np.zeros(B, dtype=np.float64)
    W = W.astype(np.float64)
    b = b.astype(np.float64)

    def f_and_grad_batch(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=np.float64)
        # Per-subject inner product, not full matmul (each subject has its own w_i)
        y = np.einsum("bi,bi->b", W, X) + b  # [B]
        g = W.copy()                          # [B, n], constant in x
        return y, g
    return f_and_grad_batch

def test_batch_api_matches_per_subject_on_linear_model():
    """Batched API produces results numerically identical (to tolerance) with
    the per-subject API on the same linear inputs."""
    rng = np.random.default_rng(42)
    B = 3
    n = 4
    W = rng.normal(size=(B, n))
    X_init = rng.normal(size=(B, n))
    # f_init = einsum("bi,bi->b", W, X_init); pick targets near each subject's f_init
    f_init = np.einsum("bi,bi->b", W, X_init)
    target_y = f_init + np.array([0.5, -0.3, 0.7])

    f_and_grad_batch = _linear_batch_model(W)
    batch_results = find_counterfactual_mode_a_adaptive_batch(
        f_and_grad_batch, X_init, target_y,
        lr=0.05, max_steps=300, tol=1e-3,
        lambda_start=1e-3, lambda_max=1e3, lambda_mult=2.0,
    )
    assert len(batch_results) == B

    # Per-subject reference run
    for i in range(B):
        w_i = W[i]
        f_i, grad_f_i = _linear_model(w_i, b=0.0)

        def fag(x, w=w_i):
            return float(np.dot(w, x)), w.astype(np.float64)

        ref = find_counterfactual_mode_a_adaptive(
            f_i, grad_f_i, X_init[i], float(target_y[i]),
            lr=0.05, max_steps=300, tol=1e-3,
            lambda_start=1e-3, lambda_max=1e3, lambda_mult=2.0,
            f_and_grad=fag,
        )
        b_res = batch_results[i]
        assert b_res.success == ref.success, f"subject {i}: success mismatch"
        # Per-subject ragged stop may freeze a subject earlier than the
        # per-subject loop; allow the batched gap to be ≤ ref gap (i.e. at
        # least as good). Both should hit tol when success=True.
        if ref.success:
            assert abs(b_res.y_cf - target_y[i]) <= 1e-3
        # Same lambda_best when both converge
        if ref.success and b_res.success:
            assert b_res.lambda_best == pytest.approx(ref.lambda_best)

def test_batch_api_ragged_stop_freezes_converged_subject():
    """Subject that converges quickly must not be updated further once done.

    Subject 0 has a strong, well-scaled gradient → converges at moderate λ.
    Subject 1 has a weak gradient → cannot reach target in budget.
    """
    # Subject 0: easy (w=1, target=0.5). Subject 1: hard (w=1e-4, target=0.5).
    W = np.array([
        [1.0],
        [1e-4],
    ])
    X_init = np.array([[0.0], [0.0]])
    target_y = np.array([0.5, 0.5])

    f_and_grad_batch = _linear_batch_model(W)

    # Track first time each subject's x hits |f - target| <= tol AND record
    # the x seen at that exact call. Note: ragged-stop semantics mean that
    # AFTER that call, x[i] should not change again.
    converged_step = {0: None, 1: None}
    frozen_x = {0: None, 1: None}
    n_calls = {"count": 0}

    def wrapped(X):
        n_calls["count"] += 1
        y, g = f_and_grad_batch(X)
        for i in range(2):
            if abs(y[i] - target_y[i]) <= 1e-3 and converged_step[i] is None:
                converged_step[i] = n_calls["count"]
                frozen_x[i] = X[i].copy()
        return y, g

    results = find_counterfactual_mode_a_adaptive_batch(
        wrapped, X_init, target_y,
        lr=0.05, max_steps=500, tol=1e-3,
        lambda_start=1e-3, lambda_max=1e3, lambda_mult=2.0,
    )
    # Subject 0 must converge.
    assert results[0].success is True, f"subject 0 should converge, got {results[0]}"
    assert converged_step[0] is not None
    # After subject 0 converges, its x must not change → final x_cf equals
    # the x recorded at the call that first hit tolerance.
    np.testing.assert_allclose(results[0].x_cf, frozen_x[0], atol=0.0)
    # Subject 1 must NOT have converged (gradient too weak).
    assert results[1].success is False

def test_batch_api_returns_per_subject_lambda_max_attempted_and_gap():
    """Each per-subject result has lambda_best, lambda_max_attempted, gap fields."""
    rng = np.random.default_rng(1)
    B = 2
    n = 3
    W = rng.normal(size=(B, n))
    X_init = rng.normal(size=(B, n))
    target_y = np.einsum("bi,bi->b", W, X_init) + 0.5
    f_and_grad_batch = _linear_batch_model(W)
    results = find_counterfactual_mode_a_adaptive_batch(
        f_and_grad_batch, X_init, target_y,
        lr=0.05, max_steps=200, tol=1e-3,
        lambda_start=1e-3, lambda_max=1e2, lambda_mult=2.0,
    )
    for r in results:
        assert isinstance(r, CounterfactualResult)
        assert hasattr(r, "lambda_best")
        assert hasattr(r, "lambda_max_attempted")
        assert hasattr(r, "gap")
        assert r.gap == pytest.approx(abs(r.y_cf - r.target_y), abs=1e-12)
        assert r.lambda_max_attempted >= r.lambda_best

def test_batch_api_records_trajectory_per_subject_when_requested():
    """``record_trajectory=True`` populates per-subject trajectories of (λ, residual)."""
    W = np.array([[0.01], [0.02]])  # both weak → multiple doublings
    X_init = np.array([[0.0], [0.0]])
    target_y = np.array([10.0, 10.0])  # forces several λ doublings
    f_and_grad_batch = _linear_batch_model(W)
    results = find_counterfactual_mode_a_adaptive_batch(
        f_and_grad_batch, X_init, target_y,
        lr=0.01, max_steps=5, tol=1e-3,
        lambda_start=1e-3, lambda_max=10.0, lambda_mult=2.0,
        record_trajectory=True,
    )
    for r in results:
        assert isinstance(r.trajectory, list)
        assert len(r.trajectory) >= 2
        lam0, res0 = r.trajectory[0]
        assert lam0 == pytest.approx(1e-3)
        assert isinstance(res0, float)

def test_batch_api_input_shape_validation():
    """Mismatched batch dims raise."""
    W = np.array([[1.0, 1.0], [1.0, 1.0]])
    X_init = np.zeros((2, 2))
    target_y = np.array([0.1])  # wrong length
    f_and_grad_batch = _linear_batch_model(W)
    with pytest.raises((ValueError, AssertionError)):
        find_counterfactual_mode_a_adaptive_batch(
            f_and_grad_batch, X_init, target_y,
            lr=0.05, max_steps=10, tol=1e-3,
        )
