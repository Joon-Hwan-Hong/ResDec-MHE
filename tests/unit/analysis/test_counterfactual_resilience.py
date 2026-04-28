"""Tests for src/analysis/counterfactual_resilience.py (Wachter Mode-A literal)."""
from __future__ import annotations

import numpy as np
import pytest

from src.analysis.counterfactual_resilience import (
    CounterfactualResult,
    find_counterfactual_mode_a_adaptive,
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
