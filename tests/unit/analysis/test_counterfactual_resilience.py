"""Tests for src/analysis/counterfactual_resilience.py."""
from __future__ import annotations

import numpy as np
import pytest

from src.analysis.counterfactual_resilience import (
    CounterfactualResult,
    batch_counterfactuals,
    find_counterfactual,
)


def _linear_model(w: np.ndarray, b: float = 0.0):
    """Return (f, grad_f) for f(x) = w·x + b."""
    def f(x: np.ndarray) -> float:
        return float(np.dot(w, x) + b)
    def grad_f(x: np.ndarray) -> np.ndarray:
        return w.astype(np.float64)
    return f, grad_f


def test_counterfactual_finds_solution_for_linear_model():
    """Wachter (small lambda): y = w·x, target reached within tol.

    Wachter equilibrium f(x*) = y_init + ||w||² / (||w||² + lambda) * (target - y_init).
    With lambda << ||w||², the equilibrium converges to ≈ target.
    """
    rng = np.random.default_rng(0)
    n = 5
    w = rng.normal(size=n)
    f, grad_f = _linear_model(w, b=0.0)
    x_init = rng.normal(size=n)
    y_init = f(x_init)
    target = y_init + 1.0
    # Small lambda so equilibrium is close to target.
    result = find_counterfactual(
        f, grad_f, x_init, target, lr=0.05, max_steps=3000,
        lambda_dist=0.001, tol=1e-2,
    )
    assert result.success, f"failed to converge: y_cf={result.y_cf:.4f}, target={target:.4f}"
    assert abs(result.y_cf - target) < 1e-2


def test_counterfactual_records_init_and_final_predictions():
    w = np.array([1.0, 2.0, 0.5])
    f, grad_f = _linear_model(w, b=0.0)
    x_init = np.array([1.0, 1.0, 1.0])
    target = 5.0
    result = find_counterfactual(
        f, grad_f, x_init, target, lr=0.01, max_steps=500, lambda_dist=0.1,
    )
    assert result.y_init == pytest.approx(np.dot(w, x_init))
    assert isinstance(result, CounterfactualResult)


def test_counterfactual_starts_at_target_succeeds_immediately():
    """Wachter targets EXACT value; if y_init == target, no perturbation needed."""
    w = np.array([1.0, 1.0])
    f, grad_f = _linear_model(w, b=0.0)
    x_init = np.array([5.0, 5.0])  # f = 10
    target = 10.0  # Already at target.
    result = find_counterfactual(f, grad_f, x_init, target, max_steps=10)
    assert result.success
    assert result.n_steps_used == 1  # First iteration check passes immediately.
    assert result.l2_distance == pytest.approx(0.0)


def test_counterfactual_records_seed_in_result():
    """seed parameter is recorded in the result dataclass."""
    w = np.array([1.0])
    f, grad_f = _linear_model(w, b=0.0)
    x_init = np.array([0.0])
    result = find_counterfactual(
        f, grad_f, x_init, target_y=0.5, seed=12345, max_steps=500,
    )
    assert result.seed == 12345
    assert result.to_dict()["seed"] == 12345


def test_counterfactual_fails_under_tight_l2_budget():
    """If l2_budget is too small, search should fail (not find a solution)."""
    w = np.array([1.0])  # 1D, simple
    f, grad_f = _linear_model(w, b=0.0)
    x_init = np.array([0.0])  # f = 0
    target = 5.0  # Need x = 5 → L2 dist 5
    result = find_counterfactual(
        f, grad_f, x_init, target, lr=0.1, max_steps=200,
        l2_budget=1.0,  # Cap perturbation at 1, but need 5
        lambda_dist=0.0,
    )
    assert not result.success
    assert result.l2_distance <= 1.0 + 1e-6


def test_counterfactual_to_dict_serializable():
    w = np.array([1.0, 1.0])
    f, grad_f = _linear_model(w, b=0.0)
    x_init = np.array([0.0, 0.0])
    result = find_counterfactual(f, grad_f, x_init, 1.0, lr=0.05, max_steps=100)
    d = result.to_dict()
    assert "perturbation" in d
    assert isinstance(d["perturbation"], list)
    assert isinstance(d["success"], bool)


def test_batch_counterfactuals_returns_list():
    w = np.array([1.0, 1.0])
    f, grad_f = _linear_model(w, b=0.0)
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    results = batch_counterfactuals(f, grad_f, X, target_y=2.0,
                                     lr=0.05, max_steps=500, lambda_dist=0.01)
    assert len(results) == 3
    assert all(isinstance(r, CounterfactualResult) for r in results)


def test_batch_counterfactuals_rejects_mismatched_seeds():
    w = np.array([1.0])
    f, grad_f = _linear_model(w, b=0.0)
    X = np.array([[0.0], [1.0]])
    with pytest.raises(ValueError, match="seeds length"):
        batch_counterfactuals(f, grad_f, X, target_y=1.0, seeds=[1, 2, 3])  # 3 vs 2
