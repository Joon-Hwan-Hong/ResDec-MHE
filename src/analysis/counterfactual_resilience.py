"""Counterfactual explanations for the resilience composite.

Given a trained model and a per-subject input ``x``, find the smallest input
perturbation (under an L2 budget) that flips the model's prediction from
"vulnerable" (low resilience) to "resilient" (high resilience). The
perturbation magnitude per feature tells us which input dimensions the model
considers most actionable for that specific subject.

This is a model-agnostic gradient-based approach (cf. Wachter et al. 2017,
"Counterfactual Explanations Without Opening the Black Box"). The objective:

    minimize_{x_cf}    || x_cf - x_init ||_2^2
    subject to         f(x_cf) >= target_y

Solved via projected gradient ascent on f with an L2-distance penalty.

The model is passed as a callable ``f(x) -> y`` (any framework), so this
module has no PyTorch / Lightning dependency and is unit-testable with a
simple linear synthetic model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class CounterfactualResult:
    """Result of a single counterfactual search."""
    x_init: np.ndarray
    x_cf: np.ndarray
    y_init: float
    y_cf: float
    target_y: float
    success: bool
    n_steps_used: int
    l2_distance: float

    def to_dict(self) -> dict:
        return {
            "y_init": self.y_init,
            "y_cf": self.y_cf,
            "target_y": self.target_y,
            "success": self.success,
            "n_steps_used": self.n_steps_used,
            "l2_distance": self.l2_distance,
            "perturbation": (self.x_cf - self.x_init).tolist(),
        }


def find_counterfactual(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x_init: np.ndarray,
    target_y: float,
    *,
    lr: float = 0.05,
    max_steps: int = 200,
    l2_budget: float | None = None,
    lambda_dist: float = 1.0,
    tol: float = 1e-4,
) -> CounterfactualResult:
    """Find x_cf near x_init such that f(x_cf) >= target_y, minimizing L2 distance.

    Algorithm: gradient ascent on the augmented objective
        L(x) = f(x) - lambda * || x - x_init ||_2^2
    until f(x) >= target_y or max_steps exhausted. Optionally project onto
    an L2 ball of radius ``l2_budget`` around x_init.

    Parameters
    ----------
    f, grad_f
        Model forward and gradient functions, both taking and returning
        numpy arrays of shape ``(n_features,)``.
    x_init
        Starting input (the subject's true features), shape ``(n_features,)``.
    target_y
        Desired model output (e.g., the "resilient" threshold).
    lr
        Learning rate for gradient steps.
    max_steps
        Maximum gradient-ascent iterations.
    l2_budget
        Optional L2 norm cap on the perturbation. If exceeded, the
        perturbation is projected back onto the L2 ball each step.
    lambda_dist
        Weight on the L2 distance penalty in the augmented objective.
    tol
        Convergence tolerance: stop early once f(x) >= target_y - tol.

    Returns
    -------
    CounterfactualResult
        Search outcome with init/final inputs, init/final predictions,
        success flag, step count, and L2 distance.
    """
    x = np.array(x_init, dtype=np.float64, copy=True)
    y_init = float(f(x_init))
    y_curr = y_init
    n_steps = 0
    success = False

    for step in range(max_steps):
        n_steps = step + 1
        if y_curr >= target_y - tol:
            success = True
            break
        # Gradient: ∇L = ∇f(x) - 2 * lambda * (x - x_init)
        g = np.asarray(grad_f(x), dtype=np.float64)
        penalty_grad = 2.0 * lambda_dist * (x - x_init)
        x = x + lr * (g - penalty_grad)
        if l2_budget is not None:
            delta = x - x_init
            d_norm = float(np.linalg.norm(delta))
            if d_norm > l2_budget:
                x = x_init + delta * (l2_budget / d_norm)
        y_curr = float(f(x))

    if y_curr >= target_y - tol:
        success = True
    return CounterfactualResult(
        x_init=np.asarray(x_init, dtype=np.float64).copy(),
        x_cf=x,
        y_init=y_init,
        y_cf=y_curr,
        target_y=float(target_y),
        success=success,
        n_steps_used=n_steps,
        l2_distance=float(np.linalg.norm(x - x_init)),
    )


def batch_counterfactuals(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    target_y: float,
    **kwargs,
) -> list[CounterfactualResult]:
    """Find counterfactuals for each row of X (loops; not vectorized)."""
    results = []
    for i in range(X.shape[0]):
        results.append(find_counterfactual(f, grad_f, X[i], target_y, **kwargs))
    return results
