"""Counterfactual explanations for the resilience composite (Wachter et al. 2017).

Given a trained model and a per-subject input ``x``, find the smallest input
perturbation that drives the model's prediction to a target value (typically
the opposite-regime threshold). The perturbation magnitude per feature tells
us which input dimensions the model considers most actionable for that
specific subject.

Implementation follows Wachter, Mittelstadt & Russell (2017), "Counterfactual
Explanations Without Opening the Black Box: Automated Decisions and the GDPR"
in the Mode-A literal form: minimize ``L(x, λ) = ‖x − x_init‖² + λ ·
(f(x) − y_target)²`` via gradient descent, with adaptive λ doubling — start
λ small (distance dominates), run inner GD for ``max_steps``, and if target
not reached double λ and reset ``x ← x_init``; terminate at success or
``λ > λ_max``. Unit-norm gradient clipping keeps the step bounded at high λ.

The model is passed as callables ``f, grad_f`` (and optionally ``f_and_grad``
to amortize one forward per step), so this module has no PyTorch / Lightning
dependency and is unit-testable with a simple linear synthetic model.
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
    lambda_used: float = 0.0  # the final λ attempted in adaptive doubling

    def to_dict(self) -> dict:
        return {
            "y_init": self.y_init,
            "y_cf": self.y_cf,
            "target_y": self.target_y,
            "success": self.success,
            "n_steps_used": self.n_steps_used,
            "l2_distance": self.l2_distance,
            "lambda_used": self.lambda_used,
            "perturbation": (self.x_cf - self.x_init).tolist(),
        }


def find_counterfactual_mode_a_adaptive(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x_init: np.ndarray,
    target_y: float,
    *,
    lr: float = 0.05,
    max_steps: int = 500,
    tol: float = 1e-3,
    lambda_start: float = 1e-3,
    lambda_max: float = 1e3,
    lambda_mult: float = 2.0,
    l2_budget: float | None = None,
    f_and_grad: Callable[[np.ndarray], tuple[float, np.ndarray]] | None = None,
) -> CounterfactualResult:
    """Mode-A adaptive-λ counterfactual search (Wachter 2017 preferred variant).

    Loss formulation:

        L(x, λ) = ||x - x_init||_2^2 + λ * (f(x) - target_y)^2

    Note that λ multiplies the *prediction* loss, not the distance. Starting
    with a small λ emphasizes distance (x stays near x_init); if target is
    not reached after ``max_steps`` of gradient descent, λ is doubled and
    the inner loop restarts from ``x_init``. Repeats until target reached or
    ``lambda_max`` exceeded.

    Gradient (d/dx of the Mode-A loss):

        grad L = 2*(x - x_init) + 2 * λ * (f(x) - target_y) * grad_f(x)

    Parameters
    ----------
    f, grad_f
        Model forward and gradient (numpy).
    x_init
        Starting input.
    target_y
        Desired model output.
    lr
        Gradient-descent step size.
    max_steps
        Inner-loop cap per λ value.
    tol
        Convergence tolerance: |f(x) - target_y| <= tol counts as success.
    lambda_start
        Initial λ (small = distance-dominant).
    lambda_max
        Doubling stops once λ exceeds this value.
    lambda_mult
        λ doubling factor (default 2.0 per Wachter).
    l2_budget
        Optional L2 norm cap on the perturbation; projected back if exceeded.

    Returns
    -------
    CounterfactualResult
        Best attempt. ``success=True`` iff target was reached for at least one
        λ; ``lambda_used`` records the λ of the returned attempt.
    """
    x_init_arr = np.asarray(x_init, dtype=np.float64).copy()
    # Use f_and_grad if provided (1 forward per step instead of 2);
    # the standalone f and grad_f remain authoritative for x_init's y.
    if f_and_grad is not None:
        y_init, _g_init = f_and_grad(x_init_arr)
        y_init = float(y_init)
    else:
        y_init = float(f(x_init_arr))
    best_x = x_init_arr.copy()
    best_y = y_init
    best_n_steps = 0
    best_lambda = lambda_start

    lam = lambda_start
    while True:
        x = x_init_arr.copy()
        y_curr = y_init
        n_steps = 0
        # If f_and_grad is provided we need (y, g) at the CURRENT x, then
        # step; the new (y, g) at new x is computed at the next iteration.
        # Initialize g_at_x for the first iteration if combined-call mode.
        if f_and_grad is not None:
            _, g_at_x = f_and_grad(x)
        for step in range(max_steps):
            n_steps = step + 1
            residual = y_curr - target_y
            if abs(residual) <= tol:
                break
            if f_and_grad is None:
                g = np.asarray(grad_f(x), dtype=np.float64)
            else:
                g = g_at_x  # already computed at current x
            grad_L = 2.0 * (x - x_init_arr) + 2.0 * lam * residual * g
            # L2 gradient clipping: keeps step size bounded at high λ where
            # raw |grad_L| scales with λ and can cause divergence. Step norm
            # is capped at 1 unit per iteration (actual step = lr * clipped).
            g_norm = float(np.linalg.norm(grad_L))
            if g_norm > 1.0:
                grad_L = grad_L / g_norm
            x = x - lr * grad_L
            if l2_budget is not None:
                delta = x - x_init_arr
                d_norm = float(np.linalg.norm(delta))
                if d_norm > l2_budget:
                    x = x_init_arr + delta * (l2_budget / d_norm)
            if f_and_grad is None:
                y_curr = float(f(x))
            else:
                # ONE forward+backward, get both new y and grad for next iter.
                y_new, g_new = f_and_grad(x)
                y_curr = float(y_new)
                g_at_x = np.asarray(g_new, dtype=np.float64)

        # Track best-so-far by proximity to target
        if abs(y_curr - target_y) < abs(best_y - target_y):
            best_x = x
            best_y = y_curr
            best_n_steps = n_steps
            best_lambda = lam

        if abs(y_curr - target_y) <= tol:
            # Success — return current λ attempt
            return CounterfactualResult(
                x_init=x_init_arr,
                x_cf=x,
                y_init=y_init,
                y_cf=y_curr,
                target_y=float(target_y),
                success=True,
                n_steps_used=n_steps,
                l2_distance=float(np.linalg.norm(x - x_init_arr)),
                lambda_used=float(lam),
            )

        if lam >= lambda_max:
            break
        lam = min(lam * lambda_mult, lambda_max)

    # Exhausted λ budget without reaching target
    return CounterfactualResult(
        x_init=x_init_arr,
        x_cf=best_x,
        y_init=y_init,
        y_cf=best_y,
        target_y=float(target_y),
        success=False,
        n_steps_used=best_n_steps,
        l2_distance=float(np.linalg.norm(best_x - x_init_arr)),
        lambda_used=float(best_lambda),
    )


