"""Counterfactual explanations for the resilience composite (Wachter et al. 2017).

Given a trained model and a per-subject input ``x``, find the smallest input
perturbation (under an L2 budget) that drives the model's prediction to a
target value (typically the "resilient" threshold). The perturbation
magnitude per feature tells us which input dimensions the model considers
most actionable for that specific subject.

Implementation follows Wachter, Mittelstadt & Russell (2017), "Counterfactual
Explanations Without Opening the Black Box: Automated Decisions and the
GDPR." The objective:

    L(x) = (f(x) - y_target)^2 + lambda * d(x, x_init)

minimized via gradient *descent*. Convergence test: |f(x) - y_target| <= tol
(reaches target) AND distance penalty has plateaued.

The model is passed as callables ``f, grad_f`` (any framework), so this
module has no PyTorch / Lightning dependency and is unit-testable with a
simple linear synthetic model. ``from_torch_model`` provides a convenience
wrapper that builds ``(f, grad_f)`` from a PyTorch nn.Module via autograd.
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
    seed: int
    lambda_used: float = 0.0  # for Mode-A adaptive: the final λ attempted

    def to_dict(self) -> dict:
        return {
            "y_init": self.y_init,
            "y_cf": self.y_cf,
            "target_y": self.target_y,
            "success": self.success,
            "n_steps_used": self.n_steps_used,
            "l2_distance": self.l2_distance,
            "seed": self.seed,
            "lambda_used": self.lambda_used,
            "perturbation": (self.x_cf - self.x_init).tolist(),
        }


def find_counterfactual(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x_init: np.ndarray,
    target_y: float,
    *,
    lr: float = 0.05,
    max_steps: int = 500,
    l2_budget: float | None = None,
    lambda_dist: float = 0.1,
    tol: float = 1e-3,
    seed: int = 42,
) -> CounterfactualResult:
    """Find x_cf near x_init driving f(x_cf) to target_y, per Wachter et al. 2017.

    Loss: ``L(x) = (f(x) - target_y)^2 + lambda_dist * ||x - x_init||_2^2``,
    minimized via gradient descent. ``grad L = 2*(f(x)-target)*grad_f -
    2*lambda*(x - x_init)``... wait this is wrong sign. Actually we minimize
    L so descent step is ``x -= lr * grad L``:

        grad L = 2*(f(x)-target_y)*grad_f(x) + 2*lambda_dist*(x - x_init)
        x_new = x - lr * grad L

    Stopping: |f(x) - target_y| <= tol AND no further distance reduction in
    last patience steps. Or max_steps exhausted.

    Parameters
    ----------
    f, grad_f
        Model forward and gradient functions.
    x_init
        Starting input, shape ``(n_features,)``.
    target_y
        Desired model output (e.g., resilient threshold).
    lr
        Learning rate.
    max_steps
        Maximum gradient-descent iterations (default 500).
    l2_budget
        Optional L2 norm cap on the perturbation; projected back if exceeded.
    lambda_dist
        Weight on the L2 distance penalty.
    tol
        Convergence tolerance: stop once |f(x) - target_y| <= tol.
    seed
        Recorded in the result for provenance (no randomness in this base
        deterministic algorithm; reserved for future stochastic extensions).

    Returns
    -------
    CounterfactualResult
        Search outcome.
    """
    x = np.array(x_init, dtype=np.float64, copy=True)
    y_init = float(f(x_init))
    y_curr = y_init
    n_steps = 0

    for step in range(max_steps):
        n_steps = step + 1
        residual = y_curr - target_y
        if abs(residual) <= tol:
            break
        # Wachter loss gradient: 2*(f-target)*grad_f + 2*lambda*(x - x_init)
        g = np.asarray(grad_f(x), dtype=np.float64)
        grad_L = 2.0 * residual * g + 2.0 * lambda_dist * (x - x_init)
        x = x - lr * grad_L
        if l2_budget is not None:
            delta = x - x_init
            d_norm = float(np.linalg.norm(delta))
            if d_norm > l2_budget:
                x = x_init + delta * (l2_budget / d_norm)
        y_curr = float(f(x))

    success = abs(y_curr - target_y) <= tol
    return CounterfactualResult(
        x_init=np.asarray(x_init, dtype=np.float64).copy(),
        x_cf=x,
        y_init=y_init,
        y_cf=y_curr,
        target_y=float(target_y),
        success=success,
        n_steps_used=n_steps,
        l2_distance=float(np.linalg.norm(x - x_init)),
        seed=int(seed),
    )


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
    seed: int = 42,
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
    seed
        Recorded in the result for provenance.

    Returns
    -------
    CounterfactualResult
        Best attempt. ``success=True`` iff target was reached for at least one
        λ; ``lambda_used`` records the λ of the returned attempt.
    """
    x_init_arr = np.asarray(x_init, dtype=np.float64).copy()
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
        for step in range(max_steps):
            n_steps = step + 1
            residual = y_curr - target_y
            if abs(residual) <= tol:
                break
            g = np.asarray(grad_f(x), dtype=np.float64)
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
            y_curr = float(f(x))

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
                seed=int(seed),
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
        seed=int(seed),
        lambda_used=float(best_lambda),
    )


def batch_counterfactuals(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    target_y: float,
    *,
    seeds: list[int] | None = None,
    **kwargs,
) -> list[CounterfactualResult]:
    """Find counterfactuals for each row of X (loops; not vectorized).

    Parameters
    ----------
    seeds
        Optional per-subject seeds. Length must match ``X.shape[0]``. If
        None, all subjects use ``kwargs.get("seed", 42)``.
    """
    if seeds is not None and len(seeds) != X.shape[0]:
        raise ValueError(
            f"seeds length {len(seeds)} != X.shape[0] {X.shape[0]}"
        )
    results = []
    for i in range(X.shape[0]):
        per_kwargs = dict(kwargs)
        if seeds is not None:
            per_kwargs["seed"] = int(seeds[i])
        results.append(find_counterfactual(f, grad_f, X[i], target_y, **per_kwargs))
    return results


def from_torch_model(model, *, device: str = "cpu") -> tuple[Callable, Callable]:
    """Convenience: build ``(f, grad_f)`` from a PyTorch nn.Module via autograd.

    The returned ``f`` and ``grad_f`` accept numpy ``(n_features,)`` arrays
    and return numpy outputs. Assumes the model is a scalar-output regressor
    (one output per input). Sets the model to eval mode but does NOT
    disable batch-norm running stats — caller must ensure model state is
    appropriate for inference.

    Example
    -------
    >>> model.eval()
    >>> f, grad_f = from_torch_model(model)
    >>> result = find_counterfactual(f, grad_f, x_init, target_y=0.5)
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch required for from_torch_model") from exc

    model.eval()
    dev = torch.device(device)

    def f(x: np.ndarray) -> float:
        with torch.no_grad():
            xt = torch.tensor(x, dtype=torch.float32, device=dev).unsqueeze(0)
            y = model(xt)
            return float(y.squeeze().detach().cpu().numpy())

    def grad_f(x: np.ndarray) -> np.ndarray:
        xt = torch.tensor(
            x, dtype=torch.float32, device=dev, requires_grad=True,
        ).unsqueeze(0)
        y = model(xt).squeeze()
        y.backward()
        return xt.grad.squeeze(0).detach().cpu().numpy().astype(np.float64)

    return f, grad_f
