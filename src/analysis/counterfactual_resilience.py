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

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class CounterfactualResult:
    """Result of a single counterfactual search.

    Fields
    ------
    x_init, x_cf, y_init, y_cf, target_y, success, n_steps_used, l2_distance
        Standard CF outputs.
    lambda_best
        λ value at which the closest (or successful) attempt was found.
    lambda_max_attempted
        Final λ value tried in the adaptive doubling search (may exceed
        ``lambda_best`` when the search continued past the best attempt
        without improvement).
    gap
        ``abs(y_cf - target_y)`` — convenience field so callers don't recompute.
    trajectory
        Optional list of ``(lam, residual_at_end)`` per λ doubling for
        diagnosis. Empty unless ``record_trajectory=True`` was passed to the
        search function.
    lambda_used
        Deprecated alias for ``lambda_best``; preserved for backward
        compatibility with existing JSON consumers.
    """
    x_init: np.ndarray
    x_cf: np.ndarray
    y_init: float
    y_cf: float
    target_y: float
    success: bool
    n_steps_used: int
    l2_distance: float
    lambda_best: float = 0.0  # λ of the closest/converging attempt
    lambda_max_attempted: float = 0.0  # final λ tried in the adaptive search
    gap: float = 0.0  # |y_cf - target_y|
    trajectory: list = field(default_factory=list)  # list[tuple[float, float]]

    @property
    def lambda_used(self) -> float:
        """Deprecated alias: returns ``lambda_best`` for backward compatibility."""
        return self.lambda_best

    def to_dict(self) -> dict:
        return {
            "y_init": self.y_init,
            "y_cf": self.y_cf,
            "target_y": self.target_y,
            "success": self.success,
            "n_steps_used": self.n_steps_used,
            "l2_distance": self.l2_distance,
            "lambda_best": self.lambda_best,
            "lambda_used": self.lambda_best,  # deprecated alias, retained
            "lambda_max_attempted": self.lambda_max_attempted,
            "gap": self.gap,
            "trajectory": list(self.trajectory),
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
    record_trajectory: bool = False,
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
    f_and_grad
        Optional combined forward+grad callable; reduces 2 evaluations to 1
        per step. ``f_and_grad(x_init)`` is cached and reused across all λ
        doublings (P1.4).
    record_trajectory
        If True, ``result.trajectory`` records ``(λ, residual_at_end_of_inner_loop)``
        for each λ value tried.

    Returns
    -------
    CounterfactualResult
        Best attempt. ``success=True`` iff target was reached for at least one
        λ; ``lambda_best`` records the λ at which the closest/successful
        attempt was found; ``lambda_max_attempted`` is the final λ tried.
    """
    x_init_arr = np.asarray(x_init, dtype=np.float64).copy()
    # P1.4: cache (y_init, g_init) from a single evaluation at x_init. Every
    # λ doubling resets x ← x_init, so the (y, g) at that point is invariant.
    if f_and_grad is not None:
        y_init_val, g_init_arr = f_and_grad(x_init_arr)
        y_init = float(y_init_val)
        g_init = np.asarray(g_init_arr, dtype=np.float64)
    else:
        y_init = float(f(x_init_arr))
        g_init = None  # not needed; grad_f(x) recomputed each step
    best_x = x_init_arr.copy()
    best_y = y_init
    best_n_steps = 0
    best_lambda = lambda_start

    trajectory: list[tuple[float, float]] = []
    lam = lambda_start
    lam_attempted = lam
    while True:
        x = x_init_arr.copy()
        y_curr = y_init
        n_steps = 0
        # If f_and_grad is provided we need (y, g) at the CURRENT x, then
        # step; the new (y, g) at new x is computed at the next iteration.
        # P1.4: at the start of every inner loop x == x_init, so reuse the
        # cached g_init instead of calling f_and_grad again.
        if f_and_grad is not None:
            g_at_x = g_init.copy()
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

        if record_trajectory:
            trajectory.append((float(lam), float(y_curr - target_y)))

        # Track best-so-far by proximity to target
        if abs(y_curr - target_y) < abs(best_y - target_y):
            best_x = x
            best_y = y_curr
            best_n_steps = n_steps
            best_lambda = lam

        if abs(y_curr - target_y) <= tol:
            # Success — return current λ attempt
            gap_val = abs(y_curr - target_y)
            return CounterfactualResult(
                x_init=x_init_arr,
                x_cf=x,
                y_init=y_init,
                y_cf=y_curr,
                target_y=float(target_y),
                success=True,
                n_steps_used=n_steps,
                l2_distance=float(np.linalg.norm(x - x_init_arr)),
                lambda_best=float(lam),
                lambda_max_attempted=float(lam),
                gap=float(gap_val),
                trajectory=trajectory,
            )

        if lam >= lambda_max:
            lam_attempted = lam
            break
        lam = min(lam * lambda_mult, lambda_max)
        lam_attempted = lam

    # Exhausted λ budget without reaching target
    gap_val = abs(best_y - target_y)
    return CounterfactualResult(
        x_init=x_init_arr,
        x_cf=best_x,
        y_init=y_init,
        y_cf=best_y,
        target_y=float(target_y),
        success=False,
        n_steps_used=best_n_steps,
        l2_distance=float(np.linalg.norm(best_x - x_init_arr)),
        lambda_best=float(best_lambda),
        lambda_max_attempted=float(lam_attempted),
        gap=float(gap_val),
        trajectory=trajectory,
    )


def find_counterfactual_mode_a_adaptive_batch(
    f_and_grad_batch: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    x_init: np.ndarray,
    target_y: np.ndarray,
    *,
    lr: float = 0.05,
    max_steps: int = 1000,
    tol: float = 1e-3,
    lambda_start: float = 1e-3,
    lambda_max: float = 1e3,
    lambda_mult: float = 2.0,
    l2_budget: float | None = None,
    record_trajectory: bool = False,
) -> list[CounterfactualResult]:
    """Batched Wachter Mode-A counterfactual search.

    Per-subject ragged stop: once subject ``i`` hits
    ``|f(x_i) - target_y_i| <= tol``, its ``x_i`` is frozen (excluded from the
    per-subject loss / grad updates of further inner GD iterations and λ
    doublings). Other subjects continue.

    Returns one ``CounterfactualResult`` per subject in the same order as
    ``x_init``.

    Parameters
    ----------
    f_and_grad_batch
        Callable taking ``[B, n_features]`` and returning ``(y: [B], g: [B, n])``.
        Per-subject Jacobian rows ``g[i] = ∂f_i/∂x_i`` evaluated at the current
        ``x[i]``. The orchestrator typically wraps a torch model's batched
        forward+backward.
    x_init
        ``[B, n_features]`` starting inputs.
    target_y
        ``[B]`` target outputs.
    lr, max_steps, tol, lambda_start, lambda_max, lambda_mult, l2_budget
        Same semantics as ``find_counterfactual_mode_a_adaptive``.
    record_trajectory
        If True, each per-subject result's ``trajectory`` is populated with
        ``(λ, residual_at_end)`` tuples per λ doubling.
    """
    X_init = np.asarray(x_init, dtype=np.float64).copy()
    if X_init.ndim != 2:
        raise ValueError(
            f"x_init must be 2D [B, n_features]; got shape {X_init.shape}"
        )
    target = np.asarray(target_y, dtype=np.float64).reshape(-1)
    B, n_features = X_init.shape
    if target.shape[0] != B:
        raise ValueError(
            f"target_y batch size {target.shape[0]} != x_init batch size {B}"
        )

    # P1.4: single evaluation at x_init reused across all λ doublings.
    y_init, g_init = f_and_grad_batch(X_init)
    y_init = np.asarray(y_init, dtype=np.float64).reshape(-1)
    g_init = np.asarray(g_init, dtype=np.float64)
    if y_init.shape != (B,):
        raise ValueError(f"f_and_grad_batch returned y of shape {y_init.shape}; expected ({B},)")
    if g_init.shape != (B, n_features):
        raise ValueError(
            f"f_and_grad_batch returned g of shape {g_init.shape}; "
            f"expected ({B}, {n_features})"
        )

    # Per-subject best-so-far state.
    best_x = X_init.copy()
    best_y = y_init.copy()
    best_n_steps = np.zeros(B, dtype=np.int64)
    best_lambda = np.full(B, lambda_start, dtype=np.float64)
    # converged_mask[i]=True ⇒ subject i is frozen (ragged stop).
    converged_mask = np.abs(y_init - target) <= tol
    # If a subject starts already converged, lock its trajectory in.
    final_y = y_init.copy()
    final_x = X_init.copy()
    final_n_steps = np.zeros(B, dtype=np.int64)
    # Per-subject trajectory (list of (lam, residual_at_end_of_inner_loop)).
    trajectories: list[list[tuple[float, float]]] = [[] for _ in range(B)]
    # Per-subject final λ attempted in the adaptive search.
    lam_attempted = np.full(B, lambda_start, dtype=np.float64)

    # Pre-record subjects that began already at-target.
    for i in range(B):
        if converged_mask[i]:
            best_x[i] = X_init[i]
            best_y[i] = y_init[i]
            best_n_steps[i] = 0
            final_y[i] = y_init[i]
            final_x[i] = X_init[i]
            final_n_steps[i] = 0

    lam = lambda_start
    while True:
        # Reset x for non-converged subjects; converged ones keep their final x.
        x = np.where(converged_mask[:, None], final_x, X_init.copy())
        y_curr = np.where(converged_mask, final_y, y_init.copy())
        # Cached g_init is at x_init; for already-converged subjects we don't
        # touch them, so their g is irrelevant. For non-converged it's the
        # gradient at the reset point (= x_init).
        g_at_x = g_init.copy()
        n_steps = np.zeros(B, dtype=np.int64)
        active = ~converged_mask  # subjects to update this λ
        if not active.any():
            # All subjects already converged → done.
            break

        for step in range(max_steps):
            # Update step counter for active subjects.
            n_steps[active] = step + 1
            residual = y_curr - target  # [B]
            # Newly-converged this step?
            newly_done = active & (np.abs(residual) <= tol)
            if newly_done.any():
                # Freeze them: record final state, mark converged, drop from active.
                converged_mask = converged_mask | newly_done
                for i in np.where(newly_done)[0]:
                    final_x[i] = x[i]
                    final_y[i] = y_curr[i]
                    final_n_steps[i] = step + 1
                    # best-so-far is at most as far as final.
                    if abs(y_curr[i] - target[i]) < abs(best_y[i] - target[i]):
                        best_x[i] = x[i]
                        best_y[i] = y_curr[i]
                        best_n_steps[i] = step + 1
                        best_lambda[i] = lam
                active = ~converged_mask
                if not active.any():
                    break

            # Compute gradient updates for active subjects.
            # grad_L_i = 2*(x_i - x_init_i) + 2*λ*(y_i - target_i)*g_i
            grad_L = 2.0 * (x - X_init) + 2.0 * lam * residual[:, None] * g_at_x
            # Per-subject unit-norm clipping: each row independent.
            row_norms = np.linalg.norm(grad_L, axis=1)  # [B]
            # Avoid division by zero on rows with grad_L == 0 (e.g. frozen).
            safe_norms = np.where(row_norms > 0.0, row_norms, 1.0)
            scale = np.where(row_norms > 1.0, 1.0 / safe_norms, 1.0)
            grad_L = grad_L * scale[:, None]
            # Apply update only to active subjects.
            mask = active[:, None]
            x = np.where(mask, x - lr * grad_L, x)

            if l2_budget is not None:
                delta = x - X_init
                d_norms = np.linalg.norm(delta, axis=1)  # [B]
                # Project rows whose perturbation exceeds budget AND active.
                over = (d_norms > l2_budget) & active
                if over.any():
                    proj_scale = np.where(over, l2_budget / np.maximum(d_norms, 1e-30), 1.0)
                    x = X_init + delta * proj_scale[:, None]

            # Re-evaluate model on the (potentially partially-frozen) batch.
            # Frozen rows pass through unchanged in x; their y/g are irrelevant
            # for active updates but we keep them current for clarity.
            y_new, g_new = f_and_grad_batch(x)
            y_curr = np.asarray(y_new, dtype=np.float64).reshape(-1)
            g_at_x = np.asarray(g_new, dtype=np.float64)

        # End of inner loop for this λ.
        # For still-active subjects: record trajectory entry, update best-so-far.
        if record_trajectory:
            for i in range(B):
                if not converged_mask[i] or len(trajectories[i]) > 0 or active[i]:
                    # record for any subject that participated this λ
                    trajectories[i].append((float(lam), float(y_curr[i] - target[i])))

        # Best-so-far update for non-converged subjects whose closest attempt
        # at this λ improved on prior λ values.
        for i in np.where(active)[0]:
            if abs(y_curr[i] - target[i]) < abs(best_y[i] - target[i]):
                best_x[i] = x[i]
                best_y[i] = y_curr[i]
                best_n_steps[i] = n_steps[i]
                best_lambda[i] = lam

        # Update lam_attempted for ACTIVE subjects (those still searching).
        for i in np.where(active)[0]:
            lam_attempted[i] = lam

        # All converged → done.
        if converged_mask.all():
            break

        # λ doubling termination check.
        if lam >= lambda_max:
            break
        lam = min(lam * lambda_mult, lambda_max)

    # Assemble per-subject results.
    results: list[CounterfactualResult] = []
    for i in range(B):
        success_i = bool(converged_mask[i])
        if success_i:
            x_cf_i = final_x[i]
            y_cf_i = float(final_y[i])
            n_used = int(final_n_steps[i])
            lam_best_i = float(best_lambda[i]) if best_lambda[i] > 0 else float(lambda_start)
            # If subject was at-target from the start, lambda_best = lambda_start;
            # set lam_attempted to at least that.
            lam_max_i = float(max(lam_attempted[i], lam_best_i))
        else:
            x_cf_i = best_x[i]
            y_cf_i = float(best_y[i])
            n_used = int(best_n_steps[i])
            lam_best_i = float(best_lambda[i])
            lam_max_i = float(lam_attempted[i])
        gap_i = abs(y_cf_i - float(target[i]))
        results.append(
            CounterfactualResult(
                x_init=X_init[i],
                x_cf=x_cf_i,
                y_init=float(y_init[i]),
                y_cf=y_cf_i,
                target_y=float(target[i]),
                success=success_i,
                n_steps_used=n_used,
                l2_distance=float(np.linalg.norm(x_cf_i - X_init[i])),
                lambda_best=lam_best_i,
                lambda_max_attempted=lam_max_i,
                gap=float(gap_i),
                trajectory=trajectories[i] if record_trajectory else [],
            )
        )
    return results
