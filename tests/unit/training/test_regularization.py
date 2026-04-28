"""Skeleton tests for attention regularization functions in
``src/training/regularization.py``.

ALL TESTS ARE CURRENTLY SKIPPED (``pytest.mark.skip``). The four
functions under test
(``attention_entropy_bonus``, ``attention_kl_to_uniform``,
``attention_coverage_penalty``, ``attention_top1_cap``)
have ``raise NotImplementedError`` bodies; until implementation is
authorized per the design doc at
``docs/plans/2026-04-28-encoder-attention-regularization-design.md``,
these tests serve as a contract / specification only.

Each test names the property it will verify. The skip-reason strings
encode the exact specification the implementation must satisfy.
"""
from __future__ import annotations

import math
from typing import Callable

import pytest
import torch

from src.training.regularization import (
    LOG_EPS,
    attention_coverage_penalty,
    attention_entropy_bonus,
    attention_kl_to_uniform,
    attention_top1_cap,
)


# ---------------------------------------------------------------------------
# Test fixtures: synthetic attention tensors with known entropy / max props
# ---------------------------------------------------------------------------

@pytest.fixture
def uniform_attention() -> torch.Tensor:
    """[2, 4, 31] all entries = 1/31 — max entropy."""
    return torch.full((2, 4, 31), 1.0 / 31)


@pytest.fixture
def concentrated_attention() -> torch.Tensor:
    """[2, 4, 31] all mass on first CT — zero entropy."""
    a = torch.zeros((2, 4, 31))
    a[..., 0] = 1.0
    return a


@pytest.fixture
def realistic_attention() -> torch.Tensor:
    """[2, 4, 31] approximation of observed attention. Head 1 concentrates
    on CT index 0 (Splatter proxy) at 0.123, rest spread evenly."""
    a = torch.full((2, 4, 31), (1.0 - 0.123) / 30)
    a[:, 1, 0] = 0.123
    return a


SKIP_REASON = "Implementation gated on user approval (design doc §12-13)"


# ---------------------------------------------------------------------------
# attention_entropy_bonus
# ---------------------------------------------------------------------------

def test_entropy_bonus_uniform_yields_max_negative(uniform_attention):
    """Entropy of uniform = log C; bonus = -weight · log C (most negative)."""
    out = attention_entropy_bonus(uniform_attention, weight=1.0)
    expected = -math.log(31)
    assert torch.allclose(out, torch.tensor(expected), atol=1e-5)


def test_entropy_bonus_concentrated_is_zero(concentrated_attention):
    """Entropy of one-hot = 0; bonus = -weight · 0 = 0."""
    out = attention_entropy_bonus(concentrated_attention, weight=1.0)
    assert torch.allclose(out, torch.tensor(0.0), atol=1e-5)


def test_entropy_bonus_zero_weight_is_zero(realistic_attention):
    """weight=0 always returns 0, regardless of input distribution."""
    out = attention_entropy_bonus(realistic_attention, weight=0.0)
    assert torch.allclose(out, torch.tensor(0.0), atol=1e-7)


def test_entropy_bonus_returns_scalar(realistic_attention):
    """Output is a 0-dim tensor (scalar) — addable to loss directly."""
    out = attention_entropy_bonus(realistic_attention, weight=1e-2)
    assert out.dim() == 0


def test_entropy_bonus_differentiable(realistic_attention):
    """Gradient flows back through the entropy computation."""
    a = realistic_attention.clone().requires_grad_(True)
    out = attention_entropy_bonus(a, weight=1.0)
    out.backward()
    assert a.grad is not None
    assert a.grad.shape == a.shape


def test_entropy_bonus_rejects_2d_input():
    """Non-3D input raises ValueError."""
    with pytest.raises(ValueError):
        attention_entropy_bonus(torch.full((4, 31), 1.0 / 31), weight=1e-2)


def test_entropy_bonus_rejects_negative_weight(realistic_attention):
    """Negative weight inverts optimization direction → reject."""
    with pytest.raises(ValueError):
        attention_entropy_bonus(realistic_attention, weight=-1e-2)


# ---------------------------------------------------------------------------
# attention_kl_to_uniform
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason=SKIP_REASON)
def test_kl_to_uniform_uniform_is_zero(uniform_attention):
    """KL(uniform || uniform) = 0."""
    out = attention_kl_to_uniform(uniform_attention, weight=1.0)
    assert torch.allclose(out, torch.tensor(0.0), atol=1e-5)


@pytest.mark.skip(reason=SKIP_REASON)
def test_kl_to_uniform_concentrated_equals_log_C(concentrated_attention):
    """KL(one-hot || uniform) = log C."""
    out = attention_kl_to_uniform(concentrated_attention, weight=1.0)
    expected = math.log(31)
    assert torch.allclose(out, torch.tensor(expected), atol=1e-3)


@pytest.mark.skip(reason=SKIP_REASON)
def test_kl_to_uniform_equals_neg_entropy_plus_logC(realistic_attention):
    """KL(p || U) = -H(p) + log C; verifies algebraic identity (entropy_bonus implemented)."""
    kl = attention_kl_to_uniform(realistic_attention, weight=1.0)
    bonus = attention_entropy_bonus(realistic_attention, weight=1.0)
    log_c = math.log(31)
    # bonus = -H, so KL = bonus + log_c
    assert torch.allclose(kl, bonus + log_c, atol=1e-5)


@pytest.mark.skip(reason=SKIP_REASON)
def test_kl_to_uniform_non_negative(realistic_attention):
    """KL is always >= 0."""
    out = attention_kl_to_uniform(realistic_attention, weight=1.0)
    assert out.item() >= 0.0


@pytest.mark.skip(reason=SKIP_REASON)
def test_kl_to_uniform_rejects_2d_input():
    with pytest.raises(ValueError):
        attention_kl_to_uniform(torch.full((4, 31), 1.0 / 31), weight=1e-2)


@pytest.mark.skip(reason=SKIP_REASON)
def test_kl_to_uniform_rejects_negative_weight(realistic_attention):
    with pytest.raises(ValueError):
        attention_kl_to_uniform(realistic_attention, weight=-1e-2)


# ---------------------------------------------------------------------------
# attention_coverage_penalty
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason=SKIP_REASON)
def test_coverage_penalty_uniform_above_floor(uniform_attention):
    """When all entries >= floor (e.g., floor = 0.5/C, uniform = 1/C), penalty = 0."""
    out = attention_coverage_penalty(uniform_attention, floor=0.5 / 31, weight=1.0)
    assert torch.allclose(out, torch.tensor(0.0), atol=1e-7)


@pytest.mark.skip(reason=SKIP_REASON)
def test_coverage_penalty_concentrated_attention(concentrated_attention):
    """One-hot at CT=0: 30 CTs at 0 each contribute floor - 0 = floor → sum = 30 · floor."""
    floor = 0.5 / 31
    out = attention_coverage_penalty(concentrated_attention, floor=floor, weight=1.0)
    expected = 30 * floor
    assert torch.allclose(out, torch.tensor(expected), atol=1e-6)


@pytest.mark.skip(reason=SKIP_REASON)
def test_coverage_penalty_zero_weight_is_zero(concentrated_attention):
    out = attention_coverage_penalty(concentrated_attention, floor=0.5 / 31, weight=0.0)
    assert torch.allclose(out, torch.tensor(0.0), atol=1e-7)


@pytest.mark.skip(reason=SKIP_REASON)
def test_coverage_penalty_rejects_floor_too_high():
    """floor > 1/C is unsatisfiable on the simplex."""
    a = torch.full((2, 4, 31), 1.0 / 31)
    with pytest.raises(ValueError):
        attention_coverage_penalty(a, floor=2.0 / 31, weight=1.0)


@pytest.mark.skip(reason=SKIP_REASON)
def test_coverage_penalty_rejects_floor_zero_or_negative():
    a = torch.full((2, 4, 31), 1.0 / 31)
    with pytest.raises(ValueError):
        attention_coverage_penalty(a, floor=0.0, weight=1.0)
    with pytest.raises(ValueError):
        attention_coverage_penalty(a, floor=-0.01, weight=1.0)


@pytest.mark.skip(reason=SKIP_REASON)
def test_coverage_penalty_rejects_negative_weight(realistic_attention):
    with pytest.raises(ValueError):
        attention_coverage_penalty(realistic_attention, floor=0.01, weight=-1e-2)


@pytest.mark.skip(reason=SKIP_REASON)
def test_coverage_penalty_rejects_2d_input():
    with pytest.raises(ValueError):
        attention_coverage_penalty(torch.full((4, 31), 1.0 / 31), floor=0.01, weight=1e-2)


@pytest.mark.skip(reason=SKIP_REASON)
def test_coverage_penalty_differentiable(concentrated_attention):
    a = concentrated_attention.clone().requires_grad_(True)
    out = attention_coverage_penalty(a, floor=0.5 / 31, weight=1.0)
    out.backward()
    assert a.grad is not None
    assert a.grad.shape == a.shape


# ---------------------------------------------------------------------------
# attention_top1_cap
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason=SKIP_REASON)
def test_top1_cap_uniform_below_cap(uniform_attention):
    """Uniform top-1 = 1/31 ≈ 0.032 << cap = 0.10 → penalty = 0."""
    out = attention_top1_cap(uniform_attention, cap=0.10, weight=1.0)
    assert torch.allclose(out, torch.tensor(0.0), atol=1e-7)


@pytest.mark.skip(reason=SKIP_REASON)
def test_top1_cap_concentrated_equals_excess(concentrated_attention):
    """One-hot top-1 = 1.0; excess = 1.0 - 0.10 = 0.90 → penalty = 0.90."""
    out = attention_top1_cap(concentrated_attention, cap=0.10, weight=1.0)
    expected = 0.90
    assert torch.allclose(out, torch.tensor(expected), atol=1e-5)


@pytest.mark.skip(reason=SKIP_REASON)
def test_top1_cap_realistic_partial_excess(realistic_attention):
    """Head 1 top-1 = 0.123 > cap=0.10; other heads top-1 = (1-0.123)/30 ≈ 0.029 < cap.
    Excess only from head 1 in each of 2 batch entries: (0.123 - 0.10) = 0.023.
    Mean over (B=2, H=4): 2 · 0.023 / 8 = 0.00575."""
    out = attention_top1_cap(realistic_attention, cap=0.10, weight=1.0)
    expected = (0.123 - 0.10) * 2 / 8
    assert torch.allclose(out, torch.tensor(expected), atol=1e-5)


@pytest.mark.skip(reason=SKIP_REASON)
def test_top1_cap_zero_weight_is_zero(concentrated_attention):
    out = attention_top1_cap(concentrated_attention, cap=0.10, weight=0.0)
    assert torch.allclose(out, torch.tensor(0.0), atol=1e-7)


@pytest.mark.skip(reason=SKIP_REASON)
def test_top1_cap_rejects_cap_below_uniform():
    """cap < 1/C is unsatisfiable (max attention >= 1/C by pigeonhole)."""
    a = torch.full((2, 4, 31), 1.0 / 31)
    with pytest.raises(ValueError):
        attention_top1_cap(a, cap=0.5 / 31, weight=1.0)


@pytest.mark.skip(reason=SKIP_REASON)
def test_top1_cap_rejects_cap_at_or_above_one():
    """cap >= 1 is vacuous (max attention <= 1 always)."""
    a = torch.full((2, 4, 31), 1.0 / 31)
    with pytest.raises(ValueError):
        attention_top1_cap(a, cap=1.0, weight=1.0)
    with pytest.raises(ValueError):
        attention_top1_cap(a, cap=1.5, weight=1.0)


@pytest.mark.skip(reason=SKIP_REASON)
def test_top1_cap_rejects_negative_weight(realistic_attention):
    with pytest.raises(ValueError):
        attention_top1_cap(realistic_attention, cap=0.10, weight=-1e-2)


@pytest.mark.skip(reason=SKIP_REASON)
def test_top1_cap_rejects_2d_input():
    with pytest.raises(ValueError):
        attention_top1_cap(torch.full((4, 31), 1.0 / 31), cap=0.10, weight=1e-2)


@pytest.mark.skip(reason=SKIP_REASON)
def test_top1_cap_differentiable(concentrated_attention):
    a = concentrated_attention.clone().requires_grad_(True)
    out = attention_top1_cap(a, cap=0.10, weight=1.0)
    out.backward()
    assert a.grad is not None
    assert a.grad.shape == a.shape


# ---------------------------------------------------------------------------
# Sanity: the LOG_EPS constant is exported and is small but positive
# ---------------------------------------------------------------------------

def test_log_eps_constant_is_positive_and_small():
    """LOG_EPS must be in (0, 1e-9] — large enough to prevent log(0), small
    enough not to bias any non-degenerate distribution. Bound tightened from
    1e-6 to 1e-9 (M17): for float32 entropy, log(1e-12) = -27.6 nats, so
    eps≪1e-9 is needed to avoid skewing the entropy of small attention values."""
    assert 0.0 < LOG_EPS <= 1e-9


# Sanity: helper to confirm the THREE STILL-SKELETON functions raise
# NotImplementedError. ``attention_entropy_bonus`` (Scheme A) is now
# implemented (2026-04-28) and was removed from this list.
@pytest.mark.parametrize(
    "fn,kwargs",
    [
        (attention_kl_to_uniform, {"weight": 1e-2}),
        (attention_coverage_penalty, {"floor": 0.01, "weight": 1e-2}),
        (attention_top1_cap, {"cap": 0.10, "weight": 1e-2}),
    ],
)
def test_skeleton_functions_raise_not_implemented(
    fn: Callable, kwargs: dict,
) -> None:
    """Confirms skeleton state for kl/coverage/top1 functions.
    Once implementation is authorized for any of them, this parametrize
    list shrinks accordingly."""
    a = torch.full((2, 4, 31), 1.0 / 31)
    with pytest.raises(NotImplementedError):
        fn(a, **kwargs)
