"""Tests for attention regularization functions in
``src/training/regularization.py``.

Currently only ``attention_entropy_bonus`` (Scheme A) is implemented.
The skeleton-only functions (``attention_kl_to_uniform``,
``attention_coverage_penalty``, ``attention_top1_cap``) were removed
2026-05-02 per the codebase no-dead-code policy. When those schemes are
authorized per the design doc at
``docs/plans/2026-04-28-encoder-attention-regularization-design.md``,
they will be added back WITH implementation + tests in a single PR.
"""
from __future__ import annotations

import math

import pytest
import torch

from src.training.regularization import (
    LOG_EPS,
    attention_entropy_bonus,
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
# Sanity: the LOG_EPS constant is exported and is small but positive
# ---------------------------------------------------------------------------

def test_log_eps_constant_is_positive_and_small():
    """LOG_EPS must be in (0, 1e-9] — large enough to prevent log(0), small
    enough not to bias any non-degenerate distribution. Bound tightened from
    1e-6 to 1e-9 (M17): for float32 entropy, log(1e-12) = -27.6 nats, so
    eps≪1e-9 is needed to avoid skewing the entropy of small attention values."""
    assert 0.0 < LOG_EPS <= 1e-9
