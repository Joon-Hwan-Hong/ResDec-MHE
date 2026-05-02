"""Targeted tests for src/training/losses.py.

Defense-in-depth: BetaNLLLoss is the loss for the Bayesian head; the head
guarantees ``std >= 1e-6`` upstream, but the loss must also defend against
zero / negative / non-finite std for standalone callers and the var clamp
at L77 of losses.py.
"""
from __future__ import annotations

import math

import pytest
import torch

from src.training.losses import BetaNLLLoss

def test_beta_nll_zero_std_raises_eagerly():
    """std.min() <= 0 must raise (the guard at L62 of losses.py)."""
    loss_fn = BetaNLLLoss(beta=0.5)
    mean = torch.zeros(8, 1)
    std = torch.zeros(8, 1)  # exactly zero — caught by the L62 guard
    target = torch.randn(8, 1) * 0.5
    with pytest.raises(ValueError, match="std must be strictly positive"):
        loss_fn(mean, std, target)

def test_beta_nll_negative_std_raises_eagerly():
    """A negative std value must be rejected by the L62 guard."""
    loss_fn = BetaNLLLoss(beta=0.5)
    mean = torch.zeros(2, 1)
    std = torch.tensor([[0.5], [-0.1]])
    target = torch.zeros(2, 1)
    with pytest.raises(ValueError, match="std must be strictly positive"):
        loss_fn(mean, std, target)

def test_beta_nll_nonfinite_std_raises_eagerly():
    """NaN or Inf std must be rejected by the L62 guard."""
    loss_fn = BetaNLLLoss(beta=0.5)
    mean = torch.zeros(2, 1)
    std = torch.tensor([[0.5], [float("inf")]])
    target = torch.zeros(2, 1)
    with pytest.raises(ValueError, match="std must be strictly positive"):
        loss_fn(mean, std, target)

def test_beta_nll_tiny_positive_std_does_not_nan():
    """A tiny positive std passes the L62 guard; the L77 var clamp must
    keep the loss finite (defense-in-depth for the upstream 1e-6 floor)."""
    loss_fn = BetaNLLLoss(beta=0.5)
    mean = torch.zeros(4, 1)
    std = torch.full((4, 1), 1e-12)  # well below 1e-6 design floor
    target = torch.full((4, 1), 0.1)
    loss = loss_fn(mean, std, target)
    assert torch.isfinite(loss), f"loss should be finite, got {loss}"

def test_beta_nll_zero_beta_reduces_to_log_var_form():
    """β=0 → β-NLL is 0.5 * log(var) + 0.5 * (y-mu)^2 / var (standard NLL minus 0.5*log(2π))."""
    loss_fn = BetaNLLLoss(beta=0.0)
    mean = torch.tensor([[0.0]])
    std = torch.tensor([[1.0]])
    target = torch.tensor([[0.5]])
    loss = loss_fn(mean, std, target)
    # The implementation drops the constant 0.5 * log(2π) term.
    expected = 0.5 * math.log(1.0) + 0.5 * 0.25 / 1.0
    assert math.isclose(loss.item(), expected, rel_tol=1e-5, abs_tol=1e-5)
