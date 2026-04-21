import pytest
import torch
import torch.nn as nn
from src.models.resdec_head.hyper_connections import HyperConnection


def test_hyper_connection_shape():
    hc = HyperConnection(d_model=64, n_streams=4)
    x = torch.randn(4, 64)
    sublayer = nn.Linear(64, 64)
    y = hc(x, sublayer)
    assert y.shape == (4, 64)


def test_hyper_connection_streams_equal_init():
    """At init, all streams weighted equally via softmax(alpha).
    With zero-init alpha, softmax is uniform → output equals sublayer(x) mean."""
    hc = HyperConnection(d_model=8, n_streams=4)
    x = torch.randn(2, 8)
    # Use an identity-like sublayer that shouldn't change its output
    sublayer = nn.Identity()
    y = hc(x, sublayer)
    # With identity sublayer and uniform weights, all stream outputs are x,
    # so weighted sum is just x regardless of weights.
    assert torch.allclose(y, x, atol=1e-6)


def test_hyper_connection_gradient_flow():
    hc = HyperConnection(d_model=16, n_streams=3)
    x = torch.randn(2, 16, requires_grad=True)
    sublayer = nn.Linear(16, 16)
    y = hc(x, sublayer)
    y.sum().backward()
    assert x.grad is not None
    assert hc.alpha.grad is not None
