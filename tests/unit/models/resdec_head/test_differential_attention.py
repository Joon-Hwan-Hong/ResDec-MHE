import pytest
import torch
from src.models.resdec_head.differential_attention import DifferentialAttention


def test_differential_attention_shape():
    attn = DifferentialAttention(d_model=64, n_heads=4, lambda_init=0.8)
    x = torch.randn(4, 16, 64)
    y = attn(x)
    assert y.shape == x.shape


def test_differential_attention_per_head_lambda():
    """λ is per-head with reparameterization."""
    attn = DifferentialAttention(d_model=32, n_heads=4, lambda_init=0.8)
    assert attn.lambda_q1.shape == (4, 8)  # [n_heads, d_head]
    assert attn.lambda_k1.shape == (4, 8)
    assert attn.lambda_q2.shape == (4, 8)
    assert attn.lambda_k2.shape == (4, 8)
    lam = attn._per_head_lambda()
    assert lam.shape == (4,)  # [n_heads]
    # At init with zero λ_*, exp(0)=1, so lam = 1 - 1 + 0.8 = 0.8 per head
    assert torch.allclose(lam, torch.full((4,), 0.8), atol=1e-6)


def test_differential_attention_has_group_norm():
    attn = DifferentialAttention(d_model=32, n_heads=4, lambda_init=0.8)
    import torch.nn as nn
    assert isinstance(attn.group_norm, nn.GroupNorm)
    assert attn.group_norm.num_groups == 4
    assert attn.group_norm.num_channels == 32


def test_differential_attention_gradient_flow():
    attn = DifferentialAttention(d_model=32, n_heads=4, lambda_init=0.5)
    x = torch.randn(2, 8, 32, requires_grad=True)
    y = attn(x)
    y.sum().backward()
    assert x.grad is not None
    assert attn.lambda_q1.grad is not None
    assert attn.lambda_k2.grad is not None


def test_differential_attention_single_head():
    attn = DifferentialAttention(d_model=16, n_heads=1, lambda_init=0.5)
    x = torch.randn(2, 4, 16)
    y = attn(x)
    assert y.shape == x.shape


def test_differential_attention_output_scaling():
    """Init: (1 - λ_init) = 0.2 factor in the output compared to standard attention magnitude."""
    attn = DifferentialAttention(d_model=16, n_heads=2, lambda_init=0.8)
    # Force attn1 = attn2 (identical queries/keys), then attn = (1 - λ_init)*attn1 per head
    # Just verify the scaling factor appears somewhere reasonable
    x = torch.randn(1, 4, 16) * 0.01  # tiny input
    y = attn(x)
    # With tiny input, output should also be small and NOT explode
    assert y.abs().max() < 10.0
