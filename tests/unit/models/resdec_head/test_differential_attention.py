import pytest
import torch
from src.models.resdec_head.differential_attention import DifferentialAttention


def test_differential_attention_shape():
    attn = DifferentialAttention(d_model=64, n_heads=4, lambda_init=0.8)
    x = torch.randn(4, 16, 64)  # [B, seq, d]
    y = attn(x)
    assert y.shape == x.shape


def test_differential_attention_learns_lambda():
    """lambda_param is a Parameter and receives gradient."""
    attn = DifferentialAttention(d_model=32, n_heads=4, lambda_init=0.5)
    x = torch.randn(2, 8, 32, requires_grad=True)
    y = attn(x)
    y.sum().backward()
    assert attn.lambda_param.grad is not None
    assert x.grad is not None


def test_differential_attention_single_head():
    attn = DifferentialAttention(d_model=16, n_heads=1, lambda_init=0.5)
    x = torch.randn(2, 4, 16)
    y = attn(x)
    assert y.shape == x.shape
