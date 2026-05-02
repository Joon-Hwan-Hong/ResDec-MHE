import pytest
import torch
from src.models.resdec_head.cross_stage_attention import CrossStageAttention

def test_cross_stage_attention_shape():
    csa = CrossStageAttention(d_subject=64, n_heads=4)
    z = torch.randn(4, 64)
    prior1 = torch.randn(4, 64)
    prior2 = torch.randn(4, 64)
    ctx = csa(z, [prior1, prior2])
    assert ctx.shape == (4, 64)

def test_cross_stage_attention_empty_priors_returns_zeros():
    """No prior stages → zero context (stage 1 case)."""
    csa = CrossStageAttention(d_subject=32, n_heads=4)
    z = torch.randn(2, 32)
    ctx = csa(z, [])
    assert ctx.shape == (2, 32)
    assert torch.allclose(ctx, torch.zeros_like(ctx))

def test_cross_stage_attention_single_prior():
    csa = CrossStageAttention(d_subject=16, n_heads=4)
    z = torch.randn(2, 16)
    prior = torch.randn(2, 16)
    ctx = csa(z, [prior])
    assert ctx.shape == (2, 16)

def test_cross_stage_attention_gradient_flow():
    csa = CrossStageAttention(d_subject=16, n_heads=4)
    z = torch.randn(2, 16, requires_grad=True)
    priors = [torch.randn(2, 16, requires_grad=True) for _ in range(2)]
    ctx = csa(z, priors)
    ctx.sum().backward()
    assert z.grad is not None
    for p in priors:
        assert p.grad is not None
