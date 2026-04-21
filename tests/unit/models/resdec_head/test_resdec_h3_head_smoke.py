import pytest
import torch
from src.models.resdec_head.resdec_h3_head import ResDecH3Head


def test_head_forward_shape_and_keys():
    head = ResDecH3Head(d_subject=64, d_metadata=8)
    z = torch.randn(4, 64)  # encoder subject embedding
    m = torch.randn(4, 8)   # metadata vector
    out = head(z, m)
    assert "prediction" in out
    assert "latent_1" in out
    assert out["prediction"].shape == (4,)
    assert out["latent_1"].shape == (4, 64)


def test_head_gradient_flow():
    head = ResDecH3Head(d_subject=32, d_metadata=8)
    z = torch.randn(6, 32, requires_grad=True)
    m = torch.randn(6, 8, requires_grad=True)
    out = head(z, m)
    out["prediction"].sum().backward()
    # Both inputs and head params should get gradients
    assert z.grad is not None
    assert m.grad is not None
    has_head_grad = any(
        p.grad is not None and p.grad.abs().max() > 0
        for p in head.parameters() if p.requires_grad
    )
    assert has_head_grad


def test_head_near_identity_film_at_init():
    """With FiLM's near-identity init, the head's behavior depends only on
    NPTStage for the first few steps — sanity check it doesn't short-circuit
    metadata completely when gradients kick in."""
    head = ResDecH3Head(d_subject=16, d_metadata=4)
    head.eval()
    z = torch.randn(4, 16)
    m_zero = torch.zeros(4, 4)
    m_rand = torch.randn(4, 4)
    with torch.no_grad():
        out_zero = head(z, m_zero)
        out_rand = head(z, m_rand)
    # Because gamma/beta init is zeros-weight-ones-bias / zeros-weight-zeros-bias,
    # FiLM(z, anything) = z at init, so both outputs should be identical at init.
    assert torch.allclose(out_zero["prediction"], out_rand["prediction"], atol=1e-5)
