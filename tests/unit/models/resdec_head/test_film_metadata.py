import pytest
import torch
from src.models.resdec_head.film_metadata import FiLMMetadata

def test_film_metadata_shape():
    film = FiLMMetadata(d_subject=64, d_metadata=8)
    z = torch.randn(4, 64)
    m = torch.randn(4, 8)
    z_cond = film(z, m)
    assert z_cond.shape == (4, 64)

def test_film_metadata_near_identity_at_init():
    """With near-identity init (γ≈1, β≈0), output ≈ z regardless of m."""
    film = FiLMMetadata(d_subject=64, d_metadata=8)
    film.eval()
    z = torch.randn(4, 64)
    m = torch.randn(4, 8)
    z_cond = film(z, m)
    # At init γ=1, β=0, so z_cond should equal z EXACTLY before any training
    assert torch.allclose(z_cond, z, atol=1e-6)

def test_film_metadata_gradient_flow():
    film = FiLMMetadata(d_subject=64, d_metadata=8)
    z = torch.randn(4, 64, requires_grad=True)
    m = torch.randn(4, 8, requires_grad=True)
    z_cond = film(z, m)
    loss = z_cond.sum()
    loss.backward()
    assert z.grad is not None
    assert m.grad is not None
    # Gamma/beta projection weights should also get gradients
    assert film.gamma_proj.weight.grad is not None
    assert film.beta_proj.weight.grad is not None
