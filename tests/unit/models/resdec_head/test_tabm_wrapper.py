import pytest
import torch
import torch.nn as nn
from src.models.resdec_head.tabm_wrapper import TabMWrapper


def test_tabm_wrapper_shape():
    sub = nn.Linear(32, 32)
    tabm = TabMWrapper(submodule=sub, d_in=32, k=8)
    x = torch.randn(4, 32)
    mean, std = tabm(x)
    assert mean.shape == (4, 32)
    assert std.shape == (4, 32)


def test_tabm_members_diverge():
    """Different k members produce different intermediate outputs."""
    sub = nn.Linear(16, 16)
    tabm = TabMWrapper(submodule=sub, d_in=16, k=4)
    # Force members to be clearly distinct via sufficient per-member scaling
    with torch.no_grad():
        tabm.s.copy_(torch.randn_like(tabm.s) * 0.5 + 1.0)
        tabm.r.copy_(torch.randn_like(tabm.r) * 0.5 + 1.0)
    x = torch.randn(2, 16)
    # std over members should be non-trivial
    _, std = tabm(x)
    assert std.abs().max() > 1e-3


def test_tabm_gradient_flow():
    sub = nn.Linear(8, 8)
    tabm = TabMWrapper(submodule=sub, d_in=8, k=3)
    x = torch.randn(2, 8, requires_grad=True)
    mean, _ = tabm(x)
    mean.sum().backward()
    assert x.grad is not None
    assert tabm.s.grad is not None
    assert tabm.r.grad is not None


def test_tabm_wrapper_output_mismatch_raises():
    """Submodule whose output dim != d_out raises a clear RuntimeError."""
    sub = nn.Linear(16, 8)  # 16 in, 8 out
    tabm = TabMWrapper(submodule=sub, d_in=16, d_out=16, k=4)  # wrong d_out
    x = torch.randn(2, 16)
    with pytest.raises(RuntimeError, match="expected submodule output dim 16"):
        tabm(x)


def test_tabm_wrapper_asymmetric_dims():
    """Explicit d_in != d_out works when submodule is compatible."""
    sub = nn.Linear(16, 8)
    tabm = TabMWrapper(submodule=sub, d_in=16, d_out=8, k=4)
    x = torch.randn(2, 16)
    mean, std = tabm(x)
    assert mean.shape == (2, 8)
    assert std.shape == (2, 8)
