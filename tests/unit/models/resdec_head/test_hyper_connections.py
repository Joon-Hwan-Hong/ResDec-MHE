import pytest
import torch
import torch.nn as nn

from src.models.resdec_head.hyper_connections import HyperConnection


def test_hyper_connection_shape():
    """Multi-stream forward preserves [B, N, d] shape."""
    hc = HyperConnection(d_model=64, n_streams=4)
    streams = torch.randn(4, 4, 64)  # [B, N, d]
    sublayer = nn.Linear(64, 64)
    y = hc(streams, sublayer)
    assert y.shape == (4, 4, 64)


def test_hyper_connection_identity_A_zero_sublayer():
    """With A = I (default init), B[m]=0 and a zero sublayer, streams pass through unchanged."""
    hc = HyperConnection(d_model=8, n_streams=4)
    # Zero out B so the sublayer output contribution vanishes; A is already I at init.
    with torch.no_grad():
        hc.B.zero_()
    streams = torch.randn(2, 4, 8)
    # Any sublayer works since its output is zeroed out by B=0.
    y = hc(streams, nn.Identity())
    assert torch.allclose(y, streams, atol=1e-6)


def test_hyper_connection_streams_diverge_under_nontrivial_A():
    """After A-matrix mixing with a non-identity A, streams that start identical
    diverge — i.e. n_streams is NOT a structural no-op."""
    hc = HyperConnection(d_model=8, n_streams=4)
    with torch.no_grad():
        # Set A to a non-identity mixing matrix (row m combines streams unequally).
        hc.A.copy_(torch.tensor([
            [1.0, 0.5, 0.0, 0.0],
            [0.0, 1.0, 0.5, 0.0],
            [0.0, 0.0, 1.0, 0.5],
            [0.5, 0.0, 0.0, 1.0],
        ]))
        # Zero B so the test isolates the stream-mixing behaviour.
        hc.B.zero_()
    # Give streams distinct values so mixing is observable.
    streams = torch.stack([
        torch.ones(2, 8) * 1.0,
        torch.ones(2, 8) * 2.0,
        torch.ones(2, 8) * 3.0,
        torch.ones(2, 8) * 4.0,
    ], dim=1)  # [2, 4, 8]
    y = hc(streams, nn.Identity())
    # The four output streams should not all be identical.
    all_equal = all(
        torch.allclose(y[:, 0, :], y[:, i, :], atol=1e-6) for i in range(1, 4)
    )
    assert not all_equal, "Streams did not diverge under non-identity A mixing"


def test_hyper_connection_gradient_flow():
    """All learnable params (A, B, alpha) receive gradients, and input gradients flow."""
    hc = HyperConnection(d_model=16, n_streams=3)
    streams = torch.randn(2, 3, 16, requires_grad=True)
    sublayer = nn.Linear(16, 16)
    y = hc(streams, sublayer)
    y.sum().backward()
    assert streams.grad is not None
    assert hc.A.grad is not None
    assert hc.B.grad is not None
    assert hc.alpha.grad is not None


def test_hyper_connection_rejects_bad_shape():
    """Passing a 2D tensor (the old signature) should raise a clear error."""
    hc = HyperConnection(d_model=8, n_streams=4)
    bad_input = torch.randn(2, 8)  # [B, d] — the old interface
    with pytest.raises(ValueError, match="\\[B, N, d\\]"):
        hc(bad_input, nn.Identity())


def test_hyper_connection_rejects_stream_count_mismatch():
    """Wrong N dimension should raise a clear error."""
    hc = HyperConnection(d_model=8, n_streams=4)
    bad_streams = torch.randn(2, 3, 8)  # N=3, expected 4
    with pytest.raises(ValueError, match="Stream count mismatch"):
        hc(bad_streams, nn.Identity())
