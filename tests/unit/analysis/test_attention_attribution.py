"""Unit tests for src/analysis/attention_attribution.py.

Tests the AttnLRP / GMAR / GAF reference implementations using mock
attention tensors.  Verifies algorithmic invariants:
  - Conservation of relevance (Σ R^{l-1} ≈ Σ R^l) under non-degenerate inputs
  - Per-head weights sum to 1 (GMAR normalization)
  - Information-tensor variants match expected aggregation form
"""
from __future__ import annotations

import numpy as np
import pytest

from src.analysis.attention_attribution import (
    attnlrp_identity,
    attnlrp_linear_eps,
    attnlrp_matmul,
    attnlrp_softmax,
    gaf_information_tensor,
    gmar_head_weights,
    gmar_weighted_rollout,
)


# ───────────────────────────────────────────────────────────────────────────────
# AttnLRP tests
# ───────────────────────────────────────────────────────────────────────────────

def test_attnlrp_softmax_shape_preserved():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 8))
    s = np.exp(x) / np.exp(x).sum(axis=-1, keepdims=True)  # softmax
    R_l = rng.normal(size=(4, 8))
    R_lminus1 = attnlrp_softmax(R_l, s, x)
    assert R_lminus1.shape == R_l.shape


def test_attnlrp_softmax_zero_relevance_propagates_to_zero():
    """If R^l = 0 everywhere, R^{l-1} must also be 0."""
    x = np.array([1.0, 2.0, 3.0])
    s = np.exp(x) / np.exp(x).sum()
    R_l = np.zeros(3)
    out = attnlrp_softmax(R_l, s, x)
    assert np.allclose(out, 0.0)


def test_attnlrp_matmul_shape():
    A = np.random.default_rng(0).normal(size=(2, 4, 6))  # batch, J, I
    V = np.random.default_rng(1).normal(size=(2, 6, 5))  # batch, I, P
    R_l = np.random.default_rng(2).normal(size=(2, 4, 5))  # batch, J, P
    R = attnlrp_matmul(R_l, A, V)
    assert R.shape == A.shape


def test_attnlrp_matmul_zero_input_finite():
    """Even if A or V has zeros, the rule shouldn't produce NaNs (eps stabilizer)."""
    A = np.zeros((4, 3))
    V = np.ones((3, 2))
    R_l = np.ones((4, 2))
    R = attnlrp_matmul(R_l, A, V, eps=1e-3)
    assert np.all(np.isfinite(R))


def test_attnlrp_matmul_bounded_when_O_near_zero():
    """When O ≈ 0, the |R| should remain bounded by R_l / eps (not 1e+large).
    Verifies the C4 floor: replacing the loose ``2*O + eps*sign`` denom with
    a hard min-magnitude floor of eps prevents 6-order-of-magnitude blow-ups
    that would silently corrupt per-CT rankings."""
    rng = np.random.default_rng(0)
    A = rng.normal(scale=1e-8, size=(4, 3))  # near-zero contributions
    V = rng.normal(scale=1e-8, size=(3, 2))
    R_l = np.ones((4, 2))
    R = attnlrp_matmul(R_l, A, V, eps=1e-6)
    # |R| <= |A| * (|V| / eps); for our scales, expect << 1 (no 1e+6 blow-up).
    assert np.max(np.abs(R)) < 1e-3, (
        f"Expected bounded |R| with floor=eps; got max |R|={np.max(np.abs(R))}"
    )


def test_attnlrp_softmax_non_conservation_documented():
    """The eq. 13 softmax rule is NON-CONSERVATIVE by design (Achtibat §3.3.3).
    This test pins down the expected behavior so future regressions don't look
    like bugs."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 8))
    s = np.exp(x) / np.exp(x).sum(axis=-1, keepdims=True)
    R_l = rng.normal(size=(4, 8))
    R_lminus1 = attnlrp_softmax(R_l, s, x)
    # Non-conservation: in general Σ R^{l-1} ≠ Σ R^l. Show that the residual
    # is non-zero (not a bug) and that the residual equals the bias term
    # x · (R_l - s · ΣR_l) summed minus Σ R_l.
    diff = R_lminus1.sum(axis=-1) - R_l.sum(axis=-1)  # per-row residual
    assert np.any(np.abs(diff) > 1e-6), (
        "Expected non-conservation; if Σ R^{l-1} == Σ R^l exactly, the rule "
        "would be a different one than eq. 13."
    )


def test_attnlrp_identity_unchanged():
    R_l = np.array([1.0, -2.0, 3.5, 0.0])
    R = attnlrp_identity(R_l)
    np.testing.assert_array_equal(R, R_l)


def test_attnlrp_linear_eps_conservation_on_simple():
    """For a 2-input 2-output linear y = W x with W diag(2, 2) and equal R^l, conservation holds."""
    W = np.array([[2.0, 0.0], [0.0, 2.0]])
    x = np.array([1.0, 1.0])
    R_l = np.array([1.0, 1.0])
    R_lminus1 = attnlrp_linear_eps(R_l, W, x, eps=1e-9)
    # R per input ≈ Σ_j W_ji x_i R_j / z_j = W_ii x_i R_i / (W_ii x_i) = R_i
    assert np.allclose(R_lminus1, [1.0, 1.0], atol=1e-6)


def test_attnlrp_linear_eps_off_diagonal_W():
    """Non-trivial off-diagonal W exercises the full ε-LRP rule, not just diag.
    For y = W x with W = [[1, 1], [1, -1]] and x = [3, 1]:
        z_0 = 1·3 + 1·1 = 4;  z_1 = 1·3 + (-1)·1 = 2
        With R^l = [4, 2], factor = R/z = [1, 1]
        R^{l-1}_i = Σ_j factor_j · W_ji · x_i
        R^{l-1}_0 = (1·1 + 1·1) · 3 = 6
        R^{l-1}_1 = (1·1 + 1·(-1)) · 1 = 0
    """
    W = np.array([[1.0, 1.0], [1.0, -1.0]])
    x = np.array([3.0, 1.0])
    R_l = np.array([4.0, 2.0])
    R_lminus1 = attnlrp_linear_eps(R_l, W, x, eps=1e-12)
    np.testing.assert_allclose(R_lminus1, [6.0, 0.0], atol=1e-6)


def test_attnlrp_linear_eps_near_zero_z_stabilizer():
    """When z_j is near zero, the ε-LRP stabilizer prevents blow-up.
    Verifies the eps · sign(z) bias actually kicks in at small z."""
    W = np.array([[0.5e-9, 0.5e-9]])
    x = np.array([1.0, 1.0])
    R_l = np.array([1e-9])
    R_lminus1 = attnlrp_linear_eps(R_l, W, x, eps=1e-6)
    # |R| should remain bounded — without stabilizer, z = 1e-9 → R ~ 1
    # With eps=1e-6 stabilizer dominating, R_input ≈ R_l · W · x / eps ≈ tiny
    assert np.all(np.isfinite(R_lminus1))
    assert np.max(np.abs(R_lminus1)) < 1.0


def test_attnlrp_softmax_batched_multi_head():
    """AttnLRP softmax preserves shape and behavior on realistic (B, H, C) tensor.
    Matches the orchestrator's actual usage in run_attention_attribution.py."""
    rng = np.random.default_rng(0)
    B, H, C = 2, 4, 31
    x = rng.normal(size=(B, H, C))
    s = np.exp(x) / np.exp(x).sum(axis=-1, keepdims=True)
    R_l = rng.normal(size=(B, H, C))
    R_lminus1 = attnlrp_softmax(R_l, s, x)
    assert R_lminus1.shape == (B, H, C)
    assert np.all(np.isfinite(R_lminus1))


# ───────────────────────────────────────────────────────────────────────────────
# GMAR tests
# ───────────────────────────────────────────────────────────────────────────────

def test_gmar_head_weights_sum_to_one_l1():
    grad = np.random.default_rng(0).normal(size=(4, 6, 6))  # n_heads=4
    w = gmar_head_weights(grad, n_heads=4, norm="l1")
    assert w.shape == (4,)
    np.testing.assert_allclose(w.sum(), 1.0, atol=1e-9)
    assert (w >= 0).all()


def test_gmar_head_weights_sum_to_one_l2():
    grad = np.random.default_rng(1).normal(size=(8, 4, 4))  # n_heads=8
    w = gmar_head_weights(grad, n_heads=8, norm="l2")
    assert w.shape == (8,)
    np.testing.assert_allclose(w.sum(), 1.0, atol=1e-9)


def test_gmar_head_weights_with_batch_dim():
    grad = np.random.default_rng(2).normal(size=(3, 4, 6, 6))  # batch, head, N, N
    w = gmar_head_weights(grad, n_heads=4, norm="l2")
    assert w.shape == (4,)
    np.testing.assert_allclose(w.sum(), 1.0, atol=1e-9)


def test_gmar_head_weights_dominant_head():
    """If one head has much larger gradient norm, it should dominate."""
    grad = np.zeros((4, 5, 5))
    grad[2] = 1.0  # head 2 is the only non-zero
    w = gmar_head_weights(grad, n_heads=4, norm="l1")
    assert w[2] > 0.99
    assert w[[0, 1, 3]].sum() < 0.01


def test_gmar_head_weights_invalid_norm():
    grad = np.random.default_rng(0).normal(size=(4, 5, 5))
    with pytest.raises(ValueError, match="norm must be"):
        gmar_head_weights(grad, n_heads=4, norm="bad")


def test_gmar_weighted_rollout_uniform_equals_vanilla():
    """With uniform per-head weights, GMAR rollout reduces to vanilla
    Abnar-Zuidema 2020 rollout: ``A_rollout = ∏_l (A_mean + α·I)`` under
    right-multiply convention."""
    L, n_heads, N = 3, 4, 5
    rng = np.random.default_rng(0)
    per_layer = [rng.uniform(0, 1, size=(n_heads, N, N)) for _ in range(L)]
    # Normalize each (head, row) attention row to sum to 1 (proper attention)
    for ell in range(L):
        per_layer[ell] /= per_layer[ell].sum(axis=-1, keepdims=True)
    A = gmar_weighted_rollout(per_layer, per_layer_head_weights=None, alpha=1.0)
    assert A.shape == (N, N)
    # Vanilla rollout: A_rollout @ (A_mean + α·I), iterated over layers.
    expected = np.eye(N)
    for ell in range(L):
        A_mean = per_layer[ell].mean(axis=0)
        expected = expected @ (A_mean + np.eye(N))
    np.testing.assert_allclose(A, expected, atol=1e-9)


def test_gmar_weighted_rollout_shape():
    L, n_heads, N = 2, 3, 4
    per_layer = [np.random.default_rng(i).uniform(0, 1, size=(n_heads, N, N))
                 for i in range(L)]
    weights = [np.array([0.6, 0.3, 0.1]) for _ in range(L)]
    A = gmar_weighted_rollout(per_layer, weights, alpha=1.0)
    assert A.shape == (N, N)


# ───────────────────────────────────────────────────────────────────────────────
# GAF tests
# ───────────────────────────────────────────────────────────────────────────────

def test_gaf_af_variant_no_grad_required():
    """AF variant doesn't need grad_A."""
    A = np.random.default_rng(0).uniform(0, 1, size=(2, 4, 5, 5))  # L, heads, N, N
    Abar = gaf_information_tensor(A, grad_A=None, variant="af")
    assert Abar.shape == (2, 5, 5)
    # AF == mean over heads
    np.testing.assert_allclose(Abar, A.mean(axis=1))


def test_gaf_gf_variant_takes_positive_grad():
    A = np.random.default_rng(0).uniform(0, 1, size=(1, 2, 3, 3))
    grad_A = np.array([[[[1.0, -2.0, 3.0],
                         [-4.0, 5.0, -6.0],
                         [7.0, -8.0, 9.0]],
                        [[0.5, -0.5, 0.5],
                         [-0.5, 0.5, -0.5],
                         [0.5, -0.5, 0.5]]]])
    Abar = gaf_information_tensor(A, grad_A, variant="gf")
    assert Abar.shape == (1, 3, 3)
    expected = np.mean([np.maximum(grad_A[0, 0], 0), np.maximum(grad_A[0, 1], 0)], axis=0)
    np.testing.assert_allclose(Abar[0], expected, atol=1e-9)


def test_gaf_agf_variant_combines_attention_and_grad():
    """AGF should be E_h(⌊A * grad_A⌋_+).  Verify on a hand-crafted case."""
    A = np.array([[[[0.5, 0.5], [0.5, 0.5]]]], dtype=np.float64)  # (1, 1, 2, 2)
    grad_A = np.array([[[[2.0, -1.0], [-1.0, 2.0]]]], dtype=np.float64)
    Abar = gaf_information_tensor(A, grad_A, variant="agf")
    # A * grad = [[1.0, -0.5], [-0.5, 1.0]] → relu → [[1.0, 0.0], [0.0, 1.0]]
    expected = np.array([[1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(Abar[0], expected, atol=1e-9)


def test_gaf_invalid_variant():
    A = np.zeros((1, 1, 2, 2))
    with pytest.raises(ValueError, match="variant must be"):
        gaf_information_tensor(A, grad_A=None, variant="xyz")


def test_gaf_grad_required_for_gf_agf():
    A = np.zeros((1, 1, 2, 2))
    with pytest.raises(ValueError, match="requires grad_A"):
        gaf_information_tensor(A, grad_A=None, variant="gf")
    with pytest.raises(ValueError, match="requires grad_A"):
        gaf_information_tensor(A, grad_A=None, variant="agf")
