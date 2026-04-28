"""Modern attention-attribution methods for transformer interpretability.

Three algorithms verified from primary PDF sources (2024-2026):

1. **AttnLRP** (Achtibat et al., ICML 2024 + Nature Machine Intelligence 2023).
   Layer-wise Relevance Propagation extended to non-linear attention.  Key rules:
     - Softmax (eq. 13):    R^{l-1}_i = x_i (R^l_i - s_i Σ_j R^l_j)   (s_i = softmax output)
     - Matmul (eq. 15):     for O_jp = Σ_i A_ji V_ip:
                            R^{l-1}_ji = Σ_p A_ji V_ip R^l_jp / (2 O_jp + ε)
     - LayerNorm/RMSNorm (eq. 19): identity rule R^{l-1}_i = R^l_i
     - Element-wise non-linearities (GELU, Swish): identity rule
     - Linear (γ-LRP for ViT): R^{l-1}_i = Σ_j W_ji x_i R^l_j / (z_j(x) + ε)
   Compute cost ≈ single backward pass (with O(1) memory checkpointing).
   Uniformly outperforms IG / SmoothGrad / GradCAM / AttnRoll / G×AttnRoll /
   AtMan / KernelSHAP / CP-LRP on faithfulness benchmarks.

2. **GMAR** (Jo & Jang, arXiv:2504.19414, April 2025) — Algorithm 1:
     - Backprop class-logit gradients; split G into per-head: G_hi = split(G, num_head)
     - Per-head importance: G_R = Σ|G_hi| (L1) or √(Σ G_hi²) (L2)  (L2 marginally better)
     - Normalize: w_h = G_R / Σ G_R
     - Weighted rollout (eq. 3):
         A_weighted = A_l ⊙ W      (W broadcast across heads)
         A_rollout  = A_rollout @ A_weighted + α · I_{N×N}     (α = residual ratio)

3. **GAF** (arXiv:2502.15765, Feb 2025) — Generalized Attention Flow.
   Three Information-Tensor variants:
     - AF  (Attention Flow):  Ā := E_h(A)
     - GF  (Grad Flow):       Ā := E_h(⌊∇A⌋_+)
     - AGF (Attention × Grad Flow):  Ā := E_h(⌊A ⊙ ∇A⌋_+)   ← the SOTA variant
   Where ⌊x⌋_+ = max(x, 0), ⊙ is Hadamard, ∇A := ∂y_t/∂A, E_h averages heads.
   Layered attribution graph 𝒢: super-source ss + super-target st with capacity u_∞;
   edge capacities u[I_(i,ℓ+1), I_(j,ℓ)] = Ā_(ℓ,i,j); cost c_(t,s) = -1.
   Solve max-flow via barrier method (eq. 12):
     min_{B^T f = 0} c^T f + ψ_μ(f),  ψ_μ(f) = -μ Σ_e [log(f_e - l_e) + log(u_e - f_e)]
   ε-approximation via μ ≤ ε / (2m).

This module provides numpy-based reference implementations of the three
algorithms.  GPU integration with the actual ResDec-MHE encoder is the next
step (separate orchestrator script in scripts/resdec_mhe/interpretability/).

Public API:

    attnlrp_softmax(R_l, s, x, eps=1e-6)
        Softmax LRP rule (eq. 13).  Returns R^{l-1}.

    attnlrp_matmul(R_l, A, V, eps=1e-6)
        Matrix-multiplication LRP rule (eq. 15).  Returns R^{l-1} for A.

    attnlrp_identity(R_l)
        LayerNorm / GELU / element-wise identity rule.

    attnlrp_linear_eps(R_l, W, x, eps=1e-6)
        ε-LRP rule for linear layer (eq. 8).  Returns R^{l-1}.

    gmar_head_weights(grad, n_heads, norm="l2")
        GMAR per-head importance (Algorithm 1, gradient-based).

    gmar_weighted_rollout(per_layer_attention, per_layer_head_weights, alpha=1.0)
        GMAR weighted-rollout aggregation (eq. 3).

    gaf_information_tensor(A, grad_A=None, variant="agf")
        GAF information tensor with 3 variants (AF / GF / AGF).
"""
from __future__ import annotations

from typing import Literal

import numpy as np


# ───────────────────────────────────────────────────────────────────────────────
# AttnLRP — Achtibat et al., ICML 2024
# ───────────────────────────────────────────────────────────────────────────────

def attnlrp_softmax(
    R_l: np.ndarray,
    s: np.ndarray,
    x: np.ndarray,
    *,
    eps: float = 1e-6,
) -> np.ndarray:
    """Softmax LRP rule (Achtibat et al., eq. 13).

    R^{l-1}_i = x_i * (R^l_i - s_i * Σ_j R^l_j)

    Parameters
    ----------
    R_l
        Relevance at the softmax OUTPUT, shape ``(..., N)``.
    s
        Softmax output values, shape ``(..., N)``.
    x
        Softmax INPUT values, shape ``(..., N)``.
    eps
        Stabilizer; only used by the caller — not consumed inside this rule
        because the eq. 13 form is bias-free.

    Returns
    -------
    np.ndarray
        Relevance at the softmax INPUT, shape ``(..., N)``.

    Notes
    -----
    The eq. 13 form is **non-conservative by design**: relevance leaks into
    a virtual bias term per Achtibat et al. §3.3.3 ("hidden bias absorbing
    vanishing residuals"). Use ``Σ R^{l-1} ≈ Σ R^l`` as a heuristic check
    (typically holds when ``Σ R^l ≠ 0`` and ``s`` is non-degenerate), NOT
    as a strict invariant. ``test_attnlrp_softmax_non_conservation_documented``
    documents the expected non-conservation behavior.
    """
    R_l = np.asarray(R_l, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    sum_R = R_l.sum(axis=-1, keepdims=True)
    return x * (R_l - s * sum_R)


def attnlrp_matmul(
    R_l: np.ndarray,
    A: np.ndarray,
    V: np.ndarray,
    *,
    eps: float = 1e-6,
) -> np.ndarray:
    """Matrix-multiplication LRP rule (Achtibat et al., eq. 15).

    For O_{jp} = Σ_i A_{ji} V_{ip} :

        R^{l-1}_{ji} = Σ_p A_{ji} V_{ip} R^l_{jp} / (2 O_{jp} + ε)

    Distributes relevance proportionally between the two operands; factor
    of 2 in denominator because both A and V contribute multiplicatively.

    Parameters
    ----------
    R_l
        Relevance at the matmul output O, shape ``(..., J, P)``.
    A
        Attention / left operand, shape ``(..., J, I)``.
    V
        Value / right operand, shape ``(..., I, P)``.
    eps
        Numerical stabilizer.

    Returns
    -------
    np.ndarray
        Relevance attributed to ``A``, shape ``(..., J, I)``.
    """
    R_l = np.asarray(R_l, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    O = A @ V  # (..., J, P)
    # Stronger floor than `2*O + eps*sign(O+eps)`: when |2 O| < eps the
    # vanilla form yields denom ≈ eps producing R magnitudes ~1e6 even when
    # the true contribution is near zero. Replace with a hard min-magnitude
    # floor of eps so near-zero O maps to bounded |R|. Sign of O preserved.
    two_O = 2.0 * O
    sign = np.where(two_O >= 0, 1.0, -1.0)
    denom = np.where(np.abs(two_O) > eps, two_O, eps * sign)
    factor = R_l / denom  # (..., J, P)
    # R_ji = Σ_p A_ji V_ip * factor_jp = A_ji * Σ_p V_ip * factor_jp
    R_a = A * (factor @ V.swapaxes(-2, -1))  # (..., J, I)
    return R_a


def attnlrp_identity(R_l: np.ndarray) -> np.ndarray:
    """LayerNorm / RMSNorm / GELU identity rule (eq. 19 + element-wise non-linearities).

    For Taylor decomposition with reference 0, the identity rule
    ``R^{l-1} = R^l`` strictly preserves conservation while excluding
    normalization from the relevance graph (per Achtibat et al. §3.3.3).
    """
    return np.asarray(R_l, dtype=np.float64).copy()


def attnlrp_linear_eps(
    R_l: np.ndarray,
    W: np.ndarray,
    x: np.ndarray,
    *,
    eps: float = 1e-6,
) -> np.ndarray:
    """ε-LRP rule for linear layer y_j = Σ_i W_ji x_i + b_j (eq. 8).

    R^{l-1}_i = Σ_j W_ji x_i R^l_j / (z_j(x) + ε)

    where z_j(x) = Σ_i W_ji x_i is the pre-bias linear output.
    """
    R_l = np.asarray(R_l, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    z = np.einsum("ji,...i->...j", W, x)
    sign = np.where(z >= 0, 1.0, -1.0)
    factor = R_l / (z + eps * sign)
    R_lminus1 = np.einsum("...j,ji->...i", factor, W) * x
    return R_lminus1


# ───────────────────────────────────────────────────────────────────────────────
# GMAR — Jo & Jang, arXiv:2504.19414 (April 2025)
# ───────────────────────────────────────────────────────────────────────────────

def gmar_head_weights(
    grad: np.ndarray,
    n_heads: int,
    *,
    norm: Literal["l1", "l2"] = "l2",
) -> np.ndarray:
    """GMAR per-head importance score (Algorithm 1, gradient-based).

    Parameters
    ----------
    grad
        Gradient of class logit wrt one attention layer, shape
        ``(B, n_heads, N, N)`` or ``(n_heads, N, N)``.
    n_heads
        Number of attention heads in the layer.
    norm
        ``"l1"``: G_R = Σ|G_hi|;  ``"l2"``: G_R = sqrt(Σ G_hi²).
        L2 marginally outperformed L1 in the original paper.

    Returns
    -------
    np.ndarray
        Per-head normalized weights, shape ``(n_heads,)``, sum to 1.
    """
    grad = np.asarray(grad, dtype=np.float64)
    if grad.shape[-3] != n_heads:
        raise ValueError(
            f"Expected n_heads={n_heads} on axis -3, got shape {grad.shape}"
        )
    # Split by head and reduce per-head
    if norm == "l1":
        G_R = np.abs(grad).reshape(*grad.shape[:-3], n_heads, -1).sum(axis=-1)
    elif norm == "l2":
        G_R = np.sqrt(
            (grad ** 2).reshape(*grad.shape[:-3], n_heads, -1).sum(axis=-1)
        )
    else:
        raise ValueError(f"norm must be 'l1' or 'l2', got {norm!r}")
    # Mean over batch if present, then normalize across heads
    while G_R.ndim > 1:
        G_R = G_R.mean(axis=0)
    if len(G_R) == 0:
        raise ValueError("GMAR head weights cannot be computed for empty grad")
    total = G_R.sum()
    if total <= 0:
        return np.ones_like(G_R) / len(G_R)
    return G_R / total


def gmar_weighted_rollout(
    per_layer_attention: list[np.ndarray],
    per_layer_head_weights: list[np.ndarray] | None = None,
    *,
    alpha: float = 1.0,
) -> np.ndarray:
    """GMAR weighted attention rollout (Jo & Jang 2025, eq. 3).

    Standard rollout (Abnar & Zuidema 2020):
        A_rollout = ∏_l (A^{(l)} + α·I)        (right-multiply convention)

    GMAR replaces the per-layer A^{(l)} with head-weighted A_weighted^{(l)}:
        A_weighted^{(l)} = Σ_h W_h · A_h^{(l)}
        A_rollout        = A_rollout @ (A_weighted + α · I_{N×N})

    where W^{(l)} is broadcast across heads and α controls the residual ratio
    (paper default α=1, matching the Abnar-Zuidema canonical convention).
    With uniform per-head weights this reduces exactly to vanilla rollout.

    Parameters
    ----------
    per_layer_attention
        List of arrays, each shape ``(n_heads, N, N)``.  No batch dim
        (this is per-subject; for batches, call once per subject).
    per_layer_head_weights
        Optional list of arrays, each shape ``(n_heads,)``.  If ``None``,
        uniform weights = vanilla rollout.
    alpha
        Residual contribution coefficient (paper default 1.0 for the
        identity matrix).

    Returns
    -------
    np.ndarray
        A_rollout of shape ``(N, N)``.
    """
    if not per_layer_attention:
        raise ValueError("per_layer_attention is empty")
    n_heads, N, _ = per_layer_attention[0].shape
    A_rollout = np.eye(N)
    for ell, A in enumerate(per_layer_attention):
        if A.shape != (n_heads, N, N):
            raise ValueError(
                f"Layer {ell}: expected ({n_heads}, {N}, {N}), got {A.shape}"
            )
        if per_layer_head_weights is None:
            w = np.ones(n_heads) / n_heads  # uniform → vanilla rollout
        else:
            w = per_layer_head_weights[ell]
        A_weighted = (w[:, None, None] * A).sum(axis=0)  # (N, N)
        # Apply (A_weighted + α·I) as the per-layer transformation, matching
        # Abnar & Zuidema 2020 vanilla rollout under uniform W.
        A_rollout = A_rollout @ (A_weighted + alpha * np.eye(N))
    return A_rollout


# ───────────────────────────────────────────────────────────────────────────────
# GAF — arXiv:2502.15765 (Feb 2025)
# ───────────────────────────────────────────────────────────────────────────────

def gaf_information_tensor(
    A: np.ndarray,
    grad_A: np.ndarray | None = None,
    *,
    variant: Literal["af", "gf", "agf"] = "agf",
) -> np.ndarray:
    """GAF Information Tensor (3 variants).

    Parameters
    ----------
    A
        Attention tensor, shape ``(L, n_heads, N, N)`` (L = #layers).
    grad_A
        Gradient ∂y_t/∂A, same shape as ``A``.  Required for ``"gf"`` and
        ``"agf"``; ignored for ``"af"``.
    variant
        ``"af"`` (Attention Flow): Ā := E_h(A)
        ``"gf"`` (Gradient Flow):  Ā := E_h(⌊∇A⌋_+)
        ``"agf"`` (Attention × Gradient Flow): Ā := E_h(⌊A ⊙ ∇A⌋_+)
        AGF is the SOTA variant per the original paper.

    Returns
    -------
    np.ndarray
        Information tensor Ā of shape ``(L, N, N)``.
    """
    if variant not in ("af", "gf", "agf"):
        raise ValueError(f"variant must be af/gf/agf, got {variant!r}")
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 4:
        raise ValueError(f"A must be 4-D (L, n_heads, N, N); got {A.shape}")
    if variant == "af":
        return A.mean(axis=1)
    if grad_A is None:
        raise ValueError(f"variant={variant!r} requires grad_A")
    grad_A = np.asarray(grad_A, dtype=np.float64)
    if grad_A.shape != A.shape:
        raise ValueError(
            f"A.shape={A.shape} ≠ grad_A.shape={grad_A.shape}"
        )
    if variant == "gf":
        return np.maximum(grad_A, 0.0).mean(axis=1)
    return np.maximum(A * grad_A, 0.0).mean(axis=1)
