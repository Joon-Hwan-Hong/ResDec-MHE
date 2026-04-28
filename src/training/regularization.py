"""Attention regularization functions for the encoder's
PathologyStratifiedAttention output.

SKELETON ONLY (2026-04-28). All function bodies raise NotImplementedError.
The corresponding tests at ``tests/unit/training/test_regularization.py``
are all skipped via ``pytest.mark.skip``. Implementation of these
functions, plus the encoder-side gradient plumbing they require (forcing
``return_attention_weights=True`` during training, and removing the
``torch.no_grad`` guard from the softmax in
``src/models/fusion/pathology_attention.py:194``), is gated on explicit
user approval per the design doc at
``docs/plans/2026-04-28-encoder-attention-regularization-design.md``.

All functions take an ``attention`` tensor of shape ``[B, H, C]`` where:
    B = batch size
    H = number of heads (default 4 in PathologyStratifiedAttention)
    C = number of cell types (31 in CELL_TYPE_ORDER)
and a scalar ``weight`` (the λ coefficient applied to the regularizer
term). Each ``attention[b, h, :]`` is a probability distribution over
cell types (rows sum to 1, since the source is the
``F.softmax(scores, dim=-1)`` at ``pathology_attention.py:194``).

Each function returns a scalar 0-dim ``torch.Tensor`` representing the
regularizer term ready to be added to the training loss in
``ResDecLightningModule.training_step``.
"""
from __future__ import annotations

import torch

# Numerical floor for log(p) computation. p · log(p + eps) is well-defined
# at p=0 (limits to 0) but log(0) is -inf. Use eps=1e-12 — small enough not
# to bias the entropy of any non-degenerate distribution while preventing
# NaN/-inf gradients from zero-attention cells (which arise from the
# cell_type_mask absent-CT path in pathology_attention.py:163).
LOG_EPS: float = 1e-12


def attention_entropy_bonus(
    attention: torch.Tensor,
    weight: float,
) -> torch.Tensor:
    """Negative-entropy bonus that ENCOURAGES high-entropy attention.

    Computes ``-weight · mean_{b,h} H(attention[b, h, :])`` where
    ``H(p) = -Σ_c p_c · log(p_c + eps)`` is the Shannon entropy in nats.
    Sign convention: this returns a scalar to be ADDED to the loss; a
    negative value DECREASES the loss when entropy is high → optimizer
    pushes attention toward uniform (max entropy = log C ≈ 3.434 nats
    for C=31).

    Args:
        attention: Tensor of shape ``[B, H, C]``; each ``attention[b, h, :]``
            must be a probability distribution (rows sum to 1, all entries
            in [0, 1]). Source: ``PathologyStratifiedAttention.forward(...,
            return_attention_weights=True)`` returns this shape.
        weight: ``λ`` coefficient. Sweep range per design doc:
            {0, 1e-3, 1e-2, 1e-1, 1.0}.

    Returns:
        Scalar 0-dim ``torch.Tensor`` of dtype matching ``attention.dtype``,
        device matching ``attention.device``. Value is
        ``-weight · mean_H``; pass directly to ``loss = loss + result``.
        When ``weight == 0`` returns a scalar zero (still differentiable;
        gradient is zero).

    Raises:
        ValueError: If ``attention.dim() != 3``.
        ValueError: If ``weight < 0`` (negative weight inverts the
            optimization direction — flag rather than silently accept).

    Implementation notes:
        - Use ``(attention + LOG_EPS).log()`` to avoid log(0).
        - Cast to float32 for numerical stability if input is bf16/fp16
          (entropy of small distributions has limited dynamic range).
        - Reduction is ``mean(dim=(0, 1))`` over batch and heads after
          per-(b, h) entropy is computed.
    """
    if attention.dim() != 3:
        raise ValueError(
            f"attention must be 3-D [B, H, C], got shape {tuple(attention.shape)}"
        )
    if weight < 0:
        raise ValueError(
            f"weight must be non-negative (got {weight}); negative weight inverts "
            "the optimization direction (encourages concentration instead of spread)."
        )
    a = attention.float()
    # Per-(b, h) entropy: H(p) = -Σ_c p_c · log(p_c + eps)
    log_a = (a + LOG_EPS).log()
    per_bh_entropy = -(a * log_a).sum(dim=-1)  # [B, H]
    mean_h = per_bh_entropy.mean()
    # Cast result back to original dtype so loss arithmetic stays consistent.
    return (-float(weight) * mean_h).to(attention.dtype)


def attention_kl_to_uniform(
    attention: torch.Tensor,
    weight: float,
) -> torch.Tensor:
    """Forward-KL penalty that PENALIZES deviation from uniform.

    Computes ``+weight · mean_{b,h} KL(attention[b, h, :] || uniform_C)``
    where ``uniform_C[c] = 1/C`` and
    ``KL(p || q) = Σ_c p_c · log(p_c / q_c)``.
    Equivalent up to a constant offset (``log C``) to the negative-entropy
    bonus; gradient is identical. Sign: returns a non-negative scalar to
    be ADDED to the loss; minimizing loss minimizes KL → pushes attention
    toward uniform.

    Note: per design doc §5(B), this is **strictly inferior** to
    ``attention_entropy_bonus`` due to the constant offset (``log C``)
    polluting the absolute loss value. Provided for completeness /
    ablation.

    Args:
        attention: Tensor of shape ``[B, H, C]``; each ``attention[b, h, :]``
            must be a probability distribution (rows sum to 1).
        weight: ``λ`` coefficient. Sweep range per design doc:
            {0, 1e-3, 1e-2, 1e-1, 1.0}.

    Returns:
        Scalar 0-dim ``torch.Tensor`` of dtype matching ``attention.dtype``,
        device matching ``attention.device``. Value is non-negative
        (``KL >= 0`` with equality iff ``attention == uniform``).

    Raises:
        ValueError: If ``attention.dim() != 3``.
        ValueError: If ``weight < 0`` (negative weight rewards deviation
            from uniform — flag rather than silently accept).

    Implementation notes (deferred):
        - ``log_C = math.log(attention.shape[-1])``.
        - Use ``(attention + LOG_EPS).log()`` to avoid log(0).
        - Per-(b, h) ``KL = Σ_c a_c · (log a_c - log(1/C))
                          = -H(a) + log C``.
        - Reduction is ``mean(dim=(0, 1))`` over batch and heads.
    """
    raise NotImplementedError(
        "attention_kl_to_uniform is a SKELETON. Implementation gated on user approval — "
        "see docs/plans/2026-04-28-encoder-attention-regularization-design.md §12-13."
    )


def attention_coverage_penalty(
    attention: torch.Tensor,
    floor: float,
    weight: float,
) -> torch.Tensor:
    """Hard per-CT minimum-share penalty (soft floor).

    Computes ``+weight · mean_{b,h} Σ_c max(0, floor - attention[b, h, c])``
    — penalizes attention BELOW ``floor`` for any CT, summed over the C
    cell types and averaged over batch and heads. Sign: returns a
    non-negative scalar to be ADDED to the loss; minimizing loss pushes
    every CT's attention up to (or above) ``floor``.

    Differs from entropy/KL: directly enforces a per-CT minimum (whereas
    entropy can be high without ANY individual CT crossing a threshold,
    e.g., a 2-CT bimodal at 0.5 each).

    Args:
        attention: Tensor of shape ``[B, H, C]``; each ``attention[b, h, :]``
            must be a probability distribution (rows sum to 1).
        floor: ``c_floor`` per-CT minimum attention share. Per design doc
            §5(C), default candidate is ``0.5 / C = 0.0161`` for C=31
            (half of uniform). Must be in (0, 1/C] — values above 1/C
            would force every CT above uniform, which is impossible since
            attention sums to 1.
        weight: ``λ`` coefficient. Sweep range per design doc:
            {0, 1e-3, 1e-2, 1e-1, 1.0}.

    Returns:
        Scalar 0-dim ``torch.Tensor`` of dtype matching ``attention.dtype``,
        device matching ``attention.device``. Value is non-negative; is
        exactly 0 when every (b, h, c) has ``attention >= floor``.

    Raises:
        ValueError: If ``attention.dim() != 3``.
        ValueError: If ``weight < 0`` (negative weight rewards deviation
            below floor — flag rather than silently accept).
        ValueError: If ``floor <= 0`` or ``floor > 1.0 / attention.shape[-1]``.
            The upper bound is necessary because the simplex constraint
            forces ``mean(a) = 1/C``, so ``floor > 1/C`` is unsatisfiable.

    Implementation notes (deferred):
        - ``deficit = (floor - attention).clamp(min=0.0)`` — element-wise.
        - Sum over CTs, mean over (B, H).
        - Subgradient at ``a = floor``: torch's ``relu`` / clamp gives
          subgradient 0 at the boundary; consistent with PyTorch
          conventions for hinge-like losses.
    """
    raise NotImplementedError(
        "attention_coverage_penalty is a SKELETON. Implementation gated on user approval — "
        "see docs/plans/2026-04-28-encoder-attention-regularization-design.md §12-13."
    )


def attention_top1_cap(
    attention: torch.Tensor,
    cap: float,
    weight: float,
) -> torch.Tensor:
    """Top-1 attention cap penalty.

    Computes ``+weight · mean_{b,h} max(0, max_c(attention[b, h, c]) - cap)``
    — penalizes the LARGEST CT's attention above ``cap`` per (batch, head),
    averaged over batch and heads. Sign: returns a non-negative scalar to
    be ADDED to the loss; minimizing loss pushes the dominant CT's
    attention down to (or below) ``cap``.

    Most direct attack on the observed concentration failure mode (head
    1 mean top-1 attention 0.123 on Splatter, vs uniform 0.0323; cap τ
    = 0.10 would push head 1's Splatter attention from 0.123 → ~0.10).

    Args:
        attention: Tensor of shape ``[B, H, C]``; each ``attention[b, h, :]``
            must be a probability distribution (rows sum to 1).
        cap: ``τ`` top-1 attention cap. Per design doc §5(D), default
            candidate is ``0.10`` (vs current head-1 top-1 of 0.123).
            Must be in [1/C, 1) — values below 1/C are unsatisfiable
            (max ≥ 1/C by pigeonhole), values ≥ 1 are vacuous.
        weight: ``λ`` coefficient. Sweep range per design doc:
            {0, 1e-3, 1e-2, 1e-1, 1.0}.

    Returns:
        Scalar 0-dim ``torch.Tensor`` of dtype matching ``attention.dtype``,
        device matching ``attention.device``. Value is non-negative; is
        exactly 0 when ``max_c attention[b, h, c] <= cap`` for all (b, h).

    Raises:
        ValueError: If ``attention.dim() != 3``.
        ValueError: If ``weight < 0``.
        ValueError: If ``cap < 1.0 / attention.shape[-1]`` (unsatisfiable)
            or ``cap >= 1.0`` (vacuous).

    Implementation notes (deferred):
        - ``top1 = attention.max(dim=-1).values``  → shape ``[B, H]``
        - ``excess = (top1 - cap).clamp(min=0.0)``  → shape ``[B, H]``
        - Reduction: ``mean(dim=(0, 1))``.
        - ``torch.max`` returns gradient only to the argmax position;
          this is the desired behavior (only the dominant CT receives
          the regularizer's gradient signal).
    """
    raise NotImplementedError(
        "attention_top1_cap is a SKELETON. Implementation gated on user approval — "
        "see docs/plans/2026-04-28-encoder-attention-regularization-design.md §12-13."
    )
