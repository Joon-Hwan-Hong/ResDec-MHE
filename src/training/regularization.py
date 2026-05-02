"""Attention regularization functions for the encoder's
PathologyStratifiedAttention output.

Currently only ``attention_entropy_bonus`` (Scheme A) is implemented.
Other regularization schemes (KL-to-uniform, coverage penalty, top-1 cap)
were skeletoned in 2026-04-28 but never implemented; the skeletons were
removed 2026-05-02 per the codebase no-dead-code policy. If/when those
schemes are needed, add them WITH implementation + tests in a single PR
per the design doc at
``docs/plans/2026-04-28-encoder-attention-regularization-design.md``.

The implemented function takes an ``attention`` tensor of shape
``[B, H, C]`` where:
    B = batch size
    H = number of heads (default 4 in PathologyStratifiedAttention)
    C = number of cell types (31 in CELL_TYPE_ORDER)
and a scalar ``weight`` (the λ coefficient applied to the regularizer
term). Each ``attention[b, h, :]`` is a probability distribution over
cell types (rows sum to 1, since the source is the
``F.softmax(scores, dim=-1)`` at ``pathology_attention.py:194``).

The function returns a scalar 0-dim ``torch.Tensor`` representing the
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
