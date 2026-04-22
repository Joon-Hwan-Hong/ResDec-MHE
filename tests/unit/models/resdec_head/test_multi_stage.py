"""Phase 3 Task 3.1 — ResDec-H3 3-stage composer tests.

Verifies:
    * forward-pass shapes / dict keys for the 3-stage composer
    * cross-stage attention is actually wired (stage_2 depends on h_1, stage_3 on both)
    * gradient detach for stage-2 and stage-3 auxiliary losses
    * TabMWrapper k_tabm=1 degenerate case

All gradient-detach tests run in fp32 so that bf16 noise cannot mask tiny non-zero
gradients that would signal a broken detach.
"""
from __future__ import annotations

import pytest
import torch

from src.models.resdec_head.resdec_h3_head import ResDecH3Head


def _mk_head(d_subject: int = 64, d_metadata: int = 8, k_tabm: int = 2) -> ResDecH3Head:
    """Smaller k_tabm keeps tests fast — the semantics are identical to k=8."""
    torch.manual_seed(0)
    head = ResDecH3Head(
        d_subject=d_subject,
        d_metadata=d_metadata,
        n_heads=4,
        n_hc_streams=2,
        lambda_init=0.8,
        k_tabm=k_tabm,
    )
    return head.to(torch.float32)


def test_forward_shapes():
    """Head accepts z_encoder [B, 64] + metadata [B, 8], returns dict with the
    full Phase-3 key set and correct shapes. Verify that ``prediction`` is the
    sum of the three stage scalars."""
    head = _mk_head(d_subject=64, d_metadata=8, k_tabm=2)
    head.eval()
    B = 4
    z = torch.randn(B, 64)
    m = torch.randn(B, 8)
    out = head(z, m)

    for key in ("prediction", "stage_1", "stage_2", "stage_3",
                "latent_1", "latent_2", "latent_3"):
        assert key in out, f"Missing key: {key!r} (got {sorted(out.keys())})"

    assert out["prediction"].shape == (B,)
    for k in ("stage_1", "stage_2", "stage_3"):
        assert out[k].shape == (B,), f"{k} shape mismatch: {out[k].shape}"
    for k in ("latent_1", "latent_2", "latent_3"):
        assert out[k].shape == (B, 64), f"{k} shape mismatch: {out[k].shape}"

    # prediction MUST equal stage_1 + stage_2 + stage_3 (composer contract)
    expected = out["stage_1"] + out["stage_2"] + out["stage_3"]
    assert torch.allclose(out["prediction"], expected, atol=1e-6), \
        "prediction != stage_1 + stage_2 + stage_3"


def test_cross_stage_attention_wired():
    """Stage 2's output must depend on stage-1's latent (because stage 2 consumes
    a cross-stage-attention context built from h_1). Stage 3 must depend on both
    h_1 and h_2. We verify this by perturbing each prior latent via a forward hook
    and asserting downstream outputs change."""
    head = _mk_head(d_subject=32, d_metadata=4, k_tabm=2)
    head.eval()
    B = 3
    z = torch.randn(B, 32)
    m = torch.randn(B, 4)

    # --- Baseline forward ---
    with torch.no_grad():
        base = head(z, m)

    # --- Perturb latent_1 (stage_1 output) via hook on the TabM wrapper that
    # computes h_1. After perturbation stage_2 and stage_3 outputs should both
    # change, confirming that stage-2 input depends on h_1 and stage-3 input
    # (which takes both h_1 and h_2 as priors) also depends on h_1. ---
    perturb = torch.randn_like(base["latent_1"]) * 3.0  # big kick

    def _hook_latent_1(module, inputs, output):
        # TabMWrapper.forward returns (mean, std); we replace the mean.
        if isinstance(output, tuple):
            return (output[0] + perturb, output[1])
        return output + perturb

    handle = head.stage_1_tabm.register_forward_hook(_hook_latent_1)
    try:
        with torch.no_grad():
            perturbed = head(z, m)
    finally:
        handle.remove()

    delta_s2 = (perturbed["stage_2"] - base["stage_2"]).abs().max().item()
    delta_s3 = (perturbed["stage_3"] - base["stage_3"]).abs().max().item()
    assert delta_s2 > 1e-5, \
        f"stage_2 did not change when latent_1 was perturbed (delta={delta_s2}); " \
        "cross-stage attention is not wired."
    assert delta_s3 > 1e-5, \
        f"stage_3 did not change when latent_1 was perturbed (delta={delta_s3}); " \
        "cross-stage attention is not wired."

    # --- Now perturb latent_2 (stage_2 output). stage_3 should change (since its
    # priors list contains h_2); stage_1 and stage_2 outputs are already past. ---
    perturb2 = torch.randn_like(base["latent_2"]) * 3.0

    def _hook_latent_2(module, inputs, output):
        if isinstance(output, tuple):
            return (output[0] + perturb2, output[1])
        return output + perturb2

    handle = head.stage_2_tabm.register_forward_hook(_hook_latent_2)
    try:
        with torch.no_grad():
            perturbed2 = head(z, m)
    finally:
        handle.remove()

    delta_s3_from_h2 = (perturbed2["stage_3"] - base["stage_3"]).abs().max().item()
    assert delta_s3_from_h2 > 1e-5, \
        f"stage_3 did not change when latent_2 was perturbed (delta={delta_s3_from_h2}); " \
        "stage 3 is not attending to h_2."


def _collect_stage_params(head: ResDecH3Head, which: str) -> list[torch.nn.Parameter]:
    """Return the leaf params belonging to ``stage_{which}``'s TabM+readout path.

    We exclude the FiLM+cross-stage-attention modules because they are SHARED
    across stages or feed all of them — their gradients aren't what detach()
    is supposed to block. The detach contract is:
        aux_k loss must not update stage_{k-1}'s TabM wrapper + its readout.

    FiLM and cross_attn_* are intentionally shared across stages, so they
    receive gradient from every aux loss — this is expected and not a detach
    violation.
    """
    wrapper = getattr(head, f"stage_{which}_tabm")
    readout = getattr(head, f"stage_{which}_readout")
    params = list(wrapper.parameters()) + list(readout.parameters())
    return params


def test_gradient_detach_stage2():
    """Aux-2 loss uses ``y - y_tabpfn - stage_1.detach()``. Backward through it
    must produce zero gradient on stage_1's TabM+readout params."""
    torch.manual_seed(0)
    head = _mk_head(d_subject=32, d_metadata=4, k_tabm=2)
    head.train()  # ensure everything requires_grad
    B = 4
    z = torch.randn(B, 32)
    m = torch.randn(B, 4)
    y = torch.randn(B)
    y_tabpfn = torch.randn(B)

    out = head(z, m)
    # aux_2 target mirrors the composer's training-step formula exactly.
    target_aux2 = y - y_tabpfn - out["stage_1"].detach()
    aux2_loss = torch.nn.functional.mse_loss(out["stage_2"], target_aux2)

    # Zero all grads, then backward ONLY the aux_2 loss.
    for p in head.parameters():
        if p.grad is not None:
            p.grad.zero_()
    aux2_loss.backward()

    stage1_params = _collect_stage_params(head, "1")
    stage1_grad_norms = [
        (0.0 if p.grad is None else p.grad.abs().max().item()) for p in stage1_params
    ]
    max_stage1_grad = max(stage1_grad_norms) if stage1_grad_norms else 0.0
    assert max_stage1_grad == 0.0, (
        f"aux_2 loss leaked gradient into stage_1 params: max |grad| = {max_stage1_grad}. "
        "Detach() on stage_1 in the aux_2 target is not blocking gradient flow."
    )

    # Sanity: stage_2 params DID get gradients (else the test itself is broken).
    stage2_params = _collect_stage_params(head, "2")
    max_stage2_grad = max(
        (0.0 if p.grad is None else p.grad.abs().max().item()) for p in stage2_params
    )
    assert max_stage2_grad > 0.0, "aux_2 loss did not flow into stage_2 (test is broken)"


def test_gradient_detach_stage3():
    """Aux-3 loss uses ``y - y_tabpfn - stage_1.detach() - stage_2.detach()``.
    Backward must produce zero gradient on stage_1 AND stage_2 TabM+readout
    params."""
    torch.manual_seed(1)
    head = _mk_head(d_subject=32, d_metadata=4, k_tabm=2)
    head.train()
    B = 4
    z = torch.randn(B, 32)
    m = torch.randn(B, 4)
    y = torch.randn(B)
    y_tabpfn = torch.randn(B)

    out = head(z, m)
    target_aux3 = y - y_tabpfn - out["stage_1"].detach() - out["stage_2"].detach()
    aux3_loss = torch.nn.functional.mse_loss(out["stage_3"], target_aux3)

    for p in head.parameters():
        if p.grad is not None:
            p.grad.zero_()
    aux3_loss.backward()

    for which in ("1", "2"):
        params = _collect_stage_params(head, which)
        max_grad = max(
            (0.0 if p.grad is None else p.grad.abs().max().item()) for p in params
        )
        assert max_grad == 0.0, (
            f"aux_3 loss leaked gradient into stage_{which} params: "
            f"max |grad| = {max_grad}. Detach() is not blocking gradient flow."
        )

    # Sanity: stage_3 DID get gradient.
    stage3_params = _collect_stage_params(head, "3")
    max_stage3_grad = max(
        (0.0 if p.grad is None else p.grad.abs().max().item()) for p in stage3_params
    )
    assert max_stage3_grad > 0.0, "aux_3 loss did not flow into stage_3 (test is broken)"


def test_tabm_k_param():
    """k_tabm=1 makes each TabMWrapper a single-member pass. Must produce valid
    shapes and finite outputs (no ensemble-dim collapse errors), and the std
    return must be exactly zero (std over a single element is zero)."""
    head = _mk_head(d_subject=32, d_metadata=4, k_tabm=1)
    head.eval()
    B = 4
    z = torch.randn(B, 32)
    m = torch.randn(B, 4)
    with torch.no_grad():
        out = head(z, m)
    assert out["prediction"].shape == (B,)
    for k in ("stage_1", "stage_2", "stage_3"):
        assert torch.isfinite(out[k]).all(), f"{k} produced non-finite values at k_tabm=1"
    for k in ("latent_1", "latent_2", "latent_3"):
        assert out[k].shape == (B, 32)
        assert torch.isfinite(out[k]).all(), f"{k} produced non-finite values at k_tabm=1"

    # Verify each TabMWrapper really has k=1.
    for which in ("1", "2", "3"):
        wrapper = getattr(head, f"stage_{which}_tabm")
        assert wrapper.k == 1, f"stage_{which}_tabm.k should be 1, got {wrapper.k}"

    # Directly call each TabMWrapper at k=1 and verify the std return is zero.
    # Rationale: torch.std over a single element (unbiased=True by default)
    # returns NaN; torch.std with unbiased=False returns 0. TabMWrapper should
    # behave consistently at k=1 (either always zero or all finite) so downstream
    # uncertainty-consumers aren't surprised. We assert finite + all-zero here.
    z_probe = torch.randn(B, 32)
    for which in ("1", "2", "3"):
        wrapper = getattr(head, f"stage_{which}_tabm")
        # stage_2 / stage_3 wrappers expect a shifted input equivalent to z_cond
        # + ctx_{2,3}; since we're only probing std behaviour, calling with any
        # valid [B, d] tensor is sufficient — TabMWrapper's std doesn't depend
        # on the upstream cross-attn context.
        with torch.no_grad():
            mean, std = wrapper(z_probe)
        assert torch.isfinite(std).all(), (
            f"stage_{which}_tabm std is non-finite at k=1 "
            f"(torch.std with unbiased=True returns NaN for a single element; "
            "check TabMWrapper.forward's reduction)."
        )
        assert torch.all(std == 0.0), (
            f"stage_{which}_tabm std is not exactly 0 at k=1, got max |std|="
            f"{std.abs().max().item()}"
        )
