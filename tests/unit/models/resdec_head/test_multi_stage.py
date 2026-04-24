"""ResDec-MHE N-stage composer tests (n_stages ∈ {1, 2, 3}).

Verifies:
    * forward-pass shapes / dict keys for n_stages=1, 2, 3 (only k <= n_stages
      keys are present; prediction = sum of present stage scalars)
    * cross-stage attention is wired (stage_2 depends on h_1 when n>=2;
      stage_3 depends on h_1 and h_2 when n>=3)
    * gradient detach for stage-2 (when n>=2) and stage-3 (when n>=3) aux losses
    * TabMWrapper k_tabm=1 degenerate case across all n_stages

All gradient-detach tests run in fp32 so that bf16 noise cannot mask tiny non-zero
gradients that would signal a broken detach.
"""
from __future__ import annotations

import pytest
import torch

from src.models.resdec_head.resdec_mhe_head import ResDecMHEHead


def _mk_head(d_subject: int = 64, d_metadata: int = 8,
             k_tabm: int = 2, n_stages: int = 3) -> ResDecMHEHead:
    """Smaller k_tabm keeps tests fast — the semantics are identical to k=8."""
    torch.manual_seed(0)
    head = ResDecMHEHead(
        d_subject=d_subject,
        d_metadata=d_metadata,
        n_heads=4,
        n_hc_streams=2,
        lambda_init=0.8,
        k_tabm=k_tabm,
        n_stages=n_stages,
    )
    return head.to(torch.float32)


# --------------------------------------------------------------------------- #
# Forward shapes per n_stages                                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_stages", [1, 2, 3])
def test_forward_shapes(n_stages):
    """Head returns dict with `stage_k` and `latent_k` only for k <= n_stages.
    `prediction` equals the sum of present stage scalars."""
    head = _mk_head(d_subject=64, d_metadata=8, k_tabm=2, n_stages=n_stages)
    head.eval()
    B = 4
    z = torch.randn(B, 64)
    m = torch.randn(B, 8)
    out = head(z, m)

    assert "prediction" in out
    assert out["prediction"].shape == (B,)

    expected_keys = {"prediction"}
    for k in range(1, n_stages + 1):
        expected_keys.add(f"stage_{k}")
        expected_keys.add(f"latent_{k}")
        assert out[f"stage_{k}"].shape == (B,), f"stage_{k} shape mismatch"
        assert out[f"latent_{k}"].shape == (B, 64), f"latent_{k} shape mismatch"

    # Absent stages must NOT be in the dict (caller uses .get() to guard).
    for k in range(n_stages + 1, 4):
        assert f"stage_{k}" not in out, f"stage_{k} should be absent for n_stages={n_stages}"
        assert f"latent_{k}" not in out, f"latent_{k} should be absent for n_stages={n_stages}"

    # prediction MUST equal sum of present stage scalars.
    expected_pred = sum(out[f"stage_{k}"] for k in range(1, n_stages + 1))
    assert torch.allclose(out["prediction"], expected_pred, atol=1e-6), \
        f"prediction != sum of present stages (n_stages={n_stages})"


# --------------------------------------------------------------------------- #
# n_stages=1 sanity: composer reduces to FiLM + single TabM[NPT] + readout    #
# --------------------------------------------------------------------------- #
def test_n_stages_1_only_builds_stage_1():
    """At n_stages=1, no cross-stage-attention or stage 2/3 modules should be
    constructed (saves params + optimizer state). Verify via state_dict."""
    head = _mk_head(d_subject=32, d_metadata=4, k_tabm=2, n_stages=1)
    sd_keys = set(head.state_dict().keys())
    # Stage 2/3 modules' submodule names should NOT appear.
    forbidden = ["stage_2_npt", "stage_2_tabm", "stage_2_readout", "stage_2_cross_attn",
                 "stage_3_npt", "stage_3_tabm", "stage_3_readout", "stage_3_cross_attn"]
    leaks = [name for name in sd_keys for f in forbidden if name.startswith(f + ".")]
    assert not leaks, f"n_stages=1 head leaked stage 2/3 params: {leaks[:5]}"


def test_invalid_n_stages_rejected():
    """Constructor must reject n_stages outside {1, 2, 3}."""
    for bad in (0, 4, -1):
        with pytest.raises(ValueError, match="n_stages must be one of"):
            ResDecMHEHead(d_subject=32, d_metadata=4, n_stages=bad)


# --------------------------------------------------------------------------- #
# Cross-stage attention wiring                                                #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_stages", [2, 3])
def test_cross_stage_attention_wired_h1(n_stages):
    """When n_stages >= 2, stage_2's output depends on stage-1's latent.
    When n_stages >= 3, stage_3 also changes (it has h_1 in its priors list)."""
    head = _mk_head(d_subject=32, d_metadata=4, k_tabm=2, n_stages=n_stages)
    head.eval()
    B = 3
    z = torch.randn(B, 32)
    m = torch.randn(B, 4)

    with torch.no_grad():
        base = head(z, m)
    perturb = torch.randn_like(base["latent_1"]) * 3.0

    def _hook(module, inputs, output):
        if isinstance(output, tuple):
            return (output[0] + perturb, output[1])
        return output + perturb

    handle = head.stage_1_tabm.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            perturbed = head(z, m)
    finally:
        handle.remove()

    delta_s2 = (perturbed["stage_2"] - base["stage_2"]).abs().max().item()
    assert delta_s2 > 1e-5, \
        f"stage_2 unchanged when latent_1 perturbed (delta={delta_s2}); cross-attn not wired."
    if n_stages >= 3:
        delta_s3 = (perturbed["stage_3"] - base["stage_3"]).abs().max().item()
        assert delta_s3 > 1e-5, \
            f"stage_3 unchanged when latent_1 perturbed (delta={delta_s3}); cross-attn not wired."


def test_cross_stage_attention_wired_h2_to_s3():
    """At n_stages=3, perturbing latent_2 must change stage_3's output."""
    head = _mk_head(d_subject=32, d_metadata=4, k_tabm=2, n_stages=3)
    head.eval()
    B = 3
    z = torch.randn(B, 32)
    m = torch.randn(B, 4)

    with torch.no_grad():
        base = head(z, m)
    perturb2 = torch.randn_like(base["latent_2"]) * 3.0

    def _hook(module, inputs, output):
        if isinstance(output, tuple):
            return (output[0] + perturb2, output[1])
        return output + perturb2

    handle = head.stage_2_tabm.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            perturbed = head(z, m)
    finally:
        handle.remove()

    delta_s3 = (perturbed["stage_3"] - base["stage_3"]).abs().max().item()
    assert delta_s3 > 1e-5, \
        f"stage_3 unchanged when latent_2 perturbed (delta={delta_s3}); stage 3 not attending to h_2."


# --------------------------------------------------------------------------- #
# Gradient detach contracts                                                   #
# --------------------------------------------------------------------------- #
def _collect_stage_params(head: ResDecMHEHead, which: str) -> list[torch.nn.Parameter]:
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
    head = _mk_head(d_subject=32, d_metadata=4, k_tabm=2, n_stages=2)
    head.train()
    B = 4
    z = torch.randn(B, 32)
    m = torch.randn(B, 4)
    y = torch.randn(B)
    y_tabpfn = torch.randn(B)

    out = head(z, m)
    target_aux2 = y - y_tabpfn - out["stage_1"].detach()
    aux2_loss = torch.nn.functional.mse_loss(out["stage_2"], target_aux2)

    for p in head.parameters():
        if p.grad is not None:
            p.grad.zero_()
    aux2_loss.backward()

    stage1_params = _collect_stage_params(head, "1")
    max_stage1_grad = max(
        (0.0 if p.grad is None else p.grad.abs().max().item()) for p in stage1_params
    )
    assert max_stage1_grad == 0.0, (
        f"aux_2 loss leaked gradient into stage_1 params: max |grad|={max_stage1_grad}. "
        "Detach() on stage_1 in the aux_2 target is not blocking gradient flow."
    )

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
    head = _mk_head(d_subject=32, d_metadata=4, k_tabm=2, n_stages=3)
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
            f"max |grad|={max_grad}. Detach() is not blocking gradient flow."
        )

    stage3_params = _collect_stage_params(head, "3")
    max_stage3_grad = max(
        (0.0 if p.grad is None else p.grad.abs().max().item()) for p in stage3_params
    )
    assert max_stage3_grad > 0.0, "aux_3 loss did not flow into stage_3 (test is broken)"


# --------------------------------------------------------------------------- #
# TabMWrapper k_tabm=1 degenerate case (parametrized over n_stages)           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_stages", [1, 2, 3])
def test_tabm_k_param(n_stages):
    """k_tabm=1 makes each TabMWrapper a single-member pass. Outputs must be finite,
    and TabM std must be exactly zero (population std over a single element)."""
    head = _mk_head(d_subject=32, d_metadata=4, k_tabm=1, n_stages=n_stages)
    head.eval()
    B = 4
    z = torch.randn(B, 32)
    m = torch.randn(B, 4)
    with torch.no_grad():
        out = head(z, m)
    assert out["prediction"].shape == (B,)
    for k in range(1, n_stages + 1):
        assert torch.isfinite(out[f"stage_{k}"]).all(), \
            f"stage_{k} non-finite at k_tabm=1"
        assert torch.isfinite(out[f"latent_{k}"]).all(), \
            f"latent_{k} non-finite at k_tabm=1"

    z_probe = torch.randn(B, 32)
    for k in range(1, n_stages + 1):
        wrapper = getattr(head, f"stage_{k}_tabm")
        assert wrapper.k == 1, f"stage_{k}_tabm.k should be 1, got {wrapper.k}"
        with torch.no_grad():
            _, std = wrapper(z_probe)
        assert torch.isfinite(std).all(), \
            f"stage_{k}_tabm std non-finite at k=1 (check TabMWrapper.forward reduction)."
        assert torch.all(std == 0.0), \
            f"stage_{k}_tabm std not exactly 0 at k=1, got max |std|={std.abs().max().item()}"
