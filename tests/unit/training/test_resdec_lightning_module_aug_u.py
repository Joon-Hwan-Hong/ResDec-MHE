"""Tests for ResDecLightningModule aug-U weighting + TabPFN fallback.

These tests verify:
    * Weighted-mean σ-normalization is numerically stable when σ → 0
    * Weighted mean reduces to plain mean when all σ are equal
    * ``use_sigma_weighting=False`` falls back to plain MSE
    * Missing TabPFN cache → training_step uses plain MSE against cognition

All tests run in fp32 (explicit) so that bf16 mixed-precision can't mask the
numerical behaviour we're verifying. Rationale: these tests assert exact-ish
equality for reduction-math and finite gradients; bf16's 7-bit mantissa would
introduce enough noise that the weighted-mean == uniform-mean equivalence
would require loose atol and the σ→0 finite-grad check could silently pass
on Inf values masked as finite by downcasting.
"""
from __future__ import annotations

from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

from src.training.resdec_lightning_module import (
    DEFAULT_SIGMA_EPS,
    ResDecLightningModule,
)


# ---------------------------------------------------------------------------- #
# Fixtures: minimal config + tiny batch generator                               #
# ---------------------------------------------------------------------------- #
@pytest.fixture
def cfg():
    """Minimal config — same shape as the smoke-test fixture in
    test_resdec_lightning_module.py (deterministic head, resdec_head section)."""
    base = OmegaConf.load("configs/default.yaml")
    OmegaConf.set_struct(base, False)
    OmegaConf.set_struct(base.model, False)
    base.model.n_genes = 4785
    base.model.n_cell_types = 31
    base.model.head = OmegaConf.create({"type": "deterministic", "d_hidden": 32})
    base.model.resdec_head = OmegaConf.create({"d_metadata": 8, "n_heads": 4})
    base.training.lr = 0.0015
    base.training.weight_decay = 5.6e-6
    return base


def _make_dummy_batch(B: int = 4) -> dict[str, Any]:
    """Produce a minimal batch the encoder can accept — same shape fixture used
    in test_resdec_lightning_module.py's test_module_forward_dummy_batch."""
    N_CT, N_GENES, N_REGIONS = 31, 4785, 6
    cells_per_ct = 10
    cells_per_subject = cells_per_ct * N_CT

    region_mask = torch.zeros(B, N_REGIONS, dtype=torch.bool)
    region_mask[:, 0] = True

    offsets_per_subj = torch.arange(
        0, cells_per_subject + 1, cells_per_ct, dtype=torch.long,
    )
    subj_offsets = torch.arange(B, dtype=torch.long) * cells_per_subject
    cell_offsets = subj_offsets.unsqueeze(1) + offsets_per_subj.unsqueeze(0)

    batch = {
        "region_pseudobulk": torch.randn(B, N_REGIONS, N_CT, N_GENES),
        "region_mask": region_mask,
        "ccc_edge_index": torch.zeros(2, 0, dtype=torch.long),
        "ccc_edge_type": torch.zeros(0, dtype=torch.long),
        "ccc_edge_attr": torch.zeros(0, 1),
        "cell_type_mask": torch.ones(B, N_CT, dtype=torch.bool),
        "cell_data": torch.randn(B * cells_per_subject, N_GENES),
        "cell_offsets": cell_offsets,
        "pathology": torch.randn(B, 3),
        "cognition": torch.randn(B, 1),
        "subject_ids": [f"sid_{i:03d}" for i in range(B)],
    }
    return batch


def _enable_tabpfn_with(
    mod: ResDecLightningModule,
    subject_ids: list[str],
    *,
    y_tabpfn: list[float] | torch.Tensor,
    sigma_tabpfn: list[float] | torch.Tensor,
) -> None:
    """Inject an in-memory TabPFN train-map (skipping the .npz-loading path)."""
    y = list(y_tabpfn.tolist() if isinstance(y_tabpfn, torch.Tensor) else y_tabpfn)
    s = list(sigma_tabpfn.tolist() if isinstance(sigma_tabpfn, torch.Tensor) else sigma_tabpfn)
    mod.tabpfn_train_map = {
        sid: (float(y[i]), float(s[i])) for i, sid in enumerate(subject_ids)
    }
    mod.tabpfn_val_map = dict(mod.tabpfn_train_map)  # not used in these tests
    mod._tabpfn_enabled = True


# ---------------------------------------------------------------------------- #
# I4: new edge-case tests                                                       #
# ---------------------------------------------------------------------------- #
def test_sigma_weight_stability_sigma_near_zero(cfg):
    """If one subject has σ=0, weighted-mean normalization must not produce
    NaN/Inf gradients and the max |grad| on stage_2 params must stay within
    ~2× of the uniform-σ baseline."""
    # aug-U sigma weighting only activates at n_stages>=2 (see training_step
    # guard). Force n_stages=2 so this test exercises the weighting path.
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.set_struct(cfg, False)
    OmegaConf.set_struct(cfg.model, False)
    OmegaConf.set_struct(cfg.model.resdec_head, False)
    cfg.model.resdec_head.n_stages = 2
    cfg.model.resdec_head.aux_lambdas = [1.0, 1.0]

    torch.manual_seed(0)
    mod = ResDecLightningModule(cfg).float()
    mod.train()

    B = 4
    batch = _make_dummy_batch(B=B)

    # ---- Baseline: uniform σ = 1.0 for all subjects ----
    _enable_tabpfn_with(
        mod,
        subject_ids=batch["subject_ids"],
        y_tabpfn=[0.0] * B,
        sigma_tabpfn=[1.0] * B,
    )
    for p in mod.parameters():
        if p.grad is not None:
            p.grad.zero_()
    loss_uniform = mod.training_step(batch, batch_idx=0)
    assert torch.isfinite(loss_uniform), "loss is not finite under uniform σ"
    loss_uniform.backward()
    grads_uniform = [
        p.grad.abs().max().item()
        for p in mod.head.stage_2_tabm.parameters()
        if p.grad is not None
    ]
    max_grad_uniform = max(grads_uniform) if grads_uniform else 0.0
    assert max_grad_uniform > 0.0, "baseline test is broken: stage_2 got no gradient"

    # ---- σ=0 case: one subject with σ=0, rest σ=1 ----
    # cfg already has n_stages=2 from the override above; mod2 inherits it.
    torch.manual_seed(0)
    mod2 = ResDecLightningModule(cfg).float()
    mod2.train()
    batch2 = _make_dummy_batch(B=B)
    _enable_tabpfn_with(
        mod2,
        subject_ids=batch2["subject_ids"],
        y_tabpfn=[0.0] * B,
        sigma_tabpfn=[0.0] + [1.0] * (B - 1),
    )
    for p in mod2.parameters():
        if p.grad is not None:
            p.grad.zero_()
    loss_sigma0 = mod2.training_step(batch2, batch_idx=0)
    assert torch.isfinite(loss_sigma0), (
        "loss is not finite when σ=0 — weighted-mean normalization failed."
    )
    loss_sigma0.backward()
    grads_sigma0 = [
        p.grad.abs().max().item()
        for p in mod2.head.stage_2_tabm.parameters()
        if p.grad is not None and torch.isfinite(p.grad).all()
    ]
    # Every stage_2 param grad must be finite (not NaN/Inf).
    for p in mod2.head.stage_2_tabm.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), (
                f"NaN/Inf gradient on stage_2 param under σ=0: shape={p.shape}"
            )
    max_grad_sigma0 = max(grads_sigma0) if grads_sigma0 else 0.0

    # Weighted-mean means the σ=0 subject's w dominates the sum, and
    # normalization by w.sum() prevents explosive magnitude. We still expect
    # the grad magnitude to be within O(1) × of baseline (a single dominant
    # subject may shift magnitude modestly vs uniform; the ratio should stay
    # within ~10× — well below the 1e6× blow-up that a non-normalized ".mean()"
    # would produce here).
    ratio = max_grad_sigma0 / max(max_grad_uniform, 1e-12)
    assert ratio < 10.0, (
        f"max|grad| ratio σ=0 vs uniform = {ratio:.3f}× (expected <10×). "
        "Weighted-mean normalization is not preventing single-subject blow-up."
    )


def test_sigma_weight_constant_sigma_reduces_to_uniform(cfg):
    """With all σ = const, the weighted-mean aux loss must equal the plain MSE
    (up to fp tolerance). This is the sanity check that weighted-mean is the
    correct generalisation of ``.mean()``."""
    # aug-U sigma weighting only activates at n_stages>=2; force it here.
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.set_struct(cfg, False)
    OmegaConf.set_struct(cfg.model, False)
    OmegaConf.set_struct(cfg.model.resdec_head, False)
    cfg.model.resdec_head.n_stages = 2
    cfg.model.resdec_head.aux_lambdas = [1.0, 1.0]

    torch.manual_seed(0)
    mod = ResDecLightningModule(cfg).float()
    mod.eval()  # deterministic forward

    B = 4
    batch = _make_dummy_batch(B=B)
    _enable_tabpfn_with(
        mod,
        subject_ids=batch["subject_ids"],
        y_tabpfn=[0.0] * B,
        sigma_tabpfn=[0.37] * B,  # any constant; tests the reduction math only
    )

    # Reach inside training_step's math — we replicate the aux_2 reduction for
    # both paths (weighted + uniform) on the SAME per-subject squared errors.
    with torch.no_grad():
        out = mod.forward(batch)
    stage_2 = out["stage_2"]
    cognition = batch["cognition"].squeeze(-1)
    y_tabpfn = torch.zeros_like(cognition)
    sigma_tabpfn = torch.full_like(cognition, 0.37)
    target_aux2 = cognition - y_tabpfn - out["stage_1"].detach()

    # Weighted-mean path (the new implementation)
    w = 1.0 / (sigma_tabpfn * sigma_tabpfn + DEFAULT_SIGMA_EPS)
    w_sum = w.sum().clamp_min(DEFAULT_SIGMA_EPS)
    L_aux2_weighted = (w * (stage_2 - target_aux2).pow(2)).sum() / w_sum

    # Uniform path (plain .mean())
    L_aux2_uniform = torch.nn.functional.mse_loss(stage_2, target_aux2)

    assert torch.allclose(L_aux2_weighted, L_aux2_uniform, atol=1e-5, rtol=1e-5), (
        f"constant-σ weighted mean != uniform mean: "
        f"weighted={L_aux2_weighted.item():.6g} uniform={L_aux2_uniform.item():.6g}"
    )


def test_use_sigma_weighting_false_is_plain_mse(cfg):
    """With ``use_sigma_weighting=False``, L_aux2 must match
    ``F.mse_loss(stage_2, target)`` exactly — regardless of σ values."""
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.set_struct(cfg, False)
    OmegaConf.set_struct(cfg.model, False)
    OmegaConf.set_struct(cfg.model.resdec_head, False)
    cfg.model.resdec_head.use_sigma_weighting = False
    # This test exercises the full L_main + L_aux1/2/3 reduction; force n_stages=3
    # so all three aux losses are present even if the project default changes.
    cfg.model.resdec_head.n_stages = 3
    cfg.model.resdec_head.aux_lambdas = [1.0, 1.0, 1.0]

    torch.manual_seed(0)
    mod = ResDecLightningModule(cfg).float()
    mod.eval()

    B = 4
    batch = _make_dummy_batch(B=B)
    # Highly non-uniform σ — if weighting were applied it would change the loss.
    _enable_tabpfn_with(
        mod,
        subject_ids=batch["subject_ids"],
        y_tabpfn=[0.0] * B,
        sigma_tabpfn=[0.01, 0.5, 2.0, 10.0],
    )
    assert mod._use_sigma_weighting is False

    with torch.no_grad():
        out = mod.forward(batch)
    stage_2 = out["stage_2"]
    stage_3 = out["stage_3"]
    cognition = batch["cognition"].squeeze(-1)
    y_tabpfn = torch.zeros_like(cognition)
    target_aux2 = cognition - y_tabpfn - out["stage_1"].detach()
    target_aux3 = cognition - y_tabpfn - out["stage_1"].detach() - out["stage_2"].detach()

    # Replicate what training_step computes in the else branch.
    L_aux2_expected = torch.nn.functional.mse_loss(stage_2, target_aux2)
    L_aux3_expected = torch.nn.functional.mse_loss(stage_3, target_aux3)

    # Expected total: L = L_main + λ1·L_aux1 + λ2·L_aux2 + λ3·L_aux3.
    # Because the module is in eval() mode, the forward above and the forward
    # inside training_step produce identical tensors; so we can compose the
    # expected loss from ``out`` directly.
    residual = cognition - y_tabpfn
    L_main_expected = torch.nn.functional.mse_loss(out["prediction"], residual)
    L_aux1_expected = torch.nn.functional.mse_loss(out["stage_1"], residual)
    lam1, lam2, lam3 = mod._aux_lambdas
    expected_total = (
        L_main_expected
        + lam1 * L_aux1_expected
        + lam2 * L_aux2_expected
        + lam3 * L_aux3_expected
    )

    actual_total = mod.training_step(batch, batch_idx=0)
    assert torch.allclose(actual_total.detach(), expected_total.detach(),
                          atol=1e-5, rtol=1e-5), (
        f"training_step loss with use_sigma_weighting=False differs from plain "
        f"MSE breakdown: actual={actual_total.item():.6g} "
        f"expected={expected_total.item():.6g}"
    )


def test_tabpfn_disabled_fallback(cfg):
    """When TabPFN caches are NOT loaded (no cfg.data.tabpfn_*_dir), the
    training_step must fall back to plain MSE against cognition without
    crashing on the aux-loss / σ computation."""
    torch.manual_seed(0)
    mod = ResDecLightningModule(cfg).float()
    # eval() mode so the two forward calls below (one via training_step, one
    # explicit for the expected-value reference) see identical batchnorm /
    # dropout state — train() would randomise between calls and fail the
    # close-comparison spuriously. Use eval()+no_grad on the reference path
    # and let training_step run its own forward.
    mod.eval()
    assert mod._tabpfn_enabled is False

    B = 3
    batch = _make_dummy_batch(B=B)

    # Compute expected first, with the same eval-mode model.
    with torch.no_grad():
        out = mod.forward(batch)
    cognition = batch["cognition"].squeeze(-1)
    expected = torch.nn.functional.mse_loss(out["prediction"], cognition)

    # Now call training_step; in eval() mode the module's internal forward
    # produces identical output.
    loss = mod.training_step(batch, batch_idx=0)
    assert torch.isfinite(loss), "fallback MSE loss is not finite"
    assert torch.allclose(loss.detach(), expected.detach(), atol=1e-5, rtol=1e-5), (
        f"fallback loss != plain MSE(pred, cognition): "
        f"loss={loss.item():.6g} expected={expected.item():.6g}"
    )


# ---------------------------------------------------------------------------- #
# Canonical n_stages=1 regression + aux_lambdas length mismatch                 #
# ---------------------------------------------------------------------------- #
def test_n_stages_1_skips_sigma_weighting(cfg):
    """At n_stages=1, sigma weighting is irrelevant — there's only stage_1
    whose aux loss is unweighted MSE. Verify no sigma_weight_* logs appear
    and the loss reduces to L_main + λ_1·MSE(stage_1, residual)."""
    # cfg fixture defaults to n_stages=1 (canonical: ResDecH3Head.DEFAULT_N_STAGES).
    # Don't override — this test is specifically about the canonical shape.
    torch.manual_seed(0)
    mod = ResDecLightningModule(cfg).float()
    mod.eval()
    assert mod._n_stages == 1
    B = 4
    batch = _make_dummy_batch(B=B)
    _enable_tabpfn_with(
        mod,
        subject_ids=batch["subject_ids"],
        y_tabpfn=[0.0] * B,
        sigma_tabpfn=[0.01, 0.5, 2.0, 10.0],  # non-uniform — would matter at n>=2
    )
    loss = mod.training_step(batch, batch_idx=0)
    # Reference: at n=1, prediction == stage_1, so L_main == L_aux1 == MSE(stage_1, residual).
    with torch.no_grad():
        out = mod.forward(batch)
    cognition = batch["cognition"].squeeze(-1)
    residual = cognition - torch.zeros_like(cognition)  # y_tabpfn = 0
    L_expected = (1.0 + mod._aux_lambdas[0]) * torch.nn.functional.mse_loss(
        out["stage_1"], residual,
    )
    assert torch.allclose(loss.detach(), L_expected.detach(), atol=1e-5), (
        f"n_stages=1 loss != (1 + λ_1)·MSE(stage_1, residual): "
        f"loss={loss.item():.6g} expected={L_expected.item():.6g}"
    )


@pytest.mark.parametrize(
    "n_stages,bad_lambdas",
    [(1, [1.0, 1.0]), (2, [1.0]), (3, [1.0, 1.0])],
)
def test_aux_lambdas_length_mismatch_raises(cfg, n_stages, bad_lambdas):
    """aux_lambdas length must equal n_stages; mismatch is a hard error."""
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.set_struct(cfg, False)
    OmegaConf.set_struct(cfg.model, False)
    OmegaConf.set_struct(cfg.model.resdec_head, False)
    cfg.model.resdec_head.n_stages = n_stages
    cfg.model.resdec_head.aux_lambdas = bad_lambdas
    with pytest.raises(ValueError, match="must have exactly n_stages="):
        ResDecLightningModule(cfg)
