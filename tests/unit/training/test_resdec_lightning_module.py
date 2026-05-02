"""Tests for ResDecLightningModule (single-stage composer).

Smoke tests that verify:
- Module builds (encoder + ResDecMHEHead) from a config extended with resdec_head.
- Forward pass on a dummy batch returns a dict with `prediction` [B] and `latent_1` [B, d_subject].
- Missing `metadata` key in the batch is handled by a zero-tensor placeholder.

These tests do NOT run .fit() — training is exercised elsewhere. They only
confirm the wiring is correct.
"""
from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from src.training.resdec_lightning_module import ResDecLightningModule
from tests.conftest import _build_canonical_batch


@pytest.fixture
def cfg(default_config_path):
    base = OmegaConf.load(default_config_path)
    OmegaConf.set_struct(base, False)
    OmegaConf.set_struct(base.model, False)
    base.model.n_genes = 4785
    base.model.n_cell_types = 31
    # Deterministic head keeps the smoke test simple: avoids Pyro SVI
    # machinery that has no bearing on the ResDecMHEHead wiring we're verifying.
    base.model.head = OmegaConf.create({"type": "deterministic", "d_hidden": 32})
    # Add required resdec_head section
    base.model.resdec_head = OmegaConf.create({"d_metadata": 8, "n_heads": 4})
    # Training overrides for the resdec optimizer path
    base.training.lr = 0.0015
    base.training.weight_decay = 5.6e-6
    return base

def test_module_builds(cfg):
    mod = ResDecLightningModule(cfg)
    assert mod.encoder is not None
    assert mod.head is not None
    # d_subject inferred from d_fused (64 in default.yaml)
    assert mod.head.d_subject == cfg.model.d_fused

def test_module_forward_dummy_batch(cfg):
    mod = ResDecLightningModule(cfg)
    mod.eval()

    B = 2
    batch = _build_canonical_batch(
        batch_size=B, n_genes=4785, cells_per_ct=10, edges_per_subj=50,
    )

    with torch.no_grad():
        out = mod(batch)

    assert "prediction" in out
    assert "latent_1" in out
    assert out["prediction"].shape == (B,)
    assert out["latent_1"].shape == (B, cfg.model.d_fused)


def test_prediction_head_is_frozen(cfg):
    mod = ResDecLightningModule(cfg)
    # All prediction_head params should have requires_grad=False
    if hasattr(mod.encoder, "prediction_head"):
        for p in mod.encoder.prediction_head.parameters():
            assert p.requires_grad is False


def test_module_handles_missing_metadata(cfg):
    """Fallback placeholder: missing `metadata` → zero tensor of shape [B, d_metadata]."""
    mod = ResDecLightningModule(cfg)
    mod.eval()

    # NOTE: no "metadata" key → zero-tensor placeholder kicks in
    # Empty CCC graphs (edges_per_subj=0) match the original test fixture.
    B = 2
    batch = _build_canonical_batch(
        batch_size=B, n_genes=4785, cells_per_ct=10, edges_per_subj=0,
    )

    with torch.no_grad():
        out = mod(batch)
    assert out["prediction"].shape == (B,)

# ---------------------------------------------------------------------------
# §31.7 fix: training-time attention path
# ---------------------------------------------------------------------------
# After the 2026-04-28 refactor, the no_grad einsum+softmax block in
# pathology_attention.py is eval-only. Training-time differentiable attention
# weights are available ONLY via the in-graph einsum path
# (compute_attention_with_grad=True), which is wired by
# ResDecLightningModule.__init__ when attention_regularization.enabled=True.
# When regularization is disabled, no attention_weights are emitted during
# training (no_grad block skipped + einsum path off).

def _build_dummy_train_batch(B: int = 2):
    """Construct a minimal train batch (cell_data path, no metadata).

    Uses cells_per_ct=2 for fast tests + empty CCC graphs.
    """
    return _build_canonical_batch(
        batch_size=B, n_genes=4785, cells_per_ct=2, edges_per_subj=0,
    )

def test_training_attention_weights_none_when_reg_disabled(cfg):
    """§31.7 fix: with attention_regularization.enabled=False (default), the
    encoder runs the canonical SDPA path during training and returns None for
    attention_weights — the no_grad einsum re-compute block is bypassed."""
    OmegaConf.set_struct(cfg, False)
    OmegaConf.set_struct(cfg.model, False)
    # Even if the user constructed with return_attention_in_training=True,
    # ResDecLightningModule.__init__ flips compute_attention_with_grad=False
    # when regularization is disabled.
    cfg.model.return_attention_in_training = True
    cfg.training.attention_regularization = OmegaConf.create({"enabled": False})

    mod = ResDecLightningModule(cfg)
    mod.train()

    # SDPA fast path retained even with the encoder flag set
    assert mod.encoder.pathology_attention.compute_attention_with_grad is False

    batch = _build_dummy_train_batch(B=2)
    out = mod(batch)

    # Training-mode forward: attention_weights must NOT be present (no_grad
    # block was the only producer in the SDPA path, and it is now eval-only).
    assert "attention_weights" not in out, (
        "attention_weights must be absent during training when "
        "attention_regularization is disabled (§31.7 fix)."
    )

def test_training_attention_weights_present_when_reg_enabled(cfg):
    """When attention_regularization.enabled=True, the lightning module wires
    compute_attention_with_grad=True so the encoder's in-graph einsum path
    emits differentiable attention weights during training."""
    OmegaConf.set_struct(cfg, False)
    OmegaConf.set_struct(cfg.model, False)
    cfg.model.return_attention_in_training = True
    cfg.training.attention_regularization = OmegaConf.create(
        {"enabled": True, "scheme": "entropy_bonus", "weight": 1e-3},
    )

    mod = ResDecLightningModule(cfg)
    mod.train()

    # In-graph einsum path is active so the regularizer's gradient flows
    assert mod.encoder.pathology_attention.compute_attention_with_grad is True

    batch = _build_dummy_train_batch(B=2)
    out = mod(batch)

    assert "attention_weights" in out, (
        "attention_weights must be returned during training when "
        "attention_regularization is enabled."
    )
    assert out["attention_weights"] is not None
    # Shape: [B, n_heads, n_cell_types]; n_heads from cfg.model.n_attention_heads
    assert out["attention_weights"].dim() == 3
    assert out["attention_weights"].shape[0] == 2
    assert out["attention_weights"].shape[2] == cfg.model.n_cell_types
    # Differentiable (in-graph einsum path)
    assert out["attention_weights"].requires_grad is True
