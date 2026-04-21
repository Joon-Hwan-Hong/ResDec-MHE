"""Tests for ResDecLightningModule (Phase 1 single-stage composer).

Smoke tests that verify:
- Module builds (encoder + ResDecH3Head) from a config extended with resdec_head.
- Forward pass on a dummy batch returns a dict with `prediction` [B] and `latent_1` [B, d_subject].
- Missing `metadata` key in the batch is handled by the Phase-1 zero-tensor placeholder.

This task (1.9a) does NOT run .fit() — training is exercised in a separate
downstream task. These tests only confirm the wiring is correct.
"""
from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from src.training.resdec_lightning_module import ResDecLightningModule


@pytest.fixture
def cfg():
    base = OmegaConf.load("configs/default.yaml")
    OmegaConf.set_struct(base, False)
    OmegaConf.set_struct(base.model, False)
    base.model.n_genes = 4785
    base.model.n_cell_types = 31
    # Deterministic head keeps the Phase-1 smoke test simple: avoids Pyro SVI
    # machinery that has no bearing on the ResDecH3Head wiring we're verifying.
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
    N_CT, N_GENES, N_REGIONS = 31, 4785, 6
    cells_per_ct = 10
    cells_per_subject = cells_per_ct * N_CT

    region_mask = torch.zeros(B, N_REGIONS, dtype=torch.bool)
    region_mask[:, 0] = True  # PFC-only, matches real data distribution

    # [B, N_CT+1] offsets: subject i starts at i * cells_per_subject
    offsets_per_subj = torch.arange(0, cells_per_subject + 1, cells_per_ct, dtype=torch.long)
    subj_offsets = torch.arange(B, dtype=torch.long) * cells_per_subject
    cell_offsets = subj_offsets.unsqueeze(1) + offsets_per_subj.unsqueeze(0)

    edges_per_subj = 50
    total_edges = B * edges_per_subj

    batch = {
        "region_pseudobulk": torch.randn(B, N_REGIONS, N_CT, N_GENES),
        "region_mask": region_mask,
        "ccc_edge_index": torch.randint(0, N_CT, (2, total_edges)),
        "ccc_edge_type": torch.randint(0, 5, (total_edges,)),
        "ccc_edge_attr": torch.rand(total_edges, 1),
        "cell_type_mask": torch.ones(B, N_CT, dtype=torch.bool),
        "cell_data": torch.randn(B * cells_per_subject, N_GENES),
        "cell_offsets": cell_offsets,
        "pathology": torch.randn(B, 3),
        "cognition": torch.randn(B, 1),
    }

    with torch.no_grad():
        out = mod(batch)

    assert "prediction" in out
    assert "latent_1" in out
    assert out["prediction"].shape == (B,)
    assert out["latent_1"].shape == (B, cfg.model.d_fused)


def test_module_handles_missing_metadata(cfg):
    """Phase-1 placeholder: missing `metadata` → zero tensor of shape [B, d_metadata]."""
    mod = ResDecLightningModule(cfg)
    mod.eval()

    B = 2
    N_CT, N_GENES, N_REGIONS = 31, 4785, 6
    cells_per_ct = 10
    cells_per_subject = cells_per_ct * N_CT

    region_mask = torch.zeros(B, N_REGIONS, dtype=torch.bool)
    region_mask[:, 0] = True
    offsets_per_subj = torch.arange(0, cells_per_subject + 1, cells_per_ct, dtype=torch.long)
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
        # NOTE: no "metadata" key → Phase-1 placeholder kicks in
    }

    with torch.no_grad():
        out = mod(batch)
    assert out["prediction"].shape == (B,)
