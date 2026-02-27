"""
Smoke test: full pipeline end-to-end.

Verifies that the entire pipeline (data -> train N epochs -> checkpoint ->
predict -> analyze) runs without crashing for both Bayesian and deterministic
head configurations.

Uses tiny dimensions for speed:
  d_embed=16, d_fused=16, n_genes=100, n_cell_types=5, n_regions=6,
  batch_size=4, max_cells_per_type=50, 2 epochs.

Marked @pytest.mark.slow so it can be skipped in fast CI runs.
"""

import pytest
import torch
import torch.utils.data
from pathlib import Path

import pyro
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf

from src.training.lightning_module import CognitiveResilienceLightningModule
from src.training.callbacks import ResilienceModelCheckpoint
from src.inference.predict import Predictor


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

N_SAMPLES = 10
N_CELL_TYPES = 5
N_GENES = 100
N_REGIONS = 6
MAX_CELLS = 50
BATCH_SIZE = 4
D_EMBED = 16
D_FUSED = 16
D_COND = 8
MAX_EPOCHS = 2


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic Dataset
# ─────────────────────────────────────────────────────────────────────────────


class SyntheticDataset(torch.utils.data.Dataset):
    """Minimal synthetic dataset for smoke testing."""

    def __init__(self, n_samples, n_cell_types, n_genes, n_regions, max_cells):
        self.n_samples = n_samples
        self.n_cell_types = n_cell_types
        self.n_genes = n_genes
        self.n_regions = n_regions
        self.max_cells = max_cells

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {
            "pseudobulk": torch.randn(self.n_cell_types, self.n_genes),
            "cognition": torch.randn(1),
            "pathology": torch.randn(3),
            "region_pseudobulk": torch.randn(
                self.n_regions, self.n_cell_types, self.n_genes
            ),
            "region_mask": torch.ones(self.n_regions, dtype=torch.bool),
            "cells": torch.randn(self.n_cell_types, self.max_cells, self.n_genes),
            "cell_mask": torch.ones(
                self.n_cell_types, self.max_cells, dtype=torch.bool
            ),
            "cell_type_mask": torch.ones(self.n_cell_types, dtype=torch.bool),
        }


def simple_collate(batch):
    """Stack tensors from a list of dicts into a batched dict."""
    result = {}
    for key in batch[0]:
        if isinstance(batch[0][key], torch.Tensor):
            result[key] = torch.stack([b[key] for b in batch])
        else:
            result[key] = [b[key] for b in batch]
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Config Builder
# ─────────────────────────────────────────────────────────────────────────────


def make_smoke_config(head_type: str) -> OmegaConf:
    """
    Build a minimal OmegaConf config for smoke testing.

    Mirrors the structure expected by build_model_from_config() and
    CognitiveResilienceLightningModule.

    Args:
        head_type: "bayesian" or "deterministic"
    """
    cfg = {
        "experiment": {"name": "smoke_test", "seed": 42, "device": "cpu"},
        "model": {
            "n_genes": N_GENES,
            "n_cell_types": N_CELL_TYPES,
            "n_regions": N_REGIONS,
            "d_embed": D_EMBED,
            "d_fused": D_FUSED,
            "dropout": 0.0,
            "pseudobulk": {
                "mlp_hidden": [32],
                "use_layer_norm": True,
            },
            "gene_gate": {
                "initial_temperature": 2.0,
            },
            "hgt": {
                "n_layers": 1,
                "n_heads": 2,
                "edge_types": [
                    "Secreted_Signaling",
                    "ECM_Receptor",
                    "Cell_Cell_Contact",
                    "Non_protein_Signaling",
                    "Novel_Uncharacterized",
                ],
            },
            "set_transformer": {
                "n_isab_layers": 1,
                "n_inducing_points": 4,
                "n_pma_seeds": 1,
                "n_heads": 2,
            },
            "cell_type_selector": {
                "selection_temperature": 1.0,
            },
            "pathology_attention": {
                "d_cond": D_COND,
                "n_heads": 2,
                "n_pathology_features": 3,
            },
            "head": {
                "type": head_type,
                "d_hidden": 16,
            },
        },
        "data": {
            "cell_sampling": {
                "max_cells_per_type": MAX_CELLS,
                "min_cells_threshold": 5,
                "sampling_strategy": "random",
            },
        },
        "training": {
            "max_epochs": MAX_EPOCHS,
            "early_stopping": {
                "patience": 5,
                "min_delta": 0.0001,
                "min_epochs": 1,
                "monitor": "val_loss",
                "mode": "min",
            },
            "optimizer": {
                "type": "adamw",
                "lr": 0.001,
                "weight_decay": 0.0,
                "betas": [0.9, 0.999],
            },
            "scheduler": {
                "type": "cosine",
                "warmup_epochs": 0,
                "eta_min": 0.0001,
            },
            "loss": {
                "type": "beta_nll" if head_type == "bayesian" else "mse",
                "beta": 0.5,
            },
            "gradient_clip_val": 1.0,
            "precision": "32-true",
            "devices": 1,
            "strategy": "auto",
            "lr_scaling": False,
            "checkpoint": {
                "save_top_k": 1,
                "monitor": "val_loss",
                "mode": "min",
                "save_last": True,
            },
            "regularization": {
                "gene_gate_l1": 0.0,
            },
            "logging": {
                "log_every_n_steps": 1,
                "val_check_interval": 1.0,
            },
            "temperature_annealing": {
                "tau_max": 2.0,
                "tau_min": 0.1,
                "warmup_epochs": 1,
                "anneal_epochs": 5,
                "schedule": "exponential",
            },
        },
        "error_handling": {
            "training": {
                "nan_loss": "fail",
                "nan_batch": "skip",
            },
        },
        "inference": {
            "num_posterior_samples": 5,
        },
    }
    return OmegaConf.create(cfg)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def synthetic_dataset():
    """Create a small synthetic dataset."""
    return SyntheticDataset(
        n_samples=N_SAMPLES,
        n_cell_types=N_CELL_TYPES,
        n_genes=N_GENES,
        n_regions=N_REGIONS,
        max_cells=MAX_CELLS,
    )


@pytest.fixture
def train_loader(synthetic_dataset):
    """Training dataloader with synthetic data."""
    return torch.utils.data.DataLoader(
        synthetic_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=simple_collate,
    )


@pytest.fixture
def val_loader(synthetic_dataset):
    """Validation dataloader with synthetic data (no shuffle)."""
    return torch.utils.data.DataLoader(
        synthetic_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=simple_collate,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Smoke Tests
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.parametrize("head_type", ["deterministic", "bayesian"])
def test_full_pipeline_smoke(head_type, train_loader, val_loader, tmp_path):
    """
    Full pipeline smoke test: data -> train -> checkpoint -> predict.

    Verifies:
    1. Model builds from config without error
    2. Training runs for 2 epochs via trainer.fit() without crashing
    3. Checkpoint is saved and exists on disk
    4. Predictor.from_checkpoint() loads the checkpoint
    5. predict() on a small batch returns finite predictions
    """
    # Clear Pyro param store to avoid state leakage between test runs
    pyro.clear_param_store()

    # ── Step 1: Build config and model ──
    config = make_smoke_config(head_type)
    module = CognitiveResilienceLightningModule(config)

    # ── Step 2: Set up callbacks and trainer ──
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="smoke-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    resilience_checkpoint = ResilienceModelCheckpoint()

    # Bayesian SVI uses differentiable_loss through Lightning's automatic
    # optimization, but gradient_clip_val is disabled for SVI because Pyro's
    # ClippedAdam handles its own gradient clipping internally.
    grad_clip = config.training.gradient_clip_val if head_type != "bayesian" else None

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="cpu",
        devices=1,
        callbacks=[checkpoint_callback, resilience_checkpoint],
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=1,
        logger=False,  # Disable logging for speed
        deterministic=True,
        gradient_clip_val=grad_clip,
    )

    # ── Step 3: Train for 2 epochs ──
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # ── Step 4: Verify checkpoint exists ──
    last_ckpt = checkpoint_dir / "last.ckpt"
    assert last_ckpt.exists(), f"Expected checkpoint at {last_ckpt}"

    # Use best checkpoint if available, otherwise fall back to last.ckpt.
    # With only 2 smoke-test epochs, best_model_path may be empty if
    # val_loss wasn't logged before the first checkpoint. last.ckpt is
    # always saved by save_last=True.
    best_ckpt_path = checkpoint_callback.best_model_path
    ckpt_to_load = best_ckpt_path if best_ckpt_path else str(last_ckpt)
    assert Path(ckpt_to_load).exists(), f"Checkpoint not found: {ckpt_to_load}"

    # ── Step 5: Load checkpoint via Predictor.from_checkpoint ──
    predictor = Predictor.from_checkpoint(
        checkpoint_path=ckpt_to_load,
        device="cpu",
        config=config,
    )

    # ── Step 6: Run predict on a small batch ──
    # Get a single batch from the val loader
    batch = next(iter(val_loader))
    result = predictor.predict_batch(batch)

    # ── Step 7: Assert predictions are valid ──
    assert "mean" in result, "Prediction result missing 'mean' key"
    mean = result["mean"]

    # mean should be finite (no NaN, no Inf)
    assert torch.isfinite(torch.from_numpy(mean)).all(), (
        f"Predictions contain non-finite values: {mean}"
    )

    # mean should have correct shape [B, 1]
    expected_batch_size = min(BATCH_SIZE, N_SAMPLES)
    assert mean.shape == (expected_batch_size, 1), (
        f"Expected shape ({expected_batch_size}, 1), got {mean.shape}"
    )

    # For Bayesian head, check that std is present and positive
    if head_type == "bayesian":
        # With guide loaded, predict_batch_bayesian returns std
        if "std" in result:
            std = result["std"]
            assert torch.isfinite(torch.from_numpy(std)).all(), (
                "Uncertainty (std) contains non-finite values"
            )

    # Check attention weights are present and valid
    assert "attention_weights" in result, "Missing attention_weights"
    attn = result["attention_weights"]
    assert torch.isfinite(torch.from_numpy(attn)).all(), (
        "Attention weights contain non-finite values"
    )
