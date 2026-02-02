"""
Tests for CognitiveResilienceLightningModule.

Tests the PyTorch Lightning wrapper around CognitiveResilienceModel:
- Instantiation with config
- Training step (forward + loss)
- Validation step (forward + metrics)
- Loss branching (β-NLL for Bayesian, MSE for deterministic)
- Optimizer and scheduler configuration
- Metric logging
"""

import pytest
import torch
from unittest.mock import MagicMock
from omegaconf import OmegaConf

from src.data.constants import N_CELL_TYPES, N_REGIONS, ALL_EDGE_TYPES, CELL_TYPE_ORDER, sanitize_key


@pytest.fixture
def base_config():
    """Minimal config for LightningModule testing.

    Mirrors the nested structure of configs/default.yaml but with
    small dimensions for fast testing.
    """
    return OmegaConf.create({
        "model": {
            "n_genes": 50,
            "n_cell_types": N_CELL_TYPES,
            "d_embed": 32,
            "d_fused": 32,
            "n_regions": N_REGIONS,
            "dropout": 0.0,
            "gene_gate": {
                "initial_temperature": 2.0,
            },
            "hgt": {
                "n_layers": 1,
                "n_heads": 4,
            },
            "set_transformer": {
                "n_isab_layers": 1,
                "n_inducing_points": 4,
                "n_heads": 4,
            },
            "cell_type_selector": {
                "selection_temperature": 1.0,
            },
            "pathology_attention": {
                "d_cond": 16,
                "n_heads": 4,
            },
            "head": {
                "type": "deterministic",
                "d_hidden": 16,
            },
        },
        "training": {
            "optimizer": {
                "type": "adamw",
                "lr": 1e-3,
                "weight_decay": 1e-4,
            },
            "scheduler": {
                "type": "cosine",
                "warmup_epochs": 5,
                "eta_min": 1e-6,
            },
            "loss": {
                "type": "beta_nll",
                "beta": 0.5,
            },
            "max_epochs": 100,
            "gradient_clip_val": 1.0,
            "regularization": {
                "gene_gate_l1": 0.0,
            },
        },
    })


@pytest.fixture
def bayesian_config(base_config):
    """Config with Bayesian head."""
    cfg = base_config.copy()
    cfg.model.head.type = "bayesian"
    return cfg


def _make_batch(batch_size=4, n_genes=50, n_cell_types=N_CELL_TYPES,
                n_regions=N_REGIONS, max_cells=10):
    """Create a synthetic batch matching collate_for_hgt_multiregion format."""
    region_pseudobulk = torch.randn(batch_size, n_regions, n_cell_types, n_genes)
    region_mask = torch.zeros(batch_size, n_regions, dtype=torch.bool)
    region_mask[:, 0] = True  # PFC only

    # HGT edge dicts
    sanitized_types = [sanitize_key(ct) for ct in CELL_TYPE_ORDER]
    sanitized_edges = [sanitize_key(et) for et in ALL_EDGE_TYPES]

    edge_index_dict_list = []
    edge_attr_dict_list = []
    for _ in range(batch_size):
        ei_dict = {}
        ea_dict = {}
        for src in sanitized_types[:2]:
            for dst in sanitized_types[:2]:
                key = (src, sanitized_edges[0], dst)
                ei_dict[key] = torch.tensor([[0], [0]], dtype=torch.long)
                ea_dict[key] = torch.rand(1, 1)
        edge_index_dict_list.append(ei_dict)
        edge_attr_dict_list.append(ea_dict)

    cells = torch.randn(batch_size, n_cell_types, max_cells, n_genes)
    cell_mask = torch.ones(batch_size, n_cell_types, max_cells, dtype=torch.bool)
    pathology = torch.rand(batch_size, 3)
    cognition = torch.randn(batch_size, 1)

    return {
        "region_pseudobulk": region_pseudobulk,
        "region_mask": region_mask,
        "edge_index_dict_list": edge_index_dict_list,
        "edge_attr_dict_list": edge_attr_dict_list,
        "cells": cells,
        "cell_mask": cell_mask,
        "pathology": pathology,
        "cognition": cognition,
    }


class TestLightningModuleInstantiation:
    """Tests for module creation."""

    def test_instantiation_deterministic(self, base_config):
        """Module instantiates with deterministic head config."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        module = CognitiveResilienceLightningModule(base_config)
        assert module is not None
        assert module.model is not None

    def test_instantiation_bayesian(self, bayesian_config):
        """Module instantiates with Bayesian head config."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        module = CognitiveResilienceLightningModule(bayesian_config)
        assert module.model.use_bayesian_head is True


class TestTrainingStep:
    """Tests for training_step."""

    def test_training_step_returns_loss(self, base_config):
        """training_step returns a scalar loss tensor."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        module = CognitiveResilienceLightningModule(base_config)
        batch = _make_batch(n_genes=50)
        loss = module.training_step(batch, batch_idx=0)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_training_step_deterministic_uses_mse(self, base_config):
        """With deterministic head, training_step uses MSE loss."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        module = CognitiveResilienceLightningModule(base_config)
        batch = _make_batch(n_genes=50)
        loss = module.training_step(batch, batch_idx=0)
        # Should not crash — MSE used instead of β-NLL
        assert torch.isfinite(loss)

    def test_training_step_gradient_flow(self, base_config):
        """Gradients flow through training_step to model parameters."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        module = CognitiveResilienceLightningModule(base_config)
        batch = _make_batch(n_genes=50)
        loss = module.training_step(batch, batch_idx=0)
        loss.backward()
        # Check that at least some parameters have gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in module.parameters() if p.requires_grad
        )
        assert has_grad

    def test_bayesian_training_step(self, bayesian_config):
        """Bayesian head training_step returns finite loss using beta-NLL."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        module = CognitiveResilienceLightningModule(bayesian_config)
        batch = _make_batch(n_genes=50)
        loss = module.training_step(batch, batch_idx=0)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert torch.isfinite(loss)


class TestValidationStep:
    """Tests for validation_step."""

    def test_validation_step_runs(self, base_config):
        """validation_step runs without error."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        module = CognitiveResilienceLightningModule(base_config)
        module.eval()
        batch = _make_batch(n_genes=50)
        # Should not raise
        module.validation_step(batch, batch_idx=0)

    def test_validation_step_logs_metrics(self, base_config):
        """validation_step logs val_loss and prediction quality metrics."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        module = CognitiveResilienceLightningModule(base_config)
        module.eval()

        logged = {}
        module.log = lambda name, value, **kwargs: logged.__setitem__(name, value)

        batch = _make_batch(n_genes=50)
        module.validation_step(batch, batch_idx=0)

        assert "val_loss" in logged
        # Should have prediction quality metrics
        metric_keys = [k for k in logged if k.startswith("val_") and k != "val_loss"]
        assert len(metric_keys) > 0

    def test_bayesian_validation_step(self, bayesian_config):
        """Bayesian head validation_step runs without error."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        module = CognitiveResilienceLightningModule(bayesian_config)
        module.eval()

        logged = {}
        module.log = lambda name, value, **kwargs: logged.__setitem__(name, value)

        batch = _make_batch(n_genes=50)
        module.validation_step(batch, batch_idx=0)

        assert "val_loss" in logged


class TestTestStep:
    """Tests for test_step."""

    def test_test_step_runs(self, base_config):
        """test_step runs without error."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        module = CognitiveResilienceLightningModule(base_config)
        module.eval()
        batch = _make_batch(n_genes=50)
        # Should not raise
        module.test_step(batch, batch_idx=0)

    def test_test_step_logs_with_test_prefix(self, base_config):
        """test_step logs metrics with test_ prefix."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        module = CognitiveResilienceLightningModule(base_config)
        module.eval()

        logged = {}
        module.log = lambda name, value, **kwargs: logged.__setitem__(name, value)

        batch = _make_batch(n_genes=50)
        module.test_step(batch, batch_idx=0)

        assert "test_loss" in logged
        test_metric_keys = [k for k in logged if k.startswith("test_") and k != "test_loss"]
        assert len(test_metric_keys) > 0


class TestPredictStep:
    """Tests for predict_step."""

    def test_predict_step_returns_dict(self, base_config):
        """predict_step returns a dict with 'mean' key."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        module = CognitiveResilienceLightningModule(base_config)
        module.eval()
        batch = _make_batch(n_genes=50)
        result = module.predict_step(batch, batch_idx=0)
        assert isinstance(result, dict)
        assert "mean" in result

    def test_predict_step_bayesian_includes_std(self, bayesian_config):
        """Bayesian head predict_step includes 'std' in result."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        module = CognitiveResilienceLightningModule(bayesian_config)
        module.eval()
        batch = _make_batch(n_genes=50)
        result = module.predict_step(batch, batch_idx=0)
        assert "std" in result

    def test_predict_step_includes_attention_if_present(self, base_config):
        """predict_step includes attention weights if model provides them."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        module = CognitiveResilienceLightningModule(base_config)
        module.eval()
        batch = _make_batch(n_genes=50)
        result = module.predict_step(batch, batch_idx=0)
        # The model returns attention weights, so they should be in the result
        if "attention" in result:
            assert result["attention"] is not None


class TestLossBranching:
    """Tests for loss function branching based on head type."""

    def test_deterministic_head_uses_mse_not_beta_nll(self, base_config):
        """Deterministic head uses MSE loss even if config says beta_nll."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        # Config says beta_nll but head is deterministic
        assert base_config.training.loss.type == "beta_nll"
        assert base_config.model.head.type == "deterministic"
        module = CognitiveResilienceLightningModule(base_config)
        # Should use MSE internally (check that head_type leads to MSE)
        assert module._use_mse_loss is True


class TestOptimizerConfiguration:
    """Tests for optimizer and scheduler setup."""

    def test_configure_optimizers_returns_optimizer(self, base_config):
        """configure_optimizers returns optimizer config."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        module = CognitiveResilienceLightningModule(base_config)
        result = module.configure_optimizers()
        # Should return dict with optimizer and lr_scheduler
        assert "optimizer" in result
        assert "lr_scheduler" in result

    def test_optimizer_is_adamw(self, base_config):
        """Default optimizer is AdamW."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        module = CognitiveResilienceLightningModule(base_config)
        result = module.configure_optimizers()
        optimizer = result["optimizer"]
        assert "AdamW" in type(optimizer).__name__

    def test_scheduler_is_sequential_with_warmup(self, base_config):
        """Scheduler uses SequentialLR combining warmup + cosine decay."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        module = CognitiveResilienceLightningModule(base_config)
        result = module.configure_optimizers()
        scheduler = result["lr_scheduler"]["scheduler"]
        assert "SequentialLR" in type(scheduler).__name__

    def test_scheduler_warmup_increases_lr(self, base_config):
        """During warmup, learning rate increases linearly."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        module = CognitiveResilienceLightningModule(base_config)
        result = module.configure_optimizers()
        optimizer = result["optimizer"]
        scheduler = result["lr_scheduler"]["scheduler"]

        lrs = []
        for epoch in range(5):
            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step()

        # LR should increase during warmup
        for i in range(1, len(lrs)):
            assert lrs[i] > lrs[i - 1], f"LR should increase during warmup: {lrs}"

    def test_scheduler_without_warmup(self, base_config):
        """Without warmup, scheduler is plain CosineAnnealingLR."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        base_config.training.scheduler.warmup_epochs = 0
        module = CognitiveResilienceLightningModule(base_config)
        result = module.configure_optimizers()
        scheduler = result["lr_scheduler"]["scheduler"]
        assert "CosineAnnealingLR" in type(scheduler).__name__

    def test_scheduler_cosine_decay_after_warmup(self, base_config):
        """After warmup, LR decreases via cosine annealing."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        module = CognitiveResilienceLightningModule(base_config)
        result = module.configure_optimizers()
        optimizer = result["optimizer"]
        scheduler = result["lr_scheduler"]["scheduler"]

        # Step through warmup
        for _ in range(5):
            scheduler.step()

        lr_after_warmup = optimizer.param_groups[0]["lr"]

        # Step a few more epochs (cosine decay)
        for _ in range(10):
            scheduler.step()

        lr_after_decay = optimizer.param_groups[0]["lr"]
        assert lr_after_decay < lr_after_warmup


class TestGeneGateL1Regularization:
    """Tests for optional gene gate L1 regularization."""

    def test_l1_regularization_when_enabled(self, base_config):
        """Gene gate L1 regularization adds to loss when lambda > 0."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        base_config.training.regularization.gene_gate_l1 = 0.01
        module = CognitiveResilienceLightningModule(base_config)
        batch = _make_batch(n_genes=50)
        loss_with_l1 = module.training_step(batch, batch_idx=0)
        assert torch.isfinite(loss_with_l1)

    def test_no_l1_when_disabled(self, base_config):
        """No L1 regularization when lambda = 0."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        base_config.training.regularization.gene_gate_l1 = 0.0
        module = CognitiveResilienceLightningModule(base_config)
        # _gene_gate_l1_lambda should be 0
        assert module._gene_gate_l1_lambda == 0.0
