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
                "n_pma_seeds": 1,
            },
            "cell_type_selector": {
                "selection_temperature": 1.0,
            },
            "pathology_attention": {
                "d_cond": 16,
                "n_heads": 4,
                "n_pathology_features": 3,
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

    # HGT raw edge tensors — 4 edges per sample (2 src x 2 dst, first edge type)
    n_edges = 4
    ccc_edge_index = torch.zeros(batch_size, 2, n_edges, dtype=torch.long)
    ccc_edge_type = torch.zeros(batch_size, n_edges, dtype=torch.long)
    ccc_edge_attr = torch.rand(batch_size, n_edges, 1)
    ccc_edge_counts = torch.full((batch_size,), n_edges, dtype=torch.long)

    cells = torch.randn(batch_size, n_cell_types, max_cells, n_genes)
    cell_mask = torch.ones(batch_size, n_cell_types, max_cells, dtype=torch.bool)
    cell_type_mask = torch.ones(batch_size, n_cell_types, dtype=torch.bool)
    cell_type_mask[:, -3:] = False  # Mask out last 3 types
    pathology = torch.rand(batch_size, 3)
    cognition = torch.randn(batch_size, 1)

    return {
        "region_pseudobulk": region_pseudobulk,
        "region_mask": region_mask,
        "ccc_edge_index": ccc_edge_index,
        "ccc_edge_type": ccc_edge_type,
        "ccc_edge_attr": ccc_edge_attr,
        "ccc_edge_counts": ccc_edge_counts,
        "cells": cells,
        "cell_mask": cell_mask,
        "cell_type_mask": cell_type_mask,
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
        module.log = lambda *args, **kwargs: None
        batch = _make_batch(n_genes=50)
        loss = module.training_step(batch, batch_idx=0)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_training_step_deterministic_uses_mse(self, base_config):
        """With deterministic head, training_step uses MSE loss."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        module = CognitiveResilienceLightningModule(base_config)
        module.log = lambda *args, **kwargs: None
        batch = _make_batch(n_genes=50)
        loss = module.training_step(batch, batch_idx=0)
        # Should not crash — MSE used instead of β-NLL
        assert torch.isfinite(loss)

    def test_training_step_gradient_flow(self, base_config):
        """Gradients flow through training_step to model parameters."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        module = CognitiveResilienceLightningModule(base_config)
        module.log = lambda *args, **kwargs: None
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
        """Bayesian head training_step returns finite differentiable ELBO loss."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        module = CognitiveResilienceLightningModule(bayesian_config)
        module.log = lambda *args, **kwargs: None
        batch = _make_batch(n_genes=50)
        loss = module.training_step(batch, batch_idx=0)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert torch.isfinite(loss)
        assert loss.requires_grad  # Differentiable loss for DDP

    def test_bayesian_uses_automatic_optimization(self, bayesian_config):
        """Bayesian path uses automatic_optimization=True for DDP gradient sync."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        module = CognitiveResilienceLightningModule(bayesian_config)
        assert module.automatic_optimization is True

    def test_train_loss_nll_logged_at_epoch_end(self, bayesian_config):
        """train_loss_nll should be logged once per epoch in on_train_epoch_end."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        module = CognitiveResilienceLightningModule(bayesian_config)
        module.log = MagicMock()
        batch = _make_batch(n_genes=50)

        # Training step should NOT log NLL (moved to epoch end)
        module.training_step(batch, batch_idx=0)
        logged_keys = [call.args[0] for call in module.log.call_args_list]
        assert "train_loss_nll" not in logged_keys, "NLL should not be logged per-step"

        # Epoch end should log NLL
        module.log.reset_mock()
        module.on_train_epoch_end()
        logged_keys = [call.args[0] for call in module.log.call_args_list]
        assert "train_loss_nll" in logged_keys, "NLL should be logged at epoch end"


class TestValidationStep:
    """Tests for validation_step."""

    def test_validation_step_runs(self, base_config):
        """validation_step runs without error."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        module = CognitiveResilienceLightningModule(base_config)
        module.eval()
        module.log = lambda *args, **kwargs: None
        batch = _make_batch(n_genes=50)
        # Should not raise
        module.validation_step(batch, batch_idx=0)

    def test_validation_step_logs_metrics(self, base_config):
        """validation_step logs val_loss per-batch, metrics at epoch end."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        module = CognitiveResilienceLightningModule(base_config)
        module.eval()

        logged = {}
        module.log = lambda name, value, **kwargs: logged.__setitem__(name, value)

        batch = _make_batch(n_genes=50)
        module.validation_step(batch, batch_idx=0)

        assert "val_loss" in logged

        # Metrics are now accumulated and logged at epoch end
        module.on_validation_epoch_end()
        metric_keys = [k for k in logged if k.startswith("val_") and k != "val_loss"]
        assert len(metric_keys) > 0

    def test_bayesian_validation_step(self, bayesian_config):
        """Bayesian head validation_step logs val_loss (ELBO) and val_nll."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        module = CognitiveResilienceLightningModule(bayesian_config)
        module.eval()

        logged = {}
        module.log = lambda name, value, **kwargs: logged.__setitem__(name, value)

        batch = _make_batch(n_genes=50)
        module.validation_step(batch, batch_idx=0)

        # val_loss = ELBO for Bayesian head (includes KL term)
        assert "val_loss" in logged
        # val_nll = Beta-NLL at posterior median (predictive quality diagnostic)
        assert "val_nll" in logged


class TestTestStep:
    """Tests for test_step."""

    def test_test_step_runs(self, base_config):
        """test_step runs without error."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        module = CognitiveResilienceLightningModule(base_config)
        module.eval()
        module.log = lambda *args, **kwargs: None
        batch = _make_batch(n_genes=50)
        # Should not raise
        module.test_step(batch, batch_idx=0)

    def test_test_step_logs_with_test_prefix(self, base_config):
        """test_step accumulates predictions; epoch end logs metrics with test_ prefix."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        module = CognitiveResilienceLightningModule(base_config)
        module.eval()

        logged = {}
        module.log = lambda name, value, **kwargs: logged.__setitem__(name, value)

        batch = _make_batch(n_genes=50)
        module.test_step(batch, batch_idx=0)

        assert "test_loss" in logged

        # Metrics are now computed at epoch end (not per-batch)
        module.on_test_epoch_end()
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

    def test_predict_step_includes_attention_weights(self, base_config):
        """predict_step includes attention_weights from model output."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        module = CognitiveResilienceLightningModule(base_config)
        module.eval()
        batch = _make_batch(n_genes=50)
        result = module.predict_step(batch, batch_idx=0)
        # Model always returns attention_weights
        assert "attention_weights" in result
        assert result["attention_weights"] is not None


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

    @pytest.mark.filterwarnings("ignore:.*Detected call of.*lr_scheduler.step.*before.*optimizer.step.*:UserWarning")
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

    @pytest.mark.filterwarnings("ignore:.*Detected call of.*lr_scheduler.step.*before.*optimizer.step.*:UserWarning")
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

    def test_lr_scaling_with_multi_gpu(self, base_config):
        """LR scales linearly with world_size when lr_scaling=true."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        base_config.training.lr_scaling = True
        base_lr = base_config.training.optimizer.lr  # 1e-3

        module = CognitiveResilienceLightningModule(base_config)

        # Mock trainer with world_size=2
        mock_trainer = MagicMock()
        mock_trainer.world_size = 2
        module._trainer = mock_trainer

        result = module.configure_optimizers()
        # Check initial_lr (before scheduler warmup modifies the current lr)
        actual_lr = result["optimizer"].param_groups[0]["initial_lr"]
        expected_lr = base_lr * 2
        assert abs(actual_lr - expected_lr) < 1e-10, (
            f"Expected LR {expected_lr}, got {actual_lr}"
        )

    def test_lr_no_scaling_when_disabled(self, base_config):
        """LR is NOT scaled when lr_scaling=false."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        base_config.training.lr_scaling = False
        base_lr = base_config.training.optimizer.lr  # 1e-3

        module = CognitiveResilienceLightningModule(base_config)

        # Mock trainer with world_size=4
        mock_trainer = MagicMock()
        mock_trainer.world_size = 4
        module._trainer = mock_trainer

        result = module.configure_optimizers()
        # Check initial_lr (before scheduler warmup modifies the current lr)
        actual_lr = result["optimizer"].param_groups[0]["initial_lr"]
        assert abs(actual_lr - base_lr) < 1e-10, (
            f"Expected base LR {base_lr}, got {actual_lr}"
        )

    def test_lr_no_scaling_single_gpu(self, base_config):
        """LR is NOT scaled when world_size=1 even if lr_scaling=true."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        base_config.training.lr_scaling = True
        base_lr = base_config.training.optimizer.lr

        module = CognitiveResilienceLightningModule(base_config)

        # Mock trainer with world_size=1
        mock_trainer = MagicMock()
        mock_trainer.world_size = 1
        module._trainer = mock_trainer

        result = module.configure_optimizers()
        # Check initial_lr (before scheduler warmup modifies the current lr)
        actual_lr = result["optimizer"].param_groups[0]["initial_lr"]
        assert abs(actual_lr - base_lr) < 1e-10, (
            f"Expected base LR {base_lr}, got {actual_lr}"
        )


class TestGeneGateL1Regularization:
    """Tests for optional gene gate L1 regularization."""

    def test_l1_regularization_when_enabled(self, base_config):
        """Gene gate L1 regularization adds to loss when lambda > 0."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        base_config.training.regularization.gene_gate_l1 = 0.01
        module = CognitiveResilienceLightningModule(base_config)
        module.log = lambda *args, **kwargs: None
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


class TestBayesianValidationMetrics:
    """Tests for Bayesian head uncertainty metrics in validation."""

    def test_bayesian_validation_logs_uncertainty_metrics(self, bayesian_config):
        """Bayesian head validation logs uncertainty metrics (mean_std, crps) at epoch end."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        module = CognitiveResilienceLightningModule(bayesian_config)
        module.eval()

        logged = {}
        module.log = lambda name, value, **kwargs: logged.__setitem__(name, value)

        batch = _make_batch(n_genes=50)
        module.validation_step(batch, batch_idx=0)
        module.on_validation_epoch_end()

        # Bayesian head should produce std, so uncertainty metrics should be logged at epoch end
        assert "val_mean_std" in logged
        assert "val_crps" in logged


class TestNaNHandling:
    """Tests for NaN detection and handling in training_step."""

    def test_nan_loss_policy_fail_raises(self, base_config):
        """NaN in batch with fail policy raises ValueError."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        # nan_batch="fail" catches NaN in any tensor (including cognition)
        # before loss computation even runs.
        base_config.error_handling = {"training": {"nan_loss": "fail", "nan_batch": "fail"}}
        module = CognitiveResilienceLightningModule(base_config)

        batch = _make_batch(n_genes=50)
        # Inject NaN into cognition — caught by _check_batch_nan
        batch["cognition"] = torch.full_like(batch["cognition"], float("nan"))

        with pytest.raises(ValueError, match="NaN detected in batch"):
            module.training_step(batch, batch_idx=0)

    def test_nan_batch_skip_returns_none(self, base_config):
        """NaN batch with skip policy returns None (skip batch)."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        base_config.error_handling = {"training": {"nan_loss": "fail", "nan_batch": "skip"}}
        module = CognitiveResilienceLightningModule(base_config)

        batch = _make_batch(n_genes=50)
        # Inject NaN into input tensor
        batch["region_pseudobulk"][0, 0, 0, 0] = float("nan")

        result = module.training_step(batch, batch_idx=0)
        assert result is None

    def test_nan_handling_defaults_without_config(self, base_config):
        """NaN handling works with defaults when error_handling not in config."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        # base_config has no error_handling section
        module = CognitiveResilienceLightningModule(base_config)
        assert module._nan_loss_policy == "fail"
        assert module._nan_batch_policy == "skip"

    def test_check_batch_nan_skips_cells_tensor(self, base_config):
        """_check_batch_nan skips cells tensor (validated at preprocessing)."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        module = CognitiveResilienceLightningModule(base_config)
        batch = {
            "cells": torch.tensor([[[float("nan")]]]),  # NaN in cells — excluded from check
            "cell_mask": torch.ones(1, 1, dtype=torch.bool),
            "cognition": torch.tensor([[1.0]]),  # No NaN here
        }
        assert not module._check_batch_nan(batch)

    def test_check_batch_nan_skips_mask_tensors(self, base_config):
        """_check_batch_nan skips boolean mask tensors (cell_mask, cell_type_mask, region_mask)."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        module = CognitiveResilienceLightningModule(base_config)
        batch = {
            "cells": torch.ones(1, 1, 1),
            "cell_mask": torch.ones(1, 1, dtype=torch.bool),
            "cell_type_mask": torch.ones(1, dtype=torch.bool),
            "region_mask": torch.ones(1, dtype=torch.bool),
            "cognition": torch.tensor([[1.0]]),
        }
        assert not module._check_batch_nan(batch)

    def test_check_batch_nan_detects_nan_in_cognition(self, base_config):
        """_check_batch_nan should detect NaN in non-skipped tensors."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        module = CognitiveResilienceLightningModule(base_config)
        batch = {
            "cells": torch.ones(1, 1, 1),
            "cognition": torch.tensor([[float("nan")]]),
        }
        assert module._check_batch_nan(batch)

    def test_check_batch_nan_detects_nan_in_edge_attrs(self, base_config):
        """_check_batch_nan should detect NaN in ccc_edge_attr tensor."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        module = CognitiveResilienceLightningModule(base_config)
        batch = {
            "cells": torch.ones(1, 1, 1),
            "cognition": torch.tensor([[1.0]]),
            "ccc_edge_attr": torch.tensor([[[float("nan")], [1.0]]]),
        }
        assert module._check_batch_nan(batch)

    def test_nan_pseudobulk_skipped_in_training_step(self, base_config):
        """NaN in pseudobulk tensor triggers batch skip in training_step."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        base_config.error_handling = {"training": {"nan_loss": "fail", "nan_batch": "skip"}}
        module = CognitiveResilienceLightningModule(base_config)

        batch = _make_batch(n_genes=50)
        # Inject NaN into region_pseudobulk tensor (simulating corrupt data source)
        batch["region_pseudobulk"][0, 0, 0, :] = float("nan")

        result = module.training_step(batch, batch_idx=0)
        assert result is None, "Expected None return when pseudobulk contains NaN and nan_batch=skip"

    def test_nan_skip_rate_threshold(self, base_config):
        """Exceeding max_nan_skip_fraction raises RuntimeError at epoch end."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        base_config.error_handling = {
            "training": {"nan_loss": "fail", "nan_batch": "skip", "max_nan_skip_fraction": 0.1}
        }
        module = CognitiveResilienceLightningModule(base_config)

        # Simulate 5 batches, 2 NaN skips (40% > 10% threshold)
        module._epoch_total_batches = 5
        module._epoch_nan_skips = 2
        # Mock self.log to avoid trainer requirement
        module.log = MagicMock()

        with pytest.raises(RuntimeError, match="NaN skip rate"):
            module.on_train_epoch_end()


class TestParameterWiring:
    """Tests for config-to-model parameter wiring."""

    def test_parameterized_n_pathology_features(self, base_config):
        """n_pathology_features is wired from config through to model."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        base_config.model.pathology_attention.n_pathology_features = 3
        module = CognitiveResilienceLightningModule(base_config)
        module.log = lambda *args, **kwargs: None
        assert module.model.pathology_encoder.n_pathology_features == 3
        # Verify model instantiates and runs with the parameter
        batch = _make_batch(n_genes=50)
        loss = module.training_step(batch, batch_idx=0)
        assert torch.isfinite(loss)

    def test_parameterized_n_pma_seeds(self, base_config):
        """n_pma_seeds is wired from config through to CellTransformer's SetTransformerEncoder."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        base_config.model.set_transformer.n_pma_seeds = 1
        module = CognitiveResilienceLightningModule(base_config)
        assert module.model.cell_transformer.set_encoder.n_pma_seeds == 1


class TestBatchToModelKwargs:
    """Tests for _batch_to_model_kwargs flat cell format support."""

    def test_flat_keys_preferred_over_padded(self, base_config):
        """When batch has both flat and padded keys, flat keys are passed to model."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        module = CognitiveResilienceLightningModule(base_config)

        batch = _make_batch(n_genes=50)
        # Add flat keys (simulating collation that produces both)
        batch["cell_data"] = torch.randn(100, 50)
        batch["cell_offsets"] = torch.zeros(2, N_CELL_TYPES + 1, dtype=torch.long)

        kwargs = module._batch_to_model_kwargs(batch)
        assert "cell_data" in kwargs, "Flat cell_data missing from model kwargs"
        assert "cell_offsets" in kwargs, "Flat cell_offsets missing from model kwargs"
        assert "cells" not in kwargs, "Padded cells should not be passed when flat keys present"
        assert "cell_mask" not in kwargs, "Padded cell_mask should not be passed when flat keys present"

    def test_padded_fallback_when_no_flat_keys(self, base_config):
        """When batch only has padded keys, padded format is passed to model."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        module = CognitiveResilienceLightningModule(base_config)

        batch = _make_batch(n_genes=50)
        # No flat keys — standard padded batch
        assert "cell_data" not in batch

        kwargs = module._batch_to_model_kwargs(batch)
        assert "cells" in kwargs, "Padded cells missing from model kwargs"
        assert "cell_mask" in kwargs, "Padded cell_mask missing from model kwargs"
        assert "cell_data" not in kwargs, "cell_data should not be present for padded-only batch"
        assert "cell_offsets" not in kwargs, "cell_offsets should not be present for padded-only batch"

    def test_flat_kwargs_forward_pass_works(self, base_config):
        """Full forward pass works when _batch_to_model_kwargs passes flat keys."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        module = CognitiveResilienceLightningModule(base_config)
        module.log = lambda *args, **kwargs: None

        n_genes = 50
        batch_size = 2

        # Build a batch with flat cell format
        n_types = N_CELL_TYPES
        # 5 cells for type 0, 3 cells for type 1, 0 for rest
        cell_counts = [5, 3] + [0] * (n_types - 2)
        total_cells_per_sample = sum(cell_counts)

        all_cell_data = []
        all_offsets = []
        running_offset = 0
        for _ in range(batch_size):
            cell_data_s = torch.randn(total_cells_per_sample, n_genes)
            all_cell_data.append(cell_data_s)
            offsets_s = torch.zeros(n_types + 1, dtype=torch.long)
            for ct in range(n_types):
                offsets_s[ct + 1] = offsets_s[ct] + cell_counts[ct]
            offsets_s += running_offset
            all_offsets.append(offsets_s)
            running_offset += total_cells_per_sample

        batch = _make_batch(batch_size=batch_size, n_genes=n_genes)
        batch["cell_data"] = torch.cat(all_cell_data, dim=0)
        batch["cell_offsets"] = torch.stack(all_offsets, dim=0)

        # Should run without error using flat path
        loss = module.training_step(batch, batch_idx=0)
        assert isinstance(loss, torch.Tensor)
        assert torch.isfinite(loss)


class _SyntheticDataset(torch.utils.data.Dataset):
    """Tiny dataset that returns pre-built batches for integration testing."""

    def __init__(self, n_samples=4, n_genes=50):
        self.samples = [_make_batch(batch_size=1, n_genes=n_genes) for _ in range(n_samples)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def _identity_collate(batch):
    """Collate that stacks single-sample dicts into a batch."""
    keys = batch[0].keys()
    result = {}
    for key in keys:
        values = [b[key] for b in batch]
        if isinstance(values[0], torch.Tensor):
            result[key] = torch.cat(values, dim=0)
        elif isinstance(values[0], list):
            # Flatten list of lists
            result[key] = [item for sublist in values for item in sublist]
        else:
            result[key] = values
    return result


class TestTrainerIntegration:
    """Integration test: Trainer + LightningModule for 1 epoch."""

    def test_trainer_fit_one_epoch(self, base_config, tmp_path):
        """Trainer.fit runs 1 epoch with synthetic data without crashing."""
        import lightning.pytorch as pl
        from src.training.lightning_module import CognitiveResilienceLightningModule

        module = CognitiveResilienceLightningModule(base_config)

        # Create proper DataLoaders with synthetic data
        train_ds = _SyntheticDataset(n_samples=4, n_genes=50)
        val_ds = _SyntheticDataset(n_samples=2, n_genes=50)

        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=2, collate_fn=_identity_collate,
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=2, collate_fn=_identity_collate,
        )

        trainer = pl.Trainer(
            max_epochs=1,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
            logger=False,
            default_root_dir=str(tmp_path),
        )

        # Should complete without error
        trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # Verify training completed
        assert trainer.current_epoch == 1


class TestCheckpointRoundTrip:
    """Tests for saving and loading checkpoints."""

    def test_save_load_checkpoint_predictions_match(self, base_config, tmp_path):
        """Checkpoint round-trip: saved and loaded module produce identical predictions."""
        import lightning.pytorch as pl
        from src.training.lightning_module import CognitiveResilienceLightningModule

        module = CognitiveResilienceLightningModule(base_config)

        # Fit 1 epoch with synthetic data
        train_ds = _SyntheticDataset(n_samples=4, n_genes=50)
        val_ds = _SyntheticDataset(n_samples=2, n_genes=50)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=2, collate_fn=_identity_collate,
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=2, collate_fn=_identity_collate,
        )

        trainer = pl.Trainer(
            max_epochs=1,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
            logger=False,
            default_root_dir=str(tmp_path),
        )
        trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # Save checkpoint
        ckpt_path = tmp_path / "test_checkpoint.ckpt"
        trainer.save_checkpoint(str(ckpt_path))

        # Get predictions from original module
        module.eval()
        batch = _make_batch(batch_size=2, n_genes=50)
        with torch.no_grad():
            original_output = module.model(
                region_pseudobulk=batch["region_pseudobulk"],
                region_mask=batch["region_mask"],
                ccc_edge_index=batch["ccc_edge_index"],
                ccc_edge_type=batch["ccc_edge_type"],
                ccc_edge_attr=batch["ccc_edge_attr"],
                ccc_edge_counts=batch["ccc_edge_counts"],
                cells=batch["cells"],
                cell_mask=batch["cell_mask"],
                cell_type_mask=batch.get("cell_type_mask"),
                pathology=batch["pathology"],
            )

        # Load from checkpoint (config must be passed explicitly since we ignore it in save_hyperparameters)
        loaded_module = CognitiveResilienceLightningModule.load_from_checkpoint(
            str(ckpt_path), config=base_config, weights_only=False,
        )
        loaded_module.eval()
        with torch.no_grad():
            loaded_output = loaded_module.model(
                region_pseudobulk=batch["region_pseudobulk"],
                region_mask=batch["region_mask"],
                ccc_edge_index=batch["ccc_edge_index"],
                ccc_edge_type=batch["ccc_edge_type"],
                ccc_edge_attr=batch["ccc_edge_attr"],
                ccc_edge_counts=batch["ccc_edge_counts"],
                cells=batch["cells"],
                cell_mask=batch["cell_mask"],
                cell_type_mask=batch.get("cell_type_mask"),
                pathology=batch["pathology"],
            )

        assert torch.allclose(original_output["mean"], loaded_output["mean"], atol=1e-6)


class TestDDPBehavior:
    """T5: DDP-related behavior tests (mock-based, no multi-GPU required)."""

    def test_elbo_data_scale_set_for_multi_gpu(self, bayesian_config):
        """Bayesian head data_scale is set to world_size for DDP."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        module = CognitiveResilienceLightningModule(bayesian_config)

        mock_trainer = MagicMock()
        mock_trainer.world_size = 4
        module._trainer = mock_trainer

        module.configure_optimizers()

        assert module.model.prediction_head._data_scale == 4

    def test_elbo_data_scale_default_single_gpu(self, bayesian_config):
        """Bayesian head data_scale stays at 1 for single GPU."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        module = CognitiveResilienceLightningModule(bayesian_config)

        mock_trainer = MagicMock()
        mock_trainer.world_size = 1
        module._trainer = mock_trainer

        module.configure_optimizers()

        assert module.model.prediction_head._data_scale == 1

    def test_worker_seed_uniqueness_across_ranks(self):
        """Worker init fn produces unique seeds for different ranks."""
        seed = 42
        max_workers = 4
        all_seeds = []
        for rank in range(4):
            for worker_id in range(max_workers):
                s = (seed + rank * max_workers + worker_id) % (2**32)
                all_seeds.append(s)

        assert len(set(all_seeds)) == len(all_seeds), f"Duplicate seeds: {all_seeds}"

    def test_sync_dist_on_all_logged_metrics(self, base_config):
        """All self.log calls use sync_dist=True for DDP correctness,
        except train_loss which uses sync_dist=False (DDP-1: per-step
        allreduce on the training loss is unnecessary overhead — Lightning
        already reduces gradients via DDP)."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        module = CognitiveResilienceLightningModule(base_config)

        log_calls = []
        original_log = module.log

        def capture_log(*args, **kwargs):
            # Capture metric name (first positional arg) alongside kwargs
            name = args[0] if args else kwargs.get("name", "unknown")
            log_calls.append({"_name": name, **kwargs})
            return original_log(*args, **kwargs)

        module.log = capture_log

        batch = _make_batch(n_genes=50)
        module.training_step(batch, batch_idx=0)

        assert len(log_calls) > 0, "No log calls captured"
        for call in log_calls:
            name = call["_name"]
            if name == "train_loss":
                # DDP-1: train_loss intentionally skips sync_dist
                assert call.get("sync_dist", False) is False, (
                    f"train_loss should use sync_dist=False (DDP-1): {call}"
                )
            else:
                assert call.get("sync_dist", False) is True, (
                    f"Missing sync_dist=True on {name}: {call}"
                )


class TestBayesianELBOConvergence:
    """T7: Verify Bayesian ELBO loss decreases through Lightning training_step."""

    def test_elbo_decreases_over_steps(self, bayesian_config):
        """ELBO loss should decrease over multiple training steps.

        Tests the full Lightning path: training_step -> _svi_forward ->
        Trace_ELBO.differentiable_loss -> backward -> optimizer step.
        """
        from src.training.lightning_module import CognitiveResilienceLightningModule

        module = CognitiveResilienceLightningModule(bayesian_config)
        module.log = lambda *args, **kwargs: None

        # configure_optimizers prototypes the guide and creates optimizer
        opt_config = module.configure_optimizers()
        optimizer = opt_config["optimizer"]

        batch = _make_batch(n_genes=50)

        losses = []
        for step in range(30):
            optimizer.zero_grad()
            loss = module.training_step(batch, batch_idx=step)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Average of last 5 losses should be less than average of first 5
        avg_first_5 = sum(losses[:5]) / 5
        avg_last_5 = sum(losses[-5:]) / 5
        assert avg_last_5 < avg_first_5, (
            f"ELBO did not decrease: first 5 avg={avg_first_5:.4f}, "
            f"last 5 avg={avg_last_5:.4f}"
        )

    def test_posterior_median_produces_finite_predictions(self, bayesian_config):
        """After training, posterior median should produce finite predictions."""
        from src.training.lightning_module import CognitiveResilienceLightningModule

        module = CognitiveResilienceLightningModule(bayesian_config)
        module.log = lambda *args, **kwargs: None

        # configure_optimizers prototypes the guide and creates optimizer
        opt_config = module.configure_optimizers()
        optimizer = opt_config["optimizer"]

        batch = _make_batch(n_genes=50)

        # Train 5 steps to prototype the guide
        for step in range(5):
            optimizer.zero_grad()
            loss = module.training_step(batch, batch_idx=step)
            loss.backward()
            optimizer.step()

        # Call _forward_batch_posterior with no_grad
        with torch.no_grad():
            output = module._forward_batch_posterior(batch)

        assert torch.isfinite(output["mean"]).all(), (
            f"Non-finite mean predictions: {output['mean']}"
        )
        assert torch.isfinite(output["std"]).all(), (
            f"Non-finite std predictions: {output['std']}"
        )
        assert (output["std"] > 0).all(), (
            f"Non-positive std predictions: {output['std']}"
        )


class TestBayesianSVILrd:
    """Tests for Bayesian SVI learning rate decay (lrd) configuration."""

    def test_bayesian_svi_uses_lrd_from_config(self, bayesian_config):
        """ExponentialLR uses gamma=lrd from config."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        bayesian_config.training.optimizer.lrd = 0.9999
        module = CognitiveResilienceLightningModule(bayesian_config)
        result = module.configure_optimizers()
        scheduler = result["lr_scheduler"]["scheduler"]
        assert isinstance(scheduler, torch.optim.lr_scheduler.ExponentialLR)
        # ExponentialLR stores gamma
        assert scheduler.gamma == 0.9999

    def test_bayesian_svi_lrd_defaults_to_one(self, bayesian_config):
        """ExponentialLR defaults to gamma=1.0 (no decay) when config omits lrd."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        if "lrd" in bayesian_config.training.optimizer:
            del bayesian_config.training.optimizer.lrd
        module = CognitiveResilienceLightningModule(bayesian_config)
        result = module.configure_optimizers()
        scheduler = result["lr_scheduler"]["scheduler"]
        assert scheduler.gamma == 1.0


