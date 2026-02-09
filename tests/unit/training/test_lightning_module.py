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
    cell_type_mask = torch.ones(batch_size, n_cell_types, dtype=torch.bool)
    cell_type_mask[:, -3:] = False  # Mask out last 3 types
    pathology = torch.rand(batch_size, 3)
    cognition = torch.randn(batch_size, 1)

    return {
        "region_pseudobulk": region_pseudobulk,
        "region_mask": region_mask,
        "edge_index_dict_list": edge_index_dict_list,
        "edge_attr_dict_list": edge_attr_dict_list,
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
        """Bayesian head training_step returns finite loss via SVI."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        module = CognitiveResilienceLightningModule(bayesian_config)
        module.log = lambda *args, **kwargs: None
        # SVI requires configure_optimizers to initialize self.svi
        module.configure_optimizers()
        batch = _make_batch(n_genes=50)
        loss = module.training_step(batch, batch_idx=0)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_bayesian_svi_logs_gradient_norms(self, bayesian_config):
        """Bayesian SVI training step should log gradient norms manually."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        module = CognitiveResilienceLightningModule(bayesian_config)
        assert hasattr(module, '_log_svi_gradient_norms')


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
        module.log = lambda *args, **kwargs: None
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
        """Bayesian head validation_step logs uncertainty metrics (mean_std, calibration_error, crps)."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        module = CognitiveResilienceLightningModule(bayesian_config)
        module.eval()

        logged = {}
        module.log = lambda name, value, **kwargs: logged.__setitem__(name, value)

        batch = _make_batch(n_genes=50)
        module.validation_step(batch, batch_idx=0)

        # Bayesian head should produce std, so uncertainty metrics should be logged
        assert "val_mean_std" in logged
        assert "val_crps" in logged


class TestNaNHandling:
    """Tests for NaN detection and handling in training_step."""

    def test_nan_loss_policy_fail_raises(self, base_config):
        """NaN loss with fail policy raises ValueError."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        # nan_batch="fail" so NaN cognition passes through to loss computation
        base_config.error_handling = {"training": {"nan_loss": "fail", "nan_batch": "fail"}}
        module = CognitiveResilienceLightningModule(base_config)

        batch = _make_batch(n_genes=50)
        # Inject NaN into cognition to cause NaN loss
        batch["cognition"] = torch.full_like(batch["cognition"], float("nan"))

        with pytest.raises(ValueError, match="NaN loss"):
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
            # Flatten list of lists (e.g., edge_index_dict_list)
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
                edge_index_dict_list=batch["edge_index_dict_list"],
                edge_attr_dict_list=batch["edge_attr_dict_list"],
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
                edge_index_dict_list=batch["edge_index_dict_list"],
                edge_attr_dict_list=batch["edge_attr_dict_list"],
                cells=batch["cells"],
                cell_mask=batch["cell_mask"],
                cell_type_mask=batch.get("cell_type_mask"),
                pathology=batch["pathology"],
            )

        assert torch.allclose(original_output["mean"], loaded_output["mean"], atol=1e-6)


class TestBayesianSVILrd:
    """Tests for Bayesian SVI learning rate decay (lrd) configuration."""

    def test_bayesian_svi_uses_lrd_from_config(self, bayesian_config):
        """ClippedAdam receives lrd value from config when specified."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        bayesian_config.training.optimizer.lrd = 0.9999
        module = CognitiveResilienceLightningModule(bayesian_config)
        module.configure_optimizers()
        # PyroClippedAdam stores optimizer args in pt_optim_args
        assert module.pyro_optim.pt_optim_args["lrd"] == 0.9999

    def test_bayesian_svi_lrd_defaults_to_one(self, bayesian_config):
        """ClippedAdam defaults to lrd=1.0 (no decay) when config omits lrd."""
        from src.training.lightning_module import CognitiveResilienceLightningModule
        # Ensure no lrd key in optimizer config
        if "lrd" in bayesian_config.training.optimizer:
            del bayesian_config.training.optimizer.lrd
        module = CognitiveResilienceLightningModule(bayesian_config)
        module.configure_optimizers()
        assert module.pyro_optim.pt_optim_args["lrd"] == 1.0
