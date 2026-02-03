"""
Tests for training callbacks.

Tests TemperatureAnnealing and GradientNormLogger callbacks:
- Temperature schedule correctness (warmup + annealing)
- Boundary conditions (epoch 0, final epoch)
- Exponential, linear, cosine schedules
- Gradient norm computation per branch
- Gradient norm ratio logging and warning thresholds
- Early stopping and checkpoint configuration
"""

import math
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, PropertyMock, patch


class TestTemperatureAnnealingSchedule:
    """Tests for TemperatureAnnealing callback schedule computation."""

    def test_temperature_at_epoch_0(self):
        """Temperature equals tau_max at start (epoch 0)."""
        from src.training.callbacks import TemperatureAnnealing
        callback = TemperatureAnnealing(
            tau_max=2.0, tau_min=0.1, warmup_epochs=5,
            anneal_epochs=50, schedule="exponential",
        )
        assert callback.get_temperature(epoch=0) == 2.0

    def test_temperature_during_warmup(self):
        """Temperature stays at tau_max during warmup epochs."""
        from src.training.callbacks import TemperatureAnnealing
        callback = TemperatureAnnealing(
            tau_max=2.0, tau_min=0.1, warmup_epochs=5,
            anneal_epochs=50, schedule="exponential",
        )
        for epoch in range(5):
            assert callback.get_temperature(epoch=epoch) == 2.0

    def test_temperature_at_final_epoch(self):
        """Temperature approximately equals tau_min at end of annealing."""
        from src.training.callbacks import TemperatureAnnealing
        callback = TemperatureAnnealing(
            tau_max=2.0, tau_min=0.1, warmup_epochs=5,
            anneal_epochs=50, schedule="exponential",
        )
        final_epoch = 5 + 50 - 1  # Last annealing epoch
        tau = callback.get_temperature(epoch=final_epoch)
        assert abs(tau - 0.1) < 0.01

    def test_temperature_after_annealing_stays_at_tau_min(self):
        """Temperature stays at tau_min after annealing is complete."""
        from src.training.callbacks import TemperatureAnnealing
        callback = TemperatureAnnealing(
            tau_max=2.0, tau_min=0.1, warmup_epochs=5,
            anneal_epochs=50, schedule="exponential",
        )
        tau = callback.get_temperature(epoch=100)
        assert abs(tau - 0.1) < 0.01

    def test_exponential_schedule_monotonically_decreasing(self):
        """Exponential schedule decreases monotonically after warmup."""
        from src.training.callbacks import TemperatureAnnealing
        callback = TemperatureAnnealing(
            tau_max=2.0, tau_min=0.1, warmup_epochs=5,
            anneal_epochs=50, schedule="exponential",
        )
        temps = [callback.get_temperature(epoch=e) for e in range(5, 55)]
        for i in range(1, len(temps)):
            assert temps[i] <= temps[i - 1], (
                f"Temperature should decrease: epoch {5+i}: {temps[i]} > {temps[i-1]}"
            )

    def test_linear_schedule(self):
        """Linear schedule produces linearly interpolated temperatures."""
        from src.training.callbacks import TemperatureAnnealing
        callback = TemperatureAnnealing(
            tau_max=2.0, tau_min=0.0, warmup_epochs=0,
            anneal_epochs=3, schedule="linear",
        )
        # anneal_epochs=3 → epoch 0: progress=0 (tau=2.0), epoch 1: progress=0.5 (tau=1.0)
        tau_mid = callback.get_temperature(epoch=1)
        assert abs(tau_mid - 1.0) < 0.01
        # epoch 2: progress=1.0 (tau=0.0)
        tau_end = callback.get_temperature(epoch=2)
        assert abs(tau_end - 0.0) < 0.01

    def test_cosine_schedule(self):
        """Cosine schedule produces smooth cosine-interpolated temperatures."""
        from src.training.callbacks import TemperatureAnnealing
        callback = TemperatureAnnealing(
            tau_max=2.0, tau_min=0.0, warmup_epochs=0,
            anneal_epochs=3, schedule="cosine",
        )
        # At midpoint (progress=0.5): cos(pi*0.5)=0, factor=0.5, tau=0+0.5*2=1.0
        tau_mid = callback.get_temperature(epoch=1)
        assert abs(tau_mid - 1.0) < 0.01

    def test_invalid_schedule_raises(self):
        """Invalid schedule type raises ValueError."""
        from src.training.callbacks import TemperatureAnnealing
        with pytest.raises(ValueError, match="schedule"):
            TemperatureAnnealing(
                tau_max=2.0, tau_min=0.1, warmup_epochs=5,
                anneal_epochs=50, schedule="invalid",
            )

    def test_on_train_epoch_start_sets_temperature(self):
        """Callback sets model gene gate temperature on epoch start."""
        from src.training.callbacks import TemperatureAnnealing
        callback = TemperatureAnnealing(
            tau_max=2.0, tau_min=0.1, warmup_epochs=0,
            anneal_epochs=50, schedule="exponential",
        )

        # Mock trainer and model
        trainer = MagicMock()
        trainer.current_epoch = 10

        pl_module = MagicMock()
        gene_gate = MagicMock()
        # Use PropertyMock to track temperature setter calls
        temp_prop = PropertyMock()
        type(gene_gate).temperature = temp_prop
        pl_module.model.pseudobulk_encoder.gene_gate = gene_gate

        callback.on_train_epoch_start(trainer, pl_module)

        # Verify temperature was set to the correct value
        expected_tau = callback.get_temperature(epoch=10)
        temp_prop.assert_called_with(expected_tau)


class TestGradientNormLogger:
    """Tests for GradientNormLogger callback."""

    def _make_simple_model(self):
        """Create a simple model with named branches for testing."""
        model = nn.Module()
        model.pseudobulk_encoder = nn.Linear(10, 5)
        model.hgt_encoder = nn.Linear(10, 5)
        model.cell_transformer = nn.Linear(10, 5)
        return model

    def test_compute_branch_norms(self):
        """Computes L2 gradient norms per branch."""
        from src.training.callbacks import GradientNormLogger
        callback = GradientNormLogger()

        model = self._make_simple_model()
        # Create dummy gradients
        x = torch.randn(2, 10)
        loss = (
            model.pseudobulk_encoder(x).sum()
            + model.hgt_encoder(x).sum()
            + model.cell_transformer(x).sum()
        )
        loss.backward()

        norms = callback.compute_branch_norms(model)
        assert "pseudobulk_encoder" in norms
        assert "hgt_encoder" in norms
        assert "cell_transformer" in norms
        # All norms should be positive (gradients exist)
        for name, norm in norms.items():
            assert norm > 0, f"Branch {name} should have positive gradient norm"

    def test_branch_norm_ratio(self):
        """Computes max/min ratio of branch norms."""
        from src.training.callbacks import GradientNormLogger
        callback = GradientNormLogger()

        norms = {"a": 10.0, "b": 2.0, "c": 5.0}
        ratio = callback.compute_norm_ratio(norms)
        assert abs(ratio - 5.0) < 1e-6  # 10 / 2 = 5

    def test_branch_norm_ratio_with_zero(self):
        """Ratio computation handles zero norms gracefully."""
        from src.training.callbacks import GradientNormLogger
        callback = GradientNormLogger()

        norms = {"a": 10.0, "b": 0.0, "c": 5.0}
        ratio = callback.compute_norm_ratio(norms)
        # Should not be infinite (epsilon protection)
        assert ratio < float('inf')

    def test_warning_at_yellow_threshold(self):
        """Logs warning when ratio is in 3-10 range (yellow zone)."""
        from src.training.callbacks import GradientNormLogger
        callback = GradientNormLogger()

        norms = {"a": 5.0, "b": 1.0, "c": 3.0}
        ratio = callback.compute_norm_ratio(norms)
        assert 3.0 <= ratio < 10.0
        assert callback.get_severity(ratio) == "yellow"

    def test_critical_at_red_threshold(self):
        """Returns critical severity when ratio >= 10 (red zone)."""
        from src.training.callbacks import GradientNormLogger
        callback = GradientNormLogger()

        norms = {"a": 50.0, "b": 1.0, "c": 10.0}
        ratio = callback.compute_norm_ratio(norms)
        assert ratio >= 10.0
        assert callback.get_severity(ratio) == "red"

    def test_normal_below_threshold(self):
        """Returns normal severity when ratio < 3."""
        from src.training.callbacks import GradientNormLogger
        callback = GradientNormLogger()

        assert callback.get_severity(2.5) == "normal"
        assert callback.get_severity(1.0) == "normal"


class TestResilienceModelCheckpoint:
    """Tests for ResilienceModelCheckpoint callback."""

    def _make_checkpoint_context(self):
        """Create mock trainer, module, and checkpoint dict."""
        from src.training.callbacks import ResilienceModelCheckpoint
        from omegaconf import OmegaConf

        callback = ResilienceModelCheckpoint()
        trainer = MagicMock()
        pl_module = MagicMock()
        pl_module.config = OmegaConf.create({
            "model": {
                "n_genes": 50,
                "d_embed": 32,
                "head": {"type": "bayesian"},
            },
            "training": {"max_epochs": 10},
        })
        checkpoint = {}
        return callback, trainer, pl_module, checkpoint

    def test_adds_checkpoint_version(self):
        """Checkpoint contains checkpoint_version string."""
        callback, trainer, pl_module, checkpoint = self._make_checkpoint_context()
        callback.on_save_checkpoint(trainer, pl_module, checkpoint)
        assert "checkpoint_version" in checkpoint
        assert isinstance(checkpoint["checkpoint_version"], str)

    def test_adds_experiment_hash(self):
        """Checkpoint contains experiment_hash (SHA-256 hex string)."""
        callback, trainer, pl_module, checkpoint = self._make_checkpoint_context()
        callback.on_save_checkpoint(trainer, pl_module, checkpoint)
        assert "experiment_hash" in checkpoint
        assert isinstance(checkpoint["experiment_hash"], str)
        assert len(checkpoint["experiment_hash"]) == 64  # SHA-256 hex length

    def test_adds_timestamp_iso_parseable(self):
        """Checkpoint contains ISO 8601 parseable timestamp."""
        from datetime import datetime
        callback, trainer, pl_module, checkpoint = self._make_checkpoint_context()
        callback.on_save_checkpoint(trainer, pl_module, checkpoint)
        assert "timestamp" in checkpoint
        # Should be parseable as ISO 8601
        parsed = datetime.fromisoformat(checkpoint["timestamp"])
        assert parsed is not None

    def test_adds_rng_states(self):
        """Checkpoint contains RNG states for python, numpy, torch."""
        callback, trainer, pl_module, checkpoint = self._make_checkpoint_context()
        callback.on_save_checkpoint(trainer, pl_module, checkpoint)
        assert "rng_states" in checkpoint
        rng = checkpoint["rng_states"]
        assert "python" in rng
        assert "numpy" in rng
        assert "torch" in rng

    def test_adds_model_config(self):
        """Checkpoint contains model_config dict."""
        callback, trainer, pl_module, checkpoint = self._make_checkpoint_context()
        callback.on_save_checkpoint(trainer, pl_module, checkpoint)
        assert "model_config" in checkpoint
        assert isinstance(checkpoint["model_config"], dict)
        assert checkpoint["model_config"]["n_genes"] == 50

    def test_on_load_checkpoint_restores_rng_states(self):
        """on_load_checkpoint restores RNG states so draws are reproducible."""
        import random
        import numpy as np

        callback, trainer, pl_module, checkpoint = self._make_checkpoint_context()

        # Save checkpoint (captures current RNG states)
        callback.on_save_checkpoint(trainer, pl_module, checkpoint)

        # Record expected draws from the saved state
        saved_python_state = checkpoint["rng_states"]["python"]
        saved_numpy_state = checkpoint["rng_states"]["numpy"]
        saved_torch_state = checkpoint["rng_states"]["torch"]

        # Advance RNG past the saved state
        random.random()
        random.random()
        np.random.random(10)
        torch.randn(10)

        # Restore from checkpoint
        callback.on_load_checkpoint(trainer, pl_module, checkpoint)

        # Verify states match the saved checkpoint
        assert random.getstate() == saved_python_state
        restored_numpy = np.random.get_state()
        assert restored_numpy[0] == saved_numpy_state[0]  # 'MT19937'
        assert (restored_numpy[1] == saved_numpy_state[1]).all()
        assert torch.equal(torch.random.get_rng_state(), saved_torch_state)

    def test_on_load_checkpoint_warns_missing_rng(self):
        """on_load_checkpoint warns when checkpoint has no rng_states."""
        import logging

        callback, trainer, pl_module, _ = self._make_checkpoint_context()
        empty_checkpoint = {}  # No rng_states key

        with patch("src.training.callbacks.logger") as mock_logger:
            callback.on_load_checkpoint(trainer, pl_module, empty_checkpoint)
            mock_logger.warning.assert_called_once()
            assert "rng_states" in mock_logger.warning.call_args[0][0]


class TestEarlyStopping:
    """Tests for early stopping configuration."""

    def test_early_stopping_patience(self):
        """Early stopping configured with correct patience from config."""
        from omegaconf import OmegaConf
        from lightning.pytorch.callbacks import EarlyStopping

        cfg = OmegaConf.load("configs/default.yaml")
        es_cfg = cfg.training.early_stopping

        # Instantiate with our config values
        es = EarlyStopping(
            monitor=es_cfg.monitor,
            patience=es_cfg.patience,
            min_delta=es_cfg.min_delta,
            mode=es_cfg.mode,
        )
        assert es.patience == 15
        # Lightning negates min_delta for mode="min", check absolute value
        assert abs(es.min_delta) == pytest.approx(0.0001)
        assert es.monitor == "val_loss"
        assert es.mode == "min"

    def test_early_stopping_min_epochs(self):
        """Training config includes min_epochs to prevent premature stopping."""
        from omegaconf import OmegaConf
        cfg = OmegaConf.load("configs/default.yaml")
        # min_epochs must be present and >= warmup + stabilization
        assert hasattr(cfg.training.early_stopping, "min_epochs"), \
            "early_stopping must include min_epochs"
        assert cfg.training.early_stopping.min_epochs >= 20, \
            "min_epochs should be >= 20 (warmup + stabilization)"


class TestMinEpochEarlyStopping:
    """Tests for MinEpochEarlyStopping callback."""

    def test_inherits_from_early_stopping(self):
        """MinEpochEarlyStopping is a subclass of EarlyStopping."""
        from src.training.callbacks import MinEpochEarlyStopping
        from lightning.pytorch.callbacks import EarlyStopping

        callback = MinEpochEarlyStopping(min_epochs=20, monitor="val_loss", patience=15)
        assert isinstance(callback, EarlyStopping)

    def test_min_epochs_default(self):
        """Default min_epochs is 20."""
        from src.training.callbacks import MinEpochEarlyStopping

        callback = MinEpochEarlyStopping(monitor="val_loss", patience=15)
        assert callback.min_epochs == 20

    def test_min_epochs_custom(self):
        """Custom min_epochs value is stored correctly."""
        from src.training.callbacks import MinEpochEarlyStopping

        callback = MinEpochEarlyStopping(min_epochs=30, monitor="val_loss", patience=15)
        assert callback.min_epochs == 30

    def test_skips_early_stopping_before_min_epochs(self):
        """on_validation_end does nothing before min_epochs is reached."""
        from src.training.callbacks import MinEpochEarlyStopping

        callback = MinEpochEarlyStopping(min_epochs=20, monitor="val_loss", patience=3)

        trainer = MagicMock()
        pl_module = MagicMock()

        # Before min_epochs: callback should NOT call parent's on_validation_end
        # We test this by verifying wait_count doesn't change
        trainer.current_epoch = 10  # < 20
        initial_wait_count = callback.wait_count

        callback.on_validation_end(trainer, pl_module)

        # wait_count should remain unchanged (parent method not called)
        assert callback.wait_count == initial_wait_count

    def test_activates_after_min_epochs(self):
        """on_validation_end calls parent after min_epochs is reached."""
        from src.training.callbacks import MinEpochEarlyStopping

        callback = MinEpochEarlyStopping(min_epochs=20, monitor="val_loss", patience=3)

        trainer = MagicMock()
        trainer.current_epoch = 25  # > 20
        trainer.callback_metrics = {"val_loss": 0.5}

        pl_module = MagicMock()

        # After min_epochs: parent method should be called
        # We mock the parent class method to verify it's called
        with patch.object(
            MinEpochEarlyStopping.__bases__[0],  # EarlyStopping
            "on_validation_end",
        ) as mock_parent:
            callback.on_validation_end(trainer, pl_module)
            mock_parent.assert_called_once_with(trainer, pl_module)

    def test_on_train_epoch_end_skips_before_min_epochs(self):
        """on_train_epoch_end does nothing before min_epochs is reached."""
        from src.training.callbacks import MinEpochEarlyStopping

        callback = MinEpochEarlyStopping(min_epochs=20, monitor="val_loss", patience=3)

        trainer = MagicMock()
        trainer.current_epoch = 5  # < 20
        pl_module = MagicMock()

        # Should not call parent
        with patch.object(
            MinEpochEarlyStopping.__bases__[0],
            "on_train_epoch_end",
        ) as mock_parent:
            callback.on_train_epoch_end(trainer, pl_module)
            mock_parent.assert_not_called()

    def test_repr(self):
        """__repr__ includes min_epochs and inherited parameters."""
        from src.training.callbacks import MinEpochEarlyStopping

        callback = MinEpochEarlyStopping(
            min_epochs=25,
            monitor="val_loss",
            patience=15,
            min_delta=0.001,
            mode="min",
        )
        repr_str = repr(callback)
        assert "min_epochs=25" in repr_str
        assert "monitor='val_loss'" in repr_str
        assert "patience=15" in repr_str

    def test_setup_callbacks_uses_min_epoch_early_stopping(self):
        """setup_callbacks in train.py uses MinEpochEarlyStopping."""
        from omegaconf import OmegaConf
        from scripts.train import setup_callbacks
        from src.training.callbacks import MinEpochEarlyStopping

        cfg = OmegaConf.load("configs/default.yaml")
        callbacks = setup_callbacks(cfg)

        # Find the MinEpochEarlyStopping callback
        es_callbacks = [cb for cb in callbacks if isinstance(cb, MinEpochEarlyStopping)]
        assert len(es_callbacks) == 1, "Should have exactly one MinEpochEarlyStopping"

        es = es_callbacks[0]
        assert es.min_epochs == cfg.training.early_stopping.min_epochs
        assert es.patience == cfg.training.early_stopping.patience


class TestModelCheckpoint:
    """Tests for model checkpoint configuration."""

    def test_model_checkpoint_saves_best(self):
        """Checkpoint config monitors val_loss and saves top-k models."""
        from omegaconf import OmegaConf
        cfg = OmegaConf.load("configs/default.yaml")
        ckpt = cfg.training.checkpoint
        assert ckpt.monitor == "val_loss"
        assert ckpt.mode == "min"
        assert ckpt.save_top_k >= 1
        assert ckpt.save_last is True
