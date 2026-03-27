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
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, PropertyMock, patch

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


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

    def test_on_train_epoch_start_sets_temperature_both_gates(self):
        """Callback sets both HGT and CT gene gate temperatures on epoch start."""
        from src.training.callbacks import TemperatureAnnealing
        callback = TemperatureAnnealing(
            tau_max=2.0, tau_min=0.1, warmup_epochs=0,
            anneal_epochs=50, schedule="exponential",
        )

        # Mock trainer and model
        trainer = MagicMock()
        trainer.current_epoch = 10

        pl_module = MagicMock()
        hgt_gate = MagicMock()
        ct_gate = MagicMock()
        # Use PropertyMock to track temperature setter calls
        hgt_temp_prop = PropertyMock()
        ct_temp_prop = PropertyMock()
        type(hgt_gate).temperature = hgt_temp_prop
        type(ct_gate).temperature = ct_temp_prop
        pl_module.model.hgt_gene_gate = hgt_gate
        pl_module.model.cell_transformer.gene_gate = ct_gate

        callback.on_train_epoch_start(trainer, pl_module)

        # Verify temperature was set to the correct value on both gates
        expected_tau = callback.get_temperature(epoch=10)
        hgt_temp_prop.assert_called_with(expected_tau)
        ct_temp_prop.assert_called_with(expected_tau)

    def test_on_train_epoch_start_works_with_only_hgt_gate(self):
        """Callback works when only HGT gene gate exists."""
        from src.training.callbacks import TemperatureAnnealing
        callback = TemperatureAnnealing(
            tau_max=2.0, tau_min=0.1, warmup_epochs=0,
            anneal_epochs=50, schedule="exponential",
        )

        trainer = MagicMock()
        trainer.current_epoch = 5

        pl_module = MagicMock()
        hgt_gate = MagicMock()
        hgt_temp_prop = PropertyMock()
        type(hgt_gate).temperature = hgt_temp_prop
        pl_module.model.hgt_gene_gate = hgt_gate
        # No CT gene gate
        pl_module.model.cell_transformer.gene_gate = None

        callback.on_train_epoch_start(trainer, pl_module)
        expected_tau = callback.get_temperature(epoch=5)
        hgt_temp_prop.assert_called_with(expected_tau)

    def test_on_train_epoch_start_raises_no_gates(self):
        """Callback raises AttributeError when no gene gates exist."""
        from src.training.callbacks import TemperatureAnnealing
        callback = TemperatureAnnealing(
            tau_max=2.0, tau_min=0.1, warmup_epochs=0,
            anneal_epochs=50, schedule="exponential",
        )

        trainer = MagicMock()
        trainer.current_epoch = 0

        pl_module = MagicMock()
        pl_module.model.hgt_gene_gate = None
        pl_module.model.cell_transformer.gene_gate = None

        with pytest.raises(AttributeError, match="at least one gene gate"):
            callback.on_train_epoch_start(trainer, pl_module)


class TestGradientNormLogger:
    """Tests for GradientNormLogger callback."""

    def _make_simple_model(self):
        """Create a simple model with named branches for testing.

        Includes hgt_gene_gate and hgt_input_proj at the model level (these are
        logically part of the HGT branch in the 2-branch architecture).
        """
        model = nn.Module()
        model.hgt_encoder = nn.Linear(10, 5)
        model.hgt_gene_gate = nn.Linear(10, 5)
        model.hgt_input_proj = nn.Linear(10, 5)
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
            model.hgt_encoder(x).sum()
            + model.hgt_gene_gate(x).sum()
            + model.hgt_input_proj(x).sum()
            + model.cell_transformer(x).sum()
        )
        loss.backward()

        norms = callback.compute_branch_norms(model)
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

    def test_disabled_branch_excluded_from_ratio(self):
        """Disabled branches (ablation) are excluded from norm ratio computation."""
        from src.training.callbacks import GradientNormLogger, _BRANCH_FLAG

        callback = GradientNormLogger(log_every_n_steps=1)

        model = self._make_simple_model()
        # Simulate ablation: cell_transformer disabled
        model.use_hgt_encoder = True
        model.use_cell_transformer = False

        # Create gradients for active branches only
        x = torch.randn(2, 10)
        loss = model.hgt_encoder(x).sum() + model.hgt_gene_gate(x).sum()
        loss.backward()

        norms = callback.compute_branch_norms(model)
        # cell_transformer has zero gradients (disabled)
        assert norms["cell_transformer"] == 0.0

        # Filter active branches as the callback does
        active_norms = {
            name: norm for name, norm in norms.items()
            if getattr(model, _BRANCH_FLAG.get(name, ""), True)
        }
        assert "cell_transformer" not in active_norms
        ratio = callback.compute_norm_ratio(active_norms)
        # Ratio should be based only on hgt (> 0), so finite and reasonable
        assert ratio < 100.0


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
        # Default: no Pyro guide (basic RNG tests don't need SVI).
        # Pyro-specific tests set these explicitly.
        pl_module.guide = None
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

        cfg = OmegaConf.load(_PROJECT_ROOT / "configs" / "default.yaml")
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
        assert es.monitor == "val_nll"
        assert es.mode == "min"

    def test_early_stopping_min_epochs(self):
        """Training config includes min_epochs to prevent premature stopping."""
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(_PROJECT_ROOT / "configs" / "default.yaml")
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

        cfg = OmegaConf.load(_PROJECT_ROOT / "configs" / "default.yaml")
        callbacks = setup_callbacks(cfg)

        # Find the MinEpochEarlyStopping callback
        es_callbacks = [cb for cb in callbacks if isinstance(cb, MinEpochEarlyStopping)]
        assert len(es_callbacks) == 1, "Should have exactly one MinEpochEarlyStopping"

        es = es_callbacks[0]
        assert es.min_epochs == cfg.training.early_stopping.min_epochs
        assert es.patience == cfg.training.early_stopping.patience


class TestPyroCheckpointRestore:
    """Tests for Pyro param store and optimizer state checkpoint restore."""

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

    def test_pyro_param_store_restored_with_device_migration(self):
        """Pyro param store loaded to CPU in on_load_checkpoint, then re-synced in on_train_start.

        Note: This is a callback-direct test with mocks — it manually calls hooks
        in the expected Lightning order (on_load_checkpoint → configure_optimizers →
        on_train_start) rather than running a real Trainer.fit(ckpt_path=...).

        After the tensor identity fix, on_train_start re-syncs the param store
        from the guide's nn.Parameters (which are the objects the optimizer tracks).
        The checkpoint values loaded by on_load_checkpoint are effectively replaced
        by the guide's authoritative tensors.
        """
        import pyro

        callback, trainer, pl_module, _ = self._make_checkpoint_context()

        # Build a checkpoint with CPU tensors in pyro_param_store
        checkpoint = {
            "pyro_param_store": {
                "auto_loc": torch.tensor([1.0, 2.0, 3.0], device="cpu"),
                "auto_scale": torch.tensor([0.1, 0.2, 0.3], device="cpu"),
            }
        }

        # Step 1: on_load_checkpoint puts params on CPU and sets resume flag
        callback.on_load_checkpoint(trainer, pl_module, checkpoint)

        store = pyro.get_param_store()
        for k in ["auto_loc", "auto_scale"]:
            param = store[k]
            assert param.device == torch.device("cpu"), (
                f"Param {k} should be on cpu after on_load_checkpoint, got {param.device}"
            )

        # Verify values are correct
        assert torch.allclose(store["auto_loc"], torch.tensor([1.0, 2.0, 3.0]))
        assert torch.allclose(store["auto_scale"], torch.tensor([0.1, 0.2, 0.3]))

        # Step 2: on_train_start re-syncs param store from guide's nn.Parameters.
        # Use a real nn.Module guide (not MagicMock) since the resync path
        # iterates guide._parameters.
        guide = nn.Module()
        guide.auto_loc = nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
        guide.auto_scale = nn.Parameter(torch.tensor([0.1, 0.2, 0.3]))

        # Add Pyro naming attributes (simulating AutoDiagonalNormal guide)
        guide._pyro_name = "AutoDiagonalNormal"
        guide._pyro_params = {}  # No PyroParams in mock

        def _pyro_get_fullname(name):
            return f"AutoDiagonalNormal.{name}"

        guide._pyro_get_fullname = _pyro_get_fullname

        pl_module.guide = guide
        pl_module.device = torch.device("cpu")
        callback.on_train_start(trainer, pl_module)

        # After on_train_start, param store entries should be the guide's
        # actual nn.Parameters (same Python objects), stored under fullnames
        store = pyro.get_param_store()
        for k in ["AutoDiagonalNormal.auto_loc", "AutoDiagonalNormal.auto_scale"]:
            param = store._params[k]
            assert param.device == torch.device("cpu"), (
                f"Param {k} should be on cpu after on_train_start, got {param.device}"
            )
        assert store._params["AutoDiagonalNormal.auto_loc"] is guide.auto_loc
        assert store._params["AutoDiagonalNormal.auto_scale"] is guide.auto_scale

    def test_legacy_checkpoint_without_rng_states_still_restores_pyro(self):
        """Legacy checkpoint without rng_states still restores Pyro param store."""
        import pyro

        callback, trainer, pl_module, _ = self._make_checkpoint_context()
        pl_module.device = torch.device("cpu")

        # Legacy checkpoint: has pyro_param_store but NO rng_states
        checkpoint = {
            "pyro_param_store": {
                "auto_loc": torch.tensor([1.0, 2.0]),
                "auto_scale": torch.tensor([0.5, 0.5]),
            }
        }

        # Should not raise, and should restore Pyro params
        with patch("src.training.callbacks.logger") as mock_logger:
            callback.on_load_checkpoint(trainer, pl_module, checkpoint)
            # Warning about missing rng_states should be logged
            mock_logger.warning.assert_called_once()
            assert "rng_states" in mock_logger.warning.call_args[0][0]

        # Pyro param store should be populated despite missing rng_states
        store = pyro.get_param_store()
        assert "auto_loc" in store
        assert "auto_scale" in store
        assert torch.allclose(store["auto_loc"], torch.tensor([1.0, 2.0]))

    def test_on_train_start_migrates_pyro_params_to_device(self):
        """on_train_start migrates Pyro param store to training device (normal startup)."""
        import pyro

        callback, trainer, pl_module, _ = self._make_checkpoint_context()
        pl_module.guide = MagicMock()
        pl_module.device = torch.device("cpu")
        # Explicitly mark as NOT resuming — MagicMock returns truthy for any attr
        pl_module._pyro_resuming_from_checkpoint = False

        pyro.clear_param_store()
        pyro.get_param_store().setdefault("test_param", torch.tensor([1.0]))

        callback.on_train_start(trainer, pl_module)

        assert pyro.get_param_store()["test_param"].device == torch.device("cpu")


class TestPyroParamStoreResync:
    """Tests for Pyro param store tensor identity re-sync on checkpoint resume.

    When resuming from a checkpoint, Lightning's load_state_dict() copies values
    into the guide's nn.Parameters in-place, but on_load_checkpoint() puts
    different tensor objects into the Pyro param store. This creates a tensor
    identity mismatch: the optimizer tracks tensor A, but the param store holds
    tensor B. The on_train_start re-sync fix must replace param store entries
    with the guide's actual nn.Parameter tensors.
    """

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

    def test_on_train_start_resyncs_param_store_after_resume(self):
        """on_train_start replaces param store entries with guide's nn.Parameters.

        Simulates the tensor identity mismatch that occurs during checkpoint resume:
        1. Guide has nn.Parameters (tensors A) — these are what the optimizer tracks
        2. Param store has cloned tensors (tensors B) — different Python objects
        3. After on_train_start, param store should point to guide's tensors (A)
        """
        import pyro

        callback, trainer, pl_module, _ = self._make_checkpoint_context()

        # Create a real nn.Module as the guide with named parameters
        guide = nn.Module()
        guide.auto_loc = nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
        guide.auto_scale = nn.Parameter(torch.tensor([0.1, 0.2, 0.3]))

        # Add Pyro naming attributes (simulating AutoDiagonalNormal guide)
        guide._pyro_name = "AutoDiagonalNormal"
        guide._pyro_params = {}  # No PyroParams in mock (both are regular nn.Parameters)

        def _pyro_get_fullname(name):
            return f"AutoDiagonalNormal.{name}"

        guide._pyro_get_fullname = _pyro_get_fullname

        pl_module.guide = guide
        pl_module.device = torch.device("cpu")

        # Simulate on_load_checkpoint: clear param store and add CLONED tensors
        # (different Python objects, simulating tensor identity mismatch)
        pyro.clear_param_store()
        store = pyro.get_param_store()
        store.setdefault("AutoDiagonalNormal.auto_loc", guide.auto_loc.data.clone())
        store.setdefault("AutoDiagonalNormal.auto_scale", guide.auto_scale.data.clone())

        # Verify: param store tensors are NOT the same objects as guide params
        assert store._params["AutoDiagonalNormal.auto_loc"] is not guide.auto_loc
        assert store._params["AutoDiagonalNormal.auto_scale"] is not guide.auto_scale

        # Set the resume flag (normally set by on_load_checkpoint)
        pl_module._pyro_resuming_from_checkpoint = True

        # Call on_train_start — should re-sync param store
        callback.on_train_start(trainer, pl_module)

        # Verify: flag is cleared
        assert pl_module._pyro_resuming_from_checkpoint is False

        # Verify: param store now has entries (re-synced from guide)
        store = pyro.get_param_store()
        assert len(store._params) == 2

        # Verify: param store tensors ARE the same objects as guide's parameters
        assert store._params["AutoDiagonalNormal.auto_loc"] is guide.auto_loc
        assert store._params["AutoDiagonalNormal.auto_scale"] is guide.auto_scale

    def test_on_load_checkpoint_sets_resume_flag(self):
        """on_load_checkpoint sets _pyro_resuming_from_checkpoint flag."""
        import pyro

        callback, trainer, pl_module, _ = self._make_checkpoint_context()

        checkpoint = {
            "pyro_param_store": {
                "auto_loc": torch.tensor([1.0, 2.0]),
                "auto_scale": torch.tensor([0.5, 0.5]),
            }
        }

        callback.on_load_checkpoint(trainer, pl_module, checkpoint)

        assert getattr(pl_module, '_pyro_resuming_from_checkpoint', False) is True

    def test_on_train_start_resyncs_pyro_params(self):
        """on_train_start correctly handles PyroParam entries (e.g., scale).

        PyroParams are stored as unconstrained nn.Parameters (name + '_unconstrained')
        but the Pyro param store key uses the base name without the suffix.
        """
        import pyro

        callback, trainer, pl_module, _ = self._make_checkpoint_context()

        # Create a guide with both regular nn.Parameter and simulated PyroParam
        guide = nn.Module()
        guide.auto_loc = nn.Parameter(torch.tensor([1.0, 2.0]))
        # PyroParam stores unconstrained value as name + '_unconstrained'
        guide.scale_unconstrained = nn.Parameter(torch.tensor([0.5, 0.5]))

        guide._pyro_name = "AutoDiagonalNormal"
        # Mark 'scale' as a PyroParam — the resync code uses this to find
        # the unconstrained parameter and register with the base name
        guide._pyro_params = {"scale": (None, 0)}  # (constraint, event_dim)

        def _pyro_get_fullname(name):
            return f"AutoDiagonalNormal.{name}"

        guide._pyro_get_fullname = _pyro_get_fullname

        pl_module.guide = guide
        pl_module.device = torch.device("cpu")

        # Simulate stale param store with cloned tensors
        pyro.clear_param_store()
        store = pyro.get_param_store()
        store._params["AutoDiagonalNormal.auto_loc"] = guide.auto_loc.data.clone()
        store._params["AutoDiagonalNormal.scale"] = guide.scale_unconstrained.data.clone()
        store._param_to_name[store._params["AutoDiagonalNormal.auto_loc"]] = "AutoDiagonalNormal.auto_loc"
        store._param_to_name[store._params["AutoDiagonalNormal.scale"]] = "AutoDiagonalNormal.scale"

        # Verify mismatch exists
        assert store._params["AutoDiagonalNormal.auto_loc"] is not guide.auto_loc
        assert store._params["AutoDiagonalNormal.scale"] is not guide.scale_unconstrained

        pl_module._pyro_resuming_from_checkpoint = True
        callback.on_train_start(trainer, pl_module)

        # Verify: regular param re-synced via identity
        store = pyro.get_param_store()
        assert store._params["AutoDiagonalNormal.auto_loc"] is guide.auto_loc

        # Verify: PyroParam re-synced — unconstrained param stored under base name
        assert store._params["AutoDiagonalNormal.scale"] is guide.scale_unconstrained

    def test_normal_startup_does_not_resync(self):
        """on_train_start without resume flag just migrates device, does not clear store."""
        import pyro

        callback, trainer, pl_module, _ = self._make_checkpoint_context()

        guide = nn.Module()
        guide.test_param = nn.Parameter(torch.tensor([1.0]))
        pl_module.guide = guide
        pl_module.device = torch.device("cpu")
        # Explicitly mark as NOT resuming — MagicMock returns truthy for any attr
        pl_module._pyro_resuming_from_checkpoint = False

        # Populate param store without setting resume flag
        pyro.clear_param_store()
        store = pyro.get_param_store()
        original_tensor = torch.tensor([42.0])
        store.setdefault("existing_param", original_tensor)

        # No resume flag set — should NOT clear and re-sync
        callback.on_train_start(trainer, pl_module)

        # Param store should still have the original entry (not guide's params)
        store = pyro.get_param_store()
        assert "existing_param" in store
        assert torch.allclose(store["existing_param"], torch.tensor([42.0]))


class TestModelCheckpoint:
    """Tests for model checkpoint configuration."""

    def test_model_checkpoint_saves_best(self):
        """Checkpoint config monitors val_nll and saves top-k models."""
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(_PROJECT_ROOT / "configs" / "default.yaml")
        ckpt = cfg.training.checkpoint
        assert ckpt.monitor == "val_nll"
        assert ckpt.mode == "min"
        assert ckpt.save_top_k >= 1
        assert ckpt.save_last is True
