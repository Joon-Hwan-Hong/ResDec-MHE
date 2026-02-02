"""
Tests for scripts/train.py helper functions.

Tests the composable pieces of the training script:
- Config loading with CLI overrides
- Callback setup from config
- Trainer configuration
- Seed setting
"""

import pytest
import torch
from omegaconf import OmegaConf
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.data.constants import N_CELL_TYPES, N_REGIONS


@pytest.fixture
def train_config():
    """Minimal config for training script testing."""
    return OmegaConf.create({
        "experiment": {
            "name": "test_run",
            "seed": 42,
            "device": "auto",
        },
        "model": {
            "n_genes": 50,
            "n_cell_types": N_CELL_TYPES,
            "d_embed": 32,
            "d_fused": 32,
            "n_regions": N_REGIONS,
            "dropout": 0.1,
            "gene_gate": {"initial_temperature": 2.0},
            "hgt": {"n_layers": 1, "n_heads": 4},
            "set_transformer": {
                "n_isab_layers": 1,
                "n_inducing_points": 4,
                "n_heads": 4,
            },
            "cell_type_selector": {"selection_temperature": 1.0},
            "pathology_attention": {"d_cond": 16, "n_heads": 4},
            "head": {"type": "deterministic", "d_hidden": 16},
        },
        "training": {
            "max_epochs": 10,
            "precision": "32",
            "gradient_clip_val": 1.0,
            "optimizer": {
                "type": "adamw",
                "lr": 1e-3,
                "weight_decay": 1e-4,
            },
            "scheduler": {
                "type": "cosine",
                "warmup_epochs": 2,
                "eta_min": 1e-6,
            },
            "loss": {"type": "mse", "beta": 0.5},
            "early_stopping": {
                "patience": 5,
                "min_delta": 0.0001,
                "min_epochs": 3,
                "monitor": "val_loss",
                "mode": "min",
            },
            "checkpoint": {
                "save_top_k": 1,
                "monitor": "val_loss",
                "mode": "min",
                "save_last": True,
            },
            "temperature_annealing": {
                "tau_max": 2.0,
                "tau_min": 0.1,
                "warmup_epochs": 2,
                "anneal_epochs": 5,
                "schedule": "exponential",
            },
            "regularization": {"gene_gate_l1": 0.0},
            "logging": {
                "log_every_n_steps": 5,
                "val_check_interval": 1.0,
            },
        },
        "data": {
            "splits": {
                "test_frac": 0.1,
                "n_folds": 5,
            },
            "dataloader": {
                "batch_size": 4,
                "num_workers": 0,
                "pin_memory": False,
                "prefetch_factor": 2,
                "use_heterodata": True,
            },
            "cell_sampling": {
                "max_cells_per_type": 100,
                "min_cells_threshold": 10,
                "sampling_strategy": "random",
            },
        },
        "paths": {
            "output_dir": "outputs/",
            "checkpoint_dir": "outputs/checkpoints/",
            "logs_dir": "outputs/logs/",
        },
    })


class TestLoadConfig:
    """Tests for config loading and merging."""

    def test_load_config_from_yaml(self, tmp_path, train_config):
        """load_config loads a YAML file into OmegaConf."""
        from scripts.train import load_config

        config_path = tmp_path / "test_config.yaml"
        OmegaConf.save(train_config, config_path)

        config = load_config(str(config_path))
        assert config.experiment.name == "test_run"
        assert config.experiment.seed == 42

    def test_load_config_with_overrides(self, tmp_path, train_config):
        """load_config applies CLI overrides to loaded config."""
        from scripts.train import load_config

        config_path = tmp_path / "test_config.yaml"
        OmegaConf.save(train_config, config_path)

        overrides = ["training.max_epochs=50", "experiment.seed=123"]
        config = load_config(str(config_path), overrides=overrides)
        assert config.training.max_epochs == 50
        assert config.experiment.seed == 123

    def test_load_config_nonexistent_file_raises(self):
        """load_config raises FileNotFoundError for missing config."""
        from scripts.train import load_config

        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.yaml")


class TestSetupCallbacks:
    """Tests for callback setup from config."""

    def test_setup_callbacks_returns_list(self, train_config):
        """setup_callbacks returns a list of Lightning callbacks."""
        from scripts.train import setup_callbacks

        callbacks = setup_callbacks(train_config)
        assert isinstance(callbacks, list)
        assert len(callbacks) > 0

    def test_setup_callbacks_includes_model_checkpoint(self, train_config):
        """Callbacks include ModelCheckpoint monitoring val_loss."""
        from scripts.train import setup_callbacks
        import lightning.pytorch as pl

        callbacks = setup_callbacks(train_config)
        checkpointers = [c for c in callbacks if isinstance(c, pl.callbacks.ModelCheckpoint)]
        assert len(checkpointers) == 1
        assert checkpointers[0].monitor == "val_loss"

    def test_setup_callbacks_includes_early_stopping(self, train_config):
        """Callbacks include EarlyStopping with configured patience."""
        from scripts.train import setup_callbacks
        import lightning.pytorch as pl

        callbacks = setup_callbacks(train_config)
        stoppers = [c for c in callbacks if isinstance(c, pl.callbacks.EarlyStopping)]
        assert len(stoppers) == 1
        assert stoppers[0].patience == 5

    def test_setup_callbacks_includes_temperature_annealing(self, train_config):
        """Callbacks include TemperatureAnnealing with config values."""
        from scripts.train import setup_callbacks
        from src.training.callbacks import TemperatureAnnealing

        callbacks = setup_callbacks(train_config)
        annealers = [c for c in callbacks if isinstance(c, TemperatureAnnealing)]
        assert len(annealers) == 1
        assert annealers[0].tau_max == 2.0
        assert annealers[0].tau_min == 0.1

    def test_setup_callbacks_includes_gradient_norm_logger(self, train_config):
        """Callbacks include GradientNormLogger."""
        from scripts.train import setup_callbacks
        from src.training.callbacks import GradientNormLogger

        callbacks = setup_callbacks(train_config)
        loggers = [c for c in callbacks if isinstance(c, GradientNormLogger)]
        assert len(loggers) == 1

    def test_setup_callbacks_includes_lr_monitor(self, train_config):
        """Callbacks include LearningRateMonitor."""
        from scripts.train import setup_callbacks
        import lightning.pytorch as pl

        callbacks = setup_callbacks(train_config)
        monitors = [c for c in callbacks if isinstance(c, pl.callbacks.LearningRateMonitor)]
        assert len(monitors) == 1


class TestSetupTrainer:
    """Tests for Trainer configuration."""

    def test_setup_trainer_returns_trainer(self, train_config, tmp_path):
        """setup_trainer returns a Lightning Trainer."""
        from scripts.train import setup_trainer
        import lightning.pytorch as pl

        train_config.paths.output_dir = str(tmp_path)
        train_config.paths.logs_dir = str(tmp_path / "logs")
        train_config.paths.checkpoint_dir = str(tmp_path / "checkpoints")

        trainer = setup_trainer(train_config)
        assert isinstance(trainer, pl.Trainer)

    def test_setup_trainer_max_epochs(self, train_config, tmp_path):
        """Trainer uses max_epochs from config."""
        from scripts.train import setup_trainer

        train_config.paths.output_dir = str(tmp_path)
        train_config.paths.logs_dir = str(tmp_path / "logs")
        train_config.paths.checkpoint_dir = str(tmp_path / "checkpoints")

        trainer = setup_trainer(train_config)
        assert trainer.max_epochs == 10

    def test_setup_trainer_gradient_clipping(self, train_config, tmp_path):
        """Trainer uses gradient_clip_val from config."""
        from scripts.train import setup_trainer

        train_config.paths.output_dir = str(tmp_path)
        train_config.paths.logs_dir = str(tmp_path / "logs")
        train_config.paths.checkpoint_dir = str(tmp_path / "checkpoints")

        trainer = setup_trainer(train_config)
        assert trainer.gradient_clip_val == 1.0

    def test_setup_trainer_has_callbacks(self, train_config, tmp_path):
        """Trainer has callbacks configured."""
        from scripts.train import setup_trainer

        train_config.paths.output_dir = str(tmp_path)
        train_config.paths.logs_dir = str(tmp_path / "logs")
        train_config.paths.checkpoint_dir = str(tmp_path / "checkpoints")

        trainer = setup_trainer(train_config)
        assert len(trainer.callbacks) > 0


class TestSetSeed:
    """Tests for reproducibility seed setting."""

    def test_set_seed_makes_torch_deterministic(self):
        """set_seed sets torch random seed for reproducibility."""
        from scripts.train import set_seed

        set_seed(42)
        a = torch.randn(5)
        set_seed(42)
        b = torch.randn(5)
        assert torch.allclose(a, b)

    def test_set_seed_different_seeds_differ(self):
        """Different seeds produce different random values."""
        from scripts.train import set_seed

        set_seed(42)
        a = torch.randn(5)
        set_seed(123)
        b = torch.randn(5)
        assert not torch.allclose(a, b)
