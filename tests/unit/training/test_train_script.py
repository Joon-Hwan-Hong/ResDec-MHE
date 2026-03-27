"""
Tests for scripts/train.py helper functions.

Tests the composable pieces of the training script:
- Config loading with CLI overrides
- Callback setup from config
- Trainer configuration
- Seed setting
- Integration test for main() with synthetic data
"""

import pytest
import torch
from omegaconf import OmegaConf
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.data.constants import N_CELL_TYPES, N_REGIONS, CELL_TYPE_ORDER, ALL_EDGE_TYPES, sanitize_key


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
                "n_pma_seeds": 1,
            },
            "cell_type_selector": {"selection_temperature": 1.0},
            "pathology_attention": {"d_cond": 16, "n_heads": 4, "n_pathology_features": 3},
            "head": {"type": "deterministic", "d_hidden": 16},
        },
        "training": {
            "max_epochs": 10,
            "precision": "32-true",
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

    def test_setup_callbacks_includes_resilience_checkpoint(self, train_config):
        """Callbacks include ResilienceModelCheckpoint."""
        from scripts.train import setup_callbacks
        from src.training.callbacks import ResilienceModelCheckpoint

        callbacks = setup_callbacks(train_config)
        ckpts = [c for c in callbacks if isinstance(c, ResilienceModelCheckpoint)]
        assert len(ckpts) == 1

    def test_setup_callbacks_includes_lr_monitor(self, train_config):
        """Callbacks include LearningRateMonitor for deterministic head."""
        from scripts.train import setup_callbacks
        import lightning.pytorch as pl

        callbacks = setup_callbacks(train_config)
        monitors = [c for c in callbacks if isinstance(c, pl.callbacks.LearningRateMonitor)]
        assert len(monitors) == 1

    def test_lr_monitor_present_for_bayesian_head(self, train_config):
        """LearningRateMonitor IS in callbacks for bayesian head (uses standard torch optimizer)."""
        from scripts.train import setup_callbacks
        import lightning.pytorch as pl

        train_config.model.head.type = "bayesian"
        callbacks = setup_callbacks(train_config)
        monitors = [c for c in callbacks if isinstance(c, pl.callbacks.LearningRateMonitor)]
        assert len(monitors) == 1

    def test_lr_monitor_present_for_deterministic_head(self, train_config):
        """LearningRateMonitor IS in callbacks when head.type is deterministic."""
        from scripts.train import setup_callbacks
        import lightning.pytorch as pl

        train_config.model.head.type = "deterministic"
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

    def test_setup_trainer_benchmark_false(self, train_config, tmp_path):
        """Trainer sets torch.backends.cudnn.benchmark=False for reproducibility."""
        from scripts.train import setup_trainer

        train_config.paths.output_dir = str(tmp_path)
        train_config.paths.logs_dir = str(tmp_path / "logs")
        train_config.paths.checkpoint_dir = str(tmp_path / "checkpoints")
        train_config.reproducibility = {"deterministic": True, "benchmark": False}

        setup_trainer(train_config)
        assert torch.backends.cudnn.benchmark is False

    def test_setup_trainer_deterministic_true(self, train_config, tmp_path):
        """Trainer sets deterministic=True for reproducibility via torch."""
        from scripts.train import setup_trainer
        train_config.paths.output_dir = str(tmp_path)
        train_config.paths.logs_dir = str(tmp_path / "logs")
        train_config.paths.checkpoint_dir = str(tmp_path / "checkpoints")
        train_config.reproducibility = {"deterministic": True, "benchmark": False}
        setup_trainer(train_config)
        assert torch.are_deterministic_algorithms_enabled() is True

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


class TestMainHelp:
    """Smoke tests for main() entry point."""

    def test_train_help_exits_clean(self, monkeypatch):
        """train.py --help exits with code 0."""
        monkeypatch.setattr("sys.argv", ["train.py", "--help"])
        with pytest.raises(SystemExit) as exc_info:
            from scripts.train import main
            main()
        assert exc_info.value.code == 0


class TestConfigToTrainerWiring:
    """Tests that config flows through to Trainer correctly."""

    def test_callbacks_and_trainer_match_config(self, train_config, tmp_path):
        """setup_callbacks + setup_trainer produce correct Trainer config."""
        from scripts.train import setup_callbacks, setup_trainer
        import lightning.pytorch as pl

        train_config.paths.output_dir = str(tmp_path)
        train_config.paths.logs_dir = str(tmp_path / "logs")
        train_config.paths.checkpoint_dir = str(tmp_path / "checkpoints")
        train_config.reproducibility = {"deterministic": True, "benchmark": False}

        callbacks = setup_callbacks(train_config)
        trainer = setup_trainer(train_config, callbacks=callbacks)

        # Check all expected callback types present
        from src.training.callbacks import (
            TemperatureAnnealing,
            GradientNormLogger,
            ResilienceModelCheckpoint,
            MinEpochEarlyStopping,
        )
        callback_types = [type(c) for c in callbacks]
        assert pl.callbacks.ModelCheckpoint in callback_types
        assert MinEpochEarlyStopping in callback_types  # Custom wrapper, not base EarlyStopping
        assert pl.callbacks.LearningRateMonitor in callback_types
        assert TemperatureAnnealing in callback_types
        assert GradientNormLogger in callback_types
        assert ResilienceModelCheckpoint in callback_types

        # Verify trainer config matches
        assert trainer.max_epochs == train_config.training.max_epochs
        assert trainer.gradient_clip_val == train_config.training.gradient_clip_val


def _make_synthetic_batch(batch_size=2, n_genes=50, n_cell_types=N_CELL_TYPES,
                          n_regions=N_REGIONS, max_cells=10):
    """Create synthetic batch matching collate_for_hgt_multiregion format."""
    region_pseudobulk = torch.randn(batch_size, n_regions, n_cell_types, n_genes)
    region_mask = torch.zeros(batch_size, n_regions, dtype=torch.bool)
    region_mask[:, 0] = True  # PFC only

    # HGT flat edge tensors — 4 edges per sample, concatenated across batch
    n_edges_per_sample = 4
    n_edges_total = batch_size * n_edges_per_sample
    per_sample_idx = torch.randint(0, n_cell_types, (2, n_edges_per_sample))
    edge_parts = [per_sample_idx + b * n_cell_types for b in range(batch_size)]
    ccc_edge_index = torch.cat(edge_parts, dim=1)  # [2, n_edges_total]
    ccc_edge_type = torch.zeros(n_edges_total, dtype=torch.long)
    ccc_edge_attr = torch.rand(n_edges_total, 1)

    # Flat cell format
    total_per_sample = n_cell_types * max_cells
    total_cells = batch_size * total_per_sample
    cell_data = torch.randn(total_cells, n_genes)
    cell_offsets = torch.zeros(batch_size, n_cell_types + 1, dtype=torch.long)
    for b in range(batch_size):
        base = b * total_per_sample
        for ct in range(n_cell_types):
            cell_offsets[b, ct + 1] = cell_offsets[b, ct] + max_cells
        cell_offsets[b] += base

    cell_type_mask = torch.ones(batch_size, n_cell_types, dtype=torch.bool)
    pathology = torch.rand(batch_size, 3)
    cognition = torch.randn(batch_size, 1)

    return {
        "region_pseudobulk": region_pseudobulk,
        "region_mask": region_mask,
        "ccc_edge_index": ccc_edge_index,
        "ccc_edge_type": ccc_edge_type,
        "ccc_edge_attr": ccc_edge_attr,
        "cell_data": cell_data,
        "cell_offsets": cell_offsets,
        "cell_type_mask": cell_type_mask,
        "pathology": pathology,
        "cognition": cognition,
    }


class SyntheticDataset(torch.utils.data.Dataset):
    """Minimal synthetic dataset for integration testing."""

    def __init__(self, n_samples=8, n_genes=50):
        self.n_samples = n_samples
        self.n_genes = n_genes

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Return single sample (not batched)
        batch = _make_synthetic_batch(batch_size=1, n_genes=self.n_genes)
        return {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v[0]
                for k, v in batch.items()}


class TestMainIntegration:
    """Integration tests for train.py main() with synthetic data."""

    def test_main_trains_one_epoch_with_synthetic_data(self, train_config, tmp_path):
        """main() completes one epoch with synthetic data (mocked loaders)."""
        import lightning.pytorch as pl
        from scripts.train import setup_callbacks
        from src.training.lightning_module import CognitiveResilienceLightningModule

        # Configure for minimal training (scheduler needs warmup < max_epochs)
        train_config.training.max_epochs = 3
        train_config.training.scheduler.warmup_epochs = 1
        train_config.training.early_stopping.min_epochs = 1
        train_config.paths.output_dir = str(tmp_path)
        train_config.paths.logs_dir = str(tmp_path / "logs")
        train_config.paths.checkpoint_dir = str(tmp_path / "checkpoints")
        (tmp_path / "checkpoints").mkdir(parents=True, exist_ok=True)

        # Create synthetic dataloaders
        train_ds = SyntheticDataset(n_samples=4, n_genes=train_config.model.n_genes)
        val_ds = SyntheticDataset(n_samples=2, n_genes=train_config.model.n_genes)

        # Custom collate that handles the dict format
        # Edge tensors use flat concatenation with node offsets (not stacking)
        _SPECIAL_KEYS = {"ccc_edge_index", "ccc_edge_type", "ccc_edge_attr", "cell_data", "cell_offsets"}
        def synthetic_collate(samples):
            batch = {}
            n_cell_types = samples[0]["region_pseudobulk"].shape[-2]
            for key in samples[0].keys():
                if not isinstance(samples[0][key], torch.Tensor):
                    batch[key] = [s[key] for s in samples]
                elif key in _SPECIAL_KEYS:
                    continue  # handle below
                else:
                    batch[key] = torch.stack([s[key] for s in samples])
            # Flat edge concatenation with node offsets
            ei_parts, et_parts, ea_parts = [], [], []
            for i, s in enumerate(samples):
                ei = s["ccc_edge_index"]
                n_edges = ei.shape[1] if ei.numel() > 0 else 0
                if n_edges > 0:
                    ei_parts.append(ei + i * n_cell_types)
                    et_parts.append(s["ccc_edge_type"])
                    ea_parts.append(s["ccc_edge_attr"])
            if ei_parts:
                batch["ccc_edge_index"] = torch.cat(ei_parts, dim=1)
                batch["ccc_edge_type"] = torch.cat(et_parts)
                batch["ccc_edge_attr"] = torch.cat(ea_parts)
            else:
                batch["ccc_edge_index"] = torch.zeros(2, 0, dtype=torch.long)
                batch["ccc_edge_type"] = torch.zeros(0, dtype=torch.long)
                batch["ccc_edge_attr"] = torch.zeros(0, 1)
            # Flat cell_data + cell_offsets with cumulative global offsets
            all_data = [s["cell_data"] for s in samples if s["cell_data"].shape[0] > 0]
            n_genes = samples[0]["region_pseudobulk"].shape[-1]
            batch["cell_data"] = torch.cat(all_data) if all_data else torch.empty(0, n_genes)
            cumulative = 0
            adjusted_offsets = []
            for s in samples:
                adjusted_offsets.append(s["cell_offsets"].unsqueeze(0) + cumulative)
                cumulative += int(s["cell_offsets"][-1].item())
            batch["cell_offsets"] = torch.cat(adjusted_offsets, dim=0)
            return batch

        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=2, collate_fn=synthetic_collate
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=2, collate_fn=synthetic_collate
        )

        # Build module and trainer
        module = CognitiveResilienceLightningModule(train_config)

        # Filter out callbacks incompatible with enable_checkpointing=False or logger=False
        callbacks = [
            cb for cb in setup_callbacks(train_config)
            if not isinstance(cb, (pl.callbacks.ModelCheckpoint, pl.callbacks.LearningRateMonitor))
        ]

        trainer = pl.Trainer(
            max_epochs=1,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
            logger=False,
            callbacks=callbacks,
        )

        # Train should complete without error
        trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # Verify training happened
        assert trainer.current_epoch == 1
        assert trainer.global_step > 0

    def test_main_logs_val_loss(self, train_config, tmp_path):
        """Training logs val_loss metric for checkpointing/early stopping."""
        import lightning.pytorch as pl
        from src.training.lightning_module import CognitiveResilienceLightningModule

        # Configure for minimal training (scheduler needs warmup < max_epochs)
        train_config.training.max_epochs = 3
        train_config.training.scheduler.warmup_epochs = 1
        train_config.paths.output_dir = str(tmp_path)
        train_config.paths.logs_dir = str(tmp_path / "logs")
        train_config.paths.checkpoint_dir = str(tmp_path / "checkpoints")
        (tmp_path / "checkpoints").mkdir(parents=True, exist_ok=True)

        train_ds = SyntheticDataset(n_samples=4, n_genes=train_config.model.n_genes)
        val_ds = SyntheticDataset(n_samples=2, n_genes=train_config.model.n_genes)

        _SPECIAL_KEYS2 = {"ccc_edge_index", "ccc_edge_type", "ccc_edge_attr", "cell_data", "cell_offsets"}
        def synthetic_collate(samples):
            batch = {}
            n_cell_types = samples[0]["region_pseudobulk"].shape[-2]
            for key in samples[0].keys():
                if not isinstance(samples[0][key], torch.Tensor):
                    batch[key] = [s[key] for s in samples]
                elif key in _SPECIAL_KEYS2:
                    continue
                else:
                    batch[key] = torch.stack([s[key] for s in samples])
            ei_parts, et_parts, ea_parts = [], [], []
            for i, s in enumerate(samples):
                ei = s["ccc_edge_index"]
                n_edges = ei.shape[1] if ei.numel() > 0 else 0
                if n_edges > 0:
                    ei_parts.append(ei + i * n_cell_types)
                    et_parts.append(s["ccc_edge_type"])
                    ea_parts.append(s["ccc_edge_attr"])
            if ei_parts:
                batch["ccc_edge_index"] = torch.cat(ei_parts, dim=1)
                batch["ccc_edge_type"] = torch.cat(et_parts)
                batch["ccc_edge_attr"] = torch.cat(ea_parts)
            else:
                batch["ccc_edge_index"] = torch.zeros(2, 0, dtype=torch.long)
                batch["ccc_edge_type"] = torch.zeros(0, dtype=torch.long)
                batch["ccc_edge_attr"] = torch.zeros(0, 1)
            # Flat cell_data + cell_offsets with cumulative global offsets
            all_data = [s["cell_data"] for s in samples if s["cell_data"].shape[0] > 0]
            n_genes = samples[0]["region_pseudobulk"].shape[-1]
            batch["cell_data"] = torch.cat(all_data) if all_data else torch.empty(0, n_genes)
            cumulative = 0
            adjusted_offsets = []
            for s in samples:
                adjusted_offsets.append(s["cell_offsets"].unsqueeze(0) + cumulative)
                cumulative += int(s["cell_offsets"][-1].item())
            batch["cell_offsets"] = torch.cat(adjusted_offsets, dim=0)
            return batch

        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=2, collate_fn=synthetic_collate
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=2, collate_fn=synthetic_collate
        )

        module = CognitiveResilienceLightningModule(train_config)
        trainer = pl.Trainer(
            max_epochs=1,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
            logger=False,
        )

        trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # val_loss should be logged
        assert "val_loss" in trainer.callback_metrics
        assert trainer.callback_metrics["val_loss"].item() > 0
