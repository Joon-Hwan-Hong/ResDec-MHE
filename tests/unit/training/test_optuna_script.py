"""
Tests for scripts/optuna_optimize.py helper functions.

Tests the composable pieces of the HP optimization script:
- Study creation with TPE sampler and Hyperband pruner
- Hyperparameter sampling from search space
- Trial config building (overriding base config with sampled params)
"""

import pytest
from omegaconf import OmegaConf
from unittest.mock import MagicMock

from src.data.constants import N_CELL_TYPES, N_REGIONS


@pytest.fixture
def optuna_config():
    """Config for Optuna script testing (includes optuna section)."""
    return OmegaConf.create({
        "experiment": {
            "name": "optuna_test",
            "seed": 42,
            "device": "auto",
        },
        "model": {
            "n_genes": 50,
            "n_cell_types": N_CELL_TYPES,
            "d_embed": 128,
            "d_fused": 128,
            "n_regions": N_REGIONS,
            "dropout": 0.1,
            "pseudobulk": {"dropout": 0.1},
            "gene_gate": {"initial_temperature": 2.0},
            "hgt": {"n_layers": 3, "n_heads": 4, "dropout": 0.1},
            "set_transformer": {
                "n_isab_layers": 2,
                "n_inducing_points": 32,
                "n_heads": 4,
                "dropout": 0.1,
            },
            "cell_type_selector": {"selection_temperature": 1.0},
            "pathology_attention": {"d_cond": 64, "n_heads": 4},
            "head": {"type": "bayesian", "d_hidden": 64},
        },
        "training": {
            "max_epochs": 100,
            "precision": "32",
            "gradient_clip_val": 1.0,
            "optimizer": {
                "type": "adamw",
                "lr": 1e-4,
                "weight_decay": 1e-4,
            },
            "scheduler": {
                "type": "cosine",
                "warmup_epochs": 5,
                "eta_min": 1e-6,
            },
            "loss": {"type": "beta_nll", "beta": 0.5},
            "early_stopping": {
                "patience": 15,
                "min_delta": 0.0001,
                "min_epochs": 20,
                "monitor": "val_loss",
                "mode": "min",
            },
            "checkpoint": {
                "save_top_k": 1,
                "monitor": "val_loss",
                "mode": "min",
                "save_last": False,
            },
            "temperature_annealing": {
                "tau_max": 2.0,
                "tau_min": 0.1,
                "warmup_epochs": 5,
                "anneal_epochs": 50,
                "schedule": "exponential",
            },
            "regularization": {"gene_gate_l1": 0.0},
            "logging": {
                "log_every_n_steps": 10,
                "val_check_interval": 1.0,
            },
        },
        "data": {
            "splits": {
                "test_frac": 0.1,
                "n_folds": 5,
            },
            "dataloader": {
                "batch_size": 16,
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
        "optuna": {
            "n_trials": 10,
            "timeout": 3600,
            "pruner": {
                "type": "hyperband",
                "min_resource": 5,
                "max_resource": 100,
                "reduction_factor": 3,
            },
            "sampler": {
                "type": "tpe",
                "seed": 42,
                "n_startup_trials": 5,
            },
            "search_space": {
                "lr": {"type": "loguniform", "low": 1e-5, "high": 1e-2},
                "d_embed": {"type": "categorical", "choices": [64, 128, 256]},
                "dropout": {"type": "uniform", "low": 0.0, "high": 0.3},
                "n_hgt_layers": {"type": "int", "low": 2, "high": 4},
                "beta": {"type": "uniform", "low": 0.0, "high": 1.0},
            },
        },
        "paths": {
            "output_dir": "outputs/",
            "checkpoint_dir": "outputs/checkpoints/",
            "logs_dir": "outputs/logs/",
        },
    })


class TestCreateStudy:
    """Tests for Optuna study creation."""

    def test_create_study_returns_study(self, optuna_config):
        """create_study returns an Optuna Study object."""
        from scripts.optuna_optimize import create_study

        study = create_study(optuna_config)
        import optuna
        assert isinstance(study, optuna.Study)

    def test_create_study_uses_tpe_sampler(self, optuna_config):
        """Study uses TPE sampler as specified."""
        from scripts.optuna_optimize import create_study
        import optuna

        study = create_study(optuna_config)
        assert isinstance(study.sampler, optuna.samplers.TPESampler)

    def test_create_study_uses_hyperband_pruner(self, optuna_config):
        """Study uses Hyperband pruner as specified."""
        from scripts.optuna_optimize import create_study
        import optuna

        study = create_study(optuna_config)
        assert isinstance(study.pruner, optuna.pruners.HyperbandPruner)

    def test_create_study_minimizes(self, optuna_config):
        """Study direction is minimize (lower val_loss = better)."""
        from scripts.optuna_optimize import create_study

        study = create_study(optuna_config)
        assert study.direction.name == "MINIMIZE"


class TestSampleHyperparameters:
    """Tests for hyperparameter sampling from trials."""

    def test_sample_hyperparameters_returns_dict(self, optuna_config):
        """sample_hyperparameters returns a dict of sampled values."""
        from scripts.optuna_optimize import sample_hyperparameters, create_study

        study = create_study(optuna_config)
        trial = study.ask()
        params = sample_hyperparameters(trial, optuna_config)
        assert isinstance(params, dict)

    def test_sample_hyperparameters_has_expected_keys(self, optuna_config):
        """Sampled params include all search space keys."""
        from scripts.optuna_optimize import sample_hyperparameters, create_study

        study = create_study(optuna_config)
        trial = study.ask()
        params = sample_hyperparameters(trial, optuna_config)

        expected_keys = set(optuna_config.optuna.search_space.keys())
        assert set(params.keys()) == expected_keys

    def test_sample_lr_in_range(self, optuna_config):
        """Sampled lr falls within configured log-uniform range."""
        from scripts.optuna_optimize import sample_hyperparameters, create_study

        study = create_study(optuna_config)
        trial = study.ask()
        params = sample_hyperparameters(trial, optuna_config)

        assert 1e-5 <= params["lr"] <= 1e-2

    def test_sample_d_embed_is_categorical(self, optuna_config):
        """Sampled d_embed is one of the configured choices."""
        from scripts.optuna_optimize import sample_hyperparameters, create_study

        study = create_study(optuna_config)
        trial = study.ask()
        params = sample_hyperparameters(trial, optuna_config)

        assert params["d_embed"] in [64, 128, 256]

    def test_sample_n_hgt_layers_is_integer_in_range(self, optuna_config):
        """Sampled n_hgt_layers is an int within configured range."""
        from scripts.optuna_optimize import sample_hyperparameters, create_study

        study = create_study(optuna_config)
        trial = study.ask()
        params = sample_hyperparameters(trial, optuna_config)

        assert isinstance(params["n_hgt_layers"], int)
        assert 2 <= params["n_hgt_layers"] <= 4


class TestObjectiveFunction:
    """Tests for objective function signature and GPU support."""

    def test_objective_accepts_gpu_id(self, optuna_config):
        """objective() accepts gpu_id parameter."""
        import inspect
        from scripts.optuna_optimize import objective
        sig = inspect.signature(objective)
        assert "gpu_id" in sig.parameters

    def test_objective_accepts_data_params(self, optuna_config):
        """objective() accepts adata, metadata, and splits parameters."""
        import inspect
        from scripts.optuna_optimize import objective
        sig = inspect.signature(objective)
        assert "adata" in sig.parameters
        assert "metadata" in sig.parameters
        assert "splits" in sig.parameters

    def test_create_study_accepts_storage(self, optuna_config):
        """create_study() accepts optional storage parameter."""
        import inspect
        from scripts.optuna_optimize import create_study
        sig = inspect.signature(create_study)
        assert "storage" in sig.parameters


class TestBuildTrialConfig:
    """Tests for overriding base config with trial parameters."""

    def test_build_trial_config_overrides_lr(self, optuna_config):
        """build_trial_config overrides optimizer lr."""
        from scripts.optuna_optimize import build_trial_config

        params = {"lr": 5e-4, "d_embed": 128, "dropout": 0.2,
                  "n_hgt_layers": 3, "beta": 0.7}
        trial_config = build_trial_config(optuna_config, params)
        assert trial_config.training.optimizer.lr == 5e-4

    def test_build_trial_config_overrides_d_embed(self, optuna_config):
        """build_trial_config overrides model d_embed."""
        from scripts.optuna_optimize import build_trial_config

        params = {"lr": 1e-4, "d_embed": 256, "dropout": 0.1,
                  "n_hgt_layers": 2, "beta": 0.5}
        trial_config = build_trial_config(optuna_config, params)
        assert trial_config.model.d_embed == 256

    def test_build_trial_config_overrides_dropout(self, optuna_config):
        """build_trial_config overrides model dropout."""
        from scripts.optuna_optimize import build_trial_config

        params = {"lr": 1e-4, "d_embed": 128, "dropout": 0.25,
                  "n_hgt_layers": 3, "beta": 0.5}
        trial_config = build_trial_config(optuna_config, params)
        assert trial_config.model.dropout == 0.25

    def test_build_trial_config_overrides_hgt_layers(self, optuna_config):
        """build_trial_config overrides model hgt n_layers."""
        from scripts.optuna_optimize import build_trial_config

        params = {"lr": 1e-4, "d_embed": 128, "dropout": 0.1,
                  "n_hgt_layers": 4, "beta": 0.5}
        trial_config = build_trial_config(optuna_config, params)
        assert trial_config.model.hgt.n_layers == 4

    def test_build_trial_config_does_not_mutate_original(self, optuna_config):
        """build_trial_config does not modify the original config."""
        from scripts.optuna_optimize import build_trial_config

        original_lr = optuna_config.training.optimizer.lr
        params = {"lr": 9e-3, "d_embed": 64, "dropout": 0.3,
                  "n_hgt_layers": 2, "beta": 0.9}
        build_trial_config(optuna_config, params)
        assert optuna_config.training.optimizer.lr == original_lr

    def test_build_trial_config_d_embed_updates_both(self, optuna_config):
        """d_embed compound mapping updates both model.d_embed and model.d_fused."""
        from scripts.optuna_optimize import build_trial_config

        params = {"d_embed": 256}
        trial_config = build_trial_config(optuna_config, params)
        assert trial_config.model.d_embed == 256
        assert trial_config.model.d_fused == 256

    def test_build_trial_config_n_heads_updates_three(self, optuna_config):
        """n_heads compound mapping updates hgt, pathology_attention, and set_transformer."""
        from scripts.optuna_optimize import build_trial_config

        params = {"n_heads": 8}
        trial_config = build_trial_config(optuna_config, params)
        assert trial_config.model.hgt.n_heads == 8
        assert trial_config.model.pathology_attention.n_heads == 8
        assert trial_config.model.set_transformer.n_heads == 8

    def test_build_trial_config_dropout_updates_four(self, optuna_config):
        """dropout compound mapping updates all 4 dropout locations."""
        from scripts.optuna_optimize import build_trial_config

        params = {"dropout": 0.25}
        trial_config = build_trial_config(optuna_config, params)
        assert trial_config.model.dropout == 0.25
        assert trial_config.model.pseudobulk.dropout == 0.25
        assert trial_config.model.hgt.dropout == 0.25
        assert trial_config.model.set_transformer.dropout == 0.25


class TestCreateStudyPrunerTypes:
    """Tests for different pruner types in study creation."""

    def test_create_study_uses_median_pruner(self, optuna_config):
        """Study uses MedianPruner when config specifies median type."""
        from scripts.optuna_optimize import create_study
        import optuna

        optuna_config.optuna.pruner = {
            "type": "median",
            "n_startup_trials": 5,
            "n_warmup_steps": 25,
            "interval_steps": 5,
        }
        study = create_study(optuna_config)
        assert isinstance(study.pruner, optuna.pruners.MedianPruner)

    def test_create_study_median_pruner_warmup_steps(self, optuna_config):
        """MedianPruner respects n_warmup_steps config."""
        from scripts.optuna_optimize import create_study

        optuna_config.optuna.pruner = {
            "type": "median",
            "n_startup_trials": 5,
            "n_warmup_steps": 25,
            "interval_steps": 5,
        }
        study = create_study(optuna_config)
        assert study.pruner._n_warmup_steps == 25
