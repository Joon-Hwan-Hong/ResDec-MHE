"""
Tests for scripts/training/hpo.py — Ray Tune HPO script helper functions.

Tests the composable pieces of the HP optimization script:
- YAML-to-Ray-Tune search space translation (_yaml_to_search_space)
- Config building from Ray-sampled HPs (build_config_from_ray)
- TuneReportCheckpointCallback factory (per-epoch val_nll reporting)
- Annealing schedule shortening (shorten_annealing_for_hpo)
"""

import pytest
import torch
from omegaconf import OmegaConf
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def search_space_config():
    """Config with hpo.search_space for testing _yaml_to_search_space."""
    return OmegaConf.create({
        "hpo": {
            "search_space": {
                "lr": {"type": "loguniform", "low": 0.0001, "high": 0.01},
                "dropout": {"type": "uniform", "low": 0.0, "high": 0.5},
                "fusion_type": {"type": "categorical", "choices": ["cross_attention", "concat"]},
                "anneal_epochs": {"type": "int", "low": 10, "high": 30},
            }
        }
    })


@pytest.fixture
def base_config():
    """Minimal base config for testing build_config_from_ray."""
    return OmegaConf.create({
        "model": {
            "d_embed": 64, "d_fused": 64, "dropout": 0.1,
            "hgt": {"n_layers": 4, "n_heads": 4},
            "pathology_attention": {"n_heads": 4},
            "set_transformer": {"n_heads": 4, "n_inducing_points": 64},
            "gene_gate": {"initial_temperature": 1.0},
            "fusion": {"type": "cross_attention", "n_heads": 4},
        },
        "training": {
            "optimizer": {"lr": 0.001, "weight_decay": 0.001, "guide_lr": 0.005},
            "loss": {"beta": 0.5},
            "temperature_annealing": {"tau_min": 0.1, "tau_max": 2.0, "warmup_epochs": 5, "anneal_epochs": 25},
            "early_stopping": {"min_epochs": 20},
            "max_epochs": 100,
        },
        "data": {"dataloader": {"batch_size": 20}},
    })


@pytest.fixture
def fake_warm_start_dir(tmp_path):
    """Create a fake Ray Tune ray_results directory with 3 completed trials,
    none of which have a d_embed key in their config (simulating HPO8 trials
    where d_embed was hardcoded outside the search space).

    Trial val_nll values are 0.40, 0.45, 0.50 — sortable for top-K selection
    in tests of inject_forced_seeds.
    """
    import json

    ray_dir = tmp_path / "ray_results"
    ray_dir.mkdir()

    # Latest experiment marker — load_warm_start_data filters trials by
    # comparing the timestamp embedded in the trial dir name to this.
    latest_ts = "2026-04-07_12-00-00"
    (ray_dir / f"experiment_state-{latest_ts}.json").write_text("{}")

    trials = [
        ("train_fn_aaa_1_lr=0.001_2026-04-07_12-00-01", 0.001, 0.40),
        ("train_fn_bbb_2_lr=0.005_2026-04-07_12-00-02", 0.005, 0.45),
        ("train_fn_ccc_3_lr=0.0001_2026-04-07_12-00-03", 0.0001, 0.50),
    ]

    for dir_name, lr, final_nll in trials:
        trial_dir = ray_dir / dir_name
        trial_dir.mkdir()
        # First JSONL line: early val_nll + full config (no d_embed key)
        early_record = {
            "val_nll": 0.9,
            "config": {
                "lr": lr,
                "dropout": 0.2,
                "beta": 0.5,
                "weight_decay": 1e-5,
                "guide_lr": 0.005,
                "anneal_epochs": 20,
            },
        }
        # Final JSONL line: final val_nll (overwrites the early one in load_warm_start_data)
        final_record = {"val_nll": final_nll}
        result_path = trial_dir / "result.json"
        result_path.write_text(
            json.dumps(early_record) + "\n" + json.dumps(final_record) + "\n"
        )

    return ray_dir


# ---------------------------------------------------------------------------
# Tests: _yaml_to_search_space
# ---------------------------------------------------------------------------


class TestYamlToSearchSpace:
    """Tests for YAML-to-Ray Tune search space translation."""

    def test_loguniform_type(self, search_space_config):
        """loguniform translates to tune.loguniform (Float with LogUniform sampler)."""
        from scripts.training.hpo import _yaml_to_search_space
        from ray.tune.search.sample import Float

        space = _yaml_to_search_space(search_space_config)
        assert isinstance(space["lr"], Float)
        # Verify it's log-uniform by checking the sampler type
        assert "LogUniform" in type(space["lr"].sampler).__name__

    def test_uniform_type(self, search_space_config):
        """uniform translates to tune.uniform (Float with Uniform sampler)."""
        from scripts.training.hpo import _yaml_to_search_space
        from ray.tune.search.sample import Float

        space = _yaml_to_search_space(search_space_config)
        assert isinstance(space["dropout"], Float)
        assert "Uniform" in type(space["dropout"].sampler).__name__
        # Exclude LogUniform — the sampler name should be exactly _Uniform or Uniform
        assert "Log" not in type(space["dropout"].sampler).__name__

    def test_categorical_type(self, search_space_config):
        """categorical translates to tune.choice (Categorical)."""
        from scripts.training.hpo import _yaml_to_search_space
        from ray.tune.search.sample import Categorical

        space = _yaml_to_search_space(search_space_config)
        assert isinstance(space["fusion_type"], Categorical)

    def test_int_type(self, search_space_config):
        """int translates to tune.randint (Integer)."""
        from scripts.training.hpo import _yaml_to_search_space
        from ray.tune.search.sample import Integer

        space = _yaml_to_search_space(search_space_config)
        assert isinstance(space["anneal_epochs"], Integer)

    def test_int_upper_bound_exclusive(self, search_space_config):
        """int type uses exclusive upper bound (spec.high + 1)."""
        from scripts.training.hpo import _yaml_to_search_space

        space = _yaml_to_search_space(search_space_config)
        # tune.randint(10, 31) — upper is spec.high(30) + 1
        assert space["anneal_epochs"].upper == 31

    def test_unknown_type_raises(self):
        """Unknown search space type raises ValueError."""
        from scripts.training.hpo import _yaml_to_search_space

        config = OmegaConf.create({
            "hpo": {
                "search_space": {
                    "bad_param": {"type": "exponential", "low": 0.1, "high": 1.0},
                }
            }
        })
        with pytest.raises(ValueError, match="Unknown search space type 'exponential'"):
            _yaml_to_search_space(config)

    def test_loguniform_samples_in_range(self, search_space_config):
        """loguniform samples fall within configured range."""
        from scripts.training.hpo import _yaml_to_search_space

        space = _yaml_to_search_space(search_space_config)
        for _ in range(50):
            val = space["lr"].sample()
            assert 0.0001 <= val <= 0.01

    def test_uniform_samples_in_range(self, search_space_config):
        """uniform samples fall within configured range."""
        from scripts.training.hpo import _yaml_to_search_space

        space = _yaml_to_search_space(search_space_config)
        for _ in range(50):
            val = space["dropout"].sample()
            assert 0.0 <= val <= 0.5

    def test_categorical_samples_from_choices(self, search_space_config):
        """categorical samples are from the configured choices."""
        from scripts.training.hpo import _yaml_to_search_space

        space = _yaml_to_search_space(search_space_config)
        for _ in range(50):
            val = space["fusion_type"].sample()
            assert val in ["cross_attention", "concat"]

    def test_int_samples_in_range(self, search_space_config):
        """int samples fall within configured range (inclusive)."""
        from scripts.training.hpo import _yaml_to_search_space

        space = _yaml_to_search_space(search_space_config)
        for _ in range(50):
            val = space["anneal_epochs"].sample()
            assert 10 <= val <= 30
            assert isinstance(val, int)

    def test_all_keys_present(self, search_space_config):
        """All search space keys are translated."""
        from scripts.training.hpo import _yaml_to_search_space

        space = _yaml_to_search_space(search_space_config)
        assert set(space.keys()) == {"lr", "dropout", "fusion_type", "anneal_epochs"}

    def test_empty_search_space(self):
        """Empty search space returns empty dict."""
        from scripts.training.hpo import _yaml_to_search_space

        config = OmegaConf.create({"hpo": {"search_space": {}}})
        space = _yaml_to_search_space(config)
        assert space == {}


# ---------------------------------------------------------------------------
# Tests: build_config_from_ray
# ---------------------------------------------------------------------------


class TestBuildConfigFromRay:
    """Tests for applying Ray-sampled HPs to base config."""

    def test_simple_param_lr(self, base_config):
        """lr maps to training.optimizer.lr."""
        from scripts.training.hpo import build_config_from_ray

        config = build_config_from_ray({"lr": 0.005}, base_config)
        assert config.training.optimizer.lr == 0.005

    def test_simple_param_weight_decay(self, base_config):
        """weight_decay maps to training.optimizer.weight_decay."""
        from scripts.training.hpo import build_config_from_ray

        config = build_config_from_ray({"weight_decay": 0.01}, base_config)
        assert config.training.optimizer.weight_decay == 0.01

    def test_simple_param_n_hgt_layers(self, base_config):
        """n_hgt_layers maps to model.hgt.n_layers."""
        from scripts.training.hpo import build_config_from_ray

        config = build_config_from_ray({"n_hgt_layers": 6}, base_config)
        assert config.model.hgt.n_layers == 6

    def test_simple_param_beta(self, base_config):
        """beta maps to training.loss.beta."""
        from scripts.training.hpo import build_config_from_ray

        config = build_config_from_ray({"beta": 0.8}, base_config)
        assert config.training.loss.beta == 0.8

    def test_simple_param_batch_size(self, base_config):
        """batch_size maps to data.dataloader.batch_size."""
        from scripts.training.hpo import build_config_from_ray

        config = build_config_from_ray({"batch_size": 32}, base_config)
        assert config.data.dataloader.batch_size == 32

    def test_simple_param_n_inducing(self, base_config):
        """n_inducing maps to model.set_transformer.n_inducing_points."""
        from scripts.training.hpo import build_config_from_ray

        config = build_config_from_ray({"n_inducing": 128}, base_config)
        assert config.model.set_transformer.n_inducing_points == 128

    def test_simple_param_gene_gate_temp(self, base_config):
        """gene_gate_temp maps to model.gene_gate.initial_temperature."""
        from scripts.training.hpo import build_config_from_ray

        config = build_config_from_ray({"gene_gate_temp": 1.5}, base_config)
        assert config.model.gene_gate.initial_temperature == 1.5

    def test_simple_param_guide_lr(self, base_config):
        """guide_lr maps to training.optimizer.guide_lr."""
        from scripts.training.hpo import build_config_from_ray

        config = build_config_from_ray({"guide_lr": 0.02}, base_config)
        assert config.training.optimizer.guide_lr == 0.02

    def test_simple_param_fusion_type(self, base_config):
        """fusion_type maps to model.fusion.type."""
        from scripts.training.hpo import build_config_from_ray

        config = build_config_from_ray({"fusion_type": "concat"}, base_config)
        assert config.model.fusion.type == "concat"

    def test_simple_param_fusion_n_heads(self, base_config):
        """fusion_n_heads maps to model.fusion.n_heads."""
        from scripts.training.hpo import build_config_from_ray

        config = build_config_from_ray({"fusion_n_heads": 8}, base_config)
        assert config.model.fusion.n_heads == 8

    def test_d_embed_updates_both(self, base_config):
        """d_embed compound mapping updates both model.d_embed and model.d_fused."""
        from scripts.training.hpo import build_config_from_ray

        config = build_config_from_ray({"d_embed": 128}, base_config)
        assert config.model.d_embed == 128
        assert config.model.d_fused == 128

    def test_n_heads_updates_three_modules(self, base_config):
        """n_heads compound mapping updates hgt, pathology_attention, and set_transformer."""
        from scripts.training.hpo import build_config_from_ray

        config = build_config_from_ray({"n_heads": 8, "d_embed": 128}, base_config)
        assert config.model.hgt.n_heads == 8
        assert config.model.pathology_attention.n_heads == 8
        assert config.model.set_transformer.n_heads == 8

    def test_dropout_updates_model_dropout(self, base_config):
        """dropout maps to model.dropout."""
        from scripts.training.hpo import build_config_from_ray

        config = build_config_from_ray({"dropout": 0.25}, base_config)
        assert config.model.dropout == 0.25

    def test_tau_min_updates_annealing(self, base_config):
        """tau_min maps to training.temperature_annealing.tau_min."""
        from scripts.training.hpo import build_config_from_ray

        config = build_config_from_ray({"tau_min": 0.05}, base_config)
        assert config.training.temperature_annealing.tau_min == 0.05

    def test_anneal_epochs_cast_to_int(self, base_config):
        """anneal_epochs is cast to int."""
        from scripts.training.hpo import build_config_from_ray

        config = build_config_from_ray({"anneal_epochs": 15.0}, base_config)
        assert config.training.temperature_annealing.anneal_epochs == 15
        assert isinstance(config.training.temperature_annealing.anneal_epochs, int)

    def test_unknown_key_warns(self, base_config):
        """Unknown keys produce a warning but do not crash."""
        from scripts.training.hpo import build_config_from_ray

        with patch("scripts.training.hpo.logger") as mock_logger:
            config = build_config_from_ray({"unknown_param": 42}, base_config)
            mock_logger.warning.assert_called_once()
            assert "unknown_param" in mock_logger.warning.call_args[0][1]
        # Config should still be valid
        assert config is not None

    def test_returns_none_when_d_embed_not_divisible_by_n_heads(self, base_config):
        """Returns None when d_embed % n_heads != 0."""
        from scripts.training.hpo import build_config_from_ray

        # d_embed=65 is not divisible by n_heads=4 (base config default)
        config = build_config_from_ray({"d_embed": 65}, base_config)
        assert config is None

    def test_returns_none_when_d_embed_not_divisible_by_fusion_n_heads(self, base_config):
        """Returns None when d_embed % fusion_n_heads != 0."""
        from scripts.training.hpo import build_config_from_ray

        # d_embed=64, fusion_n_heads=6 -> 64 % 6 != 0
        config = build_config_from_ray({"fusion_n_heads": 6}, base_config)
        assert config is None

    def test_does_not_mutate_base_config(self, base_config):
        """build_config_from_ray does not modify the base config."""
        from scripts.training.hpo import build_config_from_ray

        original_lr = base_config.training.optimizer.lr
        build_config_from_ray({"lr": 0.999}, base_config)
        assert base_config.training.optimizer.lr == original_lr

    def test_multiple_params_applied(self, base_config):
        """Multiple parameters are applied simultaneously."""
        from scripts.training.hpo import build_config_from_ray

        config = build_config_from_ray({
            "lr": 0.005,
            "dropout": 0.3,
            "beta": 0.7,
            "batch_size": 32,
        }, base_config)
        assert config.training.optimizer.lr == 0.005
        assert config.model.dropout == 0.3
        assert config.training.loss.beta == 0.7
        assert config.data.dataloader.batch_size == 32

    def test_valid_d_embed_n_heads_combination(self, base_config):
        """Valid d_embed/n_heads combination returns config (not None)."""
        from scripts.training.hpo import build_config_from_ray

        config = build_config_from_ray({"d_embed": 128, "n_heads": 8}, base_config)
        assert config is not None
        assert config.model.d_embed == 128
        assert config.model.hgt.n_heads == 8



# ---------------------------------------------------------------------------
# Tests: shorten_annealing_for_hpo
# ---------------------------------------------------------------------------


class TestShortenAnnealingForHPO:
    """Tests for proportional annealing schedule shortening."""

    def test_ratio_gte_1_returns_unchanged(self, base_config):
        """ratio >= 1.0 returns config with original values."""
        from scripts.training.hpo import shorten_annealing_for_hpo

        # max_epochs == full_max_epochs -> ratio = 1.0
        config = shorten_annealing_for_hpo(base_config, full_max_epochs=100)
        assert config.training.temperature_annealing.warmup_epochs == 5
        assert config.training.temperature_annealing.anneal_epochs == 25

    def test_ratio_gt_1_returns_unchanged(self, base_config):
        """ratio > 1.0 (more epochs than full) returns config unchanged."""
        from scripts.training.hpo import shorten_annealing_for_hpo

        # max_epochs=100, full_max_epochs=50 -> ratio=2.0
        config = shorten_annealing_for_hpo(base_config, full_max_epochs=50)
        assert config.training.temperature_annealing.warmup_epochs == 5
        assert config.training.temperature_annealing.anneal_epochs == 25

    def test_shortens_warmup_proportionally(self, base_config):
        """ratio < 1.0 shortens warmup_epochs proportionally."""
        from scripts.training.hpo import shorten_annealing_for_hpo

        base_config.training.max_epochs = 30
        config = shorten_annealing_for_hpo(base_config, full_max_epochs=100)
        # warmup_epochs = max(1, round(5 * 0.3)) = max(1, 2) = 2
        assert config.training.temperature_annealing.warmup_epochs == 2

    def test_shortens_anneal_epochs_proportionally(self, base_config):
        """ratio < 1.0 shortens anneal_epochs proportionally."""
        from scripts.training.hpo import shorten_annealing_for_hpo

        base_config.training.max_epochs = 30
        config = shorten_annealing_for_hpo(base_config, full_max_epochs=100)
        # anneal_epochs = max(1, round(25 * 0.3)) = max(1, 8) = 8
        assert config.training.temperature_annealing.anneal_epochs == 8

    def test_kl_annealing_shortened_when_enabled(self, base_config):
        """KL annealing warmup is shortened when enabled."""
        from scripts.training.hpo import shorten_annealing_for_hpo

        base_config.training.kl_annealing = {
            "enabled": True, "alpha_min": 0.01,
            "warmup_epochs": 10, "schedule": "linear",
        }
        base_config.training.max_epochs = 50
        config = shorten_annealing_for_hpo(base_config, full_max_epochs=100)
        # kl warmup_epochs = max(1, round(10 * 0.5)) = 5
        assert config.training.kl_annealing.warmup_epochs == 5

    def test_kl_annealing_not_shortened_when_disabled(self, base_config):
        """KL annealing warmup is not touched when disabled."""
        from scripts.training.hpo import shorten_annealing_for_hpo

        base_config.training.kl_annealing = {
            "enabled": False, "warmup_epochs": 10,
        }
        base_config.training.max_epochs = 50
        config = shorten_annealing_for_hpo(base_config, full_max_epochs=100)
        assert config.training.kl_annealing.warmup_epochs == 10

    def test_min_epochs_updated(self, base_config):
        """min_epochs is set to warmup + anneal (shortened)."""
        from scripts.training.hpo import shorten_annealing_for_hpo

        base_config.training.max_epochs = 30
        config = shorten_annealing_for_hpo(base_config, full_max_epochs=100)
        expected_min = (
            config.training.temperature_annealing.warmup_epochs
            + config.training.temperature_annealing.anneal_epochs
        )
        assert config.training.early_stopping.min_epochs == expected_min

    def test_minimum_one_epoch(self, base_config):
        """Shortened values are always at least 1 epoch."""
        from scripts.training.hpo import shorten_annealing_for_hpo

        base_config.training.max_epochs = 1
        config = shorten_annealing_for_hpo(base_config, full_max_epochs=100)
        assert config.training.temperature_annealing.warmup_epochs >= 1
        assert config.training.temperature_annealing.anneal_epochs >= 1


# ---------------------------------------------------------------------------
# Tests: _collect_all_subject_ids
# ---------------------------------------------------------------------------


class TestCollectAllSubjectIds:
    """Tests for subject ID collection from splits dict."""

    def test_collects_from_holdout_and_folds(self):
        """Collects IDs from holdout_test, train_val_pool, and folds."""
        from scripts.training.hpo import _collect_all_subject_ids

        splits = {
            "holdout_test": ["s1", "s2"],
            "train_val_pool": ["s3", "s4"],
            "folds": [
                {"train": ["s3", "s5"], "val": ["s4"]},
            ],
        }
        ids = _collect_all_subject_ids(splits)
        assert set(ids) == {"s1", "s2", "s3", "s4", "s5"}

    def test_deduplicates(self):
        """Duplicate IDs across sections are deduplicated."""
        from scripts.training.hpo import _collect_all_subject_ids

        splits = {
            "holdout_test": ["s1", "s2"],
            "folds": [
                {"train": ["s1", "s2"], "val": ["s1"]},
            ],
        }
        ids = _collect_all_subject_ids(splits)
        assert ids == ["s1", "s2"]

    def test_returns_sorted(self):
        """Returns sorted list."""
        from scripts.training.hpo import _collect_all_subject_ids

        splits = {
            "holdout_test": ["s3", "s1"],
            "folds": [{"train": ["s2"], "val": ["s4"]}],
        }
        ids = _collect_all_subject_ids(splits)
        assert ids == sorted(ids)

    def test_empty_splits(self):
        """Handles empty splits dict."""
        from scripts.training.hpo import _collect_all_subject_ids

        ids = _collect_all_subject_ids({})
        assert ids == []
