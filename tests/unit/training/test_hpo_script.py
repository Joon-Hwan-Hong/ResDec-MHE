"""
Tests for scripts/training/hpo.py — Ray Tune HPO script helper functions.

Tests the composable pieces of the HP optimization script:
- YAML-to-Ray-Tune search space translation (_yaml_to_search_space)
- Config building from Ray-sampled HPs (build_config_from_ray)
- Annealing schedule shortening (shorten_annealing_for_hpo)
- Subject ID collection from splits dict (_collect_all_subject_ids)
- Warm-start trial loading from Ray Tune results (load_warm_start_data)
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
    """Create a fake Ray Tune ray_results directory with 4 trials, none of
    which have a d_embed key in their config (simulating HPO8 trials where
    d_embed was hardcoded outside the search space).

    3 trials are newer than the experiment_state timestamp and should be
    loaded by ``load_warm_start_data``; 1 trial is older and should be
    filtered out by the timestamp gate at ``scripts/training/hpo.py:100-103``.

    The 3 kept trials have val_nll values 0.40, 0.45, 0.50 — sortable for
    top-K selection in tests of ``inject_forced_seeds`` (Task 5). The
    filtered-out trial has val_nll 0.30 (would be the best if not filtered),
    so if the timestamp filter regresses, downstream top-K tests will fail.

    Trial dirs use short placeholder IDs (e.g., ``aaa_1``) instead of real
    8-char hex hashes for readability. The parser only checks the
    ``train_fn_`` prefix and the trailing ``YYYY-MM-DD_HH-MM-SS`` timestamp,
    both of which are preserved.

    Returns:
        pathlib.Path to the fake ``ray_results/`` directory. Pass
        ``str(returned_path)`` to ``load_warm_start_data``.
    """
    import json  # Local import: fixture is the only consumer; keeps top-of-file imports minimal.

    # Latest experiment marker — load_warm_start_data filters trials by
    # comparing the timestamp embedded in the trial dir name to this.
    LATEST_TS = "2026-04-07_12-00-00"

    # Trial specs: (id_suffix, lr, timestamp_suffix, final_val_nll).
    # Downstream tests depend on:
    #   - The 3 kept trials being sortable by val_nll for top-K selection (Task 5)
    #   - val_nll values being exactly 0.40, 0.45, 0.50 (Task 3 asserts sorted match)
    #   - The 4th trial being filtered out by latest_ts (implicit filter coverage)
    TRIALS = [
        ("aaa_1", 0.001, "2026-04-07_12-00-01", 0.40),
        ("bbb_2", 0.005, "2026-04-07_12-00-02", 0.45),
        ("ccc_3", 0.0001, "2026-04-07_12-00-03", 0.50),
        # Stale trial from a prior experiment — should be filtered out.
        ("ddd_4", 0.01, "2026-04-06_23-59-59", 0.30),
    ]

    ray_dir = tmp_path / "ray_results"
    ray_dir.mkdir()

    # The state file's content is never read; only its filename's timestamp
    # is parsed via .stem in load_warm_start_data.
    (ray_dir / f"experiment_state-{LATEST_TS}.json").write_text("{}")

    for id_suffix, lr, ts, final_nll in TRIALS:
        dir_name = f"train_fn_{id_suffix}_lr={lr}_{ts}"
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
        # Final JSONL line: final val_nll (overwrites the early one in
        # load_warm_start_data's parser loop at hpo.py:115-126)
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


class TestLoadWarmStartData:
    """Tests for load_warm_start_data — ensures HPO warm-start survives
    when the current search space contains keys that prior trials lacked
    (e.g., adding d_embed to a search where prior HPO8 trials had d_embed
    hardcoded outside the search space)."""

    def test_survives_missing_d_embed_key(self, fake_warm_start_dir):
        """Warm-start should load trials even if current search_space has keys
        (d_embed) that prior trials lack — the missing key is filled from the
        defaults dict instead of dropping the trial.

        This is the core fix for d_embed widening: HPO8 trials had d_embed
        hardcoded at 64 (not in the search space), so their result.json
        configs have no d_embed key. Without this fix, all 50 HPO8 warm-start
        trials would be silently dropped when the new search space adds
        d_embed.

        The fixture provides 4 trials; 1 is filtered out by the timestamp
        gate, leaving 3. All 3 should be returned with d_embed=64 filled in.
        """
        from scripts.training.hpo import load_warm_start_data

        # Current search space INCLUDES d_embed (simulating widened HPO).
        search_space_keys = [
            "lr", "dropout", "beta", "weight_decay",
            "guide_lr", "anneal_epochs", "d_embed",
        ]
        points, rewards = load_warm_start_data(
            str(fake_warm_start_dir),
            search_space_keys=search_space_keys,
            defaults={"d_embed": 64},
        )

        # 3 trials survive (the 4th is filtered by timestamp gate)
        assert len(points) == 3, f"Expected 3 points (4th is stale), got {len(points)}"
        assert len(rewards) == 3, f"Expected 3 rewards (must match {len(points)} points), got {len(rewards)}"

        # Every returned point should have d_embed filled with the default
        for p in points:
            assert "d_embed" in p, f"Missing d_embed in returned point: {p}"
            assert p["d_embed"] == 64, f"Expected d_embed=64 (from defaults), got {p['d_embed']}"

        # The 3 final val_nll values match the fixture's 3 KEPT trials
        assert sorted(rewards) == [0.40, 0.45, 0.50], (
            f"Expected sorted rewards [0.40, 0.45, 0.50] from kept trials; "
            f"got {sorted(rewards)} — if this includes 0.30, the timestamp "
            f"filter regressed."
        )

    def test_backward_compat_no_defaults(self, fake_warm_start_dir):
        """Without ``defaults``, the function preserves the original
        behavior: it drops any trial whose config is missing a search-space
        key. The 3 fixture trials have all 6 baseline keys (lr, dropout,
        beta, weight_decay, guide_lr, anneal_epochs) so they all survive.
        """
        from scripts.training.hpo import load_warm_start_data

        # Search space matches the keys actually present in HPO8 trials
        search_space_keys = [
            "lr", "dropout", "beta", "weight_decay",
            "guide_lr", "anneal_epochs",
        ]
        points, rewards = load_warm_start_data(
            str(fake_warm_start_dir),
            search_space_keys=search_space_keys,
        )
        # All 3 kept trials should load (no d_embed key required)
        assert len(points) == 3
        assert len(rewards) == 3
        for p in points:
            assert "d_embed" not in p, (
                "d_embed must NOT be filled when defaults is omitted"
            )

    def test_drops_trial_when_key_missing_and_no_default(self, fake_warm_start_dir):
        """When a search_space_key is missing from BOTH the trial config
        AND the ``defaults`` dict, the trial should be dropped (preserves
        the original 'critical key missing' semantics)."""
        from scripts.training.hpo import load_warm_start_data

        # Search space includes tau_min, which the fixture trials don't have
        # AND which we don't provide a default for. All trials should be dropped.
        search_space_keys = [
            "lr", "dropout", "beta", "weight_decay",
            "guide_lr", "anneal_epochs", "d_embed", "tau_min",
        ]
        points, rewards = load_warm_start_data(
            str(fake_warm_start_dir),
            search_space_keys=search_space_keys,
            defaults={"d_embed": 64},  # tau_min has no default → all drops
        )
        assert points == []
        assert rewards == []

    def test_inject_forced_seeds_for_new_d_embed_values(self, fake_warm_start_dir):
        """After warm-start loading, we must be able to append forced-exploration
        points for new d_embed values. Top-2 HPO8 trials (by lowest val_nll)
        should be paired with d_embed=128 and d_embed=256, producing 4 new
        points whose continuous HPs are cloned from the top-2 templates.

        Why this matters: when expanding the search space with a new categorical
        axis (here d_embed: {64, 128, 256}), TPE would otherwise need many
        random samples before discovering the new axis is worth exploring.
        Forcing top-K × new-values seeds guarantees the sampler sees coverage
        of the new axis from trial 1.
        """
        from scripts.training.hpo import load_warm_start_data, inject_forced_seeds

        search_space_keys = [
            "lr", "dropout", "beta", "weight_decay",
            "guide_lr", "anneal_epochs", "d_embed",
        ]
        points, rewards = load_warm_start_data(
            str(fake_warm_start_dir),
            search_space_keys=search_space_keys,
            defaults={"d_embed": 64},
        )
        # Inject 4 forced seeds: top-2 by lowest val_nll × {128, 256}
        new_points = inject_forced_seeds(
            points, rewards,
            top_k=2,
            forced_axis="d_embed",
            forced_values=[128, 256],
        )
        # 4 new points (top-2 × 2 values)
        assert len(new_points) == 4, f"Expected 4 forced seeds, got {len(new_points)}"
        # Each new point has d_embed in {128, 256}
        assert all(p["d_embed"] in (128, 256) for p in new_points), (
            f"Some forced points have unexpected d_embed: "
            f"{[p['d_embed'] for p in new_points]}"
        )
        # The two values are balanced (2 of each)
        n_128 = sum(1 for p in new_points if p["d_embed"] == 128)
        n_256 = sum(1 for p in new_points if p["d_embed"] == 256)
        assert n_128 == 2, f"Expected 2 points with d_embed=128, got {n_128}"
        assert n_256 == 2, f"Expected 2 points with d_embed=256, got {n_256}"
        # Pin to fixture ground truth: the two lowest-val_nll trials are
        # (lr=0.001, val_nll=0.40) and (lr=0.005, val_nll=0.45). Both must
        # be represented in the forced points (each appears twice — once
        # per forced d_embed value).
        expected_lrs = {0.001, 0.005}
        actual_lrs = {p["lr"] for p in new_points}
        assert actual_lrs == expected_lrs, (
            f"Expected forced points cloned from lr={expected_lrs} "
            f"(top-2 by val_nll), got lr={actual_lrs}"
        )
        # Each new point's continuous HPs should match one of the top-2 templates (lowest val_nll = best)
        top2_points = [
            p for p, r in sorted(zip(points, rewards), key=lambda pr: pr[1])
        ][:2]
        top2_continuous = [
            {k: v for k, v in p.items() if k != "d_embed"} for p in top2_points
        ]
        new_continuous = [
            {k: v for k, v in p.items() if k != "d_embed"} for p in new_points
        ]
        for nc in new_continuous:
            assert nc in top2_continuous, (
                f"Forced point continuous HPs {nc} don't match any of the top-2 "
                f"templates {top2_continuous}"
            )
