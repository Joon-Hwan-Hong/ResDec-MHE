# tests/unit/scripts/test_extract_hpo_results.py
"""Tests for HPO result extraction script."""

import json
import csv
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def mock_ray_results(tmp_path):
    """Create a mock Ray Tune results directory with 5 trial directories."""
    ray_dir = tmp_path / "ray_results" / "experiment_name"
    ray_dir.mkdir(parents=True)

    # Write experiment state file (matches the current run's date)
    exp_state = ray_dir / "experiment_state-2026-03-25_12-36-13.json"
    exp_state.write_text(json.dumps({"runner_data": {}}))

    trials = [
        {"trial_id": "aaa11111", "val_nll": 0.42, "epochs": 10, "status": "PAUSED",
         "config": {"lr": 0.001, "dropout": 0.15, "weight_decay": 1e-5,
                    "beta": 0.5, "guide_lr": 0.005, "tau_min": 1.8,
                    "anneal_epochs": 20, "gene_gate_temp": 1.0}},
        {"trial_id": "bbb22222", "val_nll": 0.55, "epochs": 10, "status": "TERMINATED",
         "config": {"lr": 0.003, "dropout": 0.3, "weight_decay": 1e-4,
                    "beta": 0.7, "guide_lr": 0.01, "tau_min": 1.5,
                    "anneal_epochs": 15, "gene_gate_temp": 0.5}},
        {"trial_id": "ccc33333", "val_nll": 0.38, "epochs": 25, "status": "TERMINATED",
         "config": {"lr": 0.0005, "dropout": 0.2, "weight_decay": 5e-6,
                    "beta": 0.4, "guide_lr": 0.008, "tau_min": 1.7,
                    "anneal_epochs": 25, "gene_gate_temp": 1.5}},
        {"trial_id": "ddd44444", "val_nll": 0.61, "epochs": 10, "status": "TERMINATED",
         "config": {"lr": 0.005, "dropout": 0.35, "weight_decay": 3e-4,
                    "beta": 0.1, "guide_lr": 0.02, "tau_min": 1.6,
                    "anneal_epochs": 12, "gene_gate_temp": 0.8}},
        {"trial_id": "eee55555", "val_nll": 0.50, "epochs": 15, "status": "PAUSED",
         "config": {"lr": 0.002, "dropout": 0.1, "weight_decay": 2e-5,
                    "beta": 0.6, "guide_lr": 0.007, "tau_min": 1.9,
                    "anneal_epochs": 18, "gene_gate_temp": 1.2}},
    ]

    for t in trials:
        # Directory name format matches Ray Tune convention
        trial_dir_name = (
            f"train_fn_{t['trial_id']}_1_anneal_epochs={t['config']['anneal_epochs']},"
            f"beta={t['config']['beta']}_2026-03-25_12-36-13"
        )
        trial_dir = ray_dir / trial_dir_name
        trial_dir.mkdir()

        # Write per-epoch result.json (one JSON line per epoch)
        with open(trial_dir / "result.json", "w") as f:
            for epoch in range(1, t["epochs"] + 1):
                # val_nll decreases over epochs (simple linear decrease)
                epoch_nll = t["val_nll"] + (1.0 - t["val_nll"]) * (1 - epoch / t["epochs"])
                result = {
                    "val_nll": epoch_nll,
                    "training_iteration": epoch,
                    "trial_id": t["trial_id"],
                    "timestamp": 1774460000 + epoch * 100,
                    "time_this_iter_s": 30.0,
                    "time_total_s": epoch * 30.0,
                    "config": t["config"],
                    "done": epoch == t["epochs"] and t["status"] == "TERMINATED",
                }
                f.write(json.dumps(result) + "\n")

        # Write progress.csv
        with open(trial_dir / "progress.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["training_iteration", "val_nll", "time_total_s"])
            for epoch in range(1, t["epochs"] + 1):
                epoch_nll = t["val_nll"] + (1.0 - t["val_nll"]) * (1 - epoch / t["epochs"])
                writer.writerow([epoch, epoch_nll, epoch * 30.0])

        # Write params.json
        (trial_dir / "params.json").write_text(json.dumps(t["config"]))

    return ray_dir


@pytest.fixture
def mock_base_config(tmp_path):
    """Create a minimal base config YAML."""
    config = {
        "experiment": {"name": "test", "seed": 42},
        "data": {
            "dataloader": {"batch_size": 24},
            "splits": {"n_folds": 5},
        },
        "model": {
            "n_genes": 4796,
            "n_cell_types": 31,
            "d_embed": 64,
            "d_fused": 64,
            "dropout": 0.15,
            "gene_gate": {"initial_temperature": 1.0},
            "hgt": {"n_layers": 4, "n_heads": 4},
            "set_transformer": {"n_inducing_points": 64, "n_heads": 4},
            "fusion": {"type": "concat_normalized", "n_heads": 4},
            "pathology_attention": {"n_heads": 4, "d_cond": 64, "n_pathology_features": 3},
            "use_hgt_encoder": True,
            "use_cell_transformer": True,
            "head": {"type": "bayesian", "d_hidden": 64, "target_mean": None},
        },
        "training": {
            "max_epochs": 100,
            "optimizer": {"lr": 0.001, "weight_decay": 0.001, "guide_lr": 0.005},
            "loss": {"type": "beta_nll", "beta": 0.5},
            "temperature_annealing": {
                "tau_max": 2.0, "tau_min": 0.1,
                "warmup_epochs": 5, "anneal_epochs": 25, "schedule": "exponential",
            },
        },
        "paths": {"output_dir": "outputs/"},
    }
    config_path = tmp_path / "base_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


class TestParseRayResults:
    """Test parsing Ray Tune result directories."""

    def test_parse_all_trials(self, mock_ray_results):
        from scripts.extract_hpo_results import parse_ray_results
        trials = parse_ray_results(mock_ray_results)
        assert len(trials) == 5

    def test_trials_have_required_fields(self, mock_ray_results):
        from scripts.extract_hpo_results import parse_ray_results
        trials = parse_ray_results(mock_ray_results)
        required = {"trial_id", "best_val_nll", "best_epoch", "total_epochs",
                     "config", "total_time_s"}
        for t in trials:
            assert required.issubset(set(t.keys())), f"Missing fields: {required - set(t.keys())}"

    def test_best_val_nll_is_minimum(self, mock_ray_results):
        from scripts.extract_hpo_results import parse_ray_results
        trials = parse_ray_results(mock_ray_results)
        # Trial ccc33333 has best val_nll=0.38 at epoch 25
        best = min(trials, key=lambda t: t["best_val_nll"])
        assert best["trial_id"] == "ccc33333"
        assert abs(best["best_val_nll"] - 0.38) < 0.01

    def test_per_epoch_history_collected(self, mock_ray_results):
        from scripts.extract_hpo_results import parse_ray_results
        trials = parse_ray_results(mock_ray_results)
        # ccc33333 ran 25 epochs
        trial_c = [t for t in trials if t["trial_id"] == "ccc33333"][0]
        assert len(trial_c["epoch_history"]) == 25

    def test_handles_empty_directory(self, tmp_path):
        from scripts.extract_hpo_results import parse_ray_results
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        trials = parse_ray_results(empty_dir)
        assert trials == []


class TestFilterCurrentExperiment:
    """Test filtering trials to the most recent experiment."""

    def test_filters_to_latest_experiment(self, mock_ray_results):
        from scripts.extract_hpo_results import parse_ray_results, _filter_current_experiment
        trials = parse_ray_results(mock_ray_results)
        filtered = _filter_current_experiment(trials, mock_ray_results)
        # All 5 mock trials have matching timestamp, so all should be kept
        assert len(filtered) == 5

    def test_excludes_old_trials(self, mock_ray_results):
        # Add an old trial directory with a pre-experiment timestamp
        import json
        old_dir = mock_ray_results / "train_fn_old00000_0_params_2020-01-01_00-00-00"
        old_dir.mkdir()
        result = {"val_nll": 0.3, "training_iteration": 1, "trial_id": "old00000",
                  "timestamp": 1000000, "time_this_iter_s": 10, "time_total_s": 10,
                  "config": {"lr": 0.001}, "done": True}
        (old_dir / "result.json").write_text(json.dumps(result))

        from scripts.extract_hpo_results import parse_ray_results, _filter_current_experiment
        trials = parse_ray_results(mock_ray_results)
        assert len(trials) == 6  # 5 original + 1 old
        filtered = _filter_current_experiment(trials, mock_ray_results)
        assert len(filtered) == 5  # old one filtered out
        assert all(t["trial_id"] != "old00000" for t in filtered)


class TestRankTrials:
    """Test trial ranking."""

    def test_rank_by_val_nll_ascending(self, mock_ray_results):
        from scripts.extract_hpo_results import parse_ray_results, rank_trials
        trials = parse_ray_results(mock_ray_results)
        ranked = rank_trials(trials, metric="val_nll", mode="min")
        nlls = [t["best_val_nll"] for t in ranked]
        assert nlls == sorted(nlls)

    def test_top_k_selection(self, mock_ray_results):
        from scripts.extract_hpo_results import parse_ray_results, rank_trials
        trials = parse_ray_results(mock_ray_results)
        ranked = rank_trials(trials, metric="val_nll", mode="min", top_k=3)
        assert len(ranked) == 3


class TestBuildSummaryTable:
    """Test summary table generation."""

    def test_summary_has_all_trials_and_epochs(self, mock_ray_results):
        from scripts.extract_hpo_results import parse_ray_results, build_summary_table
        trials = parse_ray_results(mock_ray_results)
        df = build_summary_table(trials)
        # Total epochs: 10 + 10 + 25 + 10 + 15 = 70
        assert len(df) == 70
        assert "trial_id" in df.columns
        assert "epoch" in df.columns
        assert "val_nll" in df.columns

    def test_summary_includes_hp_columns(self, mock_ray_results):
        from scripts.extract_hpo_results import parse_ray_results, build_summary_table
        trials = parse_ray_results(mock_ray_results)
        df = build_summary_table(trials)
        hp_cols = {"lr", "dropout", "weight_decay", "beta", "guide_lr",
                   "tau_min", "anneal_epochs", "gene_gate_temp"}
        assert hp_cols.issubset(set(df.columns))


class TestExportConfigs:
    """Test config YAML export."""

    def test_export_creates_yaml_files(self, mock_ray_results, mock_base_config, tmp_path):
        from scripts.extract_hpo_results import (
            parse_ray_results, rank_trials, export_top_configs,
        )
        trials = parse_ray_results(mock_ray_results)
        ranked = rank_trials(trials, metric="val_nll", mode="min", top_k=3)
        output_dir = tmp_path / "top_configs"
        export_top_configs(ranked, str(mock_base_config), output_dir)
        yamls = list(output_dir.glob("*.yaml"))
        assert len(yamls) == 3

    def test_exported_config_has_correct_hp_values(self, mock_ray_results, mock_base_config, tmp_path):
        from scripts.extract_hpo_results import (
            parse_ray_results, rank_trials, export_top_configs,
        )
        trials = parse_ray_results(mock_ray_results)
        ranked = rank_trials(trials, metric="val_nll", mode="min", top_k=1)
        output_dir = tmp_path / "top_configs"
        export_top_configs(ranked, str(mock_base_config), output_dir)
        # Best trial is ccc33333 with lr=0.0005
        exported = yaml.safe_load((output_dir / "rank01_trial_ccc33333.yaml").read_text())
        assert abs(exported["training"]["optimizer"]["lr"] - 0.0005) < 1e-8

    def test_exported_config_preserves_non_hp_fields(self, mock_ray_results, mock_base_config, tmp_path):
        from scripts.extract_hpo_results import (
            parse_ray_results, rank_trials, export_top_configs,
        )
        trials = parse_ray_results(mock_ray_results)
        ranked = rank_trials(trials, metric="val_nll", mode="min", top_k=1)
        output_dir = tmp_path / "top_configs"
        export_top_configs(ranked, str(mock_base_config), output_dir)
        exported = yaml.safe_load((output_dir / "rank01_trial_ccc33333.yaml").read_text())
        # Non-HP fields preserved from base config
        assert exported["model"]["n_genes"] == 4796
        assert exported["model"]["hgt"]["n_layers"] == 4


class TestHPImportance:
    """Test HP importance analysis."""

    def test_importance_returns_all_hps(self, mock_ray_results):
        from scripts.extract_hpo_results import parse_ray_results, compute_hp_importance
        trials = parse_ray_results(mock_ray_results)
        importance = compute_hp_importance(trials, metric="val_nll")
        hp_names = {"lr", "dropout", "weight_decay", "beta", "guide_lr",
                    "tau_min", "anneal_epochs", "gene_gate_temp"}
        assert hp_names.issubset(set(importance.keys()))

    def test_importance_values_are_bounded(self, mock_ray_results):
        from scripts.extract_hpo_results import parse_ray_results, compute_hp_importance
        trials = parse_ray_results(mock_ray_results)
        importance = compute_hp_importance(trials, metric="val_nll")
        for name, score in importance.items():
            assert 0.0 <= score <= 1.0, f"HP {name} importance {score} out of bounds"
