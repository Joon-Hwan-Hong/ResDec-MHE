"""Tests for src/analysis/resilience_distributional.py."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import wasserstein_distance

from src.analysis.resilience_distributional import (
    _rank_biserial_correlation,
    latent_class_on_residuals,
    stability_selection,
    wasserstein_per_celltype,
)


# ---------- wasserstein_per_celltype --------------------------------------


def test_wasserstein_emits_one_entry_per_cell_type():
    rng = np.random.default_rng(0)
    expr = rng.normal(size=(40, 3, 10))  # (n_subj, n_ct, n_gene)
    is_res = np.array([True] * 20 + [False] * 20)
    out = wasserstein_per_celltype(expr, is_res)
    assert len(out["per_cell_type"]) == 3
    assert out["n_resilient"] == 20
    assert out["n_vulnerable"] == 20


def test_wasserstein_picks_up_mean_shift():
    rng = np.random.default_rng(0)
    expr = rng.normal(size=(60, 1, 5))
    is_res = np.array([True] * 30 + [False] * 30)
    # Inject a 3-sigma shift in gene 0 of the only cell type for resilient.
    expr[is_res, 0, 0] += 3.0
    out = wasserstein_per_celltype(expr, is_res, gene_names=[f"g{i}" for i in range(5)])
    top_genes = [g for g, _ in out["per_cell_type"][0]["wasserstein_per_gene_top10"]]
    assert "g0" in top_genes[:1], f"Expected shifted gene g0 to be top; got {top_genes[:3]}"


# ---------- _rank_biserial_correlation -----------------------------------


def test_rank_biserial_perfect_separation():
    x = np.arange(10).astype(float)
    y = np.arange(20, 30).astype(float)  # all > x
    rb = _rank_biserial_correlation(x, y)
    # x is fully below y: rank-biserial should be -1.
    assert rb == pytest.approx(-1.0)


def test_rank_biserial_no_separation():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    y = rng.normal(size=200)
    rb = _rank_biserial_correlation(x, y)
    assert abs(rb) < 0.1, f"Expected |rb|<0.1 for null; got {rb}"


# ---------- stability_selection -------------------------------------------


def test_stability_selection_recovers_planted_signal():
    rng = np.random.default_rng(0)
    n = 80
    n_features = 20
    x = rng.normal(size=(n, n_features))
    is_res = np.array([True] * 40 + [False] * 40)
    # Plant a strong shift in features 5 and 10.
    x[is_res, 5] += 2.0
    x[is_res, 10] += 2.0
    out = stability_selection(
        x, is_res, n_bootstrap=50, subsample_frac=0.5,
        rb_threshold=0.3, pi_threshold=0.7, seed=0,
    )
    assert 5 in out["stable_indices"]
    assert 10 in out["stable_indices"]


def test_stability_selection_no_signal_returns_few_stable():
    rng = np.random.default_rng(0)
    n = 80
    n_features = 50
    x = rng.normal(size=(n, n_features))
    is_res = np.array([True] * 40 + [False] * 40)
    out = stability_selection(
        x, is_res, n_bootstrap=50, subsample_frac=0.5,
        rb_threshold=0.3, pi_threshold=0.8, seed=0,
    )
    # Under the null, very few features should pass pi=0.8.
    assert len(out["stable_indices"]) < 5


def test_stability_selection_config_records_params():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20, 5))
    is_res = np.array([True] * 10 + [False] * 10)
    out = stability_selection(x, is_res, n_bootstrap=10, seed=42)
    assert out["config"]["n_bootstrap"] == 10
    assert out["config"]["seed"] == 42
    assert out["config"]["n_resilient"] == 10


# ---------- latent_class_on_residuals -------------------------------------


def test_latent_class_unimodal_data_picks_k1():
    rng = np.random.default_rng(0)
    residuals = rng.normal(loc=0.0, scale=1.0, size=300)
    out = latent_class_on_residuals(residuals, k_max=4, seed=0)
    assert out["best_k"] == 1
    assert out["is_unimodal"] is True


def test_latent_class_bimodal_data_picks_k_at_least_2():
    rng = np.random.default_rng(0)
    a = rng.normal(loc=-3.0, scale=0.5, size=200)
    b = rng.normal(loc=+3.0, scale=0.5, size=200)
    residuals = np.concatenate([a, b])
    out = latent_class_on_residuals(residuals, k_max=4, seed=0)
    assert out["best_k"] >= 2, f"Expected k>=2 for clearly bimodal; got {out['best_k']}"


def test_latent_class_means_sorted_ascending():
    rng = np.random.default_rng(0)
    a = rng.normal(loc=-3.0, scale=0.5, size=100)
    b = rng.normal(loc=+3.0, scale=0.5, size=100)
    residuals = np.concatenate([a, b])
    out = latent_class_on_residuals(residuals, k_max=3, seed=0)
    means = out["best_model_means"]
    assert means == sorted(means), "best_model_means should be ascending"


def test_latent_class_handles_nan_residuals():
    rng = np.random.default_rng(0)
    residuals = rng.normal(size=200)
    residuals[::20] = np.nan  # 10 NaNs
    out = latent_class_on_residuals(residuals, k_max=3, seed=0)
    # Assignments include -1 for NaN inputs.
    assignments = out["best_model_assignments"]
    assert len(assignments) == len(residuals)
    nan_positions = [i for i, r in enumerate(residuals) if not np.isfinite(r)]
    for i in nan_positions:
        assert assignments[i] == -1
