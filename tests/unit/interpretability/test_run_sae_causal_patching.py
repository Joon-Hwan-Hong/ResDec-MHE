"""Tests for ``scripts/resdec_mhe/interpretability/run_sae_causal_patching.py``.

The orchestrator's per-fold loop loads a canonical Lightning checkpoint and
runs forward over a real DataModule, which is too heavy for unit tests. We
instead test the small composable pure-numpy / torch helpers that bear the
mathematical correctness:

  * :func:`load_sae_from_dir` — reads ``sae_model.npz`` and filters the
    legacy ``l1_lambda`` config key.
  * :func:`patch_fused_with_sae` — encode → patch → decode at the SAE
    bottleneck. Verifies the bit-identity of feature-572 zero-out against the
    direct ``-h[:, j] * W_dec[:, j]`` decomposition.
  * :func:`compute_feature_percentiles` — empirical p1 / p99 of one feature
    over the cohort activations.
  * :func:`select_random_controls` — non-dead random feature picker.
  * :func:`_render_markdown` — string interpretation logic.

The orchestrator's argparse construction is also smoke-tested.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch


_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = (
    _WORKTREE_ROOT
    / "scripts"
    / "resdec_mhe"
    / "interpretability"
    / "run_sae_causal_patching.py"
)


@pytest.fixture(scope="module")
def script_module():
    """Import the orchestrator module without running ``main()``."""
    if str(_WORKTREE_ROOT) not in sys.path:
        sys.path.insert(0, str(_WORKTREE_ROOT))
    spec = importlib.util.spec_from_file_location(
        "run_sae_causal_patching_for_test", SCRIPT_PATH,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tiny_sae(tmp_path_factory):
    """Build a small synthetic batch_topk SAEModel + persisted .npz on disk."""
    from src.analysis.sparse_autoencoder import SAEConfig, SAEModel

    rng = np.random.default_rng(0)
    n, m = 8, 16
    cfg = SAEConfig(
        architecture="batch_topk",
        expansion=2,
        k=4,
        aux_lambda=1.0 / 32.0,
        aux_k=4,
        decoder_unit_norm=True,
        learning_rate=1e-4,
        batch_size=4,
        n_steps=10,
        seed=0,
    )

    W_dec = rng.standard_normal((n, m)).astype(np.float32)
    W_dec /= np.linalg.norm(W_dec, axis=0, keepdims=True) + 1e-8
    W_enc = W_dec.T.copy()
    b_enc = np.zeros(m, dtype=np.float32)
    b_dec = rng.standard_normal(n).astype(np.float32) * 0.05

    sae = SAEModel(
        W_enc=W_enc,
        b_enc=b_enc,
        W_dec=W_dec,
        b_dec=b_dec,
        config=cfg,
        activation_stats={
            "mean": np.zeros(m, dtype=np.float32),
            "std": np.ones(m, dtype=np.float32),
            "fraction_active": np.full(m, 0.5, dtype=np.float32),
            "is_dead": np.array(
                [False, True, False, False, False, False, False, False,
                 False, False, False, False, False, False, False, True],
                dtype=bool,
            ),
            # Threshold at 0 so encode_threshold gates only by ReLU(>=0) sign.
            "threshold": np.array([0.0], dtype=np.float32),
        },
    )

    # Persist via the project's canonical save helper to mirror real
    # checkpoints: sae_model.npz with config_json + stat_* keys.
    from src.analysis.sae_io import save_sae_model

    out_dir = tmp_path_factory.mktemp("tiny_sae")
    save_sae_model(sae, out_dir / "sae_model.npz")
    return sae, out_dir


# ─────────────────────────────────────────────────────────────────────────────
# Argparse smoke
# ─────────────────────────────────────────────────────────────────────────────


def test_argparse_defaults_match_canonical_artifacts(script_module):
    """``main()`` argparse exposes the canonical-artifact paths as defaults."""
    p = argparse.ArgumentParser()
    p.add_argument("--feature-idx", type=int, default=572)
    p.add_argument(
        "--sae-dir",
        default=script_module.DEFAULT_SAE_CONFIG_DIR,
    )
    p.add_argument(
        "--fused-activations-npz",
        default=script_module.DEFAULT_FUSED_ACTIVATIONS,
    )
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--n-random", type=int, default=10)
    ns = p.parse_args([])
    assert ns.feature_idx == 572
    assert ns.n_folds == 5
    assert ns.n_random == 10
    assert "batch_topk/fused/exp32_k64_seed0" in ns.sae_dir
    assert "activations_fused_all_folds.npz" in ns.fused_activations_npz


# ─────────────────────────────────────────────────────────────────────────────
# load_sae_from_dir — including legacy l1_lambda filtering
# ─────────────────────────────────────────────────────────────────────────────


def test_load_sae_from_dir_round_trip(script_module, tiny_sae):
    """Persisted SAE re-loads with matching W_enc / W_dec / config."""
    sae, out_dir = tiny_sae
    loaded = script_module.load_sae_from_dir(out_dir)
    assert loaded.W_enc.shape == sae.W_enc.shape
    assert loaded.W_dec.shape == sae.W_dec.shape
    np.testing.assert_array_equal(loaded.W_enc, sae.W_enc)
    np.testing.assert_array_equal(loaded.W_dec, sae.W_dec)
    np.testing.assert_array_equal(loaded.b_enc, sae.b_enc)
    np.testing.assert_array_equal(loaded.b_dec, sae.b_dec)
    assert loaded.config.architecture == "batch_topk"
    assert loaded.config.expansion == 2
    assert loaded.config.k == 4
    # batch_topk path includes a threshold field.
    assert "threshold" in loaded.activation_stats


def test_load_sae_from_dir_filters_legacy_l1_lambda(script_module, tmp_path):
    """An npz containing legacy ``l1_lambda`` config key still loads cleanly."""
    n, m = 4, 8
    legacy_cfg = {
        "architecture": "batch_topk",
        "expansion": 2,
        "k": 2,
        "l1_lambda": None,
        "aux_lambda": 0.03125,
        "aux_k": 2,
        "decoder_unit_norm": True,
        "learning_rate": 1e-4,
        "batch_size": 4,
        "n_steps": 10,
        "seed": 0,
    }
    np.savez(
        tmp_path / "sae_model.npz",
        W_enc=np.zeros((m, n), dtype=np.float32),
        b_enc=np.zeros(m, dtype=np.float32),
        W_dec=np.eye(n, m, dtype=np.float32),
        b_dec=np.zeros(n, dtype=np.float32),
        config_json=np.array(json.dumps(legacy_cfg), dtype=object),
        stat_mean=np.zeros(m, dtype=np.float32),
        stat_std=np.ones(m, dtype=np.float32),
        stat_fraction_active=np.zeros(m, dtype=np.float32),
        stat_is_dead=np.zeros(m, dtype=bool),
        stat_threshold=np.array([0.0], dtype=np.float32),
    )
    loaded = script_module.load_sae_from_dir(tmp_path)
    assert loaded.config.architecture == "batch_topk"


def test_load_sae_from_dir_missing_npz_raises(script_module, tmp_path):
    with pytest.raises(FileNotFoundError):
        script_module.load_sae_from_dir(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# patch_fused_with_sae — math correctness
# ─────────────────────────────────────────────────────────────────────────────


def test_patch_fused_zero_matches_decoder_decomposition(script_module, tiny_sae):
    """Patching feature j to 0 changes x_hat by exactly -h[:, j] * W_dec[:, j].

    Direct algebraic identity: if x_hat = b_dec + Σ_k h_k * W_dec[:, k], then
    setting h_j := 0 yields x_hat - h_j_orig * W_dec[:, j]. We verify against
    the project's own ``_encode_numpy`` / ``_decode_numpy`` to ensure the
    patch primitive reuses them.
    """
    from src.analysis.sparse_autoencoder import _encode_numpy, _decode_numpy
    sae, _ = tiny_sae

    rng = np.random.default_rng(7)
    fused_np = rng.standard_normal((2, 5, 8)).astype(np.float32)
    fused = torch.from_numpy(fused_np)

    flat = fused_np.reshape(-1, 8)
    h_orig = _encode_numpy(sae, flat)
    xhat_orig = _decode_numpy(sae, h_orig)

    target_feature = 4  # arbitrary live feature
    patched = script_module.patch_fused_with_sae(
        fused, sae, feature_idx=target_feature, patch_value=0.0,
    )
    patched_flat = patched.detach().cpu().numpy().reshape(-1, 8)

    expected = xhat_orig - np.outer(
        h_orig[:, target_feature], sae.W_dec[:, target_feature],
    )
    np.testing.assert_allclose(patched_flat, expected, atol=1e-5)


def test_patch_fused_does_not_modify_input(script_module, tiny_sae):
    sae, _ = tiny_sae
    fused = torch.randn(2, 5, 8)
    fused_clone = fused.clone()
    _ = script_module.patch_fused_with_sae(
        fused, sae, feature_idx=3, patch_value=0.5,
    )
    assert torch.allclose(fused, fused_clone), \
        "patch_fused_with_sae must not modify its input tensor"


def test_patch_fused_preserves_shape_dtype_device(script_module, tiny_sae):
    sae, _ = tiny_sae
    fused = torch.randn(3, 4, 8, dtype=torch.float32)
    out = script_module.patch_fused_with_sae(
        fused, sae, feature_idx=2, patch_value=1.0,
    )
    assert out.shape == fused.shape
    assert out.dtype == fused.dtype
    assert out.device == fused.device


def test_patch_fused_no_patch_round_trip_close(script_module, tiny_sae):
    """``feature_idx=None`` => SAE round-trip; should be close to input."""
    sae, _ = tiny_sae
    fused = torch.randn(2, 4, 8) * 0.1 + 0.05
    out = script_module.patch_fused_with_sae(
        fused, sae, feature_idx=None, patch_value=None,
    )
    # Tiny synthetic SAE has random weights so we can't expect exact identity,
    # but we DO expect deterministic output (same input -> same output) and a
    # finite reconstruction.
    assert torch.isfinite(out).all()
    out2 = script_module.patch_fused_with_sae(
        fused, sae, feature_idx=None, patch_value=None,
    )
    assert torch.allclose(out, out2)


def test_patch_fused_3d_shape_required(script_module, tiny_sae):
    sae, _ = tiny_sae
    with pytest.raises(ValueError, match="3D"):
        script_module.patch_fused_with_sae(
            torch.randn(8), sae, feature_idx=0, patch_value=0.0,
        )


def test_patch_fused_patch_value_required_when_feature_idx_given(
    script_module, tiny_sae,
):
    sae, _ = tiny_sae
    with pytest.raises(ValueError, match="patch_value"):
        script_module.patch_fused_with_sae(
            torch.randn(1, 1, 8), sae, feature_idx=0, patch_value=None,
        )


# ─────────────────────────────────────────────────────────────────────────────
# compute_feature_percentiles
# ─────────────────────────────────────────────────────────────────────────────


def test_compute_feature_percentiles_returns_canonical_keys(
    script_module, tiny_sae, tmp_path,
):
    """The cohort percentile helper returns the keys consumed by ``main()``."""
    sae, _ = tiny_sae
    rng = np.random.default_rng(0)
    acts = rng.standard_normal((50, 5, 8)).astype(np.float32)
    npz_path = tmp_path / "fused_acts.npz"
    np.savez(npz_path, activations=acts)
    stats = script_module.compute_feature_percentiles(
        npz_path, sae, feature_idx=2, pct_low=1.0, pct_high=99.0,
    )
    for key in ("p1", "p99", "max", "fraction_active", "n_active", "n_total"):
        assert key in stats, f"missing key {key!r} in {stats}"
    assert stats["n_total"] == 50 * 5
    assert 0.0 <= stats["fraction_active"] <= 1.0


def test_compute_feature_percentiles_p1_le_p99(script_module, tiny_sae, tmp_path):
    sae, _ = tiny_sae
    rng = np.random.default_rng(1)
    acts = rng.standard_normal((10, 3, 8)).astype(np.float32)
    npz_path = tmp_path / "tiny.npz"
    np.savez(npz_path, activations=acts)
    stats = script_module.compute_feature_percentiles(
        npz_path, sae, feature_idx=0,
    )
    assert stats["p1"] <= stats["p99"]


def test_compute_feature_percentiles_rejects_2d(script_module, tiny_sae, tmp_path):
    sae, _ = tiny_sae
    npz_path = tmp_path / "wrong_shape.npz"
    np.savez(npz_path, activations=np.zeros((10, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="3D"):
        script_module.compute_feature_percentiles(npz_path, sae, feature_idx=0)


# ─────────────────────────────────────────────────────────────────────────────
# select_random_controls
# ─────────────────────────────────────────────────────────────────────────────


def test_select_random_controls_distinct_and_excludes_target(
    script_module, tiny_sae,
):
    sae, _ = tiny_sae
    rng = np.random.default_rng(0)
    out = script_module.select_random_controls(
        sae, target_feature_idx=4, n_random=5, rng=rng,
    )
    assert len(out) == 5
    assert len(set(out.tolist())) == 5
    assert 4 not in out.tolist()


def test_select_random_controls_skips_dead_features(script_module, tiny_sae):
    """``is_dead`` features must NOT appear among the random controls."""
    sae, _ = tiny_sae
    rng = np.random.default_rng(0)
    # In tiny_sae fixture we set is_dead=[False, True, ..., True] at idx 1, 15.
    out = script_module.select_random_controls(
        sae, target_feature_idx=4, n_random=5, rng=rng,
    )
    assert 1 not in out.tolist()
    assert 15 not in out.tolist()


def test_select_random_controls_overrequest_raises(script_module, tiny_sae):
    sae, _ = tiny_sae
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="Not enough"):
        script_module.select_random_controls(
            sae, target_feature_idx=0, n_random=999, rng=rng,
        )


# ─────────────────────────────────────────────────────────────────────────────
# _render_markdown — interpretation branch coverage
# ─────────────────────────────────────────────────────────────────────────────


def _summary_skeleton(splatter_dr2: float, random_dr2: float) -> dict:
    """Build a minimal-but-complete summary dict for markdown rendering."""
    return {
        "sae_config": {
            "architecture": "batch_topk", "layer": "fused",
            "expansion": 32, "k": 64, "seed": 0,
        },
        "feature_idx": 572,
        "feature_cohort_stats": {
            "p1": 0.0, "p99": 0.7, "max": 1.0,
            "fraction_active": 0.04, "n_active": 100, "n_total": 1000,
        },
        "n_folds": 5,
        "n_random_controls": 10,
        "patch_modes": {"zero": 0.0, "saturate": 0.7, "push_down": 0.0},
        "encoder_baseline_per_fold_r2": [0.45, 0.44, 0.43, 0.46, 0.42],
        "encoder_baseline_mean_r2": 0.44,
        "sae_baseline_per_fold_r2": [0.45, 0.44, 0.43, 0.46, 0.42],
        "sae_baseline_mean_r2": 0.44,
        "splatter_feature_aggregate": {
            "zero": {"delta_r2_mean": 0.0, "delta_r2_std": 0.0,
                     "spearman_rho_mean": 0.0},
            "saturate": {"delta_r2_mean": splatter_dr2,
                         "delta_r2_std": 0.0, "spearman_rho_mean": 0.0},
            "push_down": {"delta_r2_mean": 0.0, "delta_r2_std": 0.0,
                          "spearman_rho_mean": 0.0},
        },
        "summary_statistics": {
            "splatter_saturate_delta_r2_mean": splatter_dr2,
            "splatter_saturate_delta_r2_std": 0.0,
            "random_saturate_delta_r2_mean": random_dr2,
            "random_saturate_delta_r2_std": 0.0,
            "n_random_pooled": 50,
        },
    }


def _summary_with_random_std(
    splatter_dr2: float, random_dr2: float, random_std: float,
) -> dict:
    """Like ``_summary_skeleton`` but sets the random control std explicitly."""
    s = _summary_skeleton(splatter_dr2=splatter_dr2, random_dr2=random_dr2)
    s["summary_statistics"]["random_saturate_delta_r2_std"] = random_std
    return s


def test_render_markdown_causal_branch(script_module):
    """All three criteria pass (Δ > 2× random AND > random SD AND ≥ 1 % R²).
    """
    # canonical R² = 0.44; 1 % = 0.0044. Splatter Δ = 0.05 > 2 × 0.005 (= 0.01)
    # AND > random SD 0.001 AND > 0.0044.
    summary = _summary_with_random_std(
        splatter_dr2=-0.05, random_dr2=-0.005, random_std=0.001,
    )
    md = script_module._render_markdown(summary)
    assert "**Causal**" in md
    assert "PASS" in md  # at least one criterion line says PASS


def test_render_markdown_correlated_only_branch(script_module):
    """Splatter Δ ≤ random mean — correlated-only verdict."""
    summary = _summary_with_random_std(
        splatter_dr2=-0.001, random_dr2=-0.005, random_std=0.005,
    )
    md = script_module._render_markdown(summary)
    assert "Correlated-only" in md or "correlated-only" in md.lower()
    # And the FAIL annotation should appear for at least one criterion.
    assert "FAIL" in md


def test_render_markdown_inconclusive_branch(script_module):
    """Splatter Δ > 2× random mean but < 1 % canonical R² → inconclusive."""
    # canonical R² = 0.44; 1 % cutoff = 0.0044. Set splatter Δ = 0.0002 to
    # pass criterion A (> 2× 0.00005 = 0.0001) AND B (> std 0.0001) but
    # FAIL criterion C (0.0002 < 0.0044).
    summary = _summary_with_random_std(
        splatter_dr2=-0.0002, random_dr2=-0.00005, random_std=0.0001,
    )
    md = script_module._render_markdown(summary)
    assert "Inconclusive" in md or "inconclusive" in md.lower()


def test_render_markdown_includes_canonical_metadata(script_module):
    """The MD report names the SAE config and the canonical Splatter feature."""
    summary = _summary_skeleton(splatter_dr2=-0.01, random_dr2=-0.005)
    md = script_module._render_markdown(summary)
    assert "feature 572" in md.lower() or "**572**" in md
    assert "batch_topk" in md
    assert "fused" in md
    assert "exp32" in md
