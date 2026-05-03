"""Tests for compare_trained_vs_random_sae.py.

Covers (Fix C deferred — File 7 top features index annotation):
  - ``_annotate_top_features`` joins indices to feature_report.json metadata.
  - ``_top10_decoder_cos_sim`` returns the canonical hungarian metric +
    cosine matrix + the new ``top_features_*_annotated`` lists.
  - ``_interpretable_fraction`` / ``_mw_p_cog_lt_05_count`` /
    ``_dead_fraction`` numerical correctness.
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.resdec_mhe.interpretability.compare_trained_vs_random_sae import (
    _annotate_top_features,
    _dead_fraction,
    _interpretable_fraction,
    _mw_p_cog_lt_05_count,
    _top10_decoder_cos_sim,
)


class TestAnnotateTopFeatures:

    def test_basic_join(self):
        reports = [
            {
                "flags": ["interpretable_candidate"],
                "mw_p_cognition": 0.01,
                "ct_dominance": 0.42,
                "dominant_cell_type": "Splatter",
                "fraction_active": 0.15,
            },
            {
                "flags": ["dead"],
                "mw_p_cognition": None,
                "fraction_active": 0.0,
            },
            {
                "flags": [],
                "mw_p_cognition": 0.07,
                "ct_dominance": 0.18,
                "dominant_cell_type": "Microglia",
                "fraction_active": 0.05,
            },
        ]
        top_idx = np.array([0, 2])
        out = _annotate_top_features(top_idx, reports)
        assert out[0]["feature_index"] == 0
        assert out[0]["flags"] == ["interpretable_candidate"]
        assert out[0]["dominant_cell_type"] == "Splatter"
        assert out[0]["ct_dominance"] == pytest.approx(0.42)
        assert out[1]["feature_index"] == 2
        assert out[1]["dominant_cell_type"] == "Microglia"

    def test_missing_keys_become_none(self):
        # Sparse report missing some optional keys.
        reports = [{"flags": []}]
        out = _annotate_top_features(np.array([0]), reports)
        assert out[0]["mw_p_cognition"] is None
        assert out[0]["ct_dominance"] is None
        assert out[0]["dominant_cell_type"] is None
        assert out[0]["fraction_active"] is None

    def test_index_out_of_bounds_returns_empty(self):
        # If reports is shorter than top_idx, return blank metadata for
        # the out-of-range index — defensive against schema drift.
        reports = [{"flags": ["a"], "fraction_active": 0.1}]
        out = _annotate_top_features(np.array([0, 5]), reports)
        assert out[0]["feature_index"] == 0
        assert out[0]["fraction_active"] == 0.1
        assert out[1]["feature_index"] == 5
        # All optional fields default to None / [].
        assert out[1]["flags"] == []
        assert out[1]["mw_p_cognition"] is None

    def test_preserves_top_idx_order(self):
        reports = [
            {"feature_idx_marker": 0, "fraction_active": 0.0},
            {"feature_idx_marker": 1, "fraction_active": 0.0},
            {"feature_idx_marker": 2, "fraction_active": 0.0},
        ]
        out = _annotate_top_features(np.array([2, 0, 1]), reports)
        assert [d["feature_index"] for d in out] == [2, 0, 1]


class TestTop10DecoderCosSim:

    def _make_inputs(self, K: int, n_dim: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        W_dec_a = rng.normal(size=(n_dim, K))
        W_dec_b = rng.normal(size=(n_dim, K))
        fa_a = rng.uniform(size=K)
        fa_b = rng.uniform(size=K)
        # Build trivial reports lists (one entry per feature).
        reports_a = [
            {"flags": ["interpretable_candidate"] if i % 2 == 0 else [],
             "mw_p_cognition": 0.01 * i,
             "ct_dominance": 0.1 * i,
             "dominant_cell_type": f"CT_{i}",
             "fraction_active": float(fa_a[i])}
            for i in range(K)
        ]
        reports_b = [
            {"flags": [],
             "mw_p_cognition": None,
             "fraction_active": float(fa_b[i])}
            for i in range(K)
        ]
        return W_dec_a, fa_a, reports_a, W_dec_b, fa_b, reports_b

    def test_returns_canonical_keys(self):
        wA, faA, repA, wB, faB, repB = self._make_inputs(K=10, n_dim=12)
        out = _top10_decoder_cos_sim(wA, faA, repA, wB, faB, repB)
        expected_keys = {
            "cosine_matrix", "mean_abs_cosine_off_diag",
            "hungarian_mean_diagonal_cosine", "hungarian_assignment",
            "K", "top_features_trained", "top_features_random",
            "top_features_trained_annotated", "top_features_random_annotated",
        }
        assert expected_keys <= set(out.keys())

    def test_K_caps_at_10(self):
        wA, faA, repA, wB, faB, repB = self._make_inputs(K=20, n_dim=12)
        out = _top10_decoder_cos_sim(wA, faA, repA, wB, faB, repB)
        assert out["K"] == 10

    def test_K_smaller_when_inputs_smaller(self):
        wA, faA, repA, wB, faB, repB = self._make_inputs(K=5, n_dim=12)
        out = _top10_decoder_cos_sim(wA, faA, repA, wB, faB, repB)
        assert out["K"] == 5

    def test_top_features_annotated_length_matches_K(self):
        wA, faA, repA, wB, faB, repB = self._make_inputs(K=8, n_dim=12)
        out = _top10_decoder_cos_sim(wA, faA, repA, wB, faB, repB)
        assert len(out["top_features_trained_annotated"]) == out["K"]
        assert len(out["top_features_random_annotated"]) == out["K"]

    def test_annotated_entries_match_top_features_indices(self):
        wA, faA, repA, wB, faB, repB = self._make_inputs(K=6, n_dim=8)
        out = _top10_decoder_cos_sim(wA, faA, repA, wB, faB, repB)
        for idx, ann in zip(out["top_features_trained"],
                            out["top_features_trained_annotated"]):
            assert ann["feature_index"] == idx

    def test_annotated_carries_dominant_cell_type(self):
        """The trained-side annotated entries should expose the
        dominant_cell_type field from the matched feature_report.json row."""
        wA, faA, repA, wB, faB, repB = self._make_inputs(K=6, n_dim=8)
        out = _top10_decoder_cos_sim(wA, faA, repA, wB, faB, repB)
        for ann in out["top_features_trained_annotated"]:
            i = ann["feature_index"]
            assert ann["dominant_cell_type"] == f"CT_{i}"

    def test_decoder_dim_mismatch_raises(self):
        wA = np.zeros((10, 5))
        wB = np.zeros((11, 5))
        faA = np.zeros(5)
        faB = np.zeros(5)
        with pytest.raises(ValueError, match="Decoder row dims differ"):
            _top10_decoder_cos_sim(wA, faA, [], wB, faB, [])

    def test_identical_decoders_high_hungarian_diagonal(self):
        """If the two decoders are identical and fa_* identical, the
        hungarian-aligned mean should be 1.0."""
        rng = np.random.default_rng(0)
        K = 8
        n_dim = 12
        W = rng.normal(size=(n_dim, K))
        fa = rng.uniform(size=K)
        reports = [{"flags": [], "fraction_active": float(fa[i])} for i in range(K)]
        out = _top10_decoder_cos_sim(W, fa, reports, W.copy(), fa.copy(), reports)
        assert out["hungarian_mean_diagonal_cosine"] == pytest.approx(1.0, abs=1e-6)


class TestInterpretableFraction:

    def test_basic(self):
        reports = [
            {"flags": ["interpretable_candidate"]},
            {"flags": ["dead"]},
            {"flags": ["interpretable_candidate", "ubiquitous"]},
        ]
        assert _interpretable_fraction(reports) == pytest.approx(2 / 3)

    def test_empty_returns_nan(self):
        assert np.isnan(_interpretable_fraction([]))


class TestMwPCogLt05Count:

    def test_counts_significant(self):
        reports = [
            {"mw_p_cognition": 0.01},
            {"mw_p_cognition": 0.06},
            {"mw_p_cognition": 0.04},
            {"mw_p_cognition": None},
        ]
        assert _mw_p_cog_lt_05_count(reports) == 2

    def test_missing_key_skipped(self):
        reports = [{"flags": []}, {"mw_p_cognition": 0.001}]
        assert _mw_p_cog_lt_05_count(reports) == 1


class TestDeadFraction:

    def test_basic(self):
        reports = [
            {"flags": ["dead"]},
            {"flags": []},
            {"flags": ["dead", "ubiquitous"]},
            {"flags": ["interpretable_candidate"]},
        ]
        assert _dead_fraction(reports) == pytest.approx(2 / 4)

    def test_empty_returns_nan(self):
        assert np.isnan(_dead_fraction([]))
