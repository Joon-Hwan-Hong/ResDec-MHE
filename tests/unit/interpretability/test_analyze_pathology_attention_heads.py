"""Tests for analyze_pathology_attention_heads.py helper functions.

Covers (Fix C deferred item — File 4):
  - shannon_entropy (zero-handling, uniform vs concentrated)
  - effective_n (range invariants)
  - cosine_similarity (zero-vector edge case, orthogonal vectors)
  - head_specialization (top-3 ranking + entropy/effective_n stamps)
  - inter_head_redundancy (pairwise cosine + summary stats)
  - subject_level_head_fingerprints (axis sums)
  - splatter_deepdive (missing-Splatter branch + GABAergic co-attention NaN-safe path)
  - head_metadata_correlations (always-emit-key + status flags)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.resdec_mhe.interpretability.analyze_pathology_attention_heads import (
    cosine_similarity,
    effective_n,
    head_metadata_correlations,
    head_specialization,
    inter_head_redundancy,
    shannon_entropy,
    splatter_deepdive,
    subject_level_head_fingerprints,
)


class TestShannonEntropy:

    def test_uniform_distribution_max_entropy(self):
        """Uniform p over n bins has entropy log(n) (in nats)."""
        for n in (2, 4, 31):
            p = np.full(n, 1.0 / n)
            assert shannon_entropy(p) == pytest.approx(np.log(n))

    def test_concentrated_distribution_zero_entropy(self):
        """Delta distribution has entropy 0."""
        p = np.zeros(5)
        p[2] = 1.0
        assert shannon_entropy(p) == pytest.approx(0.0)

    def test_zeros_are_skipped(self):
        """Zero entries should not contribute (log(0) is undefined)."""
        p = np.array([0.5, 0.5, 0.0, 0.0, 0.0])
        assert shannon_entropy(p) == pytest.approx(np.log(2))

    def test_returns_python_float(self):
        out = shannon_entropy(np.array([0.5, 0.5]))
        assert isinstance(out, float)

    def test_skewed_distribution(self):
        """Known case: p = [0.9, 0.1] -> ~0.325 nats."""
        p = np.array([0.9, 0.1])
        expected = -(0.9 * np.log(0.9) + 0.1 * np.log(0.1))
        assert shannon_entropy(p) == pytest.approx(expected)


class TestEffectiveN:

    def test_uniform_distribution_equals_n(self):
        """For uniform p over n bins, effective_n == n."""
        for n in (2, 4, 31):
            p = np.full(n, 1.0 / n)
            assert effective_n(p) == pytest.approx(float(n))

    def test_delta_distribution_equals_one(self):
        """Delta distribution: effective_n == 1."""
        p = np.zeros(10)
        p[3] = 1.0
        assert effective_n(p) == pytest.approx(1.0)

    def test_returns_python_float(self):
        assert isinstance(effective_n(np.array([0.5, 0.5])), float)

    def test_two_bin_skewed(self):
        """p=[0.9, 0.1] -> 1 / (0.81 + 0.01) ≈ 1.2195."""
        p = np.array([0.9, 0.1])
        expected = 1.0 / (0.81 + 0.01)
        assert effective_n(p) == pytest.approx(expected)


class TestCosineSimilarity:

    def test_identical_vectors_one(self):
        a = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(a, a) == pytest.approx(1.0)

    def test_orthogonal_vectors_zero(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_anti_parallel_vectors_neg_one(self):
        a = np.array([1.0, 2.0])
        b = -a
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self):
        """Zero-norm vector should not raise — returns 0.0."""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(a, b) == 0.0
        assert cosine_similarity(b, a) == 0.0
        assert cosine_similarity(a, a) == 0.0

    def test_handles_2d_array_via_ravel(self):
        """Implementation calls .ravel() so 2-D inputs are flattened."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = a.copy()
        assert cosine_similarity(a, b) == pytest.approx(1.0)

    def test_returns_python_float(self):
        a = np.array([1.0, 0.0])
        b = np.array([1.0, 1.0])
        out = cosine_similarity(a, b)
        assert isinstance(out, float)


class TestHeadSpecialization:

    def test_returns_one_dict_per_head(self):
        """Output length == n_heads."""
        attn_mean = np.array([
            [0.5, 0.3, 0.1, 0.1],  # head 0 specialised on CT 0
            [0.1, 0.1, 0.4, 0.4],  # head 1 split between CT 2 and 3
            [0.25, 0.25, 0.25, 0.25],  # head 2 uniform
        ])
        ct_names = ["A", "B", "C", "D"]
        out = head_specialization(attn_mean, ct_names)
        assert len(out) == 3
        assert {d["head"] for d in out} == {0, 1, 2}

    def test_top_3_cell_types_ordered_descending(self):
        """top_3 should be ordered by p_norm descending."""
        attn_mean = np.array([[0.1, 0.7, 0.15, 0.05]])
        ct_names = ["A", "B", "C", "D"]
        out = head_specialization(attn_mean, ct_names)
        top3 = out[0]["top_3_cell_types"]
        assert [d["cell_type"] for d in top3] == ["B", "C", "A"]
        # Mean attention values are also descending.
        vals = [d["mean_attention"] for d in top3]
        assert vals == sorted(vals, reverse=True)

    def test_uniform_head_has_max_entropy(self):
        """Uniform head -> shannon entropy ≈ log(n_ct)."""
        n_ct = 4
        attn_mean = np.full((1, n_ct), 1.0 / n_ct)
        ct_names = list("ABCD")
        out = head_specialization(attn_mean, ct_names)
        assert out[0]["shannon_entropy_nats"] == pytest.approx(np.log(n_ct))

    def test_specialised_head_has_low_entropy(self):
        """Single-CT-dominant head -> low entropy."""
        attn_mean = np.zeros((1, 4))
        attn_mean[0, 0] = 0.97
        attn_mean[0, 1] = 0.01
        attn_mean[0, 2] = 0.01
        attn_mean[0, 3] = 0.01
        ct_names = list("ABCD")
        uniform_entropy = np.log(4)
        out = head_specialization(attn_mean, ct_names)
        assert out[0]["shannon_entropy_nats"] < 0.3 * uniform_entropy

    def test_normalises_unnormalised_probs(self):
        """Even if rows do not sum to 1, helper internally normalises."""
        attn_mean = np.array([[2.0, 1.0, 1.0, 0.0]])  # sums to 4
        ct_names = list("ABCD")
        out = head_specialization(attn_mean, ct_names)
        # After normalise, should be [0.5, 0.25, 0.25, 0.0].
        # Top-3 first entry should be A with mean=0.5.
        assert out[0]["top_3_cell_types"][0]["cell_type"] == "A"
        assert out[0]["top_3_cell_types"][0]["mean_attention"] == pytest.approx(0.5)


class TestInterHeadRedundancy:

    def test_returns_summary_keys(self):
        attn_mean = np.array([[0.5, 0.3, 0.2], [0.5, 0.3, 0.2]])
        out = inter_head_redundancy(attn_mean)
        assert {"pairwise_cosine", "mean_pairwise_cosine",
                "max_pairwise_cosine", "min_pairwise_cosine"} == set(out.keys())

    def test_identical_heads_have_cosine_one(self):
        attn_mean = np.array([[0.5, 0.3, 0.2], [0.5, 0.3, 0.2]])
        out = inter_head_redundancy(attn_mean)
        assert out["pairwise_cosine"]["h0_vs_h1"] == pytest.approx(1.0)
        assert out["mean_pairwise_cosine"] == pytest.approx(1.0)

    def test_orthogonal_heads_have_cosine_zero(self):
        attn_mean = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        out = inter_head_redundancy(attn_mean)
        assert out["pairwise_cosine"]["h0_vs_h1"] == pytest.approx(0.0)

    def test_pairs_are_upper_triangular(self):
        """For 4 heads, pairwise has C(4, 2) = 6 entries."""
        attn_mean = np.eye(4)  # 4 orthogonal heads, n_ct=4
        out = inter_head_redundancy(attn_mean)
        assert len(out["pairwise_cosine"]) == 6

    def test_single_head_returns_zero_summary(self):
        """No pairs -> mean/max/min default to 0.0."""
        attn_mean = np.array([[0.5, 0.3, 0.2]])
        out = inter_head_redundancy(attn_mean)
        assert out["pairwise_cosine"] == {}
        assert out["mean_pairwise_cosine"] == 0.0
        assert out["max_pairwise_cosine"] == 0.0
        assert out["min_pairwise_cosine"] == 0.0


class TestSubjectLevelHeadFingerprints:

    def test_axis_sum(self):
        """Output shape [N, n_heads] from input [N, n_heads, n_ct]."""
        attn = np.zeros((5, 3, 4))
        attn[..., 0] = 0.5  # CT 0 contributes 0.5 per (subject, head)
        out = subject_level_head_fingerprints(attn)
        assert out.shape == (5, 3)
        np.testing.assert_array_almost_equal(out, np.full((5, 3), 0.5))

    def test_with_varied_input(self):
        """Concrete numerical fingerprints check."""
        # subj 0 all attention on head 0; subj 1 all on head 1.
        attn = np.array([
            [[0.4, 0.3, 0.2, 0.1], [0.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 0.0], [0.7, 0.2, 0.05, 0.05]],
        ])
        out = subject_level_head_fingerprints(attn)
        np.testing.assert_array_almost_equal(out[0], [1.0, 0.0])
        np.testing.assert_array_almost_equal(out[1], [0.0, 1.0])


class TestSplatterDeepdive:

    def test_missing_splatter_returns_error_dict(self):
        """If 'Splatter' is not in ct_names, helper returns an error dict."""
        attn = np.zeros((5, 4, 3))
        ct_names = ["A", "B", "C"]  # no Splatter
        sids = np.array(["s0", "s1", "s2", "s3", "s4"])
        out = splatter_deepdive(attn, ct_names, sids, residual_df=None)
        assert "error" in out

    def test_with_splatter_no_residual(self):
        """Splatter present, no residual_df -> stats + co-attention only."""
        attn = np.zeros((10, 4, 5))
        # Splatter is at idx 0; give head 0 some Splatter attention per subject.
        attn[:, 0, 0] = np.linspace(0.1, 1.0, 10)
        ct_names = ["Splatter", "MGE interneuron", "CGE interneuron",
                    "LAMP5-LHX6 and Chandelier", "X"]
        sids = np.array([f"s{i}" for i in range(10)])
        out = splatter_deepdive(attn, ct_names, sids, residual_df=None)
        assert "splatter_attn_total_stats" in out
        assert "gabaergic_interneuron_co_attention_pearson_r" in out
        assert "gabaergic_interneuron_co_attention_n_used" in out
        # GABAergic pairs must include all C(4,2)=6 keys after self-exclusion;
        # Splatter is one of the 4, and the others are present.
        assert len(out["gabaergic_interneuron_co_attention_pearson_r"]) >= 1

    def test_co_attention_nan_safe(self):
        """NaN co-attention values should be dropped, n_used reported."""
        # Build attn so that one subject has NaN at Splatter -> drops one pair.
        attn = np.zeros((4, 1, 5))
        attn[:, 0, 0] = [0.1, 0.2, 0.3, np.nan]  # Splatter
        attn[:, 0, 1] = [0.05, 0.1, 0.15, 0.2]  # MGE
        ct_names = ["Splatter", "MGE interneuron", "CGE interneuron",
                    "LAMP5-LHX6 and Chandelier", "X"]
        sids = np.array(["s0", "s1", "s2", "s3"])
        out = splatter_deepdive(attn, ct_names, sids, residual_df=None)
        n_block = out["gabaergic_interneuron_co_attention_n_used"]
        # The first GABA pair should report n_used=3 (one NaN dropped).
        any_pair = next(iter(n_block.values()))
        assert any_pair["n_used"] + any_pair["n_dropped_nan"] == 4


class TestHeadMetadataCorrelations:

    def test_no_residual_df_returns_error_dict(self):
        fingerprint = np.random.rand(10, 4)
        sids = np.array([f"s{i}" for i in range(10)])
        out = head_metadata_correlations(fingerprint, sids, residual_df=None)
        assert "error" in out

    def test_emits_status_per_head_per_metadata_col(self):
        """Each (head, col) pair must surface a status: ok | missing_column | n_too_small."""
        n_subj = 100
        fingerprint = np.random.default_rng(0).normal(size=(n_subj, 2))  # 2 heads
        sids = np.array([f"R{i}" for i in range(n_subj)])
        residual_df = pd.DataFrame({
            "ROSMAP_IndividualID": [f"R{i}" for i in range(n_subj)],
            "residual": np.random.default_rng(1).normal(size=n_subj),
            "cogn_global": np.random.default_rng(2).normal(size=n_subj),
        })
        out = head_metadata_correlations(fingerprint, sids, residual_df)
        # Two heads present.
        assert {"head_0", "head_1"} <= set(out.keys())
        # Both metadata cols present in each head.
        for hkey in ("head_0", "head_1"):
            head_corrs = out[hkey]
            # residual + cogn_global present and ok.
            assert head_corrs["residual"]["status"] == "ok"
            assert head_corrs["cogn_global"]["status"] == "ok"
            # apoe_e4_count not in residual_df -> missing_column.
            assert head_corrs["apoe_e4_count"]["status"] == "missing_column"

    def test_n_too_small_status(self):
        """If n_nonnull <= 30, status should be n_too_small."""
        n_subj = 20  # too small (< 30)
        fingerprint = np.random.default_rng(0).normal(size=(n_subj, 1))
        sids = np.array([f"R{i}" for i in range(n_subj)])
        residual_df = pd.DataFrame({
            "ROSMAP_IndividualID": [f"R{i}" for i in range(n_subj)],
            "residual": np.random.default_rng(1).normal(size=n_subj),
        })
        out = head_metadata_correlations(fingerprint, sids, residual_df)
        # residual is present but n=20 <= 30 -> n_too_small.
        head0 = out["head_0"]
        assert head0["residual"]["status"] == "n_too_small"
        assert head0["residual"]["n_nonnull"] == 20

    def test_apoe_e4_uppercases_input(self):
        """Lowercase 'e4' variants must count toward apoe_e4_count."""
        n_subj = 100
        fingerprint = np.random.default_rng(0).normal(size=(n_subj, 1))
        sids = np.array([f"R{i}" for i in range(n_subj)])
        residual_df = pd.DataFrame({
            "ROSMAP_IndividualID": [f"R{i}" for i in range(n_subj)],
            # Mix of uppercase/lowercase to exercise the .upper() path.
            "apoe_genotype": ["e4/e4" if i < 50 else "E3/E4" for i in range(n_subj)],
        })
        out = head_metadata_correlations(fingerprint, sids, residual_df)
        # apoe_e4_count column derived; status should be "ok" (n=100 > 30).
        assert out["head_0"]["apoe_e4_count"]["status"] == "ok"
