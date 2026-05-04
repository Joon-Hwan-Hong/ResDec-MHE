"""Unit tests for src.analysis.differential (DAE/DCR/DCCI helpers)."""
import numpy as np
import pandas as pd
import pytest
from src.analysis.differential import (
    binned_subgroup_dge_deseq2,
    differential_attribution_effect,
    differential_ccc_importance,
    differential_ct_ranking,
)


def test_dae_returns_per_pair_pvalues():
    rng = np.random.default_rng(42)
    canonical = rng.normal(0, 0.01, (5, 31, 100))
    variant = canonical + rng.normal(0, 0.001, (5, 31, 100))
    res = differential_attribution_effect(
        canonical, variant,
        ct_names=[f"CT{i}" for i in range(31)],
        gene_names=[f"GENE{j}" for j in range(100)],
    )
    assert len(res) == 31 * 100
    assert "p_wilcoxon" in res.columns
    assert "padj_bh" in res.columns
    assert "mean_diff" in res.columns
    assert "cell_type" in res.columns
    assert "gene" in res.columns


def test_dae_shape_mismatch_raises():
    canonical = np.zeros((5, 31, 100))
    variant = np.zeros((5, 31, 50))
    with pytest.raises((AssertionError, ValueError)):
        differential_attribution_effect(
            canonical, variant,
            ct_names=[f"CT{i}" for i in range(31)],
            gene_names=[f"GENE{j}" for j in range(100)],
        )


def test_dae_padj_is_monotone_in_p():
    rng = np.random.default_rng(0)
    n_ct, n_gene = 5, 10
    canonical = rng.normal(0, 1, (5, n_ct, n_gene))
    variant = canonical + rng.normal(0, 1, (5, n_ct, n_gene))
    res = differential_attribution_effect(
        canonical, variant,
        ct_names=[f"CT{i}" for i in range(n_ct)],
        gene_names=[f"G{j}" for j in range(n_gene)],
    )
    sorted_res = res.sort_values("p_wilcoxon").reset_index(drop=True)
    diffs = np.diff(sorted_res["padj_bh"].to_numpy())
    assert (diffs >= -1e-12).all(), "BH-adjusted padj must be non-decreasing in p"


def test_dcr_returns_spearman_per_method():
    canonical_ranks = {
        "method_a": list(range(31)),
        "method_b": list(range(30, -1, -1)),  # [30, 29, ..., 0] — full overlap with method_a
    }
    variant_ranks = {
        "method_a": list(range(31)),
        "method_b": list(range(31)),  # opposite order from canonical method_b
    }
    res = differential_ct_ranking(canonical_ranks, variant_ranks)
    assert res["method_a"]["spearman_rho"] > 0.99
    assert res["method_b"]["spearman_rho"] < -0.99
    assert res["method_a"]["n"] == 31
    assert res["method_b"]["n"] == 31


def test_dcr_handles_partial_overlap():
    canonical_ranks = {"m": ["A", "B", "C", "D"]}
    variant_ranks = {"m": ["B", "C", "D", "E"]}  # 3 overlap (B, C, D)
    res = differential_ct_ranking(canonical_ranks, variant_ranks)
    assert res["m"]["n"] == 3


def test_dcr_returns_nan_when_overlap_too_small():
    canonical_ranks = {"m": ["A", "B"]}
    variant_ranks = {"m": ["B", "C"]}  # only 1 overlap
    res = differential_ct_ranking(canonical_ranks, variant_ranks)
    assert np.isnan(res["m"]["spearman_rho"])
    assert res["m"]["n"] == 1


def test_dcr_skips_methods_in_one_dict_only():
    canonical_ranks = {"a": list(range(10)), "b": list(range(10))}
    variant_ranks = {"a": list(range(10)), "c": list(range(10))}
    res = differential_ct_ranking(canonical_ranks, variant_ranks)
    assert "a" in res
    assert "b" not in res
    assert "c" not in res


def test_dcci_returns_per_pair_pvalues():
    rng = np.random.default_rng(42)
    n_folds, n_ct = 5, 7
    canonical = rng.normal(0, 0.05, (n_folds, n_ct, n_ct))
    variant = canonical + rng.normal(0, 0.005, (n_folds, n_ct, n_ct))
    res = differential_ccc_importance(
        canonical, variant, ct_names=[f"CT{i}" for i in range(n_ct)],
    )
    assert len(res) == n_ct * n_ct
    assert "ct_source" in res.columns
    assert "ct_target" in res.columns
    assert "p_wilcoxon" in res.columns
    assert "padj_bh" in res.columns


def test_dcci_shape_mismatch_raises():
    canonical = np.zeros((5, 3, 3))
    variant = np.zeros((5, 3, 4))
    with pytest.raises(ValueError):
        differential_ccc_importance(
            canonical, variant, ct_names=["A", "B", "C"],
        )


def test_dcci_signal_recovers():
    rng = np.random.default_rng(0)
    n_folds, n_ct = 5, 4
    canonical = rng.normal(0, 0.01, (n_folds, n_ct, n_ct))
    variant = canonical.copy()
    # Plant: edge (0, 1) has +0.5 shift in variant
    variant[:, 0, 1] += 0.5
    res = differential_ccc_importance(
        canonical, variant, ct_names=[f"CT{i}" for i in range(n_ct)],
    )
    top = res.iloc[0]
    assert top["ct_source"] == "CT0"
    assert top["ct_target"] == "CT1"


def test_binned_subgroup_dge_deseq2_handles_filtered_ct(monkeypatch):
    """Filter branch must produce a DataFrame so pd.concat doesn't choke on dicts.

    A CT with <5 subjects per group should yield NaN-filled rows; the rest
    should run through the DESeq2 path. We monkeypatch deseq2_de to return a
    deterministic DataFrame so the test runs in milliseconds without invoking
    pydeseq2.
    """
    n_subj, n_ct, n_gene = 12, 3, 4
    rng = np.random.default_rng(0)
    counts = rng.integers(0, 50, size=(n_subj, n_ct, n_gene))
    # Drop CT 1's counts to zero — its row-sum filter rejects all subjects.
    counts[:, 1, :] = 0
    ct_names = ["CT0", "CT1_filtered", "CT2"]
    gene_names = [f"G{i}" for i in range(n_gene)]
    resilient_idx = list(range(0, 6))
    vulnerable_idx = list(range(6, 12))

    fake_de = pd.DataFrame({
        "gene": gene_names,
        "log2_fold_change": [0.1] * n_gene,
        "p_wald": [0.5] * n_gene,
        "padj_bh": [0.7] * n_gene,
    })

    monkeypatch.setattr(
        "src.analysis.de_resilience.deseq2_de",
        lambda **kw: fake_de.copy(),
    )

    df = binned_subgroup_dge_deseq2(
        raw_counts_pseudobulk=counts,
        ct_names=ct_names,
        gene_names=gene_names,
        resilient_idx=resilient_idx,
        vulnerable_idx=vulnerable_idx,
        min_cells_per_subject=1,
    )
    # 3 CTs × 4 genes = 12 rows; CT1_filtered must produce 4 NaN rows.
    assert len(df) == n_ct * n_gene
    filtered_rows = df[df["cell_type"] == "CT1_filtered"]
    assert len(filtered_rows) == n_gene
    assert filtered_rows["log2_fold_change"].isna().all()
    assert filtered_rows["padj_bh"].isna().all()
    # Non-filtered CTs got the fake DESeq2 output.
    ok_rows = df[df["cell_type"] == "CT0"]
    assert (ok_rows["log2_fold_change"] == 0.1).all()
