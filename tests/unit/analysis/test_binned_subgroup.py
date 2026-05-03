"""Unit tests for src.analysis.differential binned-subgroup helpers."""
import numpy as np
import pandas as pd
import pytest
from src.analysis.differential import (
    quartile_subgroup_indices,
    binned_subgroup_ct_importance,
    binned_subgroup_dge_wilcoxon,
)


def test_quartile_subgroup_indices_returns_top_and_bottom():
    target = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    res = quartile_subgroup_indices(target, quartile=0.25)
    assert set(res["resilient"].tolist()) == {6, 7}
    assert set(res["vulnerable"].tolist()) == {0, 1}


def test_quartile_subgroup_handles_nan():
    target = np.array([0.0, np.nan, 2.0, 3.0, 4.0, 5.0, np.nan, 7.0])
    res = quartile_subgroup_indices(target, quartile=0.25)
    assert all(not np.isnan(target[i]) for i in res["resilient"])
    assert all(not np.isnan(target[i]) for i in res["vulnerable"])


def test_quartile_subgroup_quartile_validation():
    target = np.arange(100.0)
    with pytest.raises(ValueError):
        quartile_subgroup_indices(target, quartile=0.6)  # > 0.5
    with pytest.raises(ValueError):
        quartile_subgroup_indices(target, quartile=0.0)


def test_binned_subgroup_ct_importance_returns_per_ct():
    rng = np.random.default_rng(0)
    n_subj = 100
    n_ct = 5
    attrib = rng.normal(0, 1, (n_subj, n_ct))
    resilient = np.arange(0, 25)
    vulnerable = np.arange(75, 100)
    res = binned_subgroup_ct_importance(
        attrib, resilient_idx=resilient, vulnerable_idx=vulnerable,
        ct_names=[f"CT{i}" for i in range(n_ct)],
    )
    assert len(res) == n_ct
    assert "p_wilcoxon" in res.columns
    assert "padj_bh" in res.columns
    assert "mean_resilient" in res.columns
    assert "mean_vulnerable" in res.columns
    assert "cell_type" in res.columns


def test_binned_subgroup_dge_wilcoxon_per_pair():
    rng = np.random.default_rng(0)
    n_subj = 100
    n_ct = 3
    n_gene = 5
    pseudobulk = rng.normal(0, 1, (n_subj, n_ct, n_gene))
    resilient = np.arange(0, 25)
    vulnerable = np.arange(75, 100)
    res = binned_subgroup_dge_wilcoxon(
        pseudobulk,
        resilient_idx=resilient, vulnerable_idx=vulnerable,
        ct_names=[f"CT{i}" for i in range(n_ct)],
        gene_names=[f"G{j}" for j in range(n_gene)],
    )
    assert len(res) == n_ct * n_gene
    assert "padj_bh" in res.columns


def test_binned_subgroup_dge_signal_recovers():
    """If we plant a real shift in one (CT, gene), DGE should rank it high."""
    rng = np.random.default_rng(0)
    n_subj = 100
    n_ct = 3
    n_gene = 5
    pseudobulk = rng.normal(0, 1, (n_subj, n_ct, n_gene))
    # Plant: CT=0, gene=2 has +5 shift in resilient subjects
    pseudobulk[0:25, 0, 2] += 5.0
    resilient = np.arange(0, 25)
    vulnerable = np.arange(75, 100)
    res = binned_subgroup_dge_wilcoxon(
        pseudobulk,
        resilient_idx=resilient, vulnerable_idx=vulnerable,
        ct_names=[f"CT{i}" for i in range(n_ct)],
        gene_names=[f"G{j}" for j in range(n_gene)],
    )
    top = res.iloc[0]
    assert top["cell_type"] == "CT0"
    assert top["gene"] == "G2"
    assert top["padj_bh"] < 0.05
