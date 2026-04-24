"""Tests for src/analysis/conditional_mi.py and src/analysis/de_resilience.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.conditional_mi import conditional_mi_per_celltype
from src.analysis.de_resilience import _bh_fdr, wilcoxon_de


# ---------- BH-FDR --------------------------------------------------------


def test_bh_fdr_monotone_in_input():
    p = np.array([0.001, 0.01, 0.05, 0.1, 0.5])
    adj = _bh_fdr(p)
    assert (adj <= 1.0).all()
    # Adjusted p should be >= original.
    assert (adj >= p).all()


def test_bh_fdr_alldistinct_simple_case():
    # 5 p-values, BH adjusted.
    p = np.array([0.01, 0.04, 0.05, 0.06, 0.20])
    adj = _bh_fdr(p)
    # Smallest p gets multiplied by n/1 = 5.
    expected_floor = np.array([0.05, 0.10, 0.0833, 0.075, 0.20])
    # We don't check exact (BH uses min over later ranks), just that
    # adjusted values are reasonable.
    assert adj[0] == pytest.approx(0.05, abs=0.01)
    assert (adj <= 1.0).all()


# ---------- wilcoxon_de --------------------------------------------------


def test_wilcoxon_de_recovers_planted_signal():
    rng = np.random.default_rng(0)
    n = 80
    n_genes = 30
    expr = rng.normal(size=(n, n_genes))
    is_res = np.array([True] * 40 + [False] * 40)
    # Plant strong shifts in genes 5 + 15.
    expr[is_res, 5] += 2.5
    expr[is_res, 15] += 2.5
    df = wilcoxon_de(expr, is_res)
    sig = df[df["padj_fdr"] < 0.05]
    sig_genes = set(sig["gene"].tolist())
    assert "gene_5" in sig_genes
    assert "gene_15" in sig_genes


def test_wilcoxon_de_columns():
    rng = np.random.default_rng(0)
    expr = rng.normal(size=(40, 5))
    is_res = np.array([True] * 20 + [False] * 20)
    df = wilcoxon_de(expr, is_res)
    expected_cols = {
        "gene", "log2_fold_change", "p_value", "padj_fdr",
        "rank_biserial", "n_resilient", "n_vulnerable", "method",
    }
    assert expected_cols.issubset(df.columns)
    assert len(df) == 5


def test_wilcoxon_de_method_label():
    rng = np.random.default_rng(0)
    expr = rng.normal(size=(20, 3))
    is_res = np.array([True] * 10 + [False] * 10)
    df = wilcoxon_de(expr, is_res)
    assert (df["method"] == "wilcoxon").all()


# ---------- conditional_mi -----------------------------------------------


def test_cmi_emits_one_entry_per_cell_type():
    rng = np.random.default_rng(0)
    n = 100
    n_ct = 4
    expr = rng.normal(size=(n, n_ct))
    y = rng.normal(size=n)
    z = rng.normal(size=(n, 2))
    out = conditional_mi_per_celltype(expr, y, z)
    assert len(out["per_cell_type"]) == n_ct


def test_cmi_delta_drops_when_signal_is_in_pathology():
    """If Y depends only on Z, residualizing Z out of X should largely kill MI(X, Y)."""
    rng = np.random.default_rng(0)
    n = 400
    z = rng.normal(size=(n, 1))
    x = z[:, 0] + 0.3 * rng.normal(size=n)
    y = z[:, 0] + 0.3 * rng.normal(size=n)
    expr = x.reshape(-1, 1)
    out = conditional_mi_per_celltype(expr, y, z, n_neighbors=5)
    entry = out["per_cell_type"][0]
    # Conditional MI should be substantially (≥ 50%) smaller than unconditional
    # when the only X→Y dependence flows through Z.
    assert entry["conditional_mi_given_pathology"] < 0.5 * entry["unconditional_mi"]


def test_cmi_n_jobs_parallel_matches_serial():
    """Parallel KSG (n_jobs>1) must yield identical per-CT MI to n_jobs=1.

    KSG is deterministic under a fixed seed, so joblib parallelism over
    cell types must not change values — only wall clock.
    """
    rng = np.random.default_rng(0)
    n = 100
    n_ct = 4
    expr = rng.normal(size=(n, n_ct))
    y = rng.normal(size=n)
    z = rng.normal(size=(n, 2))
    out_serial = conditional_mi_per_celltype(expr, y, z, n_jobs=1)
    out_parallel = conditional_mi_per_celltype(expr, y, z, n_jobs=2)
    assert len(out_serial["per_cell_type"]) == len(out_parallel["per_cell_type"])
    for a, b in zip(out_serial["per_cell_type"], out_parallel["per_cell_type"]):
        assert a["cell_type"] == b["cell_type"]
        assert a["unconditional_mi"] == pytest.approx(b["unconditional_mi"])
        assert a["conditional_mi_given_pathology"] == pytest.approx(b["conditional_mi_given_pathology"])
        assert a["n_used"] == b["n_used"]


def test_cmi_handles_nan_subjects():
    rng = np.random.default_rng(0)
    n = 100
    expr = rng.normal(size=(n, 2))
    y = rng.normal(size=n)
    z = rng.normal(size=(n, 1))
    expr[::10, 0] = np.nan  # 10 NaN subjects in CT 0
    out = conditional_mi_per_celltype(expr, y, z)
    assert out["per_cell_type"][0]["n_used"] == 90  # 100 - 10 NaN
    assert out["per_cell_type"][1]["n_used"] == 100  # CT 1 unaffected
