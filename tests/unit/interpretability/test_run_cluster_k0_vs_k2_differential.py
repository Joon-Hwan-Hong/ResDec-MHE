"""Smoke + unit tests for run_cluster_k0_vs_k2_differential.py."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.conftest import WORKTREE_ROOT as _WORKTREE_ROOT

# Import the module under test
import importlib.util
_SCRIPT = (
    _WORKTREE_ROOT
    / "scripts/resdec_mhe/interpretability/run_cluster_k0_vs_k2_differential.py"
)
_spec = importlib.util.spec_from_file_location("run_cluster_k0_vs_k2_differential", _SCRIPT)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)  # type: ignore

# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------
def test_fit_k4_clusters_reproduces_canonical_sizes():
    """GMM(k=4, random_state=0) on canonical residuals must yield (55, 262, 60, 139)."""
    residual_csv = (
        _WORKTREE_ROOT
        / "outputs/canonical/interpretability/residual_per_subject.csv"
    )
    if not residual_csv.is_file():
        pytest.skip("residual_per_subject.csv missing")
    df = pd.read_csv(residual_csv)
    df = df.loc[np.isfinite(df["residual"].to_numpy())].reset_index(drop=True)
    labels, means = mod.fit_k4_clusters(df["residual"].to_numpy(), random_state=0)
    sizes = np.bincount(labels, minlength=4).tolist()
    assert sizes == [55, 262, 60, 139], sizes
    # Cluster 0 should be the most-negative (vulnerable) center
    assert means[0] < -1.0, means
    # Cluster 2 should be the most-positive
    assert means[2] > 0.5, means

def test_wilcoxon_two_groups_basic():
    a = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    b = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
    u, p, log2_fc = mod.wilcoxon_two_groups(a, b)
    assert np.isfinite(u)
    assert np.isfinite(p)
    assert p < 0.05
    # FC: |0|+eps vs |5|+eps → log2 ≈ -log2(5e9) ≈ -32 (eps in numerator)
    assert log2_fc < 0

def test_wilcoxon_two_groups_too_few():
    a = np.array([1.0, 2.0])  # < 3
    b = np.array([3.0, 4.0, 5.0, 6.0])
    u, p, fc = mod.wilcoxon_two_groups(a, b)
    assert not np.isfinite(p)
    assert not np.isfinite(u)

def test_bh_correct_basic():
    p_arr = np.array([0.001, 0.002, 0.05, 0.1, 0.5])
    q = mod.bh_correct(p_arr)
    assert q.shape == p_arr.shape
    assert np.all(q >= p_arr - 1e-9)  # BH never reduces below raw p
    assert np.all(np.isfinite(q))

def test_bh_correct_with_nan():
    p_arr = np.array([0.001, np.nan, 0.05, 0.1, 0.5])
    q = mod.bh_correct(p_arr)
    assert not np.isfinite(q[1])
    assert np.all(np.isfinite(q[[0, 2, 3, 4]]))

def test_cell_count_differential_synthetic():
    """Construct counts where CT 0 is highly different and CT 1 is the same."""
    n_subj = 20
    is_k0 = np.array([True] * 10 + [False] * 10)
    is_k2 = ~is_k0
    cell_type_order = ["A", "B"]
    counts = np.zeros((n_subj, 2), dtype=np.int64)
    counts[is_k0, 0] = np.arange(10) + 50
    counts[is_k2, 0] = np.arange(10) + 5
    counts[:, 1] = 100  # identical
    df = mod.cell_count_differential(counts, is_k0, is_k2, cell_type_order)
    assert len(df) == 2
    assert df.loc[df["cell_type"] == "A", "sig_q05"].iloc[0]
    # Identical group → p≈1 (or NaN if Wilcoxon ties trigger), should not be sig
    p_b = df.loc[df["cell_type"] == "B", "p_value"].iloc[0]
    assert (not np.isfinite(p_b)) or p_b > 0.05

def test_attribution_differential_synthetic():
    rng = np.random.default_rng(0)
    n_subj, n_ct, n_gene = 20, 2, 5
    attr = rng.normal(size=(n_subj, n_ct, n_gene))
    is_k0 = np.array([True] * 10 + [False] * 10)
    is_k2 = ~is_k0
    # Inject a strong shift in one (CT, gene) pair
    attr[is_k0, 0, 2] += 5.0
    df = mod.attribution_differential(
        attr, is_k0, is_k2, ["A", "B"], ["g0", "g1", "g2", "g3", "g4"],
    )
    assert len(df) == n_ct * n_gene
    target = df.query("cell_type == 'A' and gene == 'g2'")
    assert len(target) == 1
    assert target["sig_q05"].iloc[0]

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
def test_script_runs_end_to_end(tmp_path):
    residual_csv = (
        _WORKTREE_ROOT
        / "outputs/canonical/interpretability/residual_per_subject.csv"
    )
    metadata_csv = _WORKTREE_ROOT / "data/metadata_ROSMAP/metadata.csv"
    precomputed = _WORKTREE_ROOT / "data/precomputed"
    if not residual_csv.is_file() or not metadata_csv.is_file() or not precomputed.is_dir():
        pytest.skip("required canonical data not present")

    out_json = tmp_path / "cluster.json"
    out_md = tmp_path / "cluster.md"
    out_fig = tmp_path / "fig"
    cmd = [
        sys.executable, str(_SCRIPT),
        "--residual-csv", str(residual_csv),
        "--metadata-csv", str(metadata_csv),
        "--precomputed-dir", str(precomputed),
        "--out-json", str(out_json),
        "--out-md", str(out_md),
        "--out-fig-dir", str(out_fig),
        "--top-hits-confound-n", "10",
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(_WORKTREE_ROOT),
        timeout=600,
    )
    assert result.returncode == 0, (
        f"Script failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert out_json.is_file() and out_json.stat().st_size > 100
    assert out_md.is_file() and out_md.stat().st_size > 100
    png = out_fig / "fig_cluster_k0_vs_k2.png"
    pdf = out_fig / "fig_cluster_k0_vs_k2.pdf"
    assert png.is_file() and png.stat().st_size > 1000
    assert pdf.is_file() and pdf.stat().st_size > 1000

    payload = json.loads(out_json.read_text())
    assert payload["cohort"]["n_k0"] == 55
    assert payload["cohort"]["n_k2"] == 60
    assert "cell_type_abundance" in payload
    assert "gene_pseudobulk" in payload
    assert "pathology_confound" in payload
    # 31 CTs always tested for abundance
    assert payload["cell_type_abundance"]["n_tested"] == 31
