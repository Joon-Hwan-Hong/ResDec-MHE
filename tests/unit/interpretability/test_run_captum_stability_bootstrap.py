"""Tests for run_captum_stability_bootstrap.py.

Unit tests for the statistical primitives (``canonical_top_k_per_ct``,
``bootstrap_inclusion_and_ranks``, ``median_iqr``, ``build_payload``,
``render_md``) plus a small end-to-end smoke run on a synthetic
``[N=20, C=3, G=10]`` mini-tensor saved as a temp NPZ.

We deliberately do NOT exercise the canonical 516-subject NPZ here — the full
1000-bootstrap end-to-end is the orchestrator's job, not a unit test.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import WORKTREE_ROOT as _WORKTREE_ROOT

from scripts.resdec_mhe.interpretability import (  # noqa: E402
    run_captum_stability_bootstrap as mod,
)

# =============================================================================
# canonical_top_k_per_ct
# =============================================================================

class TestCanonicalTopKPerCT:
    """``canonical_top_k_per_ct`` returns descending-ranked argsort + values."""

    def test_simple_known_ranking(self) -> None:
        # 4 subjects × 2 CTs × 5 genes; we craft per-CT mean-|.|.
        # CT 0 mean over subjects (already abs): [1, 5, 2, 4, 3]
        # CT 1 mean over subjects:                [9, 8, 7, 6, 5]
        n, c, g = 4, 2, 5
        abs_attr = np.zeros((n, c, g), dtype=np.float32)
        abs_attr[:, 0, :] = np.array([1, 5, 2, 4, 3])
        abs_attr[:, 1, :] = np.array([9, 8, 7, 6, 5])
        idx, imp = mod.canonical_top_k_per_ct(abs_attr, top_k=3)
        assert idx.shape == (2, 3)
        assert imp.shape == (2, 3)
        # CT 0 top-3 descending: gene 1 (5), gene 3 (4), gene 4 (3)
        np.testing.assert_array_equal(idx[0], np.array([1, 3, 4]))
        np.testing.assert_allclose(imp[0], [5, 4, 3])
        # CT 1 top-3 descending: gene 0, gene 1, gene 2
        np.testing.assert_array_equal(idx[1], np.array([0, 1, 2]))
        np.testing.assert_allclose(imp[1], [9, 8, 7])

    def test_takes_mean_over_subjects(self) -> None:
        # Verify it AVERAGES (not sums) along axis 0.
        abs_attr = np.zeros((10, 1, 3), dtype=np.float32)
        abs_attr[:, 0, :] = [2.0, 4.0, 6.0]
        _, imp = mod.canonical_top_k_per_ct(abs_attr, top_k=3)
        np.testing.assert_allclose(imp[0], [6.0, 4.0, 2.0])

    def test_rejects_non_3d(self) -> None:
        with pytest.raises(ValueError, match="3D"):
            mod.canonical_top_k_per_ct(np.zeros((4, 5)), top_k=2)

    def test_rejects_invalid_k(self) -> None:
        abs_attr = np.zeros((2, 1, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="K"):
            mod.canonical_top_k_per_ct(abs_attr, top_k=0)
        with pytest.raises(ValueError, match="K"):
            mod.canonical_top_k_per_ct(abs_attr, top_k=4)

# =============================================================================
# bootstrap_inclusion_and_ranks
# =============================================================================

class TestBootstrapInclusionAndRanks:
    """``bootstrap_inclusion_and_ranks`` returns inclusion and ranks."""

    def test_perfectly_stable_signal_yields_inclusion_one(self) -> None:
        # Per-CT importance is constant across subjects: every bootstrap of
        # size N draws from the same 4-row matrix → mean is identical →
        # top-K ranking is identical → inclusion = 1.0 for all canonical genes.
        n, c, g, k = 4, 2, 5, 3
        abs_attr = np.zeros((n, c, g), dtype=np.float32)
        abs_attr[:, 0, :] = [1.0, 5.0, 2.0, 4.0, 3.0]
        abs_attr[:, 1, :] = [9.0, 8.0, 7.0, 6.0, 5.0]
        canon, _ = mod.canonical_top_k_per_ct(abs_attr, top_k=k)
        rng = np.random.default_rng(0)
        incl, ranks = mod.bootstrap_inclusion_and_ranks(
            abs_attr, canon, n_boot=20, top_k=k, rng=rng,
        )
        assert incl.shape == (c, k)
        np.testing.assert_allclose(incl, 1.0)
        # Bootstrap ranks should equal canonical rank in every iteration.
        for c_i in range(c):
            for r in range(k):
                assert len(ranks[c_i][r]) == 20
                assert all(rb == r for rb in ranks[c_i][r])

    def test_swap_signal_lowers_inclusion(self) -> None:
        # Two-subject regime: subject 0 ranks gene 0 high; subject 1 ranks
        # gene 4 high.  Cohort canonical top-2 picks one of each.  Bootstrap
        # samples that ALWAYS pick subject 0 will rank gene 0 first, etc.
        n, c, g, k = 2, 1, 5, 2
        abs_attr = np.zeros((n, c, g), dtype=np.float32)
        abs_attr[0, 0, :] = [10.0, 0.0, 0.0, 0.0, 0.0]
        abs_attr[1, 0, :] = [0.0, 0.0, 0.0, 0.0, 10.0]
        canon, _ = mod.canonical_top_k_per_ct(abs_attr, top_k=k)
        # Canonical mean = [5, 0, 0, 0, 5] — top-2 is genes {0, 4}.
        assert set(canon[0].tolist()) == {0, 4}
        rng = np.random.default_rng(123)
        incl, _ = mod.bootstrap_inclusion_and_ranks(
            abs_attr, canon, n_boot=400, top_k=k, rng=rng,
        )
        # In the bootstrap of size 2 from {0, 1}: the {0,0} draw never sees
        # subject 1's signal → top-2 includes gene 0 but tied across the rest.
        # Likewise {1,1}.  Mixed draw {0,1} reproduces the canonical top-2.
        # Overall, both canonical genes are in the top-2 in ≥ 50% of boots.
        assert (incl >= 0.4).all()
        assert (incl <= 1.0).all()

    def test_seed_determinism(self) -> None:
        # Two runs with the same seed → identical inclusion matrices.
        n, c, g, k = 8, 2, 6, 2
        rng_data = np.random.default_rng(7)
        abs_attr = rng_data.random((n, c, g)).astype(np.float32)
        canon, _ = mod.canonical_top_k_per_ct(abs_attr, top_k=k)

        rng_a = np.random.default_rng(42)
        incl_a, _ = mod.bootstrap_inclusion_and_ranks(
            abs_attr, canon, n_boot=30, top_k=k, rng=rng_a,
        )
        rng_b = np.random.default_rng(42)
        incl_b, _ = mod.bootstrap_inclusion_and_ranks(
            abs_attr, canon, n_boot=30, top_k=k, rng=rng_b,
        )
        np.testing.assert_array_equal(incl_a, incl_b)

# =============================================================================
# median_iqr
# =============================================================================

class TestMedianIQR:
    """``median_iqr`` returns (median, Q1, Q3); NaN for empty."""

    def test_empty_returns_nan(self) -> None:
        m, q1, q3 = mod.median_iqr([])
        assert np.isnan(m) and np.isnan(q1) and np.isnan(q3)

    def test_simple_known_quartiles(self) -> None:
        m, q1, q3 = mod.median_iqr([0, 1, 2, 3, 4, 5, 6, 7, 8])
        assert m == pytest.approx(4.0)
        assert q1 == pytest.approx(2.0)
        assert q3 == pytest.approx(6.0)

# =============================================================================
# build_payload + render_md sanity
# =============================================================================

class TestBuildPayload:
    """``build_payload`` returns a serializable dict with summary + per-CT."""

    def test_payload_structure(self) -> None:
        # 2 CTs × top-3, with hand-set inclusion to exercise both branches.
        canon_idx = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
        canon_imp = np.array([[0.9, 0.8, 0.7], [0.6, 0.5, 0.4]],
                             dtype=np.float64)
        inclusion = np.array(
            [[1.0, 0.97, 0.4], [0.96, 0.6, 0.2]], dtype=np.float64,
        )
        boot_ranks = [
            [[0] * 100, [1] * 97, [0, 1, 2]],
            [[0] * 96, [1, 1, 2], [2]],
        ]
        ct_names = ["CT_A", "CT_B"]
        gene_names = ["g0", "g1", "g2", "g3", "g4", "g5"]
        payload = mod.build_payload(
            canonical_top_idx=canon_idx,
            canonical_top_imp=canon_imp,
            inclusion=inclusion,
            boot_ranks=boot_ranks,
            ct_names=ct_names,
            gene_names=gene_names,
            n_subjects=10,
            n_boot=100,
            top_k=3,
            seed=42,
            git="abcdef0",
        )
        assert payload["config"]["top_k"] == 3
        assert payload["config"]["n_boot"] == 100
        assert payload["summary"]["n_pairs_total"] == 6
        # rock-solid: 1.0 (CT_A r0), 0.97 (CT_A r1), 0.96 (CT_B r0) → 3
        assert payload["summary"]["n_rock_solid_pairs"] == 3
        # fragile: 0.4 (CT_A r2), 0.2 (CT_B r2) → 2
        assert payload["summary"]["n_fragile_pairs"] == 2
        # CT_A score = mean(1.0, 0.97, 0.4) ≈ 0.79; CT_B = mean(.96, .6, .2)
        assert payload["per_cell_type"][0]["stability_score"] == pytest.approx(
            np.mean([1.0, 0.97, 0.4])
        )
        assert payload["summary"]["most_stable_ct"]["cell_type"] == "CT_A"
        assert payload["summary"]["least_stable_ct"]["cell_type"] == "CT_B"

        # JSON-serializable round trip.
        s = json.dumps(payload)
        payload2 = json.loads(s)
        assert payload2["config"]["seed"] == 42

        # Top-10 lists.
        assert len(payload["top_10_most_stable_pairs"]) == 6  # only 6 total
        assert payload["top_10_most_stable_pairs"][0]["inclusion_frequency"] \
            == pytest.approx(1.0)
        assert len(payload["top_10_fragile_pairs"]) == 2

    def test_render_md_contains_key_strings(self) -> None:
        canon_idx = np.array([[0, 1]], dtype=np.int64)
        canon_imp = np.array([[0.9, 0.8]], dtype=np.float64)
        inclusion = np.array([[1.0, 0.3]], dtype=np.float64)
        boot_ranks = [[[0] * 100, [0, 1]]]
        payload = mod.build_payload(
            canonical_top_idx=canon_idx,
            canonical_top_imp=canon_imp,
            inclusion=inclusion,
            boot_ranks=boot_ranks,
            ct_names=["Splatter"],
            gene_names=["GENE_A", "GENE_B"],
            n_subjects=10,
            n_boot=100,
            top_k=2,
            seed=42,
            git="abcdef0",
        )
        md = mod.render_md(payload)
        assert "Captum IG Top-Gene Stability Bootstrap" in md
        assert "Mean per-CT stability score" in md
        assert "Splatter" in md
        assert "GENE_A" in md and "GENE_B" in md

# =============================================================================
# End-to-end smoke
# =============================================================================

class TestEndToEndSmoke:
    """Run the orchestrator on a synthetic mini-tensor."""

    def test_synthetic_run(self, tmp_path: Path) -> None:
        # Synthesize a (N, C, G) NPZ with C = len(CELL_TYPE_ORDER) = 31
        # so the schema-check passes.
        from src.data.constants import CELL_TYPE_ORDER

        n, c, g = 60, len(CELL_TYPE_ORDER), 60
        rng = np.random.default_rng(0)
        attr = rng.standard_normal((n, c, g)).astype(np.float32)
        npz_path = tmp_path / "syn.npz"
        np.savez(
            npz_path,
            subject_ids=np.array([f"R{i:06d}" for i in range(n)],
                                 dtype=object),
            attributions=attr,
            predictions_residual=np.zeros(n, dtype=np.float32),
            fold=np.zeros(n, dtype=np.int32),
        )
        gene_dir = tmp_path / "precomp"
        gene_dir.mkdir()
        np.save(
            gene_dir / "gene_names.npy",
            np.array([f"G{i:04d}" for i in range(g)], dtype=object),
        )

        out_json = tmp_path / "stability.json"
        out_md = tmp_path / "stability.md"
        out_fig = tmp_path / "fig"
        script = (_WORKTREE_ROOT
                  / "scripts/resdec_mhe/interpretability"
                  / "run_captum_stability_bootstrap.py")
        cmd = [
            sys.executable, str(script),
            "--attributions-npz", str(npz_path),
            "--precomputed-dir", str(gene_dir),
            "--out-json", str(out_json),
            "--out-md", str(out_md),
            "--out-fig-dir", str(out_fig),
            "--top-k", "10",
            "--n-boot", "30",
            "--seed", "42",
            "--focus-ct", CELL_TYPE_ORDER[0],
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=str(_WORKTREE_ROOT),
        )
        assert result.returncode == 0, (
            f"Script failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert out_json.is_file() and out_json.stat().st_size > 100
        assert out_md.is_file() and out_md.stat().st_size > 100
        assert (out_fig / "fig_captum_stability_bootstrap.png").is_file()
        assert (out_fig / "fig_captum_stability_bootstrap.pdf").is_file()
        payload = json.loads(out_json.read_text())
        assert payload["config"]["n_subjects"] == n
        assert payload["config"]["n_cell_types"] == c
        assert payload["config"]["top_k"] == 10
        assert payload["config"]["n_boot"] == 30
        assert len(payload["per_cell_type"]) == c
        assert all(
            len(entry["top_genes"]) == 10
            for entry in payload["per_cell_type"]
        )
        # Inclusion frequencies are in [0, 1] for every (CT, rank) pair.
        for entry in payload["per_cell_type"]:
            for g_rec in entry["top_genes"]:
                assert 0.0 <= g_rec["inclusion_frequency"] <= 1.0
                if g_rec["n_inclusions"] > 0:
                    assert (
                        g_rec["boot_rank_q1"]
                        <= g_rec["boot_rank_median"]
                        <= g_rec["boot_rank_q3"]
                    )
