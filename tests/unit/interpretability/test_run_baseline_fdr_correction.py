"""Unit + smoke tests for run_baseline_fdr_correction.py.

Covers:

1. ``bh_fdr`` BH-q monotonicity vs sorted p-values + agreement with
   ``statsmodels.stats.multitest.multipletests(method='fdr_bh')`` to
   machine precision (cross-implementation invariant).
2. ``bonferroni_threshold`` arithmetic (α/M).
3. ``correct_panel`` correctness on a hand-constructed 4-baseline panel
   with one "lost-to-FDR" entry.
4. End-to-end smoke: run the script as a subprocess on the canonical
   22-baseline input JSON, verify JSON + MD outputs are written and have
   the expected schema.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import WORKTREE_ROOT as _WORKTREE_ROOT

def _import_module():
    from scripts.resdec_mhe.interpretability import (  # noqa: E402
        run_baseline_fdr_correction as mod,
    )
    return mod

# ----------------------------------------------------------------------
# bh_fdr — invariants and cross-implementation agreement
# ----------------------------------------------------------------------

def test_bh_fdr_matches_statsmodels_machine_precision() -> None:
    """scipy and statsmodels BH must agree to machine precision."""
    mod = _import_module()
    rng = np.random.default_rng(seed=42)
    p = np.sort(rng.uniform(0.0, 1.0, size=50))
    q_scipy = mod.bh_fdr(p)

    # Reference: statsmodels.
    pytest.importorskip("statsmodels")
    from statsmodels.stats.multitest import multipletests

    _, q_sm, _, _ = multipletests(p, alpha=0.05, method="fdr_bh")
    np.testing.assert_allclose(q_scipy, q_sm, atol=1e-15)

def test_bh_fdr_clipped_to_one() -> None:
    """All q-values must be ≤ 1 even if M·p exceeds 1."""
    mod = _import_module()
    p = np.array([0.9, 0.95, 0.99, 0.999])
    q = mod.bh_fdr(p)
    assert np.all(q <= 1.0 + 1e-12)

def test_bh_fdr_rejects_invalid_input() -> None:
    """Out-of-[0,1] p-values must raise ValueError."""
    mod = _import_module()
    with pytest.raises(ValueError):
        mod.bh_fdr([0.5, 1.5])
    with pytest.raises(ValueError):
        mod.bh_fdr([-0.1, 0.5])
    with pytest.raises(ValueError):
        mod.bh_fdr(np.array([[0.1, 0.2], [0.3, 0.4]]))

# ----------------------------------------------------------------------
# bonferroni_threshold — arithmetic
# ----------------------------------------------------------------------

def test_bonferroni_threshold_basic() -> None:
    mod = _import_module()
    assert mod.bonferroni_threshold(0.05, 22) == pytest.approx(0.05 / 22, rel=1e-12)
    assert mod.bonferroni_threshold(0.01, 10) == pytest.approx(0.001, rel=1e-12)

def test_bonferroni_threshold_rejects_invalid() -> None:
    mod = _import_module()
    with pytest.raises(ValueError):
        mod.bonferroni_threshold(0.05, 0)
    with pytest.raises(ValueError):
        mod.bonferroni_threshold(0.05, -3)
    with pytest.raises(ValueError):
        mod.bonferroni_threshold(0.0, 10)
    with pytest.raises(ValueError):
        mod.bonferroni_threshold(1.5, 10)

# ----------------------------------------------------------------------
# correct_panel — synthetic 4-baseline panel with a lost-to-FDR entry
# ----------------------------------------------------------------------

def _make_synthetic_payload() -> dict:
    """4-baseline panel: 3 strongly-significant, 1 borderline.

    Stouffer p-values chosen so BH at α=0.05 keeps {b1,b2,b3,b4}
    when p4=0.04 (BH-q for p4 with M=4 is 4·0.04/4 = 0.04 < 0.05) and
    so that ALL pass. To exercise the lost-to-FDR branch we use a
    second panel below.
    """
    return {
        "seeds": [1, 2, 3, 4, 5],
        "per_baseline": {
            "Strong-A": {
                "stouffer_p_one_sided": 1.0e-6,
                "per_seed": {
                    str(s): {"wilcoxon_p_one_sided_greater": 0.03125}
                    for s in (1, 2, 3, 4, 5)
                },
            },
            "Strong-B": {
                "stouffer_p_one_sided": 1.0e-5,
                "per_seed": {
                    str(s): {"wilcoxon_p_one_sided_greater": 0.03125}
                    for s in (1, 2, 3, 4, 5)
                },
            },
            "Strong-C": {
                "stouffer_p_one_sided": 1.0e-4,
                "per_seed": {
                    str(s): {"wilcoxon_p_one_sided_greater": 0.03125}
                    for s in (1, 2, 3, 4, 5)
                },
            },
            "Borderline": {
                "stouffer_p_one_sided": 0.04,
                "per_seed": {
                    str(s): {"wilcoxon_p_one_sided_greater": 0.0625}
                    for s in (1, 2, 3, 4, 5)
                },
            },
        },
    }

def _make_lost_to_fdr_payload() -> dict:
    """Panel with one entry that is unadjusted-significant but lost to FDR.

    With M=4 and p = [1e-6, 0.05, 0.045, 0.04]:
        sorted p = [1e-6, 0.04, 0.045, 0.05]
        raw BH = M·p/k = [4e-6, 0.08, 0.06, 0.05]
        cummin from largest k = [4e-6, 0.05, 0.05, 0.05]

    So Borderline (p=0.04) ends up with q=0.05 (NOT < 0.05) — strict
    inequality means it is "lost to FDR" because raw p=0.04 < α=0.05
    but q=0.05 ≥ α=0.05.
    """
    return {
        "seeds": [1, 2, 3, 4, 5],
        "per_baseline": {
            "Strong": {
                "stouffer_p_one_sided": 1.0e-6,
                "per_seed": {
                    str(s): {"wilcoxon_p_one_sided_greater": 0.03125}
                    for s in (1, 2, 3, 4, 5)
                },
            },
            "Loss-A": {
                "stouffer_p_one_sided": 0.05,
                "per_seed": {
                    str(s): {"wilcoxon_p_one_sided_greater": 0.0625}
                    for s in (1, 2, 3, 4, 5)
                },
            },
            "Loss-B": {
                "stouffer_p_one_sided": 0.045,
                "per_seed": {
                    str(s): {"wilcoxon_p_one_sided_greater": 0.0625}
                    for s in (1, 2, 3, 4, 5)
                },
            },
            "Borderline": {
                "stouffer_p_one_sided": 0.04,
                "per_seed": {
                    str(s): {"wilcoxon_p_one_sided_greater": 0.0625}
                    for s in (1, 2, 3, 4, 5)
                },
            },
        },
    }

def test_correct_panel_synthetic_all_pass() -> None:
    """All-significant 4-baseline synthetic panel: BH passes all 4."""
    mod = _import_module()
    payload = _make_synthetic_payload()
    record = mod.correct_panel(payload, alpha=0.05)

    assert record["m_baselines"] == 4
    assert record["alpha"] == pytest.approx(0.05)
    assert record["bonferroni_threshold"] == pytest.approx(0.05 / 4, rel=1e-12)

    by_name = {r["baseline"]: r for r in record["per_baseline"]}
    # All four pass BH at α=0.05.
    for name in ("Strong-A", "Strong-B", "Strong-C", "Borderline"):
        assert by_name[name]["bh_q_value"] < 0.05, by_name[name]

    # Bonferroni at α/M = 0.0125 — only the three < 1.25e-2 pass.
    assert by_name["Strong-A"]["bonferroni_significant"] is True
    assert by_name["Strong-B"]["bonferroni_significant"] is True
    assert by_name["Strong-C"]["bonferroni_significant"] is True
    assert by_name["Borderline"]["bonferroni_significant"] is False

    # No lost-to-FDR entries in this panel.
    assert record["summary"]["n_lost_to_fdr"] == 0
    assert record["summary"]["n_bh_significant"] == 4
    assert record["summary"]["n_bonferroni_significant"] == 3

    # Per-seed Wilcoxon round-trip: 5 seeds preserved per baseline.
    for r in record["per_baseline"]:
        assert len(r["per_seed_wilcoxon_p_one_sided_greater"]) == 5

def test_correct_panel_lost_to_fdr_branch() -> None:
    """Constructed panel exercises the lost-to-FDR diagnostic.

    With p = [1e-6, 0.05, 0.045, 0.04] and α=0.05, BH gives
    q = [4e-6, 0.05, 0.05, 0.05]. Strict q < α excludes the three
    borderline entries.
    """
    mod = _import_module()
    payload = _make_lost_to_fdr_payload()
    record = mod.correct_panel(payload, alpha=0.05)

    by_name = {r["baseline"]: r for r in record["per_baseline"]}

    # Direct cross-check of the q-values (matches the docstring math).
    assert by_name["Strong"]["bh_q_value"] == pytest.approx(4e-6, rel=1e-9)
    for name in ("Loss-A", "Loss-B", "Borderline"):
        assert by_name[name]["bh_q_value"] == pytest.approx(0.05, rel=1e-9)

    # Strong is BH-significant (q < α). The other three are not.
    assert by_name["Strong"]["bh_q_value"] < 0.05
    for name in ("Loss-A", "Loss-B", "Borderline"):
        assert by_name[name]["bh_q_value"] >= 0.05

    # Loss-B and Borderline are "lost to FDR" (raw p < 0.05 but q ≥ 0.05).
    # Loss-A's raw p IS exactly 0.05, so raw p < α is False — not "lost".
    assert by_name["Borderline"]["lost_to_fdr"] is True
    assert by_name["Loss-B"]["lost_to_fdr"] is True
    assert by_name["Loss-A"]["lost_to_fdr"] is False
    assert by_name["Strong"]["lost_to_fdr"] is False

    assert record["summary"]["n_lost_to_fdr"] == 2
    assert record["summary"]["n_bh_significant"] == 1
    # Bonferroni at α/M = 0.0125 — only Strong (1e-6) passes.
    assert record["summary"]["n_bonferroni_significant"] == 1

# ----------------------------------------------------------------------
# End-to-end smoke test against the canonical 22-baseline JSON
# ----------------------------------------------------------------------

def test_run_baseline_fdr_correction_e2e(tmp_path: Path) -> None:
    """Run the script against the real 22-baseline JSON and validate outputs."""
    in_path = (
        _WORKTREE_ROOT
        / "outputs/canonical/interpretability/seed_variation_wilcoxon_all_baselines.json"
    )
    if not in_path.is_file():
        pytest.skip(
            "Canonical 22-baseline Stouffer JSON not present in this worktree."
        )

    out_dir = tmp_path / "out"
    script = (
        _WORKTREE_ROOT
        / "scripts/resdec_mhe/interpretability/run_baseline_fdr_correction.py"
    )
    cmd = [
        sys.executable,
        str(script),
        "--in-path",
        str(in_path),
        "--out-dir",
        str(out_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_WORKTREE_ROOT))
    assert result.returncode == 0, (
        f"Script failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )

    out_json = out_dir / "baseline_fdr_correction.json"
    out_md = out_dir / "baseline_fdr_correction.md"
    assert out_json.is_file() and out_json.stat().st_size > 1000
    assert out_md.is_file() and out_md.stat().st_size > 500

    record = json.loads(out_json.read_text())
    assert record["m_baselines"] == 22
    assert record["alpha"] == pytest.approx(0.05)
    assert record["bonferroni_threshold"] == pytest.approx(0.05 / 22, rel=1e-12)
    assert len(record["per_baseline"]) == 22

    # Every required field present on every row.
    required = {
        "baseline",
        "per_seed_wilcoxon_p_one_sided_greater",
        "stouffer_p_one_sided",
        "bh_q_value",
        "bonferroni_significant",
        "lost_to_fdr",
        "notes",
    }
    for row in record["per_baseline"]:
        assert required.issubset(row.keys()), set(row.keys())
        assert len(row["per_seed_wilcoxon_p_one_sided_greater"]) == 5
        assert 0.0 <= row["stouffer_p_one_sided"] <= 1.0
        assert 0.0 <= row["bh_q_value"] <= 1.0

    # Panel-level numeric sanity (matches the canonical 22-baseline run).
    summary = record["summary"]
    assert summary["n_bh_significant"] == 22
    assert summary["n_bonferroni_significant"] == 20
    assert summary["n_lost_to_fdr"] == 0
    assert summary["most_conservative_significant_baseline"] is not None

    # MD has the expected header row.
    md_text = out_md.read_text()
    assert "| Baseline | Stouffer p | BH-FDR q | Bonferroni-sig | Notes |" in md_text
    # The two XGBoost baselines are the ones that fail Bonferroni — sanity
    # check by verifying both XGBoost feature-set rows say 'no' in the MD.
    for line in md_text.splitlines():
        if line.startswith("| XGBoost [A] ") or line.startswith("| XGBoost [A+C+E] "):
            assert "| no |" in line, line

def test_format_p_helper() -> None:
    """``_format_p`` uses scientific notation below 1e-3, fixed-point above."""
    mod = _import_module()
    assert mod._format_p(1e-5) == "1.00e-05"
    assert mod._format_p(2.93e-5) == "2.93e-05"
    assert mod._format_p(0.05) == "0.0500"
    assert mod._format_p(0.5) == "0.5000"
    assert mod._format_p(float("nan")) == "—"
