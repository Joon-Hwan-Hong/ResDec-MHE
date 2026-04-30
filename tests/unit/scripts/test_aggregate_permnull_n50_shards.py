"""Smoke tests for aggregate_permnull_n50_shards.

Catches CC1 schema drift (the existing canonical N=10 perm summary's field
names must match what the aggregator reads) and verifies the canonical-R²
loader points at a real file.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


WT = Path(__file__).resolve().parents[3]
SCRIPT = WT / "scripts" / "resdec_mhe" / "training" / "aggregate_permnull_n50_shards.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("aggregate_permnull_n50_shards", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["aggregate_permnull_n50_shards"] = module
    spec.loader.exec_module(module)
    return module


def test_wt_resolves_to_worktree(mod):
    assert mod.WT == WT


def test_canonical_permnull_path_exists(mod):
    """The canonical N=10 perm summary that we read CANONICAL_R2 from must exist."""
    assert mod.CANONICAL_PERMNULL_PATH.exists(), (
        f"Expected canonical N=10 perm summary at {mod.CANONICAL_PERMNULL_PATH}; "
        "the N=50 aggregator depends on it for canonical R²."
    )


def test_canonical_r2_loads_finite_value(mod):
    r2 = mod._load_canonical_r2()
    assert isinstance(r2, float)
    assert 0.0 < r2 < 1.0, f"canonical R² out of plausible range: {r2}"


def test_canonical_n10_schema_has_required_fields(mod):
    """Guards against schema drift: the N=10 file must have the field names
    we read (catches future renames)."""
    summary = json.loads(mod.CANONICAL_PERMNULL_PATH.read_text())
    required = {
        "canonical_mean_r2",
        "null_mean",
        "null_std",
        "z_under_null",
        "p_value_one_sided",
        "n_perms_ge_canonical",
    }
    missing = required - set(summary.keys())
    assert not missing, f"N=10 perm summary missing fields: {missing}"


def test_main_handles_missing_shards(mod, tmp_path, monkeypatch):
    """If shard files don't exist, main() should print a message and return — no crash."""
    monkeypatch.setattr(mod, "ROOT", tmp_path / "nonexistent")
    mod.main()
