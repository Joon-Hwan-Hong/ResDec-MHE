"""Smoke + import tests for scripts/resdec_mhe/cogn_residual/build_tabpfn_cache_cogn_residual.py.

The full single-fold smoke test is `slow` because it actually runs TabPFN-2.6.
The import test checks that the per-fold callables are properly exposed by
the refactor of compute_oof / compute_outer.
"""
import sys
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]


def test_per_fold_callables_have_expected_signature():
    """Refactor of compute_oof.py / compute_outer.py exposes per-fold functions
    with the kwarg set the variant builder relies on."""
    import inspect
    from scripts.resdec_mhe.tabpfn.compute_oof import process_oof_fold
    from scripts.resdec_mhe.tabpfn.compute_outer import process_outer_fold

    expected = {
        "fold_idx", "fold_split", "features", "targets",
        "args", "device", "output_dir", "top_k_dir",
    }
    assert set(inspect.signature(process_oof_fold).parameters) == expected
    assert set(inspect.signature(process_outer_fold).parameters) == expected


def test_build_tabpfn_cache_cogn_residual_help_works(tmp_path):
    """Smoke check that --help runs without import / argparse errors."""
    res = subprocess.run(
        [sys.executable,
         str(_ROOT
             / "scripts/resdec_mhe/cogn_residual/build_tabpfn_cache_cogn_residual.py"),
         "--help"],
        capture_output=True, text=True, timeout=60,
    )
    assert res.returncode == 0, res.stderr
    assert "--variant-name" in res.stdout
    assert "--residual-cache-dir" in res.stdout
    assert "--out-dir" in res.stdout


@pytest.mark.slow
def test_build_tabpfn_cache_cogn_residual_writes_oof_and_outer_one_fold(tmp_path):
    out_dir = tmp_path / "tabpfn_cache"
    cache_dir = _ROOT / "outputs/canonical/cogn_residual/gpath_only/cache"
    if not (cache_dir / "residual_target_fold0.npz").exists():
        pytest.skip("residual cache missing; Task 2 smoke run not done")
    cmd = [
        sys.executable,
        str(_ROOT
            / "scripts/resdec_mhe/cogn_residual/build_tabpfn_cache_cogn_residual.py"),
        "--variant-name", "gpath_only",
        "--residual-cache-dir", str(cache_dir),
        "--out-dir", str(out_dir),
        "--folds", "0",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    assert res.returncode == 0, f"failed: {res.stderr[-2000:]}"
    assert (out_dir / "tabpfn_oof_fold0.npz").is_file()
    assert (out_dir / "tabpfn_outer_fold0.npz").is_file()
