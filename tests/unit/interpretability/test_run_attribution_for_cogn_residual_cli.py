"""CLI-flag smoke tests for the cogn-residual attribution orchestrator."""
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = _ROOT / "scripts/resdec_mhe/cogn_residual/run_attribution_for_cogn_residual.py"


def _help() -> str:
    res = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True, text=True, env={"PYTHONPATH": str(_ROOT), "PATH": ""},
    )
    return res.stdout + res.stderr


def test_attribution_orchestrator_exposes_pred_root_name_flag():
    assert "--pred-root-name" in _help()


def test_attribution_orchestrator_exposes_base_cache_name_flag():
    assert "--base-cache-name" in _help()


def test_attribution_orchestrator_exposes_interp_out_name_flag():
    assert "--interp-out-name" in _help()


def test_attribution_orchestrator_exposes_variant_config_flag():
    assert "--variant-config" in _help()


def test_attribution_orchestrator_help_text_mentions_base_cache_swap_path():
    """Help text exposes the cache-name flag as the base-swap mechanism."""
    h = _help()
    assert "Residual-base cache" in h or "tabpfn_cache" in h
