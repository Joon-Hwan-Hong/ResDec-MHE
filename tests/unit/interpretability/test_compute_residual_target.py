"""Smoke test for scripts/resdec_mhe/variants/compute_residual_target.py."""
import json
import sys
import subprocess
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _ROOT / "scripts/resdec_mhe/variants/compute_residual_target.py"


@pytest.mark.slow
def test_compute_residual_target_writes_npz_per_fold(tmp_path):
    out_dir = tmp_path / "gpath_only_cache"
    cmd = [
        sys.executable, str(_SCRIPT),
        "--variant-name", "gpath_only",
        "--axes", "gpath",
        "--metadata-path", str(_ROOT / "data/metadata_ROSMAP"),
        "--splits-path", str(_ROOT / "outputs/splits.json"),
        "--out-dir", str(out_dir),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, f"failed: {res.stderr}"

    for fold in range(5):
        npz = out_dir / f"residual_target_fold{fold}.npz"
        assert npz.is_file()
        d = np.load(npz, allow_pickle=True)
        assert int(d["fold"]) == fold
        assert d["target"].shape[0] == 516
        # Schema downstream (Task 4 datamodule) consumes:
        assert d["subject_ids"].shape[0] == 516
        assert d["subject_ids"].dtype == object
        assert "alpha" in d.files
        assert "beta_gpath" in d.files

    summary_json = out_dir / "summary.json"
    assert summary_json.is_file()
    summary = json.loads(summary_json.read_text())
    assert summary["variant_name"] == "gpath_only"
    assert summary["axes"] == ["gpath"]
    assert len(summary["per_fold"]) == 5
    assert "beta_gpath_mean" in summary["aggregate"]
    assert "beta_gpath_std" in summary["aggregate"]
    assert "axes" in summary["per_fold"][0]
    assert "n_val" in summary["per_fold"][0]
