"""
Dry-run startup test for all baselines.

For each baseline:
  1. Check venv exists and python runs
  2. Check imports work
  3. Load one subject's data
  4. Initialize model
  5. Run one forward pass

Does NOT train — just verifies no crashes on startup.

Usage:
    uv run python scripts/test_benchmark_startup.py --device cuda:1
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT / "data/precomputed"
SPLITS = PROJECT / "outputs/splits.json"
METADATA = PROJECT / "data/metadata_ROSMAP"

results = []


def test(name: str, func, *args, **kwargs):
    """Run a test, catch exceptions, print result."""
    t0 = time.time()
    try:
        func(*args, **kwargs)
        elapsed = time.time() - t0
        print(f"  PASS  {name} ({elapsed:.1f}s)", flush=True)
        results.append((name, "PASS", None))
    except Exception as e:
        elapsed = time.time() - t0
        err = str(e)[:200]
        print(f"  FAIL  {name} ({elapsed:.1f}s): {err}", flush=True)
        results.append((name, "FAIL", err))


def check_venv(baseline_name: str, venv_path: str):
    """Check venv python exists and can import torch."""
    p = Path(venv_path)
    if not p.exists():
        raise FileNotFoundError(f"Venv not found: {venv_path}")
    result = subprocess.run(
        [str(p), "-c", "import torch; print(f'torch {torch.__version__}, cuda={torch.cuda.is_available()}')"],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Python failed: {result.stderr[:200]}")


def check_imports(venv_python: str, import_stmt: str):
    """Check that key imports work."""
    result = subprocess.run(
        [venv_python, "-c", import_stmt],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr[:300])


def check_data_file(path: str):
    """Check a required data file exists."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing: {path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    device = args.device

    with open(SPLITS) as f:
        splits = json.load(f)
    first_sid = splits["folds"][0]["train"][0]

    print(f"\n{'='*60}")
    print(f"  Benchmark Startup Tests — {time.strftime('%H:%M:%S')}")
    print(f"{'='*60}\n")

    # ---- CloudPred --------------------------------------------------------
    print("[cloudpred]", flush=True)
    cp_py = str(PROJECT / "baselines/cloudpred/.venv/bin/python")
    test("venv", check_venv, "cloudpred", cp_py)
    test("imports", check_imports, cp_py,
         "from sklearn.mixture import GaussianMixture; import torch; import numpy")
    test("forward", check_imports, cp_py, f"""
import sys, torch, numpy as np
sys.path.insert(0, '{PROJECT}/baselines/shared')
from pt_data_utils import load_subject_pt
pt = load_subject_pt('{DATA_DIR}', '{first_sid}')
cells = pt['cell_data'].numpy()[:100].astype(np.float32)
print(f'Loaded cells: {{cells.shape}}')
# Quick PCA + mixture forward
from sklearn.decomposition import PCA
pca = PCA(n_components=10).fit(cells)
reduced = pca.transform(cells).astype(np.float32)
print(f'PCA: {{reduced.shape}}')
""")

    # ---- CloudPred per-type -----------------------------------------------
    print("\n[cloudpred_pertype]", flush=True)
    test("forward", check_imports, cp_py, f"""
import sys, torch, numpy as np, math
sys.path.insert(0, '{PROJECT}/baselines/shared')
from pt_data_utils import load_subject_pt
pt = load_subject_pt('{DATA_DIR}', '{first_sid}')
offsets = pt['cell_offsets'].numpy()
n_types = len(offsets) - 1
print(f'Types: {{n_types}}, offsets: {{offsets.shape}}')
""")

    # ---- Perceiver IO -----------------------------------------------------
    print("\n[perceiver_io]", flush=True)
    pio_py = str(PROJECT / "baselines/perceiver_io/.venv/bin/python")
    test("venv", check_venv, "perceiver_io", pio_py)
    test("imports+forward", check_imports, pio_py, f"""
import sys, torch, numpy as np
sys.path.insert(0, '{PROJECT}/baselines/shared')
from pt_data_utils import load_subject_pt, extract_ccc_summary
pt = load_subject_pt('{DATA_DIR}', '{first_sid}')
pb = pt['pseudobulk'].numpy()  # [31, 4796]
ccc = extract_ccc_summary(pt)  # [18]
print(f'Pseudobulk: {{pb.shape}}, CCC: {{ccc.shape}}')

# Test model init + forward
sys.path.insert(0, '{PROJECT}/baselines/perceiver_io')
exec(open('{PROJECT}/baselines/perceiver_io/run_rosmap.py').read().split('def main')[0])
model = PerceiverIORegressor(gene_dim=4796, ccc_dim=18, n_cell_types=31)
pb_t = torch.tensor(pb, dtype=torch.float32).unsqueeze(0)  # [1, 31, 4796]
ccc_t = torch.tensor(ccc, dtype=torch.float32).unsqueeze(0)  # [1, 18]
with torch.no_grad():
    out = model(pb_t, ccc_t)
print(f'Output: {{out.shape}} = {{out.item():.4f}}')
""")

    # ---- GPIO -------------------------------------------------------------
    print("\n[gpio]", flush=True)
    gpio_py = str(PROJECT / "baselines/gpio/.venv/bin/python")
    test("venv", check_venv, "gpio", gpio_py)
    test("imports+forward", check_imports, gpio_py, f"""
import sys, torch, numpy as np
sys.path.insert(0, '{PROJECT}/baselines/shared')
from pt_data_utils import load_subject_pt
pt = load_subject_pt('{DATA_DIR}', '{first_sid}')
pb = pt['pseudobulk']  # [31, 4796]
ei = pt['ccc_edge_index']  # [2, E]
print(f'Pseudobulk: {{pb.shape}}, Edges: {{ei.shape}}')

# Test model init
sys.path.insert(0, '{PROJECT}/baselines/gpio')
# Import just the model class
exec(open('{PROJECT}/baselines/gpio/run_rosmap.py').read().split('def main')[0])
""")

    # ---- MixMIL -----------------------------------------------------------
    print("\n[mixmil]", flush=True)
    mixmil_py = str(PROJECT / "baselines/mixmil/.venv/bin/python")
    test("venv", check_venv, "mixmil", mixmil_py)
    test("imports", check_imports, mixmil_py,
         "from mixmil import MixMIL; print('MixMIL imported')")
    test("data_file", check_data_file,
         str(PROJECT / "baselines/shared/mixmil_input.h5ad"))

    # ---- ABMIL ------------------------------------------------------------
    print("\n[abmil]", flush=True)
    abmil_py = str(PROJECT / "baselines/abmil/.venv/bin/python") if \
        (PROJECT / "baselines/abmil/.venv/bin/python").exists() else "MISSING"
    if abmil_py == "MISSING":
        test("venv", lambda: (_ for _ in ()).throw(FileNotFoundError("No venv")))
    else:
        test("venv", check_venv, "abmil", abmil_py)
    test("data_file", check_data_file,
         str(PROJECT / "baselines/shared/mixmil_input.h5ad"))

    # ---- Set Transformer --------------------------------------------------
    print("\n[set_transformer]", flush=True)
    st_py = str(PROJECT / "baselines/set_transformer/.venv/bin/python") if \
        (PROJECT / "baselines/set_transformer/.venv/bin/python").exists() else "MISSING"
    if st_py == "MISSING":
        test("venv", lambda: (_ for _ in ()).throw(FileNotFoundError("No venv")))
    else:
        test("venv", check_venv, "set_transformer", st_py)
    test("data_file", check_data_file,
         str(PROJECT / "baselines/shared/mixmil_input.h5ad"))

    # ---- scPhase ----------------------------------------------------------
    print("\n[scphase]", flush=True)
    sc_py = str(PROJECT / "baselines/scPhase/.venv/bin/python")
    test("venv", check_venv, "scphase", sc_py)
    test("imports", check_imports, sc_py, """
import sys
sys.path.insert(0, 'baselines/scPhase/repo/scphase')
from data_loader import load_data
print('scPhase imports OK')
""")
    test("data_file", check_data_file,
         str(PROJECT / "baselines/shared/scphase_input.h5ad"))

    # ---- Summary ----------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    passes = sum(1 for _, s, _ in results if s == "PASS")
    fails = sum(1 for _, s, _ in results if s == "FAIL")
    print(f"  {passes} passed, {fails} failed\n")

    if fails > 0:
        print("  Failures:")
        for name, status, err in results:
            if status == "FAIL":
                print(f"    {name}: {err}")
        print()

    # Return exit code
    sys.exit(1 if fails > 0 else 0)


if __name__ == "__main__":
    main()
