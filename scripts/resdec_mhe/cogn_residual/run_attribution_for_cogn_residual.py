"""Full attribution suite for a trained variant — Captum IG + GradientSHAP +
SmoothGrad + 5 attention methods (AttnLRP / GMAR / GAF AF/GF/AGF) + LOCO
zero-out + per-subject CCC attention + (optional) SAE 31-CT causal patching.

Wraps existing canonical scripts, redirecting --canonical-dir / --pred-root /
--out-dir / --tabpfn-dir flags + injecting variant-specific --metadata-path /
--precomputed-dir overrides. Each script-fold takes ~1-30 min depending on
method; full run on Variant A is ~3-5 GPU-hr.

For "thin" variant runs (Variant B per plan), pass --thin to skip the 5
attention methods + GradientSHAP/SmoothGrad + CCC. Captum IG + LOCO + SAE 31-CT
remain (the four most cited canonical methods).

USAGE
-----
uv run python scripts/resdec_mhe/cogn_residual/run_attribution_for_cogn_residual.py \\
    --variant-name gpath_only --device cuda:0  # full suite

uv run python scripts/resdec_mhe/cogn_residual/run_attribution_for_cogn_residual.py \\
    --variant-name multi_axis --device cuda:1 --thin
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]


def _run(cmd: list, env: dict | None = None) -> None:
    print("RUN:", " ".join(str(c) for c in cmd), flush=True)
    # Force PYTHONPATH to worktree root so subprocess imports of src.* resolve
    # to the variant-aware modules in this worktree, not master's parent-repo
    # copies (see feedback_subprocess_pythonpath_leak.md / commit notes —
    # `python script.py` sets sys.path[0] to script-dir, leaving namespace-
    # package resolution to fall through to a sibling worktree/repo).
    full_env = {**os.environ, "PYTHONPATH": str(_ROOT)}
    if env is not None:
        full_env.update(env)
    res = subprocess.run(cmd, cwd=str(_ROOT), env=full_env)
    if res.returncode != 0:
        raise RuntimeError(f"failed (exit {res.returncode}): {' '.join(str(c) for c in cmd)}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--variant-name", required=True,
                   choices=["gpath_only", "multi_axis"])
    p.add_argument("--thin", action="store_true",
                   help="Skip GradientSHAP/SmoothGrad + 5 attention methods + CCC + SAE 31-CT (Variant B per plan).")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--metadata-path", type=Path,
                   default=_ROOT / "data/metadata_ROSMAP")
    p.add_argument("--precomputed-dir", type=Path,
                   default=_ROOT / "data/precomputed")
    p.add_argument("--splits-path", type=Path,
                   default=_ROOT / "outputs/splits.json")
    args = p.parse_args()

    out_root = _ROOT / "outputs/canonical/cogn_residual" / args.variant_name
    canonical_dir = out_root / "p5_seed42"
    interp_out = out_root / "interpretability"
    interp_out.mkdir(parents=True, exist_ok=True)
    variant_config = _ROOT / "configs/resdec_mhe/cogn_residual" / f"{args.variant_name}.yaml"
    variant_tabpfn_dir = out_root / "tabpfn_cache"

    # 1. Captum IG composite attribution (always)
    _run([
        sys.executable,
        str(_ROOT / "scripts/resdec_mhe/interpretability/captum_composite_attribution.py"),
        "--config", str(variant_config),
        "--pred-root", str(canonical_dir),
        "--splits-path", str(args.splits_path),
        "--out-dir", str(interp_out / "captum_ig"),
        "--metadata-path", str(args.metadata_path),
        "--precomputed-dir", str(args.precomputed_dir),
    ])

    # 2. LOCO zero-out per CT (always)
    _run([
        sys.executable,
        str(_ROOT / "scripts/resdec_mhe/interpretability/run_loco_zero_out.py"),
        "--config", str(variant_config),
        "--canonical-dir", str(canonical_dir),
        "--tabpfn-dir", str(variant_tabpfn_dir),
        "--splits-path", str(args.splits_path),
        "--out-dir", str(interp_out / "loco_zero_out"),
        "--device", args.device,
        "--metadata-path", str(args.metadata_path),
        "--precomputed-dir", str(args.precomputed_dir),
    ])

    if not args.thin:
        # 3. GradientSHAP + SmoothGrad (combined script)
        _run([
            sys.executable,
            str(_ROOT / "scripts/resdec_mhe/interpretability/gradient_shap_smoothgrad_attribution.py"),
            "--config", str(variant_config),
            "--pred-root", str(canonical_dir),
            "--splits-path", str(args.splits_path),
            "--out-dir", str(interp_out / "captum_robustness"),
            "--metadata-path", str(args.metadata_path),
            "--precomputed-dir", str(args.precomputed_dir),
        ])

        # 4. 5 attention methods (AttnLRP / GMAR / GAF AF / GAF GF / GAF AGF)
        _run([
            sys.executable,
            str(_ROOT / "scripts/resdec_mhe/interpretability/run_attention_attribution.py"),
            "--config", str(variant_config),
            "--pred-root", str(canonical_dir),
            "--splits-path", str(args.splits_path),
            "--out-dir", str(interp_out / "attention_attribution"),
            "--metadata-path", str(args.metadata_path),
            "--precomputed-dir", str(args.precomputed_dir),
        ])

        # 5. Per-subject CCC attention (CT-CT edges)
        _run([
            sys.executable,
            str(_ROOT / "scripts/resdec_mhe/interpretability/per_subject_ccc_attention.py"),
            "--config", str(variant_config),
            "--pred-root", str(canonical_dir),
            "--splits-path", str(args.splits_path),
            "--out-dir", str(interp_out / "ccc"),
            "--metadata-path", str(args.metadata_path),
            "--precomputed-dir", str(args.precomputed_dir),
        ])

        # 6. SAE 31-CT causal patching (~1 min on 1 GPU)
        _run([
            sys.executable,
            str(_ROOT / "scripts/resdec_mhe/interpretability/run_sae_causal_patching_31ct.py"),
            "--config", str(variant_config),
            "--canonical-dir", str(canonical_dir),
            "--out-dir", str(interp_out),
            "--device", args.device,
            "--metadata-path", str(args.metadata_path),
            "--precomputed-dir", str(args.precomputed_dir),
        ])

    print(f"\nattribution suite done for variant {args.variant_name} (thin={args.thin})", flush=True)
    print("\nNOTE: Wasserstein-1 (run_distributional_resilience.py wasserstein) and CMI "
          "(run_resilience_analyses.py latent_class) need a per-subject residual CSV "
          "produced downstream. Invoke those manually after this completes.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
