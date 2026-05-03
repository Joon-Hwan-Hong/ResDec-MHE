"""Run interpretability suite on a trained variant.

Wraps existing canonical scripts, redirecting --canonical-dir / --pred-root /
--out-dir / --tabpfn-dir flags + variant-specific --metadata-path /
--precomputed-dir overrides for the sanitized configs/default.yaml.

USAGE
-----
uv run python scripts/resdec_mhe/variants/run_attribution_for_variant.py \\
    --variant-name gpath_only --device cuda:0

Add --full-suite to also run SAE 31-CT causal patching (1 GPU, ~1 min).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]


def _run(cmd: list, env: dict | None = None) -> None:
    print("RUN:", " ".join(str(c) for c in cmd), flush=True)
    res = subprocess.run(cmd, cwd=str(_ROOT), env=env)
    if res.returncode != 0:
        raise RuntimeError(f"failed (exit {res.returncode}): {' '.join(str(c) for c in cmd)}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--variant-name", required=True,
                   choices=["gpath_only", "multi_axis"])
    p.add_argument("--full-suite", action="store_true",
                   help="Also run SAE 31-CT causal patching.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--metadata-path", type=Path,
                   default=_ROOT / "data/metadata_ROSMAP")
    p.add_argument("--precomputed-dir", type=Path,
                   default=_ROOT / "data/precomputed")
    p.add_argument("--splits-path", type=Path,
                   default=_ROOT / "outputs/splits.json")
    args = p.parse_args()

    out_root = _ROOT / "outputs/canonical/variants" / args.variant_name
    canonical_dir = out_root / "p5_seed42"
    interp_out = out_root / "interpretability"
    interp_out.mkdir(parents=True, exist_ok=True)
    variant_config = _ROOT / "configs/resdec_mhe/variants" / f"{args.variant_name}.yaml"
    variant_tabpfn_dir = out_root / "tabpfn_cache"

    # 1. Captum IG composite attribution
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

    # 2. LOCO zero-out per CT
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

    if args.full_suite:
        # SAE 31-CT causal patching on variant (~1 min on 1 GPU)
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

    # NOTE: Wasserstein-1 (run_distributional_resilience.py wasserstein) needs
    # a per-subject residual CSV that does not exist for variants yet — produced
    # by a downstream aggregator that consumes Captum + ResDec predictions. CMI
    # (run_resilience_analyses.py latent_class) similarly chains downstream.
    # Invoke those manually once the prereq CSVs are produced.

    print(f"\nattribution suite done for variant {args.variant_name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
