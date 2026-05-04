"""Variant cogn-residual perm null.

For each permutation seed, shuffles each fold's residualized cognition target
(NaN-preserving), rebuilds the variant TabPFN cache on shuffled per-fold targets,
trains a variant ResDec-MHE model 5-fold using a per-perm temp config that points
at the perm-shuffled caches, and recomputes R² against the TRUE per-fold
residualized target. Mirrors run_permutation_test.py for the variant pipeline:
  - Reuses canonical top-k features (no per-perm XGBoost re-selection)
  - Reuses build_tabpfn_cache_cogn_residual.py for per-perm TabPFN cache
  - Reuses run_5fold_parallel.sh through subprocess for the 5-fold training
  - Per-fold residual target shuffle (within finite mask) replaces the canonical
    metadata.csv cogn_global shuffle.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from omegaconf import OmegaConf
from sklearn.metrics import r2_score

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))


def _shuffle_fold_residual_target(
    base_npz_path: Path, perm_seed: int, fold: int, out_npz_path: Path,
) -> None:
    """Shuffle the finite values of the per-fold residual target NPZ, NaN-safe."""
    d = np.load(base_npz_path, allow_pickle=True)
    target = d["target"].astype(float)
    finite_mask = np.isfinite(target)
    rng = np.random.default_rng(perm_seed * 100 + fold)
    permuted = target.copy()
    permuted[finite_mask] = rng.permutation(target[finite_mask])

    out_npz_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {
        "fold": int(d["fold"]),
        "subject_ids": d["subject_ids"],
        "target": permuted.astype(np.float32),
        "alpha": float(d["alpha"]),
    }
    for k in d.files:
        if k.startswith("beta_"):
            save_kwargs[k] = float(d[k])
    np.savez(out_npz_path, **save_kwargs)


def _write_variant_perm_config(
    base_config_path: Path,
    perm_residual_cache_dir: Path,
    perm_tabpfn_cache_dir: Path,
    out_config_path: Path,
) -> None:
    """Write a temp variant YAML that points at per-perm shuffled caches."""
    cfg = OmegaConf.load(base_config_path)
    cfg.data.residualize_against.cache_dir = str(perm_residual_cache_dir)
    cfg.data.tabpfn_oof_dir = str(perm_tabpfn_cache_dir)
    cfg.data.tabpfn_outer_dir = str(perm_tabpfn_cache_dir)
    out_config_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_config_path)


def _run_step(cmd: list, log_handle, step_name: str) -> None:
    log_handle.write(f"\n--- {step_name} ---\n")
    log_handle.write(f"$ {' '.join(str(c) for c in cmd)}\n")
    log_handle.flush()
    res = subprocess.run(cmd, stdout=log_handle, stderr=subprocess.STDOUT)
    log_handle.flush()
    if res.returncode != 0:
        raise RuntimeError(f"{step_name} failed with exit {res.returncode}")


def run_one_perm(
    perm_seed: int,
    output_base: Path,
    base_residual_cache_dir: Path,
    base_variant_config: Path,
    metadata_path: Path,
    precomputed_dir: Path,
    splits_path: Path,
    gpus: list[int],
) -> dict:
    perm_dir = output_base / f"perm_{perm_seed}"
    perm_dir.mkdir(parents=True, exist_ok=True)
    log_path = perm_dir / "perm.log"
    log = log_path.open("w")
    t_perm = time.time()

    try:
        # 1. Shuffle per-fold residual targets
        perm_residual_cache_dir = perm_dir / "residual_cache"
        for f in range(5):
            _shuffle_fold_residual_target(
                base_residual_cache_dir / f"residual_target_fold{f}.npz",
                perm_seed=perm_seed, fold=f,
                out_npz_path=perm_residual_cache_dir / f"residual_target_fold{f}.npz",
            )
        log.write(f"shuffled per-fold residual targets -> {perm_residual_cache_dir}\n")

        # 2. Build variant TabPFN cache on shuffled per-fold targets
        perm_tabpfn_cache_dir = perm_dir / "tabpfn_cache"
        env_inherit = {**os.environ, "PYTHONPATH": str(_ROOT)}
        tabpfn_cmd = [
            "uv", "run", "python",
            str(_ROOT / "scripts/resdec_mhe/cogn_residual/build_tabpfn_cache_cogn_residual.py"),
            "--variant-name", f"perm_{perm_seed}",
            "--residual-cache-dir", str(perm_residual_cache_dir),
            "--out-dir", str(perm_tabpfn_cache_dir),
            "--folds", "0", "1", "2", "3", "4",
            "--precomputed-dir", str(precomputed_dir),
            "--metadata-csv", str(metadata_path / "metadata.csv"),
            "--splits-path", str(splits_path),
        ]
        # Per feedback_cuda_visible_devices_subprocess.md: when len(gpus)==1 the
        # parent shell has already pinned CUDA_VISIBLE_DEVICES to a physical GPU
        # (perm-shard mode); inherit that mask. Only override when len(gpus) > 1
        # (fold-shard within-perm mode where we'd assign each child individually).
        if len(gpus) > 1:
            tabpfn_env = {**env_inherit, "CUDA_VISIBLE_DEVICES": str(gpus[0])}
        else:
            tabpfn_env = env_inherit
        _run_step_env(tabpfn_cmd, log, "build TabPFN cache", tabpfn_env)

        # 3. Write per-perm variant config pointing at perm caches
        perm_config_path = perm_dir / "variant_perm.yaml"
        _write_variant_perm_config(
            base_variant_config,
            perm_residual_cache_dir, perm_tabpfn_cache_dir,
            perm_config_path,
        )
        log.write(f"wrote perm variant config -> {perm_config_path}\n")

        # 4. Train 5 folds via run_5fold_parallel.sh (in subshell to set TMUX-bypass)
        train_outroot = perm_dir / "training"
        train_outroot.mkdir(parents=True, exist_ok=True)
        # Bypass tmux preflight by setting TMUX in env (caller is in tmux already
        # for the wrapping shard launcher; this also handles the perm-internal
        # subshell which inherits TMUX from the parent tmux session).
        gpu_list_str = ",".join(str(g) for g in gpus)
        train_env = {
            **env_inherit,
            "CONFIG": str(perm_config_path),
            "OUTROOT": str(train_outroot),
            "SEED": "42",
            "GPU_LIST": gpu_list_str,
            "RUN_REINFER": "0",
            "METADATA_PATH": str(metadata_path),
            "PRECOMPUTED_DIR": str(precomputed_dir),
            # Keep TMUX from parent so the launcher's preflight passes.
            "TMUX": os.environ.get("TMUX", "perm_subshell"),
        }
        train_cmd = [
            "bash", str(_ROOT / "scripts/resdec_mhe/training/run_5fold_parallel.sh"),
        ]
        _run_step_env(train_cmd, log, "train 5 folds", train_env)

        # 5. Eval per-fold predictions against TRUE residual target
        per_fold_r2 = []
        per_fold_n = []
        for f in range(5):
            true_npz = np.load(
                base_residual_cache_dir / f"residual_target_fold{f}.npz",
                allow_pickle=True,
            )
            true_sids = [str(s) for s in true_npz["subject_ids"]]
            true_target = true_npz["target"].astype(np.float64)
            true_map = dict(zip(true_sids, true_target))

            preds_npz = np.load(
                train_outroot / f"fold{f}/val_predictions_final.npz",
                allow_pickle=True,
            )
            pred_sids = [str(s) for s in preds_npz["subject_ids"]]
            preds = np.asarray(preds_npz["predictions"], dtype=np.float64)
            true_y = np.array([true_map[s] for s in pred_sids], dtype=np.float64)

            mask = np.isfinite(true_y) & np.isfinite(preds)
            r2 = float(r2_score(true_y[mask], preds[mask]))
            per_fold_r2.append(r2)
            per_fold_n.append(int(mask.sum()))
            log.write(f"fold {f}: R² (vs TRUE residualized target) = {r2:+.4f}, n={int(mask.sum())}\n")

        elapsed = time.time() - t_perm
        log.write(f"\n=== perm {perm_seed} TOTAL: {elapsed / 60:.2f} min ===\n")
        log.write(f"per_fold_r2 = {per_fold_r2}\n")
        log.write(f"mean R² = {float(np.mean(per_fold_r2)):+.4f}\n")
        log.close()

        return {
            "perm_seed": perm_seed,
            "per_fold_r2_true": per_fold_r2,
            "per_fold_n": per_fold_n,
            "mean_r2_true": float(np.mean(per_fold_r2)),
            "elapsed_min": elapsed / 60,
        }
    except Exception as exc:
        log.write(f"\n!!! perm {perm_seed} FAILED: {exc}\n")
        log.close()
        raise


def _run_step_env(cmd: list, log_handle, step_name: str, env: dict) -> None:
    log_handle.write(f"\n--- {step_name} ---\n")
    log_handle.write(f"$ {' '.join(str(c) for c in cmd)}\n")
    log_handle.flush()
    res = subprocess.run(cmd, stdout=log_handle, stderr=subprocess.STDOUT, env=env)
    log_handle.flush()
    if res.returncode != 0:
        raise RuntimeError(f"{step_name} failed with exit {res.returncode}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--num-perms", type=int, default=1)
    p.add_argument("--start-perm", type=int, default=0)
    p.add_argument("--output-base", type=Path, required=True)
    p.add_argument("--variant-config", type=Path, required=True,
                   help="Base variant YAML (e.g. configs/resdec_mhe/cogn_residual/gpath_only.yaml).")
    p.add_argument("--residual-cache-dir", type=Path, required=True,
                   help="Original variant residual cache (residual_target_fold{0..4}.npz).")
    p.add_argument("--metadata-path", type=Path,
                   default=_ROOT / "data/metadata_ROSMAP")
    p.add_argument("--precomputed-dir", type=Path,
                   default=_ROOT / "data/precomputed")
    p.add_argument("--splits-path", type=Path,
                   default=_ROOT / "outputs/splits.json")
    p.add_argument("--gpus", default="0,1",
                   help="Comma-separated GPU indices to use within each perm. "
                        "Set to single GPU (e.g. '0' or '1') in perm-shard mode.")
    args = p.parse_args()

    gpus = [int(g.strip()) for g in args.gpus.split(",") if g.strip()]
    if not gpus:
        raise SystemExit("--gpus must specify at least one GPU index")

    args.output_base.mkdir(parents=True, exist_ok=True)
    aggregate_path = args.output_base / "permutation_results.json"
    if aggregate_path.exists():
        results = json.loads(aggregate_path.read_text())
    else:
        results = []
    successful_seeds = {
        r["perm_seed"] for r in results
        if "error" not in r and r.get("mean_r2_true") is not None
    }
    failed_records = [r for r in results if "error" in r]
    if failed_records:
        print(f"Resume mode: {len(successful_seeds)} prior successes; "
              f"dropping {len(failed_records)} stale failure records.")
        results = [r for r in results if "error" not in r]
        aggregate_path.write_text(json.dumps(results, indent=2))

    for k in range(args.start_perm, args.start_perm + args.num_perms):
        if k in successful_seeds:
            print(f"=== perm {k} already done; skipping ===", flush=True)
            continue
        print(f"\n=== perm {k} starting @ {time.strftime('%H:%M:%S')} ===", flush=True)
        t0 = time.time()
        try:
            result = run_one_perm(
                perm_seed=k,
                output_base=args.output_base,
                base_residual_cache_dir=args.residual_cache_dir,
                base_variant_config=args.variant_config,
                metadata_path=args.metadata_path,
                precomputed_dir=args.precomputed_dir,
                splits_path=args.splits_path,
                gpus=gpus,
            )
        except Exception as exc:
            print(f"  perm {k} FAILED: {exc}", flush=True)
            results.append({
                "perm_seed": k, "error": str(exc),
                "elapsed_min": (time.time() - t0) / 60,
            })
            aggregate_path.write_text(json.dumps(results, indent=2))
            continue

        results.append(result)
        aggregate_path.write_text(json.dumps(results, indent=2))
        print(
            f"  perm {k}: mean R²(TRUE) = {result['mean_r2_true']:+.4f}, "
            f"per-fold = {[f'{x:+.3f}' for x in result['per_fold_r2_true']]}, "
            f"took {result['elapsed_min']:.1f} min",
            flush=True,
        )

    print(f"\nwrote {aggregate_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
