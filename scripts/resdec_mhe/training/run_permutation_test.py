"""Full-pipeline permutation test orchestrator (negative-control sanity check).

Per permutation k (RNG seed = k):
  1. Generate a copy of the metadata CSV with the target column (cogn_global)
     randomly permuted across subjects.
  2. Re-run XGBoost top-k feature selection on the shuffled labels.
  3. Re-run TabPFN-2.6 OOF + outer caches on the shuffled labels.
  4. Train ResDec-MHE on the shuffled residual targets (5 folds parallel on
     2 GPUs), using the per-permutation TabPFN cache + shuffled metadata.
  5. Read each fold's val_predictions_best.npz (subject_ids + predictions)
     and recompute R² against the TRUE cogn_global from the original
     metadata.

Under the null (no signal in the shuffled labels), per-fold R² should be
near zero. Across N permutations, the empirical permutation p-value for
the observed canonical R² is (1 + #perms ≥ obs) / (N + 1).

Outputs:
  <output-base>/perm_<k>/{metadata,top_k,tabpfn,folds,perm.log}
  <output-base>/permutation_results.json   — aggregate distribution
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TARGET = "cogn_global"
DEFAULT_ID_COL = "ROSMAP_IndividualID"


def generate_shuffled_metadata(
    perm_seed: int,
    base_csv: Path,
    target_col: str,
    out_csv: Path,
) -> None:
    """Write metadata.csv with target_col randomly permuted (subject IDs preserved).

    NaN values STAY in their original rows; only the finite values are
    permuted among the non-NaN positions. Preserves the missingness
    pattern so cohort validation (which rejects NaN target subjects)
    doesn't fail randomly while still randomizing labels among observed
    subjects. (Bug fix: perm 4 hit a cohort subject with shuffled-in NaN.)
    """
    df = pd.read_csv(base_csv)
    rng = np.random.default_rng(perm_seed)
    df = df.copy()
    vals = df[target_col].values.astype(np.float64)
    finite_mask = np.isfinite(vals)
    permuted = vals.copy()
    permuted[finite_mask] = rng.permutation(vals[finite_mask])
    df[target_col] = permuted
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)


def run_step(cmd: list, log_handle, step_name: str) -> float:
    """Run cmd as subprocess; tee output to log_handle; return elapsed seconds.

    Note: env inherits parent's CUDA_VISIBLE_DEVICES — non-training stages
    (top-k, TabPFN) see whatever GPU mask the launcher set, which is correct
    under perm-shard parallelization.
    """
    log_handle.write(f"\n=== {step_name} === @ {time.strftime('%H:%M:%S')}\n")
    log_handle.write(f"$ {' '.join(str(c) for c in cmd)}\n")
    log_handle.flush()
    t0 = time.time()
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    try:
        subprocess.run(cmd, stdout=log_handle, stderr=subprocess.STDOUT, env=env, check=True)
    except subprocess.CalledProcessError as exc:
        elapsed = time.time() - t0
        log_handle.write(f"=== {step_name} FAILED in {elapsed:.1f}s, exit {exc.returncode} ===\n")
        log_handle.flush()
        raise RuntimeError(f"{step_name} failed (see log); cmd: {cmd}") from exc
    elapsed = time.time() - t0
    log_handle.write(f"=== {step_name} done in {elapsed:.1f}s ===\n")
    log_handle.flush()
    return elapsed


def run_train_folds_parallel(
    folds: list[int],
    gpus: list[int],
    base_args: list,
    log_handle,
) -> float:
    """Dispatch train.py per fold across GPUs in batches; return total elapsed."""
    t0 = time.time()
    idx = 0
    while idx < len(folds):
        procs = []
        for g in gpus:
            if idx >= len(folds):
                break
            f = folds[idx]
            cmd = ["uv", "run", "python", str(ROOT / "scripts/resdec_mhe/training/train.py"),
                   "--fold", str(f)] + base_args
            log_handle.write(
                f"\n=== train fold {f} on GPU {g} === @ {time.strftime('%H:%M:%S')}\n"
            )
            log_handle.flush()
            env = {**os.environ, "PYTHONPATH": str(ROOT), "CUDA_VISIBLE_DEVICES": str(g)}
            p = subprocess.Popen(
                cmd, stdout=log_handle, stderr=subprocess.STDOUT, env=env,
            )
            procs.append((p, f, g))
            idx += 1
        for p, f, g in procs:
            ret = p.wait()
            log_handle.write(f"=== train fold {f} GPU {g} done, exit {ret} ===\n")
            log_handle.flush()
            if ret != 0:
                raise RuntimeError(f"train fold {f} failed")
    return time.time() - t0


def run_one_permutation(
    perm_seed: int,
    output_base: Path,
    base_metadata_csv: Path,
    splits_path: Path,
    precomputed_dir: Path,
    target_col: str,
    id_col: str,
    gpus: list[int] | None = None,
) -> dict:
    perm_dir = output_base / f"perm_{perm_seed}"
    perm_dir.mkdir(parents=True, exist_ok=True)
    log_path = perm_dir / "perm.log"
    log = log_path.open("w")
    t_perm = time.time()

    try:
        # 1. Shuffled metadata
        meta_dir = perm_dir / "metadata"
        shuffled_csv = meta_dir / "metadata.csv"
        generate_shuffled_metadata(perm_seed, base_metadata_csv, target_col, shuffled_csv)
        log.write(f"shuffled metadata → {shuffled_csv}\n")

        # 2. Top-k features
        topk_dir = perm_dir / "top_k"
        run_step([
            "uv", "run", "python", str(ROOT / "scripts/resdec_mhe/tabpfn/compute_top_k_features.py"),
            "--splits-path", str(splits_path),
            "--precomputed-dir", str(precomputed_dir),
            "--metadata-csv", str(shuffled_csv),
            "--output-dir", str(topk_dir),
            "--top-k", "2000",
            "--feature-set", "A",
        ], log, "top-k features")

        # 3. TabPFN OOF
        tabpfn_dir = perm_dir / "tabpfn"
        run_step([
            "uv", "run", "python", str(ROOT / "scripts/resdec_mhe/tabpfn/compute_oof.py"),
            "--splits-path", str(splits_path),
            "--precomputed-dir", str(precomputed_dir),
            "--metadata-csv", str(shuffled_csv),
            "--top-k-dir", str(topk_dir),
            "--output-dir", str(tabpfn_dir),
            "--top-k", "2000",
        ], log, "tabpfn OOF")

        # 4. TabPFN outer
        run_step([
            "uv", "run", "python", str(ROOT / "scripts/resdec_mhe/tabpfn/compute_outer.py"),
            "--splits-path", str(splits_path),
            "--precomputed-dir", str(precomputed_dir),
            "--metadata-csv", str(shuffled_csv),
            "--top-k-dir", str(topk_dir),
            "--output-dir", str(tabpfn_dir),
            "--top-k", "2000",
            "--feature-set", "A",
        ], log, "tabpfn outer")

        # 5. ResDec-MHE training × 5 folds parallel on 2 GPUs
        folds_dir = perm_dir / "folds"
        folds_dir.mkdir(parents=True, exist_ok=True)
        train_args = [
            "--config", "configs/resdec_mhe/canonical.yaml",
            "--output-dir", str(folds_dir),
            "--metadata-path", str(meta_dir),
            "--tabpfn-oof-dir", str(tabpfn_dir),
            "--tabpfn-outer-dir", str(tabpfn_dir),
        ]
        gpus_to_use = gpus or [0, 1]  # treat None or empty list as default 2-GPU
        train_elapsed = run_train_folds_parallel(
            folds=[0, 1, 2, 3, 4], gpus=gpus_to_use, base_args=train_args, log_handle=log,
        )
        log.write(f"\n=== train all folds done in {train_elapsed:.1f}s ===\n")

        # 6. Eval against TRUE labels
        base_meta = pd.read_csv(base_metadata_csv)
        true_y_map = dict(zip(base_meta[id_col].astype(str), base_meta[target_col].astype(float)))

        # Use val_predictions_final.npz (last epoch's predictions). The
        # canonical pipeline's val_predictions_best.npz is only produced by
        # the separate reinfer_best_ckpt step; for the permutation negative
        # control we expect R²≈0 either way (model trained on shuffled labels
        # has no meaningful "best" — early-stopping picks the lowest-loss
        # epoch on shuffled val targets).
        per_fold_r2 = []
        per_fold_n = []
        for f in range(5):
            npz = np.load(folds_dir / f"fold{f}" / "val_predictions_final.npz", allow_pickle=True)
            subj_ids = [str(s) for s in npz["subject_ids"]]
            preds = np.asarray(npz["predictions"], dtype=np.float64)
            true_y = np.array([true_y_map[s] for s in subj_ids], dtype=np.float64)
            r2 = float(r2_score(true_y, preds))
            per_fold_r2.append(r2)
            per_fold_n.append(len(subj_ids))
            log.write(f"fold {f}: R² (vs TRUE labels) = {r2:+.4f}, n={len(subj_ids)}\n")

        elapsed_total = time.time() - t_perm
        log.write(f"\n=== perm {perm_seed} TOTAL: {elapsed_total / 60:.2f} min ===\n")
        log.write(f"per_fold_r2 (TRUE)= {per_fold_r2}\n")
        log.write(f"mean R² = {float(np.mean(per_fold_r2)):+.4f}\n")
        log.close()

        return {
            "perm_seed": perm_seed,
            "per_fold_r2_true": per_fold_r2,
            "per_fold_n": per_fold_n,
            "mean_r2_true": float(np.mean(per_fold_r2)),
            "elapsed_min": elapsed_total / 60,
        }
    except Exception as exc:
        log.write(f"\n!!! perm {perm_seed} FAILED: {exc}\n")
        log.close()
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--num-perms", type=int, default=1)
    p.add_argument("--start-perm", type=int, default=0)
    p.add_argument("--output-base", default="outputs/canonical/permutation_test")
    p.add_argument("--base-metadata-csv",
                   default="data/metadata_ROSMAP/metadata.csv")
    p.add_argument("--splits-path", default="outputs/splits.json")
    p.add_argument("--precomputed-dir", default="data/precomputed")
    p.add_argument("--target-col", default=DEFAULT_TARGET)
    p.add_argument("--id-col", default=DEFAULT_ID_COL)
    p.add_argument(
        "--gpus", default="0,1",
        help="comma-separated GPU indices to use within each perm (default: 0,1 = "
             "fold-shard within perm). Set to '0' or '1' for perm-shard mode where "
             "the wrapper launches two processes, one per GPU, with this arg set.",
    )
    args = p.parse_args()
    gpus_list = [int(g.strip()) for g in args.gpus.split(",") if g.strip()]
    if not gpus_list:
        raise SystemExit("--gpus must specify at least one GPU index (e.g. '0' or '0,1')")

    output_base = Path(args.output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    aggregate_path = output_base / "permutation_results.json"
    if aggregate_path.exists():
        results = json.loads(aggregate_path.read_text())
    else:
        results = []

    for k in range(args.start_perm, args.start_perm + args.num_perms):
        print(f"\n=== Permutation {k} starting @ {time.strftime('%H:%M:%S')} ===", flush=True)
        t0 = time.time()
        try:
            result = run_one_permutation(
                perm_seed=k,
                output_base=output_base,
                base_metadata_csv=Path(args.base_metadata_csv),
                splits_path=Path(args.splits_path),
                precomputed_dir=Path(args.precomputed_dir),
                target_col=args.target_col,
                id_col=args.id_col,
                gpus=gpus_list,
            )
        except Exception as exc:
            print(f"  perm {k} FAILED: {exc}", flush=True)
            results.append({"perm_seed": k, "error": str(exc), "elapsed_min": (time.time() - t0) / 60})
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


if __name__ == "__main__":
    main()
