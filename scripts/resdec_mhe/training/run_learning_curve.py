"""Learning-curve orchestrator: ResDec-MHE R² vs training-set size N.

For each N ∈ ``--N-values`` (default 100, 200, 300, 400):
  1. Generate metadata.csv with training subjects subsampled to N per fold
     (validation set preserved unchanged for fair comparison).
  2. Re-run XGBoost top-k feature selection on the subsampled train set.
  3. Re-run TabPFN-2.6 OOF on subsampled train.
  4. Re-run TabPFN-2.6 outer on subsampled train → full val.
  5. Train ResDec-MHE 5 folds on subsampled train (2 GPUs parallel).
  6. Read each fold's val_predictions_final.npz + aggregate 5-fold R²
     against the TRUE cogn_global from the unchanged metadata.

N = canonical (516) is NOT re-run here — it already exists at
``outputs/redesign/p5_canonical_seed42/``. The learning-curve aggregate
reads the canonical result for the top data point.

Outputs:
  <output-base>/N_<k>/{metadata,top_k,tabpfn,folds,run.log}
  <output-base>/learning_curve_results.json   — aggregate {N → 5-fold R²}
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


def generate_subsampled_metadata(
    N: int,
    rng_seed: int,
    base_csv: Path,
    target_col: str,
    id_col: str,
    splits_path: Path,
    out_csv: Path,
) -> dict:
    """Write metadata.csv with training subjects subsampled to N per fold.

    Strategy: for each fold, identify that fold's training cohort from the
    splits file. Subsample to ``min(N, |train|)`` subjects. Mark the other
    training subjects as having NaN target (so the dataset validator
    drops them from training but keeps them queryable). Validation
    subjects remain unchanged.

    Returns a dict of ``{fold_idx: {'train_kept': N, 'train_dropped': M}}``
    for provenance.
    """
    df = pd.read_csv(base_csv)
    rng = np.random.default_rng(rng_seed)
    splits = json.loads(splits_path.read_text())

    # Collect all unique training subject IDs across folds.
    df_out = df.copy()
    df_out["__original_target__"] = df_out[target_col]
    # Global set of subjects that are used as TRAIN in ANY fold:
    train_in_any_fold: dict[str, set[int]] = {}
    val_in_any_fold: set[str] = set()
    for fi, fold in enumerate(splits["folds"]):
        for sid in fold["train"]:
            train_in_any_fold.setdefault(str(sid), set()).add(fi)
        for sid in fold["val"]:
            val_in_any_fold.add(str(sid))

    # Per-fold subsample: for each fold, pick N training subjects to KEEP.
    # A subject's target is kept if it is KEPT for at least one of its
    # train-folds; else set to NaN. (Conservative: preserves identifiability
    # across folds; fewer subjects NaN'd than a per-fold isolation.)
    kept_globally: set[str] = set()
    per_fold_info: dict[int, dict] = {}
    for fi, fold in enumerate(splits["folds"]):
        train_ids = [str(s) for s in fold["train"]]
        n_eff = min(N, len(train_ids))
        keep = list(rng.choice(train_ids, size=n_eff, replace=False))
        kept_globally.update(keep)
        per_fold_info[fi] = {
            "train_available": len(train_ids),
            "train_kept": n_eff,
            "train_dropped": len(train_ids) - n_eff,
        }

    # Any training subject NOT kept in ANY fold gets NaN target.
    # Note: val subjects are always kept.
    df_out[target_col] = df_out[target_col].astype(np.float64)
    sid_col_str = df_out[id_col].astype(str)
    drop_mask = ~sid_col_str.isin(kept_globally) & ~sid_col_str.isin(val_in_any_fold)
    df_out.loc[drop_mask, target_col] = np.nan

    # Drop helper col, write.
    df_out = df_out.drop(columns=["__original_target__"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)
    return {
        "N_requested": N,
        "N_kept_globally": len(kept_globally),
        "per_fold": per_fold_info,
        "n_subjects_nan_target": int(drop_mask.sum()),
    }


def run_step(cmd: list, log_handle, step_name: str) -> float:
    log_handle.write(f"\n=== {step_name} === @ {time.strftime('%H:%M:%S')}\n")
    log_handle.write(f"$ {' '.join(str(c) for c in cmd)}\n")
    log_handle.flush()
    t0 = time.time()
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    result = subprocess.run(cmd, stdout=log_handle, stderr=subprocess.STDOUT, env=env)
    elapsed = time.time() - t0
    log_handle.write(f"=== {step_name} done in {elapsed:.1f}s, exit {result.returncode} ===\n")
    log_handle.flush()
    if result.returncode != 0:
        raise RuntimeError(f"{step_name} failed (exit {result.returncode}); cmd: {cmd}")
    return elapsed


def run_train_folds_parallel(
    folds: list[int], gpus: list[int], base_args: list, log_handle,
) -> float:
    t0 = time.time()
    idx = 0
    while idx < len(folds):
        procs = []
        for g in gpus:
            if idx >= len(folds):
                break
            f = folds[idx]
            cmd = [
                "uv", "run", "python",
                str(ROOT / "scripts/resdec_mhe/training/train.py"),
                "--fold", str(f),
            ] + base_args
            log_handle.write(
                f"\n=== train fold {f} on GPU {g} === @ {time.strftime('%H:%M:%S')}\n"
            )
            log_handle.flush()
            env = {
                **os.environ, "PYTHONPATH": str(ROOT),
                "CUDA_VISIBLE_DEVICES": str(g),
            }
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


def run_one_N(
    N: int, rng_seed: int, output_base: Path,
    base_metadata_csv: Path, splits_path: Path, precomputed_dir: Path,
    target_col: str, id_col: str, gpus: list[int],
) -> dict:
    N_dir = output_base / f"N_{N}"
    N_dir.mkdir(parents=True, exist_ok=True)
    log_path = N_dir / "run.log"
    log = log_path.open("w")
    t_N = time.time()

    try:
        meta_dir = N_dir / "metadata"
        shuffled_csv = meta_dir / "metadata.csv"
        subsample_info = generate_subsampled_metadata(
            N, rng_seed, base_metadata_csv, target_col, id_col,
            splits_path, shuffled_csv,
        )
        log.write(f"subsampled metadata → {shuffled_csv}\n")
        log.write(f"subsample_info: {json.dumps(subsample_info, indent=2)}\n")

        topk_dir = N_dir / "top_k"
        run_step([
            "uv", "run", "python",
            str(ROOT / "scripts/resdec_mhe/tabpfn/compute_top_k_features.py"),
            "--splits-path", str(splits_path),
            "--precomputed-dir", str(precomputed_dir),
            "--metadata-csv", str(shuffled_csv),
            "--output-dir", str(topk_dir),
            "--top-k", "2000",
            "--feature-set", "A",
        ], log, "top-k features")

        tabpfn_dir = N_dir / "tabpfn"
        run_step([
            "uv", "run", "python",
            str(ROOT / "scripts/resdec_mhe/tabpfn/compute_oof.py"),
            "--splits-path", str(splits_path),
            "--precomputed-dir", str(precomputed_dir),
            "--metadata-csv", str(shuffled_csv),
            "--top-k-dir", str(topk_dir),
            "--output-dir", str(tabpfn_dir),
            "--top-k", "2000",
        ], log, "tabpfn OOF")

        run_step([
            "uv", "run", "python",
            str(ROOT / "scripts/resdec_mhe/tabpfn/compute_outer.py"),
            "--splits-path", str(splits_path),
            "--precomputed-dir", str(precomputed_dir),
            "--metadata-csv", str(shuffled_csv),
            "--top-k-dir", str(topk_dir),
            "--output-dir", str(tabpfn_dir),
            "--top-k", "2000",
            "--feature-set", "A",
        ], log, "tabpfn outer")

        folds_dir = N_dir / "folds"
        folds_dir.mkdir(parents=True, exist_ok=True)
        train_args = [
            "--config", "configs/resdec_mhe/canonical.yaml",
            "--output-dir", str(folds_dir),
            "--metadata-path", str(meta_dir),
            "--tabpfn-oof-dir", str(tabpfn_dir),
            "--tabpfn-outer-dir", str(tabpfn_dir),
        ]
        train_elapsed = run_train_folds_parallel(
            folds=[0, 1, 2, 3, 4], gpus=gpus,
            base_args=train_args, log_handle=log,
        )
        log.write(f"\n=== train all folds done in {train_elapsed:.1f}s ===\n")

        base_meta = pd.read_csv(base_metadata_csv)
        true_y_map = dict(zip(
            base_meta[id_col].astype(str),
            base_meta[target_col].astype(float),
        ))
        per_fold_r2 = []
        per_fold_n = []
        for f in range(5):
            npz = np.load(
                folds_dir / f"fold{f}" / "val_predictions_final.npz",
                allow_pickle=True,
            )
            subj_ids = [str(s) for s in npz["subject_ids"]]
            preds = np.asarray(npz["predictions"], dtype=np.float64)
            true_y = np.array([true_y_map[s] for s in subj_ids], dtype=np.float64)
            r2 = float(r2_score(true_y, preds))
            per_fold_r2.append(r2)
            per_fold_n.append(len(subj_ids))
            log.write(f"fold {f}: R² (vs TRUE) = {r2:+.4f}, n={len(subj_ids)}\n")

        elapsed_total = time.time() - t_N
        log.write(
            f"\n=== N={N} TOTAL: {elapsed_total / 60:.2f} min ===\n"
        )
        log.write(f"mean R² = {float(np.mean(per_fold_r2)):+.4f}\n")
        log.close()

        return {
            "N": N,
            "rng_seed": rng_seed,
            "subsample_info": subsample_info,
            "per_fold_r2": per_fold_r2,
            "per_fold_n": per_fold_n,
            "mean_r2": float(np.mean(per_fold_r2)),
            "std_r2": float(np.std(per_fold_r2, ddof=1)),
            "elapsed_min": elapsed_total / 60,
        }
    except Exception as exc:
        log.write(f"\n!!! N={N} FAILED: {exc}\n")
        log.close()
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--N-values", type=int, nargs="+",
                   default=[100, 200, 300, 400],
                   help="Training-set sizes to evaluate (canonical 516 not re-run).")
    p.add_argument("--output-base",
                   default="outputs/redesign/learning_curve")
    p.add_argument("--base-metadata-csv",
                   default="data/metadata_ROSMAP/metadata.csv")
    p.add_argument("--splits-path", default="outputs/splits.json")
    p.add_argument("--precomputed-dir", default="data/precomputed")
    p.add_argument("--target-col", default=DEFAULT_TARGET)
    p.add_argument("--id-col", default=DEFAULT_ID_COL)
    p.add_argument("--rng-seed", type=int, default=42)
    p.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    p.add_argument("--canonical-r2-json",
                   default="outputs/redesign/p5_canonical_seed42/best_vs_tabpfn_summary.json",
                   help="Existing canonical 5-fold summary (N=516 reference point).")
    args = p.parse_args()

    output_base = Path(args.output_base)
    output_base.mkdir(parents=True, exist_ok=True)
    aggregate_path = output_base / "learning_curve_results.json"

    results = (
        json.loads(aggregate_path.read_text())
        if aggregate_path.exists() else []
    )

    # Reference: canonical N=516 result.
    canon_path = Path(args.canonical_r2_json)
    if canon_path.exists():
        canon_data = json.loads(canon_path.read_text())
        # best_vs_tabpfn_summary.json structure: per-fold list.
        canon_r2 = canon_data.get("per_fold_composite_r2") or canon_data.get("per_fold_r2")
        if canon_r2:
            canon_entry = {
                "N": 516,
                "rng_seed": 42,
                "per_fold_r2": canon_r2,
                "mean_r2": float(np.mean(canon_r2)),
                "std_r2": float(np.std(canon_r2, ddof=1)),
                "source": "canonical (pre-existing)",
            }
            if not any(r.get("N") == 516 for r in results):
                results.append(canon_entry)

    for N in args.N_values:
        if any(r.get("N") == N and "mean_r2" in r for r in results):
            print(f"N={N} already done; skipping")
            continue
        print(f"\n=== N={N} starting @ {time.strftime('%H:%M:%S')} ===", flush=True)
        t0 = time.time()
        try:
            result = run_one_N(
                N=N, rng_seed=args.rng_seed,
                output_base=output_base,
                base_metadata_csv=Path(args.base_metadata_csv),
                splits_path=Path(args.splits_path),
                precomputed_dir=Path(args.precomputed_dir),
                target_col=args.target_col, id_col=args.id_col,
                gpus=args.gpus,
            )
        except Exception as exc:
            print(f"  N={N} FAILED: {exc}", flush=True)
            results.append({
                "N": N, "error": str(exc),
                "elapsed_min": (time.time() - t0) / 60,
            })
            aggregate_path.write_text(json.dumps(results, indent=2))
            continue

        results.append(result)
        aggregate_path.write_text(json.dumps(results, indent=2))
        print(
            f"  N={N}: mean R² = {result['mean_r2']:+.4f} ± {result['std_r2']:.3f}, "
            f"took {result['elapsed_min']:.1f} min",
            flush=True,
        )

    print(f"\nwrote {aggregate_path}")


if __name__ == "__main__":
    main()
