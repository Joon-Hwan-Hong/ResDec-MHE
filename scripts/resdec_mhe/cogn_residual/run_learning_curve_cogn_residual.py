"""Variant learning-curve orchestrator: ResDec-MHE on residualized cognition R^2 vs N.

For each (seed, sub-N) pair in the cross-product of ``--rng-seeds`` x ``--N-values``:
  1. Generate metadata.csv with training subjects subsampled to N per fold
     (validation set preserved unchanged for fair comparison; non-kept training
     subjects get NaN target, mirroring canonical run_learning_curve.py).
  2. Re-fit per-fold OLS residualization on the sub-N training set via
     compute_residual_target.py (the OLS dropna-on-target naturally excludes
     the NaN'd-out training subjects).
  3. Re-run XGBoost top-k feature selection on the subsampled train set.
  4. Re-build TabPFN-2.6 OOF + outer caches on the subsampled residualized target.
  5. Re-build RF cache on the subsampled residualized target.
  6. Build stacked (TabPFN+RF average) cache.
  7. Write a temp variant YAML config that points at sub-N caches and run
     run_5fold_parallel.sh for 5-fold ResDec-MHE training on 2 GPUs.
  8. Read each fold's val_predictions_final.npz + aggregate 5-fold R^2 against
     the sub-N residualized target (the same target the model trained on).

N = full (516) is NOT re-run here; the variant canonical at full N is at
``outputs/canonical/cogn_residual/<variant>/p5_seed{42,67,21,2000,426}/`` and is
the reference anchor (Phase 1 fills in the 4 non-42 seeds).

Outputs:
  <output-base>/seed{seed}/N_{N}/{metadata,residual_cache,top_k,tabpfn_cache,
                                  rf_cache,stacked_cache,folds,run.log,config.yaml}
  <output-base>/learning_curve_results.json   - aggregate {(N, seed) -> 5-fold R^2}
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
import yaml
from omegaconf import OmegaConf
from sklearn.metrics import r2_score

_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TARGET = "cogn_global"
DEFAULT_ID_COL = "ROSMAP_IndividualID"
TOP_K = 2000


def generate_subsampled_metadata(
    N: int,
    rng_seed: int,
    base_csv: Path,
    target_col: str,
    id_col: str,
    splits_path: Path,
    out_csv: Path,
) -> dict:
    """Per-fold subsample of N training subjects; non-kept get NaN target."""
    df = pd.read_csv(base_csv)
    rng = np.random.default_rng(rng_seed)
    splits = json.loads(splits_path.read_text())
    folds = splits.get("folds", splits.get("splits", []))

    val_in_any_fold: set[str] = set()
    for fold in folds:
        for sid in fold["val"]:
            val_in_any_fold.add(str(sid))

    kept_globally: set[str] = set()
    per_fold_info: dict[int, dict] = {}
    for fi, fold in enumerate(folds):
        train_ids = [str(s) for s in fold["train"]]
        n_eff = min(N, len(train_ids))
        keep = list(rng.choice(train_ids, size=n_eff, replace=False))
        kept_globally.update(keep)
        per_fold_info[fi] = {
            "train_available": len(train_ids),
            "train_kept": n_eff,
            "train_dropped": len(train_ids) - n_eff,
        }

    df_out = df.copy()
    df_out[target_col] = df_out[target_col].astype(np.float64)
    sid_col_str = df_out[id_col].astype(str)
    drop_mask = ~sid_col_str.isin(kept_globally) & ~sid_col_str.isin(val_in_any_fold)
    df_out.loc[drop_mask, target_col] = np.nan

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
    env = {**os.environ, "PYTHONPATH": str(_ROOT)}
    try:
        subprocess.run(
            cmd, stdout=log_handle, stderr=subprocess.STDOUT, env=env, check=True,
        )
    except subprocess.CalledProcessError as exc:
        elapsed = time.time() - t0
        log_handle.write(
            f"=== {step_name} FAILED in {elapsed:.1f}s, exit {exc.returncode} ===\n"
        )
        log_handle.flush()
        raise RuntimeError(f"{step_name} failed (see log); cmd: {cmd}") from exc
    elapsed = time.time() - t0
    log_handle.write(f"=== {step_name} done in {elapsed:.1f}s ===\n")
    log_handle.flush()
    return elapsed


def run_train_5fold(
    config_path: Path, output_dir: Path, seed: int, log_handle,
    metadata_dir: Path, precomputed_dir: Path,
) -> float:
    """Wrap run_5fold_parallel.sh; expects we're in tmux already."""
    log_handle.write(
        f"\n=== train 5-fold (config={config_path}, seed={seed}) === "
        f"@ {time.strftime('%H:%M:%S')}\n"
    )
    log_handle.flush()
    t0 = time.time()
    env = {
        **os.environ,
        "PYTHONPATH": str(_ROOT),
        "CONFIG": str(config_path.relative_to(_ROOT)) if str(config_path).startswith(str(_ROOT)) else str(config_path),
        "OUTROOT": str(output_dir.relative_to(_ROOT)) if str(output_dir).startswith(str(_ROOT)) else str(output_dir),
        "SEED": str(seed),
        "RUN_REINFER": "1",
        "METADATA_PATH": str(metadata_dir.relative_to(_ROOT)) if str(metadata_dir).startswith(str(_ROOT)) else str(metadata_dir),
        "PRECOMPUTED_DIR": str(precomputed_dir.relative_to(_ROOT)) if str(precomputed_dir).startswith(str(_ROOT)) else str(precomputed_dir),
    }
    try:
        subprocess.run(
            ["bash", str(_ROOT / "scripts/resdec_mhe/training/run_5fold_parallel.sh")],
            stdout=log_handle, stderr=subprocess.STDOUT, env=env, cwd=str(_ROOT),
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        elapsed = time.time() - t0
        log_handle.write(
            f"=== train 5-fold FAILED in {elapsed:.1f}s, exit {exc.returncode} ===\n"
        )
        log_handle.flush()
        raise RuntimeError(f"train 5-fold failed (see log)") from exc
    elapsed = time.time() - t0
    log_handle.write(f"=== train 5-fold done in {elapsed:.1f}s ===\n")
    log_handle.flush()
    return elapsed


def write_subN_variant_config(
    base_config_path: Path,
    residual_cache_dir: Path,
    stacked_cache_dir: Path,
    out_config_path: Path,
) -> None:
    """Write a temp variant YAML pointing at sub-N caches."""
    cfg = OmegaConf.load(base_config_path)
    cfg.data.residualize_against.cache_dir = str(residual_cache_dir)
    cfg.data.tabpfn_oof_dir = str(stacked_cache_dir)
    cfg.data.tabpfn_outer_dir = str(stacked_cache_dir)
    out_config_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_config_path)


def aggregate_per_fold_r2(
    folds_dir: Path,
    residual_cache_dir: Path,
    splits_path: Path,
) -> dict:
    """Compute per-fold R^2 of model predictions vs sub-N residualized target."""
    splits = json.loads(splits_path.read_text())
    fold_list = splits.get("folds", splits.get("splits", []))
    per_fold_r2 = []
    per_fold_n = []
    for f in range(5):
        npz_pred = folds_dir / f"fold{f}" / "val_predictions_final.npz"
        npz_target = residual_cache_dir / f"residual_target_fold{f}.npz"
        if not npz_pred.is_file():
            return {"error": f"missing {npz_pred}"}
        if not npz_target.is_file():
            return {"error": f"missing {npz_target}"}
        d_pred = np.load(npz_pred, allow_pickle=True)
        d_tgt = np.load(npz_target, allow_pickle=True)
        pred_subj = [str(s) for s in d_pred["subject_ids"]]
        tgt_map = dict(zip(
            [str(s) for s in d_tgt["subject_ids"]],
            np.asarray(d_tgt["target"], dtype=np.float64),
        ))
        preds = np.asarray(d_pred["predictions"], dtype=np.float64)
        true_y = np.array([tgt_map[s] for s in pred_subj], dtype=np.float64)
        finite = np.isfinite(true_y) & np.isfinite(preds)
        if finite.sum() < 5:
            return {"error": f"fold {f}: <5 finite (true_y, pred) pairs"}
        r2 = float(r2_score(true_y[finite], preds[finite]))
        per_fold_r2.append(r2)
        per_fold_n.append(int(finite.sum()))
    return {
        "per_fold_r2": per_fold_r2,
        "per_fold_n": per_fold_n,
        "mean_r2": float(np.mean(per_fold_r2)),
        "std_r2": float(np.std(per_fold_r2, ddof=1)),
    }


def run_one_seed_N(
    seed: int, N: int, output_base: Path,
    base_metadata_csv: Path, splits_path: Path, precomputed_dir: Path,
    base_variant_config: Path, axes: list[str],
    target_col: str, id_col: str,
) -> dict:
    seed_dir = output_base / f"seed{seed}" / f"N_{N}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    log_path = seed_dir / "run.log"
    log = log_path.open("w")
    t_total = time.time()

    try:
        # Step 1: subsampled metadata.csv
        meta_dir = seed_dir / "metadata"
        meta_csv = meta_dir / "metadata.csv"
        subsample_info = generate_subsampled_metadata(
            N, seed, base_metadata_csv, target_col, id_col,
            splits_path, meta_csv,
        )
        log.write(f"subsampled metadata -> {meta_csv}\n")
        log.write(f"subsample_info: {json.dumps(subsample_info, indent=2)}\n")

        # Step 2: residualization on sub-N training subjects
        residual_cache_dir = seed_dir / "residual_cache"
        run_step([
            "uv", "run", "python",
            str(_ROOT / "scripts/resdec_mhe/cogn_residual/compute_residual_target.py"),
            "--variant-name", base_variant_config.stem,
            "--axes", *axes,
            "--target", target_col,
            "--metadata-path", str(meta_dir),
            "--splits-path", str(splits_path),
            "--out-dir", str(residual_cache_dir),
        ], log, "residualization")

        # Step 3: top-k features (canonical XGBoost selector on residualized target)
        topk_dir = seed_dir / "top_k"
        run_step([
            "uv", "run", "python",
            str(_ROOT / "scripts/resdec_mhe/tabpfn/compute_top_k_features.py"),
            "--splits-path", str(splits_path),
            "--precomputed-dir", str(precomputed_dir),
            "--metadata-csv", str(meta_csv),
            "--output-dir", str(topk_dir),
            "--top-k", str(TOP_K),
            "--feature-set", "A",
            "--target-col", target_col,
        ], log, "top-k features")

        # Step 4: TabPFN cache build on sub-N residualized target
        tabpfn_cache_dir = seed_dir / "tabpfn_cache"
        run_step([
            "uv", "run", "python",
            str(_ROOT / "scripts/resdec_mhe/cogn_residual/build_tabpfn_cache_cogn_residual.py"),
            "--variant-name", base_variant_config.stem,
            "--residual-cache-dir", str(residual_cache_dir),
            "--out-dir", str(tabpfn_cache_dir),
            "--splits-path", str(splits_path),
            "--precomputed-dir", str(precomputed_dir),
            "--metadata-csv", str(meta_csv),
            "--top-k-dir", str(topk_dir),
            "--top-k", str(TOP_K),
            "--feature-set", "A",
        ], log, "tabpfn cache build")

        # Step 5: RF cache build
        rf_cache_dir = seed_dir / "rf_cache"
        run_step([
            "uv", "run", "python",
            str(_ROOT / "scripts/resdec_mhe/cogn_residual/build_rf_cache_cogn_residual.py"),
            "--residual-cache-dir", str(residual_cache_dir),
            "--out-dir", str(rf_cache_dir),
            "--splits-path", str(splits_path),
            "--precomputed-dir", str(precomputed_dir),
            "--top-k-dir", str(topk_dir),
            "--top-k", str(TOP_K),
        ], log, "rf cache build")

        # Step 6: stacked cache (avg of TabPFN + RF)
        stacked_cache_dir = seed_dir / "stacked_cache"
        run_step([
            "uv", "run", "python",
            str(_ROOT / "scripts/resdec_mhe/cogn_residual/build_stacked_cache_cogn_residual.py"),
            "--tabpfn-cache-dir", str(tabpfn_cache_dir),
            "--rf-cache-dir", str(rf_cache_dir),
            "--out-dir", str(stacked_cache_dir),
        ], log, "stacked cache build")

        # Step 7: write temp variant config + train 5-fold
        sub_config_path = seed_dir / "config.yaml"
        write_subN_variant_config(
            base_variant_config, residual_cache_dir, stacked_cache_dir,
            sub_config_path,
        )
        folds_dir = seed_dir / "folds"
        folds_dir.mkdir(parents=True, exist_ok=True)
        run_train_5fold(
            sub_config_path, folds_dir, seed, log,
            metadata_dir=meta_dir, precomputed_dir=precomputed_dir,
        )

        # Step 8: aggregate R^2 vs sub-N residualized target
        agg = aggregate_per_fold_r2(folds_dir, residual_cache_dir, splits_path)
        if "error" in agg:
            log.write(f"aggregation error: {agg['error']}\n")
            return {"seed": seed, "N": N, "error": agg["error"],
                    "elapsed_min": (time.time() - t_total) / 60}

        elapsed_total = time.time() - t_total
        log.write(
            f"\n=== seed={seed} N={N} done in {elapsed_total / 60:.2f} min, "
            f"mean R^2 = {agg['mean_r2']:+.4f} ± {agg['std_r2']:.3f} ===\n"
        )
        log.close()
        return {
            "seed": seed, "N": N,
            "subsample_info": subsample_info,
            **agg,
            "elapsed_min": elapsed_total / 60,
        }
    except Exception as exc:
        log.write(f"\n!!! seed={seed} N={N} FAILED: {exc}\n")
        log.close()
        return {"seed": seed, "N": N, "error": str(exc),
                "elapsed_min": (time.time() - t_total) / 60}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--variant-name", default="gpath_only",
                   choices=["gpath_only", "multi_axis"])
    p.add_argument("--N-values", type=int, nargs="+",
                   default=[100, 200, 300, 400])
    p.add_argument("--rng-seeds", type=int, nargs="+",
                   default=[42, 67, 21, 2000, 426])
    p.add_argument("--output-base", type=Path,
                   default=_ROOT / "outputs/canonical/cogn_residual/gpath_only/learning_curve_k5")
    p.add_argument("--base-metadata-csv", type=Path,
                   default=_ROOT / "data/metadata_ROSMAP/metadata.csv")
    p.add_argument("--splits-path", type=Path,
                   default=_ROOT / "outputs/splits.json")
    p.add_argument("--precomputed-dir", type=Path,
                   default=_ROOT / "data/precomputed")
    p.add_argument("--target-col", default=DEFAULT_TARGET)
    p.add_argument("--id-col", default=DEFAULT_ID_COL)
    p.add_argument("--variant-config", type=Path, default=None,
                   help="Path to variant YAML; default <variant_name>.yaml.")
    p.add_argument("--axes", nargs="+", default=None,
                   help="Pathology axes for residualization; default per variant.")
    args = p.parse_args()

    args.output_base.mkdir(parents=True, exist_ok=True)
    aggregate_path = args.output_base / "learning_curve_results.json"

    base_variant_config = args.variant_config or (
        _ROOT / "configs/resdec_mhe/cogn_residual" / f"{args.variant_name}.yaml"
    )
    if args.axes is None:
        if args.variant_name == "gpath_only":
            axes = ["gpath"]
        elif args.variant_name == "multi_axis":
            axes = ["gpath", "tangsqrt", "amylsqrt"]
        else:
            raise ValueError(f"unknown variant {args.variant_name}; pass --axes")
    else:
        axes = list(args.axes)

    results = (
        json.loads(aggregate_path.read_text())
        if aggregate_path.exists() else []
    )

    for seed in args.rng_seeds:
        for N in args.N_values:
            already_done = any(
                r.get("N") == N and r.get("seed") == seed
                and "mean_r2" in r and "error" not in r
                for r in results
            )
            if already_done:
                print(f"skip seed={seed} N={N} (done)", flush=True)
                continue
            print(f"\n=== seed={seed} N={N} starting @ {time.strftime('%H:%M:%S')} ===",
                  flush=True)
            t0 = time.time()
            result = run_one_seed_N(
                seed=seed, N=N,
                output_base=args.output_base,
                base_metadata_csv=args.base_metadata_csv,
                splits_path=args.splits_path,
                precomputed_dir=args.precomputed_dir,
                base_variant_config=base_variant_config,
                axes=axes,
                target_col=args.target_col,
                id_col=args.id_col,
            )
            results.append(result)
            aggregate_path.write_text(json.dumps(results, indent=2))
            if "error" in result:
                print(f"  seed={seed} N={N} FAILED: {result['error']}", flush=True)
            else:
                print(
                    f"  seed={seed} N={N}: mean R^2 = {result['mean_r2']:+.4f} ± "
                    f"{result['std_r2']:.3f} ({result['elapsed_min']:.1f} min)",
                    flush=True,
                )

    print(f"\nwrote {aggregate_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
