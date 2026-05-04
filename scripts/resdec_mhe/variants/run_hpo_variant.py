"""Variant Optuna HPO — 5-axis search on Variant A residualized cognition target.

Search space (all sampled per trial):
  1. training.lr               loguniform(5e-4, 5e-3)
  2. training.weight_decay     loguniform(1e-7, 1e-4)
  3. resdec_head.n_stages      categorical [1, 2, 3]
  4. resdec_head.aux_lambdas[0] uniform(0.0, 2.0)  (replicated to length n_stages)
  5. resdec_head.n_heads       categorical [2, 4, 8]

Per trial: train fold 0 + fold 2 sequentially on a single GPU (2-fold mean R²
matches canonical HPO convention), report mean(val/r2) as Optuna objective.
Pruning via MedianPruner (n_startup_trials=10, n_warmup_steps=1 — i.e., a
trial can be pruned after its first fold completes if its R² lies below the
running median).

Cross-worker SQLite study allows two workers (one per GPU) to pull trials
from the same study concurrently. Launch via _launch_hpo_variant.sh which
forks a worker per GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import optuna
from omegaconf import OmegaConf

_ROOT = Path(__file__).resolve().parents[3]


def _build_trial_config(
    base_config_path: Path,
    trial: optuna.Trial,
    out_path: Path,
) -> dict:
    """Sample 5 HPs and write a per-trial config file. Returns sampled-HP dict."""
    cfg = OmegaConf.merge(
        OmegaConf.load(_ROOT / "configs/default.yaml"),
        OmegaConf.load(_ROOT / "configs/resdec_mhe/canonical.yaml"),
        OmegaConf.load(base_config_path),
    )
    OmegaConf.set_struct(cfg, False)

    lr = trial.suggest_float("lr", 5e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-4, log=True)
    n_stages = trial.suggest_categorical("n_stages", [1, 2, 3])
    aux_lambda = trial.suggest_float("aux_lambdas_0", 0.0, 2.0)
    n_heads = trial.suggest_categorical("n_heads", [2, 4, 8])

    cfg.training.lr = lr
    cfg.training.weight_decay = weight_decay
    cfg.model.resdec_head.n_stages = int(n_stages)
    cfg.model.resdec_head.aux_lambdas = [float(aux_lambda)] * int(n_stages)
    cfg.model.resdec_head.n_heads = int(n_heads)
    cfg.experiment.run_name = f"variant_hpo_trial_{trial.number}"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_path)
    return {
        "lr": lr, "weight_decay": weight_decay,
        "n_stages": n_stages, "aux_lambdas_0": aux_lambda, "n_heads": n_heads,
    }


def _train_one_fold(
    config_path: Path, fold: int, output_dir: Path,
    metadata_path: Path, precomputed_dir: Path,
    tabpfn_oof_dir: Path, tabpfn_outer_dir: Path,
    log_path: Path,
) -> tuple[int, float | None]:
    """Run train.py on one fold; return (exit_code, val_best_r2_or_None)."""
    cmd = [
        "uv", "run", "python",
        str(_ROOT / "scripts/resdec_mhe/training/train.py"),
        "--config", str(config_path),
        "--fold", str(fold),
        "--output-dir", str(output_dir),
        "--metadata-path", str(metadata_path),
        "--precomputed-dir", str(precomputed_dir),
        "--tabpfn-oof-dir", str(tabpfn_oof_dir),
        "--tabpfn-outer-dir", str(tabpfn_outer_dir),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as logf:
        res = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=str(_ROOT))
    rc = res.returncode

    # Read best val R² from per-fold summary.json (last-epoch metrics)
    fold_dir = output_dir / f"fold{fold}"
    summary_path = fold_dir / "summary.json"
    if rc != 0 or not summary_path.is_file():
        return rc, None
    summary = json.loads(summary_path.read_text())
    val_results = summary.get("val_results", [])
    if not val_results:
        return rc, None
    return rc, float(val_results[0].get("val/r2", float("nan")))


def _objective(trial: optuna.Trial, args) -> float:
    trial_root = args.work_dir / f"trial_{trial.number}"
    trial_root.mkdir(parents=True, exist_ok=True)
    trial_config = trial_root / "config.yaml"
    sampled = _build_trial_config(args.base_config, trial, trial_config)
    print(f"[trial {trial.number}] sampled: {sampled}", flush=True)

    output_dir = trial_root / "training"
    fold_r2s = []
    for i, fold in enumerate(args.eval_folds):
        log = trial_root / f"fold{fold}_train.log"
        rc, r2 = _train_one_fold(
            trial_config, fold, output_dir,
            args.metadata_path, args.precomputed_dir,
            args.tabpfn_oof_dir, args.tabpfn_outer_dir, log,
        )
        if rc != 0 or r2 is None:
            print(f"[trial {trial.number}] fold {fold} FAILED rc={rc}", flush=True)
            raise optuna.exceptions.TrialPruned()
        fold_r2s.append(r2)
        # Report intermediate value after each fold; pruner sees it
        trial.report(r2, step=i)
        print(f"[trial {trial.number}] fold {fold}: R²={r2:+.4f}", flush=True)
        if trial.should_prune():
            print(f"[trial {trial.number}] pruned after fold {fold}", flush=True)
            raise optuna.exceptions.TrialPruned()

    mean_r2 = sum(fold_r2s) / len(fold_r2s)
    print(f"[trial {trial.number}] mean R² = {mean_r2:+.4f} over {args.eval_folds}", flush=True)
    return mean_r2


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--study-name", required=True)
    p.add_argument("--storage", required=True,
                   help="SQLite URL, e.g. sqlite:////absolute/path/to/study.db")
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--base-config", type=Path,
                   default=_ROOT / "configs/resdec_mhe/variants/gpath_only.yaml")
    p.add_argument("--eval-folds", nargs="+", type=int, default=[0, 2])
    p.add_argument("--work-dir", type=Path, required=True,
                   help="Per-trial scratch dir (configs + training outputs).")
    p.add_argument("--metadata-path", type=Path,
                   default=_ROOT / "data/metadata_ROSMAP")
    p.add_argument("--precomputed-dir", type=Path,
                   default=_ROOT / "data/precomputed")
    p.add_argument("--tabpfn-oof-dir", type=Path,
                   default=_ROOT / "outputs/canonical/variants/gpath_only/tabpfn_cache")
    p.add_argument("--tabpfn-outer-dir", type=Path,
                   default=_ROOT / "outputs/canonical/variants/gpath_only/tabpfn_cache")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=8)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=10, n_warmup_steps=1, interval_steps=1,
    )
    study = optuna.create_study(
        study_name=args.study_name, storage=args.storage,
        load_if_exists=True, direction="maximize",
        sampler=sampler, pruner=pruner,
    )
    print(f"worker pid={os.getpid()}: starting study={args.study_name} (n_trials={args.n_trials})", flush=True)
    study.optimize(lambda t: _objective(t, args), n_trials=args.n_trials,
                   gc_after_trial=True)
    print(f"worker pid={os.getpid()}: study done; best={study.best_value:+.4f} params={study.best_params}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
