"""Optuna HPO on residualized cognition target — 7-axis search, 5-fold per trial.

Search space (all 7 sampled per trial):
  1. training.lr                   loguniform(1e-4, 1e-2)
  2. training.weight_decay         loguniform(1e-9, 1e-3)
  3. resdec_head.n_stages          categorical [1, 2, 3, 4]
  4. resdec_head.aux_lambdas[0]    uniform(0.0, 5.0)  (replicated to length n_stages)
  5. resdec_head.n_heads           categorical [2, 4, 8, 16]
  6. dataloader.batch_size         categorical [16, 24, 32, 48]
  7. training.gradient_clip_val    uniform(0.3, 2.0)

Per trial: train all 5 folds sequentially on a single GPU; objective = mean
val/r2 across the trial's eval-folds. MedianPruner (n_startup_trials=12,
n_warmup_steps=2) reports running mean after each fold and prunes unpromising
trials after fold 3 (~12 min in) instead of running all 5.

History: original sweep (EXP-049, 2026-05-04) used 5 axes with tight bounds
(lr [5e-4, 5e-3], wd [1e-7, 1e-4], aux [0, 2], n_stages {1,2,3}, n_heads
{2,4,8}) and 2-fold (folds 0+2) objective. Best trial gave 5-fold revalidation
R² = +0.249, indistinguishable from canonical config (Δ = -0.0005). Top-5
hit lower-bound on weight_decay and upper-bound on aux_lambdas[0], motivating
the widened ranges in this version. Axes 6-7 were unexplored (fixed at 24 / 1.0
in canonical). Per-trial 5-fold replaces the noisy 2-fold proxy.

Cross-worker SQLite study allows two workers (one per GPU) to pull trials
from the same study concurrently. Launch via _launch_hpo_cogn_residual.sh which
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
    """Sample 7 HPs and write a per-trial config file. Returns sampled-HP dict."""
    cfg = OmegaConf.merge(
        OmegaConf.load(_ROOT / "configs/default.yaml"),
        OmegaConf.load(_ROOT / "configs/resdec_mhe/canonical.yaml"),
        OmegaConf.load(base_config_path),
    )
    OmegaConf.set_struct(cfg, False)

    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-9, 1e-3, log=True)
    n_stages = trial.suggest_categorical("n_stages", [1, 2, 3, 4])
    aux_lambda = trial.suggest_float("aux_lambdas_0", 0.0, 5.0)
    n_heads = trial.suggest_categorical("n_heads", [2, 4, 8, 16])
    batch_size = trial.suggest_categorical("batch_size", [16, 24, 32, 48])
    grad_clip = trial.suggest_float("gradient_clip_val", 0.3, 2.0)

    cfg.training.lr = lr
    cfg.training.weight_decay = weight_decay
    cfg.training.gradient_clip_val = grad_clip
    cfg.model.resdec_head.n_stages = int(n_stages)
    cfg.model.resdec_head.aux_lambdas = [float(aux_lambda)] * int(n_stages)
    cfg.model.resdec_head.n_heads = int(n_heads)
    cfg.data.dataloader.batch_size = int(batch_size)
    cfg.experiment.run_name = f"cogn_residual_hpo_trial_{trial.number}"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_path)
    return {
        "lr": lr, "weight_decay": weight_decay,
        "n_stages": n_stages, "aux_lambdas_0": aux_lambda, "n_heads": n_heads,
        "batch_size": batch_size, "gradient_clip_val": grad_clip,
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
    # Force PYTHONPATH to this worktree so `import src.data.datamodule` resolves
    # to the worktree's variant-aware module rather than the parent repo's
    # master-branch copy. Without this, `python script.py` sets sys.path[0] to
    # the script's directory (no src/ subdir there) and namespace-package
    # resolution falls through to /host/.../proj_ml_snrna in sys.path, silently
    # picking up master's datamodule which lacks the residualize_against branch.
    env = {**os.environ, "PYTHONPATH": str(_ROOT)}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as logf:
        res = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=str(_ROOT), env=env)
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
        # Report running mean after each fold so MedianPruner compares trials
        # apples-to-apples at the same fold count (raw per-fold R² varies by
        # fold difficulty — fold 0/2 are easier than fold 1/3/4, so reporting
        # raw R² lets MedianPruner mistake "trial-on-easy-fold" for
        # "trial-better-than-others" when other trials happen to be on hard
        # folds). Running mean smooths the per-fold variance.
        running_mean = sum(fold_r2s) / len(fold_r2s)
        trial.report(running_mean, step=i)
        print(f"[trial {trial.number}] fold {fold}: R²={r2:+.4f} (running mean={running_mean:+.4f})", flush=True)
        if trial.should_prune():
            print(f"[trial {trial.number}] pruned after fold {fold}", flush=True)
            raise optuna.exceptions.TrialPruned()

    mean_r2 = sum(fold_r2s) / len(fold_r2s)
    print(f"[trial {trial.number}] FINAL mean R² = {mean_r2:+.4f} over folds {args.eval_folds}", flush=True)
    return mean_r2


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--study-name", required=True)
    p.add_argument("--storage", required=True,
                   help="SQLite URL, e.g. sqlite:////absolute/path/to/study.db")
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--base-config", type=Path,
                   default=_ROOT / "configs/resdec_mhe/cogn_residual/gpath_only.yaml")
    p.add_argument("--eval-folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--work-dir", type=Path, required=True,
                   help="Per-trial scratch dir (configs + training outputs).")
    p.add_argument("--metadata-path", type=Path,
                   default=_ROOT / "data/metadata_ROSMAP")
    p.add_argument("--precomputed-dir", type=Path,
                   default=_ROOT / "data/precomputed")
    p.add_argument("--tabpfn-oof-dir", type=Path,
                   default=_ROOT / "outputs/canonical/cogn_residual/gpath_only/tabpfn_cache")
    p.add_argument("--tabpfn-outer-dir", type=Path,
                   default=_ROOT / "outputs/canonical/cogn_residual/gpath_only/tabpfn_cache")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=12)
    # n_warmup_steps=2 means a trial is eligible for pruning after step 2
    # (= after fold 2 finishes; running mean of 3 folds reported). With 5-fold
    # per trial that lets bad trials prune ~12 min in instead of running all 5.
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=12, n_warmup_steps=2, interval_steps=1,
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
