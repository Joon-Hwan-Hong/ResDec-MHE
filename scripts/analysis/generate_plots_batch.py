"""
Batch-run generate_plots.py across every experiment directory referenced by
the sensitivity logs, populating each run's `figures/` dir with the plots
its analysis artifacts support.

Discovery reuses the log-parsing helpers from aggregate_sensitivity_plots:
for each seed in {42, 43, 44, 45, 46}, parse
outputs/logs/sensitivity{,_seed43..46}/*.log, extract experiment-dir paths,
and invoke generate_plots.py against each.

Outputs:
    <experiment_dir>/figures/            per-run plots + manifest.{json,md}
    outputs/plots/batch_runs/<YYYYMMDD_HHMMSS>/
        MANIFEST.md / manifest.json      batch-level provenance
        per_run_status.csv               one row per (seed, config, fold) run

Usage:
    uv run python scripts/analysis/generate_plots_batch.py
    uv run python scripts/analysis/generate_plots_batch.py --limit 3    # smoke test
"""

from __future__ import annotations

import argparse
import datetime as _dt
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd

# Reuse discovery helpers from the aggregate script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_sensitivity_plots import SEED_LOG_DIRS, discover_experiment_dirs, experiment_path

from src.utils.manifest import build_manifest, file_ref, write_manifest


logger = logging.getLogger(__name__)


def run_one(
    experiment_dir: Path,
    *,
    dpi: int,
    fmt: str,
    timeout_sec: int,
) -> dict:
    figures_dir = experiment_dir / "figures"
    tb_root = experiment_dir / "logs" / "tensorboard" / "cognitive_resilience_hpo7"
    # Find the version_X subdir if present
    tb_versions = sorted(tb_root.glob("version_*")) if tb_root.exists() else []
    tb_arg = tb_versions[-1] if tb_versions else tb_root

    cmd = [
        "uv", "run", "python",
        str(Path(__file__).parent / "generate_plots.py"),
        "--experiment-dir", str(experiment_dir),
        "--output-dir", str(figures_dir),
        "--format", fmt,
        "--dpi", str(dpi),
    ]
    if tb_arg.exists():
        cmd += ["--training-log-dir", str(tb_arg)]

    start = _dt.datetime.now(_dt.timezone.utc)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=timeout_sec, check=False,
        )
        ok = result.returncode == 0
        stderr_tail = "\n".join(result.stderr.strip().splitlines()[-5:])
    except subprocess.TimeoutExpired:
        ok = False
        stderr_tail = f"TIMEOUT after {timeout_sec}s"
        result = None
    elapsed = (_dt.datetime.now(_dt.timezone.utc) - start).total_seconds()

    # Count actual PNGs produced
    n_png = 0
    if figures_dir.exists():
        n_png = len(list(figures_dir.rglob("*.png")))

    return {
        "experiment_dir": str(experiment_dir),
        "figures_dir": str(figures_dir),
        "returncode": result.returncode if result is not None else -1,
        "ok": ok,
        "n_png": n_png,
        "elapsed_sec": round(elapsed, 2),
        "stderr_tail": stderr_tail,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outputs-root", default="outputs", type=str)
    parser.add_argument("--batch-output-root", default="outputs/plots/batch_runs", type=str,
                        help="Where batch-level manifest + status CSV land")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--format", type=str, default="png")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Per-run timeout in seconds")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process this many runs (for smoke tests)")
    parser.add_argument("--seeds", type=str, nargs="+", default=None,
                        help="Subset of seeds to process (e.g. seed42 seed43)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    outputs_root = Path(args.outputs_root).resolve()
    logs_root = outputs_root / "logs"
    run_ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    batch_dir = Path(args.batch_output_root).resolve() / run_ts
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Discover (seed, config, fold) -> experiment_dir
    targets: list[dict] = []
    seeds = args.seeds or list(SEED_LOG_DIRS.keys())
    for seed in seeds:
        if seed not in SEED_LOG_DIRS:
            logger.warning("Unknown seed %s, skipping", seed)
            continue
        discovered = discover_experiment_dirs(logs_root, SEED_LOG_DIRS[seed])
        for config, folds in sorted(discovered.items()):
            for fold in sorted(folds):
                exp = folds[fold]
                exp_dir = experiment_path(outputs_root, exp)
                if not exp_dir.exists():
                    logger.warning("Experiment dir missing: %s", exp_dir)
                    continue
                targets.append({
                    "seed": seed, "config": config, "fold": fold,
                    "experiment_dir": exp_dir,
                })

    if args.limit is not None:
        targets = targets[: args.limit]

    logger.info("Dispatching %d per-run generate_plots.py invocations", len(targets))
    logger.info("Batch manifest dir: %s", batch_dir)

    rows: list[dict] = []
    n_ok, n_fail = 0, 0
    for i, t in enumerate(targets, start=1):
        logger.info("[%d/%d] %s/%s/fold%d → %s",
                    i, len(targets), t["seed"], t["config"], t["fold"], t["experiment_dir"].name)
        status = run_one(t["experiment_dir"], dpi=args.dpi, fmt=args.format, timeout_sec=args.timeout)
        row = {**t, **status}
        # Ensure JSON-safe: Path -> str
        row["experiment_dir"] = str(row["experiment_dir"])
        rows.append(row)
        if status["ok"]:
            n_ok += 1
        else:
            n_fail += 1
            logger.warning("FAIL %s/%s/fold%d: %s",
                           t["seed"], t["config"], t["fold"], status["stderr_tail"])

    status_df = pd.DataFrame(rows)
    status_csv = batch_dir / "per_run_status.csv"
    status_df.to_csv(status_csv, index=False)

    total_png = int(status_df["n_png"].sum()) if len(status_df) else 0
    total_elapsed = float(status_df["elapsed_sec"].sum()) if len(status_df) else 0.0

    warnings: list[str] = []
    if n_fail:
        warnings.append(f"{n_fail}/{len(targets)} runs failed — see per_run_status.csv stderr_tail column")

    manifest = build_manifest(
        title=f"Batch generate_plots.py — {run_ts}",
        description=(
            f"Invoked generate_plots.py across {len(targets)} experiment dirs discovered "
            "from sensitivity logs. Per-run outputs land in each experiment's figures/ dir "
            "(with its own manifest.json)."
        ),
        script_path=Path(__file__),
        argv=sys.argv,
        config={
            "outputs_root": str(outputs_root),
            "dpi": args.dpi,
            "format": args.format,
            "per_run_timeout_sec": args.timeout,
            "seeds": seeds,
            "limit": args.limit,
            "n_targets": len(targets),
            "n_ok": n_ok,
            "n_fail": n_fail,
            "total_png_generated": total_png,
            "total_elapsed_sec": round(total_elapsed, 1),
        },
        inputs=[file_ref(status_csv, label="per_run_status.csv", compute_sha=False)],
        outputs=[
            file_ref(status_csv, label="per_run_status.csv", compute_sha=False),
        ],
        warnings=warnings,
        extras={"per_run": rows},
    )
    write_manifest(batch_dir, manifest)

    logger.info("Batch complete: %d ok, %d fail, %d PNGs total (%.1fs)",
                n_ok, n_fail, total_png, total_elapsed)
    logger.info("Batch manifest: %s", batch_dir / "MANIFEST.md")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
