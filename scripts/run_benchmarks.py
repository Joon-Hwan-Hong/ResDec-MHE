"""
Benchmark orchestrator — schedules, monitors, and aggregates all baseline runs.

Manages both DL and classical baselines with:
  - Sequential or parallel scheduling (GPU-aware)
  - Crash/OOM detection and structured error logging
  - Per-benchmark timeout enforcement
  - Structured status JSON for agent-based analysis
  - Result aggregation into comparison table

Usage:
    uv run python scripts/run_benchmarks.py --device cuda:1
    uv run python scripts/run_benchmarks.py --device cuda:1 --only cloudpred perceiver_io
    uv run python scripts/run_benchmarks.py --status   # just print current status
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "outputs" / "baselines"
STATUS_FILE = PROJECT_ROOT / "outputs" / "benchmark_status.json"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"


# ---------------------------------------------------------------------------
# Benchmark definitions
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkDef:
    """Definition of a single benchmark run."""
    name: str
    command: list[str]          # full command to execute
    results_dir: str            # where results land
    results_csv: str            # expected CSV with per-fold metrics
    timeout_min: int = 120      # max runtime in minutes
    gpu: bool = False           # whether it needs GPU
    description: str = ""
    input_description: str = ""
    blockers: list[str] = field(default_factory=list)  # files that must exist
    is_data_prep: bool = False  # True for data preparation tasks (not benchmarks)


# ---------------------------------------------------------------------------
# Data preparation definitions
# ---------------------------------------------------------------------------

def get_data_prep_tasks() -> list[BenchmarkDef]:
    """Data preparation tasks that produce files needed by blocked benchmarks."""
    mixmil_h5ad = str(PROJECT_ROOT / "baselines/shared/mixmil_input.h5ad")
    scphase_h5ad = str(PROJECT_ROOT / "baselines/shared/scphase_input.h5ad")

    return [
        BenchmarkDef(
            name="prep_scphase_h5ad",
            command=[
                "uv", "run", "python", "-u",
                str(PROJECT_ROOT / "baselines/prepare_data.py"),
                "--adata", str(PROJECT_ROOT / "data/snRNAseq/adata_ROSMAP_merged.h5ad"),
                "--splits", str(PROJECT_ROOT / "outputs/splits.json"),
                "--metadata", str(PROJECT_ROOT / "data/metadata_ROSMAP/metadata.csv"),
                "--output-dir", str(PROJECT_ROOT / "baselines/shared"),
                "--methods", "scphase",
            ],
            results_dir="baselines/shared",
            results_csv=scphase_h5ad,  # "results_csv" doubles as completion marker
            timeout_min=30,
            gpu=False,
            description="Create scPhase input h5ad (raw expression, 3.9M cells)",
            is_data_prep=True,
        ),
        BenchmarkDef(
            name="prep_mixmil_h5ad",
            command=[
                "uv", "run", "python", "-u",
                str(PROJECT_ROOT / "baselines/prepare_data.py"),
                "--adata", str(PROJECT_ROOT / "data/snRNAseq/adata_ROSMAP_merged.h5ad"),
                "--splits", str(PROJECT_ROOT / "outputs/splits.json"),
                "--metadata", str(PROJECT_ROOT / "data/metadata_ROSMAP/metadata.csv"),
                "--output-dir", str(PROJECT_ROOT / "baselines/shared"),
                "--methods", "mixmil",
                "--scvi-epochs", "50",
                "--scvi-latent", "30",
            ],
            results_dir="baselines/shared",
            results_csv=mixmil_h5ad,
            timeout_min=120,  # scVI training can take a while
            gpu=True,
            description="Create MixMIL input h5ad (scVI 30-dim embeddings, 3.9M cells)",
            is_data_prep=True,
        ),
    ]


def get_all_benchmarks(device: str = "cuda:1") -> list[BenchmarkDef]:
    """Return all benchmark definitions."""
    cp_venv = str(PROJECT_ROOT / "baselines/cloudpred/.venv/bin/python")
    pio_venv = str(PROJECT_ROOT / "baselines/perceiver_io/.venv/bin/python")
    gpio_venv = str(PROJECT_ROOT / "baselines/gpio/.venv/bin/python")
    main_venv = "uv"

    common_args = [
        "--data-dir", str(PROJECT_ROOT / "data/precomputed"),
        "--splits", str(PROJECT_ROOT / "outputs/splits.json"),
        "--metadata-dir", str(PROJECT_ROOT / "data/metadata_ROSMAP"),
    ]

    return [
        BenchmarkDef(
            name="classical",
            command=[
                main_venv, "run", "python", "-u",
                str(PROJECT_ROOT / "scripts/analysis/run_baselines.py"),
                "--precomputed-dir", str(PROJECT_ROOT / "data/precomputed"),
                "--splits-path", str(PROJECT_ROOT / "outputs/splits.json"),
                "--metadata-path", str(PROJECT_ROOT / "data/metadata_ROSMAP"),
                "--output", str(PROJECT_ROOT / "outputs/baseline_results.csv"),
                "--cv-tune",
            ],
            results_dir="outputs/baseline_results.csv",
            results_csv="outputs/baseline_results.csv",
            timeout_min=240,
            gpu=True,
            description="Ridge, ElasticNet, RandomForest, XGBoost, PLS on 3 feature sets",
            input_description="Pseudobulk (148K), cell proportions (31), CCC summary (18)",
        ),
        BenchmarkDef(
            name="cloudpred",
            command=[
                cp_venv, "-u",
                str(PROJECT_ROOT / "baselines/cloudpred/run_rosmap.py"),
                *common_args,
                "--results-dir", str(RESULTS_DIR / "cloudpred"),
                "--device", device,
            ],
            results_dir="outputs/baselines/cloudpred",
            results_csv="outputs/baselines/cloudpred/results.csv",
            timeout_min=60,
            gpu=True,
            description="CloudPred (unstructured cell bag, GMM density, polynomial)",
            input_description="Raw cells PCA→10, no cell type structure",
        ),
        BenchmarkDef(
            name="cloudpred_pertype",
            command=[
                cp_venv, "-u",
                str(PROJECT_ROOT / "baselines/cloudpred/run_rosmap_pertype.py"),
                *common_args,
                "--results-dir", str(RESULTS_DIR / "cloudpred_pertype"),
                "--device", device,
                "--k-per-type", "3",
            ],
            results_dir="outputs/baselines/cloudpred_pertype",
            results_csv="outputs/baselines/cloudpred_pertype/results.csv",
            timeout_min=60,
            gpu=True,
            description="CloudPred per-type (24 cell-type GMMs, batched)",
            input_description="Raw cells PCA→10 per type, cell type structure",
        ),
        BenchmarkDef(
            name="perceiver_io",
            command=[
                pio_venv, "-u",
                str(PROJECT_ROOT / "baselines/perceiver_io/run_rosmap.py"),
                *common_args,
                "--results-dir", str(RESULTS_DIR / "perceiver_io"),
                "--device", device,
            ],
            results_dir="outputs/baselines/perceiver_io",
            results_csv="outputs/baselines/perceiver_io/results.csv",
            timeout_min=120,
            gpu=True,
            description="Perceiver IO (cross-attention on pseudobulk + CCC tokens)",
            input_description="Pseudobulk (31x4796) + CCC summary (18d)",
        ),
        BenchmarkDef(
            name="gpio",
            command=[
                gpio_venv, "-u",
                str(PROJECT_ROOT / "baselines/gpio/run_rosmap.py"),
                *common_args,
                "--results-dir", str(RESULTS_DIR / "gpio"),
                "--device", device,
            ],
            results_dir="outputs/baselines/gpio",
            results_csv="outputs/baselines/gpio/results.csv",
            timeout_min=120,
            gpu=True,
            description="GPIO (graph Perceiver IO on CCC graph with RWPE)",
            input_description="Full CCC graph + pseudobulk + RWPE",
        ),
        BenchmarkDef(
            name="mixmil",
            command=[
                str(PROJECT_ROOT / "baselines/mixmil/.venv/bin/python"), "-u",
                str(PROJECT_ROOT / "baselines/mixmil/run_rosmap.py"),
                "--data-h5ad", str(PROJECT_ROOT / "baselines/shared/mixmil_input.h5ad"),
                "--splits", str(PROJECT_ROOT / "outputs/splits.json"),
                "--results-dir", str(RESULTS_DIR / "mixmil"),
            ],
            results_dir="outputs/baselines/mixmil",
            results_csv="outputs/baselines/mixmil/results.csv",
            timeout_min=120,
            gpu=False,
            description="MixMIL (GLMM + attention MIL)",
            input_description="Unstructured cell bag, scVI→30d embeddings",
            blockers=[str(PROJECT_ROOT / "baselines/shared/mixmil_input.h5ad")],
        ),
        BenchmarkDef(
            name="abmil",
            command=[
                main_venv, "run", "python", "-u",
                str(PROJECT_ROOT / "baselines/abmil/run_rosmap.py"),
                "--data-h5ad", str(PROJECT_ROOT / "baselines/shared/mixmil_input.h5ad"),
                "--splits", str(PROJECT_ROOT / "outputs/splits.json"),
                "--results-dir", str(RESULTS_DIR / "abmil"),
                "--device", device,
            ],
            results_dir="outputs/baselines/abmil",
            results_csv="outputs/baselines/abmil/results.csv",
            timeout_min=60,
            gpu=True,
            description="ABMIL (gated attention MIL)",
            input_description="Unstructured cell bag, scVI→30d embeddings",
            blockers=[str(PROJECT_ROOT / "baselines/shared/mixmil_input.h5ad")],
        ),
        BenchmarkDef(
            name="set_transformer",
            command=[
                main_venv, "run", "python", "-u",
                str(PROJECT_ROOT / "baselines/set_transformer/run_rosmap.py"),
                "--data-h5ad", str(PROJECT_ROOT / "baselines/shared/mixmil_input.h5ad"),
                "--splits", str(PROJECT_ROOT / "outputs/splits.json"),
                "--results-dir", str(RESULTS_DIR / "set_transformer"),
                "--device", device,
            ],
            results_dir="outputs/baselines/set_transformer",
            results_csv="outputs/baselines/set_transformer/results.csv",
            timeout_min=60,
            gpu=True,
            description="Set Transformer (ISAB + PMA pooling)",
            input_description="Unstructured cell bag, scVI→30d embeddings",
            blockers=[str(PROJECT_ROOT / "baselines/shared/mixmil_input.h5ad")],
        ),
        BenchmarkDef(
            name="scphase",
            command=[
                str(PROJECT_ROOT / "baselines/scPhase/.venv/bin/python"), "-u",
                str(PROJECT_ROOT / "baselines/scPhase/run_rosmap.py"),
                "--data-h5ad", str(PROJECT_ROOT / "baselines/shared/scphase_input.h5ad"),
                "--splits", str(PROJECT_ROOT / "outputs/splits.json"),
                "--results-dir", str(RESULTS_DIR / "scphase"),
                "--device", device,
            ],
            results_dir="outputs/baselines/scphase",
            results_csv="outputs/baselines/scphase/results.csv",
            timeout_min=180,
            gpu=True,
            description="scPhase (per-cell-type attention, classification-first)",
            input_description="Raw cells (4797 genes) + cell type labels",
            blockers=[str(PROJECT_ROOT / "baselines/shared/scphase_input.h5ad")],
        ),
    ]


# ---------------------------------------------------------------------------
# Status tracking
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkStatus:
    name: str
    status: str = "pending"     # pending, blocked, running, completed, failed, timeout
    start_time: float | None = None
    end_time: float | None = None
    elapsed_s: float | None = None
    pid: int | None = None
    error: str | None = None
    metrics: dict | None = None  # mean R², etc. from results CSV
    log_file: str | None = None


def load_status() -> dict[str, dict]:
    """Load status from JSON file."""
    if STATUS_FILE.exists():
        with open(STATUS_FILE) as f:
            return json.load(f)
    return {}


def save_status(statuses: dict[str, dict]) -> None:
    """Save status to JSON file."""
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATUS_FILE, "w") as f:
        json.dump(statuses, f, indent=2)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def check_blockers(bench: BenchmarkDef) -> str | None:
    """Return first missing blocker path, or None if all present."""
    for path in bench.blockers:
        if not Path(path).exists():
            return path
    return None


def run_benchmark(bench: BenchmarkDef) -> BenchmarkStatus:
    """Run a single benchmark with timeout and error detection."""
    status = BenchmarkStatus(name=bench.name)
    log_file = LOG_DIR / f"benchmark_{bench.name}.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    status.log_file = str(log_file)

    # Check blockers
    missing = check_blockers(bench)
    if missing:
        status.status = "blocked"
        status.error = f"Missing: {missing}"
        logger.warning("[%s] BLOCKED — %s", bench.name, status.error)
        return status

    # Check if results already exist
    results_csv = PROJECT_ROOT / bench.results_csv
    if results_csv.exists():
        logger.info("[%s] Results already exist at %s — skipping", bench.name, results_csv)
        status.status = "completed"
        status.metrics = _read_metrics(results_csv, bench.name)
        return status

    logger.info("[%s] Starting: %s", bench.name, bench.description)
    logger.info("[%s] Command: %s", bench.name, " ".join(bench.command[:4]) + " ...")
    status.status = "running"
    status.start_time = time.time()

    try:
        with open(log_file, "w") as lf:
            proc = subprocess.Popen(
                bench.command,
                stdout=lf,
                stderr=subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
                preexec_fn=os.setsid,
            )
            status.pid = proc.pid
            logger.info("[%s] PID=%d, timeout=%d min, log=%s",
                        bench.name, proc.pid, bench.timeout_min, log_file)

            # Wait with timeout
            timeout_s = bench.timeout_min * 60
            try:
                returncode = proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                time.sleep(5)
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                status.status = "timeout"
                status.error = f"Exceeded {bench.timeout_min} min timeout"
                logger.error("[%s] TIMEOUT after %d min", bench.name, bench.timeout_min)
                status.end_time = time.time()
                status.elapsed_s = status.end_time - status.start_time
                return status

        status.end_time = time.time()
        status.elapsed_s = status.end_time - status.start_time

        if returncode != 0:
            # Read last 20 lines of log for error context
            tail = _tail_file(log_file, 20)
            status.status = "failed"
            status.error = f"Exit code {returncode}. Last log:\n{tail}"
            # Check for common errors
            if "OutOfMemoryError" in tail or "CUDA out of memory" in tail:
                status.error = f"OOM: {tail.split('OutOfMemoryError')[0][-100:]}"
            elif "ModuleNotFoundError" in tail:
                status.error = f"Missing dep: {tail}"
            logger.error("[%s] FAILED (exit %d) in %.1fs", bench.name, returncode,
                        status.elapsed_s)
        else:
            status.status = "completed"
            status.metrics = _read_metrics(results_csv, bench.name)
            logger.info("[%s] COMPLETED in %.1fs — %s",
                        bench.name, status.elapsed_s,
                        _format_metrics(status.metrics))

    except Exception as e:
        status.status = "failed"
        status.error = str(e)
        status.end_time = time.time()
        if status.start_time:
            status.elapsed_s = status.end_time - status.start_time
        logger.exception("[%s] Exception: %s", bench.name, e)

    return status


# ---------------------------------------------------------------------------
# Claude Code agent — analysis after each benchmark
# ---------------------------------------------------------------------------

AGENT_PROMPT_TEMPLATE = """\
You are a benchmark analyst for a cognitive resilience prediction project.

Benchmark "{name}" just finished with status: {status}.
Results dir: {results_dir}
Log file: {log_file}
Elapsed: {elapsed}
Description: {description}
Input: {input_description}

{context}

Tasks:
1. If FAILED or TIMEOUT: Read the log file, diagnose the root cause (OOM, missing dep, \
data format mismatch, numerical issue, etc.), and write a brief diagnosis to \
outputs/benchmark_analysis/{name}_diagnosis.txt. Do NOT fix the code — just diagnose.

2. If COMPLETED: Read the results CSV, verify the metrics are plausible (not NaN, \
not all identical, R² in reasonable range for this method). Check if any folds \
are outliers. Write a brief analysis to outputs/benchmark_analysis/{name}_analysis.txt.

3. Update docs/results/2026-03-30-baseline-benchmarks.md with the results \
(move from Pending to Completed, fill in the per-fold table).

4. Check research alignment: are the inputs this baseline received appropriate \
for a fair comparison? Flag any concerns.

Be concise. Write files directly, don't ask questions.
"""


def invoke_claude_agent(bench: BenchmarkDef, status: BenchmarkStatus) -> None:
    """Call Claude Code CLI to analyze a benchmark result."""
    analysis_dir = PROJECT_ROOT / "outputs" / "benchmark_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    context = ""
    if status.status == "failed":
        context = f"Error: {status.error}"
    elif status.status == "completed" and status.metrics:
        context = f"Metrics: {json.dumps(status.metrics, indent=2)}"
    elif status.status == "blocked":
        context = f"Blocked by: {status.error}"
        logger.info("[%s] Skipping agent — benchmark was blocked", bench.name)
        return

    elapsed_str = f"{status.elapsed_s:.0f}s" if status.elapsed_s else "N/A"
    prompt = AGENT_PROMPT_TEMPLATE.format(
        name=bench.name,
        status=status.status,
        results_dir=bench.results_dir,
        log_file=status.log_file or "N/A",
        elapsed=elapsed_str,
        description=bench.description,
        input_description=bench.input_description,
        context=context,
    )

    logger.info("[%s] Invoking Claude Code agent for analysis...", bench.name)
    agent_log = analysis_dir / f"{bench.name}_agent.log"

    try:
        result = subprocess.run(
            ["claude", "-p", prompt, "--output-format", "text"],
            capture_output=True,
            text=True,
            timeout=300,  # 5 min max for agent
            cwd=str(PROJECT_ROOT),
        )
        with open(agent_log, "w") as f:
            f.write(f"=== Claude Agent Output ===\n")
            f.write(f"Exit code: {result.returncode}\n\n")
            f.write(result.stdout)
            if result.stderr:
                f.write(f"\n=== stderr ===\n{result.stderr}")
        logger.info("[%s] Agent analysis complete — see %s", bench.name, agent_log)
    except subprocess.TimeoutExpired:
        logger.warning("[%s] Agent timed out (5 min)", bench.name)
    except FileNotFoundError:
        logger.warning("[%s] 'claude' CLI not found — skipping agent analysis", bench.name)
    except Exception as e:
        logger.warning("[%s] Agent error: %s", bench.name, e)


def _tail_file(path: Path, n: int = 20) -> str:
    """Read last n lines of a file."""
    try:
        with open(path) as f:
            lines = f.readlines()
            return "".join(lines[-n:])
    except Exception:
        return "<could not read log>"


def _read_metrics(csv_path: Path, name: str) -> dict | None:
    """Read results CSV and compute mean +/- std across folds."""
    try:
        df = pd.read_csv(csv_path)
        metrics = {}
        for col in ["r2", "mae", "rmse", "pearson_r", "spearman_rho"]:
            if col in df.columns:
                metrics[f"{col}_mean"] = float(df[col].mean())
                metrics[f"{col}_std"] = float(df[col].std())
        return metrics
    except Exception:
        return None


def _format_metrics(metrics: dict | None) -> str:
    """Format metrics dict for logging."""
    if not metrics:
        return "no metrics"
    r2 = metrics.get("r2_mean")
    r2_std = metrics.get("r2_std")
    if r2 is not None:
        return f"R²={r2:.4f}±{r2_std:.4f}"
    return str(metrics)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_status(statuses: dict[str, dict], benchmarks: list[BenchmarkDef]) -> None:
    """Print current status of all benchmarks."""
    print(f"\n{'='*80}")
    print(f"  Benchmark Status — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    print(f"{'Name':<20s} {'Status':<12s} {'Time':>8s} {'R² (mean±std)':>18s}  Notes")
    print("-" * 80)

    for bench in benchmarks:
        s = statuses.get(bench.name, {"status": "pending"})
        elapsed = ""
        if s.get("elapsed_s"):
            m = s["elapsed_s"] / 60
            elapsed = f"{m:.1f}m"
        r2_str = ""
        if s.get("metrics") and s["metrics"].get("r2_mean") is not None:
            r2_str = f"{s['metrics']['r2_mean']:.4f}±{s['metrics']['r2_std']:.4f}"
        notes = s.get("error", "")[:40] if s.get("error") else ""
        print(f"{bench.name:<20s} {s['status']:<12s} {elapsed:>8s} {r2_str:>18s}  {notes}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Benchmark orchestrator")
    parser.add_argument("--device", default="cuda:1", help="GPU device for DL baselines")
    parser.add_argument("--only", nargs="*", help="Run only these benchmarks (by name)")
    parser.add_argument("--skip", nargs="*", default=[], help="Skip these benchmarks")
    parser.add_argument("--status", action="store_true", help="Just print status and exit")
    parser.add_argument("--force", action="store_true", help="Re-run even if results exist")
    parser.add_argument("--no-agent", action="store_true",
                        help="Skip Claude Code agent analysis after each benchmark")
    args = parser.parse_args()

    benchmarks = get_all_benchmarks(device=args.device)
    statuses = load_status()

    if args.status:
        print_status(statuses, benchmarks)
        return

    # Filter benchmarks
    if args.only:
        benchmarks = [b for b in benchmarks if b.name in args.only]
    benchmarks = [b for b in benchmarks if b.name not in args.skip]

    if args.force:
        # Clear results to force re-run
        for b in benchmarks:
            csv = PROJECT_ROOT / b.results_csv
            if csv.exists():
                csv.unlink()
                logger.info("Removed %s for forced re-run", csv)

    logger.info("Running %d benchmarks: %s", len(benchmarks),
                ", ".join(b.name for b in benchmarks))

    # Build map: blocker file -> data prep task that produces it
    data_preps = get_data_prep_tasks()
    blocker_to_prep: dict[str, BenchmarkDef] = {}
    for prep in data_preps:
        blocker_to_prep[prep.results_csv] = prep
    completed_preps: set[str] = set()

    for bench in benchmarks:
        # Check if this benchmark is blocked and we can resolve it
        missing = check_blockers(bench)
        if missing and missing in blocker_to_prep and missing not in completed_preps:
            prep = blocker_to_prep[missing]
            if not Path(prep.results_csv).exists():
                logger.info("[%s] Blocked by %s — running data prep: %s",
                            bench.name, Path(missing).name, prep.name)
                prep_status = run_benchmark(prep)
                statuses[prep.name] = asdict(prep_status)
                save_status(statuses)
                if prep_status.status == "completed":
                    completed_preps.add(missing)
                    logger.info("[%s] Data prep succeeded — proceeding with benchmark",
                                bench.name)
                else:
                    logger.error("[%s] Data prep FAILED — skipping benchmark", bench.name)
                    statuses[bench.name] = asdict(BenchmarkStatus(
                        name=bench.name, status="blocked",
                        error=f"Data prep {prep.name} failed: {prep_status.error}",
                    ))
                    save_status(statuses)
                    continue
            else:
                completed_preps.add(missing)

        status = run_benchmark(bench)
        statuses[bench.name] = asdict(status)
        save_status(statuses)

        # Invoke Claude Code agent for analysis (unless --no-agent)
        if not args.no_agent and status.status in ("completed", "failed", "timeout"):
            invoke_claude_agent(bench, status)

    # Final summary
    print_status(statuses, get_all_benchmarks(device=args.device))

    # Aggregate comparison table
    print_comparison_table(statuses, get_all_benchmarks(device=args.device))


def print_comparison_table(statuses: dict, benchmarks: list[BenchmarkDef]) -> None:
    """Print final comparison table."""
    print(f"\n{'='*80}")
    print(f"  Comparison Table — Our model R²=0.323±0.067")
    print(f"{'='*80}")
    print(f"{'Baseline':<22s} {'R²':>12s} {'Pearson r':>12s} {'MAE':>12s}  Input")
    print("-" * 80)

    for bench in benchmarks:
        s = statuses.get(bench.name, {})
        m = s.get("metrics", {}) or {}
        r2 = f"{m['r2_mean']:.3f}±{m['r2_std']:.3f}" if m.get("r2_mean") is not None else "—"
        pr = f"{m['pearson_r_mean']:.3f}±{m['pearson_r_std']:.3f}" if m.get("pearson_r_mean") is not None else "—"
        mae = f"{m['mae_mean']:.3f}±{m['mae_std']:.3f}" if m.get("mae_mean") is not None else "—"
        inp = bench.input_description[:35]
        print(f"{bench.name:<22s} {r2:>12s} {pr:>12s} {mae:>12s}  {inp}")
    print()


if __name__ == "__main__":
    main()
