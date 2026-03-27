"""
Extract and analyze HPO results from Ray Tune experiments.

Parses Ray Tune trial directories, ranks trials by metric, exports top-K
configs as ready-to-train YAML files, and computes HP importance.

Usage:
    # Parse and extract top 10 configs (no GPU needed)
    uv run python scripts/extract_hpo_results.py \\
        --ray-dir outputs/ray_results/cognitive_resilience_hpo6/ \\
        --base-config configs/hpo_round6.yaml \\
        --top-k 10 \\
        --experiment-name hpo6 \\
        --nickname 2branch_concat

Output structure:
    outputs/hpo_analysis/{date}_{experiment_name}_{nickname}/
    ├── summary_table.csv          # All trials × all epochs
    ├── hp_importance.json         # HP importance ranking
    ├── top_configs/               # Top-K ready-to-train configs
    │   ├── rank01_trial_XXXX.yaml
    │   └── ...
    └── report.txt                 # Console summary
"""

import argparse
import json
import logging
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ray result parsing
# ---------------------------------------------------------------------------

def parse_ray_results(ray_dir: str | Path) -> list[dict]:
    """Parse all trial directories under a Ray Tune experiment directory.

    Each trial directory contains result.json (one JSON line per epoch)
    and params.json (trial hyperparameters).

    Args:
        ray_dir: Path to Ray Tune experiment directory.

    Returns:
        List of trial dicts, each containing:
        - trial_id: str
        - best_val_nll: float (minimum val_nll across epochs)
        - best_epoch: int (epoch where best_val_nll was achieved)
        - total_epochs: int
        - total_time_s: float
        - config: dict of sampled hyperparameters
        - epoch_history: list of per-epoch dicts [{epoch, val_nll, time_s, ...}]
    """
    ray_dir = Path(ray_dir)
    trials = []

    for trial_dir in sorted(ray_dir.iterdir()):
        if not trial_dir.is_dir() or not trial_dir.name.startswith("train_fn_"):
            continue

        result_file = trial_dir / "result.json"
        if not result_file.exists():
            logger.warning("No result.json in %s — skipping", trial_dir.name)
            continue

        # Parse result.json (newline-delimited JSON, one line per epoch)
        epoch_history = []
        config = {}
        trial_id = None

        with open(result_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if trial_id is None:
                    trial_id = record.get("trial_id", trial_dir.name.split("_")[2])
                if not config and "config" in record:
                    config = record["config"]

                epoch_entry = {
                    "epoch": record.get("training_iteration", 0),
                    "val_nll": record.get("val_nll", float("inf")),
                    "time_total_s": record.get("time_total_s", 0.0),
                    "time_this_iter_s": record.get("time_this_iter_s", 0.0),
                }
                # Include additional metrics if present (future HPO runs)
                for metric in ("val_r2", "val_pearson_r", "val_spearman_rho", "val_rmse"):
                    if metric in record:
                        epoch_entry[metric] = record[metric]

                epoch_history.append(epoch_entry)

        if not epoch_history:
            continue

        # Find best epoch by val_nll
        best_entry = min(epoch_history, key=lambda e: e["val_nll"])

        trials.append({
            "trial_id": trial_id,
            "best_val_nll": best_entry["val_nll"],
            "best_epoch": best_entry["epoch"],
            "total_epochs": len(epoch_history),
            "total_time_s": epoch_history[-1]["time_total_s"],
            "config": config,
            "epoch_history": epoch_history,
            "trial_dir": str(trial_dir),
        })

    logger.info("Parsed %d trials from %s", len(trials), ray_dir)
    return trials


def _filter_current_experiment(trials: list[dict], ray_dir: str | Path) -> list[dict]:
    """Filter trials to only include those from the most recent experiment.

    Ray Tune reuses the experiment directory across runs. The most recent
    experiment state file indicates the current run's start time. Only
    trial directories created after that time belong to the current run.

    Args:
        trials: All parsed trials.
        ray_dir: Ray Tune experiment directory.

    Returns:
        Filtered list of trials from the most recent experiment.
    """
    ray_dir = Path(ray_dir)
    # Find latest experiment state file
    state_files = sorted(ray_dir.glob("experiment_state-*.json"))
    if not state_files:
        return trials

    latest_state = state_files[-1]
    # Extract timestamp from filename: experiment_state-YYYY-MM-DD_HH-MM-SS.json
    ts_str = latest_state.stem.replace("experiment_state-", "")

    # Only keep trials whose directory name contains a timestamp >= experiment start.
    # Lexicographic comparison works because the YYYY-MM-DD_HH-MM-SS format is
    # monotonically ordered when compared as strings.
    filtered = []
    for t in trials:
        trial_dir_name = Path(t["trial_dir"]).name
        # Trial dir format: train_fn_<id>_<num>_<params>_YYYY-MM-DD_HH-MM-SS
        ts_match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})$', trial_dir_name)
        if ts_match:
            trial_ts = ts_match.group(1)
            if trial_ts >= ts_str:
                filtered.append(t)
        else:
            filtered.append(t)  # Can't parse timestamp, include it

    logger.info("Filtered to %d trials from latest experiment (started %s)",
                len(filtered), ts_str)
    return filtered


# ---------------------------------------------------------------------------
# Ranking and analysis
# ---------------------------------------------------------------------------

def rank_trials(
    trials: list[dict],
    metric: str = "val_nll",
    mode: str = "min",
    top_k: int | None = None,
) -> list[dict]:
    """Rank trials by best metric value.

    Args:
        trials: Parsed trial list from parse_ray_results().
        metric: Metric key to rank by (must be "val_nll" or present in epoch_history).
        mode: "min" or "max".
        top_k: Return only top K trials. None returns all.

    Returns:
        Ranked list of trials (best first), with "rank" field added.
    """
    key_fn = lambda t: t[f"best_{metric}"]
    reverse = mode == "max"
    ranked = sorted(trials, key=key_fn, reverse=reverse)

    if top_k is not None:
        ranked = ranked[:top_k]

    for i, t in enumerate(ranked):
        t["rank"] = i + 1

    return ranked


def build_summary_table(trials: list[dict]) -> pd.DataFrame:
    """Build a summary DataFrame with all trials x all epochs.

    Each row is one trial-epoch. HP columns are constant within a trial.

    Args:
        trials: Parsed trial list from parse_ray_results().

    Returns:
        DataFrame with columns: trial_id, epoch, val_nll, [additional metrics],
        lr, dropout, weight_decay, beta, guide_lr, tau_min, anneal_epochs,
        gene_gate_temp, total_time_s, best_val_nll, best_epoch, total_epochs.
    """
    rows = []
    for t in trials:
        config = t["config"]
        for entry in t["epoch_history"]:
            row = {
                "trial_id": t["trial_id"],
                "epoch": entry["epoch"],
                "val_nll": entry["val_nll"],
                "time_total_s": entry["time_total_s"],
                "time_this_epoch_s": entry["time_this_iter_s"],
                # Per-trial summary fields (repeated per epoch for easy filtering)
                "best_val_nll": t["best_val_nll"],
                "best_epoch": t["best_epoch"],
                "total_epochs": t["total_epochs"],
            }
            # Add additional metrics if present
            for metric in ("val_r2", "val_pearson_r", "val_spearman_rho", "val_rmse"):
                if metric in entry:
                    row[metric] = entry[metric]
            # Add HP columns (dynamically from config keys)
            for hp_name in sorted(config.keys()):
                row[hp_name] = config.get(hp_name)
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def compute_hp_importance(
    trials: list[dict],
    metric: str = "val_nll",
) -> dict[str, float]:
    """Compute HP importance via absolute Spearman correlation with best metric.

    Simple but robust: |Spearman rho| between each HP and best_val_nll across
    trials. Normalized to [0, 1] by dividing by max absolute correlation.

    For more sophisticated analysis (fANOVA), use Optuna's built-in
    visualization tools on the study object.

    Args:
        trials: Parsed trial list.
        metric: Metric to correlate against.

    Returns:
        Dict mapping HP name to importance score in [0, 1].
    """
    from scipy.stats import spearmanr

    hp_names = sorted(trials[0]["config"].keys()) if trials else []

    # Spearman correlation is meaningless with fewer than 3 data points
    if len(trials) < 3:
        return {hp: 0.0 for hp in hp_names}

    metric_key = f"best_{metric}"
    metric_values = np.array([t[metric_key] for t in trials])
    importance = {}

    for hp in hp_names:
        hp_values = np.array([t["config"].get(hp, 0) for t in trials])
        # Skip constant HPs
        if np.std(hp_values) < 1e-12:
            importance[hp] = 0.0
            continue
        rho, _ = spearmanr(hp_values, metric_values)
        importance[hp] = abs(rho) if not np.isnan(rho) else 0.0

    # Normalize to [0, 1]
    max_imp = max(importance.values()) if importance else 1.0
    if max_imp > 0:
        importance = {k: v / max_imp for k, v in importance.items()}

    # Sort by importance descending
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    return importance


# ---------------------------------------------------------------------------
# Config export
# ---------------------------------------------------------------------------

def export_top_configs(
    ranked_trials: list[dict],
    base_config_path: str,
    output_dir: str | Path,
) -> list[Path]:
    """Export top-K trial configs as full YAML files ready for training.

    Uses build_config_from_ray() from hpo.py to correctly map flat HP names
    to nested config paths.

    Args:
        ranked_trials: Ranked trial list (must have "rank" field).
        base_config_path: Path to base config YAML.
        output_dir: Directory to write YAML files.

    Returns:
        List of paths to exported YAML files.
    """
    from scripts.hpo import build_config_from_ray

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_config = OmegaConf.load(base_config_path)
    exported = []

    for trial in ranked_trials:
        config = build_config_from_ray(trial["config"], base_config)
        if config is None:
            logger.warning("Trial %s has invalid HP combo — skipping export",
                           trial["trial_id"])
            continue

        # Add provenance metadata
        OmegaConf.update(config, "_hpo_provenance.trial_id", trial["trial_id"])
        OmegaConf.update(config, "_hpo_provenance.best_val_nll", float(trial["best_val_nll"]))
        OmegaConf.update(config, "_hpo_provenance.best_epoch", int(trial["best_epoch"]))
        OmegaConf.update(config, "_hpo_provenance.rank", int(trial["rank"]))

        filename = f"rank{trial['rank']:02d}_trial_{trial['trial_id']}.yaml"
        out_path = output_dir / filename
        OmegaConf.save(config, out_path)
        exported.append(out_path)
        logger.info("Exported rank %d config: %s", trial["rank"], out_path)

    return exported


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    ranked_trials: list[dict],
    importance: dict[str, float],
    output_path: str | Path | None = None,
) -> str:
    """Generate human-readable summary report.

    Args:
        ranked_trials: Ranked trial list.
        importance: HP importance dict.
        output_path: Optional path to write report. If None, only returns string.

    Returns:
        Report string.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("HPO Results Summary")
    lines.append("=" * 70)
    lines.append(f"Total trials: {len(ranked_trials)}")
    lines.append("")

    # Top 10 table
    lines.append("Top Trials (by val_nll):")
    lines.append("-" * 70)
    header = f"{'Rank':<6} {'Trial ID':<12} {'val_nll':<10} {'Epoch':<8} {'Total Ep':<10} {'Time (s)':<10}"
    lines.append(header)
    lines.append("-" * 70)
    for t in ranked_trials[:10]:
        line = (f"{t['rank']:<6} {t['trial_id']:<12} {t['best_val_nll']:<10.4f} "
                f"{t['best_epoch']:<8} {t['total_epochs']:<10} {t['total_time_s']:<10.0f}")
        lines.append(line)
    lines.append("")

    # HP importance
    lines.append("HP Importance (|Spearman rho| with val_nll, normalized):")
    lines.append("-" * 40)
    for hp, score in importance.items():
        bar = "#" * int(score * 20)
        lines.append(f"  {hp:<20} {score:.3f} {bar}")
    lines.append("")

    # Best trial config
    if ranked_trials:
        best = ranked_trials[0]
        lines.append(f"Best Trial Config (rank 1, trial {best['trial_id']}):")
        lines.append("-" * 40)
        for hp, val in sorted(best["config"].items()):
            lines.append(f"  {hp:<20} {val}")

    report = "\n".join(lines)

    if output_path:
        Path(output_path).write_text(report)
        logger.info("Report saved to %s", output_path)

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract and analyze HPO results from Ray Tune experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ray-dir", type=str, required=True,
        help="Path to Ray Tune experiment directory",
    )
    parser.add_argument(
        "--base-config", type=str, required=True,
        help="Path to base config YAML (used to reconstruct full configs)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/hpo_analysis/",
        help="Base output directory (default: outputs/hpo_analysis/)",
    )
    parser.add_argument(
        "--top-k", type=int, default=10,
        help="Number of top configs to extract (default: 10)",
    )
    parser.add_argument(
        "--metric", type=str, default="val_nll",
        help="Metric to rank by (default: val_nll)",
    )
    parser.add_argument(
        "--mode", type=str, default="min", choices=["min", "max"],
        help="Optimization direction (default: min)",
    )
    parser.add_argument(
        "--experiment-name", type=str, default=None,
        help="Experiment name (e.g., hpo6). Inferred from ray-dir if omitted.",
    )
    parser.add_argument(
        "--nickname", type=str, default=None,
        help="Optional nickname suffix (e.g., 2branch_concat)",
    )
    parser.add_argument(
        "--filter-latest", action="store_true", default=True,
        help="Only include trials from the most recent experiment run (default: True)",
    )
    parser.add_argument(
        "--no-filter-latest", action="store_false", dest="filter_latest",
        help="Include all trials in the directory (including from previous runs)",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()

    # Parse trials
    trials = parse_ray_results(args.ray_dir)
    if not trials:
        logger.error("No trials found in %s", args.ray_dir)
        sys.exit(1)

    # Filter to latest experiment if requested
    if args.filter_latest:
        trials = _filter_current_experiment(trials, args.ray_dir)

    # Rank trials (call once to avoid overwriting rank values on shared dicts)
    ranked = rank_trials(trials, metric=args.metric, mode=args.mode)
    top_k = ranked[:args.top_k]

    # Build output directory name: {date}_{experiment_name}_{nickname}
    exp_name = args.experiment_name
    if exp_name is None:
        exp_name = Path(args.ray_dir).name  # e.g., "cognitive_resilience_hpo6"
    dir_parts = [date.today().isoformat(), exp_name]
    if args.nickname:
        dir_parts.append(args.nickname)
    subdir = "_".join(dir_parts)
    output_dir = Path(args.output_dir) / subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Summary table (all trials × all epochs)
    summary_df = build_summary_table(ranked)
    summary_path = output_dir / "summary_table.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info("Summary table (%d rows) saved to %s", len(summary_df), summary_path)

    # HP importance
    importance = compute_hp_importance(ranked, metric=args.metric)
    importance_path = output_dir / "hp_importance.json"
    with open(importance_path, "w") as f:
        json.dump(importance, f, indent=2)
    logger.info("HP importance saved to %s", importance_path)

    # Export top-K configs
    configs_dir = output_dir / "top_configs"
    export_top_configs(top_k, args.base_config, configs_dir)

    # Generate and print report
    report = generate_report(ranked, importance, output_dir / "report.txt")
    print(report)

    logger.info("All outputs saved to %s", output_dir)


if __name__ == "__main__":
    main()
