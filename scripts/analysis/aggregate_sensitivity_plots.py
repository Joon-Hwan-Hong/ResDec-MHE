"""
Aggregate sensitivity-analysis plots across configs × folds × seeds.

Discovers (config, fold) → experiment-dir by parsing logs in
outputs/logs/sensitivity{,_seed43..46}/, loads each run's val
predictions, and produces the 7 presentation-quality plots below.

Output layout (layout C — timestamped run subdir with `latest` symlink):
    outputs/plots/sensitivity/<data-date>/runs/<YYYYMMDD_HHMMSS>/
        ablation_bar.png               (1) 9-config R² bar chart
        per_fold_strip.png             (2) ablation bars with per-fold dots
        seed_sensitivity.png           (3) 5 seeds × 5 folds R² dots
        hpo_top5.png                   (4) HPO top-5 comparison
        pred_vs_actual_stacked.png     (5) 5-fold stacked scatter (production config)
        residual_violin.png            (6) residual distribution per ablation
        loss_curves_overlay/<config>.png (7) per-config 5-fold loss overlay
        per_run_metrics.csv            raw per-run metrics
        aggregated_metrics.csv         per-(seed, config) aggregate
        manifest.json / MANIFEST.md    provenance (git SHA, input SHA256, …)
    outputs/plots/sensitivity/<data-date>/latest → runs/<latest-ts>

Usage:
    uv run python scripts/analysis/aggregate_sensitivity_plots.py
    uv run python scripts/analysis/aggregate_sensitivity_plots.py --data-date 2026-03-30
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score

from src.utils.manifest import (
    FileRef,
    build_manifest,
    file_ref,
    write_manifest,
)
from src.visualization import (
    ACCENT_CORAL,
    ACCENT_PEACH,
    ACCENT_TEAL,
    save_figure,
    setup_seaborn_style,
)
from src.visualization.training_curves import load_tensorboard_scalars


logger = logging.getLogger(__name__)


# Config groupings (source: docs/results/2026-03-30-hpo7-ablation-interpretability.md)
ABLATION_CONFIGS: dict[str, str] = {
    "rank03_trial_b67416f2": "Full (concat_norm)",
    "ablation_no_gene_gate": "No gene gate",
    "ablation_ct_only": "CT only",
    "ablation_hgt_only": "HGT only",
    "ablation_fusion_crossfuse_blend": "Crossfuse blend",
    "ablation_fusion_cross_attention": "Cross-attention",
    "rank03_plain_concat": "Plain concat",
    "ablation_fusion_crossfuse": "Crossfuse",
    "ablation_no_pathology_attention": "No pathology attn",
}

HPO_TOP_CONFIGS: dict[str, str] = {
    "rank01_trial_98377757": "Rank 1",
    "rank02_trial_4049af7f": "Rank 2",
    "rank03_trial_b67416f2": "Rank 3 (prod)",
    "rank04_trial_24750fd3": "Rank 4",
    "rank05_trial_9b37258c": "Rank 5",
    "rank03_plain_concat": "Rank 3 plain concat",
}

PRODUCTION_CONFIG = "rank03_trial_b67416f2"

SEED_LOG_DIRS: dict[str, str] = {
    "seed42": "sensitivity",
    "seed43": "sensitivity_seed43",
    "seed44": "sensitivity_seed44",
    "seed45": "sensitivity_seed45",
    "seed46": "sensitivity_seed46",
}

EXPERIMENT_RE = re.compile(r"Experiment created: (20260\d+_\S+)")
LOG_NAME_RE = re.compile(r"^(.+)_fold(\d+)\.log$")


def discover_experiment_dirs(logs_root: Path, seed_dir_name: str) -> dict[str, dict[int, str]]:
    """Parse <logs_root>/<seed_dir_name>/*.log → {config_name: {fold: experiment_dir}}."""
    logs_dir = logs_root / seed_dir_name
    mapping: dict[str, dict[int, str]] = {}
    if not logs_dir.exists():
        return mapping
    for log in sorted(logs_dir.glob("*_fold*.log")):
        m = LOG_NAME_RE.match(log.name)
        if not m:
            continue
        config, fold = m.group(1), int(m.group(2))
        text = log.read_text()
        em = EXPERIMENT_RE.search(text)
        if not em:
            logger.warning("No Experiment-created line in %s", log)
            continue
        mapping.setdefault(config, {})[fold] = em.group(1)
    return mapping


def compute_val_metrics(pred_df: pd.DataFrame, val_ids: set[str]) -> dict | None:
    val = pred_df[pred_df["subject_id"].isin(val_ids)].copy()
    if len(val) < 2:
        return None
    actual = val["actual"].to_numpy()
    predicted = val["predicted_mean"].to_numpy()
    return {
        "r2": float(r2_score(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "pearson": float(stats.pearsonr(actual, predicted)[0]),
        "spearman": float(stats.spearmanr(actual, predicted)[0]),
        "n_val": int(len(val)),
        "val_df": val,
    }


def load_sensitivity_data(outputs_root: Path, logs_root: Path, splits: dict) -> dict:
    """Return {(seed, config, fold): {r2, rmse, pearson, spearman, n_val, val_df, experiment_dir}}."""
    results: dict = {}
    for seed, seed_dir in SEED_LOG_DIRS.items():
        discovered = discover_experiment_dirs(logs_root, seed_dir)
        for config, folds in discovered.items():
            for fold, exp in folds.items():
                parquet = outputs_root / exp / "analysis" / "predictions.parquet"
                if not parquet.exists():
                    logger.warning("Missing predictions: %s/%s/fold%d", seed, config, fold)
                    continue
                pred = pd.read_parquet(parquet)
                val_ids = set(splits["folds"][fold]["val"])
                metrics = compute_val_metrics(pred, val_ids)
                if metrics is None:
                    continue
                metrics["experiment_dir"] = exp
                results[(seed, config, fold)] = metrics
    return results


# =============================================================================
# Plots
# =============================================================================


def _filter_seed42_group(results: dict, group: dict[str, str]) -> pd.DataFrame:
    rows = [
        {"config": config, "label": group[config], "fold": fold, "r2": m["r2"]}
        for (seed, config, fold), m in results.items()
        if seed == "seed42" and config in group
    ]
    return pd.DataFrame(rows)


def plot_ablation_bar(results: dict, output_dir: Path) -> pd.DataFrame | None:
    df = _filter_seed42_group(results, ABLATION_CONFIGS)
    if df.empty:
        logger.warning("No ablation data found")
        return None
    agg = df.groupby("label")["r2"].agg(["mean", "std"]).reset_index().sort_values("mean", ascending=False)

    colors = []
    for lbl in agg["label"]:
        if lbl.startswith("Full"):
            colors.append(ACCENT_PEACH)
        elif "No pathology" in lbl:
            colors.append(ACCENT_CORAL)
        else:
            colors.append(ACCENT_TEAL)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(
        range(len(agg)), agg["mean"], yerr=agg["std"], capsize=5,
        color=colors, alpha=0.85, edgecolor="black", linewidth=0.5,
    )
    ax.set_xticks(range(len(agg)))
    ax.set_xticklabels(agg["label"], rotation=30, ha="right")
    ax.set_ylabel("Val R² (5-fold mean ± std)")
    ax.set_title("Ablation study (HPO7 rank03 HPs, seed 42)")
    ax.axhline(0, color="black", linewidth=0.5)
    for i, (mean, std) in enumerate(zip(agg["mean"], agg["std"])):
        ax.text(i, mean + std + 0.01, f"{mean:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    save_figure(fig, str(output_dir / "ablation_bar.png"), dpi=200)
    plt.close(fig)
    return agg


def plot_per_fold_strip(results: dict, output_dir: Path) -> None:
    df = _filter_seed42_group(results, ABLATION_CONFIGS)
    if df.empty:
        return
    order = df.groupby("label")["r2"].mean().sort_values(ascending=False).index.tolist()

    fig, ax = plt.subplots(figsize=(11, 6))
    rng = np.random.default_rng(42)
    for i, lbl in enumerate(order):
        vals = df[df["label"] == lbl]["r2"].to_numpy()
        mean, std = float(vals.mean()), float(vals.std())
        ax.bar(i, mean, yerr=std, capsize=5, color=ACCENT_TEAL, alpha=0.45,
               edgecolor="black", linewidth=0.5, width=0.7)
        jitter = rng.uniform(-0.15, 0.15, len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter, vals,
            color=ACCENT_CORAL, s=45, alpha=0.9, zorder=3,
            edgecolor="black", linewidth=0.5,
        )
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_ylabel("Val R²")
    ax.set_title("Ablation study — per-fold values (dots) with 5-fold mean (bars)")
    ax.axhline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    save_figure(fig, str(output_dir / "per_fold_strip.png"), dpi=200)
    plt.close(fig)


def plot_seed_sensitivity(results: dict, output_dir: Path) -> None:
    rows = [
        {"seed": seed, "fold": fold, "r2": m["r2"]}
        for (seed, config, fold), m in results.items()
        if config == PRODUCTION_CONFIG
    ]
    df = pd.DataFrame(rows)
    if df.empty:
        return
    seed_order = sorted(df["seed"].unique())

    fig, ax = plt.subplots(figsize=(9, 6))
    rng = np.random.default_rng(0)
    for i, seed in enumerate(seed_order):
        vals = df[df["seed"] == seed]["r2"].to_numpy()
        jitter = rng.uniform(-0.12, 0.12, len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter, vals, s=60, alpha=0.85,
            color=ACCENT_TEAL, edgecolor="black", linewidth=0.5,
        )
        ax.scatter(i, vals.mean(), marker="_", s=600, color=ACCENT_CORAL, linewidth=3)

    overall_mean = float(df["r2"].mean())
    seed_level_std = float(df.groupby("seed")["r2"].mean().std())
    ax.axhline(overall_mean, linestyle="--", color="gray", alpha=0.6)
    ax.set_xticks(range(len(seed_order)))
    ax.set_xticklabels(seed_order)
    ax.set_ylabel("Val R² per fold")
    ax.set_xlabel("Random seed")
    ax.set_title(
        f"Seed sensitivity ({PRODUCTION_CONFIG}) — "
        f"R² = {overall_mean:.3f} (seed-level std = {seed_level_std:.3f})"
    )
    plt.tight_layout()
    save_figure(fig, str(output_dir / "seed_sensitivity.png"), dpi=200)
    plt.close(fig)


def plot_hpo_top_comparison(results: dict, output_dir: Path) -> None:
    df = _filter_seed42_group(results, HPO_TOP_CONFIGS)
    if df.empty:
        return
    label_order = [HPO_TOP_CONFIGS[c] for c in HPO_TOP_CONFIGS if HPO_TOP_CONFIGS[c] in df["label"].unique()]
    agg = df.groupby("label")["r2"].agg(["mean", "std"]).reindex(label_order).dropna()

    colors = [ACCENT_PEACH if lbl == "Rank 3 (prod)" else ACCENT_TEAL for lbl in agg.index]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        range(len(agg)), agg["mean"], yerr=agg["std"], capsize=5,
        color=colors, alpha=0.85, edgecolor="black", linewidth=0.5,
    )
    ax.set_xticks(range(len(agg)))
    ax.set_xticklabels(agg.index, rotation=25, ha="right")
    ax.set_ylabel("Val R² (5-fold mean ± std)")
    ax.set_title("HPO Top-5 comparison (seed 42)")
    for i, (mean, std) in enumerate(zip(agg["mean"], agg["std"])):
        ax.text(i, mean + std + 0.005, f"{mean:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    save_figure(fig, str(output_dir / "hpo_top5.png"), dpi=200)
    plt.close(fig)


def plot_pred_vs_actual_stacked(
    results: dict, output_dir: Path,
    config: str = PRODUCTION_CONFIG, seed: str = "seed42",
) -> None:
    parts = []
    for fold in range(5):
        key = (seed, config, fold)
        if key not in results:
            continue
        vdf = results[key]["val_df"].copy()
        vdf["fold"] = fold
        parts.append(vdf)
    if not parts:
        return
    stacked = pd.concat(parts, ignore_index=True)
    actual = stacked["actual"].to_numpy()
    predicted = stacked["predicted_mean"].to_numpy()

    r2 = float(r2_score(actual, predicted))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    r = float(stats.pearsonr(actual, predicted)[0])

    fig, ax = plt.subplots(figsize=(7, 7))
    cmap = plt.get_cmap("tab10")
    for fold in range(5):
        f = stacked[stacked["fold"] == fold]
        if len(f) == 0:
            continue
        ax.scatter(
            f["actual"], f["predicted_mean"],
            color=cmap(fold), s=28, alpha=0.7,
            label=f"Fold {fold}", edgecolor="white", linewidth=0.3,
        )
    lo = float(min(actual.min(), predicted.min()))
    hi = float(max(actual.max(), predicted.max()))
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.6, label="y = x")
    ax.set_xlabel("Actual cogn_global")
    ax.set_ylabel("Predicted (mean)")
    ax.set_title(f"Predicted vs actual — 5 folds stacked ({config}, {seed})")
    ax.text(
        0.05, 0.95,
        f"R² = {r2:.3f}\nRMSE = {rmse:.3f}\nPearson r = {r:.3f}\nN = {len(stacked)}",
        transform=ax.transAxes, ha="left", va="top", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
    )
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    save_figure(fig, str(output_dir / "pred_vs_actual_stacked.png"), dpi=200)
    plt.close(fig)


def plot_residual_violin(results: dict, output_dir: Path) -> None:
    rows = []
    for (seed, config, fold), m in results.items():
        if seed != "seed42" or config not in ABLATION_CONFIGS:
            continue
        vdf = m["val_df"]
        resid = vdf["actual"].to_numpy() - vdf["predicted_mean"].to_numpy()
        for r in resid:
            rows.append({"label": ABLATION_CONFIGS[config], "residual": float(r)})
    df = pd.DataFrame(rows)
    if df.empty:
        return

    # Sort by ablation-bar ordering (R² mean descending) so visuals match the bar chart
    r2_df = _filter_seed42_group(results, ABLATION_CONFIGS)
    order = r2_df.groupby("label")["r2"].mean().sort_values(ascending=False).index.tolist()

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.violinplot(data=df, x="label", y="residual", ax=ax, order=order,
                   color=ACCENT_TEAL, inner="box")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_ylabel("Residual (actual − predicted)")
    ax.set_xlabel("")
    ax.set_title("Residual distribution per ablation (5-fold val sets pooled)")
    plt.tight_layout()
    save_figure(fig, str(output_dir / "residual_violin.png"), dpi=200)
    plt.close(fig)


def plot_loss_curves_overlay(results: dict, output_dir: Path, outputs_root: Path) -> None:
    overlay_dir = output_dir / "loss_curves_overlay"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    configs = sorted({
        config for (seed, config, _), _ in results.items()
        if seed == "seed42" and config in ABLATION_CONFIGS
    })
    cmap = plt.get_cmap("tab10")

    for config in configs:
        fig, ax = plt.subplots(figsize=(10, 6))
        plotted = 0
        for fold in range(5):
            key = ("seed42", config, fold)
            if key not in results:
                continue
            exp = results[key]["experiment_dir"]
            tb_dir = outputs_root / exp / "logs" / "tensorboard" / "cognitive_resilience_hpo7" / "version_0"
            if not tb_dir.exists():
                continue
            df = load_tensorboard_scalars(tb_dir)
            if df is None:
                continue
            wide = df.pivot_table(index="step", columns="tag", values="value").reset_index()
            color = cmap(fold)
            if "train_loss" in wide.columns:
                train = wide[["step", "train_loss"]].dropna()
                ax.plot(train["step"], train["train_loss"], color=color, alpha=0.55,
                        linewidth=1.2, label=f"train fold{fold}")
            if "val_loss" in wide.columns:
                val = wide[["step", "val_loss"]].dropna()
                ax.plot(val["step"], val["val_loss"], color=color, alpha=1.0,
                        linewidth=1.6, linestyle="--", label=f"val fold{fold}")
            plotted += 1
        if plotted == 0:
            plt.close(fig)
            continue
        ax.set_xlabel("Global step")
        ax.set_ylabel("Loss")
        ax.set_title(f"Loss curves — {ABLATION_CONFIGS.get(config, config)} (5 folds)")
        ax.legend(loc="upper right", fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_figure(fig, str(overlay_dir / f"{config}.png"), dpi=150)
        plt.close(fig)


def write_summary_csv(results: dict, output_dir: Path) -> None:
    rows = [
        {
            "seed": seed, "config": config, "fold": fold,
            "experiment_dir": m["experiment_dir"],
            "r2": m["r2"], "rmse": m["rmse"],
            "pearson": m["pearson"], "spearman": m["spearman"],
            "n_val": m["n_val"],
        }
        for (seed, config, fold), m in sorted(results.items())
    ]
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "per_run_metrics.csv", index=False)

    agg = df.groupby(["seed", "config"]).agg(
        r2_mean=("r2", "mean"), r2_std=("r2", "std"),
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        pearson_mean=("pearson", "mean"), pearson_std=("pearson", "std"),
        spearman_mean=("spearman", "mean"), spearman_std=("spearman", "std"),
        n_folds=("fold", "count"),
    ).reset_index()
    agg.to_csv(output_dir / "aggregated_metrics.csv", index=False)


def _update_latest_symlink(data_dir: Path, run_dir: Path) -> None:
    """Point <data_dir>/latest at the newest run."""
    latest = data_dir / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(run_dir.relative_to(data_dir))
    except OSError as e:
        logger.warning("Could not update 'latest' symlink: %s", e)


def _collect_outputs(run_dir: Path) -> list[FileRef]:
    refs: list[FileRef] = []
    for png in sorted(run_dir.rglob("*.png")):
        refs.append(file_ref(png, label=str(png.relative_to(run_dir)), compute_sha=False))
    for csv in sorted(run_dir.glob("*.csv")):
        refs.append(file_ref(csv, label=csv.name, compute_sha=False))
    return refs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outputs-root", default="outputs", type=str)
    parser.add_argument("--splits-path", default="outputs/splits.json", type=str)
    parser.add_argument("--data-date", default="2026-03-30", type=str,
                        help="Date label for the sensitivity-run data being plotted")
    parser.add_argument("--output-dir", default=None, type=str,
                        help="Override full output dir (skips layout-C timestamping)")
    parser.add_argument("--skip-tb-overlay", action="store_true",
                        help="Skip plot 7 (tensorboard loss overlay) — slow on many runs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    setup_seaborn_style()

    outputs_root = Path(args.outputs_root).resolve()
    logs_root = outputs_root / "logs"
    run_ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d_%H%M%S")

    if args.output_dir is not None:
        run_dir = Path(args.output_dir).resolve()
        data_dir = run_dir.parent
    else:
        data_dir = (outputs_root / "plots" / "sensitivity" / args.data_date).resolve()
        run_dir = data_dir / "runs" / run_ts
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(args.splits_path) as f:
        splits = json.load(f)

    logger.info("Output dir: %s", run_dir)
    logger.info("Loading sensitivity data…")
    results = load_sensitivity_data(outputs_root, logs_root, splits)
    logger.info("Loaded %d (seed, config, fold) combinations", len(results))

    logger.info("[1/7] Ablation bar chart")
    plot_ablation_bar(results, run_dir)
    logger.info("[2/7] Per-fold strip overlay")
    plot_per_fold_strip(results, run_dir)
    logger.info("[3/7] Seed sensitivity")
    plot_seed_sensitivity(results, run_dir)
    logger.info("[4/7] HPO top-5 comparison")
    plot_hpo_top_comparison(results, run_dir)
    logger.info("[5/7] Pred-vs-actual stacked")
    plot_pred_vs_actual_stacked(results, run_dir)
    logger.info("[6/7] Residual violin")
    plot_residual_violin(results, run_dir)
    if args.skip_tb_overlay:
        logger.info("[7/7] Skipped (--skip-tb-overlay)")
    else:
        logger.info("[7/7] Loss curve overlay (tensorboard)")
        plot_loss_curves_overlay(results, run_dir, outputs_root)

    logger.info("Writing summary CSVs")
    write_summary_csv(results, run_dir)

    # Manifest: SHA256 every input predictions.parquet + splits.json
    logger.info("Hashing inputs for manifest…")
    splits_ref = file_ref(args.splits_path, label="splits.json", compute_sha=True)
    input_refs: list[FileRef] = [splits_ref]
    per_run_entries: list[dict] = []
    for (seed, config, fold), m in sorted(results.items()):
        parquet = outputs_root / m["experiment_dir"] / "analysis" / "predictions.parquet"
        tb_dir = outputs_root / m["experiment_dir"] / "logs" / "tensorboard" / "cognitive_resilience_hpo7" / "version_0"
        tb_events = sorted(tb_dir.glob("events.out.tfevents.*"))
        label = f"{seed}/{config}/fold{fold}/predictions.parquet"
        input_refs.append(file_ref(parquet, label=label, compute_sha=True))
        per_run_entries.append({
            "seed": seed, "config": config, "fold": fold,
            "experiment_dir": m["experiment_dir"],
            "predictions_parquet": str(parquet),
            "tensorboard_events": [str(p) for p in tb_events],
            "val_r2": m["r2"], "val_rmse": m["rmse"],
            "val_pearson": m["pearson"], "val_spearman": m["spearman"],
            "n_val": m["n_val"],
        })

    output_refs = _collect_outputs(run_dir)

    warnings: list[str] = []
    seed_rows = [e for e in per_run_entries if e["config"] == PRODUCTION_CONFIG]
    if seed_rows:
        seed_groups = pd.DataFrame(seed_rows).groupby("seed")["val_r2"].mean()
        seed_std = float(seed_groups.std())
        if seed_std > 0.015:
            warnings.append(
                f"Seed-level R² std is {seed_std:.3f} (doc reports 0.009). "
                "On-disk predictions.parquet files for seeds 43-46 appear to differ "
                "from the values in docs/results/2026-03-30-hpo7-ablation-interpretability.md — "
                "likely from a later re-inference pass. Plots reflect current on-disk state."
            )

    manifest = build_manifest(
        title=f"Sensitivity plot aggregation — data-date {args.data_date}",
        description=(
            "Aggregate R² bars, per-fold strips, seed sensitivity, HPO top-5, stacked "
            "predicted-vs-actual, residual violins, and per-config loss overlays from the "
            "sensitivity analysis runs under outputs/logs/sensitivity*."
        ),
        script_path=Path(__file__),
        argv=sys.argv,
        config={
            "outputs_root": str(outputs_root),
            "splits_path": args.splits_path,
            "data_date": args.data_date,
            "skip_tb_overlay": args.skip_tb_overlay,
            "ablation_configs": ABLATION_CONFIGS,
            "hpo_top_configs": HPO_TOP_CONFIGS,
            "production_config": PRODUCTION_CONFIG,
            "seed_log_dirs": SEED_LOG_DIRS,
            "n_runs_loaded": len(results),
        },
        inputs=input_refs,
        outputs=output_refs,
        warnings=warnings,
        extras={"per_run": per_run_entries},
    )
    write_manifest(run_dir, manifest)

    if args.output_dir is None:
        _update_latest_symlink(data_dir, run_dir)
        logger.info("Updated symlink: %s -> %s", data_dir / "latest", run_dir.relative_to(data_dir))

    logger.info("Done. Outputs in %s", run_dir)


if __name__ == "__main__":
    main()
