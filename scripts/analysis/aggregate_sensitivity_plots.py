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

# Baseline R² from memory (2026-03-20 5-fold run). NOT VERIFIED on-disk in
# the current main-repo state (2026-04-21). Baselines were computed against
# the HPO4 model (R²=0.323) whereas the "our model" number plotted here is
# the HPO7 re-run (0.304). Mixing them understates our model relative to its
# HPO4 ancestor. See manifest warnings.
BASELINE_MEMORY_VALUES: dict[str, dict[str, float | str]] = {
    "Our model (HPO7 prod)": {"r2_mean": 0.304, "r2_std": 0.067, "source": "docs/results/2026-03-30-hpo7-ablation-interpretability.md"},
    "Ridge (flat pseudobulk)": {"r2_mean": 0.290, "r2_std": 0.0, "source": "docs/hpo-log.md:281 — single number, no std reported"},
    "MixMIL": {"r2_mean": 0.110, "r2_std": 0.038, "source": "memory 2026-03-20 (UNVERIFIED on-disk)"},
    "scPhase": {"r2_mean": -0.059, "r2_std": 0.093, "source": "memory 2026-03-20 (UNVERIFIED on-disk)"},
}
BASELINES_WARNING = (
    "Baseline R² for MixMIL and scPhase are sourced from memory notes (2026-03-20) "
    "and could not be verified against result files in the current main repo. "
    "Ridge single value from docs/hpo-log.md:281 (no std). Our-model number is HPO7 "
    "rank03 (0.304) while baselines were run against HPO4 rank3 (0.323) — subtle "
    "mismatch. Confirm numbers before use in external presentations."
)


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


# =============================================================================
# Interpretability plot helpers (use pathology_attention from one canonical run)
# =============================================================================


def load_interpretability_data(
    results: dict, outputs_root: Path,
    config: str = PRODUCTION_CONFIG, fold: int = 0, seed: str = "seed42",
) -> dict | None:
    """Load {subject_ids, cell_type_names, pathology_attention (mean over heads), metadata} from one run."""
    import h5py

    key = (seed, config, fold)
    if key not in results:
        logger.warning("No run for %s/%s/fold%d — interpretability plots skipped", seed, config, fold)
        return None
    exp = results[key]["experiment_dir"]
    h5_path = outputs_root / exp / "analysis" / "attention_weights.h5"
    if not h5_path.exists():
        logger.warning("attention_weights.h5 missing at %s", h5_path)
        return None

    with h5py.File(h5_path, "r") as f:
        if "pathology_attention" not in f:
            logger.warning("pathology_attention missing in %s", h5_path)
            return None
        patho_attn = f["pathology_attention"][:]  # (N, 4 heads, 31 cell types)
        subj_ids = [s.decode() if isinstance(s, bytes) else s for s in f["subject_ids"][:]]
        ct_names = [s.decode() if isinstance(s, bytes) else s for s in f["cell_type_names"][:]]

    # Average across heads → (N, 31)
    attn = patho_attn.mean(axis=1)

    # Merge with predictions for metadata (gpath, actual cogn, predicted_mean, predicted_std)
    pred_path = outputs_root / exp / "analysis" / "predictions.parquet"
    pred = pd.read_parquet(pred_path)
    pred_by_id = pred.set_index("subject_id")
    common = [s for s in subj_ids if s in pred_by_id.index]
    if len(common) < len(subj_ids):
        logger.warning("Only %d/%d subjects have predictions", len(common), len(subj_ids))
    aligned = pred_by_id.reindex(subj_ids)

    return {
        "experiment_dir": exp,
        "subject_ids": subj_ids,
        "cell_type_names": ct_names,
        "attention": attn,  # (N, 31)
        "metadata": aligned,
    }


def plot_baseline_comparison(output_dir: Path) -> None:
    """Bar chart comparing our model vs Ridge, MixMIL, scPhase. Excludes XGBoost/RF per user request."""
    labels = list(BASELINE_MEMORY_VALUES.keys())
    means = np.array([BASELINE_MEMORY_VALUES[l]["r2_mean"] for l in labels])
    stds = np.array([BASELINE_MEMORY_VALUES[l]["r2_std"] for l in labels])

    colors = [ACCENT_PEACH if "Our model" in l else ACCENT_TEAL for l in labels]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(range(len(labels)), means, yerr=stds, capsize=5,
                  color=colors, alpha=0.9, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Val R² (5-fold mean ± std)")
    ax.set_title("Baseline comparison on 516 subjects (XGBoost + Random Forest excluded)")
    ax.axhline(0, color="black", linewidth=0.5)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.01, f"{m:.3f}", ha="center", fontsize=10)

    plt.tight_layout()
    save_figure(fig, str(output_dir / "baseline_comparison.png"), dpi=200)
    plt.close(fig)


def plot_cell_type_importance(interp: dict, output_dir: Path) -> None:
    """Horizontal bar: 31 cell types, X=mean pathology-attention, color = corr(attn, cogn)."""
    attn = interp["attention"]  # (N, 31)
    ct_names = interp["cell_type_names"]
    meta = interp["metadata"]
    cogn = meta["actual"].to_numpy()
    # correlation of each cell-type attention with cognition
    corrs = []
    for i in range(attn.shape[1]):
        mask = ~np.isnan(attn[:, i]) & ~np.isnan(cogn)
        if mask.sum() < 3:
            corrs.append(0.0)
        else:
            r, _ = stats.pearsonr(attn[mask, i], cogn[mask])
            corrs.append(float(r))
    mean_attn = attn.mean(axis=0)
    df = pd.DataFrame({"cell_type": ct_names, "mean_attn": mean_attn, "corr_cogn": corrs})
    df = df.sort_values("mean_attn", ascending=True)

    # Diverging color map — red=negative (pathology-associated), blue=positive (resilience-associated)
    vmax = max(abs(df["corr_cogn"].min()), abs(df["corr_cogn"].max()))
    norm = plt.Normalize(vmin=-vmax, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r")
    colors = [cmap(norm(c)) for c in df["corr_cogn"]]

    fig, ax = plt.subplots(figsize=(9, 10))
    ax.barh(df["cell_type"], df["mean_attn"], color=colors, edgecolor="black", linewidth=0.4)
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row["mean_attn"] + 0.002, i, f"r={row['corr_cogn']:+.2f}",
                va="center", fontsize=8, color="#333")
    ax.set_xlabel("Mean pathology-conditioned attention weight")
    ax.set_ylabel("")
    ax.set_title("Cell-type importance (pathology attention) — colored by r(attention, cogn_global)\n"
                 "Blue = resilience-associated (↑ in cognitively preserved), Red = pathology-associated")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Pearson r(attn, cogn)")
    plt.tight_layout()
    save_figure(fig, str(output_dir / "cell_type_importance.png"), dpi=200)
    plt.close(fig)


def plot_pathology_attention_scatter(interp: dict, output_dir: Path) -> None:
    """Scatter: pathology (gpath) vs attention weight for selected cell types."""
    attn = interp["attention"]  # (N, 31)
    ct_names = interp["cell_type_names"]
    meta = interp["metadata"]
    gpath = meta["gpath"].to_numpy()
    cogn = meta["actual"].to_numpy()

    # Target cell types (from the doc's Tier-1/Tier-2 analysis)
    targets = [
        "Upper_layer_intratelencephalic",
        "Astrocyte",
        "Oligodendrocyte",
        "Vascular",
        "Fibroblast",
        "LAMP5_LHX6_and_Chandelier",
    ]
    available = [t for t in targets if t in ct_names]
    if not available:
        logger.warning("None of the target cell types found in h5 — scatter skipped")
        return

    n = len(available)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False)
    for idx, ct in enumerate(available):
        ax = axes[idx // ncols][idx % ncols]
        col = ct_names.index(ct)
        x = gpath
        y = attn[:, col]
        mask = ~np.isnan(x) & ~np.isnan(y)
        x, y = x[mask], y[mask]

        sc = ax.scatter(x, y, c=cogn[mask], cmap="viridis", s=20, alpha=0.7,
                        edgecolor="white", linewidth=0.2)
        # Trend: ordinary least squares
        if len(x) >= 3:
            slope, intercept, r, p, _ = stats.linregress(x, y)
            xx = np.linspace(x.min(), x.max(), 50)
            ax.plot(xx, slope * xx + intercept, color="black", linewidth=1.2, alpha=0.8)
            ax.text(0.05, 0.95, f"r={r:+.2f}, p={p:.2g}",
                    transform=ax.transAxes, va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        pretty = ct.replace("_", " ")
        ax.set_title(pretty, fontsize=10)
        ax.set_xlabel("gpath")
        ax.set_ylabel("Attention weight")
    # Hide extra axes
    for idx in range(len(available), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    fig.suptitle("Cell-type attention vs pathology burden (color = cogn_global)", fontsize=12, y=1.00)
    # Colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.012, 0.7])
    fig.colorbar(sc, cax=cbar_ax, label="cogn_global")
    plt.tight_layout(rect=[0, 0, 0.92, 0.97])
    save_figure(fig, str(output_dir / "pathology_attention_scatter.png"), dpi=200)
    plt.close(fig)


def plot_cell_type_dotheatmap(interp: dict, output_dir: Path) -> None:
    """scanpy-style dotplot: rows=cell types, cols=pathology tertile; dot size=mean attn, color=corr(attn, cogn) within tertile."""
    attn = interp["attention"]
    ct_names = interp["cell_type_names"]
    meta = interp["metadata"]
    gpath = meta["gpath"].to_numpy()
    cogn = meta["actual"].to_numpy()

    # Tertile bins
    tert_labels = ["low-pathology", "mid-pathology", "high-pathology"]
    q = np.quantile(gpath, [1/3, 2/3])
    bins = np.digitize(gpath, q)  # 0, 1, 2

    n_ct = len(ct_names)
    size_mat = np.zeros((n_ct, 3))
    color_mat = np.zeros((n_ct, 3))
    for b in range(3):
        mask = bins == b
        if mask.sum() < 3:
            continue
        for i in range(n_ct):
            vals = attn[mask, i]
            c_vals = cogn[mask]
            good = ~np.isnan(vals) & ~np.isnan(c_vals)
            if good.sum() < 3:
                continue
            size_mat[i, b] = vals[good].mean()
            r, _ = stats.pearsonr(vals[good], c_vals[good])
            color_mat[i, b] = r

    # Sort cell types by overall attention
    order_idx = np.argsort(-size_mat.sum(axis=1))
    ct_order = [ct_names[i] for i in order_idx]
    size_mat = size_mat[order_idx]
    color_mat = color_mat[order_idx]

    fig, ax = plt.subplots(figsize=(6, 11))
    # Dot sizes — scale by max attention
    s_scale = 800.0 / (size_mat.max() + 1e-9)
    xs, ys, ss, cs = [], [], [], []
    for i in range(n_ct):
        for b in range(3):
            xs.append(b)
            ys.append(n_ct - 1 - i)
            ss.append(size_mat[i, b] * s_scale)
            cs.append(color_mat[i, b])
    vmax = max(abs(min(cs)), abs(max(cs))) if cs else 1.0
    sc = ax.scatter(xs, ys, s=ss, c=cs, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                    edgecolor="black", linewidth=0.3)
    ax.set_xticks(range(3))
    ax.set_xticklabels(tert_labels, rotation=20, ha="right")
    ax.set_yticks(range(n_ct))
    ax.set_yticklabels(list(reversed([c.replace("_", " ") for c in ct_order])), fontsize=8)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(-0.6, n_ct - 0.4)
    ax.set_title("Cell-type attention by pathology tertile\n(size = mean attn, color = r(attn, cogn) within tertile)")
    cbar = fig.colorbar(sc, ax=ax, shrink=0.4, pad=0.02)
    cbar.set_label("Pearson r (within tertile)")
    # Size legend
    handles = []
    for v in [size_mat.max() * 0.25, size_mat.max() * 0.5, size_mat.max()]:
        handles.append(ax.scatter([], [], s=v * s_scale, c="gray",
                                  edgecolor="black", linewidth=0.3, label=f"{v:.3f}"))
    ax.legend(handles=handles, title="Mean attn", loc="lower right", fontsize=7, labelspacing=1.4)
    plt.tight_layout()
    save_figure(fig, str(output_dir / "cell_type_dotheatmap.png"), dpi=200)
    plt.close(fig)


def plot_subject_clustermap(interp: dict, output_dir: Path) -> None:
    """Hierarchically clustered heatmap: rows=subjects, cols=cell types, values=normalized attention.

    Row color annotations: cogn tertile, gpath tertile.
    Reveals subject subtypes and data-availability blocks (PFC-only cluster).
    """
    attn = interp["attention"]  # (N, 31)
    ct_names = interp["cell_type_names"]
    meta = interp["metadata"]
    gpath = meta["gpath"].to_numpy()
    cogn = meta["actual"].to_numpy()

    # Drop subjects with any NaN attention (rare)
    row_mask = ~np.isnan(attn).any(axis=1)
    A = attn[row_mask]
    g = gpath[row_mask]
    c = cogn[row_mask]

    # Row-normalize (z-score per subject) so we cluster on relative profile, not magnitude
    A_z = (A - A.mean(axis=1, keepdims=True)) / (A.std(axis=1, keepdims=True) + 1e-9)

    # Tertile row colors
    def tertile_colors(v, cmap="RdYlGn"):
        q = np.quantile(v, [1/3, 2/3])
        b = np.digitize(v, q)
        palette = plt.get_cmap(cmap)
        return [palette(0.15 if x == 0 else 0.5 if x == 1 else 0.85) for x in b]

    row_colors = pd.DataFrame({
        "gpath": tertile_colors(g, cmap="Reds"),
        "cogn": tertile_colors(c, cmap="Blues"),
    })

    # Column labels: pretty
    col_labels = [ct.replace("_", " ") for ct in ct_names]

    df = pd.DataFrame(A_z, columns=col_labels)
    cg = sns.clustermap(
        df, cmap="RdBu_r", center=0, figsize=(12, 10),
        row_colors=row_colors, col_cluster=True, row_cluster=True,
        xticklabels=col_labels, yticklabels=False,
        linewidths=0, cbar_pos=(0.02, 0.82, 0.02, 0.12),
        dendrogram_ratio=0.12,
    )
    cg.ax_heatmap.set_xticklabels(cg.ax_heatmap.get_xticklabels(), rotation=70, fontsize=8, ha="right")
    cg.ax_heatmap.set_xlabel("Cell type")
    cg.ax_heatmap.set_ylabel("Subject (clustered)")
    cg.figure.suptitle(
        "Subject × cell-type attention clustermap (per-row z-score)\n"
        "Row colors: gpath tertile (red intensity), cogn tertile (blue intensity)",
        y=1.02,
    )
    save_figure(cg.figure, str(output_dir / "subject_clustermap.png"), dpi=200)
    plt.close(cg.figure)


def plot_prediction_ridgeline(results: dict, output_dir: Path, seed: str = "seed42", config: str = PRODUCTION_CONFIG) -> None:
    """Ridgeline of predicted_mean distributions across pathology tertiles, pooled over 5 folds."""
    parts = []
    for fold in range(5):
        key = (seed, config, fold)
        if key not in results:
            continue
        parts.append(results[key]["val_df"].copy())
    if not parts:
        return
    stacked = pd.concat(parts, ignore_index=True)

    # Tertile from gpath
    q = np.quantile(stacked["gpath"], [1/3, 2/3])
    stacked["tertile"] = np.digitize(stacked["gpath"], q)
    tert_names = {0: "low-pathology", 1: "mid-pathology", 2: "high-pathology"}
    colors = {0: ACCENT_TEAL, 1: ACCENT_PEACH, 2: ACCENT_CORAL}

    fig, ax = plt.subplots(figsize=(10, 6))
    y_offsets = {2: 0.0, 1: 1.0, 0: 2.0}
    xmin, xmax = stacked["predicted_mean"].min(), stacked["predicted_mean"].max()
    xx = np.linspace(xmin - 0.5, xmax + 0.5, 400)
    for t in [2, 1, 0]:
        sub = stacked[stacked["tertile"] == t]
        if len(sub) < 3:
            continue
        kde = stats.gaussian_kde(sub["predicted_mean"])
        dens = kde(xx)
        dens = dens / dens.max() * 0.9
        ax.fill_between(xx, y_offsets[t], y_offsets[t] + dens,
                        color=colors[t], alpha=0.65, edgecolor="black", linewidth=0.6)
        mean_pred = sub["predicted_mean"].mean()
        mean_act = sub["actual"].mean()
        ax.vlines(mean_pred, y_offsets[t], y_offsets[t] + 0.9, color="black", linestyle="--", linewidth=1.2)
        ax.vlines(mean_act, y_offsets[t], y_offsets[t] + 0.9, color="darkred", linestyle=":", linewidth=1.5)
        ax.text(xmax + 0.1, y_offsets[t] + 0.45,
                f"{tert_names[t]}\nN={len(sub)}\npred={mean_pred:+.2f}, act={mean_act:+.2f}",
                va="center", fontsize=9)
    ax.set_yticks([y_offsets[0] + 0.45, y_offsets[1] + 0.45, y_offsets[2] + 0.45])
    ax.set_yticklabels([tert_names[0], tert_names[1], tert_names[2]])
    ax.set_xlabel("Predicted cogn_global")
    ax.set_xlim(xmin - 0.5, xmax + 3.5)
    ax.set_title("Prediction distribution by pathology tertile (dashed = mean predicted, red dotted = mean actual)")
    plt.tight_layout()
    save_figure(fig, str(output_dir / "prediction_ridgeline.png"), dpi=200)
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

    logger.info("[1/13] Ablation bar chart")
    plot_ablation_bar(results, run_dir)
    logger.info("[2/13] Per-fold strip overlay")
    plot_per_fold_strip(results, run_dir)
    logger.info("[3/13] Seed sensitivity")
    plot_seed_sensitivity(results, run_dir)
    logger.info("[4/13] HPO top-5 comparison")
    plot_hpo_top_comparison(results, run_dir)
    logger.info("[5/13] Pred-vs-actual stacked")
    plot_pred_vs_actual_stacked(results, run_dir)
    logger.info("[6/13] Residual violin")
    plot_residual_violin(results, run_dir)
    if args.skip_tb_overlay:
        logger.info("[7/13] Skipped (--skip-tb-overlay)")
    else:
        logger.info("[7/13] Loss curve overlay (tensorboard)")
        plot_loss_curves_overlay(results, run_dir, outputs_root)

    # New presentation-oriented interpretability plots (A, B, C, α, β, γ)
    logger.info("[8/13] Baseline comparison bar (memory values — see manifest warning)")
    plot_baseline_comparison(run_dir)

    interp = load_interpretability_data(results, outputs_root)
    if interp is not None:
        logger.info("[9/13] Cell-type importance bar")
        plot_cell_type_importance(interp, run_dir)
        logger.info("[10/13] Pathology × attention scatter")
        plot_pathology_attention_scatter(interp, run_dir)
        logger.info("[11/13] Cell-type dot-heatmap (pathology tertiles)")
        plot_cell_type_dotheatmap(interp, run_dir)
        logger.info("[12/13] Subject clustermap")
        plot_subject_clustermap(interp, run_dir)
    else:
        logger.info("[9-12/13] Skipped (interpretability data unavailable)")

    logger.info("[13/13] Prediction ridgeline by pathology tertile")
    plot_prediction_ridgeline(results, run_dir)

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

    warnings: list[str] = [BASELINES_WARNING]
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
            "baseline_memory_values": BASELINE_MEMORY_VALUES,
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
