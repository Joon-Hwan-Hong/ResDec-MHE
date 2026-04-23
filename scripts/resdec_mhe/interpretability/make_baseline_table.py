"""Paper baseline table.

Aggregates per-fold R² / MAE / RMSE / Pearson r / Spearman ρ for every
baseline and every ResDec-MHE ablation into a single CSV + Markdown table.

Sources
-------
- TabPFN-2.6 standalone    : ``data/redesign/tabpfn_outer_fold{0..4}.npz``
  (per-fold R² computed via ``src.analysis.resdec_io.compute_per_fold_r2_tabpfn``;
  other metrics computed on the fly from y_true / y_tabpfn).
- Classical baselines      : ``outputs/pipeline/baseline_results_classical.csv``
  (one row per (model, feature_set, fold); we emit one table row per
  (model, feature_set) pair).
- DL baselines             : ``outputs/baselines/<name>/results.csv`` for
  cloudpred, cloudpred_pertype, gpio, perceiver_io (folds 1..5).
- Our canonical + ablations: ``outputs/redesign/p5_*/best_vs_tabpfn_summary.json``
  (per-fold ours.{r2, mae, rmse, pearson_r, spearman_rho}).

Missing baselines or missing ablations yield a NaN row with an explanatory
note — never omitted — so the table is idempotent and re-runnable once those
experiments finish.

Output
------
- ``<out-dir>/paper_baseline_table.csv`` — full machine-readable table.
- ``<out-dir>/paper_baseline_table.md``  — paper-ready markdown table.

Usage
-----
    PYTHONPATH=<worktree-root> \\
    uv run python scripts/resdec_mhe/interpretability/make_baseline_table.py \\
        --canonical-dir outputs/redesign/p5_canonical_seed42 \\
        --ablation-root outputs/redesign \\
        --baselines-root outputs/baselines \\
        --classical-csv outputs/pipeline/baseline_results_classical.csv \\
        --tabpfn-dir data/redesign \\
        --out-dir outputs/redesign/interpretability
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr


# Make the script standalone-runnable: ensure the worktree root is on sys.path.
_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if (_WORKTREE_ROOT / "src").is_dir() and str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from src.analysis.resdec_io import load_tabpfn_outer_fold  # noqa: E402

logger = logging.getLogger(__name__)


_METRIC_KEYS: tuple[str, ...] = (
    "r2", "mae", "rmse", "pearson_r", "spearman_rho",
)
_METRIC_NAMES: dict[str, str] = {
    "r2": "R²",
    "mae": "MAE",
    "rmse": "RMSE",
    "pearson_r": "Pearson r",
    "spearman_rho": "Spearman ρ",
}

# CSV column display names: internal dict keys use sklearn-idiomatic
# ``pearson_r`` / ``spearman_rho``; the paper-facing CSV columns drop the
# ``_r`` / ``_rho`` suffix so the header reads
# ``pearson_mean, pearson_std, spearman_mean, spearman_std``.
_METRIC_DISPLAY_NAMES: dict[str, str] = {
    "r2": "r2",
    "mae": "mae",
    "rmse": "rmse",
    "pearson_r": "pearson",
    "spearman_rho": "spearman",
}

# Reference R² for the encoder-only baseline (no TabPFN residual, no head
# decomposition). Sourced from an earlier 5-fold run; per-fold data was not
# archived — only the mean is reported as a single-number reference row.
CURRENT_ENCODER_ALONE_R2_REF: float = 0.286

# Note on the no-DiffAttn ablation: the canonical run already has
# ``use_diff_attn=False``, so that ablation IS the canonical (no separate row).
# ``p5_phase3_1stage_with_tabm`` below is the INVERSE: canonical *with*
# DiffAttn, retained for the delta comparison.
#
# Non-canonical ablations requested by the paper. Order doesn't matter —
# rows are sorted by R² descending before rendering. The canonical row is
# injected separately via ``--canonical-dir`` so it can be overridden from
# the CLI without editing the table spec.
_ABLATION_NAMES: list[tuple[str, str]] = [
    ("p5_filmwired_5fold_seed42", "ResDec-MHE + FiLM with real metadata"),
    ("p5_phase3_1stage_with_tabm", "ResDec-MHE with DiffAttn"),
    ("p5_phase3_2stage", "ResDec-MHE n_stages=2"),
    ("p5_phase3_3stage", "ResDec-MHE n_stages=3"),
    ("p5_ablation_no_tabpfn", "Ablation: no TabPFN residual"),
    ("p5_ablation_k1", "Ablation: k_tabm=1"),
    ("p5_ablation_no_hyper_conn", "Ablation: no HyperConn"),
    ("p5_ablation_no_film", "Ablation: no FiLM"),
    ("p5_ablation_no_aug_u_n2", "Ablation: no aug-U n=2"),
    ("p5_ablation_topk_1000", "Ablation: top-k=1000"),
    ("p5_ablation_topk_4000", "Ablation: top-k=4000"),
    ("p5_ablation_zscore", "Ablation: per-feature z-score"),
]


# ---------------------------------------------------------------------------
# Low-level parsers
# ---------------------------------------------------------------------------

def parse_summary_json(path: Path) -> dict[str, list[float]] | None:
    """Parse a ``best_vs_tabpfn_summary.json`` into per-metric per-fold arrays.

    Returns
    -------
    dict or None
        ``{"r2": [...], "mae": [...], ...}`` with one entry per fold, in
        fold-id order. ``None`` if the file is missing or malformed.
    """
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001 — log + skip silently
        logger.warning("Failed to parse %s (%s); treating as missing.", path, exc)
        return None
    per_fold = data.get("per_fold")
    if not per_fold:
        logger.warning("%s has no 'per_fold' block; treating as missing.", path)
        return None

    try:
        per_fold_sorted = sorted(per_fold, key=lambda r: int(r["fold"]))
    except (KeyError, TypeError) as exc:
        logger.warning("%s has bad fold ids (%s); treating as missing.", path, exc)
        return None

    out: dict[str, list[float]] = {k: [] for k in _METRIC_KEYS}
    for entry in per_fold_sorted:
        ours = entry.get("ours")
        if not isinstance(ours, dict):
            logger.warning("%s fold %s missing 'ours'; skipping.", path, entry.get("fold"))
            continue
        for k in _METRIC_KEYS:
            v = ours.get(k)
            if v is None:
                logger.warning(
                    "%s fold %s missing ours.%s; filling with NaN.",
                    path, entry.get("fold"), k,
                )
                out[k].append(float("nan"))
            else:
                out[k].append(float(v))
    return out


def discover_ablation_dirs(ablation_root: Path) -> list[Path]:
    """Iterate ``p5_*`` subdirs of ``ablation_root`` with a summary JSON.

    Used by :func:`main` as a lint to flag ``p5_*`` subdirs that exist on
    disk but are not enumerated in :data:`_ABLATION_NAMES` (so they would
    silently miss the table). Main row assembly uses :func:`collect_ablation_rows`
    instead so pending/NaN rows can be emitted even for dirs that don't
    exist on disk yet.
    """
    if not ablation_root.exists():
        return []
    out: list[Path] = []
    for subdir in sorted(ablation_root.glob("p5_*")):
        if not subdir.is_dir():
            continue
        if (subdir / "best_vs_tabpfn_summary.json").exists():
            out.append(subdir)
    return out


def parse_classical_csv(path: Path) -> list[dict]:
    """Parse ``outputs/pipeline/baseline_results_classical.csv``.

    Emits one dict per (model, feature_set) pair, with ``metrics`` holding
    fold-ordered per-metric lists.

    Returns an empty list on missing or malformed input.
    """
    if not path.exists():
        logger.warning(
            "Classical baseline CSV missing at %s; classical rows will be absent.",
            path,
        )
        return []
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read %s (%s); skipping.", path, exc)
        return []

    required = {"model", "feature_set", "fold", *_METRIC_KEYS}
    missing_cols = required - set(df.columns)
    if missing_cols:
        logger.warning(
            "Classical CSV %s missing columns %s; skipping.", path, missing_cols,
        )
        return []

    rows: list[dict] = []
    for (model, feature_set), group in df.groupby(["model", "feature_set"]):
        group = group.sort_values("fold").reset_index(drop=True)
        metrics = {
            k: [
                float(v) if pd.notna(v) else float("nan")
                for v in group[k].tolist()
            ]
            for k in _METRIC_KEYS
        }
        rows.append({
            "model": str(model),
            "feature_set": str(feature_set),
            "metrics": metrics,
            "source": str(path),
        })
    return rows


def parse_dl_baseline_csv(path: Path) -> dict[str, list[float]] | None:
    """Parse a ``outputs/baselines/<name>/results.csv`` into per-metric arrays.

    Folds in these CSVs are 1-indexed; we sort on ``fold`` and drop the
    original index. Missing columns or missing file → ``None``.
    """
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read %s (%s); skipping.", path, exc)
        return None
    required = {"fold", *_METRIC_KEYS}
    missing_cols = required - set(df.columns)
    if missing_cols:
        logger.warning(
            "DL baseline CSV %s missing columns %s; skipping.", path, missing_cols,
        )
        return None
    df = df.sort_values("fold").reset_index(drop=True)
    out: dict[str, list[float]] = {}
    for k in _METRIC_KEYS:
        out[k] = [
            float(v) if pd.notna(v) else float("nan")
            for v in df[k].tolist()
        ]
    return out


def parse_tabpfn_standalone(
    tabpfn_dir: Path, n_folds: int = 5,
) -> dict[str, list[float]] | None:
    """Per-fold metrics for TabPFN-2.6 standalone, computed on the fly.

    Reads ``<tabpfn-dir>/tabpfn_outer_fold{f}.npz`` via
    :func:`src.analysis.resdec_io.load_tabpfn_outer_fold`, which yields
    ``(y_true, y_tabpfn)``. For each fold we compute R², MAE, RMSE,
    Pearson r, Spearman ρ. If any fold is missing → ``None``.
    """
    metrics: dict[str, list[float]] = {k: [] for k in _METRIC_KEYS}
    for f in range(n_folds):
        path = tabpfn_dir / f"tabpfn_outer_fold{f}.npz"
        try:
            y_true, y_pred = load_tabpfn_outer_fold(path)
        except FileNotFoundError:
            logger.warning(
                "TabPFN-2.6 outer-fold file missing: %s; TabPFN standalone row dropped.",
                path,
            )
            return None
        metrics["r2"].append(float(r2_score(y_true, y_pred)))
        metrics["mae"].append(float(mean_absolute_error(y_true, y_pred)))
        metrics["rmse"].append(float(np.sqrt(np.mean((y_true - y_pred) ** 2))))
        # Pearson/Spearman return nan if input is degenerate; catch via try/except.
        try:
            metrics["pearson_r"].append(float(pearsonr(y_true, y_pred).statistic))
        except Exception:  # noqa: BLE001
            metrics["pearson_r"].append(float("nan"))
        try:
            metrics["spearman_rho"].append(
                float(spearmanr(y_true, y_pred).statistic),
            )
        except Exception:  # noqa: BLE001
            metrics["spearman_rho"].append(float("nan"))
    return metrics


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def summarise_row(metrics: dict[str, list[float]]) -> dict[str, float]:
    """Return mean + std (ddof=1) for each metric, plus ``n_folds``.

    Empty metrics → all-NaN summary with n_folds = 0.
    """
    n = len(metrics.get("r2", []))
    summary: dict[str, float] = {"n_folds": n}
    for k in _METRIC_KEYS:
        vals = np.asarray(metrics.get(k, []), dtype=np.float64)
        # Drop NaN entries before aggregating so a single bad fold doesn't
        # poison the mean/std.
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            summary[f"{k}_mean"] = float("nan")
            summary[f"{k}_std"] = float("nan")
        else:
            summary[f"{k}_mean"] = float(finite.mean())
            # ddof=1 matches paper convention (sample std across folds).
            summary[f"{k}_std"] = (
                float(finite.std(ddof=1)) if finite.size > 1 else 0.0
            )
    return summary


def _parse_per_fold_npz(
    subdir: Path, n_folds: int = 5,
) -> dict[str, list[float]] | None:
    """Fallback loader: read per-fold ``fold{i}/val_predictions_best.npz``.

    Used when a ``best_vs_tabpfn_summary.json`` rollup is absent but the
    per-fold npz files still exist (e.g. ``p5_phase3_3stage`` was trained
    but the summary-gen step wasn't run). Returns ``None`` if ANY fold's
    npz is missing or lacks a required metric.
    """
    out: dict[str, list[float]] = {k: [] for k in _METRIC_KEYS}
    for f in range(n_folds):
        npz_path = subdir / f"fold{f}" / "val_predictions_best.npz"
        if not npz_path.exists():
            return None
        try:
            d = np.load(npz_path, allow_pickle=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read %s (%s); fallback skipped.", npz_path, exc)
            return None
        # npz keys mirror _METRIC_KEYS exactly in our training script.
        for k in _METRIC_KEYS:
            if k not in d.files:
                logger.warning(
                    "%s missing metric '%s'; fallback row skipped.", npz_path, k,
                )
                return None
            out[k].append(float(d[k]))
    return out


def collect_ablation_rows(
    ablation_root: Path,
    requested: list[tuple[str, str]],
) -> list[dict]:
    """Assemble one row per requested ablation; missing dirs → 'pending' NaN row.

    Resolution order for each ablation:
    1. ``<subdir>/best_vs_tabpfn_summary.json`` (preferred; computed by
       our training driver after re-inference).
    2. Per-fold ``<subdir>/fold{i}/val_predictions_best.npz`` (fallback;
       every metric is stored on the npz scalar header).
    3. Empty placeholder row (``pending``) if neither source resolves.

    Parameters
    ----------
    ablation_root : Path
        Root containing ``p5_*`` subdirs.
    requested : list[(model_dir_name, display_name)]
        Rows to emit, in stable order. Missing dirs still appear so the
        table is idempotent once those ablations finish.
    """
    rows: list[dict] = []
    for model_name, display in requested:
        subdir = ablation_root / model_name
        summary_path = subdir / "best_vs_tabpfn_summary.json"
        metrics = parse_summary_json(summary_path)
        source = str(summary_path)
        notes = ""
        if metrics is None:
            # Fallback: per-fold npz.
            metrics = _parse_per_fold_npz(subdir)
            if metrics is not None:
                source = f"{subdir}/fold{{0..4}}/val_predictions_best.npz"
                notes = (
                    "per-fold val_predictions_best.npz (summary JSON absent)"
                )
        if metrics is None:
            # Pending / not-yet-complete — emit a placeholder row so the table
            # is self-documenting even while some ablations are still running.
            logger.warning(
                "Ablation '%s' has no per-fold data at %s — "
                "emitting NaN row (pending).",
                model_name, subdir,
            )
            empty = {k: [] for k in _METRIC_KEYS}
            summary = summarise_row(empty)
            rows.append({
                "model": model_name,
                "display_name": display,
                **summary,
                "source_path": str(subdir),
                "notes": "pending: ablation not yet complete",
            })
            continue
        summary = summarise_row(metrics)
        rows.append({
            "model": model_name,
            "display_name": display,
            **summary,
            "source_path": source,
            "notes": notes,
        })
    return rows


def collect_tabpfn_row(tabpfn_dir: Path, n_folds: int) -> dict | None:
    """One row for TabPFN-2.6 standalone. None if any fold is missing."""
    metrics = parse_tabpfn_standalone(tabpfn_dir, n_folds=n_folds)
    if metrics is None:
        return None
    summary = summarise_row(metrics)
    return {
        "model": "tabpfn_2_6_standalone",
        "display_name": "TabPFN-2.6 standalone (top-2K features)",
        **summary,
        "source_path": str(tabpfn_dir),
        "notes": "outer-fold R² computed from tabpfn_outer_fold{f}.npz",
    }


def collect_classical_rows(classical_csv: Path) -> list[dict]:
    """Rows for Ridge/ElasticNet/PLS/RF/XGBoost across feature sets."""
    parsed = parse_classical_csv(classical_csv)
    rows: list[dict] = []
    for entry in parsed:
        model = entry["model"]
        fset = entry["feature_set"]
        summary = summarise_row(entry["metrics"])
        key = f"{model.lower()}_{fset.replace('+', '_')}"
        rows.append({
            "model": key,
            "display_name": f"{model} [{fset}]",
            **summary,
            "source_path": entry["source"],
            "notes": f"feature set: {fset}",
        })
    return rows


def collect_dl_baseline_rows(baselines_root: Path) -> list[dict]:
    """Rows for cloudpred / cloudpred_pertype / gpio / perceiver_io."""
    display_map = {
        "cloudpred": "CloudPred",
        "cloudpred_pertype": "CloudPred (per-type)",
        "gpio": "GPIO",
        "perceiver_io": "Perceiver-IO",
    }
    rows: list[dict] = []
    if not baselines_root.exists():
        logger.warning(
            "Baselines root %s does not exist; DL baseline rows will be absent.",
            baselines_root,
        )
        return rows
    for subdir in sorted(p for p in baselines_root.iterdir() if p.is_dir()):
        csv_path = subdir / "results.csv"
        metrics = parse_dl_baseline_csv(csv_path)
        if metrics is None:
            logger.warning(
                "DL baseline '%s' has no parsable results.csv; row skipped.",
                subdir.name,
            )
            continue
        summary = summarise_row(metrics)
        rows.append({
            "model": subdir.name,
            "display_name": display_map.get(
                subdir.name, subdir.name.replace("_", " ").title(),
            ),
            **summary,
            "source_path": str(csv_path),
            "notes": "",
        })
    return rows


def _try_read_rosmap_baseline_summary(results_dir: Path) -> tuple[dict, int, Path] | None:
    """Parse ``Summary_<MODEL_NAME>_ROSMAP.csv`` + ``AllFolds_<MODEL_NAME>_ROSMAP.csv``.

    These are the long-form CSVs written by ``baselines/{mixmil,scPhase}/run_rosmap.py``:
    Summary has columns ``metric, mean, std`` (one row per metric); AllFolds has per-fold
    rows with columns ``r2, mae, pearson_r, spearman_rho, fold, train_time_s``.

    Returns (metric_dict, n_folds, source_path) if the Summary file exists, else None.
    rmse is always NaN because run_rosmap.py scripts do not emit it.
    """
    summary_files = sorted(results_dir.glob("Summary_*_ROSMAP.csv"))
    if not summary_files:
        return None
    summary_path = summary_files[0]
    summary_df = pd.read_csv(summary_path)

    metric_dict: dict[str, float] = {
        f"{k}_mean": float("nan") for k in _METRIC_KEYS
    }
    metric_dict.update({f"{k}_std": float("nan") for k in _METRIC_KEYS})
    # Summary CSV rows use sklearn-idiomatic keys matching _METRIC_KEYS (r2, mae,
    # pearson_r, spearman_rho). rmse is absent from run_rosmap outputs.
    for _, row in summary_df.iterrows():
        metric = str(row["metric"])
        if metric in _METRIC_KEYS:
            metric_dict[f"{metric}_mean"] = float(row["mean"])
            metric_dict[f"{metric}_std"] = float(row["std"])

    n_folds = 0
    allfolds_files = sorted(results_dir.glob("AllFolds_*_ROSMAP.csv"))
    if allfolds_files:
        n_folds = len(pd.read_csv(allfolds_files[0]))

    return metric_dict, n_folds, summary_path


def collect_nonresult_rows(baselines_root: Path) -> list[dict]:
    """Rows for MixMIL / scPhase (auto-detects real results when present) + legacy reference.

    MixMIL and scPhase emit ``Summary_<MODEL>_ROSMAP.csv`` + ``AllFolds_<MODEL>_ROSMAP.csv``
    under ``<baselines_root>/<mixmil|scphase>/``. If those files are present, the row is
    populated from them; otherwise, a "source only, no output" placeholder row is emitted
    so the table stays idempotent across re-runs.

    "Current encoder alone" (R² = :data:`CURRENT_ENCODER_ALONE_R2_REF`) is a
    mean-only reference from an earlier 5-fold run — we emit a point-estimate
    row with NaN std.
    """
    rows: list[dict] = []

    # MixMIL — auto-detect real results, fall back to source-only placeholder.
    mixmil_dir = baselines_root / "mixmil"
    mixmil_result = _try_read_rosmap_baseline_summary(mixmil_dir)
    if mixmil_result is not None:
        metrics, n_folds, summary_path = mixmil_result
        rows.append({
            "model": "mixmil",
            "display_name": "MixMIL (Engelmann et al. 2024)",
            "n_folds": n_folds,
            **metrics,
            "source_path": str(summary_path),
            "notes": "",
        })
    else:
        rows.append({
            "model": "mixmil",
            "display_name": "MixMIL (Engelmann et al. 2024)",
            "n_folds": 0,
            **{f"{k}_mean": float("nan") for k in _METRIC_KEYS},
            **{f"{k}_std": float("nan") for k in _METRIC_KEYS},
            "source_path": "baselines/mixmil/run_rosmap.py",
            "notes": (
                "source only, no output: blocked on baselines/shared/mixmil_input.h5ad"
            ),
        })

    # scPhase — auto-detect real results, fall back to source-only placeholder.
    scphase_dir = baselines_root / "scphase"
    scphase_result = _try_read_rosmap_baseline_summary(scphase_dir)
    if scphase_result is not None:
        metrics, n_folds, summary_path = scphase_result
        rows.append({
            "model": "scphase",
            "display_name": "scPhase (Berson et al. 2025)",
            "n_folds": n_folds,
            **metrics,
            "source_path": str(summary_path),
            "notes": "",
        })
    else:
        rows.append({
            "model": "scphase",
            "display_name": "scPhase (Berson et al. 2025)",
            "n_folds": 0,
            **{f"{k}_mean": float("nan") for k in _METRIC_KEYS},
            **{f"{k}_std": float("nan") for k in _METRIC_KEYS},
            "source_path": "baselines/scPhase/run_rosmap.py",
            "notes": (
                "source only, no output: blocked on baselines/shared/scphase_input.h5ad"
            ),
        })
    # Current encoder alone — mean-only reference (legacy, no per-fold file).
    rows.append({
        "model": "current_encoder_alone",
        "display_name": "Current encoder alone (mean-only reference)",
        "n_folds": 0,
        "r2_mean": CURRENT_ENCODER_ALONE_R2_REF,
        "r2_std": float("nan"),
        **{f"{k}_mean": float("nan") for k in _METRIC_KEYS if k != "r2"},
        **{f"{k}_std": float("nan") for k in _METRIC_KEYS if k != "r2"},
        "source_path": "(legacy training run; no per-fold CSV archived)",
        "notes": (
            f"reference R² only ({CURRENT_ENCODER_ALONE_R2_REF:.3f}) from "
            "an earlier 5-fold run; per-fold data unavailable"
        ),
    })
    return rows


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _is_ours_row(model: str) -> bool:
    """Is this row an 'ours' (canonical or ablation) row?"""
    return model.startswith("p5_") or model == "current_encoder_alone"


def rows_to_dataframe(rows: list[dict]) -> pd.DataFrame:
    """Assemble rows into a DataFrame with a stable column order.

    Internal row dicts use ``pearson_r`` / ``spearman_rho`` keys (matching
    the sklearn / scipy idioms); the published CSV column names use the
    paper-facing ``pearson`` / ``spearman`` aliases (see
    :data:`_METRIC_DISPLAY_NAMES`). This function renames keys in-place to
    match the published column names.
    """
    # Column order: (r2_mean, r2_std, mae_mean, mae_std, ..., pearson_mean,
    # pearson_std, spearman_mean, spearman_std).
    cols: list[str] = ["model", "display_name", "n_folds"]
    for k in _METRIC_KEYS:
        display = _METRIC_DISPLAY_NAMES[k]
        cols.extend([f"{display}_mean", f"{display}_std"])
    cols.extend(["source_path", "notes"])

    # Rebuild rows with renamed keys so the DataFrame column names match the
    # CSV/MD contract; keep everything else.
    renamed_rows: list[dict] = []
    for row in rows:
        new_row = dict(row)
        for k in _METRIC_KEYS:
            display = _METRIC_DISPLAY_NAMES[k]
            if k == display:
                continue
            for stat in ("mean", "std"):
                old_key = f"{k}_{stat}"
                new_key = f"{display}_{stat}"
                if old_key in new_row:
                    new_row[new_key] = new_row.pop(old_key)
        renamed_rows.append(new_row)

    df = pd.DataFrame(renamed_rows)
    # Ensure every column exists (in case some rows are short).
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df[cols]


def _fmt_pair(mean: float, std: float) -> str:
    """Format a ``mean ± std`` cell for the markdown table.

    Rules:
    - NaN mean → ``"—"`` (covers pending/missing rows).
    - NaN std  → mean-only ``"0.xxxx"`` (covers size-1 / reference rows).
    - std == 0 also falls through to the ``± 0.xxxx`` branch (size-1 rows
      emitted by :func:`summarise_row` use std == 0, not NaN, when there is
      exactly one fold).
    """
    try:
        mean_f = float(mean)
    except (TypeError, ValueError):
        return "—"
    if math.isnan(mean_f):
        return "—"
    try:
        std_f = float(std)
    except (TypeError, ValueError):
        return f"{mean_f:.4f}"
    if math.isnan(std_f):
        return f"{mean_f:.4f}"
    if std_f == 0.0:
        return f"{mean_f:.4f}"
    return f"{mean_f:.4f} ± {std_f:.4f}"


def _sort_for_md(df: pd.DataFrame | list[dict]) -> pd.DataFrame | list[dict]:
    """Sort for the MD display: baselines (top) then ours (bottom).

    Within each block, sort by r2_mean descending; NaN R² rows go to the
    bottom of their block so pending/missing rows don't hide good ones.

    Accepts either a DataFrame (the main pipeline) or a list of dicts (for
    tests that want to probe sort semantics without building a full table).
    """
    if isinstance(df, list):
        def _r2(row: dict) -> float:
            v = row.get("r2_mean", float("nan"))
            try:
                return float(v)
            except (TypeError, ValueError):
                return float("nan")

        def _is_ours(row: dict) -> bool:
            return bool(
                row.get("_is_ours") or _is_ours_row(str(row.get("model", "")))
            )

        ours = [r for r in df if _is_ours(r)]
        others = [r for r in df if not _is_ours(r)]
        # NaN last within each block.
        _sort_key = lambda r: (  # noqa: E731
            math.isnan(_r2(r)), -_r2(r) if not math.isnan(_r2(r)) else 0.0,
        )
        others_sorted = sorted(others, key=_sort_key)
        ours_sorted = sorted(ours, key=_sort_key)
        return others_sorted + ours_sorted

    is_ours = df["model"].apply(_is_ours_row)
    ours = df[is_ours].copy()
    others = df[~is_ours].copy()
    # NaN sorts last when ascending=False + na_position='last'.
    others = others.sort_values("r2_mean", ascending=False, na_position="last")
    ours = ours.sort_values("r2_mean", ascending=False, na_position="last")
    return pd.concat([others, ours], ignore_index=True)


def render_markdown(df: pd.DataFrame) -> str:
    """Paper-ready markdown table ordered per the spec."""
    df = _sort_for_md(df)

    lines: list[str] = []
    lines.append("# Paper Baseline Table")
    lines.append("")
    lines.append(
        "Per-fold metrics are reported as mean ± std (ddof=1) across 5 outer "
        "folds unless noted. Rows are grouped into (1) external baselines "
        "sorted by R², (2) our canonical + ablations sorted by R². Pending "
        "or missing rows are retained so the table is idempotent across "
        "re-runs."
    )
    lines.append("")
    header = (
        "| Model | N folds | "
        + " | ".join(_METRIC_NAMES[k] for k in _METRIC_KEYS)
        + " | Source | Notes |"
    )
    sep = (
        "|---|---|"
        + "|".join("---" for _ in _METRIC_KEYS)
        + "|---|---|"
    )
    lines.append(header)
    lines.append(sep)
    for _, row in df.iterrows():
        cells = [
            str(row["display_name"]),
            str(row["n_folds"]),
        ]
        for k in _METRIC_KEYS:
            display = _METRIC_DISPLAY_NAMES[k]
            cells.append(_fmt_pair(row[f"{display}_mean"], row[f"{display}_std"]))
        cells.append(str(row["source_path"]))
        notes = row["notes"]
        if pd.isna(notes) or notes is None:
            notes = ""
        cells.append(str(notes))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def _compute_provenance(args: argparse.Namespace) -> dict:
    """Assemble a provenance JSON payload for reproducibility.

    Records the git SHA of the worktree, the CLI-resolved input paths, and
    the list of ablations requested so a downstream reader can reconstruct
    which files fed this table.
    """
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_sha = "unknown"
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_sha,
        "canonical_dir": str(args.canonical_dir),
        "ablation_root": str(args.ablation_root),
        "baselines_root": str(args.baselines_root),
        "classical_csv": str(args.classical_csv),
        "tabpfn_dir": str(args.tabpfn_dir),
        "n_folds": int(args.n_folds),
        "ablations_requested": [e[0] for e in _ABLATION_NAMES],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--canonical-dir", type=Path,
                   default=Path("outputs/redesign/p5_canonical_seed42"),
                   help="Canonical model output dir (must contain "
                        "best_vs_tabpfn_summary.json).")
    p.add_argument("--ablation-root", type=Path,
                   default=Path("outputs/redesign"),
                   help="Root holding p5_* ablation dirs.")
    p.add_argument("--baselines-root", type=Path,
                   default=Path("outputs/baselines"),
                   help="Root holding per-baseline subdirs with results.csv.")
    p.add_argument("--classical-csv", type=Path,
                   default=Path("outputs/pipeline/baseline_results_classical.csv"),
                   help="Classical baselines CSV (Ridge/ElasticNet/PLS/RF/XGBoost).")
    p.add_argument("--tabpfn-dir", type=Path, default=Path("data/redesign"),
                   help="Directory with tabpfn_outer_fold{0..4}.npz files.")
    p.add_argument("--out-dir", type=Path,
                   default=Path("outputs/redesign/interpretability"),
                   help="Output directory for the CSV + MD files.")
    p.add_argument("--n-folds", type=int, default=5,
                   help="Number of outer CV folds (default 5).")
    return p


def main(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[baseline_table] canonical-dir   = %s", args.canonical_dir)
    logger.info("[baseline_table] ablation-root   = %s", args.ablation_root)
    logger.info("[baseline_table] baselines-root  = %s", args.baselines_root)
    logger.info("[baseline_table] classical-csv   = %s", args.classical_csv)
    logger.info("[baseline_table] tabpfn-dir      = %s", args.tabpfn_dir)
    logger.info("[baseline_table] out-dir         = %s", out_dir)

    rows: list[dict] = []

    # 1. TabPFN-2.6 standalone — computed on the fly from outer-fold npz.
    tab_row = collect_tabpfn_row(args.tabpfn_dir, n_folds=args.n_folds)
    if tab_row is not None:
        rows.append(tab_row)
        logger.info(
            "[baseline_table] TabPFN-2.6 standalone R² = %.4f ± %.4f across %d folds",
            tab_row["r2_mean"], tab_row["r2_std"], tab_row["n_folds"],
        )

    # 2. Classical baselines.
    classical_rows = collect_classical_rows(args.classical_csv)
    rows.extend(classical_rows)
    logger.info(
        "[baseline_table] classical baseline rows: %d (from %s)",
        len(classical_rows), args.classical_csv,
    )

    # 3. DL baselines under <baselines-root>.
    dl_rows = collect_dl_baseline_rows(args.baselines_root)
    rows.extend(dl_rows)
    logger.info(
        "[baseline_table] DL baseline rows: %d (from %s)",
        len(dl_rows), args.baselines_root,
    )

    # 4. MixMIL / scPhase / current-encoder reference rows.
    # Verify the MixMIL / scPhase source paths still exist at runtime — warn
    # loudly if they've been moved/renamed so the provenance note in the MD
    # table doesn't silently drift.
    mixmil_src = Path("baselines/mixmil/run_rosmap.py")
    if not mixmil_src.exists():
        logger.warning("MixMIL source path %s no longer exists", mixmil_src)
    scphase_src = Path("baselines/scPhase/run_rosmap.py")
    if not scphase_src.exists():
        logger.warning("scPhase source path %s no longer exists", scphase_src)

    nonresult_rows = collect_nonresult_rows(args.baselines_root)
    rows.extend(nonresult_rows)
    logger.info(
        "[baseline_table] non-result (source-only / legacy-reference) rows: %d",
        len(nonresult_rows),
    )

    # 5. Our canonical (from --canonical-dir) + all ablations.
    #    We prepend the canonical entry so callers can override the canonical
    #    via CLI (e.g. test a seed43 canonical without editing constants).
    canonical_name = args.canonical_dir.name
    canonical_entry = (
        canonical_name,
        f"ResDec-MHE (canonical, {canonical_name})",
    )
    # De-dupe if the CLI default canonical matches a name already in
    # _ABLATION_NAMES (shouldn't, but guard just in case).
    ablation_entries = [canonical_entry] + [
        e for e in _ABLATION_NAMES if e[0] != canonical_name
    ]
    ours_rows = collect_ablation_rows(args.ablation_root, ablation_entries)
    rows.extend(ours_rows)
    logger.info(
        "[baseline_table] ours + ablation rows: %d", len(ours_rows),
    )

    # Lint: flag any ``p5_*`` subdirs that exist on disk but are NOT listed
    # in ``_ABLATION_NAMES``. The canonical dir is expected to be implicit
    # (injected via ``--canonical-dir``) and is excluded from the warning.
    found_dirs = discover_ablation_dirs(args.ablation_root)
    listed_dirs = {e[0] for e in _ABLATION_NAMES}
    unlisted = (
        {d.name for d in found_dirs}
        - listed_dirs
        - {canonical_name}
    )
    if unlisted:
        logger.warning(
            "[baseline_table] Found %d p5_* dirs not in _ABLATION_NAMES: %s. "
            "Add to _ABLATION_NAMES to include in the table.",
            len(unlisted), sorted(unlisted),
        )

    df = rows_to_dataframe(rows)

    # Write CSV in registration order, unsorted — downstream tooling can
    # re-sort. This keeps MixMIL/scPhase/current-encoder rows together with
    # their block rather than mixed across the table.
    csv_path = out_dir / "paper_baseline_table.csv"
    df.to_csv(csv_path, index=False)
    logger.info("[baseline_table] wrote %s (%d rows)", csv_path, len(df))

    # Render the markdown ONCE and reuse the string for both the file write
    # and the stdout echo. Previously this called ``render_markdown(df)``
    # twice, which re-sorted and re-formatted every row.
    md = render_markdown(df)
    md_path = out_dir / "paper_baseline_table.md"
    md_path.write_text(md)
    logger.info("[baseline_table] wrote %s", md_path)

    # Emit the provenance sidecar so downstream readers can reconstruct
    # the input tree + git SHA that produced this table.
    provenance = _compute_provenance(args)
    provenance_path = out_dir / "paper_baseline_table.provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2))
    logger.info("[baseline_table] wrote %s", provenance_path)

    # Print the MD to stdout for the caller's sanity check.
    print(md)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(build_parser().parse_args()))
