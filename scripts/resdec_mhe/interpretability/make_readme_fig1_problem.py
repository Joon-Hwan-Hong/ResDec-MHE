#!/usr/bin/env python
"""Render Figure 1 for the ResDec-MHE README redesign.

The figure frames the *question* of cognitive resilience: each ROSMAP cohort
subject is plotted as `cogn_global` vs `gpath`, colored by the per-subject
residual from an OLS regression of cogn_global on gpath. Positive residuals
(green, PiYG diverging colormap) are subjects who cognate better than their
pathology would predict ("behaviorally resilient given pathology"); negative
residuals (magenta) are subjects who cognate worse than predicted
("vulnerable").

Inputs
------
- ``outputs/splits.json``                       : 5-fold split file. Used for
  the canonical ROSMAP cohort of N=516 subjects (union over all folds).
- ``data/metadata_ROSMAP/metadata.csv``         : provides per-subject
  ``cogn_global`` (cognition composite, model target) and ``gpath`` (global
  pathology score, the standard ROSMAP resilience axis).

Pipeline
--------
1. Load splits.json; compute the union of all subject IDs across all
   folds (train + val) -- expected to be 516.
2. Load metadata.csv; subset to the 516 IDs.
3. Drop rows with NaN in ``cogn_global`` or ``gpath`` (none expected).
4. Deduplicate by ``ROSMAP_IndividualID``, keeping the latest entry per
   subject.
5. Verify the final n is exactly 516 (assert).
6. Fit OLS ``cogn_global ~ gpath`` (closed-form least squares); compute
   residuals + Spearman rho.
7. Render single-panel scatter colored by residual; PiYG diverging colormap
   centered at 0; OLS line dashed; subtle colorbar to the right; in-axes
   label with n + Spearman rho.
8. Save to ``figures/fig1_problem.png`` at 600 DPI.

Outputs
-------
- ``figures/fig1_problem.png`` (600 DPI, ~9x7 inches)
- Verification numbers printed to stdout (n_subjects, mean/std of
  cogn_global + gpath, OLS intercept + slope, Spearman rho).

Usage
-----
    PYTHONPATH=<worktree-root> \\
    uv run python scripts/resdec_mhe/interpretability/make_readme_fig1_problem.py

Idempotence
-----------
No randomness is involved (closed-form OLS, deterministic ordering from
metadata.csv after the IsIn-then-dedup-keep-last filter). PYTHONHASHSEED is
set defensively so repeated runs produce a bit-identical PNG.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Set PYTHONHASHSEED defensively for bit-identical reruns. Must be set
# before any matplotlib import in case any internal hash-based color
# selection depends on it.
os.environ.setdefault("PYTHONHASHSEED", "42")

import matplotlib

matplotlib.use("Agg")  # must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from scipy import stats

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from src.visualization.theme import (  # noqa: E402
    PALETTES,
    apply_theme,
    fmt_axes,
    save_fig,
)

logger = logging.getLogger(__name__)


def _union_subject_ids(splits_path: Path) -> list[str]:
    """Return the sorted union of subject IDs across every fold's train+val.

    The canonical ResDec-MHE pipeline uses the 516-subject pool defined by
    ``splits.json::train_val_pool``; this function recomputes that union
    from the per-fold lists rather than relying on the ``train_val_pool``
    key, so the script is robust to schema changes that drop the
    convenience key.
    """
    if not splits_path.exists():
        raise FileNotFoundError(f"splits.json not found: {splits_path}")
    payload = json.loads(splits_path.read_text())
    folds = payload.get("folds")
    if not isinstance(folds, list) or not folds:
        raise ValueError(
            f"splits.json::folds must be a non-empty list; got {type(folds).__name__}"
        )
    ids: set[str] = set()
    for i, fold in enumerate(folds):
        if not isinstance(fold, dict):
            raise ValueError(f"splits.json::folds[{i}] must be a dict; got {type(fold).__name__}")
        for key, value in fold.items():
            if isinstance(value, list):
                ids.update(str(x) for x in value)
            else:
                logger.debug("skipping non-list fold key %s.%s", i, key)
    return sorted(ids)


def _load_subject_frame(metadata_csv: Path, subject_ids: list[str]) -> pd.DataFrame:
    """Return a DataFrame of (cogn_global, gpath) for the canonical 516.

    Steps:
      1. Read metadata.csv.
      2. Subset to rows whose ``ROSMAP_IndividualID`` is in ``subject_ids``.
      3. Drop rows with NaN in cogn_global or gpath.
      4. Deduplicate by ROSMAP_IndividualID, keeping the *last* entry per
         subject (preserves the most recent measurement when multiple
         exist).
      5. Assert the final count is exactly len(subject_ids).
    """
    if not metadata_csv.exists():
        raise FileNotFoundError(f"metadata.csv not found: {metadata_csv}")
    md = pd.read_csv(metadata_csv)
    needed = {"ROSMAP_IndividualID", "cogn_global", "gpath"}
    missing = needed - set(md.columns)
    if missing:
        raise KeyError(f"metadata.csv is missing required columns: {sorted(missing)}")
    md["ROSMAP_IndividualID"] = md["ROSMAP_IndividualID"].astype(str)

    sub = md[md["ROSMAP_IndividualID"].isin(subject_ids)].copy()
    sub = sub.dropna(subset=["cogn_global", "gpath"])
    sub = sub.drop_duplicates(subset=["ROSMAP_IndividualID"], keep="last")
    sub = sub.reset_index(drop=True)

    if len(sub) != len(subject_ids):
        raise AssertionError(
            f"Filtered subject frame has n={len(sub)}; expected {len(subject_ids)}. "
            f"Check splits.json vs metadata.csv coverage."
        )
    return sub[["ROSMAP_IndividualID", "cogn_global", "gpath"]]


def _ols_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Closed-form OLS fit of y = intercept + slope * x.

    Implementation note: ``np.polyfit(x, y, 1)`` returns ``[slope, intercept]``;
    we compute the same numbers via the explicit covariance formula so the
    intent + precision is unambiguous and we don't depend on polyfit's
    internal scaling.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError(f"x and y must be 1D arrays of equal length; got {x.shape}, {y.shape}")
    x_mean = x.mean()
    y_mean = y.mean()
    xc = x - x_mean
    yc = y - y_mean
    denom = float((xc * xc).sum())
    if denom <= 0.0:
        raise ValueError("OLS denominator (sum of squared deviations of x) is 0")
    slope = float((xc * yc).sum() / denom)
    intercept = float(y_mean - slope * x_mean)
    return intercept, slope


def make_figure(
    df: pd.DataFrame,
    intercept: float,
    slope: float,
    spearman_rho: float,
) -> plt.Figure:
    """Render the single-panel residual-colored scatter."""
    apply_theme(style="paper")

    x = df["gpath"].to_numpy(dtype=np.float64)
    y = df["cogn_global"].to_numpy(dtype=np.float64)
    yhat = intercept + slope * x
    resid = y - yhat

    cmap = PALETTES["diverging"]
    # Symmetric color limits about 0 so the diverging colormap is centered:
    # green = positive residual ("resilient"), magenta = negative ("vulnerable").
    abs_max = float(np.max(np.abs(resid)))
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)

    fig, ax = plt.subplots(figsize=(9, 7))

    sc = ax.scatter(
        x, y,
        c=resid, cmap=cmap, norm=norm,
        s=42, alpha=0.85,
        edgecolors="white", linewidths=0.6,
        zorder=3,
    )

    # Dashed OLS regression line spanning the x range.
    x_line = np.linspace(float(x.min()), float(x.max()), 200)
    y_line = intercept + slope * x_line
    ax.plot(
        x_line, y_line,
        linestyle="--", color="#444444", linewidth=1.4,
        label=f"OLS: cogn_global = {intercept:.3f} + ({slope:.3f}) * gpath",
        zorder=2,
    )

    # Axis labels: keep them descriptive but compact (no title per
    # caption-only convention).
    ax.set_xlabel("gpath  (global pathology score)")
    ax.set_ylabel("cogn_global  (cognition composite)")

    # In-axes annotation: n + Spearman rho.
    ax.text(
        0.98, 0.97,
        f"n = {len(df)}\n"
        + r"$\rho_{\mathrm{Spearman}}$ = "
        + f"{spearman_rho:.3f}",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=11, family="monospace",
        bbox=dict(facecolor="white", edgecolor="#bbbbbb", alpha=0.85,
                  boxstyle="round,pad=0.4"),
        zorder=4,
    )

    # OLS line legend in the lower-left so it does not collide with the
    # n + rho annotation in the upper-right.
    ax.legend(loc="lower left", fontsize=9, frameon=True)

    fmt_axes(ax)

    # Subtle colorbar to the right, labeled by residual sign.
    cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(
        "Residual (cogn_global - OLS_pred):\n+ = resilient, - = vulnerable",
        fontsize=9,
    )
    cbar.outline.set_linewidth(0.5)
    cbar.ax.tick_params(labelsize=8)

    return fig


def _print_report(
    df: pd.DataFrame,
    intercept: float,
    slope: float,
    spearman_rho: float,
    spearman_p: float,
) -> None:
    """Echo verification numbers to stdout."""
    n = len(df)
    cog = df["cogn_global"].to_numpy(dtype=np.float64)
    gp = df["gpath"].to_numpy(dtype=np.float64)
    print("=" * 72)
    print("README Figure 1 -- the question / problem framing")
    print("=" * 72)
    print(f"  n_subjects          : {n}")
    print(f"  mean_cogn_global    : {float(cog.mean()):.6f}")
    print(f"  std_cogn_global     : {float(cog.std(ddof=1)):.6f}")
    print(f"  mean_gpath          : {float(gp.mean()):.6f}")
    print(f"  std_gpath           : {float(gp.std(ddof=1)):.6f}")
    print(f"  ols_intercept       : {intercept:.6f}")
    print(f"  ols_slope           : {slope:.6f}")
    print(f"  spearman_rho        : {spearman_rho:.6f}")
    print(f"  spearman_p_value    : {spearman_p:.6e}")
    print("=" * 72)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits-json", type=Path,
        default=_WORKTREE_ROOT / "outputs/splits.json",
        help="Path to the 5-fold splits.json (canonical ROSMAP cohort).",
    )
    parser.add_argument(
        "--metadata-csv", type=Path,
        default=_WORKTREE_ROOT / "data/metadata_ROSMAP/metadata.csv",
        help="Path to ROSMAP metadata.csv (provides cogn_global + gpath).",
    )
    parser.add_argument(
        "--out-stem", type=Path,
        default=_WORKTREE_ROOT / "figures/fig1_problem",
        help="Output path stem (without extension); save_fig will append .png.",
    )
    parser.add_argument(
        "--expected-n", type=int, default=516,
        help="Expected number of subjects after filtering (asserted).",
    )
    args = parser.parse_args()

    logger.info("[fig1] splits-json   = %s", args.splits_json)
    logger.info("[fig1] metadata-csv  = %s", args.metadata_csv)
    logger.info("[fig1] out-stem      = %s", args.out_stem)

    subject_ids = _union_subject_ids(args.splits_json)
    if len(subject_ids) != args.expected_n:
        raise AssertionError(
            f"splits.json union has n={len(subject_ids)}; expected {args.expected_n}."
        )
    logger.info("[fig1] union subject IDs: n=%d", len(subject_ids))

    df = _load_subject_frame(args.metadata_csv, subject_ids)
    logger.info("[fig1] subject frame: n=%d", len(df))

    intercept, slope = _ols_fit(
        df["gpath"].to_numpy(dtype=np.float64),
        df["cogn_global"].to_numpy(dtype=np.float64),
    )
    spearman = stats.spearmanr(
        df["cogn_global"].to_numpy(dtype=np.float64),
        df["gpath"].to_numpy(dtype=np.float64),
    )
    spearman_rho = float(spearman.statistic)
    spearman_p = float(spearman.pvalue)
    logger.info(
        "[fig1] OLS: intercept=%.6f slope=%.6f | Spearman rho=%.6f (p=%.4g)",
        intercept, slope, spearman_rho, spearman_p,
    )

    # Render figure.
    fig = make_figure(df, intercept, slope, spearman_rho)

    # Delete preexisting PNG (per spec) before writing.
    out_png = args.out_stem.with_suffix(".png")
    if out_png.exists():
        logger.info("[fig1] removing preexisting %s", out_png)
        out_png.unlink()

    # DPI matches the project visual standard (600 DPI, theme default in
    # src/visualization/theme.py::save_fig). An earlier revision used 450
    # to stay under a 1 MB GitHub-viewer ceiling, but the user explicitly
    # authorized any file size for this figure.
    written = save_fig(fig, args.out_stem, dpi=600, formats=("png",))
    plt.close(fig)
    for w in written:
        size_mb = w.stat().st_size / (1024 * 1024)
        logger.info("[fig1] wrote %s (%.3f MB)", w, size_mb)

    _print_report(df, intercept, slope, spearman_rho, spearman_p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
