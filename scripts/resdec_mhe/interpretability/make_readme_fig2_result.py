"""README Figure 2: the result + null collapse.

Renders a 2-panel figure to ``figures/fig2_result.png``:

  Left panel: predicted vs actual cogn_global, all 516 subjects (5-fold CV
              val predictions concatenated). Dashed identity line. Pooled
              R^2 annotated upper-left. Color = continuous residual
              (cogn_global - predicted), PiYG diverging colormap.

  Right panel: null collapse strip. 50 jittered dots for the permutation-
               null R^2 distribution, KDE underneath. Big bold dot at the
               canonical R^2 (BASELINE_COLORS["ResDec-MHE"]). Vertical
               dashed line at the canonical R^2. Annotated z + p.

Data sources (read fresh on every run — no hardcoded numbers):
  - outputs/canonical/p5_canonical_seed42/fold{0..4}/val_predictions_best.npz
      keys ``predictions`` (composite — already includes TabPFN base; do NOT
      add y_tabpfn again, per ``feedback_verify_y_semantics.md``) and
      ``targets`` (actual cogn_global).
  - outputs/canonical/p5_canonical_seed42/best_vs_tabpfn_summary.json
      per-fold R^2 used for the per-fold mean +- std annotation. The
      pooled R^2 is recomputed from the concatenated 516-vector.
  - outputs/canonical/permutation_test_n50_full/permutation_summary.json
      ``null_mean_r2_per_perm`` (50 floats), ``z_under_null``,
      ``p_value_one_sided``, ``null_mean``, ``null_std``,
      ``canonical_mean_r2`` (= 0.4436).

See ``docs/plans/2026-05-02-readme-redesign-design.md`` Figures section
"Figure 2 (PNG): the result" for the spec this implements.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # must precede pyplot import

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from src.visualization.theme import (
    BASELINE_COLORS,
    PALETTES,
    apply_theme,
    fmt_axes,
    save_fig,
)

logger = logging.getLogger(__name__)

# Reproducible jitter for the null-strip dots (no random seed elsewhere).
_JITTER_SEED = 42


def _load_per_fold_predictions(canonical_dir: Path, n_folds: int = 5):
    """Concatenate val predictions across folds.

    Returns
    -------
    pred : (N,) float64
    actual : (N,) float64
    fold_ids : (N,) int64

    Notes
    -----
    ``predictions`` in the .npz is the COMPOSITE (Sum f_hat + y_tabpfn),
    NOT the residual. See ``feedback_verify_y_semantics.md``.
    """
    preds, actuals, fold_ids = [], [], []
    for f in range(n_folds):
        p = canonical_dir / f"fold{f}/val_predictions_best.npz"
        if not p.exists():
            raise FileNotFoundError(f"missing per-fold predictions: {p}")
        d = np.load(p, allow_pickle=True)
        preds.append(np.asarray(d["predictions"], dtype=np.float64))
        actuals.append(np.asarray(d["targets"], dtype=np.float64))
        fold_ids.append(np.full(d["predictions"].shape[0], f, dtype=np.int64))
    return (
        np.concatenate(preds),
        np.concatenate(actuals),
        np.concatenate(fold_ids),
    )


def _pooled_r2(pred: np.ndarray, actual: np.ndarray) -> float:
    """1 - SS_res / SS_tot over the pooled vector (drops non-finite entries)."""
    valid = np.isfinite(pred) & np.isfinite(actual)
    p = pred[valid]
    a = actual[valid]
    ss_res = float(np.sum((a - p) ** 2))
    ss_tot = float(np.sum((a - a.mean()) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _per_fold_r2_from_summary(summary_json: Path) -> tuple[list[float], float, float]:
    """Read per-fold ``ours.r2`` from ``best_vs_tabpfn_summary.json``.

    Returns (per_fold_list, mean, std-with-ddof-1).
    """
    with summary_json.open() as fh:
        summary = json.load(fh)
    per_fold = [float(rec["ours"]["r2"]) for rec in summary["per_fold"]]
    arr = np.asarray(per_fold, dtype=np.float64)
    return per_fold, float(arr.mean()), float(arr.std(ddof=1))


def _draw_left_panel(ax, pred: np.ndarray, actual: np.ndarray, pooled_r2: float):
    """Predicted-vs-actual scatter colored by residual."""
    valid = np.isfinite(pred) & np.isfinite(actual)
    p = pred[valid]
    a = actual[valid]
    residual = a - p  # actual - predicted

    # Symmetric color limits around 0 so the diverging map is centered.
    vmax = float(np.nanmax(np.abs(residual)))
    cmap = PALETTES["diverging"]  # PiYG
    sc = ax.scatter(
        p, a,
        c=residual, cmap=cmap, vmin=-vmax, vmax=vmax,
        s=18, linewidths=0.4, edgecolors="white",
        zorder=3,
    )

    # Identity line (dashed) across the visible data range.
    lo = float(min(np.nanmin(p), np.nanmin(a)))
    hi = float(max(np.nanmax(p), np.nanmax(a)))
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    lo -= pad
    hi += pad
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="#555555",
            linewidth=1.0, zorder=2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")

    # Pooled R^2 annotation, upper-left corner.
    ax.text(
        0.04, 0.96,
        f"Pooled $R^2$ = {pooled_r2:.3f}\n$N$ = {valid.sum()}",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="#cccccc",
                  boxstyle="round,pad=0.3", linewidth=0.5),
        zorder=5,
    )

    ax.set_xlabel("Predicted cogn_global")
    ax.set_ylabel("Actual cogn_global")

    # Colorbar — slim, on the right side of the panel.
    cbar = ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Residual (actual $-$ predicted)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)


def _draw_right_panel(
    ax,
    null_r2: np.ndarray,
    canonical_r2: float,
    z_under_null: float,
    p_one_sided: float,
    null_mean: float,
    null_std: float,
):
    """Null R^2 strip + KDE + canonical bold dot."""
    rng = np.random.default_rng(_JITTER_SEED)

    # --- KDE underneath ---
    # Use scipy gaussian_kde (Scott's rule by default — deterministic).
    xmin = -0.6
    xmax = 0.5
    grid = np.linspace(xmin, xmax, 400)
    kde = gaussian_kde(null_r2)
    density = kde(grid)
    # Scale density so the curve sits visually below the dot strip
    # (peak around y = -0.7); the strip lives near y = 0 with jitter.
    density_max = float(density.max()) if density.max() > 0 else 1.0
    kde_y_scale = 0.6
    kde_y = -density / density_max * kde_y_scale - 0.1

    # Fill the KDE to make it more visible.
    baseline = -0.1 - kde_y_scale - 0.02  # a touch below the lowest KDE point
    ax.fill_between(
        grid, kde_y, baseline,
        color="#cccccc", alpha=0.6, linewidth=0, zorder=1,
    )
    ax.plot(grid, kde_y, color="#777777", linewidth=1.0, zorder=2)

    # --- Null dots (50 perms), jittered around y=0 ---
    jitter_strength = 0.08
    yj = rng.uniform(-jitter_strength, jitter_strength, size=null_r2.shape[0])
    ax.scatter(
        null_r2, yj,
        s=18, color="#555555", alpha=0.75,
        edgecolors="white", linewidths=0.4, zorder=4,
    )

    # --- Canonical R^2 bold dot ---
    canonical_color = BASELINE_COLORS["ResDec-MHE"]
    ax.scatter(
        [canonical_r2], [0.0],
        s=160, color=canonical_color,
        edgecolors="white", linewidths=1.4, zorder=6,
    )
    # Vertical dashed line at canonical R^2 for emphasis.
    ax.axvline(
        canonical_r2, linestyle="--", color=canonical_color,
        linewidth=1.0, alpha=0.7, zorder=3,
    )

    # --- Annotations ---
    # z + p, top-right corner.
    ax.text(
        0.97, 0.96,
        f"$z$ = {z_under_null:.2f}\n$p$ = {p_one_sided:.4f} (= 1 / {null_r2.size + 1})",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="#cccccc",
                  boxstyle="round,pad=0.3", linewidth=0.5),
        zorder=7,
    )

    # Caption-style label for the null strip — arrow points to the strip
    # cluster (around the null mean).
    ax.annotate(
        f"Permutation null ($N$ = {null_r2.size})\n"
        f"mean = {null_mean:.3f}, std = {null_std:.3f}",
        xy=(null_mean, 0.0),
        xytext=(null_mean - 0.1, 0.55),
        fontsize=8, ha="center", va="bottom",
        arrowprops=dict(arrowstyle="->", color="#555555", lw=0.7,
                        shrinkA=0, shrinkB=4),
        zorder=7,
    )

    # Caption-style label for the canonical dot.
    ax.annotate(
        f"ResDec-MHE canonical\n$R^2$ = {canonical_r2:.4f}",
        xy=(canonical_r2, 0.0),
        xytext=(canonical_r2 - 0.08, 0.55),
        fontsize=8, ha="center", va="bottom",
        color=canonical_color,
        arrowprops=dict(arrowstyle="->", color=canonical_color, lw=0.8,
                        shrinkA=0, shrinkB=8),
        zorder=7,
    )

    # Axis cosmetics: hide y-axis ticks (the y dimension is jitter-only).
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.85, 0.85)
    ax.set_yticks([])
    ax.set_xlabel("$R^2$ (cross-validation mean)")
    # Zero-line for reference (R^2 = 0 == predict-the-mean baseline).
    ax.axvline(0.0, color="#999999", linestyle=":", linewidth=0.7, zorder=1)


def _build_figure(
    pred: np.ndarray,
    actual: np.ndarray,
    pooled_r2: float,
    null_r2: np.ndarray,
    canonical_r2: float,
    z_under_null: float,
    p_one_sided: float,
    null_mean: float,
    null_std: float,
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    _draw_left_panel(axes[0], pred, actual, pooled_r2)
    _draw_right_panel(
        axes[1],
        null_r2=null_r2,
        canonical_r2=canonical_r2,
        z_under_null=z_under_null,
        p_one_sided=p_one_sided,
        null_mean=null_mean,
        null_std=null_std,
    )
    for ax in axes:
        fmt_axes(ax)
    fig.tight_layout()
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--canonical-dir",
        type=Path,
        default=_WORKTREE_ROOT / "outputs/canonical/p5_canonical_seed42",
        help="Directory holding fold{N}/val_predictions_best.npz "
             "and best_vs_tabpfn_summary.json",
    )
    parser.add_argument(
        "--permutation-summary",
        type=Path,
        default=(
            _WORKTREE_ROOT
            / "outputs/canonical/permutation_test_n50_full/permutation_summary.json"
        ),
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument(
        "--out-stem",
        type=Path,
        default=_WORKTREE_ROOT / "figures/fig2_result",
        help="Output PNG stem (extension is appended).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    apply_theme("paper")

    # --- Left panel: pooled predictions across folds ---
    canonical_dir = Path(args.canonical_dir)
    pred, actual, fold_ids = _load_per_fold_predictions(
        canonical_dir, n_folds=args.n_folds,
    )
    pooled_r2 = _pooled_r2(pred, actual)
    summary_json = canonical_dir / "best_vs_tabpfn_summary.json"
    per_fold_r2, per_fold_mean, per_fold_std = _per_fold_r2_from_summary(summary_json)

    # --- Right panel: permutation null ---
    perm_path = Path(args.permutation_summary)
    with perm_path.open() as fh:
        perm = json.load(fh)
    null_r2 = np.asarray(perm["null_mean_r2_per_perm"], dtype=np.float64)
    canonical_r2 = float(perm["canonical_mean_r2"])
    z_under_null = float(perm["z_under_null"])
    p_one_sided = float(perm["p_value_one_sided"])
    null_mean = float(perm["null_mean"])
    null_std = float(perm["null_std"])

    fig = _build_figure(
        pred=pred,
        actual=actual,
        pooled_r2=pooled_r2,
        null_r2=null_r2,
        canonical_r2=canonical_r2,
        z_under_null=z_under_null,
        p_one_sided=p_one_sided,
        null_mean=null_mean,
        null_std=null_std,
    )
    written = save_fig(fig, args.out_stem, formats=("png",))
    plt.close(fig)

    # --- Verification stdout report (PRIMARY-FILE values only) ---
    n_subjects = int((np.isfinite(pred) & np.isfinite(actual)).sum())
    print("=" * 72)
    print("README Figure 2 — verification (all values from primary files)")
    print("=" * 72)
    print(f"Left panel — predicted vs actual:")
    print(f"  N subjects (pooled, finite): {n_subjects}")
    print(f"  Per-fold N: {[int((fold_ids == f).sum()) for f in range(args.n_folds)]}")
    print(f"  Pooled R^2:                  {pooled_r2:.6f}")
    print(f"  Per-fold R^2 list:           {[round(x, 4) for x in per_fold_r2]}")
    print(f"  Per-fold mean R^2:           {per_fold_mean:.6f}")
    print(f"  Per-fold std R^2 (ddof=1):   {per_fold_std:.6f}")
    print(f"")
    print(f"Right panel — permutation null:")
    print(f"  N permutations:              {null_r2.size}")
    print(f"  Null mean R^2:               {null_mean:.6f}")
    print(f"  Null std R^2:                {null_std:.6f}")
    print(f"  z (canonical vs null):       {z_under_null:.6f}")
    print(f"  p (one-sided, empirical):    {p_one_sided:.6f}")
    print(f"  canonical_mean_r2 (file):    {canonical_r2:.6f}")
    print(f"")
    for path in written:
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"Wrote: {path}  ({size_mb:.3f} MB)")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
