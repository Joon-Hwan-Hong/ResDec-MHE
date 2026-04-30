#!/usr/bin/env python
"""Render a 4-panel calibration figure for ResDec-MHE composite predictions.

Inputs
------
- ``outputs/canonical/p5_canonical_seed42/fold{0..4}/val_predictions_best.npz``
  -> per-fold composite predictions, targets, subject ids. Has NO native
  per-subject sigma — the composite head is a deterministic regressor.
- ``data/canonical/tabpfn_outer_fold{0..4}.npz`` -> per-fold TabPFN-2.6
  outer-fold predictive ``sigma_tabpfn`` (median of [q16, q84]/2). Joined
  onto our predictions by ``ROSMAP_IndividualID``. This is the same proxy
  documented in ``statistical_rigor.json::provenance::sigma_source_note``.
- ``data/metadata_ROSMAP/metadata.csv`` -> ``cogdx`` for AD-dx coloring
  (cogdx in {4, 5} -> AD; otherwise non-AD).

Sigma source
------------
**Confirmed:** ``val_predictions_best.npz`` keys are exactly
``{subject_ids, predictions, targets, epoch, mse, mae, rmse, r2,
pearson_r, spearman_rho}`` -- there is NO ``sigma`` / ``std`` /
``uncertainty`` key. The composite head is deterministic, so calibration
must come from an external uncertainty proxy. We use the canonical TabPFN
sigma proxy (same approach the existing ``statistical_rigor.json``
calibration block uses). Documented in the JSON summary's
``provenance::sigma_source`` field.

Panels
------
A -- Actual vs predicted scatter with +/-1 sigma error bars + identity
     line + R^2 annotated. Subjects colored by AD-dx (cogdx in {4,5}).
B -- Calibration curve: empirical vs nominal coverage at multiple
     quantile levels (0.1, 0.25, 0.5, 0.68, 0.8, 0.9, 0.95) with
     diagonal "perfect calibration" reference.
C -- Residual distribution by predicted-sigma quartile (boxplot). Tests
     whether |residual| scales with sigma_tabpfn -- if calibrated, higher
     sigma quartiles should show wider residual distributions.
D -- PIT histogram. PIT_i = Phi((y_true_i - y_pred_i) / sigma_i). Under
     a Gaussian-calibrated predictor PIT is uniform on [0, 1]. Deviations
     indicate over- (U-shape) or under-confidence (peaked at 0.5).

Outputs
-------
- ``outputs/canonical/interpretability/figures/calibration/
   fig_calibration.{png,pdf}`` (600 DPI, 4-panel)
- ``outputs/canonical/interpretability/calibration_summary.json``
  with per-quantile coverage, PIT KS statistic + p-value, and
  provenance.

Usage
-----
    PYTHONPATH=<worktree-root> \\
    uv run python scripts/resdec_mhe/interpretability/make_calibration_figure.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from src.analysis.resdec_io import load_all_folds  # noqa: E402
from src.visualization.theme import apply_theme  # noqa: E402

logger = logging.getLogger(__name__)


# Quantile levels to evaluate calibration coverage.  Includes the four
# canonical levels reported in statistical_rigor.json (0.5 / 0.68 / 0.8 /
# 0.95) plus three lower-tail levels (0.1, 0.25, 0.9) for a richer curve.
DEFAULT_NOMINAL: tuple[float, ...] = (0.10, 0.25, 0.50, 0.68, 0.80, 0.90, 0.95)

# Reportable canonical levels (subset of DEFAULT_NOMINAL) -- these are the
# four levels reproduced from statistical_rigor.json and reported in the
# script's terminal output / report block.
CANONICAL_LEVELS: tuple[float, ...] = (0.50, 0.68, 0.80, 0.95)

# AD-dx threshold from ROSMAP cogdx: {4, 5} are AD-positive (with or
# without other dementia); 1/2/3/6 are non-AD (NCI / MCI / other).
AD_COGDX_VALUES: frozenset[int] = frozenset({4, 5})

# Color tokens (consistent with make_prediction_scatter_figures.py).
COLOR_AD = "#d62728"     # red
COLOR_NONAD = "#2ca02c"  # green
COLOR_REF = "#888888"    # neutral gray for reference / identity lines


@dataclass(frozen=True)
class CalibrationData:
    """Per-subject calibration tensor across all folds (already joined)."""

    subject_id: np.ndarray  # str, [N]
    fold: np.ndarray        # int,  [N]
    y_true: np.ndarray      # float, [N]
    y_pred: np.ndarray      # float, [N]   (composite predictions)
    sigma: np.ndarray       # float, [N]   (sigma_tabpfn proxy)
    is_ad: np.ndarray       # bool,  [N]   (cogdx in {4, 5})


def load_calibration_data(
    pred_root: Path, tabpfn_dir: Path, metadata_csv: Path, n_folds: int = 5,
) -> CalibrationData:
    """Build the (true, pred, sigma, is_ad) tensor across all 5 folds.

    Joins our composite predictions to TabPFN sigma via
    ``ROSMAP_IndividualID`` (same join the canonical
    ``paired_tests_and_bootstrap.py`` performs), and merges in cogdx
    from the ROSMAP metadata for AD-dx coloring.
    """
    df = load_all_folds(pred_root, tabpfn_dir, n_folds=n_folds)
    sigma_frames = []
    for f in range(n_folds):
        path = tabpfn_dir / f"tabpfn_outer_fold{f}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing TabPFN outer-fold cache: {path}")
        d = np.load(path, allow_pickle=True)
        sigma_frames.append(pd.DataFrame({
            "ROSMAP_IndividualID": d["val_subject_ids"].astype(str),
            "sigma_tabpfn": d["sigma_tabpfn"].astype(np.float64),
        }))
    sigma_df = pd.concat(sigma_frames, ignore_index=True)
    merged = df.merge(sigma_df, on="ROSMAP_IndividualID", how="inner")
    if len(merged) != len(df):
        raise RuntimeError(
            "[calibration] sigma_tabpfn join dropped subjects: "
            f"{len(df)} ours -> {len(merged)} after join."
        )

    if not metadata_csv.exists():
        raise FileNotFoundError(f"metadata.csv not found: {metadata_csv}")
    md = pd.read_csv(metadata_csv)[["ROSMAP_IndividualID", "cogdx"]]
    md["ROSMAP_IndividualID"] = md["ROSMAP_IndividualID"].astype(str)
    merged = merged.merge(md, on="ROSMAP_IndividualID", how="left")
    cogdx_int = pd.to_numeric(merged["cogdx"], errors="coerce")
    is_ad = cogdx_int.isin(AD_COGDX_VALUES).to_numpy(dtype=bool)

    finite = (
        np.isfinite(merged["y_true"].to_numpy())
        & np.isfinite(merged["y_composite"].to_numpy())
        & np.isfinite(merged["sigma_tabpfn"].to_numpy())
        & (merged["sigma_tabpfn"].to_numpy() > 0.0)
    )
    if not finite.all():
        n_drop = int((~finite).sum())
        logger.warning(
            "[calibration] dropping %d subjects with non-finite or "
            "non-positive sigma", n_drop,
        )
    merged = merged.loc[finite].reset_index(drop=True)
    is_ad = is_ad[finite]

    return CalibrationData(
        subject_id=merged["ROSMAP_IndividualID"].to_numpy(),
        fold=merged["fold"].to_numpy(dtype=np.int64),
        y_true=merged["y_true"].to_numpy(dtype=np.float64),
        y_pred=merged["y_composite"].to_numpy(dtype=np.float64),
        sigma=merged["sigma_tabpfn"].to_numpy(dtype=np.float64),
        is_ad=is_ad,
    )


def compute_calibration_metrics(
    data: CalibrationData, nominal: tuple[float, ...] = DEFAULT_NOMINAL,
) -> dict:
    """Return coverage at each nominal level + PIT KS test + summary stats.

    Coverage is defined under a Gaussian assumption: at level p,
    ``z = Phi^{-1}(0.5 + p/2)`` and ``coverage_p = mean(|y_true - y_pred|
    <= z * sigma)``.  PIT = ``Phi((y_true - y_pred) / sigma)``; under
    correct calibration PIT is uniform on [0, 1]; KS test compares the
    empirical PIT distribution to the uniform reference.
    """
    abs_resid = np.abs(data.y_true - data.y_pred)
    coverage: dict[str, float] = {}
    for p in nominal:
        if not (0.0 < p < 1.0):
            raise ValueError(f"Nominal level must be in (0, 1); got {p}")
        z = float(stats.norm.ppf(0.5 + p / 2.0))
        coverage[f"coverage_at_{p}"] = float(np.mean(abs_resid <= z * data.sigma))

    z_resid = (data.y_true - data.y_pred) / data.sigma
    pit = stats.norm.cdf(z_resid)
    ks_stat, ks_p = stats.kstest(pit, "uniform")

    # Pooled R^2 across all 5 folds (same point estimate as bootstrap_r2_ci).
    pooled_r2 = float(r2_score(data.y_true, data.y_pred))

    # Per-sigma-quartile residual stats.
    q = np.quantile(data.sigma, [0.25, 0.5, 0.75])
    q_labels = ["Q1 (low sigma)", "Q2", "Q3", "Q4 (high sigma)"]
    q_idx = np.searchsorted(q, data.sigma, side="right")  # 0..3 inclusive
    q_idx = np.clip(q_idx, 0, 3)
    per_quartile_residual: dict[str, dict[str, float]] = {}
    for i, lbl in enumerate(q_labels):
        m = q_idx == i
        per_quartile_residual[lbl] = {
            "n": int(m.sum()),
            "mean_abs_residual": float(abs_resid[m].mean()) if m.any() else float("nan"),
            "median_abs_residual": float(np.median(abs_resid[m])) if m.any() else float("nan"),
            "mean_sigma": float(data.sigma[m].mean()) if m.any() else float("nan"),
        }

    # Spearman correlation between per-subject |residual| and sigma --
    # if sigma is informative, this should be > 0.
    spearman = stats.spearmanr(abs_resid, data.sigma)
    return {
        "n": int(len(data.y_true)),
        "n_ad": int(data.is_ad.sum()),
        "pooled_r2": pooled_r2,
        "mean_sigma": float(data.sigma.mean()),
        "mean_abs_residual": float(abs_resid.mean()),
        "coverage_by_nominal": coverage,
        "pit_ks_statistic": float(ks_stat),
        "pit_ks_pvalue": float(ks_p),
        "abs_residual_vs_sigma_spearman_rho": float(spearman.statistic),
        "abs_residual_vs_sigma_spearman_pvalue": float(spearman.pvalue),
        "per_sigma_quartile_residual": per_quartile_residual,
        "nominal_levels": list(nominal),
    }


def _annotate_panel(ax: plt.Axes, label: str) -> None:
    """Place a panel label (A/B/C/D) in the upper-left corner."""
    ax.text(
        -0.15, 1.05, label,
        transform=ax.transAxes,
        fontsize=14, fontweight="bold", va="top", ha="left",
    )


def _make_panel_a(ax: plt.Axes, data: CalibrationData, metrics: dict) -> None:
    """Actual vs predicted with +-1 sigma error bars, AD-dx coloring."""
    ad_mask = data.is_ad
    nonad_mask = ~ad_mask

    # Error bars first (so dots overlay), then dots colored by AD-dx.
    for mask, color, label in [
        (nonad_mask, COLOR_NONAD, f"non-AD (n={int(nonad_mask.sum())})"),
        (ad_mask, COLOR_AD, f"AD (cogdx in {{4,5}}, n={int(ad_mask.sum())})"),
    ]:
        if not mask.any():
            continue
        ax.errorbar(
            data.y_true[mask], data.y_pred[mask],
            yerr=data.sigma[mask],
            fmt="o", markersize=4.5, alpha=0.6,
            color=color, ecolor=color, elinewidth=0.5, capsize=0,
            markeredgewidth=0.0, label=label,
        )

    # Identity line spanning the joint range.
    lo = float(min(data.y_true.min(), data.y_pred.min()))
    hi = float(max(data.y_true.max(), data.y_pred.max()))
    pad = 0.05 * (hi - lo)
    span = (lo - pad, hi + pad)
    ax.plot(span, span, linestyle="--", color=COLOR_REF, linewidth=1.0,
            label="y = x", zorder=0)
    ax.set_xlim(span)
    ax.set_ylim(span)
    ax.set_xlabel("Actual cognitive resilience (residual)")
    ax.set_ylabel("Predicted (composite)")
    ax.set_title("A. Actual vs predicted (+/-1 sigma_TabPFN)", fontsize=11)
    ax.text(
        0.03, 0.97,
        f"R^2 = {metrics['pooled_r2']:.4f}\n"
        f"N = {metrics['n']}\n"
        f"mean sigma = {metrics['mean_sigma']:.3f}",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=8, family="monospace",
        bbox=dict(facecolor="white", edgecolor="#aaaaaa", alpha=0.85,
                  boxstyle="round,pad=0.3"),
    )
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    ax.grid(True, linestyle=":", alpha=0.4)


def _make_panel_b(ax: plt.Axes, metrics: dict) -> None:
    """Calibration curve: empirical vs nominal coverage."""
    nominal = list(metrics["nominal_levels"])
    empirical = [metrics["coverage_by_nominal"][f"coverage_at_{p}"] for p in nominal]

    ax.plot([0, 1], [0, 1], linestyle="--", color=COLOR_REF, linewidth=1.0,
            label="perfect calibration")
    ax.plot(
        nominal, empirical,
        marker="o", markersize=6, linewidth=1.5, color="#3b6ea5",
        label="empirical coverage",
    )
    for xn, ye in zip(nominal, empirical):
        ax.annotate(
            f"{ye:.2f}",
            xy=(xn, ye), xytext=(5, 5), textcoords="offset points",
            fontsize=7,
        )
    ax.set_xlabel("Nominal coverage (under Gaussian sigma_TabPFN)")
    ax.set_ylabel("Empirical coverage")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", "box")
    ax.set_title("B. Calibration curve", fontsize=11)
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    ax.grid(True, linestyle=":", alpha=0.4)


def _make_panel_c(ax: plt.Axes, data: CalibrationData) -> None:
    """Residual distribution by sigma_TabPFN quartile."""
    abs_resid = np.abs(data.y_true - data.y_pred)
    q = np.quantile(data.sigma, [0.25, 0.5, 0.75])
    q_idx = np.clip(np.searchsorted(q, data.sigma, side="right"), 0, 3)
    bins = [abs_resid[q_idx == i] for i in range(4)]
    labels = [
        f"Q1\n(sigma<={q[0]:.2f})",
        f"Q2\n({q[0]:.2f}<sigma<={q[1]:.2f})",
        f"Q3\n({q[1]:.2f}<sigma<={q[2]:.2f})",
        f"Q4\n(sigma>{q[2]:.2f})",
    ]
    parts = ax.boxplot(
        bins, positions=[1, 2, 3, 4],
        widths=0.55, patch_artist=True, showfliers=False,
        medianprops={"color": "black", "linewidth": 1.4},
        whiskerprops={"color": "black", "linewidth": 1.0},
        capprops={"color": "black", "linewidth": 1.0},
        boxprops={"linewidth": 1.0},
    )
    cmap = plt.get_cmap("viridis")
    for i, box in enumerate(parts["boxes"]):
        box.set_facecolor(cmap(i / 3.0))
        box.set_alpha(0.55)
        box.set_edgecolor("black")
    # Per-subject jittered dots.
    rng = np.random.default_rng(42)
    for pos, dl in enumerate(bins, start=1):
        if len(dl) == 0:
            continue
        jitter = rng.uniform(-0.10, 0.10, size=len(dl))
        ax.scatter(
            np.full(len(dl), pos) + jitter, dl,
            s=8, color="black", alpha=0.35, edgecolors="none", zorder=3,
        )
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("|y_true - y_pred|")
    ax.set_title("C. Residual magnitude by sigma quartile", fontsize=11)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)


def _make_panel_d(ax: plt.Axes, data: CalibrationData, metrics: dict) -> None:
    """PIT histogram with KS p-value annotation + uniform reference."""
    z_resid = (data.y_true - data.y_pred) / data.sigma
    pit = stats.norm.cdf(z_resid)

    n_bins = 20
    counts, edges = np.histogram(pit, bins=n_bins, range=(0.0, 1.0))
    expected_per_bin = len(pit) / n_bins
    centers = 0.5 * (edges[:-1] + edges[1:])
    bar_w = (edges[1] - edges[0]) * 0.9

    ax.bar(
        centers, counts, width=bar_w,
        color="#3b6ea5", edgecolor="black", alpha=0.65,
        label=f"PIT (n_bins={n_bins})",
    )
    ax.axhline(
        expected_per_bin, color=COLOR_REF, linestyle="--", linewidth=1.0,
        label=f"uniform expectation ({expected_per_bin:.1f}/bin)",
    )
    ax.set_xlabel("PIT = Phi((y_true - y_pred) / sigma)")
    ax.set_ylabel("Count")
    ax.set_xlim(0, 1)
    ax.set_title("D. PIT histogram (uniform if calibrated)", fontsize=11)
    ax.text(
        0.03, 0.97,
        f"KS stat = {metrics['pit_ks_statistic']:.4f}\n"
        f"KS p    = {metrics['pit_ks_pvalue']:.4g}",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=8, family="monospace",
        bbox=dict(facecolor="white", edgecolor="#aaaaaa", alpha=0.85,
                  boxstyle="round,pad=0.3"),
    )
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    ax.grid(True, linestyle=":", alpha=0.4)


def make_figure(data: CalibrationData, metrics: dict) -> plt.Figure:
    """Render the 4-panel calibration figure (A-D in 2x2 layout)."""
    apply_theme(style="paper", use_scienceplots=True)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.5))
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    _make_panel_a(ax_a, data, metrics)
    _make_panel_b(ax_b, metrics)
    _make_panel_c(ax_c, data)
    _make_panel_d(ax_d, data, metrics)

    for ax, label in zip([ax_a, ax_b, ax_c, ax_d], ["A", "B", "C", "D"]):
        _annotate_panel(ax, label)

    fig.suptitle(
        "ResDec-MHE calibration -- composite predictions vs TabPFN-2.6 sigma proxy",
        fontsize=13, y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    return fig


def write_summary_json(
    metrics: dict, out_path: Path,
    pred_root: Path, tabpfn_dir: Path, metadata_csv: Path,
) -> None:
    """Persist the calibration summary + provenance to JSON."""
    record = dict(metrics)
    record["provenance"] = {
        "pred_root": str(pred_root),
        "tabpfn_dir": str(tabpfn_dir),
        "metadata_csv": str(metadata_csv),
        "sigma_source": (
            "TabPFN-2.6 per-subject sigma_tabpfn from "
            "data/canonical/tabpfn_outer_fold{f}.npz; "
            "val_predictions_best.npz has no sigma key (composite head is "
            "deterministic). Same proxy as statistical_rigor.json."
        ),
        "ad_cogdx_values": sorted(AD_COGDX_VALUES),
        "canonical_levels": list(CANONICAL_LEVELS),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2))


def _print_report(metrics: dict) -> None:
    """Echo the key calibration numbers to stdout."""
    print("=" * 72)
    print("Calibration summary -- ResDec-MHE composite (TabPFN sigma proxy)")
    print("=" * 72)
    print(f"  N (subjects)         : {metrics['n']}")
    print(f"  pooled R^2           : {metrics['pooled_r2']:.4f}")
    print(f"  mean sigma           : {metrics['mean_sigma']:.4f}")
    print(f"  mean |residual|      : {metrics['mean_abs_residual']:.4f}")
    print(f"  PIT KS statistic     : {metrics['pit_ks_statistic']:.4f}")
    print(f"  PIT KS p-value       : {metrics['pit_ks_pvalue']:.6g}")
    print(
        f"  Spearman(|r|, sigma) : {metrics['abs_residual_vs_sigma_spearman_rho']:.4f}"
        f"   (p={metrics['abs_residual_vs_sigma_spearman_pvalue']:.4g})"
    )
    print()
    print("Empirical coverage at canonical levels:")
    for p in CANONICAL_LEVELS:
        emp = metrics["coverage_by_nominal"][f"coverage_at_{p}"]
        delta = emp - p
        direction = "under-confident" if delta > 0 else "over-confident"
        print(f"  nominal {p:.2f}  -> empirical {emp:.4f}  (delta={delta:+.4f}, {direction})")
    print("=" * 72)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pred-root", type=Path,
        default=_WORKTREE_ROOT / "outputs/canonical/p5_canonical_seed42",
        help="Directory containing fold{0..N-1}/val_predictions_best.npz",
    )
    parser.add_argument(
        "--tabpfn-dir", type=Path,
        default=_WORKTREE_ROOT / "data/canonical",
        help="Directory containing tabpfn_outer_fold{0..N-1}.npz (provides sigma_tabpfn)",
    )
    parser.add_argument(
        "--metadata-csv", type=Path,
        default=_WORKTREE_ROOT / "data/metadata_ROSMAP/metadata.csv",
        help="ROSMAP metadata for cogdx-based AD-dx coloring",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=_WORKTREE_ROOT / "outputs/canonical/interpretability/figures/calibration",
        help="Output directory for figure files",
    )
    parser.add_argument(
        "--summary-json", type=Path,
        default=_WORKTREE_ROOT / "outputs/canonical/interpretability/calibration_summary.json",
        help="Output path for the per-quantile coverage JSON summary",
    )
    parser.add_argument(
        "--stem", type=str, default="fig_calibration",
        help="File stem (without extension) for PNG / PDF outputs",
    )
    parser.add_argument(
        "--n-folds", type=int, default=5,
        help="Number of outer folds (default 5 for canonical p5_canonical_seed42)",
    )
    parser.add_argument(
        "--nominal-coverage", type=float, nargs="+",
        default=list(DEFAULT_NOMINAL),
        help="Nominal coverage levels to evaluate (default 0.10..0.95)",
    )
    args = parser.parse_args()

    nominal = tuple(float(x) for x in args.nominal_coverage)
    logger.info("[calibration] pred-root        = %s", args.pred_root)
    logger.info("[calibration] tabpfn-dir       = %s", args.tabpfn_dir)
    logger.info("[calibration] metadata-csv     = %s", args.metadata_csv)
    logger.info("[calibration] out-dir          = %s", args.out_dir)
    logger.info("[calibration] summary-json     = %s", args.summary_json)
    logger.info("[calibration] nominal-coverage = %s", list(nominal))

    data = load_calibration_data(
        args.pred_root, args.tabpfn_dir, args.metadata_csv, n_folds=args.n_folds,
    )
    logger.info(
        "[calibration] loaded %d subjects (AD=%d, non-AD=%d)",
        len(data.y_true), int(data.is_ad.sum()), int((~data.is_ad).sum()),
    )

    metrics = compute_calibration_metrics(data, nominal=nominal)

    fig = make_figure(data, metrics)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.out_dir / f"{args.stem}.png"
    pdf_path = args.out_dir / f"{args.stem}.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    logger.info("[calibration] wrote %s", png_path)
    logger.info("[calibration] wrote %s", pdf_path)

    write_summary_json(
        metrics, args.summary_json,
        pred_root=args.pred_root,
        tabpfn_dir=args.tabpfn_dir,
        metadata_csv=args.metadata_csv,
    )
    logger.info("[calibration] wrote %s", args.summary_json)

    _print_report(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
