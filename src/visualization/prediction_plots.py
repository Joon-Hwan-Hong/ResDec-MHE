"""
Prediction and uncertainty visualization plots.

Provides publication-quality plots for:
- Predicted vs actual scatter plots
- Uncertainty calibration
- Residual analysis
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.visualization.config import (
    ACCENT_CORAL,
    ACCENT_TEAL,
    get_sequential_cmap,
    setup_seaborn_style,
    save_figure,
)


def plot_predicted_vs_actual(
    predicted_mean: np.ndarray,
    actual: np.ndarray,
    predicted_std: np.ndarray | None = None,
    figsize: tuple[float, float] = (8, 8),
    title: str = "Predicted vs Actual Cognition",
    save_path: str | Path | None = None,
    *,
    add_marginals: bool = False,
    color_by: np.ndarray | None = None,
    color_label: str | None = None,
    color_palette: dict | None = None,
    show_identity: bool = True,
    show_legend: bool = True,
    scatter_size: float = 50,
) -> plt.Figure:
    """
    Plot predicted vs actual values with optional uncertainty.

    Args:
        predicted_mean: Predicted mean values
        actual: Actual target values
        predicted_std: Predicted uncertainty (std) for error bars
        figsize: Figure size
        title: Plot title
        save_path: If provided, save figure to this path
        add_marginals: If True, add KDE marginal panels on top + right of
            the main scatter (Seaborn JointGrid-like layout).
        color_by: Optional categorical color array (len == n). When set,
            scatter and (when ``add_marginals``) marginal KDEs are split
            per category.
        color_label: Optional legend title for the color-by category axis.
        color_palette: Optional ``{category: hex_color}`` dict. If omitted
            and ``color_by`` is set, falls back to ``tab10``.

    Returns:
        Matplotlib Figure
    """
    setup_seaborn_style()

    # Filter NaN values before plotting and regression
    valid_mask = np.isfinite(actual) & np.isfinite(predicted_mean)
    if predicted_std is not None:
        valid_mask &= np.isfinite(predicted_std)
    actual = actual[valid_mask]
    predicted_mean = predicted_mean[valid_mask]
    if predicted_std is not None:
        predicted_std = predicted_std[valid_mask]
    if color_by is not None:
        color_by = np.asarray(color_by)[valid_mask]

    if add_marginals:
        # JointGrid-style layout: main + top + right.
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(
            2, 2,
            width_ratios=(4, 1),
            height_ratios=(1, 4),
            wspace=0.05,
            hspace=0.05,
        )
        ax = fig.add_subplot(gs[1, 0])
        ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax)
        # Hide tick labels on marginals to keep main axes clean.
        plt.setp(ax_top.get_xticklabels(), visible=False)
        plt.setp(ax_right.get_yticklabels(), visible=False)
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax_top = None
        ax_right = None

    # Scatter plot — three branches: color_by (categorical), predicted_std
    # (continuous uncertainty colorbar), or single accent color.
    if color_by is not None:
        categories = list(dict.fromkeys(np.asarray(color_by).tolist()))
        if color_palette is None:
            tab10 = list(plt.get_cmap("tab10").colors)
            color_palette = {c: tab10[i % len(tab10)] for i, c in enumerate(categories)}
        for cat in categories:
            m = np.asarray(color_by) == cat
            ax.scatter(
                actual[m],
                predicted_mean[m],
                alpha=0.7,
                s=scatter_size,
                color=color_palette.get(cat, "#777777"),
                label=str(cat),
                edgecolor="white",
                linewidth=0.4,
            )
    elif predicted_std is not None:
        # Color by uncertainty
        scatter = ax.scatter(
            actual,
            predicted_mean,
            c=predicted_std,
            cmap=get_sequential_cmap(),
            alpha=0.7,
            s=scatter_size,
        )
        plt.colorbar(scatter, ax=ax, label="Predicted Uncertainty (σ)")
    else:
        ax.scatter(actual, predicted_mean, alpha=0.7, s=scatter_size, color=ACCENT_CORAL)

    # Compute axis limits from data range.
    lims = [
        min(actual.min(), predicted_mean.min()),
        max(actual.max(), predicted_mean.max()),
    ]
    margin = (lims[1] - lims[0]) * 0.05
    lims = [lims[0] - margin, lims[1] + margin]

    # Identity line — gated so callers can drop it (user pref for the
    # lab-meeting prediction scatters: keep regression fit only).
    if show_identity:
        ax.plot(lims, lims, "k--", alpha=0.5, label="Identity")

    # Add regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(actual, predicted_mean)
    x_line = np.array(lims)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color=ACCENT_TEAL, label=f"Fit (R²={r_value**2:.3f})")

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Actual Cognition Score")
    ax.set_ylabel("Predicted Cognition Score")
    ax.set_title(title)
    if show_legend:
        if color_by is not None and color_label is not None:
            ax.legend(loc="lower right", title=color_label)
        else:
            ax.legend(loc="lower right")

    # Add metrics annotation
    rmse = np.sqrt(np.mean((predicted_mean - actual) ** 2))
    mae = np.mean(np.abs(predicted_mean - actual))
    ax.text(
        0.05, 0.95,
        f"RMSE: {rmse:.3f}\nMAE: {mae:.3f}\nR²: {r_value**2:.3f}",
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # KDE marginals — per-category if color_by, otherwise pooled.
    if add_marginals:
        _draw_marginal_kdes(
            ax_top=ax_top,
            ax_right=ax_right,
            actual=actual,
            predicted_mean=predicted_mean,
            color_by=color_by,
            color_palette=color_palette,
            single_color=ACCENT_CORAL,
        )
        # Strip spines + ticks from marginal panels for visual balance.
        for marg in (ax_top, ax_right):
            marg.tick_params(left=False, bottom=False, top=False, right=False,
                             which="both")
            for s in ("top", "right"):
                marg.spines[s].set_visible(False)
            # User pref: no grid + no tick labels on density axis.
            marg.grid(False)
        # Drop the [0, 1]-style tick labels on the density axis.
        ax_top.set_yticks([])
        ax_right.set_xticks([])
        ax_top.set_ylabel("")
        ax_right.set_xlabel("")
    else:
        plt.tight_layout()

    if save_path:
        save_figure(fig, str(save_path))

    return fig


def _draw_marginal_kdes(
    *,
    ax_top,
    ax_right,
    actual: np.ndarray,
    predicted_mean: np.ndarray,
    color_by: np.ndarray | None,
    color_palette: dict | None,
    single_color: str,
) -> None:
    """Draw KDE on top (over actual) and right (over predicted) marginals.

    Per-category if color_by is provided, else pooled. Uses scipy.stats.gaussian_kde
    so we have no extra dependencies. Falls back to histograms when KDE is
    degenerate (e.g., < 2 unique values in a category).
    """
    def _kde_or_hist(values: np.ndarray, color: str, axis: str, ax_):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size < 2 or float(np.std(values)) < 1e-12:
            # Degenerate — draw a thin histogram instead.
            if axis == "x":
                ax_.hist(values, bins=8, color=color, alpha=0.5, orientation="vertical")
            else:
                ax_.hist(values, bins=8, color=color, alpha=0.5, orientation="horizontal")
            return
        kde = stats.gaussian_kde(values)
        lo, hi = float(values.min()), float(values.max())
        pad = (hi - lo) * 0.1 + 1e-6
        grid = np.linspace(lo - pad, hi + pad, 200)
        density = kde(grid)
        if axis == "x":
            ax_.fill_between(grid, density, color=color, alpha=0.3)
            ax_.plot(grid, density, color=color, linewidth=1.0)
        else:
            ax_.fill_betweenx(grid, density, color=color, alpha=0.3)
            ax_.plot(density, grid, color=color, linewidth=1.0)

    if color_by is None:
        _kde_or_hist(actual, single_color, "x", ax_top)
        _kde_or_hist(predicted_mean, single_color, "y", ax_right)
        return

    categories = list(dict.fromkeys(np.asarray(color_by).tolist()))
    palette = color_palette or {}
    for cat in categories:
        m = np.asarray(color_by) == cat
        c = palette.get(cat, "#777777")
        _kde_or_hist(actual[m], c, "x", ax_top)
        _kde_or_hist(predicted_mean[m], c, "y", ax_right)


def plot_calibration_curve(
    calibration_df: pd.DataFrame,
    figsize: tuple[float, float] = (8, 6),
    title: str = "Uncertainty Calibration",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot calibration curve comparing expected vs observed coverage.

    Args:
        calibration_df: DataFrame with columns [level, expected_coverage, observed_coverage]
        figsize: Figure size
        title: Plot title
        save_path: If provided, save figure to this path

    Returns:
        Matplotlib Figure
    """
    setup_seaborn_style()

    fig, ax = plt.subplots(figsize=figsize)

    # Bar positions
    x = np.arange(len(calibration_df))
    width = 0.35

    # Plot expected and observed
    bars1 = ax.bar(
        x - width / 2,
        calibration_df["expected_coverage"],
        width,
        label="Expected",
        color=ACCENT_TEAL,
        alpha=0.8,
    )
    bars2 = ax.bar(
        x + width / 2,
        calibration_df["observed_coverage"],
        width,
        label="Observed",
        color=ACCENT_CORAL,
        alpha=0.8,
    )

    # Add calibration error annotations
    if "calibration_error" in calibration_df.columns:
        for i, (_, row) in enumerate(calibration_df.iterrows()):
            err = row["calibration_error"]
            color = "green" if abs(err) < 0.05 else "red"
            ax.annotate(
                f"{err:+.2f}",
                xy=(x[i], max(row["expected_coverage"], row["observed_coverage"]) + 0.02),
                ha="center",
                fontsize=9,
                color=color,
            )

    ax.set_xlabel("Confidence Level")
    ax.set_ylabel("Coverage")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(calibration_df["level"])
    ax.legend()
    ax.set_ylim(0, 1.1)

    # Add perfect calibration line
    ax.axhline(0.683, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(0.954, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(0.997, color="gray", linestyle=":", alpha=0.5)

    plt.tight_layout()

    if save_path:
        save_figure(fig, str(save_path))

    return fig


def plot_residuals(
    predicted_mean: np.ndarray,
    actual: np.ndarray,
    predicted_std: np.ndarray | None = None,
    figsize: tuple[float, float] = (12, 5),
    title: str = "Residual Analysis",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot residual analysis (residuals vs predicted and histogram).

    Args:
        predicted_mean: Predicted mean values
        actual: Actual target values
        predicted_std: Predicted uncertainty (std)
        figsize: Figure size
        title: Plot title
        save_path: If provided, save figure to this path

    Returns:
        Matplotlib Figure
    """
    setup_seaborn_style()

    # Filter NaN values
    valid_mask = np.isfinite(actual) & np.isfinite(predicted_mean)
    if predicted_std is not None:
        valid_mask &= np.isfinite(predicted_std)
    actual = actual[valid_mask]
    predicted_mean = predicted_mean[valid_mask]
    if predicted_std is not None:
        predicted_std = predicted_std[valid_mask]

    residuals = actual - predicted_mean

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Residuals vs Predicted
    ax1 = axes[0]
    if predicted_std is not None:
        scatter = ax1.scatter(
            predicted_mean,
            residuals,
            c=predicted_std,
            cmap=get_sequential_cmap(),
            alpha=0.7,
            s=50,
        )
        plt.colorbar(scatter, ax=ax1, label="Predicted σ")
    else:
        ax1.scatter(predicted_mean, residuals, alpha=0.7, s=50, color=ACCENT_CORAL)

    ax1.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Predicted Value")
    ax1.set_ylabel("Residual (Actual - Predicted)")
    ax1.set_title("Residuals vs Predicted")

    # Residual histogram
    ax2 = axes[1]
    ax2.hist(residuals, bins=30, color=ACCENT_CORAL, alpha=0.7, edgecolor="black")
    ax2.axvline(0, color="black", linestyle="--", alpha=0.5)
    ax2.axvline(residuals.mean(), color=ACCENT_TEAL, linestyle="-", alpha=0.8, label=f"Mean: {residuals.mean():.3f}")
    ax2.set_xlabel("Residual")
    ax2.set_ylabel("Count")
    ax2.set_title("Residual Distribution")
    ax2.legend()

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        save_figure(fig, str(save_path))

    return fig


def plot_uncertainty_vs_error(
    predicted_mean: np.ndarray,
    actual: np.ndarray,
    predicted_std: np.ndarray,
    figsize: tuple[float, float] = (8, 6),
    title: str = "Uncertainty vs Prediction Error",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot relationship between predicted uncertainty and actual error.

    For well-calibrated models, higher uncertainty should correlate with higher error.

    Args:
        predicted_mean: Predicted mean values
        actual: Actual target values
        predicted_std: Predicted uncertainty (std)
        figsize: Figure size
        title: Plot title
        save_path: If provided, save figure to this path

    Returns:
        Matplotlib Figure
    """
    setup_seaborn_style()

    # Filter NaN values
    valid_mask = np.isfinite(actual) & np.isfinite(predicted_mean) & np.isfinite(predicted_std)
    actual = actual[valid_mask]
    predicted_mean = predicted_mean[valid_mask]
    predicted_std = predicted_std[valid_mask]

    absolute_error = np.abs(actual - predicted_mean)

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(predicted_std, absolute_error, alpha=0.6, s=50, color=ACCENT_CORAL)

    # Add regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(predicted_std, absolute_error)
    x_line = np.array([predicted_std.min(), predicted_std.max()])
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color=ACCENT_TEAL, label=f"Fit (r={r_value:.3f}, p={p_value:.2e})")

    # Add identity line (perfect calibration)
    ax.plot(x_line, x_line, "k--", alpha=0.5, label="Identity (perfect calibration)")

    ax.set_xlabel("Predicted Uncertainty (σ)")
    ax.set_ylabel("Absolute Error")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()

    if save_path:
        save_figure(fig, str(save_path))

    return fig


def plot_uncertainty_correlates(
    correlates_df: pd.DataFrame,
    figsize: tuple[float, float] = (10, 6),
    title: str = "Uncertainty Correlates",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot correlation of uncertainty with covariates as bar chart.

    Args:
        correlates_df: DataFrame with columns [covariate, correlation, p_value, significant]
        figsize: Figure size
        title: Plot title
        save_path: If provided, save figure to this path

    Returns:
        Matplotlib Figure
    """
    setup_seaborn_style()

    fig, ax = plt.subplots(figsize=figsize)

    # Sort by absolute correlation
    df = correlates_df.copy()
    df["abs_correlation"] = df["correlation"].abs()
    df = df.sort_values("abs_correlation", ascending=True)

    # Color by significance
    colors = [ACCENT_CORAL if sig else "gray" for sig in df["significant"]]

    ax.barh(df["covariate"], df["correlation"], color=colors)

    # Add significance markers
    for i, (_, row) in enumerate(df.iterrows()):
        if row["significant"]:
            ax.text(
                row["correlation"] + 0.02 * np.sign(row["correlation"]),
                i,
                "*" if row["p_value"] < 0.05 else "",
                va="center",
                fontsize=12,
            )

    ax.axvline(0, color="black", linestyle="-", alpha=0.5)
    ax.set_xlabel("Spearman Correlation with Uncertainty")
    ax.set_ylabel("Covariate")
    ax.set_title(title)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=ACCENT_CORAL, label="Significant (p < 0.05)"),
        Patch(facecolor="gray", label="Not significant"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()

    if save_path:
        save_figure(fig, str(save_path))

    return fig


# ---------------------------------------------------------------------------
# ResDec-MHE composite-vs-baseline calibration plots, theme-based.
# ---------------------------------------------------------------------------

from src.visualization.theme import (
    baseline_color as _baseline_color,
    fmt_axes as _fmt_axes,
    save_fig as _theme_save_fig,
)


def plot_calibration_overlay(
    tabpfn_per_fold: list[tuple[np.ndarray, np.ndarray]],
    composite_per_fold: list[tuple[np.ndarray, np.ndarray]],
    *,
    n_bins: int = 10,
    figsize: tuple[float, float] = (7.0, 3.5),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Two-panel reliability diagram: TabPFN-only vs Composite (ResDec-MHE).

    Each panel: predicted-vs-true binned mean per quantile bin, with the
    diagonal reference. Comparing where the residual head improves vs
    hurts calibration.
    """
    if not tabpfn_per_fold or not composite_per_fold:
        raise ValueError("no per-fold predictions provided")

    def reliability(y_true_all, y_pred_all, n_bins):
        bin_edges = np.quantile(y_pred_all, np.linspace(0, 1, n_bins + 1))
        bin_edges[0] -= 1e-9
        labels = pd.cut(y_pred_all, bin_edges, labels=False, include_lowest=True)
        means_pred, means_true = [], []
        for b in range(n_bins):
            mask = labels == b
            if mask.sum() == 0:
                continue
            means_pred.append(float(y_pred_all[mask].mean()))
            means_true.append(float(y_true_all[mask].mean()))
        return np.array(means_pred), np.array(means_true)

    yt_t = np.concatenate([t for t, _ in tabpfn_per_fold])
    yp_t = np.concatenate([p for _, p in tabpfn_per_fold])
    yt_c = np.concatenate([t for t, _ in composite_per_fold])
    yp_c = np.concatenate([p for _, p in composite_per_fold])
    mp_t, mt_t = reliability(yt_t, yp_t, n_bins)
    mp_c, mt_c = reliability(yt_c, yp_c, n_bins)

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True, sharex=True)
    for ax, (mp, mt, name, color) in zip(
        axes,
        [
            (mp_t, mt_t, "TabPFN-2.6", _baseline_color("TabPFN-2.6")),
            (mp_c, mt_c, "Composite (ResDec-MHE)", _baseline_color("ResDec-MHE")),
        ],
    ):
        lo = float(min(mp.min(), mt.min()))
        hi = float(max(mp.max(), mt.max()))
        ax.plot([lo, hi], [lo, hi], color="#888", linewidth=0.6,
                linestyle="--", zorder=1)
        ax.scatter(mp, mt, color=color, s=22, edgecolor="white",
                   linewidth=0.6, zorder=3)
        ax.plot(mp, mt, color=color, linewidth=1.2, zorder=2)
        ax.set_xlabel(f"Mean predicted ({name})")
        _fmt_axes(ax)
    axes[0].set_ylabel("Mean true")
    if save_path is not None:
        _theme_save_fig(fig, save_path)
    return fig
