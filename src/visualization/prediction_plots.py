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
import seaborn as sns
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

    Returns:
        Matplotlib Figure
    """
    setup_seaborn_style()

    fig, ax = plt.subplots(figsize=figsize)

    # Scatter plot
    if predicted_std is not None:
        # Color by uncertainty
        scatter = ax.scatter(
            actual,
            predicted_mean,
            c=predicted_std,
            cmap=get_sequential_cmap(),
            alpha=0.7,
            s=50,
        )
        plt.colorbar(scatter, ax=ax, label="Predicted Uncertainty (σ)")
    else:
        ax.scatter(actual, predicted_mean, alpha=0.7, s=50, color=ACCENT_CORAL)

    # Add identity line
    lims = [
        min(actual.min(), predicted_mean.min()),
        max(actual.max(), predicted_mean.max()),
    ]
    margin = (lims[1] - lims[0]) * 0.05
    lims = [lims[0] - margin, lims[1] + margin]
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

    plt.tight_layout()

    if save_path:
        save_figure(fig, str(save_path))

    return fig


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
