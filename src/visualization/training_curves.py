"""
Training curves visualization.

Provides publication-quality plots for:
- Training and validation loss curves
- Learning rate schedules
- Metric progression over epochs
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.visualization.config import (
    ACCENT_CORAL,
    ACCENT_TEAL,
    ACCENT_PEACH,
    setup_seaborn_style,
    save_figure,
)

logger = logging.getLogger(__name__)


def load_training_logs(
    log_dir: str | Path,
    metrics_file: str = "metrics.csv",
) -> pd.DataFrame | None:
    """
    Load training logs from CSV file.

    Args:
        log_dir: Directory containing training logs
        metrics_file: Name of metrics CSV file

    Returns:
        DataFrame with training metrics or None if not found
    """
    log_dir = Path(log_dir)
    metrics_path = log_dir / metrics_file

    if metrics_path.exists():
        return pd.read_csv(metrics_path)

    # Try common alternatives
    alternatives = ["training_logs.csv", "history.csv"]
    for alt in alternatives:
        alt_path = log_dir / alt
        if alt_path.exists():
            return pd.read_csv(alt_path)

    logger.warning(f"No training metrics found in {log_dir}")
    return None


def load_tensorboard_scalars(
    log_dir: str | Path,
) -> pd.DataFrame | None:
    """
    Load scalar metrics from TensorBoard event files.

    Args:
        log_dir: Directory containing TensorBoard event files

    Returns:
        DataFrame with columns: step, tag, value
    """
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        logger.warning("tensorboard package not installed, cannot load event files")
        return None

    log_dir = Path(log_dir)

    # Find event files
    event_files = list(log_dir.glob("events.out.tfevents.*"))
    if not event_files:
        # Check subdirectories
        event_files = list(log_dir.glob("*/events.out.tfevents.*"))

    if not event_files:
        logger.warning(f"No TensorBoard event files found in {log_dir}")
        return None

    rows = []
    for event_file in event_files:
        ea = event_accumulator.EventAccumulator(str(event_file.parent))
        ea.Reload()

        for tag in ea.Tags().get("scalars", []):
            for event in ea.Scalars(tag):
                rows.append({
                    "step": event.step,
                    "tag": tag,
                    "value": event.value,
                    "wall_time": event.wall_time,
                })

    if not rows:
        return None

    return pd.DataFrame(rows)


def plot_loss_curves(
    train_loss: np.ndarray | list,
    val_loss: np.ndarray | list | None = None,
    epochs: np.ndarray | list | None = None,
    figsize: tuple[float, float] = (10, 6),
    title: str = "Training Progress",
    save_path: str | Path | None = None,
    log_scale: bool = False,
    dpi: int | None = None,
) -> plt.Figure:
    """
    Plot training and validation loss curves.

    Args:
        train_loss: Training loss values per epoch
        val_loss: Validation loss values per epoch (optional)
        epochs: Epoch numbers (defaults to 1, 2, 3, ...)
        figsize: Figure size
        title: Plot title
        save_path: If provided, save figure to this path
        log_scale: Whether to use log scale for y-axis

    Returns:
        Matplotlib Figure
    """
    setup_seaborn_style()

    train_loss = np.asarray(train_loss)
    if epochs is None:
        epochs = np.arange(1, len(train_loss) + 1)
    else:
        epochs = np.asarray(epochs)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot training loss
    ax.plot(
        epochs, train_loss,
        color=ACCENT_CORAL,
        linewidth=2,
        label="Training Loss",
        marker="o",
        markersize=4,
    )

    # Plot validation loss if provided
    if val_loss is not None:
        val_loss = np.asarray(val_loss)
        ax.plot(
            epochs[:len(val_loss)], val_loss,
            color=ACCENT_TEAL,
            linewidth=2,
            label="Validation Loss",
            marker="s",
            markersize=4,
        )

    if log_scale:
        ax.set_yscale("log")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Add annotations for min values
    min_train_idx = np.argmin(train_loss)
    ax.annotate(
        f"Min: {train_loss[min_train_idx]:.4f}",
        xy=(epochs[min_train_idx], train_loss[min_train_idx]),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=8,
        color=ACCENT_CORAL,
    )

    if val_loss is not None:
        min_val_idx = np.argmin(val_loss)
        ax.annotate(
            f"Min: {val_loss[min_val_idx]:.4f}",
            xy=(epochs[min_val_idx], val_loss[min_val_idx]),
            xytext=(10, -15),
            textcoords="offset points",
            fontsize=8,
            color=ACCENT_TEAL,
        )

    plt.tight_layout()

    if save_path:
        save_kwargs = {"dpi": dpi} if dpi is not None else {}
        save_figure(fig, str(save_path), **save_kwargs)
        logger.info(f"Saved loss curves to {save_path}")

    return fig


def plot_metric_curves(
    metrics: dict[str, np.ndarray | list],
    epochs: np.ndarray | list | None = None,
    figsize: tuple[float, float] = (10, 6),
    title: str = "Training Metrics",
    save_path: str | Path | None = None,
    dpi: int | None = None,
) -> plt.Figure:
    """
    Plot multiple metric curves on the same figure.

    Args:
        metrics: Dict mapping metric name to values per epoch
        epochs: Epoch numbers (defaults to 1, 2, 3, ...)
        figsize: Figure size
        title: Plot title
        save_path: If provided, save figure to this path

    Returns:
        Matplotlib Figure
    """
    setup_seaborn_style()

    # Determine number of epochs from first metric
    first_metric = list(metrics.values())[0]
    n_epochs = len(first_metric)
    if epochs is None:
        epochs = np.arange(1, n_epochs + 1)
    else:
        epochs = np.asarray(epochs)

    # Color palette for multiple metrics
    colors = [ACCENT_CORAL, ACCENT_TEAL, ACCENT_PEACH, "#9467BD", "#8C564B", "#E377C2"]

    fig, ax = plt.subplots(figsize=figsize)

    for i, (name, values) in enumerate(metrics.items()):
        values = np.asarray(values)
        color = colors[i % len(colors)]
        ax.plot(
            epochs[:len(values)], values,
            color=color,
            linewidth=2,
            label=name,
            marker="o",
            markersize=3,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_kwargs = {"dpi": dpi} if dpi is not None else {}
        save_figure(fig, str(save_path), **save_kwargs)
        logger.info(f"Saved metric curves to {save_path}")

    return fig


def plot_learning_rate(
    lr_values: np.ndarray | list,
    steps: np.ndarray | list | None = None,
    figsize: tuple[float, float] = (10, 4),
    title: str = "Learning Rate Schedule",
    save_path: str | Path | None = None,
    dpi: int | None = None,
) -> plt.Figure:
    """
    Plot learning rate schedule over training.

    Args:
        lr_values: Learning rate values
        steps: Step/epoch numbers
        figsize: Figure size
        title: Plot title
        save_path: If provided, save figure to this path

    Returns:
        Matplotlib Figure
    """
    setup_seaborn_style()

    lr_values = np.asarray(lr_values)
    if steps is None:
        steps = np.arange(len(lr_values))
    else:
        steps = np.asarray(steps)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        steps, lr_values,
        color=ACCENT_TEAL,
        linewidth=2,
    )

    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_kwargs = {"dpi": dpi} if dpi is not None else {}
        save_figure(fig, str(save_path), **save_kwargs)
        logger.info(f"Saved learning rate plot to {save_path}")

    return fig


def plot_training_summary(
    log_dir: str | Path,
    output_dir: str | Path | None = None,
    figsize: tuple[float, float] = (12, 8),
    fmt: str = "png",
    dpi: int | None = None,
) -> list[Path]:
    """
    Generate summary training plots from log directory.

    Attempts to load metrics from CSV or TensorBoard event files
    and generates loss curves, metric curves, and learning rate plots.

    Args:
        log_dir: Directory containing training logs
        output_dir: Directory to save plots (defaults to log_dir/plots)
        figsize: Figure size for combined plots
        fmt: Output format (png, pdf, svg)

    Returns:
        List of paths to generated plot files
    """
    log_dir = Path(log_dir)
    if output_dir is None:
        output_dir = log_dir / "plots"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = []

    # Try loading from CSV first
    df = load_training_logs(log_dir)

    if df is None:
        # Try TensorBoard
        df = load_tensorboard_scalars(log_dir)
        if df is not None:
            # Pivot TensorBoard format to wide format
            df = df.pivot_table(index="step", columns="tag", values="value").reset_index()

    if df is None:
        logger.warning("No training logs found, cannot generate training curves")
        return generated

    # Generate loss curves
    train_loss_col = None
    val_loss_col = None

    for col in df.columns:
        col_lower = col.lower()
        if "train" in col_lower and "loss" in col_lower:
            train_loss_col = col
        elif "val" in col_lower and "loss" in col_lower:
            val_loss_col = col

    if train_loss_col:
        fig = plot_loss_curves(
            train_loss=df[train_loss_col].dropna().values,
            val_loss=df[val_loss_col].dropna().values if val_loss_col else None,
            save_path=output_dir / f"loss_curves.{fmt}",
            dpi=dpi,
        )
        plt.close(fig)
        generated.append(output_dir / f"loss_curves.{fmt}")

    # Generate learning rate plot if available
    lr_col = None
    for col in df.columns:
        if "lr" in col.lower() or "learning_rate" in col.lower():
            lr_col = col
            break

    if lr_col:
        fig = plot_learning_rate(
            lr_values=df[lr_col].dropna().values,
            save_path=output_dir / f"learning_rate.{fmt}",
            dpi=dpi,
        )
        plt.close(fig)
        generated.append(output_dir / f"learning_rate.{fmt}")

    logger.info(f"Generated {len(generated)} training curve plots in {output_dir}")
    return generated
