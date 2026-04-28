"""Tests for training curves visualization."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.visualization.training_curves import (
    load_tensorboard_scalars,
    plot_loss_curves,
    plot_learning_rate,
    plot_training_summary,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_train_loss():
    # Decaying loss over 30 epochs
    return np.exp(-np.arange(30) / 10) + 0.01 * np.random.default_rng(0).standard_normal(30)


@pytest.fixture
def sample_val_loss():
    return np.exp(-np.arange(30) / 12) + 0.2 + 0.02 * np.random.default_rng(1).standard_normal(30)


@pytest.fixture
def sample_lr_values():
    # Warmup + cosine decay pattern
    steps = np.arange(100)
    warmup = np.minimum(steps / 10, 1.0)
    decay = 0.5 * (1 + np.cos(np.pi * steps / 100))
    return (warmup * decay * 1e-3).astype(float)


def _write_tfevents(log_dir, scalars: dict[str, list[float]]) -> None:
    """Write a real TensorBoard events file with the given tag→values mapping."""
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=str(log_dir))
    n = max(len(v) for v in scalars.values())
    for step in range(n):
        for tag, values in scalars.items():
            if step < len(values):
                writer.add_scalar(tag, float(values[step]), step)
    writer.flush()
    writer.close()


# =============================================================================
# load_tensorboard_scalars
# =============================================================================


class TestLoadTensorboardScalars:
    def test_empty_dir_returns_none(self, tmp_path):
        assert load_tensorboard_scalars(tmp_path) is None

    def test_loads_scalars_from_events_file(self, tmp_path):
        _write_tfevents(tmp_path, {
            "train/loss": [1.0, 0.8, 0.6, 0.4, 0.2],
            "val/loss": [1.1, 0.9, 0.7, 0.5, 0.3],
        })

        df = load_tensorboard_scalars(tmp_path)
        assert df is not None
        assert set(df.columns) == {"step", "tag", "value", "wall_time"}
        assert set(df["tag"].unique()) == {"train/loss", "val/loss"}
        assert len(df) == 10

    def test_searches_subdirectories(self, tmp_path):
        nested = tmp_path / "version_0"
        nested.mkdir()
        _write_tfevents(nested, {"loss": [1.0, 0.5]})

        df = load_tensorboard_scalars(tmp_path)
        assert df is not None
        assert len(df) == 2

    def test_no_scalars_returns_none(self, tmp_path):
        # Write an events file with no scalars — SummaryWriter still creates the file header
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(tmp_path))
        writer.flush()
        writer.close()
        # File exists but has no scalar events → returns None
        result = load_tensorboard_scalars(tmp_path)
        assert result is None


# =============================================================================
# plot_loss_curves
# =============================================================================


class TestPlotLossCurves:
    def test_train_only(self, sample_train_loss):
        fig = plot_loss_curves(train_loss=sample_train_loss)
        assert isinstance(fig, plt.Figure)
        ax = fig.get_axes()[0]
        assert ax.get_xlabel() == "Epoch"
        assert ax.get_ylabel() == "Loss"
        # Only training line
        assert len(ax.lines) == 1
        plt.close(fig)

    def test_train_and_val(self, sample_train_loss, sample_val_loss):
        fig = plot_loss_curves(train_loss=sample_train_loss, val_loss=sample_val_loss)
        ax = fig.get_axes()[0]
        assert len(ax.lines) == 2
        legend = ax.get_legend()
        assert legend is not None
        labels = [t.get_text() for t in legend.get_texts()]
        assert "Training Loss" in labels
        assert "Validation Loss" in labels
        plt.close(fig)

    def test_custom_epochs(self, sample_train_loss):
        custom_epochs = np.arange(10, 40)  # 30 values to match sample_train_loss
        fig = plot_loss_curves(train_loss=sample_train_loss, epochs=custom_epochs)
        ax = fig.get_axes()[0]
        xdata = ax.lines[0].get_xdata()
        assert xdata[0] == 10
        assert xdata[-1] == 39
        plt.close(fig)

    def test_log_scale(self, sample_train_loss):
        fig = plot_loss_curves(train_loss=sample_train_loss, log_scale=True)
        ax = fig.get_axes()[0]
        assert ax.get_yscale() == "log"
        plt.close(fig)

    def test_min_annotation_train(self, sample_train_loss):
        fig = plot_loss_curves(train_loss=sample_train_loss)
        ax = fig.get_axes()[0]
        texts = [t.get_text() for t in ax.texts]
        assert any("Min:" in t for t in texts)
        plt.close(fig)

    def test_min_annotation_val(self, sample_train_loss, sample_val_loss):
        fig = plot_loss_curves(train_loss=sample_train_loss, val_loss=sample_val_loss)
        ax = fig.get_axes()[0]
        # Two "Min:" annotations — one per curve
        texts = [t.get_text() for t in ax.texts]
        min_texts = [t for t in texts if "Min:" in t]
        assert len(min_texts) == 2
        plt.close(fig)

    def test_custom_figsize(self, sample_train_loss):
        fig = plot_loss_curves(train_loss=sample_train_loss, figsize=(12, 4))
        assert fig.get_figwidth() == 12
        assert fig.get_figheight() == 4
        plt.close(fig)

    def test_custom_title(self, sample_train_loss):
        fig = plot_loss_curves(train_loss=sample_train_loss, title="Run 1 Loss")
        assert fig.get_axes()[0].get_title() == "Run 1 Loss"
        plt.close(fig)

    def test_save_path(self, tmp_path, sample_train_loss):
        save_path = tmp_path / "loss.png"
        fig = plot_loss_curves(train_loss=sample_train_loss, save_path=save_path)
        assert save_path.exists()
        assert save_path.stat().st_size > 0
        plt.close(fig)

    def test_dpi_passed_through(self, tmp_path, sample_train_loss):
        save_path = tmp_path / "loss_dpi.png"
        fig = plot_loss_curves(
            train_loss=sample_train_loss, save_path=save_path, dpi=72
        )
        assert save_path.exists()
        plt.close(fig)

    def test_accepts_list_input(self):
        fig = plot_loss_curves(train_loss=[1.0, 0.5, 0.3, 0.1])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_val_shorter_than_train(self, sample_train_loss):
        # val has half the length — plot should truncate epochs to match
        val_short = sample_train_loss[:15]
        fig = plot_loss_curves(train_loss=sample_train_loss, val_loss=val_short)
        ax = fig.get_axes()[0]
        val_line = ax.lines[1]
        assert len(val_line.get_xdata()) == 15
        plt.close(fig)


# =============================================================================
# plot_learning_rate
# =============================================================================


class TestPlotLearningRate:
    def test_basic_plot(self, sample_lr_values):
        fig = plot_learning_rate(lr_values=sample_lr_values)
        ax = fig.get_axes()[0]
        assert ax.get_xlabel() == "Step"
        assert ax.get_ylabel() == "Learning Rate"
        assert ax.get_yscale() == "log"
        plt.close(fig)

    def test_custom_steps(self, sample_lr_values):
        custom_steps = np.arange(1000, 1000 + len(sample_lr_values))
        fig = plot_learning_rate(lr_values=sample_lr_values, steps=custom_steps)
        xdata = fig.get_axes()[0].lines[0].get_xdata()
        assert xdata[0] == 1000
        plt.close(fig)

    def test_custom_figsize(self, sample_lr_values):
        fig = plot_learning_rate(lr_values=sample_lr_values, figsize=(8, 3))
        assert fig.get_figwidth() == 8
        plt.close(fig)

    def test_custom_title(self, sample_lr_values):
        fig = plot_learning_rate(lr_values=sample_lr_values, title="LR Schedule v2")
        assert fig.get_axes()[0].get_title() == "LR Schedule v2"
        plt.close(fig)

    def test_save_path(self, tmp_path, sample_lr_values):
        save_path = tmp_path / "lr.png"
        fig = plot_learning_rate(lr_values=sample_lr_values, save_path=save_path)
        assert save_path.exists()
        assert save_path.stat().st_size > 0
        plt.close(fig)

    def test_dpi_passed_through(self, tmp_path, sample_lr_values):
        save_path = tmp_path / "lr_dpi.png"
        fig = plot_learning_rate(lr_values=sample_lr_values, save_path=save_path, dpi=72)
        assert save_path.exists()
        plt.close(fig)

    def test_accepts_list_input(self):
        fig = plot_learning_rate(lr_values=[1e-3, 5e-4, 1e-4])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# =============================================================================
# plot_training_summary
# =============================================================================


class TestPlotTrainingSummary:
    def test_empty_dir_returns_empty_list(self, tmp_path):
        generated = plot_training_summary(log_dir=tmp_path)
        assert generated == []

    def test_with_train_val_loss(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        _write_tfevents(log_dir, {
            "train_loss": list(np.linspace(1.0, 0.1, 30)),
            "val_loss": list(np.linspace(1.2, 0.3, 30)),
        })

        out_dir = tmp_path / "plots"
        generated = plot_training_summary(log_dir=log_dir, output_dir=out_dir)
        # Expect loss_curves.png but no learning_rate.png
        names = {p.name for p in generated}
        assert "loss_curves.png" in names
        assert "learning_rate.png" not in names
        assert (out_dir / "loss_curves.png").exists()

    def test_with_loss_and_lr(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        _write_tfevents(log_dir, {
            "train_loss": list(np.linspace(1.0, 0.1, 20)),
            "val_loss": list(np.linspace(1.2, 0.3, 20)),
            "lr": list(np.linspace(1e-3, 1e-5, 20)),
        })

        out_dir = tmp_path / "plots"
        generated = plot_training_summary(log_dir=log_dir, output_dir=out_dir)
        names = {p.name for p in generated}
        assert "loss_curves.png" in names
        assert "learning_rate.png" in names
        assert (out_dir / "loss_curves.png").exists()
        assert (out_dir / "learning_rate.png").exists()

    def test_default_output_dir_is_under_log_dir(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        _write_tfevents(log_dir, {
            "train_loss": [1.0, 0.5, 0.25, 0.1],
        })

        generated = plot_training_summary(log_dir=log_dir)  # no output_dir
        assert all(str(p).startswith(str(log_dir / "plots")) for p in generated)
        assert (log_dir / "plots" / "loss_curves.png").exists()

    def test_only_lr_no_loss(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        _write_tfevents(log_dir, {
            "learning_rate": list(np.linspace(1e-3, 1e-5, 10)),
        })

        out_dir = tmp_path / "plots"
        generated = plot_training_summary(log_dir=log_dir, output_dir=out_dir)
        names = {p.name for p in generated}
        assert "learning_rate.png" in names
        assert "loss_curves.png" not in names

    def test_fmt_pdf(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        _write_tfevents(log_dir, {
            "train_loss": [1.0, 0.5, 0.1],
        })

        out_dir = tmp_path / "plots"
        generated = plot_training_summary(log_dir=log_dir, output_dir=out_dir, fmt="pdf")
        assert any(p.suffix == ".pdf" for p in generated)
