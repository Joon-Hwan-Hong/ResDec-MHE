"""
Input/output utilities for saving and loading various data formats.
"""

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    epoch: int = 0,
    global_step: int = 0,
    best_val_loss: float | None = None,
    config: dict | None = None,
    rng_states: dict | None = None,
    metrics_history: dict | None = None,
    experiment_hash: str | None = None,
) -> None:
    """
    Save a full training checkpoint.

    Args:
        path: Output path for checkpoint
        model: PyTorch model
        optimizer: Optimizer (optional)
        scheduler: LR scheduler (optional)
        epoch: Current epoch
        global_step: Global training step
        best_val_loss: Best validation loss achieved
        config: Model/training configuration
        rng_states: Random number generator states
        metrics_history: Training metrics history
        experiment_hash: Experiment identifier
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "checkpoint_version": "1.0",
        "experiment_hash": experiment_hash,
        "timestamp": pd.Timestamp.now().isoformat(),
        "model_state_dict": model.state_dict(),
        "model_config": config,
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if rng_states is not None:
        checkpoint["rng_states"] = rng_states

    if metrics_history is not None:
        checkpoint["metrics_history"] = metrics_history

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    device: torch.device | str = "cpu",
) -> dict:
    """
    Load a training checkpoint.

    Args:
        path: Path to checkpoint file
        model: Model to load weights into (optional)
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        device: Device to load tensors to

    Returns:
        Checkpoint dictionary with metadata
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if model is not None:
        model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


def save_attention_weights(
    path: str | Path,
    gene_gates: np.ndarray | None = None,
    hgt_attention: np.ndarray | None = None,
    pma_attention: np.ndarray | None = None,
    pathology_attention: np.ndarray | None = None,
    metadata: dict | None = None,
    compression: str = "gzip",
) -> None:
    """
    Save attention weights to HDF5 file.

    Args:
        path: Output path for HDF5 file
        gene_gates: Gene gate weights [31, n_genes]
        hgt_attention: HGT attention [n_subjects, n_layers, n_heads, n_edges]
        pma_attention: PMA attention [n_subjects, k_selected, n_cells]
        pathology_attention: Pathology attention [n_subjects, n_heads, 31]
        metadata: Additional metadata to store as attributes
        compression: Compression algorithm ("gzip" or None)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        if gene_gates is not None:
            f.create_dataset("gene_gates", data=gene_gates)

        if hgt_attention is not None:
            f.create_dataset("hgt_attention", data=hgt_attention, compression=compression)

        if pma_attention is not None:
            f.create_dataset("pma_attention", data=pma_attention, compression=compression)

        if pathology_attention is not None:
            f.create_dataset("pathology_attention", data=pathology_attention)

        if metadata:
            for key, value in metadata.items():
                if isinstance(value, str):
                    f.attrs[key] = value
                elif isinstance(value, (int, float)):
                    f.attrs[key] = value


def load_attention_weights(path: str | Path) -> dict[str, np.ndarray | dict]:
    """
    Load attention weights from HDF5 file.

    Args:
        path: Path to HDF5 file

    Returns:
        Dictionary with attention weight arrays and metadata
    """
    result = {}

    with h5py.File(path, "r") as f:
        for key in f.keys():
            result[key] = f[key][:]

        result["metadata"] = dict(f.attrs)

    return result


def save_predictions(
    path: str | Path,
    subject_ids: list[str],
    means: np.ndarray,
    stds: np.ndarray,
    actuals: np.ndarray | None = None,
    metadata: pd.DataFrame | None = None,
) -> None:
    """
    Save model predictions to CSV.

    Args:
        path: Output path for CSV
        subject_ids: Subject identifiers
        means: Predicted means [n_subjects]
        stds: Predicted standard deviations [n_subjects]
        actuals: Actual values (if available)
        metadata: Additional metadata columns
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "subject_id": subject_ids,
        "predicted_mean": means.flatten(),
        "predicted_std": stds.flatten(),
    })

    if actuals is not None:
        df["actual"] = actuals.flatten()

    if metadata is not None:
        df = pd.concat([df, metadata.reset_index(drop=True)], axis=1)

    df.to_csv(path, index=False)


def save_json(data: dict | list, path: str | Path) -> None:
    """Save data to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: str | Path) -> dict | list:
    """Load data from JSON file."""
    with open(path) as f:
        return json.load(f)
