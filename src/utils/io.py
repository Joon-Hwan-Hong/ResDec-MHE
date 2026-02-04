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
    gene_gate: np.ndarray | None = None,
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
        gene_gate: Gene gate weights [n_cell_types, n_genes]
        hgt_attention: HGT attention [n_subjects, n_layers, n_heads, n_edges]
        pma_attention: PMA attention [n_subjects, k_selected, n_cells]
        pathology_attention: Pathology attention [n_subjects, n_heads, n_cell_types]
        metadata: Additional metadata to store as attributes
        compression: Compression algorithm ("gzip" or None)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = "1.0"

        if gene_gate is not None:
            f.create_dataset("gene_gate", data=gene_gate)

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


# =============================================================================
# DataFrame I/O (consolidated utilities)
# =============================================================================


def save_dataframe(
    df: pd.DataFrame,
    path: str | Path,
    fmt: str = "parquet",
) -> None:
    """
    Save DataFrame in specified format.

    Args:
        df: DataFrame to save
        path: Output path
        fmt: Format - "parquet" or "csv"

    Raises:
        ValueError: If format is not supported
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "parquet":
        df.to_parquet(path, index=False)
    elif fmt == "csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported format: {fmt}. Use 'parquet' or 'csv'.")


def load_dataframe(
    path: str | Path,
    fmt: str | None = None,
) -> pd.DataFrame | None:
    """
    Load DataFrame from parquet or CSV file.

    Supports flexible loading:
    - If path exists with exact name, load it
    - If path has no extension, try .parquet then .csv
    - Returns None if file not found

    Args:
        path: Path to file (with or without extension)
        fmt: Optional format override ("parquet" or "csv")

    Returns:
        DataFrame or None if file not found
    """
    path = Path(path)

    # If format specified, use it directly
    if fmt is not None:
        target = path.with_suffix(f".{fmt}") if not path.suffix else path
        if not target.exists():
            return None
        return pd.read_parquet(target) if fmt == "parquet" else pd.read_csv(target)

    # Try exact path first
    if path.exists():
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        elif path.suffix == ".csv":
            return pd.read_csv(path)
        else:
            # Unknown extension, try to infer
            try:
                return pd.read_parquet(path)
            except Exception:
                return pd.read_csv(path)

    # Try adding extensions
    parquet_path = path.with_suffix(".parquet")
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)

    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)

    return None


def save_dataframes_multi_format(
    df: pd.DataFrame,
    output_dir: str | Path,
    name: str,
    formats: list[str] | None = None,
) -> dict[str, Path]:
    """
    Save DataFrame in multiple formats.

    Args:
        df: DataFrame to save
        output_dir: Output directory
        name: Base filename (without extension)
        formats: List of formats (default: ["parquet", "csv"])

    Returns:
        Dict mapping format to saved path
    """
    if formats is None:
        formats = ["parquet", "csv"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = {}
    for fmt in formats:
        path = output_dir / f"{name}.{fmt}"
        save_dataframe(df, path, fmt)
        saved[fmt] = path

    return saved


# =============================================================================
# HDF5 I/O (consolidated utilities)
# =============================================================================


def save_array_hdf5(
    path: str | Path,
    arrays: dict[str, np.ndarray],
    attrs: dict[str, Any] | None = None,
    compression: str = "gzip",
    compression_opts: int = 4,
) -> None:
    """
    Save numpy arrays to HDF5 file with compression.

    Args:
        path: Output path
        arrays: Dict mapping dataset names to arrays
        attrs: Optional file-level attributes
        compression: Compression algorithm
        compression_opts: Compression level (1-9)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = "1.0"

        for name, arr in arrays.items():
            f.create_dataset(
                name,
                data=arr,
                compression=compression,
                compression_opts=compression_opts,
            )

        if attrs:
            for key, value in attrs.items():
                if isinstance(value, (str, int, float, bool)):
                    f.attrs[key] = value
                elif isinstance(value, (list, tuple)) and all(isinstance(x, str) for x in value):
                    f.attrs[key] = np.array(value, dtype="S64")


def load_array_hdf5(path: str | Path) -> dict[str, np.ndarray | dict]:
    """
    Load arrays and attributes from HDF5 file.

    Args:
        path: Path to HDF5 file

    Returns:
        Dict with arrays and 'attrs' key for file attributes
    """
    result = {}

    with h5py.File(path, "r") as f:
        for key in f.keys():
            result[key] = f[key][:]

        result["attrs"] = {}
        for key in f.attrs.keys():
            val = f.attrs[key]
            # Decode string arrays
            if isinstance(val, np.ndarray) and val.dtype.kind == "S":
                result["attrs"][key] = [x.decode("utf-8") for x in val]
            else:
                result["attrs"][key] = val

    return result


# =============================================================================
# JSON I/O
# =============================================================================


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
