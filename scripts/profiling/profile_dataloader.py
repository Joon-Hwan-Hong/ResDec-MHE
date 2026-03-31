#!/usr/bin/env python3
"""Profile DataLoader performance: __getitem__ and collation times.

Measures per-sample __getitem__ latency, collation time, and total
DataLoader iteration time with the real ROSMAP dataset.
"""

import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, ".")

from pathlib import Path

from omegaconf import OmegaConf

from src.data.datasets import PrecomputedDataset
from src.data.collate import create_dataloader
from src.data.splits import load_splits


def profile_getitem(dataset, n_samples=100):
    """Profile individual __getitem__ calls."""
    indices = np.random.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)
    times = []
    sizes_mb = []
    for idx in indices:
        t0 = time.perf_counter()
        sample = dataset[int(idx)]
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms
        if "cell_data" in sample:
            cd = sample["cell_data"]
            sizes_mb.append(cd.nelement() * cd.element_size() / (1024**2))

    times = np.array(times)
    print(f"\n__getitem__ ({len(times)} samples):")
    print(f"  mean:   {times.mean():.3f} ms")
    print(f"  std:    {times.std():.3f} ms")
    print(f"  min:    {times.min():.3f} ms")
    print(f"  max:    {times.max():.3f} ms")
    print(f"  median: {np.median(times):.3f} ms")
    if sizes_mb:
        sizes_mb = np.array(sizes_mb)
        print(f"  cell_data per subject: mean={sizes_mb.mean():.1f} MB, max={sizes_mb.max():.1f} MB")
    return times


def profile_dataloader(dataloader, n_batches=30, label=""):
    """Profile DataLoader batch iteration."""
    batch_times = []
    batch_sizes = []
    iterator = iter(dataloader)
    for i in range(n_batches):
        try:
            t0 = time.perf_counter()
            batch = next(iterator)
            t1 = time.perf_counter()
            batch_times.append((t1 - t0) * 1000)  # ms
            if "pseudobulk" in batch:
                batch_sizes.append(batch["pseudobulk"].shape[0])
        except StopIteration:
            break

    batch_times = np.array(batch_times)
    # Skip first batch (includes worker fork overhead)
    if len(batch_times) > 1:
        steady = batch_times[1:]
    else:
        steady = batch_times

    print(f"\nDataLoader{' (' + label + ')' if label else ''} ({len(batch_times)} batches, bs={batch_sizes[0] if batch_sizes else '?'}):")
    print(f"  First batch:  {batch_times[0]:.1f} ms (includes worker startup)")
    print(f"  Steady-state: mean={steady.mean():.1f} ms, std={steady.std():.1f} ms")
    print(f"  min={steady.min():.1f} ms, max={steady.max():.1f} ms")
    return batch_times


def main():
    cfg = OmegaConf.load("configs/default.yaml")
    data_cfg = cfg.data

    print("=" * 70)
    print("DataLoader Profiling — Full cell_data RAM caching")
    print("=" * 70)

    # Load metadata
    metadata_path = Path(data_cfg.metadata_path)
    meta_files = sorted(metadata_path.glob("*.csv"))
    metadata = pd.read_csv(meta_files[0])
    for f in meta_files[1:]:
        df = pd.read_csv(f)
        metadata = metadata.merge(df, on=data_cfg.subject_column, how="outer", suffixes=("", "_dup"))
        metadata = metadata[[c for c in metadata.columns if not c.endswith("_dup")]]

    # Load splits
    splits_path = Path("outputs/splits.json")
    splits = load_splits(splits_path)

    # Get subject IDs for fold 0
    from src.data.splits import get_fold_subjects
    train_ids = get_fold_subjects(splits, fold_idx=0, split_type="train")

    precomputed_dir = Path(data_cfg.precomputed_dir) / "rosmap"

    # Create dataset with preload_to_ram=True
    print("\nCreating dataset with preload_to_ram=True...")
    t0 = time.perf_counter()
    ds = PrecomputedDataset(
        feature_dir=precomputed_dir,
        subject_ids=train_ids,
        metadata=metadata,
        subject_column=data_cfg.subject_column,
        target_column=data_cfg.target_column,
        pathology_columns=list(data_cfg.pathology_columns),
        preload_to_ram=True,
    )
    t_init = time.perf_counter() - t0
    print(f"Dataset init: {t_init:.1f} s ({len(ds)} subjects)")

    # Check RAM usage
    if hasattr(ds, '_small_cache') and ds._small_cache is not None:
        total_bytes = 0
        cell_bytes = 0
        for sid, entry in ds._small_cache.items():
            for key, val in entry.items():
                if isinstance(val, torch.Tensor):
                    total_bytes += val.nelement() * val.element_size()
                    if key == "cell_data":
                        cell_bytes += val.nelement() * val.element_size()
        print(f"Cache total: {total_bytes / (1024**3):.1f} GB")
        print(f"  cell_data: {cell_bytes / (1024**3):.1f} GB")
        print(f"  other:     {(total_bytes - cell_bytes) / (1024**3):.2f} GB")

    # Profile __getitem__
    profile_getitem(ds, n_samples=100)

    # Create DataLoader
    dl_cfg = data_cfg.dataloader
    from src.data.collate import collate_for_hgt_multiregion as collate_subjects

    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=dl_cfg.batch_size,
        shuffle=True,
        num_workers=dl_cfg.num_workers,
        collate_fn=collate_subjects,
        pin_memory=dl_cfg.pin_memory,
        prefetch_factor=dl_cfg.get("prefetch_factor", 2),
        persistent_workers=dl_cfg.num_workers > 0,
    )

    # Profile DataLoader
    profile_dataloader(dl, n_batches=30, label="warm")

    # Full epoch timing
    print("\nFull epoch timing...")
    dl2 = torch.utils.data.DataLoader(
        ds,
        batch_size=dl_cfg.batch_size,
        shuffle=True,
        num_workers=dl_cfg.num_workers,
        collate_fn=collate_subjects,
        pin_memory=dl_cfg.pin_memory,
        prefetch_factor=dl_cfg.get("prefetch_factor", 2),
        persistent_workers=dl_cfg.num_workers > 0,
    )
    t0 = time.perf_counter()
    n_batches = 0
    for batch in dl2:
        n_batches += 1
    t1 = time.perf_counter()
    epoch_time = t1 - t0
    print(f"  {n_batches} batches in {epoch_time:.1f} s ({epoch_time/n_batches*1000:.1f} ms/batch)")

    # Second epoch (persistent workers already warm)
    print("\nSecond epoch (persistent workers warm)...")
    t0 = time.perf_counter()
    n_batches = 0
    for batch in dl2:
        n_batches += 1
    t1 = time.perf_counter()
    epoch_time = t1 - t0
    print(f"  {n_batches} batches in {epoch_time:.1f} s ({epoch_time/n_batches*1000:.1f} ms/batch)")


if __name__ == "__main__":
    main()
