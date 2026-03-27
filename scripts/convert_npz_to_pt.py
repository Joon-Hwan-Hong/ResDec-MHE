#!/usr/bin/env python
"""Convert precomputed .npz feature files to .pt format for mmap loading.

Usage:
    uv run python scripts/convert_npz_to_pt.py data/precomputed/rosmap/
    uv run python scripts/convert_npz_to_pt.py data/precomputed/rosmap/ --force
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch


# Key → dtype mapping for tensor fields
_TENSOR_DTYPES: dict[str, torch.dtype] = {
    "pseudobulk": torch.float32,
    "cell_type_mask": torch.bool,
    "cell_offsets": torch.int64,
    "cell_counts": torch.int64,
    "region_mask": torch.bool,
    "cell_data": torch.float32,
    # CCC keys get renamed (edge_* → ccc_edge_*)
    "edge_index": torch.int64,
    "edge_type": torch.int64,
    "edge_attr": torch.float32,
}

# Keys that are stored as Python lists (not tensors)
_LIST_KEYS = {"available_regions", "cell_type_order"}


def _convert_one(npz_path: Path) -> dict:
    """Convert a single .npz file to a dict suitable for torch.save."""
    data = np.load(npz_path, allow_pickle=True)
    entry: dict = {}

    for key in data.keys():
        arr = data[key]

        # List fields — convert to plain Python lists
        if key in _LIST_KEYS:
            entry[key] = arr.tolist()
            continue

        # Region pseudobulk keys (variable count per subject)
        if key.startswith("region_") and key.endswith("_pseudobulk"):
            entry[key] = torch.tensor(np.array(arr, dtype=np.float32), dtype=torch.float32)
            continue

        # CCC edge keys — rename edge_* → ccc_edge_*
        if key in ("edge_index", "edge_type", "edge_attr"):
            dtype = _TENSOR_DTYPES[key]
            target_key = f"ccc_{key}"
            if dtype in (torch.int64,):
                entry[target_key] = torch.tensor(np.array(arr, dtype=np.int64), dtype=dtype)
            else:
                entry[target_key] = torch.tensor(np.array(arr, dtype=np.float32), dtype=dtype)
            continue

        # Standard tensor fields
        if key in _TENSOR_DTYPES:
            dtype = _TENSOR_DTYPES[key]
            if dtype == torch.bool:
                entry[key] = torch.tensor(np.array(arr, dtype=bool), dtype=torch.bool)
            elif dtype == torch.int64:
                entry[key] = torch.tensor(np.array(arr, dtype=np.int64), dtype=torch.int64)
            else:
                entry[key] = torch.tensor(np.array(arr, dtype=np.float32), dtype=torch.float32)
            continue

        # Fallback: unknown key — preserve as float32 tensor
        print(f"  WARNING: unknown key '{key}' in {npz_path.name}, storing as float32")
        entry[key] = torch.tensor(np.array(arr, dtype=np.float32), dtype=torch.float32)

    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert .npz precomputed files to .pt format")
    parser.add_argument("data_dir", type=Path, help="Directory containing .npz files")
    parser.add_argument("--force", action="store_true", help="Overwrite existing .pt files")
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    if not data_dir.is_dir():
        print(f"ERROR: {data_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    npz_files = sorted(data_dir.glob("*.npz"))
    if not npz_files:
        print(f"No .npz files found in {data_dir}")
        sys.exit(0)

    print(f"Found {len(npz_files)} .npz files in {data_dir}")

    converted = 0
    skipped = 0
    t0 = time.time()

    for i, npz_path in enumerate(npz_files):
        pt_path = npz_path.with_suffix(".pt")

        if pt_path.exists() and not args.force:
            skipped += 1
            continue

        entry = _convert_one(npz_path)
        torch.save(entry, pt_path)
        converted += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{i + 1}/{len(npz_files)}] {elapsed:.1f}s elapsed, {converted} converted, {skipped} skipped")

    elapsed = time.time() - t0
    print(f"Done: {converted} converted, {skipped} skipped in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
