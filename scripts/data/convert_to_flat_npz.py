"""Convert existing padded .npz files to flat cell format.

Reads padded cells [n_types, max_cells, n_genes] + cell_mask [n_types, max_cells]
and writes cell_data [total_real_cells, n_genes] + cell_offsets [n_types + 1].

Usage:
    uv run python scripts/convert_to_flat_npz.py data/precomputed/rosmap/
    uv run python scripts/convert_to_flat_npz.py data/precomputed/rosmap/ --output-dir data/precomputed/rosmap_flat/
"""

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np


def convert_npz(src: Path, dst: Path) -> str:
    """Convert a single npz file from padded to flat cell format.

    Returns status string: "converted", "skipped" (already flat), or "error".
    """
    with np.load(src, allow_pickle=True) as data:
        arrays = dict(data)

    if "cell_data" in arrays:
        return "skipped"

    if "cells" not in arrays or "cell_mask" not in arrays:
        return "error"

    cells = arrays.pop("cells")         # [n_types, max_cells, n_genes]
    cell_mask = arrays.pop("cell_mask")  # [n_types, max_cells]
    n_types = cells.shape[0]

    cell_offsets = np.zeros(n_types + 1, dtype=np.int64)
    flat_parts = []
    for ct in range(n_types):
        n = int(cell_mask[ct].sum())
        if n > 0:
            flat_parts.append(cells[ct, :n])
        cell_offsets[ct + 1] = cell_offsets[ct] + n

    arrays["cell_data"] = (
        np.concatenate(flat_parts, axis=0)
        if flat_parts
        else np.empty((0, cells.shape[2]), dtype=np.float32)
    )
    arrays["cell_offsets"] = cell_offsets

    # Atomic write: temp file + rename
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=dst.parent, suffix=".npz", delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        np.savez_compressed(tmp_path, **arrays)
        os.replace(tmp_path, dst)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    return "converted"


def main():
    parser = argparse.ArgumentParser(
        description="Convert padded .npz files to flat cell format."
    )
    parser.add_argument("input_dir", type=Path, help="Directory with .npz files")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: convert in-place)",
    )
    args = parser.parse_args()

    out_dir = args.output_dir or args.input_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(args.input_dir.glob("*.npz"))
    # Skip non-subject files (e.g., gene_names.npy)
    files = [f for f in files if f.stem != "gene_names"]

    n_converted = 0
    n_skipped = 0
    n_errors = 0

    for i, f in enumerate(files):
        status = convert_npz(f, out_dir / f.name)
        if status == "converted":
            n_converted += 1
        elif status == "skipped":
            n_skipped += 1
        else:
            n_errors += 1
        if (i + 1) % 50 == 0 or i == len(files) - 1:
            print(
                f"[{i + 1}/{len(files)}] "
                f"converted={n_converted} skipped={n_skipped} errors={n_errors}"
            )

    print(
        f"\nDone. Converted: {n_converted}, Skipped: {n_skipped}, Errors: {n_errors}"
    )


if __name__ == "__main__":
    main()
