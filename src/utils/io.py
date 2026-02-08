"""
Input/output utilities for saving and loading various data formats.
"""

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from src.data.constants import sanitize_key


def save_attention_weights(
    path: str | Path,
    gene_gate: np.ndarray | None = None,
    pathology_attention: np.ndarray | None = None,
    cell_type_selection: np.ndarray | None = None,
    region_weights: np.ndarray | None = None,
    region_attention: np.ndarray | None = None,
    region_pseudobulk: np.ndarray | None = None,
    per_subject_pseudobulk: np.ndarray | None = None,
    hgt_attention: dict | None = None,
    pma_attention: list[np.ndarray] | None = None,
    cell_barcodes: list[list[list[str]]] | None = None,
    cell_counts: np.ndarray | None = None,
    subject_ids: list[str] | None = None,
    cell_type_names: list[str] | None = None,
    gene_names: list[str] | None = None,
    region_names: list[str] | None = None,
    embeddings: dict[str, np.ndarray] | None = None,
    metadata: dict | None = None,
    compression: str = "gzip",
    compression_opts: int = 4,
) -> None:
    """
    Save attention weights to HDF5 file (canonical saver).

    Handles flat arrays, nested HGT/PMA groups, and string datasets.

    Args:
        path: Output path for HDF5 file
        gene_gate: Gene gate weights [n_cell_types, n_genes]
        pathology_attention: Pathology attention [n_subjects, n_heads, n_cell_types]
        cell_type_selection: Cell type selection weights [n_cell_types]
        region_weights: Region importance weights [n_regions]
        region_attention: Per-subject normalized region weights [n_subjects, n_regions]
        region_pseudobulk: Mean region pseudobulk [n_regions, n_cell_types, n_genes]
        per_subject_pseudobulk: Per-subject pseudobulk [n_subjects, n_cell_types, n_genes]
        hgt_attention: HGT attention dict from aggregate_hgt_attention()
        pma_attention: PMA attention as list of per-cell-type arrays
                       [n_cell_types][n_subjects, n_heads, n_seeds, max_cells]
        cell_barcodes: Per-subject, per-cell-type barcode lists
                       [n_subjects][n_cell_types][barcodes]
        cell_counts: Cell counts per cell type [n_subjects, n_cell_types]
        embeddings: Dict of embeddings {name: array} (e.g., pseudobulk, hgt, cell, fused, attended)
        subject_ids: Subject identifier strings
        cell_type_names: Cell type name strings
        gene_names: Gene name strings
        metadata: Additional metadata to store as file-level attributes
        compression: Compression algorithm ("gzip" or None)
        compression_opts: Compression level (1-9)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = "2.0"

        # --- Flat arrays ---
        if gene_gate is not None:
            ds = f.create_dataset("gene_gate", data=gene_gate,
                                  compression=compression, compression_opts=compression_opts)
            ds.attrs["shape"] = "[n_cell_types, n_genes]"

        if pathology_attention is not None:
            ds = f.create_dataset("pathology_attention", data=pathology_attention,
                                  compression=compression, compression_opts=compression_opts)
            ds.attrs["shape"] = "[n_subjects, n_heads, n_cell_types]"

        if cell_type_selection is not None:
            ds = f.create_dataset("cell_type_selection", data=cell_type_selection)
            ds.attrs["shape"] = "[n_cell_types]"

        if region_weights is not None:
            ds = f.create_dataset("region_weights", data=region_weights)
            ds.attrs["shape"] = "[n_regions]"

        if region_attention is not None:
            ds = f.create_dataset("region_attention", data=region_attention,
                                  compression=compression, compression_opts=compression_opts)
            ds.attrs["shape"] = "[n_subjects, n_regions]"

        if region_pseudobulk is not None:
            ds = f.create_dataset("region_pseudobulk", data=region_pseudobulk,
                                  compression=compression, compression_opts=compression_opts)
            ds.attrs["shape"] = "[n_regions, n_cell_types, n_genes]"

        if per_subject_pseudobulk is not None:
            ds = f.create_dataset("per_subject_pseudobulk", data=per_subject_pseudobulk,
                                  compression=compression, compression_opts=compression_opts)
            ds.attrs["shape"] = "[n_subjects, n_cell_types, n_genes]"
            ds.attrs["description"] = "Per-subject pseudobulk expression averaged across regions"

        if cell_counts is not None:
            ds = f.create_dataset("cell_counts", data=cell_counts,
                                  compression=compression, compression_opts=compression_opts)
            ds.attrs["shape"] = "[n_subjects, n_cell_types]"

        # --- String datasets (variable-length) ---
        vlen_str = h5py.special_dtype(vlen=str)

        if subject_ids is not None:
            f.create_dataset("subject_ids",
                             data=np.array(subject_ids, dtype=object),
                             dtype=vlen_str)

        if cell_type_names is not None:
            f.create_dataset("cell_type_names",
                             data=np.array(cell_type_names, dtype=object),
                             dtype=vlen_str)

        if gene_names is not None:
            f.create_dataset("gene_names",
                             data=np.array(gene_names, dtype=object),
                             dtype=vlen_str)

        if region_names is not None:
            f.create_dataset("region_names",
                             data=np.array(region_names, dtype=object),
                             dtype=vlen_str)

        # --- Nested HGT attention group ---
        if hgt_attention is not None and isinstance(hgt_attention, dict):
            hgt_group = f.create_group("hgt_attention")
            hgt_group.attrs["description"] = "Per-layer, per-head HGT attention (edge weights)"

            if "n_samples" in hgt_attention:
                hgt_group.attrs["n_samples"] = hgt_attention["n_samples"]
            if "n_layers" in hgt_attention:
                hgt_group.attrs["n_layers"] = hgt_attention["n_layers"]

            edge_type_names = hgt_attention.get("edge_type_names", [])
            if len(edge_type_names) > 0:
                hgt_group.create_dataset(
                    "edge_type_names",
                    data=np.array(edge_type_names, dtype=object),
                    dtype=vlen_str,
                )

                # Aggregated subgroup
                agg_group = hgt_group.create_group("aggregated")
                if "mean_by_edge_type" in hgt_attention:
                    ds = agg_group.create_dataset(
                        "mean_by_edge_type",
                        data=hgt_attention["mean_by_edge_type"],
                        compression=compression, compression_opts=compression_opts,
                    )
                    ds.attrs["shape"] = "[n_edge_types, n_heads]"

                if "std_by_edge_type" in hgt_attention:
                    ds = agg_group.create_dataset(
                        "std_by_edge_type",
                        data=hgt_attention["std_by_edge_type"],
                        compression=compression, compression_opts=compression_opts,
                    )
                    ds.attrs["shape"] = "[n_edge_types, n_heads]"

                if "n_samples_per_edge_type" in hgt_attention:
                    ds = agg_group.create_dataset(
                        "n_samples_per_edge_type",
                        data=hgt_attention["n_samples_per_edge_type"],
                    )
                    ds.attrs["shape"] = "[n_edge_types]"
                    ds.attrs["description"] = "Number of non-NaN samples per edge type (coverage)"

                # Per-sample subgroup
                per_sample = hgt_attention.get("per_sample")
                if per_sample is not None and per_sample.size > 0:
                    ps_group = hgt_group.create_group("per_sample")
                    ds = ps_group.create_dataset(
                        "attention",
                        data=per_sample,
                        compression=compression, compression_opts=compression_opts,
                    )
                    ds.attrs["shape"] = "[n_samples, n_edge_types, n_layers, n_heads]"
                    ps_group.attrs["description"] = (
                        "Per-sample HGT attention summaries. Mean attention across edges "
                        "within each sample, preserving layer structure."
                    )

        # --- Nested PMA attention group ---
        if pma_attention is not None and len(pma_attention) > 0:
            from src.data.constants import CELL_TYPE_ORDER

            pma_group = f.create_group("pma_attention")
            pma_group.attrs["description"] = "Cell-level attention from Set Transformer PMA"

            first_attn = pma_attention[0]
            n_subjects_pma, n_heads_pma, n_seeds, max_cells = first_attn.shape
            pma_group.attrs["n_subjects"] = n_subjects_pma
            pma_group.attrs["n_heads"] = n_heads_pma
            pma_group.attrs["n_seeds"] = n_seeds
            pma_group.attrs["max_cells"] = max_cells

            ct_order = list(cell_type_names or CELL_TYPE_ORDER)
            per_ct_group = pma_group.create_group("per_cell_type")
            for ct_idx, ct_name in enumerate(ct_order):
                if ct_idx < len(pma_attention):
                    safe_name = sanitize_key(ct_name)
                    per_ct_group.create_dataset(
                        safe_name,
                        data=pma_attention[ct_idx],
                        compression=compression, compression_opts=compression_opts,
                    )

            # Aggregated mean across subjects, heads, seeds
            agg_group = pma_group.create_group("aggregated")
            mean_by_cell_type = np.zeros((len(pma_attention), max_cells))
            for ct_idx in range(len(pma_attention)):
                mean_by_cell_type[ct_idx] = pma_attention[ct_idx].mean(axis=(0, 1, 2))
            ds = agg_group.create_dataset(
                "mean_by_cell_type",
                data=mean_by_cell_type,
                compression=compression, compression_opts=compression_opts,
            )
            ds.attrs["shape"] = "[n_cell_types, max_cells]"

        # --- Cell barcodes group ---
        if cell_barcodes is not None:
            vlen_str_bc = h5py.special_dtype(vlen=str)
            bc_group = f.create_group("cell_barcodes")
            bc_group.attrs["description"] = "Per-subject, per-cell-type cell barcodes"
            bc_group.attrs["n_subjects"] = len(cell_barcodes)
            for subj_idx, subj_barcodes in enumerate(cell_barcodes):
                if subj_barcodes is not None:
                    for ct_idx, ct_barcodes in enumerate(subj_barcodes):
                        if ct_barcodes:
                            key = f"{subj_idx}_{ct_idx}"
                            bc_group.create_dataset(
                                key,
                                data=np.array(ct_barcodes, dtype=object),
                                dtype=vlen_str_bc,
                            )

        # --- Embeddings group ---
        if embeddings is not None:
            emb_group = f.create_group("embeddings")
            for emb_name, emb_array in embeddings.items():
                ds = emb_group.create_dataset(
                    emb_name, data=emb_array,
                    compression=compression, compression_opts=compression_opts,
                )
                if emb_array.ndim == 3:
                    ds.attrs["shape"] = "[n_subjects, n_cell_types, d_embed]"
                elif emb_array.ndim == 2:
                    ds.attrs["shape"] = "[n_subjects, d_embed]"

        # --- File-level metadata attributes ---
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, str):
                    f.attrs[key] = value
                elif isinstance(value, (int, float, bool)):
                    f.attrs[key] = value


def load_attention_weights(path: str | Path) -> dict[str, np.ndarray | dict]:
    """
    Load attention weights from HDF5 file (canonical loader).

    Handles flat arrays, nested HGT/PMA groups, and string datasets.
    String datasets (subject_ids, cell_type_names, gene_names) are decoded
    and stored both at the top level and in result["metadata"] for backward
    compatibility.

    Args:
        path: Path to HDF5 file

    Returns:
        Dictionary with attention weight arrays and metadata.
        Returns empty dict if file does not exist.
    """
    path = Path(path)
    if not path.exists():
        return {}

    result = {}

    def _decode_string_array(arr: np.ndarray) -> list[str]:
        """Decode HDF5 string array to Python list of str."""
        decoded = []
        for item in arr:
            if isinstance(item, bytes):
                decoded.append(item.decode("utf-8"))
            else:
                decoded.append(str(item))
        return decoded

    with h5py.File(path, "r") as f:
        for key in f.keys():
            if isinstance(f[key], h5py.Group):
                # Handle nested groups (e.g., hgt_attention, pma_attention)
                group_data = {"attrs": dict(f[key].attrs)}
                for subkey in f[key].keys():
                    if isinstance(f[key][subkey], h5py.Group):
                        # Nested subgroup (e.g., aggregated, per_cell_type)
                        subgroup_data = {}
                        for subsubkey in f[key][subkey].keys():
                            raw = f[key][subkey][subsubkey][:]
                            if raw.dtype.kind in ("O", "S", "U"):
                                raw = _decode_string_array(raw)
                            subgroup_data[subsubkey] = raw
                        group_data[subkey] = subgroup_data
                    else:
                        raw = f[key][subkey][:]
                        if raw.dtype.kind in ("O", "S", "U"):
                            raw = _decode_string_array(raw)
                        group_data[subkey] = raw
                result[key] = group_data
            else:
                raw = f[key][:]
                # Detect string datasets and decode them
                if raw.dtype.kind in ("O", "S", "U"):
                    result[key] = _decode_string_array(raw)
                else:
                    result[key] = raw

        # File-level attributes → metadata
        result["metadata"] = dict(f.attrs)

    # --- Backward compatibility bridging ---

    # String datasets stored as top-level datasets should also appear in metadata
    for str_key in ("subject_ids", "cell_type_names", "gene_names", "region_names"):
        if str_key in result and isinstance(result[str_key], list):
            result["metadata"][str_key] = result[str_key]

    # Alias gene_gate_weights → gene_gate for backward compat
    if "gene_gate_weights" in result and "gene_gate" not in result:
        result["gene_gate"] = result["gene_gate_weights"]

    # Clean up embeddings group: strip 'attrs' key, keep only arrays
    if "embeddings" in result and isinstance(result["embeddings"], dict):
        result["embeddings"] = {
            k: v for k, v in result["embeddings"].items()
            if k != "attrs" and isinstance(v, np.ndarray)
        }

    return result


# =============================================================================
# Attention Unpacking Utilities
# =============================================================================


def unpack_hgt_for_ccc(
    hgt_data: dict,
) -> tuple[np.ndarray | None, pd.DataFrame | None, list[str] | None]:
    """
    Unpack HGT attention dict from load_attention_weights() into CCC-ready format.

    Args:
        hgt_data: Nested dict with keys like 'aggregated', 'per_sample',
                  'edge_type_names', 'attrs'

    Returns:
        (edge_attention_scores, edge_metadata, edge_type_names)
        - edge_attention_scores: [n_samples, n_edge_types] if per_sample available
          (mean across layers and heads), else [n_edge_types] from aggregated
        - edge_metadata: DataFrame with source, target, edge_type columns
        - edge_type_names: list of "source|edge_type|target" strings (PyG convention)
    """
    # Extract edge type names
    edge_type_names = None
    raw_names = hgt_data.get("edge_type_names")
    if raw_names is not None:
        if isinstance(raw_names, np.ndarray):
            edge_type_names = [
                n.decode("utf-8") if isinstance(n, bytes) else str(n)
                for n in raw_names
            ]
        elif isinstance(raw_names, list):
            edge_type_names = raw_names
    elif "attrs" in hgt_data and "edge_type_names" in hgt_data["attrs"]:
        raw = hgt_data["attrs"]["edge_type_names"]
        if isinstance(raw, np.ndarray):
            edge_type_names = [
                n.decode("utf-8") if isinstance(n, bytes) else str(n)
                for n in raw
            ]
        else:
            edge_type_names = list(raw)

    if not edge_type_names:
        return None, None, None

    # Parse edge type names into metadata DataFrame
    rows = []
    for name in edge_type_names:
        parts = name.split("|")
        if len(parts) == 3:
            # PyG convention: (src_type, edge_type, dst_type)
            rows.append({"source": parts[0], "target": parts[2], "edge_type": parts[1]})
        else:
            rows.append({"source": name, "target": name, "edge_type": name})
    edge_metadata = pd.DataFrame(rows)

    # Extract attention scores
    edge_attention_scores = None
    per_sample = hgt_data.get("per_sample")
    if isinstance(per_sample, dict) and "attention" in per_sample:
        # [n_samples, n_edge_types, n_layers, n_heads] → mean over layers and heads
        raw_attention = per_sample["attention"]
        edge_attention_scores = np.nanmean(raw_attention, axis=(2, 3))  # [n_samples, n_edge_types]
    elif isinstance(per_sample, np.ndarray) and per_sample.ndim == 4:
        edge_attention_scores = np.nanmean(per_sample, axis=(2, 3))

    if edge_attention_scores is None:
        # Fall back to aggregated mean
        aggregated = hgt_data.get("aggregated", {})
        mean_by_edge = aggregated.get("mean_by_edge_type")
        if mean_by_edge is not None:
            # [n_edge_types, n_heads] → mean over heads
            edge_attention_scores = np.nanmean(mean_by_edge, axis=1)  # [n_edge_types]

    # Replace any remaining NaN (from all-NaN slices) with 0
    if edge_attention_scores is not None:
        edge_attention_scores = np.nan_to_num(edge_attention_scores, nan=0.0)

    return edge_attention_scores, edge_metadata, edge_type_names


def unpack_pma_attention(
    pma_data: dict,
    cell_type_names: list[str] | None = None,
) -> np.ndarray | None:
    """
    Reconstruct 3D PMA attention array from nested dict.

    Args:
        pma_data: Nested dict from load_attention_weights() with 'per_cell_type' subgroup
        cell_type_names: Cell type names to order by (uses CELL_TYPE_ORDER if None)

    Returns:
        np.ndarray [n_subjects, n_cell_types, max_cells] — aggregated across heads/seeds.
        For cell-level analysis, use run_cell_heterogeneity.py which handles this structure.
        Returns None if per_cell_type data is not found.
    """
    from src.data.constants import CELL_TYPE_ORDER

    per_ct = pma_data.get("per_cell_type", {})
    if not per_ct:
        return None

    ct_order = list(cell_type_names or CELL_TYPE_ORDER)

    # Build mapping from sanitized name to array
    sanitized_to_array = {}
    for key, arr in per_ct.items():
        sanitized_to_array[key] = arr

    # Collect arrays in cell type order
    arrays = []
    for ct_name in ct_order:
        safe_name = sanitize_key(ct_name)
        if safe_name in sanitized_to_array:
            arr = sanitized_to_array[safe_name]
            # arr shape: [n_subjects, n_heads, n_seeds, max_cells]
            # Mean over heads (axis=1) and seeds (axis=2) → [n_subjects, max_cells]
            if arr.ndim == 4:
                arrays.append(arr.mean(axis=(1, 2)))
            elif arr.ndim == 2:
                arrays.append(arr)
            else:
                arrays.append(arr.reshape(arr.shape[0], -1))

    if not arrays:
        return None

    # Stack across cell types → [n_subjects, n_cell_types, max_cells]
    return np.stack(arrays, axis=1)


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
