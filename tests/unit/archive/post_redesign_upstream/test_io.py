"""
Tests for src/utils/io.py — attention unpacking utilities.
"""

import h5py
import numpy as np
import pandas as pd
import pytest

from src.utils.io import unpack_hgt_for_ccc, save_attention_weights, load_attention_weights


@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
def test_unpack_hgt_for_ccc_nan_handling():
    """NaN entries (absent edge types) must not poison aggregated scores."""
    n_samples, n_edges, n_layers, n_heads = 5, 3, 2, 4
    per_sample = np.random.rand(n_samples, n_edges, n_layers, n_heads)
    per_sample[0, 1, :, :] = np.nan
    per_sample[1, 1, :, :] = np.nan

    edge_type_names = ["A|interacts|B", "C|interacts|D", "E|interacts|F"]

    hgt_data = {
        "per_sample": per_sample,
        "edge_type_names": np.array(edge_type_names),
    }

    scores, metadata, names = unpack_hgt_for_ccc(hgt_data)
    assert scores is not None
    assert not np.any(np.isnan(scores)), "NaN leaked through unpack_hgt_for_ccc"
    assert scores.shape == (n_samples, n_edges)


def test_hgt_coverage_roundtrip(tmp_path):
    """n_samples_per_edge_type should be saved and loaded from HDF5."""
    n_edges, n_heads = 4, 2
    hgt_attention = {
        "edge_type_names": ["A|x|B", "C|x|D", "E|x|F", "G|x|H"],
        "mean_by_edge_type": np.random.rand(n_edges, n_heads),
        "std_by_edge_type": np.random.rand(n_edges, n_heads),
        "n_samples_per_edge_type": np.array([10, 8, 10, 3]),
    }
    path = tmp_path / "test.h5"
    save_attention_weights(path, hgt_attention=hgt_attention)
    loaded = load_attention_weights(path)
    hgt = loaded["hgt_attention"]
    agg = hgt.get("aggregated", hgt)
    coverage = agg.get("n_samples_per_edge_type")
    assert coverage is not None, "coverage not persisted"
    np.testing.assert_array_equal(coverage, np.array([10, 8, 10, 3]))


def test_per_subject_pseudobulk_hdf5_roundtrip(tmp_path):
    """Per-subject pseudobulk should survive HDF5 save/load."""
    path = tmp_path / "test.h5"
    n_subjects, n_cell_types, n_genes = 10, 3, 50
    per_subject_pb = np.random.rand(n_subjects, n_cell_types, n_genes).astype(np.float32)

    save_attention_weights(
        path=path,
        gene_gate=np.random.rand(n_cell_types, n_genes),
        per_subject_pseudobulk=per_subject_pb,
    )

    loaded = load_attention_weights(path)
    assert "per_subject_pseudobulk" in loaded
    np.testing.assert_array_almost_equal(loaded["per_subject_pseudobulk"], per_subject_pb)
    assert loaded["per_subject_pseudobulk"].shape == (n_subjects, n_cell_types, n_genes)


def test_load_attention_weights_decodes_nested_strings(tmp_path):
    """Nested group string arrays should be decoded like top-level ones."""
    h5_path = tmp_path / "test_nested_strings.h5"
    with h5py.File(h5_path, "w") as f:
        f.attrs["schema_version"] = "2.0"
        grp = f.create_group("test_group")
        grp.create_dataset("names", data=np.array(["foo", "bar"], dtype="S10"))
        subgrp = grp.create_group("sub")
        subgrp.create_dataset("labels", data=np.array(["baz", "qux"], dtype="S10"))

    result = load_attention_weights(h5_path)
    # Nested strings should be decoded to str, not bytes
    names = result["test_group"]["names"]
    labels = result["test_group"]["sub"]["labels"]
    if isinstance(names, list):
        assert all(isinstance(n, str) for n in names)
    else:
        assert names.dtype.kind != "S"  # Not bytes
    if isinstance(labels, list):
        assert all(isinstance(l, str) for l in labels)
    else:
        assert labels.dtype.kind != "S"
