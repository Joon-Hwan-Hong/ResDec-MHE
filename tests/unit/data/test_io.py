"""
Tests for src/utils/io.py — attention unpacking utilities.
"""

import numpy as np
import pandas as pd
import pytest

from src.utils.io import unpack_hgt_for_ccc, save_attention_weights, load_attention_weights


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
