"""Tests for src/data/enriched_features.py — enriched flat-feature builder."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.enriched_features import (
    FEATURE_SET_SIZES,
    FEATURE_SETS,
    PATHOLOGY_COLUMNS,
    build_features,
    extract_ccc_aggregate,
    extract_ccc_dense,
    extract_composition,
    extract_region_mask,
    load_enriched_features,
    load_pathology,
)

REAL_PT = Path("data/precomputed/R1015854.pt")
META_CSV = Path("data/metadata_ROSMAP/metadata.csv")
PRECOMPUTED = Path("data/precomputed")


def _pt():
    return torch.load(REAL_PT, weights_only=False)


def test_feature_set_sizes_match_documented():
    assert FEATURE_SET_SIZES["A"] == 148_335
    assert FEATURE_SET_SIZES["A+C"] == 148_366
    assert FEATURE_SET_SIZES["A+C+E"] == 153_189
    assert FEATURE_SET_SIZES["A+C+E+P+R"] == 153_198


def test_build_features_shapes_and_dtypes():
    pt = _pt()
    pathology = np.zeros(len(PATHOLOGY_COLUMNS), dtype=np.float32)
    for fs in FEATURE_SETS:
        v = build_features(
            pt, fs,
            pathology_vec=pathology if fs == "A+C+E+P+R" else None,
        )
        assert v.shape == (FEATURE_SET_SIZES[fs],), f"wrong shape for {fs}"
        assert v.dtype == np.float32


def test_extract_composition_sums_to_one():
    comp = extract_composition(_pt())
    assert comp.shape == (31,)
    assert comp.dtype == np.float32
    assert abs(comp.sum() - 1.0) < 1e-5


def test_extract_ccc_dense_scatter_sum_matches_edge_attr_sum():
    pt = _pt()
    dense = extract_ccc_dense(pt)
    assert dense.shape == (31 * 31 * 5,)
    # Sum of scatter_add'd values = sum of edge_attr scalars.
    edge_sum = float(pt["ccc_edge_attr"][:, 0].sum().item())
    assert abs(dense.sum() - edge_sum) < 1e-3


def test_extract_ccc_aggregate_matches_baseline():
    """Must match scripts/analysis/run_baselines.extract_features_e byte-for-byte."""
    sys.path.insert(0, "scripts/analysis")
    from run_baselines import extract_features_e
    pt = _pt()
    expected = extract_features_e(pt)
    actual = extract_ccc_aggregate(pt)
    assert actual.shape == expected.shape == (18,)
    assert np.allclose(actual, expected, atol=1e-6)


def test_extract_region_mask_shape_and_values():
    rm = extract_region_mask(_pt())
    assert rm.shape == (6,)
    assert rm.dtype == np.float32
    assert set(rm.tolist()).issubset({0.0, 1.0})


def test_build_features_rejects_unknown_feature_set():
    with pytest.raises(ValueError, match="Unknown feature_set"):
        build_features(_pt(), "bogus")


def test_build_features_requires_pathology_for_pr_set():
    with pytest.raises(ValueError, match="pathology_vec is required"):
        build_features(_pt(), "A+C+E+P+R", pathology_vec=None)


def test_build_features_rejects_wrong_pathology_shape():
    bad = np.zeros(5, dtype=np.float32)
    with pytest.raises(ValueError, match="pathology_vec must be shape"):
        build_features(_pt(), "A+C+E+P+R", pathology_vec=bad)


def test_load_pathology_returns_expected_shape():
    out = load_pathology(META_CSV, ["R1015854"])
    assert "R1015854" in out
    assert out["R1015854"].shape == (len(PATHOLOGY_COLUMNS),)
    assert out["R1015854"].dtype == np.float32


def test_load_enriched_features_matches_flat_loader_for_a():
    """For feature_set='A', load_enriched_features should produce the same
    vectors as load_flat_features."""
    from src.data.feature_loaders import load_flat_features

    flat = load_flat_features(PRECOMPUTED, ["R1015854"])
    enriched = load_enriched_features(PRECOMPUTED, ["R1015854"], "A")
    assert np.array_equal(flat["R1015854"], enriched["R1015854"])


def test_load_enriched_features_drops_missing_pathology():
    """Subjects without pathology vectors should be skipped when feature_set
    requires '+P'."""
    out = load_enriched_features(
        PRECOMPUTED, ["R1015854"], "A+C+E+P+R", pathology={},
    )
    assert "R1015854" not in out
