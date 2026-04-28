"""Tests for src/data/tabpfn_input.py — flat-pseudobulk + FiLM metadata loaders."""

import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

from src.data.tabpfn_input import flatten_pseudobulk, load_metadata_vector

REAL_PT = Path("data/precomputed/R1015854.pt")
META_CSV = Path("data/metadata_ROSMAP/metadata.csv")


def test_flatten_pseudobulk_shape():
    """Flat vector is 148_335 float32 entries (31 cell-types x 4785 genes)."""
    pt = torch.load(REAL_PT, weights_only=False)
    flat = flatten_pseudobulk(pt)
    assert flat.shape == (148_335,)
    assert flat.dtype == torch.float32


def test_flatten_pseudobulk_matches_baseline():
    """flatten_pseudobulk output matches existing extract_features_a byte-for-byte."""
    sys.path.insert(0, "scripts/analysis")
    from run_baselines import extract_features_a

    pt = torch.load(REAL_PT, weights_only=False)
    flat_new = flatten_pseudobulk(pt).numpy()
    flat_old = extract_features_a(pt)
    assert flat_new.shape == flat_old.shape
    assert (flat_new == flat_old).all()


def test_load_metadata_vector_shape():
    """Metadata vector is 8-dim with the documented field names."""
    vec, field_names = load_metadata_vector("R1015854", META_CSV)
    assert vec.ndim == 1
    # 4 APOE (e2, e3, e4, missing) + 2 sex (val, missing) + 2 age (z, missing) = 8
    assert vec.shape[0] == 8
    assert len(field_names) == 8
    assert field_names == [
        "apoe_e2", "apoe_e3", "apoe_e4", "apoe_missing",
        "sex", "sex_missing",
        "age", "age_missing",
    ]


def test_load_metadata_handles_missing():
    """Unknown subject IDs yield all missingness bits set, no values written."""
    vec, _ = load_metadata_vector("R99999999", META_CSV)
    assert vec[3].item() == 1.0  # apoe_missing
    assert vec[5].item() == 1.0  # sex_missing
    assert vec[7].item() == 1.0  # age_missing
    # Actual value slots remain zero
    assert vec[0].item() == 0.0 and vec[1].item() == 0.0 and vec[2].item() == 0.0
    assert vec[4].item() == 0.0
    assert vec[6].item() == 0.0


def test_load_metadata_apoe_decode():
    """APOE 34 decodes to e3 bit and e4 bit set; e2 not set; not missing."""
    df = pd.read_csv(META_CSV)
    row_34 = df[df["apoe_genotype"] == 34.0].iloc[0]
    subj_id = row_34["ROSMAP_IndividualID"]
    vec, _ = load_metadata_vector(subj_id, META_CSV)
    assert vec[0].item() == 0.0  # no e2
    assert vec[1].item() == 1.0  # e3 present
    assert vec[2].item() == 1.0  # e4 present
    assert vec[3].item() == 0.0  # not missing
