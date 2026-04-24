"""Tests for src/data/feature_loaders.py — shared loaders used by resdec_mhe scripts."""

from pathlib import Path

import pandas as pd

from src.data.feature_loaders import (
    compute_age_stats_from_training,
    load_flat_features,
    load_targets,
)

PRECOMPUTED = Path("data/precomputed")
META = Path("data/metadata_ROSMAP/metadata.csv")


def test_load_flat_features_returns_dict():
    out = load_flat_features(PRECOMPUTED, ["R1015854"])
    assert "R1015854" in out
    assert out["R1015854"].shape == (148335,)


def test_load_flat_features_handles_missing_subject():
    out = load_flat_features(PRECOMPUTED, ["R1015854", "R_nonexistent_999"])
    assert "R1015854" in out
    assert "R_nonexistent_999" not in out


def test_load_targets_basic():
    out = load_targets(META, ["R1015854"])
    assert "R1015854" in out
    assert isinstance(out["R1015854"], float)


def test_load_targets_drops_null_and_missing():
    out = load_targets(META, ["R1015854", "R_nonexistent_999"])
    assert "R1015854" in out
    assert "R_nonexistent_999" not in out


def test_compute_age_stats_sensible_range():
    # Pick a few real subjects; test mean is in ROSMAP's plausible range
    real_ids = (
        pd.read_csv(META)["ROSMAP_IndividualID"].dropna().head(50).tolist()
    )
    mean, std = compute_age_stats_from_training(META, real_ids)
    assert 70 < mean < 100, f"unexpected age mean: {mean}"
    assert 3 < std < 15, f"unexpected age std: {std}"
