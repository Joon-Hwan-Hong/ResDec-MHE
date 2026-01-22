"""
Tests for src/data/splits.py

Tests cover:
- Stratified splitting correctness
- No data leakage between train/val/test
- K-fold consistency
- Edge cases (small datasets, rare strata)
"""

import numpy as np
import pandas as pd
import pytest


class TestCreateStratificationVariable:
    """Tests for create_stratification_variable()."""

    def test_creates_joint_bins(self):
        """Create joint pathology × cognition bins."""
        from src.data.splits import create_stratification_variable

        metadata = pd.DataFrame({
            "gpath": [0.1, 0.2, 0.5, 0.6, 0.9, 1.0],
            "cogn_global": [1.0, 0.9, 0.5, 0.4, 0.1, 0.0],
        })

        strata = create_stratification_variable(metadata)

        assert len(strata) == 6
        # Should contain combinations like "low_high", "medium_medium", etc.
        unique_strata = strata.unique()
        assert len(unique_strata) >= 2  # At least some stratification


class TestCreateStratifiedSplits:
    """Tests for create_stratified_splits()."""

    @pytest.fixture
    def mock_metadata(self):
        """Create metadata for 100 subjects."""
        np.random.seed(42)
        n = 100

        return pd.DataFrame({
            "ROSMAP_IndividualID": [f"S{i}" for i in range(n)],
            "gpath": np.random.uniform(0, 1, n),
            "cogn_global": np.random.uniform(-2, 2, n),
        })

    def test_correct_split_sizes(self, mock_metadata):
        """Verify split sizes match requested fractions."""
        from src.data.splits import create_stratified_splits

        splits = create_stratified_splits(
            mock_metadata,
            train_frac=0.8,
            val_frac=0.1,
            test_frac=0.1,
            n_folds=5,
        )

        # Test set should be ~10%
        assert len(splits["holdout_test"]) == 10

        # Train+val pool should be ~90%
        assert len(splits["train_val_pool"]) == 90

        # Each fold should have ~72 train, ~18 val
        for fold in splits["folds"]:
            assert len(fold["train"]) + len(fold["val"]) == 90
            assert 70 <= len(fold["train"]) <= 74  # Allow some variance
            assert 16 <= len(fold["val"]) <= 20

    def test_no_overlap_between_test_and_folds(self, mock_metadata):
        """Test subjects should never appear in train/val."""
        from src.data.splits import create_stratified_splits, validate_no_leakage

        splits = create_stratified_splits(mock_metadata)
        test_set = set(splits["holdout_test"])

        for fold in splits["folds"]:
            train_set = set(fold["train"])
            val_set = set(fold["val"])

            assert len(test_set & train_set) == 0, "Test subjects in train!"
            assert len(test_set & val_set) == 0, "Test subjects in val!"

        # Use validation function
        assert validate_no_leakage(splits)

    def test_no_overlap_within_folds(self, mock_metadata):
        """Train and val should not overlap within each fold."""
        from src.data.splits import create_stratified_splits

        splits = create_stratified_splits(mock_metadata)

        for i, fold in enumerate(splits["folds"]):
            train_set = set(fold["train"])
            val_set = set(fold["val"])

            assert len(train_set & val_set) == 0, f"Fold {i} has train/val overlap!"

    def test_all_subjects_accounted_for(self, mock_metadata):
        """All subjects should appear in either test or train_val_pool."""
        from src.data.splits import create_stratified_splits

        splits = create_stratified_splits(mock_metadata)

        all_subjects = set(mock_metadata["ROSMAP_IndividualID"])
        test_set = set(splits["holdout_test"])
        train_val_set = set(splits["train_val_pool"])

        assert test_set | train_val_set == all_subjects
        assert len(test_set & train_val_set) == 0  # No overlap

    def test_reproducibility_with_seed(self, mock_metadata):
        """Same seed should produce identical splits."""
        from src.data.splits import create_stratified_splits

        splits1 = create_stratified_splits(mock_metadata, random_state=42)
        splits2 = create_stratified_splits(mock_metadata, random_state=42)

        assert splits1["holdout_test"] == splits2["holdout_test"]
        for f1, f2 in zip(splits1["folds"], splits2["folds"]):
            assert f1["train"] == f2["train"]
            assert f1["val"] == f2["val"]

    def test_different_seeds_produce_different_splits(self, mock_metadata):
        """Different seeds should produce different splits."""
        from src.data.splits import create_stratified_splits

        splits1 = create_stratified_splits(mock_metadata, random_state=42)
        splits2 = create_stratified_splits(mock_metadata, random_state=123)

        # Very unlikely to be identical with different seeds
        assert splits1["holdout_test"] != splits2["holdout_test"]


class TestSaveLoadSplits:
    """Tests for save_splits() and load_splits()."""

    def test_roundtrip(self, tmp_path):
        """Save and load should produce identical splits."""
        from src.data.splits import save_splits, load_splits

        splits = {
            "holdout_test": ["S1", "S2"],
            "train_val_pool": ["S3", "S4", "S5"],
            "folds": [
                {"train": ["S3", "S4"], "val": ["S5"]},
                {"train": ["S3", "S5"], "val": ["S4"]},
            ],
            "metadata": {"n_subjects": 5, "random_state": 42},
        }

        path = tmp_path / "splits.json"
        save_splits(splits, path)
        loaded = load_splits(path)

        assert loaded["holdout_test"] == splits["holdout_test"]
        assert loaded["train_val_pool"] == splits["train_val_pool"]
        assert loaded["folds"] == splits["folds"]


class TestGetFoldSubjects:
    """Tests for get_fold_subjects()."""

    @pytest.fixture
    def mock_splits(self):
        return {
            "holdout_test": ["T1", "T2"],
            "train_val_pool": ["S1", "S2", "S3", "S4"],
            "folds": [
                {"train": ["S1", "S2", "S3"], "val": ["S4"]},
                {"train": ["S1", "S2", "S4"], "val": ["S3"]},
            ],
        }

    def test_get_train_subjects(self, mock_splits):
        """Get training subjects for specific fold."""
        from src.data.splits import get_fold_subjects

        train = get_fold_subjects(mock_splits, fold_idx=0, split_type="train")
        assert train == ["S1", "S2", "S3"]

    def test_get_val_subjects(self, mock_splits):
        """Get validation subjects for specific fold."""
        from src.data.splits import get_fold_subjects

        val = get_fold_subjects(mock_splits, fold_idx=1, split_type="val")
        assert val == ["S3"]

    def test_get_test_subjects(self, mock_splits):
        """Get test subjects (same regardless of fold)."""
        from src.data.splits import get_fold_subjects

        test = get_fold_subjects(mock_splits, fold_idx=0, split_type="test")
        assert test == ["T1", "T2"]


class TestValidateNoLeakage:
    """Tests for validate_no_leakage()."""

    def test_detects_test_in_train(self, capsys):
        """Detect when test subjects appear in training."""
        from src.data.splits import validate_no_leakage

        bad_splits = {
            "holdout_test": ["S1", "S2"],
            "train_val_pool": ["S1", "S3", "S4"],  # S1 is in both!
            "folds": [
                {"train": ["S1", "S3"], "val": ["S4"]},
            ],
        }

        assert not validate_no_leakage(bad_splits)

    def test_detects_train_val_overlap(self, capsys):
        """Detect train/val overlap within fold."""
        from src.data.splits import validate_no_leakage

        bad_splits = {
            "holdout_test": ["T1"],
            "train_val_pool": ["S1", "S2", "S3"],
            "folds": [
                {"train": ["S1", "S2"], "val": ["S2", "S3"]},  # S2 in both!
            ],
        }

        assert not validate_no_leakage(bad_splits)

    def test_passes_valid_splits(self):
        """Pass for correctly structured splits."""
        from src.data.splits import validate_no_leakage

        good_splits = {
            "holdout_test": ["T1", "T2"],
            "train_val_pool": ["S1", "S2", "S3", "S4"],
            "folds": [
                {"train": ["S1", "S2", "S3"], "val": ["S4"]},
                {"train": ["S1", "S2", "S4"], "val": ["S3"]},
            ],
        }

        assert validate_no_leakage(good_splits)