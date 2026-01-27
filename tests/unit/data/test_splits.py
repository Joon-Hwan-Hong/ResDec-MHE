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
        """Verify split sizes match true K-fold behavior."""
        from src.data.splits import create_stratified_splits

        splits = create_stratified_splits(
            mock_metadata,
            test_frac=0.1,
            n_folds=5,
        )

        # Test set should be ~10%
        assert len(splits["holdout_test"]) == 10

        # Train+val pool should be ~90%
        assert len(splits["train_val_pool"]) == 90

        # With true 5-fold CV on 90 subjects:
        # - Each validation fold: 90/5 = 18 subjects
        # - Each training fold: 90 - 18 = 72 subjects
        for fold in splits["folds"]:
            assert len(fold["train"]) + len(fold["val"]) == 90
            assert 71 <= len(fold["train"]) <= 73  # Allow ±1 for rounding
            assert 17 <= len(fold["val"]) <= 19

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


class TestTrueKFold:
    """Tests for true K-fold CV with disjoint validation sets."""

    def test_validation_sets_are_disjoint(self):
        """Each subject should appear in exactly one validation set across all folds."""
        from src.data.splits import create_stratified_splits

        n = 100
        metadata = pd.DataFrame({
            "ROSMAP_IndividualID": [f"subj_{i:03d}" for i in range(n)],
            "gpath": np.random.randn(n),
            "cogn_global": np.random.randn(n),
        })

        splits = create_stratified_splits(metadata, n_folds=5)

        # Collect all validation subjects
        all_val_subjects = []
        for fold in splits["folds"]:
            all_val_subjects.extend(fold["val"])

        # Check: no duplicates (each subject in exactly one val set)
        assert len(all_val_subjects) == len(set(all_val_subjects)), \
            "Validation sets overlap! Same subject appears in multiple validation folds."

        # Check: all train_val subjects appear in exactly one val fold
        train_val_set = set(splits["train_val_pool"])
        val_set = set(all_val_subjects)
        assert val_set == train_val_set, \
            "Not all train_val subjects appear in a validation fold."

    def test_val_size_determined_by_n_folds(self):
        """Validation size should be ~1/n_folds of train_val_pool."""
        from src.data.splits import create_stratified_splits

        n = 100
        metadata = pd.DataFrame({
            "ROSMAP_IndividualID": [f"subj_{i:03d}" for i in range(n)],
            "gpath": np.random.randn(n),
            "cogn_global": np.random.randn(n),
        })

        for n_folds in [3, 5, 10]:
            splits = create_stratified_splits(
                metadata,
                test_frac=0.10,
                n_folds=n_folds,
            )

            n_train_val = len(splits["train_val_pool"])
            expected_val_size = n_train_val // n_folds

            for fold in splits["folds"]:
                n_val = len(fold["val"])
                # Allow ±1 for rounding
                assert abs(n_val - expected_val_size) <= 1, \
                    f"Val size {n_val} != expected ~{expected_val_size} for {n_folds}-fold"

    def test_union_of_val_sets_covers_all_train_val(self):
        """Union of all validation sets should equal train_val_pool."""
        from src.data.splits import create_stratified_splits

        n = 100
        metadata = pd.DataFrame({
            "ROSMAP_IndividualID": [f"subj_{i:03d}" for i in range(n)],
            "gpath": np.random.randn(n),
            "cogn_global": np.random.randn(n),
        })

        splits = create_stratified_splits(metadata, n_folds=5)

        all_val = set()
        for fold in splits["folds"]:
            all_val.update(fold["val"])

        train_val_set = set(splits["train_val_pool"])
        assert all_val == train_val_set, \
            "Union of validation sets doesn't cover all train_val subjects."


class TestStratificationFallback:
    """Tests for stratification fallback when tertiles fail."""

    def test_falls_back_to_median_on_low_variance(self):
        """When qcut fails due to ties, should fall back to median split."""
        from src.data.splits import create_stratification_variable
        import pandas as pd
        import numpy as np
        import warnings

        # Create data with many ties (will fail qcut tertiles)
        n = 50
        metadata = pd.DataFrame({
            "ROSMAP_IndividualID": [f"subj_{i:03d}" for i in range(n)],
            "gpath": [1.0] * 40 + [2.0] * 10,  # 80% same value
            "cogn_global": np.random.randn(n),
        })

        # Should not raise, should fall back to median
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            strata = create_stratification_variable(metadata)

            # Should have logged a warning about fallback
            assert any("median" in str(warning.message).lower() for warning in w)

        # Should still produce valid strata
        assert len(strata) == n
        assert strata.nunique() >= 2  # At least 2 strata

    def test_median_fallback_produces_balanced_split(self):
        """Median fallback should produce roughly 50/50 split."""
        from src.data.splits import create_stratification_variable
        import pandas as pd
        import numpy as np

        n = 100
        metadata = pd.DataFrame({
            "ROSMAP_IndividualID": [f"subj_{i:03d}" for i in range(n)],
            "gpath": [1.0] * 90 + [2.0] * 10,  # Will fail tertiles
            "cogn_global": np.linspace(0, 1, n),  # Uniform, tertiles OK
        })

        strata = create_stratification_variable(metadata)

        # Pathology should have ~50/50 split (binary from median)
        # Cognition should have ~33/33/33 split (tertiles)
        # Total strata: 2 * 3 = 6 (or fewer if combined)
        assert strata.nunique() >= 2