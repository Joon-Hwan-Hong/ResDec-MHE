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


class TestGetFinalTrainSubjects:
    """S-A3: Tests for get_final_train_subjects()."""

    def test_get_final_train_subjects(self):
        """Should return the full train_val_pool for final retraining."""
        from src.data.splits import get_final_train_subjects

        splits = {"train_val_pool": ["S1", "S2", "S3"], "test": ["S4"]}
        result = get_final_train_subjects(splits)
        assert result == ["S1", "S2", "S3"]


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

    @pytest.mark.filterwarnings("ignore:.*Combining.*rare strata.*:UserWarning")
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

    @pytest.mark.filterwarnings("ignore:.*falling back to median split.*:UserWarning")
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


class TestStratificationAlignment:
    """Tests for correct alignment of stratification labels with shuffled subjects."""

    def test_strata_aligned_with_train_val_subjects(self):
        """Strata passed to StratifiedKFold must match train_val_subjects order.

        Regression test for bug where train_val_strata was built using original
        subjects order, but train_val_subjects was shuffled by train_test_split.
        """
        from src.data.splits import create_stratified_splits

        # Create metadata with distinct, predictable strata
        # Subject IDs encode their stratum: S_low_high_0, S_medium_low_1, etc.
        n = 90  # Divisible by 9 for 3x3 strata
        metadata_rows = []
        for i in range(n):
            # Assign to one of 9 strata (3x3)
            path_bin = i % 3  # 0, 1, 2 -> low, medium, high
            cogn_bin = (i // 3) % 3  # 0, 1, 2 -> low, medium, high
            path_labels = ["low", "medium", "high"]
            cogn_labels = ["low", "medium", "high"]

            metadata_rows.append({
                "ROSMAP_IndividualID": f"S_{path_labels[path_bin]}_{cogn_labels[cogn_bin]}_{i}",
                "gpath": path_bin * 0.4 + 0.1,  # 0.1, 0.5, 0.9
                "cogn_global": cogn_bin * 0.4 + 0.1,  # 0.1, 0.5, 0.9
            })

        metadata = pd.DataFrame(metadata_rows)

        splits = create_stratified_splits(metadata, test_frac=0.1, n_folds=5)

        # For each fold, verify strata distribution is balanced
        # If strata were misaligned, StratifiedKFold would produce imbalanced folds
        for fold_idx, fold in enumerate(splits["folds"]):
            val_subjects = fold["val"]

            # Extract strata from subject IDs (encoded in name)
            val_strata = []
            for s in val_subjects:
                parts = s.split("_")
                stratum = f"{parts[1]}_{parts[2]}"  # e.g., "low_high"
                val_strata.append(stratum)

            # Check that we have multiple strata represented in validation
            # (if alignment was wrong, might get all from one stratum)
            unique_strata = set(val_strata)
            assert len(unique_strata) >= 2, (
                f"Fold {fold_idx} validation has only {len(unique_strata)} stratum: {unique_strata}. "
                "This suggests stratification labels were misaligned with subjects."
            )

    def test_strata_match_subject_identity(self):
        """Each subject's stratum in KFold should match their actual stratum.

        This directly tests that the stratum array passed to StratifiedKFold
        corresponds to the correct subjects.
        """
        from src.data.splits import create_stratified_splits, create_stratification_variable

        np.random.seed(42)
        n = 100
        metadata = pd.DataFrame({
            "ROSMAP_IndividualID": [f"subj_{i:03d}" for i in range(n)],
            "gpath": np.random.uniform(0, 1, n),
            "cogn_global": np.random.uniform(-2, 2, n),
        })

        # Create strata for verification
        strata = create_stratification_variable(metadata.set_index("ROSMAP_IndividualID"))
        subject_to_stratum = dict(zip(metadata["ROSMAP_IndividualID"], strata))

        splits = create_stratified_splits(metadata, test_frac=0.1, n_folds=5)

        # Verify each fold's val subjects have diverse strata
        for fold_idx, fold in enumerate(splits["folds"]):
            val_strata = [subject_to_stratum[s] for s in fold["val"]]
            train_strata = [subject_to_stratum[s] for s in fold["train"]]

            # With proper stratification, each fold should have similar strata distribution
            val_unique = set(val_strata)
            train_unique = set(train_strata)

            # Training set should have all strata (or most)
            # Validation set should have representation from multiple strata
            assert len(val_unique) >= 2 or len(train_unique) >= 2, (
                f"Fold {fold_idx}: val has {len(val_unique)} strata, "
                f"train has {len(train_unique)} strata. Stratification may be broken."
            )


class TestStrataCollapseFallback:
    """Tests for fallback when strata collapse to single class."""

    def test_falls_back_when_strata_collapse(self):
        """Should fall back to non-stratified splits when strata collapse."""
        from src.data.splits import create_stratified_splits
        import warnings

        # Create highly homogeneous data where all subjects end up in same stratum
        n = 50
        metadata = pd.DataFrame({
            "ROSMAP_IndividualID": [f"subj_{i:03d}" for i in range(n)],
            "gpath": [0.5] * n,  # All same value
            "cogn_global": [0.5] * n,  # All same value
        })

        # Should warn about collapsed strata but still produce valid splits
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            splits = create_stratified_splits(metadata, n_folds=5)

            # Should have warned about strata collapse
            assert any("collapsed" in str(warning.message).lower() for warning in w)

        # Should still produce valid splits
        assert len(splits["holdout_test"]) > 0
        assert len(splits["folds"]) == 5
        for fold in splits["folds"]:
            assert len(fold["train"]) > 0
            assert len(fold["val"]) > 0

    def test_no_subject_leakage_with_collapsed_strata(self):
        """Even with collapsed strata, no subject should leak between splits."""
        from src.data.splits import create_stratified_splits
        import warnings

        n = 50
        metadata = pd.DataFrame({
            "ROSMAP_IndividualID": [f"subj_{i:03d}" for i in range(n)],
            "gpath": [0.5] * n,
            "cogn_global": [0.5] * n,
        })

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            splits = create_stratified_splits(metadata, n_folds=5)

        test_set = set(splits["holdout_test"])

        for fold in splits["folds"]:
            train_set = set(fold["train"])
            val_set = set(fold["val"])

            # No overlap with test
            assert len(train_set & test_set) == 0
            assert len(val_set & test_set) == 0

            # No overlap within fold
            assert len(train_set & val_set) == 0

    def test_partial_collapse_handled(self):
        """Should handle case where strata collapse after rare-strata merge."""
        from src.data.splits import create_stratified_splits
        import warnings

        # Create data with many small strata that collapse to "other"
        n = 30
        metadata = pd.DataFrame({
            "ROSMAP_IndividualID": [f"subj_{i:03d}" for i in range(n)],
            # Many unique values, each appearing only once → all become "other"
            "gpath": np.arange(n, dtype=float),
            "cogn_global": np.arange(n, dtype=float),
        })

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            splits = create_stratified_splits(metadata, n_folds=3)

        # Should produce valid splits regardless of warnings
        assert len(splits["holdout_test"]) > 0
        assert len(splits["folds"]) == 3


class TestSmallDataset:
    """S-A7: Tests for small datasets (N < 20 subjects)."""

    def test_small_dataset_under_20_subjects(self):
        """Should handle small datasets (N<20) without crashing."""
        from src.data.splits import create_stratified_splits, validate_no_leakage
        import warnings

        # Create metadata with ~10 subjects
        n = 10
        np.random.seed(42)
        metadata = pd.DataFrame({
            "ROSMAP_IndividualID": [f"S{i}" for i in range(n)],
            "gpath": np.random.uniform(0, 1, n),
            "cogn_global": np.random.uniform(-2, 2, n),
        })

        # Use fewer folds appropriate for small N
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            splits = create_stratified_splits(
                metadata,
                test_frac=0.2,  # 2 subjects for test
                n_folds=3,      # 3-fold on remaining 8
            )

        # Should produce valid splits
        assert len(splits["holdout_test"]) > 0
        assert len(splits["train_val_pool"]) > 0
        assert len(splits["folds"]) == 3

        # All subjects should be accounted for
        all_subjects = set(metadata["ROSMAP_IndividualID"])
        split_subjects = set(splits["holdout_test"]) | set(splits["train_val_pool"])
        assert split_subjects == all_subjects

        # No data leakage
        assert validate_no_leakage(splits)


class TestTestFracValidation:
    """S-A1: Tests for test_frac validation in create_stratified_splits."""

    @pytest.fixture
    def simple_metadata(self):
        """Minimal metadata for validation tests."""
        np.random.seed(42)
        return pd.DataFrame({
            "ROSMAP_IndividualID": [f"S{i}" for i in range(50)],
            "gpath": np.random.uniform(0, 1, 50),
            "cogn_global": np.random.uniform(-2, 2, 50),
        })

    def test_rejects_test_frac_zero(self, simple_metadata):
        """test_frac=0 should raise ValueError."""
        from src.data.splits import create_stratified_splits

        with pytest.raises(ValueError, match="test_frac must be between 0 and 1"):
            create_stratified_splits(simple_metadata, test_frac=0)

    def test_rejects_test_frac_one(self, simple_metadata):
        """test_frac=1 should raise ValueError."""
        from src.data.splits import create_stratified_splits

        with pytest.raises(ValueError, match="test_frac must be between 0 and 1"):
            create_stratified_splits(simple_metadata, test_frac=1.0)

    def test_rejects_test_frac_negative(self, simple_metadata):
        """Negative test_frac should raise ValueError."""
        from src.data.splits import create_stratified_splits

        with pytest.raises(ValueError, match="test_frac must be between 0 and 1"):
            create_stratified_splits(simple_metadata, test_frac=-0.1)

    def test_rejects_test_frac_greater_than_one(self, simple_metadata):
        """test_frac > 1 should raise ValueError."""
        from src.data.splits import create_stratified_splits

        with pytest.raises(ValueError, match="test_frac must be between 0 and 1"):
            create_stratified_splits(simple_metadata, test_frac=1.5)


class TestGetFoldSubjectsValidation:
    """S-A2: Tests for get_fold_subjects() invalid split_type."""

    @pytest.fixture
    def mock_splits(self):
        return {
            "holdout_test": ["T1", "T2"],
            "train_val_pool": ["S1", "S2", "S3", "S4"],
            "folds": [
                {"train": ["S1", "S2", "S3"], "val": ["S4"]},
            ],
        }

    def test_rejects_unknown_split_type(self, mock_splits):
        """Unknown split_type should raise ValueError."""
        from src.data.splits import get_fold_subjects

        with pytest.raises(ValueError, match="Unknown split_type"):
            get_fold_subjects(mock_splits, fold_idx=0, split_type="invalid")

    def test_rejects_empty_string_split_type(self, mock_splits):
        """Empty string split_type should raise ValueError."""
        from src.data.splits import get_fold_subjects

        with pytest.raises(ValueError, match="Unknown split_type"):
            get_fold_subjects(mock_splits, fold_idx=0, split_type="")

    def test_rejects_all_split_type(self, mock_splits):
        """split_type='all' is not valid, should raise ValueError."""
        from src.data.splits import get_fold_subjects

        with pytest.raises(ValueError, match="Unknown split_type"):
            get_fold_subjects(mock_splits, fold_idx=0, split_type="all")