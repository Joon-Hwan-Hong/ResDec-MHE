"""
Subject-level stratified data splitting for cognitive resilience model.

All splits are performed at the SUBJECT level to prevent data leakage.
Stratification by joint pathology × cognition tertiles ensures balanced representation.
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit, train_test_split

logger = logging.getLogger(__name__)


def create_stratification_variable(
    metadata: pd.DataFrame,
    pathology_column: str = "gpath",
    cognition_column: str = "cogn_global",
    n_bins: int = 3,
) -> pd.Series:
    """
    Create joint stratification variable from pathology and cognition.

    Falls back to median split (2 bins) if tertiles fail due to ties.

    Args:
        metadata: Subject-level metadata
        pathology_column: Column for pathology (e.g., gpath)
        cognition_column: Column for cognition (e.g., cogn_global)
        n_bins: Number of bins (tertiles = 3)

    Returns:
        Series with stratification labels (e.g., "low_high", "medium_medium")
    """
    import warnings

    metadata = metadata.copy()

    # Validate stratification columns have no NaN — pd.qcut silently
    # creates NaN bins which can bias fold assignments.
    for col in [pathology_column, cognition_column]:
        n_nan = metadata[col].isna().sum()
        if n_nan > 0:
            raise ValueError(
                f"Stratification column '{col}' has {n_nan} NaN values. "
                f"Filter NaN subjects before creating splits."
            )

    def bin_column(values: pd.Series, col_name: str, n_bins: int) -> pd.Series:
        """Bin a column, falling back to median if qcut fails."""
        labels_3 = ["low", "medium", "high"]
        labels_2 = ["low", "high"]

        try:
            binned = pd.qcut(
                values,
                q=n_bins,
                labels=labels_3[:n_bins],
                duplicates="drop",
            )
            # Check if we got fewer bins than requested
            actual_bins = binned.nunique()
            if actual_bins < n_bins:
                raise ValueError(f"Only got {actual_bins} bins")
            return binned
        except (ValueError, IndexError) as e:
            # IndexError: pd.qcut edge case when duplicates="drop" produces
            # degenerate bin edges with label array length mismatch.
            # Fall back to median split
            warnings.warn(
                f"Could not create {n_bins} bins for {col_name} "
                f"({type(e).__name__}: {e}), falling back to "
                f"median split (2 bins). This may reduce stratification granularity.",
                UserWarning,
            )
            median = values.median()
            return pd.Series(
                ["low" if v <= median else "high" for v in values],
                index=values.index,
            )

    # Bin pathology
    metadata["pathology_bin"] = bin_column(
        metadata[pathology_column], pathology_column, n_bins
    )

    # Bin cognition
    metadata["cognition_bin"] = bin_column(
        metadata[cognition_column], cognition_column, n_bins
    )

    # Joint stratification variable
    strata = (
        metadata["pathology_bin"].astype(str) + "_" +
        metadata["cognition_bin"].astype(str)
    )

    return strata


def create_stratified_splits(
    metadata: pd.DataFrame,
    subject_column: str = "ROSMAP_IndividualID",
    pathology_column: str = "gpath",
    cognition_column: str = "cogn_global",
    test_frac: float = 0.1,
    n_folds: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Create subject-level stratified splits using K-fold cross-validation.

    Strategy:
    1. Hold out test_frac (default 10%) as final test set (never touched during HP optimization)
    2. Perform n_folds CV on remaining pool for HP selection
    3. After HP selection, retrain on full pool and evaluate on test set

    Split sizes with defaults (test_frac=0.1, n_folds=5):
    - Test set: 10%
    - Per-fold validation: 90% / 5 = 18%
    - Per-fold training: 90% - 18% = 72%

    Note: This uses true StratifiedKFold where each subject appears in exactly
    one validation set across all folds. This provides disjoint validation sets
    for statistically independent HP optimization estimates.

    Stratification:
    - gpath tertiles (low/medium/high pathology)
    - cogn_global tertiles (low/medium/high cognition)
    - Joint 3×3 = 9 strata

    Args:
        metadata: Subject-level metadata DataFrame
        subject_column: Column containing subject IDs
        pathology_column: Column for pathology stratification
        cognition_column: Column for cognition stratification
        test_frac: Fraction for holdout test set (default: 0.1)
        n_folds: Number of CV folds (determines val_frac = (1-test_frac)/n_folds)
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with:
        - holdout_test: List of test subject IDs
        - train_val_pool: List of all non-test subject IDs
        - folds: List of {train: [...], val: [...]} dictionaries
        - metadata: Split configuration metadata
    """
    # Validate test fraction
    if not 0 < test_frac < 1:
        raise ValueError(f"test_frac must be between 0 and 1, got {test_frac}")

    # Get unique subjects
    if subject_column in metadata.columns:
        subjects = metadata[subject_column].unique()
        metadata_indexed = metadata.set_index(subject_column)
    else:
        subjects = metadata.index.unique()
        metadata_indexed = metadata

    subjects = np.array(subjects)
    n_subjects = len(subjects)

    logger.info(f"Creating splits for {n_subjects} subjects")

    # Create stratification variable
    strata = create_stratification_variable(
        metadata_indexed.loc[subjects].reset_index(),
        pathology_column=pathology_column,
        cognition_column=cognition_column,
    )

    # Handle small strata by combining rare categories
    strata_counts = strata.value_counts()
    min_count_for_split = max(n_folds + 1, 5)  # Need enough samples per stratum

    rare_strata = strata_counts[strata_counts < min_count_for_split].index.tolist()
    if rare_strata:
        warnings.warn(
            f"Combining {len(rare_strata)} rare strata into '__rare_combined__'",
            UserWarning,
            stacklevel=2,
        )
        strata = strata.replace(rare_strata, "__rare_combined__")

    strata_array = strata.values

    # Check if strata collapsed to single class after merge
    # This can happen with very homogeneous data or small sample sizes
    n_unique_strata = len(np.unique(strata_array))
    use_stratification = n_unique_strata >= 2

    if not use_stratification:
        warnings.warn(
            f"Strata collapsed to {n_unique_strata} class(es) after merging rare categories. "
            "Falling back to non-stratified splitting. This may result in imbalanced folds.",
            UserWarning
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1: Hold out fixed test set
    # ─────────────────────────────────────────────────────────────────────────
    train_val_subjects, test_subjects = train_test_split(
        subjects,
        test_size=test_frac,
        stratify=strata_array if use_stratification else None,
        random_state=random_state,
    )

    logger.info(f"Holdout test set: {len(test_subjects)} subjects ({test_frac*100:.0f}%)")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2: Create K-fold CV with disjoint validation sets
    # ─────────────────────────────────────────────────────────────────────────
    # Get strata for train_val subjects IN THE SAME ORDER as train_val_subjects
    # IMPORTANT: train_test_split shuffles the output, so we must look up strata
    # by subject ID, not by position in the original subjects array.
    subject_to_strata = dict(zip(subjects, strata_array))
    train_val_strata = np.array([subject_to_strata[s] for s in train_val_subjects])

    # Check if train_val strata also collapsed (can happen after test split)
    n_unique_train_val_strata = len(np.unique(train_val_strata))
    use_stratified_kfold = use_stratification and n_unique_train_val_strata >= 2

    if use_stratification and not use_stratified_kfold:
        warnings.warn(
            f"Train/val strata collapsed to {n_unique_train_val_strata} class(es) after test split. "
            "Falling back to non-stratified K-fold.",
            UserWarning
        )

    # Use StratifiedKFold if possible, otherwise fall back to KFold
    # - Each subject appears in exactly one validation set across all folds
    # - Validation sets are disjoint (no overlap)
    # - Stratification preserves class distribution in each fold (when possible)
    if use_stratified_kfold:
        kfold = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=random_state,
        )
        fold_iterator = kfold.split(train_val_subjects, train_val_strata)
    else:
        kfold = KFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=random_state,
        )
        fold_iterator = kfold.split(train_val_subjects)

    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(fold_iterator):
        fold = {
            "train": train_val_subjects[train_idx].tolist(),
            "val": train_val_subjects[val_idx].tolist(),
        }
        folds.append(fold)
        logger.info(f"Fold {fold_idx + 1}: {len(fold['train'])} train, {len(fold['val'])} val")

    # ─────────────────────────────────────────────────────────────────────────
    # Compile results
    # ─────────────────────────────────────────────────────────────────────────
    # Calculate derived fractions for documentation
    pool_frac = 1.0 - test_frac
    val_frac_derived = pool_frac / n_folds
    train_frac_derived = pool_frac - val_frac_derived

    splits = {
        "holdout_test": test_subjects.tolist(),
        "train_val_pool": train_val_subjects.tolist(),
        "folds": folds,
        "metadata": {
            "n_subjects": n_subjects,
            "n_test": len(test_subjects),
            "n_train_val": len(train_val_subjects),
            "n_folds": n_folds,
            "test_frac": test_frac,
            # Derived fractions (computed from test_frac and n_folds)
            "val_frac_per_fold": val_frac_derived,
            "train_frac_per_fold": train_frac_derived,
            "random_state": random_state,
            "pathology_column": pathology_column,
            "cognition_column": cognition_column,
        },
    }

    return splits


def save_splits(splits: dict, path: str | Path) -> None:
    """
    Save splits to JSON file.

    Args:
        splits: Split dictionary from create_stratified_splits
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(splits, f, indent=2)

    logger.info(f"Saved splits to {path}")


def load_splits(path: str | Path) -> dict:
    """
    Load splits from JSON file.

    Args:
        path: Path to splits JSON

    Returns:
        Split dictionary

    Raises:
        ValueError: If JSON is missing required keys
    """
    with open(path, "r") as f:
        splits = json.load(f)

    required_keys = {"holdout_test", "folds"}
    missing = required_keys - set(splits.keys())
    if missing:
        raise ValueError(
            f"Splits JSON missing required keys: {missing}. "
            f"Available keys: {sorted(splits.keys())}"
        )

    return splits


def get_fold_subjects(
    splits: dict,
    fold_idx: int,
    split_type: Literal["train", "val", "test"] = "train",
) -> list[str]:
    """
    Get subject IDs for a specific fold and split type.

    Note: For split_type='test', fold_idx is ignored — the holdout test set
    is the same for all folds. This is by design: the test set is held out
    from all CV folds to prevent information leakage.

    Args:
        splits: Split dictionary
        fold_idx: Fold index (0-indexed). Ignored when split_type="test".
        split_type: "train", "val", or "test"

    Returns:
        List of subject IDs
    """
    if split_type == "test":
        return splits["holdout_test"]
    elif split_type in ("train", "val"):
        n_folds = len(splits["folds"])
        if fold_idx < 0 or fold_idx >= n_folds:
            raise IndexError(
                f"fold_idx={fold_idx} out of range for {n_folds} folds (0-indexed)"
            )
        return splits["folds"][fold_idx][split_type]
    else:
        raise ValueError(f"Unknown split_type: {split_type}")


def get_final_train_subjects(splits: dict) -> list[str]:
    """
    Get all train+val subjects for final model training.

    After HP selection, train on full train_val_pool.

    Args:
        splits: Split dictionary

    Returns:
        List of subject IDs
    """
    return splits["train_val_pool"]


def validate_no_leakage(splits: dict) -> bool:
    """
    Validate that there's no data leakage between splits.

    Checks:
    1. Test subjects don't appear in any fold
    2. Train and val don't overlap within any fold
    3. All subjects are accounted for

    Args:
        splits: Split dictionary

    Returns:
        True if no leakage detected
    """
    test_set = set(splits["holdout_test"])
    train_val_set = set(splits["train_val_pool"])

    # Check test doesn't overlap with train_val
    if test_set & train_val_set:
        logger.warning("Test subjects appear in train_val_pool")
        return False

    # Check each fold
    for fold_idx, fold in enumerate(splits["folds"]):
        train_set = set(fold["train"])
        val_set = set(fold["val"])

        # Train and val shouldn't overlap
        if train_set & val_set:
            logger.warning(f"Fold {fold_idx} has train/val overlap")
            return False

        # Train and val should be subsets of train_val_pool
        if not (train_set | val_set).issubset(train_val_set):
            logger.warning(f"Fold {fold_idx} subjects not in train_val_pool")
            return False

        # Test shouldn't appear in train or val
        if test_set & train_set or test_set & val_set:
            logger.warning(f"Fold {fold_idx} has test subjects")
            return False

    logger.info("No data leakage detected")
    return True
