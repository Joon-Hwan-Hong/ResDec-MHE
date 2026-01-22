"""
Subject-level stratified data splitting for cognitive resilience model.

All splits are performed at the SUBJECT level to prevent data leakage.
Stratification by joint pathology × cognition tertiles ensures balanced representation.
"""

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def create_stratification_variable(
    metadata: pd.DataFrame,
    pathology_column: str = "gpath",
    cognition_column: str = "cogn_global",
    n_bins: int = 3,
) -> pd.Series:
    """
    Create joint stratification variable from pathology and cognition.

    Args:
        metadata: Subject-level metadata
        pathology_column: Column for pathology (e.g., gpath)
        cognition_column: Column for cognition (e.g., cogn_global)
        n_bins: Number of bins (tertiles = 3)

    Returns:
        Series with stratification labels (e.g., "low_high", "medium_medium")
    """
    metadata = metadata.copy()

    # Create tertile bins for pathology
    metadata["pathology_bin"] = pd.qcut(
        metadata[pathology_column],
        q=n_bins,
        labels=["low", "medium", "high"][:n_bins],
        duplicates="drop",
    )

    # Create tertile bins for cognition
    metadata["cognition_bin"] = pd.qcut(
        metadata[cognition_column],
        q=n_bins,
        labels=["low", "medium", "high"][:n_bins],
        duplicates="drop",
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
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    n_folds: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Create subject-level stratified splits.

    Strategy:
    1. Hold out 10% as final test set (never touched during HP optimization)
    2. Perform 5-fold CV on remaining 90% for HP selection
    3. After HP selection, retrain on full 90% and evaluate on test set

    Stratification:
    - gpath tertiles (low/medium/high pathology)
    - cogn_global tertiles (low/medium/high cognition)
    - Joint 3×3 = 9 strata

    Args:
        metadata: Subject-level metadata DataFrame
        subject_column: Column containing subject IDs
        pathology_column: Column for pathology stratification
        cognition_column: Column for cognition stratification
        train_frac: Fraction for training (within train+val pool)
        val_frac: Fraction for validation
        test_frac: Fraction for holdout test
        n_folds: Number of CV folds
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with:
        - holdout_test: List of test subject IDs
        - folds: List of {train: [...], val: [...]} dictionaries
        - metadata: Split configuration metadata
    """
    # Validate fractions
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Fractions must sum to 1"

    # Get unique subjects
    if subject_column in metadata.columns:
        subjects = metadata[subject_column].unique()
        metadata_indexed = metadata.set_index(subject_column)
    else:
        subjects = metadata.index.unique()
        metadata_indexed = metadata

    subjects = np.array(subjects)
    n_subjects = len(subjects)

    print(f"Creating splits for {n_subjects} subjects")

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
        print(f"Combining {len(rare_strata)} rare strata into 'other'")
        strata = strata.replace(rare_strata, "other")

    strata_array = strata.values

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1: Hold out fixed test set
    # ─────────────────────────────────────────────────────────────────────────
    train_val_subjects, test_subjects = train_test_split(
        subjects,
        test_size=test_frac,
        stratify=strata_array,
        random_state=random_state,
    )

    print(f"Holdout test set: {len(test_subjects)} subjects ({test_frac*100:.0f}%)")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2: K-fold CV on remaining subjects
    # ─────────────────────────────────────────────────────────────────────────
    # Get strata for train_val subjects
    train_val_indices = np.isin(subjects, train_val_subjects)
    train_val_strata = strata_array[train_val_indices]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_val_subjects, train_val_strata)):
        fold = {
            "train": train_val_subjects[train_idx].tolist(),
            "val": train_val_subjects[val_idx].tolist(),
        }
        folds.append(fold)
        print(f"Fold {fold_idx + 1}: {len(fold['train'])} train, {len(fold['val'])} val")

    # ─────────────────────────────────────────────────────────────────────────
    # Compile results
    # ─────────────────────────────────────────────────────────────────────────
    splits = {
        "holdout_test": test_subjects.tolist(),
        "train_val_pool": train_val_subjects.tolist(),
        "folds": folds,
        "metadata": {
            "n_subjects": n_subjects,
            "n_test": len(test_subjects),
            "n_train_val": len(train_val_subjects),
            "n_folds": n_folds,
            "train_frac": train_frac,
            "val_frac": val_frac,
            "test_frac": test_frac,
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

    print(f"Saved splits to {path}")


def load_splits(path: str | Path) -> dict:
    """
    Load splits from JSON file.

    Args:
        path: Path to splits JSON

    Returns:
        Split dictionary
    """
    with open(path, "r") as f:
        splits = json.load(f)

    return splits


def get_fold_subjects(
    splits: dict,
    fold_idx: int,
    split_type: Literal["train", "val", "test"] = "train",
) -> list[str]:
    """
    Get subject IDs for a specific fold and split type.

    Args:
        splits: Split dictionary
        fold_idx: Fold index (0-indexed)
        split_type: "train", "val", or "test"

    Returns:
        List of subject IDs
    """
    if split_type == "test":
        return splits["holdout_test"]
    elif split_type in ("train", "val"):
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
        print("ERROR: Test subjects appear in train_val_pool")
        return False

    # Check each fold
    for fold_idx, fold in enumerate(splits["folds"]):
        train_set = set(fold["train"])
        val_set = set(fold["val"])

        # Train and val shouldn't overlap
        if train_set & val_set:
            print(f"ERROR: Fold {fold_idx} has train/val overlap")
            return False

        # Train and val should be subsets of train_val_pool
        if not (train_set | val_set).issubset(train_val_set):
            print(f"ERROR: Fold {fold_idx} subjects not in train_val_pool")
            return False

        # Test shouldn't appear in train or val
        if test_set & train_set or test_set & val_set:
            print(f"ERROR: Fold {fold_idx} has test subjects")
            return False

    print("No data leakage detected")
    return True


def print_split_summary(splits: dict) -> None:
    """Print summary of splits."""
    meta = splits["metadata"]

    print("\n" + "=" * 60)
    print("SPLIT SUMMARY")
    print("=" * 60)
    print(f"Total subjects: {meta['n_subjects']}")
    print(f"Holdout test: {meta['n_test']} ({meta['test_frac']*100:.0f}%)")
    print(f"Train+Val pool: {meta['n_train_val']} ({(1-meta['test_frac'])*100:.0f}%)")
    print(f"Number of CV folds: {meta['n_folds']}")
    print(f"Random state: {meta['random_state']}")
    print()

    for i, fold in enumerate(splits["folds"]):
        n_train = len(fold["train"])
        n_val = len(fold["val"])
        print(f"  Fold {i+1}: {n_train} train, {n_val} val")

    print("=" * 60 + "\n")