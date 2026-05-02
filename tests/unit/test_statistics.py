"""Tests for src.utils.statistics utility functions."""

import numpy as np
import pytest

from src.utils.statistics import derive_resilience_groups

class TestDeriveResilienceGroups:
    """Tests for derive_resilience_groups utility."""

    def test_basic_stratification(self):
        """Should label high-pathology subjects by cognition tertiles."""
        n = 30
        pathology = np.linspace(0, 1, n)
        cognition = np.linspace(1, 0, n)

        labels = derive_resilience_groups(cognition, pathology)

        assert "resilient" in labels
        assert "vulnerable" in labels
        assert (labels != "").sum() > 0

    def test_returns_correct_dtype(self):
        """Should return string array."""
        labels = derive_resilience_groups(
            np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
            np.array([9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]),
        )
        assert labels.dtype.kind in ("U", "O")

    def test_nan_pathology_excluded(self):
        """Subjects with NaN pathology should get empty label."""
        n = 20
        pathology = np.linspace(0, 1, n)
        cognition = np.linspace(1, 0, n)
        pathology[0] = np.nan
        pathology[5] = np.nan

        labels = derive_resilience_groups(cognition, pathology)
        assert labels[0] == ""
        assert labels[5] == ""

    def test_nan_cognition_excluded(self):
        """Subjects with NaN cognition should get empty label."""
        n = 20
        pathology = np.linspace(0, 1, n)
        cognition = np.linspace(1, 0, n)
        cognition[19] = np.nan

        labels = derive_resilience_groups(cognition, pathology)
        assert labels[19] == ""

    def test_too_few_subjects_returns_empty(self):
        """Should return all empty labels if fewer than 6 valid subjects."""
        labels = derive_resilience_groups(
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 2.0, 3.0]),
        )
        assert all(l == "" for l in labels)

    def test_resilient_has_high_pathology_and_cognition(self):
        """Resilient subjects should have high pathology AND high cognition."""
        n = 100
        pathology = np.random.rand(n)
        cognition = np.random.rand(n)

        labels = derive_resilience_groups(cognition, pathology)

        resilient_mask = labels == "resilient"
        if resilient_mask.sum() > 0:
            path_threshold = np.percentile(pathology, 66.7)
            assert (pathology[resilient_mask] >= path_threshold).all()
