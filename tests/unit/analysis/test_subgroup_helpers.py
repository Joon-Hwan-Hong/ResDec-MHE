"""Unit tests for :mod:`src.analysis.subgroup_helpers`.

These helpers are shared between
``scripts/redesign/interpretability/variance_decomposition.py`` and
``scripts/redesign/interpretability/subgroup_r2.py``. The tests pin the
labeling contract (APOE ε4 count, sex string, rank-then-qcut quantiles with
NaN preserved as None) so future renames/refactors cannot silently change
subgroup membership.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.subgroup_helpers import (
    apoe_e4_count_label,
    msex_label,
    quantile_labels,
)


def test_quantile_labels_basic():
    """8 equally-spaced points should map to Q1 Q1 Q2 Q2 Q3 Q3 Q4 Q4."""
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    out = quantile_labels(s, n_quantiles=4, prefix="Q")
    expected = ["Q1", "Q1", "Q2", "Q2", "Q3", "Q3", "Q4", "Q4"]
    assert list(out) == expected


def test_quantile_labels_preserves_none_for_nan():
    """NaN entries must yield None (not a bucket label) so downstream drops them."""
    s = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0])
    out = quantile_labels(s, n_quantiles=4, prefix="Q")
    # Exactly one None at the NaN position.
    assert out.iloc[2] is None
    none_count = sum(1 for x in out if x is None)
    assert none_count == 1
    # The 7 non-NaN entries all get a Q1..Q4 label.
    labelled = [x for x in out if x is not None]
    assert len(labelled) == 7
    assert all(x in {"Q1", "Q2", "Q3", "Q4"} for x in labelled)


def test_quantile_labels_n_quantiles_3():
    """n_quantiles=3 should yield T1/T2/T3-style labels (with prefix='T')."""
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    out = quantile_labels(s, n_quantiles=3, prefix="T")
    # 6 points / 3 buckets → 2 per bucket.
    expected = ["T1", "T1", "T2", "T2", "T3", "T3"]
    assert list(out) == expected


def test_apoe_e4_count_label_codes():
    """APOE genotype two-digit codes → ε4 count as string; NaN → None."""
    assert apoe_e4_count_label(33) == "0"
    assert apoe_e4_count_label(34) == "1"
    assert apoe_e4_count_label(44) == "2"
    assert apoe_e4_count_label(23) == "0"
    assert apoe_e4_count_label(24) == "1"
    assert apoe_e4_count_label(float("nan")) is None
    assert apoe_e4_count_label(None) is None


def test_msex_label_codes():
    """msex 0/1 → '0'/'1' strings; NaN → None."""
    assert msex_label(0) == "0"
    assert msex_label(1) == "1"
    assert msex_label(0.0) == "0"
    assert msex_label(1.0) == "1"
    assert msex_label(float("nan")) is None
