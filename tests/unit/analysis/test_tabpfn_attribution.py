"""Tests for src/analysis/tabpfn_attribution.py (Captum IG on TabPFN)."""

import pytest
import numpy as np

from src.analysis.tabpfn_attribution import (
    hydrate_feature_indices,
    attribute_tabpfn_fold,
    N_GENES,
    N_CT,
)


# ---------------------------------------------------------------------------
# hydrate_feature_indices — pure Python, fast, always run
# ---------------------------------------------------------------------------


def test_hydrate_indices_basic():
    df = hydrate_feature_indices([0, 4785, 9570, 148334])
    assert len(df) == 4
    # Index 0 -> (ct=0, gene=0)
    assert df.iloc[0]["ct_id"] == 0
    assert df.iloc[0]["gene_id"] == 0
    assert df.iloc[0]["feature_idx"] == 0
    # Index 4785 -> (ct=1, gene=0)
    assert df.iloc[1]["ct_id"] == 1
    assert df.iloc[1]["gene_id"] == 0
    # Index 9570 -> (ct=2, gene=0)
    assert df.iloc[2]["ct_id"] == 2
    assert df.iloc[2]["gene_id"] == 0
    # Last possible index: 30 * 4785 + 4784 = 148_334
    assert df.iloc[3]["ct_id"] == 30
    assert df.iloc[3]["gene_id"] == 4784


def test_hydrate_indices_columns_and_schema():
    df = hydrate_feature_indices([17, 2_500])
    # Must have exactly these three columns
    assert list(df.columns) == ["feature_idx", "ct_id", "gene_id"]
    # ct_id < 31 and gene_id < 4785 always
    assert (df["ct_id"] < N_CT).all()
    assert (df["gene_id"] < N_GENES).all()
    # Round-trip: ct * 4785 + gene == feature_idx
    assert (df["ct_id"] * N_GENES + df["gene_id"] == df["feature_idx"]).all()


def test_hydrate_indices_empty():
    df = hydrate_feature_indices([])
    assert len(df) == 0
    # Schema still correct
    assert list(df.columns) == ["feature_idx", "ct_id", "gene_id"]


# ---------------------------------------------------------------------------
# End-to-end smoke — fits TabPFN on fold 0, runs attribution on 3 val subjects.
# Marked slow + cuda: skipped in default runs, run explicitly with `-m slow`.
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.cuda
def test_attribute_tabpfn_fold0_smoke():
    """Smoke-run attribution on 3 val subjects for fold 0.

    Asserts shapes, schema, and that top-attribution records are well-formed.
    """
    result = attribute_tabpfn_fold(fold_idx=0, n_val_subjects=3, device="cuda:1")
    # Shapes
    assert result["attributions"].shape == (3, 2000)
    assert result["mean_abs_attrib"].shape == (2000,)
    # Subjects
    assert len(result["val_subject_ids"]) == 3
    # Feature schema (top-2K hydrated to (ct, gene)) has 2000 rows and 3 cols
    assert result["feature_schema"].shape == (2000, 3)
    # Per-subject top-20 records
    assert len(result["top_attrib_per_subject"]) == 3
    for sid, tops in result["top_attrib_per_subject"].items():
        assert len(tops) == 20
        for t in tops:
            assert 0 <= t["ct_id"] < N_CT
            assert 0 <= t["gene_id"] < N_GENES
            assert "score" in t
            assert "feature_idx" in t
    # Method tag is present
    assert "method" in result
