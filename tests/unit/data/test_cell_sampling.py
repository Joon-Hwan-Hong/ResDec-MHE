"""
Tests for src/data/cell_sampling.py

Tests cover:
- CellSampler strategies (random, stratified)
- Correct handling of min_cells_threshold
- Cell type coverage filtering
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData


@pytest.fixture
def mock_adata():
    """Create mock AnnData with 3 cell types, varying cell counts."""
    n_cells = 200
    n_genes = 50

    X = np.random.rand(n_cells, n_genes).astype(np.float32)

    obs = pd.DataFrame({
        "supercluster_name": (
            ["Astrocyte"] * 100 +      # 100 cells
            ["Oligodendrocyte"] * 70 +  # 70 cells
            ["Microglia"] * 30          # 30 cells
        ),
    })

    var = pd.DataFrame(index=[f"Gene{i}" for i in range(n_genes)])

    return AnnData(X=X, obs=obs, var=var)


class TestCellSampler:
    """Tests for CellSampler class."""

    def test_samples_up_to_max_cells(self, mock_adata):
        """Should sample at most max_cells_per_type."""
        from src.data.cell_sampling import CellSampler

        sampler = CellSampler(max_cells_per_type=50, min_cells_threshold=10)
        indices = sampler.sample(mock_adata)

        # Astrocyte has 100 cells, should be sampled down to 50
        assert len(indices["Astrocyte"]) == 50

    def test_takes_all_if_below_max(self, mock_adata):
        """If cell type has fewer than max, take all cells."""
        from src.data.cell_sampling import CellSampler

        sampler = CellSampler(max_cells_per_type=200, min_cells_threshold=10)
        indices = sampler.sample(mock_adata)

        # All cell types have < 200 cells, should take all
        assert len(indices["Astrocyte"]) == 100
        assert len(indices["Oligodendrocyte"]) == 70
        assert len(indices["Microglia"]) == 30

    def test_respects_min_threshold(self, mock_adata):
        """Cell types below threshold should have empty arrays."""
        from src.data.cell_sampling import CellSampler

        sampler = CellSampler(max_cells_per_type=100, min_cells_threshold=50)
        indices = sampler.sample(mock_adata)

        # Microglia has only 30 cells, below threshold of 50
        assert len(indices["Microglia"]) == 0

        # Others should be sampled
        assert len(indices["Astrocyte"]) > 0
        assert len(indices["Oligodendrocyte"]) > 0

    def test_reproducibility_with_seed(self, mock_adata):
        """Same seed should produce same samples."""
        from src.data.cell_sampling import CellSampler

        sampler1 = CellSampler(max_cells_per_type=50, seed=42)
        sampler2 = CellSampler(max_cells_per_type=50, seed=42)

        indices1 = sampler1.sample(mock_adata)
        indices2 = sampler2.sample(mock_adata)

        np.testing.assert_array_equal(indices1["Astrocyte"], indices2["Astrocyte"])

    def test_different_seeds_different_samples(self, mock_adata):
        """Different seeds should produce different samples."""
        from src.data.cell_sampling import CellSampler

        sampler1 = CellSampler(max_cells_per_type=50, seed=42)
        sampler2 = CellSampler(max_cells_per_type=50, seed=123)

        indices1 = sampler1.sample(mock_adata)
        indices2 = sampler2.sample(mock_adata)

        # Very unlikely to be identical
        assert not np.array_equal(indices1["Astrocyte"], indices2["Astrocyte"])

    def test_custom_cell_types(self, mock_adata):
        """Sample only specified cell types."""
        from src.data.cell_sampling import CellSampler

        sampler = CellSampler(max_cells_per_type=50)
        indices = sampler.sample(
            mock_adata,
            cell_types=["Astrocyte", "Microglia"]  # Skip Oligodendrocyte
        )

        assert "Astrocyte" in indices
        assert "Microglia" in indices
        assert "Oligodendrocyte" not in indices


class TestSubsampleAdata:
    """Tests for subsample_adata()."""

    def test_reduces_cell_count(self, mock_adata):
        """Subsample reduces cells to max per type."""
        from src.data.cell_sampling import subsample_adata

        # Original: Astrocyte=100, Oligo=70, Microglia=30
        subsampled = subsample_adata(mock_adata, max_cells_per_type=50)

        ct_counts = subsampled.obs["supercluster_name"].value_counts()

        # Should have at most 50 per type
        assert ct_counts["Astrocyte"] == 50
        assert ct_counts["Oligodendrocyte"] == 50
        assert ct_counts["Microglia"] == 30  # Already below threshold

    def test_preserves_all_if_below_max(self, mock_adata):
        """If max is higher than all counts, preserve everything."""
        from src.data.cell_sampling import subsample_adata

        subsampled = subsample_adata(mock_adata, max_cells_per_type=1000)

        assert subsampled.n_obs == mock_adata.n_obs


class TestGetCellTypeCounts:
    """Tests for get_cell_type_counts()."""

    def test_returns_counts_per_subject(self):
        """Return nested dict of subject -> cell_type -> count."""
        from src.data.cell_sampling import get_cell_type_counts

        X = np.random.rand(10, 5)
        obs = pd.DataFrame({
            "ROSMAP_IndividualID": ["S1"] * 6 + ["S2"] * 4,
            "supercluster_name": ["A", "A", "A", "B", "B", "B", "A", "A", "B", "B"],
        })
        adata = AnnData(X=X, obs=obs)

        counts = get_cell_type_counts(adata)

        assert counts["S1"]["A"] == 3
        assert counts["S1"]["B"] == 3
        assert counts["S2"]["A"] == 2
        assert counts["S2"]["B"] == 2


class TestFilterSubjectsByCellCoverage:
    """Tests for filter_subjects_by_cell_coverage()."""

    def test_filters_by_required_types(self):
        """Only keep subjects with all required types at threshold."""
        from src.data.cell_sampling import filter_subjects_by_cell_coverage

        X = np.random.rand(100, 5)
        obs = pd.DataFrame({
            "ROSMAP_IndividualID": ["S1"] * 60 + ["S2"] * 40,
            "supercluster_name": (
                ["A"] * 30 + ["B"] * 20 + ["C"] * 10 +  # S1
                ["A"] * 35 + ["B"] * 5                    # S2: not enough B
            ),
        })
        adata = AnnData(X=X, obs=obs)

        # Require A and B with at least 10 cells each
        valid = filter_subjects_by_cell_coverage(
            adata,
            required_cell_types=["A", "B"],
            min_cells_per_type=10,
        )

        # S1 has A=30, B=20 -> valid
        # S2 has A=35, B=5 -> invalid (B < 10)
        assert valid == ["S1"]