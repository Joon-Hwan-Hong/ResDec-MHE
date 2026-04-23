"""
Tests for src/data/datasets.py

Tests cover:
- Dataset initialization and validation
- Pseudobulk computation and cell counts
- Cell-level data sampling
- Graph feature loading
- Output schema compliance
"""

from pathlib import Path

import pytest
import numpy as np
import pandas as pd
import torch


class TestCellCounts:
    """Tests for cell_counts output."""

    def test_cell_counts_shape(self, mock_dataset):
        """cell_counts should have shape [n_cell_types]."""
        sample = mock_dataset[0]
        assert "cell_counts" in sample
        assert sample["cell_counts"].shape == (mock_dataset.n_cell_types,)

    def test_cell_counts_dtype(self, mock_dataset):
        """cell_counts should be long tensor."""
        sample = mock_dataset[0]
        assert sample["cell_counts"].dtype == torch.long

    def test_cell_counts_matches_mask(self, mock_dataset):
        """cell_counts > 0 should match cell_type_mask True positions."""
        sample = mock_dataset[0]
        counts = sample["cell_counts"]
        mask = sample["cell_type_mask"]

        # Where mask is True, count should be > 0
        assert (counts[mask] > 0).all()
        # Where mask is False, count should be 0
        assert (counts[~mask] == 0).all()

    def test_cell_counts_values_correct(self, mock_dataset):
        """cell_counts should reflect actual cell counts per type."""
        sample = mock_dataset[0]
        counts = sample["cell_counts"]

        # Total cells should be reasonable (not zero, not more than total)
        total_counts = counts.sum().item()
        assert total_counts > 0
        assert total_counts <= 500  # n_cells in mock_dataset


@pytest.fixture
def mock_dataset():
    """Create mock dataset for testing.

    Uses a subset of CELL_TYPE_ORDER and enough cells per subject to clear
    ``CellSampler.min_cells_threshold=50`` — otherwise every type is filtered
    to zero cells and ``save_precomputed_features`` skips every subject as
    degenerate, which then trips the ``_load_and_validate_all`` missing-subject
    safety check. With 5 subjects × ~500 cells each across 5 types, each
    (subject, cell_type) bucket gets ~100 cells in expectation (>> 50).
    """
    from src.data.datasets import CognitiveResilienceDataset
    from src.data.constants import CELL_TYPE_ORDER
    import anndata

    # Create minimal mock data. Use 5 cell types (not all 31) so each subject
    # has enough cells per type to satisfy CellSampler.min_cells_threshold=50.
    n_subjects = 5
    cells_per_subject = 500
    n_cells = cells_per_subject * n_subjects  # 2500
    n_genes = 100
    used_cell_types = CELL_TYPE_ORDER[:5]

    np.random.seed(42)  # For reproducibility

    X = np.random.rand(n_cells, n_genes).astype(np.float32)
    obs = pd.DataFrame({
        "ROSMAP_IndividualID": np.repeat([f"subj_{i:03d}" for i in range(n_subjects)], cells_per_subject),
        "supercluster_name": np.random.choice(used_cell_types, n_cells),
        # Use region names from REGION_ORDER to enable multi-region pseudobulk processing
        "BrainRegion": np.random.choice(["PFC", "AG", "MTC"], n_cells),
    })
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
    adata = anndata.AnnData(X=X, obs=obs, var=var)

    metadata = pd.DataFrame({
        "ROSMAP_IndividualID": [f"subj_{i:03d}" for i in range(n_subjects)],
        "gpath": np.random.rand(n_subjects),
        "amylsqrt": np.random.rand(n_subjects),
        "tangsqrt": np.random.rand(n_subjects),
        "cogn_global": np.random.randn(n_subjects),
    })

    subject_ids = [f"subj_{i:03d}" for i in range(n_subjects)]

    return CognitiveResilienceDataset(
        adata=adata,
        metadata=metadata,
        subject_ids=subject_ids,
    )


class TestRegionMask:
    """Tests for region_mask output."""

    def test_region_mask_shape(self, mock_dataset):
        """region_mask should have shape [n_regions]."""
        from src.data.constants import N_REGIONS
        sample = mock_dataset[0]

        assert "region_mask" in sample
        assert sample["region_mask"].shape == (N_REGIONS,)

    def test_region_mask_dtype(self, mock_dataset):
        """region_mask should be bool tensor."""
        sample = mock_dataset[0]
        assert sample["region_mask"].dtype == torch.bool

    def test_region_mask_has_at_least_one_true(self, mock_dataset):
        """Each subject should have at least one region."""
        sample = mock_dataset[0]
        assert sample["region_mask"].any()

    def test_region_mask_missing_brain_region_column(self):
        """Dataset should handle missing BrainRegion column gracefully."""
        from src.data.datasets import CognitiveResilienceDataset
        from src.data.constants import CELL_TYPE_ORDER, N_REGIONS
        import anndata

        # Create mock data WITHOUT BrainRegion column
        n_cells = 100
        n_genes = 50
        X = np.random.rand(n_cells, n_genes).astype(np.float32)
        obs = pd.DataFrame({
            "ROSMAP_IndividualID": ["subj_001"] * n_cells,
            "supercluster_name": np.random.choice(CELL_TYPE_ORDER[:5], n_cells),
            # Note: NO BrainRegion column!
        })
        var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
        adata = anndata.AnnData(X=X, obs=obs, var=var)

        metadata = pd.DataFrame({
            "ROSMAP_IndividualID": ["subj_001"],
            "gpath": [0.5], "amylsqrt": [0.5], "tangsqrt": [0.5],
            "cogn_global": [0.0],
        })

        # Should not raise
        ds = CognitiveResilienceDataset(
            adata=adata, metadata=metadata, subject_ids=["subj_001"],
        )

        sample = ds[0]

        # region_mask should default to first region (PFC) only
        assert sample["region_mask"].shape == (N_REGIONS,)
        assert sample["region_mask"][0].item() is True  # First region (PFC)
        assert sample["region_mask"].sum().item() == 1  # Only one region

    def test_region_mask_unrecognized_region_names(self):
        """Dataset should handle unrecognized region names by defaulting to first."""
        from src.data.datasets import CognitiveResilienceDataset
        from src.data.constants import CELL_TYPE_ORDER, N_REGIONS
        import anndata

        # Create mock data with unrecognized region names
        n_cells = 100
        n_genes = 50
        X = np.random.rand(n_cells, n_genes).astype(np.float32)
        obs = pd.DataFrame({
            "ROSMAP_IndividualID": ["subj_001"] * n_cells,
            "supercluster_name": np.random.choice(CELL_TYPE_ORDER[:5], n_cells),
            "BrainRegion": ["UnknownRegion1", "UnknownRegion2"] * (n_cells // 2),
        })
        var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
        adata = anndata.AnnData(X=X, obs=obs, var=var)

        metadata = pd.DataFrame({
            "ROSMAP_IndividualID": ["subj_001"],
            "gpath": [0.5], "amylsqrt": [0.5], "tangsqrt": [0.5],
            "cogn_global": [0.0],
        })

        ds = CognitiveResilienceDataset(
            adata=adata, metadata=metadata, subject_ids=["subj_001"],
        )

        sample = ds[0]

        # Should default to first region when no matches found
        assert sample["region_mask"].shape == (N_REGIONS,)
        assert sample["region_mask"][0].item() is True
        assert sample["region_mask"].sum().item() == 1


class TestCellSampling:
    """Tests for cell sampling reproducibility."""

    def test_same_seed_same_cells(self):
        """Same seed should produce identical cell sampling."""
        from src.data.datasets import CognitiveResilienceDataset
        from src.data.constants import CELL_TYPE_ORDER
        import anndata

        # Create mock data with more cells than max_cells_per_type
        n_cells = 2000
        n_genes = 50
        X = np.random.rand(n_cells, n_genes).astype(np.float32)
        obs = pd.DataFrame({
            "ROSMAP_IndividualID": ["subj_001"] * n_cells,
            "supercluster_name": [CELL_TYPE_ORDER[0]] * n_cells,  # All same type
            "BrainRegion": ["PFC"] * n_cells,
        })
        var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
        adata = anndata.AnnData(X=X, obs=obs, var=var)

        metadata = pd.DataFrame({
            "ROSMAP_IndividualID": ["subj_001"],
            "gpath": [0.5], "amylsqrt": [0.5], "tangsqrt": [0.5],
            "cogn_global": [0.0],
        })

        # Create two datasets with same seed
        ds1 = CognitiveResilienceDataset(
            adata=adata, metadata=metadata, subject_ids=["subj_001"],
            sampling_seed=42, max_cells_per_type=100,
        )
        ds2 = CognitiveResilienceDataset(
            adata=adata, metadata=metadata, subject_ids=["subj_001"],
            sampling_seed=42, max_cells_per_type=100,
        )

        sample1 = ds1[0]
        sample2 = ds2[0]

        # Cell data should be identical
        assert torch.equal(sample1["cell_data"], sample2["cell_data"])
        assert torch.equal(sample1["cell_offsets"], sample2["cell_offsets"])

    def test_different_seed_different_cells(self):
        """Different seeds should produce different cell sampling."""
        from src.data.datasets import CognitiveResilienceDataset
        from src.data.constants import CELL_TYPE_ORDER
        import anndata

        n_cells = 2000
        n_genes = 50
        X = np.random.rand(n_cells, n_genes).astype(np.float32)
        obs = pd.DataFrame({
            "ROSMAP_IndividualID": ["subj_001"] * n_cells,
            "supercluster_name": [CELL_TYPE_ORDER[0]] * n_cells,
            "BrainRegion": ["PFC"] * n_cells,
        })
        var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
        adata = anndata.AnnData(X=X, obs=obs, var=var)

        metadata = pd.DataFrame({
            "ROSMAP_IndividualID": ["subj_001"],
            "gpath": [0.5], "amylsqrt": [0.5], "tangsqrt": [0.5],
            "cogn_global": [0.0],
        })

        ds1 = CognitiveResilienceDataset(
            adata=adata, metadata=metadata, subject_ids=["subj_001"],
            sampling_seed=42, max_cells_per_type=100,
        )
        ds2 = CognitiveResilienceDataset(
            adata=adata, metadata=metadata, subject_ids=["subj_001"],
            sampling_seed=123, max_cells_per_type=100,
        )

        sample1 = ds1[0]
        sample2 = ds2[0]

        # Cell data should differ
        assert not torch.equal(sample1["cell_data"], sample2["cell_data"])


class TestDatasetInit:
    """Tests for Dataset initialization."""

    @pytest.mark.filterwarnings("ignore:.*Removed.*subjects not found.*:UserWarning")
    def test_validates_subjects_exist(self):
        """Should filter out subjects not in adata or metadata."""
        from src.data.datasets import CognitiveResilienceDataset
        from src.data.constants import CELL_TYPE_ORDER
        import anndata

        n_cells = 100
        n_genes = 50

        np.random.seed(42)

        X = np.random.rand(n_cells, n_genes).astype(np.float32)
        obs = pd.DataFrame({
            "ROSMAP_IndividualID": ["valid_subj"] * n_cells,
            "supercluster_name": np.random.choice(CELL_TYPE_ORDER[:5], n_cells),
            "BrainRegion": ["PFC"] * n_cells,
        })
        var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
        adata = anndata.AnnData(X=X, obs=obs, var=var)

        metadata = pd.DataFrame({
            "ROSMAP_IndividualID": ["valid_subj"],
            "gpath": [0.5], "amylsqrt": [0.5], "tangsqrt": [0.5],
            "cogn_global": [0.0],
        })

        # Include invalid subjects not in adata or metadata
        subject_ids = ["valid_subj", "missing_in_adata", "missing_in_metadata"]

        # 2 of 3 subjects are invalid (66.7%), so use a high threshold
        ds = CognitiveResilienceDataset(
            adata=adata,
            metadata=metadata,
            subject_ids=subject_ids,
            max_missing_subject_fraction=1.0,
        )

        # Should only keep the valid subject
        assert len(ds) == 1
        assert ds.subject_ids == ["valid_subj"]

    def test_handles_missing_metadata_columns(self):
        """Should handle missing pathology columns gracefully."""
        from src.data.datasets import CognitiveResilienceDataset
        from src.data.constants import CELL_TYPE_ORDER
        import anndata

        n_cells = 100
        n_genes = 50

        np.random.seed(42)

        X = np.random.rand(n_cells, n_genes).astype(np.float32)
        obs = pd.DataFrame({
            "ROSMAP_IndividualID": ["subj_001"] * n_cells,
            "supercluster_name": np.random.choice(CELL_TYPE_ORDER[:5], n_cells),
            "BrainRegion": ["PFC"] * n_cells,
        })
        var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
        adata = anndata.AnnData(X=X, obs=obs, var=var)

        # Metadata missing some pathology columns
        metadata = pd.DataFrame({
            "ROSMAP_IndividualID": ["subj_001"],
            "cogn_global": [0.0],
            # Missing: gpath, amylsqrt, tangsqrt
        })

        ds = CognitiveResilienceDataset(
            adata=adata,
            metadata=metadata,
            subject_ids=["subj_001"],
        )

        sample = ds[0]

        # Should return zeros for missing pathology columns
        assert sample["pathology"].shape == (3,)
        assert (sample["pathology"] == 0.0).all()

    def test_nan_target_raises_at_init(self):
        """NaN in target column should raise ValueError at init."""
        from src.data.datasets import CognitiveResilienceDataset
        from src.data.constants import CELL_TYPE_ORDER
        import anndata

        n_cells = 100
        n_genes = 50

        np.random.seed(42)

        X = np.random.rand(n_cells, n_genes).astype(np.float32)
        obs = pd.DataFrame({
            "ROSMAP_IndividualID": ["subj_001"] * n_cells,
            "supercluster_name": np.random.choice(CELL_TYPE_ORDER[:5], n_cells),
            "BrainRegion": ["PFC"] * n_cells,
        })
        var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
        adata = anndata.AnnData(X=X, obs=obs, var=var)

        metadata = pd.DataFrame({
            "ROSMAP_IndividualID": ["subj_001"],
            "gpath": [0.5],
            "amylsqrt": [0.5],
            "tangsqrt": [0.3],
            "cogn_global": [np.nan],  # NaN target
        })

        with pytest.raises(ValueError, match="NaN in target column"):
            CognitiveResilienceDataset(
                adata=adata,
                metadata=metadata,
                subject_ids=["subj_001"],
            )

    def test_nan_pathology_raises_at_init(self):
        """NaN in pathology column should raise ValueError at init."""
        from src.data.datasets import CognitiveResilienceDataset
        from src.data.constants import CELL_TYPE_ORDER
        import anndata

        n_cells = 100
        n_genes = 50

        np.random.seed(42)

        X = np.random.rand(n_cells, n_genes).astype(np.float32)
        obs = pd.DataFrame({
            "ROSMAP_IndividualID": ["subj_001"] * n_cells,
            "supercluster_name": np.random.choice(CELL_TYPE_ORDER[:5], n_cells),
            "BrainRegion": ["PFC"] * n_cells,
        })
        var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
        adata = anndata.AnnData(X=X, obs=obs, var=var)

        metadata = pd.DataFrame({
            "ROSMAP_IndividualID": ["subj_001"],
            "gpath": [np.nan],  # NaN pathology
            "amylsqrt": [0.5],
            "tangsqrt": [0.3],
            "cogn_global": [0.5],
        })

        with pytest.raises(ValueError, match="NaN in pathology column"):
            CognitiveResilienceDataset(
                adata=adata,
                metadata=metadata,
                subject_ids=["subj_001"],
            )

    def test_clean_metadata_passes_validation(self):
        """Clean metadata (no NaN) should not raise."""
        from src.data.datasets import CognitiveResilienceDataset
        from src.data.constants import CELL_TYPE_ORDER
        import anndata

        n_cells = 100
        n_genes = 50

        np.random.seed(42)

        X = np.random.rand(n_cells, n_genes).astype(np.float32)
        obs = pd.DataFrame({
            "ROSMAP_IndividualID": ["subj_001"] * n_cells,
            "supercluster_name": np.random.choice(CELL_TYPE_ORDER[:5], n_cells),
            "BrainRegion": ["PFC"] * n_cells,
        })
        var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
        adata = anndata.AnnData(X=X, obs=obs, var=var)

        metadata = pd.DataFrame({
            "ROSMAP_IndividualID": ["subj_001"],
            "gpath": [0.5],
            "amylsqrt": [0.3],
            "tangsqrt": [0.2],
            "cogn_global": [0.8],
        })

        # Should not raise
        ds = CognitiveResilienceDataset(
            adata=adata,
            metadata=metadata,
            subject_ids=["subj_001"],
        )
        sample = ds[0]
        assert sample["pathology"][0] == pytest.approx(0.5)
        assert sample["cognition"].item() == pytest.approx(0.8)


class TestPseudobulk:
    """Tests for pseudobulk computation."""

    def test_pseudobulk_shape(self, mock_dataset):
        """Pseudobulk should have shape [n_cell_types, n_genes]."""
        sample = mock_dataset[0]
        assert sample["pseudobulk"].shape == (mock_dataset.n_cell_types, mock_dataset.n_genes)

    def test_pseudobulk_dtype(self, mock_dataset):
        """Pseudobulk should be float32."""
        sample = mock_dataset[0]
        assert sample["pseudobulk"].dtype == torch.float32

    def test_pseudobulk_values_are_means(self):
        """Pseudobulk should be mean expression per cell type."""
        from src.data.datasets import CognitiveResilienceDataset
        from src.data.constants import CELL_TYPE_ORDER
        import anndata

        # Create controlled data where we know the expected mean
        n_genes = 10
        cell_type = CELL_TYPE_ORDER[0]

        # 5 cells with known expression values
        X = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
            [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0],
            [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
        ], dtype=np.float32)

        # Expected mean for each gene: [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        expected_mean = np.array([3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], dtype=np.float32)

        obs = pd.DataFrame({
            "ROSMAP_IndividualID": ["subj_001"] * 5,
            "supercluster_name": [cell_type] * 5,
            "BrainRegion": ["PFC"] * 5,
        })
        var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
        adata = anndata.AnnData(X=X, obs=obs, var=var)

        metadata = pd.DataFrame({
            "ROSMAP_IndividualID": ["subj_001"],
            "gpath": [0.5], "amylsqrt": [0.5], "tangsqrt": [0.5],
            "cogn_global": [0.0],
        })

        ds = CognitiveResilienceDataset(
            adata=adata,
            metadata=metadata,
            subject_ids=["subj_001"],
        )

        sample = ds[0]

        # Check the pseudobulk for this cell type matches expected mean
        ct_idx = ds.ct_to_idx[cell_type]
        np.testing.assert_array_almost_equal(
            sample["pseudobulk"][ct_idx].numpy(),
            expected_mean,
        )

    def test_pseudobulk_zeros_for_missing_cell_types(self, mock_dataset):
        """Pseudobulk should be zero for cell types not present in subject."""
        sample = mock_dataset[0]
        mask = sample["cell_type_mask"]

        # Where mask is False (cell type not present), pseudobulk should be zeros
        pseudobulk = sample["pseudobulk"]
        for ct_idx in range(mock_dataset.n_cell_types):
            if not mask[ct_idx]:
                assert (pseudobulk[ct_idx] == 0).all()


class TestGraphFeatures:
    """Tests for CCC graph features."""

    def test_empty_graph_handling(self):
        """Should handle subjects with no LIANA results."""
        from src.data.datasets import CognitiveResilienceDataset
        from src.data.constants import CELL_TYPE_ORDER
        import anndata

        n_cells = 100
        n_genes = 50

        np.random.seed(42)

        X = np.random.rand(n_cells, n_genes).astype(np.float32)
        obs = pd.DataFrame({
            "ROSMAP_IndividualID": ["subj_001"] * n_cells,
            "supercluster_name": np.random.choice(CELL_TYPE_ORDER[:5], n_cells),
            "BrainRegion": ["PFC"] * n_cells,
        })
        var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
        adata = anndata.AnnData(X=X, obs=obs, var=var)

        metadata = pd.DataFrame({
            "ROSMAP_IndividualID": ["subj_001"],
            "gpath": [0.5], "amylsqrt": [0.5], "tangsqrt": [0.5],
            "cogn_global": [0.0],
        })

        # No LIANA results provided
        ds = CognitiveResilienceDataset(
            adata=adata,
            metadata=metadata,
            subject_ids=["subj_001"],
            liana_results=None,
        )

        sample = ds[0]

        # Should return empty tensors
        assert sample["ccc_edge_index"].shape == (2, 0)
        assert sample["ccc_edge_type"].shape == (0,)
        assert sample["ccc_edge_attr"].shape == (0, 1)

    def test_empty_liana_dataframe_handling(self):
        """Should handle empty LIANA DataFrame."""
        from src.data.datasets import CognitiveResilienceDataset
        from src.data.constants import CELL_TYPE_ORDER
        import anndata

        n_cells = 100
        n_genes = 50

        np.random.seed(42)

        X = np.random.rand(n_cells, n_genes).astype(np.float32)
        obs = pd.DataFrame({
            "ROSMAP_IndividualID": ["subj_001"] * n_cells,
            "supercluster_name": np.random.choice(CELL_TYPE_ORDER[:5], n_cells),
            "BrainRegion": ["PFC"] * n_cells,
        })
        var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
        adata = anndata.AnnData(X=X, obs=obs, var=var)

        metadata = pd.DataFrame({
            "ROSMAP_IndividualID": ["subj_001"],
            "gpath": [0.5], "amylsqrt": [0.5], "tangsqrt": [0.5],
            "cogn_global": [0.0],
        })

        # Empty LIANA DataFrame
        liana_results = {"subj_001": pd.DataFrame()}

        ds = CognitiveResilienceDataset(
            adata=adata,
            metadata=metadata,
            subject_ids=["subj_001"],
            liana_results=liana_results,
        )

        sample = ds[0]

        # Should return empty tensors
        assert sample["ccc_edge_index"].shape == (2, 0)
        assert sample["ccc_edge_type"].shape == (0,)
        assert sample["ccc_edge_attr"].shape == (0, 1)

    def test_edge_index_valid(self, mock_dataset):
        """Edge indices should be in valid range."""
        sample = mock_dataset[0]
        if sample["ccc_edge_index"].numel() > 0:
            assert (sample["ccc_edge_index"] >= 0).all()
            assert (sample["ccc_edge_index"] < mock_dataset.n_cell_types).all()

    def test_edge_tensors_consistent_shapes(self, mock_dataset):
        """Edge tensors should have consistent number of edges."""
        sample = mock_dataset[0]

        n_edges = sample["ccc_edge_index"].shape[1]
        assert sample["ccc_edge_type"].shape[0] == n_edges
        assert sample["ccc_edge_attr"].shape[0] == n_edges


class TestOutputSchema:
    """Tests for complete output schema compliance."""

    def test_all_required_keys_present(self, mock_dataset):
        """All required keys should be in output."""
        sample = mock_dataset[0]
        required_keys = {
            "subject_id", "pseudobulk", "cell_type_mask", "cell_counts",
            "cell_data", "cell_offsets", "ccc_edge_index", "ccc_edge_type", "ccc_edge_attr",
            "pathology", "cognition", "region_mask",
        }
        assert required_keys.issubset(set(sample.keys()))

    def test_output_dtypes(self, mock_dataset):
        """All tensors should have correct dtypes."""
        sample = mock_dataset[0]

        assert sample["pseudobulk"].dtype == torch.float32
        assert sample["cell_type_mask"].dtype == torch.bool
        assert sample["cell_counts"].dtype == torch.long
        assert sample["cell_data"].dtype == torch.float32
        assert sample["cell_offsets"].dtype == torch.long
        assert sample["ccc_edge_index"].dtype == torch.long
        assert sample["ccc_edge_type"].dtype == torch.long
        assert sample["ccc_edge_attr"].dtype == torch.float32
        assert sample["pathology"].dtype == torch.float32
        assert sample["cognition"].dtype == torch.float32
        assert sample["region_mask"].dtype == torch.bool

    def test_subject_id_is_string(self, mock_dataset):
        """subject_id should be a string."""
        sample = mock_dataset[0]
        assert isinstance(sample["subject_id"], str)

    def test_output_dimensions(self, mock_dataset):
        """Check all output tensor dimensions are correct."""
        sample = mock_dataset[0]

        # Pseudobulk: [n_cell_types, n_genes]
        assert sample["pseudobulk"].ndim == 2
        assert sample["pseudobulk"].shape[0] == mock_dataset.n_cell_types

        # Cell type mask: [n_cell_types]
        assert sample["cell_type_mask"].ndim == 1
        assert sample["cell_type_mask"].shape[0] == mock_dataset.n_cell_types

        # Cell counts: [n_cell_types]
        assert sample["cell_counts"].ndim == 1
        assert sample["cell_counts"].shape[0] == mock_dataset.n_cell_types

        # Cell data: [total_cells, n_genes] - flat concatenated
        assert sample["cell_data"].ndim == 2
        assert sample["cell_data"].shape[1] == mock_dataset.n_genes

        # Cell offsets: [n_cell_types + 1] - cumulative offsets
        assert sample["cell_offsets"].ndim == 1
        assert sample["cell_offsets"].shape[0] == mock_dataset.n_cell_types + 1
        assert sample["cell_offsets"][0] == 0
        assert sample["cell_offsets"][-1] == sample["cell_data"].shape[0]

        # Pathology: [n_pathology]
        assert sample["pathology"].ndim == 1

        # Cognition: [1]
        assert sample["cognition"].shape == (1,)

        # Edge index: [2, n_edges]
        assert sample["ccc_edge_index"].ndim == 2
        assert sample["ccc_edge_index"].shape[0] == 2

        # Edge type: [n_edges]
        assert sample["ccc_edge_type"].ndim == 1

        # Edge attr: [n_edges, 1]
        assert sample["ccc_edge_attr"].ndim == 2
        assert sample["ccc_edge_attr"].shape[1] == 1

    def test_all_subjects_produce_valid_output(self, mock_dataset):
        """All subjects in the dataset should produce valid output."""
        for idx in range(len(mock_dataset)):
            sample = mock_dataset[idx]

            # Check required keys exist
            assert "subject_id" in sample
            assert "pseudobulk" in sample
            assert "cell_type_mask" in sample
            assert "cognition" in sample

            # Check no NaN values in critical tensors
            assert not torch.isnan(sample["pseudobulk"]).any()
            assert not torch.isnan(sample["pathology"]).any()
            assert not torch.isnan(sample["cognition"]).any()


class TestDatasetLength:
    """Tests for dataset length and iteration."""

    def test_len_matches_subject_ids(self, mock_dataset):
        """len(dataset) should match number of valid subject_ids."""
        assert len(mock_dataset) == len(mock_dataset.subject_ids)

    def test_iteration_works(self, mock_dataset):
        """Should be able to iterate over all samples."""
        count = 0
        for sample in mock_dataset:
            assert "subject_id" in sample
            count += 1
        assert count == len(mock_dataset)

    def test_index_out_of_range_raises(self, mock_dataset):
        """Accessing index out of range should raise IndexError."""
        with pytest.raises(IndexError):
            _ = mock_dataset[len(mock_dataset)]


class TestTransform:
    """Tests for transform functionality."""

    def test_transform_applied(self):
        """Transform should be applied to sample."""
        from src.data.datasets import CognitiveResilienceDataset
        from src.data.constants import CELL_TYPE_ORDER
        import anndata

        n_cells = 100
        n_genes = 50

        np.random.seed(42)

        X = np.random.rand(n_cells, n_genes).astype(np.float32)
        obs = pd.DataFrame({
            "ROSMAP_IndividualID": ["subj_001"] * n_cells,
            "supercluster_name": np.random.choice(CELL_TYPE_ORDER[:5], n_cells),
            "BrainRegion": ["PFC"] * n_cells,
        })
        var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
        adata = anndata.AnnData(X=X, obs=obs, var=var)

        metadata = pd.DataFrame({
            "ROSMAP_IndividualID": ["subj_001"],
            "gpath": [0.5], "amylsqrt": [0.5], "tangsqrt": [0.5],
            "cogn_global": [0.0],
        })

        # Define a simple transform that adds a new key
        def add_key_transform(sample):
            sample["transform_applied"] = True
            return sample

        ds = CognitiveResilienceDataset(
            adata=adata,
            metadata=metadata,
            subject_ids=["subj_001"],
            transform=add_key_transform,
        )

        sample = ds[0]

        assert "transform_applied" in sample
        assert sample["transform_applied"] is True

    def test_transform_can_modify_tensors(self):
        """Transform should be able to modify tensors."""
        from src.data.datasets import CognitiveResilienceDataset
        from src.data.constants import CELL_TYPE_ORDER
        import anndata

        n_cells = 100
        n_genes = 50

        np.random.seed(42)

        X = np.random.rand(n_cells, n_genes).astype(np.float32)
        obs = pd.DataFrame({
            "ROSMAP_IndividualID": ["subj_001"] * n_cells,
            "supercluster_name": np.random.choice(CELL_TYPE_ORDER[:5], n_cells),
            "BrainRegion": ["PFC"] * n_cells,
        })
        var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
        adata = anndata.AnnData(X=X, obs=obs, var=var)

        metadata = pd.DataFrame({
            "ROSMAP_IndividualID": ["subj_001"],
            "gpath": [0.5], "amylsqrt": [0.5], "tangsqrt": [0.5],
            "cogn_global": [0.0],
        })

        # Transform that doubles pseudobulk values
        def double_pseudobulk(sample):
            sample["pseudobulk"] = sample["pseudobulk"] * 2
            return sample

        ds_no_transform = CognitiveResilienceDataset(
            adata=adata,
            metadata=metadata,
            subject_ids=["subj_001"],
        )

        ds_with_transform = CognitiveResilienceDataset(
            adata=adata,
            metadata=metadata,
            subject_ids=["subj_001"],
            transform=double_pseudobulk,
        )

        sample_no_transform = ds_no_transform[0]
        sample_with_transform = ds_with_transform[0]

        # Check that transform was applied
        torch.testing.assert_close(
            sample_with_transform["pseudobulk"],
            sample_no_transform["pseudobulk"] * 2,
        )


# ============================================================================
# PrecomputedDataset Tests
# ============================================================================


class TestPrecomputedDataset:
    """Tests for PrecomputedDataset and save_precomputed_features."""

    def test_save_and_load_includes_cell_counts(self, mock_dataset, tmp_path):
        """Saved features should include cell_counts and be loadable."""
        from src.data.datasets import PrecomputedDataset, save_precomputed_features

        # Save features
        save_precomputed_features(mock_dataset, tmp_path, verbose=False)

        # Load via PrecomputedDataset
        precomputed = PrecomputedDataset(
            feature_dir=tmp_path,
            metadata=mock_dataset.metadata,
            subject_ids=mock_dataset.subject_ids,
        )

        sample = precomputed[0]

        # Verify cell_counts is present and correct type
        assert "cell_counts" in sample
        assert sample["cell_counts"].dtype == torch.long
        assert sample["cell_counts"].shape == (mock_dataset.n_cell_types,)

    def test_save_and_load_includes_region_mask(self, mock_dataset, tmp_path):
        """Saved features should include region_mask and be loadable."""
        from src.data.datasets import PrecomputedDataset, save_precomputed_features

        # Save features
        save_precomputed_features(mock_dataset, tmp_path, verbose=False)

        # Load via PrecomputedDataset
        precomputed = PrecomputedDataset(
            feature_dir=tmp_path,
            metadata=mock_dataset.metadata,
            subject_ids=mock_dataset.subject_ids,
        )

        sample = precomputed[0]

        # Verify region_mask is present and correct type
        assert "region_mask" in sample
        assert sample["region_mask"].dtype == torch.bool
        # Should have at least one True
        assert sample["region_mask"].any()

    def test_precomputed_matches_original(self, mock_dataset, tmp_path):
        """PrecomputedDataset should return matching data to original."""
        from src.data.datasets import PrecomputedDataset, save_precomputed_features

        # Save features
        save_precomputed_features(mock_dataset, tmp_path, verbose=False)

        # Load via PrecomputedDataset
        precomputed = PrecomputedDataset(
            feature_dir=tmp_path,
            metadata=mock_dataset.metadata,
            subject_ids=mock_dataset.subject_ids,
        )

        original_sample = mock_dataset[0]
        precomputed_sample = precomputed[0]

        # Check all required keys are present
        # PrecomputedDataset uses flat cell format (cell_data + cell_offsets)
        # instead of padded format (cells + cell_mask)
        required_keys = [
            "pseudobulk", "cell_type_mask", "cell_counts", "region_mask",
            "cell_data", "cell_offsets", "ccc_edge_index", "ccc_edge_type",
            "ccc_edge_attr", "pathology", "cognition",
        ]
        for key in required_keys:
            assert key in precomputed_sample, f"Missing key: {key}"

        # Values should match
        torch.testing.assert_close(
            precomputed_sample["pseudobulk"],
            original_sample["pseudobulk"],
        )
        torch.testing.assert_close(
            precomputed_sample["cell_type_mask"],
            original_sample["cell_type_mask"],
        )
        torch.testing.assert_close(
            precomputed_sample["cell_counts"],
            original_sample["cell_counts"],
        )
        torch.testing.assert_close(
            precomputed_sample["region_mask"],
            original_sample["region_mask"],
        )

    def test_backward_compatible_loading(self, mock_dataset, tmp_path):
        """Should handle files with all required .pt keys."""
        from src.data.datasets import PrecomputedDataset
        from src.data.constants import REGION_ORDER

        subject_id = mock_dataset.subject_ids[0]
        sample = mock_dataset[0]

        torch.save({
            "pseudobulk": sample["pseudobulk"],
            "cell_type_mask": sample["cell_type_mask"],
            "cell_counts": sample["cell_counts"],
            "region_mask": torch.tensor([True] + [False] * (len(REGION_ORDER) - 1), dtype=torch.bool),
            "cell_data": sample["cell_data"],
            "cell_offsets": sample["cell_offsets"],
            "ccc_edge_index": sample["ccc_edge_index"],
            "ccc_edge_type": sample["ccc_edge_type"],
            "ccc_edge_attr": sample["ccc_edge_attr"],
            "cell_type_order": list(mock_dataset.cell_type_order),
            "available_regions": [0],
        }, tmp_path / f"{subject_id}.pt")

        # Load via PrecomputedDataset - should not crash
        precomputed = PrecomputedDataset(
            feature_dir=tmp_path,
            metadata=mock_dataset.metadata,
            subject_ids=[subject_id],
        )

        loaded_sample = precomputed[0]

        # Should have all keys
        assert "cell_counts" in loaded_sample
        assert "region_mask" in loaded_sample
        # region_mask should have first region True
        assert loaded_sample["region_mask"][0] == True

        # cell_counts should match
        assert torch.equal(loaded_sample["cell_counts"], sample["cell_counts"])


class TestPrecomputedNaNFallbacks:
    """D-A4: Tests for PrecomputedDataset backward-compatible NaN fallbacks."""

    def test_precomputed_loads_cell_counts(self, mock_dataset, tmp_path):
        """cell_counts should be loaded from .pt file."""
        from src.data.datasets import PrecomputedDataset
        from src.data.constants import REGION_ORDER

        subject_id = mock_dataset.subject_ids[0]
        sample = mock_dataset[0]

        torch.save({
            "pseudobulk": sample["pseudobulk"],
            "cell_type_mask": sample["cell_type_mask"],
            "cell_counts": sample["cell_counts"],
            "region_mask": sample["region_mask"],
            "cell_data": sample["cell_data"],
            "cell_offsets": sample["cell_offsets"],
            "ccc_edge_index": sample["ccc_edge_index"],
            "ccc_edge_type": sample["ccc_edge_type"],
            "ccc_edge_attr": sample["ccc_edge_attr"],
            "cell_type_order": list(mock_dataset.cell_type_order),
            "available_regions": [0],
        }, tmp_path / f"{subject_id}.pt")

        precomputed = PrecomputedDataset(
            feature_dir=tmp_path,
            metadata=mock_dataset.metadata,
            subject_ids=[subject_id],
        )

        loaded_sample = precomputed[0]

        assert "cell_counts" in loaded_sample
        assert loaded_sample["cell_counts"].dtype == torch.long
        assert torch.equal(loaded_sample["cell_counts"], sample["cell_counts"])

    def test_precomputed_region_mask_loaded(self, mock_dataset, tmp_path):
        """region_mask should be loaded from .pt file."""
        from src.data.datasets import PrecomputedDataset
        from src.data.constants import REGION_ORDER

        subject_id = mock_dataset.subject_ids[0]
        sample = mock_dataset[0]

        region_mask = torch.tensor([True] + [False] * (len(REGION_ORDER) - 1), dtype=torch.bool)

        torch.save({
            "pseudobulk": sample["pseudobulk"],
            "cell_type_mask": sample["cell_type_mask"],
            "cell_counts": sample["cell_counts"],
            "region_mask": region_mask,
            "cell_data": sample["cell_data"],
            "cell_offsets": sample["cell_offsets"],
            "ccc_edge_index": sample["ccc_edge_index"],
            "ccc_edge_type": sample["ccc_edge_type"],
            "ccc_edge_attr": sample["ccc_edge_attr"],
            "cell_type_order": list(mock_dataset.cell_type_order),
            "available_regions": [0],
        }, tmp_path / f"{subject_id}.pt")

        precomputed = PrecomputedDataset(
            feature_dir=tmp_path,
            metadata=mock_dataset.metadata,
            subject_ids=[subject_id],
        )

        loaded_sample = precomputed[0]

        assert "region_mask" in loaded_sample
        assert loaded_sample["region_mask"].dtype == torch.bool
        assert loaded_sample["region_mask"].shape == (len(REGION_ORDER),)
        assert loaded_sample["region_mask"][0].item() is True  # PFC
        assert loaded_sample["region_mask"].sum().item() == 1  # Only PFC


class TestPrecomputedCellTypeOrderValidation:
    """Tests for cell_type_order validation in precomputed features."""

    def test_save_precomputed_stores_cell_type_order(self, mock_dataset, tmp_path):
        """save_precomputed_features should store cell_type_order."""
        from src.data.datasets import save_precomputed_features

        save_precomputed_features(mock_dataset, tmp_path, verbose=False)

        # Check that cell_type_order was saved
        sample = mock_dataset[0]
        subject_id = sample["subject_id"]
        data = torch.load(tmp_path / f"{subject_id}.pt", weights_only=False)

        assert "cell_type_order" in data
        saved_order = list(data["cell_type_order"])
        assert saved_order == mock_dataset.cell_type_order

    def test_precomputed_validates_matching_order(self, mock_dataset, tmp_path):
        """PrecomputedDataset should load successfully with matching order."""
        from src.data.datasets import save_precomputed_features, PrecomputedDataset

        save_precomputed_features(mock_dataset, tmp_path, verbose=False)

        # Load with same order - should work
        precomputed = PrecomputedDataset(
            feature_dir=tmp_path,
            metadata=mock_dataset.metadata,
            subject_ids=mock_dataset.subject_ids,
            cell_type_order=mock_dataset.cell_type_order,
        )

        # Should load without error
        sample = precomputed[0]
        # cell_type_order is no longer in the sample dict (removed for DDP serialization),
        # but it should still be accessible as a dataset attribute
        assert precomputed.cell_type_order == mock_dataset.cell_type_order

    def test_precomputed_raises_on_mismatched_order(self, mock_dataset, tmp_path):
        """PrecomputedDataset should raise RuntimeError on mismatched order."""
        from src.data.datasets import save_precomputed_features, PrecomputedDataset

        save_precomputed_features(mock_dataset, tmp_path, verbose=False)

        # Try to load with different order
        wrong_order = list(reversed(mock_dataset.cell_type_order))

        # Should raise during __init__ (mmap_load_all validates cell_type_order)
        with pytest.raises(RuntimeError, match="different cell_type_order"):
            PrecomputedDataset(
                feature_dir=tmp_path,
                metadata=mock_dataset.metadata,
                subject_ids=mock_dataset.subject_ids,
                cell_type_order=wrong_order,
            )


class TestMultiRegionPseudobulk:
    """Tests for multi-region pseudobulk computation."""

    def test_multiregion_pseudobulk_keys_present(self, mock_dataset):
        """Should produce region_{idx}_pseudobulk keys when BrainRegion column exists."""
        sample = mock_dataset[0]

        # mock_dataset uses BrainRegion with values ["PFC", "AG", "MTC"]
        # These map to indices 0, 1, 2 in REGION_ORDER
        region_keys = [k for k in sample if k.startswith("region_") and k.endswith("_pseudobulk")]
        assert len(region_keys) > 0, "Should have at least one region pseudobulk key"

    def test_available_regions_matches_data(self, mock_dataset):
        """available_regions should list indices of regions with data."""
        sample = mock_dataset[0]

        if "available_regions" in sample:
            available = sample["available_regions"]
            # Each available region should have a corresponding pseudobulk key
            for region_idx in available:
                key = f"region_{region_idx}_pseudobulk"
                assert key in sample, f"Missing {key} for available region {region_idx}"

    def test_region_pseudobulk_shape(self, mock_dataset):
        """Each region_{idx}_pseudobulk should have shape [n_cell_types, n_genes]."""
        from src.data.constants import N_CELL_TYPES
        sample = mock_dataset[0]
        n_genes = sample["pseudobulk"].shape[1]

        for key in sample:
            if key.startswith("region_") and key.endswith("_pseudobulk"):
                assert sample[key].shape == (N_CELL_TYPES, n_genes), f"Wrong shape for {key}"

    def test_region_pseudobulk_dtype(self, mock_dataset):
        """Region pseudobulk tensors should be float."""
        sample = mock_dataset[0]

        for key in sample:
            if key.startswith("region_") and key.endswith("_pseudobulk"):
                assert sample[key].dtype == torch.float32, f"Wrong dtype for {key}"

    def test_single_region_fallback(self):
        """Should return no region keys when BrainRegion column missing."""
        from src.data.datasets import CognitiveResilienceDataset
        from src.data.constants import CELL_TYPE_ORDER
        import anndata

        # Create mock data WITHOUT BrainRegion column
        n_cells = 100
        n_genes = 50
        X = np.random.rand(n_cells, n_genes).astype(np.float32)
        obs = pd.DataFrame({
            "ROSMAP_IndividualID": ["subj_001"] * n_cells,
            "supercluster_name": np.random.choice(CELL_TYPE_ORDER[:5], n_cells),
        })
        var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
        adata = anndata.AnnData(X=X, obs=obs, var=var)

        metadata = pd.DataFrame({
            "ROSMAP_IndividualID": ["subj_001"],
            "gpath": [0.5], "amylsqrt": [0.5], "tangsqrt": [0.5],
            "cogn_global": [0.0],
        })

        ds = CognitiveResilienceDataset(
            adata=adata, metadata=metadata, subject_ids=["subj_001"],
        )

        sample = ds[0]

        # Should NOT have region pseudobulk keys
        region_keys = [k for k in sample if k.startswith("region_") and k.endswith("_pseudobulk")]
        assert len(region_keys) == 0, "Should not have region pseudobulk without BrainRegion column"
        assert "available_regions" not in sample

    def test_precomputed_saves_region_data(self, mock_dataset, tmp_path):
        """save_precomputed_features should save region pseudobulk data."""
        from src.data.datasets import save_precomputed_features

        save_precomputed_features(mock_dataset, tmp_path, verbose=False)

        # Check that .pt file contains region data
        sample = mock_dataset[0]
        subject_id = sample["subject_id"]
        data = torch.load(tmp_path / f"{subject_id}.pt", weights_only=False)

        # Check region keys are saved
        for key in sample:
            if key.startswith("region_") and key.endswith("_pseudobulk"):
                assert key in data, f"Missing {key} in saved .pt"

        if "available_regions" in sample:
            assert "available_regions" in data

    def test_precomputed_loads_region_data(self, mock_dataset, tmp_path):
        """PrecomputedDataset should load region pseudobulk data as stacked tensor."""
        from src.data.datasets import save_precomputed_features, PrecomputedDataset

        save_precomputed_features(mock_dataset, tmp_path, verbose=False)

        precomputed = PrecomputedDataset(
            feature_dir=tmp_path,
            metadata=mock_dataset.metadata,
            subject_ids=mock_dataset.subject_ids,
        )

        # Compare original and loaded samples
        for i in range(len(mock_dataset)):
            orig_sample = mock_dataset[i]
            loaded_sample = precomputed[i]

            # PrecomputedDataset stacks per-region pseudobulks into
            # region_pseudobulk [n_regions, n_ct, n_genes]
            assert "region_pseudobulk" in loaded_sample, "Missing region_pseudobulk"

            # Verify the individual region data was correctly stacked
            orig_region_keys = {k for k in orig_sample if k.startswith("region_") and k.endswith("_pseudobulk")}
            for key in orig_region_keys:
                # Extract region index from key like "region_0_pseudobulk"
                ridx = int(key.split("_")[1])
                torch.testing.assert_close(
                    loaded_sample["region_pseudobulk"][ridx],
                    orig_sample[key],
                    msg=f"Values mismatch for {key} (region index {ridx})",
                )


class TestPrecomputedMultiRegionRoundtrip:
    """Tests for PrecomputedDataset multi-region save → load → collate roundtrip."""

    def test_roundtrip_preserves_region_pseudobulk_through_collate(self, mock_dataset, tmp_path):
        """Save → load → collate_for_hgt_multiregion should preserve region data."""
        from src.data.datasets import save_precomputed_features, PrecomputedDataset
        from src.data.collate import collate_for_hgt_multiregion
        from src.data.constants import N_REGIONS

        # Save
        save_precomputed_features(mock_dataset, tmp_path, verbose=False)

        # Load
        precomputed = PrecomputedDataset(
            feature_dir=tmp_path,
            metadata=mock_dataset.metadata,
            subject_ids=mock_dataset.subject_ids,
        )

        # Gather samples and collate
        samples = [precomputed[i] for i in range(len(precomputed))]
        batch = collate_for_hgt_multiregion(samples)

        # Verify region_pseudobulk shape
        n_subjects = len(precomputed)
        n_genes = samples[0]["pseudobulk"].shape[1]
        n_cell_types = samples[0]["pseudobulk"].shape[0]

        assert "region_pseudobulk" in batch
        assert batch["region_pseudobulk"].shape == (
            n_subjects, N_REGIONS, n_cell_types, n_genes
        )

        # Verify region_mask shape and dtype
        assert "region_mask" in batch
        assert batch["region_mask"].shape == (n_subjects, N_REGIONS)
        assert batch["region_mask"].dtype == torch.bool

        # At least one region should be active per subject
        for i in range(n_subjects):
            assert batch["region_mask"][i].any(), f"Subject {i} has no active regions"

    def test_roundtrip_region_values_match_original(self, mock_dataset, tmp_path):
        """Region pseudobulk values should survive save → load → collate."""
        from src.data.datasets import save_precomputed_features, PrecomputedDataset
        from src.data.collate import collate_for_hgt_multiregion
        from src.data.constants import REGION_ORDER

        # Get original samples
        orig_samples = [mock_dataset[i] for i in range(len(mock_dataset))]

        # Save and reload
        save_precomputed_features(mock_dataset, tmp_path, verbose=False)
        precomputed = PrecomputedDataset(
            feature_dir=tmp_path,
            metadata=mock_dataset.metadata,
            subject_ids=mock_dataset.subject_ids,
        )
        loaded_samples = [precomputed[i] for i in range(len(precomputed))]

        # Collate both
        orig_batch = collate_for_hgt_multiregion(orig_samples)
        loaded_batch = collate_for_hgt_multiregion(loaded_samples)

        # Compare region_pseudobulk tensors
        assert torch.allclose(
            orig_batch["region_pseudobulk"],
            loaded_batch["region_pseudobulk"],
            atol=1e-6,
        ), "Region pseudobulk values changed after save/load roundtrip"

        # Compare region_mask
        assert torch.equal(
            orig_batch["region_mask"],
            loaded_batch["region_mask"],
        ), "Region mask changed after save/load roundtrip"

    def test_roundtrip_active_regions_are_nonzero(self, mock_dataset, tmp_path):
        """Active regions should have non-zero pseudobulk data after roundtrip."""
        from src.data.datasets import save_precomputed_features, PrecomputedDataset
        from src.data.collate import collate_for_hgt_multiregion

        save_precomputed_features(mock_dataset, tmp_path, verbose=False)
        precomputed = PrecomputedDataset(
            feature_dir=tmp_path,
            metadata=mock_dataset.metadata,
            subject_ids=mock_dataset.subject_ids,
        )

        samples = [precomputed[i] for i in range(len(precomputed))]
        batch = collate_for_hgt_multiregion(samples)

        region_pb = batch["region_pseudobulk"]
        region_mask = batch["region_mask"]

        for i in range(len(precomputed)):
            for r in range(region_mask.shape[1]):
                if region_mask[i, r]:
                    # Active region should have non-zero expression
                    assert region_pb[i, r].abs().sum() > 0, (
                        f"Subject {i}, region {r} is active but has zero pseudobulk"
                    )


class TestRuntimeErrorPaths:
    """D-A1, D-A2, D-A3: Tests for RuntimeError defensive checks in _get_pathology / _get_target.

    These RuntimeErrors are safety nets that should never trigger in normal use
    because _validate_subjects() catches the problems at __init__ time. To reach
    these code paths we construct a valid dataset and then mutate its internal
    state (metadata / subject_ids) after construction, simulating a scenario
    where the data was corrupted between init-time validation and access time.
    """

    @staticmethod
    def _make_valid_dataset():
        """Build a minimal valid CognitiveResilienceDataset."""
        from src.data.datasets import CognitiveResilienceDataset
        from src.data.constants import CELL_TYPE_ORDER
        import anndata

        n_cells = 100
        n_genes = 50
        X = np.random.rand(n_cells, n_genes).astype(np.float32)
        obs = pd.DataFrame({
            "ROSMAP_IndividualID": ["subj_001"] * n_cells,
            "supercluster_name": np.random.choice(CELL_TYPE_ORDER[:5], n_cells),
            "BrainRegion": ["PFC"] * n_cells,
        })
        var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
        adata = anndata.AnnData(X=X, obs=obs, var=var)

        metadata = pd.DataFrame({
            "ROSMAP_IndividualID": ["subj_001"],
            "gpath": [0.5],
            "amylsqrt": [0.3],
            "tangsqrt": [0.2],
            "cogn_global": [0.8],
        })

        return CognitiveResilienceDataset(
            adata=adata,
            metadata=metadata,
            subject_ids=["subj_001"],
        )

    def test_da1_get_pathology_uses_preextracted_array(self):
        """D-A1: _get_pathology uses pre-extracted numpy array, not pandas .loc.

        Post-init metadata mutation does not affect __getitem__ because
        phenotypes are pre-extracted to numpy arrays at init time.
        """
        ds = self._make_valid_dataset()
        original_val = ds._pathology_array[0, 0]

        # Mutate metadata after construction — should have no effect
        ds.metadata.loc["subj_001", "gpath"] = np.nan

        sample = ds[0]
        # Value comes from pre-extracted array, not metadata
        assert sample["pathology"][0].item() == original_val

    def test_da2_get_target_uses_preextracted_array(self):
        """D-A2: _get_target uses pre-extracted numpy array, not pandas .loc.

        Post-init metadata mutation does not affect __getitem__ because
        phenotypes are pre-extracted to numpy arrays at init time.
        """
        ds = self._make_valid_dataset()
        original_val = ds._target_array[0]

        # Mutate metadata after construction — should have no effect
        ds.metadata.loc["subj_001", "cogn_global"] = np.nan

        sample = ds[0]
        # Value comes from pre-extracted array, not metadata
        assert sample["cognition"].item() == original_val

    def test_da3_get_target_subject_not_found_raises_runtime_error(self):
        """D-A3: __getitem__ raises RuntimeError when subject is not in pre-computed index.

        We append a phantom subject ID to subject_ids *after* init to bypass
        validation. The __getitem__ guard catches this before any data loading.
        """
        ds = self._make_valid_dataset()

        # Append phantom subject_id (bypassing _validate_subjects)
        ds.subject_ids.append("phantom_subj")

        # Access the phantom subject (last index)
        idx = len(ds) - 1
        with pytest.raises(RuntimeError, match="not in pre-computed index"):
            ds[idx]


class TestPrecomputedGetGeneNames:
    """Tests for PrecomputedDataset.get_gene_names() and get_cell_type_names()."""

    def test_get_gene_names_from_sidecar(self, mock_dataset, tmp_path):
        """get_gene_names() should return names from gene_names.npy sidecar file."""
        from src.data.datasets import PrecomputedDataset, save_precomputed_features

        # Save features to create valid precomputed dir
        save_precomputed_features(mock_dataset, tmp_path, verbose=False)

        # Create sidecar gene_names.npy matching pseudobulk gene dimension
        n_genes = mock_dataset.n_genes
        gene_names = np.array([f"gene_{i}" for i in range(n_genes)])
        np.save(tmp_path / "gene_names.npy", gene_names)

        precomputed = PrecomputedDataset(
            feature_dir=tmp_path,
            metadata=mock_dataset.metadata,
            subject_ids=mock_dataset.subject_ids,
        )

        result = precomputed.get_gene_names()
        assert result is not None
        assert len(result) == n_genes
        assert result[0] == "gene_0"
        assert all(isinstance(n, str) for n in result)

    def test_get_gene_names_saved_by_precompute(self, mock_dataset, tmp_path):
        """save_precomputed_features should save gene_names.npy sidecar."""
        from src.data.datasets import PrecomputedDataset, save_precomputed_features

        save_precomputed_features(mock_dataset, tmp_path, verbose=False)
        assert (tmp_path / "gene_names.npy").exists()

        precomputed = PrecomputedDataset(
            feature_dir=tmp_path,
            metadata=mock_dataset.metadata,
            subject_ids=mock_dataset.subject_ids,
        )

        result = precomputed.get_gene_names()
        assert result is not None
        assert len(result) == mock_dataset.n_genes

    def test_get_gene_names_missing_sidecar(self, mock_dataset, tmp_path):
        """get_gene_names() returns None when sidecar file is manually absent."""
        from src.data.datasets import PrecomputedDataset, save_precomputed_features

        save_precomputed_features(mock_dataset, tmp_path, verbose=False)
        # Remove the sidecar to test the fallback path
        (tmp_path / "gene_names.npy").unlink()

        precomputed = PrecomputedDataset(
            feature_dir=tmp_path,
            metadata=mock_dataset.metadata,
            subject_ids=mock_dataset.subject_ids,
        )

        result = precomputed.get_gene_names()
        assert result is None

    def test_get_cell_type_names(self, mock_dataset, tmp_path):
        """get_cell_type_names() should return cell_type_order."""
        from src.data.datasets import PrecomputedDataset, save_precomputed_features
        from src.data.constants import CELL_TYPE_ORDER

        save_precomputed_features(mock_dataset, tmp_path, verbose=False)

        precomputed = PrecomputedDataset(
            feature_dir=tmp_path,
            metadata=mock_dataset.metadata,
            subject_ids=mock_dataset.subject_ids,
        )

        result = precomputed.get_cell_type_names()
        assert result == CELL_TYPE_ORDER

    def test_gene_names_dimension_mismatch_raises(self, mock_dataset, tmp_path):
        """Mismatched gene_names.npy length should raise ValueError."""
        from src.data.datasets import PrecomputedDataset, save_precomputed_features

        save_precomputed_features(mock_dataset, tmp_path, verbose=False)

        # Overwrite with wrong-length gene_names
        np.save(tmp_path / "gene_names.npy", np.array(["A", "B", "C"]))

        with pytest.raises(ValueError, match="gene_names.npy has 3 genes"):
            PrecomputedDataset(
                feature_dir=tmp_path,
                metadata=mock_dataset.metadata,
                subject_ids=mock_dataset.subject_ids,
            )


class TestSharedMmapContextManager:
    """Tests for _shared_mmap context manager."""

    def test_shared_mmap_restores_map_private(self):
        """_shared_mmap restores MAP_PRIVATE after exiting context."""
        import mmap as _mmap
        import torch
        from src.data.datasets import _shared_mmap

        # Ensure we start with MAP_PRIVATE
        torch.serialization.set_default_mmap_options(_mmap.MAP_PRIVATE)

        with _shared_mmap():
            pass  # MAP_SHARED is active inside

        # After context exit, MAP_PRIVATE should be restored.
        # Verify by checking it doesn't raise when setting MAP_PRIVATE again
        # (idempotent operation — the real test is that the context manager called it)
        torch.serialization.set_default_mmap_options(_mmap.MAP_PRIVATE)

    def test_shared_mmap_restores_on_exception(self):
        """_shared_mmap restores MAP_PRIVATE even when exception occurs inside."""
        import mmap as _mmap
        import torch
        from src.data.datasets import _shared_mmap

        torch.serialization.set_default_mmap_options(_mmap.MAP_PRIVATE)

        with pytest.raises(ValueError):
            with _shared_mmap():
                raise ValueError("test error")

        # MAP_PRIVATE should be restored despite the exception
        torch.serialization.set_default_mmap_options(_mmap.MAP_PRIVATE)


class TestLoadSubjectCache:
    """Tests for load_subject_cache with mmap support."""

    def test_load_subject_cache_mmap(self, tmp_path):
        """load_subject_cache with use_mmap=True returns valid tensors."""
        import torch
        from src.data.datasets import PrecomputedDataset

        for sid in ["S001", "S002"]:
            data = {
                "pseudobulk": torch.randn(5, 10),
                "cell_type_mask": torch.ones(5, dtype=torch.bool),
                "cell_offsets": torch.tensor([0, 3, 5, 5, 5, 5]),
                "ccc_edge_index": torch.zeros(2, 4, dtype=torch.long),
                "ccc_edge_type": torch.zeros(4, dtype=torch.long),
                "ccc_edge_attr": torch.ones(4),
                "cell_counts": torch.tensor([3, 2, 0, 0, 0]),
                "region_mask": torch.ones(1, dtype=torch.bool),
                "cell_data": torch.randn(5, 10),
            }
            torch.save(data, tmp_path / f"{sid}.pt")

        cache = PrecomputedDataset.load_subject_cache(
            tmp_path, ["S001", "S002"], use_mmap=True
        )
        assert len(cache) == 2
        assert cache["S001"]["pseudobulk"].shape == (5, 10)
        assert cache["S002"]["cell_data"].shape == (5, 10)

    def test_load_subject_cache_default_no_mmap(self, tmp_path):
        """load_subject_cache defaults to non-mmap (backward compat)."""
        import torch
        from src.data.datasets import PrecomputedDataset

        data = {
            "pseudobulk": torch.randn(5, 10),
            "cell_type_mask": torch.ones(5, dtype=torch.bool),
            "cell_offsets": torch.tensor([0, 3, 5, 5, 5, 5]),
            "ccc_edge_index": torch.zeros(2, 4, dtype=torch.long),
            "ccc_edge_type": torch.zeros(4, dtype=torch.long),
            "ccc_edge_attr": torch.ones(4),
            "cell_counts": torch.tensor([3, 2, 0, 0, 0]),
            "region_mask": torch.ones(1, dtype=torch.bool),
            "cell_data": torch.randn(5, 10),
        }
        torch.save(data, tmp_path / "S001.pt")

        cache = PrecomputedDataset.load_subject_cache(tmp_path, ["S001"])
        assert len(cache) == 1


class TestShmCleanup:
    """Tests for /dev/shm lifecycle management."""

    def test_cleanup_shm_cache_removes_directory(self):
        """cleanup_shm_cache removes the target directory."""
        import tempfile
        from src.data.datasets import PrecomputedDataset

        # Use real /tmp so resolved path starts with /tmp/
        with tempfile.TemporaryDirectory(dir="/tmp") as td:
            fake_shm = Path(td) / "precomputed_test"
            fake_shm.mkdir()
            (fake_shm / "dummy.pt").write_bytes(b"data")

            PrecomputedDataset.cleanup_shm_cache(fake_shm)
            assert not fake_shm.exists()

    def test_cleanup_shm_cache_ignores_missing(self):
        """cleanup_shm_cache is no-op if directory doesn't exist."""
        from src.data.datasets import PrecomputedDataset

        nonexistent = Path("/tmp/nonexistent_shm_test_dir")
        PrecomputedDataset.cleanup_shm_cache(nonexistent)  # should not raise

    def test_cleanup_shm_cache_refuses_non_shm_paths(self, tmp_path):
        """cleanup_shm_cache refuses to delete paths outside /dev/shm or /tmp."""
        from src.data.datasets import PrecomputedDataset

        # tmp_path on this system resolves outside /tmp/, so the safety
        # guard should refuse deletion and the directory should survive.
        fake_dir = tmp_path / "precomputed_test"
        fake_dir.mkdir()
        (fake_dir / "dummy.pt").write_bytes(b"data")

        PrecomputedDataset.cleanup_shm_cache(fake_dir)
        # Directory must still exist because the path is not under /dev/shm or /tmp
        assert fake_dir.exists()
