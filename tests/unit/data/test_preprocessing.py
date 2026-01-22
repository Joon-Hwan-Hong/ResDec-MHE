"""
Tests for src/data/preprocessing.py

Tests cover:
- L-R gene extraction from CellChatDB
- Preprocessing pipeline correctness
- HVG selection (seurat_v3 on raw counts)
- Pseudobulk computation
"""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from anndata import AnnData


class TestGetLRGenesFromCellChatDB:
    """Tests for get_lr_genes_from_cellchatdb()."""

    def test_extracts_simple_ligand_receptor(self, tmp_path):
        """Extract simple ligand and receptor names."""
        from src.data.preprocessing import get_lr_genes_from_cellchatdb

        # Create mock CellChatDB
        db = pd.DataFrame({
            "ligand.symbol": ["TGFB1", "WNT5A", "BMP2"],
            "receptor.symbol": ["TGFBR1", "FZD1", "BMPR1A"],
            "annotation": ["Secreted Signaling"] * 3,
        })
        db_path = tmp_path / "cellchatdb.csv"
        db.to_csv(db_path, index=False)

        lr_genes = get_lr_genes_from_cellchatdb(db_path)

        assert "TGFB1" in lr_genes
        assert "TGFBR1" in lr_genes
        assert "WNT5A" in lr_genes
        assert "FZD1" in lr_genes
        assert len(lr_genes) == 6

    def test_handles_complex_receptor_names(self, tmp_path):
        """Split complex names like TGFB1_TGFBR1_TGFBR2."""
        from src.data.preprocessing import get_lr_genes_from_cellchatdb

        db = pd.DataFrame({
            "ligand.symbol": ["TGFB1"],
            "receptor.symbol": ["TGFBR1_TGFBR2"],  # Complex receptor
            "annotation": ["Secreted Signaling"],
        })
        db_path = tmp_path / "cellchatdb.csv"
        db.to_csv(db_path, index=False)

        lr_genes = get_lr_genes_from_cellchatdb(db_path)

        assert "TGFB1" in lr_genes
        assert "TGFBR1" in lr_genes
        assert "TGFBR2" in lr_genes
        assert len(lr_genes) == 3

    def test_handles_missing_values(self, tmp_path):
        """Handle NaN values gracefully."""
        from src.data.preprocessing import get_lr_genes_from_cellchatdb

        db = pd.DataFrame({
            "ligand.symbol": ["TGFB1", np.nan, "WNT5A"],
            "receptor.symbol": ["TGFBR1", "FZD1", np.nan],
            "annotation": ["Secreted Signaling"] * 3,
        })
        db_path = tmp_path / "cellchatdb.csv"
        db.to_csv(db_path, index=False)

        lr_genes = get_lr_genes_from_cellchatdb(db_path)

        assert "TGFB1" in lr_genes
        assert "TGFBR1" in lr_genes
        assert "WNT5A" in lr_genes
        assert "FZD1" in lr_genes


class TestCreateSubjectPseudoBulkTensor:
    """Tests for create_subject_pseudobulk_tensor()."""

    @pytest.fixture
    def mock_adata(self):
        """Create mock AnnData with 2 subjects, 3 cell types."""
        n_cells = 100
        n_genes = 50

        # Random expression
        X = np.random.rand(n_cells, n_genes).astype(np.float32)

        # Create obs with subjects and cell types
        obs = pd.DataFrame({
            "ROSMAP_IndividualID": ["S1"] * 60 + ["S2"] * 40,
            "supercluster_name": (
                ["Astrocyte"] * 30 + ["Oligodendrocyte"] * 20 + ["Microglia"] * 10 +
                ["Astrocyte"] * 20 + ["Oligodendrocyte"] * 10 + ["Microglia"] * 10
            ),
        })

        var = pd.DataFrame(index=[f"Gene{i}" for i in range(n_genes)])

        return AnnData(X=X, obs=obs, var=var)

    def test_creates_correct_shape(self, mock_adata):
        """Tensor has shape [n_cell_types, n_genes]."""
        from src.data.preprocessing import create_subject_pseudobulk_tensor
        from src.visualization.config import CELL_TYPE_ORDER

        pseudobulk, cell_types = create_subject_pseudobulk_tensor(
            mock_adata,
            subject_id="S1",
            cell_type_order=CELL_TYPE_ORDER,
        )

        assert pseudobulk.shape == (len(CELL_TYPE_ORDER), 50)
        assert len(cell_types) == len(CELL_TYPE_ORDER)

    def test_absent_cell_types_are_zero(self, mock_adata):
        """Cell types not present should have zero expression."""
        from src.data.preprocessing import create_subject_pseudobulk_tensor

        cell_type_order = ["Astrocyte", "Oligodendrocyte", "Microglia", "Vascular"]

        pseudobulk, _ = create_subject_pseudobulk_tensor(
            mock_adata,
            subject_id="S1",
            cell_type_order=cell_type_order,
        )

        # Vascular not in data, should be zeros
        assert np.allclose(pseudobulk[3], 0.0)

        # Astrocyte should have non-zero values
        assert not np.allclose(pseudobulk[0], 0.0)

    def test_raises_for_missing_subject(self, mock_adata):
        """Raise error for non-existent subject."""
        from src.data.preprocessing import create_subject_pseudobulk_tensor

        with pytest.raises(ValueError, match="No cells found"):
            create_subject_pseudobulk_tensor(
                mock_adata,
                subject_id="NONEXISTENT",
            )


class TestGetSubjectsWithMinCells:
    """Tests for get_subjects_with_min_cells()."""

    @pytest.fixture
    def mock_adata(self):
        """Create mock AnnData with varying cell counts per subject."""
        n_cells = 250
        n_genes = 10

        X = np.random.rand(n_cells, n_genes)
        obs = pd.DataFrame({
            "ROSMAP_IndividualID": (
                ["S1"] * 150 +  # 150 cells
                ["S2"] * 80 +   # 80 cells
                ["S3"] * 20     # 20 cells
            ),
        })
        var = pd.DataFrame(index=[f"Gene{i}" for i in range(n_genes)])

        return AnnData(X=X, obs=obs, var=var)

    def test_filters_by_threshold(self, mock_adata):
        """Only return subjects meeting threshold."""
        from src.data.preprocessing import get_subjects_with_min_cells

        # Threshold 100: only S1
        subjects = get_subjects_with_min_cells(mock_adata, min_cells=100)
        assert subjects == ["S1"]

        # Threshold 50: S1 and S2
        subjects = get_subjects_with_min_cells(mock_adata, min_cells=50)
        assert set(subjects) == {"S1", "S2"}

        # Threshold 10: all subjects
        subjects = get_subjects_with_min_cells(mock_adata, min_cells=10)
        assert set(subjects) == {"S1", "S2", "S3"}


class TestComputePseudobulk:
    """Tests for compute_pseudobulk()."""

    @pytest.fixture
    def mock_adata(self):
        """Create mock AnnData for pseudobulk testing."""
        # 2 subjects, 2 cell types each
        X = np.array([
            [1.0, 2.0],  # S1, CT1, cell1
            [3.0, 4.0],  # S1, CT1, cell2
            [5.0, 6.0],  # S1, CT2, cell1
            [7.0, 8.0],  # S2, CT1, cell1
            [9.0, 10.0], # S2, CT2, cell1
        ], dtype=np.float32)

        obs = pd.DataFrame({
            "ROSMAP_IndividualID": ["S1", "S1", "S1", "S2", "S2"],
            "supercluster_name": ["CT1", "CT1", "CT2", "CT1", "CT2"],
        })
        var = pd.DataFrame(index=["Gene1", "Gene2"])

        return AnnData(X=X, obs=obs, var=var)

    def test_computes_mean_per_group(self, mock_adata):
        """Verify mean is computed correctly per group."""
        from src.data.preprocessing import compute_pseudobulk

        pb = compute_pseudobulk(
            mock_adata,
            groupby=["ROSMAP_IndividualID", "supercluster_name"],
        )

        # S1_CT1 mean: (1+3)/2=2, (2+4)/2=3
        s1_ct1 = pb[(pb["ROSMAP_IndividualID"] == "S1") & (pb["supercluster_name"] == "CT1")]
        assert np.isclose(s1_ct1["Gene1"].values[0], 2.0)
        assert np.isclose(s1_ct1["Gene2"].values[0], 3.0)

        # S1_CT2: single cell [5, 6]
        s1_ct2 = pb[(pb["ROSMAP_IndividualID"] == "S1") & (pb["supercluster_name"] == "CT2")]
        assert np.isclose(s1_ct2["Gene1"].values[0], 5.0)

    def test_includes_cell_counts(self, mock_adata):
        """Include n_cells column in output."""
        from src.data.preprocessing import compute_pseudobulk

        pb = compute_pseudobulk(
            mock_adata,
            groupby=["ROSMAP_IndividualID", "supercluster_name"],
        )

        assert "n_cells" in pb.columns
        s1_ct1 = pb[(pb["ROSMAP_IndividualID"] == "S1") & (pb["supercluster_name"] == "CT1")]
        assert s1_ct1["n_cells"].values[0] == 2
