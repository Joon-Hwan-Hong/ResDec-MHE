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


class TestGroupSeparator:
    """Tests for safe group key joining/splitting."""

    def test_separator_not_in_data(self):
        """Separator should not appear in typical data values."""
        from src.data.constants import GROUP_SEPARATOR, CELL_TYPE_ORDER

        for ct in CELL_TYPE_ORDER:
            assert GROUP_SEPARATOR not in ct, f"Separator found in cell type: {ct}"
