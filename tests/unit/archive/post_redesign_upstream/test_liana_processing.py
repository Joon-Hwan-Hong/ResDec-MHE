"""
Tests for src/data/liana_processing.py

Tests cover:
- Edge type mapping from CellChatDB categories
- Building CCC features from LIANA+ results
- Empty input handling
- LIANA result filtering and aggregation
- Adjacency matrix conversion
"""

import logging
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from pathlib import Path


@pytest.fixture
def mock_liana_results():
    """Create mock LIANA+ output DataFrame."""
    return pd.DataFrame({
        "source": ["Astrocyte", "Microglia", "Oligodendrocyte", "Astrocyte"],
        "target": ["Microglia", "Oligodendrocyte", "Astrocyte", "Oligodendrocyte"],
        "ligand_complex": ["TGFB1", "IL1B", "FAKE_LIGAND", "APP"],
        "receptor_complex": ["TGFBR1", "IL1R1", "FAKE_RECEPTOR", "CD74"],
        "magnitude_rank": [0.02, 0.03, 0.5, 0.01],
        "specificity_rank": [0.01, 0.02, 0.4, 0.02],
        "liana_score": [0.95, 0.90, 0.30, 0.98],
    })


@pytest.fixture
def mock_cellchatdb_csv(tmp_path):
    """Create a mock CellChatDB CSV file."""
    csv_path = tmp_path / "CellChatDB_human_interaction.csv"

    db_data = pd.DataFrame({
        "ligand.symbol": ["TGFB1", "IL1B", "APP"],
        "receptor.symbol": ["TGFBR1", "IL1R1", "CD74"],
        "annotation": ["Secreted Signaling", "Secreted Signaling", "Secreted Signaling"],
    })
    db_data.to_csv(csv_path, index=False)

    return csv_path


class TestNormalizeAnnotation:
    """L-A3: Tests for _normalize_annotation utility."""

    def test_normalize_annotation(self):
        """Should replace spaces and hyphens with underscores."""
        from src.data.liana_processing import _normalize_annotation

        assert _normalize_annotation("Secreted Signaling") == "Secreted_Signaling"
        assert _normalize_annotation("Non-protein Signaling") == "Non_protein_Signaling"
        assert _normalize_annotation("ECM-Receptor") == "ECM_Receptor"
        assert _normalize_annotation("Simple") == "Simple"


class TestLoadCellChatDBCategories:
    """Tests for loading CellChatDB category mappings."""

    def test_loads_ligand_receptor_mapping(self, mock_cellchatdb_csv):
        """Should load LR pairs and map to categories (normalized with underscores)."""
        from src.data.liana_processing import load_cellchatdb_categories

        lr_to_cat = load_cellchatdb_categories(mock_cellchatdb_csv)

        assert "TGFB1_TGFBR1" in lr_to_cat
        # Annotations are normalized: "Secreted Signaling" -> "Secreted_Signaling"
        assert lr_to_cat["TGFB1_TGFBR1"] == "Secreted_Signaling"

    def test_stores_reverse_mapping(self, mock_cellchatdb_csv):
        """Should also store reversed LR key for flexibility."""
        from src.data.liana_processing import load_cellchatdb_categories

        lr_to_cat = load_cellchatdb_categories(mock_cellchatdb_csv)

        # Both directions should be stored
        assert "TGFB1_TGFBR1" in lr_to_cat
        assert "TGFBR1_TGFB1" in lr_to_cat

    def test_handles_missing_values(self, tmp_path):
        """Should skip rows with missing ligand or receptor."""
        csv_path = tmp_path / "db_with_missing.csv"

        db_data = pd.DataFrame({
            "ligand.symbol": ["TGFB1", None, "APP"],
            "receptor.symbol": ["TGFBR1", "IL1R1", None],
            "annotation": ["Secreted Signaling", "Secreted Signaling", "ECM-Receptor"],
        })
        db_data.to_csv(csv_path, index=False)

        from src.data.liana_processing import load_cellchatdb_categories

        lr_to_cat = load_cellchatdb_categories(csv_path)

        # Only complete pairs should be loaded
        assert "TGFB1_TGFBR1" in lr_to_cat
        assert len([k for k in lr_to_cat if "IL1R1" in k]) == 0

    def test_handles_nan_annotation(self, tmp_path):
        """Should use EDGE_TYPE_NOVEL for NaN annotations."""
        from src.data.liana_processing import load_cellchatdb_categories, EDGE_TYPE_NOVEL

        csv_path = tmp_path / "db_with_nan_annotation.csv"

        db_data = pd.DataFrame({
            "ligand.symbol": ["TGFB1", "APP"],
            "receptor.symbol": ["TGFBR1", "CD74"],
            "annotation": ["Secreted Signaling", np.nan],  # Second has NaN annotation
        })
        db_data.to_csv(csv_path, index=False)

        lr_to_cat = load_cellchatdb_categories(csv_path)

        # Normal annotation should be normalized
        assert lr_to_cat["TGFB1_TGFBR1"] == "Secreted_Signaling"

        # NaN annotation should get EDGE_TYPE_NOVEL, not "nan" string
        assert lr_to_cat["APP_CD74"] == EDGE_TYPE_NOVEL
        assert lr_to_cat["APP_CD74"] != "nan"

    def test_falls_back_to_alternate_columns_when_nan(self, tmp_path):
        """Should use ligand_symbol/receptor_symbol when primary columns are NaN."""
        from src.data.liana_processing import load_cellchatdb_categories

        csv_path = tmp_path / "db_with_alternate_columns.csv"

        db_data = pd.DataFrame({
            # Primary columns have NaN, alternate columns have values
            "ligand.symbol": ["TGFB1", np.nan, np.nan],
            "receptor.symbol": ["TGFBR1", np.nan, np.nan],
            "ligand_symbol": [np.nan, "IL1B", "APP"],
            "receptor_symbol": [np.nan, "IL1R1", "CD74"],
            "annotation": ["Secreted_Signaling", "Secreted_Signaling", "ECM_Receptor"],
        })
        db_data.to_csv(csv_path, index=False)

        lr_to_cat = load_cellchatdb_categories(csv_path)

        # First row: primary columns work
        assert "TGFB1_TGFBR1" in lr_to_cat

        # Second row: should fall back to alternate columns
        assert "IL1B_IL1R1" in lr_to_cat
        assert lr_to_cat["IL1B_IL1R1"] == "Secreted_Signaling"

        # Third row: should also use alternate columns
        assert "APP_CD74" in lr_to_cat
        assert lr_to_cat["APP_CD74"] == "ECM_Receptor"

    def test_skips_when_all_symbol_columns_are_nan(self, tmp_path):
        """Should skip rows when both primary and alternate columns are NaN."""
        from src.data.liana_processing import load_cellchatdb_categories

        csv_path = tmp_path / "db_with_all_nan_symbols.csv"

        db_data = pd.DataFrame({
            "ligand.symbol": ["TGFB1", np.nan],
            "receptor.symbol": ["TGFBR1", np.nan],
            "ligand_symbol": [np.nan, np.nan],  # Both alternate also NaN
            "receptor_symbol": [np.nan, np.nan],
            "annotation": ["Secreted_Signaling", "ECM_Receptor"],
        })
        db_data.to_csv(csv_path, index=False)

        lr_to_cat = load_cellchatdb_categories(csv_path)

        # First row should work
        assert "TGFB1_TGFBR1" in lr_to_cat

        # Second row should be skipped (no valid symbols)
        # Just verify we don't crash and the first row is present
        assert len(lr_to_cat) >= 2  # At least the forward and reverse for first row


class TestAssignEdgeTypes:
    """Tests for edge type assignment."""

    def test_known_category_assigned_correctly(self, mock_liana_results, mock_cellchatdb_csv):
        """Known CellChatDB interactions should get correct category (normalized)."""
        from src.data.liana_processing import assign_edge_types

        result = assign_edge_types(
            mock_liana_results,
            cellchatdb_path=mock_cellchatdb_csv
        )

        assert "edge_type_name" in result.columns
        assert "edge_type" in result.columns

        # TGFB1_TGFBR1 is in our mock DB
        # Annotations are normalized: "Secreted Signaling" -> "Secreted_Signaling"
        tgfb_row = result[result["ligand_complex"] == "TGFB1"].iloc[0]
        assert tgfb_row["edge_type_name"] == "Secreted_Signaling"

    def test_unknown_interaction_gets_novel_category(self, mock_liana_results, mock_cellchatdb_csv):
        """Unknown L-R pairs should get Novel/Uncharacterized category."""
        from src.data.liana_processing import assign_edge_types
        from src.data.constants import EDGE_TYPE_NOVEL

        result = assign_edge_types(
            mock_liana_results,
            cellchatdb_path=mock_cellchatdb_csv
        )

        # FAKE_LIGAND_FAKE_RECEPTOR is not in DB
        fake_row = result[result["ligand_complex"] == "FAKE_LIGAND"].iloc[0]
        assert fake_row["edge_type_name"] == EDGE_TYPE_NOVEL

    def test_edge_type_indices_consistent(self, mock_liana_results, mock_cellchatdb_csv):
        """Edge type indices should match category order."""
        from src.data.liana_processing import assign_edge_types
        from src.data.constants import CELLCHATDB_EDGE_TYPES, EDGE_TYPE_NOVEL

        result = assign_edge_types(
            mock_liana_results,
            cellchatdb_path=mock_cellchatdb_csv
        )

        categories = CELLCHATDB_EDGE_TYPES + [EDGE_TYPE_NOVEL]
        category_to_idx = {cat: idx for idx, cat in enumerate(categories)}

        for _, row in result.iterrows():
            expected_idx = category_to_idx[row["edge_type_name"]]
            assert row["edge_type"] == expected_idx

    def test_handles_missing_db_file(self, mock_liana_results, tmp_path, caplog):
        """Should handle missing CellChatDB file gracefully."""
        from src.data.liana_processing import assign_edge_types
        from src.data.constants import EDGE_TYPE_NOVEL

        # Use non-existent path
        with caplog.at_level(logging.WARNING, logger="src.data.liana_processing"):
            result = assign_edge_types(
                mock_liana_results,
                cellchatdb_path=tmp_path / "nonexistent.csv"
            )

        # All interactions should get novel category
        assert (result["edge_type_name"] == EDGE_TYPE_NOVEL).all()

        # Should log warning about missing DB file
        assert any("CellChatDB not found" in msg for msg in caplog.messages)

    def test_empty_dataframe(self, mock_cellchatdb_csv):
        """Should handle empty DataFrame input."""
        from src.data.liana_processing import assign_edge_types

        empty_df = pd.DataFrame(columns=[
            "source", "target", "ligand_complex", "receptor_complex"
        ])

        result = assign_edge_types(empty_df, cellchatdb_path=mock_cellchatdb_csv)

        assert "edge_type_name" in result.columns
        assert "edge_type" in result.columns
        assert len(result) == 0

    def test_assign_edge_types_nan_ligand(self, mock_cellchatdb_csv):
        """L-A6: NaN ligand should map to novel_category."""
        from src.data.liana_processing import assign_edge_types
        from src.data.constants import EDGE_TYPE_NOVEL

        # Create DataFrame with NaN in ligand column
        df_with_nan = pd.DataFrame({
            "source": ["Astrocyte", "Microglia", "Astrocyte"],
            "target": ["Microglia", "Astrocyte", "Oligodendrocyte"],
            "ligand_complex": [np.nan, "IL1B", "TGFB1"],
            "receptor_complex": ["TGFBR1", np.nan, "TGFBR1"],
            "magnitude_rank": [0.1, 0.2, 0.3],
        })

        result = assign_edge_types(df_with_nan, cellchatdb_path=mock_cellchatdb_csv)

        # Row 0: NaN ligand -> novel_category
        assert result.iloc[0]["edge_type_name"] == EDGE_TYPE_NOVEL
        # Row 1: NaN receptor -> novel_category
        assert result.iloc[1]["edge_type_name"] == EDGE_TYPE_NOVEL
        # Row 2: valid ligand+receptor (TGFB1_TGFBR1 in DB) -> Secreted_Signaling
        assert result.iloc[2]["edge_type_name"] == "Secreted_Signaling"


class TestGetEdgeTypeMetadata:
    """Tests for edge type metadata."""

    def test_returns_all_required_keys(self):
        """Metadata should contain all required keys."""
        from src.data.liana_processing import get_edge_type_metadata

        meta = get_edge_type_metadata()

        assert "categories" in meta
        assert "n_edge_types" in meta
        assert "category_to_idx" in meta
        assert "idx_to_category" in meta
        assert "source" in meta

    def test_all_edge_types_indexed(self):
        """All edge types should have consistent indices."""
        from src.data.liana_processing import get_edge_type_metadata
        from src.data.constants import ALL_EDGE_TYPES

        meta = get_edge_type_metadata()

        assert meta["n_edge_types"] == len(ALL_EDGE_TYPES)
        assert set(meta["categories"]) == set(ALL_EDGE_TYPES)

        # Verify bidirectional mapping
        for cat, idx in meta["category_to_idx"].items():
            assert meta["idx_to_category"][idx] == cat

    def test_indices_are_consecutive(self):
        """Indices should be 0, 1, 2, ... n-1."""
        from src.data.liana_processing import get_edge_type_metadata

        meta = get_edge_type_metadata()

        indices = sorted(meta["idx_to_category"].keys())
        expected = list(range(meta["n_edge_types"]))
        assert indices == expected


class TestFilterLianaResults:
    """Tests for filtering LIANA+ results."""

    def test_filters_by_magnitude_rank(self, mock_liana_results):
        """Should filter by magnitude rank threshold."""
        from src.data.liana_processing import filter_liana_results

        # Only rows with magnitude_rank <= 0.02
        filtered = filter_liana_results(
            mock_liana_results,
            magnitude_rank_threshold=0.02,
            specificity_rank_threshold=1.0,  # No filtering on specificity
        )

        assert len(filtered) < len(mock_liana_results)
        assert (filtered["magnitude_rank"] <= 0.02).all()

    def test_filters_by_specificity_rank(self, mock_liana_results):
        """Should filter by specificity rank threshold."""
        from src.data.liana_processing import filter_liana_results

        filtered = filter_liana_results(
            mock_liana_results,
            magnitude_rank_threshold=1.0,  # No filtering on magnitude
            specificity_rank_threshold=0.02,
        )

        assert (filtered["specificity_rank"] <= 0.02).all()

    def test_filters_by_min_score(self, mock_liana_results):
        """Should filter by minimum liana score."""
        from src.data.liana_processing import filter_liana_results

        filtered = filter_liana_results(
            mock_liana_results,
            magnitude_rank_threshold=1.0,
            specificity_rank_threshold=1.0,
            min_score=0.9,
        )

        assert (filtered["liana_score"] >= 0.9).all()

    def test_combined_filters(self, mock_liana_results):
        """Should apply all filters together."""
        from src.data.liana_processing import filter_liana_results

        filtered = filter_liana_results(
            mock_liana_results,
            magnitude_rank_threshold=0.05,
            specificity_rank_threshold=0.05,
            min_score=0.9,
        )

        # Should satisfy all conditions
        assert (filtered["magnitude_rank"] <= 0.05).all()
        assert (filtered["specificity_rank"] <= 0.05).all()
        assert (filtered["liana_score"] >= 0.9).all()


class TestAggregateLianaByCelltypePair:
    """Tests for aggregating LIANA results by cell type pair."""

    def test_aggregates_by_mean(self, mock_liana_results):
        """Should aggregate scores by mean."""
        from src.data.liana_processing import aggregate_liana_by_celltype_pair

        aggregated = aggregate_liana_by_celltype_pair(
            mock_liana_results,
            score_col="magnitude_rank",
            agg_func="mean",
        )

        assert "source" in aggregated.columns
        assert "target" in aggregated.columns
        assert "magnitude_rank_mean" in aggregated.columns

    def test_aggregates_by_count(self, mock_liana_results):
        """Should count interactions per cell type pair."""
        from src.data.liana_processing import aggregate_liana_by_celltype_pair

        aggregated = aggregate_liana_by_celltype_pair(
            mock_liana_results,
            agg_func="count",
        )

        assert "interaction_count" in aggregated.columns

    def test_aggregates_by_max(self, mock_liana_results):
        """Should get max score per cell type pair."""
        from src.data.liana_processing import aggregate_liana_by_celltype_pair

        aggregated = aggregate_liana_by_celltype_pair(
            mock_liana_results,
            score_col="liana_score",
            agg_func="max",
        )

        assert "liana_score_max" in aggregated.columns


class TestLianaToAdjacencyMatrix:
    """Tests for converting LIANA results to adjacency matrix."""

    def test_correct_shape(self, mock_liana_results):
        """Adjacency matrix should be [n_types, n_types]."""
        from src.data.liana_processing import liana_to_adjacency_matrix

        cell_types = ["Astrocyte", "Microglia", "Oligodendrocyte"]

        adj = liana_to_adjacency_matrix(mock_liana_results, cell_types)

        assert adj.shape == (3, 3)

    def test_fill_value_for_missing(self, mock_liana_results):
        """Missing interactions should have fill value."""
        from src.data.liana_processing import liana_to_adjacency_matrix

        cell_types = ["Astrocyte", "Microglia", "Oligodendrocyte", "ExtraType"]
        fill_value = 1.0

        adj = liana_to_adjacency_matrix(
            mock_liana_results,
            cell_types,
            fill_value=fill_value,
        )

        # ExtraType has no interactions, row/col should be fill_value
        extra_idx = cell_types.index("ExtraType")
        assert np.allclose(adj[extra_idx, :], fill_value)
        assert np.allclose(adj[:, extra_idx], fill_value)

    def test_keeps_minimum_score(self, mock_liana_results):
        """For multiple interactions, should keep minimum rank."""
        from src.data.liana_processing import liana_to_adjacency_matrix

        # Add duplicate interaction with different scores
        df = mock_liana_results.copy()
        df = pd.concat([df, pd.DataFrame({
            "source": ["Astrocyte"],
            "target": ["Microglia"],
            "ligand_complex": ["OTHER"],
            "receptor_complex": ["RECEPTOR"],
            "magnitude_rank": [0.5],  # Higher than existing 0.02
        })], ignore_index=True)

        cell_types = ["Astrocyte", "Microglia", "Oligodendrocyte"]

        adj = liana_to_adjacency_matrix(df, cell_types)

        # Should keep the minimum (0.02, not 0.5).
        # Use approx because adj is float32 and 0.02 is not exactly representable.
        astro_idx = cell_types.index("Astrocyte")
        micro_idx = cell_types.index("Microglia")
        assert adj[astro_idx, micro_idx] == pytest.approx(0.02)

    def test_handles_unknown_cell_types(self, mock_liana_results):
        """Should skip interactions with unknown cell types."""
        from src.data.liana_processing import liana_to_adjacency_matrix

        # Cell types list doesn't include Oligodendrocyte
        cell_types = ["Astrocyte", "Microglia"]

        adj = liana_to_adjacency_matrix(mock_liana_results, cell_types)

        # Should only have Astrocyte -> Microglia interaction
        assert adj.shape == (2, 2)

    def test_skips_invalid_magnitude_rank(self):
        """Should skip edges with NaN or out-of-range magnitude_rank."""
        from src.data.liana_processing import liana_to_adjacency_matrix

        # Create data with invalid magnitude_rank values
        df = pd.DataFrame({
            "source": ["Astrocyte", "Microglia", "Oligodendrocyte", "Astrocyte"],
            "target": ["Microglia", "Oligodendrocyte", "Astrocyte", "Oligodendrocyte"],
            "magnitude_rank": [0.3, np.nan, -0.1, 1.5],  # Valid, NaN, negative, >1
        })

        cell_types = ["Astrocyte", "Microglia", "Oligodendrocyte"]
        fill_value = 1.0

        adj = liana_to_adjacency_matrix(df, cell_types, fill_value=fill_value)

        # Only Astrocyte -> Microglia (0.3) should be included
        astro_idx = cell_types.index("Astrocyte")
        micro_idx = cell_types.index("Microglia")
        oligo_idx = cell_types.index("Oligodendrocyte")

        # Valid edge: Astrocyte -> Microglia = 0.3.
        # Use approx because adj is float32 and 0.3 is not exactly representable.
        assert adj[astro_idx, micro_idx] == pytest.approx(0.3)

        # Invalid edges: should have fill_value
        assert adj[micro_idx, oligo_idx] == fill_value  # NaN
        assert adj[oligo_idx, astro_idx] == fill_value  # negative
        assert adj[astro_idx, oligo_idx] == fill_value  # >1


class TestBuildSubjectCCCFeatures:
    """Tests for building CCC feature matrices."""

    def test_edge_index_shape(self, mock_liana_results, mock_cellchatdb_csv):
        """Edge index should be [2, n_edges]."""
        from src.data.liana_processing import build_subject_ccc_features

        cell_types = ["Astrocyte", "Microglia", "Oligodendrocyte"]

        # First assign edge types
        from src.data.liana_processing import assign_edge_types
        df_with_types = assign_edge_types(
            mock_liana_results,
            cellchatdb_path=mock_cellchatdb_csv
        )

        features = build_subject_ccc_features(df_with_types, cell_types)

        assert features["edge_index"].shape[0] == 2
        assert features["edge_index"].shape[1] == features["n_edges"]

    def test_edge_type_shape(self, mock_liana_results, mock_cellchatdb_csv):
        """Edge type should be [n_edges]."""
        from src.data.liana_processing import build_subject_ccc_features, assign_edge_types

        cell_types = ["Astrocyte", "Microglia", "Oligodendrocyte"]
        df_with_types = assign_edge_types(
            mock_liana_results,
            cellchatdb_path=mock_cellchatdb_csv
        )

        features = build_subject_ccc_features(df_with_types, cell_types)

        assert features["edge_type"].shape == (features["n_edges"],)

    def test_edge_attr_shape(self, mock_liana_results, mock_cellchatdb_csv):
        """Edge attr should be [n_edges, 1]."""
        from src.data.liana_processing import build_subject_ccc_features, assign_edge_types

        cell_types = ["Astrocyte", "Microglia", "Oligodendrocyte"]
        df_with_types = assign_edge_types(
            mock_liana_results,
            cellchatdb_path=mock_cellchatdb_csv
        )

        features = build_subject_ccc_features(df_with_types, cell_types)

        assert features["edge_attr"].shape == (features["n_edges"], 1)

    def test_adjacency_shape(self, mock_liana_results, mock_cellchatdb_csv):
        """Adjacency matrix should be [n_types, n_types]."""
        from src.data.liana_processing import build_subject_ccc_features, assign_edge_types

        cell_types = ["Astrocyte", "Microglia", "Oligodendrocyte"]
        df_with_types = assign_edge_types(
            mock_liana_results,
            cellchatdb_path=mock_cellchatdb_csv
        )

        features = build_subject_ccc_features(df_with_types, cell_types)

        assert features["adjacency"].shape == (3, 3)

    def test_empty_liana_results(self):
        """Should handle empty LIANA results gracefully."""
        from src.data.liana_processing import build_subject_ccc_features

        cell_types = ["Astrocyte", "Microglia", "Oligodendrocyte"]
        empty_df = pd.DataFrame()

        features = build_subject_ccc_features(empty_df, cell_types)

        assert features["edge_index"].shape == (2, 0)
        assert features["edge_type"].shape == (0,)
        assert features["edge_attr"].shape == (0, 1)
        assert features["n_edges"] == 0

    def test_edge_indices_within_bounds(self, mock_liana_results, mock_cellchatdb_csv):
        """Edge indices should be valid cell type indices."""
        from src.data.liana_processing import build_subject_ccc_features, assign_edge_types

        cell_types = ["Astrocyte", "Microglia", "Oligodendrocyte"]
        df_with_types = assign_edge_types(
            mock_liana_results,
            cellchatdb_path=mock_cellchatdb_csv
        )

        features = build_subject_ccc_features(df_with_types, cell_types)

        # All indices should be in range [0, n_types)
        assert features["edge_index"].min() >= 0
        assert features["edge_index"].max() < len(cell_types)

    def test_edge_types_within_bounds(self, mock_liana_results, mock_cellchatdb_csv):
        """Edge type indices should be valid."""
        from src.data.liana_processing import build_subject_ccc_features, assign_edge_types
        from src.data.constants import ALL_EDGE_TYPES

        cell_types = ["Astrocyte", "Microglia", "Oligodendrocyte"]
        df_with_types = assign_edge_types(
            mock_liana_results,
            cellchatdb_path=mock_cellchatdb_csv
        )

        features = build_subject_ccc_features(df_with_types, cell_types)

        # All edge type indices should be in range
        assert features["edge_type"].min() >= 0
        assert features["edge_type"].max() < len(ALL_EDGE_TYPES)

    def test_assigns_edge_types_if_missing(self, mock_liana_results):
        """Should auto-assign edge types if not present."""
        from src.data.liana_processing import build_subject_ccc_features

        cell_types = ["Astrocyte", "Microglia", "Oligodendrocyte"]

        # Input doesn't have edge_type_name column
        assert "edge_type_name" not in mock_liana_results.columns

        features = build_subject_ccc_features(mock_liana_results, cell_types)

        # Should still produce valid output
        assert features["n_edges"] > 0
        assert features["edge_type"].shape == (features["n_edges"],)

    def test_skips_unknown_cell_types(self, mock_liana_results, mock_cellchatdb_csv):
        """Should skip edges involving unknown cell types."""
        from src.data.liana_processing import build_subject_ccc_features, assign_edge_types

        # Cell types list is missing Oligodendrocyte
        cell_types = ["Astrocyte", "Microglia"]
        df_with_types = assign_edge_types(
            mock_liana_results,
            cellchatdb_path=mock_cellchatdb_csv
        )

        features = build_subject_ccc_features(df_with_types, cell_types)

        # Only Astrocyte -> Microglia edge should be included
        assert features["n_edges"] == 1

    def test_correct_dtypes(self, mock_liana_results, mock_cellchatdb_csv):
        """Arrays should have correct dtypes."""
        from src.data.liana_processing import build_subject_ccc_features, assign_edge_types

        cell_types = ["Astrocyte", "Microglia", "Oligodendrocyte"]
        df_with_types = assign_edge_types(
            mock_liana_results,
            cellchatdb_path=mock_cellchatdb_csv
        )

        features = build_subject_ccc_features(df_with_types, cell_types)

        assert features["edge_index"].dtype == np.int64
        assert features["edge_type"].dtype == np.int64
        assert features["edge_attr"].dtype == np.float32
        assert features["adjacency"].dtype == np.float32

    def test_edge_attr_inverts_magnitude_rank(self):
        """edge_attr should be 1.0 - magnitude_rank (higher = stronger)."""
        from src.data.liana_processing import build_subject_ccc_features

        # Create controlled test data with known magnitude_rank values
        liana_results = pd.DataFrame({
            "source": ["Astrocyte", "Microglia"],
            "target": ["Microglia", "Astrocyte"],
            "ligand_complex": ["TGFB1", "IL1B"],
            "receptor_complex": ["TGFBR1", "IL1R1"],
            "magnitude_rank": [0.1, 0.9],  # Known values
            "edge_type_name": ["Secreted_Signaling", "Secreted_Signaling"],
        })
        cell_types = ["Astrocyte", "Microglia"]

        features = build_subject_ccc_features(liana_results, cell_types)

        # edge_attr should be inverted: 1.0 - magnitude_rank
        # magnitude_rank 0.1 -> edge_attr 0.9 (strong interaction)
        # magnitude_rank 0.9 -> edge_attr 0.1 (weak interaction)
        expected_attrs = np.array([[0.9], [0.1]], dtype=np.float32)
        np.testing.assert_array_almost_equal(features["edge_attr"], expected_attrs)

    def test_edge_attr_bounded_zero_one(self, mock_liana_results, mock_cellchatdb_csv):
        """edge_attr values should be in [0, 1] range."""
        from src.data.liana_processing import build_subject_ccc_features, assign_edge_types

        cell_types = ["Astrocyte", "Microglia", "Oligodendrocyte"]
        df_with_types = assign_edge_types(
            mock_liana_results,
            cellchatdb_path=mock_cellchatdb_csv
        )

        features = build_subject_ccc_features(df_with_types, cell_types)

        if features["n_edges"] > 0:
            assert (features["edge_attr"] >= 0.0).all()
            assert (features["edge_attr"] <= 1.0).all()


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_interaction(self, mock_cellchatdb_csv):
        """Should handle single interaction correctly."""
        from src.data.liana_processing import build_subject_ccc_features, assign_edge_types

        single_df = pd.DataFrame({
            "source": ["Astrocyte"],
            "target": ["Microglia"],
            "ligand_complex": ["TGFB1"],
            "receptor_complex": ["TGFBR1"],
            "magnitude_rank": [0.05],
        })

        cell_types = ["Astrocyte", "Microglia"]
        df_with_types = assign_edge_types(single_df, cellchatdb_path=mock_cellchatdb_csv)
        features = build_subject_ccc_features(df_with_types, cell_types)

        assert features["n_edges"] == 1
        assert features["edge_index"].shape == (2, 1)

    def test_self_loop_interaction(self, mock_cellchatdb_csv):
        """Should handle self-loop (same source and target) correctly."""
        from src.data.liana_processing import build_subject_ccc_features, assign_edge_types

        self_loop_df = pd.DataFrame({
            "source": ["Astrocyte"],
            "target": ["Astrocyte"],
            "ligand_complex": ["TGFB1"],
            "receptor_complex": ["TGFBR1"],
            "magnitude_rank": [0.05],
        })

        cell_types = ["Astrocyte", "Microglia"]
        df_with_types = assign_edge_types(self_loop_df, cellchatdb_path=mock_cellchatdb_csv)
        features = build_subject_ccc_features(df_with_types, cell_types)

        assert features["n_edges"] == 1
        # Both source and target should be Astrocyte (index 0)
        assert features["edge_index"][0, 0] == 0
        assert features["edge_index"][1, 0] == 0

    def test_nan_in_magnitude_rank(self, mock_cellchatdb_csv):
        """Edges with NaN magnitude_rank should be skipped.

        We only include edges where LIANA+ computed valid scores.
        This is more conservative than imputing missing values.
        """
        from src.data.liana_processing import build_subject_ccc_features, assign_edge_types

        df_with_nan = pd.DataFrame({
            "source": ["Astrocyte", "Microglia"],
            "target": ["Microglia", "Astrocyte"],
            "ligand_complex": ["TGFB1", "IL1B"],
            "receptor_complex": ["TGFBR1", "IL1R1"],
            "magnitude_rank": [np.nan, 0.3],  # First is NaN, second is valid
        })

        cell_types = ["Astrocyte", "Microglia"]
        df_with_types = assign_edge_types(df_with_nan, cellchatdb_path=mock_cellchatdb_csv)
        features = build_subject_ccc_features(df_with_types, cell_types)

        # Should only include the valid edge, skip the NaN one
        assert features["n_edges"] == 1
        # The valid edge should have edge_attr = 1.0 - 0.3 = 0.7
        assert np.isclose(features["edge_attr"][0, 0], 0.7)

    def test_out_of_range_magnitude_rank_skipped(self, mock_cellchatdb_csv):
        """Edges with magnitude_rank outside [0, 1] should be skipped."""
        from src.data.liana_processing import build_subject_ccc_features, assign_edge_types

        df_with_invalid = pd.DataFrame({
            "source": ["Astrocyte", "Microglia", "Astrocyte"],
            "target": ["Microglia", "Astrocyte", "Astrocyte"],
            "ligand_complex": ["TGFB1", "IL1B", "NGF"],
            "receptor_complex": ["TGFBR1", "IL1R1", "NGFR"],
            "magnitude_rank": [-0.5, 1.5, 0.5],  # Two invalid, one valid
        })

        cell_types = ["Astrocyte", "Microglia"]
        df_with_types = assign_edge_types(df_with_invalid, cellchatdb_path=mock_cellchatdb_csv)
        features = build_subject_ccc_features(df_with_types, cell_types)

        # Should only include the valid edge (0.5), skip the out-of-range ones
        assert features["n_edges"] == 1
        # The valid edge should have edge_attr = 1.0 - 0.5 = 0.5
        assert np.isclose(features["edge_attr"][0, 0], 0.5)

    def test_large_number_of_edges(self, mock_cellchatdb_csv):
        """Should handle large number of edges efficiently."""
        from src.data.liana_processing import build_subject_ccc_features, assign_edge_types

        n_edges = 10000
        large_df = pd.DataFrame({
            "source": np.random.choice(["Astrocyte", "Microglia", "Oligodendrocyte"], n_edges),
            "target": np.random.choice(["Astrocyte", "Microglia", "Oligodendrocyte"], n_edges),
            "ligand_complex": [f"L{i}" for i in range(n_edges)],
            "receptor_complex": [f"R{i}" for i in range(n_edges)],
            "magnitude_rank": np.random.rand(n_edges),
        })

        cell_types = ["Astrocyte", "Microglia", "Oligodendrocyte"]
        df_with_types = assign_edge_types(large_df, cellchatdb_path=mock_cellchatdb_csv)
        features = build_subject_ccc_features(df_with_types, cell_types)

        assert features["n_edges"] == n_edges
        assert features["edge_index"].shape == (2, n_edges)


class TestRunLianaAnalysis:
    """L-A1: Tests for run_liana_analysis() function."""

    def test_import_error_when_liana_not_installed(self):
        """Should raise ImportError with helpful message when liana not installed."""
        from anndata import AnnData
        from src.data.liana_processing import run_liana_analysis

        # Create minimal AnnData
        adata = AnnData(np.random.rand(10, 5))
        adata.obs["supercluster_name"] = ["TypeA"] * 5 + ["TypeB"] * 5

        # Setting sys.modules["liana"] = None causes `import liana` to raise ImportError
        with patch.dict(sys.modules, {"liana": None}):
            with pytest.raises(ImportError, match="LIANA"):
                run_liana_analysis(adata)

    def test_calls_rank_aggregate_with_correct_params(self):
        """Should call li.mt.rank_aggregate with correct parameters."""
        from anndata import AnnData

        adata = AnnData(np.random.rand(20, 10))
        adata.obs["supercluster_name"] = ["TypeA"] * 10 + ["TypeB"] * 10

        # Mock liana module
        mock_li = MagicMock()
        mock_liana_res = pd.DataFrame({
            "source": ["TypeA"], "target": ["TypeB"],
            "ligand_complex": ["L1"], "receptor_complex": ["R1"],
            "magnitude_rank": [0.1],
        })

        # rank_aggregate stores results in adata.uns
        def fake_rank_aggregate(adata_arg, **kwargs):
            adata_arg.uns["liana_res"] = mock_liana_res

        mock_li.mt.rank_aggregate = MagicMock(side_effect=fake_rank_aggregate)

        with patch.dict(sys.modules, {"liana": mock_li}):
            from src.data.liana_processing import run_liana_analysis
            result = run_liana_analysis(
                adata,
                cell_type_column="supercluster_name",
                resource_name="CellChatDB",
                verbose=False,
            )

        # Verify rank_aggregate was called
        mock_li.mt.rank_aggregate.assert_called_once()
        call_kwargs = mock_li.mt.rank_aggregate.call_args
        # Check groupby parameter (could be positional or keyword)
        if call_kwargs.kwargs:
            assert call_kwargs.kwargs.get("groupby") == "supercluster_name"
        else:
            assert call_kwargs[1]["groupby"] == "supercluster_name"

        # Verify result is a DataFrame
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_returns_copy_of_results(self):
        """Should return a copy, not a reference to adata.uns."""
        from anndata import AnnData

        adata = AnnData(np.random.rand(20, 10))
        adata.obs["supercluster_name"] = ["TypeA"] * 10 + ["TypeB"] * 10

        mock_li = MagicMock()
        original_df = pd.DataFrame({"source": ["TypeA"], "target": ["TypeB"]})

        def fake_rank_aggregate(adata_arg, **kwargs):
            adata_arg.uns["liana_res"] = original_df

        mock_li.mt.rank_aggregate = MagicMock(side_effect=fake_rank_aggregate)

        with patch.dict(sys.modules, {"liana": mock_li}):
            from src.data.liana_processing import run_liana_analysis
            result = run_liana_analysis(adata, verbose=False)

        # Modifying result should not affect original
        result["new_col"] = 1
        assert "new_col" not in adata.uns["liana_res"].columns


class TestRunLianaPerSubject:
    """L-A2: Tests for run_liana_per_subject() function."""

    def test_skips_subject_with_too_few_cell_types(self):
        """Should skip subjects with < 2 valid cell types and return empty DataFrame."""
        from anndata import AnnData

        # Create AnnData with 2 subjects, one has only 1 cell type
        n_cells = 30
        adata = AnnData(np.random.rand(n_cells, 10))
        adata.obs["subject_id"] = ["S1"] * 20 + ["S2"] * 10
        adata.obs["supercluster_name"] = (
            ["TypeA"] * 10 + ["TypeB"] * 10 +  # S1: 2 types
            ["TypeA"] * 10                       # S2: only 1 type
        )

        mock_li = MagicMock()
        mock_result = pd.DataFrame({
            "source": ["TypeA"], "target": ["TypeB"],
            "magnitude_rank": [0.1],
        })

        def fake_rank_aggregate(adata_arg, **kwargs):
            adata_arg.uns["liana_res"] = mock_result

        mock_li.mt.rank_aggregate = MagicMock(side_effect=fake_rank_aggregate)

        with patch.dict(sys.modules, {"liana": mock_li}):
            from src.data.liana_processing import run_liana_per_subject
            results = run_liana_per_subject(
                adata,
                subject_column="subject_id",
                cell_type_column="supercluster_name",
                min_cells_per_type=10,
                verbose=False,
            )

        # S1 should have results, S2 should be empty
        assert "S1" in results
        assert "S2" in results
        assert len(results["S2"]) == 0  # Skipped

    def test_caches_results_to_parquet(self, tmp_path):
        """Should cache results to parquet files."""
        from anndata import AnnData

        adata = AnnData(np.random.rand(20, 10))
        adata.obs["subject_id"] = ["S1"] * 10 + ["S2"] * 10
        adata.obs["supercluster_name"] = (
            ["TypeA"] * 5 + ["TypeB"] * 5 + ["TypeA"] * 5 + ["TypeB"] * 5
        )

        mock_li = MagicMock()
        call_count = [0]

        def fake_rank_aggregate(adata_arg, **kwargs):
            call_count[0] += 1
            adata_arg.uns["liana_res"] = pd.DataFrame({
                "source": ["TypeA"], "target": ["TypeB"],
                "magnitude_rank": [0.1],
            })

        mock_li.mt.rank_aggregate = MagicMock(side_effect=fake_rank_aggregate)

        cache_dir = tmp_path / "liana_cache"

        with patch.dict(sys.modules, {"liana": mock_li}):
            from src.data.liana_processing import run_liana_per_subject

            # First run should compute and cache
            results1 = run_liana_per_subject(
                adata,
                subject_column="subject_id",
                min_cells_per_type=5,
                cache_dir=cache_dir,
                verbose=False,
            )
            first_call_count = call_count[0]

            # Second run should load from cache
            results2 = run_liana_per_subject(
                adata,
                subject_column="subject_id",
                min_cells_per_type=5,
                cache_dir=cache_dir,
                verbose=False,
            )

        # Cache files should exist
        assert (cache_dir / "liana_S1.parquet").exists()
        assert (cache_dir / "liana_S2.parquet").exists()

        # Second run should not have called rank_aggregate again
        assert call_count[0] == first_call_count

    def test_handles_analysis_error_gracefully(self):
        """Should store empty DataFrame on error and continue to next subject."""
        from anndata import AnnData

        adata = AnnData(np.random.rand(20, 10))
        adata.obs["subject_id"] = ["S1"] * 10 + ["S2"] * 10
        adata.obs["supercluster_name"] = (
            ["TypeA"] * 5 + ["TypeB"] * 5 + ["TypeA"] * 5 + ["TypeB"] * 5
        )

        mock_li = MagicMock()
        call_idx = [0]

        def fake_rank_aggregate(adata_arg, **kwargs):
            call_idx[0] += 1
            if call_idx[0] == 1:
                raise RuntimeError("LIANA analysis failed")
            adata_arg.uns["liana_res"] = pd.DataFrame({
                "source": ["TypeA"], "target": ["TypeB"],
                "magnitude_rank": [0.1],
            })

        mock_li.mt.rank_aggregate = MagicMock(side_effect=fake_rank_aggregate)

        with patch.dict(sys.modules, {"liana": mock_li}):
            from src.data.liana_processing import run_liana_per_subject
            results = run_liana_per_subject(
                adata,
                subject_column="subject_id",
                min_cells_per_type=5,
                verbose=False,
            )

        # Both subjects should have entries
        assert len(results) == 2
        # One should be empty (error), other should have data
        empty_count = sum(1 for v in results.values() if len(v) == 0)
        assert empty_count == 1

    def test_adds_subject_id_column(self):
        """Should add subject_id column to results."""
        from anndata import AnnData

        adata = AnnData(np.random.rand(20, 10))
        adata.obs["subject_id"] = ["S1"] * 10 + ["S2"] * 10
        adata.obs["supercluster_name"] = (
            ["TypeA"] * 5 + ["TypeB"] * 5 + ["TypeA"] * 5 + ["TypeB"] * 5
        )

        mock_li = MagicMock()

        def fake_rank_aggregate(adata_arg, **kwargs):
            adata_arg.uns["liana_res"] = pd.DataFrame({
                "source": ["TypeA"], "target": ["TypeB"],
                "magnitude_rank": [0.1],
            })

        mock_li.mt.rank_aggregate = MagicMock(side_effect=fake_rank_aggregate)

        with patch.dict(sys.modules, {"liana": mock_li}):
            from src.data.liana_processing import run_liana_per_subject
            results = run_liana_per_subject(
                adata,
                subject_column="subject_id",
                min_cells_per_type=5,
                verbose=False,
            )

        # Non-empty results should have subject_id column
        for subject_id, df in results.items():
            if len(df) > 0:
                assert "subject_id" in df.columns
                assert (df["subject_id"] == subject_id).all()
