"""
Tests for src/data/liana_processing.py

Tests cover:
- Edge type mapping from CellChatDB categories
- Building CCC features from LIANA+ results
- Empty input handling
- LIANA result filtering and aggregation
- Adjacency matrix conversion
"""

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
        from src.data.constants import NOVEL_CATEGORY

        result = assign_edge_types(
            mock_liana_results,
            cellchatdb_path=mock_cellchatdb_csv
        )

        # FAKE_LIGAND_FAKE_RECEPTOR is not in DB
        fake_row = result[result["ligand_complex"] == "FAKE_LIGAND"].iloc[0]
        assert fake_row["edge_type_name"] == NOVEL_CATEGORY

    def test_edge_type_indices_consistent(self, mock_liana_results, mock_cellchatdb_csv):
        """Edge type indices should match category order."""
        from src.data.liana_processing import assign_edge_types
        from src.data.constants import CELLCHATDB_CATEGORIES, NOVEL_CATEGORY

        result = assign_edge_types(
            mock_liana_results,
            cellchatdb_path=mock_cellchatdb_csv
        )

        categories = CELLCHATDB_CATEGORIES + [NOVEL_CATEGORY]
        category_to_idx = {cat: idx for idx, cat in enumerate(categories)}

        for _, row in result.iterrows():
            expected_idx = category_to_idx[row["edge_type_name"]]
            assert row["edge_type"] == expected_idx

    def test_handles_missing_db_file(self, mock_liana_results, tmp_path, capsys):
        """Should handle missing CellChatDB file gracefully."""
        from src.data.liana_processing import assign_edge_types
        from src.data.constants import NOVEL_CATEGORY

        # Use non-existent path
        result = assign_edge_types(
            mock_liana_results,
            cellchatdb_path=tmp_path / "nonexistent.csv"
        )

        # All interactions should get novel category
        assert (result["edge_type_name"] == NOVEL_CATEGORY).all()

        # Should print warning
        captured = capsys.readouterr()
        assert "Warning" in captured.out

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

        # Should keep the minimum (0.02, not 0.5)
        astro_idx = cell_types.index("Astrocyte")
        micro_idx = cell_types.index("Microglia")
        assert adj[astro_idx, micro_idx] == 0.02

    def test_handles_unknown_cell_types(self, mock_liana_results):
        """Should skip interactions with unknown cell types."""
        from src.data.liana_processing import liana_to_adjacency_matrix

        # Cell types list doesn't include Oligodendrocyte
        cell_types = ["Astrocyte", "Microglia"]

        adj = liana_to_adjacency_matrix(mock_liana_results, cell_types)

        # Should only have Astrocyte -> Microglia interaction
        assert adj.shape == (2, 2)


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
        """Should handle NaN magnitude_rank."""
        from src.data.liana_processing import build_subject_ccc_features, assign_edge_types

        df_with_nan = pd.DataFrame({
            "source": ["Astrocyte"],
            "target": ["Microglia"],
            "ligand_complex": ["TGFB1"],
            "receptor_complex": ["TGFBR1"],
            "magnitude_rank": [np.nan],
        })

        cell_types = ["Astrocyte", "Microglia"]
        df_with_types = assign_edge_types(df_with_nan, cellchatdb_path=mock_cellchatdb_csv)
        features = build_subject_ccc_features(df_with_types, cell_types)

        # Should still build features (NaN becomes edge attr)
        assert features["n_edges"] == 1

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
