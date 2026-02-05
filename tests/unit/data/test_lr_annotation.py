"""
Tests for ligand-receptor pair annotation functions.

Tests extract_lr_pairs_by_edge() and aggregate_lr_mapping_across_subjects()
which provide L-R pair context for high-attention edges in CCC analysis.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.liana_processing import (
    extract_lr_pairs_by_edge,
    aggregate_lr_mapping_across_subjects,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_liana_results() -> pd.DataFrame:
    """Create sample LIANA+ results for testing."""
    return pd.DataFrame({
        "source": [
            "Microglia", "Microglia", "Microglia",
            "Astrocyte", "Astrocyte",
            "Neuron", "Neuron",
        ],
        "target": [
            "Astrocyte", "Astrocyte", "Astrocyte",
            "Microglia", "Oligodendrocyte",
            "Astrocyte", "Microglia",
        ],
        "ligand_complex": [
            "IL1B", "TNF", "CCL2",
            "BDNF", "NGF",
            "NRXN1", "NLGN1",
        ],
        "receptor_complex": [
            "IL1R1", "TNFR1", "CCR2",
            "NTRK2", "NGFR",
            "NLGN1", "NRXN1",
        ],
        "magnitude_rank": [0.01, 0.05, 0.1, 0.02, 0.15, 0.03, 0.08],
    })


@pytest.fixture
def sample_liana_with_edge_types(sample_liana_results) -> pd.DataFrame:
    """Sample LIANA+ results with pre-assigned edge types."""
    df = sample_liana_results.copy()
    df["edge_type_name"] = [
        "Secreted_Signaling", "Secreted_Signaling", "Secreted_Signaling",
        "Secreted_Signaling", "Secreted_Signaling",
        "Cell_Cell_Contact", "Cell_Cell_Contact",
    ]
    return df


@pytest.fixture
def multi_subject_liana() -> dict[str, pd.DataFrame]:
    """Create LIANA+ results for multiple subjects."""
    # Subject 1 - has IL1B-IL1R1 and TNF-TNFR1
    subject1 = pd.DataFrame({
        "source": ["Microglia", "Microglia", "Astrocyte"],
        "target": ["Astrocyte", "Astrocyte", "Neuron"],
        "ligand_complex": ["IL1B", "TNF", "BDNF"],
        "receptor_complex": ["IL1R1", "TNFR1", "NTRK2"],
        "magnitude_rank": [0.01, 0.05, 0.02],
        "edge_type_name": ["Secreted_Signaling", "Secreted_Signaling", "Secreted_Signaling"],
    })

    # Subject 2 - has IL1B-IL1R1 and CCL2-CCR2 (no TNF-TNFR1)
    subject2 = pd.DataFrame({
        "source": ["Microglia", "Microglia", "Astrocyte"],
        "target": ["Astrocyte", "Astrocyte", "Neuron"],
        "ligand_complex": ["IL1B", "CCL2", "GDNF"],
        "receptor_complex": ["IL1R1", "CCR2", "RET"],
        "magnitude_rank": [0.02, 0.08, 0.03],
        "edge_type_name": ["Secreted_Signaling", "Secreted_Signaling", "Secreted_Signaling"],
    })

    # Subject 3 - has all three: IL1B-IL1R1, TNF-TNFR1, CCL2-CCR2
    subject3 = pd.DataFrame({
        "source": ["Microglia", "Microglia", "Microglia"],
        "target": ["Astrocyte", "Astrocyte", "Astrocyte"],
        "ligand_complex": ["IL1B", "TNF", "CCL2"],
        "receptor_complex": ["IL1R1", "TNFR1", "CCR2"],
        "magnitude_rank": [0.01, 0.03, 0.06],
        "edge_type_name": ["Secreted_Signaling", "Secreted_Signaling", "Secreted_Signaling"],
    })

    return {"subject1": subject1, "subject2": subject2, "subject3": subject3}


# ─────────────────────────────────────────────────────────────────────────────
# Tests for extract_lr_pairs_by_edge
# ─────────────────────────────────────────────────────────────────────────────


class TestExtractLRPairsByEdge:
    """Test extract_lr_pairs_by_edge function."""

    def test_basic_extraction(self, sample_liana_with_edge_types):
        """Test basic L-R pair extraction."""
        lr_mapping = extract_lr_pairs_by_edge(sample_liana_with_edge_types)

        # Should have entries for the edges
        assert len(lr_mapping) > 0

        # Check Microglia -> Astrocyte Secreted_Signaling edge
        mic_ast_key = "Microglia|Astrocyte|Secreted_Signaling"
        assert mic_ast_key in lr_mapping

        # Should have 3 L-R pairs for this edge
        lr_pairs = lr_mapping[mic_ast_key]
        assert len(lr_pairs) == 3
        assert "IL1B_IL1R1" in lr_pairs
        assert "TNF_TNFR1" in lr_pairs
        assert "CCL2_CCR2" in lr_pairs

    def test_ordering_by_magnitude(self, sample_liana_with_edge_types):
        """Test that L-R pairs are ordered by magnitude (best first)."""
        lr_mapping = extract_lr_pairs_by_edge(sample_liana_with_edge_types)

        mic_ast_key = "Microglia|Astrocyte|Secreted_Signaling"
        lr_pairs = lr_mapping[mic_ast_key]

        # IL1B_IL1R1 has magnitude_rank 0.01 (best)
        # TNF_TNFR1 has magnitude_rank 0.05
        # CCL2_CCR2 has magnitude_rank 0.1
        assert lr_pairs[0] == "IL1B_IL1R1"
        assert lr_pairs[1] == "TNF_TNFR1"
        assert lr_pairs[2] == "CCL2_CCR2"

    def test_max_pairs_per_edge(self, sample_liana_with_edge_types):
        """Test max_pairs_per_edge limit."""
        lr_mapping = extract_lr_pairs_by_edge(
            sample_liana_with_edge_types,
            max_pairs_per_edge=2
        )

        mic_ast_key = "Microglia|Astrocyte|Secreted_Signaling"
        lr_pairs = lr_mapping[mic_ast_key]

        # Should be limited to 2
        assert len(lr_pairs) == 2
        # Should keep the best ones
        assert "IL1B_IL1R1" in lr_pairs
        assert "TNF_TNFR1" in lr_pairs
        assert "CCL2_CCR2" not in lr_pairs  # Excluded (worst magnitude)

    def test_cell_type_filtering(self, sample_liana_with_edge_types):
        """Test filtering by cell types."""
        # Only include Microglia and Astrocyte
        lr_mapping = extract_lr_pairs_by_edge(
            sample_liana_with_edge_types,
            cell_types=["Microglia", "Astrocyte"]
        )

        # Should have Microglia -> Astrocyte
        assert "Microglia|Astrocyte|Secreted_Signaling" in lr_mapping

        # Should have Astrocyte -> Microglia
        assert "Astrocyte|Microglia|Secreted_Signaling" in lr_mapping

        # Should NOT have edges involving Oligodendrocyte or Neuron
        for key in lr_mapping.keys():
            assert "Oligodendrocyte" not in key
            assert "Neuron" not in key

    def test_multiple_edge_types(self, sample_liana_with_edge_types):
        """Test extraction with multiple edge types."""
        lr_mapping = extract_lr_pairs_by_edge(sample_liana_with_edge_types)

        # Should have both edge types
        secreted_edges = [k for k in lr_mapping if "Secreted_Signaling" in k]
        contact_edges = [k for k in lr_mapping if "Cell_Cell_Contact" in k]

        assert len(secreted_edges) > 0
        assert len(contact_edges) > 0

    def test_no_duplicates(self, sample_liana_with_edge_types):
        """Test that L-R pairs are not duplicated within an edge."""
        # Add duplicate entry
        df = sample_liana_with_edge_types.copy()
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

        lr_mapping = extract_lr_pairs_by_edge(df)

        # Should still have no duplicates
        for lr_pairs in lr_mapping.values():
            assert len(lr_pairs) == len(set(lr_pairs))

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        empty_df = pd.DataFrame({
            "source": [],
            "target": [],
            "ligand_complex": [],
            "receptor_complex": [],
            "magnitude_rank": [],
        })

        lr_mapping = extract_lr_pairs_by_edge(empty_df)
        assert lr_mapping == {}

    def test_missing_ligand_receptor(self):
        """Test handling of missing ligand/receptor values."""
        df = pd.DataFrame({
            "source": ["Microglia", "Microglia", "Microglia"],
            "target": ["Astrocyte", "Astrocyte", "Astrocyte"],
            "ligand_complex": ["IL1B", None, "CCL2"],
            "receptor_complex": ["IL1R1", "TNFR1", None],
            "magnitude_rank": [0.01, 0.05, 0.1],
            "edge_type_name": ["Secreted_Signaling", "Secreted_Signaling", "Secreted_Signaling"],
        })

        lr_mapping = extract_lr_pairs_by_edge(df)

        # Should only have the valid L-R pair
        mic_ast_key = "Microglia|Astrocyte|Secreted_Signaling"
        assert mic_ast_key in lr_mapping
        assert len(lr_mapping[mic_ast_key]) == 1
        assert lr_mapping[mic_ast_key][0] == "IL1B_IL1R1"

    def test_edge_key_format(self, sample_liana_with_edge_types):
        """Test that edge keys follow correct format."""
        lr_mapping = extract_lr_pairs_by_edge(sample_liana_with_edge_types)

        for edge_key in lr_mapping.keys():
            parts = edge_key.split("|")
            assert len(parts) == 3  # source|target|edge_type
            # Parts should be non-empty
            assert all(p for p in parts)

    def test_edge_key_sanitizes_cell_type_names(self):
        """Cell type names with special characters are sanitized in edge keys (F3)."""
        df = pd.DataFrame({
            "source": ["Upper-layer intratelencephalic", "L6b/CT"],
            "target": ["Microglia", "Upper-layer intratelencephalic"],
            "ligand_complex": ["IL1B", "BDNF"],
            "receptor_complex": ["IL1R1", "NTRK2"],
            "magnitude_rank": [0.01, 0.02],
            "edge_type_name": ["Secreted_Signaling", "Secreted_Signaling"],
        })

        lr_mapping = extract_lr_pairs_by_edge(df)

        # Hyphens and slashes should be sanitized to underscores
        assert "Upper_layer_intratelencephalic|Microglia|Secreted_Signaling" in lr_mapping
        assert "L6b_CT|Upper_layer_intratelencephalic|Secreted_Signaling" in lr_mapping

        # Raw (unsanitized) keys should NOT be present
        assert "Upper-layer intratelencephalic|Microglia|Secreted_Signaling" not in lr_mapping
        assert "L6b/CT|Upper-layer intratelencephalic|Secreted_Signaling" not in lr_mapping


# ─────────────────────────────────────────────────────────────────────────────
# Tests for aggregate_lr_mapping_across_subjects
# ─────────────────────────────────────────────────────────────────────────────


class TestAggregateLRMappingAcrossSubjects:
    """Test aggregate_lr_mapping_across_subjects function."""

    def test_basic_aggregation(self, multi_subject_liana):
        """Test basic aggregation across subjects."""
        # Extract L-R mappings for each subject
        subject_mappings = {
            subj: extract_lr_pairs_by_edge(df)
            for subj, df in multi_subject_liana.items()
        }

        aggregated = aggregate_lr_mapping_across_subjects(subject_mappings)

        # Should have the Microglia->Astrocyte edge
        mic_ast_key = "Microglia|Astrocyte|Secreted_Signaling"
        assert mic_ast_key in aggregated

    def test_frequency_ordering(self, multi_subject_liana):
        """Test that L-R pairs are ordered by frequency."""
        subject_mappings = {
            subj: extract_lr_pairs_by_edge(df)
            for subj, df in multi_subject_liana.items()
        }

        aggregated = aggregate_lr_mapping_across_subjects(subject_mappings)

        mic_ast_key = "Microglia|Astrocyte|Secreted_Signaling"
        lr_pairs = aggregated[mic_ast_key]

        # IL1B_IL1R1 appears in all 3 subjects (frequency 3) - should be first
        assert lr_pairs[0] == "IL1B_IL1R1"

        # CCL2_CCR2 and TNF_TNFR1 appear in 2 subjects each
        # They should come after IL1B_IL1R1
        assert "TNF_TNFR1" in lr_pairs
        assert "CCL2_CCR2" in lr_pairs

    def test_min_subjects_filter(self, multi_subject_liana):
        """Test min_subjects filtering."""
        subject_mappings = {
            subj: extract_lr_pairs_by_edge(df)
            for subj, df in multi_subject_liana.items()
        }

        # Require L-R pair to appear in at least 3 subjects
        aggregated = aggregate_lr_mapping_across_subjects(
            subject_mappings,
            min_subjects=3
        )

        mic_ast_key = "Microglia|Astrocyte|Secreted_Signaling"
        lr_pairs = aggregated.get(mic_ast_key, [])

        # Only IL1B_IL1R1 appears in all 3 subjects
        assert "IL1B_IL1R1" in lr_pairs
        # TNF_TNFR1 only appears in 2 subjects - should be excluded
        assert "TNF_TNFR1" not in lr_pairs
        # CCL2_CCR2 only appears in 2 subjects - should be excluded
        assert "CCL2_CCR2" not in lr_pairs

    def test_min_subjects_two(self, multi_subject_liana):
        """Test min_subjects=2 includes pairs in 2+ subjects."""
        subject_mappings = {
            subj: extract_lr_pairs_by_edge(df)
            for subj, df in multi_subject_liana.items()
        }

        aggregated = aggregate_lr_mapping_across_subjects(
            subject_mappings,
            min_subjects=2
        )

        mic_ast_key = "Microglia|Astrocyte|Secreted_Signaling"
        lr_pairs = aggregated[mic_ast_key]

        # All three L-R pairs appear in >= 2 subjects
        assert "IL1B_IL1R1" in lr_pairs  # 3 subjects
        assert "TNF_TNFR1" in lr_pairs   # 2 subjects
        assert "CCL2_CCR2" in lr_pairs   # 2 subjects

    def test_empty_input(self):
        """Test with empty input."""
        aggregated = aggregate_lr_mapping_across_subjects({})
        assert aggregated == {}

    def test_single_subject(self):
        """Test with single subject."""
        single_mapping = {
            "subject1": {
                "Microglia|Astrocyte|Secreted_Signaling": ["IL1B_IL1R1", "TNF_TNFR1"]
            }
        }

        aggregated = aggregate_lr_mapping_across_subjects(single_mapping)

        # Should include all pairs (min_subjects=1 by default)
        mic_ast_key = "Microglia|Astrocyte|Secreted_Signaling"
        assert mic_ast_key in aggregated
        assert "IL1B_IL1R1" in aggregated[mic_ast_key]
        assert "TNF_TNFR1" in aggregated[mic_ast_key]

    def test_no_overlap_between_subjects(self):
        """Test when subjects have no overlapping edges."""
        mappings = {
            "subject1": {"Microglia|Astrocyte|Secreted_Signaling": ["IL1B_IL1R1"]},
            "subject2": {"Neuron|Oligodendrocyte|Cell_Cell_Contact": ["NRXN1_NLGN1"]},
        }

        aggregated = aggregate_lr_mapping_across_subjects(mappings)

        # Both edges should be included
        assert "Microglia|Astrocyte|Secreted_Signaling" in aggregated
        assert "Neuron|Oligodendrocyte|Cell_Cell_Contact" in aggregated

    def test_alphabetical_tiebreaker(self):
        """Test that ties in frequency are broken alphabetically."""
        # Create mappings where two L-R pairs have same frequency
        mappings = {
            "subject1": {"A|B|X": ["ZZZ_gene", "AAA_gene"]},
            "subject2": {"A|B|X": ["ZZZ_gene", "AAA_gene"]},
        }

        aggregated = aggregate_lr_mapping_across_subjects(mappings)

        # Both have frequency 2, should be ordered alphabetically
        lr_pairs = aggregated["A|B|X"]
        assert lr_pairs == ["AAA_gene", "ZZZ_gene"]


# ─────────────────────────────────────────────────────────────────────────────
# Integration Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestLRAnnotationIntegration:
    """Integration tests for L-R annotation workflow."""

    def test_full_workflow(self, multi_subject_liana):
        """Test complete workflow from LIANA results to aggregated mapping."""
        # Step 1: Extract L-R pairs for each subject
        subject_mappings = {}
        for subject_id, liana_df in multi_subject_liana.items():
            subject_mappings[subject_id] = extract_lr_pairs_by_edge(
                liana_df,
                max_pairs_per_edge=5
            )

        # Step 2: Aggregate across subjects
        consensus = aggregate_lr_mapping_across_subjects(
            subject_mappings,
            min_subjects=2
        )

        # Verify results
        assert len(consensus) > 0

        # Check that most frequent pairs are included
        mic_ast_key = "Microglia|Astrocyte|Secreted_Signaling"
        if mic_ast_key in consensus:
            assert "IL1B_IL1R1" in consensus[mic_ast_key]

    def test_workflow_with_cell_type_filter(self, multi_subject_liana):
        """Test workflow with cell type filtering."""
        cell_types = ["Microglia", "Astrocyte"]

        subject_mappings = {}
        for subject_id, liana_df in multi_subject_liana.items():
            subject_mappings[subject_id] = extract_lr_pairs_by_edge(
                liana_df,
                cell_types=cell_types
            )

        consensus = aggregate_lr_mapping_across_subjects(subject_mappings)

        # Should only have edges between Microglia and Astrocyte
        for edge_key in consensus.keys():
            parts = edge_key.split("|")
            assert parts[0] in cell_types
            assert parts[1] in cell_types
