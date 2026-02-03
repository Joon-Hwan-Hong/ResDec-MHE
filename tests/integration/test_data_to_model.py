"""
Integration tests for data pipeline → model forward pass.

Tests verify that:
- Dataset output format matches collate expectations
- Collate output format matches model input expectations
- End-to-end forward pass produces valid outputs
- Cross-branch dimension consistency for fusion
- Gradient flow through entire pipeline
- Edge cases across full pipeline
"""

import torch
import pytest

from src.data.constants import N_CELL_TYPES, N_REGIONS, CELL_TYPE_ORDER, ALL_EDGE_TYPES


def create_mock_dataset_sample(
    n_genes: int = 100,
    max_cells: int = 50,
    n_edges: int = 20,
) -> dict:
    """Create a sample matching CognitiveResilienceDataset output format."""
    return {
        "subject_id": "TEST_SUBJECT",
        "pseudobulk": torch.randn(N_CELL_TYPES, n_genes),
        "cell_type_mask": torch.ones(N_CELL_TYPES, dtype=torch.bool),
        "cell_counts": torch.randint(10, 100, (N_CELL_TYPES,)),
        "cells": torch.randn(N_CELL_TYPES, max_cells, n_genes),
        "cell_mask": torch.ones(N_CELL_TYPES, max_cells, dtype=torch.bool),
        "ccc_edge_index": torch.randint(0, N_CELL_TYPES, (2, n_edges)),
        "ccc_edge_type": torch.randint(0, 5, (n_edges,)),
        "ccc_edge_attr": torch.rand(n_edges, 1),
        "pathology": torch.rand(3),
        "cognition": torch.randn(1),
        "region_mask": torch.ones(N_REGIONS, dtype=torch.bool),
    }


def create_mock_sample_with_empty_cell_types(
    n_genes: int = 100,
    max_cells: int = 50,
    n_edges: int = 20,
    empty_types: list[int] = None,
) -> dict:
    """Create sample with some cell types having all cells masked."""
    sample = create_mock_dataset_sample(n_genes, max_cells, n_edges)
    if empty_types:
        for ct_idx in empty_types:
            sample["cell_mask"][ct_idx, :] = False
            sample["cell_type_mask"][ct_idx] = False
    return sample


class TestDatasetToCollate:
    """Test that dataset output format works with collate functions."""

    def test_collate_fn_accepts_dataset_format(self):
        """collate_fn should accept dataset output format."""
        from src.data.collate import collate_fn

        batch = [create_mock_dataset_sample() for _ in range(4)]
        result = collate_fn(batch)

        assert result["batch_size"] == 4
        assert result["pseudobulk"].shape[0] == 4
        assert result["cells"].shape[0] == 4

    def test_collate_for_hgt_accepts_dataset_format(self):
        """collate_for_hgt should accept dataset output format."""
        from src.data.collate import collate_for_hgt

        batch = [create_mock_dataset_sample() for _ in range(4)]
        result = collate_for_hgt(batch)

        assert result["batch_size"] == 4
        assert "edge_index_dict_list" in result
        assert "edge_attr_dict_list" in result
        assert len(result["edge_index_dict_list"]) == 4


class TestCollateToCellTransformer:
    """Test that collate output works with CellTransformer."""

    def test_cell_transformer_accepts_collate_output(self):
        """CellTransformer should accept cells from collate output."""
        from src.data.collate import collate_fn
        from src.models.branches.cell_transformer import CellTransformer

        n_genes = 100
        batch = [create_mock_dataset_sample(n_genes=n_genes) for _ in range(4)]
        collated = collate_fn(batch)

        # Create CellTransformer with matching dimensions
        transformer = CellTransformer(
            n_genes=n_genes,
            n_cell_types=N_CELL_TYPES,
            d_model=64,
            n_heads=4,
            n_isab_layers=2,
            n_inducing=16,
        )

        # Forward pass should work
        embeddings, selection_weights, _ = transformer(
            collated["cells"],
            collated["cell_mask"],
        )

        # Verify output shapes
        assert embeddings.shape == (4, N_CELL_TYPES, 64)
        assert selection_weights.shape == (N_CELL_TYPES,)

    def test_cell_transformer_gradient_flow_from_collate(self):
        """Gradients should flow through CellTransformer with collate data."""
        from src.data.collate import collate_fn
        from src.models.branches.cell_transformer import CellTransformer

        n_genes = 100
        batch = [create_mock_dataset_sample(n_genes=n_genes) for _ in range(4)]
        collated = collate_fn(batch)

        transformer = CellTransformer(
            n_genes=n_genes,
            n_cell_types=N_CELL_TYPES,
            d_model=64,
            n_heads=4,
            n_isab_layers=2,
            n_inducing=16,
        )

        # Enable gradients for input
        cells = collated["cells"].clone().requires_grad_(True)

        embeddings, _, _ = transformer(cells, collated["cell_mask"])
        loss = embeddings.sum()
        loss.backward()

        # Gradients should flow
        assert cells.grad is not None
        assert not torch.all(cells.grad == 0)


class TestEndToEndPipeline:
    """Test complete pipeline from dataset format to model output."""

    def test_full_pipeline_produces_valid_output(self):
        """Full pipeline: mock dataset → collate → CellTransformer."""
        from src.data.collate import collate_fn
        from src.models.branches.cell_transformer import CellTransformer

        n_genes = 100
        d_model = 64

        # Simulate dataset batch
        batch = [create_mock_dataset_sample(n_genes=n_genes) for _ in range(8)]

        # Collate
        collated = collate_fn(batch)

        # CellTransformer forward
        transformer = CellTransformer(
            n_genes=n_genes,
            n_cell_types=N_CELL_TYPES,
            d_model=d_model,
            n_heads=4,
            n_isab_layers=2,
            n_inducing=16,
        )

        embeddings, selection_weights, attention = transformer(
            collated["cells"],
            collated["cell_mask"],
            return_attention=True,
        )

        # Verify outputs are valid
        assert torch.isfinite(embeddings).all()
        assert torch.isfinite(selection_weights).all()
        assert attention is not None
        assert len(attention) == N_CELL_TYPES

    def test_pipeline_with_sparse_data(self):
        """Pipeline should handle sparse cell data correctly."""
        from src.data.collate import collate_fn
        from src.models.branches.cell_transformer import CellTransformer

        n_genes = 100

        # Create samples with sparse cell masks
        samples = []
        for i in range(4):
            sample = create_mock_dataset_sample(n_genes=n_genes, max_cells=100)
            # Make cell mask sparse - only 10 valid cells per type
            sample["cell_mask"] = torch.zeros(N_CELL_TYPES, 100, dtype=torch.bool)
            sample["cell_mask"][:, :10] = True
            samples.append(sample)

        collated = collate_fn(samples)

        transformer = CellTransformer(
            n_genes=n_genes,
            n_cell_types=N_CELL_TYPES,
            d_model=64,
            n_heads=4,
            n_isab_layers=2,
            n_inducing=16,
        )

        embeddings, _, _ = transformer(collated["cells"], collated["cell_mask"])

        # Should produce valid outputs despite sparse data
        assert torch.isfinite(embeddings).all()

    def test_pipeline_with_mixed_valid_cells(self):
        """Pipeline handles samples with different numbers of valid cells."""
        from src.data.collate import collate_fn
        from src.models.branches.cell_transformer import CellTransformer

        n_genes = 100

        samples = []
        for i in range(4):
            sample = create_mock_dataset_sample(n_genes=n_genes, max_cells=100)
            # Vary number of valid cells per sample
            n_valid = (i + 1) * 20  # 20, 40, 60, 80
            sample["cell_mask"] = torch.zeros(N_CELL_TYPES, 100, dtype=torch.bool)
            sample["cell_mask"][:, :n_valid] = True
            samples.append(sample)

        collated = collate_fn(samples)

        transformer = CellTransformer(
            n_genes=n_genes,
            n_cell_types=N_CELL_TYPES,
            d_model=64,
            n_heads=4,
            n_isab_layers=2,
            n_inducing=16,
        )

        embeddings, _, _ = transformer(collated["cells"], collated["cell_mask"])

        # All samples should have valid embeddings
        assert embeddings.shape[0] == 4
        assert torch.isfinite(embeddings).all()


# =============================================================================
# HGT Encoder Integration Tests
# =============================================================================


class TestCollateToHGTEncoder:
    """Test that collate_for_hgt output works with HGTEncoder and HGTEncoderBatched."""

    def test_collate_for_hgt_produces_valid_format(self):
        """collate_for_hgt should produce dict lists compatible with HGTEncoderBatched."""
        from src.data.collate import collate_for_hgt, build_x_dict_list_from_embeddings
        from src.models.branches.hgt_encoder import HGTEncoder

        d_input = 100
        d_hidden = 64

        batch = [create_mock_dataset_sample(n_genes=d_input, n_edges=30) for _ in range(4)]
        collated = collate_for_hgt(batch)

        # Verify dict list structure
        assert "edge_index_dict_list" in collated
        assert "edge_attr_dict_list" in collated
        assert len(collated["edge_index_dict_list"]) == 4
        assert len(collated["edge_attr_dict_list"]) == 4

        # Build x_dict_list from pseudobulk for standalone HGT testing
        x_dict_list = build_x_dict_list_from_embeddings(
            collated["pseudobulk"], collated["node_types"]
        )

        # Verify each sample's x_dict has all cell types
        assert len(x_dict_list) == 4
        for x_dict in x_dict_list:
            assert len(x_dict) == N_CELL_TYPES
            for node_type, x in x_dict.items():
                assert x.shape == (1, d_input)

        # Verify encoder can be created with correct structure
        encoder = HGTEncoder(
            d_input=d_input,
            d_hidden=d_hidden,
            d_output=d_hidden,
            n_heads=4,
            n_layers=2,
        )
        assert encoder.n_node_types == N_CELL_TYPES
        assert encoder.n_edge_types == len(ALL_EDGE_TYPES)

    def test_hgt_encoder_batched_with_collate_for_hgt(self):
        """End-to-end: collate_for_hgt output should run through HGTEncoderBatched."""
        from src.data.collate import collate_for_hgt, build_x_dict_list_from_embeddings
        from src.models.branches.hgt_encoder import HGTEncoderBatched

        d_input = 100
        d_hidden = 64
        d_output = 64

        # Create batch and collate
        batch = [create_mock_dataset_sample(n_genes=d_input, n_edges=30) for _ in range(4)]
        collated = collate_for_hgt(batch)

        # Build x_dict_list from pseudobulk (as the full model would after encoding)
        x_dict_list = build_x_dict_list_from_embeddings(
            collated["pseudobulk"], collated["node_types"]
        )

        # Create encoder with sanitized node types from collate
        encoder = HGTEncoderBatched(
            d_input=d_input,
            d_hidden=d_hidden,
            d_output=d_output,
            n_heads=4,
            n_layers=2,
            node_types=collated["node_types"],
            edge_categories=collated["edge_types"],
        )

        # Forward pass using built x_dict_list and collate edge dicts
        output_dict, attention = encoder(
            x_dict_list,
            collated["edge_index_dict_list"],
            collated["edge_attr_dict_list"],
            return_attention=True,
        )

        # Verify outputs
        assert len(output_dict) == N_CELL_TYPES
        for node_type, out in output_dict.items():
            assert out.shape == (4, 1, d_output)  # (batch, n_nodes, d_output)
            assert torch.isfinite(out).all()

        assert attention is not None
        assert len(attention) == 4  # One per batch sample

    def test_hgt_encoder_forward_with_manual_dict_format(self):
        """Test HGT forward pass with manually constructed dict format."""
        from src.models.branches.hgt_encoder import HGTEncoder

        d_input = 64
        d_hidden = 32
        d_output = 32

        encoder = HGTEncoder(
            d_input=d_input,
            d_hidden=d_hidden,
            d_output=d_output,
            n_heads=4,
            n_layers=2,
        )

        # Create x_dict with one node per cell type
        x_dict = {
            ct: torch.randn(1, d_input) for ct in CELL_TYPE_ORDER
        }

        # Create some edges
        edge_index_dict = {}
        edge_attr_dict = {}

        # Add edges between first few cell types
        src_ct = CELL_TYPE_ORDER[0]
        dst_ct = CELL_TYPE_ORDER[1]
        relation = ALL_EDGE_TYPES[0]
        triplet = (src_ct, relation, dst_ct)

        edge_index_dict[triplet] = torch.tensor([[0], [0]], dtype=torch.long)
        edge_attr_dict[triplet] = torch.rand(1, 1)

        # Forward pass
        output_dict, attention = encoder(x_dict, edge_index_dict, edge_attr_dict)

        # Verify outputs
        assert len(output_dict) == N_CELL_TYPES
        for ct, out in output_dict.items():
            assert out.shape == (1, d_output)
            assert torch.isfinite(out).all()

    def test_hgt_encoder_with_sanitized_node_names(self):
        """HGTEncoder should handle sanitized node names from collate_for_hgt.

        This tests the fix for the bug where HGT residual update iterated over
        unsanitized self.node_types but h_dict had sanitized keys, causing most
        cell types to be silently skipped.
        """
        from src.models.branches.hgt_encoder import HGTEncoder

        d_input = 64
        d_hidden = 32
        d_output = 32

        encoder = HGTEncoder(
            d_input=d_input,
            d_hidden=d_hidden,
            d_output=d_output,
            n_heads=4,
            n_layers=2,
        )

        # Sanitize function matching collate_for_hgt
        def sanitize_name(name: str) -> str:
            return name.replace(" ", "_").replace("/", "_").replace("-", "_")

        # Create x_dict with SANITIZED keys (as collate_for_hgt does)
        sanitized_cell_types = [sanitize_name(ct) for ct in CELL_TYPE_ORDER]
        x_dict = {
            sanitized_ct: torch.randn(1, d_input)
            for sanitized_ct in sanitized_cell_types
        }

        # Create edges with sanitized triplets
        edge_index_dict = {}
        edge_attr_dict = {}

        src_ct = sanitize_name(CELL_TYPE_ORDER[0])  # e.g., "Astrocyte"
        dst_ct = sanitize_name(CELL_TYPE_ORDER[1])  # e.g., "Oligodendrocyte"
        relation = ALL_EDGE_TYPES[0]  # Already uses underscores
        triplet = (src_ct, relation, dst_ct)

        edge_index_dict[triplet] = torch.tensor([[0], [0]], dtype=torch.long)
        edge_attr_dict[triplet] = torch.rand(1, 1)

        # Forward pass with sanitized keys
        output_dict, attention = encoder(x_dict, edge_index_dict, edge_attr_dict)

        # CRITICAL: Verify ALL cell types were processed (not silently skipped)
        assert len(output_dict) == N_CELL_TYPES, (
            f"Expected {N_CELL_TYPES} outputs but got {len(output_dict)}. "
            "This indicates sanitized node names are not being handled correctly."
        )

        # Verify output keys match input keys (sanitized)
        for sanitized_ct in sanitized_cell_types:
            assert sanitized_ct in output_dict, f"Missing output for {sanitized_ct}"
            assert output_dict[sanitized_ct].shape == (1, d_output)
            assert torch.isfinite(output_dict[sanitized_ct]).all()

    def test_hgt_encoder_gradient_flow_with_sanitized_names(self):
        """Gradients should flow correctly with sanitized node names."""
        from src.models.branches.hgt_encoder import HGTEncoder

        d_input = 64
        d_hidden = 32
        d_output = 32

        encoder = HGTEncoder(
            d_input=d_input,
            d_hidden=d_hidden,
            d_output=d_output,
            n_heads=4,
            n_layers=2,
        )

        def sanitize_name(name: str) -> str:
            return name.replace(" ", "_").replace("/", "_").replace("-", "_")

        # Create x_dict with sanitized keys and gradient tracking
        sanitized_cell_types = [sanitize_name(ct) for ct in CELL_TYPE_ORDER]
        x_dict = {
            sanitized_ct: torch.randn(1, d_input, requires_grad=True)
            for sanitized_ct in sanitized_cell_types
        }

        # Create edges
        src_ct = sanitized_cell_types[0]
        dst_ct = sanitized_cell_types[1]
        triplet = (src_ct, ALL_EDGE_TYPES[0], dst_ct)

        edge_index_dict = {triplet: torch.tensor([[0], [0]], dtype=torch.long)}
        edge_attr_dict = {triplet: torch.rand(1, 1)}

        # Forward pass
        output_dict, _ = encoder(x_dict, edge_index_dict, edge_attr_dict)

        # Compute loss and backward
        loss = sum(out.sum() for out in output_dict.values())
        loss.backward()

        # Verify gradients flow to ALL input cell types
        for sanitized_ct in sanitized_cell_types:
            assert x_dict[sanitized_ct].grad is not None, (
                f"No gradient for {sanitized_ct}"
            )


# =============================================================================
# Pseudobulk Encoder Integration Tests
# =============================================================================


class TestCollateToPseudobulkEncoder:
    """Test that collate output works with PseudobulkEncoder."""

    def test_pseudobulk_encoder_accepts_collate_output(self):
        """PseudobulkEncoder should accept pseudobulk from collate."""
        from src.data.collate import collate_fn
        from src.models.branches.pseudobulk_encoder import PseudobulkEncoder

        n_genes = 100
        d_embed = 64

        batch = [create_mock_dataset_sample(n_genes=n_genes) for _ in range(4)]
        collated = collate_fn(batch)

        encoder = PseudobulkEncoder(
            n_genes=n_genes,
            n_cell_types=N_CELL_TYPES,
            d_embed=d_embed,
        )

        # Forward pass
        embeddings = encoder(collated["pseudobulk"])

        # Verify output shape: [batch, n_cell_types, d_embed]
        assert embeddings.shape == (4, N_CELL_TYPES, d_embed)
        assert torch.isfinite(embeddings).all()

    def test_pseudobulk_encoder_gradient_flow(self):
        """Gradients should flow through PseudobulkEncoder."""
        from src.data.collate import collate_fn
        from src.models.branches.pseudobulk_encoder import PseudobulkEncoder

        n_genes = 100
        batch = [create_mock_dataset_sample(n_genes=n_genes) for _ in range(4)]
        collated = collate_fn(batch)

        encoder = PseudobulkEncoder(
            n_genes=n_genes,
            n_cell_types=N_CELL_TYPES,
            d_embed=64,
        )

        pseudobulk = collated["pseudobulk"].clone().requires_grad_(True)
        embeddings = encoder(pseudobulk)
        loss = embeddings.sum()
        loss.backward()

        assert pseudobulk.grad is not None
        assert not torch.all(pseudobulk.grad == 0)


# =============================================================================
# Cross-Branch Consistency Tests
# =============================================================================


class TestCrossBranchConsistency:
    """Test that all branches produce compatible outputs for fusion."""

    def test_all_branches_same_embed_dimension(self):
        """All branches should output the same embedding dimension."""
        from src.data.collate import collate_fn
        from src.models.branches.cell_transformer import CellTransformer
        from src.models.branches.pseudobulk_encoder import PseudobulkEncoder
        from src.models.branches.hgt_encoder import HGTEncoder

        n_genes = 100
        d_embed = 64  # Same for all branches

        batch = [create_mock_dataset_sample(n_genes=n_genes) for _ in range(4)]
        collated = collate_fn(batch)

        # Branch 1: Pseudobulk
        pb_encoder = PseudobulkEncoder(
            n_genes=n_genes,
            n_cell_types=N_CELL_TYPES,
            d_embed=d_embed,
        )
        pb_out = pb_encoder(collated["pseudobulk"])

        # Branch 3: CellTransformer
        cell_transformer = CellTransformer(
            n_genes=n_genes,
            n_cell_types=N_CELL_TYPES,
            d_model=d_embed,
            n_heads=4,
            n_isab_layers=2,
            n_inducing=16,
        )
        cell_out, _, _ = cell_transformer(collated["cells"], collated["cell_mask"])

        # Branch 2: HGT (verify output dimension matches)
        hgt_encoder = HGTEncoder(
            d_input=d_embed,  # Takes embeddings as input
            d_hidden=d_embed,
            d_output=d_embed,
            n_heads=4,
            n_layers=2,
        )

        # Verify dimensions match for fusion
        assert pb_out.shape[-1] == d_embed
        assert cell_out.shape[-1] == d_embed
        assert hgt_encoder.d_output == d_embed

    def test_all_branches_same_cell_type_dimension(self):
        """All branches should have same n_cell_types dimension."""
        from src.data.collate import collate_fn
        from src.models.branches.cell_transformer import CellTransformer
        from src.models.branches.pseudobulk_encoder import PseudobulkEncoder

        n_genes = 100
        batch = [create_mock_dataset_sample(n_genes=n_genes) for _ in range(4)]
        collated = collate_fn(batch)

        pb_encoder = PseudobulkEncoder(n_genes=n_genes, n_cell_types=N_CELL_TYPES)
        cell_transformer = CellTransformer(n_genes=n_genes, n_cell_types=N_CELL_TYPES)

        pb_out = pb_encoder(collated["pseudobulk"])
        cell_out, _, _ = cell_transformer(collated["cells"], collated["cell_mask"])

        # Both should have n_cell_types in dimension 1
        assert pb_out.shape[1] == N_CELL_TYPES
        assert cell_out.shape[1] == N_CELL_TYPES


# =============================================================================
# DataLoader Integration Tests
# =============================================================================


class TestDataLoaderIntegration:
    """Test DataLoader with collate functions."""

    def test_create_dataloader_with_collate_fn(self):
        """create_dataloader should work with custom dataset."""
        from src.data.collate import collate_fn
        from torch.utils.data import Dataset, DataLoader

        class MockDataset(Dataset):
            def __init__(self, n_samples=10, n_genes=100):
                self.n_samples = n_samples
                self.n_genes = n_genes

            def __len__(self):
                return self.n_samples

            def __getitem__(self, idx):
                return create_mock_dataset_sample(n_genes=self.n_genes)

        dataset = MockDataset(n_samples=16, n_genes=100)
        loader = DataLoader(
            dataset,
            batch_size=4,
            collate_fn=collate_fn,
            num_workers=0,
        )

        # Iterate through loader
        for batch in loader:
            assert batch["batch_size"] == 4
            assert batch["pseudobulk"].shape[0] == 4
            break

    def test_dataloader_iteration_produces_valid_batches(self):
        """All batches from DataLoader should be valid."""
        from src.data.collate import collate_fn
        from torch.utils.data import Dataset, DataLoader

        class MockDataset(Dataset):
            def __init__(self, n_samples=10, n_genes=100):
                self.n_samples = n_samples
                self.n_genes = n_genes

            def __len__(self):
                return self.n_samples

            def __getitem__(self, idx):
                return create_mock_dataset_sample(n_genes=self.n_genes)

        dataset = MockDataset(n_samples=10, n_genes=100)
        loader = DataLoader(dataset, batch_size=3, collate_fn=collate_fn, num_workers=0)

        n_batches = 0
        for batch in loader:
            assert torch.isfinite(batch["pseudobulk"]).all()
            assert torch.isfinite(batch["cells"]).all()
            n_batches += 1

        assert n_batches == 4  # 10 samples / 3 = 4 batches (last has 1 sample)


# =============================================================================
# Gradient Flow Audit Tests
# =============================================================================


class TestGradientFlowAudit:
    """Verify gradients flow to all learnable parameters."""

    def test_gradients_reach_cell_type_selector(self):
        """Gradients should reach CellTypeSelector logits."""
        from src.data.collate import collate_fn
        from src.models.branches.cell_transformer import CellTransformer

        n_genes = 100
        batch = [create_mock_dataset_sample(n_genes=n_genes) for _ in range(4)]
        collated = collate_fn(batch)

        transformer = CellTransformer(
            n_genes=n_genes,
            n_cell_types=N_CELL_TYPES,
            d_model=64,
            n_heads=4,
            n_isab_layers=2,
            n_inducing=16,
        )

        embeddings, _, _ = transformer(collated["cells"], collated["cell_mask"])
        loss = embeddings.sum()
        loss.backward()

        # Selector logits should have gradients
        assert transformer.selector.selection_logits.grad is not None
        assert not torch.all(transformer.selector.selection_logits.grad == 0)

    def test_gradients_reach_hgt_layer_scales(self):
        """Gradients should reach HGT LayerScale parameters."""
        from src.models.branches.hgt_encoder import HGTEncoder

        d_input = 64
        encoder = HGTEncoder(
            d_input=d_input,
            d_hidden=64,
            d_output=64,
            n_heads=4,
            n_layers=2,
        )

        # Create minimal input
        x_dict = {ct: torch.randn(1, d_input, requires_grad=True) for ct in CELL_TYPE_ORDER}

        # Create one edge
        triplet = (CELL_TYPE_ORDER[0], ALL_EDGE_TYPES[0], CELL_TYPE_ORDER[1])
        edge_index_dict = {triplet: torch.tensor([[0], [0]], dtype=torch.long)}
        edge_attr_dict = {triplet: torch.rand(1, 1)}

        output_dict, _ = encoder(x_dict, edge_index_dict, edge_attr_dict)

        # Compute loss and backprop
        loss = sum(out.sum() for out in output_dict.values())
        loss.backward()

        # LayerScale parameters should have gradients
        for layer_scale in encoder.layer_scales:
            assert layer_scale.grad is not None

    def test_gradients_reach_set_transformer_inducing_points(self):
        """Gradients should reach SetTransformer inducing points."""
        from src.data.collate import collate_fn
        from src.models.branches.cell_transformer import CellTransformer

        n_genes = 100
        batch = [create_mock_dataset_sample(n_genes=n_genes) for _ in range(4)]
        collated = collate_fn(batch)

        transformer = CellTransformer(
            n_genes=n_genes,
            n_cell_types=N_CELL_TYPES,
            d_model=64,
            n_heads=4,
            n_isab_layers=2,
            n_inducing=16,
        )

        embeddings, _, _ = transformer(collated["cells"], collated["cell_mask"])
        loss = embeddings.sum()
        loss.backward()

        # Check ISAB inducing points have gradients
        for isab in transformer.set_encoder.isab_layers:
            assert isab.inducing_points.grad is not None


# =============================================================================
# Edge Cases Across Full Pipeline
# =============================================================================


class TestPipelineEdgeCases:
    """Test edge cases across the full pipeline."""

    def test_pipeline_with_all_masked_cell_types(self):
        """Pipeline should handle samples with all-masked cell types."""
        from src.data.collate import collate_fn
        from src.models.branches.cell_transformer import CellTransformer

        n_genes = 100
        samples = []
        for i in range(4):
            sample = create_mock_sample_with_empty_cell_types(
                n_genes=n_genes,
                empty_types=[0, 5, 10],  # Make some types empty
            )
            samples.append(sample)

        collated = collate_fn(samples)

        transformer = CellTransformer(
            n_genes=n_genes,
            n_cell_types=N_CELL_TYPES,
            d_model=64,
            n_heads=4,
            n_isab_layers=2,
            n_inducing=16,
        )

        embeddings, _, _ = transformer(collated["cells"], collated["cell_mask"])

        # Should produce finite outputs despite empty cell types
        assert torch.isfinite(embeddings).all()

    def test_pipeline_with_empty_graphs(self):
        """Pipeline should handle samples with no CCC edges."""
        from src.data.collate import collate_fn

        samples = []
        for i in range(4):
            sample = create_mock_dataset_sample(n_edges=0)
            sample["ccc_edge_index"] = torch.zeros((2, 0), dtype=torch.long)
            sample["ccc_edge_type"] = torch.zeros((0,), dtype=torch.long)
            sample["ccc_edge_attr"] = torch.zeros((0, 1))
            samples.append(sample)

        collated = collate_fn(samples)

        # Should handle empty graphs
        assert collated["ccc_edge_index"].shape == (2, 0)
        assert collated["batch_size"] == 4

    def test_pipeline_with_single_sample_batch(self):
        """Pipeline should handle batch of size 1."""
        from src.data.collate import collate_fn
        from src.models.branches.cell_transformer import CellTransformer

        n_genes = 100
        batch = [create_mock_dataset_sample(n_genes=n_genes)]
        collated = collate_fn(batch)

        transformer = CellTransformer(
            n_genes=n_genes,
            n_cell_types=N_CELL_TYPES,
            d_model=64,
            n_heads=4,
            n_isab_layers=2,
            n_inducing=16,
        )

        embeddings, _, _ = transformer(collated["cells"], collated["cell_mask"])

        assert embeddings.shape[0] == 1
        assert torch.isfinite(embeddings).all()


# =============================================================================
# Numerical Stability End-to-End
# =============================================================================


class TestNumericalStabilityEndToEnd:
    """Test numerical stability across the full pipeline."""

    def test_large_input_values_through_pipeline(self):
        """Large input values should not cause NaN/Inf."""
        from src.data.collate import collate_fn
        from src.models.branches.cell_transformer import CellTransformer

        n_genes = 100
        samples = []
        for i in range(4):
            sample = create_mock_dataset_sample(n_genes=n_genes)
            sample["pseudobulk"] = sample["pseudobulk"] * 100
            sample["cells"] = sample["cells"] * 100
            samples.append(sample)

        collated = collate_fn(samples)

        transformer = CellTransformer(
            n_genes=n_genes,
            n_cell_types=N_CELL_TYPES,
            d_model=64,
            n_heads=4,
            n_isab_layers=2,
            n_inducing=16,
        )

        embeddings, _, _ = transformer(collated["cells"], collated["cell_mask"])

        assert torch.isfinite(embeddings).all()

    def test_small_input_values_through_pipeline(self):
        """Small input values should not cause numerical issues."""
        from src.data.collate import collate_fn
        from src.models.branches.cell_transformer import CellTransformer

        n_genes = 100
        samples = []
        for i in range(4):
            sample = create_mock_dataset_sample(n_genes=n_genes)
            sample["pseudobulk"] = sample["pseudobulk"] * 1e-6
            sample["cells"] = sample["cells"] * 1e-6
            samples.append(sample)

        collated = collate_fn(samples)

        transformer = CellTransformer(
            n_genes=n_genes,
            n_cell_types=N_CELL_TYPES,
            d_model=64,
            n_heads=4,
            n_isab_layers=2,
            n_inducing=16,
        )

        embeddings, _, _ = transformer(collated["cells"], collated["cell_mask"])

        assert torch.isfinite(embeddings).all()

    def test_no_nan_in_any_output_tensor(self):
        """No NaN should appear in any output tensor."""
        from src.data.collate import collate_fn
        from src.models.branches.cell_transformer import CellTransformer
        from src.models.branches.pseudobulk_encoder import PseudobulkEncoder

        n_genes = 100
        batch = [create_mock_dataset_sample(n_genes=n_genes) for _ in range(8)]
        collated = collate_fn(batch)

        # Test all branches
        pb_encoder = PseudobulkEncoder(n_genes=n_genes, n_cell_types=N_CELL_TYPES)
        cell_transformer = CellTransformer(
            n_genes=n_genes, n_cell_types=N_CELL_TYPES,
            d_model=64, n_heads=4, n_isab_layers=2, n_inducing=16,
        )

        pb_out = pb_encoder(collated["pseudobulk"])
        cell_out, weights, attention = cell_transformer(
            collated["cells"], collated["cell_mask"], return_attention=True
        )

        # Check all outputs
        assert not torch.isnan(pb_out).any()
        assert not torch.isnan(cell_out).any()
        assert not torch.isnan(weights).any()
        for attn in attention:
            assert not torch.isnan(attn).any()
