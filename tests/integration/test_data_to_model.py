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
        assert "ccc_edge_index" in result
        assert "ccc_edge_type" in result
        assert "ccc_edge_attr" in result
        assert "ccc_edge_counts" not in result


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
        assert attention.shape[1] == N_CELL_TYPES

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


class TestCollateToHGTEncoderTensor:
    """Test that collate_for_hgt output works with HGTEncoderTensor."""

    def test_collate_for_hgt_produces_valid_tensor_format(self):
        """collate_for_hgt should produce padded tensors compatible with HGTEncoderTensor."""
        from src.data.collate import collate_for_hgt
        from src.models.branches.hgt_encoder_tensor import HGTEncoderTensor

        d_input = 100
        d_hidden = 64

        batch = [create_mock_dataset_sample(n_genes=d_input, n_edges=30) for _ in range(4)]
        collated = collate_for_hgt(batch)

        # Verify flat edge tensor structure
        assert "ccc_edge_index" in collated
        assert "ccc_edge_type" in collated
        assert "ccc_edge_attr" in collated
        assert "ccc_edge_counts" not in collated
        E_total = 4 * 30
        assert collated["ccc_edge_index"].shape == (2, E_total)

        # Verify encoder can be created with correct structure
        encoder = HGTEncoderTensor(
            d_input=d_input,
            d_hidden=d_hidden,
            d_output=d_hidden,
            n_heads=4,
            n_layers=2,
            n_node_types=N_CELL_TYPES,
            n_edge_types=len(ALL_EDGE_TYPES),
            edge_dim=1,
        )

        # Forward pass with collated tensors (flat format, no edge_counts)
        x = collated["pseudobulk"]  # [4, N_CELL_TYPES, d_input]
        out = encoder(
            x,
            collated["ccc_edge_index"],
            collated["ccc_edge_type"],
            collated["ccc_edge_attr"],
        )
        assert out.shape == (4, N_CELL_TYPES, d_hidden)
        assert torch.isfinite(out).all()

    def test_hgt_encoder_tensor_gradient_flow(self):
        """Gradients should flow correctly through HGTEncoderTensor."""
        from src.models.branches.hgt_encoder_tensor import HGTEncoderTensor

        B = 2
        d_input = 64
        d_output = 32

        encoder = HGTEncoderTensor(
            d_input=d_input,
            d_hidden=32,
            d_output=d_output,
            n_heads=4,
            n_layers=2,
            n_node_types=N_CELL_TYPES,
            n_edge_types=len(ALL_EDGE_TYPES),
            edge_dim=1,
        )

        x = torch.randn(B, N_CELL_TYPES, d_input, requires_grad=True)
        n_edges = 5
        src_parts, dst_parts, type_parts = [], [], []
        for b in range(B):
            offset = b * N_CELL_TYPES
            src_parts.append(torch.randint(0, N_CELL_TYPES, (n_edges,)) + offset)
            dst_parts.append(torch.randint(0, N_CELL_TYPES, (n_edges,)) + offset)
            type_parts.append(torch.randint(0, len(ALL_EDGE_TYPES), (n_edges,)))
        edge_index = torch.stack([torch.cat(src_parts), torch.cat(dst_parts)])
        edge_type = torch.cat(type_parts)
        edge_attr = torch.rand(B * n_edges, 1)

        out = encoder(x, edge_index, edge_type, edge_attr)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None


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
        from src.models.branches.hgt_encoder_tensor import HGTEncoderTensor

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
        hgt_encoder = HGTEncoderTensor(
            d_input=d_embed,  # Takes embeddings as input
            d_hidden=d_embed,
            d_output=d_embed,
            n_heads=4,
            n_layers=2,
            n_node_types=N_CELL_TYPES,
            n_edge_types=len(ALL_EDGE_TYPES),
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
        from src.models.branches.hgt_encoder_tensor import HGTEncoderTensor
        from src.data.constants import N_EDGE_TYPES

        d_input = 64
        encoder = HGTEncoderTensor(
            d_input=d_input,
            d_hidden=64,
            d_output=64,
            n_heads=4,
            n_layers=2,
            n_node_types=N_CELL_TYPES,
            n_edge_types=N_EDGE_TYPES,
        )

        B = 1
        n_edges = 3
        x = torch.randn(B, N_CELL_TYPES, d_input, requires_grad=True)
        src_parts, dst_parts, type_parts = [], [], []
        for b in range(B):
            offset = b * N_CELL_TYPES
            src_parts.append(torch.randint(0, N_CELL_TYPES, (n_edges,)) + offset)
            dst_parts.append(torch.randint(0, N_CELL_TYPES, (n_edges,)) + offset)
            type_parts.append(torch.randint(0, N_EDGE_TYPES, (n_edges,)))
        edge_index = torch.stack([torch.cat(src_parts), torch.cat(dst_parts)])
        edge_type = torch.cat(type_parts)
        edge_attr = torch.rand(B * n_edges, 1)

        output = encoder(x, edge_index, edge_type, edge_attr)

        # Compute loss and backprop
        loss = output.sum()
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
        assert not torch.isnan(attention).any()
