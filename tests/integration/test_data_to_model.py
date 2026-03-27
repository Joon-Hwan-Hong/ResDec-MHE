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
    """Create a sample matching CognitiveResilienceDataset output format (flat cell format)."""
    # Flat cell format: cell_data [total_cells, n_genes], cell_offsets [N_CELL_TYPES + 1]
    total_cells = N_CELL_TYPES * max_cells
    cell_data = torch.randn(total_cells, n_genes)
    cell_offsets = torch.arange(
        0, (N_CELL_TYPES + 1) * max_cells, max_cells, dtype=torch.long,
    )

    return {
        "subject_id": "TEST_SUBJECT",
        "pseudobulk": torch.randn(N_CELL_TYPES, n_genes),
        "cell_type_mask": torch.ones(N_CELL_TYPES, dtype=torch.bool),
        "cell_counts": torch.randint(10, 100, (N_CELL_TYPES,)),
        "cell_data": cell_data,
        "cell_offsets": cell_offsets,
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
    """Create sample with some cell types having zero cells (flat format).

    For empty cell types, the cell_offsets for that type have equal start/end
    (zero cells), and cell_type_mask is set to False.
    """
    if empty_types is None:
        empty_types = []

    # Build per-type cell counts
    cells_per_type = []
    for ct in range(N_CELL_TYPES):
        if ct in empty_types:
            cells_per_type.append(0)
        else:
            cells_per_type.append(max_cells)

    total_cells = sum(cells_per_type)
    cell_data = torch.randn(max(total_cells, 0), n_genes) if total_cells > 0 else torch.empty(0, n_genes)

    # Build offsets from per-type counts
    offsets = [0]
    for c in cells_per_type:
        offsets.append(offsets[-1] + c)
    cell_offsets = torch.tensor(offsets, dtype=torch.long)

    cell_type_mask = torch.ones(N_CELL_TYPES, dtype=torch.bool)
    for ct_idx in empty_types:
        cell_type_mask[ct_idx] = False

    return {
        "subject_id": "TEST_SUBJECT",
        "pseudobulk": torch.randn(N_CELL_TYPES, n_genes),
        "cell_type_mask": cell_type_mask,
        "cell_counts": torch.tensor(cells_per_type, dtype=torch.long),
        "cell_data": cell_data,
        "cell_offsets": cell_offsets,
        "ccc_edge_index": torch.randint(0, N_CELL_TYPES, (2, n_edges)),
        "ccc_edge_type": torch.randint(0, 5, (n_edges,)),
        "ccc_edge_attr": torch.rand(n_edges, 1),
        "pathology": torch.rand(3),
        "cognition": torch.randn(1),
        "region_mask": torch.ones(N_REGIONS, dtype=torch.bool),
    }


class TestDatasetToCollate:
    """Test that dataset output format works with collate functions."""

    def test_collate_fn_accepts_dataset_format(self):
        """collate_fn should accept dataset output format."""
        from src.data.collate import collate_fn

        batch = [create_mock_dataset_sample() for _ in range(4)]
        result = collate_fn(batch)

        assert result["batch_size"] == 4
        assert result["pseudobulk"].shape[0] == 4
        assert "cell_data" in result
        assert "cell_offsets" in result

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
        embeddings, _ = transformer(
            collated["cell_data"],
            collated["cell_offsets"],
        )

        # Verify output shape
        assert embeddings.shape == (4, N_CELL_TYPES, 64)

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
        cells = collated["cell_data"].clone().requires_grad_(True)

        embeddings, _ = transformer(cells, collated["cell_offsets"])
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

        embeddings, attention = transformer(
            collated["cell_data"],
            collated["cell_offsets"],
            return_attention=True,
        )

        # Verify outputs are valid
        assert torch.isfinite(embeddings).all()
        assert attention is not None
        assert attention.shape[1] == N_CELL_TYPES

    def test_pipeline_with_sparse_data(self):
        """Pipeline should handle sparse cell data correctly."""
        from src.data.collate import collate_fn
        from src.models.branches.cell_transformer import CellTransformer

        n_genes = 100

        # Create samples with few cells per type (sparse)
        samples = []
        for i in range(4):
            sample = create_mock_dataset_sample(n_genes=n_genes, max_cells=10)
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

        embeddings, _ = transformer(collated["cell_data"], collated["cell_offsets"])

        # Should produce valid outputs despite sparse data
        assert torch.isfinite(embeddings).all()

    def test_pipeline_with_mixed_valid_cells(self):
        """Pipeline handles samples with different numbers of valid cells."""
        from src.data.collate import collate_fn
        from src.models.branches.cell_transformer import CellTransformer

        n_genes = 100

        # Create samples with varying numbers of cells per type
        samples = []
        for i in range(4):
            max_cells = (i + 1) * 20  # 20, 40, 60, 80
            sample = create_mock_dataset_sample(n_genes=n_genes, max_cells=max_cells)
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

        embeddings, _ = transformer(collated["cell_data"], collated["cell_offsets"])

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
# Cross-Branch Consistency Tests
# =============================================================================


class TestCrossBranchConsistency:
    """Test that all branches produce compatible outputs for fusion."""

    def test_all_branches_same_embed_dimension(self):
        """Both branches should output the same embedding dimension."""
        from src.data.collate import collate_fn
        from src.models.branches.cell_transformer import CellTransformer
        from src.models.components.gene_attention_gate import GeneAttentionGate
        from src.models.branches.hgt_encoder_tensor import HGTEncoderTensor

        n_genes = 100
        d_embed = 64  # Same for all branches

        batch = [create_mock_dataset_sample(n_genes=n_genes) for _ in range(4)]
        collated = collate_fn(batch)

        # HGT input pipeline: GeneAttentionGate + Linear projection
        gate = GeneAttentionGate(n_cell_types=N_CELL_TYPES, n_genes=n_genes, temperature=2.0)
        proj = torch.nn.Linear(n_genes, d_embed)
        hgt_input = proj(gate(collated["pseudobulk"]))

        # Branch 2: CellTransformer
        cell_transformer = CellTransformer(
            n_genes=n_genes,
            n_cell_types=N_CELL_TYPES,
            d_model=d_embed,
            n_heads=4,
            n_isab_layers=2,
            n_inducing=16,
        )
        cell_out, _ = cell_transformer(collated["cell_data"], collated["cell_offsets"])

        # Branch 1: HGT (verify output dimension matches)
        hgt_encoder = HGTEncoderTensor(
            d_input=d_embed,
            d_hidden=d_embed,
            d_output=d_embed,
            n_heads=4,
            n_layers=2,
            n_node_types=N_CELL_TYPES,
            n_edge_types=len(ALL_EDGE_TYPES),
        )

        # Verify dimensions match for fusion
        assert hgt_input.shape[-1] == d_embed
        assert cell_out.shape[-1] == d_embed
        assert hgt_encoder.d_output == d_embed

    def test_all_branches_same_cell_type_dimension(self):
        """Both branches should have same n_cell_types dimension."""
        from src.data.collate import collate_fn
        from src.models.branches.cell_transformer import CellTransformer
        from src.models.components.gene_attention_gate import GeneAttentionGate

        n_genes = 100
        d_embed = 64
        batch = [create_mock_dataset_sample(n_genes=n_genes) for _ in range(4)]
        collated = collate_fn(batch)

        gate = GeneAttentionGate(n_cell_types=N_CELL_TYPES, n_genes=n_genes, temperature=2.0)
        proj = torch.nn.Linear(n_genes, d_embed)
        cell_transformer = CellTransformer(n_genes=n_genes, n_cell_types=N_CELL_TYPES, d_model=d_embed)

        hgt_input = proj(gate(collated["pseudobulk"]))
        cell_out, _ = cell_transformer(collated["cell_data"], collated["cell_offsets"])

        # Both should have n_cell_types in dimension 1
        assert hgt_input.shape[1] == N_CELL_TYPES
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
            assert torch.isfinite(batch["cell_data"]).all()
            n_batches += 1

        assert n_batches == 4  # 10 samples / 3 = 4 batches (last has 1 sample)


# =============================================================================
# Gradient Flow Audit Tests
# =============================================================================


class TestGradientFlowAudit:
    """Verify gradients flow to all learnable parameters."""

    def test_gradients_reach_gene_attention_gate(self):
        """Gradients should reach GeneAttentionGate logits."""
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

        embeddings, _ = transformer(collated["cell_data"], collated["cell_offsets"])
        loss = embeddings.sum()
        loss.backward()

        # Gene attention gate logits should have gradients
        assert transformer.gene_gate.gate_logits.grad is not None
        assert not torch.all(transformer.gene_gate.gate_logits.grad == 0)

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

        embeddings, _ = transformer(collated["cell_data"], collated["cell_offsets"])
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

        embeddings, _ = transformer(collated["cell_data"], collated["cell_offsets"])

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

        embeddings, _ = transformer(collated["cell_data"], collated["cell_offsets"])

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
            sample["cell_data"] = sample["cell_data"] * 100
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

        embeddings, _ = transformer(collated["cell_data"], collated["cell_offsets"])

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
            sample["cell_data"] = sample["cell_data"] * 1e-6
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

        embeddings, _ = transformer(collated["cell_data"], collated["cell_offsets"])

        assert torch.isfinite(embeddings).all()

    def test_no_nan_in_any_output_tensor(self):
        """No NaN should appear in any output tensor."""
        from src.data.collate import collate_fn
        from src.models.branches.cell_transformer import CellTransformer
        from src.models.components.gene_attention_gate import GeneAttentionGate

        n_genes = 100
        batch = [create_mock_dataset_sample(n_genes=n_genes) for _ in range(8)]
        collated = collate_fn(batch)

        # Test HGT input pipeline (GeneAttentionGate + Linear projection)
        gate = GeneAttentionGate(n_cell_types=N_CELL_TYPES, n_genes=n_genes, temperature=2.0)
        proj = torch.nn.Linear(n_genes, 64)
        gated = gate(collated["pseudobulk"])
        hgt_input = proj(gated)

        cell_transformer = CellTransformer(
            n_genes=n_genes, n_cell_types=N_CELL_TYPES,
            d_model=64, n_heads=4, n_isab_layers=2, n_inducing=16,
        )

        cell_out, attention = cell_transformer(
            collated["cell_data"], collated["cell_offsets"], return_attention=True
        )

        # Check all outputs
        assert not torch.isnan(hgt_input).any()
        assert not torch.isnan(cell_out).any()
        if attention is not None:
            assert not torch.isnan(attention).any()
