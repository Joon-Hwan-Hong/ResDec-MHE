"""
Integration test: RegionHandler with various upstream components.

Tests the multi-region encoding flow with:
1. PseudobulkEncoder + RegionHandler
2. CellTransformer output dimensions feeding into RegionHandler
3. RegionHandler output feeding into FusionLayer / PathologyEncoder
"""

import pytest
import torch

from src.data.constants import N_CELL_TYPES, N_REGIONS


class TestPseudobulkRegionHandlerIntegration:
    """Integration tests for PseudobulkEncoder + RegionHandler."""

    def test_full_multiregion_pipeline(self):
        """Test complete multi-region encoding pipeline."""
        from src.models.branches.pseudobulk_encoder import PseudobulkEncoder
        from src.models.components.region_handler import RegionHandler

        # Setup
        n_cell_types = N_CELL_TYPES
        n_genes = 100
        d_embed = 64
        n_regions = N_REGIONS
        batch_size = 4

        encoder = PseudobulkEncoder(
            n_cell_types=n_cell_types,
            n_genes=n_genes,
            d_embed=d_embed,
        )
        region_handler = RegionHandler(d_model=d_embed, n_regions=n_regions)

        # Input: multi-region pseudobulk
        region_pseudobulk = torch.randn(batch_size, n_regions, n_cell_types, n_genes)
        region_mask = torch.ones(batch_size, n_regions, dtype=torch.bool)

        # Forward pass (as it would be in full model)
        B, R, C, G = region_pseudobulk.shape

        # Encode each region
        encoded = encoder(region_pseudobulk.view(B * R, C, G))
        encoded = encoded.view(B, R, C, -1)

        # Pool across regions
        pooled, region_context = region_handler(encoded, region_mask)

        # Verify shapes
        assert pooled.shape == (batch_size, n_cell_types, d_embed)
        assert region_context.shape == (batch_size, d_embed)

    def test_single_region_same_as_direct_encoding(self):
        """Single-region through pipeline should equal direct encoding."""
        from src.models.branches.pseudobulk_encoder import PseudobulkEncoder
        from src.models.components.region_handler import RegionHandler

        n_cell_types = N_CELL_TYPES
        n_genes = 100
        d_embed = 64
        batch_size = 2

        encoder = PseudobulkEncoder(
            n_cell_types=n_cell_types,
            n_genes=n_genes,
            d_embed=d_embed,
        )
        region_handler = RegionHandler(d_model=d_embed, n_regions=N_REGIONS)

        # Set to eval mode to disable dropout for deterministic comparison
        encoder.eval()
        region_handler.eval()

        # Single-region input (PFC only)
        single_pseudobulk = torch.randn(batch_size, n_cell_types, n_genes)

        # Direct encoding
        direct_encoded = encoder(single_pseudobulk)

        # Through multi-region pipeline
        region_pseudobulk = torch.zeros(batch_size, N_REGIONS, n_cell_types, n_genes)
        region_pseudobulk[:, 0] = single_pseudobulk  # Only PFC
        region_mask = torch.zeros(batch_size, N_REGIONS, dtype=torch.bool)
        region_mask[:, 0] = True

        B, R, C, G = region_pseudobulk.shape
        encoded = encoder(region_pseudobulk.view(B * R, C, G))
        encoded = encoded.view(B, R, C, -1)
        pooled, _ = region_handler(encoded, region_mask)

        # Should be identical
        assert torch.allclose(pooled, direct_encoded, atol=1e-5)

    def test_gradient_flows_through_pipeline(self):
        """Gradients should flow through entire pipeline."""
        from src.models.branches.pseudobulk_encoder import PseudobulkEncoder
        from src.models.components.region_handler import RegionHandler

        encoder = PseudobulkEncoder(n_cell_types=N_CELL_TYPES, n_genes=100, d_embed=64)
        region_handler = RegionHandler(d_model=64, n_regions=N_REGIONS)

        region_pseudobulk = torch.randn(2, N_REGIONS, N_CELL_TYPES, 100, requires_grad=True)
        region_mask = torch.ones(2, N_REGIONS, dtype=torch.bool)

        B, R, C, G = region_pseudobulk.shape
        encoded = encoder(region_pseudobulk.view(B * R, C, G))
        encoded = encoded.view(B, R, C, -1)
        pooled, region_context = region_handler(encoded, region_mask)

        loss = pooled.sum() + region_context.sum()
        loss.backward()

        # Gradients should reach input
        assert region_pseudobulk.grad is not None

        # Gradients should reach encoder parameters
        assert encoder.gene_gate.gate_logits.grad is not None

        # Gradients should reach region_handler parameters
        assert region_handler.region_weights.grad is not None


class TestRegionHandlerWithCellTransformerOutput:
    """Integration tests: CellTransformer-shaped output + RegionHandler.

    CellTransformer produces [B, C, D] per region; when stacked to [B, R, C, D]
    the RegionHandler should pool across regions correctly.
    """

    def test_cell_transformer_shaped_output_through_region_handler(self):
        """RegionHandler should accept stacked CellTransformer-shaped embeddings."""
        from src.models.components.region_handler import RegionHandler

        d_embed = 64
        batch_size = 3

        region_handler = RegionHandler(d_model=d_embed, n_regions=N_REGIONS)

        # Simulate stacked CellTransformer outputs across regions: [B, R, C, D]
        stacked_cell_embs = torch.randn(batch_size, N_REGIONS, N_CELL_TYPES, d_embed)
        region_mask = torch.ones(batch_size, N_REGIONS, dtype=torch.bool)

        pooled, region_context = region_handler(stacked_cell_embs, region_mask)

        assert pooled.shape == (batch_size, N_CELL_TYPES, d_embed)
        assert region_context.shape == (batch_size, d_embed)
        assert torch.isfinite(pooled).all()
        assert torch.isfinite(region_context).all()

    def test_partial_region_mask_with_cell_transformer_output(self):
        """RegionHandler with partial mask gives different output from full mask."""
        from src.models.components.region_handler import RegionHandler

        d_embed = 64
        batch_size = 2

        region_handler = RegionHandler(d_model=d_embed, n_regions=N_REGIONS)
        region_handler.eval()

        stacked = torch.randn(batch_size, N_REGIONS, N_CELL_TYPES, d_embed)

        full_mask = torch.ones(batch_size, N_REGIONS, dtype=torch.bool)
        partial_mask = torch.zeros(batch_size, N_REGIONS, dtype=torch.bool)
        partial_mask[:, 0] = True  # PFC only

        pooled_full, _ = region_handler(stacked, full_mask)
        pooled_partial, _ = region_handler(stacked, partial_mask)

        assert not torch.allclose(pooled_full, pooled_partial, atol=1e-6), \
            "Full-mask and partial-mask outputs should differ"


class TestRegionHandlerToDownstreamComponents:
    """Integration tests: RegionHandler output -> PathologyEncoder / FusionLayer."""

    def test_region_context_feeds_into_pathology_encoder(self):
        """region_context from RegionHandler should be accepted by PathologyEncoder."""
        from src.models.components.region_handler import RegionHandler
        from src.models.fusion.pathology_encoder import PathologyEncoder

        d_embed = 64
        d_cond = 32
        batch_size = 3

        handler = RegionHandler(d_model=d_embed, n_regions=N_REGIONS)
        encoder = PathologyEncoder(
            n_pathology_features=3,
            d_region=d_embed,
            d_cond=d_cond,
        )

        x = torch.randn(batch_size, N_REGIONS, N_CELL_TYPES, d_embed)
        mask = torch.ones(batch_size, N_REGIONS, dtype=torch.bool)
        pathology = torch.randn(batch_size, 3)

        _, region_context = handler(x, mask)
        path_emb = encoder(pathology, region_context)

        assert path_emb.shape == (batch_size, d_cond)
        assert torch.isfinite(path_emb).all()

        # Gradient should flow back through both components
        loss = path_emb.sum()
        loss.backward()

        assert handler.region_embedding.weight.grad is not None
        assert encoder.region_proj.weight.grad is not None

    def test_region_handler_pooled_feeds_into_fusion_layer(self):
        """pooled output [B, C, D] should be compatible with FusionLayer inputs."""
        from src.models.components.region_handler import RegionHandler
        from src.models.fusion.fusion_layer import FusionLayer

        d_embed = 64
        d_fused = 64
        batch_size = 2

        handler = RegionHandler(d_model=d_embed, n_regions=N_REGIONS)
        fusion = FusionLayer(d_embed=d_embed, d_fused=d_fused, n_cell_types=N_CELL_TYPES)

        x = torch.randn(batch_size, N_REGIONS, N_CELL_TYPES, d_embed)
        mask = torch.ones(batch_size, N_REGIONS, dtype=torch.bool)

        pooled, _ = handler(x, mask)

        # pooled can serve as one of the three branch inputs to FusionLayer
        hgt_emb = torch.randn(batch_size, N_CELL_TYPES, d_embed)
        cell_emb = torch.randn(batch_size, N_CELL_TYPES, d_embed)

        fused = fusion(pooled, hgt_emb, cell_emb)

        assert fused.shape == (batch_size, N_CELL_TYPES, d_fused)
        assert torch.isfinite(fused).all()
