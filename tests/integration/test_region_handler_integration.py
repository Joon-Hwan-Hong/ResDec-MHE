"""
Integration test: PseudobulkEncoder + RegionHandler pipeline.

Tests the full multi-region encoding flow:
    region_pseudobulk [B, R, C, G]
    -> reshape -> PseudobulkEncoder -> reshape
    -> RegionHandler -> pooled [B, C, D] + region_context [B, D]
"""

import pytest
import torch


class TestPseudobulkRegionHandlerIntegration:
    """Integration tests for PseudobulkEncoder + RegionHandler."""

    def test_full_multiregion_pipeline(self):
        """Test complete multi-region encoding pipeline."""
        from src.models.branches.pseudobulk_encoder import PseudobulkEncoder
        from src.models.components.region_handler import RegionHandler

        # Setup
        n_cell_types = 31
        n_genes = 100
        d_embed = 64
        n_regions = 6
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

        n_cell_types = 31
        n_genes = 100
        d_embed = 64
        batch_size = 2

        encoder = PseudobulkEncoder(
            n_cell_types=n_cell_types,
            n_genes=n_genes,
            d_embed=d_embed,
        )
        region_handler = RegionHandler(d_model=d_embed, n_regions=6)

        # Set to eval mode to disable dropout for deterministic comparison
        encoder.eval()
        region_handler.eval()

        # Single-region input (PFC only)
        single_pseudobulk = torch.randn(batch_size, n_cell_types, n_genes)

        # Direct encoding
        direct_encoded = encoder(single_pseudobulk)

        # Through multi-region pipeline
        region_pseudobulk = torch.zeros(batch_size, 6, n_cell_types, n_genes)
        region_pseudobulk[:, 0] = single_pseudobulk  # Only PFC
        region_mask = torch.zeros(batch_size, 6, dtype=torch.bool)
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

        encoder = PseudobulkEncoder(n_cell_types=31, n_genes=100, d_embed=64)
        region_handler = RegionHandler(d_model=64, n_regions=6)

        region_pseudobulk = torch.randn(2, 6, 31, 100, requires_grad=True)
        region_mask = torch.ones(2, 6, dtype=torch.bool)

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
