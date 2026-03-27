"""
Integration test: RegionHandler with various upstream components.

Tests the multi-region encoding flow with:
1. CellTransformer output dimensions feeding into RegionHandler
2. RegionHandler output feeding into FusionLayer / PathologyEncoder
"""

import pytest
import torch

from src.data.constants import N_CELL_TYPES, N_REGIONS


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

        pooled, region_context, _ = region_handler(stacked_cell_embs, region_mask)

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

        pooled_full, _, _ = region_handler(stacked, full_mask)
        pooled_partial, _, _ = region_handler(stacked, partial_mask)

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

        _, region_context, _ = handler(x, mask)
        path_emb = encoder(pathology, region_context)

        assert path_emb.shape == (batch_size, d_cond)
        assert torch.isfinite(path_emb).all()

        # Gradient should flow back through both components
        loss = path_emb.sum()
        loss.backward()

        assert handler.region_embedding.weight.grad is not None
        assert encoder.region_proj.weight.grad is not None

    def test_region_handler_pooled_feeds_into_fusion_layer(self):
        """pooled output [B, C, D] should be compatible with FusionLayer inputs.

        In 2-branch architecture, pooled feeds into HGT encoder, and the HGT
        output is one of two FusionLayer inputs (hgt_emb, cell_emb).
        """
        from src.models.components.region_handler import RegionHandler
        from src.models.fusion.fusion_layer import FusionLayer

        d_embed = 64
        d_fused = 64
        batch_size = 2

        handler = RegionHandler(d_model=d_embed, n_regions=N_REGIONS)
        fusion = FusionLayer(d_embed=d_embed, d_fused=d_fused, n_cell_types=N_CELL_TYPES)

        x = torch.randn(batch_size, N_REGIONS, N_CELL_TYPES, d_embed)
        mask = torch.ones(batch_size, N_REGIONS, dtype=torch.bool)

        pooled, _, _ = handler(x, mask)

        # pooled serves as HGT input; FusionLayer takes 2 inputs (hgt_emb, cell_emb)
        cell_emb = torch.randn(batch_size, N_CELL_TYPES, d_embed)

        fused = fusion(pooled, cell_emb)

        assert fused.shape == (batch_size, N_CELL_TYPES, d_fused)
        assert torch.isfinite(fused).all()
