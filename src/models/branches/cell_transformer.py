"""
Cell Transformer (Branch 3) for cell-level heterogeneity modeling.

Combines CellTypeSelector with SetTransformerEncoder to capture within-cell-type
variation for all cell types, weighted by learned importance.

Architecture:
    Input: Cell-level data [batch, n_cell_types, max_cells, n_genes]
    → SetTransformerEncoder (per cell type, all 31 types)
    → CellTypeSelector weights (soft attention, differentiable)
    → Output: Weighted embeddings [batch, n_cell_types, d_embed]

Design decision (2026-01-27):
    Uses soft attention weighting instead of hard top-k selection.
    This makes cell type selection fully differentiable, allowing the model
    to learn which cell types are most relevant for predicting cognitive resilience.
"""

from typing import Optional

import torch
import torch.nn as nn

from src.models.components.cell_type_selector import CellTypeSelector
from src.models.components.set_transformer import SetTransformerEncoder


class CellTransformer(nn.Module):
    """
    Cell-level transformer for modeling within-cell-type heterogeneity.

    Processes ALL cell types through SetTransformer and weights their
    contributions using learned soft attention. This is fully differentiable,
    allowing the model to learn which cell types are most relevant.

    Args:
        n_genes: Number of input genes
        n_cell_types: Total number of cell types (default: 31)
        d_model: Model dimension for Set Transformer
        n_heads: Number of attention heads
        n_isab_layers: Number of ISAB blocks
        n_inducing: Number of inducing points for ISAB
        n_pma_seeds: Number of seed vectors for PMA pooling
        dropout: Dropout probability
        selection_temperature: Temperature for cell type selection (higher = softer)

    Shape:
        - cells: (batch, n_cell_types, max_cells, n_genes) cell expression
        - cell_mask: (batch, n_cell_types, max_cells) valid cell mask
        - Output: (batch, n_cell_types, d_model) weighted embeddings for ALL types

    Note:
        Unlike the previous hard top-k selection, this version:
        1. Processes all 31 cell types (not just k selected)
        2. Weights each embedding by learned soft attention
        3. Is fully differentiable (gradients flow to selection_logits)
    """

    def __init__(
        self,
        n_genes: int,
        n_cell_types: int = 31,
        d_model: int = 128,
        n_heads: int = 4,
        n_isab_layers: int = 2,
        n_inducing: int = 32,
        n_pma_seeds: int = 1,
        dropout: float = 0.1,
        selection_temperature: float = 1.0,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()

        if n_genes <= 0:
            raise ValueError(f"n_genes must be positive, got {n_genes}")
        if n_cell_types <= 0:
            raise ValueError(f"n_cell_types must be positive, got {n_cell_types}")

        self.n_genes = n_genes
        self.n_cell_types = n_cell_types
        self.d_model = d_model

        # Cell type selector (soft attention weights, differentiable)
        self.selector = CellTypeSelector(
            n_cell_types=n_cell_types,
            temperature=selection_temperature,
        )

        # Set Transformer encoder (shared across all cell types)
        self.set_encoder = SetTransformerEncoder(
            d_input=n_genes,
            d_model=d_model,
            n_heads=n_heads,
            n_isab_layers=n_isab_layers,
            n_inducing=n_inducing,
            n_pma_seeds=n_pma_seeds,
            dropout=dropout,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

    @property
    def selection_temperature(self) -> float:
        """Current temperature for cell type selection."""
        return self.selector.temperature

    @selection_temperature.setter
    def selection_temperature(self, value: float) -> None:
        """Set temperature for cell type selection."""
        self.selector.temperature = value

    def forward(
        self,
        cells: torch.Tensor,
        cell_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        apply_selection_weights: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode cell-level data for ALL cell types with soft selection weighting.

        Uses batched processing for efficiency: all cell types are processed
        together in a single forward pass through SetTransformerEncoder, which
        is ~31x faster than sequential processing.

        Args:
            cells: Cell expression (batch, n_cell_types, max_cells, n_genes)
            cell_mask: Valid cell mask (batch, n_cell_types, max_cells), True=valid
            return_attention: Whether to return PMA attention weights
            apply_selection_weights: Whether to scale embeddings by selection weights.
                Setting False disables gradient flow to CellTypeSelector.selection_logits
                (the parameter will not learn). Use only for ablation studies.
                CognitiveResilienceModel always passes True.

        Returns:
            embeddings: (batch, n_cell_types, d_model) weighted embeddings for ALL types
            selection_weights: (n_cell_types,) soft attention weights (sum to 1)
            attention: Attention weights [B, n_cell_types, n_heads, n_seeds, max_cells] (if requested)

        Note:
            Selection weights are differentiable - gradients flow back to
            selector.selection_logits, allowing the model to learn which
            cell types are most important for the prediction task.
        """
        if cells.dim() != 4:
            raise ValueError(
                f"Expected 4D cells (batch, n_cell_types, max_cells, n_genes), "
                f"got shape {cells.shape}"
            )

        batch_size, n_ct, max_cells, n_genes = cells.shape

        if n_ct != self.n_cell_types:
            raise ValueError(
                f"Expected {self.n_cell_types} cell types, got {n_ct}"
            )
        if n_genes != self.n_genes:
            raise ValueError(
                f"Expected {self.n_genes} genes, got {n_genes}"
            )

        # Get soft selection weights (differentiable)
        selection_weights = self.selector.get_selection_weights()  # (n_cell_types,)

        # ─────────────────────────────────────────────────────────────────────
        # BATCHED PROCESSING: Process all cell types in one forward pass
        # ─────────────────────────────────────────────────────────────────────
        # Reshape from [B, n_cell_types, max_cells, n_genes]
        #           to [B * n_cell_types, max_cells, n_genes]
        cells_flat = cells.view(batch_size * self.n_cell_types, max_cells, n_genes)

        # Similarly reshape mask
        mask_flat = None
        if cell_mask is not None:
            mask_flat = cell_mask.view(batch_size * self.n_cell_types, max_cells)

        # Single forward pass through Set Transformer
        embeddings_flat, attention_flat = self.set_encoder(
            cells_flat, mask=mask_flat, return_attention=return_attention
        )
        # embeddings_flat: [B * n_cell_types, d_model]
        # attention_flat: [B * n_cell_types, n_heads, n_seeds, max_cells] or None

        # Reshape back to [B, n_cell_types, d_model]
        embeddings = embeddings_flat.view(batch_size, self.n_cell_types, self.d_model)

        # Handle attention weights for interpretability
        attention_out = None
        if return_attention and attention_flat is not None:
            # Reshape: [B * n_cell_types, n_heads, n_seeds, max_cells] -> [B, n_cell_types, ...]
            attn_shape = attention_flat.shape[1:]
            attention_out = attention_flat.view(batch_size, self.n_cell_types, *attn_shape)

        # Apply selection weights (differentiable scaling)
        if apply_selection_weights:
            # Scale each cell type's embedding by its selection weight
            # selection_weights: (n_cell_types,) -> (1, n_cell_types, 1)
            weights = selection_weights.view(1, -1, 1)
            embeddings = embeddings * weights

        return embeddings, selection_weights.detach(), attention_out

    def get_selection_weights(self) -> torch.Tensor:
        """
        Get soft selection weights for all cell types.

        Returns:
            Tensor of shape (n_cell_types,) with selection probabilities
        """
        return self.selector.get_selection_weights()

    def extra_repr(self) -> str:
        return (
            f"n_genes={self.n_genes}, n_cell_types={self.n_cell_types}, "
            f"d_model={self.d_model}, temperature={self.selection_temperature}"
        )