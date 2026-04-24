"""
Cell Transformer branch for cell-level heterogeneity modeling.

Uses SetTransformerEncoder to capture within-cell-type variation for all cell types.

Architecture:
    Input: Flat cell data [total_cells, n_genes] + offsets [B, n_types + 1]
    -> Gene gate -> input_proj on flat data -> Pad to [B*n_types, max_cells, d_model]
    -> SetTransformerEncoder (per cell type, all 31 types)
    -> Output: Embeddings [batch, n_cell_types, d_embed]
"""

from typing import Optional

import torch
import torch.nn as nn

from src.models.components.gene_attention_gate import GeneAttentionGate
from src.models.components.set_transformer import SetTransformerEncoder


class CellTransformer(nn.Module):
    """
    Cell-level transformer for modeling within-cell-type heterogeneity.

    Processes ALL cell types through SetTransformerEncoder to produce
    per-cell-type embeddings.

    Args:
        n_genes: Number of input genes
        n_cell_types: Total number of cell types (default: 31)
        d_model: Model dimension for Set Transformer
        n_heads: Number of attention heads
        n_isab_layers: Number of ISAB blocks
        n_inducing: Number of inducing points for ISAB
        n_pma_seeds: Number of seed vectors for PMA pooling
        dropout: Dropout probability
        use_gradient_checkpointing: Whether to use gradient checkpointing for memory savings
        condition_on_cell_type: Whether to use cell-type-conditioned inducing points.
            When True (default), each cell type gets a learned offset added to the
            shared inducing points, allowing the encoder to specialize per type.
    Shape:
        - cell_data: (total_cells, n_genes) concatenated cell expression
        - cell_offsets: (B, n_cell_types + 1) cumulative offsets into cell_data
        - Output: (B, n_cell_types, n_pma_seeds * d_model) embeddings for ALL types
            When n_pma_seeds=1, output is (B, n_cell_types, d_model).
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
        use_gradient_checkpointing: bool = False,
        condition_on_cell_type: bool = True,
        gene_gate_temperature: float = 2.0,  # Unused — kept for config backward compat
    ):
        super().__init__()

        if n_genes <= 0:
            raise ValueError(f"n_genes must be positive, got {n_genes}")
        if n_cell_types <= 0:
            raise ValueError(f"n_cell_types must be positive, got {n_cell_types}")

        self.n_genes = n_genes
        self.n_cell_types = n_cell_types
        self.d_model = d_model
        self.n_pma_seeds = n_pma_seeds
        self.condition_on_cell_type = condition_on_cell_type

        # Gene attention gate: cell-type-specific gene re-weighting
        self.gene_gate = GeneAttentionGate(
            n_cell_types=n_cell_types,
            n_genes=n_genes,
            temperature=gene_gate_temperature,
        )

        # Input projection: applied to flat cell data BEFORE padding.
        # This projects [total_cells, n_genes] -> [total_cells, d_model] so
        # the padded tensor is [B*n_types, max_cells, d_model] (~0.13 GB)
        # instead of [B*n_types, max_cells, n_genes] (~9.5 GB).
        self.input_proj = nn.Sequential(
            nn.Linear(n_genes, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        # Set Transformer encoder (shared across all cell types)
        # external_proj=True: input_proj is applied here in CellTransformer
        # before padding, so SetTransformerEncoder skips its own projection.
        self.set_encoder = SetTransformerEncoder(
            d_input=d_model,
            d_model=d_model,
            n_heads=n_heads,
            n_isab_layers=n_isab_layers,
            n_inducing=n_inducing,
            n_pma_seeds=n_pma_seeds,
            dropout=dropout,
            use_gradient_checkpointing=use_gradient_checkpointing,
            n_cell_types=n_cell_types if condition_on_cell_type else None,
            external_proj=True,
        )

    def forward(
        self,
        cell_data: torch.Tensor,       # [total_cells, n_genes]
        cell_offsets: torch.Tensor,     # [B, n_types + 1]
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass from flat cell representation.

        Reconstructs padded groups from flat representation and processes
        through SetTransformerEncoder.

        Args:
            cell_data: [total_cells, n_genes] concatenated cell expressions
            cell_offsets: [B, n_types + 1] absolute offsets into cell_data
            return_attention: Whether to return PMA attention weights

        Returns:
            embeddings: (B, n_cell_types, n_pma_seeds * d_model) embeddings.
                Each cell type carries n_pma_seeds distinct subpopulation summaries
                concatenated along the feature dim. When n_pma_seeds=1, shape is
                (B, n_cell_types, d_model).
            attention: Attention weights (if requested), else None
        """
        B = cell_offsets.shape[0]
        n_types = self.n_cell_types
        device = cell_data.device

        # Compute per-(sample, type) cell counts
        counts = cell_offsets[:, 1:] - cell_offsets[:, :-1]  # [B, n_types]

        total_cells = cell_data.shape[0]

        # Apply gene gate before projection
        if total_cells > 0:
            counts_flat = counts.reshape(-1)  # [B * n_types]
            scaled_gate = self.gene_gate.get_gate_weights()
            type_indices = torch.arange(n_types, device=device).repeat(B)
            per_cell_type = torch.repeat_interleave(type_indices, counts_flat)
            cell_data = cell_data * scaled_gate[per_cell_type].to(cell_data.dtype)

        # Project flat cell data BEFORE padding: [total_cells, n_genes] -> [total_cells, d_model]
        # Projecting before padding keeps the padded tensor at [B*n_types, max_cells, d_model]
        # instead of [B*n_types, max_cells, n_genes], which is orders of magnitude smaller
        # once n_genes is in the thousands.
        if total_cells > 0:
            cell_data = self.input_proj(cell_data)

        max_cells = max(int(counts.max().item()), 1) if counts.numel() > 0 else 1

        # Build padded tensor [B * n_types, max_cells, d_model]
        pad_dim = self.d_model
        cells_grouped = torch.zeros(
            B * n_types, max_cells, pad_dim,
            device=device, dtype=cell_data.dtype if total_cells > 0 else torch.float32,
        )
        mask_grouped = torch.zeros(
            B * n_types, max_cells,
            device=device, dtype=torch.bool,
        )

        total_cells = cell_data.shape[0]
        if total_cells > 0:
            starts = cell_offsets[:, :-1].reshape(-1)    # [B * n_types]
            counts_flat = counts.reshape(-1)             # [B * n_types]

            # Row: which group (b*n_types + t) each cell belongs to
            group_idx = torch.arange(B * n_types, device=device)
            row_idx = torch.repeat_interleave(group_idx, counts_flat)

            # Col: position within its group (0, 1, ..., count-1)
            group_starts = torch.repeat_interleave(starts, counts_flat)
            col_idx = torch.arange(total_cells, device=device) - group_starts

            cells_grouped[row_idx, col_idx] = cell_data
            mask_grouped[row_idx, col_idx] = True

        # Generate cell type indices for conditioned inducing points
        ct_idx = None
        if self.condition_on_cell_type:
            ct_idx = torch.arange(
                n_types, device=cell_data.device
            ).repeat(B)

        # SetTransformerEncoder forward
        embeddings_flat, attention_flat = self.set_encoder(
            cells_grouped, mask=mask_grouped, return_attention=return_attention,
            ct_idx=ct_idx,
        )

        # When n_pma_seeds > 1, SetTransformerEncoder returns
        # (B*n_types, n_pma_seeds, d_model). Reshape to concatenate seeds
        # along feature dim: (B, n_types, n_pma_seeds * d_model).
        # This preserves each seed's distinct subpopulation summary for
        # the fusion layer to learn how to integrate.
        if self.n_pma_seeds > 1:
            embeddings_flat = embeddings_flat.reshape(
                B * n_types, self.n_pma_seeds * self.d_model
            )  # (B*n_types, n_pma_seeds * d_model)
        embeddings = embeddings_flat.reshape(B, n_types, -1)

        attention_out = None
        if return_attention and attention_flat is not None:
            attn_shape = attention_flat.shape[1:]
            attention_out = attention_flat.view(B, n_types, *attn_shape)

        return embeddings, attention_out

    @property
    def gene_gate_temperature(self) -> float:
        return self.gene_gate.temperature

    @gene_gate_temperature.setter
    def gene_gate_temperature(self, value: float) -> None:
        self.gene_gate.temperature = value

    def extra_repr(self) -> str:
        return (
            f"n_genes={self.n_genes}, n_cell_types={self.n_cell_types}, "
            f"d_model={self.d_model}, "
            f"condition_on_cell_type={self.condition_on_cell_type}"
        )