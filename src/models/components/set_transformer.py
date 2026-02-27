"""
Set Transformer components for cell-level modeling.

Implements the Set Transformer architecture from Lee et al. (2019):
"Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks"

Key components:
- MultiheadAttentionBlock (MAB): Core attention building block
- ISAB: Induced Set Attention Block for O(n*m) complexity
- PMA: Pooling by Multihead Attention for permutation-invariant aggregation
- SetTransformerEncoder: Full encoder combining ISAB layers + PMA pooling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MultiheadAttentionBlock(nn.Module):
    """
    Multihead attention with layer norm and residual connection.

    Used as the core building block for ISAB and PMA.
    Follows the standard transformer pattern:
        Attention → Add & Norm → FFN → Add & Norm

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        dropout: Dropout probability
        use_ffn: Whether to include feed-forward network
        ffn_expansion: FFN hidden dimension multiplier

    Shape:
        - query: (batch, n_query, d_model)
        - key_value: (batch, n_kv, d_model)
        - output: (batch, n_query, d_model)
        - attention: (batch, n_heads, n_query, n_kv) if return_attention=True
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        use_ffn: bool = True,
        ffn_expansion: int = 4,
    ):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )

        self.d_model = d_model
        self.n_heads = n_heads

        # Multihead attention
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model) if use_ffn else None

        # Optional feed-forward network
        self.use_ffn = use_ffn
        if use_ffn:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_model * ffn_expansion),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * ffn_expansion, d_model),
                nn.Dropout(dropout),
            )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with optional attention weight extraction.

        Args:
            query: Query tensor (batch, n_query, d_model)
            key_value: Key/value tensor (batch, n_kv, d_model)
            key_padding_mask: Boolean mask (batch, n_kv), True = ignore
            return_attention: Whether to return attention weights

        Returns:
            output: (batch, n_query, d_model)
            attention: (batch, n_heads, n_query, n_kv) or None
        """
        # Multihead attention
        attn_output, attn_weights = self.attention(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask,
            need_weights=return_attention,
            average_attn_weights=False,  # Keep per-head weights
        )

        # Residual + LayerNorm
        x = self.norm1(query + self.dropout(attn_output))

        # Optional FFN with residual
        if self.use_ffn:
            x = self.norm2(x + self.ffn(x))

        if return_attention:
            return x, attn_weights
        return x, None


class ISAB(nn.Module):
    """
    Induced Set Attention Block.

    Uses m inducing points as a bottleneck for O(n*m) complexity
    instead of O(n²) for full self-attention.

    The two-step process:
    1. Inducing points attend to input set → compressed representation
    2. Input set attends to inducing points → updated set

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        n_inducing: Number of inducing points (m)
        dropout: Dropout probability

    Shape:
        - Input: (batch, n_cells, d_model)
        - Output: (batch, n_cells, d_model)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_inducing: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_inducing = n_inducing

        # Learnable inducing points
        self.inducing_points = nn.Parameter(
            torch.randn(n_inducing, d_model) * 0.02
        )

        # Two MABs: one for each attention direction
        self.mab1 = MultiheadAttentionBlock(d_model, n_heads, dropout=dropout)
        self.mab2 = MultiheadAttentionBlock(d_model, n_heads, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, n_cells, d_model)
            mask: Boolean mask (batch, n_cells), True = valid cell

        Returns:
            Updated tensor (batch, n_cells, d_model)
        """
        batch_size = x.size(0)

        # Expand inducing points for batch
        inducing = self.inducing_points.unsqueeze(0).expand(batch_size, -1, -1)

        # Create padding mask for attention (True = ignore)
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask  # Invert: True in mask = valid

        # Step 1: Inducing points attend to cells → compressed
        h, _ = self.mab1(inducing, x, key_padding_mask=key_padding_mask)

        # Step 2: Cells attend to inducing points → updated cells
        output, _ = self.mab2(x, h)

        return output


class PMA(nn.Module):
    """
    Pooling by Multihead Attention.

    Learnable seed vectors attend to cells for permutation-invariant pooling.
    Attention weights are interpretable as cell importance.

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        n_seeds: Number of seed vectors (output dimension along set axis)
        dropout: Dropout probability

    Shape:
        - Input: (batch, n_cells, d_model)
        - Output: (batch, n_seeds, d_model)
        - Attention: (batch, n_heads, n_seeds, n_cells) if return_attention=True
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_seeds: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_seeds = n_seeds

        # Learnable seed vectors (Xavier initialization for proper scale)
        self.seed_vectors = nn.Parameter(torch.empty(n_seeds, d_model))
        nn.init.xavier_uniform_(self.seed_vectors)

        # MAB for pooling attention
        self.mab = MultiheadAttentionBlock(d_model, n_heads, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Pool set elements using attention.

        Args:
            x: Input tensor (batch, n_cells, d_model)
            mask: Boolean mask (batch, n_cells), True = valid cell
            return_attention: Whether to return attention weights

        Returns:
            pooled: (batch, n_seeds, d_model)
            attention: (batch, n_heads, n_seeds, n_cells) or None
        """
        batch_size = x.size(0)

        # Expand seeds for batch
        seeds = self.seed_vectors.unsqueeze(0).expand(batch_size, -1, -1)

        # Create padding mask (True = ignore)
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask  # Invert: True in mask = valid

        # Seeds attend to cells
        pooled, attention = self.mab(
            seeds, x,
            key_padding_mask=key_padding_mask,
            return_attention=return_attention,
        )

        return pooled, attention


class SetTransformerEncoder(nn.Module):
    """
    Complete Set Transformer encoder for cell-level modeling.

    Architecture:
        Input embedding → ISAB × n_layers → PMA → Output

    Handles edge cases:
        - All-masked cell types: Returns learned "empty" embedding instead of NaN
        - Empty sets: Same handling as all-masked

    Args:
        d_input: Input dimension (n_genes)
        d_model: Model dimension
        n_heads: Number of attention heads
        n_isab_layers: Number of ISAB blocks
        n_inducing: Number of inducing points for ISAB
        n_pma_seeds: Number of seed vectors for PMA pooling
        dropout: Dropout probability

    Shape:
        - Input: (batch, n_cells, d_input)
        - Mask: (batch, n_cells) boolean, True = valid cell
        - Output: (batch, n_pma_seeds, d_model) or (batch, d_model) if n_pma_seeds=1
    """

    def __init__(
        self,
        d_input: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_isab_layers: int = 2,
        n_inducing: int = 32,
        n_pma_seeds: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_input = d_input
        self.d_model = d_model
        self.n_pma_seeds = n_pma_seeds

        # Input embedding
        self.input_proj = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        # Stack of ISAB blocks
        self.isab_layers = nn.ModuleList([
            ISAB(d_model, n_heads, n_inducing, dropout)
            for _ in range(n_isab_layers)
        ])

        # PMA for pooling
        self.pma = PMA(d_model, n_heads, n_pma_seeds, dropout)

        # Learned embedding for empty sets (all cells masked)
        # This prevents NaN from attention on all-masked inputs
        self.empty_embedding = nn.Parameter(torch.zeros(d_model))

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode cell set to fixed-size representation.

        Args:
            x: Cell expression (batch, n_cells, d_input)
            mask: Valid cell mask (batch, n_cells), True = valid
            return_attention: Whether to return PMA attention weights

        Returns:
            embedding: (batch, d_model) if n_pma_seeds=1, else (batch, n_seeds, d_model)
            attention: PMA attention weights or None

        Note:
            Handles all-masked inputs (empty sets) by returning a learned
            empty_embedding instead of producing NaN from attention.
        """
        batch_size = x.size(0)
        device = x.device

        # Detect which samples have all cells masked (empty sets)
        if mask is not None:
            # has_valid_cells: True if at least one cell is valid
            has_valid_cells = mask.any(dim=1)  # (batch,)
            all_empty = ~has_valid_cells.any()  # All samples are empty
        else:
            has_valid_cells = torch.ones(batch_size, dtype=torch.bool, device=device)
            all_empty = False

        # If ALL samples are empty, return empty embeddings directly
        # This avoids any attention computation on all-masked inputs
        if all_empty:
            if self.n_pma_seeds == 1:
                pooled = self.empty_embedding.to(device).unsqueeze(0).expand(batch_size, -1)
            else:
                pooled = self.empty_embedding.to(device).unsqueeze(0).unsqueeze(0).expand(
                    batch_size, self.n_pma_seeds, -1
                )
            attention = None
            if return_attention:
                # Return zero attention weights for empty sets
                n_cells = x.size(1)
                attention = torch.zeros(
                    batch_size, self.pma.mab.n_heads, self.n_pma_seeds, n_cells,
                    device=device, dtype=x.dtype
                )
            return pooled, attention

        # Split batch into valid and empty sub-batches to prevent NaN gradient
        # propagation. nn.MultiheadAttention produces NaN for fully-masked inputs,
        # and torch.where doesn't block gradient flow through the NaN branch.
        if mask is not None and not has_valid_cells.all():
            # Mixed batch: some samples have valid cells, some don't
            valid_indices = has_valid_cells.nonzero(as_tuple=True)[0]
            n_valid = valid_indices.size(0)

            # Process only valid samples through attention
            x_valid = x[valid_indices]
            mask_valid = mask[valid_indices]

            h_valid = self.input_proj(x_valid)
            for isab in self.isab_layers:
                h_valid = isab(h_valid, mask_valid)
            pooled_valid, attention_valid = self.pma(
                h_valid, mask_valid, return_attention=return_attention
            )
            if self.n_pma_seeds == 1:
                pooled_valid = pooled_valid.squeeze(1)  # (n_valid, d_model)

            # Allocate full-batch output with empty_embedding for all samples
            if self.n_pma_seeds == 1:
                pooled = self.empty_embedding.unsqueeze(0).expand(batch_size, -1).clone()
            else:
                pooled = self.empty_embedding.unsqueeze(0).unsqueeze(0).expand(
                    batch_size, self.n_pma_seeds, -1
                ).clone()

            # Scatter valid results back into full-batch tensor
            pooled[valid_indices] = pooled_valid

            # Handle attention
            attention = None
            if return_attention and attention_valid is not None:
                n_cells = x.size(1)
                attention = torch.zeros(
                    batch_size, self.pma.mab.n_heads, self.n_pma_seeds, n_cells,
                    device=device, dtype=x.dtype
                )
                attention[valid_indices] = attention_valid
        else:
            # All samples have valid cells — process entire batch
            h = self.input_proj(x)
            for isab in self.isab_layers:
                h = isab(h, mask)
            pooled, attention = self.pma(h, mask, return_attention=return_attention)
            if self.n_pma_seeds == 1:
                pooled = pooled.squeeze(1)  # (batch, d_model)

        return pooled, attention
