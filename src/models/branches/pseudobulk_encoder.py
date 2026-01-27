"""
Pseudobulk Encoder (Branch 1) for cell-type-specific expression encoding.

Combines GeneAttentionGate with a shared MLP to encode pseudobulk expression
profiles into cell-type embeddings.

Architecture:
    Input [batch, 31, n_genes]
    → GeneAttentionGate (cell-type-specific gene weighting)
    → Shared MLP (n_genes → 512 → 256 → d_embed)
    → Output [batch, 31, d_embed]
"""

import torch
import torch.nn as nn

from src.models.components.gene_attention_gate import GeneAttentionGate


class PseudobulkEncoder(nn.Module):
    """
    Pseudobulk encoder combining gene attention gating with shared MLP.

    The encoder learns cell-type-specific gene importance via the attention gate,
    then projects all cell types through a shared MLP to preserve cross-cell-type
    relationships.

    Args:
        n_cell_types: Number of cell types (default: 31 for Allen ABC)
        n_genes: Number of input genes
        d_embed: Output embedding dimension
        mlp_hidden: Hidden layer dimensions for shared MLP
        dropout: Dropout probability
        temperature: Initial temperature for gene attention gate
        use_layer_norm: Whether to use LayerNorm in MLP

    Shape:
        - Input: (batch, n_cell_types, n_genes)
        - Output: (batch, n_cell_types, d_embed)
    """

    def __init__(
        self,
        n_cell_types: int,
        n_genes: int,
        d_embed: int = 128,
        mlp_hidden: list[int] | None = None,
        dropout: float = 0.1,
        temperature: float = 1.0,
        use_layer_norm: bool = True,
    ):
        super().__init__()

        if n_cell_types <= 0:
            raise ValueError(f"n_cell_types must be positive, got {n_cell_types}")
        if n_genes <= 0:
            raise ValueError(f"n_genes must be positive, got {n_genes}")
        if d_embed <= 0:
            raise ValueError(f"d_embed must be positive, got {d_embed}")

        self.n_cell_types = n_cell_types
        self.n_genes = n_genes
        self.d_embed = d_embed

        # Default MLP hidden dimensions from spec
        if mlp_hidden is None:
            mlp_hidden = [512, 256]
        self.mlp_hidden = mlp_hidden

        # Gene attention gate (cell-type-specific)
        self.gene_gate = GeneAttentionGate(
            n_cell_types=n_cell_types,
            n_genes=n_genes,
            temperature=temperature,
        )

        # Build shared MLP
        # Architecture: n_genes → 512 → 256 → d_embed (with LN + GELU)
        layers = []
        in_dim = n_genes

        for hidden_dim in mlp_hidden:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        # Final projection to d_embed (no activation after final layer)
        layers.append(nn.Linear(in_dim, d_embed))

        self.shared_mlp = nn.Sequential(*layers)

    @property
    def temperature(self) -> float:
        """Current temperature of the gene attention gate."""
        return self.gene_gate.temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        """Set temperature of the gene attention gate."""
        self.gene_gate.temperature = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode pseudobulk expression to cell-type embeddings.

        Args:
            x: Pseudobulk expression tensor (batch, n_cell_types, n_genes)

        Returns:
            Cell-type embeddings (batch, n_cell_types, d_embed)
        """
        if x.dim() != 3:
            raise ValueError(
                f"Expected 3D input (batch, n_cell_types, n_genes), got shape {x.shape}"
            )
        if x.size(1) != self.n_cell_types:
            raise ValueError(
                f"Expected {self.n_cell_types} cell types, got {x.size(1)}"
            )
        if x.size(2) != self.n_genes:
            raise ValueError(
                f"Expected {self.n_genes} genes, got {x.size(2)}"
            )

        # Apply gene attention gating
        # [batch, n_cell_types, n_genes] → [batch, n_cell_types, n_genes]
        gated = self.gene_gate(x)

        # Apply shared MLP to each cell type
        # The MLP is applied to the last dimension (genes → d_embed)
        # [batch, n_cell_types, n_genes] → [batch, n_cell_types, d_embed]
        batch_size = x.size(0)

        # Reshape for batch processing through MLP
        # [batch, n_cell_types, n_genes] → [batch * n_cell_types, n_genes]
        gated_flat = gated.view(-1, self.n_genes)

        # Apply MLP
        embeddings_flat = self.shared_mlp(gated_flat)

        # Reshape back
        # [batch * n_cell_types, d_embed] → [batch, n_cell_types, d_embed]
        embeddings = embeddings_flat.view(batch_size, self.n_cell_types, self.d_embed)

        return embeddings

    def get_gene_weights(self) -> torch.Tensor:
        """
        Get current gene attention weights for interpretability.

        Returns:
            Gate weights (n_cell_types, n_genes), each row sums to 1
        """
        return self.gene_gate.get_gate_weights()

    def get_top_genes_per_cell_type(
        self, k: int = 100, gene_names: list[str] | None = None
    ) -> dict[int, list[tuple[int | str, float]]]:
        """
        Get top-k genes by attention weight for each cell type.

        Args:
            k: Number of top genes to return per cell type
            gene_names: Optional list of gene names for readable output

        Returns:
            Dict mapping cell type index to list of (gene_id/name, weight) tuples
        """
        return self.gene_gate.get_top_genes_per_cell_type(k=k, gene_names=gene_names)

    def extra_repr(self) -> str:
        return (
            f"n_cell_types={self.n_cell_types}, "
            f"n_genes={self.n_genes}, "
            f"d_embed={self.d_embed}, "
            f"mlp_hidden={self.mlp_hidden}"
        )