"""
Gene Attention Gate for cell-type-specific gene weighting.

The gate learns which genes are important for each cell type independently,
using softmax attention to weight gene expression values.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneAttentionGate(nn.Module):
    """
    Cell-type-specific gene attention gate.

    Each cell type learns which genes to attend to.
    Softmax ensures weights sum to 1 per cell type.
    Temperature annealing: high τ (soft) → low τ (sharp/sparse).

    Args:
        n_cell_types: Number of cell types (default: 31 for Allen ABC)
        n_genes: Number of genes in input
        temperature: Initial temperature for softmax (higher = softer)
        init_uniform: If True, initialize logits to 0 for uniform attention

    Shape:
        - Input: (batch, n_cell_types, n_genes)
        - Output: (batch, n_cell_types, n_genes)
    """

    def __init__(
        self,
        n_cell_types: int,
        n_genes: int,
        temperature: float = 1.0,
        init_uniform: bool = True,
    ):
        super().__init__()

        if n_cell_types <= 0:
            raise ValueError(f"n_cell_types must be positive, got {n_cell_types}")
        if n_genes <= 0:
            raise ValueError(f"n_genes must be positive, got {n_genes}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        self.n_cell_types = n_cell_types
        self.n_genes = n_genes
        self.register_buffer("_temperature_buf", torch.tensor(max(float(temperature), 0.05)))

        # Gate logits: learned parameter for each (cell_type, gene) pair
        # Initialize to zeros for uniform attention at start
        if init_uniform:
            self.gate_logits = nn.Parameter(torch.zeros(n_cell_types, n_genes))
        else:
            self.gate_logits = nn.Parameter(
                torch.randn(n_cell_types, n_genes) * 0.01
            )

    @property
    def temperature(self) -> float:
        """Current temperature value."""
        return self._temperature_buf.item()

    @temperature.setter
    def temperature(self, value: float) -> None:
        """Set temperature (with validation). Clamped to minimum 0.05."""
        if value <= 0:
            raise ValueError(f"temperature must be positive, got {value}")
        self._temperature_buf.fill_(max(value, 0.05))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply cell-type-specific gene gating.

        Args:
            x: Expression tensor of shape (batch, n_cell_types, n_genes)

        Returns:
            Gated expression tensor of same shape
        """
        if x.dim() != 3:
            raise ValueError(
                f"Expected 3D input (batch, n_cell_types, n_genes), got shape {x.shape}"
            )
        if x.size(1) != self.n_cell_types or x.size(2) != self.n_genes:
            raise ValueError(
                f"Input shape mismatch: expected (*, {self.n_cell_types}, {self.n_genes}), "
                f"got {x.shape}"
            )

        # Compute gate weights with temperature-controlled softmax
        gate = self.get_gate_weights()

        # Scale by n_genes to preserve input magnitude.
        # Softmax produces weights ~1/n_genes at initialization, which would
        # shrink inputs by ~4000x. Scaling by n_genes gives ~1.0 per gene at
        # init, so the downstream MLP receives properly-scaled inputs.
        # This also eliminates weight decay asymmetry on the first MLP layer
        # (see review discussion: Option A for gene gate scaling).
        scaled_gate = gate * self.n_genes

        # Apply gating (broadcast over batch dimension)
        # scaled_gate: [n_cell_types, n_genes] -> [1, n_cell_types, n_genes]
        # Cast gate to input dtype to avoid unnecessary bf16→float32→bf16 round-trip:
        # get_gate_weights() returns float32 (for softmax stability), but the
        # downstream MLP's nn.Linear will re-downcast under autocast anyway.
        return x * scaled_gate.unsqueeze(0).to(x.dtype)

    def get_gate_weights(self) -> torch.Tensor:
        """
        Get current gate weights as probabilities for interpretability.

        Returns probabilities (sum to 1 per cell type), NOT the scaled values
        used in forward(). For the actual scaling applied during forward pass,
        these weights are multiplied by n_genes.

        Returns:
            Gate weights of shape (n_cell_types, n_genes), each row sums to 1
        """
        # Use scalar tensor directly (no .item()) to avoid torch.compile graph break.
        # Division by 0-d tensor works via broadcasting — numerically identical.
        # Promote to float32 for softmax stability under AMP.
        # At low temperature (tau→0.05), logits/tau amplifies by 20x.
        # Float16 exp() overflows at ~11.09 — any logit > 0.55 would produce NaN.
        return F.softmax((self.gate_logits / self._temperature_buf).float(), dim=-1)

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
        weights = self.get_gate_weights().detach()
        k = min(k, self.n_genes)

        results = {}
        for ct_idx in range(self.n_cell_types):
            ct_weights = weights[ct_idx]
            top_values, top_indices = torch.topk(ct_weights, k)

            if gene_names is not None:
                genes = [
                    (gene_names[idx.item()], val.item())
                    for idx, val in zip(top_indices, top_values)
                ]
            else:
                genes = [
                    (idx.item(), val.item())
                    for idx, val in zip(top_indices, top_values)
                ]

            results[ct_idx] = genes

        return results

    def extra_repr(self) -> str:
        return (
            f"n_cell_types={self.n_cell_types}, "
            f"n_genes={self.n_genes}, "
            f"temperature={self._temperature_buf.item()}"
        )