"""
Gene Attention Gate for cell-type-specific gene weighting.

The gate learns which genes are important for each cell type independently,
using sigmoid activation to produce per-gene gate values in [0, 1].

Gate value interpretation:
    - 0.0: gene is fully filtered (no signal passes)
    - 0.5: neutral (initialization point, maximum gradient)
    - 1.0: gene is fully passed

Design choice: Sigmoid replaces the earlier softmax gate.
Softmax produced uniform weights due to (1) zero init + high temperature
creating vanishing gradients over thousands of genes, and (2) the downstream
linear layer absorbing gene selection, removing gradient signal from the gate.
Sigmoid avoids both issues: zero init places all genes at the steepest
gradient point (sigmoid'(0) = 0.25), and independent gating avoids the
competitive dilution of softmax over thousands of genes.
"""

import torch
import torch.nn as nn


class GeneAttentionGate(nn.Module):
    """
    Cell-type-specific gene attention gate using sigmoid activation.

    Each cell type independently learns which genes to pass or filter.
    Gate values are in [0, 1] — directly interpretable as the fraction
    of gene signal passed through.

    Args:
        n_cell_types: Number of cell types (default: 31 for Allen ABC)
        n_genes: Number of genes in input
        temperature: Unused (kept for config backward compatibility). Ignored.
        init_uniform: If True (default), initialize logits to 0 (sigmoid=0.5).

    Shape:
        - Input: (batch, n_cell_types, n_genes)
        - Output: (batch, n_cell_types, n_genes)
    """

    def __init__(
        self,
        n_cell_types: int,
        n_genes: int,
        temperature: float = 1.0,  # Ignored — kept for config compatibility
        init_uniform: bool = True,
    ):
        super().__init__()

        if n_cell_types <= 0:
            raise ValueError(f"n_cell_types must be positive, got {n_cell_types}")
        if n_genes <= 0:
            raise ValueError(f"n_genes must be positive, got {n_genes}")
        # Reject ``temperature < 0.05`` outright instead of silently
        # smashing it to the 0.05 floor (the previous ``max(temperature,
        # 0.05)`` clamp made caller intent invisible). 0.05 is the lower
        # bound used by the temperature annealing schedule.
        if temperature < 0.05:
            raise ValueError(
                f"temperature must be >= 0.05 (got {temperature}); set "
                "0.05 as the minimum or pass a larger value."
            )

        self.n_cell_types = n_cell_types
        self.n_genes = n_genes

        # Gate logits: learned parameter for each (cell_type, gene) pair
        # Zero init: sigmoid(0) = 0.5 = maximum gradient point
        if init_uniform:
            self.gate_logits = nn.Parameter(torch.zeros(n_cell_types, n_genes))
        else:
            self.gate_logits = nn.Parameter(
                torch.randn(n_cell_types, n_genes) * 0.01
            )

        # Temperature buffer kept for backward compatibility with checkpoints
        # and the TemperatureAnnealing callback. Setting it is a no-op for
        # sigmoid gating, but we store it to avoid breaking checkpoint loading.
        # The constructor enforces ``temperature >= 0.05`` above, so no
        # additional clamp is needed.
        self.register_buffer("_temperature_buf", torch.tensor(float(temperature)))

    @property
    def temperature(self) -> float:
        """Temperature value (no-op for sigmoid gate, kept for compatibility)."""
        return self._temperature_buf.item()

    @temperature.setter
    def temperature(self, value: float) -> None:
        """Set temperature (no-op for sigmoid gate, kept for compatibility)."""
        if value < 0.05:
            raise ValueError(
                f"temperature must be >= 0.05 (got {value}); the previous "
                "clamp behaviour silently smashed values into the "
                "[0, 0.05] band."
            )
        self._temperature_buf.fill_(float(value))

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

        # Sigmoid gate: each gene independently gated in [0, 1]
        gate = self.get_gate_weights()

        # Apply gating (broadcast over batch dimension)
        # Cast gate to input dtype to avoid bf16/float32 issues under AMP
        return x * gate.unsqueeze(0).to(x.dtype)

    def get_gate_weights(self) -> torch.Tensor:
        """
        Get current gate weights for interpretability.

        Returns:
            Gate weights of shape (n_cell_types, n_genes), each value in [0, 1].
            Values > 0.5 = gene passes more signal, < 0.5 = gene is filtered.
        """
        return torch.sigmoid(self.gate_logits.float())

    def get_top_genes_per_cell_type(
        self, k: int = 100, gene_names: list[str] | None = None
    ) -> dict[int, list[tuple[int | str, float]]]:
        """
        Get top-k genes by gate weight for each cell type.

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

    def get_selected_genes(
        self, threshold: float = 0.5, gene_names: list[str] | None = None
    ) -> dict[int, list[tuple[int | str, float]]]:
        """
        Get genes above threshold for each cell type.

        Args:
            threshold: Gate value threshold (default: 0.5 = above initialization)
            gene_names: Optional list of gene names

        Returns:
            Dict mapping cell type index to list of (gene_id/name, weight) tuples
        """
        weights = self.get_gate_weights().detach()

        results = {}
        for ct_idx in range(self.n_cell_types):
            ct_weights = weights[ct_idx]
            mask = ct_weights > threshold
            indices = mask.nonzero(as_tuple=True)[0]
            values = ct_weights[indices]

            # Sort by weight descending
            sorted_order = values.argsort(descending=True)
            indices = indices[sorted_order]
            values = values[sorted_order]

            if gene_names is not None:
                genes = [
                    (gene_names[idx.item()], val.item())
                    for idx, val in zip(indices, values)
                ]
            else:
                genes = [
                    (idx.item(), val.item())
                    for idx, val in zip(indices, values)
                ]

            results[ct_idx] = genes

        return results

    def extra_repr(self) -> str:
        return (
            f"n_cell_types={self.n_cell_types}, "
            f"n_genes={self.n_genes}, "
            f"activation=sigmoid"
        )
