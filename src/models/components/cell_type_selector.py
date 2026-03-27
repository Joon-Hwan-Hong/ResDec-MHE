"""
Cell Type Selector for choosing which cell types get cell-level modeling.

Learns which cell types benefit from fine-grained Set Transformer processing
versus just using pseudobulk representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CellTypeSelector(nn.Module):
    """
    Learns which cell types need fine-grained cell-level modeling.

    Uses attention-based selection with temperature-controlled softmax.
    All 31 cell types are processed through Set Transformer with soft
    attention weighting. After training, get_selected_types(k) can extract
    the top-k most important types for interpretability analysis.

    Args:
        n_cell_types: Number of cell types (default: 31 for Allen ABC)
        temperature: Temperature for softmax (higher = softer selection)
        init_uniform: If True, initialize for uniform selection

    Shape:
        - get_selection_weights(): Returns (n_cell_types,) selection probabilities
        - get_selected_types(k): Returns (k,) indices of top-k selected types
    """

    def __init__(
        self,
        n_cell_types: int = 31,
        temperature: float = 1.0,
        # Note: Fixed temperature by design. Annealing deferred per design doc recommendation
        # (Part 2, § Training Callbacks). Revisit if training diagnostics show the model
        # struggles to differentiate cell type importance.
        init_uniform: bool = True,
    ):
        super().__init__()

        if n_cell_types <= 0:
            raise ValueError(f"n_cell_types must be positive, got {n_cell_types}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        self.n_cell_types = n_cell_types
        self.register_buffer("_temperature_buf", torch.tensor(float(temperature)))

        # Selection logits: learned importance for each cell type
        if init_uniform:
            self.selection_logits = nn.Parameter(torch.zeros(n_cell_types))
        else:
            self.selection_logits = nn.Parameter(
                torch.randn(n_cell_types) * 0.01
            )

    @property
    def temperature(self) -> float:
        """Current temperature value."""
        return self._temperature_buf.item()

    @temperature.setter
    def temperature(self, value: float) -> None:
        """Set temperature (with validation)."""
        if value <= 0:
            raise ValueError(f"temperature must be positive, got {value}")
        self._temperature_buf.fill_(value)

    def forward(self) -> torch.Tensor:
        """
        Get soft selection weights (probabilities).

        Returns:
            Selection weights (n_cell_types,) summing to 1
        """
        return self.get_selection_weights()

    def get_selection_weights(self) -> torch.Tensor:
        """
        Get soft selection weights for all cell types.

        Returns:
            Tensor of shape (n_cell_types,) with selection probabilities
        """
        # AMP note: selection_logits is an nn.Parameter (always float32, not
        # autocasted), and _temperature_buf is a float32 buffer. The entire
        # computation stays float32. PyTorch autocast also promotes softmax
        # to float32 natively, so no explicit .float() promotion is needed.
        return F.softmax(self.selection_logits / self._temperature_buf, dim=0)

    def get_selected_types(self, k: int) -> torch.Tensor:
        """
        Get indices of top-k selected cell types.

        Args:
            k: Number of cell types to select

        Returns:
            Tensor of shape (k,) with cell type indices, sorted by importance
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if k > self.n_cell_types:
            raise ValueError(
                f"k ({k}) cannot exceed n_cell_types ({self.n_cell_types})"
            )

        _, indices = torch.topk(self.selection_logits, k)
        return indices

    def get_ranking(self) -> torch.Tensor:
        """
        Get cell type indices sorted by selection importance (descending).

        Returns:
            Tensor of shape (n_cell_types,) with sorted cell type indices
        """
        _, indices = torch.sort(self.selection_logits, descending=True)
        return indices

    def extra_repr(self) -> str:
        return f"n_cell_types={self.n_cell_types}, temperature={self._temperature_buf.item()}"
