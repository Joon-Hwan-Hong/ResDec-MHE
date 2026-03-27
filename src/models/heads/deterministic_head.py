"""
Deterministic prediction head for cognition prediction without uncertainty quantification.

This is a simple fallback when Bayesian inference is too slow or uncertainty
quantification is not needed. Uses a 3-layer MLP with GELU activations.

Design Decision (2026-01-28):
- 3 Linear layers: 2-layer MLP is minimum for universal approximation;
  3 layers adds capacity without being excessive
- d_hidden bottleneck: Compresses features, acts as regularization
- GELU activation: Smoother than ReLU, matches transformer conventions
- No final activation: Regression output should be unbounded
- Dropout between layers: Configurable regularization for small sample sizes
"""

import torch
import torch.nn as nn


class DeterministicPredictionHead(nn.Module):
    """
    Simple deterministic prediction head (fallback if Bayesian is too slow).

    Architecture: Linear -> GELU -> Dropout -> Linear -> GELU -> Dropout -> Linear

    Design rationale:
    - 3 Linear layers: 2-layer MLP is minimum for universal approximation;
      3 layers adds capacity without being excessive
    - d_hidden bottleneck: Compresses features, acts as regularization
    - GELU activation: Smoother than ReLU, matches transformer conventions
    - No final activation: Regression output should be unbounded
    - Dropout between layers: Configurable regularization for small sample sizes (~400 subjects)

    Args:
        d_input: Input feature dimension (from attended features)
        d_hidden: Hidden layer dimension (default: 64)
        dropout: Dropout probability between layers (default: 0.1). Must be in [0, 1).

    Example:
        >>> head = DeterministicPredictionHead(d_input=128, d_hidden=64, dropout=0.1)
        >>> x = torch.randn(8, 128)  # batch of attended features
        >>> prediction = head(x)
        >>> print(prediction.shape)  # [8, 1]
    """

    def __init__(self, d_input: int, d_hidden: int = 64, dropout: float = 0.1):
        super().__init__()

        # Input validation
        if d_input <= 0:
            raise ValueError(f"d_input must be positive, got {d_input}")
        if d_hidden <= 0:
            raise ValueError(f"d_hidden must be positive, got {d_hidden}")
        if not (0 <= dropout < 1):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.d_input = d_input
        self.d_hidden = d_hidden
        self.dropout_rate = dropout

        # No LayerNorm by design: upstream attended features are already
        # normalized, and the 3-layer MLP with dropout provides sufficient
        # regularization for ~400 training subjects.
        self.mlp = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [B, d_input] - attended features

        Returns:
            prediction: [B, 1]

        Raises:
            ValueError: If input dimensions are incorrect
        """
        # Input validation
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input, got shape {x.shape}")
        if x.size(1) != self.d_input:
            raise ValueError(f"Expected d_input={self.d_input}, got {x.size(1)}")

        return self.mlp(x)

    def extra_repr(self) -> str:
        """Return extra representation string for this module."""
        return f"d_input={self.d_input}, d_hidden={self.d_hidden}, dropout={self.dropout_rate}"
