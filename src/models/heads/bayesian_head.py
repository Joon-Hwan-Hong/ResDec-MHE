"""
Bayesian prediction head for cognition prediction with uncertainty quantification.

Uses Pyro for variational inference with weight uncertainty. The design separates
epistemic uncertainty (from weight priors on fc1, fc2, fc_mean) from aleatoric
uncertainty (learned from data via deterministic fc_log_std).

Design Decision (2026-01-27):
- Priors on fc1, fc2, fc_mean layers for epistemic uncertainty
- fc_log_std is deterministic to maintain clean separation of uncertainties
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.constants import EPSILON_POSITIVE_FLOOR
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample


def _make_normal_prior(shape: list[int], head_module: "BayesianPredictionHead"):
    """Create a device-aware Normal prior that follows the module's device.

    Uses a sentinel buffer registered on head_module that moves with .to().
    The prior samples on whatever device the sentinel is currently on.
    """
    def prior_fn(module):
        device = head_module._device_sentinel.device
        return dist.Normal(
            torch.zeros(shape, device=device),
            torch.ones(shape, device=device),
        ).to_event(len(shape))
    return prior_fn


class BayesianPredictionHead(PyroModule):
    """
    Bayesian regression head for cognition prediction.

    Outputs:
    - mean: Point prediction of cognition
    - std: Uncertainty estimate (epistemic + aleatoric)

    Uses Pyro for variational inference with weight uncertainty.

    Prior placement (2026-01-27 design):
    - fc1, fc2, fc_mean: Have priors for epistemic uncertainty
    - fc_log_std: Deterministic, learns aleatoric uncertainty from data

    GPU support:
        Priors are device-aware and follow ``.to(device)`` calls. A sentinel
        buffer (``_device_sentinel``) is registered so that ``.to(cuda)`` moves
        it alongside other parameters; prior factory functions read the
        sentinel's device at sample time to create distribution tensors on the
        correct device.

    Args:
        d_input: Input feature dimension (from attended features)
        d_hidden: Hidden layer dimension (default: 64)

    Example:
        >>> head = BayesianPredictionHead(d_input=128, d_hidden=64)
        >>> x = torch.randn(8, 128)  # batch of attended features
        >>> mean, std = head(x)  # prediction with uncertainty
        >>> print(mean.shape, std.shape)  # [8, 1], [8, 1]
    """

    def __init__(self, d_input: int, d_hidden: int = 64):
        super().__init__()

        # Validate inputs
        if d_input <= 0:
            raise ValueError(f"d_input must be positive, got {d_input}")
        if d_hidden <= 0:
            raise ValueError(f"d_hidden must be positive, got {d_hidden}")

        self.d_input = d_input
        self.d_hidden = d_hidden

        # No LayerNorm by design: the upstream attended features are already
        # normalized (PseudoBulkEncoder uses LayerNorm). Adding LayerNorm here
        # would mask the signal scale that fc_log_std needs to learn aleatoric
        # uncertainty, since LayerNorm collapses variance information.
        self.fc1 = PyroModule[nn.Linear](d_input, d_hidden)
        self.fc2 = PyroModule[nn.Linear](d_hidden, d_hidden)
        self.fc_mean = PyroModule[nn.Linear](d_hidden, 1)

        # fc_log_std is deterministic (aleatoric uncertainty only)
        self.fc_log_std = nn.Linear(d_hidden, 1)

        # ELBO likelihood scaling factor (set to world_size for DDP)
        self._data_scale = 1.0

        # Device sentinel — moves with .to(device), priors read its device at sample time
        self.register_buffer("_device_sentinel", torch.empty(0))

        # Device-aware priors: sample on current device at call time
        self.fc1.weight = PyroSample(_make_normal_prior([d_hidden, d_input], self))
        self.fc1.bias = PyroSample(_make_normal_prior([d_hidden], self))

        self.fc2.weight = PyroSample(_make_normal_prior([d_hidden, d_hidden], self))
        self.fc2.bias = PyroSample(_make_normal_prior([d_hidden], self))

        self.fc_mean.weight = PyroSample(_make_normal_prior([1, d_hidden], self))
        self.fc_mean.bias = PyroSample(_make_normal_prior([1], self))

    def set_data_scale(self, scale: float) -> None:
        """Set ELBO likelihood scaling factor (world_size for DDP)."""
        self._data_scale = scale

    def forward(self, x: torch.Tensor, y: torch.Tensor = None):
        """
        Forward pass with optional observation for training.

        Args:
            x: [B, d_input] - attended features from PathologyStratifiedAttention
            y: [B, 1] observed cognition (None for prediction)

        Returns:
            mean: [B, 1] - point prediction
            std: [B, 1] - uncertainty estimate

        Raises:
            ValueError: If input dimensions are incorrect
        """
        # Input validation
        if x.dim() != 2:
            raise ValueError(
                f"Expected 2D input [B, d_input], got {x.dim()}D tensor"
            )
        if x.size(1) != self.d_input:
            raise ValueError(
                f"Expected d_input={self.d_input}, got {x.size(1)}"
            )

        # Validate y if provided
        if y is not None:
            if y.dim() != 2:
                raise ValueError(
                    f"Expected 2D y [B, 1], got {y.dim()}D tensor"
                )
            if y.size(0) != x.size(0):
                raise ValueError(
                    f"Batch size mismatch: x has {x.size(0)}, y has {y.size(0)}"
                )
            if y.size(1) != 1:
                raise ValueError(
                    f"Expected y feature dim=1, got {y.size(1)}"
                )

        # Forward pass
        h = F.gelu(self.fc1(x))
        h = F.gelu(self.fc2(h))

        mean = self.fc_mean(h)
        log_std = self.fc_log_std(h)
        std = F.softplus(log_std) + EPSILON_POSITIVE_FLOOR  # Ensure positive with minimum

        with pyro.plate("data", x.size(0)):
            with pyro.poutine.scale(scale=self._data_scale):
                pyro.sample("obs", dist.Normal(mean, std).to_event(1), obs=y)

        return mean, std

    def extra_repr(self) -> str:
        """Return extra representation string for this module."""
        return f"d_input={self.d_input}, d_hidden={self.d_hidden}"
