"""
OGM-GE gradient modulation (Peng et al., CVPR 2022) adapted for multi-branch regression.

Original paper uses per-modality softmax prediction scores (classification) as the
discrepancy ratio. Our adaptation uses gradient norms since we have regression with
a Bayesian head.

Core principle: Monitor per-branch gradient norm discrepancy. Suppress dominant
branches (k < 1). Leave lagging branches unchanged (k = 1).

GE (Generalization Enhancement) injects noise computed from a second forward pass
with different dropout masks to improve generalization.
"""

import logging
import math
from contextlib import nullcontext

import torch
import lightning.pytorch as pl

from src.training.callbacks import BRANCH_NAMES

logger = logging.getLogger(__name__)


class OGMGEModulator:
    """Compute OGM-GE modulation coefficients from branch gradient norms.

    Implements the adapted OGM formula:
        rho_i = g_i / mean(g)
        k_i = 1 - tanh(alpha * rho_i) if rho_i > 1 (dominant -> suppressed)
        k_i = 1                       otherwise    (lagging -> unchanged)
    """

    def __init__(self, alpha: float = 1.0, branch_names: tuple[str, ...] = BRANCH_NAMES):
        self.alpha = alpha
        self.branch_names = branch_names

    def compute_k(self, branch_norms: dict[str, float]) -> dict[str, float]:
        """Compute modulation coefficients from branch gradient norms.

        Args:
            branch_norms: Dict mapping branch name to L2 gradient norm.

        Returns:
            Dict mapping branch name to modulation coefficient k_i.
        """
        values = [branch_norms[name] for name in self.branch_names]
        mean_norm = sum(values) / len(values) if values else 0.0

        # All-zero norms: no modulation
        if mean_norm == 0.0:
            return {name: 1.0 for name in self.branch_names}

        k = {}
        for name in self.branch_names:
            rho = branch_norms[name] / mean_norm
            if rho > 1.0:
                # Dominant branch: suppress
                k[name] = 1.0 - math.tanh(self.alpha * rho)
            else:
                # Lagging or equal branch: leave unchanged
                k[name] = 1.0
        return k


class GradientModulationCallback(pl.Callback):
    """
    Lightning callback implementing adapted OGM-GE.

    Runs in on_before_optimizer_step (after DDP allreduce, before optimizer step).
    Computes per-branch gradient norms, applies OGM modulation coefficients,
    and optionally injects GE noise for generalization.

    Args:
        alpha: Suppression sensitivity for OGM (higher = stronger suppression).
        ge_enabled: Whether to apply GE noise injection.
        log_modulation: Whether to log k_i values.
        log_every_n_steps: Logging frequency.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        ge_enabled: bool = True,
        log_modulation: bool = True,
        log_every_n_steps: int = 10,
    ):
        super().__init__()
        self.modulator = OGMGEModulator(alpha=alpha)
        self.ge_enabled = ge_enabled
        self.log_modulation = log_modulation
        self.log_every_n_steps = log_every_n_steps
        self._branch_params: dict[str, list[torch.nn.Parameter]] | None = None

    def _build_param_cache(self, model: torch.nn.Module) -> None:
        """Build parameter-to-branch mapping once, reused every call."""
        self._branch_params = {name: [] for name in BRANCH_NAMES}
        for param_name, param in model.named_parameters():
            for branch_name in BRANCH_NAMES:
                if branch_name in param_name:
                    self._branch_params[branch_name].append(param)
                    break

    def _compute_branch_norms(self, model: torch.nn.Module) -> dict[str, float]:
        """Compute L2 gradient norm for each encoder branch."""
        if self._branch_params is None:
            self._build_param_cache(model)

        branch_norms = {}
        for branch_name, params in self._branch_params.items():
            grad_params = [p for p in params if p.grad is not None]
            if grad_params:
                grad_norms = torch.stack(
                    [p.grad.data.norm(2) for p in grad_params]
                )
                branch_norms[branch_name] = grad_norms.norm(2).item()
            else:
                branch_norms[branch_name] = 0.0
        return branch_norms

    def on_before_optimizer_step(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, optimizer,
    ) -> None:
        """
        Apply OGM-GE gradient modulation.

        Steps:
        1. Compute per-branch gradient norms
        2. Compute modulation coefficients k_i
        3. Save unscaled branch gradients (for GE noise)
        4. Scale branch parameter gradients by k_i (OGM)
        5. Apply GE noise injection (if enabled)
        6. Log k_i values
        """
        # Only apply to Bayesian SVI training
        if not getattr(pl_module, '_use_bayesian_svi', False):
            return

        model = pl_module.model

        # Step 1: Compute per-branch gradient norms
        if self._branch_params is None:
            self._build_param_cache(model)
        branch_norms = self._compute_branch_norms(model)

        # Step 2: Compute modulation coefficients
        k = self.modulator.compute_k(branch_norms)

        # Step 3: Save unscaled branch gradients (for GE noise computation)
        unscaled_grads = {}
        if self.ge_enabled:
            for branch_name, params in self._branch_params.items():
                unscaled_grads[branch_name] = [
                    p.grad.data.clone() if p.grad is not None else None
                    for p in params
                ]

        # Step 4: Scale branch parameter gradients by k_i (OGM)
        for branch_name, k_val in k.items():
            if k_val < 1.0 and branch_name in self._branch_params:
                for p in self._branch_params[branch_name]:
                    if p.grad is not None:
                        p.grad.data.mul_(k_val)

        # Step 5: Apply GE noise injection (if enabled)
        if self.ge_enabled:
            self._apply_ge_noise(pl_module, k, unscaled_grads)

        # Step 6: Log k_i values
        if self.log_modulation and trainer.global_step % self.log_every_n_steps == 0:
            if trainer.is_global_zero:
                for branch_name, k_val in k.items():
                    pl_module.log(
                        f"gradients/ogm_k/{branch_name}",
                        k_val,
                        on_step=True,
                        on_epoch=False,
                        rank_zero_only=True,
                    )

    def _apply_ge_noise(
        self,
        pl_module: pl.LightningModule,
        k: dict[str, float],
        unscaled_grads: dict[str, list[torch.Tensor | None]],
    ) -> None:
        """
        GE noise injection (paper Equations 12-17).

        1. Save ALL current gradients (OGM-scaled for branches, original for others)
        2. Identify branch parameter IDs
        3. Zero all gradients
        4. Prevent DDP allreduce with no_sync() context
        5. Re-evaluate loss with different dropout (torch.random.fork_rng)
        6. Set _is_ge_reevaluation = True on pl_module to suppress logging
        7. loss.backward()
        8. For branch params: effective = k*g1 + (g2 - g1) / sqrt(2)
        9. For non-branch params: restore saved gradients
        """
        batch = getattr(pl_module, '_current_batch', None)
        if batch is None:
            return

        model = pl_module.model

        # Step 1: Save ALL current gradients
        saved_grads = {}
        branch_param_ids = set()
        for branch_name, params in self._branch_params.items():
            for p in params:
                branch_param_ids.add(id(p))

        for p in model.parameters():
            if p.grad is not None:
                saved_grads[id(p)] = p.grad.data.clone()

        # Step 3: Zero all gradients
        model.zero_grad(set_to_none=False)
        if pl_module.guide is not None:
            pl_module.guide.zero_grad(set_to_none=False)

        # Step 4-7: Re-evaluate with different dropout mask
        # Use no_sync to prevent DDP allreduce on this auxiliary backward
        strategy = pl_module.trainer.strategy
        if hasattr(strategy, 'model') and hasattr(strategy.model, 'no_sync'):
            no_sync_ctx = strategy.model.no_sync()
        else:
            no_sync_ctx = nullcontext()

        # Get CUDA devices for fork_rng
        cuda_devices = []
        if torch.cuda.is_available():
            cuda_devices = [pl_module.device.index] if pl_module.device.type == 'cuda' else []

        with no_sync_ctx:
            # Fork RNG to get different dropout mask
            with torch.random.fork_rng(devices=cuda_devices, enabled=True):
                # Advance RNG state so dropout produces a different mask
                torch.manual_seed(torch.randint(0, 2**32, (1,)).item())
                if cuda_devices:
                    torch.cuda.manual_seed(torch.randint(0, 2**32, (1,)).item())

                # Suppress logging during re-evaluation
                pl_module._is_ge_reevaluation = True
                try:
                    loss2 = pl_module._svi_forward(batch)
                    loss2.backward()
                finally:
                    pl_module._is_ge_reevaluation = False

        # Step 8: For branch params: effective = k*g1 + (g2 - g1) / sqrt(2)
        sqrt2 = math.sqrt(2.0)
        for branch_name, params in self._branch_params.items():
            k_val = k[branch_name]
            for i, p in enumerate(params):
                g1 = unscaled_grads[branch_name][i]
                if g1 is None:
                    continue
                g2 = p.grad.data if p.grad is not None else torch.zeros_like(g1)
                noise = (g2 - g1) / sqrt2
                # effective = k * g1 + noise
                p.grad.data = k_val * g1 + noise

        # Step 9: For non-branch params, restore saved gradients
        for p in model.parameters():
            if id(p) not in branch_param_ids and id(p) in saved_grads:
                if p.grad is not None:
                    p.grad.data.copy_(saved_grads[id(p)])
                else:
                    p.grad = saved_grads[id(p)]

        # Also restore guide parameter gradients (not branch params)
        if pl_module.guide is not None:
            for p in pl_module.guide.parameters():
                if id(p) in saved_grads:
                    if p.grad is not None:
                        p.grad.data.copy_(saved_grads[id(p)])
                    else:
                        p.grad = saved_grads[id(p)]

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        """Clear cached batch reference to free memory."""
        pl_module._current_batch = None
