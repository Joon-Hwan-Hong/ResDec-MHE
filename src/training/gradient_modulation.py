"""
Adapted OGM-GE gradient modulation for multi-branch regression.

Adapts Peng et al., "Balanced Multimodal Learning via On-the-fly Gradient
Modulation," CVPR 2022 (Oral). Original uses per-modality softmax prediction
scores for the discrepancy ratio (classification-specific). Our adaptation
uses gradient norms as the discrepancy ratio since the model is regression
with a Bayesian head — no class labels, no per-branch softmax predictions.

Core principle: monitor per-branch contribution discrepancy, suppress
dominant branches, leave lagging branches unchanged.

OGM component (paper Equation 10):
    k_i = 1 - tanh(alpha * rho_i) if rho_i > 1 (dominant -> suppressed)
    k_i = 1                  otherwise    (lagging -> unchanged)
    where rho_i = g_i / mean(g) is the gradient-norm discrepancy ratio.

GE component (paper Equations 12-17):
    Adds noise h ~ N(0, Sigma^sgd) to recover the SGD noise that OGM's gradient
    scaling reduces. Approximated by re-evaluating the loss on the same batch
    with a different dropout mask and computing noise = (g_2 - g_1) / sqrt(2).

Reference: https://github.com/GeWu-Lab/OGM-GE_CVPR2022
Fallback: GradNorm (Chen et al., ICML 2018) — see design doc S1 Out of Scope.
"""

import logging
import math
from contextlib import nullcontext

import torch
import lightning.pytorch as pl

from src.training.callbacks import BRANCH_NAMES
from src.training._branch_utils import (
    BRANCH_PREFIXES,
    build_branch_param_cache,
    compute_branch_norms,
)

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
        """Build parameter-to-branch mapping once, reused every call.

        Delegates to the canonical helper in src.training._branch_utils so
        both this callback and GradientNormLogger pick up branch-prefix
        changes (e.g., a new HGT submodule alias) in lockstep.
        """
        self._branch_params = build_branch_param_cache(
            model, BRANCH_NAMES, branch_prefixes=BRANCH_PREFIXES,
        )

    def _compute_branch_norms(self, model: torch.nn.Module) -> dict[str, float]:
        """Compute L2 gradient norm for each encoder branch."""
        if self._branch_params is None:
            self._build_param_cache(model)
        return compute_branch_norms(self._branch_params)

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

        # Also save guide gradients (they must be restored after GE re-evaluation)
        if pl_module.guide is not None:
            for p in pl_module.guide.parameters():
                if p.grad is not None:
                    saved_grads[id(p)] = p.grad.data.clone()

        if not saved_grads:
            # Defensive: every parameter had grad=None (e.g., on rank 0 right
            # after a NaN-skip). The downstream noise computation would
            # silently no-op on every branch param, so log a debug-level note
            # for visibility without polluting normal training logs.
            logger.debug(
                "GE noise: saved_grads is empty (all parameters have grad=None). "
                "Skipping GE re-evaluation."
            )
            return

        # Step 3: Zero all gradients
        # set_to_none=False so p.grad exists for the noise computation below
        # (the noise formula needs g2 = p.grad.data from the re-evaluation pass)
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

        # Get CUDA devices for fork_rng. Skip when device.index is None
        # (pre-DDP CPU-only paths), which torch.random.fork_rng rejects
        # with a ValueError.
        cuda_devices: list[int] = []
        if torch.cuda.is_available() and pl_module.device.type == 'cuda':
            dev_idx = pl_module.device.index
            if dev_idx is not None:
                cuda_devices = [dev_idx]

        with no_sync_ctx:
            # Fork RNG to get different dropout mask. Derive the seed
            # deterministically from (global_seed, global_step, rank) so
            # GE noise is reproducible across runs while still being
            # different from the dropout mask used in the primary
            # forward pass.
            with torch.random.fork_rng(devices=cuda_devices, enabled=True):
                base_seed = int(
                    pl_module.config.get("experiment", {}).get("seed", 42)
                )
                global_step = int(getattr(pl_module.trainer, "global_step", 0))
                rank = int(getattr(pl_module.trainer, "global_rank", 0))
                # Salt the seed so the GE re-eval RNG is decoupled from the
                # main run's RNG schedule but still deterministic.
                ge_seed = (
                    base_seed
                    + global_step * 1_000_003
                    + rank * 1_000_000_007
                ) % (2**32 - 1)
                torch.manual_seed(ge_seed)
                if cuda_devices:
                    torch.cuda.manual_seed(ge_seed)

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
            n_noised = 0
            for i, p in enumerate(params):
                g1 = unscaled_grads[branch_name][i]
                if g1 is None:
                    continue
                g2 = p.grad.data if p.grad is not None else torch.zeros_like(g1)
                noise = (g2 - g1) / sqrt2
                # effective = k * g1 + noise
                p.grad.data = k_val * g1 + noise
                n_noised += 1
            if n_noised == 0:
                # Branch had no grad-having parameters this step (rare; e.g.,
                # converged at boundary or all-NaN re-eval). Surface for
                # diagnostic visibility.
                logger.debug(
                    "GE noise: branch %s had 0 grad-having parameters this step.",
                    branch_name,
                )

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
