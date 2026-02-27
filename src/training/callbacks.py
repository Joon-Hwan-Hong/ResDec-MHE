"""
Training callbacks for cognitive resilience model.

MinEpochEarlyStopping: EarlyStopping wrapper that enforces a minimum number of
  epochs before patience counting begins. Protects LR warmup and temperature
  annealing phases from premature termination.

TemperatureAnnealing: Anneals gene attention gate temperature during training.
  - Warmup phase: keeps tau_max for stability
  - Anneal phase: exponential/linear/cosine decay from tau_max to tau_min
  - Post-anneal: clamps at tau_min

GradientNormLogger: Monitors per-branch gradient norms to detect training
  imbalances across the three encoder branches (pseudobulk, HGT, cell transformer).
"""

import logging
import math
from datetime import datetime, timezone

import torch
import lightning.pytorch as pl
from omegaconf import OmegaConf

from lightning.pytorch.callbacks import EarlyStopping

from src.data.constants import EPSILON_DIVISION
from src.utils.hashing import hash_config
from src.utils.reproducibility import get_rng_states, set_rng_states

logger = logging.getLogger(__name__)

BRANCH_NAMES = ("pseudobulk_encoder", "hgt_encoder", "cell_transformer")


class MinEpochEarlyStopping(EarlyStopping):
    """
    EarlyStopping with enforced minimum epochs before patience counting.

    Standard Lightning EarlyStopping starts monitoring from epoch 0. This wrapper
    skips all early stopping logic until min_epochs is reached, protecting warmup
    phases (LR warmup, temperature annealing) from premature termination.

    Design doc requirement:
        "Minimum 20 epochs before early stopping activates (allow warmup + initial exploration)"

    Args:
        min_epochs: Minimum epochs before early stopping can trigger (default: 20)
        **kwargs: All standard EarlyStopping arguments (monitor, patience, min_delta, mode, etc.)

    Example:
        >>> callback = MinEpochEarlyStopping(
        ...     min_epochs=20,
        ...     monitor="val_loss",
        ...     patience=15,
        ...     min_delta=0.0001,
        ...     mode="min",
        ... )
    """

    def __init__(self, min_epochs: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.min_epochs = min_epochs

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Skip early stopping check if below min_epochs threshold."""
        if trainer.current_epoch < self.min_epochs:
            return  # Skip all early stopping logic during warmup
        super().on_validation_end(trainer, pl_module)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Skip early stopping check if below min_epochs threshold."""
        if trainer.current_epoch < self.min_epochs:
            return  # Skip all early stopping logic during warmup
        super().on_train_epoch_end(trainer, pl_module)

    def __repr__(self) -> str:
        return (
            f"MinEpochEarlyStopping(min_epochs={self.min_epochs}, "
            f"monitor='{self.monitor}', patience={self.patience}, "
            f"min_delta={self.min_delta}, mode='{self.mode}')"
        )


class TemperatureAnnealing(pl.Callback):
    """
    Anneal gene attention gate temperature during training.

    Schedule:
    - Epochs [0, warmup_epochs): temperature = tau_max
    - Epochs [warmup_epochs, warmup_epochs + anneal_epochs): decay tau_max -> tau_min
    - Epochs >= warmup_epochs + anneal_epochs: temperature = tau_min

    Args:
        tau_max: Starting temperature (soft attention)
        tau_min: Final temperature (sharp attention)
        warmup_epochs: Number of epochs to hold tau_max
        anneal_epochs: Number of epochs to anneal from tau_max to tau_min
        schedule: Annealing curve type ("exponential", "linear", "cosine")
    """

    VALID_SCHEDULES = ("exponential", "linear", "cosine")

    def __init__(
        self,
        tau_max: float = 2.0,
        tau_min: float = 0.1,
        warmup_epochs: int = 5,
        anneal_epochs: int = 50,
        schedule: str = "exponential",
    ):
        super().__init__()
        if schedule not in self.VALID_SCHEDULES:
            raise ValueError(
                f"Unknown schedule '{schedule}', must be one of {self.VALID_SCHEDULES}"
            )
        self.tau_max = tau_max
        self.tau_min = tau_min
        self.warmup_epochs = warmup_epochs
        self.anneal_epochs = anneal_epochs
        self.schedule = schedule

    def get_temperature(self, epoch: int) -> float:
        """
        Compute temperature for a given epoch.

        Args:
            epoch: Current training epoch (0-indexed)

        Returns:
            Temperature value for this epoch
        """
        # Warmup: hold at tau_max
        if epoch < self.warmup_epochs:
            return self.tau_max

        # Post-anneal: clamp at tau_min
        anneal_epoch = epoch - self.warmup_epochs
        if anneal_epoch >= self.anneal_epochs:
            return self.tau_min

        # Progress through annealing [0, 1]
        progress = anneal_epoch / (self.anneal_epochs - 1) if self.anneal_epochs > 1 else 1.0

        if self.schedule == "exponential":
            # Exponential interpolation in log space
            log_tau_max = math.log(self.tau_max)
            log_tau_min = math.log(self.tau_min)
            return math.exp(log_tau_max + progress * (log_tau_min - log_tau_max))

        elif self.schedule == "linear":
            return self.tau_max + progress * (self.tau_min - self.tau_max)

        elif self.schedule == "cosine":
            # Cosine interpolation: slow start and end, fast middle
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.tau_min + cosine_factor * (self.tau_max - self.tau_min)

        # Should be unreachable due to __init__ validation
        raise ValueError(f"Unknown schedule: {self.schedule}")

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Set gene gate temperature at the start of each epoch."""
        epoch = trainer.current_epoch
        tau = self.get_temperature(epoch)
        pl_module.model.pseudobulk_encoder.gene_gate.temperature = tau
        pl_module.log("gene_gate_temperature", tau, sync_dist=True)

    def __repr__(self) -> str:
        return (
            f"TemperatureAnnealing(tau_max={self.tau_max}, tau_min={self.tau_min}, "
            f"warmup_epochs={self.warmup_epochs}, anneal_epochs={self.anneal_epochs}, "
            f"schedule='{self.schedule}')"
        )


class GradientNormLogger(pl.Callback):
    """
    Log per-branch gradient L2 norms to detect training imbalances.

    Monitors pseudobulk_encoder, hgt_encoder, and cell_transformer branches.
    Logs individual norms and the max/min ratio.

    Severity levels based on max/min ratio:
    - normal (< 3): No action needed
    - yellow (3-10): Warning, continue monitoring
    - red (>= 10): Critical imbalance, intervention recommended

    Args:
        warning_threshold: Ratio threshold for yellow warning (default: 3.0)
        critical_threshold: Ratio threshold for red/critical alert (default: 10.0)
        log_every_n_steps: Only compute and log every N steps (default: 10)
    """

    def __init__(
        self,
        warning_threshold: float = 3.0,
        critical_threshold: float = 10.0,
        log_every_n_steps: int = 10,
    ):
        super().__init__()
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.log_every_n_steps = log_every_n_steps

    def compute_branch_norms(self, model: torch.nn.Module) -> dict[str, float]:
        """
        Compute L2 gradient norm for each encoder branch.

        Args:
            model: The CognitiveResilienceModel

        Returns:
            Dict mapping branch name to L2 gradient norm
        """
        branch_norms = {}
        for branch_name in BRANCH_NAMES:
            branch_params = [
                p for n, p in model.named_parameters()
                if branch_name in n and p.grad is not None
            ]
            if branch_params:
                total_norm = torch.sqrt(
                    sum(p.grad.data.norm(2) ** 2 for p in branch_params)
                )
                branch_norms[branch_name] = total_norm.item()
            else:
                branch_norms[branch_name] = 0.0
        return branch_norms

    @staticmethod
    def compute_norm_ratio(norms: dict[str, float]) -> float:
        """
        Compute max/min ratio of branch norms.

        Args:
            norms: Dict mapping branch name to gradient norm

        Returns:
            Ratio of max norm to min norm (epsilon-protected)
        """
        values = list(norms.values())
        if not values:
            return 0.0
        return max(values) / (min(values) + EPSILON_DIVISION)

    def get_severity(self, ratio: float) -> str:
        """
        Classify gradient norm ratio into severity level.

        Args:
            ratio: max/min branch gradient norm ratio

        Returns:
            "normal" (< 3), "yellow" (3-10), or "red" (>= 10)
        """
        if ratio >= self.critical_threshold:
            return "red"
        elif ratio >= self.warning_threshold:
            return "yellow"
        return "normal"

    def on_before_optimizer_step(self, trainer: pl.Trainer, pl_module: pl.LightningModule, optimizer) -> None:
        """Log branch gradient norms after DDP sync, before optimizer step (every N steps)."""
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        norms = self.compute_branch_norms(pl_module.model)

        for branch_name, norm in norms.items():
            pl_module.log(
                f"gradients/branch_norm/{branch_name}",
                norm,
                on_step=True,
                on_epoch=False,
            )

        ratio = self.compute_norm_ratio(norms)
        pl_module.log(
            "gradients/branch_norm_ratio",
            ratio,
            on_step=True,
            on_epoch=False,
        )

        severity = self.get_severity(ratio)
        norms_str = {k: f"{v:.4f}" for k, v in norms.items()}
        if severity == "red":
            logger.error(
                "CRITICAL gradient norm imbalance (ratio=%.1f, intervention recommended): %s",
                ratio, norms_str,
            )
        elif severity == "yellow":
            logger.warning(
                "Gradient norm imbalance detected (ratio=%.1f): %s",
                ratio, norms_str,
            )

    def __repr__(self) -> str:
        return (
            f"GradientNormLogger(warning_threshold={self.warning_threshold}, "
            f"critical_threshold={self.critical_threshold}, "
            f"log_every_n_steps={self.log_every_n_steps})"
        )


CHECKPOINT_VERSION = "1.0"


class ResilienceModelCheckpoint(pl.Callback):
    """
    Add custom metadata to Lightning checkpoints for reproducibility.

    Metadata added to every checkpoint:
    - checkpoint_version: Schema version for future compatibility
    - experiment_hash: SHA-256 of model config for experiment tracking
    - timestamp: ISO 8601 UTC timestamp of checkpoint creation
    - rng_states: Python, NumPy, PyTorch (and CUDA) RNG states
    - model_config: Full model configuration dict
    """

    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: dict,
    ) -> None:
        """Add custom metadata to checkpoint dict."""
        config = pl_module.config

        # Convert OmegaConf to plain dict if needed
        if OmegaConf.is_config(config):
            config_dict = OmegaConf.to_container(config, resolve=True)
        else:
            config_dict = dict(config) if not isinstance(config, dict) else config

        # Experiment hash from model config (uses same hashing as ExperimentManager)
        model_config = config_dict.get("model", config_dict)
        experiment_hash = hash_config(model_config)

        # RNG states (via shared utility — single source of truth)
        rng_states = get_rng_states()

        checkpoint["checkpoint_version"] = CHECKPOINT_VERSION
        checkpoint["experiment_hash"] = experiment_hash
        checkpoint["timestamp"] = datetime.now(timezone.utc).isoformat()
        checkpoint["rng_states"] = rng_states
        checkpoint["model_config"] = model_config
        checkpoint["full_config"] = config_dict

        # Save guide state for Bayesian SVI
        if hasattr(pl_module, 'guide') and pl_module.guide is not None:
            checkpoint["guide_state_dict"] = pl_module.guide.state_dict()
            # Also save Pyro param store (contains variational parameters)
            import pyro
            checkpoint["pyro_param_store"] = {
                k: v.detach().cpu()
                for k, v in pyro.get_param_store().items()
            }

    def on_load_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: dict,
    ) -> None:
        """Restore RNG states and Pyro state from checkpoint."""
        # Restore RNG states (optional — legacy checkpoints may not have them)
        rng_states = checkpoint.get("rng_states")
        if rng_states is None:
            logger.warning(
                "Checkpoint has no rng_states — RNG state not restored. "
                "Training will not be exactly reproducible from this checkpoint."
            )
        else:
            set_rng_states(rng_states)

        # Restore Pyro param store for Bayesian SVI (independent of RNG restore)
        if "pyro_param_store" in checkpoint:
            import pyro
            # Use CPU during checkpoint load — Lightning hasn't moved model to
            # accelerator device yet. Pyro params will migrate with model.
            device = torch.device("cpu")
            pyro.clear_param_store()
            for k, v in checkpoint["pyro_param_store"].items():
                pyro.get_param_store().setdefault(k, v.to(device))

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Migrate Pyro param store to training device if needed."""
        if hasattr(pl_module, 'guide') and pl_module.guide is not None:
            import pyro
            target_device = pl_module.device
            store = pyro.get_param_store()
            for k in list(store.keys()):
                v = store[k]
                if v.device != target_device:
                    store[k] = v.to(target_device)
