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
        """Set gene gate temperature at the start of each epoch.

        On checkpoint resume, the temperature buffer may contain a stale
        value. This callback is authoritative — it recomputes from
        trainer.current_epoch, which Lightning restores correctly.
        """
        epoch = trainer.current_epoch
        tau = self.get_temperature(epoch)
        gate = getattr(
            getattr(getattr(pl_module, "model", None), "pseudobulk_encoder", None),
            "gene_gate", None,
        )
        if gate is None:
            raise AttributeError(
                "TemperatureAnnealing requires model.pseudobulk_encoder.gene_gate "
                "but the attribute path does not exist on the current model."
            )
        gate.temperature = tau
        pl_module.log("gene_gate_temperature", tau, rank_zero_only=True)

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
        log_every_n_steps: int = 50,
    ):
        super().__init__()
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.log_every_n_steps = log_every_n_steps
        # Cached parameter-to-branch mapping (built on first call)
        self._branch_params: dict[str, list[torch.nn.Parameter]] | None = None

    def _build_param_cache(self, model: torch.nn.Module) -> None:
        """Build parameter-to-branch mapping once, reused every call."""
        self._branch_params = {name: [] for name in BRANCH_NAMES}
        for param_name, param in model.named_parameters():
            for branch_name in BRANCH_NAMES:
                if branch_name in param_name:
                    self._branch_params[branch_name].append(param)
                    break

    def compute_branch_norms(self, model: torch.nn.Module) -> dict[str, float]:
        """
        Compute L2 gradient norm for each encoder branch.

        Args:
            model: The CognitiveResilienceModel

        Returns:
            Dict mapping branch name to L2 gradient norm
        """
        if self._branch_params is None:
            self._build_param_cache(model)

        branch_norms = {}
        for branch_name, params in self._branch_params.items():
            grad_params = [p for p in params if p.grad is not None]
            if grad_params:
                # Under bf16-mixed and 32-true, p.grad contains unscaled gradients
                # (no GradScaler). Under 16-mixed (float16), Lightning uses GradScaler
                # and p.grad here contains SCALED gradients — logged norms would
                # include the scale factor.
                # Stack per-param norms and reduce in 2 ops instead of N.
                grad_norms = torch.stack(
                    [p.grad.data.norm(2) for p in grad_params]
                )
                branch_norms[branch_name] = grad_norms.norm(2).item()
            else:
                branch_norms[branch_name] = 0.0
        return branch_norms

    @staticmethod
    def compute_norm_ratio(norms: dict[str, float]) -> float:
        """
        Compute max/min ratio of branch norms.

        All-zero norms (e.g., frozen branches or pre-first-step) return ratio 0.0
        because max(values) == 0 and EPSILON_DIVISION prevents division by zero.

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
        """Log branch gradient norms after DDP sync, before optimizer step (every N steps).

        Under DDP, gradients are already synchronized (allreduce) before this
        hook fires, so all ranks have identical gradient values. Computation
        and logging runs on rank 0 only — avoids .item() CUDA sync on non-zero ranks.
        """
        if trainer.global_step % self.log_every_n_steps != 0:
            return
        if not trainer.is_global_zero:
            return

        norms = self.compute_branch_norms(pl_module.model)

        for branch_name, norm in norms.items():
            pl_module.log(
                f"gradients/branch_norm/{branch_name}",
                norm,
                on_step=True,
                on_epoch=False,
                rank_zero_only=True,
            )

        ratio = self.compute_norm_ratio(norms)
        pl_module.log(
            "gradients/branch_norm_ratio",
            ratio,
            on_step=True,
            on_epoch=False,
            rank_zero_only=True,
        )

        severity = self.get_severity(ratio)
        norms_str = {k: f"{v:.4f}" for k, v in norms.items()}
        if severity == "red" and trainer.is_global_zero:
            logger.error(
                "CRITICAL gradient norm imbalance (ratio=%.1f, intervention recommended): %s",
                ratio, norms_str,
            )
        elif severity == "yellow" and trainer.is_global_zero:
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


class KLAnnealingCallback(pl.Callback):
    """
    Anneal KL divergence weight during Bayesian SVI training.

    Ramps kl_weight from alpha_min to 1.0 over warmup_epochs using a linear
    schedule. This allows the model to learn the data distribution before
    prior regularization reaches full strength.

    Schedule:
    - Epochs [0, warmup_epochs): linear ramp from alpha_min to 1.0
    - Epochs >= warmup_epochs: kl_weight = 1.0

    Args:
        alpha_min: Floor KL weight (>0 to maintain minimal regularization)
        warmup_epochs: Number of epochs to ramp from alpha_min to 1.0
        schedule: Annealing schedule type (currently only "linear")
    """

    def __init__(
        self,
        alpha_min: float = 0.01,
        warmup_epochs: int = 5,
        schedule: str = "linear",
    ):
        super().__init__()
        if alpha_min < 0:
            raise ValueError(f"alpha_min must be >= 0, got {alpha_min}")
        self.alpha_min = alpha_min
        self.warmup_epochs = warmup_epochs
        self.schedule = schedule

    def get_kl_weight(self, epoch: int) -> float:
        """Compute KL weight for a given epoch.

        Args:
            epoch: Current training epoch (0-indexed)

        Returns:
            KL weight in [alpha_min, 1.0]
        """
        if epoch >= self.warmup_epochs:
            return 1.0
        if self.warmup_epochs <= 0:
            return 1.0
        progress = epoch / self.warmup_epochs
        # Linear ramp: alpha_min at epoch 0, 1.0 at epoch warmup_epochs
        return self.alpha_min + progress * (1.0 - self.alpha_min)

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule,
    ) -> None:
        """Set KL weight on the ELBO at the start of each epoch."""
        elbo = getattr(pl_module, "elbo", None)
        if elbo is None or not hasattr(elbo, "kl_weight"):
            return
        kl_weight = self.get_kl_weight(trainer.current_epoch)
        elbo.kl_weight = kl_weight
        pl_module.log("kl_weight", kl_weight, rank_zero_only=True)

    def __repr__(self) -> str:
        return (
            f"KLAnnealingCallback(alpha_min={self.alpha_min}, "
            f"warmup_epochs={self.warmup_epochs}, "
            f"schedule='{self.schedule}')"
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
        """Add custom metadata to checkpoint dict.

        Called only on rank 0 by Lightning's ModelCheckpoint. RNG states are
        rank-0 only; other ranks re-initialize from global seed on resume
        (see _make_worker_init_fn for rank-aware data worker seeding).
        """
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
        """Restore RNG states and Pyro state from checkpoint.

        Checkpoint resume limitations (not bit-reproducible):
        - RNG states are restored here but may drift before training resumes
          due to Lightning's internal operations (device moves, sampler creation).
        - DataLoader position within the interrupted epoch is not saved —
          the epoch restarts from the beginning on resume.
        - CellSampler per-worker RNG states (with persistent_workers=True) are
          not accessible from the main process and cannot be checkpointed. On
          resume, workers re-initialize from the fixed seed rather than continuing
          from their advanced state. This only affects CognitiveResilienceDataset
          (on-the-fly mode); PrecomputedDataset has no CellSampler.

        Result: resumed training converges to the same quality but the exact
        gradient trajectory differs from an uninterrupted run.
        """
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
                pyro.get_param_store()[k] = v.to(device)

            # Flag that we're resuming — on_train_start needs to re-sync
            # param store with guide's nn.Parameters for tensor identity
            pl_module._pyro_resuming_from_checkpoint = True

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Migrate Pyro param store to training device and sync tensor identity.

        After checkpoint resume, the Pyro param store holds tensor objects from
        on_load_checkpoint(), but the guide's nn.Parameters (which the optimizer
        tracks) are different objects restored by Lightning's load_state_dict().
        We must re-sync the param store to point to the guide's actual tensors
        so that optimizer updates are visible to Pyro's trace mechanism.
        """
        if hasattr(pl_module, 'guide') and pl_module.guide is not None:
            import pyro
            target_device = pl_module.device
            store = pyro.get_param_store()

            if getattr(pl_module, '_pyro_resuming_from_checkpoint', False):
                # Re-sync param store from guide's nn.Parameters.
                # After Lightning's load_state_dict(), the guide's parameters have
                # the correct checkpoint values AND are the same objects the optimizer
                # tracks. Replace param store entries with these authoritative tensors.
                #
                # Key naming: Pyro uses fullnames (e.g., "AutoDiagonalNormal.loc")
                # not nn.Module names ("loc"). We must use _pyro_get_fullname() to
                # build correct keys. We also directly set store._params to avoid
                # going through constraint transforms which would create new tensors
                # and break the identity link with the optimizer.
                # Tested against Pyro 1.8.x–1.9.x — store._params and store._param_to_name
                # are private but no public API preserves tensor identity with optimizer.
                _TESTED_PYRO_VERSIONS = ("1.8", "1.9")
                pyro_version = ".".join(pyro.__version__.split(".")[:2])
                if pyro_version not in _TESTED_PYRO_VERSIONS:
                    logger.warning(
                        "Pyro param store re-sync uses private APIs tested on Pyro %s. "
                        "Current version: %s. Verify store._params/_param_to_name.",
                        ", ".join(_TESTED_PYRO_VERSIONS), pyro.__version__,
                    )
                guide = pl_module.guide

                # Clean slate: remove all old entries before re-registering
                store._params.clear()
                store._param_to_name.clear()

                # Re-register regular nn.Parameters (e.g., loc)
                # Note: .to(target_device) is a no-op when param is already on
                # target_device (Lightning moves model before on_train_start),
                # so tensor identity with the optimizer is preserved.
                for name in list(guide._parameters.keys()):
                    param = guide._parameters[name]
                    if param is None or name.endswith('_unconstrained'):
                        continue
                    fullname = guide._pyro_get_fullname(name)
                    live_param = param.to(target_device)
                    store._params[fullname] = live_param
                    store._param_to_name[live_param] = fullname

                # Re-register PyroParams (e.g., scale → stored as scale_unconstrained)
                if hasattr(guide, '_pyro_params'):
                    for name in guide._pyro_params:
                        unconstrained = guide._parameters.get(name + '_unconstrained')
                        if unconstrained is not None:
                            fullname = guide._pyro_get_fullname(name)
                            live_param = unconstrained.to(target_device)
                            store._params[fullname] = live_param
                            store._param_to_name[live_param] = fullname

                pl_module._pyro_resuming_from_checkpoint = False

                # Sanity check: param count should be non-zero if guide has params
                expected_n = sum(1 for p in guide.parameters() if p is not None)
                actual_n = len(store._params)
                if actual_n == 0 and expected_n > 0:
                    raise RuntimeError(
                        f"Pyro param store re-sync produced 0 params but guide has "
                        f"{expected_n}. Pyro internal API may have changed "
                        f"(version: {pyro.__version__})."
                    )
                logger.info(
                    "Pyro param store re-synced from guide after checkpoint resume: "
                    "%d params", len(store._params)
                )
            else:
                # Normal startup — just migrate to device
                for k in list(store.keys()):
                    v = store[k]
                    if v.device != target_device:
                        store[k] = v.to(target_device)
