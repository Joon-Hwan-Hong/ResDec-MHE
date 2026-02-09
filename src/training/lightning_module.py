"""
PyTorch Lightning module wrapping CognitiveResilienceModel.

Handles:
- Loss branching: β-NLL for Bayesian head, MSE for deterministic head
- Optimizer and scheduler configuration
- Metric logging (training and validation)
- Optional gene gate L1 regularization
"""

import logging
from typing import Any

import torch
import lightning.pytorch as pl
from omegaconf import DictConfig

import pyro
import pyro.poutine
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.optim import ClippedAdam as PyroClippedAdam

from src.models.full_model import CognitiveResilienceModel, build_model_from_config
from src.training.losses import BetaNLLLoss, mse_loss
from src.training.metrics import ResilienceMetrics

logger = logging.getLogger(__name__)


class CognitiveResilienceLightningModule(pl.LightningModule):
    """
    Lightning wrapper for CognitiveResilienceModel.

    Loss branching:
    - Bayesian head (outputs mean + std): uses β-NLL loss
    - Deterministic head (outputs mean only): uses MSE loss
      If config specifies beta_nll with deterministic head, a warning is logged
      and MSE is used automatically.

    Args:
        config: OmegaConf config with 'model' and 'training' sections

    Checkpoint Loading:
        Config is NOT saved via save_hyperparameters() to avoid duplication with
        ResilienceModelCheckpoint. When loading from checkpoint, pass config explicitly::

            checkpoint = torch.load("path/to/checkpoint.ckpt")
            config = OmegaConf.create(checkpoint["model_config"])
            module = CognitiveResilienceLightningModule.load_from_checkpoint(
                "path/to/checkpoint.ckpt",
                config=config,
            )
    """

    def __init__(self, config: DictConfig):
        super().__init__()
        # Ignore config in save_hyperparameters — ResilienceModelCheckpoint handles
        # config persistence. On load_from_checkpoint(), pass config explicitly.
        self.save_hyperparameters(ignore=["config"])
        self.config = config

        # Build model from config (shared factory ensures training/inference parity)
        model_cfg = config.model
        use_bayesian = model_cfg.head.type == "bayesian"
        self.model = build_model_from_config(model_cfg)

        # SVI setup for Bayesian head
        self._use_bayesian_svi = use_bayesian
        self.guide = None
        self.svi = None
        self.pyro_optim = None

        if self._use_bayesian_svi:
            pyro.clear_param_store()
            self.guide = AutoDiagonalNormal(self.model)
            # Disable automatic optimization — SVI manages its own backward + step
            self.automatic_optimization = False

        # Loss function setup with branching
        train_cfg = config.training
        self._use_mse_loss = not use_bayesian

        if use_bayesian:
            self.loss_fn = BetaNLLLoss(beta=train_cfg.loss.beta)
        else:
            if train_cfg.loss.type == "beta_nll":
                logger.warning(
                    "Config specifies beta_nll loss but head is deterministic. "
                    "Falling back to MSE loss."
                )
            self.loss_fn = None  # Will use mse_loss function

        # Gene gate L1 regularization
        self._gene_gate_l1_lambda = train_cfg.regularization.get("gene_gate_l1", 0.0)

        # NaN handling policy from config
        error_cfg = config.get("error_handling", {}).get("training", {})
        self._nan_loss_policy = error_cfg.get("nan_loss", "fail")
        self._nan_batch_policy = error_cfg.get("nan_batch", "skip")

        # Metrics
        self.metrics = ResilienceMetrics()

    def _compute_loss(self, output: dict, cognition: torch.Tensor) -> torch.Tensor:
        """Compute loss with branching based on head type."""
        if self._use_mse_loss:
            loss = mse_loss(output["mean"], cognition)
        else:
            loss = self.loss_fn(output["mean"], output["std"], cognition)

        # Optional gene gate L1 regularization
        if self._gene_gate_l1_lambda > 0:
            gate_logits = self.model.pseudobulk_encoder.gene_gate.gate_logits
            l1_penalty = self._gene_gate_l1_lambda * gate_logits.abs().mean()
            loss = loss + l1_penalty

        return loss

    def _forward_batch(self, batch: dict) -> dict:
        """Run forward pass extracting inputs from batch dict."""
        return self.model(
            region_pseudobulk=batch.get("region_pseudobulk"),
            region_mask=batch.get("region_mask"),
            pseudobulk=batch.get("pseudobulk"),
            edge_index_dict_list=batch.get("edge_index_dict_list"),
            edge_attr_dict_list=batch.get("edge_attr_dict_list"),
            cells=batch.get("cells"),
            cell_mask=batch.get("cell_mask"),
            cell_type_mask=batch.get("cell_type_mask"),
            pathology=batch.get("pathology"),
            cognition=batch.get("cognition"),
        )

    def _svi_forward(self, batch: dict[str, Any]) -> torch.Tensor:
        """
        Run SVI step for Bayesian training.

        SVI.step() internally runs model forward, guide forward, computes ELBO,
        computes gradients, and steps the Pyro optimizer. Returns scalar loss.
        """
        loss = self.svi.step(
            region_pseudobulk=batch.get("region_pseudobulk"),
            region_mask=batch.get("region_mask"),
            pseudobulk=batch.get("pseudobulk"),
            edge_index_dict_list=batch.get("edge_index_dict_list"),
            edge_attr_dict_list=batch.get("edge_attr_dict_list"),
            cells=batch.get("cells"),
            cell_mask=batch.get("cell_mask"),
            cell_type_mask=batch.get("cell_type_mask"),
            pathology=batch.get("pathology"),
            cognition=batch.get("cognition"),
        )
        return torch.tensor(loss, device=self.device)

    def _forward_batch_posterior(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
        Forward pass using posterior median (MAP estimate) from guide.

        Used for validation/test to get deterministic predictions from the
        learned posterior, avoiding sampling noise.

        Falls back to standard forward if guide hasn't been initialized yet
        (e.g., validation before first training step).
        """
        if self.guide.prototype_trace is None:
            # Guide hasn't been prototyped yet (no SVI step has run).
            # Fall back to standard forward pass with prior samples.
            return self._forward_batch(batch)
        median = self.guide.median()
        conditioned = pyro.poutine.condition(self.model, data=median)
        return conditioned(
            region_pseudobulk=batch.get("region_pseudobulk"),
            region_mask=batch.get("region_mask"),
            pseudobulk=batch.get("pseudobulk"),
            edge_index_dict_list=batch.get("edge_index_dict_list"),
            edge_attr_dict_list=batch.get("edge_attr_dict_list"),
            cells=batch.get("cells"),
            cell_mask=batch.get("cell_mask"),
            cell_type_mask=batch.get("cell_type_mask"),
            pathology=batch.get("pathology"),
            cognition=batch.get("cognition"),
        )

    def _log_svi_gradient_norms(self) -> None:
        """Log branch gradient norms after SVI step (manual substitute for on_after_backward)."""
        branch_names = ["pseudobulk_encoder", "hgt_encoder", "cell_transformer"]
        norms = {}
        for branch_name in branch_names:
            branch_params = [
                p for n, p in self.model.named_parameters()
                if branch_name in n and p.grad is not None
            ]
            if branch_params:
                total_norm = torch.sqrt(
                    sum(p.grad.data.norm(2) ** 2 for p in branch_params)
                )
                norms[branch_name] = total_norm.item()
            else:
                norms[branch_name] = 0.0

        for branch_name, norm in norms.items():
            self.log(
                f"gradients/branch_norm/{branch_name}",
                norm,
                on_step=True,
                on_epoch=False,
            )

        # Log ratio
        values = list(norms.values())
        if values and min(values) > 0:
            ratio = max(values) / min(values)
            self.log("gradients/branch_norm_ratio", ratio, on_step=True, on_epoch=False)

    def _check_batch_nan(self, batch: dict) -> bool:
        """Check if batch contains NaN values. Returns True if NaN detected."""
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and torch.isnan(value).any():
                return True
        return False

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor | None:
        """Training step with SVI branching for Bayesian head."""
        if self._nan_batch_policy == "skip" and self._check_batch_nan(batch):
            logger.warning("NaN detected in batch %d — skipping", batch_idx)
            return None

        if self._use_bayesian_svi:
            # SVI handles forward, loss, backward, and optimizer step internally
            loss = self._svi_forward(batch)
            # Log gradient norms manually — SVI bypasses Lightning's backward,
            # so on_after_backward callbacks never fire.
            try:
                should_log = self.trainer.global_step % 10 == 0
            except RuntimeError:
                should_log = False
            if should_log:
                self._log_svi_gradient_norms()
        else:
            output = self._forward_batch(batch)
            loss = self._compute_loss(output, batch["cognition"])

        # Check for NaN loss
        if torch.isnan(loss):
            if self._nan_loss_policy == "fail":
                raise ValueError(f"NaN loss detected at batch {batch_idx}")
            else:
                logger.warning("NaN loss at batch %d — skipping", batch_idx)
                return None

        bs = batch["cognition"].shape[0]
        self.log("train_loss", loss, prog_bar=True, sync_dist=True, batch_size=bs)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        """Validation step using posterior median for Bayesian head."""
        if self._use_bayesian_svi:
            output = self._forward_batch_posterior(batch)
        else:
            output = self._forward_batch(batch)
        loss = self._compute_loss(output, batch["cognition"])
        bs = batch["cognition"].shape[0]
        self.log("val_loss", loss, prog_bar=True, sync_dist=True, batch_size=bs)

        std = output.get("std")
        metrics = self.metrics.compute(output["mean"], std, batch["cognition"])
        for name, value in metrics.items():
            if not (isinstance(value, float) and value != value):
                self.log(f"val_{name}", value, sync_dist=True, batch_size=bs)

    def test_step(self, batch: dict, batch_idx: int) -> None:
        """Test step using posterior median for Bayesian head."""
        if self._use_bayesian_svi:
            output = self._forward_batch_posterior(batch)
        else:
            output = self._forward_batch(batch)
        loss = self._compute_loss(output, batch["cognition"])
        bs = batch["cognition"].shape[0]
        self.log("test_loss", loss, prog_bar=True, sync_dist=True, batch_size=bs)

        std = output.get("std")
        metrics = self.metrics.compute(output["mean"], std, batch["cognition"])
        for name, value in metrics.items():
            if not (isinstance(value, float) and value != value):
                self.log(f"test_{name}", value, sync_dist=True, batch_size=bs)

    def predict_step(self, batch: dict, batch_idx: int) -> dict[str, Any]:
        """Predict step using posterior median for Bayesian head.

        Returns:
            Dict with keys:
            - mean: [B, 1] predicted values
            - std: [B, 1] predicted uncertainty (if Bayesian head)
            - attention_weights: [B, n_heads, n_cell_types] pathology attention (if present)
        """
        if self._use_bayesian_svi:
            output = self._forward_batch_posterior(batch)
        else:
            output = self._forward_batch(batch)
        result = {"mean": output["mean"]}
        if "std" in output and output["std"] is not None:
            result["std"] = output["std"]
        if "attention_weights" in output and output["attention_weights"] is not None:
            result["attention_weights"] = output["attention_weights"]
        return result

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and learning rate scheduler."""
        train_cfg = self.config.training
        opt_cfg = train_cfg.optimizer

        # Linear LR scaling for multi-GPU (Goyal et al. 2017)
        # effective_lr = base_lr * n_gpus when using DDP
        base_lr = opt_cfg.lr
        lr_scaling_enabled = train_cfg.get("lr_scaling", True)

        if lr_scaling_enabled:
            try:
                world_size = self.trainer.world_size
            except RuntimeError:
                world_size = 1
            if world_size > 1:
                scaled_lr = base_lr * world_size
                logger.info(
                    f"LR scaling: {base_lr} × {world_size} GPUs = {scaled_lr}"
                )
                base_lr = scaled_lr

        effective_lr = base_lr

        if self._use_bayesian_svi:
            # Pyro optimizer handles BOTH model and guide parameters.
            #
            # LR decay strategy: ClippedAdam's built-in `lrd` (per-step multiplicative
            # decay). With lrd=0.9999 and ~25k steps, final LR ≈ 8% of initial — smooth
            # exponential decay comparable to cosine annealing.
            #
            # Why not fixed LR (Option 1): Risks ELBO oscillation late in training,
            # prevents tight posterior convergence, produces noisier epistemic uncertainty.
            #
            # Why not PyroLRScheduler + cosine (Option 3): Three layers of wrapping
            # (PyTorch optimizer → scheduler → Pyro wrapper), requires manual
            # scheduler.step() with automatic_optimization=False, and is over-engineered
            # for AutoDiagonalNormal mean-field VI. The lrd parameter is the idiomatic
            # Pyro approach — built into ClippedAdam specifically for SVI decay.
            self.pyro_optim = PyroClippedAdam({
                "lr": effective_lr,
                "weight_decay": opt_cfg.weight_decay,
                "betas": tuple(opt_cfg.get("betas", [0.9, 0.999])),
                "clip_norm": train_cfg.get("gradient_clip_val", 1.0),
                "lrd": opt_cfg.get("lrd", 1.0),  # 1.0 = no decay (backward compat)
            })
            self.svi = SVI(
                model=self.model,
                guide=self.guide,
                optim=self.pyro_optim,
                loss=Trace_ELBO(),
            )
            # SVI manages optimization internally; LR decay handled by lrd above.
            # Lightning requires a return value from configure_optimizers
            # when automatic_optimization=False, but won't step it.
            dummy_opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0)
            return dummy_opt

        # Standard optimizer for deterministic head
        if opt_cfg.type == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=effective_lr,
                weight_decay=opt_cfg.weight_decay,
                betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
            )
        elif opt_cfg.type == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=effective_lr,
                weight_decay=opt_cfg.get("weight_decay", 0),
            )
        else:
            raise ValueError(f"Unknown optimizer type: {opt_cfg.type}")

        # Scheduler: cosine annealing with optional linear warmup
        sched_cfg = train_cfg.scheduler
        warmup_epochs = sched_cfg.get("warmup_epochs", 0)
        eta_min = sched_cfg.get("eta_min", 1e-6)

        if sched_cfg.type == "cosine":
            t_max = train_cfg.max_epochs - warmup_epochs
            if t_max <= 0:
                raise ValueError(
                    f"warmup_epochs ({warmup_epochs}) must be less than "
                    f"max_epochs ({train_cfg.max_epochs}) for cosine scheduler"
                )
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=t_max,
                eta_min=eta_min,
            )

            if warmup_epochs > 0:
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=0.01,
                    end_factor=1.0,
                    total_iters=warmup_epochs,
                )
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[warmup_epochs],
                )
            else:
                scheduler = cosine_scheduler
        else:
            raise ValueError(f"Unknown scheduler type: {sched_cfg.type}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
