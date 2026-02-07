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

    def _check_batch_nan(self, batch: dict) -> bool:
        """Check if batch contains NaN values. Returns True if NaN detected."""
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and torch.isnan(value).any():
                return True
        return False

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Training step: forward pass + loss computation with NaN handling."""
        # Check for NaN in batch inputs
        if self._nan_batch_policy == "skip" and self._check_batch_nan(batch):
            logger.warning("NaN detected in batch %d — skipping", batch_idx)
            return None

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
        """Validation step: forward pass + loss + metrics."""
        output = self._forward_batch(batch)
        loss = self._compute_loss(output, batch["cognition"])
        bs = batch["cognition"].shape[0]
        self.log("val_loss", loss, prog_bar=True, sync_dist=True, batch_size=bs)

        # Compute and log metrics
        std = output.get("std")
        metrics = self.metrics.compute(output["mean"], std, batch["cognition"])
        for name, value in metrics.items():
            if not (isinstance(value, float) and value != value):  # skip NaN
                self.log(f"val_{name}", value, sync_dist=True, batch_size=bs)

    def test_step(self, batch: dict, batch_idx: int) -> None:
        """Test step: forward pass + loss + metrics (same as validation, test_ prefix)."""
        output = self._forward_batch(batch)
        loss = self._compute_loss(output, batch["cognition"])
        bs = batch["cognition"].shape[0]
        self.log("test_loss", loss, prog_bar=True, sync_dist=True, batch_size=bs)

        # Compute and log metrics
        std = output.get("std")
        metrics = self.metrics.compute(output["mean"], std, batch["cognition"])
        for name, value in metrics.items():
            if not (isinstance(value, float) and value != value):  # skip NaN
                self.log(f"test_{name}", value, sync_dist=True, batch_size=bs)

    def predict_step(self, batch: dict, batch_idx: int) -> dict[str, Any]:
        """Predict step: forward pass returning predictions dict.

        Returns:
            Dict with keys:
            - mean: [B, 1] predicted values
            - std: [B, 1] predicted uncertainty (if Bayesian head)
            - attention_weights: [B, n_heads, n_cell_types] pathology attention (if present)
        """
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

        # Optimizer
        if opt_cfg.type == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=opt_cfg.lr,
                weight_decay=opt_cfg.weight_decay,
                betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
            )
        elif opt_cfg.type == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=opt_cfg.lr,
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
