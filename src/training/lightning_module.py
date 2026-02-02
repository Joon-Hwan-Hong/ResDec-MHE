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

from src.models.full_model import CognitiveResilienceModel
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
    """

    def __init__(self, config: DictConfig):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

        # Build model
        model_cfg = config.model
        use_bayesian = model_cfg.head.type == "bayesian"

        self.model = CognitiveResilienceModel(
            n_genes=model_cfg.n_genes,
            n_cell_types=model_cfg.n_cell_types,
            d_embed=model_cfg.d_embed,
            d_fused=model_cfg.d_fused,
            d_cond=model_cfg.pathology_attention.d_cond,
            n_regions=model_cfg.get("n_regions", 6),
            n_hgt_layers=model_cfg.hgt.n_layers,
            n_hgt_heads=model_cfg.hgt.n_heads,
            n_cell_transformer_heads=model_cfg.set_transformer.get("n_heads", 4),
            n_isab_layers=model_cfg.set_transformer.n_isab_layers,
            n_inducing_points=model_cfg.set_transformer.n_inducing_points,
            n_attention_heads=model_cfg.pathology_attention.n_heads,
            gene_gate_temperature=model_cfg.gene_gate.get("initial_temperature", 2.0),
            selection_temperature=model_cfg.cell_type_selector.get("selection_temperature", 1.0),
            use_bayesian_head=use_bayesian,
            d_head_hidden=model_cfg.head.d_hidden,
            dropout=model_cfg.get("dropout", 0.1),
        )

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

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Training step: forward pass + loss computation."""
        output = self._forward_batch(batch)
        loss = self._compute_loss(output, batch["cognition"])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        """Validation step: forward pass + loss + metrics."""
        output = self._forward_batch(batch)
        loss = self._compute_loss(output, batch["cognition"])
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        # Compute and log metrics
        std = output.get("std")
        metrics = self.metrics.compute(output["mean"], std, batch["cognition"])
        for name, value in metrics.items():
            if not (isinstance(value, float) and value != value):  # skip NaN
                self.log(f"val_{name}", value, sync_dist=True)

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
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=train_cfg.max_epochs - warmup_epochs,
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
