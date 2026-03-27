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
from pyro.infer import Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal

from src.data.constants import N_REGIONS
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

        # Bayesian head setup
        self._use_bayesian_svi = use_bayesian
        self.guide = None

        if self._use_bayesian_svi:
            # Safe to clear globally: the CV loop creates exactly one module at a
            # time per fold and does not hold references to previous fold modules
            # when constructing the next one (see scripts/optuna_optimize.py fold loop).
            pyro.clear_param_store()
            self.guide = AutoDiagonalNormal(self.model)
            self.elbo = Trace_ELBO()
            # automatic_optimization stays True (default) —
            # differentiable_loss returns a loss tensor that flows through
            # Lightning's standard backward + optimizer step + DDP gradient sync.

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
        """Compute differentiable ELBO loss for Bayesian training.

        Returns a differentiable loss tensor that flows through Lightning's
        standard backward pass, enabling DDP gradient synchronization.
        """
        loss = self.elbo.differentiable_loss(
            self.model,
            self.guide,
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
        return loss

    def _forward_batch_posterior(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
        Forward pass using posterior median (MAP estimate) from guide.

        Used for validation/test to get deterministic predictions from the
        learned posterior, avoiding sampling noise.

        Falls back to standard forward if guide hasn't been prototyped yet
        (safety net only — guide is prototyped in configure_optimizers).
        """
        if getattr(self.guide, 'prototype_trace', None) is None:
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

    def _check_batch_nan(self, batch: dict) -> bool:
        """Check if batch contains NaN values. Returns True if NaN detected.

        Uses sum-based check (isfinite on sum) to avoid allocating a boolean
        tensor the size of the input. Only checks tensors likely to contain NaN
        from data loading — cells/cell_mask come from preprocessing and are
        validated at dataset construction time.
        """
        # Keys that are validated during dataset construction / preprocessing.
        # Note: edge_index_dict_list and edge_attr_dict_list are Python lists
        # (not tensors), so isinstance(value, torch.Tensor) already skips them.
        _skip_keys = {"cells", "cell_mask", "cell_type_mask", "region_mask"}
        for key, value in batch.items():
            if key in _skip_keys:
                continue
            if isinstance(value, torch.Tensor) and value.is_floating_point() and not torch.isfinite(value.sum()):
                return True
        # Also check nested edge attribute structures
        edge_attr_list = batch.get("edge_attr_dict_list")
        if edge_attr_list is not None:
            for edge_dict in edge_attr_list:
                if isinstance(edge_dict, dict):
                    for v in edge_dict.values():
                        if isinstance(v, torch.Tensor) and v.is_floating_point() and not torch.isfinite(v.sum()):
                            return True
        return False

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor | None:
        """Training step with ELBO loss for Bayesian head."""
        if self._nan_batch_policy == "skip" and self._check_batch_nan(batch):
            logger.warning("NaN detected in batch %d — skipping", batch_idx)
            return None

        if self._use_bayesian_svi:
            loss = self._svi_forward(batch)
            # Apply gene gate L1 regularization (also needed in SVI path)
            if self._gene_gate_l1_lambda > 0:
                gate_logits = self.model.pseudobulk_encoder.gene_gate.gate_logits
                l1_penalty = self._gene_gate_l1_lambda * gate_logits.abs().mean()
                loss = loss + l1_penalty
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

        # For Bayesian path, periodically log NLL (comparable scale to val_loss).
        # Only every 50 steps to avoid doubling training time — each NLL computation
        # requires a full posterior-median forward pass through all three branches.
        if self._use_bayesian_svi and (batch_idx % 50 == 0):
            with torch.no_grad():
                nll_output = self._forward_batch_posterior(batch)
                nll_loss = self._compute_loss(nll_output, batch["cognition"])
            self.log("train_loss_nll", nll_loss, sync_dist=True, batch_size=bs)

        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        """Validation step using posterior median for Bayesian head."""
        if self._use_bayesian_svi:
            output = self._forward_batch_posterior(batch)
            # Log ELBO for monitoring posterior convergence (F2).
            # val_loss (beta-NLL) remains primary metric for model selection.
            # Only compute on first val batch to avoid doubling validation time.
            if batch_idx == 0:
                with torch.no_grad():
                    val_elbo = self._svi_forward(batch)
                bs = batch["cognition"].shape[0]
                self.log("val_elbo", val_elbo, prog_bar=False, sync_dist=True, batch_size=bs)
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

        # Query world_size once (used for LR scaling and ELBO scaling)
        try:
            world_size = self.trainer.world_size
        except RuntimeError:
            world_size = 1

        # Linear LR scaling for multi-GPU (Goyal et al. 2017)
        # effective_lr = base_lr * n_gpus when using DDP
        base_lr = opt_cfg.lr
        lr_scaling_enabled = train_cfg.get("lr_scaling", True)

        if lr_scaling_enabled and world_size > 1:
            scaled_lr = base_lr * world_size
            logger.info(
                f"LR scaling: {base_lr} × {world_size} GPUs = {scaled_lr}"
            )
            base_lr = scaled_lr

        effective_lr = base_lr

        if self._use_bayesian_svi:
            # Set ELBO likelihood scaling for DDP (world_size > 1)
            if world_size > 1:
                self.model.prediction_head.set_data_scale(float(world_size))
                logger.info(f"Bayesian ELBO scaling: data_scale={world_size} for DDP")

            # Prototype the guide so AutoDiagonalNormal creates its variational
            # parameters (loc, scale). Without this, guide.parameters() returns []
            # and the optimizer never updates the posterior (F1 fix).
            model_cfg = self.config.model
            dummy_batch = {
                "region_pseudobulk": torch.zeros(
                    1, N_REGIONS, model_cfg.n_cell_types, model_cfg.n_genes,
                    device=self.device,
                ),
                "region_mask": torch.ones(1, N_REGIONS, dtype=torch.bool, device=self.device),
                "cells": torch.zeros(
                    1, model_cfg.n_cell_types, 1, model_cfg.n_genes,
                    device=self.device,
                ),
                "cell_mask": torch.ones(
                    1, model_cfg.n_cell_types, 1, dtype=torch.bool,
                    device=self.device,
                ),
                "cell_type_mask": torch.ones(
                    1, model_cfg.n_cell_types, dtype=torch.bool,
                    device=self.device,
                ),
                "pathology": torch.zeros(
                    1,
                    model_cfg.get("pathology_attention", {}).get("n_pathology_features", 3),
                    device=self.device,
                ),
                "edge_index_dict_list": [{}],
                "edge_attr_dict_list": [{}],
                "cognition": torch.zeros(1, 1, device=self.device),
            }
            with torch.no_grad():
                self._svi_forward(dummy_batch)
            n_guide_params = sum(1 for _ in self.guide.parameters())
            logger.info(
                f"Guide prototyped in configure_optimizers: {n_guide_params} parameter tensors"
            )

            # Collect model + guide parameters
            all_params = list(self.model.parameters()) + list(self.guide.parameters())
            optimizer = torch.optim.Adam(
                all_params,
                lr=effective_lr,
                weight_decay=opt_cfg.get("weight_decay", 0),
                betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
            )
            # ExponentialLR replicates ClippedAdam's lrd parameter
            lrd = opt_cfg.get("lrd", 1.0)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lrd)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }

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
