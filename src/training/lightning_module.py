"""
PyTorch Lightning module wrapping CognitiveResilienceModel.

Handles:
- Loss branching: β-NLL for Bayesian head, MSE for deterministic head
- Optimizer and scheduler configuration
- Metric logging (training and validation)
- Optional gene gate L1 regularization

Metric logging strategy (Bayesian head):
    The Bayesian head trains via SVI, optimizing the ELBO (evidence lower bound).
    The ELBO decomposes as: ELBO = E[log p(y|x,θ)] - KL(q(θ) || p(θ)), where the
    KL term regularizes the variational posterior q(θ) toward the prior p(θ).

    For model selection (early stopping + checkpointing), we monitor val_nll
    (predictive Beta-NLL loss at posterior median). With 1/N KL normalization
    (Graves 2011, Blundell et al. 2015), the KL term is properly scaled but
    remains a regularizer, not a measure of predictive quality. Monitoring
    val_loss (ELBO) was found to prevent early stopping from firing because KL
    monotonically decreases, masking prediction quality degradation
    (FixAttempt2 diagnostic, 2026-03-12).

    The full ELBO is still logged as val_loss / val_elbo for posterior health
    diagnostics. If posterior collapse occurs (KL → 0 while NLL increases),
    this will be visible in val_elbo diverging from val_nll.

    For the deterministic head, val_loss is simply MSE — no ambiguity.
"""

import logging
import math
from typing import Any

import torch
import lightning.pytorch as pl
from omegaconf import DictConfig

import pyro
import pyro.poutine
from src.training.kl_annealing import KLAnnealedELBO
from pyro.infer.autoguide import AutoDiagonalNormal

from src.data.constants import N_REGIONS
from src.models.full_model import CognitiveResilienceModel, build_model_from_config
from src.training.losses import BetaNLLLoss, mse_loss
from src.training.metrics import ResilienceMetrics

logger = logging.getLogger(__name__)

# Float tensor keys that can contain NaN from data sources.
# cells/cell_mask are validated at preprocessing (dataset construction).
# Boolean masks and integer tensors cannot be NaN.
_NAN_CHECK_KEYS = ("pseudobulk", "pathology", "cognition", "ccc_edge_attr",
                    "region_pseudobulk")


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
            # when constructing the next one (see scripts/hpo.py train_fn).
            # Constraint: only one CognitiveResilienceLightning instance may exist
            # per process at a time. Under DDP, each rank is a separate process,
            # so this is safe across ranks.
            pyro.clear_param_store()
            self.guide = AutoDiagonalNormal(self.model)
            # Disable Pyro's runtime validation (shape/support/constraint checks
            # on every pyro.sample call). This is a GLOBAL Pyro setting that persists
            # for the process lifetime. Safe for training (one module per process).
            # Re-enable with pyro.enable_validation(True) for debugging.
            pyro.enable_validation(False)
            kl_cfg = config.training.get("kl_annealing", {})
            kl_temp = kl_cfg.get("temperature", 1.0) if kl_cfg else 1.0
            self.elbo = KLAnnealedELBO(kl_weight=1.0, temperature=kl_temp)
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
        self._max_nan_skip_fraction = error_cfg.get("max_nan_skip_fraction", 0.1)
        # Per-epoch NaN counters — intentionally NOT checkpointed.
        # On resume, Lightning restarts the interrupted epoch from the
        # beginning, so these correctly start at 0 for the full re-run.
        self._epoch_nan_skips = 0
        self._epoch_total_batches = 0

        # Metrics
        self.metrics = ResilienceMetrics()

        # Epoch-level accumulators for validation metrics (P3, P15)
        self._val_means: list[torch.Tensor] = []
        self._val_targets: list[torch.Tensor] = []
        self._val_stds: list[torch.Tensor] = []
        self._val_elbos: list[float] = []

        # Epoch-level accumulators for test metrics (same pattern as val)
        self._test_means: list[torch.Tensor] = []
        self._test_targets: list[torch.Tensor] = []
        self._test_stds: list[torch.Tensor] = []

        # CPU-cached last training batch for epoch-end NLL computation (P7)
        self._last_train_batch_ref: dict | None = None

        # Batch reference for GE noise re-evaluation (set in training_step,
        # cleared by GradientModulationCallback.on_train_batch_end).
        self._current_batch: dict | None = None
        self._is_ge_reevaluation: bool = False

    def _gene_gate_l1_penalty(self) -> torch.Tensor:
        """Compute L1 penalty on all gene gate logits (HGT + CT gates).

        Returns:
            Scalar L1 penalty tensor (mean of absolute gate logits across all gates).
        """
        logits_list = []
        hgt_gate = getattr(self.model, "hgt_gene_gate", None)
        if hgt_gate is not None and hasattr(hgt_gate, "gate_logits"):
            logits_list.append(hgt_gate.gate_logits)
        ct = getattr(self.model, "cell_transformer", None)
        ct_gate = getattr(ct, "gene_gate", None) if ct is not None else None
        if ct_gate is not None and hasattr(ct_gate, "gate_logits"):
            logits_list.append(ct_gate.gate_logits)
        if not logits_list:
            return torch.tensor(0.0, device=self.device)
        all_logits = torch.cat([l.flatten() for l in logits_list])
        return self._gene_gate_l1_lambda * all_logits.abs().mean()

    def _compute_loss(self, output: dict, cognition: torch.Tensor) -> torch.Tensor:
        """Compute loss with branching based on head type."""
        if self._use_mse_loss:
            loss = mse_loss(output["mean"], cognition)
        else:
            loss = self.loss_fn(output["mean"], output["std"], cognition)

        # Optional gene gate L1 regularization (both gates)
        if self._gene_gate_l1_lambda > 0:
            loss = loss + self._gene_gate_l1_penalty()

        return loss

    @staticmethod
    def _batch_to_model_kwargs(batch: dict) -> dict:
        """Extract model forward-pass kwargs from batch dict.

        Single source of truth for batch -> model argument mapping.
        Used by _forward_batch, _svi_forward, and _forward_batch_posterior.

        Prefers flat cell format (cell_data + cell_offsets) over padded
        (cells + cell_mask) when both are present in the batch.
        """
        kwargs = {
            "region_pseudobulk": batch.get("region_pseudobulk"),
            "region_mask": batch.get("region_mask"),
            "pseudobulk": batch.get("pseudobulk"),
            "ccc_edge_index": batch.get("ccc_edge_index"),
            "ccc_edge_type": batch.get("ccc_edge_type"),
            "ccc_edge_attr": batch.get("ccc_edge_attr"),
            "cell_type_mask": batch.get("cell_type_mask"),
            "pathology": batch.get("pathology"),
            "cognition": batch.get("cognition"),
        }
        # Flat cell format
        kwargs["cell_data"] = batch["cell_data"]
        kwargs["cell_offsets"] = batch["cell_offsets"]
        return kwargs

    def _forward_batch(self, batch: dict) -> dict:
        """Run forward pass extracting inputs from batch dict."""
        return self.model(**self._batch_to_model_kwargs(batch))

    def _svi_forward(self, batch: dict[str, Any]) -> torch.Tensor:
        """Compute differentiable ELBO loss for Bayesian training.

        Returns a differentiable loss tensor that flows through Lightning's
        standard backward pass, enabling DDP gradient synchronization.

        When the ELBO supports differentiable_loss_with_parts (KLAnnealedELBO),
        also logs decomposed NLL and KL terms for diagnostics.

        Runs in float32 regardless of AMP autocast state: Pyro's log_prob()
        computes std**2 and log(std) which underflow in bf16 when posterior
        scales are small (e.g. std=1e-6 → std**2=1e-12, below bf16 min ~1e-8).
        """
        # Disable autocast for ELBO: log-probability arithmetic requires float32
        # precision. Under bf16, small posterior scales cause std**2 underflow
        # and corrupt KL divergence terms.
        device_type = self.device.type if self.device.type in ("cuda", "cpu") else "cpu"
        with torch.amp.autocast(device_type, enabled=False):
            if hasattr(self.elbo, 'differentiable_loss_with_parts'):
                nll, kl, loss = self.elbo.differentiable_loss_with_parts(
                    self.model, self.guide,
                    **self._batch_to_model_kwargs(batch),
                )
                if (self.training
                        and not getattr(self, '_is_prototyping', False)
                        and not getattr(self, '_is_ge_reevaluation', False)):
                    bs = batch["cognition"].shape[0]
                    self.log("train_nll", nll.detach(), prog_bar=False, sync_dist=True, batch_size=bs)
                    self.log("train_kl_weighted", kl.detach(), prog_bar=False, sync_dist=True, batch_size=bs)
            else:
                loss = self.elbo.differentiable_loss(
                    self.model, self.guide,
                    **self._batch_to_model_kwargs(batch),
                )
        return loss

    def _forward_batch_posterior(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
        Forward pass using posterior median (MAP estimate) from guide.

        Used for validation/test to get deterministic predictions from the
        learned posterior, avoiding sampling noise.

        AMP note: this runs UNDER autocast (unlike _svi_forward). The
        pyro.sample call in BayesianPredictionHead records log_prob at bf16
        precision, but this does not affect the returned mean/std values.
        Validation ELBO is computed separately via _svi_forward with autocast
        disabled (line 198).

        Falls back to standard forward if guide hasn't been prototyped yet
        (safety net only — guide is prototyped in configure_optimizers).
        """
        if getattr(self.guide, 'prototype_trace', None) is None:
            return self._forward_batch(batch)
        median = self.guide.median()
        conditioned = pyro.poutine.condition(self.model, data=median)
        return conditioned(**self._batch_to_model_kwargs(batch))

    def _check_batch_nan(self, batch: dict) -> bool:
        """Check if batch contains NaN values in data-source tensors.

        Only checks float tensors from external data sources (metadata, LIANA
        scores). Tensors validated at preprocessing (cells, cell_mask) are excluded.
        """
        for key in _NAN_CHECK_KEYS:
            value = batch.get(key)
            if value is not None and not torch.isfinite(value.sum()):
                return True
        return False

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor | None:
        """Training step with ELBO loss for Bayesian head.

        Returns None on NaN skip (nan_batch="skip") in single-GPU mode only.
        Under DDP, NaN-skip is forced to "fail" to prevent rank desync: if one
        rank returns None (skipping backward) while others proceed with
        loss.backward(), the proceeding ranks hang at allreduce indefinitely.
        """
        self._epoch_total_batches += 1

        # Under DDP, NaN-skip is unsafe: if only one rank returns None while
        # others call loss.backward(), the proceeding ranks hang at allreduce.
        # Force fail policy so NaN crashes immediately with a clear error.
        # Fix NaN data upstream (precompute_features validation) instead.
        nan_batch_policy = self._nan_batch_policy
        nan_loss_policy = self._nan_loss_policy
        world_size = self.trainer.world_size if self._trainer is not None else 1
        if world_size > 1:
            nan_batch_policy = "fail"
            nan_loss_policy = "fail"

        if nan_batch_policy == "skip" and self._check_batch_nan(batch):
            self._epoch_nan_skips += 1
            subject_ids = batch.get("subject_ids", ["unknown"])
            logger.warning("NaN detected in batch %d (subjects=%s) — skipping", batch_idx, subject_ids)
            return None
        elif nan_batch_policy == "fail" and self._check_batch_nan(batch):
            subject_ids = batch.get("subject_ids", ["unknown"])
            raise ValueError(
                f"NaN detected in batch {batch_idx} (subjects={subject_ids}). "
                f"Under DDP, NaN-skip is disabled to prevent rank desync. "
                f"Validate data with precompute_features before multi-GPU training."
            )

        if self._use_bayesian_svi:
            loss = self._svi_forward(batch)
            # Apply gene gate L1 regularization (also needed in SVI path)
            if self._gene_gate_l1_lambda > 0:
                loss = loss + self._gene_gate_l1_penalty()
        else:
            output = self._forward_batch(batch)
            loss = self._compute_loss(output, batch["cognition"])

        # Check for NaN loss
        if torch.isnan(loss):
            if nan_loss_policy == "fail":
                raise ValueError(f"NaN loss detected at batch {batch_idx}")
            else:
                self._epoch_nan_skips += 1
                logger.warning("NaN loss at batch %d — skipping", batch_idx)
                return None

        # Cache batch for GE noise re-evaluation (GradientModulationCallback).
        # Cleared after CPU copy below to avoid holding an extra GPU batch ref
        # when gradient modulation is disabled (the callback would clear it).
        self._current_batch = batch

        bs = batch["cognition"].shape[0]
        # For Bayesian head: train_loss = ELBO (the actual optimization target).
        # For deterministic head: train_loss = MSE.
        # DDP-1: sync_dist=False — per-step allreduce on training loss is
        # unnecessary overhead. Lightning already reduces gradients via DDP.
        # Each rank logs its local loss; Lightning averages across steps.
        self.log("train_loss", loss, prog_bar=True, batch_size=bs)

        # Cache last batch on CPU for epoch-end NLL computation.
        # Moved to CPU to release ~10 GB GPU memory during validation.
        # on_train_epoch_end moves it back to GPU (~50ms over PCIe4).
        # Under DDP, each rank caches its own last batch. train_loss_nll is
        # logged with sync_dist=True, averaging each rank's NLL estimate.
        if self._use_bayesian_svi:
            self._last_train_batch_ref = {
                k: v.cpu() if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
                if isinstance(v, torch.Tensor) or k == "subject_ids"
            }

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        """Release GPU batch ref cached for GradientModulationCallback.

        GradientModulationCallback.on_after_backward reads _current_batch
        (fires before this hook). When gradient modulation is disabled, no
        callback clears it, so we do it here unconditionally.
        """
        self._current_batch = None

    def on_train_epoch_end(self) -> None:
        """Check NaN skip rate and compute NLL on last batch (Bayesian only)."""
        # Check NaN skip rate for the epoch.
        # Under DDP, nan_batch_policy is forced to "fail" (see training_step),
        # so _epoch_nan_skips is always 0 and this branch is unreachable.
        # The threshold check exists for single-GPU NaN-skip diagnostics.
        if self._epoch_total_batches > 0 and self._epoch_nan_skips > 0:
            skip_frac = self._epoch_nan_skips / self._epoch_total_batches
            self.log("nan_skip_fraction", skip_frac, sync_dist=True)
            if skip_frac > self._max_nan_skip_fraction:
                raise RuntimeError(
                    f"NaN skip rate {skip_frac:.1%} ({self._epoch_nan_skips}/{self._epoch_total_batches} batches) "
                    f"exceeds threshold {self._max_nan_skip_fraction:.1%}. "
                    f"This indicates a data pipeline issue. "
                    f"Configure error_handling.training.max_nan_skip_fraction to adjust."
                )
        self._epoch_nan_skips = 0
        self._epoch_total_batches = 0

        if self._use_bayesian_svi and self._last_train_batch_ref is not None:
            # Move cached batch back to GPU for NLL computation.
            # Cost: ~50ms for 10 GB over PCIe4, negligible vs training time.
            device = self.device
            gpu_batch = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in self._last_train_batch_ref.items()
            }
            with torch.no_grad():
                nll_output = self._forward_batch_posterior(gpu_batch)
                nll_loss = self._compute_loss(nll_output, gpu_batch["cognition"])
            bs = gpu_batch["cognition"].shape[0]
            self.log("train_loss_nll", nll_loss, sync_dist=True, batch_size=bs)
            self._last_train_batch_ref = None

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        """Validation step — accumulates predictions for epoch-level metrics.

        For Bayesian head: val_loss = ELBO (matches training objective, includes KL
        term for posterior health monitoring). Beta-NLL logged separately as val_nll.
        For deterministic head: val_loss = MSE.
        """
        if self._use_bayesian_svi:
            output = self._forward_batch_posterior(batch)
            # ELBO on val set — used as val_loss for early stopping/checkpointing.
            # See module docstring for rationale.
            with torch.no_grad():
                val_elbo = self._svi_forward(batch)
            self._val_elbos.append(val_elbo.detach().item())
            # Beta-NLL (predictive quality at posterior median) logged as val_nll
            nll_loss = self._compute_loss(output, batch["cognition"])
            bs = batch["cognition"].shape[0]
            self.log("val_nll", nll_loss, prog_bar=False, sync_dist=True, batch_size=bs)
            self.log("val_loss", val_elbo, prog_bar=True, sync_dist=True, batch_size=bs)
        else:
            output = self._forward_batch(batch)
            loss = self._compute_loss(output, batch["cognition"])
            bs = batch["cognition"].shape[0]
            self.log("val_loss", loss, prog_bar=True, sync_dist=True, batch_size=bs)

        # Accumulate predictions for epoch-level correlation metrics.
        # .cpu() moves to host immediately, freeing GPU memory that would
        # otherwise accumulate across the full validation epoch.
        self._val_means.append(output["mean"].detach().cpu())
        self._val_targets.append(batch["cognition"].detach().cpu())
        if output.get("std") is not None:
            self._val_stds.append(output["std"].detach().cpu())

    def _get_real_dataset_size(self, prefix: str) -> int | None:
        """Get actual dataset size (before DistributedSampler padding).

        Returns None if the dataset size cannot be determined (e.g., unit tests
        without a real trainer/datamodule).
        """
        try:
            if prefix == "val":
                dl = self.trainer.val_dataloaders
            elif prefix == "test":
                dl = self.trainer.test_dataloaders
            else:
                return None
            if dl is None:
                return None
            # Lightning may return a single DataLoader or a list
            if isinstance(dl, (list, tuple)):
                dl = dl[0]
            # len(dl.dataset) returns the original (unpadded) size because
            # DataLoader.dataset returns the raw dataset, not the
            # DistributedSampler wrapper. Padded length is len(dl.sampler).
            return len(dl.dataset)
        except (RuntimeError, AttributeError):
            return None

    def _gather_and_compute_metrics(
        self,
        means_list: list[torch.Tensor],
        targets_list: list[torch.Tensor],
        stds_list: list[torch.Tensor],
        prefix: str,
    ) -> None:
        """Gather predictions across DDP ranks and compute epoch-level metrics.

        Under DDP, each rank only sees its shard of the data. Correlation metrics
        (Pearson r, Spearman rho, R²) computed on partial data and then averaged
        are biased. Instead, we all_gather predictions and compute once on the
        full dataset (rank 0 only to avoid duplicate logging).
        """
        if not means_list:
            return

        all_means = torch.cat(means_list, dim=0)
        all_targets = torch.cat(targets_list, dim=0)
        all_stds = torch.cat(stds_list, dim=0) if stds_list else None

        # Gather across DDP ranks for correct correlation computation
        try:
            world_size = self.trainer.world_size
            is_global_zero = self.trainer.is_global_zero
        except RuntimeError:
            # Module not attached to Trainer (unit tests)
            world_size = 1
            is_global_zero = True

        if world_size > 1:
            all_means = self.all_gather(all_means).reshape(-1, all_means.shape[-1])
            all_targets = self.all_gather(all_targets).reshape(-1, all_targets.shape[-1])
            if all_stds is not None:
                all_stds = self.all_gather(all_stds).reshape(-1, all_stds.shape[-1])

            # DistributedSampler pads the dataset to make it evenly divisible
            # across ranks, duplicating some samples. Truncate to the real dataset
            # size to prevent biased correlation metrics (Pearson r, Spearman, R²).
            real_n = self._get_real_dataset_size(prefix)
            if real_n is not None and real_n < all_means.shape[0]:
                all_means = all_means[:real_n]
                all_targets = all_targets[:real_n]
                if all_stds is not None:
                    all_stds = all_stds[:real_n]

        # Compute on rank 0 only to avoid duplicate logs.
        # Note: val_pearson_r, val_spearman_rho, etc. are only available in
        # trainer.callback_metrics on rank 0. Lightning's ModelCheckpoint and
        # EarlyStopping read from rank 0, so this is correct.
        if is_global_zero:
            metrics = self.metrics.compute(all_means, all_stds, all_targets)
            for name, value in metrics.items():
                if not (isinstance(value, float) and math.isnan(value)):
                    self.log(f"{prefix}_{name}", value, rank_zero_only=True, sync_dist=True)

            # Bootstrap CI on R² (validation only, ~50ms for 1000 resamples on N=93)
            if prefix == "val":
                ci = self.metrics.bootstrap_ci(all_means, target=all_targets, metrics=["r2"])
                if "r2" in ci:
                    self.log("val_r2_ci_lower", ci["r2"][0], rank_zero_only=True, sync_dist=True)
                    self.log("val_r2_ci_upper", ci["r2"][1], rank_zero_only=True, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        """Compute epoch-level metrics from accumulated predictions."""
        self._gather_and_compute_metrics(
            self._val_means, self._val_targets, self._val_stds, "val",
        )

        # Log mean ELBO across all validation batches (diagnostic, separate from
        # the per-step val_loss which is also ELBO for the Bayesian head).
        # Under DDP, each rank computes mean from its local shard. sync_dist=True
        # averages across ranks. DistributedSampler padding may introduce a minor
        # bias (< 1 sample for typical setup), acceptable for a diagnostic metric.
        if self._val_elbos:
            mean_elbo = math.fsum(self._val_elbos) / len(self._val_elbos)
            self.log("val_elbo", mean_elbo, prog_bar=False, sync_dist=True)

        # Clear accumulators
        self._val_means.clear()
        self._val_targets.clear()
        self._val_stds.clear()
        self._val_elbos.clear()

    def test_step(self, batch: dict, batch_idx: int) -> None:
        """Test step — accumulates predictions for epoch-level metrics."""
        if self._use_bayesian_svi:
            output = self._forward_batch_posterior(batch)
        else:
            output = self._forward_batch(batch)
        loss = self._compute_loss(output, batch["cognition"])
        bs = batch["cognition"].shape[0]
        self.log("test_loss", loss, prog_bar=True, sync_dist=True, batch_size=bs)

        # Accumulate for epoch-level computation (same as validation).
        # .cpu() moves to host immediately, freeing GPU memory.
        self._test_means.append(output["mean"].detach().cpu())
        self._test_targets.append(batch["cognition"].detach().cpu())
        if output.get("std") is not None:
            self._test_stds.append(output["std"].detach().cpu())

    def on_test_epoch_end(self) -> None:
        """Compute epoch-level test metrics from accumulated predictions."""
        self._gather_and_compute_metrics(
            self._test_means, self._test_targets, self._test_stds, "test",
        )
        self._test_means.clear()
        self._test_targets.clear()
        self._test_stds.clear()

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

    def _prototype_guide_if_needed(self, caller: str = "") -> bool:
        """Prototype Bayesian guide with a dummy forward pass if not already done.

        AutoDiagonalNormal creates variational parameters (loc, scale_unconstrained)
        lazily during the first forward pass. This helper ensures the guide has
        parameters before they're needed (e.g., by the optimizer or load_state_dict).

        Uses self.device when available (after DDP setup), falls back to CPU.

        Args:
            caller: Name of the calling method (for logging).

        Returns:
            True if prototyping was performed, False if already prototyped.
        """
        if not self._use_bayesian_svi or self.guide is None:
            return False

        if getattr(self.guide, 'prototype_trace', None) is not None:
            return False

        model_cfg = self.config.model
        # Use self.device when available (configure_optimizers, post-DDP);
        # fall back to CPU (on_load_checkpoint, which runs before device setup).
        try:
            device = self.device
        except RuntimeError:
            device = torch.device("cpu")

        dummy_batch = {
            "region_pseudobulk": torch.zeros(
                1, N_REGIONS, model_cfg.n_cell_types, model_cfg.n_genes,
                device=device,
            ),
            "region_mask": torch.ones(1, N_REGIONS, dtype=torch.bool, device=device),
            "cell_data": torch.zeros(
                0, model_cfg.n_genes,
                device=device,
            ),
            "cell_offsets": torch.zeros(
                1, model_cfg.n_cell_types + 1, dtype=torch.long,
                device=device,
            ),
            "cell_type_mask": torch.ones(
                1, model_cfg.n_cell_types, dtype=torch.bool,
                device=device,
            ),
            "pathology": torch.zeros(
                1,
                model_cfg.get("pathology_attention", {}).get("n_pathology_features", 3),
                device=device,
            ),
            "ccc_edge_index": torch.zeros(2, 0, dtype=torch.long, device=device),
            "ccc_edge_type": torch.zeros(0, dtype=torch.long, device=device),
            "ccc_edge_attr": torch.zeros(0, 1, device=device),
            "cognition": torch.zeros(1, 1, device=device),
        }
        self._is_prototyping = True
        try:
            with torch.no_grad():
                self._svi_forward(dummy_batch)
        finally:
            self._is_prototyping = False

        n_params = sum(1 for _ in self.guide.parameters())
        logger.info("Guide prototyped in %s: %d parameter tensors", caller, n_params)
        return True

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Pre-initialize Bayesian guide before load_state_dict.

        AutoDiagonalNormal creates loc/scale_unconstrained lazily during the
        first forward pass (_setup_prototype). On checkpoint resume,
        load_state_dict(strict=True) runs BEFORE configure_optimizers, so the
        guide has no parameters yet → 'unexpected keys' error. We prototype
        the guide here (this hook runs right before load_state_dict).
        """
        if not self._use_bayesian_svi or self.guide is None:
            return

        state_dict = checkpoint.get("state_dict", {})
        has_guide_keys = any(k.startswith("guide.") for k in state_dict)
        if not has_guide_keys:
            return

        self._prototype_guide_if_needed(caller="on_load_checkpoint")

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
            # Bayesian SVI uses Adam + ExponentialLR (Pyro convention).
            # training.optimizer.type and training.scheduler.* are intentionally
            # ignored here — validate_config() warns if non-defaults are set.
            # See: https://pyro.ai/examples/svi_part_iv.html

            # Set ELBO likelihood scaling for DDP (world_size > 1)
            if world_size > 1:
                self.model.prediction_head.set_data_scale(float(world_size))
                logger.info(f"Bayesian ELBO scaling: data_scale={world_size} for DDP")

            # Set 1/N KL normalization (Graves 2011, Blundell et al. 2015).
            # KL complexity cost applies once per dataset, not once per sample.
            train_ds = self.trainer.datamodule.train_dataset
            if train_ds is None or len(train_ds) == 0:
                raise RuntimeError(
                    "Cannot determine training set size for 1/N KL scaling: "
                    "datamodule.train_dataset is None or empty. "
                    "Ensure datamodule.setup('fit') has been called."
                )
            n_train = len(train_ds)
            self.elbo.n_train = n_train
            logger.info(f"KL 1/N normalization: n_train={n_train}")

            # Prototype the guide so AutoDiagonalNormal creates its variational
            # parameters (loc, scale). Without this, guide.parameters() returns []
            # and the optimizer never updates the posterior.
            # NOTE: This must happen in configure_optimizers (not later) because
            # Lightning calls configure_optimizers() BEFORE DDP wrapping. Guide
            # parameters created here are included in DDP's parameter list.
            # If Lightning ever reorders this, guide gradients won't sync across
            # ranks. The _prototype_guide_if_needed helper is idempotent — if
            # on_load_checkpoint already prototyped (checkpoint resume), this is a no-op.
            self._prototype_guide_if_needed(caller="configure_optimizers")

            # Separate param groups: encoder (model) and guide (variational posterior).
            # Guide params start from loc=0 and need higher LR to converge within
            # training budget. Standard SVI practice: Pyro's ClippedAdam uses lr=0.01.
            guide_lr = opt_cfg.get("guide_lr", None)
            if guide_lr is None:
                guide_lr = effective_lr
            logger.info(f"Optimizer LR: encoder={effective_lr}, guide={guide_lr}")

            weight_decay = opt_cfg.get("weight_decay", 0)
            betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))
            optimizer = torch.optim.Adam([
                {"params": list(self.model.parameters()), "lr": effective_lr},
                {"params": list(self.guide.parameters()), "lr": guide_lr},
            ], weight_decay=weight_decay, betas=betas)
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
            # Allow explicit T_max override (e.g., to keep schedule calibrated
            # at 100 epochs while training for 150 — epochs beyond T_max train
            # at eta_min, acting as a low-LR fine-tuning phase).
            t_max = sched_cfg.get("T_max", train_cfg.max_epochs - warmup_epochs)
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
