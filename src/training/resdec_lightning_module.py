"""PyTorch Lightning wrapper composing the existing CognitiveResilienceModel
encoder with the new ResDec-H3 head (Phase 1 single-stage composer).

Phase 1 scope
-------------
- Encoder: existing ``CognitiveResilienceModel`` (unchanged), built via
  :func:`build_model_from_config`. Forward returns a dict with ``attended``
  ``[B, d_fused]`` — this is the subject embedding consumed by the head.
- Head: :class:`ResDecH3Head` (FiLM + single NPTStage + scalar readout).
- Loss: MSE against ``cognition`` (deterministic head — the Bayesian SVI
  machinery in :class:`CognitiveResilienceLightningModule` is not needed here
  because the ResDec-H3 head produces its own scalar readout).
- Optimizer: AdamW with cosine annealing + linear warmup, following the same
  pattern as the existing deterministic-head path in
  :class:`CognitiveResilienceLightningModule`.

Metadata wiring
---------------
ResDecH3Head consumes an 8-dim metadata vector (APOE/sex/age FiLM conditioning).
The current datamodule does not yet produce a ``metadata`` key, so Phase 1 uses
a zero placeholder (FiLM initialises near-identity, so zeros → no-op). Proper
wiring is deferred to Phase 4 — see the TODO in :meth:`ResDecLightningModule.forward`.

This task (1.9a) only writes + unit-tests the wrapper. Training is exercised
downstream in task 1.9b.
"""
from __future__ import annotations

import logging
import math
from typing import Any

import lightning.pytorch as pl
import torch
from omegaconf import DictConfig

from src.models.full_model import build_model_from_config
from src.models.resdec_head.resdec_h3_head import ResDecH3Head
from src.training.losses import mse_loss

logger = logging.getLogger(__name__)

# Keys required by the encoder's forward (mirrors
# CognitiveResilienceLightningModule._batch_to_model_kwargs).
_ENCODER_KWARG_KEYS = (
    "region_pseudobulk",
    "region_mask",
    "pseudobulk",
    "ccc_edge_index",
    "ccc_edge_type",
    "ccc_edge_attr",
    "cell_type_mask",
    "pathology",
    "cognition",
    "cell_data",
    "cell_offsets",
)


class ResDecLightningModule(pl.LightningModule):
    """Lightning wrapper: encoder (unchanged) → ResDec-H3 head.

    Args:
        config: OmegaConf DictConfig with ``model`` and ``training`` sections.
            ``config.model.resdec_head.d_metadata`` controls the FiLM input
            dimension. ``d_subject`` for the head is inferred from
            ``config.model.d_fused`` (the encoder's attended-vector dim).
    """

    def __init__(self, config: DictConfig):
        super().__init__()
        # ResDec-H3 runs with the deterministic head; no guide/config to
        # persist via Lightning's hparams machinery.
        self.save_hyperparameters(ignore=["config"])
        self.config = config

        # Build encoder — existing model, no modifications. The deterministic
        # prediction head is still built (it lives at self.encoder.prediction_head),
        # but we ignore its scalar output; the ResDec-H3 head reads `attended`
        # directly.
        model_cfg = config.model
        self.encoder = build_model_from_config(model_cfg)

        # d_subject == d_fused: the encoder's PathologyStratifiedAttention
        # returns `attended` of shape [B, d_fused]. (Not d_embed * 2 — verified
        # against src/models/fusion/pathology_attention.py.)
        d_subject = int(model_cfg.d_fused)
        resdec_cfg = model_cfg.get("resdec_head", {}) or {}
        d_metadata = int(resdec_cfg.get("d_metadata", 8))
        n_heads = int(resdec_cfg.get("n_heads", 4))
        n_hc_streams = int(resdec_cfg.get("n_hc_streams", 4))
        lambda_init = float(resdec_cfg.get("lambda_init", 0.8))

        self.head = ResDecH3Head(
            d_subject=d_subject,
            d_metadata=d_metadata,
            n_heads=n_heads,
            n_hc_streams=n_hc_streams,
            lambda_init=lambda_init,
        )
        self._d_metadata = d_metadata

        # Validation accumulators for epoch-level R² / MSE.
        self._val_preds: list[torch.Tensor] = []
        self._val_targets: list[torch.Tensor] = []

    # ------------------------------------------------------------------ #
    # Forward                                                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _batch_to_encoder_kwargs(batch: dict) -> dict:
        """Select encoder-relevant keys from the batch dict."""
        return {k: batch.get(k) for k in _ENCODER_KWARG_KEYS}

    def _get_metadata(self, batch: dict, batch_size: int) -> torch.Tensor:
        """Return [B, d_metadata] FiLM conditioning vector.

        # TODO(phase4): wire metadata from datamodule via
        #   src.data.tabpfn_input.load_metadata_vector
        # Phase 1 placeholder: zero tensor. FiLM is initialised so that
        # gamma=1, beta=0 → zero metadata leaves z unchanged (near-identity).
        """
        md = batch.get("metadata")
        if md is None:
            device = batch["cognition"].device if batch.get("cognition") is not None else self.device
            return torch.zeros(batch_size, self._d_metadata, device=device)
        if md.dim() == 1:
            md = md.unsqueeze(0)
        if md.shape[-1] != self._d_metadata:
            raise ValueError(
                f"Metadata last-dim mismatch: expected {self._d_metadata}, got {md.shape[-1]}"
            )
        return md

    def forward(self, batch: dict) -> dict:
        """Run encoder → extract `attended` → run ResDec-H3 head.

        Returns dict with:
            prediction: [B] scalar cognition prediction from the head
            latent_1:   [B, d_subject] stage-1 latent (reserved for cross-stage attention in Phase 3)
            attended:   [B, d_subject] encoder output, for downstream debugging/logging
            attention_weights: [B, n_heads, n_cell_types] pathology attention (from encoder)
        """
        enc_out = self.encoder(**self._batch_to_encoder_kwargs(batch))
        z = enc_out["attended"]  # [B, d_subject]
        B = z.shape[0]
        metadata = self._get_metadata(batch, B)
        head_out = self.head(z, metadata)

        out: dict[str, Any] = {
            "prediction": head_out["prediction"],
            "latent_1": head_out["latent_1"],
            "attended": z,
        }
        if enc_out.get("attention_weights") is not None:
            out["attention_weights"] = enc_out["attention_weights"]
        return out

    # ------------------------------------------------------------------ #
    # Train / Val steps                                                  #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """MSE between [B] prediction and [B, 1] or [B] target.

        Centralised here so training_step and validation_step agree on shapes.
        """
        if target.dim() == 2 and target.shape[-1] == 1:
            target = target.squeeze(-1)
        # Delegate the actual reduction to the shared loss implementation —
        # mse_loss handles [B] ↔ [B] MSE just fine.
        return mse_loss(pred, target)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        out = self.forward(batch)
        loss = self._mse(out["prediction"], batch["cognition"])
        bs = batch["cognition"].shape[0]
        self.log("train/mse", loss, prog_bar=True, batch_size=bs, sync_dist=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        out = self.forward(batch)
        pred = out["prediction"].detach()
        target = batch["cognition"].detach()
        if target.dim() == 2 and target.shape[-1] == 1:
            target = target.squeeze(-1)

        loss = torch.nn.functional.mse_loss(pred, target)
        bs = target.shape[0]
        self.log("val/mse_batch", loss, prog_bar=False, batch_size=bs, sync_dist=True)

        self._val_preds.append(pred.cpu())
        self._val_targets.append(target.cpu())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        preds = torch.cat(self._val_preds, dim=0)
        targets = torch.cat(self._val_targets, dim=0)

        # Epoch MSE over full val set (not batch-mean, which is sample-size biased).
        mse = torch.mean((preds - targets) ** 2)

        # R² = 1 - SS_res / SS_tot. If SS_tot == 0 (constant targets), R² is
        # undefined — log NaN so it's visible rather than silently 0.
        ss_res = torch.sum((targets - preds) ** 2)
        ss_tot = torch.sum((targets - targets.mean()) ** 2)
        if ss_tot > 0:
            r2 = (1.0 - (ss_res / ss_tot)).item()
        else:
            r2 = float("nan")

        self.log("val/mse", mse, prog_bar=True, sync_dist=True)
        if not math.isnan(r2):
            self.log("val/r2", r2, prog_bar=True, sync_dist=True)

        self._val_preds.clear()
        self._val_targets.clear()

    # ------------------------------------------------------------------ #
    # Optimizer                                                          #
    # ------------------------------------------------------------------ #
    def configure_optimizers(self) -> dict[str, Any]:
        train_cfg = self.config.training

        # lr and weight_decay: Phase 1 config sets them at the top-level of
        # `training`, but the project default lives under `training.optimizer`.
        # Honor both so this module works against either shape.
        lr = train_cfg.get("lr")
        if lr is None:
            lr = train_cfg.optimizer.lr
        weight_decay = train_cfg.get("weight_decay")
        if weight_decay is None:
            weight_decay = train_cfg.optimizer.get("weight_decay", 0.0)

        betas = tuple(train_cfg.get("betas", (0.9, 0.999)))

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(lr),
            weight_decay=float(weight_decay),
            betas=betas,
        )

        # Cosine annealing with 5-epoch linear warmup — matches the pattern in
        # CognitiveResilienceLightningModule.configure_optimizers for the
        # deterministic head.
        sched_cfg = train_cfg.get("scheduler", {}) or {}
        warmup_epochs = int(sched_cfg.get("warmup_epochs", 5))
        eta_min = float(sched_cfg.get("eta_min", 1e-6))
        max_epochs = int(train_cfg.get("max_epochs", 60))
        t_max = max(1, max_epochs - warmup_epochs)

        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=eta_min,
        )
        if warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs],
            )
        else:
            scheduler = cosine

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
