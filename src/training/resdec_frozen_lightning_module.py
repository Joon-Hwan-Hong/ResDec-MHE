"""ResDec-MHE Lightning module for frozen-encoder training.

Consumes cached ``attended`` embeddings directly. No encoder forward at
train time. Head stack is the same :class:`ResDecH3Head` as the live-encoder
module — only the data path differs.

Motivation: full-cohort NPT attention on the live encoder is memory-heavy.
Pre-encoding every subject once and caching its 64-dim embedding lets us
train the head at full-cohort batch size without re-running the large
encoder per step.
"""
from __future__ import annotations

import math
from typing import Any

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from src.data.tabpfn_input import METADATA_FIELDS
from src.models.resdec_head.resdec_h3_head import ResDecH3Head


class ResDecFrozenLightningModule(pl.LightningModule):
    """Train the ResDec-MHE head on cached encoder embeddings."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        # Avoid persisting the full OmegaConf via save_hyperparameters (it can
        # include non-picklable objects); we hold a reference instead.
        self.cfg = cfg
        self.d_subject = int(cfg.model.d_fused)
        resdec_cfg = cfg.model.get("resdec_head", {}) or {}
        self._d_metadata = int(resdec_cfg.get("d_metadata", len(METADATA_FIELDS)))
        n_heads = int(resdec_cfg.get("n_heads", 4))
        n_hc_streams = int(resdec_cfg.get("n_hc_streams", 4))
        lambda_init = float(resdec_cfg.get("lambda_init", 0.8))

        self.head = ResDecH3Head(
            d_subject=self.d_subject,
            d_metadata=self._d_metadata,
            n_heads=n_heads,
            n_hc_streams=n_hc_streams,
            lambda_init=lambda_init,
        )

        # Validation accumulators for epoch-level R² / MSE.
        self._val_preds: list[torch.Tensor] = []
        self._val_targets: list[torch.Tensor] = []

    # ------------------------------------------------------------------ #
    # Forward                                                            #
    # ------------------------------------------------------------------ #
    def forward(self, batch: dict) -> dict:
        attended = batch["attended"]   # [B, d_subject]
        metadata = batch["metadata"]   # [B, d_metadata]
        if metadata.shape[-1] != self._d_metadata:
            raise ValueError(
                f"metadata last-dim mismatch: expected {self._d_metadata}, "
                f"got {metadata.shape[-1]}"
            )
        return self.head(attended, metadata)

    # ------------------------------------------------------------------ #
    # Train / Val steps                                                  #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _squeeze_target(target: torch.Tensor) -> torch.Tensor:
        if target.dim() == 2 and target.shape[-1] == 1:
            return target.squeeze(-1)
        return target

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        out = self.forward(batch)
        target = self._squeeze_target(batch["cognition"])
        loss = F.mse_loss(out["prediction"], target)
        self.log(
            "train/mse", loss,
            on_step=False, on_epoch=True, prog_bar=True,
            batch_size=target.shape[0],
        )
        return loss

    def on_validation_epoch_start(self) -> None:
        self._val_preds = []
        self._val_targets = []

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        out = self.forward(batch)
        target = self._squeeze_target(batch["cognition"])
        pred = out["prediction"].detach()
        # Note: per-batch val/mse is intentionally not logged here — the
        # epoch-level val/mse (logged in on_validation_epoch_end) is the
        # correct aggregate for early-stopping and model selection.
        self._val_preds.append(pred.cpu())
        self._val_targets.append(target.detach().cpu())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        preds = torch.cat(self._val_preds, dim=0)
        targets = torch.cat(self._val_targets, dim=0)

        # Epoch MSE over full val set (not batch-mean).
        mse = torch.mean((preds - targets) ** 2)
        ss_res = torch.sum((targets - preds) ** 2)
        ss_tot = torch.sum((targets - targets.mean()) ** 2)
        if ss_tot > 0:
            r2 = (1.0 - (ss_res / ss_tot)).item()
        else:
            r2 = float("nan")

        self.log("val/mse", mse, prog_bar=True)
        if not math.isnan(r2):
            self.log("val/r2", r2, prog_bar=True)

        self._val_preds.clear()
        self._val_targets.clear()

    # ------------------------------------------------------------------ #
    # Optimizer                                                          #
    # ------------------------------------------------------------------ #
    def configure_optimizers(self) -> dict[str, Any]:
        train_cfg = self.cfg.training
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
        return {"optimizer": optimizer}


__all__ = ["ResDecFrozenLightningModule"]
