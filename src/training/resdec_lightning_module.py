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
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import numpy as np
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

        # Encoder's own prediction_head is bypassed under ResDec-H3: we consume the
        # 'attended' subject embedding directly and feed it to self.head. Freeze the
        # prediction_head to avoid wasted optimizer state (verified: grad is always zero).
        if hasattr(self.encoder, "prediction_head"):
            for p in self.encoder.prediction_head.parameters():
                p.requires_grad_(False)

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
        # Phase 2 (Task 2.2): TabPFN residual base                            #
        # ------------------------------------------------------------------ #
        # If cfg.data.tabpfn_oof_dir / tabpfn_outer_dir are provided, load the
        # fold-specific cached TabPFN predictions and build subject_id -> (y, σ)
        # lookup dicts. The head then trains on residuals y - y_tabpfn_oof and
        # validates on composite prediction ŷ = y_tabpfn_outer + f̂_1.
        # When the paths are absent we fall back to plain MSE (Phase 1 behaviour).
        self.tabpfn_train_map: dict[str, tuple[float, float]] = {}
        self.tabpfn_val_map: dict[str, tuple[float, float]] = {}
        self._tabpfn_enabled = False
        data_cfg = config.get("data", {}) or {}
        oof_dir = data_cfg.get("tabpfn_oof_dir", None)
        outer_dir = data_cfg.get("tabpfn_outer_dir", None)
        fold = data_cfg.get("fold", None)
        if oof_dir is not None and outer_dir is not None:
            if fold is None:
                raise ValueError(
                    "TabPFN residual base requires cfg.data.fold to be set "
                    "so the correct fold's .npz files can be loaded."
                )
            self._load_tabpfn_caches(
                oof_dir=Path(str(oof_dir)),
                outer_dir=Path(str(outer_dir)),
                fold=int(fold),
            )
            self._tabpfn_enabled = True
            logger.info(
                "TabPFN residual base enabled (fold=%d): %d train-OOF subjects, "
                "%d outer-val subjects",
                int(fold), len(self.tabpfn_train_map), len(self.tabpfn_val_map),
            )

    # ------------------------------------------------------------------ #
    # TabPFN cache loading (Phase 2 Task 2.2)                             #
    # ------------------------------------------------------------------ #
    def _load_tabpfn_caches(
        self, oof_dir: Path, outer_dir: Path, fold: int,
    ) -> None:
        """Load cached TabPFN OOF + outer-fold predictions for this fold.

        OOF (training) file keys: subject_ids, y_true, y_tabpfn_oof, sigma_tabpfn_oof.
        Outer (validation) file keys: val_subject_ids, y_true, y_tabpfn, sigma_tabpfn.
        """
        oof_path = oof_dir / f"tabpfn_oof_fold{fold}.npz"
        outer_path = outer_dir / f"tabpfn_outer_fold{fold}.npz"
        if not oof_path.exists():
            raise FileNotFoundError(f"TabPFN OOF cache not found: {oof_path}")
        if not outer_path.exists():
            raise FileNotFoundError(f"TabPFN outer cache not found: {outer_path}")

        oof = np.load(oof_path, allow_pickle=True)
        outer = np.load(outer_path, allow_pickle=True)

        oof_sids = [str(s) for s in oof["subject_ids"]]
        self.tabpfn_train_map = {
            sid: (float(oof["y_tabpfn_oof"][i]), float(oof["sigma_tabpfn_oof"][i]))
            for i, sid in enumerate(oof_sids)
        }

        outer_sids = [str(s) for s in outer["val_subject_ids"]]
        self.tabpfn_val_map = {
            sid: (float(outer["y_tabpfn"][i]), float(outer["sigma_tabpfn"][i]))
            for i, sid in enumerate(outer_sids)
        }

    def _tabpfn_train_batch(
        self, subject_ids: list[str], device: torch.device, dtype: torch.dtype,
    ) -> torch.Tensor:
        """Gather TabPFN OOF predictions for training subjects in batch order."""
        try:
            values = [self.tabpfn_train_map[sid][0] for sid in subject_ids]
        except KeyError as exc:
            raise KeyError(
                f"Subject {exc.args[0]!r} is missing from the TabPFN OOF cache "
                f"(fold cache has {len(self.tabpfn_train_map)} subjects). "
                f"Check cfg.data.fold matches the DataModule fold_idx."
            ) from exc
        return torch.tensor(values, device=device, dtype=dtype)

    def _tabpfn_val_batch(
        self, subject_ids: list[str], device: torch.device, dtype: torch.dtype,
    ) -> torch.Tensor:
        """Gather TabPFN outer predictions for val subjects in batch order."""
        try:
            values = [self.tabpfn_val_map[sid][0] for sid in subject_ids]
        except KeyError as exc:
            raise KeyError(
                f"Val subject {exc.args[0]!r} is missing from the TabPFN outer "
                f"cache (fold cache has {len(self.tabpfn_val_map)} subjects). "
                f"Check cfg.data.fold matches the DataModule fold_idx."
            ) from exc
        return torch.tensor(values, device=device, dtype=dtype)

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
        pred = out["prediction"]  # [B] — head output
        cognition = batch["cognition"]
        if cognition.dim() == 2 and cognition.shape[-1] == 1:
            cognition = cognition.squeeze(-1)
        bs = cognition.shape[0]

        if self._tabpfn_enabled:
            # Phase 2: train on residual target y - y_tabpfn_oof.
            # Composite prediction = ŷ_tabpfn + f̂_1 (for monitoring).
            subject_ids = batch["subject_ids"]  # list[str] from collate_for_hgt
            y_tabpfn = self._tabpfn_train_batch(
                subject_ids, device=cognition.device, dtype=cognition.dtype,
            )
            residual_target = cognition - y_tabpfn
            loss = torch.nn.functional.mse_loss(pred, residual_target)
            composite = pred.detach() + y_tabpfn
            comp_mse = torch.nn.functional.mse_loss(composite, cognition)
            self.log("train/residual_mse", loss, prog_bar=True, batch_size=bs, sync_dist=True)
            self.log("train/composite_mse", comp_mse, prog_bar=False, batch_size=bs, sync_dist=True)
        else:
            # Phase 1 fallback: plain MSE against cognition.
            loss = torch.nn.functional.mse_loss(pred, cognition)
            self.log("train/mse", loss, prog_bar=True, batch_size=bs, sync_dist=True)

        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        out = self.forward(batch)
        pred = out["prediction"].detach()
        target = batch["cognition"].detach()
        if target.dim() == 2 and target.shape[-1] == 1:
            target = target.squeeze(-1)

        if self._tabpfn_enabled:
            # Phase 2: composite prediction ŷ = y_tabpfn_outer + f̂_1.
            subject_ids = batch["subject_ids"]
            y_tabpfn = self._tabpfn_val_batch(
                subject_ids, device=target.device, dtype=target.dtype,
            )
            pred = pred + y_tabpfn  # composite

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
