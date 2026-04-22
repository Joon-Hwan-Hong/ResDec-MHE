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
        # Phase-3 H3 extension knobs — ablation-friendly defaults match plan spec.
        k_tabm = int(resdec_cfg.get("k_tabm", 8))
        aux_lambdas = list(resdec_cfg.get("aux_lambdas", [1.0, 1.0, 1.0]))
        if len(aux_lambdas) != 3:
            raise ValueError(
                f"resdec_head.aux_lambdas must have exactly 3 entries "
                f"(stage_1/2/3); got {len(aux_lambdas)}: {aux_lambdas}"
            )
        self._aux_lambdas: tuple[float, float, float] = (
            float(aux_lambdas[0]), float(aux_lambdas[1]), float(aux_lambdas[2]),
        )
        self._use_sigma_weighting = bool(resdec_cfg.get("use_sigma_weighting", True))
        # Numerical floor for aug-U: w(σ) = 1 / (σ² + eps).
        self._sigma_eps = float(resdec_cfg.get("sigma_eps", 1e-6))

        self.head = ResDecH3Head(
            d_subject=d_subject,
            d_metadata=d_metadata,
            n_heads=n_heads,
            n_hc_streams=n_hc_streams,
            lambda_init=lambda_init,
            k_tabm=k_tabm,
        )
        self._d_metadata = d_metadata

        # Validation accumulators for epoch-level R² / MSE / full metric suite.
        self._val_preds: list[torch.Tensor] = []
        self._val_targets: list[torch.Tensor] = []
        self._val_subject_ids: list[str] = []

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather TabPFN OOF ``(y, σ)`` for training subjects in batch order.

        Returns ``(y_tabpfn [B], sigma_tabpfn [B])``. σ is needed for the
        aug-U weighting ``w(σ) = 1 / (σ² + eps)`` applied to aux_2/aux_3 losses.
        """
        try:
            y_vals = [self.tabpfn_train_map[sid][0] for sid in subject_ids]
            sigma_vals = [self.tabpfn_train_map[sid][1] for sid in subject_ids]
        except KeyError as exc:
            raise KeyError(
                f"Subject {exc.args[0]!r} is missing from the TabPFN OOF cache "
                f"(fold cache has {len(self.tabpfn_train_map)} subjects). "
                f"Check cfg.data.fold matches the DataModule fold_idx."
            ) from exc
        y = torch.tensor(y_vals, device=device, dtype=dtype)
        sigma = torch.tensor(sigma_vals, device=device, dtype=dtype)
        return y, sigma

    def _tabpfn_val_batch(
        self, subject_ids: list[str], device: torch.device, dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather TabPFN outer ``(y, σ)`` for val subjects in batch order."""
        try:
            y_vals = [self.tabpfn_val_map[sid][0] for sid in subject_ids]
            sigma_vals = [self.tabpfn_val_map[sid][1] for sid in subject_ids]
        except KeyError as exc:
            raise KeyError(
                f"Val subject {exc.args[0]!r} is missing from the TabPFN outer "
                f"cache (fold cache has {len(self.tabpfn_val_map)} subjects). "
                f"Check cfg.data.fold matches the DataModule fold_idx."
            ) from exc
        y = torch.tensor(y_vals, device=device, dtype=dtype)
        sigma = torch.tensor(sigma_vals, device=device, dtype=dtype)
        return y, sigma

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
            prediction: [B] composer output = ``f̂_1 + f̂_2 + f̂_3`` (residual sum,
                        not yet composite with ŷ_tabpfn — that happens in
                        validation_step for the final composite prediction)
            stage_1/2/3: [B] per-stage scalars for aux-loss construction
            latent_1/2/3: [B, d_subject] per-stage pre-readout latents
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
            "stage_1": head_out["stage_1"],
            "stage_2": head_out["stage_2"],
            "stage_3": head_out["stage_3"],
            "latent_1": head_out["latent_1"],
            "latent_2": head_out["latent_2"],
            "latent_3": head_out["latent_3"],
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
        pred = out["prediction"]  # [B] — f̂_1 + f̂_2 + f̂_3 (residual sum)
        stage_1 = out["stage_1"]
        stage_2 = out["stage_2"]
        stage_3 = out["stage_3"]
        cognition = batch["cognition"]
        if cognition.dim() == 2 and cognition.shape[-1] == 1:
            cognition = cognition.squeeze(-1)
        bs = cognition.shape[0]

        if self._tabpfn_enabled:
            # ============================================================= #
            # Phase 3 — H3 boosting with aug-U uncertainty-weighted aux loss.#
            # ============================================================= #
            #
            # L = L_main
            #   + λ_1 · MSE(f̂_1, y − ŷ_tabpfn)
            #   + λ_2 · MSE(f̂_2, y − ŷ_tabpfn − f̂_1.detach()) · w(σ)
            #   + λ_3 · MSE(f̂_3, y − ŷ_tabpfn − f̂_1.detach() − f̂_2.detach()) · w(σ)
            #
            # L_main = MSE(f̂_1+f̂_2+f̂_3, y − ŷ_tabpfn) — the composite residual
            # target. This is equivalent in gradient to MSE of composite prediction
            # ŷ_tabpfn + (f̂_1+f̂_2+f̂_3) against y, since y_tabpfn is a constant.
            # w(σ) = 1 / (σ² + eps) is the aug-U per-subject weighting.
            #
            # All four loss terms share a single backward pass — this is the
            # "single backward" guarantee in docs/plans/2026-04-21 §Phase 3.

            subject_ids = batch["subject_ids"]  # list[str] from collate_for_hgt
            y_tabpfn, sigma_tabpfn = self._tabpfn_train_batch(
                subject_ids, device=cognition.device, dtype=cognition.dtype,
            )
            residual_target = cognition - y_tabpfn  # [B]

            # Main loss: composite residual MSE (no per-sample weighting).
            L_main = torch.nn.functional.mse_loss(pred, residual_target)

            # Stage-1 aux: unweighted residual-target MSE.
            L_aux1 = torch.nn.functional.mse_loss(stage_1, residual_target)

            # Stage-2/3 aux: aug-U per-subject weighting w(σ) = 1/(σ²+eps),
            # mean-reduced by hand so the weighting applies before reduction.
            target_aux2 = residual_target - stage_1.detach()
            target_aux3 = residual_target - stage_1.detach() - stage_2.detach()

            if self._use_sigma_weighting:
                w = 1.0 / (sigma_tabpfn * sigma_tabpfn + self._sigma_eps)  # [B]
                # Mean of weighted per-subject squared error.
                L_aux2 = (w * (stage_2 - target_aux2).pow(2)).mean()
                L_aux3 = (w * (stage_3 - target_aux3).pow(2)).mean()
            else:
                # Ablation knob: uniform weighting.
                L_aux2 = torch.nn.functional.mse_loss(stage_2, target_aux2)
                L_aux3 = torch.nn.functional.mse_loss(stage_3, target_aux3)

            lam1, lam2, lam3 = self._aux_lambdas
            loss = L_main + lam1 * L_aux1 + lam2 * L_aux2 + lam3 * L_aux3

            # Composite MSE (detached, for monitoring parity with Phase 2).
            composite = pred.detach() + y_tabpfn
            comp_mse = torch.nn.functional.mse_loss(composite, cognition)

            self.log("train/loss", loss, prog_bar=True, batch_size=bs, sync_dist=True)
            self.log("train/L_main", L_main, prog_bar=False, batch_size=bs, sync_dist=True)
            self.log("train/L_aux1", L_aux1, prog_bar=False, batch_size=bs, sync_dist=True)
            self.log("train/L_aux2", L_aux2, prog_bar=False, batch_size=bs, sync_dist=True)
            self.log("train/L_aux3", L_aux3, prog_bar=False, batch_size=bs, sync_dist=True)
            self.log("train/residual_mse", L_main, prog_bar=False, batch_size=bs, sync_dist=True)
            self.log("train/composite_mse", comp_mse, prog_bar=False, batch_size=bs, sync_dist=True)
        else:
            # Phase 1 fallback: plain MSE against cognition (no TabPFN cache).
            loss = torch.nn.functional.mse_loss(pred, cognition)
            self.log("train/mse", loss, prog_bar=True, batch_size=bs, sync_dist=True)

        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        out = self.forward(batch)
        # Phase 3: pred["prediction"] is the sum f̂_1+f̂_2+f̂_3. Composite prediction
        # at val time is ŷ_tabpfn + f̂_1 + f̂_2 + f̂_3.
        pred = out["prediction"].detach()
        target = batch["cognition"].detach()
        if target.dim() == 2 and target.shape[-1] == 1:
            target = target.squeeze(-1)

        if self._tabpfn_enabled:
            subject_ids = batch["subject_ids"]
            y_tabpfn, _sigma = self._tabpfn_val_batch(
                subject_ids, device=target.device, dtype=target.dtype,
            )
            pred = pred + y_tabpfn  # composite

        loss = torch.nn.functional.mse_loss(pred, target)
        bs = target.shape[0]
        self.log("val/mse_batch", loss, prog_bar=False, batch_size=bs, sync_dist=True)

        self._val_preds.append(pred.cpu())
        self._val_targets.append(target.cpu())
        if "subject_ids" in batch:
            self._val_subject_ids.extend(list(batch["subject_ids"]))

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        preds = torch.cat(self._val_preds, dim=0)
        targets = torch.cat(self._val_targets, dim=0)

        # Epoch MSE over full val set (not batch-mean, which is sample-size biased).
        mse = torch.mean((preds - targets) ** 2).item()
        mae = torch.mean(torch.abs(preds - targets)).item()
        rmse = math.sqrt(mse)

        # R² = 1 - SS_res / SS_tot. If SS_tot == 0 (constant targets), R² is
        # undefined — log NaN so it's visible rather than silently 0.
        ss_res = torch.sum((targets - preds) ** 2)
        ss_tot = torch.sum((targets - targets.mean()) ** 2)
        if ss_tot > 0:
            r2 = (1.0 - (ss_res / ss_tot)).item()
        else:
            r2 = float("nan")

        # Pearson r (linear correlation) + Spearman ρ (rank correlation)
        # via numpy-on-CPU — simpler than torch-corrcoef broadcasting.
        import numpy as _np
        p_np = preds.detach().numpy()
        t_np = targets.detach().numpy()
        if p_np.std() > 0 and t_np.std() > 0:
            pearson_r = float(_np.corrcoef(p_np, t_np)[0, 1])
            # Spearman via rank-corr on np.argsort orderings
            from scipy.stats import spearmanr as _spearmanr  # noqa: PLC0415
            spearman_rho = float(_spearmanr(p_np, t_np).correlation)
        else:
            pearson_r = float("nan")
            spearman_rho = float("nan")

        self.log("val/mse", mse, prog_bar=True, sync_dist=True)
        self.log("val/mae", mae, prog_bar=False, sync_dist=True)
        self.log("val/rmse", rmse, prog_bar=False, sync_dist=True)
        if not math.isnan(r2):
            self.log("val/r2", r2, prog_bar=True, sync_dist=True)
        if not math.isnan(pearson_r):
            self.log("val/pearson_r", pearson_r, prog_bar=False, sync_dist=True)
        if not math.isnan(spearman_rho):
            self.log("val/spearman_rho", spearman_rho, prog_bar=False, sync_dist=True)

        # Persist per-subject predictions for downstream full-metric recomputation
        # and interpretability. Overwritten each val epoch; the final .npz reflects
        # the final epoch. Best-epoch predictions can be recovered via ModelCheckpoint.
        if self._val_subject_ids and len(self._val_subject_ids) == len(p_np):
            try:
                log_dir = None
                try:
                    log_dir = Path(self.trainer.log_dir) if self.trainer.log_dir else None
                except Exception:
                    log_dir = None
                if log_dir is None:
                    log_dir = Path("outputs/redesign/val_predictions")
                log_dir.mkdir(parents=True, exist_ok=True)
                _np.savez(
                    log_dir / "val_predictions_final.npz",
                    subject_ids=_np.array(self._val_subject_ids, dtype=object),
                    predictions=p_np.astype(_np.float32),
                    targets=t_np.astype(_np.float32),
                    epoch=int(self.current_epoch),
                    mse=mse, mae=mae, rmse=rmse,
                    r2=r2, pearson_r=pearson_r, spearman_rho=spearman_rho,
                )
            except Exception as e:  # do not let IO crash training
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "Failed to persist val predictions: %s", e,
                )

        self._val_preds.clear()
        self._val_targets.clear()
        self._val_subject_ids.clear()

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
