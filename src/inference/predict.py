"""
Inference module for running trained cognitive resilience model.

Provides batch inference with:
- Checkpoint loading (Lightning or raw PyTorch)
- Automatic device placement
- Attention weight extraction
- Output to Parquet (predictions) and HDF5 (attention tensors)

Usage:
    from src.inference.predict import Predictor

    predictor = Predictor.from_checkpoint("path/to/checkpoint.ckpt")
    results = predictor.predict(dataloader)
    predictor.save_predictions(results, output_dir)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.constants import CELL_TYPE_ORDER, N_REGIONS, REGION_ORDER
from src.models.full_model import CognitiveResilienceModel, build_model_from_config
from src.training.lightning_module import CognitiveResilienceLightningModule
from src.utils.io import save_attention_weights as _io_save_attention_weights

logger = logging.getLogger(__name__)


def _hgt_attention_to_cpu(hgt_attention: list[dict]) -> list[dict]:
    """Move HGT attention tensors to CPU.

    HGT attention is list[dict[tuple, Tensor]] (per-layer, per-edge-type).
    Moves to CPU to prevent GPU OOM during batch accumulation.
    """
    return [
        {
            edge_type: attn.cpu() if isinstance(attn, torch.Tensor) else attn
            for edge_type, attn in layer_dict.items()
        }
        for layer_dict in hgt_attention
    ]


@dataclass
class PredictionResult:
    """
    Container for prediction outputs.

    Attributes:
        subject_ids: List of subject identifiers
        mean: Predicted mean values [n_subjects, 1]
        std: Predicted uncertainty (std) [n_subjects, 1], None if deterministic head
        actual: Actual cognition values [n_subjects, 1], None if not provided
        pathology: Pathology features [n_subjects, 3]
        attention_weights: Pathology attention [n_subjects, n_heads, n_cell_types]
        gene_gate_weights: Static gene gate weights [n_cell_types, n_genes] (shared)
        hgt_attention: List of per-sample HGT attention dicts, None if not extracted
        pma_attention: List of per-cell-type PMA attention [n_cell_types][n_subjects, n_heads, n_seeds, max_cells]
        per_subject_pseudobulk: Per-subject pseudobulk averaged across regions [n_subjects, n_cell_types, n_genes]
        metadata: Dict of additional metadata (config, checkpoint path, etc.)
    """
    subject_ids: list[str]
    mean: np.ndarray
    std: np.ndarray | None
    actual: np.ndarray | None
    pathology: np.ndarray | None
    attention_weights: np.ndarray
    gene_gate_weights: np.ndarray
    hgt_attention: list[dict] | None = None
    pma_attention: list[np.ndarray] | None = None
    region_weights: np.ndarray | None = None
    region_attention: np.ndarray | None = None  # [n_subjects, n_regions]
    region_pseudobulk_mean: np.ndarray | None = None  # [n_regions, n_cell_types, n_genes]
    per_subject_pseudobulk: np.ndarray | None = None  # [n_subjects, n_cell_types, n_genes]
    cell_barcodes: list[list[list[str]]] | None = None  # [n_subjects][n_cell_types][barcodes]
    cell_counts: np.ndarray | None = None  # [n_subjects, n_cell_types] cell counts per type
    gene_names: list[str] | None = None
    cell_type_selection: np.ndarray | None = None  # [n_cell_types] selection weights
    epistemic_std: np.ndarray | None = None  # [N, 1] epistemic uncertainty
    aleatoric_std: np.ndarray | None = None  # [N, 1] aleatoric uncertainty
    embeddings: dict[str, np.ndarray] | None = None  # {name: array} branch/fused/attended
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_subjects(self) -> int:
        return len(self.subject_ids)

    @property
    def has_uncertainty(self) -> bool:
        return self.std is not None

    @property
    def has_actual(self) -> bool:
        return self.actual is not None


class Predictor:
    """
    Inference engine for cognitive resilience model.

    Handles checkpoint loading, batch inference, and output formatting.

    Example:
        >>> predictor = Predictor.from_checkpoint("checkpoint.ckpt", device="cuda")
        >>> results = predictor.predict(val_loader, extract_hgt_attention=True)
        >>> predictor.save_predictions(results, "outputs/")
    """

    def __init__(
        self,
        model: CognitiveResilienceModel,
        config: DictConfig,
        device: torch.device | str = "auto",
        checkpoint_path: str | None = None,
        guide=None,
    ):
        """
        Initialize predictor with model and config.

        Args:
            model: Trained CognitiveResilienceModel
            config: Model configuration
            device: Device for inference ("auto", "cuda", "cpu")
            checkpoint_path: Optional path to checkpoint (for metadata)
            guide: Optional Pyro guide for Bayesian posterior sampling
        """
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.guide = guide

        # Resolve device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Error handling policy for inference
        error_cfg = {}
        if hasattr(config, "error_handling") and hasattr(config.error_handling, "inference"):
            error_cfg = config.error_handling.inference
        self._missing_field_policy = error_cfg.get("missing_field", "skip") if error_cfg else "skip"

        self.model = model.to(self.device)
        self.model.eval()

        logger.info(f"Predictor initialized on {self.device}")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str = "auto",
        config: DictConfig | None = None,
    ) -> "Predictor":
        """
        Load predictor from Lightning checkpoint.

        Args:
            checkpoint_path: Path to .ckpt file
            device: Device for inference
            config: Optional config override (if not in checkpoint)

        Returns:
            Configured Predictor instance
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Extract config from checkpoint or use provided
        if config is None:
            if "full_config" in checkpoint:
                # Full config from ResilienceModelCheckpoint (v1.1+)
                config = OmegaConf.create(checkpoint["full_config"])
            elif "model_config" in checkpoint:
                # Legacy: model-only config from older checkpoints
                config = OmegaConf.create({"model": checkpoint["model_config"]})
                # Add training section with defaults for head type
                if "training" not in config:
                    config.training = OmegaConf.create({
                        "loss": {"type": "beta_nll", "beta": 0.5}
                    })
            elif "hyper_parameters" in checkpoint and "config" in checkpoint["hyper_parameters"]:
                config = checkpoint["hyper_parameters"]["config"]
            else:
                raise ValueError(
                    "No config found in checkpoint. Provide config explicitly."
                )

        # Build model from checkpoint config (shared factory ensures training/inference parity)
        model_cfg = config.model
        try:
            model = build_model_from_config(model_cfg)
            use_bayesian = model_cfg.head.type == "bayesian"
        except (AttributeError, KeyError) as e:
            raise ValueError(
                f"Checkpoint config missing required model parameter: {e}. "
                f"Available keys: {list(model_cfg.keys()) if hasattr(model_cfg, 'keys') else 'N/A'}. "
                f"Provide a complete config via the config= argument."
            ) from e

        # Load weights
        if "state_dict" in checkpoint:
            # Lightning checkpoint - strip 'model.' prefix.
            # For Bayesian models, the state_dict may also contain guide
            # parameters (e.g., 'guide.loc', 'guide.scale_unconstrained')
            # which are NOT part of the model — skip them here; they are
            # loaded separately below via guide_state_dict / param store.
            state_dict = {}
            for k, v in checkpoint["state_dict"].items():
                if k.startswith("model."):
                    state_dict[k[6:]] = v  # Remove 'model.' prefix
                # Skip non-model keys (guide params, etc.)
            model.load_state_dict(state_dict, strict=True)
        elif "model_state_dict" in checkpoint:
            # io.save_checkpoint format
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        else:
            # Raw PyTorch checkpoint (just tensors)
            model.load_state_dict(checkpoint, strict=True)

        logger.info("Model weights loaded successfully")

        # Load guide for Bayesian posterior sampling
        guide = None
        if use_bayesian and "guide_state_dict" in checkpoint:
            from pyro.infer.autoguide import AutoDiagonalNormal
            import pyro

            guide = AutoDiagonalNormal(model)

            # Prototype the guide so it creates loc/scale parameters.
            # AutoDiagonalNormal needs a forward pass through the model to
            # discover all sample sites before it can initialize parameters.
            # We run a dummy forward to trigger this initialization.
            try:
                n_ct = model_cfg.n_cell_types
                n_genes = model_cfg.n_genes
                n_regions = N_REGIONS
                n_pathology = model_cfg.pathology_attention.get("n_pathology_features", 3)
                dummy_kwargs = {
                    "region_pseudobulk": torch.zeros(1, n_regions, n_ct, n_genes),
                    "region_mask": torch.ones(1, n_regions, dtype=torch.bool),
                    "cells": torch.zeros(1, n_ct, 1, n_genes),
                    "cell_mask": torch.ones(1, n_ct, 1, dtype=torch.bool),
                    "pathology": torch.zeros(1, n_pathology),
                    "cognition": torch.zeros(1, 1),
                }
                guide(**dummy_kwargs)
            except Exception as e:
                logger.warning("Guide prototype forward failed: %s. Param store load may fail if guide params are uninitialized.", e)

            # Restore Pyro param store (primary mechanism — guide params live here)
            if "pyro_param_store" in checkpoint:
                # Resolve target device before restoring params
                if device == "auto":
                    target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                else:
                    target_device = torch.device(device)
                pyro.clear_param_store()
                for k, v in checkpoint["pyro_param_store"].items():
                    pyro.get_param_store().setdefault(k, v.to(target_device))

            # Also load guide state dict (secondary — for internal guide state)
            try:
                guide.load_state_dict(checkpoint["guide_state_dict"])
            except Exception as e:
                # Guide may not be prototyped yet; param store is sufficient
                logger.warning("Could not load guide state_dict; using param store only: %s", e)

        return cls(
            model=model,
            config=config,
            device=device,
            checkpoint_path=str(checkpoint_path),
            guide=guide,
        )

    @classmethod
    def from_lightning_module(
        cls,
        module: CognitiveResilienceLightningModule,
        device: str = "auto",
    ) -> "Predictor":
        """
        Create predictor from an existing Lightning module.

        Args:
            module: Trained Lightning module
            device: Device for inference

        Returns:
            Configured Predictor instance
        """
        return cls(
            model=module.model,
            config=module.config,
            device=device,
            guide=getattr(module, "guide", None),
        )

    def _move_batch_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move batch tensors to device."""
        from src.utils.device import move_batch_to_device
        return move_batch_to_device(batch, self.device)

    @torch.no_grad()
    def _predict_batch_bayesian(
        self,
        batch: dict[str, Any],
        num_samples: int = 100,
        return_hgt_attention: bool = False,
        return_pma_attention: bool = False,
        return_region_attention: bool = False,
        return_embeddings: bool = False,
    ) -> dict[str, Any]:
        """
        Bayesian prediction with posterior sampling.

        Uses guide.median() for point estimate + aleatoric std,
        then posterior sampling loop for epistemic std.

        Args:
            batch: Batch dict from DataLoader
            num_samples: Number of posterior samples for epistemic uncertainty
            return_hgt_attention: Whether to extract HGT attention weights
            return_pma_attention: Whether to extract PMA cell-level attention weights
            return_region_attention: Whether to extract per-subject region attention
            return_embeddings: Whether to extract branch/fused/attended embeddings

        Returns:
            Dict with predictions, attention weights, and uncertainty estimates
        """
        import pyro
        import pyro.poutine

        batch = self._move_batch_to_device(batch)

        model_kwargs = dict(
            region_pseudobulk=batch.get("region_pseudobulk"),
            region_mask=batch.get("region_mask"),
            pseudobulk=batch.get("pseudobulk"),
            edge_index_dict_list=batch.get("edge_index_dict_list"),
            edge_attr_dict_list=batch.get("edge_attr_dict_list"),
            cells=batch.get("cells"),
            cell_mask=batch.get("cell_mask"),
            cell_type_mask=batch.get("cell_type_mask"),
            pathology=batch.get("pathology"),
            return_hgt_attention=return_hgt_attention,
            return_pma_attention=return_pma_attention,
            return_region_attention=return_region_attention,
            return_embeddings=return_embeddings,
        )

        # Point estimate from posterior median
        median = self.guide.median()
        conditioned = pyro.poutine.condition(self.model, data=median)
        output_median = conditioned(**model_kwargs)

        # Collect posterior samples for epistemic uncertainty
        means = []
        for _ in range(num_samples):
            guide_trace = pyro.poutine.trace(self.guide).get_trace(**model_kwargs)
            conditioned_sample = pyro.poutine.replay(self.model, trace=guide_trace)
            with torch.no_grad():
                out = conditioned_sample(**model_kwargs)
            means.append(out["mean"].detach().clone())
            # Free Pyro trace objects and model output to prevent GPU memory accumulation
            del out, guide_trace, conditioned_sample

        means_stacked = torch.stack(means, dim=0)  # [num_samples, B, 1]
        epistemic_std = means_stacked.std(dim=0)    # [B, 1]

        result = {
            "mean": output_median["mean"].cpu().numpy(),
            "attention_weights": output_median["attention_weights"].cpu().numpy(),
        }

        # Aleatoric std from Bayesian head
        if "std" in output_median:
            result["aleatoric_std"] = output_median["std"].cpu().numpy()

        result["epistemic_std"] = epistemic_std.cpu().numpy()

        # Total uncertainty = sqrt(epistemic^2 + aleatoric^2)
        if "aleatoric_std" in result:
            result["std"] = np.sqrt(
                result["aleatoric_std"] ** 2 + result["epistemic_std"] ** 2
            )
        else:
            result["std"] = result["epistemic_std"]

        # Extract optional outputs from median prediction (same logic as predict_batch)
        if return_hgt_attention and "hgt_attention" in output_median:
            result["hgt_attention"] = _hgt_attention_to_cpu(output_median["hgt_attention"])

        if return_pma_attention and "pma_attention" in output_median:
            result["pma_attention"] = [
                attn.cpu().numpy() for attn in output_median["pma_attention"]
            ]

        if return_region_attention and "region_attention" in output_median:
            result["region_attention"] = output_median["region_attention"].cpu().numpy()

        if return_embeddings and "embeddings" in output_median:
            result["embeddings"] = {
                name: emb.cpu().numpy()
                for name, emb in output_median["embeddings"].items()
            }

        # Batch metadata
        if "subject_ids" in batch:
            result["subject_ids"] = batch["subject_ids"]
        if "cognition" in batch:
            result["actual"] = batch["cognition"].cpu().numpy()
        if "pathology" in batch:
            result["pathology"] = batch["pathology"].cpu().numpy()
        if "cell_barcodes" in batch:
            result["cell_barcodes"] = batch["cell_barcodes"]

        return result

    @torch.no_grad()
    def predict_batch(
        self,
        batch: dict[str, Any],
        extract_hgt_attention: bool = False,
        extract_pma_attention: bool = False,
        extract_region_attention: bool = False,
        extract_embeddings: bool = False,
    ) -> dict[str, Any]:
        """
        Run inference on a single batch.

        Args:
            batch: Batch dict from DataLoader
            extract_hgt_attention: Whether to extract HGT attention weights
            extract_pma_attention: Whether to extract PMA cell-level attention weights
            extract_region_attention: Whether to extract per-subject region attention
            extract_embeddings: Whether to extract branch/fused/attended embeddings

        Returns:
            Dict with predictions and attention weights
        """
        if self.guide is not None:
            num_samples = 100
            if hasattr(self.config, "inference"):
                num_samples = self.config.inference.get("num_posterior_samples", 100)
            return self._predict_batch_bayesian(
                batch, num_samples=num_samples,
                return_hgt_attention=extract_hgt_attention,
                return_pma_attention=extract_pma_attention,
                return_region_attention=extract_region_attention,
                return_embeddings=extract_embeddings,
            )

        batch = self._move_batch_to_device(batch)

        output = self.model(
            region_pseudobulk=batch.get("region_pseudobulk"),
            region_mask=batch.get("region_mask"),
            pseudobulk=batch.get("pseudobulk"),
            edge_index_dict_list=batch.get("edge_index_dict_list"),
            edge_attr_dict_list=batch.get("edge_attr_dict_list"),
            cells=batch.get("cells"),
            cell_mask=batch.get("cell_mask"),
            cell_type_mask=batch.get("cell_type_mask"),
            pathology=batch.get("pathology"),
            return_hgt_attention=extract_hgt_attention,
            return_pma_attention=extract_pma_attention,
            return_region_attention=extract_region_attention,
            return_embeddings=extract_embeddings,
        )

        result = {
            "mean": output["mean"].cpu().numpy(),
            "attention_weights": output["attention_weights"].cpu().numpy(),
        }

        if "std" in output:
            result["std"] = output["std"].cpu().numpy()

        if extract_hgt_attention and "hgt_attention" in output:
            result["hgt_attention"] = _hgt_attention_to_cpu(output["hgt_attention"])

        if extract_pma_attention and "pma_attention" in output:
            # Convert list of tensors to list of numpy arrays
            result["pma_attention"] = [
                attn.cpu().numpy() for attn in output["pma_attention"]
            ]

        if extract_region_attention and "region_attention" in output:
            result["region_attention"] = output["region_attention"].cpu().numpy()

        if extract_embeddings and "embeddings" in output:
            result["embeddings"] = {
                name: emb.cpu().numpy()
                for name, emb in output["embeddings"].items()
            }

        # Include metadata from batch
        if "subject_ids" in batch:
            result["subject_ids"] = batch["subject_ids"]
        if "cognition" in batch:
            result["actual"] = batch["cognition"].cpu().numpy()
        if "pathology" in batch:
            result["pathology"] = batch["pathology"].cpu().numpy()
        if "cell_barcodes" in batch:
            result["cell_barcodes"] = batch["cell_barcodes"]

        return result

    @torch.no_grad()
    def predict(
        self,
        dataloader: DataLoader,
        extract_hgt_attention: bool = False,
        extract_pma_attention: bool = False,
        extract_region_attention: bool = False,
        extract_embeddings: bool = False,
        show_progress: bool = True,
    ) -> PredictionResult:
        """
        Run inference on full dataset.

        Args:
            dataloader: DataLoader yielding batches
            extract_hgt_attention: Whether to extract HGT attention weights
            extract_pma_attention: Whether to extract PMA cell-level attention weights
            extract_region_attention: Whether to extract per-subject region attention
            extract_embeddings: Whether to extract branch/fused/attended embeddings
            show_progress: Whether to show progress bar

        Returns:
            PredictionResult with all predictions and attention
        """
        all_subject_ids = []
        all_mean = []
        all_std = []
        all_epistemic_std = []
        all_aleatoric_std = []
        all_actual = []
        all_pathology = []
        all_attention = []
        all_hgt_attention = [] if extract_hgt_attention else None
        # PMA attention: list of lists, outer = cell types, inner = batches
        all_pma_attention: list[list[np.ndarray]] | None = None
        if extract_pma_attention:
            all_pma_attention = [[] for _ in range(self.model.n_cell_types)]

        # Incremental region pseudobulk aggregation (avoid storing all raw data)
        region_pseudobulk_sum = None   # Running sum for mean computation
        region_pseudobulk_count = None # Per-region count of valid subjects
        region_pseudobulk_has_mask = False  # Whether any mask was seen
        all_region_mask = []  # Track which regions are valid per subject
        per_subject_pseudobulk_parts = []  # Per-subject mean across regions (small)
        all_region_attention = [] if extract_region_attention else None
        all_cell_barcodes = []
        all_cell_counts = []
        _embedding_names = ['pseudobulk', 'hgt', 'cell', 'fused', 'attended']
        all_embeddings: dict[str, list[np.ndarray]] | None = None
        if extract_embeddings:
            all_embeddings = {name: [] for name in _embedding_names}

        iterator: Iterator = dataloader
        if show_progress:
            iterator = tqdm(dataloader, desc="Predicting", unit="batch")

        for batch in iterator:
            try:
                result = self.predict_batch(
                    batch, extract_hgt_attention, extract_pma_attention,
                    extract_region_attention, extract_embeddings,
                )
            except (KeyError, ValueError) as e:
                if self._missing_field_policy == "skip":
                    batch_ids = batch.get("subject_ids", ["unknown"])
                    logger.warning("Skipping batch (input error %s): subjects=%s", e, batch_ids)
                    continue
                else:
                    raise

            all_mean.append(result["mean"])
            all_attention.append(result["attention_weights"])

            if "std" in result:
                all_std.append(result["std"])
            if "epistemic_std" in result:
                all_epistemic_std.append(result["epistemic_std"])
            if "aleatoric_std" in result:
                all_aleatoric_std.append(result["aleatoric_std"])
            if "actual" in result:
                all_actual.append(result["actual"])
            if "pathology" in result:
                all_pathology.append(result["pathology"])
            if "subject_ids" in result:
                all_subject_ids.extend(result["subject_ids"])
            if extract_hgt_attention and "hgt_attention" in result:
                all_hgt_attention.extend(result["hgt_attention"])
            if extract_pma_attention and "pma_attention" in result:
                # result["pma_attention"] is list of [B, n_heads, n_seeds, max_cells] per cell type
                for ct_idx, ct_attn in enumerate(result["pma_attention"]):
                    all_pma_attention[ct_idx].append(ct_attn)
            if extract_region_attention and "region_attention" in result:
                all_region_attention.append(result["region_attention"])
            if "cell_barcodes" in result and result["cell_barcodes"] is not None:
                all_cell_barcodes.extend(result["cell_barcodes"])
            # Use pre-computed cell counts from dataset (not clipped by max_cells_per_type)
            if "cell_counts" in batch:
                cc = batch["cell_counts"]
                if isinstance(cc, torch.Tensor):
                    cc = cc.cpu().numpy()
                all_cell_counts.append(cc)
            elif "cell_mask" in batch:
                # Fallback: derive from cell_mask (may undercount clipped types)
                cm = batch["cell_mask"]
                if isinstance(cm, torch.Tensor):
                    cm = cm.cpu()
                batch_cell_counts = cm.sum(dim=-1).numpy()  # [B, n_cell_types]
                all_cell_counts.append(batch_cell_counts)
            if extract_embeddings and "embeddings" in result:
                for name in _embedding_names:
                    if name in result["embeddings"]:
                        all_embeddings[name].append(result["embeddings"][name])

            # Incremental region pseudobulk aggregation (avoid storing all raw data)
            if "region_pseudobulk" in batch:
                rpb = batch["region_pseudobulk"]
                if isinstance(rpb, torch.Tensor):
                    rpb = rpb.cpu().numpy()
                # Capture region_mask to know which regions are valid per subject
                rmask = None
                if "region_mask" in batch:
                    rm = batch["region_mask"]
                    if isinstance(rm, torch.Tensor):
                        rm = rm.cpu().numpy()
                    rmask = rm
                    all_region_mask.append(rm)
                    region_pseudobulk_has_mask = True
                # Accumulate running sum/count for region_pseudobulk_mean
                if rmask is not None:
                    mask_expanded = rmask[:, :, np.newaxis, np.newaxis]
                    masked_rpb = np.where(mask_expanded, rpb, 0.0)
                    batch_sum = masked_rpb.sum(axis=0)
                    batch_count = mask_expanded.sum(axis=0)
                    # Per-subject pseudobulk: mean across valid regions
                    masked_nan = np.where(mask_expanded, rpb, np.nan)
                    per_subj = np.nanmean(masked_nan, axis=1)
                    per_subj = np.nan_to_num(per_subj, nan=0.0)
                    per_subject_pseudobulk_parts.append(per_subj)
                else:
                    batch_sum = rpb.sum(axis=0)
                    batch_count = rpb.shape[0]
                    per_subject_pseudobulk_parts.append(rpb.mean(axis=1))
                if region_pseudobulk_sum is None:
                    region_pseudobulk_sum = batch_sum
                    region_pseudobulk_count = batch_count
                else:
                    region_pseudobulk_sum += batch_sum
                    region_pseudobulk_count += batch_count
            elif "pseudobulk" in batch:
                # Single-region fallback: wrap pseudobulk as 1-region
                pb = batch["pseudobulk"]
                if isinstance(pb, torch.Tensor):
                    pb = pb.cpu().numpy()
                rpb = pb[:, np.newaxis, :, :]
                rmask = np.ones((pb.shape[0], 1), dtype=bool)
                all_region_mask.append(rmask)
                region_pseudobulk_has_mask = True
                mask_expanded = rmask[:, :, np.newaxis, np.newaxis]
                masked_rpb = np.where(mask_expanded, rpb, 0.0)
                batch_sum = masked_rpb.sum(axis=0)
                batch_count = mask_expanded.sum(axis=0)
                if region_pseudobulk_sum is None:
                    region_pseudobulk_sum = batch_sum
                    region_pseudobulk_count = batch_count
                else:
                    region_pseudobulk_sum += batch_sum
                    region_pseudobulk_count += batch_count
                # Per-subject pseudobulk: for single region, just the pseudobulk itself
                per_subject_pseudobulk_parts.append(pb)

        # Guard: if all batches were skipped, provide actionable error
        if not all_mean:
            raise RuntimeError(
                "All batches were skipped during inference. This likely means every "
                "batch failed input validation under missing_field_policy='skip'. "
                "Check your data for missing required fields (pseudobulk, region_pseudobulk, etc.)."
            )

        # Concatenate results
        mean = np.concatenate(all_mean, axis=0)
        attention_weights = np.concatenate(all_attention, axis=0)
        std = np.concatenate(all_std, axis=0) if all_std else None
        epistemic_std = np.concatenate(all_epistemic_std, axis=0) if all_epistemic_std else None
        aleatoric_std = np.concatenate(all_aleatoric_std, axis=0) if all_aleatoric_std else None
        actual = np.concatenate(all_actual, axis=0) if all_actual else None
        pathology = np.concatenate(all_pathology, axis=0) if all_pathology else None

        # Get static gene gate weights
        gene_gate_weights = self.model.pseudobulk_encoder.gene_gate.get_gate_weights()
        gene_gate_weights = gene_gate_weights.cpu().numpy()

        # Get cell type selection weights
        cell_type_selection = self.model.cell_transformer.get_selection_weights()
        cell_type_selection = cell_type_selection.cpu().numpy()

        # Get region importance weights
        region_importance = self.model.get_region_importance()
        region_weights = np.array(list(region_importance.values()), dtype=np.float32)

        # Extract gene names from dataset if available
        gene_names = None
        dataset = getattr(dataloader, "dataset", None)
        if dataset is not None and hasattr(dataset, "get_gene_names"):
            gene_names = dataset.get_gene_names()

        # Extract per-subject metadata (region, split) from dataset if available
        subject_metadata = {}
        if dataset is not None and hasattr(dataset, "metadata"):
            meta_df = dataset.metadata  # Already indexed by subject_id
            for col in ["region", "split"]:
                if col in meta_df.columns:
                    aligned = []
                    for sid in all_subject_ids:
                        if sid in meta_df.index:
                            val = meta_df.loc[sid, col]
                            aligned.append(val if pd.notna(val) else None)
                        else:
                            aligned.append(None)
                    subject_metadata[col] = aligned

        # Concatenate cell counts
        cell_counts = np.concatenate(all_cell_counts, axis=0) if all_cell_counts else None

        # Concatenate PMA attention per cell type
        pma_attention = None
        if extract_pma_attention and all_pma_attention:
            pma_attention = [
                np.concatenate(ct_batches, axis=0) for ct_batches in all_pma_attention
            ]

        # Concatenate region attention per subject
        region_attention = None
        if extract_region_attention and all_region_attention:
            region_attention = np.concatenate(all_region_attention, axis=0)  # [N, n_regions]

        # Compute mean region_pseudobulk from incrementally accumulated sum/count
        region_pseudobulk_mean = None
        if region_pseudobulk_sum is not None:
            if region_pseudobulk_has_mask:
                # Divide accumulated sum by count; use 0 where no valid subjects existed
                region_pseudobulk_mean = np.where(
                    region_pseudobulk_count > 0,
                    region_pseudobulk_sum / region_pseudobulk_count,
                    0.0,
                )
            else:
                region_pseudobulk_mean = region_pseudobulk_sum / max(region_pseudobulk_count, 1)

        # Per-subject pseudobulk was computed incrementally in the batch loop
        per_subject_pseudobulk = None
        if per_subject_pseudobulk_parts:
            per_subject_pseudobulk = np.concatenate(per_subject_pseudobulk_parts, axis=0)

        # Concatenate embeddings
        embeddings_dict = None
        if extract_embeddings and all_embeddings:
            embeddings_dict = {}
            for name in _embedding_names:
                if all_embeddings[name]:
                    embeddings_dict[name] = np.concatenate(all_embeddings[name], axis=0)

        # Build metadata
        metadata = {
            "checkpoint_path": self.checkpoint_path,
            "device": str(self.device),
            "n_subjects": len(all_subject_ids),
            "has_uncertainty": std is not None,
            "extracted_hgt_attention": extract_hgt_attention,
            "extracted_pma_attention": extract_pma_attention,
        }
        if self.config:
            metadata["model_config"] = OmegaConf.to_container(self.config.model, resolve=True)
            # Store pathology columns if available in data config
            if hasattr(self.config, "data") and hasattr(self.config.data, "pathology_columns"):
                metadata["pathology_columns"] = list(self.config.data.pathology_columns)
            # Pass region names through metadata for save_predictions
            if hasattr(self.config, "data") and hasattr(self.config.data, "region_names"):
                metadata["region_names"] = list(self.config.data.region_names)
        if subject_metadata:
            metadata["subject_metadata"] = subject_metadata

        return PredictionResult(
            subject_ids=all_subject_ids,
            mean=mean,
            std=std,
            actual=actual,
            pathology=pathology,
            attention_weights=attention_weights,
            gene_gate_weights=gene_gate_weights,
            cell_type_selection=cell_type_selection,
            hgt_attention=all_hgt_attention,
            pma_attention=pma_attention,
            region_weights=region_weights,
            region_attention=region_attention,
            region_pseudobulk_mean=region_pseudobulk_mean,
            per_subject_pseudobulk=per_subject_pseudobulk,
            cell_barcodes=all_cell_barcodes if all_cell_barcodes else None,
            cell_counts=cell_counts,
            gene_names=gene_names,
            epistemic_std=epistemic_std,
            aleatoric_std=aleatoric_std,
            embeddings=embeddings_dict,
            metadata=metadata,
        )

    def save_predictions(
        self,
        results: PredictionResult,
        output_dir: str | Path,
        save_parquet: bool = True,
        save_csv: bool = True,
        save_hdf5: bool = True,
    ) -> dict[str, Path]:
        """
        Save prediction results to files.

        Args:
            results: PredictionResult from predict()
            output_dir: Output directory
            save_parquet: Save predictions as Parquet (recommended)
            save_csv: Save predictions as CSV (human-readable)
            save_hdf5: Save attention tensors as HDF5

        Returns:
            Dict mapping output type to file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        # Build predictions DataFrame
        df_data = {
            "subject_id": results.subject_ids,
            "predicted_mean": results.mean.flatten(),
        }

        if results.has_uncertainty:
            df_data["predicted_std"] = results.std.flatten()

        if results.has_actual:
            df_data["actual"] = results.actual.flatten()
            df_data["residual"] = results.actual.flatten() - results.mean.flatten()

        if results.pathology is not None:
            # Default pathology column names (can be customized via metadata)
            default_pathology_names = ["gpath", "amylsqrt", "tangsqrt"]
            pathology_names = results.metadata.get("pathology_columns", default_pathology_names)

            # Handle both 1D and 2D pathology arrays with bounds checking
            n_pathology = results.pathology.shape[-1] if results.pathology.ndim > 1 else 1
            for i in range(min(n_pathology, len(pathology_names))):
                col_name = pathology_names[i]
                if results.pathology.ndim > 1:
                    df_data[col_name] = results.pathology[:, i]
                else:
                    df_data[col_name] = results.pathology

        # Add per-subject metadata columns (region, split) if available
        subject_meta = results.metadata.get("subject_metadata", {})
        for col in ["region", "split"]:
            if col in subject_meta and len(subject_meta[col]) == len(results.subject_ids):
                df_data[col] = subject_meta[col]

        df = pd.DataFrame(df_data)

        # Save Parquet (primary format)
        if save_parquet:
            parquet_path = output_dir / "predictions.parquet"
            df.to_parquet(parquet_path, index=False)
            saved_files["parquet"] = parquet_path
            logger.info(f"Saved predictions to {parquet_path}")

        # Save CSV (human-readable)
        if save_csv:
            csv_path = output_dir / "predictions.csv"
            df.to_csv(csv_path, index=False, float_format="%.6f")
            saved_files["csv"] = csv_path
            logger.info(f"Saved predictions to {csv_path}")

        # Save HDF5 (attention tensors)
        if save_hdf5:
            h5_path = output_dir / "attention_weights.h5"
            self._save_attention_hdf5(results, h5_path)
            saved_files["hdf5"] = h5_path
            logger.info(f"Saved attention weights to {h5_path}")

        return saved_files

    def _save_attention_hdf5(
        self,
        results: PredictionResult,
        path: Path,
    ) -> None:
        """Save attention weights to HDF5 file via canonical io.save_attention_weights()."""
        # Aggregate HGT attention if available
        hgt_agg = None
        if results.hgt_attention is not None and len(results.hgt_attention) > 0:
            from src.inference.extract_attention import aggregate_hgt_attention
            hgt_agg = aggregate_hgt_attention(results.hgt_attention, include_per_sample=True)

        _io_save_attention_weights(
            path=path,
            gene_gate=results.gene_gate_weights,
            cell_type_selection=results.cell_type_selection,
            pathology_attention=results.attention_weights,
            region_weights=results.region_weights,
            region_attention=results.region_attention,
            region_pseudobulk=results.region_pseudobulk_mean,
            per_subject_pseudobulk=results.per_subject_pseudobulk,
            hgt_attention=hgt_agg,
            pma_attention=results.pma_attention,
            cell_barcodes=results.cell_barcodes,
            cell_counts=results.cell_counts,
            subject_ids=results.subject_ids,
            cell_type_names=list(CELL_TYPE_ORDER),
            gene_names=results.gene_names,
            region_names=(
                results.metadata.get("region_names", list(REGION_ORDER))
                if results.region_pseudobulk_mean is not None
                else None
            ),
            embeddings=results.embeddings,
            metadata={
                "n_subjects": results.n_subjects,
                "checkpoint_path": results.metadata.get("checkpoint_path", ""),
            },
        )


def predict_from_checkpoint(
    checkpoint_path: str | Path,
    dataloader: DataLoader,
    output_dir: str | Path,
    device: str = "auto",
    extract_hgt_attention: bool = True,
    extract_pma_attention: bool = True,
    extract_region_attention: bool = True,
    extract_embeddings: bool = True,
) -> PredictionResult:
    """
    Convenience function: load checkpoint, run inference, save results.

    Args:
        checkpoint_path: Path to model checkpoint
        dataloader: DataLoader for inference
        output_dir: Directory for output files
        device: Device for inference
        extract_hgt_attention: Whether to extract HGT attention
        extract_pma_attention: Whether to extract PMA cell-level attention
        extract_region_attention: Whether to extract per-subject region attention
        extract_embeddings: Whether to extract branch/fused/attended embeddings

    Returns:
        PredictionResult with all outputs
    """
    predictor = Predictor.from_checkpoint(checkpoint_path, device=device)
    results = predictor.predict(
        dataloader,
        extract_hgt_attention=extract_hgt_attention,
        extract_pma_attention=extract_pma_attention,
        extract_region_attention=extract_region_attention,
        extract_embeddings=extract_embeddings,
    )
    predictor.save_predictions(results, output_dir)
    return results



