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
from typing import Any, Iterator, Sequence

import h5py
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.constants import CELL_TYPE_ORDER
from src.models.full_model import CognitiveResilienceModel
from src.training.lightning_module import CognitiveResilienceLightningModule

logger = logging.getLogger(__name__)


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
        metadata: Dict of additional metadata (config, checkpoint path, etc.)
    """
    subject_ids: list[str]
    mean: np.ndarray
    std: np.ndarray | None
    actual: np.ndarray | None
    pathology: np.ndarray
    attention_weights: np.ndarray
    gene_gate_weights: np.ndarray
    hgt_attention: list[dict] | None = None
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
    ):
        """
        Initialize predictor with model and config.

        Args:
            model: Trained CognitiveResilienceModel
            config: Model configuration
            device: Device for inference ("auto", "cuda", "cpu")
            checkpoint_path: Optional path to checkpoint (for metadata)
        """
        self.config = config
        self.checkpoint_path = checkpoint_path

        # Resolve device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

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
            if "model_config" in checkpoint:
                # From ResilienceModelCheckpoint
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

        # Build model
        model_cfg = config.model
        use_bayesian = model_cfg.head.type == "bayesian"

        model = CognitiveResilienceModel(
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
            n_pathology_features=model_cfg.pathology_attention.get("n_pathology_features", 3),
            n_pma_seeds=model_cfg.set_transformer.get("n_pma_seeds", 1),
            mlp_hidden=list(model_cfg.pseudobulk.mlp_hidden) if model_cfg.get("pseudobulk", {}).get("mlp_hidden") else None,
            use_layer_norm=model_cfg.get("pseudobulk", {}).get("use_layer_norm", True),
        )

        # Load weights
        if "state_dict" in checkpoint:
            # Lightning checkpoint - strip 'model.' prefix
            state_dict = {}
            for k, v in checkpoint["state_dict"].items():
                if k.startswith("model."):
                    state_dict[k[6:]] = v  # Remove 'model.' prefix
                else:
                    state_dict[k] = v
            model.load_state_dict(state_dict, strict=True)
        else:
            # Raw PyTorch checkpoint
            model.load_state_dict(checkpoint, strict=True)

        logger.info("Model weights loaded successfully")

        return cls(
            model=model,
            config=config,
            device=device,
            checkpoint_path=str(checkpoint_path),
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
        )

    def _move_batch_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move batch tensors to device."""
        moved = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                moved[k] = v.to(self.device)
            elif isinstance(v, list) and len(v) > 0:
                # Handle list of dicts (edge_index_dict_list, edge_attr_dict_list)
                if isinstance(v[0], dict):
                    moved[k] = [
                        {kk: vv.to(self.device) if isinstance(vv, torch.Tensor) else vv
                         for kk, vv in d.items()}
                        for d in v
                    ]
                else:
                    moved[k] = v
            else:
                moved[k] = v
        return moved

    @torch.no_grad()
    def predict_batch(
        self,
        batch: dict[str, Any],
        extract_hgt_attention: bool = False,
    ) -> dict[str, Any]:
        """
        Run inference on a single batch.

        Args:
            batch: Batch dict from DataLoader
            extract_hgt_attention: Whether to extract HGT attention weights

        Returns:
            Dict with predictions and attention weights
        """
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
        )

        result = {
            "mean": output["mean"].cpu().numpy(),
            "attention_weights": output["attention_weights"].cpu().numpy(),
        }

        if "std" in output:
            result["std"] = output["std"].cpu().numpy()

        if extract_hgt_attention and "hgt_attention" in output:
            result["hgt_attention"] = output["hgt_attention"]

        # Include metadata from batch
        if "subject_id" in batch:
            result["subject_ids"] = batch["subject_id"]
        if "cognition" in batch:
            result["actual"] = batch["cognition"].cpu().numpy()
        if "pathology" in batch:
            result["pathology"] = batch["pathology"].cpu().numpy()

        return result

    @torch.no_grad()
    def predict(
        self,
        dataloader: DataLoader,
        extract_hgt_attention: bool = False,
        show_progress: bool = True,
    ) -> PredictionResult:
        """
        Run inference on full dataset.

        Args:
            dataloader: DataLoader yielding batches
            extract_hgt_attention: Whether to extract HGT attention weights
            show_progress: Whether to show progress bar

        Returns:
            PredictionResult with all predictions and attention
        """
        all_subject_ids = []
        all_mean = []
        all_std = []
        all_actual = []
        all_pathology = []
        all_attention = []
        all_hgt_attention = [] if extract_hgt_attention else None

        iterator: Iterator = dataloader
        if show_progress:
            iterator = tqdm(dataloader, desc="Predicting", unit="batch")

        for batch in iterator:
            result = self.predict_batch(batch, extract_hgt_attention)

            all_mean.append(result["mean"])
            all_attention.append(result["attention_weights"])

            if "std" in result:
                all_std.append(result["std"])
            if "actual" in result:
                all_actual.append(result["actual"])
            if "pathology" in result:
                all_pathology.append(result["pathology"])
            if "subject_ids" in result:
                all_subject_ids.extend(result["subject_ids"])
            if extract_hgt_attention and "hgt_attention" in result:
                all_hgt_attention.extend(result["hgt_attention"])

        # Concatenate results
        mean = np.concatenate(all_mean, axis=0)
        attention_weights = np.concatenate(all_attention, axis=0)
        std = np.concatenate(all_std, axis=0) if all_std else None
        actual = np.concatenate(all_actual, axis=0) if all_actual else None
        pathology = np.concatenate(all_pathology, axis=0) if all_pathology else None

        # Get static gene gate weights
        gene_gate_weights = self.model.pseudobulk_encoder.gene_gate.get_gate_weights()
        gene_gate_weights = gene_gate_weights.cpu().numpy()

        # Build metadata
        metadata = {
            "checkpoint_path": self.checkpoint_path,
            "device": str(self.device),
            "n_subjects": len(all_subject_ids),
            "has_uncertainty": std is not None,
            "extracted_hgt_attention": extract_hgt_attention,
        }
        if self.config:
            metadata["model_config"] = OmegaConf.to_container(self.config.model, resolve=True)

        return PredictionResult(
            subject_ids=all_subject_ids,
            mean=mean,
            std=std,
            actual=actual,
            pathology=pathology,
            attention_weights=attention_weights,
            gene_gate_weights=gene_gate_weights,
            hgt_attention=all_hgt_attention,
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
            df_data["gpath"] = results.pathology[:, 0]
            df_data["amylsqrt"] = results.pathology[:, 1]
            df_data["tangsqrt"] = results.pathology[:, 2]

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
        """Save attention weights to HDF5 file."""
        with h5py.File(path, "w") as f:
            # Schema version for future compatibility
            f.attrs["schema_version"] = "1.0"
            f.attrs["n_subjects"] = results.n_subjects
            f.attrs["checkpoint_path"] = results.metadata.get("checkpoint_path", "")

            # Gene gate weights (static, shared across subjects)
            # Shape: [n_cell_types, n_genes]
            f.create_dataset(
                "gene_gate_weights",
                data=results.gene_gate_weights,
                compression="gzip",
                compression_opts=4,
            )
            f["gene_gate_weights"].attrs["shape"] = "[n_cell_types, n_genes]"
            f["gene_gate_weights"].attrs["description"] = "Learned gene attention weights per cell type"

            # Pathology attention weights (per subject)
            # Shape: [n_subjects, n_heads, n_cell_types]
            f.create_dataset(
                "pathology_attention",
                data=results.attention_weights,
                compression="gzip",
                compression_opts=4,
            )
            f["pathology_attention"].attrs["shape"] = "[n_subjects, n_heads, n_cell_types]"
            f["pathology_attention"].attrs["description"] = "Cell type attention conditioned on pathology"

            # Subject IDs (as fixed-length strings)
            subject_ids_encoded = np.array(results.subject_ids, dtype="S64")
            f.create_dataset("subject_ids", data=subject_ids_encoded)

            # Cell type names for reference
            cell_types_encoded = np.array(list(CELL_TYPE_ORDER), dtype="S64")
            f.create_dataset("cell_type_names", data=cell_types_encoded)

            # HGT attention (if extracted)
            if results.hgt_attention is not None:
                hgt_group = f.create_group("hgt_attention")
                hgt_group.attrs["description"] = "Per-layer, per-head HGT attention (edge weights)"
                # Note: HGT attention structure is complex (per-sample, per-edge-type dicts)
                # For now, we just flag that it's available - full extraction in extract_attention.py
                hgt_group.attrs["available"] = True
                hgt_group.attrs["n_samples"] = len(results.hgt_attention)


def predict_from_checkpoint(
    checkpoint_path: str | Path,
    dataloader: DataLoader,
    output_dir: str | Path,
    device: str = "auto",
    extract_hgt_attention: bool = False,
) -> PredictionResult:
    """
    Convenience function: load checkpoint, run inference, save results.

    Args:
        checkpoint_path: Path to model checkpoint
        dataloader: DataLoader for inference
        output_dir: Directory for output files
        device: Device for inference
        extract_hgt_attention: Whether to extract HGT attention

    Returns:
        PredictionResult with all outputs
    """
    predictor = Predictor.from_checkpoint(checkpoint_path, device=device)
    results = predictor.predict(dataloader, extract_hgt_attention=extract_hgt_attention)
    predictor.save_predictions(results, output_dir)
    return results


def save_predictions_parquet(
    results: PredictionResult,
    path: str | Path,
) -> None:
    """
    Save predictions to Parquet file.

    Args:
        results: PredictionResult to save
        path: Output Parquet path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df_data = {
        "subject_id": results.subject_ids,
        "predicted_mean": results.mean.flatten(),
    }

    if results.has_uncertainty:
        df_data["predicted_std"] = results.std.flatten()

    if results.has_actual:
        df_data["actual"] = results.actual.flatten()
    else:
        df_data["actual"] = [np.nan] * len(results.subject_ids)

    if results.pathology is not None:
        df_data["pathology"] = results.pathology[:, 0] if results.pathology.ndim > 1 else results.pathology

    df = pd.DataFrame(df_data)
    df.to_parquet(path, index=False)
    logger.info(f"Saved predictions to {path}")


def save_predictions_hdf5(
    results: PredictionResult,
    path: str | Path,
) -> None:
    """
    Save prediction attention weights to HDF5 file.

    Args:
        results: PredictionResult to save
        path: Output HDF5 path
    """
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        # Schema version
        f.attrs["schema_version"] = "1.0"
        f.attrs["n_subjects"] = results.n_subjects

        # Predictions
        f.create_dataset("mean", data=results.mean)
        if results.std is not None:
            f.create_dataset("std", data=results.std)

        # Attention weights
        f.create_dataset(
            "attention_weights",
            data=results.attention_weights,
            compression="gzip",
            compression_opts=4,
        )

        f.create_dataset(
            "gene_gate_weights",
            data=results.gene_gate_weights,
            compression="gzip",
            compression_opts=4,
        )

        # Subject IDs
        subject_ids_encoded = np.array(results.subject_ids, dtype="S64")
        f.create_dataset("subject_ids", data=subject_ids_encoded)

        # Metadata as JSON attribute
        if results.metadata:
            try:
                f.attrs["metadata"] = json.dumps(results.metadata, default=str)
            except (TypeError, ValueError):
                # Skip unserializable metadata
                pass

    logger.info(f"Saved predictions HDF5 to {path}")


def load_predictions_parquet(path: str | Path) -> pd.DataFrame:
    """
    Load predictions from Parquet file.

    Args:
        path: Path to Parquet file

    Returns:
        DataFrame with predictions
    """
    return pd.read_parquet(path)
