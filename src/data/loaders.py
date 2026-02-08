"""Shared data loading utilities for training scripts."""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from omegaconf import DictConfig

from src.data.splits import get_fold_subjects
from src.data.datasets import CognitiveResilienceDataset, PrecomputedDataset
from src.data.collate import create_dataloader

if TYPE_CHECKING:
    import anndata
    import pandas as pd


def create_fold_dataloaders(
    config: DictConfig,
    adata: anndata.AnnData | None,
    metadata: pd.DataFrame,
    splits: dict,
    fold_idx: int,
    precomputed_dir: str | Path | None = None,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train and validation DataLoaders for a CV fold.

    Args:
        config: Full experiment config (needs data section)
        adata: AnnData object (required if precomputed_dir is None)
        metadata: Subject-level metadata DataFrame
        splits: Splits dict from create_stratified_splits or load_splits
        fold_idx: CV fold index (0-indexed)
        precomputed_dir: If provided, use PrecomputedDataset

    Returns:
        (train_dataloader, val_dataloader)

    Raises:
        ValueError: If adata is None and precomputed_dir is None
    """
    data_cfg = config.data
    dl_cfg = data_cfg.dataloader

    train_subjects = get_fold_subjects(splits, fold_idx=fold_idx, split_type="train")
    val_subjects = get_fold_subjects(splits, fold_idx=fold_idx, split_type="val")

    if precomputed_dir is not None:
        train_ds = PrecomputedDataset(
            feature_dir=precomputed_dir, subject_ids=train_subjects, metadata=metadata,
            subject_column=data_cfg.get("subject_column", "ROSMAP_IndividualID"),
            target_column=data_cfg.get("target_column", "cogn_global"),
            pathology_columns=list(data_cfg.get("pathology_columns", [])),
        )
        val_ds = PrecomputedDataset(
            feature_dir=precomputed_dir, subject_ids=val_subjects, metadata=metadata,
            subject_column=data_cfg.get("subject_column", "ROSMAP_IndividualID"),
            target_column=data_cfg.get("target_column", "cogn_global"),
            pathology_columns=list(data_cfg.get("pathology_columns", [])),
        )
    else:
        if adata is None:
            raise ValueError("adata is required when precomputed_dir is not provided")
        train_ds = CognitiveResilienceDataset(
            adata, metadata, train_subjects,
            cell_type_column=data_cfg.get("cell_type_column", "supercluster_name"),
            subject_column=data_cfg.get("subject_column", "ROSMAP_IndividualID"),
            target_column=data_cfg.get("target_column", "cogn_global"),
            pathology_columns=list(data_cfg.get("pathology_columns", [])),
            max_cells_per_type=data_cfg.cell_sampling.get("max_cells_per_type", 1000),
            min_cells_threshold=data_cfg.cell_sampling.get("min_cells_threshold", 50),
            sampling_strategy=data_cfg.cell_sampling.get("sampling_strategy", "random"),
        )
        val_ds = CognitiveResilienceDataset(
            adata, metadata, val_subjects,
            cell_type_column=data_cfg.get("cell_type_column", "supercluster_name"),
            subject_column=data_cfg.get("subject_column", "ROSMAP_IndividualID"),
            target_column=data_cfg.get("target_column", "cogn_global"),
            pathology_columns=list(data_cfg.get("pathology_columns", [])),
            max_cells_per_type=data_cfg.cell_sampling.get("max_cells_per_type", 1000),
            min_cells_threshold=data_cfg.cell_sampling.get("min_cells_threshold", 50),
            sampling_strategy=data_cfg.cell_sampling.get("sampling_strategy", "random"),
        )

    train_loader = create_dataloader(
        train_ds, batch_size=dl_cfg.batch_size, shuffle=True,
        num_workers=dl_cfg.get("num_workers", 4),
        pin_memory=dl_cfg.get("pin_memory", True),
        multiregion=True, use_hgt_format=True,
        prefetch_factor=dl_cfg.get("prefetch_factor", 2),
    )
    val_loader = create_dataloader(
        val_ds, batch_size=dl_cfg.batch_size, shuffle=False,
        num_workers=dl_cfg.get("num_workers", 4),
        pin_memory=dl_cfg.get("pin_memory", True),
        multiregion=True, use_hgt_format=True,
        prefetch_factor=dl_cfg.get("prefetch_factor", 2),
    )
    return train_loader, val_loader
