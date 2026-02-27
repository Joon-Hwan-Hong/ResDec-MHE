"""LightningDataModule for DDP-safe data loading."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import lightning.pytorch as pl
import torch
from omegaconf import DictConfig

from src.data.collate import create_dataloader, _deterministic_worker_init_fn
from src.data.datasets import CognitiveResilienceDataset, PrecomputedDataset
from src.data.splits import get_fold_subjects, get_final_train_subjects

if TYPE_CHECKING:
    import anndata
    import pandas as pd

logger = logging.getLogger(__name__)


class CognitiveResilienceDataModule(pl.LightningDataModule):
    """LightningDataModule for cognitive resilience prediction.

    Wraps dataset creation and DataLoader construction for DDP-safe training.
    Lightning automatically adds DistributedSampler when strategy="ddp".

    Note: For reproducible validation, use PrecomputedDataset (precomputed .npz features).
    On-the-fly CognitiveResilienceDataset uses random cell sampling which introduces
    stochastic variation in val/test metrics.

    Args:
        config: Full experiment config (needs data section)
        metadata: Subject-level metadata DataFrame
        splits: Splits dict from create_stratified_splits or load_splits
        fold_idx: CV fold index (0-indexed)
        precomputed_dir: If provided, use PrecomputedDataset
        adata: AnnData object (required if precomputed_dir is None)
        final_mode: If True, train on full train_val_pool, test on holdout
    """

    def __init__(
        self,
        config: DictConfig,
        metadata: pd.DataFrame,
        splits: dict,
        fold_idx: int,
        precomputed_dir: str | Path | None = None,
        adata: anndata.AnnData | None = None,
        final_mode: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.metadata = metadata
        self.splits = splits
        self.fold_idx = fold_idx
        self.precomputed_dir = Path(precomputed_dir) if precomputed_dir is not None else None
        self.adata = adata
        self.final_mode = final_mode

        self._data_cfg = config.data
        self._dl_cfg = self._data_cfg.dataloader
        self.batch_size = self._dl_cfg.batch_size

        self._train_ds = None
        self._val_ds = None
        self._test_ds = None

    def setup(self, stage: str | None = None) -> None:
        """Create datasets for the appropriate splits."""
        if self.final_mode:
            train_subjects = get_final_train_subjects(self.splits)
            test_subjects = self.splits["holdout_test"]

            if stage in ("fit", None):
                self._train_ds = self._make_dataset(train_subjects)
                logger.info(
                    "Final mode: %d train subjects", len(train_subjects)
                )
            if stage in ("test", None):
                self._test_ds = self._make_dataset(test_subjects)
                logger.info(
                    "Final mode: %d holdout test subjects", len(test_subjects)
                )
        else:
            train_subjects = get_fold_subjects(
                self.splits, fold_idx=self.fold_idx, split_type="train"
            )
            val_subjects = get_fold_subjects(
                self.splits, fold_idx=self.fold_idx, split_type="val"
            )

            if stage in ("fit", None):
                self._train_ds = self._make_dataset(train_subjects)
                self._val_ds = self._make_dataset(val_subjects)
                logger.info(
                    "Fold %d: %d train, %d val subjects",
                    self.fold_idx, len(train_subjects), len(val_subjects),
                )
            if stage in ("test", None):
                test_subjects = get_fold_subjects(
                    self.splits, fold_idx=self.fold_idx, split_type="test"
                )
                self._test_ds = self._make_dataset(test_subjects)

    def _make_dataset(self, subject_ids: list[str]):
        """Create a dataset for the given subject IDs."""
        if self.precomputed_dir is not None:
            return PrecomputedDataset(
                feature_dir=self.precomputed_dir,
                subject_ids=subject_ids,
                metadata=self.metadata,
                subject_column=self._data_cfg.get(
                    "subject_column", "ROSMAP_IndividualID"
                ),
                target_column=self._data_cfg.get(
                    "target_column", "cogn_global"
                ),
                pathology_columns=list(
                    self._data_cfg.get("pathology_columns", [])
                ),
            )
        else:
            if self.adata is None:
                raise ValueError(
                    "adata is required when precomputed_dir is not provided"
                )
            return CognitiveResilienceDataset(
                self.adata,
                self.metadata,
                subject_ids,
                cell_type_column=self._data_cfg.get(
                    "cell_type_column", "supercluster_name"
                ),
                subject_column=self._data_cfg.get(
                    "subject_column", "ROSMAP_IndividualID"
                ),
                target_column=self._data_cfg.get(
                    "target_column", "cogn_global"
                ),
                pathology_columns=list(
                    self._data_cfg.get("pathology_columns", [])
                ),
                max_cells_per_type=self._data_cfg.cell_sampling.get(
                    "max_cells_per_type", 1000
                ),
                min_cells_threshold=self._data_cfg.cell_sampling.get(
                    "min_cells_threshold", 50
                ),
                sampling_strategy=self._data_cfg.cell_sampling.get(
                    "sampling_strategy", "random"
                ),
            )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Create training DataLoader.

        Lightning automatically replaces shuffle with DistributedSampler
        when using strategy="ddp".
        """
        return create_dataloader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self._dl_cfg.get("num_workers", 4),
            pin_memory=self._dl_cfg.get("pin_memory", True),
            multiregion=True,
            use_hgt_format=True,
            prefetch_factor=self._dl_cfg.get("prefetch_factor", 2),
            worker_init_fn=self._make_worker_init_fn(),
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Create validation DataLoader with deterministic cell sampling."""
        return create_dataloader(
            self._val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self._dl_cfg.get("num_workers", 4),
            pin_memory=self._dl_cfg.get("pin_memory", True),
            multiregion=True,
            use_hgt_format=True,
            prefetch_factor=self._dl_cfg.get("prefetch_factor", 2),
            worker_init_fn=_deterministic_worker_init_fn,
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """Create test DataLoader with deterministic cell sampling."""
        return create_dataloader(
            self._test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self._dl_cfg.get("num_workers", 4),
            pin_memory=self._dl_cfg.get("pin_memory", True),
            multiregion=True,
            use_hgt_format=True,
            prefetch_factor=self._dl_cfg.get("prefetch_factor", 2),
            worker_init_fn=_deterministic_worker_init_fn,
        )

    def _make_worker_init_fn(self):
        """Create a rank-aware worker init function for DDP reproducibility.

        Standard _worker_init_fn uses (global_seed + worker_id). Under DDP,
        multiple ranks share the same worker_ids (0..num_workers-1), so
        without rank offset, all ranks produce identical worker samples.

        This factory incorporates global_rank so each rank's workers get
        unique seeds: global_seed + global_rank * max_workers + worker_id.
        """
        global_rank = self.trainer.global_rank if self.trainer is not None else 0
        max_workers = max(self._dl_cfg.get("num_workers", 4), 1)
        global_seed = self.config.experiment.get("seed", 42)

        def _rank_aware_worker_init_fn(worker_id: int) -> None:
            import random

            worker_seed = (global_seed + global_rank * max_workers + worker_id) % (2**32)
            np.random.seed(worker_seed)
            random.seed(worker_seed)

            # Re-seed CellSampler's RNG if the dataset has one
            worker_info = torch.utils.data.get_worker_info()
            dataset = worker_info.dataset
            if hasattr(dataset, "sampler") and hasattr(dataset.sampler, "rng"):
                dataset.sampler.rng = np.random.default_rng(worker_seed)

        return _rank_aware_worker_init_fn
