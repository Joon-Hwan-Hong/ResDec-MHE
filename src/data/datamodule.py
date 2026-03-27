"""LightningDataModule for DDP-safe data loading."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import lightning.pytorch as pl
import torch
from omegaconf import DictConfig

from src.data.collate import create_dataloader
from src.data.datasets import CognitiveResilienceDataset, PrecomputedDataset
from src.data.samplers import EdgeCountBucketBatchSampler
from src.data.splits import get_fold_subjects, get_final_train_subjects

if TYPE_CHECKING:
    import anndata
    import pandas as pd

logger = logging.getLogger(__name__)


class CognitiveResilienceDataModule(pl.LightningDataModule):
    """LightningDataModule for cognitive resilience prediction.

    Wraps dataset creation and DataLoader construction for DDP-safe training.
    Lightning automatically adds DistributedSampler when strategy="ddp".

    Note: For reproducible validation, use PrecomputedDataset (precomputed .pt features).
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
        liana_results: Dict mapping subject_id -> LIANA+ DataFrame for CCC edges
            (only used with on-the-fly CognitiveResilienceDataset, not PrecomputedDataset)
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
        liana_results: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.metadata = metadata
        self.splits = splits
        self.fold_idx = fold_idx
        self.precomputed_dir = Path(precomputed_dir) if precomputed_dir is not None else None
        self.adata = adata
        self.final_mode = final_mode
        self.liana_results = liana_results

        self._data_cfg = config.data
        self._dl_cfg = self._data_cfg.dataloader
        self.batch_size = self._dl_cfg.batch_size

        self._train_ds = None
        self._val_ds = None
        self._test_ds = None

    def setup(self, stage: str | None = None) -> None:
        """Create datasets for the appropriate splits.

        Note: No idempotency guard (e.g., ``if self._train_ds is not None: return``)
        because Lightning may call setup() multiple times with different stages,
        and the CV loop legitimately recreates DataModules with different fold_idx.
        Recreating datasets is cheap (index-only, no data copy).
        """
        if self.final_mode:
            train_subjects = get_final_train_subjects(self.splits)
            test_subjects = self.splits["holdout_test"]

            if stage in ("fit", None):
                self._train_ds = self._make_dataset(train_subjects)
                if len(self._train_ds) == 0:
                    raise ValueError(
                        f"Training dataset is empty after filtering. "
                        f"Original: {len(train_subjects)} subjects. "
                        f"Check that .pt files exist or adata contains these subjects."
                    )
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
                if len(self._train_ds) == 0:
                    raise ValueError(
                        f"Training dataset is empty after filtering (fold {self.fold_idx}). "
                        f"Original: {len(train_subjects)} subjects. "
                        f"Check that .pt files exist or adata contains these subjects."
                    )
                if len(self._val_ds) == 0:
                    raise ValueError(
                        f"Validation dataset is empty after filtering (fold {self.fold_idx}). "
                        f"Original: {len(val_subjects)} subjects."
                    )
                logger.info(
                    "Fold %d: %d train, %d val subjects",
                    self.fold_idx, len(train_subjects), len(val_subjects),
                )
            if stage in ("test", None):
                test_subjects = get_fold_subjects(
                    self.splits, fold_idx=self.fold_idx, split_type="test"
                )
                self._test_ds = self._make_dataset(test_subjects)

    @property
    def train_target_mean(self) -> float | None:
        """Compute mean of training targets for data-driven prior centering.

        Returns None if training dataset has not been created yet (setup not called)
        or if the dataset is empty.
        """
        if self._train_ds is None or len(self._train_ds) == 0:
            return None
        targets = []
        for i in range(len(self._train_ds)):
            sample = self._train_ds[i]
            targets.append(sample["cognition"].item())
        return float(np.mean(targets))

    def on_train_epoch_start(self) -> None:
        """Update bucket sampler epoch for deterministic batch-order shuffling."""
        dl = self.trainer.train_dataloader
        if dl is not None and hasattr(dl, "batch_sampler"):
            bs = dl.batch_sampler
            if hasattr(bs, "set_epoch"):
                bs.set_epoch(self.trainer.current_epoch)

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
                liana_results=self.liana_results,
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
                sampling_seed=self.config.experiment.get("seed", 42),
                region_column=self._data_cfg.get(
                    "region_column", "BrainRegion"
                ),
                max_missing_subject_fraction=self._data_cfg.get(
                    "max_missing_subject_fraction", 0.1
                ),
            )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Create training DataLoader.

        When bucket_batching is enabled (default for PrecomputedDataset),
        uses EdgeCountBucketBatchSampler to group subjects with similar
        edge counts, reducing padding waste. This replaces both shuffle
        and DistributedSampler — the bucket sampler handles DDP internally.

        When bucket_batching is disabled, Lightning automatically replaces
        shuffle with DistributedSampler for DDP.
        """
        use_bucket = self._dl_cfg.get("bucket_batching", True)
        has_edge_counts = hasattr(self._train_ds, "get_edge_counts")

        if use_bucket and has_edge_counts:
            return self._make_bucket_dataloader(self._train_ds, shuffle=True)

        return create_dataloader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self._dl_cfg.get("num_workers", 4),
            pin_memory=self._dl_cfg.get("pin_memory", True),
            multiregion=True,
            use_hgt_format=True,
            prefetch_factor=self._dl_cfg.get("prefetch_factor", 2),
            worker_init_fn=self._make_worker_init_fn(),
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader | None:
        """Create validation DataLoader with deterministic cell sampling.

        DDP note: DistributedSampler pads dataset to make it evenly divisible
        across ranks. _gather_and_compute_metrics truncates to real dataset
        size for correct correlation metrics. drop_last defaults to False,
        which is correct here (padding handles equal batch counts).

        Returns None in final_mode (no validation set exists).
        """
        if self._val_ds is None:
            return None
        return create_dataloader(
            self._val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self._dl_cfg.get("num_workers", 4),
            pin_memory=self._dl_cfg.get("pin_memory", True),
            multiregion=True,
            use_hgt_format=True,
            prefetch_factor=self._dl_cfg.get("prefetch_factor", 2),
            worker_init_fn=self._make_deterministic_worker_init_fn(),
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
            worker_init_fn=self._make_deterministic_worker_init_fn(),
        )

    def _make_worker_init_fn(self):
        """Create a rank-aware worker init function for DDP reproducibility.

        Standard _worker_init_fn uses (global_seed + worker_id). Under DDP,
        multiple ranks share the same worker_ids (0..num_workers-1), so
        without rank offset, all ranks produce identical worker samples.

        This factory incorporates global_rank so each rank's workers get
        unique seeds: global_seed + global_rank * max_workers + worker_id.

        With persistent_workers=True, this init runs once at DataLoader creation.
        CellSampler RNGs advance naturally across epochs, providing different cell
        samples each epoch (data augmentation). The sampling sequence is reproducible
        only when num_workers, batch_size, and DDP world_size are held constant.
        """
        global_rank = self.trainer.global_rank if self.trainer is not None else 0
        max_workers = max(self._dl_cfg.get("num_workers", 4), 1)
        global_seed = self.config.experiment.get("seed", 42)

        def _rank_aware_worker_init_fn(worker_id: int) -> None:
            import random

            worker_seed = (global_seed + global_rank * max_workers + worker_id) % (2**32)
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            torch.manual_seed(worker_seed)

            # Re-seed CellSampler's RNG if the dataset has one
            worker_info = torch.utils.data.get_worker_info()
            dataset = worker_info.dataset
            # PrecomputedDataset has no sampler — hasattr is a no-op for it
            if hasattr(dataset, "sampler") and hasattr(dataset.sampler, "rng"):
                dataset.sampler.rng = np.random.default_rng(worker_seed)

        return _rank_aware_worker_init_fn

    def _make_deterministic_worker_init_fn(self):
        """Create a deterministic worker init function for val/test DataLoaders.

        Uses experiment seed (not hardcoded 42) for consistency with the rest
        of the reproducibility pipeline. Val/test workers get the same seed
        every epoch so evaluation is reproducible within and across runs.

        DDP note: Does not incorporate global_rank. Under DDP, all ranks'
        val/test workers use the same seed. This is correct because
        DistributedSampler gives each rank different subjects, and with
        PrecomputedDataset (recommended) cell sampling does not apply.
        """
        global_seed = self.config.experiment.get("seed", 42)

        def _det_worker_init_fn(worker_id: int) -> None:
            import random

            seed = global_seed + worker_id
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)

            worker_info = torch.utils.data.get_worker_info()
            dataset = worker_info.dataset
            # PrecomputedDataset has no sampler — hasattr is a no-op for it
            if hasattr(dataset, "sampler") and hasattr(dataset.sampler, "rng"):
                dataset.sampler.rng = np.random.default_rng(seed)

        return _det_worker_init_fn

    def _make_bucket_dataloader(
        self, dataset, shuffle: bool = True,
    ) -> torch.utils.data.DataLoader:
        """Create DataLoader with EdgeCountBucketBatchSampler.

        The batch_sampler handles batching AND DDP distribution, so we pass
        batch_size=1 and batch_sampler to DataLoader (mutually exclusive with
        batch_size, shuffle, sampler, and drop_last).
        """
        from src.data.collate import collate_for_hgt_multiregion, _worker_init_fn

        global_rank = self.trainer.global_rank if self.trainer is not None else 0
        world_size = self.trainer.world_size if self.trainer is not None else 1
        seed = self.config.experiment.get("seed", 42)
        num_workers = self._dl_cfg.get("num_workers", 4)
        prefetch = self._dl_cfg.get("prefetch_factor", 2) if num_workers > 0 else None

        batch_sampler = EdgeCountBucketBatchSampler(
            edge_counts=dataset.get_edge_counts(),
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=shuffle,
            seed=seed,
            rank=global_rank,
            world_size=world_size,
        )

        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=self._dl_cfg.get("pin_memory", True),
            collate_fn=collate_for_hgt_multiregion,
            persistent_workers=num_workers > 0,
            worker_init_fn=self._make_worker_init_fn() if num_workers > 0 else None,
            prefetch_factor=prefetch,
        )
