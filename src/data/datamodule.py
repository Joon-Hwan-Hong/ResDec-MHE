"""LightningDataModule for DDP-safe data loading."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import lightning.pytorch as pl
import torch
from omegaconf import DictConfig

from src.data.collate import create_dataloader
from src.data.datasets import CognitiveResilienceDataset, PrecomputedDataset
from src.data.feature_loaders import load_residualized_targets
from src.data.prefetch import ThreadedPrefetcher
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
        preloaded_cache: Pre-loaded subject tensors from
            ``PrecomputedDataset.load_subject_cache``. Eliminates per-trial disk I/O
            during HPO. Only used with PrecomputedDataset.
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
        preloaded_cache: dict[str, dict] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.metadata = metadata
        self.splits = splits
        self.fold_idx = fold_idx
        self._prefetchers: list[ThreadedPrefetcher] = []
        self.precomputed_dir = Path(precomputed_dir) if precomputed_dir is not None else None
        self.adata = adata
        self.final_mode = final_mode
        self.liana_results = liana_results
        self.preloaded_cache = preloaded_cache

        self._data_cfg = config.data
        self._dl_cfg = self._data_cfg.dataloader
        self.batch_size = self._dl_cfg.batch_size

        self._train_ds = None
        self._val_ds = None
        self._test_ds = None

        # Train-only age stats for FiLM metadata z-scoring. Populated in
        # ``setup()`` from the fold's train subjects — val/test use the same
        # stats to avoid leakage.
        self._train_age_mean: float | None = None
        self._train_age_std: float | None = None

        # Variant-target override: when cfg.data.residualize_against is set,
        # setup() injects fold-specific residual targets as a synthetic
        # metadata column and stores the column name here so _make_dataset
        # passes it as target_column. Empty string means use raw cogn_global.
        self._target_column_override: str = ""

    @property
    def _effective_num_workers(self) -> int:
        """Return 0 workers when using precomputed (heap-loaded) data.

        With heap-loaded .pt files, __getitem__ is a dict lookup on
        in-memory tensors (O(1), no disk I/O).  DataLoader workers
        add no throughput benefit but their fork() fails under
        overcommit_memory=2 because process-private pages are counted
        toward the commit limit.  This applies regardless of GPU count.
        """
        if self.precomputed_dir is not None:
            return 0
        return self._dl_cfg.get("num_workers", 4)

    def setup(self, stage: str | None = None) -> None:
        """Create datasets for the appropriate splits.

        Note: No idempotency guard (e.g., ``if self._train_ds is not None: return``)
        because Lightning may call setup() multiple times with different stages,
        and the CV loop legitimately recreates DataModules with different fold_idx.
        Recreating datasets is cheap (index-only, no data copy).
        """
        # Variant-target injection: when cfg.data.residualize_against is set,
        # the dataset classes read targets from metadata.loc[sid, target_column],
        # so the cleanest wire-up for a per-fold residual target is to inject
        # the fold's residuals as a synthetic metadata column and override
        # target_column. (Plan Task 4 originally proposed swapping a
        # load_targets() call, but the datamodule does not own that call —
        # the dataset does.)
        rcfg = self._data_cfg.get("residualize_against")
        if rcfg is not None and self.final_mode:
            raise NotImplementedError(
                "final_mode=True with cfg.data.residualize_against is not "
                "supported: residual cache is per-fold; final_mode trains on "
                "the full train_val_pool with no fold structure. Use per-fold "
                "training and aggregate, or extend "
                "scripts/resdec_mhe/variants/compute_residual_target.py to "
                "emit a final-mode cache."
            )
        if rcfg is not None:
            cache_dir = Path(rcfg.cache_dir)
            override_col = f"_residual_target_fold{self.fold_idx}"
            stale = [
                c for c in self.metadata.columns
                if c.startswith("_residual_target_fold")
            ]
            if stale:
                self.metadata = self.metadata.drop(columns=stale)
            residuals = load_residualized_targets(
                subject_ids=self.metadata["ROSMAP_IndividualID"].tolist(),
                cache_dir=cache_dir, fold_idx=self.fold_idx,
            )
            self.metadata = self.metadata.copy()
            self.metadata[override_col] = self.metadata[
                "ROSMAP_IndividualID"
            ].map(residuals)
            self._target_column_override = override_col
            logger.info(
                "Variant residualized target injected: fold=%d cache=%s "
                "n_subjects=%d (axes=%s)",
                self.fold_idx, cache_dir, len(residuals),
                list(rcfg.axes),
            )

        if self.final_mode:
            train_subjects = get_final_train_subjects(self.splits)
            test_subjects = self.splits["holdout_test"]

            # Compute TRAIN-only age stats for FiLM metadata z-scoring.
            # Val/test datasets reuse these same stats to avoid leakage.
            self._compute_train_age_stats(train_subjects)

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

            # Compute TRAIN-only age stats for FiLM metadata z-scoring.
            # Val/test datasets reuse these same stats to avoid leakage.
            self._compute_train_age_stats(train_subjects)

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
    def train_dataset(self):
        """Public access to the training dataset (used by LightningModule for 1/N KL scaling)."""
        return self._train_ds

    @property
    def _is_ddp(self) -> bool:
        """True iff a Trainer is attached and ``world_size > 1`` (DDP active).

        Centralises the "trainer is attached and CUDA is in use under DDP"
        check that the train/val/test dataloader builders previously
        repeated three times.
        """
        return (
            torch.cuda.is_available()
            and self.trainer is not None
            and self.trainer.world_size > 1
        )

    @property
    def train_target_mean(self) -> float | None:
        """Compute mean of training targets for data-driven prior centering.

        Returns None if training dataset has not been created yet (setup not called)
        or if the dataset is empty.
        """
        if self._train_ds is None or len(self._train_ds) == 0:
            return None
        # Direct array access avoids N __getitem__ calls + dict construction.
        # Both CognitiveResilienceDataset and PrecomputedDataset expose
        # ``_target_array``; gate on hasattr to fail soft against future
        # dataset replacements (caller can recover by walking __getitem__).
        if not hasattr(self._train_ds, "_target_array"):
            return None
        return float(np.mean(self._train_ds._target_array))

    def _make_dataset(self, subject_ids: list[str]):
        """Create a dataset for the given subject IDs."""
        meta_csv = self._resolve_meta_csv()
        target_column = self._target_column_override or self._data_cfg.get(
            "target_column", "cogn_global"
        )
        # Variant override fallback: holdout_test subjects are NOT in the
        # per-fold residual cache (cache spans train_val_pool only). If any
        # of these subjects are missing from the override column, fall back
        # to the canonical raw target column rather than crashing on NaN.
        if self._target_column_override:
            mcol = self.metadata.set_index("ROSMAP_IndividualID")[
                self._target_column_override
            ]
            requested = [s for s in subject_ids if s in mcol.index]
            if mcol.loc[requested].isna().any():
                target_column = self._data_cfg.get(
                    "target_column", "cogn_global"
                )
                logger.warning(
                    "Variant override has NaN for some requested subjects "
                    "(likely holdout_test); falling back to %s for this dataset.",
                    target_column,
                )
        if self.precomputed_dir is not None:
            return PrecomputedDataset(
                feature_dir=self.precomputed_dir,
                subject_ids=subject_ids,
                metadata=self.metadata,
                subject_column=self._data_cfg.get(
                    "subject_column", "ROSMAP_IndividualID"
                ),
                target_column=target_column,
                pathology_columns=list(
                    self._data_cfg.get("pathology_columns", [])
                ),
                preloaded_cache=self.preloaded_cache,
                meta_csv=meta_csv,
                age_mean=self._train_age_mean,
                age_std=self._train_age_std,
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
                target_column=target_column,
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
                meta_csv=meta_csv,
                age_mean=self._train_age_mean,
                age_std=self._train_age_std,
            )

    def _resolve_meta_csv(self) -> Path | None:
        """Return the path to metadata.csv, or None if not configured.

        Reads ``data.metadata_path`` from the config. The value may point at
        a directory (the legacy convention from configs/default.yaml, in
        which case ``metadata.csv`` is appended) or directly at a CSV file.
        Returns None if the resolved path does not exist — the dataset then
        skips FiLM metadata wiring and the lightning module's None→zeros
        fallback kicks in. Emits a ``logger.warning`` when ``metadata_path``
        was configured but resolves to a nonexistent file so misconfiguration
        (e.g. a typo) doesn't silently degrade FiLM to a no-op.
        """
        meta_path_str = self._data_cfg.get("metadata_path", None)
        if meta_path_str is None:
            return None
        meta_path = Path(meta_path_str)
        meta_csv = meta_path / "metadata.csv" if meta_path.is_dir() else meta_path
        if not meta_csv.exists():
            logger.warning(
                "metadata_path=%r resolves to %s which does not exist — "
                "FiLM will run with zero metadata (no-op). Check data.metadata_path.",
                meta_path_str, meta_csv,
            )
            return None
        return meta_csv

    def _compute_train_age_stats(self, train_subjects: list[str]) -> None:
        """Compute train-only age_mean / age_std for FiLM z-scoring.

        Reads ``age_death`` from ``self.metadata`` restricted to the fold's
        train subject IDs. Val/test datasets reuse these stats (set via
        ``self._train_age_mean`` / ``self._train_age_std``) to prevent val
        leakage through per-fold z-score statistics. Falls back to leaving
        the stats as None when ``age_death`` is missing — the downstream
        ``load_metadata_vector`` default (cohort-wide approx) then applies.
        """
        if "age_death" not in self.metadata.columns:
            return
        subject_column = self._data_cfg.get(
            "subject_column", "ROSMAP_IndividualID"
        )
        if subject_column in self.metadata.columns:
            mask = self.metadata[subject_column].isin(train_subjects)
            train_ages = self.metadata.loc[mask, "age_death"].dropna()
        else:
            train_ages = self.metadata.loc[
                self.metadata.index.isin(train_subjects), "age_death"
            ].dropna()
        if len(train_ages) == 0:
            return
        self._train_age_mean = float(train_ages.mean())
        # ddof=0 matches the default in load_metadata_vector's reference stats
        # (population std). Using ddof=1 would subtly shift FiLM inputs.
        self._train_age_std = float(train_ages.std(ddof=0))

    def train_dataloader(self) -> torch.utils.data.DataLoader | ThreadedPrefetcher:
        """Create training DataLoader, optionally wrapped with ThreadedPrefetcher.

        On CUDA, wraps the DataLoader with ThreadedPrefetcher to overlap batch
        collation (torch.cat of ~1.4 GB cell_data) with GPU forward/backward.
        This reduces data loading time from 628ms to 103ms (2-GPU DDP) and
        158ms to ~0ms (1-GPU).

        When wrapping with ThreadedPrefetcher, DistributedSampler is added
        manually because Lightning only auto-adds sampler to DataLoader
        instances (ThreadedPrefetcher is not a DataLoader).  Val/test loaders
        return plain DataLoaders and get Lightning's automatic sampler.

        ThreadedPrefetcher also moves batches to the target CUDA device.
        Lightning's transfer_batch_to_device sees tensors already on device
        and becomes a no-op (torch .to() on same-device tensor returns self).
        """
        nw = self._effective_num_workers

        # Under DDP on CUDA, add DistributedSampler ourselves (see docstring)
        sampler = None
        if self._is_ddp:
            from torch.utils.data.distributed import DistributedSampler

            sampler = DistributedSampler(
                self._train_ds,
                num_replicas=self.trainer.world_size,
                rank=self.trainer.global_rank,
                shuffle=True,
                seed=self.config.experiment.get("seed", 42),
            )

        # When ThreadedPrefetcher will wrap the DataLoader, disable pin_memory
        # since the prefetcher does its own .to(device) — double-pinning wastes
        # a synchronous memcpy.
        use_prefetcher = torch.cuda.is_available() and self.trainer is not None
        pin = False if use_prefetcher else self._dl_cfg.get("pin_memory", True)

        # drop_last is normally True (avoid partial final batches) but when the
        # train cohort fits inside a single batch (full-cohort NPT mode used
        # by the ResDec-MHE head), dropping that single partial batch would
        # empty the loader. Auto-disable drop_last in that regime.
        drop_last = len(self._train_ds) > self.batch_size

        dl = create_dataloader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=drop_last,
            num_workers=nw,
            pin_memory=pin,
            multiregion=True,
            use_hgt_format=True,
            prefetch_factor=self._dl_cfg.get("prefetch_factor", 2) if nw > 0 else None,
            worker_init_fn=self._make_worker_init_fn() if nw > 0 else None,
            sampler=sampler,
            seed=self.config.experiment.get("seed", 42),
        )

        if use_prefetcher:
            device = torch.device(f"cuda:{self.trainer.local_rank}")
            pf = ThreadedPrefetcher(dl, device, prefetch_count=2)
            self._prefetchers.append(pf)
            return pf

        return dl

    def val_dataloader(self) -> torch.utils.data.DataLoader | ThreadedPrefetcher | None:
        """Create validation DataLoader with deterministic cell sampling.

        DDP note: DistributedSampler pads dataset to make it evenly divisible
        across ranks. _gather_and_compute_metrics truncates to real dataset
        size for correct correlation metrics. Lightning handles
        DistributedSampler automatically (use_distributed_sampler=True).

        On CUDA, wraps with ThreadedPrefetcher (same as train_dataloader)
        to overlap collation with GPU compute.

        Returns None in final_mode (no validation set exists).
        """
        if self._val_ds is None:
            return None
        nw = self._effective_num_workers
        use_prefetcher = torch.cuda.is_available() and self.trainer is not None
        pin = False if use_prefetcher else self._dl_cfg.get("pin_memory", True)
        dl = create_dataloader(
            self._val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=nw,
            pin_memory=pin,
            multiregion=True,
            use_hgt_format=True,
            prefetch_factor=self._dl_cfg.get("prefetch_factor", 2) if nw > 0 else None,
            worker_init_fn=self._make_deterministic_worker_init_fn() if nw > 0 else None,
            seed=self.config.experiment.get("seed", 42),
        )
        if use_prefetcher:
            device = torch.device(f"cuda:{self.trainer.local_rank}")
            pf = ThreadedPrefetcher(dl, device, prefetch_count=2)
            self._prefetchers.append(pf)
            return pf
        return dl

    def test_dataloader(self) -> torch.utils.data.DataLoader | ThreadedPrefetcher:
        """Create test DataLoader with deterministic cell sampling.

        On CUDA, wraps with ThreadedPrefetcher (same as train/val) to overlap
        collation with GPU compute.
        """
        nw = self._effective_num_workers
        use_prefetcher = torch.cuda.is_available() and self.trainer is not None
        pin = False if use_prefetcher else self._dl_cfg.get("pin_memory", True)
        dl = create_dataloader(
            self._test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=nw,
            pin_memory=pin,
            multiregion=True,
            use_hgt_format=True,
            prefetch_factor=self._dl_cfg.get("prefetch_factor", 2) if nw > 0 else None,
            worker_init_fn=self._make_deterministic_worker_init_fn() if nw > 0 else None,
            seed=self.config.experiment.get("seed", 42),
        )
        if use_prefetcher:
            device = torch.device(f"cuda:{self.trainer.local_rank}")
            pf = ThreadedPrefetcher(dl, device, prefetch_count=2)
            self._prefetchers.append(pf)
            return pf
        return dl

    def shutdown_prefetchers(self) -> None:
        """Shut down all ThreadedPrefetcher instances created by this DataModule.

        Call before deleting the DataModule to ensure daemon producer threads
        release their GPU tensor references. Replaces the fragile
        gc.get_objects() scan pattern.
        """
        for pf in self._prefetchers:
            pf.shutdown()
        self._prefetchers.clear()

    @staticmethod
    def _seed_worker(worker_seed: int) -> None:
        """Seed numpy / random / torch and the per-dataset CellSampler RNG.

        Shared between the rank-aware (train) and deterministic (val/test)
        worker init factories below, replacing two near-duplicate closures.
        """
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset
        # PrecomputedDataset has no sampler — hasattr is a no-op for it.
        if hasattr(dataset, "sampler") and hasattr(dataset.sampler, "rng"):
            dataset.sampler.rng = np.random.default_rng(worker_seed)

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
        # ``max_workers >= 1`` so the per-rank stride never collapses to 0
        # when num_workers=0; without this clamp, all ranks would land on
        # the same worker_seed for worker_id=0.
        max_workers = max(self._dl_cfg.get("num_workers", 4), 1)
        global_seed = self.config.experiment.get("seed", 42)

        def _rank_aware_worker_init_fn(worker_id: int) -> None:
            worker_seed = (
                global_seed + global_rank * max_workers + worker_id
            ) % (2**32)
            CognitiveResilienceDataModule._seed_worker(worker_seed)

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
            seed = global_seed + worker_id
            CognitiveResilienceDataModule._seed_worker(seed)

        return _det_worker_init_fn


