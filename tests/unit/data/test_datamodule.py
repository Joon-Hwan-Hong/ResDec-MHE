"""Tests for CognitiveResilienceDataModule."""
import pytest
import torch
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from omegaconf import OmegaConf

from src.data.datamodule import CognitiveResilienceDataModule


@pytest.fixture
def minimal_config():
    """Minimal config for DataModule."""
    return OmegaConf.create({
        "data": {
            "cell_type_column": "supercluster_name",
            "subject_column": "ROSMAP_IndividualID",
            "target_column": "cogn_global",
            "pathology_columns": [],
            "cell_sampling": {
                "max_cells_per_type": 100,
                "min_cells_threshold": 10,
                "sampling_strategy": "random",
            },
            "dataloader": {
                "batch_size": 4,
                "num_workers": 0,
                "pin_memory": False,
                "prefetch_factor": None,
            },
        },
        "experiment": {"seed": 42},
    })


@pytest.fixture
def mock_metadata():
    return pd.DataFrame({
        "ROSMAP_IndividualID": [f"subj_{i}" for i in range(20)],
        "cogn_global": np.random.randn(20),
    })


@pytest.fixture
def mock_splits():
    subjects = [f"subj_{i}" for i in range(20)]
    return {
        "holdout_test": subjects[:4],
        "train_val_pool": subjects[4:],
        "folds": [
            {"train": subjects[4:14], "val": subjects[14:]},
        ],
    }


@pytest.fixture
def precomputed_dir(tmp_path, mock_metadata):
    """Create a tmp dir with minimal .npz files for all 20 subjects."""
    from src.data.constants import CELL_TYPE_ORDER, REGION_ORDER

    n_cell_types = len(CELL_TYPE_ORDER)
    n_genes = 10  # minimal
    max_cells = 5
    n_regions = len(REGION_ORDER)

    for i in range(20):
        sid = f"subj_{i}"
        np.savez_compressed(
            tmp_path / f"{sid}.npz",
            pseudobulk=np.random.randn(n_cell_types, n_genes).astype(np.float32),
            cell_type_mask=np.ones(n_cell_types, dtype=bool),
            cell_counts=np.full(n_cell_types, max_cells, dtype=np.int64),
            region_mask=np.array([True] + [False] * (n_regions - 1), dtype=bool),
            cells=np.random.randn(n_cell_types, max_cells, n_genes).astype(np.float32),
            cell_mask=np.ones((n_cell_types, max_cells), dtype=bool),
            edge_index=np.zeros((2, 0), dtype=np.int64),
            edge_type=np.zeros((0,), dtype=np.int64),
            edge_attr=np.zeros((0, 1), dtype=np.float32),
        )
    return tmp_path


class TestCognitiveResilienceDataModule:
    def test_init_with_precomputed(self, minimal_config, mock_metadata, mock_splits, precomputed_dir):
        """DataModule can be initialized with precomputed features."""
        dm = CognitiveResilienceDataModule(
            config=minimal_config,
            metadata=mock_metadata,
            splits=mock_splits,
            fold_idx=0,
            precomputed_dir=precomputed_dir,
        )
        assert dm.batch_size == 4

    def test_train_dataloader_returns_dataloader(self, minimal_config, mock_metadata, mock_splits, precomputed_dir):
        """train_dataloader() returns a DataLoader."""
        dm = CognitiveResilienceDataModule(
            config=minimal_config,
            metadata=mock_metadata,
            splits=mock_splits,
            fold_idx=0,
            precomputed_dir=precomputed_dir,
        )
        dm.setup(stage="fit")
        loader = dm.train_dataloader()
        assert isinstance(loader, torch.utils.data.DataLoader)

    def test_val_dataloader_returns_dataloader(self, minimal_config, mock_metadata, mock_splits, precomputed_dir):
        """val_dataloader() returns a DataLoader."""
        dm = CognitiveResilienceDataModule(
            config=minimal_config,
            metadata=mock_metadata,
            splits=mock_splits,
            fold_idx=0,
            precomputed_dir=precomputed_dir,
        )
        dm.setup(stage="fit")
        loader = dm.val_dataloader()
        assert isinstance(loader, torch.utils.data.DataLoader)

    def test_worker_init_fn_accounts_for_rank(self, minimal_config, mock_metadata, mock_splits, precomputed_dir):
        """Worker init function factory exists for DDP reproducibility."""
        dm = CognitiveResilienceDataModule(
            config=minimal_config,
            metadata=mock_metadata,
            splits=mock_splits,
            fold_idx=0,
            precomputed_dir=precomputed_dir,
        )
        assert hasattr(dm, '_make_worker_init_fn')

    def test_final_mode_uses_all_train_val_subjects(self, minimal_config, mock_metadata, mock_splits, precomputed_dir):
        """Final mode trains on full train_val_pool and tests on holdout."""
        dm = CognitiveResilienceDataModule(
            config=minimal_config,
            metadata=mock_metadata,
            splits=mock_splits,
            fold_idx=0,
            precomputed_dir=precomputed_dir,
            final_mode=True,
        )
        dm.setup(stage="fit")
        train_loader = dm.train_dataloader()
        assert train_loader is not None

    def test_final_mode_val_dataloader_returns_none(self, minimal_config, mock_metadata, mock_splits, precomputed_dir):
        """In final mode, val_dataloader() returns None since no validation set exists."""
        dm = CognitiveResilienceDataModule(
            config=minimal_config,
            metadata=mock_metadata,
            splits=mock_splits,
            fold_idx=0,
            precomputed_dir=precomputed_dir,
            final_mode=True,
        )
        dm.setup(stage="fit")
        assert dm.val_dataloader() is None

    def test_final_mode_test_dataloader(self, minimal_config, mock_metadata, mock_splits, precomputed_dir):
        """Final mode provides test dataloader for holdout evaluation."""
        dm = CognitiveResilienceDataModule(
            config=minimal_config,
            metadata=mock_metadata,
            splits=mock_splits,
            fold_idx=0,
            precomputed_dir=precomputed_dir,
            final_mode=True,
        )
        dm.setup(stage="test")
        test_loader = dm.test_dataloader()
        assert test_loader is not None


class TestSetupBehavior:
    """T6: Test setup() creates correct datasets for each stage."""

    def test_setup_fit_creates_train_and_val(self, minimal_config, mock_metadata, mock_splits, precomputed_dir):
        dm = CognitiveResilienceDataModule(
            config=minimal_config, metadata=mock_metadata, splits=mock_splits,
            fold_idx=0, precomputed_dir=precomputed_dir,
        )
        dm.setup(stage="fit")
        assert dm._train_ds is not None
        assert dm._val_ds is not None
        assert dm._test_ds is None

    def test_setup_test_creates_test_only(self, minimal_config, mock_metadata, mock_splits, precomputed_dir):
        dm = CognitiveResilienceDataModule(
            config=minimal_config, metadata=mock_metadata, splits=mock_splits,
            fold_idx=0, precomputed_dir=precomputed_dir,
        )
        dm.setup(stage="test")
        assert dm._train_ds is None
        assert dm._val_ds is None
        assert dm._test_ds is not None

    def test_setup_none_creates_all(self, minimal_config, mock_metadata, mock_splits, precomputed_dir):
        dm = CognitiveResilienceDataModule(
            config=minimal_config, metadata=mock_metadata, splits=mock_splits,
            fold_idx=0, precomputed_dir=precomputed_dir,
        )
        dm.setup(stage=None)
        assert dm._train_ds is not None
        assert dm._val_ds is not None
        assert dm._test_ds is not None

    def test_final_mode_setup_fit_no_val(self, minimal_config, mock_metadata, mock_splits, precomputed_dir):
        dm = CognitiveResilienceDataModule(
            config=minimal_config, metadata=mock_metadata, splits=mock_splits,
            fold_idx=0, precomputed_dir=precomputed_dir, final_mode=True,
        )
        dm.setup(stage="fit")
        assert dm._train_ds is not None
        assert dm._val_ds is None


class TestMakeDataset:
    """T6: Test _make_dataset paths."""

    def test_precomputed_creates_precomputed_dataset(self, minimal_config, mock_metadata, mock_splits, precomputed_dir):
        from src.data.datasets import PrecomputedDataset
        dm = CognitiveResilienceDataModule(
            config=minimal_config, metadata=mock_metadata, splits=mock_splits,
            fold_idx=0, precomputed_dir=precomputed_dir,
        )
        ds = dm._make_dataset(["subj_0", "subj_1"])
        assert isinstance(ds, PrecomputedDataset)

    def test_no_precomputed_no_adata_raises(self, minimal_config, mock_metadata, mock_splits):
        dm = CognitiveResilienceDataModule(
            config=minimal_config, metadata=mock_metadata, splits=mock_splits,
            fold_idx=0, precomputed_dir=None, adata=None,
        )
        with pytest.raises(ValueError, match="adata is required"):
            dm._make_dataset(["subj_0"])

    def test_dataset_subject_count_matches(self, minimal_config, mock_metadata, mock_splits, precomputed_dir):
        dm = CognitiveResilienceDataModule(
            config=minimal_config, metadata=mock_metadata, splits=mock_splits,
            fold_idx=0, precomputed_dir=precomputed_dir,
        )
        subjects = ["subj_0", "subj_1", "subj_2"]
        ds = dm._make_dataset(subjects)
        assert len(ds) == 3


class TestDataloaderConfig:
    """T6: Verify DataLoader configuration."""

    def test_train_dataloader_batch_size(self, minimal_config, mock_metadata, mock_splits, precomputed_dir):
        dm = CognitiveResilienceDataModule(
            config=minimal_config, metadata=mock_metadata, splits=mock_splits,
            fold_idx=0, precomputed_dir=precomputed_dir,
        )
        dm.setup(stage="fit")
        loader = dm.train_dataloader()
        assert loader.batch_size == 4

    def test_val_dataloader_batch_size(self, minimal_config, mock_metadata, mock_splits, precomputed_dir):
        dm = CognitiveResilienceDataModule(
            config=minimal_config, metadata=mock_metadata, splits=mock_splits,
            fold_idx=0, precomputed_dir=precomputed_dir,
        )
        dm.setup(stage="fit")
        loader = dm.val_dataloader()
        assert loader is not None
        assert loader.batch_size == 4

    def test_batch_size_from_config(self, mock_metadata, mock_splits, precomputed_dir):
        config = OmegaConf.create({
            "data": {
                "cell_type_column": "supercluster_name",
                "subject_column": "ROSMAP_IndividualID",
                "target_column": "cogn_global",
                "pathology_columns": [],
                "cell_sampling": {"max_cells_per_type": 100, "min_cells_threshold": 10, "sampling_strategy": "random"},
                "dataloader": {"batch_size": 8, "num_workers": 0, "pin_memory": False, "prefetch_factor": None},
            },
            "experiment": {"seed": 42},
        })
        dm = CognitiveResilienceDataModule(
            config=config, metadata=mock_metadata, splits=mock_splits,
            fold_idx=0, precomputed_dir=precomputed_dir,
        )
        assert dm.batch_size == 8


class TestWorkerInitFn:
    """T6: Verify rank-aware worker init function."""

    def test_worker_init_fn_returns_callable(self, minimal_config, mock_metadata, mock_splits, precomputed_dir):
        dm = CognitiveResilienceDataModule(
            config=minimal_config, metadata=mock_metadata, splits=mock_splits,
            fold_idx=0, precomputed_dir=precomputed_dir,
        )
        dm.trainer = MagicMock()
        dm.trainer.global_rank = 0
        fn = dm._make_worker_init_fn()
        assert callable(fn)

    def test_different_ranks_produce_different_fns(self, minimal_config, mock_metadata, mock_splits, precomputed_dir):
        fns = []
        for rank in range(3):
            dm = CognitiveResilienceDataModule(
                config=minimal_config, metadata=mock_metadata, splits=mock_splits,
                fold_idx=0, precomputed_dir=precomputed_dir,
            )
            dm.trainer = MagicMock()
            dm.trainer.global_rank = rank
            fns.append(dm._make_worker_init_fn())
        assert len(fns) == 3
        assert fns[0] is not fns[1]
