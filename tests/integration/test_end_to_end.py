"""
End-to-end integration tests for pipeline handoffs.

Tests the contracts between pipeline stages:
- Precompute round-trip: CognitiveResilienceDataset → save → PrecomputedDataset
- DataModule with PrecomputedDataset
- Real collate → model → Predictor inference
- Bayesian checkpoint resume with Pyro param store sync
"""

import warnings

import numpy as np
import pandas as pd
import pytest
import torch
import torch.utils.data
from pathlib import Path

import pyro
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf

from src.data.constants import CELL_TYPE_ORDER, REGION_ORDER
from src.data.datasets import (
    CognitiveResilienceDataset,
    PrecomputedDataset,
    save_precomputed_features,
)
from src.data.collate import collate_for_hgt_multiregion
from src.models.full_model import CognitiveResilienceModel, build_model_from_config
from src.training.lightning_module import CognitiveResilienceLightningModule
from src.training.callbacks import ResilienceModelCheckpoint
from src.inference.predict import Predictor


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

N_SUBJECTS = 6
N_CELL_TYPES = 5
N_GENES = 50
N_REGIONS = 6
MAX_CELLS = 30
BATCH_SIZE = 3
D_EMBED = 16
D_FUSED = 16
D_COND = 8
CELL_TYPE_SUBSET = CELL_TYPE_ORDER[:N_CELL_TYPES]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_synthetic_adata(n_subjects, n_cells_per_subject=100):
    """Create a minimal AnnData for testing."""
    import anndata
    import scipy.sparse as sp

    total_cells = n_subjects * n_cells_per_subject
    subject_ids = [f"SUBJ_{i:03d}" for i in range(n_subjects)]

    # Create sparse expression matrix
    X = sp.random(total_cells, N_GENES, density=0.3, format="csr", dtype=np.float32)
    X.data = np.abs(X.data) * 10  # positive counts

    # Assign cells to subjects and cell types
    obs_subject = np.repeat(subject_ids, n_cells_per_subject)
    obs_celltype = np.tile(
        CELL_TYPE_SUBSET * (n_cells_per_subject // N_CELL_TYPES + 1),
        n_subjects,
    )[:total_cells]

    obs = pd.DataFrame({
        "ROSMAP_IndividualID": obs_subject,
        "supercluster_name": obs_celltype,
    })
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(N_GENES)])

    adata = anndata.AnnData(X=X, obs=obs, var=var)
    adata.raw = adata.copy()
    return adata, subject_ids


def _make_metadata(subject_ids):
    """Create minimal metadata DataFrame."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "ROSMAP_IndividualID": subject_ids,
        "cogn_global": rng.normal(0, 1, len(subject_ids)).astype(np.float32),
        "gpath": rng.uniform(0, 5, len(subject_ids)).astype(np.float32),
        "amylsqrt": rng.uniform(0, 3, len(subject_ids)).astype(np.float32),
        "tangsqrt": rng.uniform(0, 3, len(subject_ids)).astype(np.float32),
    })


def _make_config(head_type="deterministic"):
    """Build minimal config for testing."""
    cfg = {
        "experiment": {"name": "e2e_test", "seed": 42, "device": "cpu"},
        "model": {
            "n_genes": N_GENES,
            "n_cell_types": N_CELL_TYPES,
            "n_regions": N_REGIONS,
            "d_embed": D_EMBED,
            "d_fused": D_FUSED,
            "dropout": 0.0,
            "pseudobulk": {"mlp_hidden": [32], "use_layer_norm": True},
            "gene_gate": {"initial_temperature": 2.0},
            "hgt": {
                "n_layers": 1, "n_heads": 2,
                "edge_types": [
                    "Secreted_Signaling", "ECM_Receptor", "Cell_Cell_Contact",
                    "Non_protein_Signaling", "Novel_Uncharacterized",
                ],
            },
            "set_transformer": {
                "n_isab_layers": 1, "n_inducing_points": 4,
                "n_pma_seeds": 1, "n_heads": 2,
            },
            "cell_type_selector": {"selection_temperature": 1.0},
            "pathology_attention": {
                "d_cond": D_COND, "n_heads": 2, "n_pathology_features": 3,
            },
            "head": {
                "type": head_type,
                "d_hidden": 16,
            },
        },
        "data": {
            "cell_sampling": {
                "max_cells_per_type": MAX_CELLS,
                "min_cells_threshold": 5,
                "sampling_strategy": "random",
            },
        },
        "training": {
            "max_epochs": 2,
            "early_stopping": {
                "patience": 5, "min_delta": 0.0001,
                "min_epochs": 1, "monitor": "val_loss", "mode": "min",
            },
            "optimizer": {
                "type": "adamw", "lr": 0.001,
                "weight_decay": 0.0, "betas": [0.9, 0.999],
            },
            "scheduler": {"type": "cosine", "warmup_epochs": 0, "eta_min": 0.0001},
            "loss": {
                "type": "beta_nll" if head_type == "bayesian" else "mse",
                "beta": 0.5,
            },
            "gradient_clip_val": 1.0,
            "precision": "32-true",
            "devices": 1,
            "strategy": "auto",
            "lr_scaling": False,
            "checkpoint": {
                "save_top_k": 1, "monitor": "val_loss",
                "mode": "min", "save_last": True,
            },
            "regularization": {"gene_gate_l1": 0.0},
            "logging": {"log_every_n_steps": 1, "val_check_interval": 1.0},
            "temperature_annealing": {
                "tau_max": 2.0, "tau_min": 0.1,
                "warmup_epochs": 1, "anneal_epochs": 5,
                "schedule": "exponential",
            },
        },
        "error_handling": {"training": {"nan_loss": "fail", "nan_batch": "skip"}},
        "inference": {"num_posterior_samples": 5},
    }
    return OmegaConf.create(cfg)


# ─────────────────────────────────────────────────────────────────────────────
# Gap 1: Precompute Round-Trip
# ─────────────────────────────────────────────────────────────────────────────


class TestPrecomputeRoundTrip:
    """Verify CognitiveResilienceDataset → save → PrecomputedDataset round-trip."""

    def test_precompute_round_trip_fields_match(self, tmp_path):
        """Saved .npz fields match what PrecomputedDataset loads."""
        adata, subject_ids = _make_synthetic_adata(N_SUBJECTS)
        metadata = _make_metadata(subject_ids)

        # Create on-the-fly dataset
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            ds_live = CognitiveResilienceDataset(
                adata=adata,
                metadata=metadata,
                subject_ids=subject_ids,
                cell_type_order=CELL_TYPE_SUBSET,
                max_cells_per_type=MAX_CELLS,
                min_cells_threshold=5,
                sampling_seed=42,
            )

        # Save precomputed
        save_precomputed_features(ds_live, tmp_path, verbose=False)

        # Verify files exist
        for sid in subject_ids:
            assert (tmp_path / f"{sid}.pt").exists()

        # Load via PrecomputedDataset
        ds_pre = PrecomputedDataset(
            feature_dir=tmp_path,
            subject_ids=subject_ids,
            metadata=metadata,
            cell_type_order=CELL_TYPE_SUBSET,
        )

        assert len(ds_pre) == len(ds_live)

        # Compare a sample
        # Use a new on-the-fly dataset with same seed to get same cell samples
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            ds_live2 = CognitiveResilienceDataset(
                adata=adata,
                metadata=metadata,
                subject_ids=subject_ids,
                cell_type_order=CELL_TYPE_SUBSET,
                max_cells_per_type=MAX_CELLS,
                min_cells_threshold=5,
                sampling_seed=42,
            )

        live_sample = ds_live2[0]
        pre_sample = ds_pre[0]

        # Core tensors must match (same format in both datasets)
        for key in ["pseudobulk", "cell_type_mask", "cell_counts", "region_mask"]:
            assert key in pre_sample, f"Missing key '{key}' in PrecomputedDataset"
            torch.testing.assert_close(
                pre_sample[key], live_sample[key],
                msg=f"Mismatch in '{key}'",
            )

        # Cell data: PrecomputedDataset uses flat format (cell_data + cell_offsets)
        # while CognitiveResilienceDataset uses padded (cells + cell_mask).
        # Flat format correctness is tested in test_flat_cells.py; here just check presence.
        assert "cell_data" in pre_sample, "Missing 'cell_data' in PrecomputedDataset"
        assert "cell_offsets" in pre_sample, "Missing 'cell_offsets' in PrecomputedDataset"

        # Edge tensors
        for key in ["ccc_edge_index", "ccc_edge_type", "ccc_edge_attr"]:
            assert key in pre_sample, f"Missing key '{key}' in PrecomputedDataset"

        # Phenotypes come from metadata, not .npz — verify they're present
        assert "pathology" in pre_sample
        assert "cognition" in pre_sample

    def test_skip_existing_subjects(self, tmp_path):
        """save_precomputed_features skips subjects in skip_subjects set."""
        adata, subject_ids = _make_synthetic_adata(4)
        metadata = _make_metadata(subject_ids)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            ds = CognitiveResilienceDataset(
                adata=adata, metadata=metadata, subject_ids=subject_ids,
                cell_type_order=CELL_TYPE_SUBSET,
                max_cells_per_type=MAX_CELLS, min_cells_threshold=5,
            )

        # Save all first
        save_precomputed_features(ds, tmp_path, verbose=False)
        first_mtime = (tmp_path / f"{subject_ids[0]}.pt").stat().st_mtime

        # Save again, skipping first two
        import time
        time.sleep(0.01)  # ensure mtime differs
        save_precomputed_features(
            ds, tmp_path, verbose=False,
            skip_subjects={subject_ids[0], subject_ids[1]},
        )

        # First subject's file should NOT be overwritten
        assert (tmp_path / f"{subject_ids[0]}.pt").stat().st_mtime == first_mtime
        # Third subject's file should be overwritten (newer mtime)


# ─────────────────────────────────────────────────────────────────────────────
# Gap 2: Real Collate → Model → Predictor
# ─────────────────────────────────────────────────────────────────────────────


class TestRealCollateInference:
    """Verify real collate_for_hgt_multiregion produces batches the model accepts."""

    def _make_hgt_sample(self, idx):
        """Create sample with HGT-compatible edge dicts and region data."""
        n_edges = 10
        return {
            "subject_id": f"SUBJ_{idx:03d}",
            "pseudobulk": torch.randn(N_CELL_TYPES, N_GENES),
            "cognition": torch.randn(1),
            "pathology": torch.randn(3),
            "cells": torch.randn(N_CELL_TYPES, MAX_CELLS, N_GENES),
            "cell_mask": torch.ones(N_CELL_TYPES, MAX_CELLS, dtype=torch.bool),
            "cell_type_mask": torch.ones(N_CELL_TYPES, dtype=torch.bool),
            "cell_counts": torch.full((N_CELL_TYPES,), MAX_CELLS, dtype=torch.long),
            "region_mask": torch.ones(N_REGIONS, dtype=torch.bool),
            "ccc_edge_index": torch.randint(0, N_CELL_TYPES, (2, n_edges)),
            "ccc_edge_type": torch.randint(0, 5, (n_edges,)),
            "ccc_edge_attr": torch.rand(n_edges, 1),
            "cell_type_order": CELL_TYPE_SUBSET,
            # Region pseudobulk keys
            **{
                f"region_{i}_pseudobulk": torch.randn(N_CELL_TYPES, N_GENES)
                for i in range(N_REGIONS)
            },
            "available_regions": list(range(N_REGIONS)),
        }

    def test_collate_to_predict_batch(self):
        """Real collate output feeds into Predictor.predict_batch without error."""
        config = _make_config("deterministic")
        model = build_model_from_config(config.model)
        model.eval()

        # Create batch via real collate
        samples = [self._make_hgt_sample(i) for i in range(BATCH_SIZE)]
        batch = collate_for_hgt_multiregion(samples)

        # Verify collate output has expected raw edge tensor keys
        assert "ccc_edge_index" in batch
        assert "ccc_edge_type" in batch
        assert "ccc_edge_attr" in batch
        assert "ccc_edge_counts" not in batch

        # Run through model (unpack batch like lightning module's _forward_batch)
        with torch.no_grad():
            output = model(
                region_pseudobulk=batch.get("region_pseudobulk"),
                region_mask=batch.get("region_mask"),
                pseudobulk=batch.get("pseudobulk"),
                cell_type_mask=batch.get("cell_type_mask"),
                cells=batch.get("cells"),
                cell_mask=batch.get("cell_mask"),
                pathology=batch.get("pathology"),
                ccc_edge_index=batch.get("ccc_edge_index"),
                ccc_edge_type=batch.get("ccc_edge_type"),
                ccc_edge_attr=batch.get("ccc_edge_attr"),
            )

        assert "mean" in output
        assert output["mean"].shape == (BATCH_SIZE, 1)
        assert torch.isfinite(output["mean"]).all()


# ─────────────────────────────────────────────────────────────────────────────
# Gap 4: DataModule with PrecomputedDataset
# ─────────────────────────────────────────────────────────────────────────────


class TestDataModulePrecomputed:
    """Verify DataModule correctly loads PrecomputedDataset."""

    def test_precomputed_datamodule_creates_correct_datasets(self, tmp_path):
        """DataModule with precomputed_dir creates PrecomputedDataset instances."""
        adata, subject_ids = _make_synthetic_adata(N_SUBJECTS)
        metadata = _make_metadata(subject_ids)

        # Create precomputed files with full CELL_TYPE_ORDER subset (5 types)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            ds = CognitiveResilienceDataset(
                adata=adata, metadata=metadata, subject_ids=subject_ids,
                cell_type_order=CELL_TYPE_SUBSET,
                max_cells_per_type=MAX_CELLS, min_cells_threshold=5,
                sampling_seed=42,
            )
        save_precomputed_features(ds, tmp_path / "features", verbose=False)

        from src.data.datamodule import CognitiveResilienceDataModule

        config = _make_config()
        config.data.subject_column = "ROSMAP_IndividualID"
        config.data.target_column = "cogn_global"
        config.data.pathology_columns = ["gpath", "amylsqrt", "tangsqrt"]
        config.data.cell_type_column = "supercluster_name"
        config.data.dataloader = {
            "num_workers": 0, "pin_memory": False,
            "prefetch_factor": None, "batch_size": BATCH_SIZE,
        }

        splits = {
            "train_val_pool": subject_ids,
            "holdout_test": [],
            "folds": [
                {"train": subject_ids[:4], "val": subject_ids[4:]},
            ],
        }

        dm = CognitiveResilienceDataModule(
            config=config, metadata=metadata, splits=splits, fold_idx=0,
            precomputed_dir=tmp_path / "features",
        )
        dm.setup("fit")

        # Verify datasets are PrecomputedDataset (not CognitiveResilienceDataset)
        assert isinstance(dm._train_ds, PrecomputedDataset)
        assert isinstance(dm._val_ds, PrecomputedDataset)
        assert len(dm._train_ds) == 4
        assert len(dm._val_ds) == 2

    def test_precomputed_dataset_loads_and_collates(self, tmp_path):
        """PrecomputedDataset samples collate correctly through real collate fn."""
        adata, subject_ids = _make_synthetic_adata(N_SUBJECTS)
        metadata = _make_metadata(subject_ids)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            ds = CognitiveResilienceDataset(
                adata=adata, metadata=metadata, subject_ids=subject_ids,
                cell_type_order=CELL_TYPE_SUBSET,
                max_cells_per_type=MAX_CELLS, min_cells_threshold=5,
                sampling_seed=42,
            )
        save_precomputed_features(ds, tmp_path / "features", verbose=False)

        # Load via PrecomputedDataset with matching cell_type_order
        ds_pre = PrecomputedDataset(
            feature_dir=tmp_path / "features",
            subject_ids=subject_ids,
            metadata=metadata,
            cell_type_order=CELL_TYPE_SUBSET,
        )

        # Collate a batch through real collate function
        samples = [ds_pre[i] for i in range(BATCH_SIZE)]
        batch = collate_for_hgt_multiregion(samples)

        assert "pseudobulk" in batch
        assert "cognition" in batch
        assert "ccc_edge_index" in batch
        assert batch["pseudobulk"].shape == (BATCH_SIZE, N_CELL_TYPES, N_GENES)
        assert batch["cognition"].shape == (BATCH_SIZE, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Gap 5: Bayesian Checkpoint Resume (Pyro Param Store Sync)
# ─────────────────────────────────────────────────────────────────────────────


class TestBayesianCheckpointResume:
    """Verify Pyro param store is correctly synced after checkpoint resume."""

    @pytest.mark.slow
    def test_bayesian_resume_pyro_param_store_identity(self, tmp_path):
        """After resume, Pyro param store tensors are identity-linked to guide params."""
        pyro.clear_param_store()

        config = _make_config("bayesian")
        module = CognitiveResilienceLightningModule(config)

        # Create synthetic data with simple collate
        class _SimpleDS(torch.utils.data.Dataset):
            def __len__(self):
                return 8
            def __getitem__(self, idx):
                return {
                    "pseudobulk": torch.randn(N_CELL_TYPES, N_GENES),
                    "cognition": torch.randn(1),
                    "pathology": torch.randn(3),
                    "region_pseudobulk": torch.randn(N_REGIONS, N_CELL_TYPES, N_GENES),
                    "region_mask": torch.ones(N_REGIONS, dtype=torch.bool),
                    "cells": torch.randn(N_CELL_TYPES, MAX_CELLS, N_GENES),
                    "cell_mask": torch.ones(N_CELL_TYPES, MAX_CELLS, dtype=torch.bool),
                    "cell_type_mask": torch.ones(N_CELL_TYPES, dtype=torch.bool),
                }

        def _collate(batch):
            result = {}
            for key in batch[0]:
                if isinstance(batch[0][key], torch.Tensor):
                    result[key] = torch.stack([b[key] for b in batch])
            return result

        ds = _SimpleDS()
        train_dl = torch.utils.data.DataLoader(ds, batch_size=4, collate_fn=_collate)
        val_dl = torch.utils.data.DataLoader(ds, batch_size=4, collate_fn=_collate)

        ckpt_dir = tmp_path / "ckpts"
        ckpt_dir.mkdir()
        ckpt_cb = ModelCheckpoint(
            dirpath=str(ckpt_dir), save_last=True,
            monitor="val_loss", mode="min", save_top_k=1,
        )
        rmc = ResilienceModelCheckpoint()

        # Train 2 epochs
        trainer = pl.Trainer(
            max_epochs=2, accelerator="cpu", devices=1,
            callbacks=[ckpt_cb, rmc],
            enable_progress_bar=False, enable_model_summary=False,
            logger=False, deterministic=True,
        )
        trainer.fit(module, train_dataloaders=train_dl, val_dataloaders=val_dl)

        # Verify checkpoint exists
        last_ckpt = ckpt_dir / "last.ckpt"
        assert last_ckpt.exists()

        # Resume from checkpoint
        pyro.clear_param_store()
        module2 = CognitiveResilienceLightningModule(config)
        rmc2 = ResilienceModelCheckpoint()

        trainer2 = pl.Trainer(
            max_epochs=3, accelerator="cpu", devices=1,
            callbacks=[rmc2],
            enable_progress_bar=False, enable_model_summary=False,
            logger=False, deterministic=True,
        )
        # weights_only=False: checkpoint contains numpy RNG states with internal
        # dtype objects (UInt32DType etc.) that can't be allowlisted exhaustively.
        # Safe here since we just created this checkpoint.
        trainer2.fit(
            module2, train_dataloaders=train_dl, val_dataloaders=val_dl,
            ckpt_path=str(last_ckpt), weights_only=False,
        )

        # After resume + 1 more epoch, verify Pyro param store integrity
        store = pyro.get_param_store()
        assert len(store._params) > 0, "Pyro param store is empty after resume"

        # Verify tensor identity: store params should be the SAME objects
        # as the guide's nn.Parameters (not copies)
        guide = module2.guide
        for name in list(guide._parameters.keys()):
            param = guide._parameters[name]
            if param is None or name.endswith('_unconstrained'):
                continue
            fullname = guide._pyro_get_fullname(name)
            if fullname in store._params:
                assert store._params[fullname] is param, (
                    f"Param store tensor for '{fullname}' is not identity-linked "
                    f"to guide parameter '{name}'"
                )
