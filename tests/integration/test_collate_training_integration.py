"""Integration test: real collate_for_hgt_multiregion -> Lightning training_step.

Verifies the full data pipeline from per-sample dicts through the production
collate function into the Lightning module's training_step, including:
- Composite CCC edge key grouping into dict-list format
- Dynamic cell padding across variable-sized samples
- Region pseudobulk assembly from per-region keys
- training_step produces finite loss
"""

import pytest
import torch
import numpy as np
from omegaconf import OmegaConf

from src.data.collate import collate_for_hgt_multiregion
from src.data.constants import CELL_TYPE_ORDER, REGION_ORDER, ALL_EDGE_TYPES
from src.training.lightning_module import CognitiveResilienceLightningModule


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_GENES = 50
N_CELL_TYPES = len(CELL_TYPE_ORDER)
N_REGIONS = len(REGION_ORDER)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sample(
    subject_id: str,
    n_genes: int = N_GENES,
    n_cell_types: int = N_CELL_TYPES,
    n_regions: int = N_REGIONS,
    max_cells: int = 20,
    n_edges: int = 5,
    available_regions: list[int] | None = None,
) -> dict:
    """Create a single sample dict matching PrecomputedDataset output format.

    Parameters
    ----------
    subject_id : str
        Subject identifier.
    n_genes : int
        Number of genes per cell type.
    n_cell_types : int
        Number of cell types (default: all 31).
    n_regions : int
        Number of brain regions (default: 6).
    max_cells : int
        Maximum number of cells per cell type.
    n_edges : int
        Number of CCC edges (0 for no edges).
    available_regions : list[int] | None
        Region indices with data. Defaults to all regions.
    """
    if available_regions is None:
        available_regions = list(range(n_regions))

    sample = {
        "subject_id": subject_id,
        "pseudobulk": torch.randn(n_cell_types, n_genes),
        "cell_type_mask": torch.ones(n_cell_types, dtype=torch.bool),
        "cell_counts": torch.full((n_cell_types,), max_cells, dtype=torch.long),
        "pathology": torch.randn(3),
        "cognition": torch.randn(1),
        "cells": torch.randn(n_cell_types, max_cells, n_genes),
        "cell_mask": torch.ones(n_cell_types, max_cells, dtype=torch.bool),
        "region_mask": torch.zeros(n_regions, dtype=torch.bool),
    }

    # Mark available regions and provide per-region pseudobulk
    for r in available_regions:
        sample["region_mask"][r] = True
        sample[f"region_{r}_pseudobulk"] = torch.randn(n_cell_types, n_genes)

    sample["available_regions"] = available_regions

    # CCC edges
    if n_edges > 0:
        sample["ccc_edge_index"] = torch.randint(0, n_cell_types, (2, n_edges))
        sample["ccc_edge_type"] = torch.randint(0, len(ALL_EDGE_TYPES), (n_edges,))
        sample["ccc_edge_attr"] = torch.rand(n_edges, 1)
    else:
        sample["ccc_edge_index"] = torch.zeros(2, 0, dtype=torch.long)
        sample["ccc_edge_type"] = torch.zeros(0, dtype=torch.long)
        sample["ccc_edge_attr"] = torch.zeros(0, 1)

    return sample


def _make_config(head_type: str = "deterministic") -> OmegaConf:
    """Create OmegaConf config for CognitiveResilienceLightningModule.

    Parameters
    ----------
    head_type : str
        "deterministic" or "bayesian".
    """
    cfg = {
        "experiment": {"name": "collate_integration_test", "seed": 42, "device": "cpu"},
        "model": {
            "n_genes": N_GENES,
            "n_cell_types": N_CELL_TYPES,
            "n_regions": N_REGIONS,
            "d_embed": 16,
            "d_fused": 16,
            "dropout": 0.0,
            "pseudobulk": {
                "mlp_hidden": [32],
                "use_layer_norm": True,
            },
            "gene_gate": {
                "initial_temperature": 2.0,
            },
            "hgt": {
                "n_layers": 1,
                "n_heads": 2,
            },
            "set_transformer": {
                "n_isab_layers": 1,
                "n_inducing_points": 8,
                "n_pma_seeds": 1,
                "n_heads": 2,
            },
            "cell_type_selector": {
                "selection_temperature": 1.0,
            },
            "pathology_attention": {
                "d_cond": 16,
                "n_heads": 2,
                "n_pathology_features": 3,
            },
            "region": {},
            "cell_selector": {},
            "fusion": {},
            "head": {
                "type": head_type,
                "d_hidden": 16,
            },
        },
        "training": {
            "max_epochs": 1,
            "optimizer": {
                "type": "adamw",
                "lr": 1e-3,
                "weight_decay": 0.0,
                "betas": [0.9, 0.999],
            },
            "scheduler": {
                "type": "cosine",
                "warmup_epochs": 0,
                "eta_min": 1e-6,
            },
            "loss": {
                "type": "beta_nll" if head_type == "bayesian" else "mse",
                "beta": 0.5,
            },
            "regularization": {
                "gene_gate_l1": 0.0,
            },
        },
        "error_handling": {
            "training": {
                "nan_loss": "fail",
                "nan_batch": "fail",
            },
        },
    }
    return OmegaConf.create(cfg)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCollateToTrainingStep:
    """Integration tests: collate_for_hgt_multiregion -> training_step."""

    def test_deterministic_forward_produces_finite_loss(self):
        """4 samples -> collate -> training_step -> finite loss."""
        config = _make_config("deterministic")
        module = CognitiveResilienceLightningModule(config)
        module.eval()  # avoid dropout variance

        samples = [_make_sample(f"subj_{i}") for i in range(4)]
        batch = collate_for_hgt_multiregion(samples)

        loss = module.training_step(batch, batch_idx=0)

        assert loss is not None
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    def test_bayesian_forward_produces_finite_loss(self):
        """Bayesian head: configure_optimizers -> training_step -> finite loss."""
        import pyro
        pyro.clear_param_store()

        config = _make_config("bayesian")
        module = CognitiveResilienceLightningModule(config)

        # Prototype the guide (normally done inside configure_optimizers)
        module.configure_optimizers()

        samples = [_make_sample(f"subj_{i}") for i in range(4)]
        batch = collate_for_hgt_multiregion(samples)

        loss = module.training_step(batch, batch_idx=0)

        assert loss is not None
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    def test_collate_edge_dicts_have_tuple_keys(self):
        """Edge dicts should have (src, rel, dst) tuple keys."""
        samples = [_make_sample(f"subj_{i}", n_edges=10) for i in range(4)]
        batch = collate_for_hgt_multiregion(samples)

        edge_index_dict_list = batch["edge_index_dict_list"]
        edge_attr_dict_list = batch["edge_attr_dict_list"]

        assert isinstance(edge_index_dict_list, list)
        assert len(edge_index_dict_list) == 4

        # At least one sample should have edges (all have n_edges=10)
        has_edges = False
        for edge_index_dict in edge_index_dict_list:
            assert isinstance(edge_index_dict, dict)
            for key in edge_index_dict:
                assert isinstance(key, tuple), f"Key should be tuple, got {type(key)}"
                assert len(key) == 3, f"Tuple key should have 3 elements, got {len(key)}"
                src, rel, dst = key
                assert isinstance(src, str)
                assert isinstance(rel, str)
                assert isinstance(dst, str)
                has_edges = True

        assert has_edges, "Expected at least one edge dict to have edges"

    def test_collate_region_pseudobulk_assembled(self):
        """Region pseudobulk assembled for available_regions=[0,2,4]."""
        available = [0, 2, 4]
        samples = [
            _make_sample(f"subj_{i}", available_regions=available)
            for i in range(4)
        ]
        batch = collate_for_hgt_multiregion(samples)

        region_pseudobulk = batch["region_pseudobulk"]
        region_mask = batch["region_mask"]

        # Shape: [batch, n_regions, n_cell_types, n_genes]
        assert region_pseudobulk.shape == (4, N_REGIONS, N_CELL_TYPES, N_GENES)
        assert region_mask.shape == (4, N_REGIONS)

        # Available regions should be True, others False
        for b in range(4):
            for r in range(N_REGIONS):
                if r in available:
                    assert region_mask[b, r].item(), (
                        f"Region {r} should be True for sample {b}"
                    )
                    # Data should be non-zero (random normal)
                    assert region_pseudobulk[b, r].abs().sum() > 0
                else:
                    assert not region_mask[b, r].item(), (
                        f"Region {r} should be False for sample {b}"
                    )

    def test_variable_cell_counts_padded_correctly(self):
        """Samples with max_cells=5 and max_cells=15 padded to 15."""
        s1 = _make_sample("subj_0", max_cells=5)
        s2 = _make_sample("subj_1", max_cells=15)
        batch = collate_for_hgt_multiregion([s1, s2])

        cells = batch["cells"]
        cell_mask = batch["cell_mask"]

        # Padded to max in batch (15)
        assert cells.shape[1] == N_CELL_TYPES
        assert cells.shape[2] == 15, f"Expected max_cells=15, got {cells.shape[2]}"
        assert cell_mask.shape[2] == 15

        # Sample 0 (originally 5 cells): mask True for [:5], False for [5:]
        assert cell_mask[0, :, :5].all(), "First 5 cells should be valid for sample 0"
        assert not cell_mask[0, :, 5:].any(), (
            "Padded positions should be False for sample 0"
        )

        # Sample 1 (originally 15 cells): all True
        assert cell_mask[1, :, :15].all(), "All 15 cells should be valid for sample 1"

    def test_no_edges_still_works(self):
        """Samples with n_edges=0 -> collate -> training_step -> finite loss."""
        config = _make_config("deterministic")
        module = CognitiveResilienceLightningModule(config)
        module.eval()

        samples = [_make_sample(f"subj_{i}", n_edges=0) for i in range(4)]
        batch = collate_for_hgt_multiregion(samples)

        # Verify collate produces empty edge dicts
        for edge_dict in batch["edge_index_dict_list"]:
            assert len(edge_dict) == 0, "Expected empty edge dict for n_edges=0"

        loss = module.training_step(batch, batch_idx=0)

        assert loss is not None
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
