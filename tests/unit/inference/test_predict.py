"""
Tests for src/inference/predict.py.

Test coverage includes:
- PredictionResult dataclass behavior
- Predictor initialization and checkpoint loading
- Batch prediction correctness
- Output format validation
- Schema validation for saved outputs
- Edge cases (single sample, missing data)
"""

import tempfile
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st
from unittest.mock import MagicMock, patch
import torch

from src.data.constants import N_CELL_TYPES
from src.inference.predict import (
    PredictionResult,
    Predictor,
    predict_from_checkpoint,
)
from src.training.lightning_module import CognitiveResilienceLightningModule


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_prediction_result():
    """Sample PredictionResult for testing."""
    n_samples = 10
    np.random.seed(42)
    return PredictionResult(
        subject_ids=[f"subj_{i}" for i in range(n_samples)],
        mean=np.random.randn(n_samples).astype(np.float32),
        std=np.abs(np.random.randn(n_samples).astype(np.float32)) + 0.1,
        actual=np.random.randn(n_samples).astype(np.float32),
        pathology=np.random.rand(n_samples).astype(np.float32),
        attention_weights=np.random.rand(n_samples, 4, N_CELL_TYPES).astype(np.float32),
        gene_gate_weights=np.random.rand(N_CELL_TYPES, 100).astype(np.float32),
        hgt_attention=None,
        metadata={"model_version": "1.0", "config": {"lr": 1e-4}},
    )


@pytest.fixture
def mock_config():
    """Mock configuration for Predictor."""
    from omegaconf import OmegaConf
    return OmegaConf.create({
        "model": {
            "n_genes": 100,
            "n_cell_types": N_CELL_TYPES,
            "d_embed": 128,
            "d_fused": 128,
            "dropout": 0.1,
            "head": {"type": "bayesian", "d_hidden": 64},
            "hgt": {"n_layers": 2, "n_heads": 4},
            "set_transformer": {"n_heads": 4, "n_isab_layers": 2, "n_inducing_points": 32},
            "pathology_attention": {"d_cond": 64, "n_heads": 4},
            "gene_gate": {"initial_temperature": 2.0},
            "cell_type_selector": {"selection_temperature": 1.0},
        }
    })


@pytest.fixture
def mock_model():
    """Mock CognitiveResilienceModel."""
    model = MagicMock()
    model.n_cell_types = N_CELL_TYPES
    model.n_genes = 100

    # Mock forward pass
    def mock_forward(**kwargs):
        # Handle keyword arguments from actual model.forward() call
        batch_size = 4  # Default
        if "pseudobulk" in kwargs and kwargs["pseudobulk"] is not None:
            batch_size = kwargs["pseudobulk"].shape[0]
        elif "region_pseudobulk" in kwargs and kwargs["region_pseudobulk"] is not None:
            batch_size = kwargs["region_pseudobulk"].shape[0]
        output = {
            "mean": torch.randn(batch_size, 1),
            "std": torch.abs(torch.randn(batch_size, 1)) + 0.1,
            "attention_weights": torch.rand(batch_size, 4, N_CELL_TYPES),
        }
        if kwargs.get("return_embeddings"):
            d_embed = 128
            output["embeddings"] = {
                "pseudobulk": torch.randn(batch_size, N_CELL_TYPES, d_embed),
                "hgt": torch.randn(batch_size, N_CELL_TYPES, d_embed),
                "cell": torch.randn(batch_size, N_CELL_TYPES, d_embed),
                "fused": torch.randn(batch_size, N_CELL_TYPES, d_embed),
                "attended": torch.randn(batch_size, d_embed),
            }
        return output

    model.side_effect = mock_forward
    model.__call__ = MagicMock(side_effect=mock_forward)
    model.eval.return_value = model
    model.to.return_value = model

    # Mock gene gate weights
    model.pseudobulk_encoder.gene_gate.get_gate_weights.return_value = torch.rand(N_CELL_TYPES, 100)

    # Mock cell type selection weights
    model.cell_transformer.get_selection_weights.return_value = torch.rand(N_CELL_TYPES)

    return model




# ============================================================================
# PredictionResult Dataclass Tests
# ============================================================================


class TestPredictionResultDataclass:
    """Tests for PredictionResult dataclass."""

    def test_init_minimal(self):
        """PredictionResult can be initialized with required fields."""
        result = PredictionResult(
            subject_ids=["a", "b"],
            mean=np.array([1.0, 2.0]),
            std=np.array([0.1, 0.2]),
            actual=None,
            pathology=np.array([0.5, 0.6]),
            attention_weights=np.zeros((2, 4, 31)),
            gene_gate_weights=np.zeros((31, 100)),
        )
        assert len(result.subject_ids) == 2
        assert result.actual is None

    def test_metadata_defaults_to_empty_dict(self):
        """metadata defaults to empty dict if not provided."""
        result = PredictionResult(
            subject_ids=["a"],
            mean=np.array([1.0]),
            std=np.array([0.1]),
            actual=None,
            pathology=np.array([0.5]),
            attention_weights=np.zeros((1, 4, 31)),
            gene_gate_weights=np.zeros((31, 100)),
        )
        assert result.metadata == {}

    def test_all_fields_present(self, sample_prediction_result):
        """All expected fields are present in PredictionResult."""
        expected_fields = {
            "subject_ids", "mean", "std", "actual", "pathology",
            "attention_weights", "gene_gate_weights", "cell_type_selection",
            "hgt_attention", "pma_attention", "region_weights",
            "region_pseudobulk_mean", "per_subject_pseudobulk",
            "region_attention", "cell_barcodes",
            "cell_counts", "gene_names", "epistemic_std", "aleatoric_std",
            "embeddings", "metadata"
        }
        actual_fields = {f.name for f in fields(sample_prediction_result)}
        assert expected_fields == actual_fields


# ============================================================================
# Predictor Class Tests
# ============================================================================


class TestPredictor:
    """Tests for Predictor class."""

    def test_init_sets_model_to_eval(self, mock_model, mock_config):
        """Predictor sets model to eval mode."""
        predictor = Predictor(mock_model, config=mock_config, device="cpu")
        mock_model.eval.assert_called()

    def test_init_moves_model_to_device(self, mock_model, mock_config):
        """Predictor moves model to specified device."""
        predictor = Predictor(mock_model, config=mock_config, device="cpu")
        mock_model.to.assert_called_with(torch.device("cpu"))

    def test_predict_batch_returns_dict(self, mock_model, mock_config):
        """predict_batch returns dictionary with expected keys."""
        predictor = Predictor(mock_model, config=mock_config, device="cpu")

        batch = {
            "pseudobulk": torch.randn(4, N_CELL_TYPES, 100),
            "subject_ids": ["a", "b", "c", "d"],
        }

        result = predictor.predict_batch(batch)
        assert "mean" in result
        assert "attention_weights" in result

    def test_predict_batch_shapes(self, mock_model, mock_config):
        """predict_batch returns correct shapes."""
        predictor = Predictor(mock_model, config=mock_config, device="cpu")

        batch = {
            "pseudobulk": torch.randn(4, N_CELL_TYPES, 100),
            "subject_ids": ["a", "b", "c", "d"],
        }

        result = predictor.predict_batch(batch)
        assert result["mean"].shape == (4, 1)

    def test_predict_returns_prediction_result(self, mock_model, mock_config):
        """predict returns PredictionResult instance."""
        predictor = Predictor(mock_model, config=mock_config, device="cpu")

        # Mock dataloader
        batch1 = {
            "pseudobulk": torch.randn(2, N_CELL_TYPES, 100),
            "subject_ids": ["a", "b"],
            "pathology": torch.rand(2, 3),
        }
        batch2 = {
            "pseudobulk": torch.randn(2, N_CELL_TYPES, 100),
            "subject_ids": ["c", "d"],
            "pathology": torch.rand(2, 3),
        }
        dataloader = [batch1, batch2]

        result = predictor.predict(dataloader, show_progress=False)
        assert isinstance(result, PredictionResult)
        assert len(result.subject_ids) == 4

    def test_predict_preserves_subject_order(self, mock_model, mock_config):
        """predict preserves subject ID order from dataloader."""
        predictor = Predictor(mock_model, config=mock_config, device="cpu")

        batch = {
            "pseudobulk": torch.randn(3, N_CELL_TYPES, 100),
            "subject_ids": ["z", "a", "m"],
            "pathology": torch.rand(3, 3),
        }
        dataloader = [batch]

        result = predictor.predict(dataloader, show_progress=False)
        assert result.subject_ids == ["z", "a", "m"]

    def test_from_checkpoint_handles_model_state_dict_key(self, tmp_path, mock_config):
        """from_checkpoint() should handle checkpoints saved with model_state_dict key."""
        from src.models.full_model import CognitiveResilienceModel

        # Build a real model from mock_config so we can get its state_dict
        model_cfg = mock_config.model
        model = CognitiveResilienceModel(
            n_genes=model_cfg.n_genes,
            n_cell_types=model_cfg.n_cell_types,
            d_embed=model_cfg.d_embed,
            d_fused=model_cfg.d_fused,
            d_cond=model_cfg.pathology_attention.d_cond,
            n_hgt_layers=model_cfg.hgt.n_layers,
            n_hgt_heads=model_cfg.hgt.n_heads,
            n_cell_transformer_heads=model_cfg.set_transformer.get("n_heads", 4),
            n_isab_layers=model_cfg.set_transformer.n_isab_layers,
            n_inducing_points=model_cfg.set_transformer.n_inducing_points,
            n_attention_heads=model_cfg.pathology_attention.n_heads,
            gene_gate_temperature=model_cfg.gene_gate.get("initial_temperature", 2.0),
            selection_temperature=model_cfg.cell_type_selector.get("selection_temperature", 1.0),
            use_bayesian_head=(model_cfg.head.type == "bayesian"),
            d_head_hidden=model_cfg.head.d_hidden,
            dropout=model_cfg.get("dropout", 0.1),
        )

        # Save checkpoint using model_state_dict key (matching io.save_checkpoint format)
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_config": dict(model_cfg),
            "epoch": 0,
            "optimizer_state_dict": {},
        }
        ckpt_path = tmp_path / "test_ckpt.pt"
        torch.save(checkpoint, ckpt_path)

        # Load via from_checkpoint — should not fail
        predictor = Predictor.from_checkpoint(ckpt_path, device="cpu", config=mock_config)
        assert predictor is not None
        assert predictor.model is not None


class TestFromCheckpointConfigRecovery:
    """Test config recovery from checkpoint (full_config vs model_config)."""

    def test_full_config_recovery(self, tmp_path, mock_config):
        """from_checkpoint recovers full_config when present in checkpoint."""
        from omegaconf import OmegaConf
        from src.models.full_model import CognitiveResilienceModel

        model_cfg = mock_config.model
        model = CognitiveResilienceModel(
            n_genes=model_cfg.n_genes,
            n_cell_types=model_cfg.n_cell_types,
            d_embed=model_cfg.d_embed,
            d_fused=model_cfg.d_fused,
            d_cond=model_cfg.pathology_attention.d_cond,
            n_hgt_layers=model_cfg.hgt.n_layers,
            n_hgt_heads=model_cfg.hgt.n_heads,
            n_cell_transformer_heads=model_cfg.set_transformer.get("n_heads", 4),
            n_isab_layers=model_cfg.set_transformer.n_isab_layers,
            n_inducing_points=model_cfg.set_transformer.n_inducing_points,
            n_attention_heads=model_cfg.pathology_attention.n_heads,
            gene_gate_temperature=model_cfg.gene_gate.get("initial_temperature", 2.0),
            selection_temperature=model_cfg.cell_type_selector.get("selection_temperature", 1.0),
            use_bayesian_head=(model_cfg.head.type == "bayesian"),
            d_head_hidden=model_cfg.head.d_hidden,
            dropout=model_cfg.get("dropout", 0.1),
        )

        # Save checkpoint with full_config (new v1.1+ format)
        full_config_dict = OmegaConf.to_container(mock_config, resolve=True)
        full_config_dict["data"] = {"adata_path": "/fake/path.h5ad"}
        full_config_dict["training"] = {"loss": {"type": "beta_nll", "beta": 0.5}}

        checkpoint = {
            "state_dict": {"model." + k: v for k, v in model.state_dict().items()},
            "full_config": full_config_dict,
            "model_config": full_config_dict["model"],
        }
        ckpt_path = tmp_path / "full_config_ckpt.pt"
        torch.save(checkpoint, ckpt_path)

        # Load WITHOUT providing config — should recover from full_config
        predictor = Predictor.from_checkpoint(ckpt_path, device="cpu")
        assert predictor is not None
        assert predictor.model is not None
        # full_config should include data section
        assert hasattr(predictor.config, "data")
        assert predictor.config.data.adata_path == "/fake/path.h5ad"

    def test_legacy_model_config_fallback(self, tmp_path, mock_config):
        """from_checkpoint falls back to model_config when full_config absent."""
        from omegaconf import OmegaConf
        from src.models.full_model import CognitiveResilienceModel

        model_cfg = mock_config.model
        model = CognitiveResilienceModel(
            n_genes=model_cfg.n_genes,
            n_cell_types=model_cfg.n_cell_types,
            d_embed=model_cfg.d_embed,
            d_fused=model_cfg.d_fused,
            d_cond=model_cfg.pathology_attention.d_cond,
            n_hgt_layers=model_cfg.hgt.n_layers,
            n_hgt_heads=model_cfg.hgt.n_heads,
            n_cell_transformer_heads=model_cfg.set_transformer.get("n_heads", 4),
            n_isab_layers=model_cfg.set_transformer.n_isab_layers,
            n_inducing_points=model_cfg.set_transformer.n_inducing_points,
            n_attention_heads=model_cfg.pathology_attention.n_heads,
            gene_gate_temperature=model_cfg.gene_gate.get("initial_temperature", 2.0),
            selection_temperature=model_cfg.cell_type_selector.get("selection_temperature", 1.0),
            use_bayesian_head=(model_cfg.head.type == "bayesian"),
            d_head_hidden=model_cfg.head.d_hidden,
            dropout=model_cfg.get("dropout", 0.1),
        )

        # Save checkpoint with ONLY model_config (legacy format, no full_config)
        model_config_dict = OmegaConf.to_container(model_cfg, resolve=True)
        checkpoint = {
            "state_dict": {"model." + k: v for k, v in model.state_dict().items()},
            "model_config": model_config_dict,
        }
        ckpt_path = tmp_path / "legacy_ckpt.pt"
        torch.save(checkpoint, ckpt_path)

        # Load WITHOUT providing config — should fall back to model_config
        predictor = Predictor.from_checkpoint(ckpt_path, device="cpu")
        assert predictor is not None
        assert predictor.model is not None
        # Legacy path wraps under {"model": ...}, should NOT have data section
        assert not hasattr(predictor.config, "data") or predictor.config.get("data") is None


# ============================================================================
# Bayesian Device Migration Tests
# ============================================================================


class TestFromCheckpointBayesianDeviceMigration:
    """Tests for Pyro param store device migration in from_checkpoint."""

    def test_from_checkpoint_bayesian_device_migration(self, tmp_path, mock_config):
        """Pyro param store tensors are migrated to target device in from_checkpoint."""
        import pyro
        from omegaconf import OmegaConf
        from src.models.full_model import CognitiveResilienceModel

        model_cfg = mock_config.model
        model = CognitiveResilienceModel(
            n_genes=model_cfg.n_genes,
            n_cell_types=model_cfg.n_cell_types,
            d_embed=model_cfg.d_embed,
            d_fused=model_cfg.d_fused,
            d_cond=model_cfg.pathology_attention.d_cond,
            n_hgt_layers=model_cfg.hgt.n_layers,
            n_hgt_heads=model_cfg.hgt.n_heads,
            n_cell_transformer_heads=model_cfg.set_transformer.get("n_heads", 4),
            n_isab_layers=model_cfg.set_transformer.n_isab_layers,
            n_inducing_points=model_cfg.set_transformer.n_inducing_points,
            n_attention_heads=model_cfg.pathology_attention.n_heads,
            gene_gate_temperature=model_cfg.gene_gate.get("initial_temperature", 2.0),
            selection_temperature=model_cfg.cell_type_selector.get("selection_temperature", 1.0),
            use_bayesian_head=(model_cfg.head.type == "bayesian"),
            d_head_hidden=model_cfg.head.d_hidden,
            dropout=model_cfg.get("dropout", 0.1),
        )

        # Create a guide and prototype it to populate param store
        from pyro.infer.autoguide import AutoDiagonalNormal
        pyro.clear_param_store()
        guide = AutoDiagonalNormal(model)

        # Prototype the guide
        n_ct = model_cfg.n_cell_types
        n_genes = model_cfg.n_genes
        n_regions = model_cfg.get("n_regions", 6)
        dummy_kwargs = {
            "region_pseudobulk": torch.zeros(1, n_regions, n_ct, n_genes),
            "region_mask": torch.ones(1, n_regions, dtype=torch.bool),
            "cells": torch.zeros(1, n_ct, 1, n_genes),
            "cell_mask": torch.ones(1, n_ct, 1, dtype=torch.bool),
            "pathology": torch.zeros(1, 3),
            "cognition": torch.zeros(1, 1),
        }
        try:
            guide(**dummy_kwargs)
        except Exception:
            pass

        # Save param store values as CPU tensors
        param_store_cpu = {
            k: v.detach().cpu()
            for k, v in pyro.get_param_store().items()
        }

        # Save checkpoint with pyro_param_store and guide_state_dict
        checkpoint = {
            "state_dict": {"model." + k: v for k, v in model.state_dict().items()},
            "model_config": OmegaConf.to_container(model_cfg, resolve=True),
            "full_config": OmegaConf.to_container(mock_config, resolve=True),
            "guide_state_dict": guide.state_dict(),
            "pyro_param_store": param_store_cpu,
        }
        ckpt_path = tmp_path / "bayesian_ckpt.pt"
        torch.save(checkpoint, ckpt_path)

        # Clear param store before loading
        pyro.clear_param_store()

        # Load from checkpoint with device="cpu"
        predictor = Predictor.from_checkpoint(ckpt_path, device="cpu", config=mock_config)
        assert predictor is not None

        # Verify param store tensors are on CPU
        store = pyro.get_param_store()
        for k in param_store_cpu:
            assert store[k].device == torch.device("cpu"), (
                f"Param {k} should be on cpu after from_checkpoint, got {store[k].device}"
            )

    def test_guide_prototype_uses_config_pathology_dim(self, tmp_path):
        """Guide prototype reads n_pathology_features from config (not hardcoded 3)."""
        import pyro
        from omegaconf import OmegaConf
        from src.models.full_model import CognitiveResilienceModel

        # Config with n_pathology_features=5 (non-default)
        config = OmegaConf.create({
            "model": {
                "n_genes": 100,
                "n_cell_types": N_CELL_TYPES,
                "d_embed": 128,
                "d_fused": 128,
                "dropout": 0.1,
                "head": {"type": "bayesian", "d_hidden": 64},
                "hgt": {"n_layers": 2, "n_heads": 4},
                "set_transformer": {"n_heads": 4, "n_isab_layers": 2, "n_inducing_points": 32},
                "pathology_attention": {"d_cond": 64, "n_heads": 4, "n_pathology_features": 5},
                "gene_gate": {"initial_temperature": 2.0},
                "cell_type_selector": {"selection_temperature": 1.0},
            }
        })
        model_cfg = config.model

        model = CognitiveResilienceModel(
            n_genes=model_cfg.n_genes,
            n_cell_types=model_cfg.n_cell_types,
            d_embed=model_cfg.d_embed,
            d_fused=model_cfg.d_fused,
            d_cond=model_cfg.pathology_attention.d_cond,
            n_hgt_layers=model_cfg.hgt.n_layers,
            n_hgt_heads=model_cfg.hgt.n_heads,
            n_cell_transformer_heads=model_cfg.set_transformer.get("n_heads", 4),
            n_isab_layers=model_cfg.set_transformer.n_isab_layers,
            n_inducing_points=model_cfg.set_transformer.n_inducing_points,
            n_attention_heads=model_cfg.pathology_attention.n_heads,
            n_pathology_features=model_cfg.pathology_attention.n_pathology_features,
            gene_gate_temperature=model_cfg.gene_gate.get("initial_temperature", 2.0),
            selection_temperature=model_cfg.cell_type_selector.get("selection_temperature", 1.0),
            use_bayesian_head=True,
            d_head_hidden=model_cfg.head.d_hidden,
            dropout=model_cfg.get("dropout", 0.1),
        )

        # Create guide, prototype, and save checkpoint
        from pyro.infer.autoguide import AutoDiagonalNormal
        pyro.clear_param_store()
        guide = AutoDiagonalNormal(model)

        n_ct = model_cfg.n_cell_types
        n_genes = model_cfg.n_genes
        n_regions = model_cfg.get("n_regions", 6)
        dummy_kwargs = {
            "region_pseudobulk": torch.zeros(1, n_regions, n_ct, n_genes),
            "region_mask": torch.ones(1, n_regions, dtype=torch.bool),
            "cells": torch.zeros(1, n_ct, 1, n_genes),
            "cell_mask": torch.ones(1, n_ct, 1, dtype=torch.bool),
            "pathology": torch.zeros(1, 5),  # 5 pathology features
            "cognition": torch.zeros(1, 1),
        }
        try:
            guide(**dummy_kwargs)
        except Exception:
            pass

        param_store_cpu = {
            k: v.detach().cpu()
            for k, v in pyro.get_param_store().items()
        }

        checkpoint = {
            "state_dict": {"model." + k: v for k, v in model.state_dict().items()},
            "model_config": OmegaConf.to_container(model_cfg, resolve=True),
            "full_config": OmegaConf.to_container(config, resolve=True),
            "guide_state_dict": guide.state_dict(),
            "pyro_param_store": param_store_cpu,
        }
        ckpt_path = tmp_path / "pathology5_ckpt.pt"
        torch.save(checkpoint, ckpt_path)

        pyro.clear_param_store()

        # Load from checkpoint — should NOT crash despite n_pathology_features=5
        predictor = Predictor.from_checkpoint(ckpt_path, device="cpu", config=config)
        assert predictor is not None
        assert predictor.model is not None
        # Model's pathology encoder should have 5 features
        assert predictor.model.pathology_encoder.n_pathology_features == 5


# ============================================================================
# Serialization Tests
# ============================================================================




# ============================================================================
# Property-Based Tests (Hypothesis)
# ============================================================================


class TestPropertyBased:
    """Property-based tests using Hypothesis."""

    @given(n_samples=st.integers(min_value=1, max_value=50))
    @settings(max_examples=20)
    def test_prediction_result_length_consistency(self, n_samples):
        """All array fields have consistent length with subject_ids."""
        result = PredictionResult(
            subject_ids=[f"s{i}" for i in range(n_samples)],
            mean=np.random.randn(n_samples).astype(np.float32),
            std=np.abs(np.random.randn(n_samples)).astype(np.float32) + 0.01,
            actual=np.random.randn(n_samples).astype(np.float32),
            pathology=np.random.rand(n_samples).astype(np.float32),
            attention_weights=np.random.rand(n_samples, 4, 31).astype(np.float32),
            gene_gate_weights=np.random.rand(31, 100).astype(np.float32),
        )

        assert len(result.subject_ids) == n_samples
        assert len(result.mean) == n_samples
        assert len(result.std) == n_samples
        assert result.attention_weights.shape[0] == n_samples



# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Edge case tests."""

    def test_single_sample_prediction(self, mock_model, mock_config):
        """Handles single sample prediction."""
        predictor = Predictor(mock_model, config=mock_config, device="cpu")

        batch = {
            "pseudobulk": torch.randn(1, N_CELL_TYPES, 100),
            "subject_ids": ["single"],
            "pathology": torch.rand(1, 3),
        }
        dataloader = [batch]

        result = predictor.predict(dataloader, show_progress=False)
        assert len(result.subject_ids) == 1
        assert result.mean.shape[0] == 1

    def test_prediction_without_actual_values(self, mock_model, mock_config):
        """Handles prediction without actual target values."""
        predictor = Predictor(mock_model, config=mock_config, device="cpu")

        batch = {
            "pseudobulk": torch.randn(3, N_CELL_TYPES, 100),
            "subject_ids": ["a", "b", "c"],
            "pathology": torch.rand(3, 3),
            # No "cognition" key
        }
        dataloader = [batch]

        result = predictor.predict(dataloader, show_progress=False)
        assert result.actual is None



# ============================================================================
# Region Pseudobulk Masking Tests
# ============================================================================


class TestRegionPseudobulkMasking:
    """Tests for region_pseudobulk_mean masked computation (Finding 1)."""

    def test_masked_mean_ignores_zero_padded_regions(self):
        """region_pseudobulk_mean should ignore regions where region_mask is False."""
        # Simulate 3 subjects, 4 regions, 2 cell types, 3 genes
        # Subject 0: only regions 0,1 valid. Subject 1: all 4. Subject 2: only region 0.
        stacked = np.zeros((3, 4, 2, 3), dtype=np.float32)
        stacked_mask = np.zeros((3, 4), dtype=bool)

        # Subject 0: regions 0,1 valid with value 1.0
        stacked[0, 0] = 1.0
        stacked[0, 1] = 1.0
        stacked_mask[0, :2] = True

        # Subject 1: all regions valid with value 2.0
        stacked[1] = 2.0
        stacked_mask[1] = True

        # Subject 2: only region 0 valid with value 3.0
        stacked[2, 0] = 3.0
        stacked_mask[2, 0] = True

        # Compute masked mean (same logic as predict.py)
        mask_expanded = stacked_mask[:, :, np.newaxis, np.newaxis]
        masked = np.where(mask_expanded, stacked, np.nan)
        result = np.nanmean(masked, axis=0)  # [R, C, G]
        result = np.nan_to_num(result, nan=0.0)

        # Region 0: all 3 subjects have it -> mean of 1.0, 2.0, 3.0 = 2.0
        np.testing.assert_allclose(result[0], 2.0)
        # Region 1: subjects 0,1 have it -> mean of 1.0, 2.0 = 1.5
        np.testing.assert_allclose(result[1], 1.5)
        # Region 2: only subject 1 has it -> 2.0
        np.testing.assert_allclose(result[2], 2.0)
        # Region 3: only subject 1 has it -> 2.0
        np.testing.assert_allclose(result[3], 2.0)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_all_masked_region_gets_zero(self):
        """A region masked for ALL subjects should get 0.0 (not NaN)."""
        stacked = np.ones((2, 3, 2, 2), dtype=np.float32)
        stacked_mask = np.zeros((2, 3), dtype=bool)
        stacked_mask[:, :2] = True  # Only first 2 regions valid

        mask_expanded = stacked_mask[:, :, np.newaxis, np.newaxis]
        masked = np.where(mask_expanded, stacked, np.nan)
        result = np.nanmean(masked, axis=0)
        result = np.nan_to_num(result, nan=0.0)

        # Region 2 should be 0.0 (all masked)
        np.testing.assert_allclose(result[2], 0.0)
        # Regions 0,1 should be 1.0
        np.testing.assert_allclose(result[0], 1.0)
        np.testing.assert_allclose(result[1], 1.0)

    def test_unmasked_mean_would_be_biased(self):
        """Without masking, zeros from padded regions bias the mean downward."""
        # Subject 0: region 1 valid (10.0), region 0 padded (0.0)
        # Subject 1: both regions valid (10.0)
        stacked = np.zeros((2, 2, 1, 1), dtype=np.float32)
        stacked_mask = np.zeros((2, 2), dtype=bool)

        stacked[0, 1] = 10.0
        stacked_mask[0, 1] = True
        stacked[1, :] = 10.0
        stacked_mask[1, :] = True

        # Unmasked mean for region 0: (0 + 10) / 2 = 5.0 — biased!
        # (Subject 0 has 0.0 because region 0 is padded)
        unmasked = stacked.mean(axis=0)
        assert unmasked[0, 0, 0] == 5.0  # biased downward

        # Masked mean for region 0: only subject 1 → 10.0
        mask_expanded = stacked_mask[:, :, np.newaxis, np.newaxis]
        masked = np.where(mask_expanded, stacked, np.nan)
        correct = np.nanmean(masked, axis=0)
        correct = np.nan_to_num(correct, nan=0.0)
        assert correct[0, 0, 0] == 10.0  # correct


# ============================================================================
# Single-Region Fallback Tests (Phase 6 Review Round 5)
# ============================================================================


class TestSingleRegionFallback:
    """Test that single-region models populate region_pseudobulk_mean."""

    def test_pseudobulk_without_region_creates_single_region(self):
        """When batch has pseudobulk but no region_pseudobulk, fallback creates [B,1,C,G]."""
        # Simulate what predict() does with the batch loop logic
        n_subjects = 4
        n_cell_types = 3
        n_genes = 10

        # Create batch with pseudobulk but NO region_pseudobulk
        pseudobulk = torch.randn(n_subjects, n_cell_types, n_genes)

        all_region_pseudobulk = []
        all_region_mask = []

        # Replicate the predict() logic
        batch = {"pseudobulk": pseudobulk}

        if "region_pseudobulk" in batch:
            pass  # Not present
        elif not all_region_pseudobulk and "pseudobulk" in batch:
            pb = batch["pseudobulk"]
            if isinstance(pb, torch.Tensor):
                pb = pb.cpu().numpy()
            all_region_pseudobulk.append(pb[:, np.newaxis, :, :])
            all_region_mask.append(np.ones((pb.shape[0], 1), dtype=bool))

        assert len(all_region_pseudobulk) == 1
        assert all_region_pseudobulk[0].shape == (n_subjects, 1, n_cell_types, n_genes)
        assert all_region_mask[0].shape == (n_subjects, 1)
        assert all_region_mask[0].all()

    def test_single_region_mean_computation(self):
        """Single-region fallback should produce valid region_pseudobulk_mean."""
        n_subjects = 4
        n_cell_types = 3
        n_genes = 10

        # Simulated single-region data
        stacked = np.random.rand(n_subjects, 1, n_cell_types, n_genes)
        stacked_mask = np.ones((n_subjects, 1), dtype=bool)

        mask_expanded = stacked_mask[:, :, np.newaxis, np.newaxis]
        masked = np.where(mask_expanded, stacked, np.nan)
        region_pseudobulk_mean = np.nanmean(masked, axis=0)
        region_pseudobulk_mean = np.nan_to_num(region_pseudobulk_mean, nan=0.0)

        assert region_pseudobulk_mean.shape == (1, n_cell_types, n_genes)
        # With all-valid mask, should equal simple mean
        np.testing.assert_allclose(
            region_pseudobulk_mean[0],
            stacked[:, 0, :, :].mean(axis=0),
            rtol=1e-5,
        )


# ============================================================================
# Subject ID Key Tests (Phase 6 Review Round 6 — F1)
# ============================================================================


class TestSubjectIdExtraction:
    """Tests for subject ID extraction from batch dicts."""

    def test_predict_batch_extracts_subject_ids(self, mock_model, mock_config):
        """predict_batch extracts subject_ids from batch using plural key."""
        predictor = Predictor(mock_model, config=mock_config, device="cpu")

        batch = {
            "pseudobulk": torch.randn(3, N_CELL_TYPES, 100),
            "subject_ids": ["subj_A", "subj_B", "subj_C"],
            "pathology": torch.rand(3, 3),
        }

        result = predictor.predict_batch(batch)
        assert "subject_ids" in result
        assert result["subject_ids"] == ["subj_A", "subj_B", "subj_C"]

    def test_predict_populates_subject_ids_from_dataloader(self, mock_model, mock_config):
        """predict() populates PredictionResult.subject_ids from multi-batch dataloader."""
        predictor = Predictor(mock_model, config=mock_config, device="cpu")

        batch1 = {
            "pseudobulk": torch.randn(2, N_CELL_TYPES, 100),
            "subject_ids": ["s1", "s2"],
            "pathology": torch.rand(2, 3),
        }
        batch2 = {
            "pseudobulk": torch.randn(2, N_CELL_TYPES, 100),
            "subject_ids": ["s3", "s4"],
            "pathology": torch.rand(2, 3),
        }
        dataloader = [batch1, batch2]

        result = predictor.predict(dataloader, show_progress=False)
        assert result.subject_ids == ["s1", "s2", "s3", "s4"]

    def test_predict_batch_without_subject_ids(self, mock_model, mock_config):
        """predict_batch gracefully handles missing subject_ids key."""
        predictor = Predictor(mock_model, config=mock_config, device="cpu")

        batch = {
            "pseudobulk": torch.randn(2, N_CELL_TYPES, 100),
            "pathology": torch.rand(2, 3),
        }

        result = predictor.predict_batch(batch)
        assert "subject_ids" not in result


# ============================================================================
# Embedding Pipeline Tests (Phase 6 Review Round 6 — F2)
# ============================================================================


class TestEmbeddingPipeline:
    """Tests for embedding extraction pipeline."""

    def test_predict_batch_extracts_embeddings(self, mock_model, mock_config):
        """predict_batch with extract_embeddings=True returns embeddings dict."""
        predictor = Predictor(mock_model, config=mock_config, device="cpu")

        batch = {
            "pseudobulk": torch.randn(3, N_CELL_TYPES, 100),
            "subject_ids": ["a", "b", "c"],
            "pathology": torch.rand(3, 3),
        }

        result = predictor.predict_batch(batch, extract_embeddings=True)
        assert "embeddings" in result
        assert set(result["embeddings"].keys()) == {"pseudobulk", "hgt", "cell", "fused", "attended"}
        # 3D branch embeddings
        assert result["embeddings"]["pseudobulk"].shape == (3, N_CELL_TYPES, 128)
        # 2D attended embedding
        assert result["embeddings"]["attended"].shape == (3, 128)

    def test_predict_batch_no_embeddings_by_default(self, mock_model, mock_config):
        """predict_batch without extract_embeddings does not include embeddings."""
        predictor = Predictor(mock_model, config=mock_config, device="cpu")

        batch = {
            "pseudobulk": torch.randn(2, N_CELL_TYPES, 100),
            "pathology": torch.rand(2, 3),
        }

        result = predictor.predict_batch(batch)
        assert "embeddings" not in result

    def test_predict_accumulates_embeddings(self, mock_model, mock_config):
        """predict() concatenates embeddings across batches."""
        predictor = Predictor(mock_model, config=mock_config, device="cpu")

        batch1 = {
            "pseudobulk": torch.randn(2, N_CELL_TYPES, 100),
            "subject_ids": ["s1", "s2"],
            "pathology": torch.rand(2, 3),
        }
        batch2 = {
            "pseudobulk": torch.randn(3, N_CELL_TYPES, 100),
            "subject_ids": ["s3", "s4", "s5"],
            "pathology": torch.rand(3, 3),
        }

        result = predictor.predict(
            [batch1, batch2], extract_embeddings=True, show_progress=False,
        )
        assert result.embeddings is not None
        assert result.embeddings["pseudobulk"].shape[0] == 5  # 2 + 3
        assert result.embeddings["attended"].shape[0] == 5

    def test_predict_embeddings_none_by_default(self, mock_model, mock_config):
        """predict() without extract_embeddings has embeddings=None."""
        predictor = Predictor(mock_model, config=mock_config, device="cpu")

        batch = {
            "pseudobulk": torch.randn(2, N_CELL_TYPES, 100),
            "subject_ids": ["s1", "s2"],
            "pathology": torch.rand(2, 3),
        }

        result = predictor.predict([batch], show_progress=False)
        assert result.embeddings is None


class TestEmbeddingsHDF5Roundtrip:
    """Tests for embedding HDF5 serialization."""

    def test_embeddings_save_load_roundtrip(self):
        """Embeddings round-trip through HDF5 save/load."""
        from src.utils.io import save_attention_weights, load_attention_weights

        embeddings = {
            "pseudobulk": np.random.rand(10, N_CELL_TYPES, 128).astype(np.float32),
            "hgt": np.random.rand(10, N_CELL_TYPES, 128).astype(np.float32),
            "cell": np.random.rand(10, N_CELL_TYPES, 128).astype(np.float32),
            "fused": np.random.rand(10, N_CELL_TYPES, 128).astype(np.float32),
            "attended": np.random.rand(10, 128).astype(np.float32),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "attn.h5"
            save_attention_weights(path, embeddings=embeddings)
            loaded = load_attention_weights(path)

        assert "embeddings" in loaded
        for name in ["pseudobulk", "hgt", "cell", "fused", "attended"]:
            np.testing.assert_array_almost_equal(
                loaded["embeddings"][name], embeddings[name]
            )

    def test_embeddings_shapes_preserved(self):
        """3D and 2D embedding shapes are preserved through save/load."""
        from src.utils.io import save_attention_weights, load_attention_weights

        embeddings = {
            "branch_3d": np.random.rand(5, 10, 64).astype(np.float32),
            "subject_2d": np.random.rand(5, 64).astype(np.float32),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "attn.h5"
            save_attention_weights(path, embeddings=embeddings)
            loaded = load_attention_weights(path)

        assert loaded["embeddings"]["branch_3d"].shape == (5, 10, 64)
        assert loaded["embeddings"]["subject_2d"].shape == (5, 64)


# ============================================================================
# Cell Type Selection Tests (F5)
# ============================================================================


class TestCellTypeSelection:
    """Tests for cell_type_selection extraction and persistence."""

    def test_cell_type_selection_in_prediction_result(self, mock_model, mock_config):
        """predict() populates cell_type_selection with correct shape."""
        predictor = Predictor(mock_model, config=mock_config, device="cpu")

        batch = {
            "pseudobulk": torch.randn(3, N_CELL_TYPES, 100),
            "subject_ids": ["a", "b", "c"],
            "pathology": torch.rand(3, 3),
        }
        dataloader = [batch]

        result = predictor.predict(dataloader, show_progress=False)
        assert result.cell_type_selection is not None
        assert result.cell_type_selection.shape == (N_CELL_TYPES,)

    def test_cell_type_selection_persisted_hdf5(self):
        """cell_type_selection survives HDF5 save/load roundtrip."""
        from src.utils.io import save_attention_weights, load_attention_weights

        selection = np.random.rand(N_CELL_TYPES).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "attn.h5"
            save_attention_weights(path, cell_type_selection=selection)
            loaded = load_attention_weights(path)

        assert "cell_type_selection" in loaded
        np.testing.assert_array_almost_equal(
            loaded["cell_type_selection"], selection
        )


# ============================================================================
# Regional Attention Default Tests (F4)
# ============================================================================


class TestPredictFromCheckpointDefaults:
    """Tests for predict_from_checkpoint default parameter values."""

    def test_predict_from_checkpoint_defaults_region_attention_true(self):
        """predict_from_checkpoint has extract_region_attention=True by default."""
        import inspect
        sig = inspect.signature(predict_from_checkpoint)
        param = sig.parameters["extract_region_attention"]
        assert param.default is True

    def test_predict_from_checkpoint_defaults_embeddings_true(self):
        """predict_from_checkpoint has extract_embeddings=True by default."""
        import inspect
        sig = inspect.signature(predict_from_checkpoint)
        param = sig.parameters["extract_embeddings"]
        assert param.default is True

    def test_predict_from_checkpoint_defaults_hgt_attention_true(self):
        """predict_from_checkpoint has extract_hgt_attention=True by default."""
        import inspect
        sig = inspect.signature(predict_from_checkpoint)
        param = sig.parameters["extract_hgt_attention"]
        assert param.default is True

    def test_predict_from_checkpoint_defaults_pma_attention_true(self):
        """predict_from_checkpoint has extract_pma_attention=True by default."""
        import inspect
        sig = inspect.signature(predict_from_checkpoint)
        param = sig.parameters["extract_pma_attention"]
        assert param.default is True


# ============================================================================
# Cell Counts Tests (Phase 6 Review Round 7 — M2)
# ============================================================================


class TestCellCounts:
    """Tests for cell_counts derivation and persistence."""

    def test_cell_counts_from_cell_mask(self, mock_model, mock_config):
        """predict() derives cell_counts from cell_mask in batch."""
        predictor = Predictor(mock_model, config=mock_config, device="cpu")

        # Create batch with cell_mask: [B, n_cell_types, max_cells]
        max_cells = 50
        cell_mask = torch.zeros(3, N_CELL_TYPES, max_cells, dtype=torch.bool)
        # Subject 0: 10 cells in type 0, 5 in type 1
        cell_mask[0, 0, :10] = True
        cell_mask[0, 1, :5] = True
        # Subject 1: 20 cells in type 0
        cell_mask[1, 0, :20] = True
        # Subject 2: all zeros

        batch = {
            "pseudobulk": torch.randn(3, N_CELL_TYPES, 100),
            "subject_ids": ["a", "b", "c"],
            "pathology": torch.rand(3, 3),
            "cell_mask": cell_mask,
        }

        result = predictor.predict([batch], show_progress=False)
        assert result.cell_counts is not None
        assert result.cell_counts.shape == (3, N_CELL_TYPES)
        assert result.cell_counts[0, 0] == 10
        assert result.cell_counts[0, 1] == 5
        assert result.cell_counts[1, 0] == 20
        assert result.cell_counts[2, 0] == 0

    def test_cell_counts_none_without_cell_mask(self, mock_model, mock_config):
        """predict() returns cell_counts=None when no cell_mask in batch."""
        predictor = Predictor(mock_model, config=mock_config, device="cpu")

        batch = {
            "pseudobulk": torch.randn(2, N_CELL_TYPES, 100),
            "subject_ids": ["a", "b"],
            "pathology": torch.rand(2, 3),
        }

        result = predictor.predict([batch], show_progress=False)
        assert result.cell_counts is None

    def test_cell_counts_hdf5_roundtrip(self):
        """cell_counts survives HDF5 save/load roundtrip."""
        from src.utils.io import save_attention_weights, load_attention_weights

        cell_counts = np.array([[10, 5, 0], [20, 15, 8]], dtype=np.int64)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "attn.h5"
            save_attention_weights(path, cell_counts=cell_counts)
            loaded = load_attention_weights(path)

        assert "cell_counts" in loaded
        np.testing.assert_array_equal(loaded["cell_counts"], cell_counts)


# ============================================================================
# Phase 6 Review Round 8 — H2: Use batch cell_counts over cell_mask
# ============================================================================


class TestCellCountsFromBatch:
    """Tests that predict() prefers batch['cell_counts'] over deriving from cell_mask."""

    def test_prefers_batch_cell_counts_over_cell_mask(self, mock_model, mock_config):
        """When both cell_counts and cell_mask are present, cell_counts is used."""
        predictor = Predictor(mock_model, config=mock_config, device="cpu")

        max_cells = 50
        # cell_mask says 10 cells for type 0 (clipped)
        cell_mask = torch.zeros(2, N_CELL_TYPES, max_cells, dtype=torch.bool)
        cell_mask[0, 0, :10] = True
        cell_mask[1, 0, :10] = True

        # But true cell_counts say 200 cells (pre-clipping)
        true_counts = torch.zeros(2, N_CELL_TYPES, dtype=torch.long)
        true_counts[0, 0] = 200
        true_counts[1, 0] = 150

        batch = {
            "pseudobulk": torch.randn(2, N_CELL_TYPES, 100),
            "subject_ids": ["a", "b"],
            "pathology": torch.rand(2, 3),
            "cell_mask": cell_mask,
            "cell_counts": true_counts,
        }

        result = predictor.predict([batch], show_progress=False)
        assert result.cell_counts is not None
        # Should use true counts, not clipped cell_mask counts
        assert result.cell_counts[0, 0] == 200
        assert result.cell_counts[1, 0] == 150

    def test_falls_back_to_cell_mask_when_no_cell_counts(self, mock_model, mock_config):
        """When cell_counts is absent but cell_mask is present, derives from mask."""
        predictor = Predictor(mock_model, config=mock_config, device="cpu")

        max_cells = 50
        cell_mask = torch.zeros(2, N_CELL_TYPES, max_cells, dtype=torch.bool)
        cell_mask[0, 0, :10] = True

        batch = {
            "pseudobulk": torch.randn(2, N_CELL_TYPES, 100),
            "subject_ids": ["a", "b"],
            "pathology": torch.rand(2, 3),
            "cell_mask": cell_mask,
        }

        result = predictor.predict([batch], show_progress=False)
        assert result.cell_counts is not None
        assert result.cell_counts[0, 0] == 10


# ============================================================================
# Per-Subject Pseudobulk Tests
# ============================================================================


class TestPerSubjectPseudobulk:
    """Tests for per_subject_pseudobulk field in PredictionResult."""

    def test_per_subject_pseudobulk_field(self):
        """PredictionResult should have per_subject_pseudobulk field."""
        n_subjects, n_cell_types, n_genes = 5, 3, 10
        result = PredictionResult(
            subject_ids=[f"S{i}" for i in range(n_subjects)],
            mean=np.zeros((n_subjects, 1)),
            std=None,
            actual=None,
            pathology=np.zeros((n_subjects, 3)),
            attention_weights=np.zeros((n_subjects, 1, n_cell_types)),
            gene_gate_weights=np.zeros((n_cell_types, n_genes)),
            per_subject_pseudobulk=np.random.rand(n_subjects, n_cell_types, n_genes),
        )
        assert result.per_subject_pseudobulk.shape == (n_subjects, n_cell_types, n_genes)

    def test_per_subject_pseudobulk_default_none(self):
        """per_subject_pseudobulk should default to None."""
        result = PredictionResult(
            subject_ids=["S0"],
            mean=np.zeros((1, 1)),
            std=None,
            actual=None,
            pathology=np.zeros((1, 3)),
            attention_weights=np.zeros((1, 1, 3)),
            gene_gate_weights=np.zeros((3, 10)),
        )
        assert result.per_subject_pseudobulk is None


# ============================================================================
# Save Predictions Metadata Tests (Phase 6 Review Round 12 — M1)
# ============================================================================


class TestSavePredictionsMetadata:
    """Test that save_predictions includes metadata columns."""

    def test_save_predictions_includes_metadata_columns(self, tmp_path):
        """Predictions DataFrame should include region/split if available in metadata."""
        result = PredictionResult(
            subject_ids=["S1", "S2"],
            mean=np.array([[0.5], [0.6]]),
            std=None,
            actual=None,
            pathology=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
            attention_weights=np.zeros((2, 2, 31)),
            gene_gate_weights=np.zeros((31, 100)),
            metadata={
                "subject_metadata": {
                    "region": ["PFC", "EC"],
                    "split": ["train", "val"],
                },
            },
        )

        # Use Predictor.save_predictions via a mock Predictor instance
        predictor = MagicMock(spec=Predictor)
        predictor.save_predictions = Predictor.save_predictions.__get__(predictor)
        predictor._save_attention_hdf5 = MagicMock()  # skip HDF5 save
        predictor.checkpoint_path = None
        predictor.config = None

        saved = predictor.save_predictions(result, tmp_path, save_hdf5=False)

        df = pd.read_parquet(tmp_path / "predictions.parquet")
        assert "region" in df.columns, "region column missing from predictions"
        assert "split" in df.columns, "split column missing from predictions"
        assert list(df["region"]) == ["PFC", "EC"]
        assert list(df["split"]) == ["train", "val"]


# ============================================================================
# from_lightning_module Guide Propagation Tests
# ============================================================================


class TestFromLightningModuleGuide:
    """Tests that from_lightning_module passes the guide to Predictor."""

    def test_from_lightning_module_passes_guide(self, mock_config):
        """Bayesian module's guide is propagated to Predictor."""
        module = MagicMock(spec=CognitiveResilienceLightningModule)
        module.model = MagicMock()
        module.model.eval.return_value = module.model
        module.model.to.return_value = module.model
        module.config = mock_config
        module.guide = MagicMock(name="mock_guide")

        predictor = Predictor.from_lightning_module(module, device="cpu")
        assert predictor.guide is module.guide

    def test_from_lightning_module_guide_none_deterministic(self, mock_config):
        """Deterministic module (guide=None) results in predictor.guide=None."""
        module = MagicMock(spec=CognitiveResilienceLightningModule)
        module.model = MagicMock()
        module.model.eval.return_value = module.model
        module.model.to.return_value = module.model
        module.config = mock_config
        module.guide = None

        predictor = Predictor.from_lightning_module(module, device="cpu")
        assert predictor.guide is None

    def test_from_lightning_module_missing_guide_attr(self, mock_config):
        """Module without guide attribute results in predictor.guide=None."""
        module = MagicMock(spec=CognitiveResilienceLightningModule)
        module.model = MagicMock()
        module.model.eval.return_value = module.model
        module.model.to.return_value = module.model
        module.config = mock_config
        # Delete guide attr so getattr fallback triggers
        del module.guide

        predictor = Predictor.from_lightning_module(module, device="cpu")
        assert predictor.guide is None
