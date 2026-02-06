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
            "cell_counts", "gene_names", "embeddings", "metadata"
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
