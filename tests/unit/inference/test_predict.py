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
    save_predictions_parquet,
    save_predictions_hdf5,
    load_predictions_parquet,
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
        return {
            "mean": torch.randn(batch_size, 1),
            "std": torch.abs(torch.randn(batch_size, 1)) + 0.1,
            "attention_weights": torch.rand(batch_size, 4, N_CELL_TYPES),
        }

    model.side_effect = mock_forward
    model.__call__ = MagicMock(side_effect=mock_forward)
    model.eval.return_value = model
    model.to.return_value = model

    # Mock gene gate weights
    model.pseudobulk_encoder.gene_gate.get_gate_weights.return_value = torch.rand(N_CELL_TYPES, 100)

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
            "attention_weights", "gene_gate_weights", "hgt_attention", "metadata"
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
            "subject_id": ["a", "b", "c", "d"],
        }

        result = predictor.predict_batch(batch)
        assert "mean" in result
        assert "attention_weights" in result

    def test_predict_batch_shapes(self, mock_model, mock_config):
        """predict_batch returns correct shapes."""
        predictor = Predictor(mock_model, config=mock_config, device="cpu")

        batch = {
            "pseudobulk": torch.randn(4, N_CELL_TYPES, 100),
            "subject_id": ["a", "b", "c", "d"],
        }

        result = predictor.predict_batch(batch)
        assert result["mean"].shape == (4, 1)

    def test_predict_returns_prediction_result(self, mock_model, mock_config):
        """predict returns PredictionResult instance."""
        predictor = Predictor(mock_model, config=mock_config, device="cpu")

        # Mock dataloader
        batch1 = {
            "pseudobulk": torch.randn(2, N_CELL_TYPES, 100),
            "subject_id": ["a", "b"],
            "pathology": torch.rand(2, 3),
        }
        batch2 = {
            "pseudobulk": torch.randn(2, N_CELL_TYPES, 100),
            "subject_id": ["c", "d"],
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
            "subject_id": ["z", "a", "m"],
            "pathology": torch.rand(3, 3),
        }
        dataloader = [batch]

        result = predictor.predict(dataloader, show_progress=False)
        assert result.subject_ids == ["z", "a", "m"]


# ============================================================================
# Serialization Tests
# ============================================================================


class TestSerializationParquet:
    """Tests for Parquet save/load."""

    def test_save_predictions_parquet_creates_file(self, sample_prediction_result):
        """save_predictions_parquet creates a parquet file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.parquet"
            save_predictions_parquet(sample_prediction_result, path)
            assert path.exists()

    def test_save_load_parquet_roundtrip(self, sample_prediction_result):
        """Parquet save/load preserves prediction data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.parquet"
            save_predictions_parquet(sample_prediction_result, path)
            df = load_predictions_parquet(path)

        assert len(df) == len(sample_prediction_result.subject_ids)
        assert "subject_id" in df.columns
        assert "predicted_mean" in df.columns
        assert "predicted_std" in df.columns

    def test_parquet_contains_all_fields(self, sample_prediction_result):
        """Parquet file contains all expected columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.parquet"
            save_predictions_parquet(sample_prediction_result, path)
            df = load_predictions_parquet(path)

        expected_columns = {
            "subject_id", "predicted_mean", "predicted_std",
            "actual", "pathology"
        }
        assert expected_columns.issubset(set(df.columns))


class TestSerializationHDF5:
    """Tests for HDF5 save/load."""

    def test_save_predictions_hdf5_creates_file(self, sample_prediction_result):
        """save_predictions_hdf5 creates an HDF5 file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.h5"
            save_predictions_hdf5(sample_prediction_result, path)
            assert path.exists()

    def test_hdf5_contains_attention_weights(self, sample_prediction_result):
        """HDF5 file contains attention weights."""
        import h5py

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.h5"
            save_predictions_hdf5(sample_prediction_result, path)

            with h5py.File(path, "r") as f:
                assert "attention_weights" in f
                assert "gene_gate_weights" in f

    def test_hdf5_attention_shapes_preserved(self, sample_prediction_result):
        """HDF5 preserves attention weight shapes."""
        import h5py

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.h5"
            save_predictions_hdf5(sample_prediction_result, path)

            with h5py.File(path, "r") as f:
                assert f["attention_weights"].shape == sample_prediction_result.attention_weights.shape
                assert f["gene_gate_weights"].shape == sample_prediction_result.gene_gate_weights.shape


# ============================================================================
# Schema Validation Tests
# ============================================================================


class TestOutputSchemaValidation:
    """Tests validating output schemas."""

    def test_parquet_schema(self, sample_prediction_result):
        """Parquet output has expected schema."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.parquet"
            save_predictions_parquet(sample_prediction_result, path)
            df = load_predictions_parquet(path)

        # Type validation
        assert df["subject_id"].dtype == object
        assert np.issubdtype(df["predicted_mean"].dtype, np.floating)
        assert np.issubdtype(df["predicted_std"].dtype, np.floating)

    def test_parquet_no_missing_subject_ids(self, sample_prediction_result):
        """Parquet output has no missing subject IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.parquet"
            save_predictions_parquet(sample_prediction_result, path)
            df = load_predictions_parquet(path)

        assert df["subject_id"].notna().all()

    def test_parquet_std_positive(self, sample_prediction_result):
        """Predicted std values are positive."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.parquet"
            save_predictions_parquet(sample_prediction_result, path)
            df = load_predictions_parquet(path)

        assert (df["predicted_std"] > 0).all()


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

    @given(n_samples=st.integers(min_value=1, max_value=20))
    @settings(max_examples=10)
    def test_parquet_roundtrip_preserves_length(self, n_samples):
        """Parquet round-trip preserves number of samples."""
        result = PredictionResult(
            subject_ids=[f"s{i}" for i in range(n_samples)],
            mean=np.random.randn(n_samples).astype(np.float32),
            std=np.abs(np.random.randn(n_samples)).astype(np.float32) + 0.01,
            actual=None,
            pathology=np.random.rand(n_samples).astype(np.float32),
            attention_weights=np.random.rand(n_samples, 4, 31).astype(np.float32),
            gene_gate_weights=np.random.rand(31, 100).astype(np.float32),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pred.parquet"
            save_predictions_parquet(result, path)
            df = load_predictions_parquet(path)

        assert len(df) == n_samples


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
            "subject_id": ["single"],
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
            "subject_id": ["a", "b", "c"],
            "pathology": torch.rand(3, 3),
            # No "cognition" key
        }
        dataloader = [batch]

        result = predictor.predict(dataloader, show_progress=False)
        assert result.actual is None

    def test_parquet_with_none_actual(self):
        """Parquet save/load handles None actual values."""
        result = PredictionResult(
            subject_ids=["a", "b"],
            mean=np.array([1.0, 2.0]),
            std=np.array([0.1, 0.2]),
            actual=None,  # No ground truth
            pathology=np.array([0.5, 0.6]),
            attention_weights=np.zeros((2, 4, 31)),
            gene_gate_weights=np.zeros((31, 100)),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pred.parquet"
            save_predictions_parquet(result, path)
            df = load_predictions_parquet(path)

        # actual column should exist but have NaN values
        assert "actual" in df.columns
        assert df["actual"].isna().all()

    def test_empty_metadata(self, sample_prediction_result):
        """Handles empty metadata dict."""
        sample_prediction_result.metadata = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.h5"
            save_predictions_hdf5(sample_prediction_result, path)
            assert path.exists()

    def test_large_metadata(self, sample_prediction_result):
        """Handles large metadata dict."""
        sample_prediction_result.metadata = {
            "config": {"nested": {"deeply": {"value": 123}}},
            "history": [{"epoch": i, "loss": 0.1 * i} for i in range(100)],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.h5"
            save_predictions_hdf5(sample_prediction_result, path)
            assert path.exists()
