"""Tests for run_analysis.py script."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pandas as pd
import pytest


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_experiment_dir(tmp_path):
    """Create mock experiment directory structure."""
    exp_dir = tmp_path / "experiments" / "20260113_test"
    analysis_dir = exp_dir / "analysis"
    analysis_dir.mkdir(parents=True)

    # Create mock predictions
    predictions = pd.DataFrame({
        "subject_id": [f"ROSMAP_{i:03d}" for i in range(20)],
        "predicted_mean": np.random.randn(20) * 2 + 5,
        "predicted_std": np.abs(np.random.randn(20)) * 0.5 + 0.3,
        "actual": np.random.randn(20) * 2 + 5,
    })
    predictions.to_parquet(analysis_dir / "predictions.parquet")

    # Create mock attention weights
    with h5py.File(analysis_dir / "attention_weights.h5", "w") as f:
        f.create_dataset("gene_gate", data=np.random.rand(8, 100))
        f.create_dataset("pathology_attention", data=np.random.rand(20, 4, 8))
        f.create_dataset("region_weights", data=np.random.rand(6))
        f.attrs["cell_type_names"] = ["Ast", "Mic", "Oli", "OPC", "Exc", "Inh", "End", "Per"]
        f.attrs["gene_names"] = [f"GENE{i}" for i in range(100)]
        f.attrs["subject_ids"] = [f"ROSMAP_{i:03d}" for i in range(20)]

    return exp_dir


@pytest.fixture
def mock_metadata(tmp_path):
    """Create mock metadata file."""
    metadata = pd.DataFrame({
        "subject_id": [f"ROSMAP_{i:03d}" for i in range(20)],
        "pathology": np.random.rand(20) * 10,
        "age": np.random.randint(60, 95, 20),
    })
    path = tmp_path / "metadata.csv"
    metadata.to_csv(path, index=False)
    return path


# =============================================================================
# Script Import Tests
# =============================================================================


class TestRunAnalysisImports:
    """Test that script imports work correctly."""

    def test_script_is_importable(self):
        """Test that script can be imported."""
        # Import the module functions directly
        from scripts.run_analysis import (
            parse_args,
            load_predictions,
            run_cell_type_importance,
            run_gene_importance,
            run_uncertainty_analysis,
        )
        # load_attention_weights is now imported from src.utils.io
        from src.utils.io import load_attention_weights

        assert callable(parse_args)
        assert callable(load_predictions)
        assert callable(load_attention_weights)
        assert callable(run_cell_type_importance)
        assert callable(run_gene_importance)
        assert callable(run_uncertainty_analysis)


# =============================================================================
# Function Unit Tests
# =============================================================================


class TestLoadPredictions:
    """Test load_predictions function."""

    def test_load_parquet(self, tmp_path):
        """Test loading parquet file."""
        from scripts.run_analysis import load_predictions

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        path = tmp_path / "test.parquet"
        df.to_parquet(path)

        loaded = load_predictions(path)
        pd.testing.assert_frame_equal(df, loaded)

    def test_load_csv(self, tmp_path):
        """Test loading CSV file."""
        from scripts.run_analysis import load_predictions

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        path = tmp_path / "test.csv"
        df.to_csv(path, index=False)

        loaded = load_predictions(path)
        pd.testing.assert_frame_equal(df, loaded)

    def test_unsupported_format(self, tmp_path):
        """Test error on unsupported format."""
        from scripts.run_analysis import load_predictions

        path = tmp_path / "test.xyz"
        path.touch()

        with pytest.raises(ValueError, match="Unsupported"):
            load_predictions(path)


class TestLoadAttentionWeights:
    """Test load_attention_weights function (now from src.utils.io)."""

    def test_load_hdf5(self, tmp_path):
        """Test loading HDF5 file."""
        from src.utils.io import load_attention_weights

        path = tmp_path / "attention.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("gene_gate", data=np.random.rand(8, 100))
            f.create_dataset("pathology_attention", data=np.random.rand(20, 4, 8))
            f.attrs["cell_type_names"] = ["A", "B", "C"]

        weights = load_attention_weights(Path(path))

        assert "gene_gate" in weights
        assert "pathology_attention" in weights
        # Metadata is stored under "metadata" key in the shared function
        assert "metadata" in weights
        assert "cell_type_names" in weights["metadata"]
        assert weights["gene_gate"].shape == (8, 100)

    def test_load_nonexistent(self, tmp_path):
        """Test loading nonexistent file raises FileNotFoundError."""
        from src.utils.io import load_attention_weights

        path = Path(tmp_path) / "nonexistent.h5"
        # The shared function raises FileNotFoundError for nonexistent files
        with pytest.raises(FileNotFoundError):
            load_attention_weights(path)


# =============================================================================
# Analysis Function Tests
# =============================================================================


class TestRunCellTypeImportance:
    """Test run_cell_type_importance function."""

    def test_runs_without_error(self, tmp_path):
        """Test function runs without error."""
        from scripts.run_analysis import run_cell_type_importance

        pathology_attention = np.random.rand(20, 4, 8)
        cell_type_names = ["A", "B", "C", "D", "E", "F", "G", "H"]

        # Should not raise
        run_cell_type_importance(
            pathology_attention=pathology_attention,
            cell_type_names=cell_type_names,
            output_dir=tmp_path,
            formats=["csv"],
        )

        # Check output exists
        assert (tmp_path / "cell_type_importance.csv").exists()


class TestRunGeneImportance:
    """Test run_gene_importance function."""

    def test_runs_without_error(self, tmp_path):
        """Test function runs without error."""
        from scripts.run_analysis import run_gene_importance

        gene_gate = np.random.rand(8, 100)
        cell_type_names = ["A", "B", "C", "D", "E", "F", "G", "H"]
        gene_names = [f"GENE{i}" for i in range(100)]

        run_gene_importance(
            gene_gate_weights=gene_gate,
            cell_type_names=cell_type_names,
            gene_names=gene_names,
            top_k=10,
            output_dir=tmp_path,
            formats=["csv"],
        )

        assert (tmp_path / "gene_importance_by_celltype.csv").exists()


class TestRunUncertaintyAnalysis:
    """Test run_uncertainty_analysis function."""

    def test_runs_without_error(self, tmp_path):
        """Test function runs without error."""
        from scripts.run_analysis import run_uncertainty_analysis

        np.random.seed(42)
        n = 50
        actual = np.random.randn(n) * 2 + 5
        predicted_mean = actual + np.random.randn(n) * 0.5
        predicted_std = np.abs(np.random.randn(n)) * 0.3 + 0.2

        run_uncertainty_analysis(
            predicted_mean=predicted_mean,
            predicted_std=predicted_std,
            actual=actual,
            output_dir=tmp_path,
            formats=["csv"],
        )

        assert (tmp_path / "prediction_uncertainty.csv").exists()
        assert (tmp_path / "calibration_summary.csv").exists()


# =============================================================================
# Integration Tests
# =============================================================================


class TestScriptIntegration:
    """Integration tests for run_analysis.py script."""

    def test_help_flag(self):
        """Test --help flag works."""
        result = subprocess.run(
            [sys.executable, "scripts/run_analysis.py", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Run post-hoc analysis" in result.stdout

    def test_requires_input(self):
        """Test script fails without input."""
        result = subprocess.run(
            [sys.executable, "scripts/run_analysis.py"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

    def test_with_experiment_dir(self, mock_experiment_dir):
        """Test script runs with experiment directory."""
        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_analysis.py",
                "--experiment-dir", str(mock_experiment_dir),
                "--skip-ccc",
                "--skip-resilience",
            ],
            capture_output=True,
            text=True,
        )

        # Script should complete (may warn about missing data)
        # Check for no Python errors
        assert "Traceback" not in result.stderr or "Error" not in result.stderr

    def test_skip_flags(self, mock_experiment_dir):
        """Test skip flags work."""
        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_analysis.py",
                "--experiment-dir", str(mock_experiment_dir),
                "--skip-cell-type",
                "--skip-gene",
                "--skip-ccc",
                "--skip-resilience",
                "--skip-regional",
                "--skip-uncertainty",
            ],
            capture_output=True,
            text=True,
        )

        # Should complete quickly with all analyses skipped
        assert "Completed 0 analyses" in result.stdout or result.returncode == 0


# =============================================================================
# Edge Cases
# =============================================================================


class TestRunAnalysisEdgeCases:
    """Test edge cases."""

    def test_empty_analysis_dir(self, tmp_path):
        """Test with empty analysis directory."""
        from src.utils.io import load_attention_weights

        # Shared function raises FileNotFoundError for nonexistent files
        # This is the expected behavior - callers should check existence first
        path = Path(tmp_path) / "nonexistent.h5"
        with pytest.raises(FileNotFoundError):
            load_attention_weights(path)

    def test_partial_data(self, tmp_path):
        """Test with partial data (only predictions, no attention)."""
        # Create only predictions
        predictions = pd.DataFrame({
            "subject_id": [f"S{i}" for i in range(10)],
            "predicted_mean": np.random.randn(10),
            "predicted_std": np.abs(np.random.randn(10)) + 0.1,
        })
        predictions.to_parquet(tmp_path / "predictions.parquet")

        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_analysis.py",
                "--predictions-path", str(tmp_path / "predictions.parquet"),
                "--output-dir", str(tmp_path / "output"),
                "--skip-cell-type",
                "--skip-gene",
                "--skip-ccc",
                "--skip-resilience",
                "--skip-regional",
            ],
            capture_output=True,
            text=True,
        )

        # Should handle missing attention gracefully
        assert "Traceback" not in result.stderr
