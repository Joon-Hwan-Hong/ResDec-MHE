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

    # Create mock attention weights with nested groups matching predictor output
    with h5py.File(analysis_dir / "attention_weights.h5", "w") as f:
        f.attrs["schema_version"] = "2.0"
        f.create_dataset("gene_gate", data=np.random.rand(8, 100))
        f.create_dataset("pathology_attention", data=np.random.rand(20, 4, 8))
        f.create_dataset("region_weights", data=np.random.rand(6))
        f.create_dataset("region_pseudobulk", data=np.random.rand(6, 8, 100))
        f.create_dataset("region_attention", data=np.random.rand(20, 6))
        vlen_str = h5py.special_dtype(vlen=str)
        f.create_dataset("cell_type_names", data=np.array(
            ["Ast", "Mic", "Oli", "OPC", "Exc", "Inh", "End", "Per"], dtype=object
        ), dtype=vlen_str)
        f.create_dataset("gene_names", data=np.array(
            [f"GENE{i}" for i in range(100)], dtype=object
        ), dtype=vlen_str)
        f.create_dataset("subject_ids", data=np.array(
            [f"ROSMAP_{i:03d}" for i in range(20)], dtype=object
        ), dtype=vlen_str)

        # HGT attention as nested group
        hgt_group = f.create_group("hgt_attention")
        n_edge_types = 5
        # PyG convention: src|edge_type|dst
        hgt_group.create_dataset("edge_type_names", data=np.array([
            "Ast|Secreted_Signaling|Mic", "Mic|Secreted_Signaling|Ast",
            "Oli|ECM_Receptor|Exc", "Exc|Cell_Cell_Contact|Inh",
            "Ast|Secreted_Signaling|Oli",
        ], dtype=object), dtype=vlen_str)
        agg_group = hgt_group.create_group("aggregated")
        agg_group.create_dataset("mean_by_edge_type", data=np.random.rand(n_edge_types, 4))
        agg_group.create_dataset("std_by_edge_type", data=np.random.rand(n_edge_types, 4))

    return exp_dir


@pytest.fixture
def mock_metadata(tmp_path):
    """Create mock metadata file."""
    metadata = pd.DataFrame({
        "subject_id": [f"ROSMAP_{i:03d}" for i in range(20)],
        "pathology": np.random.rand(20) * 10,
        "gpath": np.random.rand(20) * 10,
        "cogn_global": np.random.randn(20) * 2 + 5,
        "age": np.random.randint(60, 95, 20),
        "cell_count": np.random.randint(100, 5000, 20),
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
        vlen_str = h5py.special_dtype(vlen=str)
        with h5py.File(path, "w") as f:
            f.create_dataset("gene_gate", data=np.random.rand(8, 100))
            f.create_dataset("pathology_attention", data=np.random.rand(20, 4, 8))
            f.create_dataset("cell_type_names", data=np.array(["A", "B", "C"], dtype=object), dtype=vlen_str)

        weights = load_attention_weights(Path(path))

        assert "gene_gate" in weights
        assert "pathology_attention" in weights
        # Metadata is stored under "metadata" key in the shared function
        assert "metadata" in weights
        assert "cell_type_names" in weights["metadata"]
        assert weights["gene_gate"].shape == (8, 100)
        # String datasets are also at the top level as lists
        assert "cell_type_names" in weights
        assert isinstance(weights["cell_type_names"], list)
        assert weights["cell_type_names"] == ["A", "B", "C"]

    def test_load_nonexistent(self, tmp_path):
        """Test loading nonexistent file returns empty dict."""
        from src.utils.io import load_attention_weights

        path = Path(tmp_path) / "nonexistent.h5"
        # The shared function returns empty dict for nonexistent files
        result = load_attention_weights(path)
        assert result == {}


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


class TestRunCCCImportance:
    """Test run_ccc_importance function."""

    def test_runs_without_error(self, tmp_path):
        """Test function runs without error."""
        from scripts.run_analysis import run_ccc_importance

        # Create mock edge attention scores [n_subjects, n_edges]
        edge_attention = np.random.rand(20, 50)
        cell_type_names = ["Ast", "Mic", "Oli", "OPC", "Exc", "Inh", "End", "Per"]

        run_ccc_importance(
            edge_attention_scores=edge_attention,
            cell_type_names=cell_type_names,
            output_dir=tmp_path,
            formats=["csv"],
        )

        assert (tmp_path / "ccc_importance.csv").exists()

    def test_with_edge_metadata(self, tmp_path):
        """Test function with edge metadata."""
        from scripts.run_analysis import run_ccc_importance

        edge_attention = np.random.rand(20, 10)
        edge_metadata = pd.DataFrame({
            "source": ["Ast", "Mic", "Ast", "Oli", "Exc"] * 2,
            "target": ["Mic", "Ast", "Exc", "Inh", "Inh"] * 2,
            "edge_type": ["secreted_signaling"] * 10,
        })

        run_ccc_importance(
            edge_attention_scores=edge_attention,
            edge_metadata=edge_metadata,
            output_dir=tmp_path,
            formats=["csv"],
        )

        assert (tmp_path / "ccc_importance.csv").exists()


class TestRunResilienceSignature:
    """Test run_resilience_signature function."""

    def test_runs_without_error(self, tmp_path):
        """Test function runs without error."""
        from scripts.run_analysis import run_resilience_signature

        np.random.seed(42)
        n_subjects = 30
        n_cell_types = 8

        # Create mock attention [n_subjects, n_heads, n_cell_types]
        attention = np.random.rand(n_subjects, 4, n_cell_types)
        pathology_scores = np.random.rand(n_subjects) * 10
        cognition_scores = np.random.rand(n_subjects) * 100
        cell_type_names = ["Ast", "Mic", "Oli", "OPC", "Exc", "Inh", "End", "Per"]

        run_resilience_signature(
            attention=attention,
            pathology_scores=pathology_scores,
            cognition_scores=cognition_scores,
            cell_type_names=cell_type_names,
            n_permutations=10,  # Few permutations for speed
            output_dir=tmp_path,
            formats=["csv"],
        )

        assert (tmp_path / "resilience_signature.csv").exists()


class TestRunRegionalAnalysis:
    """Test run_regional_analysis function."""

    def test_runs_without_error(self, tmp_path):
        """Test function runs without error."""
        from scripts.run_analysis import run_regional_analysis

        np.random.seed(42)
        n_regions = 6
        n_cell_types = 8
        n_genes = 100

        region_weights = np.random.rand(n_regions)
        gene_gate_weights = np.random.rand(n_cell_types, n_genes)
        region_names = ["PFC", "AG", "MTC", "EC", "HC", "TH"]
        cell_type_names = ["Ast", "Mic", "Oli", "OPC", "Exc", "Inh", "End", "Per"]
        gene_names = [f"GENE{i}" for i in range(n_genes)]

        run_regional_analysis(
            region_weights=region_weights,
            gene_gate_weights=gene_gate_weights,
            region_names=region_names,
            cell_type_names=cell_type_names,
            gene_names=gene_names,
            top_k_genes=10,
            output_dir=tmp_path,
            formats=["csv"],
        )

        # Regional analysis saves multiple outputs
        assert (tmp_path / "region_contribution.csv").exists()


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

    def test_with_experiment_dir_full(self, mock_experiment_dir):
        """Test script runs with experiment directory including CCC (uses nested HGT groups)."""
        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_analysis.py",
                "--experiment-dir", str(mock_experiment_dir),
                "--skip-resilience",
                "--skip-embedding",
            ],
            capture_output=True,
            text=True,
        )

        # Script should complete without Python errors
        assert "Traceback" not in result.stderr

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

        # Shared function returns empty dict for nonexistent files
        path = Path(tmp_path) / "nonexistent.h5"
        result = load_attention_weights(path)
        assert result == {}

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


# =============================================================================
# New Tests for Phase 6 Round 3
# =============================================================================


class TestAlignArrayBySubjectId:
    """Test _align_array_by_subject_id helper."""

    def test_basic_alignment(self):
        """Test basic subject-aligned extraction."""
        from scripts.run_analysis import _align_array_by_subject_id

        source_df = pd.DataFrame({
            "subject_id": ["S1", "S2", "S3"],
            "gpath": [1.0, 2.0, 3.0],
        })
        target_ids = ["S3", "S1", "S2"]
        result = _align_array_by_subject_id(source_df, target_ids, "gpath")
        np.testing.assert_array_equal(result, [3.0, 1.0, 2.0])

    def test_missing_subjects_get_nan(self):
        """Missing subjects get NaN values."""
        from scripts.run_analysis import _align_array_by_subject_id

        source_df = pd.DataFrame({
            "subject_id": ["S1", "S2"],
            "gpath": [1.0, 2.0],
        })
        target_ids = ["S1", "S3", "S2"]
        result = _align_array_by_subject_id(source_df, target_ids, "gpath")
        assert result[0] == 1.0
        assert np.isnan(result[1])
        assert result[2] == 2.0

    def test_column_not_found_returns_none(self):
        """Returns None when column doesn't exist."""
        from scripts.run_analysis import _align_array_by_subject_id

        source_df = pd.DataFrame({"subject_id": ["S1"], "other": [1.0]})
        result = _align_array_by_subject_id(source_df, ["S1"], "nonexistent")
        assert result is None

    def test_no_subject_id_column_returns_none(self):
        """Returns None when source_id_column doesn't exist."""
        from scripts.run_analysis import _align_array_by_subject_id

        source_df = pd.DataFrame({"id": ["S1"], "gpath": [1.0]})
        result = _align_array_by_subject_id(source_df, ["S1"], "gpath")
        assert result is None


class TestResilienceWithAblation:
    """Test run_resilience_signature with ablation flags."""

    def test_ablation_params_accepted(self, tmp_path):
        """Test run_resilience_signature accepts ablation parameters."""
        from scripts.run_analysis import run_resilience_signature

        np.random.seed(42)
        n_subjects = 30
        n_cell_types = 8

        attention = np.random.rand(n_subjects, 4, n_cell_types)
        pathology_scores = np.random.rand(n_subjects) * 10
        cognition_scores = np.random.rand(n_subjects) * 100
        cell_type_names = ["Ast", "Mic", "Oli", "OPC", "Exc", "Inh", "End", "Per"]

        run_resilience_signature(
            attention=attention,
            pathology_scores=pathology_scores,
            cognition_scores=cognition_scores,
            cell_type_names=cell_type_names,
            n_permutations=10,
            run_ablation=True,
            ablation_method="zero_embedding",
            output_dir=tmp_path,
            formats=["csv"],
        )

        assert (tmp_path / "resilience_signature.csv").exists()


class TestAblationCLIFlags:
    """Test ablation CLI flags are accepted by argparse."""

    def test_ablation_flags_accepted(self, mock_experiment_dir, mock_metadata):
        """Test --run-ablation and --ablation-method are accepted."""
        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_analysis.py",
                "--experiment-dir", str(mock_experiment_dir),
                "--metadata-path", str(mock_metadata),
                "--skip-cell-type", "--skip-gene", "--skip-ccc",
                "--skip-regional", "--skip-uncertainty", "--skip-embedding",
                "--run-ablation",
                "--ablation-method", "zero_embedding",
            ],
            capture_output=True,
            text=True,
        )
        assert "Traceback" not in result.stderr


class TestCognitionFallback:
    """Test resilience analysis cognition fallback to metadata."""

    def test_resilience_uses_metadata_cognition(self, tmp_path):
        """Test resilience runs using cogn_global from metadata when predictions lack actual."""
        # Create predictions WITHOUT "actual" column
        predictions = pd.DataFrame({
            "subject_id": [f"S{i}" for i in range(30)],
            "predicted_mean": np.random.randn(30) * 2 + 5,
            "predicted_std": np.abs(np.random.randn(30)) * 0.5 + 0.3,
        })
        predictions.to_parquet(tmp_path / "predictions.parquet")

        # Create metadata WITH cognition column
        metadata = pd.DataFrame({
            "subject_id": [f"S{i}" for i in range(30)],
            "cogn_global": np.random.randn(30) * 2 + 5,
            "gpath": np.random.rand(30) * 10,
        })
        metadata.to_csv(tmp_path / "metadata.csv", index=False)

        # Create attention weights
        with h5py.File(tmp_path / "attention.h5", "w") as f:
            f.attrs["schema_version"] = "2.0"
            f.create_dataset("gene_gate", data=np.random.rand(8, 50))
            f.create_dataset("pathology_attention", data=np.random.rand(30, 4, 8))
            vlen_str = h5py.special_dtype(vlen=str)
            f.create_dataset("subject_ids", data=np.array(
                [f"S{i}" for i in range(30)], dtype=object
            ), dtype=vlen_str)
            f.create_dataset("cell_type_names", data=np.array(
                ["Ast", "Mic", "Oli", "OPC", "Exc", "Inh", "End", "Per"], dtype=object
            ), dtype=vlen_str)

        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_analysis.py",
                "--predictions-path", str(tmp_path / "predictions.parquet"),
                "--attention-path", str(tmp_path / "attention.h5"),
                "--metadata-path", str(tmp_path / "metadata.csv"),
                "--output-dir", str(tmp_path / "output"),
                "--skip-cell-type", "--skip-gene", "--skip-ccc",
                "--skip-regional", "--skip-uncertainty", "--skip-embedding",
                "--n-permutations", "10",
            ],
            capture_output=True,
            text=True,
        )

        assert "Traceback" not in result.stderr
        assert "cogn_global" in result.stderr


class TestRegionPseudobulkRoundtrip:
    """Test region_pseudobulk save/load roundtrip."""

    def test_region_pseudobulk_saved_to_hdf5(self, tmp_path):
        """region_pseudobulk is saved and loadable from HDF5."""
        from src.utils.io import save_attention_weights, load_attention_weights

        gene_gate = np.random.rand(8, 100).astype(np.float32)
        region_pseudobulk = np.random.rand(6, 8, 100).astype(np.float32)

        path = tmp_path / "attention.h5"
        save_attention_weights(
            path=path,
            gene_gate=gene_gate,
            region_pseudobulk=region_pseudobulk,
        )

        loaded = load_attention_weights(path)
        assert "region_pseudobulk" in loaded
        np.testing.assert_array_almost_equal(
            loaded["region_pseudobulk"], region_pseudobulk
        )


# =============================================================================
# Gene Importance with Region Pseudobulk Tests
# =============================================================================


class TestGeneImportanceWithRegionPseudobulk:
    """Test run_gene_importance with region_pseudobulk produces by-region file (Finding 3)."""

    def test_produces_gene_importance_by_region(self, tmp_path):
        """run_gene_importance with region_pseudobulk should produce by-region CSV."""
        from scripts.run_analysis import run_gene_importance

        np.random.seed(42)
        n_cell_types = 8
        n_genes = 50
        n_regions = 6

        gene_gate = np.random.rand(n_cell_types, n_genes).astype(np.float32)
        cell_type_names = ["Ast", "Mic", "Oli", "OPC", "Exc", "Inh", "End", "Per"]
        gene_names = [f"GENE{i}" for i in range(n_genes)]
        region_names = ["PFC", "AG", "MTC", "EC", "HC", "TH"]

        # Create region pseudobulk dict: {region_name: [n_cell_types, n_genes]}
        region_pseudobulk = {
            name: np.random.rand(n_cell_types, n_genes).astype(np.float32)
            for name in region_names
        }

        run_gene_importance(
            gene_gate_weights=gene_gate,
            cell_type_names=cell_type_names,
            gene_names=gene_names,
            top_k=5,
            output_dir=tmp_path,
            formats=["csv"],
            region_pseudobulk=region_pseudobulk,
        )

        # Should produce gene_importance_by_celltype.csv (always)
        assert (tmp_path / "gene_importance_by_celltype.csv").exists()
        # Should also produce gene_importance_by_region.csv or regional_gene_importance.csv
        has_by_region = (
            (tmp_path / "gene_importance_by_region.csv").exists()
            or (tmp_path / "regional_gene_importance.csv").exists()
        )
        assert has_by_region, "Expected gene_importance_by_region.csv or regional_gene_importance.csv"

    def test_without_region_pseudobulk_no_region_file(self, tmp_path):
        """run_gene_importance without region_pseudobulk should NOT produce by-region file."""
        from scripts.run_analysis import run_gene_importance

        gene_gate = np.random.rand(8, 50).astype(np.float32)
        cell_type_names = ["Ast", "Mic", "Oli", "OPC", "Exc", "Inh", "End", "Per"]
        gene_names = [f"GENE{i}" for i in range(50)]

        run_gene_importance(
            gene_gate_weights=gene_gate,
            cell_type_names=cell_type_names,
            gene_names=gene_names,
            top_k=5,
            output_dir=tmp_path,
            formats=["csv"],
        )

        assert (tmp_path / "gene_importance_by_celltype.csv").exists()
        assert not (tmp_path / "gene_importance_by_region.csv").exists()
        assert not (tmp_path / "regional_gene_importance.csv").exists()
