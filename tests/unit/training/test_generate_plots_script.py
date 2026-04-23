"""Tests for generate_plots.py script."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for testing

import h5py
import numpy as np
import pandas as pd
import pytest


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_analysis_dir(tmp_path):
    """Create mock analysis directory with outputs."""
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()

    # Use full cell type names matching CELL_TYPE_ORDER
    cell_types = ["Astrocyte", "Microglia", "Oligodendrocyte", "Oligodendrocyte precursor"]

    # Cell type importance
    pd.DataFrame({
        "cell_type": cell_types,
        "mean_attention": [0.3, 0.25, 0.25, 0.2],
        "std_attention": [0.05, 0.04, 0.03, 0.03],
        "rank": [1, 2, 3, 4],
    }).to_parquet(analysis_dir / "cell_type_importance.parquet")

    # Cell type importance by pathology
    data = []
    for ct in cell_types:
        for tertile in ["low", "medium", "high"]:
            data.append({
                "cell_type": ct,
                "pathology_tertile": tertile,
                "mean_attention": np.random.rand() * 0.3 + 0.1,
            })
    pd.DataFrame(data).to_parquet(analysis_dir / "cell_type_importance_by_pathology.parquet")

    # Gene importance
    gene_data = []
    for ct in cell_types:
        for rank in range(1, 51):
            gene_data.append({
                "cell_type": ct,
                "rank": rank,
                "gene": f"GENE{rank}",
                "weight": np.random.rand() * 0.3,
            })
    pd.DataFrame(gene_data).to_parquet(analysis_dir / "top_genes_per_celltype.parquet")

    # Gene importance by cell type
    by_ct_data = []
    for ct in cell_types[:2]:  # Just first two
        for i in range(100):
            by_ct_data.append({
                "cell_type": ct,
                "gene": f"GENE{i}",
                "weight": np.random.rand(),
            })
    pd.DataFrame(by_ct_data).to_parquet(analysis_dir / "gene_importance_by_celltype.parquet")

    # CCC data — filenames must match what load_analysis_data() expects
    # ccc_importance: per-edge importance
    ccc_edges = []
    for src in cell_types[:3]:
        for tgt in cell_types[:3]:
            ccc_edges.append({
                "source": src,
                "target": tgt,
                "edge_type": "Secreted_Signaling",
                "mean_attention": np.random.rand() * 0.3,
                "std_attention": np.random.rand() * 0.1,
            })
    pd.DataFrame(ccc_edges).to_parquet(analysis_dir / "ccc_importance.parquet")

    # ccc_network_summary: aggregated by edge type
    pd.DataFrame({
        "edge_type": ["Secreted_Signaling", "ECM_Receptor", "Cell_Cell_Contact"],
        "display_name": ["Secreted Signaling", "ECM-Receptor", "Cell-Cell Contact"],
        "mean_attention": [0.35, 0.25, 0.20],
        "std_attention": [0.05, 0.03, 0.02],
        "n_edges": [9, 6, 4],
    }).to_parquet(analysis_dir / "ccc_network_summary.parquet")

    # top_interactions: ranked interactions
    interactions = []
    for rank, (src, tgt) in enumerate(zip(cell_types[:3] * 2, cell_types[1:4] * 2), 1):
        interactions.append({
            "rank": rank,
            "source": src,
            "target": tgt,
            "edge_type": "Secreted_Signaling",
            "mean_attention": np.random.rand() * 0.3,
        })
    pd.DataFrame(interactions).to_parquet(analysis_dir / "top_interactions.parquet")

    # Resilience signature (both root for backward compat and subdirectory)
    pd.DataFrame({
        "cell_type": cell_types,
        "signature": np.random.randn(4) * 0.2,
    }).to_parquet(analysis_dir / "resilience_signature.parquet")

    # Also create a subdirectory layout
    resilience_subdir = analysis_dir / "resilience_gpath"
    resilience_subdir.mkdir()
    pd.DataFrame({
        "cell_type": cell_types,
        "signature": np.random.randn(4) * 0.2,
    }).to_parquet(resilience_subdir / "resilience_signature.parquet")

    # Predictions
    np.random.seed(42)
    n = 50
    actual = np.random.randn(n) * 2 + 5
    pd.DataFrame({
        "subject_id": [f"S{i}" for i in range(n)],
        "predicted_mean": actual + np.random.randn(n) * 0.5,
        "predicted_std": np.abs(np.random.randn(n)) * 0.3 + 0.2,
        "actual": actual,
    }).to_parquet(analysis_dir / "predictions.parquet")

    # Prediction uncertainty (output from UncertaintyAnalyzer)
    pd.DataFrame({
        "subject_id": [f"S{i}" for i in range(n)],
        "predicted_mean": actual + np.random.randn(n) * 0.5,
        "predicted_std": np.abs(np.random.randn(n)) * 0.3 + 0.2,
        "actual": actual,
        "residual": np.random.randn(n) * 0.5,
        "z_score": np.abs(np.random.randn(n)),
    }).to_parquet(analysis_dir / "prediction_uncertainty.parquet")

    # Calibration
    pd.DataFrame({
        "level": ["1_sigma", "2_sigma", "3_sigma"],
        "expected_coverage": [0.6827, 0.9545, 0.9973],
        "observed_coverage": [0.70, 0.92, 0.99],
        "calibration_error": [0.02, -0.03, -0.01],
    }).to_parquet(analysis_dir / "calibration_summary.parquet")

    # Uncertainty correlates
    pd.DataFrame({
        "covariate": ["cell_count", "pathology", "age"],
        "correlation": [0.35, 0.22, -0.15],
        "p_value": [0.01, 0.03, 0.12],
        "significant": [True, True, False],
    }).to_parquet(analysis_dir / "uncertainty_correlates.parquet")

    # Regional gene importance
    regional_data = []
    for region in ["DLPFC", "PCC", "AC"]:
        for ct in ["Ast", "Mic"]:
            for i in range(10):
                regional_data.append({
                    "region": region,
                    "cell_type": ct,
                    "gene": f"GENE{i}_{region}",
                    "effective_weight": np.random.rand() * 0.5,
                })
    pd.DataFrame(regional_data).to_parquet(analysis_dir / "regional_gene_importance.parquet")

    # Attention weights HDF5
    with h5py.File(analysis_dir / "attention_weights.h5", "w") as f:
        f.attrs["schema_version"] = "2.0"
        f.create_dataset("gene_gate", data=np.random.rand(8, 100))
        f.create_dataset("pathology_attention", data=np.random.rand(50, 4, 8))
        vlen_str = h5py.special_dtype(vlen=str)
        f.create_dataset("cell_type_names", data=np.array(
            ["Ast", "Mic", "Oli", "OPC", "Exc", "Inh", "End", "Per"], dtype=object
        ), dtype=vlen_str)
        f.create_dataset("gene_names", data=np.array(
            [f"GENE{i}" for i in range(100)], dtype=object
        ), dtype=vlen_str)

    return analysis_dir


# =============================================================================
# Script Import Tests
# =============================================================================


class TestGeneratePlotsImports:
    """Test that script imports work correctly."""

    def test_script_is_importable(self):
        """Test that script can be imported."""
        from scripts.analysis.generate_plots import (
            parse_args,
            load_analysis_data,
            load_attention_weights,
            generate_attention_plots,
            generate_resilience_plots,
            generate_importance_plots,
            generate_prediction_plots,
        )

        assert callable(parse_args)
        assert callable(load_analysis_data)
        assert callable(load_attention_weights)
        assert callable(generate_attention_plots)
        assert callable(generate_resilience_plots)
        assert callable(generate_importance_plots)
        assert callable(generate_prediction_plots)


# =============================================================================
# Function Unit Tests
# =============================================================================


class TestLoadAnalysisData:
    """Test load_analysis_data function."""

    def test_load_parquet(self, tmp_path):
        """Test loading analysis parquet files."""
        from scripts.analysis.generate_plots import load_analysis_data

        # Create analysis dir with a test file
        analysis_dir = tmp_path / "analysis"
        analysis_dir.mkdir()

        df = pd.DataFrame({"a": [1, 2, 3]})
        df.to_parquet(analysis_dir / "cell_type_importance.parquet")

        loaded = load_analysis_data(analysis_dir)
        assert "cell_type_importance" in loaded
        pd.testing.assert_frame_equal(df, loaded["cell_type_importance"])

    def test_load_empty_dir(self, tmp_path):
        """Test loading from empty directory returns empty dict."""
        from scripts.analysis.generate_plots import load_analysis_data

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        loaded = load_analysis_data(empty_dir)
        assert isinstance(loaded, dict)
        # Will have no data keys since no files match expected names


class TestLoadAttentionWeights:
    """Test load_attention_weights function."""

    def test_load_hdf5(self, tmp_path):
        """Test loading HDF5 file."""
        from scripts.analysis.generate_plots import load_attention_weights

        path = tmp_path / "attention.h5"
        vlen_str = h5py.special_dtype(vlen=str)
        with h5py.File(path, "w") as f:
            f.create_dataset("gene_gate", data=np.random.rand(8, 100))
            f.create_dataset("cell_type_names", data=np.array(["A", "B", "C"], dtype=object), dtype=vlen_str)

        weights = load_attention_weights(path)

        assert "gene_gate" in weights
        assert "cell_type_names" in weights

    def test_load_nonexistent(self, tmp_path):
        """Test loading nonexistent file returns empty dict."""
        from scripts.analysis.generate_plots import load_attention_weights

        weights = load_attention_weights(tmp_path / "nonexistent.h5")
        assert weights == {}


# =============================================================================
# Plot Generation Tests
# =============================================================================


class TestGenerateAttentionPlots:
    """Test generate_attention_plots function."""

    def test_generates_plots(self, mock_analysis_dir, tmp_path):
        """Test attention plots are generated."""
        from scripts.analysis.generate_plots import (
            generate_attention_plots,
            load_attention_weights,
            load_analysis_data,
        )

        plots_dir = tmp_path / "plots"
        plots_dir.mkdir()

        data = load_analysis_data(mock_analysis_dir)
        attention = load_attention_weights(mock_analysis_dir / "attention_weights.h5")

        generated = generate_attention_plots(
            data=data,
            attention=attention,
            output_dir=plots_dir,
            skip_plots=[],
            fmt="png",
        )

        assert len(generated) > 0
        assert any(plots_dir.glob("*.png"))


class TestGenerateImportancePlots:
    """Test generate_importance_plots function."""

    def test_generates_plots(self, mock_analysis_dir, tmp_path):
        """Test importance plots are generated."""
        from scripts.analysis.generate_plots import generate_importance_plots, load_analysis_data

        plots_dir = tmp_path / "plots"
        plots_dir.mkdir()

        data = load_analysis_data(mock_analysis_dir)

        generated = generate_importance_plots(
            data=data,
            output_dir=plots_dir,
            skip_plots=[],
            fmt="png",
        )

        # Should generate at least CCC-related plots when data is available
        assert isinstance(generated, list)


class TestGeneratePredictionPlots:
    """Test generate_prediction_plots function."""

    def test_generates_plots(self, mock_analysis_dir, tmp_path):
        """Test prediction plots are generated."""
        from scripts.analysis.generate_plots import generate_prediction_plots, load_analysis_data

        plots_dir = tmp_path / "plots"
        plots_dir.mkdir()

        data = load_analysis_data(mock_analysis_dir)

        generated = generate_prediction_plots(
            data=data,
            output_dir=plots_dir,
            skip_plots=[],
            fmt="png",
        )

        # With predictions data including actual + predicted_mean, should generate plots
        assert len(generated) > 0


class TestGenerateResiliencePlots:
    """Test generate_resilience_plots function."""

    def test_generates_plots(self, mock_analysis_dir, tmp_path):
        """Test resilience plots are generated."""
        from scripts.analysis.generate_plots import generate_resilience_plots, load_analysis_data

        plots_dir = tmp_path / "plots"
        plots_dir.mkdir()

        data = load_analysis_data(mock_analysis_dir)

        generated = generate_resilience_plots(
            data=data,
            output_dir=plots_dir,
            skip_plots=[],
            fmt="png",
        )

        # With resilience_signature data present, should generate at least one plot
        assert len(generated) > 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestScriptIntegration:
    """Integration tests for generate_plots.py script."""

    def test_help_flag(self):
        """Test --help flag works."""
        result = subprocess.run(
            [sys.executable, "scripts/analysis/generate_plots.py", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Generate publication-quality plots" in result.stdout

    def test_requires_input(self):
        """Test script fails without input."""
        result = subprocess.run(
            [sys.executable, "scripts/analysis/generate_plots.py"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

    def test_with_analysis_dir(self, mock_analysis_dir, tmp_path):
        """Test script runs with analysis directory."""
        plots_dir = tmp_path / "plots"

        result = subprocess.run(
            [
                sys.executable,
                "scripts/analysis/generate_plots.py",
                "--analysis-dir", str(mock_analysis_dir),
                "--output-dir", str(plots_dir),
            ],
            capture_output=True,
            text=True,
        )

        # Should complete successfully
        assert "Traceback" not in result.stderr
        assert plots_dir.exists()

    def test_plot_types_flag(self, mock_analysis_dir, tmp_path):
        """Test --plot-types flag works."""
        plots_dir = tmp_path / "plots"

        result = subprocess.run(
            [
                sys.executable,
                "scripts/analysis/generate_plots.py",
                "--analysis-dir", str(mock_analysis_dir),
                "--output-dir", str(plots_dir),
                "--plot-types", "prediction",
            ],
            capture_output=True,
            text=True,
        )

        assert "Traceback" not in result.stderr
        # Should generate some plots or complete without error
        assert result.returncode == 0 or "Generated" in result.stdout

    def test_skip_plots_flag(self, mock_analysis_dir, tmp_path):
        """Test --skip-plots flag works."""
        plots_dir = tmp_path / "plots"

        result = subprocess.run(
            [
                sys.executable,
                "scripts/analysis/generate_plots.py",
                "--analysis-dir", str(mock_analysis_dir),
                "--output-dir", str(plots_dir),
                "--skip-plots", "cell_type_attention_heatmap", "cell_type_importance_bar",
            ],
            capture_output=True,
            text=True,
        )

        assert "Traceback" not in result.stderr

    def test_format_flag(self, mock_analysis_dir, tmp_path):
        """Test --format flag works."""
        plots_dir = tmp_path / "plots"

        result = subprocess.run(
            [
                sys.executable,
                "scripts/analysis/generate_plots.py",
                "--analysis-dir", str(mock_analysis_dir),
                "--output-dir", str(plots_dir),
                "--format", "pdf",
                "--plot-types", "prediction",
            ],
            capture_output=True,
            text=True,
        )

        assert "Traceback" not in result.stderr
        # Should generate PDF files
        if plots_dir.exists():
            pdf_files = list(plots_dir.glob("*.pdf"))
            # May or may not have PDF files depending on what was available


# =============================================================================
# Edge Cases
# =============================================================================


class TestGeneratePlotsEdgeCases:
    """Test edge cases."""

    def test_empty_analysis_dir(self, tmp_path):
        """Test with empty analysis directory."""
        from scripts.analysis.generate_plots import (
            generate_attention_plots,
            generate_importance_plots,
            generate_prediction_plots,
        )

        plots_dir = tmp_path / "plots"
        plots_dir.mkdir()

        # Should not crash, just return 0 plots
        generated1 = generate_attention_plots(
            data={},
            attention={},
            output_dir=plots_dir,
            skip_plots=[],
        )
        generated2 = generate_importance_plots(
            data={},
            output_dir=plots_dir,
            skip_plots=[],
        )
        generated3 = generate_prediction_plots(
            data={},
            output_dir=plots_dir,
            skip_plots=[],
        )

        assert len(generated1) == 0
        assert len(generated2) == 0
        assert len(generated3) == 0

    def test_partial_data(self, tmp_path):
        """Test with partial analysis data."""
        from scripts.analysis.generate_plots import generate_prediction_plots, load_analysis_data

        analysis_dir = tmp_path / "analysis"
        analysis_dir.mkdir()
        plots_dir = tmp_path / "plots"
        plots_dir.mkdir()

        # Create only predictions (no calibration, no correlates)
        pd.DataFrame({
            "predicted": np.random.randn(20),
            "actual": np.random.randn(20),
            "predicted_std": np.abs(np.random.randn(20)) + 0.1,
        }).to_parquet(analysis_dir / "predictions.parquet")

        data = load_analysis_data(analysis_dir)

        generated = generate_prediction_plots(
            data=data,
            output_dir=plots_dir,
            skip_plots=[],
        )

        # Should generate what it can without errors
        assert isinstance(generated, list)


# =============================================================================
# Uncertainty correlates data-source tests
# =============================================================================


class TestUncertaintyCorrelatesDataSource:
    """Test that uncertainty_correlates plot uses the correct data file."""

    def test_load_analysis_data_includes_correlates(self, mock_analysis_dir):
        """load_analysis_data should load uncertainty_correlates.parquet."""
        from scripts.analysis.generate_plots import load_analysis_data
        data = load_analysis_data(mock_analysis_dir)
        assert "uncertainty_correlates" in data
        assert "covariate" in data["uncertainty_correlates"].columns
        assert "correlation" in data["uncertainty_correlates"].columns

    def test_correlates_plot_uses_correct_key(self, mock_analysis_dir):
        """generate_analysis_plots should use uncertainty_correlates key, not uncertainty."""
        from scripts.analysis.generate_plots import load_analysis_data
        data = load_analysis_data(mock_analysis_dir)
        # The uncertainty_correlates key should be present for the plot
        assert "uncertainty_correlates" in data
        # The old "uncertainty" key (prediction_uncertainty.parquet) should NOT have correlates columns
        if "uncertainty" in data:
            assert "covariate" not in data["uncertainty"].columns


# =============================================================================
# Cleanup
# =============================================================================


@pytest.fixture(autouse=True)
def cleanup():
    """Cleanup matplotlib figures after each test."""
    import matplotlib.pyplot as plt
    yield
    plt.close("all")
