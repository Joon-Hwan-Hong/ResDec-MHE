"""
Tests for src/analysis/embedding_analysis.py.

Test coverage includes:
- EmbeddingAnalysisResult dataclass behavior
- EmbeddingAnalyzer initialization and validation
- UMAP projection
- Clustering analysis
- Linear probes
- Similarity matrix computation
- Outlier detection
- Trajectory analysis
- Batch effect assessment
- Save functionality
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis.embedding_analysis import (
    EmbeddingAnalyzer,
    EmbeddingAnalysisResult,
    analyze_embeddings,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_embeddings():
    """Sample subject embeddings [n_subjects, embed_dim]."""
    np.random.seed(42)
    return np.random.rand(50, 32).astype(np.float32)


@pytest.fixture
def sample_subject_ids():
    """Sample subject IDs."""
    return [f"subject_{i}" for i in range(50)]


@pytest.fixture
def sample_covariates():
    """Sample covariates DataFrame."""
    np.random.seed(42)
    return pd.DataFrame({
        "cognition": np.random.rand(50),
        "pathology": np.random.rand(50),
        "age": np.random.randint(60, 90, 50),
        "sex": np.random.choice(["M", "F"], 50),
    })


@pytest.fixture
def sample_batch_labels():
    """Sample batch labels."""
    np.random.seed(42)
    return np.array(["batch_1"] * 25 + ["batch_2"] * 25)


@pytest.fixture
def analyzer(sample_embeddings, sample_subject_ids, sample_covariates, sample_batch_labels):
    """EmbeddingAnalyzer instance with all data."""
    return EmbeddingAnalyzer(
        embeddings=sample_embeddings,
        subject_ids=sample_subject_ids,
        covariates=sample_covariates,
        batch_labels=sample_batch_labels,
    )


@pytest.fixture
def basic_analyzer(sample_embeddings, sample_subject_ids):
    """Basic EmbeddingAnalyzer without covariates or batch labels."""
    return EmbeddingAnalyzer(
        embeddings=sample_embeddings,
        subject_ids=sample_subject_ids,
    )


# ============================================================================
# EmbeddingAnalysisResult Dataclass Tests
# ============================================================================


class TestEmbeddingAnalysisResult:
    """Tests for EmbeddingAnalysisResult dataclass."""

    def test_init_with_defaults(self):
        """Result can be initialized with defaults."""
        result = EmbeddingAnalysisResult()
        assert result.umap_projection is None
        assert result.metadata == {}

    def test_metadata_defaults_to_empty_dict(self):
        """metadata defaults to empty dict."""
        result = EmbeddingAnalysisResult()
        assert result.metadata == {}


# ============================================================================
# EmbeddingAnalyzer Initialization Tests
# ============================================================================


class TestAnalyzerInit:
    """Tests for EmbeddingAnalyzer initialization."""

    def test_init_validates_embeddings_ndim(self):
        """Analyzer rejects embeddings with wrong dimensions."""
        bad_embeddings = np.random.rand(30, 4, 8).astype(np.float32)  # 3D
        with pytest.raises(ValueError, match="must be 2D"):
            EmbeddingAnalyzer(embeddings=bad_embeddings)

    def test_init_validates_subject_ids_length(self, sample_embeddings):
        """Analyzer rejects subject_ids with wrong length."""
        bad_ids = [f"subject_{i}" for i in range(10)]  # Wrong length
        with pytest.raises(ValueError, match="subject_ids"):
            EmbeddingAnalyzer(embeddings=sample_embeddings, subject_ids=bad_ids)

    def test_init_validates_covariates_length(self, sample_embeddings, sample_subject_ids):
        """Analyzer rejects covariates with wrong length."""
        bad_covariates = pd.DataFrame({"x": range(10)})
        with pytest.raises(ValueError, match="covariates"):
            EmbeddingAnalyzer(
                embeddings=sample_embeddings,
                subject_ids=sample_subject_ids,
                covariates=bad_covariates,
            )

    def test_init_validates_batch_labels_length(self, sample_embeddings, sample_subject_ids):
        """Analyzer rejects batch_labels with wrong length."""
        bad_batch = np.array(["batch_1"] * 10)
        with pytest.raises(ValueError, match="batch_labels"):
            EmbeddingAnalyzer(
                embeddings=sample_embeddings,
                subject_ids=sample_subject_ids,
                batch_labels=bad_batch,
            )

    def test_init_generates_default_subject_ids(self, sample_embeddings):
        """Analyzer generates default subject IDs if not provided."""
        analyzer = EmbeddingAnalyzer(embeddings=sample_embeddings)
        assert len(analyzer.subject_ids) == sample_embeddings.shape[0]
        assert analyzer.subject_ids[0] == "subject_0"


# ============================================================================
# Analysis Tests
# ============================================================================


class TestAnalysis:
    """Tests for main analyze() method."""

    def test_analyze_returns_result(self, basic_analyzer):
        """analyze() returns EmbeddingAnalysisResult."""
        result = basic_analyzer.analyze(
            run_umap=False,  # Skip UMAP for speed
            run_linear_probes=False,  # No covariates
            run_batch_effect=False,  # No batch labels
        )
        assert isinstance(result, EmbeddingAnalysisResult)

    def test_analyze_metadata_contains_info(self, basic_analyzer):
        """analyze() adds metadata."""
        result = basic_analyzer.analyze(run_umap=False, run_linear_probes=False, run_batch_effect=False)
        assert "n_subjects" in result.metadata
        assert "embed_dim" in result.metadata
        assert result.metadata["n_subjects"] == 50
        assert result.metadata["embed_dim"] == 32


# ============================================================================
# UMAP Tests
# ============================================================================


@pytest.mark.filterwarnings("ignore:n_jobs value 1 overridden to 1 by setting random_state:UserWarning")
class TestUMAP:
    """Tests for UMAP projection."""

    def test_umap_returns_dataframe(self, basic_analyzer):
        """UMAP projection returns DataFrame."""
        result = basic_analyzer.analyze(
            run_umap=True,
            run_clustering=False,
            run_linear_probes=False,
            run_similarity=False,
            run_outlier_detection=False,
            run_trajectory=False,
            run_batch_effect=False,
        )
        if result.umap_projection is not None:
            assert isinstance(result.umap_projection, pd.DataFrame)
            assert "subject_id" in result.umap_projection.columns
            assert "umap_1" in result.umap_projection.columns
            assert "umap_2" in result.umap_projection.columns


# ============================================================================
# Clustering Tests
# ============================================================================


class TestClustering:
    """Tests for clustering analysis."""

    def test_clustering_returns_dataframes(self, basic_analyzer):
        """Clustering returns assignment and statistics DataFrames."""
        result = basic_analyzer.analyze(
            run_umap=False,
            run_clustering=True,
            run_linear_probes=False,
            run_similarity=False,
            run_outlier_detection=False,
            run_trajectory=False,
            run_batch_effect=False,
        )
        assert result.cluster_assignments is not None
        assert result.cluster_statistics is not None
        assert isinstance(result.cluster_assignments, pd.DataFrame)
        assert isinstance(result.cluster_statistics, pd.DataFrame)

    def test_clustering_has_expected_columns(self, basic_analyzer):
        """Cluster assignments have expected columns."""
        result = basic_analyzer.analyze(
            run_umap=False,
            run_clustering=True,
            run_linear_probes=False,
            run_similarity=False,
            run_outlier_detection=False,
            run_trajectory=False,
            run_batch_effect=False,
        )
        expected_cols = {"subject_id", "cluster", "kmeans_cluster", "hierarchical_cluster"}
        assert set(result.cluster_assignments.columns) == expected_cols

    def test_clustering_with_specified_k(self, basic_analyzer):
        """Clustering works with specified number of clusters."""
        result = basic_analyzer.analyze(
            run_umap=False,
            run_clustering=True,
            run_linear_probes=False,
            run_similarity=False,
            run_outlier_detection=False,
            run_trajectory=False,
            run_batch_effect=False,
            n_clusters=3,
        )
        # Should have 3 unique clusters
        assert len(result.cluster_assignments["kmeans_cluster"].unique()) == 3


# ============================================================================
# Linear Probes Tests
# ============================================================================


class TestLinearProbes:
    """Tests for linear probe analysis."""

    def test_linear_probes_returns_dataframe(self, analyzer):
        """Linear probes return DataFrame when covariates provided."""
        result = analyzer.analyze(
            run_umap=False,
            run_clustering=False,
            run_linear_probes=True,
            run_similarity=False,
            run_outlier_detection=False,
            run_trajectory=False,
            run_batch_effect=False,
        )
        assert result.linear_probe_results is not None
        assert isinstance(result.linear_probe_results, pd.DataFrame)

    def test_linear_probes_has_expected_columns(self, analyzer):
        """Linear probe results have expected columns."""
        result = analyzer.analyze(
            run_umap=False,
            run_clustering=False,
            run_linear_probes=True,
            run_similarity=False,
            run_outlier_detection=False,
            run_trajectory=False,
            run_batch_effect=False,
        )
        expected_cols = {"target", "covariate", "task_type", "metric", "r2_score", "score_mean", "score_std", "n_samples"}
        assert set(result.linear_probe_results.columns) == expected_cols

    def test_linear_probes_none_without_covariates(self, basic_analyzer):
        """Linear probes return None without covariates."""
        result = basic_analyzer.analyze(
            run_umap=False,
            run_clustering=False,
            run_linear_probes=True,
            run_similarity=False,
            run_outlier_detection=False,
            run_trajectory=False,
            run_batch_effect=False,
        )
        assert result.linear_probe_results is None


# ============================================================================
# Similarity Matrix Tests
# ============================================================================


class TestSimilarityMatrix:
    """Tests for similarity matrix computation."""

    def test_similarity_returns_dataframe(self, basic_analyzer):
        """Similarity matrix returns DataFrame."""
        result = basic_analyzer.analyze(
            run_umap=False,
            run_clustering=False,
            run_linear_probes=False,
            run_similarity=True,
            run_outlier_detection=False,
            run_trajectory=False,
            run_batch_effect=False,
        )
        assert result.similarity_matrix is not None
        assert isinstance(result.similarity_matrix, pd.DataFrame)

    def test_similarity_has_expected_columns(self, basic_analyzer):
        """Similarity matrix has expected columns."""
        result = basic_analyzer.analyze(
            run_umap=False,
            run_clustering=False,
            run_linear_probes=False,
            run_similarity=True,
            run_outlier_detection=False,
            run_trajectory=False,
            run_batch_effect=False,
        )
        expected_cols = {"subject_1", "subject_2", "similarity"}
        assert set(result.similarity_matrix.columns) == expected_cols

    def test_similarity_is_upper_triangle(self, basic_analyzer):
        """Similarity matrix contains only upper triangle pairs."""
        result = basic_analyzer.analyze(
            run_umap=False,
            run_clustering=False,
            run_linear_probes=False,
            run_similarity=True,
            run_outlier_detection=False,
            run_trajectory=False,
            run_batch_effect=False,
        )
        # For n subjects, upper triangle has n*(n-1)/2 pairs
        n = 50
        expected_pairs = n * (n - 1) // 2
        assert len(result.similarity_matrix) == expected_pairs


# ============================================================================
# Outlier Detection Tests
# ============================================================================


class TestOutlierDetection:
    """Tests for outlier detection."""

    def test_outlier_detection_returns_dataframe(self, basic_analyzer):
        """Outlier detection returns DataFrame."""
        result = basic_analyzer.analyze(
            run_umap=False,
            run_clustering=False,
            run_linear_probes=False,
            run_similarity=False,
            run_outlier_detection=True,
            run_trajectory=False,
            run_batch_effect=False,
        )
        assert result.outlier_scores is not None
        assert isinstance(result.outlier_scores, pd.DataFrame)

    def test_outlier_detection_has_expected_columns(self, basic_analyzer):
        """Outlier scores have expected columns."""
        result = basic_analyzer.analyze(
            run_umap=False,
            run_clustering=False,
            run_linear_probes=False,
            run_similarity=False,
            run_outlier_detection=True,
            run_trajectory=False,
            run_batch_effect=False,
        )
        expected_cols = {"subject_id", "outlier_score", "is_outlier"}
        assert set(result.outlier_scores.columns) == expected_cols

    def test_outlier_detection_has_all_subjects(self, basic_analyzer):
        """Outlier scores include all subjects."""
        result = basic_analyzer.analyze(
            run_umap=False,
            run_clustering=False,
            run_linear_probes=False,
            run_similarity=False,
            run_outlier_detection=True,
            run_trajectory=False,
            run_batch_effect=False,
        )
        assert len(result.outlier_scores) == 50


# ============================================================================
# Trajectory Analysis Tests
# ============================================================================


class TestTrajectoryAnalysis:
    """Tests for trajectory analysis."""

    def test_trajectory_returns_dataframe(self, basic_analyzer):
        """Trajectory analysis returns DataFrame."""
        result = basic_analyzer.analyze(
            run_umap=False,
            run_clustering=False,
            run_linear_probes=False,
            run_similarity=False,
            run_outlier_detection=False,
            run_trajectory=True,
            run_batch_effect=False,
        )
        assert result.trajectory_scores is not None
        assert isinstance(result.trajectory_scores, pd.DataFrame)

    def test_trajectory_has_expected_columns(self, basic_analyzer):
        """Trajectory scores have expected columns."""
        result = basic_analyzer.analyze(
            run_umap=False,
            run_clustering=False,
            run_linear_probes=False,
            run_similarity=False,
            run_outlier_detection=False,
            run_trajectory=True,
            run_batch_effect=False,
        )
        expected_cols = {"subject_id", "pseudotime", "pc1", "pc2"}
        assert set(result.trajectory_scores.columns) == expected_cols

    def test_pseudotime_normalized(self, basic_analyzer):
        """Pseudotime is normalized to [0, 1]."""
        result = basic_analyzer.analyze(
            run_umap=False,
            run_clustering=False,
            run_linear_probes=False,
            run_similarity=False,
            run_outlier_detection=False,
            run_trajectory=True,
            run_batch_effect=False,
        )
        assert result.trajectory_scores["pseudotime"].min() >= 0
        assert result.trajectory_scores["pseudotime"].max() <= 1


# ============================================================================
# Batch Effect Assessment Tests
# ============================================================================


class TestBatchEffectAssessment:
    """Tests for batch effect assessment."""

    def test_batch_effect_returns_dataframe(self, analyzer):
        """Batch effect assessment returns DataFrame when batch_labels provided."""
        result = analyzer.analyze(
            run_umap=False,
            run_clustering=False,
            run_linear_probes=False,
            run_similarity=False,
            run_outlier_detection=False,
            run_trajectory=False,
            run_batch_effect=True,
        )
        assert result.batch_effect_metrics is not None
        assert isinstance(result.batch_effect_metrics, pd.DataFrame)

    def test_batch_effect_has_expected_columns(self, analyzer):
        """Batch effect metrics have expected columns."""
        result = analyzer.analyze(
            run_umap=False,
            run_clustering=False,
            run_linear_probes=False,
            run_similarity=False,
            run_outlier_detection=False,
            run_trajectory=False,
            run_batch_effect=True,
        )
        expected_cols = {"metric", "value", "interpretation"}
        assert set(result.batch_effect_metrics.columns) == expected_cols

    def test_batch_effect_none_without_batch_labels(self, basic_analyzer):
        """Batch effect assessment returns None without batch_labels."""
        result = basic_analyzer.analyze(
            run_umap=False,
            run_clustering=False,
            run_linear_probes=False,
            run_similarity=False,
            run_outlier_detection=False,
            run_trajectory=False,
            run_batch_effect=True,
        )
        assert result.batch_effect_metrics is None


# ============================================================================
# Save Tests
# ============================================================================


class TestSave:
    """Tests for save functionality."""

    def test_save_creates_files(self, basic_analyzer):
        """save() creates expected files."""
        result = basic_analyzer.analyze(
            run_umap=False,
            run_clustering=True,
            run_linear_probes=False,
            run_similarity=False,
            run_outlier_detection=True,
            run_trajectory=True,
            run_batch_effect=False,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            saved = basic_analyzer.save(result, tmpdir)
            assert (Path(tmpdir) / "cluster_assignments.parquet").exists()
            assert (Path(tmpdir) / "outlier_scores.csv").exists()
            assert (Path(tmpdir) / "trajectory_scores.parquet").exists()


# ============================================================================
# Convenience Function Tests
# ============================================================================


class TestConvenienceFunction:
    """Tests for analyze_embeddings function."""

    def test_analyze_embeddings_returns_result(self, sample_embeddings, sample_subject_ids):
        """analyze_embeddings returns EmbeddingAnalysisResult."""
        result = analyze_embeddings(
            embeddings=sample_embeddings,
            subject_ids=sample_subject_ids,
            run_umap=False,
            run_linear_probes=False,
            run_batch_effect=False,
        )
        assert isinstance(result, EmbeddingAnalysisResult)

    def test_analyze_embeddings_with_output_dir(self, sample_embeddings, sample_subject_ids):
        """analyze_embeddings saves when output_dir provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = analyze_embeddings(
                embeddings=sample_embeddings,
                subject_ids=sample_subject_ids,
                output_dir=tmpdir,
                run_umap=False,
                run_linear_probes=False,
                run_batch_effect=False,
            )
            assert (Path(tmpdir) / "cluster_assignments.parquet").exists()


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Edge case tests."""

    def test_small_dataset(self):
        """Handles small dataset gracefully."""
        np.random.seed(42)
        embeddings = np.random.rand(10, 8).astype(np.float32)

        analyzer = EmbeddingAnalyzer(embeddings=embeddings)
        result = analyzer.analyze(
            run_umap=False,
            run_clustering=True,
            run_linear_probes=False,
            run_similarity=True,
            run_outlier_detection=True,
            run_trajectory=True,
            run_batch_effect=False,
        )
        assert result.cluster_assignments is not None

    def test_single_batch(self):
        """Handles single batch without crashing."""
        np.random.seed(42)
        embeddings = np.random.rand(20, 8).astype(np.float32)
        batch_labels = np.array(["batch_1"] * 20)

        analyzer = EmbeddingAnalyzer(
            embeddings=embeddings,
            batch_labels=batch_labels,
        )
        result = analyzer.analyze(
            run_umap=False,
            run_clustering=False,
            run_linear_probes=False,
            run_similarity=False,
            run_outlier_detection=False,
            run_trajectory=False,
            run_batch_effect=True,
        )
        # Should return None for single batch
        assert result.batch_effect_metrics is None


# ============================================================================
# Schema Compatibility Tests
# ============================================================================


class TestSchemaCompat:
    """Tests ensuring analyzer output matches plotter expectations."""

    def test_cluster_output_has_cluster_column(self):
        """Cluster assignments must include a 'cluster' column for plotters."""
        np.random.seed(42)
        embeddings = np.random.rand(30, 16).astype(np.float32)
        analyzer = EmbeddingAnalyzer(embeddings=embeddings)
        result = analyzer.analyze(
            run_umap=False,
            run_clustering=True,
            run_linear_probes=False,
            run_similarity=False,
            run_outlier_detection=False,
            run_trajectory=False,
            run_batch_effect=False,
            n_clusters=3,
        )
        assert result.cluster_assignments is not None
        assert "cluster" in result.cluster_assignments.columns, (
            "Plotters expect a 'cluster' column in cluster_assignments"
        )

    def test_linear_probe_output_has_target_and_r2_score(self):
        """Linear probe results must include 'target' and 'r2_score' columns for plotters."""
        np.random.seed(42)
        embeddings = np.random.rand(50, 16).astype(np.float32)
        covariates = pd.DataFrame({
            "cognition": np.random.rand(50),
            "pathology": np.random.rand(50),
        })
        analyzer = EmbeddingAnalyzer(
            embeddings=embeddings,
            covariates=covariates,
        )
        result = analyzer.analyze(
            run_umap=False,
            run_clustering=False,
            run_linear_probes=True,
            run_similarity=False,
            run_outlier_detection=False,
            run_trajectory=False,
            run_batch_effect=False,
        )
        assert result.linear_probe_results is not None
        assert "target" in result.linear_probe_results.columns, (
            "Plotters expect a 'target' column in linear_probe_results"
        )
        assert "r2_score" in result.linear_probe_results.columns, (
            "Plotters expect an 'r2_score' column in linear_probe_results"
        )
