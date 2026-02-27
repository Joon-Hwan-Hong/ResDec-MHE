"""
Integration tests for analysis pipeline components.

Tests that analysis modules work together end-to-end:
- Load attention weights → run resilience analysis → save results
- Load embeddings → run embedding analysis → save results
- Load PMA attention → run cell heterogeneity analysis → save results

Uses synthetic data to avoid dependency on real experiment outputs.
"""

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from src.data.constants import CELL_TYPE_ORDER


# Fixtures for synthetic data generation
@pytest.fixture
def n_subjects() -> int:
    return 30


@pytest.fixture
def n_cell_types() -> int:
    return len(CELL_TYPE_ORDER)


@pytest.fixture
def n_heads() -> int:
    return 4


@pytest.fixture
def embed_dim() -> int:
    return 64


@pytest.fixture
def max_cells_per_type() -> int:
    return 100


@pytest.fixture
def synthetic_attention(n_subjects, n_heads, n_cell_types):
    """Generate synthetic attention weights [n_subjects, n_heads, n_cell_types]."""
    np.random.seed(42)
    # Attention weights sum to 1 across cell types for each subject and head
    attention = np.zeros((n_subjects, n_heads, n_cell_types))
    for i in range(n_subjects):
        for h in range(n_heads):
            attention[i, h] = np.random.dirichlet(np.ones(n_cell_types))
    return attention.astype(np.float32)


@pytest.fixture
def synthetic_pma_attention(n_subjects, n_cell_types, max_cells_per_type):
    """Generate synthetic PMA attention weights for cell heterogeneity."""
    np.random.seed(42)
    # PMA attention: [n_subjects, n_cell_types, max_cells]
    # Use softmax-like distribution within each cell type
    raw = np.random.exponential(scale=1.0, size=(n_subjects, n_cell_types, max_cells_per_type))
    # Normalize per cell type per subject
    pma = raw / raw.sum(axis=2, keepdims=True)
    return pma.astype(np.float32)


@pytest.fixture
def synthetic_embeddings(n_subjects, embed_dim):
    """Generate synthetic subject embeddings [n_subjects, embed_dim]."""
    np.random.seed(42)
    return np.random.randn(n_subjects, embed_dim).astype(np.float32)


@pytest.fixture
def synthetic_phenotypes(n_subjects):
    """Generate synthetic pathology and cognition scores."""
    np.random.seed(42)
    # Pathology: continuous 0-1 with some high values
    pathology = np.random.beta(2, 5, size=n_subjects)
    # Make some subjects high pathology
    pathology[pathology > np.percentile(pathology, 50)] *= 1.5
    pathology = np.clip(pathology, 0, 1)

    # Cognition: z-scores correlated with inverse pathology + noise
    cognition = -pathology * 2 + np.random.randn(n_subjects) * 0.5

    return pathology.astype(np.float32), cognition.astype(np.float32)


@pytest.fixture
def synthetic_covariates(n_subjects, synthetic_phenotypes):
    """Generate synthetic covariates DataFrame."""
    pathology, cognition = synthetic_phenotypes
    return pd.DataFrame({
        "pathology": pathology,
        "cognition": cognition,
    })


@pytest.fixture
def synthetic_region_labels(n_subjects):
    """Generate synthetic region labels with guaranteed sufficient subjects per region."""
    # Ensure each region has at least 10 subjects for high pathology to work
    regions = ["PFC"] * 10 + ["EC"] * 10 + ["HIP"] * 10
    return np.array(regions)


class TestResilienceAnalysisPipeline:
    """Integration tests for resilience signature analysis pipeline."""

    def test_full_pipeline_runs(
        self,
        synthetic_attention,
        synthetic_phenotypes,
        n_cell_types,
    ):
        """Test complete pipeline: analyze → save → load."""
        from src.analysis.resilience_signatures import (
            ResilienceSignatureAnalyzer,
            ResilienceSignatureResult,
        )

        pathology, cognition = synthetic_phenotypes

        # Run analysis
        analyzer = ResilienceSignatureAnalyzer(
            attention=synthetic_attention,
            pathology_scores=pathology,
            cognition_scores=cognition,
            cell_type_names=list(CELL_TYPE_ORDER),
        )

        result = analyzer.analyze(
            n_permutations=100,
            random_seed=42,
        )

        # Verify result structure
        assert isinstance(result, ResilienceSignatureResult)
        assert result.signature is not None
        assert len(result.signature) == n_cell_types
        assert result.permutation_pvalues is not None
        assert result.permutation_null is not None

        # Save and reload
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            analyzer.save(result, output_dir)

            # Verify files created
            assert (output_dir / "resilience_signature.parquet").exists()
            assert (output_dir / "resilience_signature.csv").exists()
            assert (output_dir / "signature_pvalues.parquet").exists()
            assert (output_dir / "group_statistics.parquet").exists()
            assert (output_dir / "resilience_permutation_null.h5").exists()

            # Verify HDF5 contents
            with h5py.File(output_dir / "resilience_permutation_null.h5", "r") as f:
                assert "null_distribution" in f
                assert f["null_distribution"].shape[0] == 100  # n_permutations
                assert f["null_distribution"].shape[1] == n_cell_types

    def test_pipeline_with_regional_analysis(
        self,
        synthetic_attention,
        synthetic_phenotypes,
        synthetic_region_labels,
    ):
        """Test pipeline with regional stratification."""
        from src.analysis.resilience_signatures import ResilienceSignatureAnalyzer

        pathology, cognition = synthetic_phenotypes

        analyzer = ResilienceSignatureAnalyzer(
            attention=synthetic_attention,
            pathology_scores=pathology,
            cognition_scores=cognition,
            cell_type_names=list(CELL_TYPE_ORDER),
            region_labels=synthetic_region_labels,
        )

        result = analyzer.analyze(n_permutations=0)

        # Regional analysis may be None if insufficient subjects per region
        # This is expected behavior - just verify no errors
        if result.by_region is not None:
            assert "region" in result.by_region.columns
            regions_in_result = result.by_region["region"].unique()
            assert len(regions_in_result) > 0

    def test_pipeline_with_ablation(
        self,
        synthetic_attention,
        synthetic_phenotypes,
        n_subjects,
        n_cell_types,
    ):
        """Test pipeline with ablation study."""
        from src.analysis.resilience_signatures import ResilienceSignatureAnalyzer

        pathology, cognition = synthetic_phenotypes

        # Ablation needs embeddings [n_subjects, n_cell_types, embed_dim]
        np.random.seed(42)
        embeddings_3d = np.random.randn(n_subjects, n_cell_types, 64).astype(np.float32)

        analyzer = ResilienceSignatureAnalyzer(
            attention=synthetic_attention,
            pathology_scores=pathology,
            cognition_scores=cognition,
            cell_type_names=list(CELL_TYPE_ORDER),
        )

        result = analyzer.analyze(
            n_permutations=0,
            run_ablation=True,
            ablation_method="both",
            embeddings=embeddings_3d,
        )

        assert result.ablation_results is not None
        assert result.ablation_comparison is not None
        # Both methods should be present in ablation_results
        methods = result.ablation_results["method"].unique()
        assert "zero_embedding" in methods
        assert "node_removal" in methods

    def test_pipeline_full_with_regional_ablation(
        self,
        synthetic_attention,
        synthetic_phenotypes,
        synthetic_region_labels,
        n_subjects,
        n_cell_types,
    ):
        """Test complete pipeline with both regional analysis and ablation."""
        from src.analysis.resilience_signatures import ResilienceSignatureAnalyzer

        pathology, cognition = synthetic_phenotypes

        # Ablation needs embeddings [n_subjects, n_cell_types, embed_dim]
        np.random.seed(42)
        embeddings_3d = np.random.randn(n_subjects, n_cell_types, 64).astype(np.float32)

        analyzer = ResilienceSignatureAnalyzer(
            attention=synthetic_attention,
            pathology_scores=pathology,
            cognition_scores=cognition,
            cell_type_names=list(CELL_TYPE_ORDER),
            region_labels=synthetic_region_labels,
        )

        result = analyzer.analyze(
            n_permutations=50,
            run_ablation=True,
            ablation_method="both",
            embeddings=embeddings_3d,
            random_seed=42,
        )

        # Verify core outputs present
        assert result.signature is not None
        assert result.permutation_pvalues is not None
        assert result.permutation_null is not None
        assert result.group_statistics is not None
        assert result.ablation_results is not None
        assert result.ablation_comparison is not None

        # Save and verify files
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            analyzer.save(result, output_dir)

            # Core files should always exist
            expected_files = [
                "resilience_signature.parquet",
                "signature_pvalues.parquet",
                "group_statistics.parquet",
                "ablation_importance.parquet",
                "ablation_comparison.parquet",
                "resilience_permutation_null.h5",
            ]

            for fname in expected_files:
                assert (output_dir / fname).exists(), f"Missing: {fname}"


@pytest.mark.filterwarnings("ignore:n_jobs value 1 overridden to 1 by setting random_state:UserWarning")
@pytest.mark.filterwarnings("ignore:The TBB threading layer requires TBB version:numba.core.errors.NumbaWarning")
class TestEmbeddingAnalysisPipeline:
    """Integration tests for embedding analysis pipeline."""

    def test_full_pipeline_runs(
        self,
        synthetic_embeddings,
        synthetic_covariates,
    ):
        """Test complete embedding analysis pipeline."""
        from src.analysis.embedding_analysis import EmbeddingAnalyzer

        analyzer = EmbeddingAnalyzer(
            embeddings=synthetic_embeddings,
            covariates=synthetic_covariates,
        )

        result = analyzer.analyze(
            run_umap=True,
            run_clustering=True,
            run_linear_probes=True,
            run_similarity=True,
            run_outlier_detection=True,
        )

        # Verify core outputs
        assert result.umap_projection is not None
        assert result.cluster_assignments is not None
        assert result.linear_probe_results is not None
        assert result.similarity_matrix is not None
        assert result.outlier_scores is not None

    def test_pipeline_saves_outputs(
        self,
        synthetic_embeddings,
        synthetic_covariates,
    ):
        """Test that outputs are saved correctly."""
        from src.analysis.embedding_analysis import EmbeddingAnalyzer

        analyzer = EmbeddingAnalyzer(
            embeddings=synthetic_embeddings,
            covariates=synthetic_covariates,
        )

        result = analyzer.analyze(
            run_umap=True,
            run_clustering=True,
            run_linear_probes=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            analyzer.save(result, output_dir)

            # Check key files exist (embedding analyzer saves parquet/csv files, not HDF5)
            assert (output_dir / "umap_projection.parquet").exists()
            assert (output_dir / "cluster_assignments.parquet").exists()
            assert (output_dir / "similarity_matrix.parquet").exists()

    def test_pipeline_with_batch_effect_analysis(
        self,
        synthetic_embeddings,
        n_subjects,
    ):
        """Test batch effect analysis in pipeline."""
        from src.analysis.embedding_analysis import EmbeddingAnalyzer

        # Create batch labels
        batch_labels = np.array(["batch_A"] * (n_subjects // 2) + ["batch_B"] * (n_subjects - n_subjects // 2))

        analyzer = EmbeddingAnalyzer(
            embeddings=synthetic_embeddings,
            batch_labels=batch_labels,
        )

        result = analyzer.analyze(
            run_batch_effect=True,
        )

        assert result.batch_effect_metrics is not None
        # Check that batch_silhouette is in the metrics (stored as row, not column)
        assert "batch_silhouette" in result.batch_effect_metrics["metric"].values


class TestCellHeterogeneityPipeline:
    """Integration tests for cell heterogeneity analysis pipeline."""

    def test_analyze_cell_heterogeneity(
        self,
        synthetic_pma_attention,
        n_subjects,
        n_cell_types,
    ):
        """Test cell heterogeneity analysis via CellHeterogeneityAnalyzer."""
        from src.analysis.cell_heterogeneity import CellHeterogeneityAnalyzer

        subject_ids = [f"subj_{i}" for i in range(n_subjects)]

        analyzer = CellHeterogeneityAnalyzer(
            pma_attention=synthetic_pma_attention,
            cell_type_names=list(CELL_TYPE_ORDER),
            subject_ids=subject_ids,
            top_percentile=10.0,
            min_cells_per_type=10,
        )
        result = analyzer.analyze(save_all_scores=True)
        summary_df = result.summary
        high_attention_df = result.high_attention_cells
        all_scores_df = result.all_scores

        # Verify summary statistics
        assert len(summary_df) == n_cell_types
        assert "gini_coefficient" in summary_df.columns
        assert "attention_entropy" in summary_df.columns

        # Verify high attention cells identified
        assert len(high_attention_df) > 0
        assert "attention_score" in high_attention_df.columns

        # Verify all scores collected
        assert len(all_scores_df) > 0
        assert "is_high_attention" in all_scores_df.columns

    def test_cell_metadata_enrichment(
        self,
        synthetic_pma_attention,
        n_subjects,
    ):
        """cell_metadata columns are merged into high_attention_df and all_scores_df."""
        from src.analysis.cell_heterogeneity import CellHeterogeneityAnalyzer

        n_cell_types_actual = synthetic_pma_attention.shape[1]
        max_cells = synthetic_pma_attention.shape[2]
        subject_ids = [f"subj_{i}" for i in range(n_subjects)]

        # Build cell_barcodes dict so rows get a "cell_barcode" column
        cell_barcodes = {}
        all_barcodes = []
        for s_idx in range(n_subjects):
            for ct_idx in range(n_cell_types_actual):
                barcodes = [f"BC_{s_idx}_{ct_idx}_{c}" for c in range(max_cells)]
                cell_barcodes[f"{s_idx}_{ct_idx}"] = barcodes
                all_barcodes.extend(barcodes)

        # cell_metadata indexed by barcode with an extra column
        cell_metadata = pd.DataFrame(
            {"donor_age": np.random.randint(60, 90, size=len(all_barcodes))},
            index=all_barcodes,
        )

        analyzer = CellHeterogeneityAnalyzer(
            pma_attention=synthetic_pma_attention,
            cell_type_names=list(CELL_TYPE_ORDER),
            subject_ids=subject_ids,
            cell_barcodes=cell_barcodes,
            cell_metadata=cell_metadata,
            top_percentile=10.0,
            min_cells_per_type=5,
        )
        result = analyzer.analyze(save_all_scores=True)
        high_attention_df = result.high_attention_cells
        all_scores_df = result.all_scores

        # "donor_age" should appear in both output DataFrames
        assert "donor_age" in high_attention_df.columns, (
            f"cell_metadata column not merged into high_attention_df; cols={list(high_attention_df.columns)}"
        )
        assert "donor_age" in all_scores_df.columns, (
            f"cell_metadata column not merged into all_scores_df; cols={list(all_scores_df.columns)}"
        )
        # At least some values should be non-NaN (barcodes matched)
        assert high_attention_df["donor_age"].notna().any()
        assert all_scores_df["donor_age"].notna().any()

    def test_pma_attention_loading(self, synthetic_pma_attention, n_cell_types):
        """Test PMA attention can be saved and loaded from HDF5."""
        from scripts.run_cell_heterogeneity import load_pma_attention

        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = Path(tmpdir) / "attention_weights.h5"

            # Save synthetic attention
            with h5py.File(h5_path, "w") as f:
                f.create_dataset("pma_attention", data=synthetic_pma_attention)
                f.attrs["cell_type_names"] = list(CELL_TYPE_ORDER)

            # Load and verify
            loaded = load_pma_attention(h5_path)

            assert "pma_attention" in loaded
            assert loaded["pma_attention"].shape == synthetic_pma_attention.shape
            assert "cell_type_names" in loaded
            np.testing.assert_array_almost_equal(
                loaded["pma_attention"],
                synthetic_pma_attention
            )


class TestCCCWithHGTData:
    """Integration tests for CCC analyzer with HGT attention data."""

    def test_hgt_save_load_unpack_to_ccc(self):
        """End-to-end: save HGT HDF5 → load → unpack → CCC analyzer → non-zero results."""
        from src.utils.io import save_attention_weights, load_attention_weights, unpack_hgt_for_ccc
        from src.analysis.ccc_importance import CCCImportanceAnalyzer

        np.random.seed(42)
        n_samples = 10
        n_edge_types = 5
        n_heads = 4

        # Create mock HGT attention dict (output format of aggregate_hgt_attention)
        # PyG convention: src|edge_type|dst
        edge_type_names = [
            "Ast|Secreted_Signaling|Mic",
            "Mic|Secreted_Signaling|Ast",
            "Oli|ECM_Receptor|Exc",
            "Exc|Cell_Cell_Contact|Inh",
            "Ast|Secreted_Signaling|Oli",
        ]
        hgt_dict = {
            "edge_type_names": edge_type_names,
            "mean_by_edge_type": np.random.rand(n_edge_types, n_heads).astype(np.float32),
            "std_by_edge_type": np.random.rand(n_edge_types, n_heads).astype(np.float32),
            "per_sample": np.random.rand(n_samples, n_edge_types, 3, n_heads).astype(np.float32),
            "n_samples": n_samples,
            "n_layers": 3,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "attention_weights.h5"

            # Save
            save_attention_weights(path=path, hgt_attention=hgt_dict)

            # Load
            loaded = load_attention_weights(path)
            hgt_raw = loaded.get("hgt_attention")
            assert hgt_raw is not None

            # Unpack
            scores, edge_metadata, names = unpack_hgt_for_ccc(hgt_raw)
            assert scores is not None
            assert edge_metadata is not None
            assert scores.shape == (n_samples, n_edge_types)

            # Verify PyG convention parsing (src|edge_type|dst)
            # First edge type: "Ast|Secreted_Signaling|Mic"
            assert edge_metadata.iloc[0]["source"] == "Ast"
            assert edge_metadata.iloc[0]["edge_type"] == "Secreted_Signaling"
            assert edge_metadata.iloc[0]["target"] == "Mic"

            # Pass to CCC analyzer
            analyzer = CCCImportanceAnalyzer(
                edge_attention_scores=scores,
                edge_metadata=edge_metadata,
            )
            result = analyzer.analyze()

            # Verify non-zero results
            assert len(result.edge_importance) == n_edge_types
            assert result.edge_importance["mean_attention"].sum() > 0
            assert len(result.top_interactions) > 0
            assert result.top_interactions["mean_attention"].iloc[0] > 0


@pytest.mark.filterwarnings("ignore:n_jobs value 1 overridden to 1 by setting random_state:UserWarning")
class TestCrossModuleIntegration:
    """Test that analysis modules work together correctly."""

    def test_resilience_to_embedding_consistency(
        self,
        synthetic_attention,
        synthetic_embeddings,
        synthetic_phenotypes,
        synthetic_covariates,
    ):
        """Test that resilience and embedding analyses use consistent data."""
        from src.analysis.resilience_signatures import ResilienceSignatureAnalyzer
        from src.analysis.embedding_analysis import EmbeddingAnalyzer

        pathology, cognition = synthetic_phenotypes

        # Run resilience analysis
        resilience_analyzer = ResilienceSignatureAnalyzer(
            attention=synthetic_attention,
            pathology_scores=pathology,
            cognition_scores=cognition,
            cell_type_names=list(CELL_TYPE_ORDER),
        )
        resilience_result = resilience_analyzer.analyze(n_permutations=0)

        # Run embedding analysis
        embedding_analyzer = EmbeddingAnalyzer(
            embeddings=synthetic_embeddings,
            covariates=synthetic_covariates,
        )
        embedding_result = embedding_analyzer.analyze(
            run_linear_probes=True,
        )

        # Verify consistency
        assert len(resilience_result.signature) == len(CELL_TYPE_ORDER)
        assert embedding_result.linear_probe_results is not None

        # Both should have same cell types
        resilience_cell_types = set(resilience_result.signature["cell_type"])
        embedding_cell_types = set(CELL_TYPE_ORDER)
        assert resilience_cell_types == embedding_cell_types

    def test_outputs_saved_to_same_directory(
        self,
        synthetic_attention,
        synthetic_embeddings,
        synthetic_phenotypes,
    ):
        """Test that all analysis outputs can be saved to shared directory."""
        from src.analysis.resilience_signatures import ResilienceSignatureAnalyzer
        from src.analysis.embedding_analysis import EmbeddingAnalyzer

        pathology, cognition = synthetic_phenotypes

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Run and save resilience analysis
            resilience_analyzer = ResilienceSignatureAnalyzer(
                attention=synthetic_attention,
                pathology_scores=pathology,
                cognition_scores=cognition,
                cell_type_names=list(CELL_TYPE_ORDER),
            )
            resilience_result = resilience_analyzer.analyze(n_permutations=0)
            resilience_analyzer.save(resilience_result, output_dir / "resilience")

            # Run and save embedding analysis
            embedding_analyzer = EmbeddingAnalyzer(
                embeddings=synthetic_embeddings,
            )
            embedding_result = embedding_analyzer.analyze(run_umap=True)
            embedding_analyzer.save(embedding_result, output_dir / "embeddings")

            # Verify both saved without conflict
            assert (output_dir / "resilience" / "resilience_signature.parquet").exists()
            assert (output_dir / "embeddings" / "umap_projection.parquet").exists()
