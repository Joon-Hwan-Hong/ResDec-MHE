"""
Embedding extraction and analysis utilities.

Provides tools for analyzing learned embeddings from the cognitive resilience model:
- UMAP projection for visualization
- Clustering (k-means, hierarchical)
- Linear probes to assess embedding quality
- Similarity networks between subjects/cell types
- Outlier detection
- Trajectory analysis (pseudotime, progression)
- Batch effect assessment

Output format: Tidy DataFrames saved as Parquet (primary) and CSV (human-readable).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.neighbors import LocalOutlierFactor
from src.data.constants import EPSILON_DIVISION
from src.utils.io import save_dataframe

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingAnalysisResult:
    """
    Container for embedding analysis results.

    Attributes:
        umap_projection: UMAP 2D/3D coordinates per subject
        cluster_assignments: Cluster labels per subject
        cluster_statistics: Statistics for each cluster
        linear_probe_results: Linear probe prediction performance
        similarity_matrix: Subject-subject similarity matrix
        outlier_scores: Outlier detection scores per subject
        trajectory_scores: Pseudotime/trajectory scores per subject
        batch_effect_metrics: Batch effect assessment results
        metadata: Additional analysis metadata
    """

    umap_projection: pd.DataFrame | None = None
    cluster_assignments: pd.DataFrame | None = None
    cluster_statistics: pd.DataFrame | None = None
    linear_probe_results: pd.DataFrame | None = None
    similarity_matrix: pd.DataFrame | None = None
    outlier_scores: pd.DataFrame | None = None
    trajectory_scores: pd.DataFrame | None = None
    batch_effect_metrics: pd.DataFrame | None = None
    metadata: dict = field(default_factory=dict)


class EmbeddingAnalyzer:
    """
    Analyze learned embeddings from the cognitive resilience model.

    Provides comprehensive analysis tools including dimensionality reduction,
    clustering, quality assessment via linear probes, and batch effect detection.

    Example:
        >>> analyzer = EmbeddingAnalyzer(
        ...     embeddings=subject_embeddings,  # [n_subjects, embed_dim]
        ...     subject_ids=subject_ids,
        ...     covariates=metadata_df,
        ... )
        >>> result = analyzer.analyze()
        >>> analyzer.save(result, output_dir)
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        subject_ids: list[str] | None = None,
        covariates: pd.DataFrame | None = None,
        batch_labels: np.ndarray | None = None,
    ):
        """
        Initialize analyzer with embeddings and metadata.

        Args:
            embeddings: Subject embeddings [n_subjects, embed_dim]
            subject_ids: Subject identifiers
            covariates: DataFrame with covariates (cognition, pathology, etc.)
            batch_labels: Batch labels for batch effect assessment
        """
        self.embeddings = embeddings
        if embeddings.ndim != 2:
            raise ValueError(
                f"embeddings must be 2D [n_subjects, embed_dim], "
                f"got shape {embeddings.shape}"
            )
        self.n_subjects, self.embed_dim = embeddings.shape
        self.subject_ids = subject_ids or [f"subject_{i}" for i in range(self.n_subjects)]
        self.covariates = covariates
        self.batch_labels = batch_labels

        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """Validate input array shapes and consistency."""
        if len(self.subject_ids) != self.n_subjects:
            raise ValueError(
                f"subject_ids has {len(self.subject_ids)} entries "
                f"but embeddings has {self.n_subjects} subjects"
            )

        if self.covariates is not None and len(self.covariates) != self.n_subjects:
            raise ValueError(
                f"covariates has {len(self.covariates)} rows "
                f"but embeddings has {self.n_subjects} subjects"
            )

        if self.batch_labels is not None and len(self.batch_labels) != self.n_subjects:
            raise ValueError(
                f"batch_labels has {len(self.batch_labels)} entries "
                f"but embeddings has {self.n_subjects} subjects"
            )

    def analyze(
        self,
        run_umap: bool = True,
        run_clustering: bool = True,
        run_linear_probes: bool = True,
        run_similarity: bool = True,
        run_outlier_detection: bool = True,
        run_trajectory: bool = True,
        run_batch_effect: bool = True,
        n_clusters: int | None = None,
        umap_n_components: int = 2,
        random_seed: int = 42,
    ) -> EmbeddingAnalysisResult:
        """
        Run embedding analysis.

        Args:
            run_umap: Whether to run UMAP projection
            run_clustering: Whether to run clustering
            run_linear_probes: Whether to run linear probes
            run_similarity: Whether to compute similarity matrix
            run_outlier_detection: Whether to run outlier detection
            run_trajectory: Whether to run trajectory analysis
            run_batch_effect: Whether to run batch effect assessment
            n_clusters: Number of clusters (auto-detected if None)
            umap_n_components: UMAP output dimensions (2 or 3)
            random_seed: Random seed for reproducibility

        Returns:
            EmbeddingAnalysisResult with analysis outputs
        """
        metadata = {
            "n_subjects": self.n_subjects,
            "embed_dim": self.embed_dim,
            "random_seed": random_seed,
        }

        result = EmbeddingAnalysisResult(metadata=metadata)

        # UMAP projection
        if run_umap:
            result.umap_projection = self._run_umap(
                n_components=umap_n_components,
                random_seed=random_seed,
            )
            metadata["umap_n_components"] = umap_n_components

        # Clustering
        if run_clustering:
            result.cluster_assignments, result.cluster_statistics = self._run_clustering(
                n_clusters=n_clusters,
                random_seed=random_seed,
            )
            if n_clusters is not None:
                metadata["n_clusters"] = n_clusters

        # Linear probes
        if run_linear_probes and self.covariates is not None:
            result.linear_probe_results = self._run_linear_probes(random_seed=random_seed)

        # Similarity matrix
        if run_similarity:
            result.similarity_matrix = self._compute_similarity_matrix()

        # Outlier detection
        if run_outlier_detection:
            result.outlier_scores = self._detect_outliers()

        # Trajectory analysis
        if run_trajectory:
            result.trajectory_scores = self._analyze_trajectory()

        # Batch effect assessment
        if run_batch_effect and self.batch_labels is not None:
            result.batch_effect_metrics = self._assess_batch_effect(random_seed=random_seed)

        return result

    def _run_umap(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        random_seed: int = 42,
    ) -> pd.DataFrame | None:
        """
        Run UMAP dimensionality reduction.

        Args:
            n_components: Output dimensions (2 or 3)
            n_neighbors: UMAP n_neighbors parameter
            min_dist: UMAP min_dist parameter
            random_seed: Random seed

        Returns:
            DataFrame with subject_id and UMAP coordinates, or None if UMAP unavailable
        """
        try:
            import umap
        except ImportError:
            logger.warning("UMAP not available. Install with: pip install umap-learn")
            return None

        logger.info(f"Running UMAP projection to {n_components}D...")

        # Adjust n_neighbors if we have few subjects
        actual_n_neighbors = min(n_neighbors, self.n_subjects - 1)

        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=actual_n_neighbors,
            min_dist=min_dist,
            random_state=random_seed,
        )

        coords = reducer.fit_transform(self.embeddings)

        # Build DataFrame
        df_dict = {"subject_id": self.subject_ids}
        for i in range(n_components):
            df_dict[f"umap_{i+1}"] = coords[:, i]

        return pd.DataFrame(df_dict)

    def _run_clustering(
        self,
        n_clusters: int | None = None,
        random_seed: int = 42,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run clustering analysis (k-means and hierarchical).

        Args:
            n_clusters: Number of clusters (auto-detected if None)
            random_seed: Random seed

        Returns:
            Tuple of (cluster_assignments_df, cluster_statistics_df)
        """
        logger.info("Running clustering analysis...")

        # Auto-detect optimal k using silhouette score if not specified
        if n_clusters is None:
            n_clusters = self._find_optimal_clusters(
                max_k=min(10, self.n_subjects // 3),
                random_seed=random_seed,
            )

        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10)
        kmeans_labels = kmeans.fit_predict(self.embeddings)

        # Hierarchical clustering
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
        hier_labels = hierarchical.fit_predict(self.embeddings)

        # Cluster assignments DataFrame
        assignments_df = pd.DataFrame({
            "subject_id": self.subject_ids,
            "cluster": kmeans_labels,  # Default cluster (used by plots)
            "kmeans_cluster": kmeans_labels,
            "hierarchical_cluster": hier_labels,
        })

        # Cluster statistics
        stats_rows = []
        for method, labels in [("kmeans", kmeans_labels), ("hierarchical", hier_labels)]:
            for cluster_id in range(n_clusters):
                mask = labels == cluster_id
                cluster_embeddings = self.embeddings[mask]

                stats_rows.append({
                    "method": method,
                    "cluster_id": cluster_id,
                    "n_subjects": int(mask.sum()),
                    "centroid_norm": float(np.linalg.norm(cluster_embeddings.mean(axis=0))),
                    "intra_cluster_variance": float(cluster_embeddings.var()),
                })

        # Add silhouette scores
        if len(np.unique(kmeans_labels)) > 1:
            kmeans_silhouette = silhouette_score(self.embeddings, kmeans_labels)
            hier_silhouette = silhouette_score(self.embeddings, hier_labels)
        else:
            kmeans_silhouette = hier_silhouette = 0.0

        stats_df = pd.DataFrame(stats_rows)

        # Add overall silhouette to metadata
        logger.info(f"  K-means silhouette: {kmeans_silhouette:.3f}")
        logger.info(f"  Hierarchical silhouette: {hier_silhouette:.3f}")

        return assignments_df, stats_df

    def _find_optimal_clusters(
        self,
        max_k: int = 10,
        random_seed: int = 42,
    ) -> int:
        """Find optimal number of clusters using silhouette score."""
        if max_k < 2:
            return 2

        best_k = 2
        best_score = -1

        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=random_seed, n_init=10)
            labels = kmeans.fit_predict(self.embeddings)
            score = silhouette_score(self.embeddings, labels)
            if score > best_score:
                best_score = score
                best_k = k

        logger.info(f"  Optimal k={best_k} (silhouette={best_score:.3f})")
        return best_k

    def _run_linear_probes(
        self,
        n_folds: int = 5,
        random_seed: int = 42,
    ) -> pd.DataFrame | None:
        """
        Run linear probes to assess embedding quality.

        Tests how well embeddings predict various covariates using
        simple linear models (Ridge for continuous, LogisticRegression for categorical).

        Args:
            n_folds: Number of cross-validation folds
            random_seed: Random seed

        Returns:
            DataFrame with covariate, task_type, metric, score
        """
        logger.info("Running linear probes...")

        if self.covariates is None:
            return None

        rows = []

        for col in self.covariates.columns:
            y = self.covariates[col].values

            # Skip columns with too many missing values
            valid_mask = ~pd.isna(y)
            if valid_mask.sum() < n_folds * 2:
                continue

            X = self.embeddings[valid_mask]
            y_valid = y[valid_mask]

            # Determine task type
            if pd.api.types.is_numeric_dtype(self.covariates[col]):
                # Regression task
                task_type = "regression"
                model = Ridge(alpha=1.0)
                cv = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
                scores = cross_val_score(model, X, y_valid, cv=cv, scoring="r2")

                rows.append({
                    "target": col,
                    "task_type": task_type,
                    "metric": "r2",
                    "score_mean": float(scores.mean()),
                    "score_std": float(scores.std()),
                    "n_samples": int(valid_mask.sum()),
                })
            else:
                # Classification task
                task_type = "classification"
                n_classes = len(np.unique(y_valid))

                if n_classes < 2 or n_classes > self.n_subjects // 2:
                    continue

                model = LogisticRegression(max_iter=1000, random_state=random_seed)
                cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

                try:
                    scores = cross_val_score(model, X, y_valid, cv=cv, scoring="accuracy")
                    rows.append({
                        "target": col,
                        "task_type": task_type,
                        "metric": "accuracy",
                        "score_mean": float(scores.mean()),
                        "score_std": float(scores.std()),
                        "n_samples": int(valid_mask.sum()),
                    })
                except ValueError:
                    # Skip if not enough samples per class
                    continue

        return pd.DataFrame(rows) if rows else None

    def _compute_similarity_matrix(
        self,
        metric: str = "cosine",
    ) -> pd.DataFrame:
        """
        Compute subject-subject similarity matrix.

        Args:
            metric: Distance metric (cosine, euclidean, correlation)

        Returns:
            DataFrame with long-format similarity (subject_1, subject_2, similarity)
        """
        logger.info("Computing similarity matrix...")

        # Compute pairwise distances
        if metric == "cosine":
            # Cosine similarity = 1 - cosine distance
            distances = pdist(self.embeddings, metric="cosine")
            similarities = 1 - squareform(distances)
        elif metric == "correlation":
            distances = pdist(self.embeddings, metric="correlation")
            similarities = 1 - squareform(distances)
        else:  # euclidean
            distances = pdist(self.embeddings, metric="euclidean")
            # Convert distance to similarity
            max_dist = distances.max() if distances.max() > 0 else 1
            similarities = 1 - squareform(distances) / max_dist

        # Vectorized upper-triangle extraction
        triu_idx = np.triu_indices(self.n_subjects, k=1)
        return pd.DataFrame({
            "subject_1": np.array(self.subject_ids)[triu_idx[0]],
            "subject_2": np.array(self.subject_ids)[triu_idx[1]],
            "similarity": similarities[triu_idx],
        })

    def _detect_outliers(
        self,
        n_neighbors: int = 20,
        contamination: float = 0.1,
    ) -> pd.DataFrame:
        """
        Detect outliers using Local Outlier Factor.

        Args:
            n_neighbors: LOF n_neighbors parameter
            contamination: Expected proportion of outliers

        Returns:
            DataFrame with subject_id, outlier_score, is_outlier
        """
        logger.info("Running outlier detection...")

        # Adjust n_neighbors for small datasets
        actual_n_neighbors = min(n_neighbors, self.n_subjects - 1)

        lof = LocalOutlierFactor(
            n_neighbors=actual_n_neighbors,
            contamination=contamination,
        )

        # LOF returns -1 for outliers, 1 for inliers
        predictions = lof.fit_predict(self.embeddings)
        scores = -lof.negative_outlier_factor_  # Higher = more outlier-like

        return pd.DataFrame({
            "subject_id": self.subject_ids,
            "outlier_score": scores,
            "is_outlier": predictions == -1,
        })

    def _analyze_trajectory(self) -> pd.DataFrame:
        """
        Analyze embedding trajectory/progression.

        Uses first principal component as a proxy for biological progression,
        which often captures disease severity or aging trajectories.

        Returns:
            DataFrame with subject_id, pseudotime, pc1, pc2
        """
        logger.info("Analyzing embedding trajectory...")

        # PCA for trajectory
        pca = PCA(n_components=min(5, self.embed_dim))
        pca_coords = pca.fit_transform(self.embeddings)

        # Use PC1 as pseudotime (normalized to [0, 1])
        pc1 = pca_coords[:, 0]
        pseudotime = (pc1 - pc1.min()) / (pc1.max() - pc1.min() + EPSILON_DIVISION)

        df = pd.DataFrame({
            "subject_id": self.subject_ids,
            "pseudotime": pseudotime,
            "pc1": pca_coords[:, 0],
            "pc2": pca_coords[:, 1] if pca_coords.shape[1] > 1 else 0,
        })

        # Add variance explained
        logger.info(f"  PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}% variance")

        return df

    def _assess_batch_effect(
        self,
        random_seed: int = 42,
    ) -> pd.DataFrame | None:
        """
        Assess batch effects in embeddings.

        Uses multiple metrics:
        - Silhouette score (higher = more batch separation = worse)
        - Logistic regression accuracy for batch prediction
        - kBET-like mixing metric

        Args:
            random_seed: Random seed

        Returns:
            DataFrame with metric, value, interpretation
        """
        logger.info("Assessing batch effects...")

        if self.batch_labels is None:
            return None

        unique_batches = np.unique(self.batch_labels)
        n_batches = len(unique_batches)

        if n_batches < 2:
            logger.warning("Need at least 2 batches for batch effect assessment")
            return None

        rows = []

        # 1. Silhouette score (higher = more batch-specific clustering = worse)
        if n_batches < self.n_subjects:
            batch_silhouette = silhouette_score(self.embeddings, self.batch_labels)
            rows.append({
                "metric": "batch_silhouette",
                "value": float(batch_silhouette),
                "interpretation": "low" if batch_silhouette < 0.2 else "medium" if batch_silhouette < 0.5 else "high",
            })

        # 2. Batch prediction accuracy (lower = better mixing)
        if n_batches >= 2:
            model = LogisticRegression(max_iter=1000, random_state=random_seed)
            # Convert batch labels to numeric for bincount
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            batch_numeric = le.fit_transform(self.batch_labels)
            n_folds = min(5, min(np.bincount(batch_numeric)))
            if n_folds >= 2:
                cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
                try:
                    scores = cross_val_score(
                        model, self.embeddings, self.batch_labels, cv=cv, scoring="accuracy"
                    )
                    batch_pred_acc = scores.mean()
                    # Baseline is 1/n_batches (random guessing)
                    baseline = 1.0 / n_batches

                    rows.append({
                        "metric": "batch_prediction_accuracy",
                        "value": float(batch_pred_acc),
                        "interpretation": "low" if batch_pred_acc < baseline * 1.5 else "medium" if batch_pred_acc < baseline * 2 else "high",
                    })
                    rows.append({
                        "metric": "batch_prediction_baseline",
                        "value": float(baseline),
                        "interpretation": "reference",
                    })
                except ValueError as e:
                    logger.debug("Batch prediction accuracy skipped: %s", e)

        # 3. Local mixing score (inspired by kBET)
        # Measures if local neighborhoods have batch proportions similar to global
        local_mixing = self._compute_local_mixing_score()
        rows.append({
            "metric": "local_mixing_score",
            "value": float(local_mixing),
            "interpretation": "good" if local_mixing > 0.8 else "moderate" if local_mixing > 0.5 else "poor",
        })

        return pd.DataFrame(rows)

    def _compute_local_mixing_score(
        self,
        n_neighbors: int = 15,
    ) -> float:
        """
        Compute local mixing score (kBET-inspired).

        Measures whether local neighborhoods have batch compositions
        similar to the global batch composition.

        Note: kneighbors() with the training set as query includes the
        query point itself (distance=0). This is standard in Python kBET
        implementations and introduces a small bias of ~1/k toward the
        query's own batch. For qualitative assessment (good/moderate/poor
        categories) this does not affect interpretation.

        Returns:
            Score between 0 (no mixing) and 1 (perfect mixing)
        """
        from sklearn.neighbors import NearestNeighbors

        # Adjust n_neighbors for small datasets
        k = min(n_neighbors, self.n_subjects - 1)

        # Find nearest neighbors
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(self.embeddings)
        _, indices = nn.kneighbors(self.embeddings)

        # Global batch proportions
        unique_batches = np.unique(self.batch_labels)
        global_props = {b: (self.batch_labels == b).mean() for b in unique_batches}

        # Check local neighborhoods
        chi_sq_values = []
        for i in range(self.n_subjects):
            neighbor_batches = self.batch_labels[indices[i]]
            local_props = {b: (neighbor_batches == b).mean() for b in unique_batches}

            # Chi-squared statistic comparing local to global
            chi_sq = sum(
                (local_props.get(b, 0) - global_props[b])**2 / (global_props[b] + EPSILON_DIVISION)
                for b in unique_batches
            )
            chi_sq_values.append(chi_sq)

        # Convert to score (lower chi-sq = better mixing = higher score)
        mean_chi_sq = np.mean(chi_sq_values)
        # Normalize: score of 1 when chi_sq=0, approaches 0 for large chi_sq
        mixing_score = np.exp(-mean_chi_sq)

        return mixing_score

    def save(
        self,
        result: EmbeddingAnalysisResult,
        output_dir: str | Path,
        formats: list[Literal["parquet", "csv"]] | None = None,
    ) -> dict[str, Path]:
        """
        Save analysis results to files.

        Args:
            result: EmbeddingAnalysisResult to save
            output_dir: Directory for output files
            formats: Output formats (default: ["parquet", "csv"])

        Returns:
            Dict mapping output name to file path
        """
        if formats is None:
            formats = ["parquet", "csv"]

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        # Save each result component
        components = [
            ("umap_projection", result.umap_projection),
            ("cluster_assignments", result.cluster_assignments),
            ("cluster_statistics", result.cluster_statistics),
            ("linear_probe_results", result.linear_probe_results),
            ("similarity_matrix", result.similarity_matrix),
            ("outlier_scores", result.outlier_scores),
            ("trajectory_scores", result.trajectory_scores),
            ("batch_effect_metrics", result.batch_effect_metrics),
        ]

        for name, df in components:
            if df is not None:
                for fmt in formats:
                    path = output_dir / f"{name}.{fmt}"
                    save_dataframe(df, path, fmt)
                    saved_files[f"{name}_{fmt}"] = path

        logger.info(f"Saved embedding analysis to {output_dir}")
        return saved_files


def analyze_embeddings(
    embeddings: np.ndarray,
    subject_ids: list[str] | None = None,
    covariates: pd.DataFrame | None = None,
    batch_labels: np.ndarray | None = None,
    output_dir: str | Path | None = None,
    **kwargs,
) -> EmbeddingAnalysisResult:
    """
    Convenience function to analyze embeddings.

    Args:
        embeddings: Subject embeddings [n_subjects, embed_dim]
        subject_ids: Subject identifiers
        covariates: DataFrame with covariates
        batch_labels: Batch labels for batch effect assessment
        output_dir: If provided, save results to this directory
        **kwargs: Additional arguments passed to analyze()

    Returns:
        EmbeddingAnalysisResult with analysis results
    """
    analyzer = EmbeddingAnalyzer(
        embeddings=embeddings,
        subject_ids=subject_ids,
        covariates=covariates,
        batch_labels=batch_labels,
    )

    result = analyzer.analyze(**kwargs)

    if output_dir is not None:
        analyzer.save(result, output_dir)

    return result
