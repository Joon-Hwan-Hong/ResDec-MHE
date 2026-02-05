"""
Resilience signature extraction from attention patterns.

Primary Method: Attention Difference
1. Subset to high pathology subjects (top tertile gpath)
2. Split into resilient (top tertile cognition) vs vulnerable (bottom tertile)
3. Resilience signature = mean(attention_resilient) - mean(attention_vulnerable)
4. Statistical significance via permutation test

Secondary Method: Ablation Study (Optional)
- Zero out attention for specific cell types
- Measure prediction change

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

from src.data.constants import CELL_TYPE_ORDER, N_CELL_TYPES
from src.utils.io import save_dataframe
from src.utils.statistics import benjamini_hochberg, cohens_d_with_ci, attention_entropy

logger = logging.getLogger(__name__)


@dataclass
class ResilienceSignatureResult:
    """
    Container for resilience signature analysis results.

    Attributes:
        signature: DataFrame with resilience signature per cell type
        permutation_pvalues: P-values from permutation test
        permutation_null: Full null distribution [n_cell_types, n_permutations]
        group_statistics: Statistics for resilient vs vulnerable groups
        by_region: DataFrame with resilience signatures per brain region
        ablation_results: Optional ablation study results
        ablation_comparison: Optional comparison of ablation methods
        ablation_by_region: Optional regional ablation results
        metadata: Additional analysis metadata
    """

    signature: pd.DataFrame
    permutation_pvalues: pd.DataFrame | None = None
    permutation_null: np.ndarray | None = None
    group_statistics: pd.DataFrame | None = None
    by_region: pd.DataFrame | None = None
    ablation_results: pd.DataFrame | None = None
    ablation_comparison: pd.DataFrame | None = None
    ablation_by_region: pd.DataFrame | None = None
    metadata: dict = field(default_factory=dict)


class ResilienceSignatureAnalyzer:
    """
    Extract resilience signatures from attention patterns.

    Identifies cell types differentially important between cognitively
    resilient and vulnerable subjects with high pathology burden.

    Example:
        >>> analyzer = ResilienceSignatureAnalyzer(
        ...     attention=pathology_attention,     # [n_subjects, n_heads, n_cell_types]
        ...     pathology_scores=gpath,            # [n_subjects]
        ...     cognition_scores=cogn_global,      # [n_subjects]
        ... )
        >>> result = analyzer.analyze(n_permutations=1000)
        >>> analyzer.save(result, output_dir)
    """

    def __init__(
        self,
        attention: np.ndarray,
        pathology_scores: np.ndarray,
        cognition_scores: np.ndarray,
        subject_ids: list[str] | None = None,
        cell_type_names: list[str] | None = None,
        region_labels: np.ndarray | None = None,
        pathology_threshold_percentile: float = 66.7,
    ):
        """
        Initialize analyzer with attention and clinical data.

        Args:
            attention: Pathology attention weights [n_subjects, n_heads, n_cell_types]
            pathology_scores: Pathology burden scores [n_subjects] (higher = more pathology)
            cognition_scores: Cognitive scores [n_subjects] (higher = better cognition)
            subject_ids: Subject identifiers
            cell_type_names: Cell type names (defaults to CELL_TYPE_ORDER)
            region_labels: Brain region labels for regional analysis [n_subjects]
            pathology_threshold_percentile: Percentile for high pathology (default: top 1/3)
        """
        self.attention = attention
        self.pathology_scores = pathology_scores
        self.cognition_scores = cognition_scores
        self.subject_ids = subject_ids
        self.cell_type_names = cell_type_names or list(CELL_TYPE_ORDER)
        self.region_labels = region_labels
        self.pathology_threshold_percentile = pathology_threshold_percentile

        self._validate_inputs()

        # Pre-compute subject groups
        self._identify_groups()

    def _validate_inputs(self) -> None:
        """Validate input array shapes and consistency."""
        if self.attention.ndim != 3:
            raise ValueError(
                f"attention must be 3D [n_subjects, n_heads, n_cell_types], "
                f"got shape {self.attention.shape}"
            )

        n_subjects = self.attention.shape[0]

        if len(self.pathology_scores) != n_subjects:
            raise ValueError(
                f"pathology_scores has {len(self.pathology_scores)} entries "
                f"but attention has {n_subjects} subjects"
            )

        if len(self.cognition_scores) != n_subjects:
            raise ValueError(
                f"cognition_scores has {len(self.cognition_scores)} entries "
                f"but attention has {n_subjects} subjects"
            )

        n_cell_types = self.attention.shape[2]
        if len(self.cell_type_names) != n_cell_types:
            raise ValueError(
                f"cell_type_names has {len(self.cell_type_names)} entries "
                f"but attention has {n_cell_types} cell types"
            )

        if self.region_labels is not None and len(self.region_labels) != n_subjects:
            raise ValueError(
                f"region_labels has {len(self.region_labels)} entries "
                f"but attention has {n_subjects} subjects"
            )

        # Warn about NaN values (will be excluded from group assignment)
        n_nan_path = int(np.sum(np.isnan(self.pathology_scores)))
        n_nan_cog = int(np.sum(np.isnan(self.cognition_scores)))
        if n_nan_path > 0 or n_nan_cog > 0:
            logger.warning(
                f"Input arrays contain NaN values: {n_nan_path} pathology, "
                f"{n_nan_cog} cognition. NaN subjects will be excluded from "
                f"group assignment."
            )

    def _identify_groups(self) -> None:
        """Identify resilient and vulnerable subject groups."""
        n_subjects = len(self.pathology_scores)

        # High pathology threshold (top tertile)
        pathology_threshold = np.nanpercentile(
            self.pathology_scores, self.pathology_threshold_percentile
        )
        self.high_pathology_mask = self.pathology_scores >= pathology_threshold

        # Pathology tertile masks (low/medium/high)
        path_33 = np.nanpercentile(self.pathology_scores, 33.3)
        path_67 = np.nanpercentile(self.pathology_scores, 66.7)
        self.pathology_low_mask = self.pathology_scores < path_33
        self.pathology_med_mask = (self.pathology_scores >= path_33) & (self.pathology_scores < path_67)
        self.pathology_high_mask = self.pathology_scores >= path_67

        # Within high pathology subjects, split by cognition
        high_path_indices = np.where(self.high_pathology_mask)[0]
        high_path_cognition = self.cognition_scores[high_path_indices]

        # Resilient = top tertile cognition among high pathology
        # Vulnerable = bottom tertile cognition among high pathology
        cog_33 = np.nanpercentile(high_path_cognition, 33.3)
        cog_67 = np.nanpercentile(high_path_cognition, 66.7)

        self.resilient_mask = np.zeros(n_subjects, dtype=bool)
        self.vulnerable_mask = np.zeros(n_subjects, dtype=bool)

        for idx in high_path_indices:
            cog = self.cognition_scores[idx]
            if cog >= cog_67:
                self.resilient_mask[idx] = True
            elif cog <= cog_33:
                self.vulnerable_mask[idx] = True

        self.n_resilient = self.resilient_mask.sum()
        self.n_vulnerable = self.vulnerable_mask.sum()

        logger.info(
            f"Identified {self.n_resilient} resilient and {self.n_vulnerable} vulnerable "
            f"subjects from {self.high_pathology_mask.sum()} high-pathology subjects"
        )
        logger.info(
            f"Pathology tertiles: low={self.pathology_low_mask.sum()}, "
            f"medium={self.pathology_med_mask.sum()}, high={self.pathology_high_mask.sum()}"
        )

    def analyze(
        self,
        n_permutations: int = 1000,
        random_seed: int | None = 42,
        run_ablation: bool = False,
        ablation_method: Literal["both", "zero_embedding", "node_removal"] = "both",
        embeddings: np.ndarray | None = None,
        apply_fdr_correction: bool = True,
    ) -> ResilienceSignatureResult:
        """
        Run resilience signature analysis.

        Args:
            n_permutations: Number of permutations for significance testing
            random_seed: Random seed for permutation test
            run_ablation: Whether to run ablation study
            ablation_method: Which ablation method(s) to run
            embeddings: Optional embeddings for ablation [n_subjects, n_cell_types, embed_dim]
            apply_fdr_correction: Whether to apply Benjamini-Hochberg FDR correction (default: True)

        Returns:
            ResilienceSignatureResult with signature and statistics
        """
        # Compute signature (attention difference)
        signature_df = self._compute_signature()

        # Permutation test for significance
        pvalues_df = None
        permutation_null = None
        if n_permutations > 0 and self.n_resilient > 0 and self.n_vulnerable > 0:
            pvalues_df, permutation_null = self._permutation_test(
                n_permutations, random_seed, apply_fdr_correction
            )

        # Group statistics
        group_stats_df = self._compute_group_statistics()

        # Regional analysis (if region labels provided)
        by_region_df = self._compute_signature_by_region()

        # Ablation study (if requested)
        ablation_df = None
        ablation_comparison_df = None
        ablation_by_region_df = None
        if run_ablation:
            ablation_df, ablation_comparison_df = self.run_ablation_study(
                method=ablation_method,
                embeddings=embeddings,
            )
            # Regional ablation (if region labels provided)
            if self.region_labels is not None:
                ablation_by_region_df = self._run_regional_ablation(
                    method=ablation_method,
                    embeddings=embeddings,
                )

        metadata = {
            "n_subjects": len(self.pathology_scores),
            "n_high_pathology": int(self.high_pathology_mask.sum()),
            "n_resilient": int(self.n_resilient),
            "n_vulnerable": int(self.n_vulnerable),
            "pathology_threshold_percentile": self.pathology_threshold_percentile,
            "n_permutations": n_permutations,
        }

        # Add regional metadata if available
        if by_region_df is not None:
            unique_regions = by_region_df["region"].unique()
            metadata["n_regions_analyzed"] = len(unique_regions)
            metadata["regions"] = list(unique_regions)

        # Add ablation metadata if available
        if ablation_df is not None:
            metadata["ablation_method"] = ablation_method
            if ablation_comparison_df is not None:
                corr = ablation_comparison_df["methods_correlation"].iloc[0]
                metadata["ablation_methods_correlation"] = float(corr)

        return ResilienceSignatureResult(
            signature=signature_df,
            permutation_pvalues=pvalues_df,
            permutation_null=permutation_null,
            group_statistics=group_stats_df,
            by_region=by_region_df,
            ablation_results=ablation_df,
            ablation_comparison=ablation_comparison_df,
            ablation_by_region=ablation_by_region_df,
            metadata=metadata,
        )

    def _compute_signature(self) -> pd.DataFrame:
        """
        Compute resilience signature as attention difference.

        Returns:
            DataFrame with columns: cell_type, signature, resilient_mean, vulnerable_mean,
                                   cohens_d, ci_lower, ci_upper, rank
        """
        # Average attention across heads: [n_subjects, n_cell_types]
        attention_per_subject = self.attention.mean(axis=1)

        # Get group data
        resilient_data = attention_per_subject[self.resilient_mask] if self.n_resilient > 0 else None
        vulnerable_data = attention_per_subject[self.vulnerable_mask] if self.n_vulnerable > 0 else None

        # Mean attention per group
        if resilient_data is not None:
            resilient_mean = resilient_data.mean(axis=0)
            resilient_std = resilient_data.std(axis=0, ddof=1)
        else:
            resilient_mean = np.zeros(len(self.cell_type_names))
            resilient_std = np.zeros(len(self.cell_type_names))

        if vulnerable_data is not None:
            vulnerable_mean = vulnerable_data.mean(axis=0)
            vulnerable_std = vulnerable_data.std(axis=0, ddof=1)
        else:
            vulnerable_mean = np.zeros(len(self.cell_type_names))
            vulnerable_std = np.zeros(len(self.cell_type_names))

        # Signature = resilient - vulnerable
        # Positive = more important for resilience
        signature = resilient_mean - vulnerable_mean

        # Cohen's d: (mean1 - mean2) / pooled_std
        # Pooled std: sqrt(((n1-1)*s1^2 + (n2-1)*s2^2) / (n1+n2-2))
        # Note: This is a vectorized implementation of the formula in
        # src.utils.statistics.cohens_d_with_ci for performance across cell types.
        cohens_d = np.zeros(len(self.cell_type_names))
        ci_lower = np.zeros(len(self.cell_type_names))
        ci_upper = np.zeros(len(self.cell_type_names))
        cohens_d_ci_lower = np.zeros(len(self.cell_type_names))
        cohens_d_ci_upper = np.zeros(len(self.cell_type_names))

        if self.n_resilient > 1 and self.n_vulnerable > 1:
            n1, n2 = self.n_resilient, self.n_vulnerable
            pooled_var = ((n1 - 1) * resilient_std**2 + (n2 - 1) * vulnerable_std**2) / (n1 + n2 - 2)
            pooled_std = np.sqrt(pooled_var)

            # Avoid division by zero
            nonzero_mask = pooled_std > 1e-10
            cohens_d[nonzero_mask] = signature[nonzero_mask] / pooled_std[nonzero_mask]

            # 95% CI for mean difference using pooled SE
            # SE = pooled_std * sqrt(1/n1 + 1/n2)
            se = pooled_std * np.sqrt(1/n1 + 1/n2)
            # t critical value for 95% CI with df = n1 + n2 - 2
            df = n1 + n2 - 2
            t_crit = stats.t.ppf(0.975, df)
            ci_lower = signature - t_crit * se
            ci_upper = signature + t_crit * se

            # 95% CI for Cohen's d using Hedges & Olkin (1985) approximation
            # SE(d) ≈ sqrt((n1+n2)/(n1*n2) + d^2/(2*(n1+n2)))
            se_d = np.sqrt((n1 + n2) / (n1 * n2) + cohens_d**2 / (2 * (n1 + n2)))
            cohens_d_ci_lower = cohens_d - t_crit * se_d
            cohens_d_ci_upper = cohens_d + t_crit * se_d

        df = pd.DataFrame({
            "cell_type": self.cell_type_names,
            "signature": signature,
            "resilient_mean": resilient_mean,
            "vulnerable_mean": vulnerable_mean,
            "cohens_d": cohens_d,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "cohens_d_ci_lower": cohens_d_ci_lower,
            "cohens_d_ci_upper": cohens_d_ci_upper,
        })

        # Sort by absolute signature (most differential first)
        df["abs_signature"] = np.abs(df["signature"])
        df = df.sort_values("abs_signature", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)

        return df[["cell_type", "signature", "resilient_mean", "vulnerable_mean",
                   "cohens_d", "ci_lower", "ci_upper", "cohens_d_ci_lower",
                   "cohens_d_ci_upper", "rank"]]

    def _permutation_test(
        self,
        n_permutations: int,
        random_seed: int | None,
        apply_fdr_correction: bool = True,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Permutation test for signature significance.

        Shuffles resilient/vulnerable labels and recomputes signature.

        Args:
            n_permutations: Number of permutations
            random_seed: Random seed
            apply_fdr_correction: Whether to apply Benjamini-Hochberg FDR correction

        Returns:
            Tuple of (DataFrame with p-values, null distribution array [n_permutations, n_cell_types])
        """
        rng = np.random.default_rng(random_seed)

        # Original signature
        attention_per_subject = self.attention.mean(axis=1)
        original_signature = (
            attention_per_subject[self.resilient_mask].mean(axis=0) -
            attention_per_subject[self.vulnerable_mask].mean(axis=0)
        )

        # Get indices for permutation (only high pathology subjects)
        high_path_indices = np.where(self.high_pathology_mask)[0]
        n_high_path = len(high_path_indices)

        # Permutation distribution
        perm_signatures = np.zeros((n_permutations, len(self.cell_type_names)))

        for i in range(n_permutations):
            # Shuffle labels within high pathology group
            shuffled_indices = rng.permutation(high_path_indices)

            # Re-assign to resilient/vulnerable based on original group sizes
            perm_resilient = shuffled_indices[:self.n_resilient]
            perm_vulnerable = shuffled_indices[-self.n_vulnerable:]

            perm_resilient_mean = attention_per_subject[perm_resilient].mean(axis=0)
            perm_vulnerable_mean = attention_per_subject[perm_vulnerable].mean(axis=0)

            perm_signatures[i] = perm_resilient_mean - perm_vulnerable_mean

        # Two-tailed p-value
        p_values = np.zeros(len(self.cell_type_names))
        for ct_idx in range(len(self.cell_type_names)):
            obs = np.abs(original_signature[ct_idx])
            null_dist = np.abs(perm_signatures[:, ct_idx])
            p_values[ct_idx] = (null_dist >= obs).mean()

        # FDR correction (Benjamini-Hochberg) - optional
        if apply_fdr_correction:
            fdr_corrected, _ = benjamini_hochberg(p_values)
            df = pd.DataFrame({
                "cell_type": self.cell_type_names,
                "p_value": p_values,
                "fdr_corrected": fdr_corrected,
                "significant_005": fdr_corrected < 0.05,
                "significant_001": fdr_corrected < 0.01,
            })
        else:
            # Use uncorrected p-values
            df = pd.DataFrame({
                "cell_type": self.cell_type_names,
                "p_value": p_values,
                "fdr_corrected": p_values,  # Same as uncorrected for compatibility
                "significant_005": p_values < 0.05,
                "significant_001": p_values < 0.01,
            })
            logger.warning("FDR correction disabled - reporting uncorrected p-values")

        return df, perm_signatures

    def _compute_signature_by_region(self) -> pd.DataFrame | None:
        """
        Compute resilience signatures separately for each brain region.

        Uses the same tertile-based approach as global signature, but
        stratified by region. Only regions with sufficient subjects
        (>= 3 per group) are included.

        Returns:
            DataFrame with columns: region, cell_type, signature, resilient_mean,
                                   vulnerable_mean, cohens_d, ci_lower, ci_upper,
                                   n_resilient, n_vulnerable
            Returns None if region_labels not provided.
        """
        if self.region_labels is None:
            return None

        # Average attention across heads: [n_subjects, n_cell_types]
        attention_per_subject = self.attention.mean(axis=1)

        unique_regions = np.unique(self.region_labels)
        rows = []

        for region in unique_regions:
            # Get indices for this region
            region_mask = self.region_labels == region

            # Filter to high pathology subjects in this region
            region_high_path = region_mask & self.high_pathology_mask
            n_region_high_path = region_high_path.sum()

            if n_region_high_path < 6:  # Need at least some subjects per group
                logger.warning(
                    f"Region '{region}' has only {n_region_high_path} high-pathology "
                    f"subjects, skipping regional analysis"
                )
                continue

            # Get cognition scores for high-pathology subjects in this region
            region_indices = np.where(region_high_path)[0]
            region_cognition = self.cognition_scores[region_indices]

            # Split into resilient/vulnerable by cognition tertiles within region
            cog_33 = np.nanpercentile(region_cognition, 33.3)
            cog_67 = np.nanpercentile(region_cognition, 66.7)

            region_resilient_mask = np.zeros(len(self.pathology_scores), dtype=bool)
            region_vulnerable_mask = np.zeros(len(self.pathology_scores), dtype=bool)

            for idx in region_indices:
                cog = self.cognition_scores[idx]
                if cog >= cog_67:
                    region_resilient_mask[idx] = True
                elif cog <= cog_33:
                    region_vulnerable_mask[idx] = True

            n_resilient = region_resilient_mask.sum()
            n_vulnerable = region_vulnerable_mask.sum()

            if n_resilient < 2 or n_vulnerable < 2:
                logger.warning(
                    f"Region '{region}' has insufficient group sizes "
                    f"(n_resilient={n_resilient}, n_vulnerable={n_vulnerable}), skipping"
                )
                continue

            # Compute signature for this region
            resilient_data = attention_per_subject[region_resilient_mask]
            vulnerable_data = attention_per_subject[region_vulnerable_mask]

            resilient_mean = resilient_data.mean(axis=0)
            resilient_std = resilient_data.std(axis=0, ddof=1)
            vulnerable_mean = vulnerable_data.mean(axis=0)
            vulnerable_std = vulnerable_data.std(axis=0, ddof=1)

            signature = resilient_mean - vulnerable_mean

            # Cohen's d and 95% CI
            n1, n2 = n_resilient, n_vulnerable
            pooled_var = ((n1 - 1) * resilient_std**2 + (n2 - 1) * vulnerable_std**2) / (n1 + n2 - 2)
            pooled_std = np.sqrt(pooled_var)

            cohens_d = np.zeros(len(self.cell_type_names))
            nonzero_mask = pooled_std > 1e-10
            cohens_d[nonzero_mask] = signature[nonzero_mask] / pooled_std[nonzero_mask]

            se = pooled_std * np.sqrt(1/n1 + 1/n2)
            df = n1 + n2 - 2
            t_crit = stats.t.ppf(0.975, df)
            ci_lower = signature - t_crit * se
            ci_upper = signature + t_crit * se

            # Cohen's d CI using Hedges & Olkin approximation
            se_d = np.sqrt((n1 + n2) / (n1 * n2) + cohens_d**2 / (2 * (n1 + n2)))
            cohens_d_ci_lower = cohens_d - t_crit * se_d
            cohens_d_ci_upper = cohens_d + t_crit * se_d

            # Add rows for each cell type in this region
            for ct_idx, ct_name in enumerate(self.cell_type_names):
                rows.append({
                    "region": region,
                    "cell_type": ct_name,
                    "signature": signature[ct_idx],
                    "resilient_mean": resilient_mean[ct_idx],
                    "vulnerable_mean": vulnerable_mean[ct_idx],
                    "cohens_d": cohens_d[ct_idx],
                    "ci_lower": ci_lower[ct_idx],
                    "ci_upper": ci_upper[ct_idx],
                    "cohens_d_ci_lower": cohens_d_ci_lower[ct_idx],
                    "cohens_d_ci_upper": cohens_d_ci_upper[ct_idx],
                    "n_resilient": n_resilient,
                    "n_vulnerable": n_vulnerable,
                })

        if not rows:
            logger.warning("No regions had sufficient subjects for regional analysis")
            return None

        df = pd.DataFrame(rows)

        # Sort by region, then by absolute signature within region
        df["abs_signature"] = np.abs(df["signature"])
        df = df.sort_values(
            ["region", "abs_signature"],
            ascending=[True, False]
        ).reset_index(drop=True)

        return df[["region", "cell_type", "signature", "resilient_mean", "vulnerable_mean",
                   "cohens_d", "ci_lower", "ci_upper", "cohens_d_ci_lower", "cohens_d_ci_upper",
                   "n_resilient", "n_vulnerable"]]

    def run_ablation_study(
        self,
        method: Literal["both", "zero_embedding", "node_removal"] = "both",
        embeddings: np.ndarray | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """
        Run ablation study to measure importance of each cell type.

        NOTE: This is an attention-based ablation proxy, not true causal ablation.
        True causal ablation would require running model forward passes with masked
        inputs. This implementation analyzes how attention patterns change when
        cell types are masked, providing a computationally efficient approximation.

        Implements two ablation methods:
        - Option A (Zero Embedding): Zero out attention for each cell type, renormalize,
          measure deviation in weighted representation. Captures "transcriptional"
          importance - how much this cell type contributes to the weighted embedding.
        - Option B (Node Removal): Remove cell type entirely from consideration,
          measure impact on attention distribution via attention magnitude and
          entropy change. Captures "structural" importance.

        Interpretation:
        - "transcriptional" (higher in Option A): Cell type's expression profile matters
        - "structural" (higher in Option B): Cell type's presence in the network matters
        - "consistent": Both methods agree on importance

        Args:
            method: Which ablation method(s) to run
            embeddings: Optional cell type embeddings [n_subjects, n_cell_types, embed_dim].
                       If provided, Option A computes actual embedding deviation.
                       If None, uses attention-based proxy.

        Returns:
            Tuple of (ablation_results_df, comparison_df or None)
        """
        logger.info(f"Running ablation study with method={method}...")

        results = {}

        if method in ("both", "zero_embedding"):
            results["zero_embedding"] = self._ablation_zero_embedding(embeddings)

        if method in ("both", "node_removal"):
            results["node_removal"] = self._ablation_node_removal()

        # Combine results into single DataFrame
        rows = []
        for ablation_method, method_df in results.items():
            for _, row in method_df.iterrows():
                rows.append({
                    "cell_type": row["cell_type"],
                    "method": ablation_method,
                    "importance": row["importance"],
                    "importance_std": row["importance_std"],
                    "importance_high_pathology": row.get("importance_high_pathology", np.nan),
                    "importance_low_pathology": row.get("importance_low_pathology", np.nan),
                    "importance_low_tertile": row.get("importance_low_tertile", np.nan),
                    "importance_med_tertile": row.get("importance_med_tertile", np.nan),
                    "importance_high_tertile": row.get("importance_high_tertile", np.nan),
                    "rank": row["rank"],
                })

        ablation_df = pd.DataFrame(rows)

        # Generate comparison if both methods were run
        comparison_df = None
        if method == "both" and len(results) == 2:
            comparison_df = self._compare_ablation_methods(
                results["zero_embedding"],
                results["node_removal"],
            )

        return ablation_df, comparison_df

    def _ablation_zero_embedding(
        self,
        embeddings: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """
        Option A: Zero embedding ablation.

        For each cell type:
        1. Zero out attention for that cell type
        2. Renormalize remaining attention (sum to 1)
        3. Compute deviation from original weighted representation

        Args:
            embeddings: Optional [n_subjects, n_cell_types, embed_dim]

        Returns:
            DataFrame with cell_type, importance, importance_std, rank
        """
        # Average attention across heads: [n_subjects, n_cell_types]
        attention = self.attention.mean(axis=1)
        n_subjects, n_cell_types = attention.shape

        # Compute importance for each cell type
        importances = []

        for ct_idx in range(n_cell_types):
            # Create masked attention (zero out this cell type)
            masked_attention = attention.copy()
            masked_attention[:, ct_idx] = 0

            # Renormalize (handle case where all attention could be 0)
            row_sums = masked_attention.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums > 0, row_sums, 1)  # Avoid division by zero
            masked_attention = masked_attention / row_sums

            if embeddings is not None:
                # Compute actual embedding deviation
                # Original weighted embedding: sum over cell types of (attention * embedding)
                # Shape: [n_subjects, embed_dim]
                original_weighted = np.einsum("sc,scd->sd", attention, embeddings)
                ablated_weighted = np.einsum("sc,scd->sd", masked_attention, embeddings)

                # L2 distance between original and ablated
                deviation = np.linalg.norm(original_weighted - ablated_weighted, axis=1)
            else:
                # Use attention-based proxy: measure change in attention distribution
                # KL divergence or simpler: sum of absolute differences
                deviation = np.abs(attention - masked_attention).sum(axis=1)

            importances.append(deviation)

        importances = np.array(importances)  # [n_cell_types, n_subjects]

        # Compute mean/std importance per cell type
        mean_importance = importances.mean(axis=1)
        std_importance = importances.std(axis=1)

        # Stratified by pathology tertiles
        df = self._build_ablation_dataframe(importances, mean_importance, std_importance, n_cell_types)
        return df

    def _ablation_node_removal(self) -> pd.DataFrame:
        """
        Option B: Node removal ablation.

        For each cell type:
        1. Remove the cell type entirely (as if it doesn't exist)
        2. Measure how much the remaining attention distribution changes
        3. This captures structural importance in the attention mechanism

        This differs from zero_embedding in that it completely removes the node
        rather than setting its contribution to zero and renormalizing.

        Returns:
            DataFrame with cell_type, importance, importance_std, rank
        """
        # Average attention across heads: [n_subjects, n_cell_types]
        attention = self.attention.mean(axis=1)
        n_subjects, n_cell_types = attention.shape

        # Compute importance for each cell type
        importances = []

        for ct_idx in range(n_cell_types):
            # Create mask for remaining cell types
            remaining_mask = np.ones(n_cell_types, dtype=bool)
            remaining_mask[ct_idx] = False

            # Original attention on remaining cell types (before renorm)
            original_remaining = attention[:, remaining_mask]

            # After node removal, the attention on remaining nodes would need to
            # redistribute. We measure how much the original attention on this node
            # was contributing relative to others.

            # Node importance = how much attention was on this node
            node_attention = attention[:, ct_idx]

            # Additionally, measure entropy change when removing this node
            # Higher entropy change = more structurally important
            original_entropy = attention_entropy(attention, axis=1)

            # Renormalized attention without this node
            remaining_sum = original_remaining.sum(axis=1, keepdims=True)
            remaining_sum = np.where(remaining_sum > 0, remaining_sum, 1)
            renorm_remaining = original_remaining / remaining_sum

            # Pad back to original size for entropy comparison
            renorm_full = np.zeros_like(attention)
            renorm_full[:, remaining_mask] = renorm_remaining
            ablated_entropy = attention_entropy(renorm_full[:, remaining_mask], axis=1)

            # Importance combines: attention magnitude + entropy change
            # Normalize each component to [0, 1] range for fair combination
            attention_component = node_attention
            entropy_change = np.abs(original_entropy - ablated_entropy)

            # Combined importance (attention magnitude is primary)
            deviation = attention_component + 0.1 * entropy_change

            importances.append(deviation)

        importances = np.array(importances)  # [n_cell_types, n_subjects]

        # Compute mean/std importance per cell type
        mean_importance = importances.mean(axis=1)
        std_importance = importances.std(axis=1)

        # Stratified by pathology tertiles
        df = self._build_ablation_dataframe(importances, mean_importance, std_importance, n_cell_types)
        return df

    def _build_ablation_dataframe(
        self,
        importances: np.ndarray,
        mean_importance: np.ndarray,
        std_importance: np.ndarray,
        n_cell_types: int,
    ) -> pd.DataFrame:
        """Build ablation result DataFrame with pathology tertile stratification.

        Args:
            importances: [n_cell_types, n_subjects] raw importance scores
            mean_importance: [n_cell_types] mean across subjects
            std_importance: [n_cell_types] std across subjects
            n_cell_types: number of cell types
        """
        # Binary stratification (backward compat)
        importance_high_path = (
            importances[:, self.high_pathology_mask].mean(axis=1)
            if self.high_pathology_mask.any() else np.zeros(n_cell_types)
        )
        importance_low_path = (
            importances[:, ~self.high_pathology_mask].mean(axis=1)
            if (~self.high_pathology_mask).any() else np.zeros(n_cell_types)
        )

        # Tertile stratification (low/medium/high)
        importance_low_tertile = (
            importances[:, self.pathology_low_mask].mean(axis=1)
            if self.pathology_low_mask.any() else np.zeros(n_cell_types)
        )
        importance_med_tertile = (
            importances[:, self.pathology_med_mask].mean(axis=1)
            if self.pathology_med_mask.any() else np.zeros(n_cell_types)
        )
        importance_high_tertile = (
            importances[:, self.pathology_high_mask].mean(axis=1)
            if self.pathology_high_mask.any() else np.zeros(n_cell_types)
        )

        df = pd.DataFrame({
            "cell_type": self.cell_type_names,
            "importance": mean_importance,
            "importance_std": std_importance,
            "importance_high_pathology": importance_high_path,
            "importance_low_pathology": importance_low_path,
            "importance_low_tertile": importance_low_tertile,
            "importance_med_tertile": importance_med_tertile,
            "importance_high_tertile": importance_high_tertile,
        })

        df = df.sort_values("importance", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)
        return df

    def _compare_ablation_methods(
        self,
        zero_embedding_df: pd.DataFrame,
        node_removal_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compare results from both ablation methods.

        Identifies cell types where the methods agree/disagree on importance.

        Args:
            zero_embedding_df: Results from zero embedding ablation
            node_removal_df: Results from node removal ablation

        Returns:
            DataFrame with comparison metrics
        """
        # Merge on cell type
        merged = zero_embedding_df.merge(
            node_removal_df,
            on="cell_type",
            suffixes=("_zero", "_node"),
        )

        # Compute rank difference (positive = more important in zero_embedding)
        merged["rank_diff"] = merged["rank_node"] - merged["rank_zero"]

        # Compute importance ratio
        merged["importance_ratio"] = merged["importance_zero"] / (merged["importance_node"] + 1e-10)

        # Agreement score: 1 if ranks are within 3 of each other
        merged["rank_agreement"] = np.abs(merged["rank_diff"]) <= 3

        # Compute Spearman correlation between methods (scalar)
        from scipy.stats import spearmanr
        corr, pval = spearmanr(merged["rank_zero"], merged["rank_node"])

        # Categorize: "transcriptional" (higher in zero_embedding) vs "structural" (higher in node_removal)
        def categorize(row):
            if row["rank_diff"] > 3:
                return "structural"  # More important in node removal
            elif row["rank_diff"] < -3:
                return "transcriptional"  # More important in zero embedding
            else:
                return "consistent"

        merged["importance_type"] = merged.apply(categorize, axis=1)

        # Select and rename columns
        result = merged[[
            "cell_type",
            "importance_zero",
            "importance_node",
            "rank_zero",
            "rank_node",
            "rank_diff",
            "importance_ratio",
            "rank_agreement",
            "importance_type",
        ]].copy()

        result.columns = [
            "cell_type",
            "importance_zero_embedding",
            "importance_node_removal",
            "rank_zero_embedding",
            "rank_node_removal",
            "rank_difference",
            "importance_ratio",
            "methods_agree",
            "importance_type",
        ]

        # Add correlation as metadata column (same value for all rows)
        result["methods_correlation"] = corr
        result["correlation_pvalue"] = pval

        return result

    def _run_regional_ablation(
        self,
        method: Literal["both", "zero_embedding", "node_removal"] = "both",
        embeddings: np.ndarray | None = None,
    ) -> pd.DataFrame | None:
        """
        Run ablation study stratified by brain region.

        For each region, runs the ablation analysis on subjects from that region only.

        Args:
            method: Which ablation method(s) to run
            embeddings: Optional embeddings for zero-embedding ablation

        Returns:
            DataFrame with region, cell_type, method, importance, etc.
            Returns None if region_labels not provided.
        """
        if self.region_labels is None:
            return None

        unique_regions = np.unique(self.region_labels)
        rows = []

        for region in unique_regions:
            # Get subjects from this region
            region_mask = self.region_labels == region
            n_region = region_mask.sum()

            if n_region < 6:
                logger.warning(f"Region '{region}' has only {n_region} subjects, skipping ablation")
                continue

            # Get attention for this region's subjects
            region_attention = self.attention[region_mask]  # [n_region, n_heads, n_cell_types]
            attention_per_subject = region_attention.mean(axis=1)  # [n_region, n_cell_types]
            n_subjects, n_cell_types = attention_per_subject.shape

            # Run zero embedding ablation for this region
            if method in ("both", "zero_embedding"):
                for ct_idx, ct_name in enumerate(self.cell_type_names):
                    # Create masked attention
                    masked_attention = attention_per_subject.copy()
                    masked_attention[:, ct_idx] = 0
                    row_sums = masked_attention.sum(axis=1, keepdims=True)
                    row_sums = np.where(row_sums > 0, row_sums, 1)
                    masked_attention = masked_attention / row_sums

                    # Measure deviation
                    if embeddings is not None:
                        region_embeddings = embeddings[region_mask]
                        original_weighted = np.einsum("sc,scd->sd", attention_per_subject, region_embeddings)
                        ablated_weighted = np.einsum("sc,scd->sd", masked_attention, region_embeddings)
                        deviation = np.linalg.norm(original_weighted - ablated_weighted, axis=1)
                    else:
                        deviation = np.abs(attention_per_subject - masked_attention).sum(axis=1)

                    rows.append({
                        "region": region,
                        "cell_type": ct_name,
                        "method": "zero_embedding",
                        "importance": float(deviation.mean()),
                        "importance_std": float(deviation.std()),
                        "n_subjects": n_subjects,
                    })

            # Run node removal ablation for this region
            if method in ("both", "node_removal"):
                for ct_idx, ct_name in enumerate(self.cell_type_names):
                    # Node importance = attention magnitude on this node
                    node_attention = attention_per_subject[:, ct_idx]

                    rows.append({
                        "region": region,
                        "cell_type": ct_name,
                        "method": "node_removal",
                        "importance": float(node_attention.mean()),
                        "importance_std": float(node_attention.std()),
                        "n_subjects": n_subjects,
                    })

        if not rows:
            return None

        df = pd.DataFrame(rows)

        # Add rank within each region-method combination
        df["rank"] = df.groupby(["region", "method"])["importance"].rank(ascending=False).astype(int)

        return df.sort_values(["region", "method", "rank"]).reset_index(drop=True)

    def _compute_group_statistics(self) -> pd.DataFrame:
        """
        Compute descriptive statistics for resilient vs vulnerable groups.

        Returns:
            DataFrame with group-level statistics
        """
        rows = []

        # Resilient group
        if self.n_resilient > 0:
            rows.append({
                "group": "resilient",
                "n_subjects": int(self.n_resilient),
                "mean_pathology": float(self.pathology_scores[self.resilient_mask].mean()),
                "std_pathology": float(self.pathology_scores[self.resilient_mask].std()),
                "mean_cognition": float(self.cognition_scores[self.resilient_mask].mean()),
                "std_cognition": float(self.cognition_scores[self.resilient_mask].std()),
            })

        # Vulnerable group
        if self.n_vulnerable > 0:
            rows.append({
                "group": "vulnerable",
                "n_subjects": int(self.n_vulnerable),
                "mean_pathology": float(self.pathology_scores[self.vulnerable_mask].mean()),
                "std_pathology": float(self.pathology_scores[self.vulnerable_mask].std()),
                "mean_cognition": float(self.cognition_scores[self.vulnerable_mask].mean()),
                "std_cognition": float(self.cognition_scores[self.vulnerable_mask].std()),
            })

        return pd.DataFrame(rows)

    def save(
        self,
        result: ResilienceSignatureResult,
        output_dir: str | Path,
        formats: list[Literal["parquet", "csv"]] | None = None,
    ) -> dict[str, Path]:
        """
        Save analysis results to files.

        Args:
            result: ResilienceSignatureResult to save
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

        # Save signature
        for fmt in formats:
            path = output_dir / f"resilience_signature.{fmt}"
            save_dataframe(result.signature, path, fmt)
            saved_files[f"signature_{fmt}"] = path

        # Save p-values (if available)
        if result.permutation_pvalues is not None:
            for fmt in formats:
                path = output_dir / f"signature_pvalues.{fmt}"
                save_dataframe(result.permutation_pvalues, path, fmt)
                saved_files[f"pvalues_{fmt}"] = path

        # Save permutation null distribution to HDF5 (if available)
        if result.permutation_null is not None:
            import h5py
            path = output_dir / "resilience_permutation_null.h5"
            with h5py.File(path, "w") as f:
                f.attrs["schema_version"] = "2.0"
                f.create_dataset(
                    "null_distribution",
                    data=result.permutation_null,
                    compression="gzip",
                    compression_opts=4,
                )
                f.attrs["n_permutations"] = result.permutation_null.shape[0]
                f.attrs["n_cell_types"] = result.permutation_null.shape[1]
                f.attrs["cell_type_names"] = self.cell_type_names
            saved_files["permutation_null_h5"] = path

        # Save group statistics
        if result.group_statistics is not None:
            for fmt in formats:
                path = output_dir / f"group_statistics.{fmt}"
                save_dataframe(result.group_statistics, path, fmt)
                saved_files[f"group_stats_{fmt}"] = path

        # Save regional results (if available)
        if result.by_region is not None:
            for fmt in formats:
                path = output_dir / f"resilience_signature_by_region.{fmt}"
                save_dataframe(result.by_region, path, fmt)
                saved_files[f"by_region_{fmt}"] = path

        # Save ablation results (if available)
        if result.ablation_results is not None:
            for fmt in formats:
                path = output_dir / f"ablation_importance.{fmt}"
                save_dataframe(result.ablation_results, path, fmt)
                saved_files[f"ablation_{fmt}"] = path

        # Save ablation comparison (if available)
        if result.ablation_comparison is not None:
            for fmt in formats:
                path = output_dir / f"ablation_comparison.{fmt}"
                save_dataframe(result.ablation_comparison, path, fmt)
                saved_files[f"ablation_comparison_{fmt}"] = path

        # Save regional ablation results (if available)
        if result.ablation_by_region is not None:
            for fmt in formats:
                path = output_dir / f"ablation_by_region.{fmt}"
                save_dataframe(result.ablation_by_region, path, fmt)
                saved_files[f"ablation_by_region_{fmt}"] = path

        logger.info(f"Saved resilience signature analysis to {output_dir}")
        return saved_files


def compute_resilience_signature(
    attention: np.ndarray,
    pathology_scores: np.ndarray,
    cognition_scores: np.ndarray,
    subject_ids: list[str] | None = None,
    cell_type_names: list[str] | None = None,
    n_permutations: int = 1000,
    output_dir: str | Path | None = None,
) -> ResilienceSignatureResult:
    """
    Convenience function to compute and optionally save resilience signature.

    Args:
        attention: Pathology attention weights [n_subjects, n_heads, n_cell_types]
        pathology_scores: Pathology burden scores [n_subjects]
        cognition_scores: Cognitive scores [n_subjects]
        subject_ids: Subject identifiers
        cell_type_names: Cell type names
        n_permutations: Number of permutations for significance testing
        output_dir: If provided, save results to this directory

    Returns:
        ResilienceSignatureResult with analysis results
    """
    analyzer = ResilienceSignatureAnalyzer(
        attention=attention,
        pathology_scores=pathology_scores,
        cognition_scores=cognition_scores,
        subject_ids=subject_ids,
        cell_type_names=cell_type_names,
    )

    result = analyzer.analyze(n_permutations=n_permutations)

    if output_dir is not None:
        analyzer.save(result, output_dir)

    return result
