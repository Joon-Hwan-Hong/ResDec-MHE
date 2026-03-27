"""
Subgroup uncertainty analysis — stratify predictive uncertainty by demographics,
pathology, and cognition.

Produces:
1. subgroup_stats — per-scheme per-group summary (n, mean/std of uncertainty, R², calibration)
2. between_group_tests — Kruskal-Wallis + Cohen's d for 2-group schemes
3. covariate_correlations — Spearman rho of uncertainty with continuous covariates
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import pandas as pd
import torch
from scipy.stats import kruskal, spearmanr

from src.training.metrics import ResilienceMetrics
from src.utils.io import save_dataframe
from src.utils.statistics import (
    benjamini_hochberg,
    cohens_d,
    derive_resilience_groups,
)

logger = logging.getLogger(__name__)

# Minimum group size for inclusion in statistical tests
MIN_GROUP_SIZE = 3


# =============================================================================
# Default schemes and covariates
# =============================================================================

DEFAULT_CATEGORICAL_SCHEMES: dict[str, str | Callable] = {
    "resilience": lambda df: derive_resilience_groups(
        df["cogn_global"].values, df["gpath"].values
    ),
    "sex": "msex",
    "apoe_e4_carrier": lambda df: np.where(
        df["apoe_genotype"].astype(str).str.contains("4"), "E4+", "E4-"
    ),
    "apoe_genotype": "apoe_genotype",
    "age_tertile": lambda df: pd.qcut(
        df["age_death"], 3, labels=["young", "middle", "old"], duplicates="drop"
    ),
    "gpath_tertile": lambda df: pd.qcut(
        df["gpath"], 3, labels=["low", "mid", "high"], duplicates="drop"
    ),
    "amyloid_tertile": lambda df: pd.qcut(
        df["amylsqrt"], 3, labels=["low", "mid", "high"], duplicates="drop"
    ),
    "tau_tertile": lambda df: pd.qcut(
        df["tangsqrt"], 3, labels=["low", "mid", "high"], duplicates="drop"
    ),
    "braak": "braaksc",
    "cerad": "ceradsc",
    "nia_reagan": "niareagansc",
    "cogdx": "cogdx",
}

DEFAULT_CONTINUOUS_COVARIATES: list[str] = [
    "cogn_ep_lv",
    "cogn_po_lv",
    "cogn_ps_lv",
    "cogn_se_lv",
    "cogn_wo_lv",
    "cogn_global",
    "cts_mmse30_lv",
    "gpath",
    "amylsqrt",
    "tangsqrt",
    "age_death",
]


# =============================================================================
# Result dataclass
# =============================================================================


@dataclass
class SubgroupUncertaintyResult:
    """Container for subgroup uncertainty analysis results.

    Attributes:
        subgroup_stats: Per-scheme per-group summary statistics.
            Columns: scheme, group, n, mean_std, std_std,
                     mean_epistemic, std_epistemic, mean_aleatoric, std_aleatoric,
                     r2, calibration_error
        between_group_tests: Between-group statistical tests.
            Columns: scheme, test_name, test_stat, pvalue, effect_size
        covariate_correlations: Spearman correlations with continuous covariates.
            Columns: covariate, uncertainty_type, spearman_rho, pvalue,
                     pvalue_fdr, significant_fdr
        metadata: Analysis metadata dict.
    """

    subgroup_stats: pd.DataFrame
    between_group_tests: pd.DataFrame
    covariate_correlations: pd.DataFrame
    metadata: dict = field(default_factory=dict)


# =============================================================================
# Analyzer
# =============================================================================


class SubgroupUncertaintyAnalyzer:
    """Stratify predictive uncertainty by categorical subgroups and continuous covariates."""

    def __init__(
        self,
        predicted_mean: np.ndarray,
        predicted_std: np.ndarray,
        actual: np.ndarray | None,
        subject_metadata: pd.DataFrame,
        subject_ids: list[str],
        epistemic_std: np.ndarray | None = None,
        aleatoric_std: np.ndarray | None = None,
    ):
        self.predicted_mean = np.asarray(predicted_mean).flatten()
        self.predicted_std = np.asarray(predicted_std).flatten()
        self.actual = np.asarray(actual).flatten() if actual is not None else None
        self.subject_metadata = subject_metadata.copy()
        self.subject_ids = list(subject_ids)
        self.epistemic_std = (
            np.asarray(epistemic_std).flatten() if epistemic_std is not None else None
        )
        self.aleatoric_std = (
            np.asarray(aleatoric_std).flatten() if aleatoric_std is not None else None
        )

        self._validate_inputs()
        # Build aligned metadata indexed by subject_ids order
        self._aligned_metadata = self._align_metadata()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_inputs(self) -> None:
        n = len(self.predicted_mean)

        if len(self.predicted_std) != n:
            raise ValueError(
                f"predicted_std length ({len(self.predicted_std)}) "
                f"!= predicted_mean length ({n})"
            )
        if self.actual is not None and len(self.actual) != n:
            raise ValueError(
                f"actual length ({len(self.actual)}) != predicted_mean length ({n})"
            )
        if len(self.subject_ids) != n:
            raise ValueError(
                f"subject_ids length ({len(self.subject_ids)}) "
                f"!= predicted_mean length ({n})"
            )
        if self.epistemic_std is not None and len(self.epistemic_std) != n:
            raise ValueError(
                f"epistemic_std length ({len(self.epistemic_std)}) "
                f"!= predicted_mean length ({n})"
            )
        if self.aleatoric_std is not None and len(self.aleatoric_std) != n:
            raise ValueError(
                f"aleatoric_std length ({len(self.aleatoric_std)}) "
                f"!= predicted_mean length ({n})"
            )
        if (self.predicted_std <= 0).any():
            raise ValueError("predicted_std must be positive")

    def _align_metadata(self) -> pd.DataFrame:
        """Build a metadata DataFrame aligned to ``subject_ids`` order.

        The caller guarantees that ``subject_ids`` entries exist in
        ``subject_metadata`` (either as the index or in a ``subject_id`` /
        ``projid`` column). Rows are re-ordered to match ``self.subject_ids``.
        """
        meta = self.subject_metadata.copy()

        # Try to find a subject-id column to use as join key
        for col in ("subject_id", "projid"):
            if col in meta.columns:
                meta = meta.set_index(col)
                break

        # Re-index to match subject_ids order.  If not all ids are found the
        # corresponding rows will be NaN — downstream code handles NaN
        # gracefully per scheme.
        aligned = meta.reindex(self.subject_ids)
        aligned.index = list(range(len(self.subject_ids)))
        return aligned

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        categorical_schemes: dict[str, str | Callable] | None = None,
        continuous_covariates: list[str] | None = None,
    ) -> SubgroupUncertaintyResult:
        """Run the full subgroup uncertainty analysis.

        Args:
            categorical_schemes: Mapping of scheme_name to either a column
                name (str) or a callable(df) -> array-like of group labels.
                ``None`` uses ``DEFAULT_CATEGORICAL_SCHEMES``.
            continuous_covariates: List of column names for Spearman
                correlation.  ``None`` uses ``DEFAULT_CONTINUOUS_COVARIATES``.

        Returns:
            ``SubgroupUncertaintyResult`` with all DataFrames populated.
        """
        if categorical_schemes is None:
            categorical_schemes = DEFAULT_CATEGORICAL_SCHEMES
        if continuous_covariates is None:
            continuous_covariates = DEFAULT_CONTINUOUS_COVARIATES

        # --- categorical ---
        subgroup_rows: list[dict] = []
        test_rows: list[dict] = []

        for scheme_name, grouper in categorical_schemes.items():
            group_labels = self._resolve_groups(scheme_name, grouper)
            if group_labels is None:
                continue  # missing column — already logged

            self._analyze_scheme(
                scheme_name, group_labels, subgroup_rows, test_rows
            )

        subgroup_stats = pd.DataFrame(subgroup_rows)
        between_group_tests = pd.DataFrame(test_rows)

        # --- continuous ---
        covariate_correlations = self._analyze_continuous(continuous_covariates)

        metadata = {
            "n_subjects": len(self.predicted_mean),
            "n_categorical_schemes": len(categorical_schemes),
            "n_continuous_covariates": len(continuous_covariates),
            "has_actual": self.actual is not None,
            "has_epistemic": self.epistemic_std is not None,
            "has_aleatoric": self.aleatoric_std is not None,
            "schemes_analyzed": list(
                subgroup_stats["scheme"].unique()
            )
            if len(subgroup_stats) > 0
            else [],
        }

        return SubgroupUncertaintyResult(
            subgroup_stats=subgroup_stats,
            between_group_tests=between_group_tests,
            covariate_correlations=covariate_correlations,
            metadata=metadata,
        )

    def save(
        self,
        result: SubgroupUncertaintyResult,
        output_dir: str | Path,
        formats: list[Literal["parquet", "csv"]] | None = None,
    ) -> dict[str, Path]:
        """Save results to disk.

        Args:
            result: Analysis result to persist.
            output_dir: Target directory (created if absent).
            formats: File formats (default: ``["parquet", "csv"]``).

        Returns:
            Dict mapping logical name -> saved file path.
        """
        if formats is None:
            formats = ["parquet", "csv"]

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved: dict[str, Path] = {}

        for name, df in [
            ("subgroup_stats", result.subgroup_stats),
            ("between_group_tests", result.between_group_tests),
            ("covariate_correlations", result.covariate_correlations),
        ]:
            if df is not None and len(df) > 0:
                for fmt in formats:
                    path = output_dir / f"subgroup_{name}.{fmt}"
                    save_dataframe(df, path, fmt)
                    saved[f"{name}_{fmt}"] = path

        logger.info("Saved subgroup uncertainty analysis to %s", output_dir)
        return saved

    # ------------------------------------------------------------------
    # Internals — group resolution
    # ------------------------------------------------------------------

    def _resolve_groups(
        self, scheme_name: str, grouper: str | Callable
    ) -> np.ndarray | None:
        """Resolve a grouper specification into an array of string labels.

        Returns ``None`` when required columns are missing (logged as warning).
        """
        try:
            if callable(grouper):
                labels = grouper(self._aligned_metadata)
            else:
                # grouper is a column name
                if grouper not in self._aligned_metadata.columns:
                    logger.warning(
                        "Scheme '%s': column '%s' not in metadata — skipping",
                        scheme_name,
                        grouper,
                    )
                    return None
                labels = self._aligned_metadata[grouper].values
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning(
                "Scheme '%s' could not be resolved (%s) — skipping",
                scheme_name,
                exc,
            )
            return None

        # Coerce to string array for uniform handling
        labels = np.asarray(labels, dtype=str)
        return labels

    # ------------------------------------------------------------------
    # Internals — per-scheme analysis
    # ------------------------------------------------------------------

    def _analyze_scheme(
        self,
        scheme_name: str,
        group_labels: np.ndarray,
        subgroup_rows: list[dict],
        test_rows: list[dict],
    ) -> None:
        """Compute per-group stats and between-group tests for one scheme."""
        # Drop subjects whose label is NaN / empty / "nan"
        valid_mask = ~np.isin(group_labels, ["", "nan", "None", "NaN"])
        if valid_mask.sum() == 0:
            logger.warning(
                "Scheme '%s': no valid group labels — skipping", scheme_name
            )
            return

        unique_labels = np.unique(group_labels[valid_mask])

        metrics_calculator = ResilienceMetrics()

        # --- per-group stats ---
        for label in unique_labels:
            mask = (group_labels == label) & valid_mask
            n = int(mask.sum())
            if n == 0:
                continue

            row: dict = {
                "scheme": scheme_name,
                "group": label,
                "n": n,
                "mean_std": float(self.predicted_std[mask].mean()),
                "std_std": float(self.predicted_std[mask].std()),
            }

            # Epistemic
            if self.epistemic_std is not None:
                row["mean_epistemic"] = float(self.epistemic_std[mask].mean())
                row["std_epistemic"] = float(self.epistemic_std[mask].std())
            else:
                row["mean_epistemic"] = np.nan
                row["std_epistemic"] = np.nan

            # Aleatoric
            if self.aleatoric_std is not None:
                row["mean_aleatoric"] = float(self.aleatoric_std[mask].mean())
                row["std_aleatoric"] = float(self.aleatoric_std[mask].std())
            else:
                row["mean_aleatoric"] = np.nan
                row["std_aleatoric"] = np.nan

            # R² and calibration within group (require actual values)
            if self.actual is not None and n >= MIN_GROUP_SIZE:
                group_mean = self.predicted_mean[mask]
                group_target = self.actual[mask]
                group_std = self.predicted_std[mask]

                metrics = metrics_calculator.compute(
                    mean=torch.tensor(group_mean.reshape(-1, 1)),
                    std=torch.tensor(group_std.reshape(-1, 1)),
                    target=torch.tensor(group_target.reshape(-1, 1)),
                )
                row["r2"] = metrics["r2"]
                row["calibration_error"] = metrics["calibration_error"]
            else:
                row["r2"] = np.nan
                row["calibration_error"] = np.nan

            subgroup_rows.append(row)

        # --- between-group tests ---
        # Collect per-group uncertainty arrays (only groups with n >= MIN_GROUP_SIZE)
        testable_labels = [
            lbl
            for lbl in unique_labels
            if ((group_labels == lbl) & valid_mask).sum() >= MIN_GROUP_SIZE
        ]

        if len(testable_labels) < 2:
            return

        group_arrays = [
            self.predicted_std[(group_labels == lbl) & valid_mask]
            for lbl in testable_labels
        ]

        # Kruskal-Wallis
        stat, pval = kruskal(*group_arrays)
        test_rows.append(
            {
                "scheme": scheme_name,
                "test_name": "kruskal_wallis",
                "test_stat": float(stat),
                "pvalue": float(pval),
                "effect_size": np.nan,  # KW doesn't have a simple effect size
                "groups_compared": ", ".join(testable_labels),
            }
        )

        # Cohen's d for 2-group schemes (all pairwise if >2)
        if len(testable_labels) == 2:
            d = cohens_d(group_arrays[0], group_arrays[1])
            test_rows.append(
                {
                    "scheme": scheme_name,
                    "test_name": "cohens_d",
                    "test_stat": np.nan,
                    "pvalue": np.nan,
                    "effect_size": float(d),
                    "groups_compared": f"{testable_labels[0]} vs {testable_labels[1]}",
                }
            )
        elif len(testable_labels) > 2:
            # Pairwise Cohen's d
            for i, j in combinations(range(len(testable_labels)), 2):
                d = cohens_d(group_arrays[i], group_arrays[j])
                test_rows.append(
                    {
                        "scheme": scheme_name,
                        "test_name": "cohens_d",
                        "test_stat": np.nan,
                        "pvalue": np.nan,
                        "effect_size": float(d),
                        "groups_compared": (
                            f"{testable_labels[i]} vs {testable_labels[j]}"
                        ),
                    }
                )

    # ------------------------------------------------------------------
    # Internals — continuous covariates
    # ------------------------------------------------------------------

    def _analyze_continuous(
        self, covariates: list[str]
    ) -> pd.DataFrame:
        """Compute Spearman correlations between uncertainty and continuous covariates."""
        rows: list[dict] = []

        uncertainty_types: dict[str, np.ndarray | None] = {
            "total": self.predicted_std,
            "epistemic": self.epistemic_std,
            "aleatoric": self.aleatoric_std,
        }

        for cov_name in covariates:
            if cov_name not in self._aligned_metadata.columns:
                logger.warning(
                    "Continuous covariate '%s' not in metadata — skipping",
                    cov_name,
                )
                continue

            values = self._aligned_metadata[cov_name].values

            # Must be numeric
            if not np.issubdtype(values.dtype, np.number):
                try:
                    values = values.astype(float)
                except (ValueError, TypeError):
                    logger.warning(
                        "Covariate '%s' is not numeric — skipping", cov_name
                    )
                    continue

            for unc_type, unc_arr in uncertainty_types.items():
                if unc_arr is None:
                    continue

                valid = ~(np.isnan(values) | np.isnan(unc_arr))
                if valid.sum() < MIN_GROUP_SIZE:
                    continue

                # Skip constant columns
                if np.std(values[valid]) == 0 or np.std(unc_arr[valid]) == 0:
                    continue

                rho, pval = spearmanr(values[valid], unc_arr[valid])
                rows.append(
                    {
                        "covariate": cov_name,
                        "uncertainty_type": unc_type,
                        "spearman_rho": float(rho),
                        "pvalue": float(pval),
                    }
                )

        df = pd.DataFrame(rows)

        if len(df) > 0:
            # FDR correction across all tests
            fdr_values, sig_mask = benjamini_hochberg(df["pvalue"].values)
            df["pvalue_fdr"] = fdr_values
            df["significant_fdr"] = sig_mask
            df = df.sort_values("pvalue").reset_index(drop=True)

        return df
