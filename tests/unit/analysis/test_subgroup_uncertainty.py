"""Tests for subgroup uncertainty analysis module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.subgroup_uncertainty import (
    DEFAULT_CATEGORICAL_SCHEMES,
    DEFAULT_CONTINUOUS_COVARIATES,
    MIN_GROUP_SIZE,
    SubgroupUncertaintyAnalyzer,
    SubgroupUncertaintyResult,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def synthetic_data():
    """Create 100 synthetic subjects with known subgroup structure.

    Groups:
    - sex (msex): 0 or 1 — 50/50 split
    - cogdx: 1, 2, or 3 — roughly 34/33/33 split
    - braaksc: 0..5

    Uncertainty deliberately differs between sex groups so that Kruskal-Wallis
    should detect a difference.
    """
    np.random.seed(42)
    n = 100

    subject_ids = [f"ROSMAP_{i:04d}" for i in range(n)]
    sex = np.array([0] * 50 + [1] * 50)
    cogdx = np.array([1] * 34 + [2] * 33 + [3] * 33)
    braaksc = np.random.choice([0, 1, 2, 3, 4, 5], size=n)

    # Continuous covariates
    cogn_global = np.random.randn(n) * 0.5 + 0.1
    gpath = np.random.rand(n) * 3.0
    age_death = np.random.uniform(65, 95, n)
    amylsqrt = np.random.rand(n) * 2.0
    tangsqrt = np.random.rand(n) * 2.0

    # Predictions — sex=1 group has higher uncertainty (injected signal)
    predicted_mean = cogn_global + np.random.randn(n) * 0.2
    base_std = np.abs(np.random.randn(n)) * 0.3 + 0.2
    predicted_std = base_std.copy()
    predicted_std[sex == 1] += 0.5  # inject difference

    actual = cogn_global

    epistemic_std = predicted_std * 0.6
    aleatoric_std = predicted_std * 0.4

    metadata = pd.DataFrame(
        {
            "subject_id": subject_ids,
            "msex": sex,
            "cogdx": cogdx,
            "braaksc": braaksc,
            "cogn_global": cogn_global,
            "gpath": gpath,
            "age_death": age_death,
            "amylsqrt": amylsqrt,
            "tangsqrt": tangsqrt,
            "apoe_genotype": np.random.choice(["33", "34", "44", "23"], size=n),
            "ceradsc": np.random.choice([1, 2, 3, 4], size=n),
            "niareagansc": np.random.choice([1, 2, 3, 4], size=n),
        }
    )

    return {
        "predicted_mean": predicted_mean,
        "predicted_std": predicted_std,
        "actual": actual,
        "subject_ids": subject_ids,
        "metadata": metadata,
        "epistemic_std": epistemic_std,
        "aleatoric_std": aleatoric_std,
    }


# =============================================================================
# Initialization
# =============================================================================


class TestSubgroupUncertaintyAnalyzerInit:
    """Test SubgroupUncertaintyAnalyzer initialization and validation."""

    def test_basic_init(self, synthetic_data):
        analyzer = SubgroupUncertaintyAnalyzer(
            predicted_mean=synthetic_data["predicted_mean"],
            predicted_std=synthetic_data["predicted_std"],
            actual=synthetic_data["actual"],
            subject_metadata=synthetic_data["metadata"],
            subject_ids=synthetic_data["subject_ids"],
        )
        assert len(analyzer.predicted_mean) == 100
        assert analyzer.epistemic_std is None
        assert analyzer.aleatoric_std is None

    def test_init_with_epistemic_aleatoric(self, synthetic_data):
        analyzer = SubgroupUncertaintyAnalyzer(
            predicted_mean=synthetic_data["predicted_mean"],
            predicted_std=synthetic_data["predicted_std"],
            actual=synthetic_data["actual"],
            subject_metadata=synthetic_data["metadata"],
            subject_ids=synthetic_data["subject_ids"],
            epistemic_std=synthetic_data["epistemic_std"],
            aleatoric_std=synthetic_data["aleatoric_std"],
        )
        assert analyzer.epistemic_std is not None
        assert analyzer.aleatoric_std is not None

    def test_mismatched_std_length(self, synthetic_data):
        with pytest.raises(ValueError, match="predicted_std"):
            SubgroupUncertaintyAnalyzer(
                predicted_mean=synthetic_data["predicted_mean"],
                predicted_std=synthetic_data["predicted_std"][:50],
                actual=synthetic_data["actual"],
                subject_metadata=synthetic_data["metadata"],
                subject_ids=synthetic_data["subject_ids"],
            )

    def test_mismatched_actual_length(self, synthetic_data):
        with pytest.raises(ValueError, match="actual"):
            SubgroupUncertaintyAnalyzer(
                predicted_mean=synthetic_data["predicted_mean"],
                predicted_std=synthetic_data["predicted_std"],
                actual=synthetic_data["actual"][:50],
                subject_metadata=synthetic_data["metadata"],
                subject_ids=synthetic_data["subject_ids"],
            )

    def test_mismatched_subject_ids(self, synthetic_data):
        with pytest.raises(ValueError, match="subject_ids"):
            SubgroupUncertaintyAnalyzer(
                predicted_mean=synthetic_data["predicted_mean"],
                predicted_std=synthetic_data["predicted_std"],
                actual=synthetic_data["actual"],
                subject_metadata=synthetic_data["metadata"],
                subject_ids=synthetic_data["subject_ids"][:50],
            )

    def test_negative_std_rejected(self, synthetic_data):
        bad_std = synthetic_data["predicted_std"].copy()
        bad_std[0] = -0.1
        with pytest.raises(ValueError, match="positive"):
            SubgroupUncertaintyAnalyzer(
                predicted_mean=synthetic_data["predicted_mean"],
                predicted_std=bad_std,
                actual=synthetic_data["actual"],
                subject_metadata=synthetic_data["metadata"],
                subject_ids=synthetic_data["subject_ids"],
            )

    def test_mismatched_epistemic_length(self, synthetic_data):
        with pytest.raises(ValueError, match="epistemic_std"):
            SubgroupUncertaintyAnalyzer(
                predicted_mean=synthetic_data["predicted_mean"],
                predicted_std=synthetic_data["predicted_std"],
                actual=synthetic_data["actual"],
                subject_metadata=synthetic_data["metadata"],
                subject_ids=synthetic_data["subject_ids"],
                epistemic_std=synthetic_data["epistemic_std"][:50],
            )


# =============================================================================
# Subgroup Stats
# =============================================================================


class TestSubgroupStats:
    """Test that per-group statistics are correctly computed."""

    def test_correct_group_counts(self, synthetic_data):
        """Group counts should match the injected structure."""
        analyzer = SubgroupUncertaintyAnalyzer(
            predicted_mean=synthetic_data["predicted_mean"],
            predicted_std=synthetic_data["predicted_std"],
            actual=synthetic_data["actual"],
            subject_metadata=synthetic_data["metadata"],
            subject_ids=synthetic_data["subject_ids"],
        )

        result = analyzer.analyze(
            categorical_schemes={"sex": "msex"},
            continuous_covariates=[],
        )
        stats = result.subgroup_stats
        assert len(stats) == 2  # two sex groups
        counts = stats.set_index("group")["n"]
        assert counts["0"] == 50
        assert counts["1"] == 50

    def test_multi_group_counts(self, synthetic_data):
        analyzer = SubgroupUncertaintyAnalyzer(
            predicted_mean=synthetic_data["predicted_mean"],
            predicted_std=synthetic_data["predicted_std"],
            actual=None,
            subject_metadata=synthetic_data["metadata"],
            subject_ids=synthetic_data["subject_ids"],
        )
        result = analyzer.analyze(
            categorical_schemes={"cogdx": "cogdx"},
            continuous_covariates=[],
        )
        stats = result.subgroup_stats
        assert set(stats["group"]) == {"1", "2", "3"}
        total = stats["n"].sum()
        assert total == 100

    def test_stats_include_uncertainty_columns(self, synthetic_data):
        analyzer = SubgroupUncertaintyAnalyzer(
            predicted_mean=synthetic_data["predicted_mean"],
            predicted_std=synthetic_data["predicted_std"],
            actual=synthetic_data["actual"],
            subject_metadata=synthetic_data["metadata"],
            subject_ids=synthetic_data["subject_ids"],
            epistemic_std=synthetic_data["epistemic_std"],
            aleatoric_std=synthetic_data["aleatoric_std"],
        )
        result = analyzer.analyze(
            categorical_schemes={"sex": "msex"},
            continuous_covariates=[],
        )
        cols = set(result.subgroup_stats.columns)
        for expected in [
            "scheme",
            "group",
            "n",
            "mean_std",
            "std_std",
            "mean_epistemic",
            "std_epistemic",
            "mean_aleatoric",
            "std_aleatoric",
            "r2",
            "calibration_error",
        ]:
            assert expected in cols, f"Missing column: {expected}"

    def test_epistemic_nan_when_not_provided(self, synthetic_data):
        analyzer = SubgroupUncertaintyAnalyzer(
            predicted_mean=synthetic_data["predicted_mean"],
            predicted_std=synthetic_data["predicted_std"],
            actual=None,
            subject_metadata=synthetic_data["metadata"],
            subject_ids=synthetic_data["subject_ids"],
        )
        result = analyzer.analyze(
            categorical_schemes={"sex": "msex"},
            continuous_covariates=[],
        )
        assert result.subgroup_stats["mean_epistemic"].isna().all()
        assert result.subgroup_stats["mean_aleatoric"].isna().all()

    def test_r2_nan_without_actual(self, synthetic_data):
        analyzer = SubgroupUncertaintyAnalyzer(
            predicted_mean=synthetic_data["predicted_mean"],
            predicted_std=synthetic_data["predicted_std"],
            actual=None,
            subject_metadata=synthetic_data["metadata"],
            subject_ids=synthetic_data["subject_ids"],
        )
        result = analyzer.analyze(
            categorical_schemes={"sex": "msex"},
            continuous_covariates=[],
        )
        assert result.subgroup_stats["r2"].isna().all()
        assert result.subgroup_stats["calibration_error"].isna().all()

    def test_r2_computed_with_actual(self, synthetic_data):
        analyzer = SubgroupUncertaintyAnalyzer(
            predicted_mean=synthetic_data["predicted_mean"],
            predicted_std=synthetic_data["predicted_std"],
            actual=synthetic_data["actual"],
            subject_metadata=synthetic_data["metadata"],
            subject_ids=synthetic_data["subject_ids"],
        )
        result = analyzer.analyze(
            categorical_schemes={"sex": "msex"},
            continuous_covariates=[],
        )
        # R² should be a number (not NaN) for groups with enough subjects
        assert not result.subgroup_stats["r2"].isna().all()


# =============================================================================
# Between-group tests
# =============================================================================


class TestBetweenGroupTests:
    """Test Kruskal-Wallis and Cohen's d."""

    def test_kruskal_wallis_for_multi_group(self, synthetic_data):
        analyzer = SubgroupUncertaintyAnalyzer(
            predicted_mean=synthetic_data["predicted_mean"],
            predicted_std=synthetic_data["predicted_std"],
            actual=None,
            subject_metadata=synthetic_data["metadata"],
            subject_ids=synthetic_data["subject_ids"],
        )
        result = analyzer.analyze(
            categorical_schemes={"cogdx": "cogdx"},
            continuous_covariates=[],
        )
        kw_rows = result.between_group_tests[
            result.between_group_tests["test_name"] == "kruskal_wallis"
        ]
        assert len(kw_rows) == 1
        assert kw_rows.iloc[0]["pvalue"] >= 0

    def test_kruskal_wallis_detects_injected_signal(self, synthetic_data):
        """Sex groups have injected uncertainty difference -> p should be small."""
        analyzer = SubgroupUncertaintyAnalyzer(
            predicted_mean=synthetic_data["predicted_mean"],
            predicted_std=synthetic_data["predicted_std"],
            actual=None,
            subject_metadata=synthetic_data["metadata"],
            subject_ids=synthetic_data["subject_ids"],
        )
        result = analyzer.analyze(
            categorical_schemes={"sex": "msex"},
            continuous_covariates=[],
        )
        kw_rows = result.between_group_tests[
            result.between_group_tests["test_name"] == "kruskal_wallis"
        ]
        assert len(kw_rows) == 1
        assert kw_rows.iloc[0]["pvalue"] < 0.05

    def test_cohens_d_for_two_groups(self, synthetic_data):
        analyzer = SubgroupUncertaintyAnalyzer(
            predicted_mean=synthetic_data["predicted_mean"],
            predicted_std=synthetic_data["predicted_std"],
            actual=None,
            subject_metadata=synthetic_data["metadata"],
            subject_ids=synthetic_data["subject_ids"],
        )
        result = analyzer.analyze(
            categorical_schemes={"sex": "msex"},
            continuous_covariates=[],
        )
        cd_rows = result.between_group_tests[
            result.between_group_tests["test_name"] == "cohens_d"
        ]
        assert len(cd_rows) == 1
        # Injected signal should give a non-trivial effect size
        assert abs(cd_rows.iloc[0]["effect_size"]) > 0.3

    def test_pairwise_cohens_d_for_three_groups(self, synthetic_data):
        analyzer = SubgroupUncertaintyAnalyzer(
            predicted_mean=synthetic_data["predicted_mean"],
            predicted_std=synthetic_data["predicted_std"],
            actual=None,
            subject_metadata=synthetic_data["metadata"],
            subject_ids=synthetic_data["subject_ids"],
        )
        result = analyzer.analyze(
            categorical_schemes={"cogdx": "cogdx"},
            continuous_covariates=[],
        )
        cd_rows = result.between_group_tests[
            result.between_group_tests["test_name"] == "cohens_d"
        ]
        # 3 groups => C(3,2) = 3 pairwise comparisons
        assert len(cd_rows) == 3


# =============================================================================
# Continuous covariates (Spearman correlations)
# =============================================================================


class TestCovariateCorrelations:
    """Test Spearman correlations with continuous covariates."""

    def test_correlations_computed(self, synthetic_data):
        analyzer = SubgroupUncertaintyAnalyzer(
            predicted_mean=synthetic_data["predicted_mean"],
            predicted_std=synthetic_data["predicted_std"],
            actual=None,
            subject_metadata=synthetic_data["metadata"],
            subject_ids=synthetic_data["subject_ids"],
        )
        result = analyzer.analyze(
            categorical_schemes={},
            continuous_covariates=["cogn_global", "age_death"],
        )
        corr = result.covariate_correlations
        assert len(corr) > 0
        assert "spearman_rho" in corr.columns
        assert "pvalue" in corr.columns

    def test_multiple_uncertainty_types(self, synthetic_data):
        """When epistemic + aleatoric provided, correlations for all three types."""
        analyzer = SubgroupUncertaintyAnalyzer(
            predicted_mean=synthetic_data["predicted_mean"],
            predicted_std=synthetic_data["predicted_std"],
            actual=None,
            subject_metadata=synthetic_data["metadata"],
            subject_ids=synthetic_data["subject_ids"],
            epistemic_std=synthetic_data["epistemic_std"],
            aleatoric_std=synthetic_data["aleatoric_std"],
        )
        result = analyzer.analyze(
            categorical_schemes={},
            continuous_covariates=["cogn_global"],
        )
        corr = result.covariate_correlations
        types_present = set(corr["uncertainty_type"])
        assert types_present == {"total", "epistemic", "aleatoric"}

    def test_fdr_correction_applied(self, synthetic_data):
        analyzer = SubgroupUncertaintyAnalyzer(
            predicted_mean=synthetic_data["predicted_mean"],
            predicted_std=synthetic_data["predicted_std"],
            actual=None,
            subject_metadata=synthetic_data["metadata"],
            subject_ids=synthetic_data["subject_ids"],
        )
        result = analyzer.analyze(
            categorical_schemes={},
            continuous_covariates=["cogn_global", "age_death", "gpath"],
        )
        corr = result.covariate_correlations
        assert "pvalue_fdr" in corr.columns
        assert "significant_fdr" in corr.columns

    def test_sorted_by_pvalue(self, synthetic_data):
        analyzer = SubgroupUncertaintyAnalyzer(
            predicted_mean=synthetic_data["predicted_mean"],
            predicted_std=synthetic_data["predicted_std"],
            actual=None,
            subject_metadata=synthetic_data["metadata"],
            subject_ids=synthetic_data["subject_ids"],
        )
        result = analyzer.analyze(
            categorical_schemes={},
            continuous_covariates=["cogn_global", "age_death", "gpath"],
        )
        corr = result.covariate_correlations
        pvals = corr["pvalue"].values
        assert (pvals[:-1] <= pvals[1:]).all()


# =============================================================================
# Missing columns handled gracefully
# =============================================================================


class TestMissingColumns:
    """Missing metadata columns should be skipped, not crash."""

    def test_missing_categorical_column_skipped(self, synthetic_data):
        analyzer = SubgroupUncertaintyAnalyzer(
            predicted_mean=synthetic_data["predicted_mean"],
            predicted_std=synthetic_data["predicted_std"],
            actual=None,
            subject_metadata=synthetic_data["metadata"],
            subject_ids=synthetic_data["subject_ids"],
        )
        result = analyzer.analyze(
            categorical_schemes={"nonexistent_col": "does_not_exist"},
            continuous_covariates=[],
        )
        # Should produce empty stats (no crash)
        assert len(result.subgroup_stats) == 0

    def test_missing_continuous_column_skipped(self, synthetic_data):
        analyzer = SubgroupUncertaintyAnalyzer(
            predicted_mean=synthetic_data["predicted_mean"],
            predicted_std=synthetic_data["predicted_std"],
            actual=None,
            subject_metadata=synthetic_data["metadata"],
            subject_ids=synthetic_data["subject_ids"],
        )
        result = analyzer.analyze(
            categorical_schemes={},
            continuous_covariates=["nonexistent_column"],
        )
        assert len(result.covariate_correlations) == 0

    def test_partial_missing_columns(self, synthetic_data):
        """Mix of valid and invalid columns — valid ones should still work."""
        analyzer = SubgroupUncertaintyAnalyzer(
            predicted_mean=synthetic_data["predicted_mean"],
            predicted_std=synthetic_data["predicted_std"],
            actual=None,
            subject_metadata=synthetic_data["metadata"],
            subject_ids=synthetic_data["subject_ids"],
        )
        result = analyzer.analyze(
            categorical_schemes={
                "sex": "msex",
                "nonexistent": "missing_col",
            },
            continuous_covariates=["cogn_global", "nope"],
        )
        # sex scheme should produce results
        assert len(result.subgroup_stats) > 0
        assert set(result.subgroup_stats["scheme"]) == {"sex"}
        # cogn_global should produce correlations
        assert len(result.covariate_correlations) > 0
        assert "cogn_global" in result.covariate_correlations["covariate"].values

    def test_callable_scheme_with_missing_column(self, synthetic_data):
        """A lambda scheme that references a missing column should be skipped."""
        analyzer = SubgroupUncertaintyAnalyzer(
            predicted_mean=synthetic_data["predicted_mean"],
            predicted_std=synthetic_data["predicted_std"],
            actual=None,
            subject_metadata=synthetic_data["metadata"],
            subject_ids=synthetic_data["subject_ids"],
        )
        result = analyzer.analyze(
            categorical_schemes={
                "bad_lambda": lambda df: df["no_such_column"],
            },
            continuous_covariates=[],
        )
        assert len(result.subgroup_stats) == 0


# =============================================================================
# Small groups (n < MIN_GROUP_SIZE) excluded from tests
# =============================================================================


class TestSmallGroupExclusion:
    """Groups with n < MIN_GROUP_SIZE are excluded from statistical tests."""

    def test_tiny_groups_excluded_from_kruskal(self):
        """Only 2 subjects in one group — that group should not participate in KW."""
        np.random.seed(99)
        n = 20
        subject_ids = [f"S_{i}" for i in range(n)]

        # Group A: 18 subjects, group B: 2 subjects (below MIN_GROUP_SIZE=3)
        groups = np.array(["A"] * 18 + ["B"] * 2)
        metadata = pd.DataFrame({"subject_id": subject_ids, "grp": groups})

        analyzer = SubgroupUncertaintyAnalyzer(
            predicted_mean=np.random.randn(n),
            predicted_std=np.abs(np.random.randn(n)) + 0.1,
            actual=None,
            subject_metadata=metadata,
            subject_ids=subject_ids,
        )
        result = analyzer.analyze(
            categorical_schemes={"grp": "grp"},
            continuous_covariates=[],
        )

        # Group B has n=2 < 3 so only A is testable — cannot run KW with 1 group
        assert len(result.between_group_tests) == 0

    def test_two_adequate_groups_run_kw(self):
        """Both groups meet MIN_GROUP_SIZE — KW should run."""
        np.random.seed(99)
        n = 20
        subject_ids = [f"S_{i}" for i in range(n)]
        groups = np.array(["A"] * 10 + ["B"] * 10)
        metadata = pd.DataFrame({"subject_id": subject_ids, "grp": groups})

        analyzer = SubgroupUncertaintyAnalyzer(
            predicted_mean=np.random.randn(n),
            predicted_std=np.abs(np.random.randn(n)) + 0.1,
            actual=None,
            subject_metadata=metadata,
            subject_ids=subject_ids,
        )
        result = analyzer.analyze(
            categorical_schemes={"grp": "grp"},
            continuous_covariates=[],
        )
        kw_rows = result.between_group_tests[
            result.between_group_tests["test_name"] == "kruskal_wallis"
        ]
        assert len(kw_rows) == 1


# =============================================================================
# NaN handling in metadata
# =============================================================================


class TestNaNHandling:
    """NaN values in metadata should be dropped for the affected scheme."""

    def test_nan_labels_excluded(self):
        np.random.seed(42)
        n = 20
        subject_ids = [f"S_{i}" for i in range(n)]
        groups = np.array(["A"] * 8 + ["B"] * 8 + [np.nan] * 4, dtype=object)
        metadata = pd.DataFrame({"subject_id": subject_ids, "grp": groups})

        analyzer = SubgroupUncertaintyAnalyzer(
            predicted_mean=np.random.randn(n),
            predicted_std=np.abs(np.random.randn(n)) + 0.1,
            actual=None,
            subject_metadata=metadata,
            subject_ids=subject_ids,
        )
        result = analyzer.analyze(
            categorical_schemes={"grp": "grp"},
            continuous_covariates=[],
        )
        total_n = result.subgroup_stats["n"].sum()
        assert total_n == 16  # 4 NaN subjects excluded

    def test_nan_continuous_covariate(self):
        """NaN in continuous covariate should not crash Spearman."""
        np.random.seed(42)
        n = 30
        subject_ids = [f"S_{i}" for i in range(n)]
        values = np.random.randn(n)
        values[:5] = np.nan

        metadata = pd.DataFrame({"subject_id": subject_ids, "some_cov": values})

        analyzer = SubgroupUncertaintyAnalyzer(
            predicted_mean=np.random.randn(n),
            predicted_std=np.abs(np.random.randn(n)) + 0.1,
            actual=None,
            subject_metadata=metadata,
            subject_ids=subject_ids,
        )
        result = analyzer.analyze(
            categorical_schemes={},
            continuous_covariates=["some_cov"],
        )
        assert len(result.covariate_correlations) > 0
        assert not result.covariate_correlations["spearman_rho"].isna().any()


# =============================================================================
# Save
# =============================================================================


class TestSave:
    """Test that save() creates expected files."""

    def test_save_creates_files(self, tmp_path, synthetic_data):
        analyzer = SubgroupUncertaintyAnalyzer(
            predicted_mean=synthetic_data["predicted_mean"],
            predicted_std=synthetic_data["predicted_std"],
            actual=synthetic_data["actual"],
            subject_metadata=synthetic_data["metadata"],
            subject_ids=synthetic_data["subject_ids"],
        )
        result = analyzer.analyze(
            categorical_schemes={"sex": "msex", "cogdx": "cogdx"},
            continuous_covariates=["cogn_global", "age_death"],
        )
        saved = analyzer.save(result, tmp_path)

        assert "subgroup_stats_csv" in saved
        assert "subgroup_stats_parquet" in saved
        assert "between_group_tests_csv" in saved
        assert "covariate_correlations_csv" in saved

        # Files actually exist
        for path in saved.values():
            assert path.exists()

    def test_save_csv_only(self, tmp_path, synthetic_data):
        analyzer = SubgroupUncertaintyAnalyzer(
            predicted_mean=synthetic_data["predicted_mean"],
            predicted_std=synthetic_data["predicted_std"],
            actual=None,
            subject_metadata=synthetic_data["metadata"],
            subject_ids=synthetic_data["subject_ids"],
        )
        result = analyzer.analyze(
            categorical_schemes={"sex": "msex"},
            continuous_covariates=["cogn_global"],
        )
        saved = analyzer.save(result, tmp_path, formats=["csv"])

        assert "subgroup_stats_csv" in saved
        assert "subgroup_stats_parquet" not in saved

    def test_save_empty_results(self, tmp_path, synthetic_data):
        """Empty result DataFrames should not create files."""
        analyzer = SubgroupUncertaintyAnalyzer(
            predicted_mean=synthetic_data["predicted_mean"],
            predicted_std=synthetic_data["predicted_std"],
            actual=None,
            subject_metadata=synthetic_data["metadata"],
            subject_ids=synthetic_data["subject_ids"],
        )
        result = analyzer.analyze(
            categorical_schemes={"missing": "no_col"},
            continuous_covariates=["no_col"],
        )
        saved = analyzer.save(result, tmp_path)
        # No files should be created for empty DataFrames
        assert len(saved) == 0


# =============================================================================
# Metadata
# =============================================================================


class TestMetadata:
    """Test metadata dict in result."""

    def test_metadata_fields(self, synthetic_data):
        analyzer = SubgroupUncertaintyAnalyzer(
            predicted_mean=synthetic_data["predicted_mean"],
            predicted_std=synthetic_data["predicted_std"],
            actual=synthetic_data["actual"],
            subject_metadata=synthetic_data["metadata"],
            subject_ids=synthetic_data["subject_ids"],
            epistemic_std=synthetic_data["epistemic_std"],
        )
        result = analyzer.analyze(
            categorical_schemes={"sex": "msex"},
            continuous_covariates=["cogn_global"],
        )
        m = result.metadata
        assert m["n_subjects"] == 100
        assert m["has_actual"] is True
        assert m["has_epistemic"] is True
        assert m["has_aleatoric"] is False
        assert "sex" in m["schemes_analyzed"]


# =============================================================================
# Default schemes integration
# =============================================================================


class TestDefaultSchemes:
    """Test that default schemes can be applied to synthetic data that has the columns."""

    def test_default_schemes_with_available_columns(self, synthetic_data):
        """Schemes that reference available columns should work; others should skip."""
        analyzer = SubgroupUncertaintyAnalyzer(
            predicted_mean=synthetic_data["predicted_mean"],
            predicted_std=synthetic_data["predicted_std"],
            actual=synthetic_data["actual"],
            subject_metadata=synthetic_data["metadata"],
            subject_ids=synthetic_data["subject_ids"],
        )
        # Use all defaults (some will be missing and skipped gracefully)
        result = analyzer.analyze()

        # Some schemes should have succeeded
        assert len(result.subgroup_stats) > 0
        schemes = set(result.subgroup_stats["scheme"])
        # These columns exist in synthetic data
        assert "sex" in schemes
        assert "cogdx" in schemes
        assert "braak" in schemes
