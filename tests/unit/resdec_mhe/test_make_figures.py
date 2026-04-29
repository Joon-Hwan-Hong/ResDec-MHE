"""Unit tests for scripts/resdec_mhe/interpretability/make_figures.py.

Each make_figX_* function accepts pre-loaded data (DataFrames / dicts) and
returns a ``matplotlib.figure.Figure``; missing inputs raise ``SkipFigure``
so the orchestrator can log a WARNING and skip that figure without
aborting the whole run.

``matplotlib.use("Agg")`` is set at module level so tests do not require
an X display.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # must be set before pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


# Make the script importable via scripts.resdec_mhe.interpretability.
_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from scripts.resdec_mhe.interpretability import make_figures as mod


# ---------------------------------------------------------------------------
# Fixtures: minimal canonical-shaped inputs for each figure.
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_baseline_table() -> pd.DataFrame:
    """3-row baseline table: 1 baseline, 1 ours-canonical, 1 pending ablation."""
    return pd.DataFrame([
        {
            "model": "tabpfn_2_6_standalone",
            "display_name": "TabPFN-2.6 standalone",
            "n_folds": 5,
            "r2_mean": 0.40,
            "r2_std": 0.10,
            "mae_mean": 0.7, "mae_std": 0.04, "rmse_mean": 0.9, "rmse_std": 0.06,
            "pearson_mean": 0.64, "pearson_std": 0.07,
            "spearman_mean": 0.61, "spearman_std": 0.04,
            "source_path": "data/canonical", "notes": "",
        },
        {
            "model": "p5_canonical_seed42",
            "display_name": "ResDec-MHE (canonical, p5_canonical_seed42)",
            "n_folds": 5,
            "r2_mean": 0.4436,
            "r2_std": 0.10,
            "mae_mean": 0.67, "mae_std": 0.05, "rmse_mean": 0.86, "rmse_std": 0.06,
            "pearson_mean": 0.67, "pearson_std": 0.07,
            "spearman_mean": 0.66, "spearman_std": 0.05,
            "source_path": "outputs/canonical/p5_canonical_seed42",
            "notes": "",
        },
        {
            "model": "p5_ablation_topk_4000",
            "display_name": "Ablation: top-k=4000",
            "n_folds": 0,
            "r2_mean": float("nan"),
            "r2_std": float("nan"),
            "mae_mean": float("nan"), "mae_std": float("nan"),
            "rmse_mean": float("nan"), "rmse_std": float("nan"),
            "pearson_mean": float("nan"), "pearson_std": float("nan"),
            "spearman_mean": float("nan"), "spearman_std": float("nan"),
            "source_path": "outputs/canonical/p5_ablation_topk_4000",
            "notes": "pending: ablation not yet complete",
        },
    ])


@pytest.fixture
def mock_resilience_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 60
    y_true = rng.normal(0.0, 1.0, size=n)
    y_pred = y_true + rng.normal(0.0, 0.6, size=n)
    return pd.DataFrame({
        "ROSMAP_IndividualID": [f"R{i:07d}" for i in range(n)],
        "fold": rng.integers(0, 5, size=n),
        "y_true": y_true,
        "y_composite": y_pred,
        "y_tabpfn": y_true + rng.normal(0.0, 0.7, size=n),
        "f1_residual": rng.normal(0.0, 0.5, size=n),
    })


@pytest.fixture
def mock_captum_summary() -> dict:
    return {
        "n_subjects": 516,
        "n_cell_types": 31,
        "n_genes": 4785,
        "top_cell_type_gene_pairs": [
            {"cell_type": "Splatter", "gene": "SCN3B",
             "mean_abs_attribution": 0.0049},
            {"cell_type": "Splatter", "gene": "KCNIP4",
             "mean_abs_attribution": 0.0041},
            {"cell_type": "Microglia", "gene": "APOE",
             "mean_abs_attribution": 0.0038},
            {"cell_type": "Astrocyte", "gene": "GFAP",
             "mean_abs_attribution": 0.0032},
            {"cell_type": "Oligodendrocyte", "gene": "MBP",
             "mean_abs_attribution": 0.0029},
        ],
        "top_global_genes": [
            {"gene": "MT-CO3", "mean_abs_attribution": 0.00033},
            {"gene": "MT-CO2", "mean_abs_attribution": 0.00030},
        ],
        "top_genes_per_cell_type": {
            "Splatter": [
                {"gene": "SCN3B", "mean_abs_attribution": 0.0049},
                {"gene": "KCNIP4", "mean_abs_attribution": 0.0041},
            ],
            "Microglia": [
                {"gene": "APOE", "mean_abs_attribution": 0.0038},
            ],
        },
        "cell_types_ranked_by_total_attribution": [
            {"cell_type": "Splatter", "total_abs_attribution": 1.42},
            {"cell_type": "Microglia", "total_abs_attribution": 0.85},
        ],
    }


@pytest.fixture
def mock_head_analysis() -> dict:
    return {
        "n_heads": 4,
        "n_cell_types": 31,
        "uniform_baseline_per_cell_type": 0.0322,
        "max_entropy_nats": 3.434,
        "head_specialization": [
            {
                "head": 0,
                "shannon_entropy_nats": 3.04,
                "effective_n_cell_types": 17.9,
                "top_3_cell_types": [
                    {"cell_type": "LAMP5-LHX6 and Chandelier",
                     "mean_attention": 0.106},
                    {"cell_type": "Fibroblast", "mean_attention": 0.091},
                    {"cell_type": "Microglia", "mean_attention": 0.075},
                ],
            },
            {
                "head": 1,
                "shannon_entropy_nats": 2.96,
                "effective_n_cell_types": 15.1,
                "top_3_cell_types": [
                    {"cell_type": "Splatter", "mean_attention": 0.123},
                    {"cell_type": "LAMP5-LHX6 and Chandelier",
                     "mean_attention": 0.118},
                    {"cell_type": "Fibroblast", "mean_attention": 0.107},
                ],
            },
            {
                "head": 2,
                "shannon_entropy_nats": 2.99,
                "effective_n_cell_types": 17.0,
                "top_3_cell_types": [
                    {"cell_type": "Fibroblast", "mean_attention": 0.112},
                    {"cell_type": "Vascular", "mean_attention": 0.085},
                    {"cell_type": "Committed oligodendrocyte precursor",
                     "mean_attention": 0.068},
                ],
            },
            {
                "head": 3,
                "shannon_entropy_nats": 2.98,
                "effective_n_cell_types": 15.5,
                "top_3_cell_types": [
                    {"cell_type": "Vascular", "mean_attention": 0.107},
                    {"cell_type": "Miscellaneous", "mean_attention": 0.105},
                    {"cell_type": "Fibroblast", "mean_attention": 0.104},
                ],
            },
        ],
    }


@pytest.fixture
def mock_subgroup_metrics() -> dict:
    return {
        "APOE_e4_0": {"n": 380, "r2": 0.36,
                      "r2_ci": [0.24, 0.43]},
        "APOE_e4_1": {"n": 127, "r2": 0.51,
                      "r2_ci": [0.40, 0.61]},
        "APOE_e4_2": {"n": 8, "r2": -0.04,
                      "r2_ci": [-8.0, 0.57]},
        "msex_0": {"n": 334, "r2": 0.48,
                   "r2_ci": [0.42, 0.53]},
        "msex_1": {"n": 182, "r2": 0.36,
                   "r2_ci": [0.15, 0.51]},
        "age_quartile_Q1": {"n": 129, "r2": 0.44,
                            "r2_ci": [0.29, 0.56]},
        "age_quartile_Q2": {"n": 129, "r2": 0.45,
                            "r2_ci": [0.27, 0.58]},
        "age_quartile_Q3": {"n": 129, "r2": 0.40,
                            "r2_ci": [0.25, 0.50]},
        "age_quartile_Q4": {"n": 129, "r2": 0.41,
                            "r2_ci": [0.22, 0.55]},
        "pathology_quartile_Q1": {"n": 129, "r2": 0.13,
                                  "r2_ci": [-0.20, 0.32]},
        "pathology_quartile_Q2": {"n": 129, "r2": 0.31,
                                  "r2_ci": [0.09, 0.45]},
        "pathology_quartile_Q3": {"n": 129, "r2": 0.40,
                                  "r2_ci": [0.23, 0.51]},
        "pathology_quartile_Q4": {"n": 129, "r2": 0.29,
                                  "r2_ci": [0.16, 0.45]},
    }


@pytest.fixture
def mock_statistical_rigor() -> dict:
    return {
        "calibration_coverage": {
            "coverage_at_0.5": 0.43,
            "coverage_at_0.68": 0.63,
            "coverage_at_0.8": 0.75,
            "coverage_at_0.95": 0.92,
            "mean_sigma": 0.73,
            "mean_abs_residual": 0.67,
            "n": 516,
        }
    }


@pytest.fixture
def mock_calibration_per_subject() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 100
    return pd.DataFrame({
        "abs_residual": np.abs(rng.normal(0, 0.67, n)),
        "sigma_tabpfn": np.abs(rng.normal(0.73, 0.1, n)),
    })


# ---------------------------------------------------------------------------
# Per-figure tests
# ---------------------------------------------------------------------------


def test_make_fig1_ablation_bar_returns_figure(mock_baseline_table):
    fig = mod.make_fig1_ablation_bar(
        table=mock_baseline_table, canonical_r2=0.4436
    )
    assert isinstance(fig, plt.Figure)
    # At least one axes, at least one bar (rectangle) drawn.
    ax = fig.axes[0]
    assert len(ax.patches) >= 1
    plt.close(fig)


def test_make_fig1_ablation_bar_sorts_desc(mock_baseline_table):
    """Bars should be sorted by r2_mean descending (NaN / pending last)."""
    fig = mod.make_fig1_ablation_bar(
        table=mock_baseline_table, canonical_r2=0.4436
    )
    ax = fig.axes[0]
    # Each tick label corresponds to one bar, in draw order
    labels = [t.get_text() for t in ax.get_xticklabels()]
    # canonical (0.4436) should come before tabpfn (0.40)
    canonical_label_idx = next(
        i for i, ll in enumerate(labels) if "canonical" in ll.lower()
    )
    tabpfn_idx = next(
        i for i, ll in enumerate(labels) if "tabpfn" in ll.lower()
    )
    assert canonical_label_idx < tabpfn_idx, (labels, canonical_label_idx, tabpfn_idx)
    plt.close(fig)


def test_make_fig2_resilience_scatter_returns_figure(mock_resilience_df):
    fig = mod.make_fig2_resilience_scatter(df=mock_resilience_df)
    assert isinstance(fig, plt.Figure)
    ax = fig.axes[0]
    # Scatter drawn: at least one PathCollection.
    assert len(ax.collections) >= 1
    plt.close(fig)


def test_make_fig3_celltype_gene_heatmap_returns_figure(mock_captum_summary):
    fig = mod.make_fig3_celltype_gene_heatmap(summary=mock_captum_summary)
    assert isinstance(fig, plt.Figure)
    # At least one imshow image
    ax = fig.axes[0]
    assert len(ax.images) >= 1
    plt.close(fig)


def test_make_fig4_head_specialization_returns_figure(mock_head_analysis):
    fig = mod.make_fig4_head_specialization(
        head_summary=mock_head_analysis,
        splatter_lamp5_corr=-0.16,
    )
    assert isinstance(fig, plt.Figure)
    ax = fig.axes[0]
    # 4 heads * 3 top cell types = at least 12 bar patches in stacked bar
    assert len(ax.patches) >= 4
    plt.close(fig)


def test_make_fig5_subgroup_r2_returns_figure(mock_subgroup_metrics):
    fig = mod.make_fig5_subgroup_r2(
        metrics=mock_subgroup_metrics, canonical_r2=0.4436
    )
    assert isinstance(fig, plt.Figure)
    ax = fig.axes[0]
    # At least one bar per subgroup family represented
    assert len(ax.patches) >= 1
    plt.close(fig)


def test_make_fig6_calibration_returns_figure(
    mock_statistical_rigor, mock_calibration_per_subject
):
    fig = mod.make_fig6_calibration(
        stat_rigor=mock_statistical_rigor,
        per_subject=mock_calibration_per_subject,
    )
    assert isinstance(fig, plt.Figure)
    # Two subplots: residual vs sigma, nominal vs empirical coverage
    assert len(fig.axes) == 2
    plt.close(fig)


# ---------------------------------------------------------------------------
# Missing-input handling
# ---------------------------------------------------------------------------


def test_missing_input_raises_skipfigure_fig1():
    with pytest.raises(mod.SkipFigure):
        mod.make_fig1_ablation_bar(table=None, canonical_r2=0.44)


def test_missing_input_raises_skipfigure_fig1_all_nan(mock_baseline_table):
    """If every ablation has NaN r2, should raise SkipFigure."""
    table = mock_baseline_table.copy()
    table["r2_mean"] = float("nan")
    table["n_folds"] = 0
    with pytest.raises(mod.SkipFigure):
        mod.make_fig1_ablation_bar(table=table, canonical_r2=0.44)


def test_missing_input_raises_skipfigure_fig2():
    with pytest.raises(mod.SkipFigure):
        mod.make_fig2_resilience_scatter(df=None)


def test_missing_input_raises_skipfigure_fig3():
    with pytest.raises(mod.SkipFigure):
        mod.make_fig3_celltype_gene_heatmap(summary=None)


def test_missing_input_raises_skipfigure_fig4():
    with pytest.raises(mod.SkipFigure):
        mod.make_fig4_head_specialization(head_summary=None)


def test_missing_input_raises_skipfigure_fig5():
    with pytest.raises(mod.SkipFigure):
        mod.make_fig5_subgroup_r2(metrics=None, canonical_r2=0.44)


def test_missing_input_raises_skipfigure_fig6():
    with pytest.raises(mod.SkipFigure):
        mod.make_fig6_calibration(stat_rigor=None, per_subject=None)


# ---------------------------------------------------------------------------
# Output saving
# ---------------------------------------------------------------------------


def test_save_drops_pdf_format(tmp_path, mock_baseline_table):
    """mod.save_figure now writes PNG only — PDFs intentionally dropped per
    user pref. Legacy callers passing ``formats=("png", "pdf")`` get PNG-only
    output (the "pdf" entry is silently filtered)."""
    fig = mod.make_fig1_ablation_bar(
        table=mock_baseline_table, canonical_r2=0.4436
    )
    out = mod.save_figure(
        fig, tmp_path, "fig_test", formats=("png", "pdf"), dpi=72
    )
    assert (tmp_path / "fig_test.png").is_file()
    assert not (tmp_path / "fig_test.pdf").is_file()
    assert len(out) == 1
    plt.close(fig)


def test_skipfigure_has_message():
    """SkipFigure error message names the missing input."""
    try:
        mod.make_fig1_ablation_bar(table=None, canonical_r2=0.44)
    except mod.SkipFigure as e:
        assert "table" in str(e).lower() or "baseline" in str(e).lower()
    else:
        pytest.fail("SkipFigure not raised")


# ---------------------------------------------------------------------------
# Label-content regression tests (M8)
# ---------------------------------------------------------------------------


def test_fig5_subgroup_labels_stripped(mock_subgroup_metrics):
    """After C1 fix the APOE/sex/age/pathology prefixes are stripped and
    underscores replaced with spaces in the x-tick labels.

    NB: subgroups with n<10 (default ``min_subgroup_n``) are now omitted from
    the plot entirely (user pref — small-n CIs distort the y-axis range), so
    the e4_2 bar disappears here. To keep the label-stripping test focused
    on the rendering logic, ``min_subgroup_n=0`` is passed so all subgroups
    appear and stripping can be checked across every prefix family.
    """
    fig = mod.make_fig5_subgroup_r2(
        metrics=mock_subgroup_metrics, canonical_r2=0.4436,
        min_subgroup_n=0,
    )
    ax = fig.axes[0]
    # Strip any '†' truncation marker (I6) for label equivalence checks.
    labels = [t.get_text().rstrip("†") for t in ax.get_xticklabels()]
    # Positive: expected stripped forms appear.
    assert "e4 0" in labels
    assert "e4 1" in labels
    assert "e4 2" in labels
    assert "0" in labels and "1" in labels  # msex_0 / msex_1
    assert "Q1" in labels
    assert "Q4" in labels
    # Negative: no raw "APOE_e4_0" / "msex_0" / "age_quartile_Q1" leaked.
    for bad in ("APOE_e4_0", "msex_0", "age_quartile_Q1",
                "pathology_quartile_Q1"):
        assert bad not in labels, (bad, labels)
    plt.close(fig)


def test_fig5_truncated_ci_gets_dagger(mock_subgroup_metrics):
    """I6: APOE_e4_2 has ci_lo=-8.0 (< -1.5 clip) → label carries '†'
    and the footnote about axis range is present.

    Pass ``min_subgroup_n=0`` so the small-n e4_2 subgroup is included
    (default behaviour now omits it). The truncation-marker logic still
    fires for any included subgroup whose CI lower bound is below the
    visual clip.
    """
    fig = mod.make_fig5_subgroup_r2(
        metrics=mock_subgroup_metrics, canonical_r2=0.4436,
        min_subgroup_n=0,
    )
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_xticklabels()]
    # APOE_e4_2 → stripped to "e4 2", with '†' appended.
    assert any(ll.startswith("e4 2") and ll.endswith("†") for ll in labels), labels
    # Footnote is drawn as a figure-level text; check presence.
    footnote_texts = [t.get_text() for t in fig.texts]
    assert any("CI lower bound extends" in t for t in footnote_texts)
    plt.close(fig)


def test_fig5_filters_small_n_subgroups(mock_subgroup_metrics):
    """Default ``min_subgroup_n=10`` filters subgroups with fewer subjects.

    The mock fixture has ``APOE_e4_2`` at n=8, which produces an absurd
    CI and is omitted from the plot under the default. The omitted-list
    footnote MUST list the filtered subgroup so the audience can see what
    was dropped.
    """
    fig = mod.make_fig5_subgroup_r2(
        metrics=mock_subgroup_metrics, canonical_r2=0.4436,
    )
    ax = fig.axes[0]
    labels = [t.get_text().rstrip("†") for t in ax.get_xticklabels()]
    # e4 2 should NOT appear as a tick label under default filtering.
    assert "e4 2" not in labels, labels
    # Footnote names the omitted subgroup (and its n).
    footnote_texts = [t.get_text() for t in fig.texts]
    assert any("APOE_e4_2" in t and "n=8" in t for t in footnote_texts), (
        footnote_texts
    )
    plt.close(fig)


def test_fig1_has_canonical_line(mock_baseline_table):
    """M8: figure 1 draws a horizontal line at the canonical R² value."""
    fig = mod.make_fig1_ablation_bar(
        table=mock_baseline_table, canonical_r2=0.4436,
    )
    found = False
    for ax in fig.axes:
        for line in ax.get_lines():
            ydata = line.get_ydata()
            # axhline lines are horizontal: all y values identical.
            if len(ydata) >= 2 and abs(ydata[0] - 0.4436) < 1e-6:
                found = True
                break
        if found:
            break
    assert found, "No axhline at canonical_r2=0.4436 found in fig1"
    plt.close(fig)


def test_fig2_quadrant_labels_canonical(mock_resilience_df):
    """C2: after fix, fig2 has canonical-resilience quadrant labels.

    - "Resilient" label uses y_true>y_pred semantics (not pathology).
    - "Overestimated" label uses y_pred>y_true semantics.
    - No "high pathology" / "low pathology" phrasing remains.
    """
    fig = mod.make_fig2_resilience_scatter(df=mock_resilience_df)
    ax = fig.axes[0]
    texts = [t.get_text() for t in ax.texts]
    joined = "\n".join(texts)
    assert "Resilient" in joined
    assert "Overestimated" in joined
    assert "y_true > y_pred" in joined
    assert "y_pred > y_true" in joined
    # Negative: the old pathology phrasing must be gone (C2 deviation).
    assert "high pathology" not in joined
    assert "low pathology" not in joined
    plt.close(fig)


def test_fig2_title_says_pooled_r2(mock_resilience_df):
    """M9: fig2 title identifies the R² as pooled (not mean-per-fold)."""
    fig = mod.make_fig2_resilience_scatter(df=mock_resilience_df)
    ax = fig.axes[0]
    assert "pooled R" in ax.get_title(), ax.get_title()
    plt.close(fig)


def test_make_fig7_k_sensitivity_returns_figure():
    """Fig 7: 3-point k-sensitivity line plot with bootstrap CI band."""
    fig = mod.make_fig7_k_sensitivity(
        k_values=[1000, 2000, 4000],
        r2_means=[0.4499, 0.4436, 0.4404],
        r2_stds=[0.079, 0.100, 0.067],
        bootstrap_ci=(0.39, 0.51),
    )
    assert isinstance(fig, plt.Figure)
    ax = fig.axes[0]
    # errorbar draws at least one Line2D (the connecting line); at least
    # one scatter / Line2D for the highlight marker.
    assert len(ax.get_lines()) >= 1
    # Log x-scale per design decision.
    assert ax.get_xscale() == "log"
    # X-tick labels are the k values.
    tick_labels = [t.get_text() for t in ax.get_xticklabels()]
    assert "1000" in tick_labels
    assert "2000" in tick_labels
    assert "4000" in tick_labels
    # Title matches spec.
    assert "feature-count sensitivity" in ax.get_title().lower()
    # Caption mentions the CI bounds.
    caption = "\n".join(t.get_text() for t in fig.texts)
    assert "0.39" in caption and "0.51" in caption
    plt.close(fig)


def test_make_fig7_skipfigure_on_nan():
    """Fig 7: NaN in r2_means or r2_stds raises SkipFigure."""
    with pytest.raises(mod.SkipFigure):
        mod.make_fig7_k_sensitivity(
            k_values=[1000, 2000, 4000],
            r2_means=[0.45, float("nan"), 0.44],
            r2_stds=[0.08, 0.10, 0.07],
        )
    with pytest.raises(mod.SkipFigure):
        mod.make_fig7_k_sensitivity(
            k_values=[1000, 2000, 4000],
            r2_means=[0.45, 0.44, 0.44],
            r2_stds=[0.08, float("nan"), 0.07],
        )
    # Empty inputs also raise.
    with pytest.raises(mod.SkipFigure):
        mod.make_fig7_k_sensitivity(k_values=[], r2_means=[], r2_stds=[])
    # Length-mismatch also raises.
    with pytest.raises(mod.SkipFigure):
        mod.make_fig7_k_sensitivity(
            k_values=[1000, 2000], r2_means=[0.45], r2_stds=[0.08],
        )


def test_fig1_nan_nfolds_does_not_crash():
    """I3: a NaN n_folds entry must not crash label-building."""
    df = pd.DataFrame([
        {
            "model": "p5_canonical_seed42",
            "display_name": "ResDec-MHE canonical",
            "n_folds": 5, "r2_mean": 0.44, "r2_std": 0.10,
            "mae_mean": 0.67, "mae_std": 0.05,
            "rmse_mean": 0.86, "rmse_std": 0.06,
            "pearson_mean": 0.67, "pearson_std": 0.07,
            "spearman_mean": 0.66, "spearman_std": 0.05,
            "source_path": "", "notes": "",
        },
        {
            "model": "some_pending_ablation",
            "display_name": "Pending (NaN n_folds)",
            "n_folds": float("nan"),
            "r2_mean": float("nan"), "r2_std": float("nan"),
            "mae_mean": float("nan"), "mae_std": float("nan"),
            "rmse_mean": float("nan"), "rmse_std": float("nan"),
            "pearson_mean": float("nan"), "pearson_std": float("nan"),
            "spearman_mean": float("nan"), "spearman_std": float("nan"),
            "source_path": "", "notes": "pending",
        },
    ])
    fig = mod.make_fig1_ablation_bar(table=df, canonical_r2=0.4436)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
