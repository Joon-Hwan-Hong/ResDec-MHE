"""Tests for attribution visualization plots (Captum × DE concordance).

Smoke / contract tests for the per-cell-type and aggregate concordance
plot; other functions in ``attribution_plots`` are covered by integration
via the orchestrators (pre-existing gap, not a regression from this file).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.visualization.attribution_plots import plot_captum_de_concordance


@pytest.fixture
def small_attributions():
    """Tiny attribution cube: 20 subjects × 4 CTs × 50 genes."""
    rng = np.random.default_rng(0)
    return rng.normal(size=(20, 4, 50)).astype(np.float32)


@pytest.fixture
def small_de_per_ct():
    """Per-CT DE DataFrames for 4 cell types × 50 shared genes."""
    rng = np.random.default_rng(1)
    genes = [f"gene_{i}" for i in range(50)]
    return [
        pd.DataFrame(
            {
                "gene": genes,
                "p_value": rng.uniform(0.0, 1.0, size=50),
                "log2_fold_change": rng.normal(0, 0.5, size=50),
            },
        )
        for _ in range(4)
    ]


class TestPlotCaptumDeConcordance:
    """Smoke tests for ``plot_captum_de_concordance``."""

    def test_basic_plot(self, small_attributions, small_de_per_ct):
        ct_names = [f"CT_{i}" for i in range(4)]
        gene_names = [f"gene_{i}" for i in range(50)]
        fig = plot_captum_de_concordance(
            small_attributions, small_de_per_ct, ct_names, gene_names,
        )
        assert isinstance(fig, plt.Figure)
        assert len(fig.get_axes()) >= 2
        plt.close(fig)

    def test_save_path(self, tmp_path, small_attributions, small_de_per_ct):
        ct_names = [f"CT_{i}" for i in range(4)]
        gene_names = [f"gene_{i}" for i in range(50)]
        save_path = tmp_path / "concordance"
        fig = plot_captum_de_concordance(
            small_attributions, small_de_per_ct, ct_names, gene_names,
            save_path=save_path,
        )
        assert save_path.with_suffix(".png").exists()
        plt.close(fig)

    def test_ct_count_mismatch_raises(
        self, small_attributions, small_de_per_ct,
    ):
        wrong_names = [f"CT_{i}" for i in range(3)]  # attr has 4
        gene_names = [f"gene_{i}" for i in range(50)]
        with pytest.raises(ValueError, match="n_ct mismatch"):
            plot_captum_de_concordance(
                small_attributions, small_de_per_ct, wrong_names, gene_names,
            )

    def test_de_count_mismatch_raises(self, small_attributions):
        ct_names = [f"CT_{i}" for i in range(4)]
        gene_names = [f"gene_{i}" for i in range(50)]
        short_de = [
            pd.DataFrame({"gene": gene_names, "p_value": [0.5] * 50}),
        ] * 2  # only 2 CTs of DE, 4 expected
        with pytest.raises(ValueError, match="DE count"):
            plot_captum_de_concordance(
                small_attributions, short_de, ct_names, gene_names,
            )

    def test_gene_count_mismatch_raises(
        self, small_attributions, small_de_per_ct,
    ):
        ct_names = [f"CT_{i}" for i in range(4)]
        wrong_genes = [f"gene_{i}" for i in range(10)]  # attr has 50
        with pytest.raises(ValueError, match="n_gene mismatch"):
            plot_captum_de_concordance(
                small_attributions, small_de_per_ct, ct_names, wrong_genes,
            )

    def test_none_entries_skipped(
        self, small_attributions, small_de_per_ct,
    ):
        """A ``None`` DE slot is skipped; plot still works if ≥1 CT has data."""
        de_with_none = list(small_de_per_ct)
        de_with_none[0] = None
        de_with_none[1] = None
        ct_names = [f"CT_{i}" for i in range(4)]
        gene_names = [f"gene_{i}" for i in range(50)]
        fig = plot_captum_de_concordance(
            small_attributions, de_with_none, ct_names, gene_names,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_all_none_raises(self, small_attributions):
        ct_names = [f"CT_{i}" for i in range(4)]
        gene_names = [f"gene_{i}" for i in range(50)]
        with pytest.raises(ValueError, match="no cell types"):
            plot_captum_de_concordance(
                small_attributions, [None] * 4, ct_names, gene_names,
            )


@pytest.fixture(autouse=True)
def cleanup():
    yield
    plt.close("all")
