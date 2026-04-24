"""Tests for counterfactual visualization plots."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.visualization.counterfactual_plots import (
    plot_counterfactual_ct_aggregate,
    plot_counterfactual_movement,
    plot_counterfactual_top_pairs,
)


class TestPlotCounterfactualMovement:
    def test_basic(self):
        sids = [f"R{i:06d}" for i in range(10)]
        frac = np.linspace(0.1, 0.9, 10)
        regime = ["resilient"] * 5 + ["vulnerable"] * 5
        fig = plot_counterfactual_movement(sids, frac, regime)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_save_path(self, tmp_path):
        save = tmp_path / "cf_mov"
        fig = plot_counterfactual_movement(
            ["R1", "R2"], [0.5, 0.7], ["resilient", "vulnerable"],
            save_path=save,
        )
        assert save.with_suffix(".png").exists()
        plt.close(fig)

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="length mismatch"):
            plot_counterfactual_movement(
                ["a", "b"], [0.1, 0.2, 0.3], ["resilient", "vulnerable"],
            )

    def test_empty(self):
        with pytest.raises(ValueError, match="no subjects"):
            plot_counterfactual_movement([], [], [])


class TestPlotCounterfactualCtAggregate:
    def test_basic(self):
        d = {"Splatter": 433, "Ependymal": 86, "Hippocampal CA4": 73}
        fig = plot_counterfactual_ct_aggregate(d)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_total(self):
        d = {"Splatter": 433, "Ependymal": 86}
        fig = plot_counterfactual_ct_aggregate(d, total=1000)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_save_path(self, tmp_path):
        save = tmp_path / "cf_ct"
        fig = plot_counterfactual_ct_aggregate({"Splatter": 10}, save_path=save)
        assert save.with_suffix(".png").exists()
        plt.close(fig)

    def test_empty(self):
        with pytest.raises(ValueError, match="empty"):
            plot_counterfactual_ct_aggregate({})


class TestPlotCounterfactualTopPairs:
    def test_basic(self):
        d = {("Splatter", "TPH2"): 11, ("Splatter", "PPP1R17"): 10,
             ("Hippocampal CA4", "HMSD"): 8}
        fig = plot_counterfactual_top_pairs(d, top_n=5)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_save_path(self, tmp_path):
        save = tmp_path / "cf_pairs"
        fig = plot_counterfactual_top_pairs(
            {("Splatter", "TPH2"): 11}, save_path=save,
        )
        assert save.with_suffix(".png").exists()
        plt.close(fig)

    def test_empty(self):
        with pytest.raises(ValueError, match="empty"):
            plot_counterfactual_top_pairs({})


@pytest.fixture(autouse=True)
def cleanup():
    yield
    plt.close("all")
