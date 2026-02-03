"""Tests for visualization configuration module."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for testing

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pytest

from src.visualization.config import (
    FIGURE_DPI,
    FIGURE_FORMAT,
    ACCENT_TEAL,
    ACCENT_CORAL,
    CELL_TYPE_COLORS,
    EDGE_TYPE_COLORS,
    get_sequential_cmap,
    get_diverging_cmap,
    get_cell_type_color,
    get_edge_type_color,
    get_edge_type_display_name,
    setup_seaborn_style,
    setup_matplotlib_defaults,
    save_figure,
    get_color_palette,
    register_colormaps,
    validate_cell_type_colors,
)


# =============================================================================
# Constants Tests
# =============================================================================


class TestConfigConstants:
    """Test configuration constants."""

    def test_figure_dpi(self):
        """Test FIGURE_DPI is publication-quality."""
        assert FIGURE_DPI >= 300  # Minimum for print
        assert FIGURE_DPI == 600  # Expected value

    def test_figure_format(self):
        """Test FIGURE_FORMAT is valid."""
        assert FIGURE_FORMAT in ["png", "pdf", "svg", "eps"]

    def test_accent_colors_are_hex(self):
        """Test accent colors are valid hex."""
        assert ACCENT_TEAL.startswith("#")
        assert len(ACCENT_TEAL) == 7
        assert ACCENT_CORAL.startswith("#")
        assert len(ACCENT_CORAL) == 7

    def test_cell_type_colors_not_empty(self):
        """Test cell type colors dict is populated."""
        assert len(CELL_TYPE_COLORS) > 0
        assert all(c.startswith("#") for c in CELL_TYPE_COLORS.values())

    def test_edge_type_colors_not_empty(self):
        """Test edge type colors dict is populated."""
        assert len(EDGE_TYPE_COLORS) > 0
        assert all(c.startswith("#") for c in EDGE_TYPE_COLORS.values())


# =============================================================================
# Colormap Tests
# =============================================================================


class TestColormaps:
    """Test colormap creation functions."""

    def test_sequential_cmap_type(self):
        """Test sequential colormap returns correct type."""
        cmap = get_sequential_cmap()
        assert isinstance(cmap, LinearSegmentedColormap)

    def test_sequential_cmap_name(self):
        """Test sequential colormap default name."""
        cmap = get_sequential_cmap()
        assert cmap.name == "resilience_seq"

    def test_sequential_cmap_custom_name(self):
        """Test sequential colormap custom name."""
        cmap = get_sequential_cmap(name="custom_seq")
        assert cmap.name == "custom_seq"

    def test_diverging_cmap_type(self):
        """Test diverging colormap returns correct type."""
        cmap = get_diverging_cmap()
        assert isinstance(cmap, LinearSegmentedColormap)

    def test_diverging_cmap_name(self):
        """Test diverging colormap default name."""
        cmap = get_diverging_cmap()
        assert cmap.name == "resilience_div"

    def test_diverging_cmap_custom_name(self):
        """Test diverging colormap custom name."""
        cmap = get_diverging_cmap(name="custom_div")
        assert cmap.name == "custom_div"

    def test_sequential_cmap_values(self):
        """Test sequential colormap produces valid values."""
        cmap = get_sequential_cmap()
        # Test at endpoints and middle
        for val in [0.0, 0.5, 1.0]:
            rgba = cmap(val)
            assert len(rgba) == 4
            assert all(0 <= c <= 1 for c in rgba)

    def test_diverging_cmap_values(self):
        """Test diverging colormap produces valid values."""
        cmap = get_diverging_cmap()
        for val in [0.0, 0.25, 0.5, 0.75, 1.0]:
            rgba = cmap(val)
            assert len(rgba) == 4
            assert all(0 <= c <= 1 for c in rgba)

    def test_register_colormaps(self):
        """Test colormap registration doesn't raise."""
        # Should not raise even if already registered
        register_colormaps()
        register_colormaps()  # Call twice to test idempotency


# =============================================================================
# Color Getter Tests
# =============================================================================


class TestColorGetters:
    """Test color getter functions."""

    def test_get_cell_type_color_known(self):
        """Test getting color for known cell type."""
        color = get_cell_type_color("Astrocyte")
        assert color.startswith("#")
        assert color == CELL_TYPE_COLORS["Astrocyte"]

    def test_get_cell_type_color_unknown(self):
        """Test fallback for unknown cell type."""
        color = get_cell_type_color("UnknownCellType123")
        assert color == "#808080"  # Gray fallback

    def test_get_edge_type_color_known(self):
        """Test getting color for known edge type."""
        color = get_edge_type_color("Secreted_Signaling")
        assert color.startswith("#")
        assert color == EDGE_TYPE_COLORS["Secreted_Signaling"]

    def test_get_edge_type_color_unknown(self):
        """Test fallback for unknown edge type."""
        color = get_edge_type_color("UnknownEdgeType123")
        assert color == "#808080"  # Gray fallback

    def test_get_edge_type_display_name_known(self):
        """Test getting display name for known edge type."""
        display = get_edge_type_display_name("Secreted_Signaling")
        # Should return human-readable name
        assert isinstance(display, str)

    def test_get_edge_type_display_name_unknown(self):
        """Test fallback for unknown edge type."""
        display = get_edge_type_display_name("Unknown_Type")
        assert display == "Unknown_Type"  # Returns input as fallback


# =============================================================================
# Style Setup Tests
# =============================================================================


class TestStyleSetup:
    """Test style setup functions."""

    def test_setup_seaborn_style_runs(self):
        """Test setup_seaborn_style doesn't raise."""
        setup_seaborn_style()

    def test_setup_seaborn_style_custom(self):
        """Test setup_seaborn_style with custom params."""
        setup_seaborn_style(
            style="white",
            context="talk",
            font_scale=1.5,
        )

    def test_setup_matplotlib_defaults_runs(self):
        """Test setup_matplotlib_defaults doesn't raise."""
        setup_matplotlib_defaults()

    def test_matplotlib_defaults_applied(self):
        """Test that defaults are actually applied."""
        setup_matplotlib_defaults()
        assert plt.rcParams["savefig.dpi"] == FIGURE_DPI


# =============================================================================
# Figure Saving Tests
# =============================================================================


class TestSaveFigure:
    """Test figure saving function."""

    def test_save_figure_png(self, tmp_path):
        """Test saving figure as PNG."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        path = tmp_path / "test_figure.png"
        save_figure(fig, str(path))

        assert path.exists()
        plt.close(fig)

    def test_save_figure_custom_dpi(self, tmp_path):
        """Test saving figure with custom DPI."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        path = tmp_path / "test_figure_dpi.png"
        save_figure(fig, str(path), dpi=300)

        assert path.exists()
        plt.close(fig)

    def test_save_figure_custom_format(self, tmp_path):
        """Test saving figure with custom format."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        path = tmp_path / "test_figure.pdf"
        save_figure(fig, str(path), format="pdf")

        assert path.exists()
        plt.close(fig)

    def test_save_figure_transparent(self, tmp_path):
        """Test saving figure with transparent background."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        path = tmp_path / "test_figure_transparent.png"
        save_figure(fig, str(path), transparent=True)

        assert path.exists()
        plt.close(fig)


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestUtilityFunctions:
    """Test utility functions."""

    def test_get_color_palette_default(self):
        """Test getting default color palette."""
        palette = get_color_palette()
        assert len(palette) == 10
        assert all(c.startswith("#") for c in palette)

    def test_get_color_palette_custom_size(self):
        """Test getting custom size palette."""
        palette = get_color_palette(n_colors=5)
        assert len(palette) == 5

    def test_get_color_palette_large(self):
        """Test getting large palette (uses husl)."""
        palette = get_color_palette(n_colors=15)
        assert len(palette) == 15
        assert all(c.startswith("#") for c in palette)

    def test_validate_cell_type_colors(self):
        """Test cell type color validation."""
        missing = validate_cell_type_colors()
        # Function should return list (possibly empty)
        assert isinstance(missing, list)


# =============================================================================
# Integration Tests
# =============================================================================


class TestConfigIntegration:
    """Integration tests for config module."""

    def test_full_workflow(self, tmp_path):
        """Test complete workflow from setup to save."""
        # Setup styles
        setup_seaborn_style()
        setup_matplotlib_defaults()

        # Create figure with custom colormap
        fig, ax = plt.subplots()
        data = np.random.rand(10, 10)
        im = ax.imshow(data, cmap=get_sequential_cmap())
        plt.colorbar(im, ax=ax)

        # Save
        path = tmp_path / "workflow_test.png"
        save_figure(fig, str(path))

        assert path.exists()
        plt.close(fig)

    def test_colors_in_plot(self, tmp_path):
        """Test using cell type colors in a plot."""
        setup_seaborn_style()

        fig, ax = plt.subplots()
        cell_types = ["Astrocyte", "Microglia", "Oligodendrocyte"]
        values = [0.3, 0.4, 0.3]
        colors = [get_cell_type_color(ct) for ct in cell_types]

        ax.bar(cell_types, values, color=colors)

        path = tmp_path / "colored_bar.png"
        save_figure(fig, str(path))

        assert path.exists()
        plt.close(fig)


# =============================================================================
# Cleanup
# =============================================================================


@pytest.fixture(autouse=True)
def cleanup():
    """Cleanup matplotlib figures after each test."""
    yield
    plt.close("all")
