"""
Visualization configuration: colors, styling, and plotting defaults.

Color scheme designed for publication-quality figures with:
- Sequential colormap for attention/importance (white → coral)
- Diverging colormap for resilience signatures (teal ↔ coral)
"""

from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns


# ─────────────────────────────────────────────────────────────────────────────
# Publication Defaults
# ─────────────────────────────────────────────────────────────────────────────

FIGURE_DPI = 600
FIGURE_FORMAT = "png"
DEFAULT_FIGSIZE = (8, 6)


# ─────────────────────────────────────────────────────────────────────────────
# Color Palette
# ─────────────────────────────────────────────────────────────────────────────

# Primary accent colors
ACCENT_TEAL = "#189584"    # Negative / Resilience / Low values
ACCENT_CORAL = "#E76A7B"   # Positive / Vulnerability / High values
ACCENT_PEACH = "#F6A987"   # Neutral / Medium values
WHITE = "#FFFFFF"
LIGHT_TEAL = "#8BCAC2"

# Sequential colormap: White → Peach → Coral (for unidirectional data)
SEQUENTIAL_COLORS = [WHITE, ACCENT_PEACH, ACCENT_CORAL]

# Diverging colormap: Teal → White → Coral (for bidirectional data)
DIVERGING_COLORS = [ACCENT_TEAL, LIGHT_TEAL, WHITE, ACCENT_PEACH, ACCENT_CORAL]


# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib Colormaps
# ─────────────────────────────────────────────────────────────────────────────

def get_sequential_cmap(name: str = "resilience_seq") -> LinearSegmentedColormap:
    """
    Get sequential colormap for attention weights, gene importance, etc.

    White → Peach → Coral
    """
    return LinearSegmentedColormap.from_list(name, SEQUENTIAL_COLORS)


def get_diverging_cmap(name: str = "resilience_div") -> LinearSegmentedColormap:
    """
    Get diverging colormap for resilience signatures, differential attention, etc.

    Teal (negative) → White (zero) → Coral (positive)
    """
    return LinearSegmentedColormap.from_list(name, DIVERGING_COLORS)


def register_colormaps() -> None:
    """Register custom colormaps with matplotlib."""
    try:
        mpl.colormaps.register(cmap=get_sequential_cmap(), name="resilience_seq")
        mpl.colormaps.register(cmap=get_diverging_cmap(), name="resilience_div")
    except ValueError:
        # Already registered
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Plotly Colorscales
# ─────────────────────────────────────────────────────────────────────────────

def get_plotly_sequential_colorscale() -> list[list[float | str]]:
    """Get sequential colorscale for Plotly."""
    return [
        [0.0, WHITE],
        [0.5, ACCENT_PEACH],
        [1.0, ACCENT_CORAL],
    ]


def get_plotly_diverging_colorscale() -> list[list[float | str]]:
    """Get diverging colorscale for Plotly."""
    return [
        [0.00, ACCENT_TEAL],
        [0.25, LIGHT_TEAL],
        [0.50, WHITE],
        [0.75, ACCENT_PEACH],
        [1.00, ACCENT_CORAL],
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Cell Type Colors
# ─────────────────────────────────────────────────────────────────────────────

# Allen ABC 31 supercluster cell types (from MapMyCells)
# Colors grouped by major class for visual consistency
CELL_TYPE_COLORS = {
    # Glial cells (orange/brown tones)
    "Oligodendrocyte": "#8C564B",
    "Astrocyte": "#FF7F0E",
    "Microglia": "#9467BD",
    "Oligodendrocyte precursor": "#C49C94",
    "Committed oligodendrocyte precursor": "#D2691E",
    "Bergmann glia": "#DEB887",

    # Cortical excitatory neurons (warm red/coral tones)
    "Upper-layer intratelencephalic": "#E41A1C",
    "Deep-layer intratelencephalic": "#FF6B6B",
    "Deep-layer corticothalamic and 6b": "#FA8072",
    "Deep-layer near-projecting": "#FF7F50",

    # Cortical inhibitory neurons (cool blue/green tones)
    "CGE interneuron": "#1F77B4",
    "MGE interneuron": "#2CA02C",
    "LAMP5-LHX6 and Chandelier": "#17BECF",
    "Midbrain-derived inhibitory": "#4A90D9",

    # Hippocampal neurons (purple tones)
    "Hippocampal dentate gyrus": "#7B68EE",
    "Hippocampal CA1-3": "#9370DB",
    "Hippocampal CA4": "#BA55D3",

    # Subcortical/other excitatory (yellow/gold tones)
    "Amygdala excitatory": "#FFD700",
    "Thalamic excitatory": "#FFA500",
    "Mammillary body": "#DAA520",

    # Striatal neurons (green tones)
    "Medium spiny neuron": "#228B22",
    "Eccentric medium spiny neuron": "#32CD32",

    # Cerebellar/rhombic lip (teal/cyan tones)
    "Upper rhombic lip": "#20B2AA",
    "Lower rhombic lip": "#48D1CC",
    "Cerebellar inhibitory": "#40E0D0",

    # Vascular/structural cells (gray/pink tones)
    "Vascular": "#E377C2",
    "Fibroblast": "#7F7F7F",
    "Ependymal": "#BC8F8F",
    "Choroid plexus": "#F0E68C",

    # Quality/other categories
    "Miscellaneous": "#BCBD22",
    "Splatter": "#D3D3D3",
}

# Canonical order for consistent plotting
CELL_TYPE_ORDER = [
    # Glial
    "Astrocyte",
    "Oligodendrocyte",
    "Oligodendrocyte precursor",
    "Committed oligodendrocyte precursor",
    "Microglia",
    "Bergmann glia",
    # Cortical excitatory
    "Upper-layer intratelencephalic",
    "Deep-layer intratelencephalic",
    "Deep-layer corticothalamic and 6b",
    "Deep-layer near-projecting",
    # Cortical inhibitory
    "CGE interneuron",
    "MGE interneuron",
    "LAMP5-LHX6 and Chandelier",
    "Midbrain-derived inhibitory",
    # Hippocampal
    "Hippocampal dentate gyrus",
    "Hippocampal CA1-3",
    "Hippocampal CA4",
    # Subcortical excitatory
    "Amygdala excitatory",
    "Thalamic excitatory",
    "Mammillary body",
    # Striatal
    "Medium spiny neuron",
    "Eccentric medium spiny neuron",
    # Cerebellar
    "Upper rhombic lip",
    "Lower rhombic lip",
    "Cerebellar inhibitory",
    # Vascular/structural
    "Vascular",
    "Fibroblast",
    "Ependymal",
    "Choroid plexus",
    # Other
    "Miscellaneous",
    "Splatter",
]


def get_cell_type_color(cell_type: str) -> str:
    """Get color for a cell type, with fallback to gray."""
    return CELL_TYPE_COLORS.get(cell_type, "#808080")


# ─────────────────────────────────────────────────────────────────────────────
# Seaborn Style Configuration
# ─────────────────────────────────────────────────────────────────────────────

def setup_seaborn_style(
    style: str = "whitegrid",
    context: str = "paper",
    font_scale: float = 1.2,
) -> None:
    """
    Configure seaborn style for publication-quality figures.

    Args:
        style: Seaborn style ("whitegrid", "white", "darkgrid", "dark", "ticks")
        context: Seaborn context ("paper", "notebook", "talk", "poster")
        font_scale: Font scale multiplier
    """
    # Register custom colormaps
    register_colormaps()

    # Set seaborn theme
    sns.set_theme(
        style=style,
        context=context,
        font_scale=font_scale,
        rc={
            "savefig.dpi": FIGURE_DPI,
            "figure.figsize": DEFAULT_FIGSIZE,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.3,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "axes.titleweight": "bold",
            "axes.labelweight": "normal",
        }
    )


def setup_matplotlib_defaults() -> None:
    """Set matplotlib defaults for consistent styling."""
    plt.rcParams.update({
        "savefig.dpi": FIGURE_DPI,
        "savefig.format": FIGURE_FORMAT,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "figure.figsize": DEFAULT_FIGSIZE,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 0.8,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Figure Saving Utilities
# ─────────────────────────────────────────────────────────────────────────────

def save_figure(
    fig: plt.Figure,
    path: str,
    dpi: int = FIGURE_DPI,
    format: str = FIGURE_FORMAT,
    transparent: bool = False,
) -> None:
    """
    Save figure with publication-quality settings.

    Args:
        fig: Matplotlib figure
        path: Output path
        dpi: Resolution (default: 600)
        format: Output format (default: "png")
        transparent: Whether background should be transparent
    """
    fig.savefig(
        path,
        dpi=dpi,
        format=format,
        bbox_inches="tight",
        pad_inches=0.1,
        facecolor="white" if not transparent else "none",
        edgecolor="none",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────────────────────────────────────

def get_color_palette(n_colors: int = 10) -> list[str]:
    """Get a color palette suitable for categorical data."""
    if n_colors <= 10:
        return sns.color_palette("Set2", n_colors).as_hex()
    else:
        return sns.color_palette("husl", n_colors).as_hex()


def truncate_colormap(
    cmap: LinearSegmentedColormap,
    minval: float = 0.0,
    maxval: float = 1.0,
    n: int = 256,
) -> LinearSegmentedColormap:
    """
    Truncate a colormap to a subset of its range.

    Useful for adjusting color intensity.
    """
    new_cmap = LinearSegmentedColormap.from_list(
        f"trunc({cmap.name},{minval:.2f},{maxval:.2f})",
        cmap(mpl.colors.Normalize(minval, maxval)(range(n)))
    )
    return new_cmap


# Initialize on import
register_colormaps()
