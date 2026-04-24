"""Project-wide visual theme for paper-quality figures.

Codifies user-decided conventions:
  - Palettes: tab10 (categorical), Dark2 (paired), viridis (sequential),
    PiYG (diverging)
  - Fonts: Helvetica with DejaVu Sans fallback
  - Spines: top + right hidden by default
  - Tick direction: out
  - Grid: major only
  - Markers: filled circles primary, with white edge for overlapping data
  - Error bars: caps for bars/points, ribbons for curves
  - Save: 600 DPI; both .png and .pdf
  - No in-figure title (caption-only)

Public API:
    apply_theme(style="paper") -> None
        Apply matplotlib rcParams + scienceplots base if available.

    PALETTES: dict[str, list[str] | colormap]
        Project palettes by name.

    BASELINE_COLORS: dict[str, str]
        Stable hex colors per baseline so identity is consistent across figures.

    fmt_axes(ax, *, hide_spines=("top", "right"), tick_dir="out",
             grid_major=True, grid_minor=False) -> None
        Per-axes styling.

    save_fig(fig, path_stem, *, dpi=600, formats=("png", "pdf"),
             bbox_inches="tight") -> list[Path]
        Save fig as <path_stem>.<ext> for each ext; returns list of paths.

    errorbar_caps(ax, x, y, yerr, *, color=None, label=None, **kwargs)
        Caps-style errorbar (bars / points).

    errorbar_ribbon(ax, x, y, yerr, *, color=None, label=None,
                    line_kwargs=None, fill_alpha=0.2)
        Ribbon-style errorbar (curves).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


PALETTES: dict[str, object] = {
    "categorical": list(plt.get_cmap("tab10").colors),
    "categorical_paired": list(plt.get_cmap("Dark2").colors),
    "sequential": plt.get_cmap("viridis"),
    "diverging": plt.get_cmap("PiYG"),
    # Fold colors: first 5 of tab10 — consistent across all per-fold plots.
    "fold_colors": list(plt.get_cmap("tab10").colors)[:5],
}

# Stable per-baseline colors so identity is consistent across all figures.
# Choose from tab10 + Dark2 to maintain visual coherence.
BASELINE_COLORS: dict[str, str] = {
    "ResDec-MHE":             "#1f77b4",   # tab10 blue (canonical)
    "ResDec-MHE (canonical)": "#1f77b4",
    "TabPFN-2.6":             "#d62728",   # tab10 red — primary baseline
    "TabPFN-2.6 standalone":  "#d62728",
    "XGBoost":                "#ff7f0e",   # tab10 orange — strongest classical
    "XGBoost [A]":            "#ff7f0e",
    "XGBoost [A+C+E]":        "#ff7f0e",
    "XGBoost [C]":            "#ffbb78",   # lighter orange for [C] feature set
    "RandomForest":           "#2ca02c",   # tab10 green
    "Ridge":                  "#9467bd",   # tab10 purple
    "ElasticNet":             "#8c564b",   # tab10 brown
    "PLS":                    "#e377c2",   # tab10 pink
    "MixMIL":                 "#17becf",   # tab10 cyan
    "scPhase":                "#7f7f7f",   # tab10 gray
    "CloudPred":              "#bcbd22",   # tab10 olive
    "CloudPred (per-type)":   "#dbdb8d",   # lighter olive
    "GPIO":                   "#1b9e77",   # Dark2 teal
    "Perceiver-IO":           "#7570b3",   # Dark2 purple
}


def apply_theme(style: str = "paper", use_scienceplots: bool = True) -> str:
    """Register matplotlib rcParams for the chosen style.

    Parameters
    ----------
    style
        "paper" (default), "talk" (larger fonts), or "draft" (faster).
    use_scienceplots
        If True and ``scienceplots`` is installed, layer the ``science``
        + ``nature`` styles before our overrides.

    Returns
    -------
    str
        Either ``"scienceplots"`` (if scienceplots was loaded) or
        ``"manual"`` (rcParams only). Useful for logging which path
        produced a figure.
    """
    used_scienceplots = False
    if use_scienceplots:
        try:
            import scienceplots  # noqa: F401  (registers styles via import)
            plt.style.use(["science", "nature"])
            used_scienceplots = True
        except ImportError:
            pass  # Fall through to manual rcParams

    base_size = {"paper": 8, "talk": 12, "draft": 7}.get(style, 8)

    rc = {
        # Fonts: prefer Helvetica; fall back to Arial then DejaVu Sans.
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size":          base_size,
        "axes.titlesize":     base_size + 1,
        "axes.labelsize":     base_size,
        "xtick.labelsize":    base_size - 1,
        "ytick.labelsize":    base_size - 1,
        "legend.fontsize":    base_size - 1,
        "figure.titlesize":   base_size + 2,

        # Math via matplotlib's built-in mathtext (LaTeX-style without LaTeX).
        # text.usetex=False overrides scienceplots' default LaTeX rendering,
        # which doesn't have unicode (∈, ℝ, etc.) bound to glyphs by default
        # and is fragile for arbitrary string content.
        "text.usetex":        False,
        "mathtext.fontset":   "dejavusans",

        # Spines: hide top + right by default. Per-axes can re-enable via fmt_axes.
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.linewidth":     0.8,

        # Ticks: out, with consistent length and width.
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "xtick.major.size":   3.0,
        "ytick.major.size":   3.0,
        "xtick.major.width":  0.8,
        "ytick.major.width":  0.8,
        "xtick.minor.size":   1.5,
        "ytick.minor.size":   1.5,

        # Grid: major only, light gray, behind data.
        "axes.grid":          True,
        "axes.grid.which":    "major",
        "axes.grid.axis":     "both",
        "grid.color":         "#e6e6e6",
        "grid.linewidth":     0.6,
        "grid.alpha":         1.0,
        "axes.axisbelow":     True,

        # Lines + markers.
        "lines.linewidth":    1.4,
        "lines.markersize":   4.5,
        "lines.markeredgewidth": 0.8,

        # Save: high DPI, tight bounding box, transparent disabled (white bg
        # for paper figures).
        "savefig.dpi":        600,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.05,
        "savefig.transparent": False,

        # Figure: white background, sensible defaults. Note we do NOT enable
        # constrained_layout here because save_fig() uses bbox_inches="tight"
        # which conflicts with constrained_layout (matplotlib emits a warning
        # + the layouts can fight). Callers who want constrained_layout in an
        # interactive notebook can set it manually.
        "figure.dpi":         150,
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
        "figure.autolayout":  False,
        "figure.constrained_layout.use": False,

        # Legend: clean frame, no shadow.
        "legend.frameon":     True,
        "legend.framealpha":  0.95,
        "legend.edgecolor":   "#cccccc",
        "legend.fancybox":    False,
    }
    mpl.rcParams.update(rc)
    return "scienceplots" if used_scienceplots else "manual"


def fmt_axes(
    ax,
    *,
    hide_spines: Sequence[str] = ("top", "right"),
    tick_dir: str = "out",
    grid_major: bool = True,
    grid_minor: bool = False,
) -> None:
    """Apply per-axes styling overrides (in case rcParams were modified)."""
    for spine_name in ("top", "right", "bottom", "left"):
        ax.spines[spine_name].set_visible(spine_name not in hide_spines)
    ax.tick_params(direction=tick_dir, which="both")
    if grid_major:
        ax.grid(True, which="major", linewidth=0.6, color="#e6e6e6", zorder=0)
    if grid_minor:
        ax.grid(True, which="minor", linewidth=0.4, color="#f0f0f0", zorder=0)
    if not grid_major and not grid_minor:
        ax.grid(False)
    ax.set_axisbelow(True)


def save_fig(
    fig,
    path_stem: str | Path,
    *,
    dpi: int = 600,
    formats: Iterable[str] = ("png", "pdf"),
    bbox_inches: str | None = "tight",
) -> list[Path]:
    """Save fig as <stem>.<ext> for each ext; return list of written paths."""
    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in formats:
        out = stem.with_suffix(f".{ext}")
        fig.savefig(out, dpi=dpi, bbox_inches=bbox_inches)
        written.append(out)
    return written


def errorbar_caps(
    ax,
    x,
    y,
    yerr,
    *,
    color=None,
    label=None,
    capsize: float = 3.0,
    **kwargs,
):
    """Caps-style error bars; suitable for bars / point estimates."""
    return ax.errorbar(
        x, y, yerr=yerr,
        fmt="o" if "fmt" not in kwargs else kwargs.pop("fmt"),
        color=color,
        ecolor=kwargs.pop("ecolor", "black"),
        elinewidth=kwargs.pop("elinewidth", 0.8),
        capsize=capsize,
        capthick=kwargs.pop("capthick", 0.8),
        label=label,
        zorder=kwargs.pop("zorder", 3),
        markeredgewidth=kwargs.pop("markeredgewidth", 0.8),
        markeredgecolor=kwargs.pop("markeredgecolor", "white"),
        **kwargs,
    )


def errorbar_ribbon(
    ax,
    x,
    y,
    yerr,
    *,
    color=None,
    label=None,
    line_kwargs: dict | None = None,
    fill_alpha: float = 0.2,
):
    """Ribbon-style error bar; suitable for curves / time series."""
    line_kwargs = line_kwargs or {}
    y_arr = np.asarray(y, dtype=float)
    err_arr = np.asarray(yerr, dtype=float)
    line, = ax.plot(x, y_arr, color=color, label=label, **line_kwargs)
    ax.fill_between(
        x,
        y_arr - err_arr,
        y_arr + err_arr,
        color=color or line.get_color(),
        alpha=fill_alpha,
        linewidth=0,
        zorder=line.get_zorder() - 0.5,
    )
    return line


def baseline_color(name: str, default: str = "#777777") -> str:
    """Look up a baseline's stable color; case-insensitive prefix match."""
    if name in BASELINE_COLORS:
        return BASELINE_COLORS[name]
    name_lower = name.lower()
    for k, v in BASELINE_COLORS.items():
        if name_lower.startswith(k.lower()):
            return v
    return default
