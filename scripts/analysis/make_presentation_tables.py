"""
Render presentation tables + architecture sketch as PNG drop-ins for the
lab talk slide deck. All outputs land in
outputs/plots/sensitivity/<data-date>/slides/ alongside the existing
figure symlinks.

Tables produced:
    slide03_data_summary_table.png     Data card (ROSMAP, N, regions, genes)
    slide07_baselines_table.png        Methods × metrics (detailed baseline)
    slide09_tier_biology_table.png     Tier 1/2 cell types with published refs
    backup_hpo_config_table.png        HPO7 rank3 hyperparameters (Q&A backup)

Architecture (matplotlib sketch, not publication-ready — starting point
for a PowerPoint/Inkscape rebuild):
    architecture_sketch.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.visualization import (
    ACCENT_CORAL,
    ACCENT_PEACH,
    ACCENT_TEAL,
    save_figure,
    setup_seaborn_style,
)

# Biology green (we don't have this in the palette; define locally)
BIO_GREEN = "#6dbf8a"
HEADER_BG = "#2b3a4a"
HEADER_FG = "#ffffff"
ROW_ALT_BG = "#f2f4f7"


# =============================================================================
# Table rendering helper
# =============================================================================


def render_table(
    rows: list[list[str]],
    header: list[str],
    output_path: Path,
    *,
    title: str | None = None,
    col_widths: list[float] | None = None,
    row_highlights: dict[int, str] | None = None,
    figsize: tuple[float, float] = (12, 6),
    fontsize: int = 11,
) -> None:
    """Render a table as a PNG using matplotlib.

    Args:
        rows: list of row-value lists (each same length as header)
        header: column header labels
        output_path: where to save the PNG
        title: optional figure-level title
        col_widths: optional relative column widths
        row_highlights: optional {row_index: color} for specific row backgrounds
    """
    n_rows = len(rows)
    n_cols = len(header)
    if col_widths is None:
        col_widths = [1.0 / n_cols] * n_cols
    else:
        total = sum(col_widths)
        col_widths = [w / total for w in col_widths]

    row_highlights = row_highlights or {}

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=fontsize + 4, fontweight="bold", y=0.97)

    table = ax.table(
        cellText=rows,
        colLabels=header,
        cellLoc="center",
        colWidths=col_widths,
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, 1.8)

    # Style header row
    for c in range(n_cols):
        cell = table[(0, c)]
        cell.set_facecolor(HEADER_BG)
        cell.set_text_props(color=HEADER_FG, fontweight="bold")
        cell.set_edgecolor("white")

    # Alternate body rows + highlights
    for r in range(n_rows):
        for c in range(n_cols):
            cell = table[(r + 1, c)]
            if r in row_highlights:
                cell.set_facecolor(row_highlights[r])
            elif r % 2 == 0:
                cell.set_facecolor(ROW_ALT_BG)
            else:
                cell.set_facecolor("white")
            cell.set_edgecolor("#d0d5dc")

    plt.tight_layout(rect=[0, 0, 1, 0.95 if title else 1])
    save_figure(fig, str(output_path), dpi=200)
    plt.close(fig)


# =============================================================================
# Tables
# =============================================================================


def table_data_summary(slides_dir: Path) -> None:
    rows = [
        ["Subjects",          "516"],
        ["Nuclei",            "~3.9 M"],
        ["Cell types",        "31"],
        ["Brain regions",     "6 (PFC + 5 others)"],
        ["PFC-only subjects", "452 (87.6 %)"],
        ["Multi-region subjects", "64 (12.4 %)"],
        ["Genes (HVG)",       "4,796"],
        ["Target",            "cogn_global (continuous)"],
        ["CV protocol",       "5-fold, no holdout"],
    ]
    render_table(
        rows,
        header=["Quantity", "Value"],
        output_path=slides_dir / "slide03_data_summary_table.png",
        title="ROSMAP — data at a glance",
        col_widths=[0.45, 0.55],
        figsize=(9, 5.5),
    )


def table_baselines(slides_dir: Path) -> None:
    # R² / Pearson / Spearman (memory values for MIL baselines — not verified on disk)
    # Our model: HPO7 rank03 (docs/results/2026-03-30-hpo7-ablation-interpretability.md)
    rows = [
        ["Our model", "2-branch + pathology attn", "0.304 ± 0.067", "0.556 ± 0.051", "0.481 ± 0.054", "Per cell type"],
        ["Ridge",     "Flat pseudobulk (148k)",    "0.290*",         "—",             "—",             "Per feature"],
        ["MixMIL",    "scVI (30-dim) bag of cells","0.110 ± 0.038*", "0.359 ± 0.072*","0.344 ± 0.037*", "None"],
        ["scPhase",   "Raw cells per cell type",   "−0.059 ± 0.093*","−0.010 ± 0.103*","0.025 ± 0.122*","Per cell type"],
    ]
    render_table(
        rows,
        header=["Method", "Input", "R²", "Pearson r", "Spearman ρ", "Interpretability"],
        output_path=slides_dir / "slide07_baselines_table.png",
        title="Baseline comparison — 516 subjects, 5-fold CV\n(* unverified memory values; see manifest)",
        col_widths=[0.12, 0.22, 0.16, 0.16, 0.16, 0.18],
        row_highlights={0: "#ffe0c2"},  # peach for our model
        figsize=(14, 4.5),
    )


def table_tier_biology(slides_dir: Path) -> None:
    # From docs/results/2026-03-30-hpo7-ablation-interpretability.md sections 3 + 11
    rows = [
        # Tier 1: ↑ attention in resilient subjects
        ["Upper-layer IT",         "T1", "0.065", "+0.465", "−0.708", "1642", "Leng et al., Nature 2021"],
        ["Oligodendrocyte",        "T1", "0.043", "+0.412", "−0.536", "1712", "Bartzokis, Neurobiol Aging 2004"],
        ["MGE interneuron",        "T1", "0.056", "+0.376", "−0.570", "457",  "Palop & Bhatt, Nat Neurosci 2013"],
        ["CGE interneuron",        "T1", "0.042", "+0.361", "−0.495", "489",  "Palop & Bhatt, Nat Neurosci 2013"],
        ["OPC",                    "T1", "0.039", "+0.347", "−0.504", "350",  "— (novel)"],
        ["Deep-layer IT",          "T1", "0.040", "+0.296", "−0.420", "638",  "— (novel)"],
        ["Astrocyte",              "T1", "0.038", "+0.201", "−0.311", "834",  "Liddelow Nature 2017, Cellformer 2025"],
        # Tier 2: ↑ attention with pathology
        ["Vascular",               "T2", "0.051", "−0.354", "+0.444", "64",   "Iadecola, Neuron 2017"],
        ["Miscellaneous",          "T2", "0.062", "−0.286", "+0.469", "55",   "— (data-driven cluster)"],
        ["Fibroblast",             "T2", "0.049", "−0.251", "+0.342", "34",   "Vanlandewijck et al., Nature 2018"],
        ["LAMP5-LHX6/Chandelier",  "T2", "0.069", "−0.159", "+0.305", "90",   "— (novel)"],
        ["Hippocampal CA1-3",      "T2", "0.027", "−0.165", "+0.240", "65",   "Gómez-Isla et al., J Neurosci 1996"],
    ]
    render_table(
        rows,
        header=["Cell type", "Tier", "Mean attn", "r(attn, cogn)", "r(attn, gpath)", "Mean cells", "Prior work"],
        output_path=slides_dir / "slide09_tier_biology_table.png",
        title="Attention-implicated cell types with published corroboration\n"
              "Tier 1 = ↑ attention in resilient subjects · Tier 2 = ↑ attention with pathology",
        col_widths=[0.17, 0.05, 0.09, 0.11, 0.11, 0.10, 0.37],
        row_highlights={
            **{i: "#e0f4eb" for i in range(7)},                 # T1 green tint
            **{i: "#fde3e1" for i in range(7, 12)},             # T2 red tint
        },
        figsize=(16, 7),
        fontsize=10,
    )


def table_hpo_config(slides_dir: Path) -> None:
    # From docs/results/2026-03-30-hpo7-ablation-interpretability.md §4
    rows = [
        ["Learning rate (encoders)", "1.5 × 10⁻³"],
        ["Learning rate (guide)",    "4.4 × 10⁻³"],
        ["Dropout",                  "0.234"],
        ["Weight decay",             "5.24 × 10⁻⁶"],
        ["β (Beta-NLL)",             "0.422"],
        ["τ_min (temperature floor)","1.808"],
        ["Anneal epochs",            "18"],
        ["Gene-gate temperature",    "0.936"],
        ["Fusion",                   "concat_normalized"],
        ["Embedding dim",            "64"],
        ["HGT layers",               "4"],
        ["Attention heads",          "4"],
        ["Min epochs",               "23"],
        ["Early-stop patience",      "15"],
    ]
    render_table(
        rows,
        header=["Hyperparameter", "Value"],
        output_path=slides_dir / "backup_hpo_config_table.png",
        title="HPO7 rank 3 (production config) — hyperparameters",
        col_widths=[0.55, 0.45],
        figsize=(8, 7),
        fontsize=10,
    )


# =============================================================================
# Architecture sketch (matplotlib starting point)
# =============================================================================


def architecture_sketch(slides_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")

    def box(x, y, w, h, text, color, edge="#333", fontsize=11, fontweight="normal", text_color="#222"):
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.3,rounding_size=1.2",
            linewidth=1.2, edgecolor=edge, facecolor=color, alpha=0.95,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, fontweight=fontweight, color=text_color)

    def arrow(x1, y1, x2, y2, color="#444", lw=1.5, style="->"):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle=style, mutation_scale=18,
            linewidth=lw, color=color,
        ))

    # Stage labels (top row)
    for sx, label in [(7, "Biology inputs"), (32, "Two encoders"),
                      (59, "Pathology re-weighting"), (86, "Uncertainty output")]:
        ax.text(sx, 47.5, label, ha="center", va="center",
                fontsize=11, fontweight="bold", color="#555")

    # ------- Stage 1: inputs (green) -------
    box(2, 30, 17, 8,
        "Pseudobulk\n(31 cell types × 4,796 genes)",
        BIO_GREEN, fontsize=10)
    box(2, 12, 17, 8,
        "Cell-cell signaling graph\n(LIANA ligand–receptor)",
        BIO_GREEN, fontsize=10)
    ax.text(10.5, 6.5, "Subject i", ha="center", va="center",
            fontsize=11, fontweight="bold", color="#333")

    # ------- Stage 2: encoders (blue/teal) -------
    box(27, 30, 18, 8,
        "Cell-type attention\nencoder",
        ACCENT_TEAL, fontsize=11)
    box(27, 12, 18, 8,
        "Graph network\n(over signaling edges)",
        ACCENT_TEAL, fontsize=11)

    # Intermediate label
    ax.text(48.5, 25, "→ cell-type\nembeddings\n(31 × 64)",
            ha="center", va="center", fontsize=9, style="italic", color="#555")

    # ------- Stage 3: pathology re-weighting (peach — the key layer) -------
    box(54, 17, 23, 16,
        "Pathology-aware\ncell-type\nre-weighting",
        ACCENT_PEACH, edge="#c76a2d", fontsize=13, fontweight="bold")
    # Pathology side input
    ax.text(65, 6, "gpath, amyl, tau", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#333",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff6d6", edgecolor="#c5a200", linewidth=1))
    arrow(65, 9.2, 65, 16.5, color="#c5a200", lw=2)

    # Callout
    ax.annotate(
        "ablation\n→ R² drops from\n0.30 to 0.04",
        xy=(65.5, 33), xytext=(80, 43),
        fontsize=9, color="#c76a2d", fontweight="bold", ha="center",
        arrowprops=dict(arrowstyle="->", color="#c76a2d", lw=1.2),
    )

    # ------- Stage 4: Bayesian head + output (coral) -------
    box(82, 22, 16, 9,
        "Bayesian\nregression head",
        ACCENT_CORAL, fontsize=11)

    # Gaussian icon + output label
    from numpy import linspace, exp
    xs = linspace(-2, 2, 80)
    ys = exp(-0.5 * xs ** 2)
    xs_plot = 82 + (xs + 2) * 4   # 82..98
    ys_plot = 12 + ys * 5         # 12..17
    ax.plot(xs_plot, ys_plot, color=ACCENT_CORAL, linewidth=2.2)
    ax.fill_between(xs_plot, 12, ys_plot, color=ACCENT_CORAL, alpha=0.2)
    ax.text(90, 9, "predicted cogn_global\n± uncertainty",
            ha="center", va="center", fontsize=10, fontweight="bold", color="#333")

    # ------- Arrows between stages -------
    # Stage 1 → Stage 2
    arrow(19.5, 34, 26.5, 34)
    arrow(19.5, 16, 26.5, 16)
    # Stage 2 → Stage 3 (converge into pathology box)
    arrow(45.5, 34, 54, 29)
    arrow(45.5, 16, 54, 22)
    # Stage 3 → Stage 4
    arrow(77.5, 25, 82, 26)

    # Title + source footer
    fig.suptitle(
        "Model architecture — two biological views, combined and re-weighted by pathology",
        fontsize=14, fontweight="bold", y=0.98,
    )
    ax.text(
        50, -2,
        "matplotlib draft sketch · rebuild in PowerPoint/Inkscape for slide-quality typography",
        ha="center", va="center", fontsize=8, color="#888", style="italic",
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    save_figure(fig, str(slides_dir / "architecture_sketch.png"), dpi=200)
    plt.close(fig)


# =============================================================================
# README update with markdown table blocks
# =============================================================================


MARKDOWN_TABLES = """

## Markdown tables (copy into PowerPoint if you want native table styling)

### Data summary (slide 3)

| Quantity | Value |
|---|---|
| Subjects | 516 |
| Nuclei | ~3.9 M |
| Cell types | 31 |
| Brain regions | 6 (PFC + 5 others) |
| PFC-only subjects | 452 (87.6 %) |
| Multi-region subjects | 64 (12.4 %) |
| Genes (HVG) | 4,796 |
| Target | cogn_global (continuous) |
| CV protocol | 5-fold, no holdout |

### Baselines (slide 7)

Asterisks mark unverified memory values (see manifest warning).

| Method | Input | R² | Pearson r | Spearman ρ | Interpretability |
|---|---|---|---|---|---|
| **Our model** | 2-branch + pathology attn | **0.304 ± 0.067** | **0.556 ± 0.051** | **0.481 ± 0.054** | Per cell type |
| Ridge | Flat pseudobulk (148k) | 0.290* | — | — | Per feature |
| MixMIL | scVI (30-dim) bag of cells | 0.110 ± 0.038* | 0.359 ± 0.072* | 0.344 ± 0.037* | None |
| scPhase | Raw cells per cell type | −0.059 ± 0.093* | −0.010 ± 0.103* | 0.025 ± 0.122* | Per cell type |

### Cell-type tiers with published corroboration (slide 9)

Tier 1 = ↑ attention in resilient subjects · Tier 2 = ↑ attention with pathology

| Cell type | Tier | Mean attn | r(attn, cogn) | r(attn, gpath) | Mean cells | Prior work |
|---|---|---|---|---|---|---|
| Upper-layer IT | T1 | 0.065 | +0.465 | −0.708 | 1642 | Leng et al., Nature 2021 |
| Oligodendrocyte | T1 | 0.043 | +0.412 | −0.536 | 1712 | Bartzokis, Neurobiol Aging 2004 |
| MGE interneuron | T1 | 0.056 | +0.376 | −0.570 | 457 | Palop & Bhatt, Nat Neurosci 2013 |
| CGE interneuron | T1 | 0.042 | +0.361 | −0.495 | 489 | Palop & Bhatt, Nat Neurosci 2013 |
| OPC | T1 | 0.039 | +0.347 | −0.504 | 350 | — (novel) |
| Deep-layer IT | T1 | 0.040 | +0.296 | −0.420 | 638 | — (novel) |
| Astrocyte | T1 | 0.038 | +0.201 | −0.311 | 834 | Liddelow Nature 2017; Cellformer 2025 |
| Vascular | T2 | 0.051 | −0.354 | +0.444 | 64 | Iadecola, Neuron 2017 |
| Miscellaneous | T2 | 0.062 | −0.286 | +0.469 | 55 | — (data-driven cluster) |
| Fibroblast | T2 | 0.049 | −0.251 | +0.342 | 34 | Vanlandewijck et al., Nature 2018 |
| LAMP5-LHX6/Chandelier | T2 | 0.069 | −0.159 | +0.305 | 90 | — (novel) |
| Hippocampal CA1-3 | T2 | 0.027 | −0.165 | +0.240 | 65 | Gómez-Isla et al., J Neurosci 1996 |

### HPO7 rank 3 config (backup / Q&A)

| Hyperparameter | Value |
|---|---|
| Learning rate (encoders) | 1.5 × 10⁻³ |
| Learning rate (guide) | 4.4 × 10⁻³ |
| Dropout | 0.234 |
| Weight decay | 5.24 × 10⁻⁶ |
| β (Beta-NLL) | 0.422 |
| τ_min (temperature floor) | 1.808 |
| Anneal epochs | 18 |
| Gene-gate temperature | 0.936 |
| Fusion | concat_normalized |
| Embedding dim | 64 |
| HGT layers | 4 |
| Attention heads | 4 |
| Min epochs | 23 |
| Early-stop patience | 15 |
"""


def append_markdown_to_readme(slides_dir: Path) -> None:
    readme = slides_dir / "README.md"
    if not readme.exists():
        return
    current = readme.read_text()
    marker = "## Markdown tables"
    if marker in current:
        # Already appended — rewrite section to keep latest
        current = current.split(marker)[0].rstrip() + "\n"
    readme.write_text(current + MARKDOWN_TABLES)


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slides-dir",
                        default="outputs/plots/sensitivity/2026-03-30/slides",
                        type=str)
    args = parser.parse_args()

    setup_seaborn_style()
    slides_dir = Path(args.slides_dir).resolve()
    slides_dir.mkdir(parents=True, exist_ok=True)

    print(f"Rendering presentation assets into {slides_dir}")
    architecture_sketch(slides_dir)
    print("  architecture_sketch.png")
    table_data_summary(slides_dir)
    print("  slide03_data_summary_table.png")
    table_baselines(slides_dir)
    print("  slide07_baselines_table.png")
    table_tier_biology(slides_dir)
    print("  slide09_tier_biology_table.png")
    table_hpo_config(slides_dir)
    print("  backup_hpo_config_table.png")
    append_markdown_to_readme(slides_dir)
    print("  README.md updated with markdown tables")


if __name__ == "__main__":
    main()
