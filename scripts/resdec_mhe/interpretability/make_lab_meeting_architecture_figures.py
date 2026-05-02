"""Orchestrator: render the 2 architecture-diagram lab-meeting figures.

Produces two architecture-diagram figures used in the 2026-04-29 lab meeting
that prior orchestrators do NOT cover:

  Slot 3.4 — fig_slot3_4_fusion_stack.{png,pdf}
      Single-panel technical block diagram of the END of the architecture
      pipeline: fusion + pathology-stratified attention + composite-with-
      TabPFN-residual stack.  Tensor shapes annotated on data-flow arrows;
      colour-coded per-module rectangles using ACCENT_TEAL / ACCENT_CORAL +
      categorical palette.  Figsize (10, 8).

  Slot 3.1-3.3 — fig_slot3_full_architecture_hybrid.{png,pdf}
      Single-panel hybrid bio-iconographic full architecture.  Cell-type
      branch (left) + cell-cell-interaction branch (right) -> fusion ->
      pathology attention -> composite prediction.  Biology-leaning
      labels: "Cell-by-cell pooling" (instead of ISAB / Set Transformer),
      "Cell-cell signaling" (instead of HGT), etc.  Abstract circle/graph
      iconography (no real biology icons).  Figsize (14, 10).

Default I/O paths are resolved from the project worktree root and can be
overridden via env-vars or argparse per project conventions.

Usage:
    uv run python scripts/resdec_mhe/interpretability/make_lab_meeting_architecture_figures.py
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # must precede pyplot import

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from src.visualization.config import ACCENT_CORAL, ACCENT_TEAL  # noqa: E402
from src.visualization.theme import (  # noqa: E402
    PALETTES,
    apply_theme,
    fmt_axes,
    save_fig,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# Defaults (env-var / argparse driven per project rules)
# ===========================================================================

_DEFAULT_OUT_DIR = os.environ.get(
    "LAB_MEETING_ARCH_OUT_DIR",
    str(_WORKTREE_ROOT
        / "outputs/canonical/interpretability/figures/lab_meeting"),
)


# Canonical model dimensions (from configs/default.yaml — d_embed=64,
# d_fused=64, d_cond=64).  Hardcoded here to keep the figure self-contained
# and avoid an expensive OmegaConf load just for two annotations.
_D_EMBED = 64
_D_FUSED = 64
_D_COND = 64
_N_CT = 31  # number of cell-type tokens output by both branches

# Module palette — mapped to user-requested colour roles.
_COLOR_CELL_BRANCH = ACCENT_TEAL          # cell-type expression branch
_COLOR_HGT_BRANCH = ACCENT_CORAL          # cell-cell signaling branch
_COLOR_FUSION = PALETTES["categorical"][2]   # green-ish (3rd)
_COLOR_PATHOLOGY = PALETTES["categorical"][4]  # purple-ish (5th)
_COLOR_HEAD = PALETTES["categorical"][1]     # orange (head + composite)
_COLOR_DATA = "#f5f5f5"                      # light gray for data plates
_COLOR_TABPFN = "#d62728"                    # tab10 red — TabPFN identity


# ===========================================================================
# Helpers (rounded box, arrow, tensor-shape annotation)
# ===========================================================================


def _block(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    fc: str = "#dceaf5",
    ec: str = "#222222",
    fontsize: float = 8,
    fontweight: str = "normal",
    text_color: str = "#111111",
):
    """Draw a rounded box with centered text."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        linewidth=1.0, edgecolor=ec, facecolor=fc, zorder=2,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight,
            color=text_color, wrap=True, zorder=3)


def _arrow(
    ax,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    color: str = "#444444",
    linewidth: float = 1.1,
    label: str | None = None,
    label_offset: tuple[float, float] = (0.0, 0.012),
    label_fontsize: float = 7,
    label_color: str = "#444444",
    connectionstyle: str = "arc3,rad=0",
):
    """Draw an arrow with optional shape-annotation label at midpoint."""
    arr = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="-|>",
        mutation_scale=11,
        color=color,
        linewidth=linewidth,
        zorder=2,
        connectionstyle=connectionstyle,
    )
    ax.add_patch(arr)
    if label is not None:
        mx, my = (x0 + x1) / 2 + label_offset[0], (y0 + y1) / 2 + label_offset[1]
        ax.text(mx, my, label,
                ha="center", va="center",
                fontsize=label_fontsize, color=label_color,
                fontstyle="italic", zorder=4,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.85))


# ===========================================================================
# Slot 3.4 — fusion-stack technical block diagram
# ===========================================================================


def build_slot3_4_fusion_stack() -> plt.Figure:
    """Block diagram of fusion + pathology-stratified attention + composite.

    Layout: top-to-bottom flow.
      Row 1 (top): two branch-output tensors (cell + HGT) -> fusion box
      Row 2 (mid): fused tensor + pathology side path -> attention -> head
      Row 3 (bot): TabPFN residual stack -> composite prediction
    """
    fig, ax = plt.subplots(figsize=(10.0, 8.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    ax.axis("off")

    # -----------------------------------------------------------------
    # Row 1 — Branch outputs + fusion (top)
    # -----------------------------------------------------------------
    # Cell branch tensor (top-left) and HGT branch tensor (top-right).
    cell_box = (0.04, 0.84, 0.26, 0.10)  # x, y, w, h
    hgt_box = (0.70, 0.84, 0.26, 0.10)
    _block(ax, *cell_box,
           f"Cell branch output\n[B, {_N_CT}, {_D_EMBED}]",
           fc=_COLOR_CELL_BRANCH, ec="#0f6e62",
           fontsize=9, fontweight="bold", text_color="white")
    _block(ax, *hgt_box,
           f"HGT branch output\n[B, {_N_CT}, {_D_EMBED}]",
           fc=_COLOR_HGT_BRANCH, ec="#a23e4f",
           fontsize=9, fontweight="bold", text_color="white")

    # Fusion box (centred, slightly wider).
    fuse_box = (0.30, 0.66, 0.40, 0.12)
    _block(ax, *fuse_box,
           "FusionLayer\n(concat / cross-attention)",
           fc=_COLOR_FUSION, ec="#1f6e1f",
           fontsize=10, fontweight="bold", text_color="white")

    # Arrows: cell branch -> fusion (down + right diagonal)
    _arrow(ax,
           cell_box[0] + cell_box[2] / 2, cell_box[1],
           fuse_box[0] + fuse_box[2] * 0.30, fuse_box[1] + fuse_box[3],
           label=f"[B, {_N_CT}, {_D_EMBED}]",
           label_offset=(-0.045, 0.0))
    # HGT branch -> fusion (down + left diagonal)
    _arrow(ax,
           hgt_box[0] + hgt_box[2] / 2, hgt_box[1],
           fuse_box[0] + fuse_box[2] * 0.70, fuse_box[1] + fuse_box[3],
           label=f"[B, {_N_CT}, {_D_EMBED}]",
           label_offset=(0.045, 0.0))

    # -----------------------------------------------------------------
    # Row 2 — fused tensor + pathology side path -> attention -> head
    # -----------------------------------------------------------------
    # Pathology side path (right column, parallel to fusion).
    pathology_input_box = (0.72, 0.50, 0.24, 0.07)
    pathology_enc_box = (0.72, 0.36, 0.24, 0.10)
    _block(ax, *pathology_input_box,
           "Pathology covariates\n[B, 3]",
           fc="#fff2cc", ec="#8c6d18", fontsize=8)
    _block(ax, *pathology_enc_box,
           f"PathologyEncoder\n-> path_emb [B, {_D_COND}]",
           fc=_COLOR_PATHOLOGY, ec="#3d2a55",
           fontsize=9, fontweight="bold", text_color="white")
    # Pathology arrow (within side path, top-down).
    _arrow(ax,
           pathology_input_box[0] + pathology_input_box[2] / 2,
           pathology_input_box[1],
           pathology_enc_box[0] + pathology_enc_box[2] / 2,
           pathology_enc_box[1] + pathology_enc_box[3])

    # Pathology-stratified attention (centred, beneath fusion).
    attn_box = (0.18, 0.36, 0.52, 0.10)
    _block(ax, *attn_box,
           f"MultiHeadAttention\nQ = path_emb,  K = V = fused\n-> attended [B, {_D_FUSED}]",
           fc="#cfe2f3", ec="#1f4e79",
           fontsize=9, fontweight="bold")

    # Arrow: fusion -> attention (centre column, top-down with shape label).
    _arrow(ax,
           fuse_box[0] + fuse_box[2] / 2, fuse_box[1],
           attn_box[0] + attn_box[2] * 0.55, attn_box[1] + attn_box[3],
           label=f"fused [B, {_N_CT}, {_D_FUSED}]",
           label_offset=(0.0, 0.0))

    # Arrow: pathology encoder -> attention (right -> attn).
    # No mid-arrow label — "path_emb [B, d_cond]" is already named inside the
    # PathologyEncoder box itself, so the arrow stays clean.
    _arrow(ax,
           pathology_enc_box[0],
           pathology_enc_box[1] + pathology_enc_box[3] * 0.40,
           attn_box[0] + attn_box[2], attn_box[1] + attn_box[3] * 0.50)

    # Head box (below attention).
    head_box = (0.24, 0.20, 0.40, 0.10)
    _block(ax, *head_box,
           "ResDecMHEHead\n-> y_deep [B]",
           fc=_COLOR_HEAD, ec="#a04500",
           fontsize=10, fontweight="bold", text_color="white")
    # Arrow: attention -> head.
    _arrow(ax,
           attn_box[0] + attn_box[2] / 2, attn_box[1],
           head_box[0] + head_box[2] / 2, head_box[1] + head_box[3],
           label=f"attended [B, {_D_FUSED}]",
           label_offset=(0.0, 0.0))

    # -----------------------------------------------------------------
    # Row 3 — TabPFN residual + composite
    # -----------------------------------------------------------------
    tabpfn_box = (0.04, 0.06, 0.28, 0.08)
    plus_box = (0.36, 0.06, 0.10, 0.08)
    composite_box = (0.50, 0.04, 0.46, 0.12)
    _block(ax, *tabpfn_box,
           "TabPFN(metadata)\n-> y_TabPFN [B]",
           fc=_COLOR_TABPFN, ec="#7b1717",
           fontsize=9, fontweight="bold", text_color="white")
    _block(ax, *plus_box,
           "+",
           fc="white", ec="#222222",
           fontsize=18, fontweight="bold")
    _block(ax, *composite_box,
           "Composite prediction\n"
           r"$\hat{y} = \hat{y}_{\mathrm{TabPFN}} + \hat{y}_{\mathrm{deep}}$"
           "\n(vs cognitive_residual target)",
           fc="#d4eee0", ec="#1f6e1f",
           fontsize=10, fontweight="bold")

    # Arrow: head -> plus (right + down diagonal).
    _arrow(ax,
           head_box[0] + head_box[2] / 2, head_box[1],
           plus_box[0] + plus_box[2] / 2, plus_box[1] + plus_box[3],
           label="y_deep [B]",
           label_offset=(0.0, 0.0))
    # Arrow: TabPFN -> plus (left -> right).
    _arrow(ax,
           tabpfn_box[0] + tabpfn_box[2],
           tabpfn_box[1] + tabpfn_box[3] / 2,
           plus_box[0],
           plus_box[1] + plus_box[3] / 2,
           label="y_TabPFN [B]",
           label_offset=(0.0, 0.012))
    # Arrow: plus -> composite (right -> right).
    _arrow(ax,
           plus_box[0] + plus_box[2],
           plus_box[1] + plus_box[3] / 2,
           composite_box[0],
           composite_box[1] + composite_box[3] / 2)

    # Title strip at top (caption-style).
    ax.text(
        0.5, 0.985,
        "Fusion + pathology-stratified attention + TabPFN residual stack",
        ha="center", va="top", fontsize=11, fontweight="bold")

    fmt_axes(ax, hide_spines=("top", "right", "bottom", "left"),
             grid_major=False, grid_minor=False)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


# ===========================================================================
# Slot 3.1-3.3 — hybrid bio-iconographic full architecture
# ===========================================================================


def _draw_cell_cluster(
    ax,
    cx: float,
    cy: float,
    *,
    radius_outer: float = 0.06,
    n_cells: int = 10,
    palette: list[str] | None = None,
    cell_radius: float = 0.012,
    rng_seed: int = 17,
):
    """Draw a stylized cluster of snRNA-seq cells (small filled circles)."""
    if palette is None:
        palette = list(PALETTES["categorical"])[:6]
    rng = np.random.default_rng(rng_seed)
    for i in range(n_cells):
        # Sample radial offset (closer to center) and angle.
        r = radius_outer * np.sqrt(rng.random())
        theta = 2 * np.pi * rng.random()
        x = cx + r * np.cos(theta)
        y = cy + r * np.sin(theta)
        c = palette[i % len(palette)]
        circ = Circle((x, y), cell_radius, facecolor=c,
                      edgecolor="white", linewidth=0.5, zorder=4)
        ax.add_patch(circ)


def _draw_ct_graph(
    ax,
    cx: float,
    cy: float,
    *,
    radius: float = 0.075,
    n_nodes: int = 7,
    palette: list[str] | None = None,
    node_radius: float = 0.014,
    edge_color: str = "#777777",
    rng_seed: int = 23,
):
    """Draw a small CT-CT signaling graph (nodes = CT colors, edges = LR pairs)."""
    if palette is None:
        palette = list(PALETTES["categorical"])[:n_nodes]
    rng = np.random.default_rng(rng_seed)
    # Equally spaced node positions on a circle.
    positions = []
    for i in range(n_nodes):
        theta = 2 * np.pi * i / n_nodes - np.pi / 2
        positions.append((cx + radius * np.cos(theta),
                          cy + radius * np.sin(theta)))
    # Random subset of edges (LR-pair connections), but consistent each call.
    n_edges = min(n_nodes + 3, n_nodes * (n_nodes - 1) // 2)
    pairs_seen: set[tuple[int, int]] = set()
    while len(pairs_seen) < n_edges:
        i, j = sorted(rng.choice(n_nodes, size=2, replace=False))
        if i == j:
            continue
        pairs_seen.add((int(i), int(j)))
    for i, j in pairs_seen:
        ax.plot([positions[i][0], positions[j][0]],
                [positions[i][1], positions[j][1]],
                color=edge_color, linewidth=0.7, alpha=0.7,
                zorder=3, solid_capstyle="round")
    for i, (x, y) in enumerate(positions):
        c = palette[i % len(palette)]
        circ = Circle((x, y), node_radius, facecolor=c,
                      edgecolor="black", linewidth=0.6, zorder=4)
        ax.add_patch(circ)


def build_slot3_full_architecture_hybrid() -> plt.Figure:
    """Hybrid bio-iconographic full-architecture diagram.

    Layout: top -> bottom.
      Row 1 (top)         input data plate (snRNA-seq cells)
      Row 2 (upper-mid)   two branches (cell-by-cell pooling + cell-cell signaling)
      Row 3 (mid)         fusion + pathology-conditioned attention
      Row 4 (bot)         composite prediction (TabPFN + DeepModel)
      Caption strip at bottom.
    """
    fig, ax = plt.subplots(figsize=(14.0, 10.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    ax.axis("off")

    palette_ct = list(PALETTES["categorical"])

    # -----------------------------------------------------------------
    # Row 1 — Input data plate (no extra side icons here; the bigger
    # individual-cells / CellChatDB graph icons further down do the visual
    # work for each branch and adding more clusters here just makes the
    # top of the figure noisy).
    # -----------------------------------------------------------------
    input_box = (0.28, 0.86, 0.44, 0.09)
    _block(ax, *input_box,
           "snRNA-seq cells from N=516 ROSMAP subjects\n(post-mortem PFC tissue)",
           fc="#fff8e1", ec="#8c6d18", fontsize=11, fontweight="bold")

    # -----------------------------------------------------------------
    # Row 2 — Two branches (cell-by-cell pooling + cell-cell signaling)
    # -----------------------------------------------------------------
    # LEFT branch: Cell-by-cell pooling (cell-type expression).
    # Plate: cells of various colors -> pooler box.
    # Cell-cluster icon (left), big version with caption below.
    _draw_cell_cluster(ax, 0.13, 0.755, radius_outer=0.040,
                       n_cells=18, palette=palette_ct, cell_radius=0.010,
                       rng_seed=11)
    ax.text(0.13, 0.668,
            "individual cells",
            ha="center", va="center", fontsize=8, fontstyle="italic",
            color="#555555")

    cell_branch_box = (0.05, 0.50, 0.33, 0.10)
    _block(ax, *cell_branch_box,
           "Cell-by-cell pooling (Set Transformer)\n"
           f"-> {_N_CT} cell-type vectors per subject",
           fc=_COLOR_CELL_BRANCH, ec="#0f6e62",
           fontsize=10, fontweight="bold", text_color="white")
    # Arrow input plate -> cell-icons -> cell branch box.
    _arrow(ax, input_box[0] + input_box[2] * 0.20, input_box[1],
           0.13, 0.79)
    _arrow(ax, 0.13, 0.685, 0.13, cell_branch_box[1] + cell_branch_box[3])

    # RIGHT branch: Cell-cell signaling (HGT).
    # Graph icon (right), big version with caption.
    _draw_ct_graph(ax, 0.87, 0.755, radius=0.045, n_nodes=8,
                   palette=palette_ct, rng_seed=31)
    ax.text(0.87, 0.668,
            "CellChatDB\nligand-receptor pathways",
            ha="center", va="center", fontsize=8, fontstyle="italic",
            color="#555555")

    hgt_branch_box = (0.62, 0.50, 0.33, 0.10)
    _block(ax, *hgt_branch_box,
           "Cell-cell signaling (Heterog. Graph Transformer)\n"
           f"-> {_N_CT} cell-type vectors w/ cross-CT context",
           fc=_COLOR_HGT_BRANCH, ec="#a23e4f",
           fontsize=10, fontweight="bold", text_color="white")
    # Arrow input plate -> graph -> HGT branch box.
    _arrow(ax, input_box[0] + input_box[2] * 0.80, input_box[1],
           0.87, 0.79)
    _arrow(ax, 0.87, 0.685, 0.87, hgt_branch_box[1] + hgt_branch_box[3])

    # -----------------------------------------------------------------
    # Row 3 — Fusion + pathology-conditioned attention
    # -----------------------------------------------------------------
    fusion_box = (0.30, 0.34, 0.40, 0.08)
    _block(ax, *fusion_box,
           "Combined cell-type embedding\n(fusion: concat + cross-attention)",
           fc=_COLOR_FUSION, ec="#1f6e1f",
           fontsize=10, fontweight="bold", text_color="white")
    # Arrows: each branch -> fusion (with shape).
    _arrow(ax,
           cell_branch_box[0] + cell_branch_box[2] / 2,
           cell_branch_box[1],
           fusion_box[0] + fusion_box[2] * 0.30,
           fusion_box[1] + fusion_box[3],
           label=f"[B, {_N_CT}, {_D_EMBED}]")
    _arrow(ax,
           hgt_branch_box[0] + hgt_branch_box[2] / 2,
           hgt_branch_box[1],
           fusion_box[0] + fusion_box[2] * 0.70,
           fusion_box[1] + fusion_box[3],
           label=f"[B, {_N_CT}, {_D_EMBED}]")

    # Pathology side icon + box (right of attention).
    pathology_box = (0.74, 0.22, 0.22, 0.08)
    _block(ax, *pathology_box,
           "Pathology covariates\n(amyloid + tau + TDP-43)",
           fc=_COLOR_PATHOLOGY, ec="#3d2a55",
           fontsize=9, fontweight="bold", text_color="white")

    attention_box = (0.20, 0.22, 0.50, 0.08)
    _block(ax, *attention_box,
           "Pathology-conditioned attention\n"
           "(reweights cell-type contributions by pathology context)",
           fc="#cfe2f3", ec="#1f4e79",
           fontsize=10, fontweight="bold")
    # Arrow: fusion -> attention.
    _arrow(ax,
           fusion_box[0] + fusion_box[2] / 2, fusion_box[1],
           attention_box[0] + attention_box[2] * 0.55,
           attention_box[1] + attention_box[3],
           label=f"fused [B, {_N_CT}, {_D_FUSED}]")
    # Arrow: pathology -> attention.
    _arrow(ax,
           pathology_box[0],
           pathology_box[1] + pathology_box[3] / 2,
           attention_box[0] + attention_box[2],
           attention_box[1] + attention_box[3] / 2,
           label=f"path_emb [B, {_D_COND}]",
           label_offset=(0.0, 0.012))

    # -----------------------------------------------------------------
    # Row 4 — Composite prediction
    # -----------------------------------------------------------------
    deep_box = (0.04, 0.08, 0.30, 0.07)
    tabpfn_box = (0.36, 0.08, 0.30, 0.07)
    composite_box = (0.68, 0.07, 0.30, 0.10)
    _block(ax, *deep_box,
           "DeepModel(snRNA-seq)\n"
           r"-> $\hat{y}_{\mathrm{deep}}$",
           fc=_COLOR_HEAD, ec="#a04500",
           fontsize=10, fontweight="bold", text_color="white")
    _block(ax, *tabpfn_box,
           "TabPFN(metadata)\n"
           r"-> $\hat{y}_{\mathrm{TabPFN}}$",
           fc=_COLOR_TABPFN, ec="#7b1717",
           fontsize=10, fontweight="bold", text_color="white")
    _block(ax, *composite_box,
           "Composite prediction\n"
           r"$\hat{y} = \hat{y}_{\mathrm{TabPFN}} + \hat{y}_{\mathrm{deep}}$",
           fc="#d4eee0", ec="#1f6e1f",
           fontsize=11, fontweight="bold")
    # Arrow: attention -> deep model.
    _arrow(ax,
           attention_box[0] + attention_box[2] * 0.30, attention_box[1],
           deep_box[0] + deep_box[2] / 2, deep_box[1] + deep_box[3])
    # Arrow: input plate -> tabpfn (curved long arc, hugging the centre).
    _arrow(ax,
           input_box[0] + input_box[2] * 0.50, input_box[1],
           tabpfn_box[0] + tabpfn_box[2] / 2, tabpfn_box[1] + tabpfn_box[3],
           connectionstyle="arc3,rad=0.18",
           color="#888888",
           linewidth=0.9,
           label="(subject metadata)",
           label_offset=(0.06, 0.0),
           label_fontsize=8)
    # Arrow: deep -> composite + tabpfn -> composite.
    _arrow(ax,
           deep_box[0] + deep_box[2],
           deep_box[1] + deep_box[3] / 2,
           composite_box[0],
           composite_box[1] + composite_box[3] / 2)
    _arrow(ax,
           tabpfn_box[0] + tabpfn_box[2],
           tabpfn_box[1] + tabpfn_box[3] / 2,
           composite_box[0],
           composite_box[1] + composite_box[3] / 2)

    # -----------------------------------------------------------------
    # Caption strip at the very bottom.
    # -----------------------------------------------------------------
    caption = (
        "Two-branch architecture; both branches output 31 cell-type vectors "
        "per subject; fusion + pathology-conditioned attention + TabPFN "
        "residual stack."
    )
    ax.text(0.5, 0.005, caption,
            ha="center", va="bottom", fontsize=9,
            color="#444444", fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fafafa",
                      edgecolor="#cccccc", linewidth=0.6))

    fmt_axes(ax, hide_spines=("top", "right", "bottom", "left"),
             grid_major=False, grid_minor=False)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


# ===========================================================================
# Top-level orchestrator
# ===========================================================================


def build_all_figures(*, out_dir: Path | str) -> dict[str, list[Path]]:
    """Build + save the 2 architecture-diagram lab-meeting figures.

    Returns
    -------
    dict
        Mapping ``slot_name -> [paths_written]`` for each of the 2 figures.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    apply_theme()

    written: dict[str, list[Path]] = {}

    # Slot 3.4 — fusion stack
    fig_a = build_slot3_4_fusion_stack()
    written["slot3_4"] = save_fig(
        fig_a, out_dir / "fig_slot3_4_fusion_stack", dpi=600,
    )
    plt.close(fig_a)

    # Slot 3.1-3.3 — full architecture hybrid
    fig_b = build_slot3_full_architecture_hybrid()
    written["slot3_full"] = save_fig(
        fig_b, out_dir / "fig_slot3_full_architecture_hybrid", dpi=600,
    )
    plt.close(fig_b)

    for slot_name, paths in written.items():
        for p in paths:
            logger.info("wrote %s (%.1f KB)", p, p.stat().st_size / 1024.0)

    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-dir", default=_DEFAULT_OUT_DIR)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    build_all_figures(out_dir=Path(args.out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
