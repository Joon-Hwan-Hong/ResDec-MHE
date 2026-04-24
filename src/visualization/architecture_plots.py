"""Architecture plots: model schematic + structural summaries.

Functions:

  - ``plot_architecture_diagram`` — pure-matplotlib block diagram of the
    ResDec-MHE encoder + head + TabPFN residual base, with arrows showing
    the data-flow path from input pseudobulk to composite prediction.

  - ``plot_hgt_celltype_network`` — directed graph of top-K HGT edges
    between cell types, with edge thickness = mean attention and color =
    edge type.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.visualization.theme import PALETTES, baseline_color, save_fig


def plot_architecture_diagram(
    *,
    figsize: tuple[float, float] = (7.0, 5.0),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Pure-matplotlib model schematic of ResDec-MHE.

    Boxes for each component (HGT, CellTransformer, PMA, RegionHandler,
    PathologyAttention, gene_gate, ResDecMHE head, TabPFN base) with
    arrows showing data flow. No real data dependency.
    """
    fig, ax = plt.subplots(figsize=figsize)

    def box(x, y, w, h, label, color):
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="black",
                             linewidth=0.8, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=7, fontweight="bold", zorder=3)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=0.7, color="#444"),
                    zorder=1)

    box(0.05, 0.75, 0.20, 0.10, "Per-subject\npseudobulk", "#e6e6e6")
    box(0.30, 0.85, 0.20, 0.10, "HGT (CCC graph)", PALETTES["categorical"][0])
    box(0.30, 0.65, 0.20, 0.10, "CellTransformer", PALETTES["categorical"][1])
    box(0.55, 0.75, 0.18, 0.10, "PMA pool", PALETTES["categorical"][2])
    box(0.55, 0.55, 0.18, 0.10, "RegionHandler\n(6-scalar)", PALETTES["categorical"][3])
    box(0.55, 0.35, 0.18, 0.10, "Pathology\nAttention", PALETTES["categorical"][4])
    box(0.55, 0.15, 0.18, 0.10, "gene_gate", PALETTES["categorical"][5])
    box(0.78, 0.45, 0.18, 0.20, "z ∈ ℝ⁶⁴", "#fff2cc")
    box(0.78, 0.10, 0.18, 0.20,
        "ResDecMHE Head\n(NPT + TabM\nk=8 ensemble\n+ HyperConn\n+ FiLM)",
        baseline_color("ResDec-MHE"))
    box(0.05, 0.20, 0.35, 0.10, "TabPFN-2.6 (in-context, top-2K features)",
        baseline_color("TabPFN-2.6"))
    box(0.45, 0.20, 0.20, 0.10, "ŷ_tabpfn", "#ffd9d9")
    box(0.45, 0.02, 0.20, 0.10, "f̂_residual", "#d9ffd9")
    box(0.78, 0.02, 0.18, 0.10, "ŷ = ŷ_tabpfn + f̂", "#cce5ff")

    arrow(0.25, 0.80, 0.30, 0.90)
    arrow(0.25, 0.80, 0.30, 0.70)
    arrow(0.50, 0.90, 0.55, 0.80)
    arrow(0.50, 0.70, 0.55, 0.60)
    arrow(0.50, 0.65, 0.55, 0.40)
    arrow(0.73, 0.80, 0.78, 0.60)
    arrow(0.73, 0.60, 0.78, 0.55)
    arrow(0.73, 0.40, 0.78, 0.50)
    arrow(0.73, 0.20, 0.78, 0.45)
    arrow(0.78, 0.50, 0.78, 0.30)
    arrow(0.40, 0.25, 0.45, 0.25)
    arrow(0.78, 0.10, 0.65, 0.07)
    arrow(0.65, 0.07, 0.78, 0.07)
    arrow(0.55, 0.25, 0.78, 0.07)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_axis_off()
    if save_path is not None:
        save_fig(fig, save_path)
    return fig


def plot_hgt_celltype_network(
    edge_attention_df: pd.DataFrame,
    *,
    top_k_edges: int = 50,
    edge_type_col: str = "edge_type_name",
    sender_col: str = "source_ct",
    receiver_col: str = "target_ct",
    weight_col: str = "mean_attention",
    figsize: tuple[float, float] = (6.5, 5.5),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Network: top-K HGT edges as a directed graph between cell types."""
    try:
        import networkx as nx
    except ImportError as exc:
        raise ImportError("networkx required for plot_hgt_celltype_network") from exc
    if edge_attention_df.empty:
        raise ValueError("empty edge_attention_df")
    df = edge_attention_df.sort_values(weight_col, ascending=False).head(top_k_edges)
    G = nx.DiGraph()
    edge_types = (
        sorted(df[edge_type_col].unique()) if edge_type_col in df.columns else ["edge"]
    )
    edge_color_map = {
        et: PALETTES["categorical_paired"][i % len(PALETTES["categorical_paired"])]
        for i, et in enumerate(edge_types)
    }
    for _, row in df.iterrows():
        s = row[sender_col]
        r = row[receiver_col]
        et = row.get(edge_type_col, "edge")
        w = float(row[weight_col])
        G.add_edge(s, r, weight=w, color=edge_color_map.get(et, "#888"))
    pos = nx.spring_layout(G, seed=42, k=1.5 / max(1, len(G) ** 0.5))
    fig, ax = plt.subplots(figsize=figsize)
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    colors = [G[u][v]["color"] for u, v in G.edges()]
    if not weights:
        raise ValueError("no edges after filtering")
    w_arr = np.array(weights)
    w_norm = (w_arr - w_arr.min()) / (max(1e-9, w_arr.max() - w_arr.min()))
    nx.draw_networkx_nodes(G, pos, node_color="#cccccc", node_size=300,
                           edgecolors="black", linewidths=0.6, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color=colors, width=0.8 + 2.5 * w_norm,
                           arrows=True, arrowsize=10, alpha=0.7, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=6, ax=ax)
    handles = [
        plt.Line2D([0], [0], color=color, lw=2, label=et)
        for et, color in edge_color_map.items()
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=7, title="Edge type")
    ax.set_axis_off()
    if save_path is not None:
        save_fig(fig, save_path)
    return fig
