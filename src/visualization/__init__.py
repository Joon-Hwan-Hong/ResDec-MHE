"""Visualization and plotting modules.

All plot_* functions return matplotlib Figure objects. Callers are responsible
for calling plt.close(fig) after saving/displaying to prevent memory leaks
in batch contexts.
"""

from src.visualization.config import (
    FIGURE_DPI,
    FIGURE_FORMAT,
    ACCENT_TEAL,
    ACCENT_CORAL,
    ACCENT_PEACH,
    get_sequential_cmap,
    get_diverging_cmap,
    get_cell_type_color,
    get_edge_type_color,
    setup_seaborn_style,
    setup_matplotlib_defaults,
    save_figure,
    CELL_TYPE_COLORS,
    EDGE_TYPE_COLORS,
)

from src.visualization.attention_plots import (
    plot_cell_type_attention_heatmap,
    plot_cell_type_importance_bar,
    plot_attention_distribution,
    plot_gene_gate_heatmap,
    plot_resilience_signature_heatmap,
)

from src.visualization.importance_plots import (
    plot_top_genes_per_cell_type,
    plot_gene_importance_volcano,
    plot_ccc_network_summary,
    plot_top_interactions_heatmap,
    plot_regional_gene_importance,
)

from src.visualization.prediction_plots import (
    plot_predicted_vs_actual,
    plot_calibration_curve,
    plot_residuals,
    plot_uncertainty_vs_error,
    plot_uncertainty_correlates,
)

from src.visualization.training_curves import (
    load_tensorboard_scalars,
    plot_loss_curves,
    plot_learning_rate,
    plot_training_summary,
)

from src.visualization.embedding_plots import (
    plot_umap_scatter,
    plot_cluster_composition,
    plot_linear_probe_results,
    plot_embedding_summary,
)

__all__ = [
    # Config
    "FIGURE_DPI",
    "FIGURE_FORMAT",
    "ACCENT_TEAL",
    "ACCENT_CORAL",
    "ACCENT_PEACH",
    "get_sequential_cmap",
    "get_diverging_cmap",
    "get_cell_type_color",
    "get_edge_type_color",
    "setup_seaborn_style",
    "setup_matplotlib_defaults",
    "save_figure",
    "CELL_TYPE_COLORS",
    "EDGE_TYPE_COLORS",
    # Attention plots
    "plot_cell_type_attention_heatmap",
    "plot_cell_type_importance_bar",
    "plot_attention_distribution",
    "plot_gene_gate_heatmap",
    "plot_resilience_signature_heatmap",
    # Importance plots
    "plot_top_genes_per_cell_type",
    "plot_gene_importance_volcano",
    "plot_ccc_network_summary",
    "plot_top_interactions_heatmap",
    "plot_regional_gene_importance",
    # Prediction plots
    "plot_predicted_vs_actual",
    "plot_calibration_curve",
    "plot_residuals",
    "plot_uncertainty_vs_error",
    "plot_uncertainty_correlates",
    # Training curves
    "load_tensorboard_scalars",
    "plot_loss_curves",
    "plot_learning_rate",
    "plot_training_summary",
    # Embedding plots
    "plot_umap_scatter",
    "plot_cluster_composition",
    "plot_linear_probe_results",
    "plot_embedding_summary",
]