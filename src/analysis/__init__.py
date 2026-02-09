"""Post-hoc analysis and interpretability modules."""

from src.analysis.cell_type_importance import (
    CellTypeImportanceAnalyzer,
    CellTypeImportanceResult,
    compute_cell_type_importance,
    load_cell_type_importance,
)
from src.analysis.gene_importance import (
    GeneImportanceAnalyzer,
    GeneImportanceResult,
    compute_gene_importance,
    load_gene_gate_weights_hdf5,
)
from src.analysis.ccc_importance import (
    CCCImportanceAnalyzer,
    CCCImportanceResult,
    compute_ccc_importance,
    create_edge_metadata_from_graph,
)
from src.analysis.resilience_signatures import (
    ResilienceSignatureAnalyzer,
    ResilienceSignatureResult,
    compute_resilience_signature,
)
from src.analysis.regional_analysis import (
    RegionalAnalyzer,
    RegionalAnalysisResult,
    compute_regional_analysis,
)
from src.analysis.uncertainty_analysis import (
    UncertaintyAnalyzer,
    UncertaintyAnalysisResult,
    compute_uncertainty_analysis,
    compute_ece_regression,
)
from src.utils.statistics import CALIBRATION_LEVELS
from src.analysis.cell_heterogeneity import (
    CellHeterogeneityAnalyzer,
    CellHeterogeneityResult,
    compute_cell_heterogeneity,
)
from src.analysis.embedding_analysis import (
    EmbeddingAnalyzer,
    EmbeddingAnalysisResult,
    analyze_embeddings,
)
__all__ = [
    # Cell type importance
    "CellTypeImportanceAnalyzer",
    "CellTypeImportanceResult",
    "compute_cell_type_importance",
    "load_cell_type_importance",
    # Gene importance
    "GeneImportanceAnalyzer",
    "GeneImportanceResult",
    "compute_gene_importance",
    "load_gene_gate_weights_hdf5",
    # CCC importance
    "CCCImportanceAnalyzer",
    "CCCImportanceResult",
    "compute_ccc_importance",
    "create_edge_metadata_from_graph",
    # Resilience signatures
    "ResilienceSignatureAnalyzer",
    "ResilienceSignatureResult",
    "compute_resilience_signature",
    # Regional analysis
    "RegionalAnalyzer",
    "RegionalAnalysisResult",
    "compute_regional_analysis",
    # Uncertainty analysis
    "UncertaintyAnalyzer",
    "UncertaintyAnalysisResult",
    "compute_uncertainty_analysis",
    "compute_ece_regression",
    "CALIBRATION_LEVELS",
    # Cell heterogeneity
    "CellHeterogeneityAnalyzer",
    "CellHeterogeneityResult",
    "compute_cell_heterogeneity",
    # Embedding analysis
    "EmbeddingAnalyzer",
    "EmbeddingAnalysisResult",
    "analyze_embeddings",
]