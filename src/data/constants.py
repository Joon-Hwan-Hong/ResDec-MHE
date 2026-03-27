"""
Central registry for data schema constants.

Single source of truth for cell types, edge types, and regions.
All data modules import from here.
"""

from pathlib import Path

# Project root: resolve from this file's location (src/data/constants.py → project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# CellChatDB database path (anchored to project root)
CELLCHATDB_PATH: Path = _PROJECT_ROOT / "data" / "database" / "CellChatDB_human_interaction.csv"

# 31 cell types from Allen Brain Atlas taxonomy
CELL_TYPE_ORDER: list[str] = [
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

# 5 CellChatDB interaction categories
# Use underscores for code compatibility (PyG edge type tuples, config keys)
EDGE_TYPE_SECRETED_SIGNALING: str = "Secreted_Signaling"
EDGE_TYPE_ECM_RECEPTOR: str = "ECM_Receptor"
EDGE_TYPE_CELL_CELL_CONTACT: str = "Cell_Cell_Contact"
EDGE_TYPE_NON_PROTEIN_SIGNALING: str = "Non_protein_Signaling"
EDGE_TYPE_NOVEL: str = "Novel_Uncharacterized"

CELLCHATDB_EDGE_TYPES: list[str] = [
    EDGE_TYPE_SECRETED_SIGNALING,
    EDGE_TYPE_ECM_RECEPTOR,
    EDGE_TYPE_CELL_CELL_CONTACT,
    EDGE_TYPE_NON_PROTEIN_SIGNALING,
]

ALL_EDGE_TYPES: list[str] = CELLCHATDB_EDGE_TYPES + [EDGE_TYPE_NOVEL]

# Display names for visualization (human-readable with spaces)
EDGE_TYPE_DISPLAY_NAMES: dict[str, str] = {
    EDGE_TYPE_SECRETED_SIGNALING: "Secreted Signaling",
    EDGE_TYPE_ECM_RECEPTOR: "ECM-Receptor",
    EDGE_TYPE_CELL_CELL_CONTACT: "Cell-Cell Contact",
    EDGE_TYPE_NON_PROTEIN_SIGNALING: "Non-protein Signaling",
    EDGE_TYPE_NOVEL: "Novel/Uncharacterized",
}

# 6 ROSMAP brain regions
# Names match design doc (2026-01-27-region-handler-design.md) and RegionHandler.REGIONS
REGION_ORDER: list[str] = [
    "PFC",  # Prefrontal cortex (Region 0, primary)
    "AG",   # Angular gyrus
    "MTC",  # Midtemporal cortex
    "EC",   # Entorhinal cortex
    "HC",   # Hippocampus
    "TH",   # Anterior thalamus
]

# Derived constants
N_CELL_TYPES: int = len(CELL_TYPE_ORDER)  # 31
N_EDGE_TYPES: int = len(ALL_EDGE_TYPES)   # 5
N_REGIONS: int = len(REGION_ORDER)        # 6

# Separator for composite keys (must not appear in cell type names or subject IDs)
GROUP_SEPARATOR: str = "||"

# Index of the primary region (PFC) within REGION_ORDER
PFC_REGION_IDX: int = REGION_ORDER.index("PFC")  # 0


# Numerical stability constants (three tiers by magnitude and intent)
EPSILON_DIVISION: float = 1e-10     # Division-by-zero guard for float64 computations
EPSILON_SOFTMAX: float = 1e-8       # Softmax/normalization denominator guard
EPSILON_POSITIVE_FLOOR: float = 1e-6  # Minimum positive value floor (e.g., std, scores)


def sanitize_key(name: str) -> str:
    """
    Sanitize a name for use as a PyTorch ModuleDict/ParameterDict key.

    PyTorch requires keys to be valid Python identifiers. This function
    replaces characters that appear in cell type names, edge type names,
    and region names with underscores.

    This is the single canonical implementation — all modules that need
    sanitized keys (HGTConvTensor, HGTEncoderTensor, collate_for_hgt,
    CognitiveResilienceModel) must import and use this function.

    Args:
        name: Original name (e.g., "Oligodendrocyte precursor", "ECM-Receptor")

    Returns:
        Sanitized string safe for use as a dict key
    """
    sanitized = name.replace(" ", "_").replace("-", "_").replace("/", "_")
    if sanitized and sanitized[0].isdigit():
        sanitized = "_" + sanitized
    return sanitized


# Pre-computed sanitized names for collate functions.
# Avoids 36 sanitize_key() calls per batch (31 cell types + 5 edge types).
SANITIZED_CELL_TYPE_ORDER: list[str] = [sanitize_key(ct) for ct in CELL_TYPE_ORDER]
SANITIZED_EDGE_TYPES: list[str] = [sanitize_key(et) for et in ALL_EDGE_TYPES]
