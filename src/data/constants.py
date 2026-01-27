"""
Central registry for data schema constants.

Single source of truth for cell types, edge types, and regions.
All data modules import from here.
"""

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

# Legacy aliases for backwards compatibility
CELLCHATDB_CATEGORIES = CELLCHATDB_EDGE_TYPES
NOVEL_CATEGORY = EDGE_TYPE_NOVEL

# 6 ROSMAP brain regions
REGION_ORDER: list[str] = [
    "PFC", "EC", "HC", "TH", "AG", "MTC"
]

# Derived constants
N_CELL_TYPES: int = len(CELL_TYPE_ORDER)  # 31
N_EDGE_TYPES: int = len(ALL_EDGE_TYPES)   # 5
N_REGIONS: int = len(REGION_ORDER)        # 6

# Separator for composite keys (must not appear in cell type names or subject IDs)
GROUP_SEPARATOR: str = "||"
