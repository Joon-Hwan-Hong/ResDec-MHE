"""
LIANA+ processing for cell-cell communication analysis.

Provides edge type assignment based on CellChatDB categories and
utilities for building heterogeneous graphs from LIANA+ results.
"""

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from anndata import AnnData

from src.data.constants import CELLCHATDB_EDGE_TYPES, CELLCHATDB_PATH, EDGE_TYPE_NOVEL, ALL_EDGE_TYPES


def _normalize_annotation(annotation: str) -> str:
    """
    Normalize CellChatDB annotation to our standard format.

    CellChatDB uses spaces (e.g., "Secreted Signaling") but we use
    underscores (e.g., "Secreted_Signaling") for code compatibility
    with PyG edge type tuples and config keys.

    Args:
        annotation: Raw annotation from CellChatDB

    Returns:
        Normalized annotation with underscores
    """
    # Replace spaces and hyphens with underscores to match our standard format
    # (e.g., "Secreted Signaling" → "Secreted_Signaling",
    #  "Non-protein Signaling" → "Non_protein_Signaling")
    return annotation.replace(" ", "_").replace("-", "_")


def load_cellchatdb_categories(
    db_path: str | Path,
) -> dict[str, str]:
    """
    Load CellChatDB ligand-receptor to category mapping.

    Args:
        db_path: Path to CellChatDB_human_interaction.csv

    Returns:
        Dict mapping "LIGAND_RECEPTOR" -> category name (normalized with underscores)
    """
    db = pd.read_csv(db_path)

    lr_to_category = {}
    for _, row in db.iterrows():
        # Get ligand and receptor symbols with proper NaN handling
        # row.get() returns NaN if column exists but value is NaN, and NaN
        # is truthy so "NaN or fallback" doesn't work as expected.
        ligand = row.get("ligand.symbol")
        if pd.isna(ligand):
            ligand = row.get("ligand_symbol", "")
        if pd.isna(ligand):
            ligand = ""

        receptor = row.get("receptor.symbol")
        if pd.isna(receptor):
            receptor = row.get("receptor_symbol", "")
        if pd.isna(receptor):
            receptor = ""

        # row.get returns NaN if column exists but value is NaN;
        # the default is only used when key is missing entirely
        annotation = row.get("annotation", EDGE_TYPE_NOVEL)
        if pd.isna(annotation):
            annotation = EDGE_TYPE_NOVEL

        if not ligand or not receptor:
            continue

        # Normalize annotation to our standard format (underscores)
        normalized_annotation = _normalize_annotation(str(annotation))

        # Create standardized LR key (gene symbols)
        lr_key = f"{ligand}_{receptor}"
        lr_to_category[lr_key] = normalized_annotation

        # Also store reversed for flexibility
        # (some tools might report receptor_ligand)
        lr_key_rev = f"{receptor}_{ligand}"
        if lr_key_rev not in lr_to_category:
            lr_to_category[lr_key_rev] = normalized_annotation

    return lr_to_category


def assign_edge_types(
    liana_results: pd.DataFrame,
    cellchatdb_path: str | Path = CELLCHATDB_PATH,
    novel_category: str = EDGE_TYPE_NOVEL,
    ligand_col: str = "ligand_complex",
    receptor_col: str = "receptor_complex",
) -> pd.DataFrame:
    """
    Assign edge types to LIANA+ results using CellChatDB categories.

    Args:
        liana_results: LIANA+ output DataFrame with columns:
            source, target, ligand_complex, receptor_complex,
            magnitude_rank, specificity_rank, etc.
        cellchatdb_path: Path to CellChatDB CSV
        novel_category: Category for LR pairs not in CellChatDB
        ligand_col: Column name for ligand
        receptor_col: Column name for receptor

    Returns:
        liana_results with added 'edge_type' and 'edge_type_name' columns
    """
    # Load CellChatDB mapping
    cellchatdb_path = Path(cellchatdb_path)
    if cellchatdb_path.exists():
        lr_to_category = load_cellchatdb_categories(cellchatdb_path)
    else:
        print(f"Warning: CellChatDB not found at {cellchatdb_path}")
        lr_to_category = {}

    # Get unique categories for encoding
    categories = CELLCHATDB_EDGE_TYPES + [novel_category]
    category_to_idx = {cat: idx for idx, cat in enumerate(categories)}

    # Assign edge types
    def get_edge_type_name(row):
        ligand = row.get(ligand_col, "")
        receptor = row.get(receptor_col, "")

        if pd.isna(ligand) or pd.isna(receptor):
            return novel_category

        lr_key = f"{ligand}_{receptor}"

        if lr_key in lr_to_category:
            return lr_to_category[lr_key]
        else:
            return novel_category

    liana_results = liana_results.copy()
    liana_results["edge_type_name"] = liana_results.apply(get_edge_type_name, axis=1)
    liana_results["edge_type"] = liana_results["edge_type_name"].map(category_to_idx)

    return liana_results


def get_edge_type_metadata() -> dict:
    """
    Return edge type metadata for model configuration and interpretation.

    Returns:
        Dictionary with edge type information
    """
    return {
        "categories": ALL_EDGE_TYPES,
        "n_edge_types": len(ALL_EDGE_TYPES),
        "category_to_idx": {cat: idx for idx, cat in enumerate(ALL_EDGE_TYPES)},
        "idx_to_category": {idx: cat for idx, cat in enumerate(ALL_EDGE_TYPES)},
        "source": "CellChatDB + LIANA+ (constrained data-driven)",
    }


# ─────────────────────────────────────────────────────────────────────────────
# LIANA+ Analysis
# ─────────────────────────────────────────────────────────────────────────────


def run_liana_analysis(
    adata: AnnData,
    cell_type_column: str = "supercluster_name",
    resource_name: str = "CellChatDB",
    expr_prop: float = 0.1,
    min_cells: int = 10,
    use_raw: bool = False,
    verbose: bool = True,
    n_perms: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run LIANA+ cell-cell communication analysis.

    Args:
        adata: Preprocessed AnnData (normalized, log-transformed)
        cell_type_column: Column containing cell type labels
        resource_name: Ligand-receptor resource to use
        expr_prop: Minimum proportion of cells expressing ligand/receptor
        min_cells: Minimum cells per cell type
        use_raw: Whether to use raw counts (adata.raw)
        verbose: Print progress
        n_perms: Number of permutations for significance testing
        seed: Random seed

    Returns:
        DataFrame with LIANA+ results
    """
    try:
        import liana as li
    except ImportError:
        raise ImportError(
            "LIANA+ is required for CCI analysis. "
            "Install with: pip install liana"
        )

    if verbose:
        print(f"Running LIANA+ with {resource_name} resource...")
        print(f"Cell type column: {cell_type_column}")
        print(f"Number of cell types: {adata.obs[cell_type_column].nunique()}")

    # Run LIANA
    li.mt.rank_aggregate(
        adata,
        groupby=cell_type_column,
        resource_name=resource_name,
        expr_prop=expr_prop,
        min_cells=min_cells,
        use_raw=use_raw,
        n_perms=n_perms,
        seed=seed,
        verbose=verbose,
    )

    # Extract results
    liana_results = adata.uns["liana_res"].copy()

    if verbose:
        print(f"LIANA+ found {len(liana_results):,} interactions")

    return liana_results


def filter_liana_results(
    liana_results: pd.DataFrame,
    magnitude_rank_threshold: float = 0.05,
    specificity_rank_threshold: float = 0.05,
    min_score: float | None = None,
) -> pd.DataFrame:
    """
    Filter LIANA+ results by significance thresholds.

    Args:
        liana_results: Raw LIANA+ output
        magnitude_rank_threshold: Keep interactions with magnitude_rank <= threshold
        specificity_rank_threshold: Keep interactions with specificity_rank <= threshold
        min_score: Optional minimum aggregate score

    Returns:
        Filtered DataFrame
    """
    filtered = liana_results.copy()

    # Filter by magnitude rank
    if "magnitude_rank" in filtered.columns:
        filtered = filtered[filtered["magnitude_rank"] <= magnitude_rank_threshold]

    # Filter by specificity rank
    if "specificity_rank" in filtered.columns:
        filtered = filtered[filtered["specificity_rank"] <= specificity_rank_threshold]

    # Filter by aggregate score
    if min_score is not None and "liana_score" in filtered.columns:
        filtered = filtered[filtered["liana_score"] >= min_score]

    return filtered


def aggregate_liana_by_celltype_pair(
    liana_results: pd.DataFrame,
    source_col: str = "source",
    target_col: str = "target",
    score_col: str = "magnitude_rank",
    agg_func: Literal["mean", "sum", "max", "count"] = "mean",
) -> pd.DataFrame:
    """
    Aggregate LIANA+ results by cell type pair.

    Creates a summary matrix of interactions between cell types.

    Args:
        liana_results: LIANA+ output
        source_col: Column for source cell type
        target_col: Column for target cell type
        score_col: Column to aggregate
        agg_func: Aggregation function

    Returns:
        DataFrame with aggregated scores per cell type pair
    """
    grouped = liana_results.groupby([source_col, target_col])

    if agg_func == "count":
        aggregated = grouped.size().reset_index(name="interaction_count")
    else:
        aggregated = grouped[score_col].agg(agg_func).reset_index()
        aggregated.columns = [source_col, target_col, f"{score_col}_{agg_func}"]

    return aggregated


def liana_to_adjacency_matrix(
    liana_results: pd.DataFrame,
    cell_types: list[str],
    source_col: str = "source",
    target_col: str = "target",
    score_col: str = "magnitude_rank",
    fill_value: float = 1.0,
) -> np.ndarray:
    """
    Convert LIANA+ results to adjacency matrix.

    Only includes edges with valid magnitude_rank values (not NaN, in [0, 1]).
    This matches the filtering in build_subject_ccc_features for consistency.

    Args:
        liana_results: LIANA+ output
        cell_types: Ordered list of cell types
        source_col: Column for source cell type
        target_col: Column for target cell type
        score_col: Column for edge weights (lower is better for ranks)
        fill_value: Value for missing edges (and invalid edges)

    Returns:
        Adjacency matrix [n_cell_types, n_cell_types]
    """
    n_types = len(cell_types)
    ct_to_idx = {ct: idx for idx, ct in enumerate(cell_types)}

    # Initialize with fill value (no interaction)
    adj_matrix = np.full((n_types, n_types), fill_value, dtype=np.float32)

    # Fill in observed interactions
    for _, row in liana_results.iterrows():
        src = row[source_col]
        tgt = row[target_col]

        if src in ct_to_idx and tgt in ct_to_idx:
            score = row.get(score_col)

            # Skip invalid magnitude_rank values - consistent with edge building
            if score is None or pd.isna(score):
                continue
            if score < 0.0 or score > 1.0:
                continue

            src_idx = ct_to_idx[src]
            tgt_idx = ct_to_idx[tgt]

            # For ranks, keep minimum (most significant)
            adj_matrix[src_idx, tgt_idx] = min(adj_matrix[src_idx, tgt_idx], score)

    return adj_matrix


# ─────────────────────────────────────────────────────────────────────────────
# Per-Subject LIANA Processing
# ─────────────────────────────────────────────────────────────────────────────


def run_liana_per_subject(
    adata: AnnData,
    subject_column: str = "ROSMAP_IndividualID",
    cell_type_column: str = "supercluster_name",
    resource_name: str = "CellChatDB",
    min_cells_per_type: int = 10,
    cache_dir: str | Path | None = None,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Run LIANA+ analysis for each subject separately.

    Args:
        adata: Preprocessed AnnData
        subject_column: Column containing subject IDs
        cell_type_column: Column containing cell type labels
        resource_name: Ligand-receptor resource
        min_cells_per_type: Minimum cells per cell type for analysis
        cache_dir: Directory to cache results (optional)
        verbose: Print progress

    Returns:
        Dictionary mapping subject_id -> LIANA results DataFrame
    """
    subjects = adata.obs[subject_column].unique()
    results = {}

    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    for i, subject_id in enumerate(subjects):
        if verbose:
            print(f"Processing subject {i+1}/{len(subjects)}: {subject_id}")

        # Check cache
        if cache_dir:
            cache_file = cache_dir / f"liana_{subject_id}.parquet"
            if cache_file.exists():
                results[subject_id] = pd.read_parquet(cache_file)
                if verbose:
                    print(f"  Loaded from cache")
                continue

        # Subset to subject
        subject_mask = adata.obs[subject_column] == subject_id
        adata_subject = adata[subject_mask].copy()

        # Check cell type counts
        ct_counts = adata_subject.obs[cell_type_column].value_counts()
        valid_cts = ct_counts[ct_counts >= min_cells_per_type].index.tolist()

        if len(valid_cts) < 2:
            if verbose:
                print(f"  Skipping: only {len(valid_cts)} cell types with >= {min_cells_per_type} cells")
            results[subject_id] = pd.DataFrame()
            continue

        # Filter to valid cell types
        adata_subject = adata_subject[adata_subject.obs[cell_type_column].isin(valid_cts)]

        try:
            liana_result = run_liana_analysis(
                adata_subject,
                cell_type_column=cell_type_column,
                resource_name=resource_name,
                min_cells=min_cells_per_type,
                verbose=False,
            )
            liana_result["subject_id"] = subject_id
            results[subject_id] = liana_result

            # Cache results
            if cache_dir:
                liana_result.to_parquet(cache_file)

        except Exception as e:
            if verbose:
                print(f"  Error: {e}")
            results[subject_id] = pd.DataFrame()

    return results


def build_subject_ccc_features(
    liana_results: pd.DataFrame,
    cell_types: list[str],
    edge_types: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """
    Build CCC feature matrices for a single subject.

    Args:
        liana_results: LIANA+ results for one subject
        cell_types: Ordered list of cell types
        edge_types: List of edge type names (default: ALL_EDGE_TYPES)

    Returns:
        Dictionary with:
        - 'adjacency': [n_cell_types, n_cell_types] interaction strength (raw magnitude_rank)
        - 'edge_index': [2, n_edges] edge indices
        - 'edge_type': [n_edges] edge type indices
        - 'edge_attr': [n_edges, 1] edge attributes (1.0 - magnitude_rank, so higher = stronger)

    Note:
        edge_attr is inverted from LIANA+'s magnitude_rank convention:
        - LIANA+: lower magnitude_rank = stronger interaction (0 = strongest)
        - edge_attr: higher value = stronger interaction (1 = strongest)
        This matches HGTConvWithEdgeAttr's expectation that higher edge values
        should increase attention and message strength.
    """
    if edge_types is None:
        edge_types = ALL_EDGE_TYPES

    n_types = len(cell_types)
    ct_to_idx = {ct: idx for idx, ct in enumerate(cell_types)}
    et_to_idx = {et: idx for idx, et in enumerate(edge_types)}

    # Assign edge types if not already done
    if "edge_type_name" not in liana_results.columns:
        liana_results = assign_edge_types(liana_results)

    # Build edge lists
    edge_src = []
    edge_dst = []
    edge_type_list = []
    edge_attr_list = []

    for _, row in liana_results.iterrows():
        src = row.get("source", "")
        tgt = row.get("target", "")
        et_name = row.get("edge_type_name", EDGE_TYPE_NOVEL)

        if src not in ct_to_idx or tgt not in ct_to_idx:
            continue

        # Validate magnitude_rank before adding edge
        # Skip edges with invalid scores - only include edges where LIANA+ has
        # full confidence. This is more conservative than imputing missing values.
        #
        # Why skip rather than impute:
        #   - NaN means LIANA+ couldn't compute reliable statistics (sparse data,
        #     too few cells, numerical issues)
        #   - Including with neutral weight (0.5) would treat uncertain edges
        #     as "average" - inflating low-quality data
        #   - Including with weak weight (0.0) preserves topology but adds noise
        #   - Skipping is cleanest: only trust edges with valid LIANA+ scores
        #
        # Why skip out-of-range values:
        #   - magnitude_rank should be in [0, 1] by LIANA+ definition
        #   - Values outside this range indicate data quality issues
        #   - Rather than clamp and guess, skip these unreliable edges
        magnitude_rank = row.get("magnitude_rank")
        if magnitude_rank is None or pd.isna(magnitude_rank):
            continue
        if magnitude_rank < 0.0 or magnitude_rank > 1.0:
            continue

        edge_src.append(ct_to_idx[src])
        edge_dst.append(ct_to_idx[tgt])
        edge_type_list.append(et_to_idx.get(et_name, et_to_idx[EDGE_TYPE_NOVEL]))

        # Transform magnitude_rank to edge attribute
        # LIANA+ convention: lower magnitude_rank = stronger interaction (0 = strongest)
        # HGT convention: higher edge_attr = more attention/influence
        # Solution: invert so that stronger interactions get higher values
        edge_attr = 1.0 - float(magnitude_rank)  # Now higher = stronger
        edge_attr_list.append([edge_attr])

    # Convert to arrays
    if len(edge_src) > 0:
        edge_index = np.array([edge_src, edge_dst], dtype=np.int64)
        edge_type = np.array(edge_type_list, dtype=np.int64)
        edge_attr = np.array(edge_attr_list, dtype=np.float32)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_type = np.zeros((0,), dtype=np.int64)
        edge_attr = np.zeros((0, 1), dtype=np.float32)

    # Also build dense adjacency matrix
    adjacency = liana_to_adjacency_matrix(
        liana_results, cell_types,
        fill_value=1.0  # No interaction = rank 1.0
    )

    return {
        "adjacency": adjacency,
        "edge_index": edge_index,
        "edge_type": edge_type,
        "edge_attr": edge_attr,
        "n_edges": len(edge_src),
    }
