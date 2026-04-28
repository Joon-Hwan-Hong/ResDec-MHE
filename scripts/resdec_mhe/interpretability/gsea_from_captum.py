"""GSEA adapter on Captum-IG top-attributed gene lists.

Reads the Captum composite-attribution summary produced by
``captum_composite_attribution.py`` and runs Enrichr gene-set over-representation
against Hallmark / Reactome / KEGG, plus a hypergeometric overlap against a
curated AD-GWAS reference set (Bellenguez 2022 + Wightman 2021).

Four gene lists are queried:

1. ``top_<K>_global`` — top-K global genes by mean |attribution| across all
   subjects and cell types (default K=200).
2. ``top_<K_ct>_<CellType>`` — top-K genes per each of the top-3 cell types
   ranked by ``cell_types_ranked_by_total_attribution`` in the summary
   (default K=50 per cell type).

For each gene list × each database, the script records the top-10 enriched
terms (by adjusted p-value) in a per-database CSV and a combined JSON summary.
A separate ``ad_gwas_overlap.csv`` reports the AD-GWAS overlap size,
overlap genes, and one-sided hypergeometric p-value per gene list.

**AD-GWAS gene list (manually curated).**

The ``AD_GWAS_GENES`` constant below combines the lead genes from two
recent, widely-cited AD GWAS meta-analyses:

- Bellenguez C. et al. **"New insights into the genetic etiology of Alzheimer's
  disease and related dementias."** *Nat Genet* 54, 412–436 (2022). Reports
  75 AD/ADRD-associated loci (mostly new). Gene assignments here are the
  proximal / eQTL-prioritised genes reported in their Table 1 / Supplementary
  Table 5.
- Wightman D.P. et al. **"A genome-wide association study with 1,126,563
  individuals identifies new risk loci for Alzheimer's disease."** *Nat Genet*
  53, 1276–1282 (2021). Reports 38 genome-wide-significant loci.

The list is the **union** of lead genes from both papers, deduplicated and
uppercased to HUGO convention. Entries are confirmed AD risk genes only;
no speculative hits were added. For the hypergeometric test we restrict the
GWAS set to those genes present in the 4785 ROSMAP HVG universe that the
model actually saw.

Usage
-----

    PYTHONPATH=<worktree-root> \\
    uv run python scripts/resdec_mhe/interpretability/gsea_from_captum.py \\
        --captum-npz outputs/canonical/interpretability/captum_ig/composite_attributions.npz \\
        --out-dir outputs/canonical/interpretability/gsea

Outputs (default ``outputs/canonical/interpretability/gsea/``):

- ``gsea_<database>_<list_name>.csv`` — per (list, database) top-10 terms.
- ``gsea_summary.json`` — all (list, database) top-10 terms in one file.
- ``ad_gwas_overlap.csv`` — per-list AD-GWAS overlap + hypergeometric p-value.
- ``gene_lists.json`` — the actual gene lists used (for reproducibility).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

# Bootstrap sys.path so we can import src.* from the worktree root.
_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if (_WORKTREE_ROOT / "src").is_dir() and str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AD-GWAS reference gene list (Bellenguez 2022 + Wightman 2021, curated)
# ---------------------------------------------------------------------------

# Union of reported lead genes from:
#   - Bellenguez 2022 (Nat Genet, 75 loci, Table 1 / Supp Table 5 gene assignments)
#   - Wightman 2021  (Nat Genet, 38 loci, Table 1 gene assignments)
# Deduplicated, uppercased, confirmed AD risk genes only.
AD_GWAS_GENES: frozenset[str] = frozenset({
    # Core well-established (in both papers)
    "APOE", "APP", "PSEN1", "PSEN2",
    # APOE / TOMM40 / APOC1 cluster (chr19q13)
    "TOMM40", "APOC1",
    # Immune / microglial
    "TREM2", "CD33", "MS4A6A", "MS4A4A", "CR1", "ABCA7", "CLU",
    "HLA-DRB1", "HLA-DRB5", "INPP5D", "PLCG2", "SPI1",
    # Endocytosis / trafficking
    "BIN1", "PICALM", "SORL1", "PTK2B", "CD2AP", "EPHA1",
    "USP6NL", "ECHDC3", "RIN3", "AP4E1", "AP4M1", "BLNK",
    # Lipid metabolism / cholesterol
    "ABCA1", "LILRB2",
    # Tau / synaptic
    "CASS4", "FERMT2", "SLC24A4", "CELF1", "NME8",
    # Bellenguez 2022 novel loci (selected, high-confidence gene assignments)
    "ADAM17", "ADAMTS1", "ADAM10", "ANKH", "APH1B", "BCKDK", "CLNK",
    "COX7C", "CTSB", "CTSH", "DOC2A", "EED", "FOXF1", "GRN",
    "HAVCR2", "HS3ST5", "ICA1", "IDUA", "IL34", "IQCK", "JAZF1",
    "KAT8", "KLF16", "LILRB4", "MAF", "MAPT", "MINDY2", "MME",
    "MYO15A", "NCK2", "PLEKHA1", "PRDM7", "PRKD3", "RBCK1",
    "RHOH", "SCIMP", "SEC61G", "SHARPIN", "SIGLEC11", "SIGLEC14",
    "SNX1", "SORT1", "TMEM106B", "TNIP1", "TPCN1", "TSPAN14",
    "TSPOAP1", "UMAD1", "UNC5CL", "USP6NL", "WDR12", "WDR81",
    "WNT3", "ZCWPW1",
    # Wightman 2021 additions (non-overlapping with Bellenguez)
    "ADAMTS4", "HESX1", "CNTNAP2", "AGRN", "KAT8",
})


# ---------------------------------------------------------------------------
# Gene-list helpers (pure functions, tested in
#   tests/unit/analysis/test_gsea_from_captum.py)
# ---------------------------------------------------------------------------


def rank_cell_types_from_summary(
    summary: dict, top_n: int
) -> list[str]:
    """Return the top-N cell types by total |attribution|.

    Reads ``cell_types_ranked_by_total_attribution`` from the Captum summary.
    """
    ranked = summary["cell_types_ranked_by_total_attribution"]
    names = [entry["cell_type"] for entry in ranked[:top_n]]
    return names


def build_gene_list_from_summary(
    summary: dict,
    scope: str,
    top_k: int,
    cell_type: str | None = None,
) -> list[str]:
    """Extract a top-K gene list from the Captum summary.

    Args:
        summary: Parsed ``composite_attribution_summary.json`` dict.
        scope: ``"global"`` (reads ``top_global_genes``) or ``"cell_type"``
            (reads ``top_genes_per_cell_type[cell_type]``).
        top_k: Number of genes to return (truncated to what's available).
        cell_type: Required when ``scope == "cell_type"``.

    Returns:
        List of HUGO gene symbols, preserving the ordering in the summary
        (which is descending by ``mean_abs_attribution``).

    Raises:
        ValueError: Invalid scope or missing cell_type for cell_type scope.
        KeyError: cell_type not present in summary.
    """
    if scope == "global":
        entries = summary["top_global_genes"]
    elif scope == "cell_type":
        if cell_type is None:
            raise ValueError("cell_type must be provided when scope='cell_type'")
        per_ct = summary["top_genes_per_cell_type"]
        if cell_type not in per_ct:
            raise KeyError(
                f"Cell type {cell_type!r} not in top_genes_per_cell_type. "
                f"Available: {list(per_ct.keys())}"
            )
        entries = per_ct[cell_type]
    else:
        raise ValueError(
            f"scope must be 'global' or 'cell_type', got {scope!r}"
        )

    return [entry["gene"] for entry in entries[:top_k]]


def build_gene_list_from_npz(
    attributions: np.ndarray,
    gene_names: list[str],
    cell_type_names: list[str],
    scope: str,
    top_k: int,
    cell_type: str | None = None,
) -> list[str]:
    """Extract a top-K gene list directly from the Captum attributions tensor.

    Unlike ``build_gene_list_from_summary``, this is not limited by the
    summary's pre-computed top-N — it operates on the raw ``[N, C, G]``
    tensor and can produce arbitrarily-long top-K lists.

    Ranking rule: ``mean_abs_attribution`` averaged across subjects (first
    axis). For ``scope="global"``, we further average across cell types.
    For ``scope="cell_type"``, we select that one cell-type slice.

    Args:
        attributions: ``[N_subjects, N_cell_types, N_genes]`` array.
        gene_names: Length ``N_genes`` list of HUGO symbols.
        cell_type_names: Length ``N_cell_types`` list of cell-type names.
        scope: ``"global"`` or ``"cell_type"``.
        top_k: Number of top genes to return.
        cell_type: Required when ``scope == "cell_type"``.

    Returns:
        Top-K HUGO symbols in descending order of mean |attribution|.

    Raises:
        ValueError: Invalid scope or shape mismatch.
        KeyError: ``cell_type`` not in ``cell_type_names``.
    """
    if attributions.ndim != 3:
        raise ValueError(
            f"attributions must be 3D [N, C, G], got shape {attributions.shape}"
        )
    n_sub, n_ct, n_genes = attributions.shape
    if len(gene_names) != n_genes:
        raise ValueError(
            f"gene_names has {len(gene_names)} entries but attributions has "
            f"{n_genes} genes"
        )
    if len(cell_type_names) != n_ct:
        raise ValueError(
            f"cell_type_names has {len(cell_type_names)} entries but "
            f"attributions has {n_ct} cell types"
        )

    abs_attr = np.abs(attributions)
    if scope == "global":
        per_gene = abs_attr.mean(axis=(0, 1))  # [G]
    elif scope == "cell_type":
        if cell_type is None:
            raise ValueError(
                "cell_type must be provided when scope='cell_type'"
            )
        if cell_type not in cell_type_names:
            raise KeyError(
                f"Cell type {cell_type!r} not in cell_type_names. "
                f"Available: {cell_type_names}"
            )
        ct_idx = cell_type_names.index(cell_type)
        per_gene = abs_attr[:, ct_idx, :].mean(axis=0)  # [G]
    else:
        raise ValueError(
            f"scope must be 'global' or 'cell_type', got {scope!r}"
        )

    top_k_clipped = min(top_k, n_genes)
    # argsort descending (stable, so ties preserve gene order)
    order = np.argsort(-per_gene, kind="stable")[:top_k_clipped]
    return [gene_names[i] for i in order]


# ---------------------------------------------------------------------------
# AD-GWAS hypergeometric overlap
# ---------------------------------------------------------------------------


def hypergeometric_overlap_pvalue(
    overlap: int,
    sample_size: int,
    n_successes_pop: int,
    pop_size: int,
) -> float:
    """One-sided hypergeometric p-value for observing >= ``overlap``.

    Computes ``P(X >= overlap)`` where ``X ~ Hypergeometric(N=pop_size,
    K=n_successes_pop, n=sample_size)``.

    Uses ``scipy.stats.hypergeom.sf(overlap - 1, ...)`` which is numerically
    exact (no chaining through CDF).

    Args:
        overlap: Number of successes observed (e.g., #AD-GWAS ∩ top-K).
        sample_size: Size of the draw (e.g., top-K gene-list size).
        n_successes_pop: Number of successes in population (e.g., AD-GWAS
            genes present in the HVG universe).
        pop_size: Population size (e.g., 4785 HVG genes).

    Raises:
        ValueError: Any invalid combination (negative counts,
            overlap > sample_size, n_successes_pop > pop_size, etc.).
    """
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if sample_size < 0:
        raise ValueError(f"sample_size must be >= 0, got {sample_size}")
    if n_successes_pop < 0 or n_successes_pop > pop_size:
        raise ValueError(
            f"n_successes_pop must be in [0, pop_size={pop_size}], "
            f"got {n_successes_pop}"
        )
    if overlap > sample_size:
        raise ValueError(
            f"overlap ({overlap}) cannot exceed sample_size ({sample_size})"
        )
    if overlap > n_successes_pop:
        raise ValueError(
            f"overlap ({overlap}) cannot exceed n_successes_pop "
            f"({n_successes_pop})"
        )
    if sample_size > pop_size:
        raise ValueError(
            f"sample_size ({sample_size}) cannot exceed pop_size ({pop_size})"
        )

    if overlap == 0:
        return 1.0
    # P(X >= overlap) = sf(overlap - 1)
    return float(hypergeom.sf(overlap - 1, pop_size, n_successes_pop, sample_size))


def compute_ad_gwas_overlap(
    gene_list: Iterable[str],
    universe: Iterable[str],
    gwas_genes: Iterable[str],
) -> dict:
    """Compute AD-GWAS overlap + hypergeometric p-value for a gene list.

    Restricts both ``gene_list`` and ``gwas_genes`` to the universe before
    counting; genes outside the HVG universe cannot contribute to overlap.

    Args:
        gene_list: Attribution top-K gene symbols (e.g., top 200 global).
        universe: Full set of genes the model saw (the 4785 ROSMAP HVGs).
        gwas_genes: Reference AD-GWAS gene set.

    Returns:
        Dict with keys:
            - ``n_overlap`` (int)
            - ``overlap_genes`` (sorted list[str])
            - ``p_hypergeometric`` (float)
            - ``sample_size`` (int): size of gene_list in universe
            - ``n_gwas_in_universe`` (int)
            - ``universe_size`` (int)
    """
    universe_set = set(universe)
    gwas_in_universe = set(gwas_genes) & universe_set
    gene_list_in_universe = [g for g in gene_list if g in universe_set]
    overlap_set = set(gene_list_in_universe) & gwas_in_universe

    sample_size = len(gene_list_in_universe)
    n_overlap = len(overlap_set)
    n_gwas_in_universe = len(gwas_in_universe)
    pop_size = len(universe_set)

    p = hypergeometric_overlap_pvalue(
        overlap=n_overlap,
        sample_size=sample_size,
        n_successes_pop=n_gwas_in_universe,
        pop_size=pop_size,
    )

    return {
        "n_overlap": n_overlap,
        "overlap_genes": sorted(overlap_set),
        "p_hypergeometric": p,
        "sample_size": sample_size,
        "n_gwas_in_universe": n_gwas_in_universe,
        "universe_size": pop_size,
    }


# ---------------------------------------------------------------------------
# Enrichr runner (network-hitting; not unit-tested)
# ---------------------------------------------------------------------------


def run_enrichr_for_gene_list(
    gene_list: list[str],
    database: str,
    background: list[str] | None = None,
    min_overlap: int = 3,
) -> pd.DataFrame:
    """Run Enrichr (via gseapy) for a gene list against one database.

    Args:
        gene_list: Top-K gene symbols.
        database: Enrichr library name, e.g. ``"MSigDB_Hallmark_2020"``.
        background: Full HVG universe. NOTE: when ``background`` is supplied
            alongside an Enrichr library name, gseapy switches to its *local*
            over-representation path (hypergeometric vs. the supplied
            background, not the web Enrichr default). That path drops the
            ``Overlap`` column but yields a more biologically appropriate
            statistic — the HVG universe is what the model actually saw, not
            the global 20k-gene Enrichr background.
        min_overlap: Minimum gene overlap to include a term in the output.
            Filters using the ``Genes`` column when ``Overlap`` is absent.

    Returns:
        DataFrame with columns:
            ``term``, ``overlap``, ``p_value``, ``adjusted_p_value``,
            ``odds_ratio``, ``combined_score``, ``genes``, ``database``.
        Sorted ascending by ``adjusted_p_value``.
    """
    import gseapy as gp

    empty_schema = [
        "term", "overlap", "p_value", "adjusted_p_value",
        "odds_ratio", "combined_score", "genes", "database",
    ]

    logger.info(
        "Enrichr: list of %d genes vs %s%s",
        len(gene_list),
        database,
        f" (background={len(background)})" if background else "",
    )
    try:
        enr = gp.enrichr(
            gene_list=list(gene_list),
            gene_sets=database,
            organism="human",
            background=background,
            outdir=None,
            no_plot=True,
            verbose=False,
        )
    except Exception as e:
        logger.error(
            "Enrichr failed for database=%s: %s", database, e
        )
        return pd.DataFrame(columns=empty_schema)

    if enr.results is None or enr.results.empty:
        return pd.DataFrame(columns=empty_schema)

    df = enr.results.copy()

    # Two possible output schemas depending on whether ``background`` was
    # passed. Standard Enrichr path has an ``Overlap`` column ("k/n"),
    # local background path has no ``Overlap`` column but still has
    # ``Genes`` (semicolon-separated).
    if "Overlap" in df.columns:
        def _parse_overlap(s: str) -> int:
            return int(str(s).split("/")[0])
        df["_overlap_int"] = df["Overlap"].apply(_parse_overlap)
    else:
        def _count_genes(s: str) -> int:
            if pd.isna(s) or s == "":
                return 0
            return len(str(s).split(";"))
        df["_overlap_int"] = df["Genes"].apply(_count_genes)
        df["Overlap"] = df["_overlap_int"].astype(str)

    df = df[df["_overlap_int"] >= min_overlap].copy()
    df = df.drop(columns=["_overlap_int"])

    # Normalise column names to lower_snake_case.
    # NOTE: gseapy uses 'Old adjusted P-value' (lowercase 'a') when background
    # is supplied vs. 'Old Adjusted P-value' without — not relevant to us but
    # documented here.
    df = df.rename(columns={
        "Term": "term",
        "Overlap": "overlap",
        "P-value": "p_value",
        "Adjusted P-value": "adjusted_p_value",
        "Odds Ratio": "odds_ratio",
        "Combined Score": "combined_score",
        "Genes": "genes",
    })
    keep = [
        "term", "overlap", "p_value", "adjusted_p_value",
        "odds_ratio", "combined_score", "genes",
    ]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()
    df["database"] = database
    df = df.sort_values("adjusted_p_value").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "GSEA (Enrichr) + AD-GWAS overlap for Captum-IG top-attributed "
            "gene lists."
        )
    )
    p.add_argument(
        "--captum-npz",
        type=Path,
        default=Path(
            "outputs/canonical/interpretability/captum_ig/"
            "composite_attributions.npz"
        ),
        help=(
            "Path to composite_attributions.npz (only used for metadata — "
            "the gene lists come from the summary JSON)."
        ),
    )
    p.add_argument(
        "--captum-summary",
        type=Path,
        default=Path(
            "outputs/canonical/interpretability/captum_ig/"
            "composite_attribution_summary.json"
        ),
        help="Path to composite_attribution_summary.json.",
    )
    p.add_argument(
        "--gene-names-npy",
        type=Path,
        default=Path("data/precomputed/gene_names.npy"),
        help="Path to gene_names.npy (HVG universe).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/canonical/interpretability/gsea"),
        help="Output directory.",
    )
    p.add_argument(
        "--top-k-global",
        type=int,
        default=200,
        help="Top-K genes for the global list.",
    )
    p.add_argument(
        "--top-k-per-celltype",
        type=int,
        default=50,
        help="Top-K genes per per-cell-type list.",
    )
    p.add_argument(
        "--n-top-celltypes",
        type=int,
        default=3,
        help=(
            "Number of top cell types (by total |attribution|) to run "
            "per-cell-type enrichment on."
        ),
    )
    p.add_argument(
        "--databases",
        nargs="+",
        default=[
            "MSigDB_Hallmark_2020",
            "Reactome_2022",
            "KEGG_2021_Human",
        ],
        help="Enrichr library names.",
    )
    p.add_argument(
        "--min-overlap",
        type=int,
        default=3,
        help="Minimum gene overlap to report an Enrichr term.",
    )
    p.add_argument(
        "--top-terms-per-db",
        type=int,
        default=10,
        help="Top-N terms (by adjusted p) to keep per (list, database).",
    )
    p.add_argument(
        "--skip-enrichr",
        action="store_true",
        help=(
            "Skip Enrichr network calls; run AD-GWAS overlap only. Useful "
            "for offline smoke tests."
        ),
    )
    return p.parse_args()


def _safe_list_name(name: str) -> str:
    """Sanitise a cell-type or scope name for use in file paths."""
    bad = " /\\,()[]"
    out = name
    for ch in bad:
        out = out.replace(ch, "_")
    # Collapse repeated underscores.
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = _parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load inputs ------------------------------------------------------
    logger.info("Loading summary: %s", args.captum_summary)
    summary = json.loads(args.captum_summary.read_text())

    logger.info("Loading gene names: %s", args.gene_names_npy)
    gene_names_arr = np.load(args.gene_names_npy, allow_pickle=True)
    gene_names = [str(g) for g in gene_names_arr]
    universe = set(gene_names)
    logger.info(
        "HVG universe: %d genes; AD-GWAS reference: %d genes (%d in universe)",
        len(universe),
        len(AD_GWAS_GENES),
        len(AD_GWAS_GENES & universe),
    )

    # Load the attributions tensor — enables arbitrary top-K beyond what the
    # summary pre-computes. Cell-type order matches ``CELL_TYPE_ORDER`` in
    # the canonical model (verified against the summary JSON).
    from src.data.constants import CELL_TYPE_ORDER

    logger.info("Loading attributions npz: %s", args.captum_npz)
    npz = np.load(args.captum_npz, allow_pickle=True)
    attributions = npz["attributions"]  # [N, C, G]
    logger.info("attributions shape: %s", attributions.shape)
    if attributions.shape[-1] != len(gene_names):
        raise RuntimeError(
            f"attributions last dim = {attributions.shape[-1]} but "
            f"gene_names has {len(gene_names)} entries — mismatch."
        )
    if attributions.shape[1] != len(CELL_TYPE_ORDER):
        raise RuntimeError(
            f"attributions middle dim = {attributions.shape[1]} but "
            f"CELL_TYPE_ORDER has {len(CELL_TYPE_ORDER)} entries — mismatch."
        )

    # --- Build the gene lists --------------------------------------------
    gene_lists: dict[str, list[str]] = {}

    k_global = args.top_k_global
    gene_lists[f"top_{k_global}_global"] = build_gene_list_from_npz(
        attributions=attributions,
        gene_names=gene_names,
        cell_type_names=list(CELL_TYPE_ORDER),
        scope="global",
        top_k=k_global,
    )

    top_cts = rank_cell_types_from_summary(summary, top_n=args.n_top_celltypes)
    logger.info("Top %d cell types: %s", len(top_cts), top_cts)
    k_ct = args.top_k_per_celltype
    for ct in top_cts:
        safe_ct = _safe_list_name(ct)
        key = f"top_{k_ct}_{safe_ct}"
        gene_lists[key] = build_gene_list_from_npz(
            attributions=attributions,
            gene_names=gene_names,
            cell_type_names=list(CELL_TYPE_ORDER),
            scope="cell_type",
            cell_type=ct,
            top_k=k_ct,
        )

    # Record gene lists for reproducibility
    (out_dir / "gene_lists.json").write_text(
        json.dumps(gene_lists, indent=2)
    )
    logger.info("Wrote gene_lists.json (%d lists)", len(gene_lists))

    # --- AD-GWAS overlap (always runs; no network) ------------------------
    overlap_rows: list[dict] = []
    for list_name, genes in gene_lists.items():
        r = compute_ad_gwas_overlap(
            gene_list=genes, universe=universe, gwas_genes=AD_GWAS_GENES
        )
        overlap_rows.append({
            "gene_list": list_name,
            "sample_size": r["sample_size"],
            "n_overlap": r["n_overlap"],
            "n_gwas_in_universe": r["n_gwas_in_universe"],
            "universe_size": r["universe_size"],
            "p_hypergeometric": r["p_hypergeometric"],
            "overlap_genes": ";".join(r["overlap_genes"]),
        })

    overlap_df = pd.DataFrame(overlap_rows)
    overlap_df.to_csv(out_dir / "ad_gwas_overlap.csv", index=False)
    logger.info(
        "Wrote ad_gwas_overlap.csv:\n%s",
        overlap_df[[
            "gene_list", "sample_size", "n_overlap", "p_hypergeometric"
        ]].to_string(index=False),
    )

    # --- Enrichr (network) -----------------------------------------------
    if args.skip_enrichr:
        logger.warning("Skipping Enrichr per --skip-enrichr")
        gsea_summary: dict[str, dict[str, list[dict]]] = {}
    else:
        gsea_summary = {}
        for list_name, genes in gene_lists.items():
            gsea_summary[list_name] = {}
            for db in args.databases:
                df = run_enrichr_for_gene_list(
                    gene_list=genes,
                    database=db,
                    background=gene_names,
                    min_overlap=args.min_overlap,
                )
                # Per-(list, db) CSV
                csv_path = (
                    out_dir
                    / f"gsea_{_safe_list_name(db)}_{list_name}.csv"
                )
                df.to_csv(csv_path, index=False)
                logger.info(
                    "Wrote %s (%d terms)", csv_path.name, len(df)
                )

                # Keep top-N for combined JSON
                top_df = df.head(args.top_terms_per_db)
                gsea_summary[list_name][db] = top_df.to_dict(orient="records")

        # Combined top-10 summary JSON
        (out_dir / "gsea_summary.json").write_text(
            json.dumps(gsea_summary, indent=2, default=str)
        )
        logger.info("Wrote gsea_summary.json")

    # --- Provenance -------------------------------------------------------
    provenance = {
        "captum_summary": str(args.captum_summary),
        "captum_npz": str(args.captum_npz),
        "gene_names_npy": str(args.gene_names_npy),
        "databases": args.databases,
        "top_k_global": k_global,
        "top_k_per_celltype": k_ct,
        "n_top_celltypes": args.n_top_celltypes,
        "min_overlap": args.min_overlap,
        "top_terms_per_db": args.top_terms_per_db,
        "ad_gwas_ref_size": len(AD_GWAS_GENES),
        "ad_gwas_in_universe": len(AD_GWAS_GENES & universe),
        "universe_size": len(universe),
        "n_gene_lists": len(gene_lists),
        "gene_list_names": list(gene_lists.keys()),
        "enrichr_skipped": bool(args.skip_enrichr),
    }
    (out_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2)
    )
    logger.info("Wrote provenance.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
