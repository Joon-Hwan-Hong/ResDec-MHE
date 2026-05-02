"""
Gene set enrichment analysis on gene gate weights.

Uses two complementary tools:
1. **decoupler-py** — ULM + consensus scoring against MSigDB (Hallmark, KEGG,
   Reactome, GO:BP), PROGENy (14 pathways), and CollecTRI (TF networks)
2. **gseapy** — pre-ranked GSEA for leading edge genes

Input: gene gate weights [n_cell_types, n_genes] with continuous scores (0-1).
Analysis runs per cell type across all gene set collections.

Output format: Tidy DataFrames saved as Parquet (primary) and CSV (human-readable).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

import decoupler as dc
import gseapy as gp

from src.utils.io import save_dataframe

logger = logging.getLogger(__name__)

# MSigDB collection name mappings.
# Keys are the user-facing names used in gene_set_collections;
# values are the collection identifiers used in decoupler's MSigDB DataFrame.
MSIGDB_COLLECTION_MAP: dict[str, str] = {
    "hallmark": "hallmark",
    "kegg": "kegg_pathways",
    "reactome": "reactome_pathways",
    "go_bp": "go_biological_process",
}

# Default gene set collections to query.
DEFAULT_GENE_SET_COLLECTIONS: list[str] = [
    "hallmark",
    "kegg",
    "reactome",
    "go_bp",
]


@dataclass
class GeneEnrichmentResult:
    """
    Container for gene enrichment analysis results.

    Attributes:
        decoupler_scores: DataFrame with columns
            [cell_type, source, score, pvalue, method, collection]
        gsea_results: DataFrame with columns
            [cell_type, term, es, nes, pvalue, fdr, leading_edge, collection]
        consensus: DataFrame with columns
            [cell_type, source, consensus_score, consensus_pvalue, collection]
        metadata: Additional analysis metadata
    """

    decoupler_scores: pd.DataFrame
    gsea_results: pd.DataFrame
    consensus: pd.DataFrame
    metadata: dict = field(default_factory=dict)


def _fetch_msigdb_network(
    collection_key: str,
) -> pd.DataFrame:
    """
    Fetch an MSigDB gene set collection as a decoupler network DataFrame.

    Args:
        collection_key: One of the keys in MSIGDB_COLLECTION_MAP.

    Returns:
        DataFrame with columns [source, target, weight] suitable for
        decoupler methods.

    Raises:
        ValueError: If collection_key is not recognized.
    """
    if collection_key not in MSIGDB_COLLECTION_MAP:
        raise ValueError(
            f"Unknown MSigDB collection: {collection_key}. "
            f"Available: {list(MSIGDB_COLLECTION_MAP.keys())}"
        )

    msigdb = dc.op.resource("MSigDB", organism="human")
    collection_id = MSIGDB_COLLECTION_MAP[collection_key]

    # Filter to the desired collection.
    # MSigDB resource has columns: genesymbol, geneset, collection, ...
    filtered = msigdb[
        msigdb["collection"].str.lower().str.contains(
            collection_id.replace("_", " "), case=False, na=False
        )
    ].copy()

    if filtered.empty:
        # Fallback: try exact match
        filtered = msigdb[msigdb["collection"] == collection_id].copy()

    if filtered.empty:
        logger.warning(
            "No gene sets found for collection '%s' (mapped to '%s'). "
            "Available collections: %s",
            collection_key,
            collection_id,
            msigdb["collection"].unique().tolist()[:20],
        )
        return pd.DataFrame(columns=["source", "target", "weight"])

    # Rename to decoupler network format
    net = filtered.rename(
        columns={"geneset": "source", "genesymbol": "target"}
    )
    if "weight" not in net.columns:
        net["weight"] = 1.0

    return net[["source", "target", "weight"]].drop_duplicates()


def _fetch_progeny_network() -> pd.DataFrame:
    """Fetch PROGENy pathway network (14 signaling pathways with signed weights)."""
    net = dc.op.progeny(organism="human")
    return net


def _fetch_collectri_network() -> pd.DataFrame:
    """Fetch CollecTRI TF regulatory network."""
    net = dc.op.collectri(organism="human")
    return net


def _convert_net_to_gseapy_dict(net: pd.DataFrame) -> dict[str, list[str]]:
    """
    Convert a decoupler network DataFrame to a gseapy gene_sets dict.

    Args:
        net: DataFrame with at least 'source' and 'target' columns.

    Returns:
        Dict mapping source (gene set name) to list of target genes.
    """
    gene_sets: dict[str, list[str]] = {}
    for source, group in net.groupby("source"):
        gene_sets[source] = group["target"].unique().tolist()
    return gene_sets


class GeneEnrichmentAnalyzer:
    """
    Analyze gene gate weights via gene set enrichment.

    Runs decoupler ULM + consensus and gseapy prerank per cell type against
    MSigDB, PROGENy, and CollecTRI gene set collections.

    Example:
        >>> analyzer = GeneEnrichmentAnalyzer(
        ...     gene_gate_weights=weights,  # [n_cell_types, n_genes]
        ...     gene_names=gene_names,
        ...     cell_type_names=cell_type_names,
        ... )
        >>> result = analyzer.analyze()
        >>> analyzer.save(result, output_dir)
    """

    def __init__(
        self,
        gene_gate_weights: np.ndarray,
        gene_names: list[str],
        cell_type_names: list[str] | None = None,
        gene_set_collections: list[str] | None = None,
    ):
        """
        Initialize analyzer with gene gate weights.

        Args:
            gene_gate_weights: Gene gate attention weights [n_cell_types, n_genes].
                Values should be continuous (0-1).
            gene_names: List of gene names corresponding to columns.
            cell_type_names: Cell type names corresponding to rows.
                Defaults to ["cell_type_0", "cell_type_1", ...].
            gene_set_collections: Which gene set collections to analyze.
                Defaults to DEFAULT_GENE_SET_COLLECTIONS (hallmark, kegg, reactome, go_bp).
                Additional special collections: "progeny", "collectri".
        """
        if gene_gate_weights.ndim != 2:
            raise ValueError(
                f"gene_gate_weights must be 2D [n_cell_types, n_genes], "
                f"got shape {gene_gate_weights.shape}"
            )

        self.gene_gate_weights = gene_gate_weights
        self.n_cell_types, self.n_genes = gene_gate_weights.shape
        self.gene_names = gene_names

        if len(gene_names) != self.n_genes:
            raise ValueError(
                f"gene_names has {len(gene_names)} entries but "
                f"gene_gate_weights has {self.n_genes} genes"
            )

        self.cell_type_names = cell_type_names or [
            f"cell_type_{i}" for i in range(self.n_cell_types)
        ]
        if len(self.cell_type_names) != self.n_cell_types:
            raise ValueError(
                f"cell_type_names has {len(self.cell_type_names)} entries but "
                f"gene_gate_weights has {self.n_cell_types} cell types"
            )

        self.gene_set_collections = gene_set_collections or list(
            DEFAULT_GENE_SET_COLLECTIONS
        )

    def analyze(self, top_k_ora: int = 100) -> GeneEnrichmentResult:
        """
        Run gene enrichment analysis across all collections and cell types.

        Args:
            top_k_ora: Number of top genes per cell type for ORA (unused by
                the current ULM + prerank workflow, retained for interface
                compatibility).

        Returns:
            GeneEnrichmentResult containing decoupler scores, GSEA results,
            and consensus scores.
        """
        # Build the weight matrix as DataFrame for decoupler
        mat = pd.DataFrame(
            self.gene_gate_weights,
            index=self.cell_type_names,
            columns=self.gene_names,
        )

        all_decoupler_rows: list[dict] = []
        all_gsea_rows: list[dict] = []
        all_consensus_rows: list[dict] = []

        # Process each gene set collection
        for collection in self.gene_set_collections:
            logger.info("Processing collection: %s", collection)

            net = self._fetch_network(collection)
            if net.empty:
                logger.warning(
                    "Empty network for collection '%s' — skipping", collection
                )
                continue

            # --- decoupler: ULM + MLM + consensus ---
            dc_rows, cons_rows = self._run_decoupler(
                mat, net, collection
            )
            all_decoupler_rows.extend(dc_rows)
            all_consensus_rows.extend(cons_rows)

            # --- gseapy: prerank per cell type ---
            gsea_rows = self._run_gseapy_prerank(net, collection)
            all_gsea_rows.extend(gsea_rows)

        # Assemble result DataFrames
        decoupler_scores = pd.DataFrame(all_decoupler_rows)
        gsea_results = pd.DataFrame(all_gsea_rows)
        consensus = pd.DataFrame(all_consensus_rows)

        metadata = {
            "n_cell_types": self.n_cell_types,
            "n_genes": self.n_genes,
            "collections": self.gene_set_collections,
            "top_k_ora": top_k_ora,
        }

        return GeneEnrichmentResult(
            decoupler_scores=decoupler_scores,
            gsea_results=gsea_results,
            consensus=consensus,
            metadata=metadata,
        )

    def _fetch_network(self, collection: str) -> pd.DataFrame:
        """
        Fetch the gene set network for a given collection.

        Args:
            collection: Collection key (e.g., "hallmark", "progeny", "collectri").

        Returns:
            Network DataFrame with [source, target, weight] columns.
        """
        if collection == "progeny":
            return _fetch_progeny_network()
        elif collection == "collectri":
            return _fetch_collectri_network()
        elif collection in MSIGDB_COLLECTION_MAP:
            return _fetch_msigdb_network(collection)
        else:
            raise ValueError(
                f"Unknown gene set collection: {collection}. "
                f"Available: {list(MSIGDB_COLLECTION_MAP.keys()) + ['progeny', 'collectri']}"
            )

    def _run_decoupler(
        self,
        mat: pd.DataFrame,
        net: pd.DataFrame,
        collection: str,
    ) -> tuple[list[dict], list[dict]]:
        """
        Run decoupler ULM + MLM and consensus on the weight matrix.

        Args:
            mat: Weight matrix (cell_types x genes) as DataFrame.
            net: Network DataFrame with [source, target, weight] columns.
            collection: Collection name for labeling results.

        Returns:
            Tuple of (decoupler_rows, consensus_rows) as lists of dicts.
        """
        decoupler_rows: list[dict] = []
        consensus_rows: list[dict] = []

        # Run decouple with ULM + MLM for consensus
        try:
            result = dc.mt.decouple(
                mat, net=net, methods=["ulm", "mlm"], tmin=5, verbose=False
            )
        except (AssertionError, ValueError) as e:
            logger.warning(
                "decoupler failed for collection '%s': %s", collection, e
            )
            return decoupler_rows, consensus_rows

        # Extract per-method scores
        for method in ["ulm", "mlm"]:
            score_key = f"score_{method}"
            padj_key = f"padj_{method}"
            if score_key not in result or padj_key not in result:
                continue

            scores_df = result[score_key]
            pvals_df = result[padj_key]

            for cell_type in scores_df.index:
                for source in scores_df.columns:
                    decoupler_rows.append({
                        "cell_type": cell_type,
                        "source": source,
                        "score": float(scores_df.loc[cell_type, source]),
                        "pvalue": float(pvals_df.loc[cell_type, source]),
                        "method": method,
                        "collection": collection,
                    })

        # Run consensus across methods
        try:
            cons_scores, cons_pvals = dc.mt.consensus(result)
            for cell_type in cons_scores.index:
                for source in cons_scores.columns:
                    consensus_rows.append({
                        "cell_type": cell_type,
                        "source": source,
                        "consensus_score": float(
                            cons_scores.loc[cell_type, source]
                        ),
                        "consensus_pvalue": float(
                            cons_pvals.loc[cell_type, source]
                        ),
                        "collection": collection,
                    })
        except (ValueError, KeyError) as e:
            logger.warning(
                "Consensus failed for collection '%s': %s", collection, e
            )

        return decoupler_rows, consensus_rows

    def _run_gseapy_prerank(
        self,
        net: pd.DataFrame,
        collection: str,
    ) -> list[dict]:
        """
        Run gseapy prerank per cell type.

        Args:
            net: Network DataFrame with [source, target, weight] columns.
            collection: Collection name for labeling results.

        Returns:
            List of result dicts with GSEA fields.
        """
        # Convert network to gseapy dict format
        gene_sets = _convert_net_to_gseapy_dict(net)

        if not gene_sets:
            logger.warning(
                "No gene sets for gseapy prerank in collection '%s'",
                collection,
            )
            return []

        gsea_rows: list[dict] = []

        for ct_idx, ct_name in enumerate(self.cell_type_names):
            # Ranked gene list: gene_name -> gate_weight
            rnk = pd.Series(
                self.gene_gate_weights[ct_idx],
                index=self.gene_names,
            ).sort_values(ascending=False)

            try:
                result = gp.prerank(
                    rnk=rnk,
                    gene_sets=gene_sets,
                    min_size=5,
                    max_size=500,
                    permutation_num=1000,
                    seed=42,
                    no_plot=True,
                    verbose=False,
                    threads=1,
                )
            except (ValueError, IndexError, RuntimeError, KeyError) as e:
                # Narrow except: gseapy.prerank raises ValueError for empty
                # rankings, IndexError for all-zero genes, and RuntimeError
                # for various network/cache failures. KeyboardInterrupt and
                # SystemExit propagate normally.
                logger.warning(
                    "gseapy prerank failed for cell type '%s', "
                    "collection '%s': %s",
                    ct_name,
                    collection,
                    e,
                )
                continue

            if result.res2d is None or result.res2d.empty:
                continue

            for _, row in result.res2d.iterrows():
                gsea_rows.append({
                    "cell_type": ct_name,
                    "term": row["Term"],
                    "es": float(row["ES"]),
                    "nes": float(row["NES"]),
                    "pvalue": float(row["NOM p-val"]),
                    "fdr": float(row["FDR q-val"]),
                    "leading_edge": row.get("Lead_genes", ""),
                    "collection": collection,
                })

        return gsea_rows

    def save(
        self,
        result: GeneEnrichmentResult,
        output_dir: str | Path,
        formats: list[Literal["parquet", "csv"]] | None = None,
    ) -> dict[str, Path]:
        """
        Save analysis results to files.

        Args:
            result: GeneEnrichmentResult to save.
            output_dir: Directory for output files.
            formats: Output formats (default: ["parquet", "csv"]).

        Returns:
            Dict mapping output name to file path.
        """
        if formats is None:
            formats = ["parquet", "csv"]

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files: dict[str, Path] = {}

        # Save decoupler scores
        if not result.decoupler_scores.empty:
            for fmt in formats:
                path = output_dir / f"decoupler_scores.{fmt}"
                save_dataframe(result.decoupler_scores, path, fmt)
                saved_files[f"decoupler_scores_{fmt}"] = path

        # Save GSEA results
        if not result.gsea_results.empty:
            for fmt in formats:
                path = output_dir / f"gsea_results.{fmt}"
                save_dataframe(result.gsea_results, path, fmt)
                saved_files[f"gsea_results_{fmt}"] = path

        # Save consensus scores
        if not result.consensus.empty:
            for fmt in formats:
                path = output_dir / f"consensus_scores.{fmt}"
                save_dataframe(result.consensus, path, fmt)
                saved_files[f"consensus_scores_{fmt}"] = path

        # Save metadata as JSON
        meta_path = output_dir / "gene_enrichment_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(result.metadata, f, indent=2, default=str)
        saved_files["metadata"] = meta_path

        logger.info("Saved gene enrichment analysis to %s", output_dir)
        return saved_files


def compute_gene_enrichment(
    gene_gate_weights: np.ndarray,
    gene_names: list[str],
    cell_type_names: list[str] | None = None,
    gene_set_collections: list[str] | None = None,
    top_k_ora: int = 100,
    output_dir: str | Path | None = None,
    formats: list[Literal["parquet", "csv"]] | None = None,
) -> GeneEnrichmentResult:
    """
    Convenience function to compute and optionally save gene enrichment analysis.

    Args:
        gene_gate_weights: Gene gate attention weights [n_cell_types, n_genes].
        gene_names: List of gene names.
        cell_type_names: Cell type names (defaults to generic names).
        gene_set_collections: Which collections to use
            (default: hallmark, kegg, reactome, go_bp).
        top_k_ora: Number of top genes for ORA (interface compatibility).
        output_dir: If provided, save results to this directory.
        formats: Output formats (default: ["parquet", "csv"]).

    Returns:
        GeneEnrichmentResult with analysis results.
    """
    analyzer = GeneEnrichmentAnalyzer(
        gene_gate_weights=gene_gate_weights,
        gene_names=gene_names,
        cell_type_names=cell_type_names,
        gene_set_collections=gene_set_collections,
    )

    result = analyzer.analyze(top_k_ora=top_k_ora)

    if output_dir is not None:
        analyzer.save(result, output_dir, formats=formats)

    return result
