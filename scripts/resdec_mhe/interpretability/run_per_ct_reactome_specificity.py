"""Per-CT Reactome pathway specificity check (audit Finding 5 follow-up).

Tests whether the dominant "neurotransmitter-release SNARE-triad"
convergence seen for Splatter's top-50 genes is *Splatter-specific* or is
a generic artifact of the model's top-attribution gene rankings (i.e.,
also dominates other top-attribution cell types).

For each non-Splatter top cell type {Fibroblast, Committed OPC, Vascular,
MGE interneuron, Deep-IT}, this script:

1. Loads the Captum top-50 gene list for the CT from the canonical Captum
   composite-attribution summary
   (``outputs/canonical/interpretability/captum_ig/composite_attribution_summary.json``).
   This matches the gene-list source already used for the Splatter
   reference enrichment (``gsea_Reactome_2022_top_50_Splatter.csv``) so
   the per-CT comparison is apples-to-apples.

2. Runs ``gseapy.enrichr`` against ``Reactome_2022`` with the HVG universe
   as background (mirroring ``gsea_from_captum.py``).

3. Saves per-CT top-10 Reactome pathways to a single combined JSON, and
   also writes per-(CT, db) CSVs alongside the existing GSEA artifacts
   for reproducibility.

Outputs
-------
- ``--out-dir``/``per_ct_reactome_top10.json`` — combined per-CT top-10
  pathways with p_value + adjusted_p_value + overlap genes.
- ``--out-dir``/``gsea_Reactome_2022_top_50_<safe_ct>.csv`` (per CT) —
  full enrichment table (only non-Splatter CTs we add; Splatter is
  already in the gsea/ dir from the canonical gsea_from_captum.py run).
- ``--out-dir``/``per_ct_reactome_provenance.json``.

Cell-type axis-alignment
------------------------
We use ``CELL_TYPE_ORDER`` from ``src.data.constants`` (the canonical
axis-aligned list). The summary's
``cell_types_ranked_by_total_attribution`` is sorted by attribution
magnitude, NOT by axis index, so it is fine to use cell-type names from
either, but we stick to the same convention as
``gsea_from_captum.py``: we read ``top_genes_per_cell_type[<ct>]`` by
name.

Usage
-----

    PYTHONPATH=<worktree-root> \\
    uv run python scripts/resdec_mhe/interpretability/run_per_ct_reactome_specificity.py \\
        --captum-summary outputs/canonical/interpretability/captum_ig/composite_attribution_summary.json \\
        --gene-names-npy data/precomputed/gene_names.npy \\
        --out-dir outputs/canonical/interpretability/gsea
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if (_WORKTREE_ROOT / "src").is_dir() and str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from scripts.resdec_mhe.interpretability.gsea_from_captum import (  # noqa: E402
    _safe_list_name,
    build_gene_list_from_summary,
    run_enrichr_for_gene_list,
)

logger = logging.getLogger(__name__)


# Per the audit Finding 5, we test the 5 non-Splatter top cell types
# against the same Reactome library used for Splatter. Names match
# CELL_TYPE_ORDER / Captum summary keys exactly.
TARGET_CELL_TYPES: tuple[str, ...] = (
    "Fibroblast",
    "Committed oligodendrocyte precursor",  # task: "Committed OPC"
    "Vascular",
    "MGE interneuron",
    "Deep-layer intratelencephalic",  # task: "Deep-IT"
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
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
        help="Path to gene_names.npy (HVG universe used as Enrichr background).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/canonical/interpretability/gsea"),
        help="Output directory.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-K genes per CT (matches Splatter reference top-50).",
    )
    p.add_argument(
        "--database",
        default="Reactome_2022",
        help=(
            "Enrichr library name. Default Reactome_2022 mirrors the "
            "Splatter reference."
        ),
    )
    p.add_argument(
        "--min-overlap",
        type=int,
        default=3,
        help="Minimum gene overlap to report an Enrichr term.",
    )
    p.add_argument(
        "--top-terms",
        type=int,
        default=10,
        help="Top-N terms (by adjusted p) to keep per CT for combined JSON.",
    )
    p.add_argument(
        "--cell-types",
        nargs="+",
        default=list(TARGET_CELL_TYPES),
        help="Cell types to enrich (default: 5 non-Splatter top CTs).",
    )
    return p.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = _parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading Captum summary: %s", args.captum_summary)
    summary = json.loads(args.captum_summary.read_text())

    logger.info("Loading gene names (HVG universe): %s", args.gene_names_npy)
    gene_names = [str(g) for g in np.load(args.gene_names_npy, allow_pickle=True)]
    logger.info("HVG universe: %d genes", len(gene_names))

    per_ct_payload: dict[str, dict] = {}
    t0 = time.time()
    for ct in args.cell_types:
        logger.info("=== %s ===", ct)
        try:
            genes = build_gene_list_from_summary(
                summary, scope="cell_type", top_k=args.top_k, cell_type=ct,
            )
        except KeyError as e:
            logger.error("Skipping %s: %s", ct, e)
            continue
        logger.info("  top-%d genes head: %s", len(genes), genes[:5])

        df = run_enrichr_for_gene_list(
            gene_list=genes,
            database=args.database,
            background=gene_names,
            min_overlap=args.min_overlap,
        )
        safe_ct = _safe_list_name(ct)
        csv_path = (
            out_dir / f"gsea_{_safe_list_name(args.database)}_top_{args.top_k}_{safe_ct}.csv"
        )
        df.to_csv(csv_path, index=False)
        logger.info("  wrote %s (%d terms)", csv_path.name, len(df))

        # Top-N for combined JSON. Sort by raw p_value to match the
        # ranking convention used in MASTER-INFO §32.2 for Splatter.
        df_sorted = df.sort_values("p_value", kind="stable").head(args.top_terms)
        per_ct_payload[ct] = {
            "n_top_genes": len(genes),
            "top_genes_head_5": genes[:5],
            "n_terms_total": int(len(df)),
            "top_terms": df_sorted.to_dict(orient="records"),
        }

    combined = {
        "database": args.database,
        "top_k_genes_per_ct": args.top_k,
        "min_overlap": args.min_overlap,
        "top_terms_per_ct": args.top_terms,
        "cell_types": list(per_ct_payload.keys()),
        "per_cell_type": per_ct_payload,
        "elapsed_s": round(time.time() - t0, 1),
    }
    combined_path = out_dir / "per_ct_reactome_top10.json"
    combined_path.write_text(json.dumps(combined, indent=2, default=str))
    logger.info("Wrote %s", combined_path)

    # Provenance
    prov = {
        "captum_summary": str(args.captum_summary),
        "gene_names_npy": str(args.gene_names_npy),
        "database": args.database,
        "top_k_genes_per_ct": args.top_k,
        "min_overlap": args.min_overlap,
        "top_terms_per_ct": args.top_terms,
        "cell_types": list(per_ct_payload.keys()),
        "n_genes_universe": len(gene_names),
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out_dir / "per_ct_reactome_provenance.json").write_text(
        json.dumps(prov, indent=2)
    )
    logger.info("Wrote per_ct_reactome_provenance.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
