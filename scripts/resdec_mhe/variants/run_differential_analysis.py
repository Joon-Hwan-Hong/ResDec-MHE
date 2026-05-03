"""DAE / DCR orchestrator: compare canonical Captum IG attributions to one or
more variants' Captum IG attributions, emit per (CT, gene) BH-FDR table + per-
method Spearman rank correlation tables.

Per src/analysis/differential.py:
- DAE = paired Wilcoxon per (CT, gene) on per-fold mean attribution magnitude.
- DCR = Spearman rho between canonical and variant per-method CT rank lists.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))

from src.analysis.differential import (  # noqa: E402
    differential_attribution_effect,
    differential_ct_ranking,
)
from src.data.constants import CELL_TYPE_ORDER  # noqa: E402
from src.utils.gene_names import load_gene_names  # noqa: E402


def _load_per_fold_mean_attribution(npz_path: Path) -> np.ndarray:
    """Stack per-fold mean attribution magnitude into shape (n_folds, n_ct, n_gene).

    composite_attributions.npz schema (per captum_composite_attribution.py):
      - attributions: (n_subjects, n_ct, n_gene)
      - fold: (n_subjects,) per-subject fold assignment
    Per-fold mean is computed by averaging |attribution| over subjects in each fold.
    """
    d = np.load(npz_path, allow_pickle=True)
    attr = d["attributions"]
    folds = d["fold"]
    fold_ids = sorted(set(int(f) for f in folds))
    out = []
    for fid in fold_ids:
        mask = folds == fid
        mean_abs = np.abs(attr[mask]).mean(axis=0)
        out.append(mean_abs)
    return np.stack(out, axis=0)


def _ct_ranking_from_attribution(npz_path: Path) -> list[str]:
    """Rank CTs by total absolute attribution mass (descending)."""
    d = np.load(npz_path, allow_pickle=True)
    attr = np.abs(d["attributions"])  # (n_subj, n_ct, n_gene)
    ct_total = attr.sum(axis=(0, 2))   # (n_ct,)
    n_ct = ct_total.shape[0]
    ct_names = list(CELL_TYPE_ORDER)[:n_ct]
    order = np.argsort(-ct_total)
    return [ct_names[i] for i in order]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--canonical-attr-npz", type=Path,
        default=_ROOT / "outputs/canonical/interpretability/captum_ig/composite_attributions.npz",
    )
    p.add_argument(
        "--variant-attr-npz", type=Path, action="append", required=True,
        help="Path to variant Captum IG composite_attributions.npz (repeatable).",
    )
    p.add_argument(
        "--variant-name", type=str, action="append", required=True,
        help="Tag for the corresponding --variant-attr-npz (repeat in same order).",
    )
    p.add_argument(
        "--precomputed-dir", type=Path,
        default=_ROOT / "data/precomputed",
        help="Directory holding gene_names.npy / feature_names.json for "
             "real-symbol lookup; falls back to gene_<i> placeholders.",
    )
    p.add_argument(
        "--out-dir", type=Path,
        default=_ROOT / "outputs/canonical/variants/differential",
    )
    args = p.parse_args()
    if len(args.variant_attr_npz) != len(args.variant_name):
        raise SystemExit(
            f"--variant-attr-npz count ({len(args.variant_attr_npz)}) "
            f"!= --variant-name count ({len(args.variant_name)})"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    canonical = _load_per_fold_mean_attribution(args.canonical_attr_npz)
    n_folds, n_ct, n_gene = canonical.shape
    ct_names = list(CELL_TYPE_ORDER)[:n_ct]
    gene_names, _ = load_gene_names(args.precomputed_dir, n_gene)

    canonical_rank = _ct_ranking_from_attribution(args.canonical_attr_npz)
    canonical_ranks = {"captum_ig": canonical_rank}

    summary = {
        "canonical_attr_npz": str(args.canonical_attr_npz),
        "n_folds": n_folds, "n_ct": n_ct, "n_gene": n_gene,
        "variants": {},
    }

    for vname, vpath in zip(args.variant_name, args.variant_attr_npz):
        variant = _load_per_fold_mean_attribution(vpath)
        if variant.shape != canonical.shape:
            raise SystemExit(
                f"variant {vname} shape {variant.shape} != canonical {canonical.shape}"
            )
        # DAE table
        dae = differential_attribution_effect(
            canonical, variant,
            ct_names=ct_names, gene_names=gene_names,
        )
        dae_path = args.out_dir / f"dae_canonical_vs_{vname}.csv"
        dae.to_csv(dae_path, index=False)

        # DCR
        variant_ranks = {"captum_ig": _ct_ranking_from_attribution(vpath)}
        dcr = differential_ct_ranking(canonical_ranks, variant_ranks)
        dcr_path = args.out_dir / f"dcr_canonical_vs_{vname}.json"
        dcr_path.write_text(json.dumps(dcr, indent=2))

        n_sig = int((dae["padj_bh"] < 0.05).sum())
        summary["variants"][vname] = {
            "variant_attr_npz": str(vpath),
            "dae_csv": str(dae_path),
            "dcr_json": str(dcr_path),
            "n_pairs_padj_lt_005": n_sig,
            "spearman_rho_captum_ig": dcr["captum_ig"]["spearman_rho"],
        }
        print(
            f"variant={vname}: DAE table written to {dae_path} "
            f"({n_sig}/{n_ct * n_gene} pairs at padj<0.05); "
            f"DCR Spearman rho (captum_ig) = {dcr['captum_ig']['spearman_rho']:+.4f}",
            flush=True,
        )

    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
