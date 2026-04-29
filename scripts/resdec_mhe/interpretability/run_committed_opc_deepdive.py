"""F8 deep-dive: Committed OPC × {LMOD3, LRAT, PCDHGB4, SCN10A} effect sizes.

For each of the 4 genes that anchor Committed OPC (multi-method convergence
at the perm-null floor p=1/1001), reports Cohen's d, log2 fold change,
mean expression in resilient vs vulnerable, and the existing Wilcoxon
p / padj / rank-biserial / Storey q from the canonical DE table.

Inputs (canonical):
  - residual_per_subject.csv (resilient/vulnerable groupings via top/bottom
    quartile of residual)
  - data/precomputed/{R<id>}.pt pseudobulk
  - de_resilient_vs_vulnerable/CT_03_de.csv (Wilcoxon stats)
  - de_storey_and_permutation/perm_pvalues_per_ct_top50.csv

Output JSON:
  outputs/canonical/interpretability/committed_opc_stable_genes_deepdive.json

CPU only; pathlib + argparse.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from src.analysis.pseudobulk_io import load_pseudobulk_matrix
from src.data.constants import CELL_TYPE_ORDER

logger = logging.getLogger(__name__)

GENES = ["LMOD3", "LRAT", "PCDHGB4", "SCN10A"]
TARGET_CT_NAME = "Committed oligodendrocyte precursor"


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d (pooled SD) for resilient (a) vs vulnerable (b)."""
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    n_a, n_b = len(a), len(b)
    s_a = a.var(ddof=1)
    s_b = b.var(ddof=1)
    pooled_sd = np.sqrt(((n_a - 1) * s_a + (n_b - 1) * s_b) / (n_a + n_b - 2))
    if pooled_sd == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled_sd)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--residual-csv",
        default="outputs/canonical/interpretability/residual_per_subject.csv",
    )
    p.add_argument("--precomputed-dir", default="data/precomputed")
    p.add_argument("--gene-names-npy", default="data/precomputed/gene_names.npy")
    p.add_argument(
        "--de-csv",
        default="outputs/canonical/interpretability/de_resilient_vs_vulnerable/CT_03_de.csv",
    )
    p.add_argument(
        "--perm-csv",
        default="outputs/canonical/interpretability/de_storey_and_permutation/perm_pvalues_per_ct_top50.csv",
    )
    p.add_argument(
        "--out-json",
        default="outputs/canonical/interpretability/committed_opc_stable_genes_deepdive.json",
    )
    p.add_argument("--quartile-fraction", type=float, default=0.25)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    root = _WORKTREE_ROOT
    residual_csv = (root / args.residual_csv).resolve()
    precomp_dir = (root / args.precomputed_dir).resolve()
    gene_names_npy = (root / args.gene_names_npy).resolve()
    de_csv = (root / args.de_csv).resolve()
    perm_csv = (root / args.perm_csv).resolve()
    out_json = (root / args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)

    # Reproduce DE-orchestrator quartile split.
    res_df = pd.read_csv(residual_csv)
    id_col = "ROSMAP_IndividualID" if "ROSMAP_IndividualID" in res_df.columns else res_df.columns[0]
    res_df = res_df.rename(columns={id_col: "subject_id"})
    finite = np.isfinite(res_df["residual"])
    q_lo = res_df.loc[finite, "residual"].quantile(args.quartile_fraction)
    q_hi = res_df.loc[finite, "residual"].quantile(1 - args.quartile_fraction)
    res_df["group"] = "middle"
    res_df.loc[res_df["residual"] >= q_hi, "group"] = "resilient"
    res_df.loc[res_df["residual"] <= q_lo, "group"] = "vulnerable"
    keep = res_df[res_df["group"].isin(("resilient", "vulnerable"))].copy()
    n_res = int((keep["group"] == "resilient").sum())
    n_vul = int((keep["group"] == "vulnerable").sum())
    logger.info("split: %d resilient + %d vulnerable", n_res, n_vul)

    keep_ids = keep["subject_id"].astype(str).tolist()
    is_resilient = (keep["group"] == "resilient").to_numpy()

    pb = load_pseudobulk_matrix(precomp_dir, keep_ids)
    n_subj, n_ct, n_gene = pb.shape
    logger.info("pseudobulk shape: %s", pb.shape)

    gene_names = list(np.load(gene_names_npy, allow_pickle=True))
    if len(gene_names) != n_gene:
        raise RuntimeError(f"gene_names length {len(gene_names)} != n_gene {n_gene}")

    if n_ct != len(CELL_TYPE_ORDER):
        raise RuntimeError(
            f"n_ct={n_ct} mismatches CELL_TYPE_ORDER len={len(CELL_TYPE_ORDER)}"
        )
    ct_idx = list(CELL_TYPE_ORDER).index(TARGET_CT_NAME)
    logger.info("target CT '%s' index: %d", TARGET_CT_NAME, ct_idx)

    # Existing DE rows for Committed OPC.
    de_df = pd.read_csv(de_csv)
    de_by_gene = {row["gene"]: row for _, row in de_df.iterrows()}

    perm_df = pd.read_csv(perm_csv)
    perm_df = perm_df[perm_df["cell_type"] == TARGET_CT_NAME]
    perm_by_gene = {row["gene"]: row for _, row in perm_df.iterrows()}

    expr_ct = pb[:, ct_idx, :]  # (n_subj_kept, n_gene)
    finite_subj_mask = np.isfinite(expr_ct).all(axis=1)
    if not finite_subj_mask.all():
        n_drop = int((~finite_subj_mask).sum())
        logger.warning("dropping %d subjects with NaN rows in CT %d", n_drop, ct_idx)
    expr_ct = expr_ct[finite_subj_mask]
    is_resilient_ct = is_resilient[finite_subj_mask]
    logger.info(
        "CT %d expression for %d subjects (res=%d, vul=%d)",
        ct_idx, expr_ct.shape[0], int(is_resilient_ct.sum()),
        int((~is_resilient_ct).sum()),
    )

    per_gene = []
    for g in GENES:
        if g not in gene_names:
            logger.error("gene %s not in HVG panel", g)
            per_gene.append({"gene": g, "in_hvg_panel": False})
            continue
        g_idx = gene_names.index(g)
        x = expr_ct[:, g_idx]
        x_res = x[is_resilient_ct]
        x_vul = x[~is_resilient_ct]
        d = cohens_d(x_res, x_vul)
        # log2 FC on log1p data is the difference of means in log1p space
        # (linearized ratio). The DE table also uses input_scale=log1p so
        # mean diff is what we report.
        log2_fc_via_means = float(x_res.mean() - x_vul.mean())
        de_row = de_by_gene.get(g)
        perm_row = perm_by_gene.get(g)
        per_gene.append({
            "gene": g,
            "in_hvg_panel": True,
            "gene_index": int(g_idx),
            "cell_type_index": ct_idx,
            "cell_type": TARGET_CT_NAME,
            "n_resilient": int(is_resilient_ct.sum()),
            "n_vulnerable": int((~is_resilient_ct).sum()),
            "mean_resilient_log1p": float(x_res.mean()),
            "mean_vulnerable_log1p": float(x_vul.mean()),
            "std_resilient_log1p": float(x_res.std(ddof=1)),
            "std_vulnerable_log1p": float(x_vul.std(ddof=1)),
            "cohens_d": d,
            "log2_fc_recomputed_diffmeans_log1p": log2_fc_via_means,
            "log2_fc_from_de_table": (
                float(de_row["log2_fold_change"]) if de_row is not None else None
            ),
            "p_value_wilcoxon": (
                float(de_row["p_value"]) if de_row is not None else None
            ),
            "padj_fdr_wilcoxon": (
                float(de_row["padj_fdr"]) if de_row is not None else None
            ),
            "rank_biserial": (
                float(de_row["rank_biserial"]) if de_row is not None else None
            ),
            "p_perm_1000": (
                float(perm_row["p_perm"]) if perm_row is not None else None
            ),
            "wilcoxon_U_observed": (
                float(perm_row["wilcoxon_U_observed"]) if perm_row is not None else None
            ),
        })
        logger.info(
            "%s: d=%.3f, log2FC=%.4f (DE_table), mean_res=%.4f, mean_vul=%.4f, "
            "p_wilcoxon=%.2e, padj=%.3f, p_perm=%.4f",
            g, d,
            de_row["log2_fold_change"] if de_row is not None else float("nan"),
            x_res.mean(), x_vul.mean(),
            de_row["p_value"] if de_row is not None else float("nan"),
            de_row["padj_fdr"] if de_row is not None else float("nan"),
            perm_row["p_perm"] if perm_row is not None else float("nan"),
        )

    out = {
        "cell_type": TARGET_CT_NAME,
        "cell_type_index": ct_idx,
        "n_resilient": int(is_resilient_ct.sum()),
        "n_vulnerable": int((~is_resilient_ct).sum()),
        "quartile_fraction": args.quartile_fraction,
        "input_scale": "log1p",
        "genes": per_gene,
        "provenance": {
            "residual_csv": str(residual_csv),
            "precomputed_dir": str(precomp_dir),
            "gene_names_npy": str(gene_names_npy),
            "de_csv": str(de_csv),
            "perm_csv": str(perm_csv),
        },
    }
    out_json.write_text(json.dumps(out, indent=2))
    logger.info("wrote %s", out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
