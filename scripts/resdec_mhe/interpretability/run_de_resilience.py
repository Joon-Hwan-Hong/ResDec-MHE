"""Differential expression: resilient vs vulnerable subjects, per cell type.

Loads per-subject pseudobulk from ``data/precomputed/R<subject_id>.pt``,
splits subjects into resilient (top quartile of residual) vs vulnerable
(bottom quartile), and runs DE per cell type via either Wilcoxon (default,
fast) or pydeseq2 (--method deseq2, slower).

Outputs (in --out-dir, default outputs/redesign/interpretability/de_resilient_vs_vulnerable/):
  - per-CT CSV: <celltype>_de.csv with columns from src.analysis.de_resilience
  - cross-CT summary: top_genes_per_ct.csv (top-20 sig genes per CT, padj<0.05)
  - provenance.json: run config + inputs + n subjects per group + git SHA
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from src.analysis.de_resilience import deseq2_de, wilcoxon_de
from src.analysis.pseudobulk_io import load_pseudobulk_matrix
from src.data.constants import CELL_TYPE_ORDER
from src.utils.provenance import git_sha

logger = logging.getLogger(__name__)


def _safe_filename(name: str) -> str:
    """Replace characters that are awkward in filenames (spaces, /, parens)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--residual-csv",
        default="outputs/redesign/interpretability/residual_per_subject.csv",
    )
    p.add_argument("--precomputed-dir", default="data/precomputed")
    p.add_argument("--gene-names-npy", default="data/precomputed/gene_names.npy")
    p.add_argument(
        "--out-dir",
        default="outputs/redesign/interpretability/de_resilient_vs_vulnerable",
    )
    p.add_argument(
        "--cell-type-names-source",
        default="outputs/redesign/interpretability/captum_ig/composite_attribution_summary.json",
        help="JSON containing cell_types_ranked_by_total_attribution.",
    )
    p.add_argument("--method", choices=["wilcoxon", "deseq2"], default="wilcoxon")
    p.add_argument("--quartile-fraction", type=float, default=0.25,
                   help="Top/bottom fraction of residual distribution (default 0.25).")
    p.add_argument(
        "--input-scale", choices=["counts", "log1p", "raw"], default="log1p",
        help="How to compute LFC (default log1p — pseudobulk is typically log-normalized).",
    )
    p.add_argument("--bootstrap-ci", type=int, default=None,
                   help="If set, bootstrap LFC CIs (Wilcoxon only).")
    p.add_argument("--top-k-export", type=int, default=20,
                   help="Top-K significant genes per CT to write to top_genes_per_ct.csv.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load residuals.
    res_df = pd.read_csv(args.residual_csv)
    id_col = "ROSMAP_IndividualID" if "ROSMAP_IndividualID" in res_df.columns else res_df.columns[0]
    res_df = res_df.rename(columns={id_col: "subject_id"})
    finite_mask = np.isfinite(res_df["residual"])
    q_lo = res_df.loc[finite_mask, "residual"].quantile(args.quartile_fraction)
    q_hi = res_df.loc[finite_mask, "residual"].quantile(1 - args.quartile_fraction)
    res_df["group"] = "middle"
    res_df.loc[res_df["residual"] >= q_hi, "group"] = "resilient"
    res_df.loc[res_df["residual"] <= q_lo, "group"] = "vulnerable"
    keep_df = res_df[res_df["group"].isin(("resilient", "vulnerable"))].copy()
    n_res = int((keep_df["group"] == "resilient").sum())
    n_vul = int((keep_df["group"] == "vulnerable").sum())
    logger.info(
        "DE split: %d resilient + %d vulnerable (dropped middle %d)",
        n_res, n_vul, len(res_df) - n_res - n_vul,
    )

    # Load pseudobulk for kept subjects only.
    keep_ids = keep_df["subject_id"].astype(str).tolist()
    is_resilient = (keep_df["group"] == "resilient").to_numpy()
    pb = load_pseudobulk_matrix(Path(args.precomputed_dir), keep_ids)
    n_subj, n_ct, n_gene = pb.shape
    logger.info("pseudobulk loaded: shape=%s", pb.shape)

    # Gene + cell type names.
    gene_names = list(np.load(args.gene_names_npy, allow_pickle=True))
    if len(gene_names) != n_gene:
        logger.warning(
            "gene_names length %d != n_gene %d; using placeholders",
            len(gene_names), n_gene,
        )
        gene_names = [f"gene_{j}" for j in range(n_gene)]
    # Axis-aligned CT names from the authoritative constant. The captum
    # summary JSON's "cell_types_ranked_by_total_attribution" is ordered
    # by attribution magnitude (not CT index) so we don't use it for
    # labeling — report it separately if needed.
    if n_ct != len(CELL_TYPE_ORDER):
        logger.warning(
            "n_ct=%d != len(CELL_TYPE_ORDER)=%d; truncating/padding",
            n_ct, len(CELL_TYPE_ORDER),
        )
    ct_names = list(CELL_TYPE_ORDER[:n_ct])
    src = Path(args.cell_type_names_source)
    if src.exists():
        s = json.loads(src.read_text())
        raw = s.get("cell_types_ranked_by_total_attribution") or s.get("cell_types")
        if isinstance(raw, list) and raw and isinstance(raw[0], dict):
            logger.info(
                "cell_types_ranked_by_attribution (top 5, NOT axis-aligned): %s",
                [d["cell_type"] for d in raw[:5]],
            )

    # Per-CT DE.
    per_ct_summary = []
    top_sig_rows = []        # padj < 0.05 (often empty for high-dim genomic data)
    top_by_pvalue_rows = []  # top-K by raw p-value (always populated)
    t_start = time.time()
    for ct in range(n_ct):
        expr = pb[:, ct, :]  # (n_subj, n_gene)
        if not np.isfinite(expr).any():
            logger.warning("CT %d: all-NaN expression; skipping", ct)
            continue
        if args.method == "wilcoxon":
            df = wilcoxon_de(
                expr, is_resilient, gene_names=gene_names,
                input_scale=args.input_scale,
                bootstrap_ci=args.bootstrap_ci,
                seed=args.seed,
            )
        else:
            try:
                df = deseq2_de(
                    expr, is_resilient, gene_names=gene_names, n_cpus=4,
                )
            except ImportError as exc:
                logger.error("DESeq2 not available; falling back to Wilcoxon for CT %d (%s)", ct, exc)
                df = wilcoxon_de(expr, is_resilient, gene_names=gene_names,
                                 input_scale=args.input_scale, seed=args.seed)
        # Write per-CT CSV.
        ct_label = f"CT_{ct:02d}"
        out_csv = out_dir / f"{ct_label}_de.csv"
        df.to_csv(out_csv, index=False)
        ct_name = ct_names[ct] if ct < len(ct_names) else ct_label
        # Top-K passing padj<0.05 (often empty at this dimensionality).
        sig = df[df["padj_fdr"] < 0.05].sort_values("padj_fdr").head(args.top_k_export)
        for _, row in sig.iterrows():
            top_sig_rows.append({
                "cell_type_index": ct,
                "cell_type": ct_name,
                "gene": row["gene"],
                "log2_fold_change": row["log2_fold_change"],
                "padj_fdr": row["padj_fdr"],
                "rank_biserial": row["rank_biserial"],
            })
        # Top-K by raw p-value (always populated; useful for Captum × DE
        # concordance even when no gene passes BH correction).
        top_p = df.sort_values("p_value").head(args.top_k_export)
        for _, row in top_p.iterrows():
            top_by_pvalue_rows.append({
                "cell_type_index": ct,
                "cell_type": ct_name,
                "gene": row["gene"],
                "log2_fold_change": row["log2_fold_change"],
                "p_value": row["p_value"],
                "padj_fdr": row["padj_fdr"],
                "rank_biserial": row["rank_biserial"],
            })
        per_ct_summary.append({
            "cell_type_index": ct,
            "cell_type": ct_name,
            "n_genes_tested": int(np.isfinite(df["p_value"]).sum()),
            "n_sig_padj005": int((df["padj_fdr"] < 0.05).sum()),
            "min_padj": float(df["padj_fdr"].min(skipna=True)),
            "min_pvalue": float(df["p_value"].min(skipna=True)),
        })
        if (ct + 1) % 5 == 0:
            elapsed = time.time() - t_start
            logger.info("DE: %d/%d CTs done in %.1fs", ct + 1, n_ct, elapsed)

    pd.DataFrame(top_sig_rows).to_csv(out_dir / "top_sig_genes_per_ct_padj005.csv", index=False)
    pd.DataFrame(top_by_pvalue_rows).to_csv(out_dir / "top_genes_per_ct_by_pvalue.csv", index=False)
    pd.DataFrame(per_ct_summary).to_csv(out_dir / "per_ct_summary.csv", index=False)

    provenance = {
        "method": args.method,
        "input_scale": args.input_scale,
        "bootstrap_ci": args.bootstrap_ci,
        "n_resilient": n_res,
        "n_vulnerable": n_vul,
        "quartile_fraction": args.quartile_fraction,
        "n_cell_types": int(n_ct),
        "n_genes": int(n_gene),
        "seed": args.seed,
        "git_commit": git_sha(_WORKTREE_ROOT),
        "elapsed_min": round((time.time() - t_start) / 60, 2),
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
    logger.info("wrote per-CT DE + summaries to %s", out_dir)
    logger.info("provenance: %s", provenance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
