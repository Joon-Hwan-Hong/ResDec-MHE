"""Distributional resilience analyses on raw per-subject pseudobulk.

Model-free complement to ``run_resilience_analyses.py`` (which operates on
Captum attributions). Here we treat raw pseudobulk as the signal and
answer: does resilient vs. vulnerable expression differ distributionally
(Wasserstein) or is any gene reproducibly selected across subject
resamples (stability selection)?

Subcommands (each writes a JSON to ``--out-dir``):

    wasserstein  — per cell type, per gene Wasserstein-1 distance between
                   resilient and vulnerable raw expression distributions.

    stability    — per cell type stability selection (|rank-biserial| ≥
                   threshold in ≥ pi_threshold of resamples).

Inputs (defaults; CLI-overridable):
    --residual-csv    outputs/canonical/interpretability/residual_per_subject.csv
    --precomputed-dir data/precomputed
    --gene-names-npy  data/precomputed/gene_names.npy
    --cell-type-names-source outputs/canonical/interpretability/captum_ig/composite_attribution_summary.json
    --out-dir         outputs/canonical/interpretability/distributional_resilience

Provenance JSON lists input paths, quartile config, n_resilient /
n_vulnerable, seed, and git SHA.
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
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from src.analysis.pseudobulk_io import load_pseudobulk_matrix
from src.analysis.resilience_distributional import (
    stability_selection,
    wasserstein_per_celltype,
)
from src.data.constants import CELL_TYPE_ORDER
from src.utils.provenance import git_sha

logger = logging.getLogger(__name__)


def _load_cell_type_names(src_path: Path, n_ct: int) -> list[str]:
    """Return axis-aligned cell-type names.

    Uses the authoritative ``src.data.constants.CELL_TYPE_ORDER`` — this is
    the same CT index convention used everywhere else in the codebase
    (pseudobulk loader, model forward, Captum orchestrator). The
    ``src_path`` argument is accepted for backward compatibility but
    ignored: captum_summary.json's ``cell_types_ranked_by_total_attribution``
    is NOT axis-aligned (it's sorted by attribution magnitude).
    """
    del src_path  # unused; kept for backward compat.
    if n_ct != len(CELL_TYPE_ORDER):
        logger.warning(
            "n_ct=%d != len(CELL_TYPE_ORDER)=%d; truncating/padding",
            n_ct, len(CELL_TYPE_ORDER),
        )
    return list(CELL_TYPE_ORDER[:n_ct])


def _split_resilient_vs_vulnerable(
    residual_csv: Path, quartile_fraction: float,
) -> tuple[list[str], np.ndarray, int, int]:
    df = pd.read_csv(residual_csv)
    id_col = (
        "ROSMAP_IndividualID"
        if "ROSMAP_IndividualID" in df.columns
        else df.columns[0]
    )
    df = df.rename(columns={id_col: "subject_id"})
    finite = np.isfinite(df["residual"])
    q_lo = df.loc[finite, "residual"].quantile(quartile_fraction)
    q_hi = df.loc[finite, "residual"].quantile(1 - quartile_fraction)
    df["group"] = "middle"
    df.loc[df["residual"] >= q_hi, "group"] = "resilient"
    df.loc[df["residual"] <= q_lo, "group"] = "vulnerable"
    keep = df[df["group"].isin(("resilient", "vulnerable"))].copy()
    ids = keep["subject_id"].astype(str).tolist()
    is_resilient = (keep["group"] == "resilient").to_numpy()
    return ids, is_resilient, int(is_resilient.sum()), int((~is_resilient).sum())


def _run_wasserstein(args: argparse.Namespace) -> dict:
    ids, is_resilient, n_res, n_vul = _split_resilient_vs_vulnerable(
        Path(args.residual_csv), args.quartile_fraction,
    )
    logger.info(
        "wasserstein split: %d resilient + %d vulnerable", n_res, n_vul,
    )
    pb = load_pseudobulk_matrix(Path(args.precomputed_dir), ids)
    _, n_ct, n_gene = pb.shape
    logger.info("pseudobulk loaded: %s", pb.shape)
    gene_names = list(np.load(args.gene_names_npy, allow_pickle=True))
    if len(gene_names) != n_gene:
        logger.warning(
            "gene_names length %d != n_gene %d; using placeholders",
            len(gene_names), n_gene,
        )
        gene_names = [f"gene_{j}" for j in range(n_gene)]
    ct_names = _load_cell_type_names(Path(args.cell_type_names_source), n_ct)
    t0 = time.time()
    result = wasserstein_per_celltype(
        pb, is_resilient,
        cell_type_names=ct_names, gene_names=gene_names,
    )
    result["provenance"] = {
        "analysis": "wasserstein",
        "source": "pseudobulk",
        "quartile_fraction": args.quartile_fraction,
        "n_resilient": n_res,
        "n_vulnerable": n_vul,
        "n_cell_types": int(n_ct),
        "n_genes": int(n_gene),
        "precomputed_dir": str(args.precomputed_dir),
        "elapsed_s": round(time.time() - t0, 1),
        "git_commit": git_sha(_WORKTREE_ROOT),
    }
    return result


def _run_stability(args: argparse.Namespace) -> dict:
    ids, is_resilient, n_res, n_vul = _split_resilient_vs_vulnerable(
        Path(args.residual_csv), args.quartile_fraction,
    )
    logger.info(
        "stability split: %d resilient + %d vulnerable", n_res, n_vul,
    )
    pb = load_pseudobulk_matrix(Path(args.precomputed_dir), ids)
    _, n_ct, n_gene = pb.shape
    logger.info("pseudobulk loaded: %s", pb.shape)
    gene_names = list(np.load(args.gene_names_npy, allow_pickle=True))
    if len(gene_names) != n_gene:
        logger.warning(
            "gene_names length %d != n_gene %d; using placeholders",
            len(gene_names), n_gene,
        )
        gene_names = [f"gene_{j}" for j in range(n_gene)]
    ct_names = _load_cell_type_names(Path(args.cell_type_names_source), n_ct)
    t0 = time.time()
    per_ct: list[dict] = []
    for ct in range(n_ct):
        expr = pb[:, ct, :]
        ok = np.isfinite(expr).all(axis=1)
        if ok.sum() < 8:
            logger.warning("CT %d: too few finite rows; skipping", ct)
            continue
        expr_ok = expr[ok]
        is_res_ok = is_resilient[ok]
        if is_res_ok.sum() < 3 or (~is_res_ok).sum() < 3:
            continue
        res = stability_selection(
            expr_ok, is_res_ok,
            n_bootstrap=args.n_bootstrap,
            subsample_frac=args.subsample_frac,
            rb_threshold=args.rb_threshold,
            pi_threshold=args.pi_threshold,
            seed=args.seed,
            gene_names=gene_names,
        )
        per_ct.append({
            "cell_type_index": ct,
            "cell_type": str(ct_names[ct]),
            "n_stable": len(res["stable_indices"]),
            "stable_genes": res["stable_genes"],
        })
    return {
        "per_cell_type": per_ct,
        "provenance": {
            "analysis": "stability_selection",
            "source": "pseudobulk",
            "quartile_fraction": args.quartile_fraction,
            "n_resilient": n_res,
            "n_vulnerable": n_vul,
            "n_cell_types": int(n_ct),
            "n_genes": int(n_gene),
            "n_bootstrap": args.n_bootstrap,
            "subsample_frac": args.subsample_frac,
            "rb_threshold": args.rb_threshold,
            "pi_threshold": args.pi_threshold,
            "seed": args.seed,
            "elapsed_s": round(time.time() - t0, 1),
            "git_commit": git_sha(_WORKTREE_ROOT),
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "subcommand", choices=["wasserstein", "stability"],
        help="Which analysis to run.",
    )
    p.add_argument(
        "--residual-csv",
        default="outputs/canonical/interpretability/residual_per_subject.csv",
    )
    p.add_argument("--precomputed-dir", default="data/precomputed")
    p.add_argument("--gene-names-npy", default="data/precomputed/gene_names.npy")
    p.add_argument(
        "--cell-type-names-source",
        default="outputs/canonical/interpretability/captum_ig/"
        "composite_attribution_summary.json",
    )
    p.add_argument(
        "--out-dir",
        default="outputs/canonical/interpretability/distributional_resilience",
    )
    p.add_argument("--quartile-fraction", type=float, default=0.25)
    # stability-specific:
    p.add_argument("--n-bootstrap", type=int, default=100)
    p.add_argument("--subsample-frac", type=float, default=0.5)
    p.add_argument("--rb-threshold", type=float, default=0.2)
    p.add_argument("--pi-threshold", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.subcommand == "wasserstein":
        result = _run_wasserstein(args)
        out_path = out_dir / "wasserstein_per_celltype_pseudobulk.json"
    else:
        result = _run_stability(args)
        out_path = out_dir / "stability_selection_pseudobulk.json"
    out_path.write_text(json.dumps(result, indent=2))
    logger.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
