"""Orchestrator: run resilient-vs-vulnerable analyses on canonical artefacts.

Subcommands (each writes a JSON to outputs/redesign/interpretability/):

    latent_class            — Gaussian mixture on per-subject residuals
                              (BIC + AIC; tests whether resilience is
                               continuous vs discrete).

    wasserstein             — per cell type, per gene Wasserstein-1 distance
                              between resilient and vulnerable Captum-attribution
                              distributions. Flags cell types with the largest
                              distributional shift.

    stability               — stability selection over Captum attributions
                              (resampling subjects + per-gene rank-biserial)
                              with optional threshold-path sweep.

    cmi                     — conditional mutual information of cell-type-mean
                              Captum attribution with the resilience composite,
                              given pathology covariates (gpath, amyloid,
                              tangles).

Inputs (defaults; overridable via CLI):
  --canonical-dir   outputs/redesign/p5_canonical_seed42
  --captum-npz      outputs/redesign/interpretability/captum_ig/composite_attributions.npz
  --residual-csv    outputs/redesign/interpretability/residual_per_subject.csv
  --metadata-csv    data/metadata_ROSMAP/metadata.csv
  --out-dir         outputs/redesign/interpretability/

Note on attribution-as-proxy: these analyses operate on Captum attributions
(what the MODEL says is important) rather than on raw pseudobulk. This is
an interpretation of the model's internal signal, not a model-free DE test.
The DE module (de_resilience.py) handles raw-pseudobulk DE separately when
the orchestrator pulls counts from precomputed_dataset.pt.
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

from src.analysis.conditional_mi import conditional_mi_per_celltype  # noqa: E402
from src.analysis.resilience_distributional import (  # noqa: E402
    latent_class_on_residuals,
    stability_selection,
    wasserstein_per_celltype,
)


logger = logging.getLogger(__name__)


def _load_residuals(csv_path: Path) -> pd.DataFrame:
    """Load residual_per_subject.csv with subject_id + residual columns."""
    if not csv_path.exists():
        raise FileNotFoundError(f"residual CSV missing: {csv_path}")
    df = pd.read_csv(csv_path)
    id_col = "ROSMAP_IndividualID" if "ROSMAP_IndividualID" in df.columns else df.columns[0]
    df = df.rename(columns={id_col: "subject_id"})
    if "residual" not in df.columns:
        raise ValueError(f"residual column missing in {csv_path}; columns: {df.columns.tolist()}")
    return df[["subject_id", "residual"]]


def _load_captum(npz_path: Path) -> dict:
    """Load captum_ig/composite_attributions.npz, return dict of arrays."""
    if not npz_path.exists():
        raise FileNotFoundError(f"captum npz missing: {npz_path}")
    d = np.load(npz_path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _resilient_split(residuals: np.ndarray, fraction: float = 0.25) -> np.ndarray:
    """Top fraction = resilient (True), bottom fraction = vulnerable (False).

    Subjects in the middle are dropped via NaN sentinel.
    """
    finite = np.isfinite(residuals)
    q_lo, q_hi = np.quantile(residuals[finite], [fraction, 1 - fraction])
    is_res = np.full(residuals.shape, False, dtype=bool)
    drop = np.full(residuals.shape, False, dtype=bool)
    is_res[residuals >= q_hi] = True
    drop_mask = (residuals < q_hi) & (residuals > q_lo)
    drop[drop_mask] = True
    # Combine: keep top + bottom only.
    keep = ~drop & finite
    return is_res, keep


# ----------------------------- subcommands -------------------------------


def cmd_latent_class(args):
    """Fit Gaussian mixtures on residuals; pick K by BIC."""
    df = _load_residuals(Path(args.residual_csv))
    residuals = df["residual"].to_numpy(dtype=np.float64)
    out = latent_class_on_residuals(
        residuals, k_max=int(args.k_max), n_init=int(args.n_init), seed=int(args.seed),
    )
    out_path = Path(args.out_dir) / "latent_class_on_residuals.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    logger.info(
        "wrote %s — best_k=%d, BIC[k=%d]=%.2f, is_unimodal=%s",
        out_path, out["best_k"], out["best_k"],
        out["bic_per_k"][out["best_k"] - 1], out["is_unimodal"],
    )


def cmd_wasserstein(args):
    """Per-CT, per-gene Wasserstein distance between resilient/vulnerable Captum attrs."""
    captum = _load_captum(Path(args.captum_npz))
    attrs = np.asarray(captum["attributions"], dtype=np.float64)  # (n_subj, n_ct, n_gene)
    subj_ids = [str(s) for s in captum["subject_ids"]]
    res_df = _load_residuals(Path(args.residual_csv))
    res_map = dict(zip(res_df["subject_id"].astype(str), res_df["residual"].astype(float)))
    residuals = np.array([res_map.get(s, np.nan) for s in subj_ids], dtype=np.float64)

    is_res_full, keep_mask = _resilient_split(residuals, fraction=float(args.fraction))
    is_res = is_res_full[keep_mask]
    attrs_kept = attrs[keep_mask]
    logger.info(
        "wasserstein: %d resilient + %d vulnerable kept (dropped middle %d)",
        int(is_res.sum()), int((~is_res).sum()),
        int((~keep_mask).sum()),
    )

    # Cell type names from captum_summary if available.
    summary_path = Path(args.captum_npz).parent / "composite_attribution_summary.json"
    ct_names = None
    gene_names = None
    if summary_path.exists():
        s = json.loads(summary_path.read_text())
        raw = s.get("cell_types_ranked_by_total_attribution") or s.get("cell_types")
        # cell_types_ranked_by_total_attribution is a list of dicts; extract the
        # 'cell_type' string from each. NOTE: this list is RANKED by total attr,
        # NOT in original index order — for per-CT figures that need axis-aligned
        # CT names, we'd need a separate ordered list. Here we just use it for
        # display names (and reorder downstream if needed).
        if isinstance(raw, list) and raw and isinstance(raw[0], dict):
            ct_names = [d["cell_type"] for d in raw]
        else:
            ct_names = raw
    gene_names_path = Path("data/precomputed/gene_names.npy")
    if gene_names_path.exists():
        gene_names = list(np.load(gene_names_path, allow_pickle=True))

    out = wasserstein_per_celltype(attrs_kept, is_res, ct_names, gene_names)
    out_path = Path(args.out_dir) / "wasserstein_per_celltype.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    # Top 5 CT by mean per-gene W-1.
    sorted_ct = sorted(out["per_cell_type"], key=lambda r: -r["wasserstein_per_gene_mean"])
    logger.info("top-5 CT by mean per-gene W-1:")
    for r in sorted_ct[:5]:
        logger.info("  %s: %.5f", r["cell_type"], r["wasserstein_per_gene_mean"])
    logger.info("wrote %s", out_path)


def cmd_stability(args):
    """Stability selection over flattened (CT × gene) Captum attributions."""
    captum = _load_captum(Path(args.captum_npz))
    attrs = np.asarray(captum["attributions"], dtype=np.float64)
    n_subj, n_ct, n_gene = attrs.shape
    subj_ids = [str(s) for s in captum["subject_ids"]]
    res_df = _load_residuals(Path(args.residual_csv))
    res_map = dict(zip(res_df["subject_id"].astype(str), res_df["residual"].astype(float)))
    residuals = np.array([res_map.get(s, np.nan) for s in subj_ids], dtype=np.float64)

    is_res_full, keep_mask = _resilient_split(residuals, fraction=float(args.fraction))
    is_res = is_res_full[keep_mask]
    flat = attrs[keep_mask].reshape(int(keep_mask.sum()), n_ct * n_gene)
    feature_names = [
        f"CT{c}_gene{g}" for c in range(n_ct) for g in range(n_gene)
    ]
    logger.info(
        "stability: %d kept subjects × %d features", flat.shape[0], flat.shape[1],
    )
    out = stability_selection(
        flat, is_res,
        n_bootstrap=int(args.n_bootstrap),
        rb_threshold=float(args.rb_threshold),
        pi_threshold=float(args.pi_threshold),
        seed=int(args.seed),
        gene_names=feature_names,
    )
    # Drop the full selection_probability vector from the JSON to keep it small;
    # write the stable subset + config + probability summary stats only.
    probs = np.asarray(out["selection_probability"])
    summary = {
        "stable_indices": out["stable_indices"],
        "stable_genes": out["stable_genes"],
        "n_stable": len(out["stable_indices"]),
        "selection_probability_summary": {
            "mean": float(probs.mean()),
            "max": float(probs.max()),
            "p99": float(np.percentile(probs, 99)),
        },
        "config": out["config"],
    }
    out_path = Path(args.out_dir) / "stability_selection_attributions.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    logger.info(
        "wrote %s — %d stable (CT,gene) features at pi=%.2f, rb=%.2f",
        out_path, len(out["stable_indices"]), summary["config"]["pi_threshold"],
        summary["config"]["rb_threshold"],
    )


def cmd_cmi(args):
    """Conditional MI of per-CT mean attribution with composite ŷ given pathology."""
    captum = _load_captum(Path(args.captum_npz))
    attrs = np.asarray(captum["attributions"], dtype=np.float64)
    subj_ids = [str(s) for s in captum["subject_ids"]]
    # Use composite predictions as Y (from val_predictions_best.npz across folds).
    canon = Path(args.canonical_dir)
    composite_subj = []
    composite_pred = []
    for f in range(5):
        p = canon / f"fold{f}/val_predictions_best.npz"
        if not p.exists():
            continue
        d = np.load(p, allow_pickle=True)
        composite_subj.extend([str(s) for s in d["subject_ids"]])
        composite_pred.extend(np.asarray(d["predictions"], dtype=np.float64).tolist())
    pred_map = dict(zip(composite_subj, composite_pred))
    Y = np.array([pred_map.get(s, np.nan) for s in subj_ids], dtype=np.float64)

    # Pathology covariates from metadata. Column names match the canonical
    # ROSMAP schema: gpath (global pathology), amylsqrt (sqrt-transformed
    # amyloid), tangsqrt (sqrt-transformed tangles).
    md = pd.read_csv(args.metadata_csv)
    md_map = md.set_index("ROSMAP_IndividualID").to_dict("index")
    Z_cols = ["gpath", "amylsqrt", "tangsqrt"]
    Z = np.array([
        [md_map.get(s, {}).get(c, np.nan) for c in Z_cols] for s in subj_ids
    ], dtype=np.float64)

    # Per-CT mean attribution (collapse genes).
    ct_mean_attr = attrs.mean(axis=2)  # (n_subj, n_ct)

    # Cell type names from captum_summary. The list is structured as
    # [{cell_type, total_abs_attribution}, ...]; extract names.
    summary_path = Path(args.captum_npz).parent / "composite_attribution_summary.json"
    ct_names = None
    if summary_path.exists():
        s = json.loads(summary_path.read_text())
        raw = s.get("cell_types_ranked_by_total_attribution") or s.get("cell_types")
        if isinstance(raw, list) and raw and isinstance(raw[0], dict):
            ct_names = [d["cell_type"] for d in raw]
        else:
            ct_names = raw

    out = conditional_mi_per_celltype(
        ct_mean_attr, Y, Z,
        cell_type_names=ct_names,
        seed=int(args.seed),
        n_neighbors=int(args.n_neighbors),
        regressor=str(args.regressor),
        aggregation="mean",
    )
    out_path = Path(args.out_dir) / "conditional_mi_per_celltype.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    sorted_ct = sorted(
        out["per_cell_type"], key=lambda r: -(r["conditional_mi_given_pathology"] or 0),
    )
    logger.info("top-5 CT by conditional MI:")
    for r in sorted_ct[:5]:
        logger.info(
            "  %s: cond=%.4f, unc=%.4f (delta=%.4f)",
            r["cell_type"], r["conditional_mi_given_pathology"],
            r["unconditional_mi"], r["delta"],
        )
    logger.info("wrote %s", out_path)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--canonical-dir", default="outputs/redesign/p5_canonical_seed42",
    )
    p.add_argument(
        "--captum-npz",
        default="outputs/redesign/interpretability/captum_ig/composite_attributions.npz",
    )
    p.add_argument(
        "--residual-csv",
        default="outputs/redesign/interpretability/residual_per_subject.csv",
    )
    p.add_argument("--metadata-csv", default="data/metadata_ROSMAP/metadata.csv")
    p.add_argument(
        "--out-dir", default="outputs/redesign/interpretability",
    )
    p.add_argument("--seed", type=int, default=42)

    sub = p.add_subparsers(dest="cmd", required=True)

    p_lc = sub.add_parser("latent_class", help="Latent class GMM on residuals")
    p_lc.add_argument("--k-max", type=int, default=5)
    p_lc.add_argument("--n-init", type=int, default=10)
    p_lc.set_defaults(func=cmd_latent_class)

    p_w = sub.add_parser("wasserstein", help="Wasserstein per CT (resilient vs vulnerable)")
    p_w.add_argument("--fraction", type=float, default=0.25,
                     help="Top/bottom fraction of residual distribution (default 0.25 = quartiles)")
    p_w.set_defaults(func=cmd_wasserstein)

    p_s = sub.add_parser("stability", help="Stability selection over Captum attributions")
    p_s.add_argument("--fraction", type=float, default=0.25)
    p_s.add_argument("--n-bootstrap", type=int, default=100)
    p_s.add_argument("--rb-threshold", type=float, default=0.2)
    p_s.add_argument("--pi-threshold", type=float, default=0.8)
    p_s.set_defaults(func=cmd_stability)

    p_c = sub.add_parser("cmi", help="Conditional MI per CT given pathology")
    p_c.add_argument("--n-neighbors", type=int, default=5)
    p_c.add_argument("--regressor", choices=["linear", "rf"], default="linear")
    p_c.set_defaults(func=cmd_cmi)

    args = p.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
