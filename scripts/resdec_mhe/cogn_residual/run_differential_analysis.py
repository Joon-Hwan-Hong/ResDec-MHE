"""Multi-method DAE / DCR / DCCI orchestrator: compares canonical attribution
outputs vs variant attribution outputs across all available methods.

Standard interp-dir layout discovered:
  - captum_ig/composite_attributions.npz
      (n_subj, n_ct, n_gene) → DAE + DCR(captum_ig)
  - captum_robustness/gradientshap_attributions.npz
      (n_subj, n_ct, n_gene) → DAE + DCR(gradient_shap)
  - captum_robustness/smoothgrad_attributions.npz (when present)
      (n_subj, n_ct, n_gene) → DAE + DCR(smoothgrad)
  - attention_attribution/per_subject_attribution.npz
      keys attnlrp/gmar/gaf_af/gaf_gf/gaf_agf each (n_subj, n_ct)
      → DCR per attention key (no DAE; no gene axis)
  - ccc/per_subject_ccc_attention.npz
      (n_subj, n_ct, n_ct[, n_edge]) → DCCI

Outputs to <out-dir>:
  - dae_canonical_vs_<variant>__<method>.csv  per gradient method
  - dcr_canonical_vs_<variant>.json           per-method Spearman rho table
  - dcci_canonical_vs_<variant>.csv           per CT-CT edge BH-FDR table
  - summary.json                               consolidated counts + headlines
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
    differential_ccc_importance,
    differential_ct_ranking,
)
from src.data.constants import CELL_TYPE_ORDER  # noqa: E402
from src.utils.gene_names import load_gene_names  # noqa: E402


def _per_fold_mean_abs(arr_per_subj: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Stack per-fold mean |attribution| → shape (n_folds, *attribution_shape)."""
    fold_ids = sorted(set(int(f) for f in folds))
    out = []
    for fid in fold_ids:
        mask = folds == fid
        out.append(np.abs(arr_per_subj[mask]).mean(axis=0))
    return np.stack(out, axis=0)


def _ct_ranking_from_per_subject_per_ct(
    arr_per_subj_per_ct: np.ndarray, ct_names: list[str],
) -> list[str]:
    # Collapse all non-CT axes — supports both (n_subj, n_ct) and (n_subj, n_ct, n_gene).
    if arr_per_subj_per_ct.ndim == 3:
        ct_total = np.abs(arr_per_subj_per_ct).sum(axis=(0, 2))
    elif arr_per_subj_per_ct.ndim == 2:
        ct_total = np.abs(arr_per_subj_per_ct).sum(axis=0)
    else:
        raise ValueError(f"unexpected ndim={arr_per_subj_per_ct.ndim}")
    order = np.argsort(-ct_total)
    return [ct_names[i] for i in order]


def _load_method_npz(path: Path) -> dict:
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _ccc_per_fold_mean_abs(
    arr_per_subj: np.ndarray, folds: np.ndarray,
) -> np.ndarray:
    """CCC: arr is (n_subj, n_ct, n_ct, n_edge?). Collapse edge dim then per-fold mean."""
    if arr_per_subj.ndim == 4:
        # average over edge types (dim -1)
        arr = np.abs(arr_per_subj).mean(axis=-1)
    elif arr_per_subj.ndim == 3:
        arr = np.abs(arr_per_subj)
    else:
        raise ValueError(f"unexpected CCC ndim={arr_per_subj.ndim}")
    fold_ids = sorted(set(int(f) for f in folds))
    out = []
    for fid in fold_ids:
        mask = folds == fid
        out.append(arr[mask].mean(axis=0))
    return np.stack(out, axis=0)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--canonical-interp-dir", type=Path,
                   default=_ROOT / "outputs/canonical/interpretability")
    p.add_argument("--variant-interp-dir", type=Path, required=True)
    p.add_argument("--variant-name", required=True)
    p.add_argument("--precomputed-dir", type=Path,
                   default=_ROOT / "data/precomputed",
                   help="For real-symbol gene names; placeholder fallback.")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {
        "variant_name": args.variant_name,
        "canonical_interp_dir": str(args.canonical_interp_dir),
        "variant_interp_dir": str(args.variant_interp_dir),
        "dae_per_method": {},
        "dcr": {},
        "dcci": None,
        "skipped_methods": [],
    }

    n_ct_default = len(CELL_TYPE_ORDER)
    canonical_ranks: dict[str, list[str]] = {}
    variant_ranks: dict[str, list[str]] = {}

    gradient_methods = [
        ("captum_ig", "captum_ig/composite_attributions.npz"),
        ("gradient_shap", "captum_robustness/gradientshap_attributions.npz"),
        ("smoothgrad", "captum_robustness/smoothgrad_attributions.npz"),
    ]
    for method, rel in gradient_methods:
        canon_path = args.canonical_interp_dir / rel
        var_path = args.variant_interp_dir / rel
        if not canon_path.is_file() or not var_path.is_file():
            summary["skipped_methods"].append(
                {"method": method, "reason": "canonical or variant NPZ missing",
                 "canon_exists": canon_path.is_file(),
                 "variant_exists": var_path.is_file()}
            )
            continue
        canon = _load_method_npz(canon_path)
        var = _load_method_npz(var_path)
        canon_pf = _per_fold_mean_abs(canon["attributions"], canon["fold"])
        var_pf = _per_fold_mean_abs(var["attributions"], var["fold"])
        if canon_pf.shape != var_pf.shape:
            summary["skipped_methods"].append({
                "method": method, "reason": "shape mismatch",
                "canon_shape": list(canon_pf.shape), "var_shape": list(var_pf.shape),
            })
            continue
        n_folds, n_ct, n_gene = canon_pf.shape
        if n_ct > len(CELL_TYPE_ORDER):
            raise ValueError(f"n_ct={n_ct} exceeds CELL_TYPE_ORDER length {len(CELL_TYPE_ORDER)}")
        ct_names = list(CELL_TYPE_ORDER)[:n_ct]
        gene_names, used_real = load_gene_names(args.precomputed_dir, n_gene)
        dae = differential_attribution_effect(
            canon_pf, var_pf, ct_names=ct_names, gene_names=gene_names,
        )
        dae_path = args.out_dir / f"dae_canonical_vs_{args.variant_name}__{method}.csv"
        dae.to_csv(dae_path, index=False)
        n_sig = int((dae["padj_bh"] < 0.05).sum())
        summary["dae_per_method"][method] = {
            "csv": str(dae_path),
            "n_sig_padj_lt_005": n_sig,
            "total_pairs": n_ct * n_gene,
            "used_real_gene_names": used_real,
        }
        canonical_ranks[method] = _ct_ranking_from_per_subject_per_ct(
            canon["attributions"], ct_names,
        )
        variant_ranks[method] = _ct_ranking_from_per_subject_per_ct(
            var["attributions"], ct_names,
        )
        print(f"  DAE {method}: {n_sig}/{n_ct * n_gene} (CT, gene) pairs at padj<0.05")

    # Attention methods (DCR only — no gene axis, no DAE)
    attn_path_canon = args.canonical_interp_dir / "attention_attribution/per_subject_attribution.npz"
    attn_path_var = args.variant_interp_dir / "attention_attribution/per_subject_attribution.npz"
    if attn_path_canon.is_file() and attn_path_var.is_file():
        canon_attn = _load_method_npz(attn_path_canon)
        var_attn = _load_method_npz(attn_path_var)
        for key in ("attnlrp", "gmar", "gaf_af", "gaf_gf", "gaf_agf"):
            if key not in canon_attn or key not in var_attn:
                summary["skipped_methods"].append(
                    {"method": key, "reason": f"key missing in canonical or variant attention NPZ"}
                )
                continue
            n_ct = canon_attn[key].shape[1]
            if n_ct > len(CELL_TYPE_ORDER):
            raise ValueError(f"n_ct={n_ct} exceeds CELL_TYPE_ORDER length {len(CELL_TYPE_ORDER)}")
        ct_names = list(CELL_TYPE_ORDER)[:n_ct]
            canonical_ranks[key] = _ct_ranking_from_per_subject_per_ct(canon_attn[key], ct_names)
            variant_ranks[key] = _ct_ranking_from_per_subject_per_ct(var_attn[key], ct_names)
            print(f"  DCR registered for attention method {key}")
    else:
        summary["skipped_methods"].append({
            "method": "attention_attribution_5methods",
            "reason": "canonical or variant attention NPZ missing",
            "canon_exists": attn_path_canon.is_file(),
            "variant_exists": attn_path_var.is_file(),
        })

    # DCR across all methods that registered
    dcr = differential_ct_ranking(canonical_ranks, variant_ranks)
    dcr_path = args.out_dir / f"dcr_canonical_vs_{args.variant_name}.json"
    dcr_path.write_text(json.dumps(dcr, indent=2))
    summary["dcr"] = {
        "json": str(dcr_path),
        "n_methods": len(dcr),
        "method_rhos": {m: r["spearman_rho"] for m, r in dcr.items()},
    }
    for m, r in dcr.items():
        print(f"  DCR {m}: rho={r['spearman_rho']:+.4f} (n={r['n']})")

    # DCCI
    ccc_path_canon = args.canonical_interp_dir / "ccc/per_subject_ccc_attention.npz"
    ccc_path_var = args.variant_interp_dir / "ccc/per_subject_ccc_attention.npz"
    if ccc_path_canon.is_file() and ccc_path_var.is_file():
        canon_ccc = _load_method_npz(ccc_path_canon)
        var_ccc = _load_method_npz(ccc_path_var)
        canon_pf = _ccc_per_fold_mean_abs(canon_ccc["attention"], canon_ccc["folds"])
        var_pf = _ccc_per_fold_mean_abs(var_ccc["attention"], var_ccc["folds"])
        if canon_pf.shape == var_pf.shape:
            n_folds, n_ct, n_ct2 = canon_pf.shape
            if n_ct > len(CELL_TYPE_ORDER):
            raise ValueError(f"n_ct={n_ct} exceeds CELL_TYPE_ORDER length {len(CELL_TYPE_ORDER)}")
        ct_names = list(CELL_TYPE_ORDER)[:n_ct]
            dcci = differential_ccc_importance(canon_pf, var_pf, ct_names=ct_names)
            dcci_path = args.out_dir / f"dcci_canonical_vs_{args.variant_name}.csv"
            dcci.to_csv(dcci_path, index=False)
            n_sig = int((dcci["padj_bh"] < 0.05).sum())
            summary["dcci"] = {
                "csv": str(dcci_path),
                "n_sig_padj_lt_005": n_sig,
                "total_edges": n_ct * n_ct,
            }
            print(f"  DCCI: {n_sig}/{n_ct * n_ct} CT-CT edges at padj<0.05")
        else:
            summary["dcci"] = {"error": f"shape mismatch canon={canon_pf.shape} var={var_pf.shape}"}
    else:
        summary["dcci"] = {"error": "canonical or variant CCC NPZ missing"}

    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
