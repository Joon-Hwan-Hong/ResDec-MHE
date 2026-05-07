"""Binned-subgroup analyses (top vs bottom quartile) for canonical or variant.

For one variant (canonical or {gpath_only, multi_axis}):
  1. Load per-fold residual targets (or for canonical: cogn_global from metadata).
  2. Pool targets across folds → quartile split (default 25%) → resilient + vulnerable
     subject indices.
  3. Run three differential analyses on the subgroups:
     - DGE Wilcoxon per (CT, gene)  (raw pseudobulk; large n_subj)
     - DGE DESeq2 per (CT, gene)    (raw integer counts; same subgroups)
     - Differential CT importance (paired test on Captum IG attribution
       magnitude resilient vs vulnerable)
     - Differential CCC (paired test on per-subject CCC attention)
  4. Write per-analysis CSV + summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))

from src.analysis.differential import (  # noqa: E402
    binned_subgroup_ccc,
    binned_subgroup_ct_importance,
    binned_subgroup_dge_deseq2,
    binned_subgroup_dge_wilcoxon,
    quartile_subgroup_indices,
)
from src.data.constants import CELL_TYPE_ORDER  # noqa: E402
from src.utils.gene_names import load_gene_names  # noqa: E402


def _build_pooled_target(
    variant: str, splits: dict, residual_cache_dir: Path | None,
    metadata_csv: Path,
) -> tuple[np.ndarray, list[str]]:
    """Return (pooled_target_per_subject, subject_ids in cohort order).

    For canonical: target = metadata['cogn_global'].
    For variants: target = per-fold residual averaged across all 5 folds. The
    residualization (α + β·gpath) is fit on train-only subjects each fold, so
    a given subject's residual differs slightly across folds; averaging gives a
    stable per-subject estimate suitable for quartile binning.
    """
    cohort_ids = sorted({s for f in splits["folds"] for s in f["train"] + f["val"]})
    if variant == "canonical":
        meta = pd.read_csv(metadata_csv)
        m = meta.set_index("ROSMAP_IndividualID")["cogn_global"]
        return np.array([m.get(s, np.nan) for s in cohort_ids], dtype=float), cohort_ids

    # Variant: load per-fold residual NPZs, average per-subject
    fold_targets = []
    for f in range(len(splits["folds"])):
        npz = np.load(residual_cache_dir / f"residual_target_fold{f}.npz", allow_pickle=True)
        sids = npz["subject_ids"].tolist()
        tgt = npz["target"].astype(float)
        per_subj = {s: t for s, t in zip(sids, tgt)}
        fold_targets.append([per_subj.get(s, np.nan) for s in cohort_ids])
    fold_arr = np.array(fold_targets)  # (n_folds, n_subj)
    return np.nanmean(fold_arr, axis=0), cohort_ids


def _load_pseudobulk(precomputed_dir: Path, subject_ids: list[str]) -> np.ndarray:
    """Load (n_subj, n_ct, n_gene) raw pseudobulk for subjects in order."""
    out = []
    keep = []
    for sid in subject_ids:
        pt = precomputed_dir / f"{sid}.pt"
        if not pt.exists():
            continue
        d = torch.load(pt, weights_only=False)
        out.append(d["pseudobulk"].numpy())
        keep.append(sid)
    return np.stack(out, axis=0), keep


def _load_attribution(npz_path: Path, subject_ids: list[str]) -> np.ndarray | None:
    if not npz_path.is_file():
        return None
    d = np.load(npz_path, allow_pickle=True)
    sid_to_idx = {str(s): i for i, s in enumerate(d["subject_ids"])}
    rows = [d["attributions"][sid_to_idx[s]] for s in subject_ids if s in sid_to_idx]
    if not rows:
        return None
    arr = np.stack(rows, axis=0)
    # Sum |attribution| over genes → per-CT importance per subject
    if arr.ndim == 3:
        return np.abs(arr).sum(axis=2)
    return arr


def _load_ccc_attention(npz_path: Path, subject_ids: list[str]) -> np.ndarray | None:
    if not npz_path.is_file():
        return None
    d = np.load(npz_path, allow_pickle=True)
    sid_to_idx = {str(s): i for i, s in enumerate(d["subject_ids"])}
    rows = [d["attention"][sid_to_idx[s]] for s in subject_ids if s in sid_to_idx]
    if not rows:
        return None
    arr = np.stack(rows, axis=0)
    if arr.ndim == 4:
        # Average over edge-type axis
        arr = np.abs(arr).mean(axis=-1)
    else:
        arr = np.abs(arr)
    return arr


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--variant", required=True,
                   choices=["canonical", "gpath_only", "multi_axis"])
    p.add_argument("--quartile", type=float, default=0.25)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--splits-path", type=Path,
                   default=_ROOT / "outputs/splits.json")
    p.add_argument("--metadata-csv", type=Path,
                   default=_ROOT / "data/metadata_ROSMAP/metadata.csv")
    p.add_argument("--precomputed-dir", type=Path,
                   default=_ROOT / "data/precomputed")
    p.add_argument("--captum-npz", type=Path, default=None,
                   help="Override path to Captum IG composite_attributions.npz "
                        "(default: outputs/canonical/.../captum_ig/composite_attributions.npz "
                        "for canonical, outputs/canonical/cogn_residual/<v>/interpretability/captum_ig/... for variants).")
    p.add_argument("--ccc-npz", type=Path, default=None)
    p.add_argument("--skip-deseq2", action="store_true",
                   help="Skip slow DESeq2 step.")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    splits = json.loads(args.splits_path.read_text())

    # Default attribution + CCC paths per variant
    if args.variant == "canonical":
        residual_cache_dir = None
        captum_default = _ROOT / "outputs/canonical/interpretability/captum_ig/composite_attributions.npz"
        ccc_default = _ROOT / "outputs/canonical/interpretability/ccc/per_subject_ccc_attention.npz"
    else:
        residual_cache_dir = _ROOT / f"outputs/canonical/cogn_residual/{args.variant}/cache"
        captum_default = _ROOT / f"outputs/canonical/cogn_residual/{args.variant}/interpretability/captum_ig/composite_attributions.npz"
        ccc_default = _ROOT / f"outputs/canonical/cogn_residual/{args.variant}/interpretability/ccc/per_subject_ccc_attention.npz"
    captum_path = args.captum_npz or captum_default
    ccc_path = args.ccc_npz or ccc_default

    target, cohort_ids = _build_pooled_target(
        args.variant, splits, residual_cache_dir, args.metadata_csv,
    )
    qres = quartile_subgroup_indices(target, quartile=args.quartile)
    res_idx = qres["resilient"]
    vul_idx = qres["vulnerable"]
    print(f"variant={args.variant}: cohort n={len(cohort_ids)}, "
          f"resilient n={len(res_idx)}, vulnerable n={len(vul_idx)} (quartile={args.quartile})")

    print("loading pseudobulk...")
    pseudobulk, kept_ids = _load_pseudobulk(args.precomputed_dir, cohort_ids)
    n_ct = pseudobulk.shape[1]
    n_gene = pseudobulk.shape[2]
    if n_ct > len(CELL_TYPE_ORDER):
        raise ValueError(f"n_ct={n_ct} exceeds CELL_TYPE_ORDER length {len(CELL_TYPE_ORDER)}")
    ct_names = list(CELL_TYPE_ORDER)[:n_ct]
    gene_names, used_real_genes = load_gene_names(args.precomputed_dir, n_gene)
    print(f"pseudobulk: {pseudobulk.shape}; kept {len(kept_ids)}/{len(cohort_ids)} subjects")

    summary = {
        "variant": args.variant, "quartile": args.quartile,
        "n_cohort": len(cohort_ids), "n_kept": len(kept_ids),
        "n_resilient": int(len(res_idx)), "n_vulnerable": int(len(vul_idx)),
        "used_real_gene_names": used_real_genes,
    }

    # Reindex resilient/vulnerable arrays to kept_ids row positions
    keep_set = set(kept_ids)
    cohort_to_keep_idx = {s: i for i, s in enumerate(kept_ids)}
    res_subjects = [cohort_ids[i] for i in res_idx if cohort_ids[i] in keep_set]
    vul_subjects = [cohort_ids[i] for i in vul_idx if cohort_ids[i] in keep_set]
    res_kept = np.array([cohort_to_keep_idx[s] for s in res_subjects], dtype=int)
    vul_kept = np.array([cohort_to_keep_idx[s] for s in vul_subjects], dtype=int)

    # 1. DGE Wilcoxon
    print("computing DGE Wilcoxon...")
    dge_wx = binned_subgroup_dge_wilcoxon(
        pseudobulk, resilient_idx=res_kept, vulnerable_idx=vul_kept,
        ct_names=ct_names, gene_names=gene_names,
    )
    out_wx = args.out_dir / "dge_wilcoxon.csv"
    dge_wx.to_csv(out_wx, index=False)
    n_wx = int((dge_wx["padj_bh"] < 0.05).sum())
    summary["dge_wilcoxon"] = {"csv": str(out_wx), "n_sig_padj_lt_005": n_wx,
                               "total_pairs": n_ct * n_gene}
    print(f"  Wilcoxon: {n_wx}/{n_ct * n_gene} (CT, gene) pairs at padj<0.05")

    # 2. DGE DESeq2
    if not args.skip_deseq2:
        print("computing DGE DESeq2... (slow; use --skip-deseq2 to bypass)")
        dge_ds = binned_subgroup_dge_deseq2(
            pseudobulk, resilient_idx=res_kept, vulnerable_idx=vul_kept,
            ct_names=ct_names, gene_names=gene_names,
        )
        out_ds = args.out_dir / "dge_deseq2.csv"
        dge_ds.to_csv(out_ds, index=False)
        n_ds = int((dge_ds["padj_bh"] < 0.05).sum()) if "padj_bh" in dge_ds.columns else 0
        summary["dge_deseq2"] = {"csv": str(out_ds), "n_sig_padj_lt_005": n_ds,
                                 "total_pairs": n_ct * n_gene}
        print(f"  DESeq2: {n_ds}/{n_ct * n_gene} (CT, gene) pairs at padj<0.05")
    else:
        summary["dge_deseq2"] = {"skipped": True}

    # 3. Differential CT importance via Captum IG
    captum = _load_attribution(captum_path, kept_ids)
    if captum is not None:
        print(f"computing differential CT importance from {captum_path}...")
        dct = binned_subgroup_ct_importance(
            captum, resilient_idx=res_kept, vulnerable_idx=vul_kept,
            ct_names=ct_names,
        )
        out_dct = args.out_dir / "ct_importance_captum.csv"
        dct.to_csv(out_dct, index=False)
        n_dct = int((dct["padj_bh"] < 0.05).sum())
        summary["ct_importance_captum"] = {"csv": str(out_dct), "n_sig_padj_lt_005": n_dct,
                                           "n_ct": n_ct, "captum_npz": str(captum_path)}
        print(f"  CT importance: {n_dct}/{n_ct} CTs at padj<0.05")
    else:
        summary["ct_importance_captum"] = {"skipped": True,
                                           "reason": f"Captum IG NPZ not at {captum_path}"}

    # 4. Differential CCC
    ccc = _load_ccc_attention(ccc_path, kept_ids)
    if ccc is not None:
        print(f"computing differential CCC from {ccc_path}...")
        dccc = binned_subgroup_ccc(
            ccc, resilient_idx=res_kept, vulnerable_idx=vul_kept,
            ct_names=ct_names,
        )
        out_dccc = args.out_dir / "differential_ccc.csv"
        dccc.to_csv(out_dccc, index=False)
        n_dccc = int((dccc["padj_bh"] < 0.05).sum())
        summary["differential_ccc"] = {"csv": str(out_dccc),
                                       "n_sig_padj_lt_005": n_dccc,
                                       "total_edges": n_ct * n_ct,
                                       "ccc_npz": str(ccc_path)}
        print(f"  Differential CCC: {n_dccc}/{n_ct * n_ct} CT-CT edges at padj<0.05")
    else:
        summary["differential_ccc"] = {"skipped": True,
                                       "reason": f"CCC NPZ not at {ccc_path}"}

    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
