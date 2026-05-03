"""Differential analyses across ResDec-MHE variants.

DAE — Differential Attribution Effect: per (CT, gene) paired Wilcoxon on
attribution magnitude (e.g., Captum IG) between canonical and a variant.

DCR — Differential CT Ranking: per-method Spearman rho between canonical and
variant CT-rank lists.

DCCI hooks (CT-CT edge attention magnitude shift) live alongside as a thin
wrapper over the DAE primitive on the (CT, CT) axis instead of (CT, gene).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def differential_attribution_effect(
    canonical: np.ndarray,
    variant: np.ndarray,
    *,
    ct_names: Sequence[str],
    gene_names: Sequence[str],
) -> pd.DataFrame:
    """Per (CT, gene) paired Wilcoxon between canonical and variant attributions.

    Parameters
    ----------
    canonical : (n_folds, n_ct, n_gene) array of attribution magnitudes.
    variant   : (n_folds, n_ct, n_gene) array of attribution magnitudes.

    Returns
    -------
    DataFrame with columns: cell_type, gene, mean_diff, p_wilcoxon, padj_bh.
    Sorted ascending by padj_bh.
    """
    if canonical.shape != variant.shape:
        raise ValueError(
            f"shape mismatch: canonical {canonical.shape} vs variant {variant.shape}"
        )
    if canonical.ndim != 3:
        raise ValueError(f"expected 3D arrays, got ndim={canonical.ndim}")
    n_folds, n_ct, n_gene = canonical.shape
    if len(ct_names) != n_ct:
        raise ValueError(f"ct_names len {len(ct_names)} != n_ct {n_ct}")
    if len(gene_names) != n_gene:
        raise ValueError(f"gene_names len {len(gene_names)} != n_gene {n_gene}")

    rows = []
    for ct_idx in range(n_ct):
        for g_idx in range(n_gene):
            x = canonical[:, ct_idx, g_idx]
            y = variant[:, ct_idx, g_idx]
            # scipy.stats.wilcoxon raises when all paired diffs are zero or
            # when n_diff < 1; in either case the test is uninformative and
            # we record p=1.0.
            try:
                _, p = stats.wilcoxon(x, y)
                if not np.isfinite(p):
                    p = 1.0
            except ValueError:
                p = 1.0
            rows.append({
                "cell_type": ct_names[ct_idx],
                "gene":      gene_names[g_idx],
                "mean_diff": float((y - x).mean()),
                "p_wilcoxon": float(p),
            })
    df = pd.DataFrame(rows)
    _, padj, _, _ = multipletests(df["p_wilcoxon"], method="fdr_bh")
    df["padj_bh"] = padj
    return df.sort_values(["padj_bh", "p_wilcoxon"]).reset_index(drop=True)


def quartile_subgroup_indices(
    target: np.ndarray,
    *,
    quartile: float = 0.25,
) -> dict[str, np.ndarray]:
    """Split subjects into top-quartile (resilient) and bottom-quartile (vulnerable).

    NaN targets are excluded from both subgroups.

    Parameters
    ----------
    target : 1D array of per-subject targets (e.g. residualized cogn_global).
    quartile : fraction in (0, 0.5] for the top/bottom slices. 0.25 = 25%.

    Returns
    -------
    dict with "resilient" (highest-target subjects) and "vulnerable"
    (lowest-target subjects) integer-index arrays into `target`.
    """
    if not (0.0 < quartile <= 0.5):
        raise ValueError(f"quartile must be in (0, 0.5], got {quartile}")
    target = np.asarray(target, dtype=float)
    valid_mask = ~np.isnan(target)
    valid_idx = np.flatnonzero(valid_mask)
    valid_targets = target[valid_idx]
    n_take = max(1, int(round(len(valid_idx) * quartile)))
    order = np.argsort(valid_targets)
    bottom_local = order[:n_take]
    top_local = order[-n_take:]
    return {
        "resilient": valid_idx[top_local],
        "vulnerable": valid_idx[bottom_local],
    }


def binned_subgroup_ct_importance(
    attribution: np.ndarray,
    *,
    resilient_idx: np.ndarray,
    vulnerable_idx: np.ndarray,
    ct_names: Sequence[str],
) -> pd.DataFrame:
    """Per-CT Mann-Whitney U test of attribution magnitude resilient vs vulnerable.

    Parameters
    ----------
    attribution : (n_subjects, n_ct) per-subject per-CT attribution magnitudes.
    resilient_idx, vulnerable_idx : index arrays into attribution's first axis.

    Returns
    -------
    DataFrame with columns: cell_type, mean_resilient, mean_vulnerable,
    p_wilcoxon, padj_bh.  Sorted by padj_bh ascending.
    """
    if attribution.ndim != 2:
        raise ValueError(f"expected 2D, got ndim={attribution.ndim}")
    n_subj, n_ct = attribution.shape
    if len(ct_names) != n_ct:
        raise ValueError(f"ct_names len {len(ct_names)} != n_ct {n_ct}")

    rows = []
    for ct_idx in range(n_ct):
        x_res = attribution[resilient_idx, ct_idx]
        x_vul = attribution[vulnerable_idx, ct_idx]
        try:
            _, p = stats.mannwhitneyu(x_res, x_vul, alternative="two-sided")
        except ValueError:
            p = 1.0
        rows.append({
            "cell_type": ct_names[ct_idx],
            "mean_resilient": float(np.nanmean(x_res)),
            "mean_vulnerable": float(np.nanmean(x_vul)),
            "p_wilcoxon": float(p),
        })
    df = pd.DataFrame(rows)
    _, padj, _, _ = multipletests(df["p_wilcoxon"], method="fdr_bh")
    df["padj_bh"] = padj
    return df.sort_values(["padj_bh", "p_wilcoxon"]).reset_index(drop=True)


def binned_subgroup_dge_wilcoxon(
    pseudobulk: np.ndarray,
    *,
    resilient_idx: np.ndarray,
    vulnerable_idx: np.ndarray,
    ct_names: Sequence[str],
    gene_names: Sequence[str],
) -> pd.DataFrame:
    """Per (CT, gene) Mann-Whitney U DGE between resilient and vulnerable subgroups.

    Parameters
    ----------
    pseudobulk : (n_subjects, n_ct, n_gene) per-(subject, CT, gene) expression.
    resilient_idx, vulnerable_idx : index arrays into pseudobulk's first axis.

    Returns
    -------
    DataFrame with columns: cell_type, gene, mean_resilient, mean_vulnerable,
    p_wilcoxon, padj_bh. Sorted by padj_bh ascending.
    """
    if pseudobulk.ndim != 3:
        raise ValueError(f"expected 3D, got ndim={pseudobulk.ndim}")
    n_subj, n_ct, n_gene = pseudobulk.shape
    if len(ct_names) != n_ct:
        raise ValueError(f"ct_names len {len(ct_names)} != n_ct {n_ct}")
    if len(gene_names) != n_gene:
        raise ValueError(f"gene_names len {len(gene_names)} != n_gene {n_gene}")

    rows = []
    for ct_idx in range(n_ct):
        for g_idx in range(n_gene):
            x_res = pseudobulk[resilient_idx, ct_idx, g_idx]
            x_vul = pseudobulk[vulnerable_idx, ct_idx, g_idx]
            try:
                _, p = stats.mannwhitneyu(x_res, x_vul, alternative="two-sided")
            except ValueError:
                p = 1.0
            rows.append({
                "cell_type": ct_names[ct_idx],
                "gene":      gene_names[g_idx],
                "mean_resilient": float(np.nanmean(x_res)),
                "mean_vulnerable": float(np.nanmean(x_vul)),
                "p_wilcoxon": float(p),
            })
    df = pd.DataFrame(rows)
    _, padj, _, _ = multipletests(df["p_wilcoxon"], method="fdr_bh")
    df["padj_bh"] = padj
    return df.sort_values(["padj_bh", "p_wilcoxon"]).reset_index(drop=True)


def differential_ccc_importance(
    canonical: np.ndarray,
    variant: np.ndarray,
    *,
    ct_names: Sequence[str],
) -> pd.DataFrame:
    """Per (CT_source, CT_target) paired Wilcoxon on CCC attention magnitude.

    Parameters
    ----------
    canonical, variant : (n_folds, n_ct, n_ct) per-fold mean CCC attention.
    """
    if canonical.shape != variant.shape:
        raise ValueError(
            f"shape mismatch: canonical {canonical.shape} vs variant {variant.shape}"
        )
    if canonical.ndim != 3:
        raise ValueError(f"expected 3D (n_folds, n_ct, n_ct), got ndim={canonical.ndim}")
    n_folds, n_ct, n_ct2 = canonical.shape
    if n_ct != n_ct2:
        raise ValueError(f"CT axes must match: ({n_ct}, {n_ct2})")
    if len(ct_names) != n_ct:
        raise ValueError(f"ct_names len {len(ct_names)} != n_ct {n_ct}")

    rows = []
    for src_idx in range(n_ct):
        for tgt_idx in range(n_ct):
            x = canonical[:, src_idx, tgt_idx]
            y = variant[:, src_idx, tgt_idx]
            try:
                _, p = stats.wilcoxon(x, y)
                if not np.isfinite(p):
                    p = 1.0
            except ValueError:
                p = 1.0
            rows.append({
                "ct_source": ct_names[src_idx],
                "ct_target": ct_names[tgt_idx],
                "mean_diff": float((y - x).mean()),
                "p_wilcoxon": float(p),
            })
    df = pd.DataFrame(rows)
    _, padj, _, _ = multipletests(df["p_wilcoxon"], method="fdr_bh")
    df["padj_bh"] = padj
    return df.sort_values(["padj_bh", "p_wilcoxon"]).reset_index(drop=True)


def binned_subgroup_ccc(
    ccc_attention: np.ndarray,
    *,
    resilient_idx: np.ndarray,
    vulnerable_idx: np.ndarray,
    ct_names: Sequence[str],
) -> pd.DataFrame:
    """Per (CT_source, CT_target) Mann-Whitney U on CCC attention between
    resilient and vulnerable subjects (top vs bottom quartile).

    Parameters
    ----------
    ccc_attention : (n_subjects, n_ct, n_ct) per-subject CCC attention magnitude.
    """
    if ccc_attention.ndim != 3:
        raise ValueError(f"expected 3D (n_subj, n_ct, n_ct), got ndim={ccc_attention.ndim}")
    n_subj, n_ct, n_ct2 = ccc_attention.shape
    if n_ct != n_ct2:
        raise ValueError(f"CT axes must match: ({n_ct}, {n_ct2})")
    if len(ct_names) != n_ct:
        raise ValueError(f"ct_names len {len(ct_names)} != n_ct {n_ct}")

    rows = []
    for src_idx in range(n_ct):
        for tgt_idx in range(n_ct):
            x_res = ccc_attention[resilient_idx, src_idx, tgt_idx]
            x_vul = ccc_attention[vulnerable_idx, src_idx, tgt_idx]
            try:
                _, p = stats.mannwhitneyu(x_res, x_vul, alternative="two-sided")
            except ValueError:
                p = 1.0
            rows.append({
                "ct_source": ct_names[src_idx],
                "ct_target": ct_names[tgt_idx],
                "mean_resilient": float(np.nanmean(x_res)),
                "mean_vulnerable": float(np.nanmean(x_vul)),
                "p_wilcoxon": float(p),
            })
    df = pd.DataFrame(rows)
    _, padj, _, _ = multipletests(df["p_wilcoxon"], method="fdr_bh")
    df["padj_bh"] = padj
    return df.sort_values(["padj_bh", "p_wilcoxon"]).reset_index(drop=True)


def binned_subgroup_dge_deseq2(
    raw_counts_pseudobulk: np.ndarray,
    *,
    resilient_idx: np.ndarray,
    vulnerable_idx: np.ndarray,
    ct_names: Sequence[str],
    gene_names: Sequence[str],
    min_cells_per_subject: int = 1,
) -> pd.DataFrame:
    """Per CT, run pydeseq2 between resilient and vulnerable subjects.

    Wraps src.analysis.de_resilience.deseq2_de — assumes raw INTEGER count
    pseudobulk (sum across cells, NOT log-normalized). Per (CT, gene) padj
    + shrunken log2 fold change (lfc_shrink).
    """
    from src.analysis.de_resilience import deseq2_de

    if raw_counts_pseudobulk.ndim != 3:
        raise ValueError(
            f"expected 3D raw counts (n_subj, n_ct, n_gene), got ndim={raw_counts_pseudobulk.ndim}"
        )
    n_subj, n_ct, n_gene = raw_counts_pseudobulk.shape
    if len(ct_names) != n_ct:
        raise ValueError(f"ct_names len {len(ct_names)} != n_ct {n_ct}")
    if len(gene_names) != n_gene:
        raise ValueError(f"gene_names len {len(gene_names)} != n_gene {n_gene}")

    all_rows = []
    for ct_idx in range(n_ct):
        # Per-CT counts: (n_subj, n_gene). DESeq2 wants integer counts.
        counts_ct = raw_counts_pseudobulk[:, ct_idx, :].astype(int)
        # Filter subjects with no cells of this CT (zero counts everywhere).
        nonzero = counts_ct.sum(axis=1) >= min_cells_per_subject
        res_keep = np.array([i for i in resilient_idx if nonzero[i]], dtype=int)
        vul_keep = np.array([i for i in vulnerable_idx if nonzero[i]], dtype=int)
        if len(res_keep) < 5 or len(vul_keep) < 5:
            for g_idx in range(n_gene):
                all_rows.append({
                    "cell_type": ct_names[ct_idx],
                    "gene": gene_names[g_idx],
                    "log2_fold_change": float("nan"),
                    "p_wald": float("nan"),
                    "padj_bh": float("nan"),
                    "n_resilient": len(res_keep),
                    "n_vulnerable": len(vul_keep),
                })
            continue

        keep = np.concatenate([res_keep, vul_keep])
        is_resilient = np.concatenate([
            np.ones(len(res_keep), dtype=bool),
            np.zeros(len(vul_keep), dtype=bool),
        ])
        per_gene = deseq2_de(
            counts=counts_ct[keep],
            is_resilient=is_resilient,
            gene_names=list(gene_names),
        )
        per_gene["cell_type"] = ct_names[ct_idx]
        per_gene["n_resilient"] = len(res_keep)
        per_gene["n_vulnerable"] = len(vul_keep)
        all_rows.append(per_gene)

    if not all_rows:
        return pd.DataFrame()
    if isinstance(all_rows[0], dict):
        df = pd.DataFrame(all_rows)
    else:
        df = pd.concat(all_rows, ignore_index=True)
    if "padj_bh" not in df.columns and "padj" in df.columns:
        df = df.rename(columns={"padj": "padj_bh"})
    sort_col = "padj_bh" if "padj_bh" in df.columns else df.columns[0]
    return df.sort_values(sort_col).reset_index(drop=True)


def differential_ct_ranking(
    canonical_ranks: dict[str, list],
    variant_ranks: dict[str, list],
    *,
    min_overlap: int = 3,
) -> dict[str, dict[str, float]]:
    """Per-method Spearman rho between canonical and variant CT rank lists.

    canonical_ranks / variant_ranks keys are method names; values are
    sequences of CT identifiers ordered from highest-rank to lowest. Methods
    present in only one dict are skipped (asymmetric coverage).
    """
    results: dict[str, dict[str, float]] = {}
    for method in sorted(canonical_ranks.keys() & variant_ranks.keys()):
        c = list(canonical_ranks[method])
        v = list(variant_ranks[method])
        c_map = {ct: r for r, ct in enumerate(c)}
        v_map = {ct: r for r, ct in enumerate(v)}
        common = sorted(set(c_map) & set(v_map))
        if len(common) < min_overlap:
            results[method] = {
                "spearman_rho": float("nan"),
                "p": float("nan"),
                "n": len(common),
            }
            continue
        rho, p = stats.spearmanr(
            [c_map[k] for k in common],
            [v_map[k] for k in common],
        )
        results[method] = {
            "spearman_rho": float(rho),
            "p": float(p),
            "n": len(common),
        }
    return results
