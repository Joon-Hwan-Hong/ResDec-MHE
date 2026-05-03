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
    return df.sort_values("padj_bh").reset_index(drop=True)


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
