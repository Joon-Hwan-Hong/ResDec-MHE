"""Differential expression: resilient vs vulnerable subjects, per cell type.

Two methods, kept side-by-side for concordance reporting:

  - ``wilcoxon_de``: per-gene Mann-Whitney U test (rank-sum) with rank-biserial
    effect size + Benjamini-Hochberg FDR correction. Fast, distribution-free,
    minimal dependencies (scipy + numpy only).

  - ``deseq2_de``: pydeseq2 on raw integer pseudobulk counts. Field standard
    for bulk RNA-seq DE; treats subject as a sample with one observation per
    cell type. Requires ``pydeseq2`` package; raises ImportError if missing.

Both functions return a per-gene DataFrame with shared columns
(gene, log2_fold_change, p_value, padj_fdr, effect_size_or_lfc_se,
n_resilient, n_vulnerable, method) so downstream plotting (volcano)
can consume either schema interchangeably.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from src.analysis.resilience_distributional import _rank_biserial_correlation


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted p-values."""
    p = np.asarray(pvals, dtype=np.float64)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    # adj[i] = min over j>=i of (n / (j+1)) * ranked[j], capped at 1.
    factors = n / (np.arange(n) + 1.0)
    adj_sorted = np.minimum.accumulate((ranked * factors)[::-1])[::-1]
    adj_sorted = np.minimum(adj_sorted, 1.0)
    adj = np.empty(n, dtype=np.float64)
    adj[order] = adj_sorted
    return adj


def wilcoxon_de(
    expression: np.ndarray,
    is_resilient: np.ndarray,
    *,
    gene_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Per-gene Mann-Whitney U + BH-FDR.

    Parameters
    ----------
    expression
        Shape ``(n_subjects, n_genes)`` — typically per-subject pseudobulk
        for ONE cell type (call once per cell type; concatenate downstream
        if you want all-CTs at once).
    is_resilient
        Boolean ``(n_subjects,)``.
    gene_names
        Optional names; default ``gene_<j>``.

    Returns
    -------
    pd.DataFrame
        Columns: ``gene, log2_fold_change, p_value, padj_fdr,
        rank_biserial, n_resilient, n_vulnerable, method``.
    """
    expression = np.asarray(expression, dtype=np.float64)
    is_res = np.asarray(is_resilient, dtype=bool)
    n_subj, n_gene = expression.shape
    if gene_names is None:
        gene_names = [f"gene_{j}" for j in range(n_gene)]
    n_res = int(is_res.sum())
    n_vul = int((~is_res).sum())

    pvals = np.full(n_gene, 1.0, dtype=np.float64)
    log2fcs = np.full(n_gene, 0.0, dtype=np.float64)
    rbs = np.full(n_gene, 0.0, dtype=np.float64)
    for g in range(n_gene):
        x = expression[is_res, g]
        y = expression[~is_res, g]
        x = x[np.isfinite(x)]
        y = y[np.isfinite(y)]
        if x.size < 2 or y.size < 2:
            continue
        # Mean-of-resilient / mean-of-vulnerable, log2 with pseudocount.
        eps = 1e-9
        log2fcs[g] = float(
            np.log2(max(x.mean(), eps) + 1.0) - np.log2(max(y.mean(), eps) + 1.0)
        )
        rbs[g] = _rank_biserial_correlation(x, y)
        try:
            res = mannwhitneyu(x, y, alternative="two-sided")
            pvals[g] = float(res.pvalue)
        except ValueError:
            # Constant input; leave p = 1.0.
            continue
    padj = _bh_fdr(pvals)

    return pd.DataFrame({
        "gene": list(gene_names),
        "log2_fold_change": log2fcs,
        "p_value": pvals,
        "padj_fdr": padj,
        "rank_biserial": rbs,
        "n_resilient": n_res,
        "n_vulnerable": n_vul,
        "method": "wilcoxon",
    })


def deseq2_de(
    counts: np.ndarray,
    is_resilient: np.ndarray,
    *,
    gene_names: Sequence[str] | None = None,
    n_cpus: int = 4,
) -> pd.DataFrame:
    """Pydeseq2 DE on pseudobulk counts (resilient vs vulnerable).

    Requires ``pydeseq2``. Raises ImportError otherwise.

    Parameters
    ----------
    counts
        Shape ``(n_subjects, n_genes)`` — INTEGER pseudobulk counts. Rounded
        to int internally if floats are passed.
    is_resilient
        Boolean ``(n_subjects,)``.
    gene_names
        Optional gene names.
    n_cpus
        Threads for pydeseq2 (default 4).

    Returns
    -------
    pd.DataFrame
        Same columns as ``wilcoxon_de`` (with ``log2_fold_change`` from
        DESeq2's lfc_shrink and ``rank_biserial`` filled with NaN since
        DESeq2's effect size is the LFC standard error / Wald statistic
        — kept here for column parity).
    """
    try:
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.ds import DeseqStats
        from pydeseq2.default_inference import DefaultInference
    except ImportError as exc:
        raise ImportError(
            "pydeseq2 not installed. uv pip install pydeseq2"
        ) from exc

    counts = np.asarray(counts)
    if not np.issubdtype(counts.dtype, np.integer):
        counts = np.rint(counts).astype(np.int64)
    counts = np.maximum(counts, 0)  # No negative counts allowed.
    is_res = np.asarray(is_resilient, dtype=bool)
    n_subj, n_gene = counts.shape
    if gene_names is None:
        gene_names = [f"gene_{j}" for j in range(n_gene)]

    metadata = pd.DataFrame({
        "condition": np.where(is_res, "resilient", "vulnerable"),
    }, index=[f"S{i:04d}" for i in range(n_subj)])
    counts_df = pd.DataFrame(
        counts, index=metadata.index, columns=list(gene_names),
    )

    inference = DefaultInference(n_cpus=n_cpus)
    dds = DeseqDataSet(
        counts=counts_df,
        metadata=metadata,
        design_factors="condition",
        refit_cooks=True,
        inference=inference,
        quiet=True,
    )
    dds.deseq2()
    stats = DeseqStats(
        dds,
        contrast=["condition", "resilient", "vulnerable"],
        inference=inference,
        quiet=True,
    )
    stats.summary()
    df = stats.results_df.copy()

    return pd.DataFrame({
        "gene": df.index.astype(str).tolist(),
        "log2_fold_change": df["log2FoldChange"].astype(float).tolist(),
        "p_value": df["pvalue"].astype(float).tolist(),
        "padj_fdr": df["padj"].astype(float).tolist(),
        "rank_biserial": [float("nan")] * len(df),
        "n_resilient": int(is_res.sum()),
        "n_vulnerable": int((~is_res).sum()),
        "method": "deseq2",
    })
