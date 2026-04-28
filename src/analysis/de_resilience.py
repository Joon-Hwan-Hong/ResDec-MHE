"""Differential expression: resilient vs vulnerable subjects, per cell type.

Two methods, kept side-by-side for concordance reporting:

  - ``wilcoxon_de``: per-gene Mann-Whitney U test (rank-sum) with rank-biserial
    effect size + Benjamini-Hochberg FDR correction. Optional bootstrap CIs
    on the LFC. Fast, distribution-free, minimal dependencies (scipy + numpy).

  - ``deseq2_de``: pydeseq2 on raw integer pseudobulk counts. Field standard
    for bulk RNA-seq DE. Calls ``lfc_shrink`` (apeglm-style) so that
    ``log2_fold_change`` is the SHRUNKEN posterior estimate — better
    calibrated than the raw MLE LFC for downstream interpretation.

Both functions return a per-gene DataFrame with shared columns so downstream
plotting (volcano) can consume either schema interchangeably.
"""
from __future__ import annotations

import logging
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

logger = logging.getLogger(__name__)


def rank_biserial_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Rank-biserial correlation between two groups (Wilcoxon effect size).

    Equivalent to ``2*U / (n_x * n_y) - 1`` where U is the Mann-Whitney U
    statistic. Bounded in [-1, +1]; sign indicates direction of stochastic
    dominance (+ means group ``x`` tends to exceed group ``y``).

    Public function — preferred over the underscore-prefixed alias in
    ``resilience_distributional.py`` for cross-module use.
    """
    from scipy.stats import rankdata
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size < 1 or y.size < 1:
        return float("nan")
    pooled = np.concatenate([x, y])
    ranks = rankdata(pooled)
    rank_x = ranks[: x.size]
    U = float(rank_x.sum() - x.size * (x.size + 1) / 2.0)
    return float(2.0 * U / (x.size * y.size) - 1.0)


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted p-values; NaN-safe.

    NaN input p-values pass through as NaN in the output.
    """
    p = np.asarray(pvals, dtype=np.float64)
    n_total = p.size
    finite_mask = np.isfinite(p)
    n = int(finite_mask.sum())
    out = np.full(n_total, np.nan, dtype=np.float64)
    if n == 0:
        return out
    p_finite = p[finite_mask]
    order = np.argsort(p_finite)
    ranked = p_finite[order]
    factors = n / (np.arange(n) + 1.0)
    adj_sorted = np.minimum.accumulate((ranked * factors)[::-1])[::-1]
    adj_sorted = np.minimum(adj_sorted, 1.0)
    adj = np.empty(n, dtype=np.float64)
    adj[order] = adj_sorted
    out[finite_mask] = adj
    return out


def _compute_lfc(
    x: np.ndarray, y: np.ndarray, scale: Literal["counts", "log1p", "raw"],
) -> float:
    """Compute log2 fold-change between resilient ``x`` and vulnerable ``y``.

    Behavior depends on ``scale``:
      - ``"counts"``: log2(mean(x)+1) - log2(mean(y)+1). Standard for raw
        UMI/read counts where many cells = 0.
      - ``"log1p"``: mean(x) - mean(y). Input already on log1p scale, so
        difference of means IS the log2 fold-change up to log base.
      - ``"raw"``: log2(mean(x)) - log2(mean(y)). Input is on linear scale
        but no zero-inflation expected (e.g., normalized expression).
    """
    if scale == "counts":
        mx, my = float(x.mean()), float(y.mean())
        if mx < 0.0 or my < 0.0:
            raise ValueError(
                f"counts scale requires non-negative input means; "
                f"got mean(x)={mx}, mean(y)={my}. Did you pass log1p data?"
            )
        return float(np.log2(mx + 1.0) - np.log2(my + 1.0))
    if scale == "log1p":
        return float(x.mean() - y.mean())
    if scale == "raw":
        mx, my = float(x.mean()), float(y.mean())
        if mx <= 0.0 or my <= 0.0:
            raise ValueError(
                f"raw scale requires strictly-positive input means; "
                f"got mean(x)={mx}, mean(y)={my}. Use --input-scale counts "
                f"or log1p for non-positive data."
            )
        return float(np.log2(mx) - np.log2(my))
    raise ValueError(f"Unknown scale: {scale!r}; expected counts/log1p/raw")


def _bootstrap_lfc_ci(
    x: np.ndarray,
    y: np.ndarray,
    scale: Literal["counts", "log1p", "raw"],
    n_boot: int,
    conf: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Percentile bootstrap CI on the LFC; returns (lo, hi)."""
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        xb = rng.choice(x, size=x.size, replace=True)
        yb = rng.choice(y, size=y.size, replace=True)
        boots[b] = _compute_lfc(xb, yb, scale)
    alpha = 1.0 - conf
    lo = float(np.quantile(boots, alpha / 2.0))
    hi = float(np.quantile(boots, 1.0 - alpha / 2.0))
    return lo, hi


def wilcoxon_de(
    expression: np.ndarray,
    is_resilient: np.ndarray,
    *,
    gene_names: Sequence[str] | None = None,
    input_scale: Literal["counts", "log1p", "raw"] = "log1p",
    bootstrap_ci: int | None = None,
    bootstrap_conf: float = 0.95,
    seed: int = 42,
) -> pd.DataFrame:
    """Per-gene Mann-Whitney U + BH-FDR + optional LFC bootstrap CI.

    Parameters
    ----------
    expression
        Shape ``(n_subjects, n_genes)``.
    is_resilient
        Boolean ``(n_subjects,)``.
    gene_names
        Optional names; default ``gene_<j>``.
    input_scale
        How to compute log2 fold-change. ``"log1p"`` (default) treats input
        as already log-normalized; ``"counts"`` applies pseudocount; ``"raw"``
        takes log of mean directly.
    bootstrap_ci
        If not None, number of bootstrap resamples for LFC 95% CIs.
    bootstrap_conf
        Confidence level for the CI (default 0.95).
    seed
        RNG seed for the bootstrap.

    Returns
    -------
    pd.DataFrame
        Columns: ``gene, log2_fold_change, lfc_ci_lo, lfc_ci_hi, p_value,
        padj_fdr, rank_biserial, n_resilient, n_vulnerable, method``.
        ``lfc_ci_lo``/``hi`` are NaN if ``bootstrap_ci is None``.
    """
    expression = np.asarray(expression, dtype=np.float64)
    is_res = np.asarray(is_resilient, dtype=bool)
    _, n_gene = expression.shape
    if gene_names is None:
        gene_names = [f"gene_{j}" for j in range(n_gene)]
    n_res = int(is_res.sum())
    n_vul = int((~is_res).sum())
    rng = np.random.default_rng(int(seed))

    pvals = np.full(n_gene, 1.0, dtype=np.float64)
    log2fcs = np.full(n_gene, 0.0, dtype=np.float64)
    rbs = np.full(n_gene, 0.0, dtype=np.float64)
    ci_lo = np.full(n_gene, np.nan, dtype=np.float64)
    ci_hi = np.full(n_gene, np.nan, dtype=np.float64)
    n_res_finite = np.zeros(n_gene, dtype=np.int32)
    n_vul_finite = np.zeros(n_gene, dtype=np.int32)
    for g in range(n_gene):
        x = expression[is_res, g]
        y = expression[~is_res, g]
        x = x[np.isfinite(x)]
        y = y[np.isfinite(y)]
        n_res_finite[g] = x.size
        n_vul_finite[g] = y.size
        if x.size < 2 or y.size < 2:
            continue
        log2fcs[g] = _compute_lfc(x, y, input_scale)
        rbs[g] = rank_biserial_correlation(x, y)
        try:
            res = mannwhitneyu(x, y, alternative="two-sided")
            pvals[g] = float(res.pvalue)
        except ValueError:
            continue
        if bootstrap_ci is not None:
            ci_lo[g], ci_hi[g] = _bootstrap_lfc_ci(
                x, y, input_scale, n_boot=int(bootstrap_ci),
                conf=float(bootstrap_conf), rng=rng,
            )
    padj = _bh_fdr(pvals)

    return pd.DataFrame({
        "gene": list(gene_names),
        "log2_fold_change": log2fcs,
        "lfc_ci_lo": ci_lo,
        "lfc_ci_hi": ci_hi,
        "p_value": pvals,
        "padj_fdr": padj,
        "rank_biserial": rbs,
        "n_resilient": n_res,
        "n_vulnerable": n_vul,
        "n_resilient_finite": n_res_finite,
        "n_vulnerable_finite": n_vul_finite,
        "method": "wilcoxon",
    })


def deseq2_de(
    counts: np.ndarray,
    is_resilient: np.ndarray,
    *,
    gene_names: Sequence[str] | None = None,
    n_cpus: int = 4,
    cooks_cutoff: float | None = None,
    independent_filter: bool = True,
    lfc_shrink: bool = True,
) -> pd.DataFrame:
    """Pydeseq2 DE on pseudobulk counts (resilient vs vulnerable).

    Calls ``stats.lfc_shrink`` by default so the returned ``log2_fold_change``
    is the SHRUNKEN (apeglm-style) posterior estimate, which is better
    calibrated than the raw MLE LFC for downstream interpretation +
    visualization. Set ``lfc_shrink=False`` to report raw MLE LFCs.

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
    cooks_cutoff
        Pass-through to DeseqStats; ``None`` = pydeseq2 default.
    independent_filter
        Pass-through to DeseqStats (default True).
    lfc_shrink
        If True (default), call ``stats.lfc_shrink`` for shrunken LFCs.

    Returns
    -------
    pd.DataFrame
        Same columns as ``wilcoxon_de``. ``rank_biserial``/``lfc_ci_*`` are
        NaN since DESeq2 reports its own LFC standard error in a different
        coordinate; we keep column parity for downstream consumers.
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
    counts = np.maximum(counts, 0)
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
    stats_kwargs = {
        "contrast": ["condition", "resilient", "vulnerable"],
        "inference": inference,
        "quiet": True,
        "independent_filter": independent_filter,
    }
    if cooks_cutoff is not None:
        stats_kwargs["cooks_filter"] = True
        stats_kwargs["cooks_cutoff_value"] = float(cooks_cutoff)
    stats = DeseqStats(dds, **stats_kwargs)
    stats.summary()
    shrink_succeeded = False
    if lfc_shrink:
        try:
            stats.lfc_shrink(coeff="condition_resilient_vs_vulnerable")
            shrink_succeeded = True
        except (KeyError, ValueError) as exc:
            # Fall back to no-shrink if the contrast/coeff name doesn't match
            # the installed pydeseq2 version's API. Log so callers know.
            logger.warning(
                "lfc_shrink failed (%s); returning MLE LFCs instead. "
                "Method label will reflect this.", exc,
            )
    df = stats.results_df.copy()

    method_label = (
        "deseq2_lfc_shrink" if (lfc_shrink and shrink_succeeded) else "deseq2_mle"
    )
    return pd.DataFrame({
        "gene": df.index.astype(str).tolist(),
        "log2_fold_change": df["log2FoldChange"].astype(float).tolist(),
        "lfc_ci_lo": [float("nan")] * len(df),
        "lfc_ci_hi": [float("nan")] * len(df),
        "p_value": df["pvalue"].astype(float).tolist(),
        "padj_fdr": df["padj"].astype(float).tolist(),
        "rank_biserial": [float("nan")] * len(df),
        "n_resilient": int(is_res.sum()),
        "n_vulnerable": int((~is_res).sum()),
        "method": method_label,
    })
