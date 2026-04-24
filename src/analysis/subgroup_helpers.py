"""Shared subgroup-labeling helpers for ResDec-MHE interpretability scripts.

Consolidates the label/quartile helpers previously duplicated across
``scripts/resdec_mhe/interpretability/variance_decomposition.py`` and
``scripts/resdec_mhe/interpretability/subgroup_r2.py``. Keeping one authoritative
implementation prevents drift: APOE-ε4 count, sex string, and generic
rank-then-qcut quantile labels must be computed identically across the two
scripts so the same subjects land in the same buckets.

All helpers preserve ``None`` for NaN / missing entries so the downstream
subgroup logic drops those rows rather than silently bucketing them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def quantile_labels(
    series: pd.Series,
    n_quantiles: int = 4,
    prefix: str = "Q",
) -> pd.Series:
    """Assign ``{prefix}1..{prefix}n_quantiles`` labels via rank-then-qcut.

    ``rank(method="first")`` breaks ties so ``qcut`` always produces
    equal-sized buckets (modulo one bucket absorbing the remainder when the
    valid count is not divisible by ``n_quantiles``).

    Parameters
    ----------
    series : pd.Series
        Numeric series to quantile. NaN entries receive ``None`` labels.
    n_quantiles : int, default 4
        Number of equal-sized buckets to form over the non-null subset.
    prefix : str, default "Q"
        Label prefix; output labels are ``f"{prefix}{i}"`` for
        ``i ∈ 1..n_quantiles``.

    Returns
    -------
    pd.Series (dtype=object)
        Same index as ``series``; None for NaN entries, otherwise
        ``f"{prefix}{k}"`` for the assigned bucket.
    """
    labels = pd.Series([None] * len(series), index=series.index, dtype=object)
    valid = series.notna()
    if valid.sum() >= n_quantiles:
        q = pd.qcut(
            series.loc[valid].rank(method="first"),
            q=n_quantiles,
            labels=[f"{prefix}{i + 1}" for i in range(n_quantiles)],
        )
        labels.loc[valid] = q.astype(str).to_numpy()
    return labels


def _apoe_e4_count(genotype: object) -> object:
    """Return the number of ε4 alleles in an APOE genotype string (0/1/2) or None.

    APOE genotypes are encoded as two-digit concatenations of allele numbers
    (22, 23, 24, 33, 34, 44) — the number of "4"s is the ε4 count.
    """
    if genotype is None:
        return None
    try:
        g_float = float(genotype)
        if np.isnan(g_float):
            return None
        g_str = str(int(g_float))
    except (TypeError, ValueError):
        g_str = str(genotype)
    return g_str.count("4")


def apoe_e4_count_label(genotype: object) -> str | None:
    """Stringify the ε4 count, preserving None for missing genotypes."""
    c = _apoe_e4_count(genotype)
    return str(c) if c is not None else None


def msex_label(x: object) -> str | None:
    """Stringify msex (0/1), preserving None for NaN entries."""
    if pd.isna(x):
        return None
    return str(int(x))


__all__ = ["quantile_labels", "apoe_e4_count_label", "msex_label"]
